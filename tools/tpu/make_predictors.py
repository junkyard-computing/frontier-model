#!/usr/bin/env python3
"""Package the int8 prefetch predictors for the Edge TPU (finch), offline.

Input: train.py --tpu-export DIR (tpu_pred.json plus int8 weights/biases) and the
trace the predictors were trained on. Each predictor is one int8 FC (2048 -> 512) or
a fused chain (2048 -> 1024 -> 512). Output in OUT:

  pred_<l>.tflite      finch fc-tflite / fc-chain-tflite
  pred_<l>.bin         finch compile: ONE executable, from scratch (no closed compiler)
  pred_<l>.s<k>.w.i8 / .s<k>.b.i32   stage k's int8 weights and bias (CPU fallback in llama.cpp)
  pred_<l>.in.i8       N test inputs (real held-out router inputs, quantized)
  pred_<l>.expect.i8   the bit-exact int8 outputs the host reference computes
  pred_<b>.expect_on_<a>.i8  predictor b's outputs on predictor a's inputs (the 2nd predictor, on
                       the 1st one's inputs: for smoke.c's finch_update check)
  manifest.txt         header "span S"; then per predictor:
                       layer n_stages d0 d1 [d2] a0 a1 [a2] w1 [w2]
                       (layer widths d, activation scales a, weight scales w)

The TPU requantizes with a 16-bit multiplier, so outputs may differ from the
reference by 1 LSB; the phone checks report exact and +-1 agreement.

usage: make_predictors.py TPU_EXPORT_DIR TRACE_DIR OUT [--finch PATH] [--n 16]
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../predictor"))
from train import int8_fc  # noqa: E402


def run(cmd):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("export")
    p.add_argument("trace")
    p.add_argument("out")
    p.add_argument("--finch", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "../../build-finch/release/finch"))
    p.add_argument("--n", type=int, default=16, help="test vectors per predictor (held-out tokens)")
    a = p.parse_args()

    meta = json.load(open(os.path.join(a.export, "tpu_pred.json")))
    tm = json.load(open(os.path.join(a.trace, "meta.json")))
    HL, D, N = tm["hidden_layers"], tm["n_embd"], tm["n_tokens"]
    hidden = np.memmap(os.path.join(a.trace, "hidden.f16"), dtype=np.float16, mode="r", shape=(N, len(HL), D))
    os.makedirs(a.out, exist_ok=True)
    rows = []
    first_xq = None          # predictor 0's test inputs, for the finch_update check (smoke.c)
    for pi, pr in enumerate(meta["predictors"]):
        l = pr["layer"]
        # older single-FC exports have no "stages"/"act_scales"
        stages = pr.get("stages") or [{"in": pr["in"], "out": pr["out"], "w_scale": pr["w_scale"]}]
        acts = pr.get("act_scales") or [pr["in_scale"], pr["out_scale"]]
        tags = [""] if len(stages) == 1 else [f".fc{k + 1}" for k in range(len(stages))]
        ws = [os.path.join(a.export, f"pred_{l}{t}.w.i8") for t in tags]
        bs = [os.path.join(a.export, f"pred_{l}{t}.b.i32") for t in tags]
        dims = [stages[0]["in"]] + [s["out"] for s in stages]

        tfl = os.path.join(a.out, f"pred_{l}.tflite")
        binp = os.path.join(a.out, f"pred_{l}.bin")
        if len(stages) == 1:
            run([a.finch, "fc-tflite", "--in-features", str(dims[0]), "--out-features", str(dims[1]),
                 "--weights", ws[0], "--bias", bs[0], "--in-scale", repr(acts[0]),
                 "--w-scale", repr(stages[0]["w_scale"]), "--out-scale", repr(acts[1]), tfl])
            with tempfile.TemporaryDirectory() as td:
                run([a.finch, "compile", tfl, "--out-dir", td])
                tiles = sorted(f for f in os.listdir(td) if f.startswith("tile_"))
                if len(tiles) != 1:
                    raise SystemExit(f"layer {l}: finch compiled {len(tiles)} tiles, want one executable")
                shutil.copy(os.path.join(td, tiles[0]), binp)
        else:
            run([a.finch, "fc-chain-tflite", "--dims", ",".join(map(str, dims)),
                 "--weights", ",".join(ws), "--bias", ",".join(bs),
                 "--act-scales", ",".join(repr(x) for x in acts),
                 "--w-scales", ",".join(repr(s["w_scale"]) for s in stages), tfl])
            run([a.finch, "compile", tfl, "--out-bin", binp])

        # test vectors from the last tokens (held out from training), through the int8 reference
        h = np.asarray(hidden[N - a.n:, HL.index(l)], dtype=np.float64)
        xq = np.clip(np.rint(h / acts[0]), -128, 127).astype(np.int8)
        for k, s in enumerate(stages):
            Wq = np.fromfile(ws[k], dtype=np.int8).reshape(s["out"], s["in"])
            bq = np.fromfile(bs[k], dtype="<i4")
            yq = int8_fc(h, acts[k], Wq, bq, s["w_scale"], acts[k + 1])
            h = yq.astype(np.float64) * acts[k + 1]
            Wq.tofile(os.path.join(a.out, f"pred_{l}.s{k}.w.i8"))
            bq.astype("<i4").tofile(os.path.join(a.out, f"pred_{l}.s{k}.b.i32"))
        xq.tofile(os.path.join(a.out, f"pred_{l}.in.i8"))
        yq.tofile(os.path.join(a.out, f"pred_{l}.expect.i8"))
        if pi == 0:
            first_xq, first_l = xq, l
        elif pi == 1:
            # smoke.c swaps executables under one input: predictor 1's output on predictor 0's inputs
            h = first_xq.astype(np.float64) * acts[0]
            for k, s in enumerate(stages):
                Wq = np.fromfile(ws[k], dtype=np.int8).reshape(s["out"], s["in"])
                bq = np.fromfile(bs[k], dtype="<i4")
                y2 = int8_fc(h, acts[k], Wq, bq, s["w_scale"], acts[k + 1])
                h = y2.astype(np.float64) * acts[k + 1]
            y2.tofile(os.path.join(a.out, f"pred_{l}.expect_on_{first_l}.i8"))
        rows.append(" ".join([str(l), str(len(stages))] + [str(d) for d in dims] + [repr(x) for x in acts] +
                             [repr(s["w_scale"]) for s in stages]))
        print(f"layer {l:2d}: {'x'.join(map(str, dims))}  tflite {os.path.getsize(tfl):>8} B  exe {os.path.getsize(binp):>8} B")
    with open(os.path.join(a.out, "manifest.txt"), "w") as f:
        f.write(f"span {meta['span']}\n")
        f.write("\n".join(rows) + "\n")
    print(f"wrote {len(rows)} predictors to {a.out} (one executable each, compiled by finch)")


if __name__ == "__main__":
    main()
