#!/usr/bin/env python3
"""Repack a MoE GGUF's routed experts into a flash-streaming pack.

Stock GGUF stores each layer's experts as three 3-D tensors
(ffn_{gate,up,down}_exps, expert outermost), so one expert is three
scattered reads. The pack stores each expert as one bundle:

    bundle(l, e) = [gate | pad | up | pad | down | pad]

  * every bundle starts on a --bundle-align boundary (4 KiB: O_DIRECT
    offset/length/memory alignment on UFS)
  * every projection starts on a --sub-align boundary and is padded to
    a multiple of it (64 B: the panvk decode fast path needs 64-byte
    aligned descriptor address and size)

Bytes are copied verbatim (same quant types); the original GGUF still
provides the resident tensors. The index (JSON) records, per layer, the
bundle stride and each projection's offset/size/type/shape.

Split models: pass part 1 (name-00001-of-0000N.gguf); the other parts are found.

usage: pack_experts.py model.gguf out.pack        (writes out.pack + out.pack.json)
       pack_experts.py --verify model.gguf out.pack
"""
import argparse
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gguf_parts import field, open_parts, tensors  # noqa: E402

EXPS = re.compile(r"^blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$")
PROJ_ORDER = ("gate", "up", "down")


def align(x, a):
    return (x + a - 1) // a * a


def expert_tensors(parts):
    """{layer: {proj: (part_index, tensor)}} across all split parts."""
    layers = {}
    for pi, t in tensors(parts):
        m = EXPS.match(t.name)
        if m:
            layers.setdefault(int(m.group(1)), {})[m.group(2)] = (pi, t)
    for il, d in layers.items():
        missing = set(PROJ_ORDER) - set(d)
        if missing:
            raise SystemExit(f"layer {il}: missing {missing} (fused gate_up not supported yet)")
    return dict(sorted(layers.items()))


def build_index(parts, bundle_align, sub_align):
    arch = field(parts[0][1], "general.architecture")
    layers = expert_tensors(parts)
    idx = {"source": None, "arch": arch, "bundle_align": bundle_align, "sub_align": sub_align, "layers": []}
    off = 0
    for il, d in layers.items():
        n_expert = int(d["gate"][1].shape[-1])
        projs, sub = [], 0
        for name in PROJ_ORDER:
            pi, t = d[name]
            assert int(t.shape[-1]) == n_expert and t.n_bytes % n_expert == 0, t.name
            size = t.n_bytes // n_expert
            projs.append({"name": name, "tensor": t.name, "type": t.tensor_type.name,
                          "shape": [int(x) for x in t.shape[:-1]],
                          "offset": sub, "size": size, "src_file": pi, "src_offset": int(t.data_offset)})
            sub = align(sub + size, sub_align)
        stride = align(sub, bundle_align)
        idx["layers"].append({"layer": il, "n_expert": n_expert, "base": off,
                              "bundle_stride": stride, "bundle_bytes": sub, "proj": projs})
        # one preadv can scatter the bundle into three packed per-projection
        # slot pools (what the Vulkan MoE kernels need) only if the projections
        # are back to back and each is a whole number of O_DIRECT blocks
        scatter = all(p["size"] % bundle_align == 0 for p in projs) and \
            all(projs[i + 1]["offset"] == projs[i]["offset"] + projs[i]["size"] for i in range(len(projs) - 1))
        idx["layers"][-1]["scatter_ok"] = scatter
        off += stride * n_expert
    idx["total_bytes"] = off
    idx["source_files"] = [os.path.basename(fp) for fp, _ in parts]
    idx["scatter_ok"] = all(L["scatter_ok"] for L in idx["layers"])
    return idx


def src_slice(mms, p, e):
    s = p["src_offset"] + e * p["size"]
    return mms[p.get("src_file", 0)][s:s + p["size"]]


def write_pack(model, out, bundle_align, sub_align):
    parts = open_parts(model)
    idx = build_index(parts, bundle_align, sub_align)
    idx["source"] = os.path.basename(model)
    mm = [np.memmap(fp, dtype=np.uint8, mode="r") for fp, _ in parts]
    with open(out, "wb") as f:
        for L in idx["layers"]:
            for e in range(L["n_expert"]):
                buf = np.zeros(L["bundle_stride"], dtype=np.uint8)
                for p in L["proj"]:
                    buf[p["offset"]:p["offset"] + p["size"]] = src_slice(mm, p, e)
                assert f.tell() == L["base"] + e * L["bundle_stride"]
                f.write(buf.tobytes())
            print(f"\rlayer {L['layer']}", end="", file=sys.stderr)
    print(file=sys.stderr)
    with open(out + ".json", "w") as f:
        json.dump(idx, f, indent=1)
    write_txt_index(idx, out + ".idx")
    g = sum(int(t.n_bytes) for L in expert_tensors(parts).values() for _, t in L.values())
    print(f"wrote {out}: {idx['total_bytes'] / 1e9:.3f} GB ({idx['total_bytes'] / g - 1:+.2%} padding vs "
          f"{g / 1e9:.3f} GB of experts in the GGUF)")
    print(f"scatter_ok (single preadv into per-projection pools): {idx['scatter_ok']}")


def write_txt_index(idx, path):
    """Flat index for the C runtime (runtime/estore.c)."""
    with open(path, "w") as f:
        f.write(f"fmpack 2\nbundle_align {idx['bundle_align']}\nn_layer {len(idx['layers'])}\n")
        for L in idx["layers"]:
            f.write(f"layer {L['layer']} {L['n_expert']} {L['base']} {L['bundle_stride']} {len(L['proj'])}\n")
            for p in L["proj"]:
                f.write(f"proj {p['name']} {p['type']} {p['size']} {p['offset']}\n")


def verify(model, out):
    idx = json.load(open(out + ".json"))
    from gguf_parts import part_paths
    mm = [np.memmap(fp, dtype=np.uint8, mode="r") for fp in part_paths(model)]
    pk = np.memmap(out, dtype=np.uint8, mode="r")
    assert pk.size == idx["total_bytes"], "pack size mismatch"
    ba, sa = idx["bundle_align"], idx["sub_align"]
    for L in idx["layers"]:
        assert L["base"] % ba == 0 and L["bundle_stride"] % ba == 0
        for e in range(L["n_expert"]):
            b = L["base"] + e * L["bundle_stride"]
            for p in L["proj"]:
                assert (b + p["offset"]) % sa == 0 and align(p["size"], sa) >= p["size"]
                got = pk[b + p["offset"]:b + p["offset"] + p["size"]]
                if not np.array_equal(got, src_slice(mm, p, e)):
                    raise SystemExit(f"MISMATCH layer {L['layer']} expert {e} {p['name']}")
        print(f"\rverified layer {L['layer']}", end="", file=sys.stderr)
    print(file=sys.stderr)
    print(f"OK: {sum(L['n_expert'] for L in idx['layers'])} bundles byte-identical to the GGUF, "
          f"{ba}-byte bundle / {sa}-byte projection alignment")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model")
    p.add_argument("out")
    p.add_argument("--verify", action="store_true")
    p.add_argument("--index-only", action="store_true", help="rewrite .json/.idx without rewriting the pack")
    p.add_argument("--bundle-align", type=int, default=4096)
    p.add_argument("--sub-align", type=int, default=64)
    a = p.parse_args()
    if a.verify:
        verify(a.model, a.out)
    elif a.index_only:
        idx = build_index(open_parts(a.model), a.bundle_align, a.sub_align)
        idx["source"] = os.path.basename(a.model)
        json.dump(idx, open(a.out + ".json", "w"), indent=1)
        write_txt_index(idx, a.out + ".idx")
        print(f"wrote {a.out}.json, {a.out}.idx (scatter_ok {idx['scatter_ok']})")
    else:
        write_pack(a.model, a.out, a.bundle_align, a.sub_align)


if __name__ == "__main__":
    main()
