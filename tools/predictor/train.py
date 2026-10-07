#!/usr/bin/env python3
"""Train expert-prefetch predictors from an fm-trace.

One predictor per recorded hidden layer l (meta.hidden_layers). It reads the
router input of layer l (ffn_norm, n_embd wide) and predicts which experts
layers l+1 .. l+span will route to, as one multi-label head of span * n_expert
outputs.

Shapes are constrained to what finch can compile for the Edge TPU today
(tensor-tpu RESULTS.md, research/compiler-re.md):
  * FC input width must be 1024..4096  -> hidden input n_embd (2048/4096) is
    fine; a bottleneck (--rank) must be >= 1024.
  * FC output widths 64 and 128 are refused -> never a bare n_expert (128)
    head; span >= 2 gives 256+ outputs.
  * int8 only -> --qat fake-quantises weights and activations (per-tensor,
    symmetric weights, learned-scale-free for now).

Evaluation is on the last --eval-frac of the trace (contiguous, no leakage)
and reports, per lookahead distance:
  recall@K      fraction of the true K experts in the predicted top K
  prev-token    the same metric for "reuse the previous token's experts"

--emit writes the eval-split predictions (top --emit-m per predicted layer)
for cachesim.py --predictions.

--export writes the weights (f32, bottleneck folded) for llama.cpp FM_ESTORE_PRED.

usage: train.py TRACE_DIR [--span 4] [--rank 0] [--epochs 6] [--emit out.npz] [--export DIR]
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def fake_quant(x, bits=8, scale=None):
    """Symmetric per-tensor int8 (zero point 0, as finch fc-tflite emits). scale=None: from max|x|."""
    qmax = 2 ** (bits - 1) - 1
    if scale is None:
        scale = x.detach().abs().max().clamp(min=1e-8) / qmax
    q = torch.clamp(torch.round(x / scale), -qmax - 1, qmax) * scale
    return x + (q - x).detach()       # straight-through estimator


class Predictor(nn.Module):
    def __init__(self, d_in, n_out, rank, qat, in_scale=None):
        super().__init__()
        self.qat = qat
        self.in_scale = in_scale     # static input scale (the TPU cannot compute one per call)
        if rank:
            assert 1024 <= rank <= 4096, "finch FC input width must be 1024..4096"
            self.fc1 = nn.Linear(d_in, rank, bias=True)
            self.fc2 = nn.Linear(rank, n_out, bias=True)
        else:
            self.fc1 = None
            self.fc2 = nn.Linear(d_in, n_out, bias=True)

    def _lin(self, fc, x, in_scale=None):
        if not self.qat:
            return fc(x)
        return F.linear(fake_quant(x, scale=in_scale), fake_quant(fc.weight), fc.bias)

    def forward(self, x):
        if self.fc1 is not None:
            x = self._lin(self.fc1, x, self.in_scale)
            return self._lin(self.fc2, x)
        return self._lin(self.fc2, x, self.in_scale)


def int8_fc(x, in_s, Wq, bq, w_s, out_s):
    """Bit-exact symmetric int8 FC as TFLite/finch run it: int32 accumulate, requantize, clamp."""
    xq = np.clip(np.rint(x / in_s), -128, 127).astype(np.int64)
    acc = xq @ Wq.astype(np.int64).T + bq.astype(np.int64)
    return np.clip(np.rint(acc * (in_s * w_s / out_s)), -128, 127).astype(np.int8)


def recall_at_k(logits, target_ids, E, K):
    # logits [n, span*E]; target_ids [n, span, K] (0xFFFF = no MoE layer)
    n, span = target_ids.shape[:2]
    top = logits.view(n, span, E).topk(K, dim=-1).indices
    out = []
    for s in range(span):
        t = target_ids[:, s]
        valid = (t != 0xFFFF).all(dim=1)
        if valid.sum() == 0:
            out.append(float("nan"))
            continue
        hit = (top[valid, s, :, None] == t[valid, None, :]).any(dim=1).float().sum(dim=1)
        out.append((hit / K).mean().item())
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("trace")
    p.add_argument("--span", type=int, default=4)
    p.add_argument("--rank", type=int, default=0, help="0 = single FC; else >= 1024")
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--qat", action="store_true")
    p.add_argument("--eval-frac", type=float, default=0.2)
    p.add_argument("--eval-trace", help="evaluate on this trace (same model) instead of the last --eval-frac;"
                   " TRACE is then all training data. --emit predictions cover this trace")
    p.add_argument("--emit", help="write eval predictions (.npz) for cachesim")
    p.add_argument("--emit-m", type=int, default=12, help="experts emitted per predicted layer")
    p.add_argument("--emit-p", type=float, default=0.0, help="emit only experts with probability > this")
    p.add_argument("--export", help="directory for predictor weights (llama.cpp FM_ESTORE_PRED)")
    p.add_argument("--tpu-export", help="directory for int8 TPU predictors (finch fc-tflite inputs); needs --rank 0 --qat."
                   " Evaluation and --emit then use the bit-exact int8 path")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    assert args.span * 128 not in (64, 128), "finch refuses FC output width 64/128"
    if args.tpu_export:
        assert args.qat and args.rank in (0, 1024), \
            "--tpu-export needs --qat and --rank 0 (one FC) or --rank 1024 (finch fc-chain 2048 -> 1024 -> 512)"
    tpu_meta = []
    torch.manual_seed(args.seed)

    meta = json.load(open(os.path.join(args.trace, "meta.json")))
    L, E, K, D = meta["n_layer"], meta["n_expert"], meta["n_expert_used"], meta["n_embd"]
    HL = meta["hidden_layers"]
    topk = np.fromfile(os.path.join(args.trace, "topk.u16"), dtype=np.uint16).reshape(-1, L, K)
    N_tr = topk.shape[0]
    hiddens = [np.memmap(os.path.join(args.trace, "hidden.f16"), dtype=np.float16, mode="r", shape=(N_tr, len(HL), D))]
    if args.eval_trace:
        # train on all of TRACE, evaluate on a separate trace (appended after it)
        em = json.load(open(os.path.join(args.eval_trace, "meta.json")))
        assert (em["n_layer"], em["n_embd"], em["hidden_layers"]) == (L, D, HL), "eval trace shape differs"
        tk_e = np.fromfile(os.path.join(args.eval_trace, "topk.u16"), dtype=np.uint16).reshape(-1, L, K)
        hiddens.append(np.memmap(os.path.join(args.eval_trace, "hidden.f16"), dtype=np.float16, mode="r",
                                 shape=(tk_e.shape[0], len(HL), D)))
        topk = np.concatenate([topk, tk_e])
    N = topk.shape[0]

    class _Hidden:                    # hidden[:, j] across the training and evaluation traces
        def __getitem__(self, ix):
            return np.concatenate([np.asarray(h[ix]) for h in hiddens])
    hidden = _Hidden()
    n_eval = N - N_tr if args.eval_trace else int(N * args.eval_frac)
    n_train = N - n_eval
    print(f"trace: {N} tokens ({n_train} train / {n_eval} eval), {L} layers, {K}/{E} experts, "
          f"hidden layers {HL}, span {args.span}, rank {args.rank or 'none'}, qat {args.qat}")

    # targets for predictor j: layers HL[j]+1 .. HL[j]+span (0xFFFF past the end)
    def targets(j):
        l0 = HL[j] + 1
        t = np.full((N, args.span, K), 0xFFFF, dtype=np.int64)
        for s in range(args.span):
            if l0 + s < L:
                t[:, s] = topk[:, l0 + s]
        return torch.from_numpy(t)

    def multi_hot(t):
        n = t.shape[0]
        y = torch.zeros(n, args.span, E + 1)
        y.scatter_(2, torch.where(t == 0xFFFF, E, t), 1.0)
        return y[:, :, :E].reshape(n, args.span * E)

    emit = np.full((n_eval, L, args.emit_m), 0xFFFF, dtype=np.uint16)
    summary = []
    THRESHOLDS = (0.3, 0.5, 0.7, 0.9)
    pr_stats = {t: [0.0, 0.0, 0.0] for t in THRESHOLDS}
    for j, l in enumerate(HL):
        x = torch.from_numpy(np.asarray(hidden[:, j], dtype=np.float32))
        tgt = targets(j)
        mask = (tgt != 0xFFFF).all(dim=2).float()           # [N, span] layer exists
        mask_rep = mask.repeat_interleave(E, dim=1)
        xtr, xev = x[:n_train], x[n_train:]
        ttr, tev = tgt[:n_train], tgt[n_train:]

        in_scale = None
        if args.tpu_export:
            in_scale = float(np.quantile(np.abs(xtr.numpy()), 0.9999) / 127.0)
        model = Predictor(D, args.span * E, args.rank, args.qat, in_scale)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs * ((n_train + args.batch - 1) // args.batch))
        for ep in range(args.epochs):
            perm = torch.randperm(n_train)
            for b in range(0, n_train, args.batch):
                idx = perm[b:b + args.batch]
                logits = model(xtr[idx])
                y = multi_hot(ttr[idx])
                loss = (F.binary_cross_entropy_with_logits(logits, y, reduction="none") * mask_rep[idx]).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()

        model.eval()
        with torch.no_grad():
            lev = model(xev)
        if args.tpu_export:
            # quantize exactly as finch fc-tflite / fc-chain-tflite will, then evaluate with the int8 path
            fcs = [model.fc2] if model.fc1 is None else [model.fc1, model.fc2]
            with torch.no_grad():
                xs = xtr[:4096]
                acts = [xs] if model.fc1 is None else [xs, model.fc1(xs)]   # float input of each stage
                ltr = model(xs).numpy()
            scales = [in_scale] + [float(np.quantile(np.abs(a.numpy()), 0.9999) / 127.0) for a in acts[1:]] + \
                     [float(np.quantile(np.abs(ltr), 0.9999) / 127.0)]      # act scales: in, (mid,) out
            os.makedirs(args.tpu_export, exist_ok=True)
            stages, h = [], xev.numpy().astype(np.float64)
            for k, fc in enumerate(fcs):
                W = fc.weight.detach().numpy().astype(np.float64)
                b = fc.bias.detach().numpy().astype(np.float64)
                w_s = float(np.abs(W).max() / 127.0)
                Wq = np.clip(np.rint(W / w_s), -127, 127).astype(np.int8)
                bq = np.rint(b / (scales[k] * w_s)).astype(np.int32)
                h = int8_fc(h, scales[k], Wq, bq, w_s, scales[k + 1]).astype(np.float64) * scales[k + 1]
                tag = "" if len(fcs) == 1 else f".fc{k + 1}"
                Wq.tofile(os.path.join(args.tpu_export, f"pred_{l}{tag}.w.i8"))          # [out, in] row-major
                bq.astype("<i4").tofile(os.path.join(args.tpu_export, f"pred_{l}{tag}.b.i32"))
                stages.append({"in": W.shape[1], "out": W.shape[0], "w_scale": w_s})
            lev = torch.from_numpy(h.astype(np.float32))
            tpu_meta.append({"layer": l, "in": D, "out": args.span * E, "act_scales": scales,
                             "stages": stages,
                             # single-FC keys kept for tools that predate chains
                             "in_scale": scales[0], "w_scale": stages[-1]["w_scale"] if len(stages) == 1 else None,
                             "out_scale": scales[-1]})
        if args.export:
            # CPU form: fold the (linear) bottleneck into one [span*E, D] matrix
            with torch.no_grad():
                W, b = model.fc2.weight, model.fc2.bias
                if model.fc1 is not None:
                    b = W @ model.fc1.bias + b
                    W = W @ model.fc1.weight
            os.makedirs(args.export, exist_ok=True)
            with open(os.path.join(args.export, f"pred_{l}.f32"), "wb") as f:
                f.write(W.detach().numpy().astype(np.float32).tobytes())
                f.write(b.detach().numpy().astype(np.float32).tobytes())
        rec = recall_at_k(lev, tev, E, K)

        # baseline: previous token's experts at the same target layer
        base = []
        for s in range(args.span):
            lt = l + 1 + s
            if lt >= L:
                base.append(float("nan"))
                continue
            cur = topk[n_train + 1:, lt].astype(np.int64)
            prev = topk[n_train:-1, lt].astype(np.int64)
            base.append(float((cur[:, :, None] == prev[:, None, :]).any(axis=2).mean()))

        # precision/recall of "prefetch every expert with p > thr"
        prob = torch.sigmoid(lev.view(n_eval, args.span, E))
        y = multi_hot(tev).view(n_eval, args.span, E)
        m = mask[n_train:].unsqueeze(-1)
        for thr in THRESHOLDS:
            sel = (prob > thr).float() * m
            pr_stats[thr][0] += (sel * y).sum().item()      # true positives
            pr_stats[thr][1] += sel.sum().item()            # predicted
            pr_stats[thr][2] += (y * m).sum().item()        # actual

        tv = lev.view(n_eval, args.span, E).topk(args.emit_m, dim=-1)
        top = tv.indices.numpy()
        if args.emit_p > 0:
            top = np.where(torch.sigmoid(tv.values).numpy() > args.emit_p, top, 0xFFFF)
        for s in range(args.span):
            if l + 1 + s < L:
                emit[:, l + 1 + s] = top[:, s]

        line = " ".join(f"+{s + 1}:{r:.3f}/{b:.3f}" for s, (r, b) in enumerate(zip(rec, base)))
        print(f"layer {l:3d}  recall@{K} (pred/prev-token)  {line}")
        summary.append({"layer": l, "recall": rec, "prev_token": base})

    valid = [(r, b) for s in summary for r, b in zip(s["recall"], s["prev_token"]) if r == r]
    print(f"mean recall@{K}: predictor {np.mean([r for r, _ in valid]):.3f}, "
          f"prev-token {np.mean([b for _, b in valid]):.3f}")
    n_slots = sum(1 for s_ in summary for r in s_["recall"] if r == r) * n_eval
    for thr in THRESHOLDS:
        tp, npred, nact = pr_stats[thr]
        print(f"p > {thr}: precision {tp / max(npred, 1):.3f}, recall {tp / max(nact, 1):.3f}, "
              f"{npred / max(n_slots, 1):.2f} prefetches per predicted layer")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params per predictor: {n_params:,} ({n_params / 1e6:.1f} MB int8); total "
          f"{n_params * len(HL) / 1e6:.1f} MB for {len(HL)} predictors")

    if args.tpu_export:
        json.dump({"span": args.span, "n_expert": E, "hidden_layers": HL, "predictors": tpu_meta},
                  open(os.path.join(args.tpu_export, "tpu_pred.json"), "w"), indent=1)
        print(f"exported {len(tpu_meta)} int8 TPU predictors to {args.tpu_export} (metrics above use the int8 path)")

    if args.export:
        with open(os.path.join(args.export, "pred.txt"), "w") as f:
            f.write(f"fmpred 1\n{D} {E} {args.span} {len(HL)}\n{' '.join(map(str, HL))}\n")
        print(f"exported {len(HL)} predictors to {args.export}")

    if args.emit:
        eval_start = 0 if args.eval_trace else n_train   # index into the trace the predictions cover
        np.savez_compressed(args.emit, preds=emit, eval_start=eval_start, span=args.span,
                            hidden_layers=np.asarray(HL))
        # raw copy for the C replay harness (runtime/replay.c)
        emit.tofile(args.emit + ".u16")
        with open(args.emit + ".txt", "w") as f:
            f.write(f"{eval_start} {n_eval} {L} {args.emit_m} {args.span} {len(HL)}\n{' '.join(map(str, HL))}\n")
        print(f"wrote {args.emit}")


if __name__ == "__main__":
    main()
