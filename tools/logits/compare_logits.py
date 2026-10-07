#!/usr/bin/env python3
"""Compare a variant's logits with a reference run of fm-logits on the same text.

Reports perplexity of both (logits[i] predicts token i+1), mean and p99 KL(ref || variant) per
position, and how often the two agree on the top-1 token. Used to price quality trade-offs such as
routing to fewer experts per token.

usage: compare_logits.py REF.f32 VARIANT.f32 [--vocab N] [--skip K]
       (fm-logits writes OUT and OUT.tok; both runs must use the same text and --n)
"""
import argparse
import os

import numpy as np


def log_softmax(x):
    m = x.max(axis=-1, keepdims=True)
    z = x - m
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ref")
    p.add_argument("var")
    p.add_argument("--vocab", type=int, default=151936, help="n_vocab (Qwen3: 151936)")
    p.add_argument("--skip", type=int, default=16, help="ignore the first K positions (little context)")
    a = p.parse_args()
    toks = np.fromfile(a.ref + ".tok", dtype=np.int32)
    if os.path.exists(a.var + ".tok") and not np.array_equal(toks, np.fromfile(a.var + ".tok", dtype=np.int32)):
        raise SystemExit("the two runs tokenized different text")
    n = len(toks)
    R = np.memmap(a.ref, dtype=np.float32, mode="r", shape=(n, a.vocab))
    V = np.memmap(a.var, dtype=np.float32, mode="r", shape=(n, a.vocab))
    kl, nll_r, nll_v, agree = [], [], [], 0
    for i in range(a.skip, n - 1):
        lr, lv = log_softmax(R[i].astype(np.float64)), log_softmax(V[i].astype(np.float64))
        kl.append(float((np.exp(lr) * (lr - lv)).sum()))
        nll_r.append(-lr[toks[i + 1]])
        nll_v.append(-lv[toks[i + 1]])
        agree += int(np.argmax(lr) == np.argmax(lv))
    m = len(kl)
    kl = np.array(kl)
    print(f"{os.path.basename(a.var)} vs {os.path.basename(a.ref)} over {m} positions: "
          f"ppl {np.exp(np.mean(nll_r)):.3f} -> {np.exp(np.mean(nll_v)):.3f}, "
          f"KL mean {kl.mean():.4f} p99 {np.percentile(kl, 99):.3f}, top-1 agree {agree / m * 100:.1f}%")


if __name__ == "__main__":
    main()
