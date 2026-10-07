#!/usr/bin/env python3
"""Would speculative decoding cut expert reads? Simulate it on a routing trace.

A verify pass over the last token plus K draft tokens runs K+1 positions through every layer in one
batch, so each layer loads the union of their experts once. If a drafts are accepted the pass yields a+1
tokens. Speculation only reads fewer bytes per generated token if neighbouring tokens share experts more
than the per-layer LRU cache already exploits, so this replays the trace through the same per-layer LRU
for plain decode (K=0) and for verify passes with K drafts, each draft accepted independently with
probability p. Accepted drafts route like the true next tokens; a rejected draft is a different token, so
it (and every draft after it) takes the routing of a random other position in the trace (--optimistic
uses the true next tokens instead, which wrongly turns rejected drafts into perfect prefetches).

Reports flash misses per generated token and the ratio to plain decode, plus the raw union statistic
(distinct experts per layer over K+1 consecutive tokens, as a fraction of 8(K+1)).

usage: spec_overlap.py TRACE_DIR [--slots 9] [--k 1,2,3,4,6] [--p 0.3,0.5,0.7,0.9] [--tokens 8192]
"""
import argparse
import collections
import json
import os

import numpy as np


def load(trace, n_max):
    meta = json.load(open(os.path.join(trace, "meta.json")))
    L, U = meta["n_layer"], meta["n_expert_used"]
    topk = np.fromfile(os.path.join(trace, "topk.u16"), dtype=np.uint16).reshape(-1, L, U)
    layers = meta["moe_layers"]
    return topk[:n_max][:, layers, :], meta


class LayerLRU:
    def __init__(self, slots):
        self.slots, self.d = slots, collections.OrderedDict()

    def access(self, experts):
        """Load a set of experts for one pass; returns how many were misses."""
        miss = 0
        for e in experts:
            if e in self.d:
                self.d.move_to_end(e)
            else:
                miss += 1
                self.d[e] = True
        while len(self.d) > max(self.slots, len(experts)):
            self.d.popitem(last=False)
        return miss


def simulate(topk, slots, K, p, rng, optimistic=False):
    N, L, _ = topk.shape
    caches = [LayerLRU(slots) for _ in range(L)]
    t, made, miss, passes = 0, 0, 0, 0
    while t < N:
        n_pos = min(K + 1, N - t)
        a = 0
        while a < n_pos - 1 and rng.random() < p:
            a += 1
        rows = list(range(t, t + n_pos)) if optimistic else \
            list(range(t, t + a + 1)) + rng.integers(0, N, n_pos - a - 1).tolist()
        for l in range(L):
            miss += caches[l].access(set(topk[rows, l, :].ravel().tolist()))
        made += a + 1
        t += a + 1
        passes += 1
    return miss / made, made / passes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace")
    ap.add_argument("--slots", type=int, default=9, help="cache slots per layer (phone pool 5.8 GB / 6.97 MB / 94 ~ 9)")
    ap.add_argument("--k", default="1,2,3,4,6")
    ap.add_argument("--p", default="0.3,0.5,0.7,0.9")
    ap.add_argument("--tokens", type=int, default=8192)
    ap.add_argument("--optimistic", action="store_true", help="rejected drafts route like the true next tokens")
    a = ap.parse_args()
    topk, meta = load(a.trace, a.tokens)
    N, L, U = topk.shape
    print(f"{a.trace}: {N} tokens, {L} MoE layers, top-{U}, {a.slots} LRU slots per layer")

    print("\nunion of experts over W consecutive tokens (fraction of U*W):")
    for W in [1] + [k + 1 for k in map(int, a.k.split(","))]:
        u = [len(set(topk[t:t + W, l, :].ravel().tolist())) for t in range(0, N - W, W) for l in range(0, L, 4)]
        print(f"  W={W}: {np.mean(u):6.2f} distinct / {U * W}  = {np.mean(u) / (U * W):.3f}")

    rng = np.random.default_rng(0)
    base, _ = simulate(topk, a.slots, 0, 0.0, rng)
    print(f"\nplain decode: {base:.1f} misses per token ({base / (L * U):.3f} of accesses)")
    print("speculative verify: misses per generated token, ratio to plain decode (tokens per pass)")
    print("   K  " + "".join(f"   p={p:<14}" for p in a.p.split(",")))
    for K in map(int, a.k.split(",")):
        row = []
        for p in map(float, a.p.split(",")):
            m, tpp = simulate(topk, a.slots, K, p, rng, a.optimistic)
            row.append(f"{m:6.1f} x{m / base:.2f} ({tpp:.2f})")
        print(f"  {K:2d}  " + "  ".join(row))


if __name__ == "__main__":
    main()
