#!/usr/bin/env python3
"""Hit-rate analysis of per-layer expert caches (no I/O timing; see cachesim.py for that).

estore's LAYER policy gives every MoE layer its own share of slots and keeps each share in LRU order.
For a fixed share per layer, LRU hits are a function of each access's reuse distance: the number of
other distinct experts of the same layer used since this expert's last use. One pass over a trace
gives every layer's LRU hit curve for every share size at once, so allocations can be compared
without re-simulating.

    policy_lab.py curves TRACE OUT.npz [--warm-frac F]    per-layer reuse-distance histograms
    policy_lab.py alloc TRAIN.npz EVAL.npz --slots N        equal shares vs shares chosen on TRAIN,
                                                            both scored on EVAL
"""
import argparse
import json
import os

import numpy as np


def load_topk(d):
    meta = json.load(open(os.path.join(d, "meta.json")))
    T, L, K = meta["n_tokens"], meta["n_layer"], meta["n_expert_used"]
    topk = np.memmap(os.path.join(d, "topk.u16"), dtype=np.uint16, mode="r", shape=(T, L, K))
    return meta, topk


def curves(args):
    meta, topk = load_topk(args.trace)
    T, L, K = topk.shape
    E = meta["n_expert"]
    warm = int(T * args.warm_frac)
    # hist[l, d]: accesses of layer l with reuse distance d (d = E means first use: always a miss)
    hist = np.zeros((L, E + 1), dtype=np.int64)
    # time of each expert's last use, -1 = never. A token's 8 experts get distinct times in estore's
    # insertion order (reverse router order: the top-ranked expert is newest), so the other experts of
    # the same token count against the share; with one time per token a 1-slot share would "hold" the
    # whole previous token
    last = np.full((L, E), -1, dtype=np.int64)
    rows = np.arange(L)[:, None]
    order = (K - 1 - np.arange(K, dtype=np.int64))[None, :]   # rank 0 (top) newest
    for t in range(T):
        sel = topk[t].astype(np.int64)                 # [L, K]
        lt = last[rows, sel]                           # [L, K] last use of the selected experts
        # distinct other experts used after that: experts whose last use is later
        dist = (last[:, None, :] > lt[:, :, None]).sum(-1)
        dist[lt < 0] = E
        if t >= warm:
            np.add.at(hist, (np.broadcast_to(rows, dist.shape), dist), 1)
        last[rows, sel] = t * K + order
        if (t + 1) % 4096 == 0:
            print(f"\r{t + 1}/{T}", end="", flush=True)
    print()
    np.savez(args.out, hist=hist, n_tokens=T - warm, n_expert=E, n_used=K, trace=args.trace)
    acc = hist.sum()
    print(f"{args.out}: {T - warm} tokens after {warm} warm-up, {acc} accesses, "
          f"{hist[:, E].sum() / acc:.4f} first-use")


def hit_table(hist, cmax):
    """hits[l, c] = LRU hits of layer l with a share of c slots (c = 0..cmax)."""
    cum = np.cumsum(hist[:, :-1], axis=1)              # hits with distance <= d
    hits = np.zeros((hist.shape[0], cmax + 1), dtype=np.int64)
    n = min(cmax, cum.shape[1])
    hits[:, 1:n + 1] = cum[:, :n]                      # share c holds distances 0..c-1
    hits[:, n + 1:] = cum[:, -1:]
    return hits


def best_alloc(hits, budget):
    """Shares maximizing total hits with sum(shares) <= budget (exact DP over layers)."""
    L, C1 = hits.shape
    NEG = np.iinfo(np.int64).min // 4
    dp = np.full(budget + 1, NEG, dtype=np.int64)
    dp[0] = 0
    choice = np.zeros((L, budget + 1), dtype=np.int32)
    for l in range(L):
        nd = np.full(budget + 1, NEG, dtype=np.int64)
        for c in range(C1):
            cand = np.full(budget + 1, NEG, dtype=np.int64)
            cand[c:] = dp[:budget + 1 - c] + hits[l, c]
            better = cand > nd
            nd[better] = cand[better]
            choice[l, better] = c
        dp = nd
    b = int(np.argmax(dp))
    shares = np.zeros(L, dtype=np.int32)
    for l in range(L - 1, -1, -1):
        shares[l] = choice[l, b]
        b -= shares[l]
    return shares


def alloc(args):
    tr, ev = np.load(args.train), np.load(args.eval)
    cmax = args.cmax
    htr, hev = hit_table(tr["hist"], cmax), hit_table(ev["hist"], cmax)
    L = htr.shape[0]
    acc_ev = ev["hist"].sum()
    budget = args.slots - args.reserve
    eq = np.full(L, budget // L, dtype=np.int32)
    eq[: budget - eq.sum()] += 1                       # spread the remainder
    opt_tr = best_alloc(htr, budget)
    opt_ev = best_alloc(hev, budget)                   # oracle: fitted on the eval trace itself
    rows = np.arange(L)

    def score(shares, hits):
        return hits[rows, shares].sum() / acc_ev

    print(f"slots {args.slots} ({budget} for shares after a {args.reserve}-slot reserve), {L} layers")
    print(f"  equal shares ({eq.min()}-{eq.max()} per layer):      eval hit {score(eq, hev):.4f}")
    print(f"  shares fitted on train ({opt_tr.min()}-{opt_tr.max()}): eval hit {score(opt_tr, hev):.4f}")
    print(f"  shares fitted on eval (oracle):    eval hit {score(opt_ev, hev):.4f}")
    if args.show:
        print("  layer: train-fitted share")
        for l in range(0, L, 8):
            print("   " + " ".join(f"{i:2d}:{opt_tr[i]:2d}" for i in range(l, min(L, l + 8))))
    if args.save:
        np.savetxt(args.save, opt_tr[None, :], fmt="%d")
        print(f"  saved train-fitted shares to {args.save}")


def belady(args):
    """Hit rate of equal per-layer shares with optimal (Belady) eviction inside each share: the
    ceiling for any smarter within-layer policy, e.g. predictor- or frequency-informed eviction."""
    meta, topk = load_topk(args.trace)
    T, L, K = topk.shape
    warm = int(T * args.warm_frac)
    budget = args.slots - args.reserve
    caps = [budget // L + (1 if l < budget - (budget // L) * L else 0) for l in range(L)]
    hits = acc = 0
    INF = 1 << 60
    for l in range(L):
        seq = np.asarray(topk[:, l, :], dtype=np.int64)          # [T, K]
        # next use (token index) of each (t, k) access
        nxt = np.full((T, K), INF, dtype=np.int64)
        last_seen = {}
        for t in range(T - 1, -1, -1):
            for k in range(K):
                e = int(seq[t, k])
                nxt[t, k] = last_seen.get(e, INF)
            for k in range(K):
                last_seen[int(seq[t, k])] = t
        cache = {}                                                 # expert -> next use
        cap = caps[l]
        for t in range(T):
            cur = set(int(e) for e in seq[t])
            for k in range(K):
                e = int(seq[t, k])
                if t >= warm:
                    acc += 1
                    hits += e in cache
                cache[e] = int(nxt[t, k])
            # trim back to the share: drop the experts needed furthest in the future (current ones last)
            while len(cache) > cap:
                victim = max(cache, key=lambda x: (x not in cur, cache[x]))
                del cache[victim]
        print(f"\rlayer {l + 1}/{L}", end="", flush=True)
    print(f"\n{args.trace}: equal shares at {args.slots} slots, per-layer Belady: hit {hits / acc:.4f}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("curves")
    c.add_argument("trace")
    c.add_argument("out")
    c.add_argument("--warm-frac", type=float, default=0.2)
    a = sub.add_parser("alloc")
    a.add_argument("train")
    a.add_argument("eval")
    a.add_argument("--slots", type=int, required=True, help="total expert slots (estore: n_slots)")
    a.add_argument("--reserve", type=int, default=16, help="slots kept for the layer being computed")
    a.add_argument("--cmax", type=int, default=64, help="largest share considered per layer")
    a.add_argument("--show", action="store_true")
    a.add_argument("--save")
    b = sub.add_parser("belady")
    b.add_argument("trace")
    b.add_argument("--slots", type=int, required=True)
    b.add_argument("--reserve", type=int, default=16)
    b.add_argument("--warm-frac", type=float, default=0.2)
    args = p.parse_args()
    {"curves": curves, "alloc": alloc, "belady": belady}[args.cmd](args)


if __name__ == "__main__":
    main()
