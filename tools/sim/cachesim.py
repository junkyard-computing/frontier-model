#!/usr/bin/env python3
"""Expert-cache simulator for flash-streamed MoE decode.

Replays a routing trace (fm-trace output) token by token, layer by layer, as
single-stream decode on one phone, and reports expert-cache hit rate, flash
bytes per token and an estimated tokens/s.

Timing model (deliberately simple; every constant is a flag):
  * Flash is one FIFO server: a read costs expert_bytes / flash_gbps plus
    flash_lat_us / qd (per-request overhead amortised over the queue depth).
  * Decode compute is DRAM-bandwidth bound: a layer costs
    (resident layer bytes + bytes of its active experts) / mem_gbps; the
    global tensors (embeddings, output head) are charged once per token.
  * A layer starts when all of its experts are resident.
  * --demand-priority: a demand miss starts after the read in progress,
    ahead of queued prefetches (which are pushed back).
  * Prefetch (--lookahead K): at layer l we issue reads for the predicted
    experts of layer l+K (same token only: the next token is not known until
    this one is sampled). The predictor is modelled as the true routing with
    per-expert recall --recall plus --false-pos wrong guesses per layer, until
    real predictor outputs replace it.

Policies:
  lru     least recently used
  hot     a static pinned set (the most-used experts over the first
          --warm-frac of the trace) filling --pin-frac of the cache, LRU
          for the rest
  layer   estore's LAYER policy: each MoE layer keeps its own most recent
          experts in an equal share of the cache (after a 16-slot reserve),
          inserted in reverse router order and trimmed back to the share once
          the layer is computed. Unlike global LRU, layers stop evicting each
          other once a token's working set (layers x top-k) exceeds the cache
  belady  optimal offline eviction (upper bound; no prefetch)

The first --warm-frac of the trace warms the cache; statistics cover the rest.

usage: cachesim.py TRACE_DIR PROFILE.json [options]
"""
import argparse
import heapq
import json
import os
from collections import OrderedDict

import numpy as np

GB = 1e9


def load_trace(d, max_tokens=0):
    meta = json.load(open(os.path.join(d, "meta.json")))
    L, K = meta["n_layer"], meta["n_expert_used"]
    topk = np.fromfile(os.path.join(d, "topk.u16"), dtype=np.uint16)
    topk = topk.reshape(-1, L, K)
    if max_tokens:
        topk = topk[:max_tokens]
    return meta, topk


class Flash:
    def __init__(self, gbps, lat_us, qd):
        self.bps = gbps * GB
        self.overhead = lat_us * 1e-6 / qd
        self.free_at = 0.0
        self.bytes = 0
        self.pending = []      # [start, end, key] of prefetch reads
        self.demand_free_at = 0.0

    def read(self, now, nbytes, key=None):
        start = max(self.free_at, now)
        self.free_at = start + nbytes / self.bps + self.overhead
        self.bytes += nbytes
        if key is not None:
            self.pending.append([start, self.free_at, key])
        return self.free_at

    def read_demand(self, now, nbytes, inflight):
        """Demand read that overtakes queued (not yet started) prefetches."""
        self.pending = [p for p in self.pending if p[1] > now]
        cur_end = max([p[1] for p in self.pending if p[0] <= now], default=now)
        svc = nbytes / self.bps + self.overhead
        start = max(now, cur_end, self.demand_free_at)   # demand reads still serialise
        end = start + svc
        self.demand_free_at = end
        for p in self.pending:
            if p[0] >= start:
                p[0] += svc
                p[1] += svc
                if p[2] in inflight:
                    inflight[p[2]] = p[1]
        self.free_at = max(self.free_at + svc, end)
        self.bytes += nbytes
        return end


def simulate(args, meta, prof, topk, preds=None, pred_layers=(), pred_span=0):
    L = meta["n_layer"]
    E = meta["n_expert"]
    moe = meta["moe_layers"]
    ebytes = prof["expert_bytes"]
    mem_bps = args.mem_gbps * GB

    resident = prof["total_resident_bytes"]
    budget = args.ram_gb * GB - args.reserve_gb * GB - resident
    if budget <= 0:
        raise SystemExit(f"resident weights ({resident / GB:.2f} GB) exceed the RAM budget")
    total_experts = prof["total_expert_bytes"]

    rng = np.random.default_rng(args.seed)
    flash = Flash(args.flash_gbps, args.flash_lat_us, args.qd)
    N = topk.shape[0]

    # cache state: key = l * E + e
    lru = OrderedDict()
    used = 0.0
    inflight = {}          # key -> completion time
    pinned = set()

    # the first --warm-frac of the trace warms the cache (and picks the pinned
    # set for "hot"); statistics cover the rest
    warm = int(N * args.warm_frac)
    if args.policy == "hot":
        counts = np.zeros(L * E, dtype=np.int64)
        for l in moe:
            np.add.at(counts, l * E + topk[:warm, l, :].ravel().astype(np.int64), 1)
        pin_budget = budget * args.pin_frac
        for key in np.argsort(-counts):
            if counts[key] == 0:
                break
            b = ebytes[key // E]
            if used + b > pin_budget:
                break
            pinned.add(int(key))
            used += b

    # layer policy: per-layer LRU partitions, each with the same capacity (in experts)
    part = None
    if args.policy == "layer":
        avg = sum(ebytes[l] for l in moe) / len(moe)
        # same rule as estore's LAYER policy: an equal share per layer after a 16-slot reserve for
        # the layer being computed; a layer is trimmed back to its share once computed
        part_cap = max(1, int((budget / avg - 16 - args.pf_buffer) / len(moe)))
        part = {l: OrderedDict() for l in moe}
        caps = {l: part_cap for l in moe}
        if args.shares:
            # per-layer shares (policy_lab.py alloc --save), split like estore: in proportion to the
            # weights, leftover slots to the largest remainders, at least 1 per layer
            w = np.loadtxt(args.shares, dtype=np.int64).ravel()
            total = max(0, int(budget / avg - 16 - args.pf_buffer))
            ws = sum(int(w[l]) for l in moe) or 1
            caps = {l: int(w[l]) * total // ws for l in moe}
            rem = sorted(moe, key=lambda l: -(int(w[l]) * total - caps[l] * ws))
            for l in rem[: total - sum(caps.values())]:
                caps[l] += 1
            caps = {l: max(1, c) for l, c in caps.items()}
    pfbuf = OrderedDict()      # --pf-buffer: prefetched experts, outside the layer shares

    next_use = None
    if args.policy == "belady":
        # order of accesses: token-major, then layer, then expert
        keys = (np.asarray(moe, dtype=np.int64)[None, :, None] * E
                + topk[:, moe, :].astype(np.int64)).ravel()
        nu = np.empty_like(keys)
        last = {}
        for i in range(len(keys) - 1, -1, -1):
            k = int(keys[i])
            nu[i] = last.get(k, 1 << 62)
            last[k] = i
        next_use = nu
        heap = []          # (-next_use, key)
        cur_next = {}      # key -> next use of the resident copy

    def evict_for(nbytes, protect):
        nonlocal used
        while used + nbytes > budget:
            if args.policy == "belady":
                skipped = []
                while True:
                    if not heap:
                        raise SystemExit("cache too small for one layer's working set")
                    nuk, key = heapq.heappop(heap)
                    if cur_next.get(key) != -nuk:
                        continue            # stale heap entry
                    if key in protect:
                        skipped.append((nuk, key))
                        continue
                    break
                for s in skipped:
                    heapq.heappush(heap, s)
                del cur_next[key]
            else:
                for key in lru:
                    if key not in protect:
                        break
                else:
                    raise SystemExit("cache too small for one layer's working set")
                del lru[key]
            used -= ebytes[key // E]

    def part_insert(l, key, protect):
        """insert key into layer l's partition, evicting its oldest unprotected entries"""
        d = part[l]
        while len(d) >= caps[l]:
            victim = next((k for k in d if k not in protect and k not in inflight), None)
            if victim is None:
                break               # this layer's own working set overflows its share for now
            del d[victim]
        d[key] = True

    clock = 0.0
    stall = 0.0
    hits = accesses = pref_hits = 0
    pref_bytes = useful_pref = 0
    access_i = 0
    t_measure0 = None
    bytes_measure0 = 0

    for t in range(N):
        if t == warm:
            t_measure0 = clock
            bytes_measure0 = flash.bytes
            hits = accesses = pref_hits = 0
            stall = 0.0
        for li, l in enumerate(moe):
            needed = [l * E + int(e) for e in topk[t, l]]
            protect = set(needed)

            # prefetch: (layer, guessed keys) pairs to issue at this layer
            issue = []
            if preds is not None:
                if l in pred_layers:
                    for lp in range(l + 1, min(l + 1 + pred_span, L)):
                        issue.append((lp, [lp * E + int(e) for e in preds[t, lp, :args.pred_m] if e != 0xFFFF]))
            elif args.lookahead and args.policy != "belady":
                lp = l + args.lookahead
                if lp < L and lp in meta_moe_set:
                    true = [lp * E + int(e) for e in topk[t, lp]]
                    guess = [k for k in true if rng.random() < args.recall]
                    extra = rng.choice(E, size=args.false_pos, replace=False) if args.false_pos else []
                    guess += [lp * E + int(e) for e in extra if lp * E + int(e) not in true]
                    issue.append((lp, guess))
            for lp, guess in issue:
                true = {lp * E + int(e) for e in topk[t, lp]}
                if part is not None and args.pf_span and lp > l + args.pf_span:
                    continue
                for key in guess:
                    if part is not None and args.pf_buffer:
                        if key in part[lp] or key in inflight or key in pfbuf:
                            continue
                        while len(pfbuf) >= args.pf_buffer:     # drop the oldest unused prefetch
                            victim = next((k for k in pfbuf if k not in inflight), None)
                            if victim is None:
                                break
                            del pfbuf[victim]
                        if len(pfbuf) >= args.pf_buffer:
                            break
                        pfbuf[key] = True
                        b = ebytes[lp]
                        inflight[key] = flash.read(clock, b, key)
                        pref_bytes += b
                        if key in true:
                            useful_pref += b
                        continue
                    if part is not None:
                        if key in part[lp] or key in inflight:
                            continue
                        part_insert(lp, key, protect)
                        b = ebytes[lp]
                        inflight[key] = flash.read(clock, b, key)
                        pref_bytes += b
                        if key in true:
                            useful_pref += b
                        continue
                    if key in pinned or key in lru or key in inflight:
                        continue
                    b = ebytes[lp]
                    evict_for(b, protect | set(inflight))
                    lru[key] = True
                    used += b
                    inflight[key] = flash.read(clock, b)
                    pref_bytes += b
                    if key in true:
                        useful_pref += b

            ready = clock
            # LAYER policy inserts in reverse router order (the top-ranked expert ends up newest)
            for key in (needed[::-1] if part is not None else needed):
                accesses += 1
                b = ebytes[l]
                if args.policy == "belady":
                    nu = int(next_use[access_i])
                    access_i += 1
                    if key in cur_next:
                        hits += 1
                    else:
                        evict_for(b, protect)
                        used += b
                        ready = max(ready, flash.read(clock, b))
                    cur_next[key] = nu
                    heapq.heappush(heap, (-nu, key))
                    continue
                if part is not None:
                    d = part[l]
                    if key in pfbuf:            # prefetched into the buffer: move into the layer's share
                        pref_hits += 1
                        ready = max(ready, inflight.get(key, ready))
                        del pfbuf[key]
                        part_insert(l, key, protect)
                    elif key in inflight:
                        pref_hits += 1
                        ready = max(ready, inflight[key])
                        d.move_to_end(key)
                    elif key in d:
                        hits += 1
                        d.move_to_end(key)
                    else:
                        part_insert(l, key, protect)
                        rd = flash.read_demand(clock, b, inflight) if args.demand_priority else flash.read(clock, b)
                        ready = max(ready, rd)
                    continue
                if key in pinned:
                    hits += 1
                elif key in inflight:
                    pref_hits += 1
                    ready = max(ready, inflight[key])
                    lru.move_to_end(key)
                elif key in lru:
                    hits += 1
                    lru.move_to_end(key)
                else:
                    evict_for(b, protect | set(inflight))
                    lru[key] = True
                    used += b
                    rd = flash.read_demand(clock, b, inflight) if args.demand_priority else flash.read(clock, b)
                    ready = max(ready, rd)
            # drop completed in-flight entries
            for key in [k for k, c in inflight.items() if c <= ready]:
                del inflight[key]

            if part is not None:        # layer done: trim it back to its share, oldest first
                d = part[l]
                while len(d) > caps[l]:
                    victim = next((k for k in d if k not in inflight), None)
                    if victim is None:
                        break
                    del d[victim]
            stall += ready - clock
            compute = (prof["resident_layer_bytes"][l] + len(needed) * ebytes[l]) / mem_bps
            clock = ready + compute
        # dense layers without experts and global tensors
        dense = sum(prof["resident_layer_bytes"][l] for l in range(L) if l not in meta_moe_set)
        clock += (dense + prof["resident_global_bytes"] * args.global_frac) / mem_bps

    measured = N - warm
    elapsed = clock - (t_measure0 or 0.0)
    read = flash.bytes - bytes_measure0
    return {
        "policy": args.policy,
        "tokens": measured,
        "cache_gb": round(budget / GB, 2),
        "slots_per_layer": part_cap if part is not None else None,
        "cache_frac_of_experts": round(budget / total_experts, 3),
        "hit_rate": round(hits / accesses, 4),
        "prefetch_hit_rate": round(pref_hits / accesses, 4),
        "flash_mb_per_token": round(read / measured / 1e6, 1),
        "stall_frac": round(stall / elapsed, 3),
        "prefetch_useful_frac": round(useful_pref / pref_bytes, 3) if pref_bytes else None,
        "tok_per_s": round(measured / elapsed, 3),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("trace")
    p.add_argument("profile")
    p.add_argument("--policy", choices=["lru", "hot", "layer", "belady"], default="lru")
    p.add_argument("--ram-gb", type=float, default=9.0, help="RAM for weights + cache + reserve")
    p.add_argument("--reserve-gb", type=float, default=1.0, help="KV cache, scratch, runtime")
    p.add_argument("--flash-gbps", type=float, default=2.0)
    p.add_argument("--flash-lat-us", type=float, default=100.0)
    p.add_argument("--qd", type=int, default=8)
    p.add_argument("--mem-gbps", type=float, default=25.0)
    p.add_argument("--global-frac", type=float, default=1.0,
                   help="fraction of global tensors read per token (embedding rows are ~free)")
    p.add_argument("--demand-priority", action="store_true", help="demand reads overtake queued prefetches")
    p.add_argument("--lookahead", type=int, default=0)
    p.add_argument("--recall", type=float, default=1.0)
    p.add_argument("--false-pos", type=int, default=0)
    p.add_argument("--pin-frac", type=float, default=0.5)
    p.add_argument("--warm-frac", type=float, default=0.2)
    p.add_argument("--max-tokens", type=int, default=0)
    p.add_argument("--predictions", help="train.py --emit output; replaces --lookahead/--recall/--false-pos")
    p.add_argument("--pred-m", type=int, default=8, help="predicted experts prefetched per layer")
    p.add_argument("--pf-buffer", type=int, default=0,
                   help="layer policy: prefetch into a buffer of this many slots (taken from the cache) instead of the layer shares")
    p.add_argument("--shares", help="layer policy: per-layer shares file (tools/sim/policy_lab.py alloc --save)")
    p.add_argument("--pf-span", type=int, default=0, help="layer policy: only prefetch for the next N layers (0 = all predicted)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    meta, topk = load_trace(args.trace, 0)
    preds, pred_layers, pred_span = None, set(), 0
    if args.predictions:
        z = np.load(args.predictions)
        preds = z["preds"]
        start = int(z["eval_start"])
        topk = topk[start:start + len(preds)]
        pred_layers, pred_span = set(int(x) for x in z["hidden_layers"]), int(z["span"])
        assert args.pred_m <= preds.shape[2], "--pred-m exceeds emitted experts"
    if args.max_tokens:
        topk, preds = topk[:args.max_tokens], (preds[:args.max_tokens] if preds is not None else None)
    prof = json.load(open(args.profile))
    assert prof["n_layer"] == meta["n_layer"] and prof["n_expert"] == meta["n_expert"], "trace/profile mismatch"
    global meta_moe_set
    meta_moe_set = set(meta["moe_layers"])
    print(json.dumps(simulate(args, meta, prof, topk, preds, pred_layers, pred_span)))


if __name__ == "__main__":
    main()
