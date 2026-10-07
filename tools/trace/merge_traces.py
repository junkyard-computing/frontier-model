#!/usr/bin/env python3
"""Concatenate fm-trace outputs (same model and hidden layers) into one trace.

For a trace recorded in parts, e.g. an interrupted run continued with
`fm-trace --start-window N`. Parts are appended in the order given.

usage: merge_traces.py OUT PART [PART ...]
"""
import json
import os
import shutil
import sys

FILES = ("tokens.i32", "topk.u16", "hidden.f16")
SHAPE_KEYS = ("arch", "n_layer", "n_embd", "n_expert", "n_expert_used", "hidden_layers", "moe_layers")


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    out, parts = sys.argv[1], sys.argv[2:]
    metas = [json.load(open(os.path.join(p, "meta.json"))) for p in parts]
    for p, m in zip(parts[1:], metas[1:]):
        for k in SHAPE_KEYS:
            if m.get(k) != metas[0].get(k):
                raise SystemExit(f"{p}: {k} differs from {parts[0]}")
    os.makedirs(out, exist_ok=True)
    for name in FILES:
        if not all(os.path.exists(os.path.join(p, name)) for p in parts):
            continue
        with open(os.path.join(out, name), "wb") as fo:
            for p in parts:
                with open(os.path.join(p, name), "rb") as fi:
                    shutil.copyfileobj(fi, fo, 64 << 20)
    meta = dict(metas[0])
    meta["n_tokens"] = sum(m["n_tokens"] for m in metas)
    meta["parts"] = [{"dir": p, "n_tokens": m["n_tokens"]} for p, m in zip(parts, metas)]
    meta.pop("start_token", None)
    meta.pop("note", None)
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=2)
    # sanity: every file holds exactly n_tokens rows
    L, K, D, H = meta["n_layer"], meta["n_expert_used"], meta["n_embd"], len(meta["hidden_layers"])
    for name, row in (("tokens.i32", 4), ("topk.u16", L * K * 2), ("hidden.f16", H * D * 2)):
        f = os.path.join(out, name)
        if os.path.exists(f) and os.path.getsize(f) != meta["n_tokens"] * row:
            raise SystemExit(f"{name}: {os.path.getsize(f)} bytes, expected {meta['n_tokens'] * row}")
    print(f"merged {len(parts)} parts into {out}: {meta['n_tokens']} tokens")


if __name__ == "__main__":
    main()
