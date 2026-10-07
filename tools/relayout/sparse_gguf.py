#!/usr/bin/env python3
"""Copy a GGUF with holes where its routed-expert tensors are (sparse files).

With expert streaming (FM_ESTORE_PACK) llama.cpp never reads the model's own *_exps
tensors (they are marked lazy and the pools stand in for them), so a device only needs
the GGUF's metadata and resident tensors plus the expert pack. The copy has the same
size and offsets as the original, so it loads unchanged, but the expert ranges are
holes: on disk it costs only the resident weights. Transfer it with `rsync --sparse`
(or `tar -S`) to keep the holes.

Uses the expert pack's index (pack_experts.py) for the expert ranges.

usage: sparse_gguf.py MODEL.gguf PACK OUT_DIR      (split models: pass part 1)
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gguf_parts import part_paths  # noqa: E402

CHUNK = 64 << 20


def main():
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    model, pack, out = sys.argv[1:]
    idx = json.load(open(pack + ".json"))
    parts = part_paths(model)
    holes = {i: [] for i in range(len(parts))}
    for L in idx["layers"]:
        for p in L["proj"]:
            start = p["src_offset"]
            holes[p.get("src_file", 0)].append((start, start + p["size"] * L["n_expert"]))
    os.makedirs(out, exist_ok=True)
    total_kept = 0
    for i, src in enumerate(parts):
        size = os.path.getsize(src)
        dst = os.path.join(out, os.path.basename(src))
        # keep everything outside the expert ranges, rounded to 4 KiB so holes are whole blocks
        keep, pos = [], 0
        for a, b in sorted(holes[i]):
            a4, b4 = (a + 4095) // 4096 * 4096, b // 4096 * 4096
            if a4 > pos:
                keep.append((pos, a4))
            pos = max(pos, b4)
        keep.append((pos, size))
        with open(src, "rb") as fi, open(dst, "wb") as fo:
            for a, b in keep:
                fi.seek(a)
                fo.seek(a)
                while a < b:
                    n = min(CHUNK, b - a)
                    fo.write(fi.read(n))
                    a += n
            fo.truncate(size)
        kept = sum(b - a for a, b in keep)
        total_kept += kept
        print(f"{os.path.basename(dst)}: {size / 1e9:.2f} GB apparent, {kept / 1e9:.2f} GB data")
    print(f"total data {total_kept / 1e9:.2f} GB (expert ranges are holes)")


if __name__ == "__main__":
    main()
