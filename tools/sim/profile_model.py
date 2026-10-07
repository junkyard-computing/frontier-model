#!/usr/bin/env python3
"""Byte profile of a MoE GGUF: per-layer expert bytes vs resident bytes.

Writes JSON consumed by cachesim.py. Expert bytes per layer are the combined
size of one expert's gate/up/down slices (the unit we stream from flash);
everything else (attention, norms, router, embeddings, output) is resident.

usage: profile_model.py model.gguf > profile.json   (split models: pass part 1)
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../relayout"))
from gguf_parts import field, open_parts, tensors  # noqa: E402

EXPS = re.compile(r"^blk\.(\d+)\.ffn_(gate|up|down|gate_up)_exps\.weight$")
BLK = re.compile(r"^blk\.(\d+)\.")


def main():
    parts = open_parts(sys.argv[1])
    r = parts[0][1]
    arch = field(r, "general.architecture")
    n_layer = field(r, f"{arch}.block_count")
    n_expert = field(r, f"{arch}.expert_count")

    expert_bytes = [0] * n_layer          # one expert, all projections
    expert_types = [dict() for _ in range(n_layer)]
    resident_layer = [0] * n_layer
    resident_global = 0
    for _, t in tensors(parts):
        m = EXPS.match(t.name)
        if m:
            il = int(m.group(1))
            assert t.n_bytes % n_expert == 0, t.name
            expert_bytes[il] += t.n_bytes // n_expert
            expert_types[il][m.group(2)] = t.tensor_type.name
            continue
        m = BLK.match(t.name)
        if m:
            resident_layer[int(m.group(1))] += t.n_bytes
        else:
            resident_global += t.n_bytes

    json.dump({
        "model": os.path.basename(sys.argv[1]),
        "arch": arch,
        "n_layer": n_layer,
        "n_expert": n_expert,
        "n_expert_used": field(r, f"{arch}.expert_used_count"),
        "n_embd": field(r, f"{arch}.embedding_length"),
        "expert_bytes": expert_bytes,
        "expert_types": expert_types,
        "resident_layer_bytes": resident_layer,
        "resident_global_bytes": resident_global,
        "total_expert_bytes": sum(expert_bytes) * n_expert,
        "total_resident_bytes": sum(resident_layer) + resident_global,
    }, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
