"""Open a GGUF that may be split (llama.cpp gguf-split: name-00001-of-0000N.gguf).

Returns the parts in order, each with its GGUFReader. Metadata comes from part 1;
each tensor lives in exactly one part, at an offset within that part's file.
"""
import os
import re
import sys

_here = os.path.dirname(os.path.abspath(__file__))
for _p in ("../../third_party/llama.cpp/gguf-py", "../../../llama.cpp/gguf-py"):
    if os.path.isdir(os.path.join(_here, _p)):
        sys.path.insert(0, os.path.join(_here, _p))
        break
from gguf import GGUFReader  # noqa: E402

SPLIT = re.compile(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$")


def part_paths(path):
    m = SPLIT.match(path)
    if not m:
        return [path]
    n = int(m.group(3))
    paths = [f"{m.group(1)}-{i:05d}-of-{n:05d}.gguf" for i in range(1, n + 1)]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"missing split parts: {missing}")
    return paths


def open_parts(path):
    """[(file_path, GGUFReader)] for every part of the model."""
    return [(p, GGUFReader(p)) for p in part_paths(path)]


def field(reader, key):
    f = reader.fields.get(key)
    if f is None:
        return None
    v = f.parts[f.data[0]]
    return v.tobytes().decode() if f.types[0].name == "STRING" else v.tolist()[0]


def tensors(parts):
    """Yield (part_index, tensor) over all parts."""
    for i, (_, r) in enumerate(parts):
        for t in r.tensors:
            yield i, t
