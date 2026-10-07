#!/usr/bin/env python3
"""Expert-tensor quant types of a remote (Hugging Face) GGUF, without downloading it.

Fetches only the first --mb MB of each part with an HTTP range request (the header: metadata and the
tensor table) and reports, per expert projection, how many layers use each type, plus the size of one
expert at each layer's mix. Used to pick a lower-bit build for the expert pack.

usage: gguf_remote_types.py REPO SUBPATH-PREFIX [--mb 24]
  e.g. gguf_remote_types.py unsloth/Qwen3-235B-A22B-Instruct-2507-GGUF UD-Q2_K_XL/
"""
import argparse
import collections
import json
import struct
import urllib.request

# ggml type id -> (name, block size, bytes per block)
TYPES = {0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20), 6: ("Q5_0", 32, 22),
         7: ("Q5_1", 32, 24), 8: ("Q8_0", 32, 34), 10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110),
         12: ("Q4_K", 256, 144), 13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210), 15: ("Q8_K", 256, 292),
         16: ("IQ2_XXS", 256, 66), 17: ("IQ2_XS", 256, 74), 18: ("IQ3_XXS", 256, 98), 19: ("IQ1_S", 256, 50),
         20: ("IQ4_NL", 32, 18), 21: ("IQ3_S", 256, 110), 22: ("IQ2_S", 256, 82), 23: ("IQ4_XS", 256, 136),
         29: ("IQ1_M", 256, 56), 30: ("BF16", 1, 2)}


class Buf:
    def __init__(self, b):
        self.b, self.o = b, 0

    def take(self, fmt):
        v = struct.unpack_from("<" + fmt, self.b, self.o)
        self.o += struct.calcsize("<" + fmt)
        return v[0] if len(v) == 1 else v

    def string(self):
        n = self.take("Q")
        s = self.b[self.o:self.o + n].decode("utf-8", "replace")
        self.o += n
        return s


def skip_value(r, t):
    sizes = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
    if t in sizes:
        r.take(sizes[t])
    elif t == 8:
        r.string()
    elif t == 9:
        et, n = r.take("I"), r.take("Q")
        for _ in range(n):
            skip_value(r, et)
    else:
        raise ValueError(f"unknown GGUF value type {t}")


def tensor_types(url, mb):
    req = urllib.request.Request(url, headers={"Range": f"bytes=0-{mb * 1024 * 1024 - 1}"})
    data = urllib.request.urlopen(req, timeout=120).read()
    r = Buf(data)
    if r.take("I") != 0x46554747:
        raise ValueError("not a GGUF file")
    r.take("I")                                   # version
    n_tensors, n_kv = r.take("Q"), r.take("Q")
    for _ in range(n_kv):
        r.string()
        skip_value(r, r.take("I"))
    out = []
    for _ in range(n_tensors):
        name = r.string()
        nd = r.take("I")
        dims = [r.take("Q") for _ in range(nd)]
        t = r.take("I")
        r.take("Q")                               # offset
        out.append((name, dims, t))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("repo")
    p.add_argument("prefix")
    p.add_argument("--mb", type=int, default=24)
    a = p.parse_args()
    tree = json.load(urllib.request.urlopen(f"https://huggingface.co/api/models/{a.repo}/tree/main?recursive=1", timeout=60))
    parts = sorted(f["path"] for f in tree if f["type"] == "file" and f["path"].startswith(a.prefix) and f["path"].endswith(".gguf"))
    per_layer = collections.defaultdict(dict)     # layer -> proj -> (type, bytes per expert)
    for part in parts:
        for name, dims, t in tensor_types(f"https://huggingface.co/{a.repo}/resolve/main/{part}", a.mb):
            if "_exps." not in name:
                continue
            layer = int(name.split(".")[1])
            proj = name.split(".")[2].replace("ffn_", "").replace("_exps", "")
            tn, bs, bb = TYPES.get(t, (f"type{t}", 1, 0))
            per_expert = dims[0] * dims[1] // bs * bb
            per_layer[layer][proj] = (tn, per_expert)
    mixes = collections.Counter()
    sizes = []
    for layer, projs in sorted(per_layer.items()):
        mixes[tuple(sorted((k, v[0]) for k, v in projs.items()))] += 1
        sizes.append(sum(v[1] for v in projs.values()))
    print(f"{a.repo} {a.prefix}: {len(parts)} parts, {len(per_layer)} MoE layers")
    for mix, n in mixes.most_common():
        print(f"  {n:3d} layers: " + ", ".join(f"{k}={v}" for k, v in mix))
    if sizes:
        print(f"  expert size {min(sizes) / 1e6:.2f}-{max(sizes) / 1e6:.2f} MB, mean {sum(sizes) / len(sizes) / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
