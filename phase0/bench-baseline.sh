#!/bin/sh
# bench-baseline.sh LLAMA_BIN_DIR MODEL.gguf [OUTDIR] — Phase 1 baselines.
#
# llama-bench prompt processing (pp512) and token generation (tg128):
#   * CPU (-ngl 0) at 2 threads (X1 only), 4 (X1 + A78) and 8 (all cores),
#     pinned: gs201 is cpu0-3 A55, cpu4-5 A78, cpu6-7 X1
#   * panvk Vulkan (-ngl 99)
# The model must fit in RAM (Qwen3-30B-A3B at ~3 bits, or a smaller one); it
# is loaded with mmap, so an oversized model measures page-cache thrash rather
# than compute — check MemAvailable first. Run under guard.sh.
set -e
BIN=$1
MODEL=$2
OUT=${3:-bench-$(date +%Y%m%d-%H%M%S)}
[ -x "$BIN/llama-bench" ] && [ -r "$MODEL" ] || { echo "usage: $0 LLAMA_BIN_DIR MODEL.gguf [OUTDIR]" >&2; exit 2; }
mkdir -p "$OUT"

avail_kb=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
model_kb=$(( $(stat -c %s "$MODEL") / 1024 ))
echo "MemAvailable ${avail_kb} kB, model ${model_kb} kB" | tee "$OUT/mem.txt"
[ "$model_kb" -lt "$avail_kb" ] || echo "WARNING: model larger than MemAvailable; results will be I/O bound" | tee -a "$OUT/mem.txt"

bench() {  # name cpus threads ngl
    taskset -c "$2" "$BIN/llama-bench" -m "$MODEL" -p 512 -n 128 -r 3 -t "$3" -ngl "$4" -o json \
        > "$OUT/$1.json" 2> "$OUT/$1.log"
    python3 - "$OUT/$1.json" "$1" <<'EOF'
import json, sys
for r in json.load(open(sys.argv[1])):
    test = f"pp{r['n_prompt']}" if r["n_prompt"] else f"tg{r['n_gen']}"
    print(f"{sys.argv[2]:10s} {test:6s} {r['avg_ts']:8.2f} ± {r['stddev_ts']:.2f} t/s")
EOF
}

{
    bench cpu-x1    6-7 2 0
    bench cpu-big   4-7 4 0
    bench cpu-all   0-7 8 0
    bench vulkan    4-7 4 99
} | tee "$OUT/summary.txt"
