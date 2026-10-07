#!/bin/sh
# cache-aware routing + skipping weak uncached experts: UD-Q2_K_XL, gamma $G, simulated LRU $SLOTS slots, FM_CACHE_SKIP=tau
cd "$(dirname "$0")/../.."
O=${OUT:-out/quality}; mkdir -p "$O"   # needs $O/q2xl-top8.f32: the plain top-8 reference logits (FM_CACHE_ROUTE unset)
M=models/q235b-q2xl/Qwen3-235B-A22B-Instruct-2507-UD-Q2_K_XL-00001-of-00002.gguf
G=${G:-1}; SLOTS=${SLOTS:-9}
for tau in "$@"; do
  echo "== gamma $G skip $tau slots $SLOTS $(date +%T)"
  env FM_CPU_ONLY=1 FM_NO_REPACK=1 FM_CACHE_ROUTE=$G FM_CACHE_SKIP=$tau FM_CACHE_ROUTE_SLOTS=$SLOTS nice -n 10 build-tpu/fm-logits -m $M -f data/wikitext-2-raw/wiki.test.raw -o $O/cs-g$G-t$tau.f32 --n 512 -t 8 2>&1 | grep -E "fm-cache-route|error|abort"
  python3 tools/logits/compare_logits.py $O/q2xl-top8.f32 $O/cs-g$G-t$tau.f32
done
