#!/bin/sh
# cache-aware routing quality: UD-Q2_K_XL top-8 with FM_CACHE_ROUTE=gamma (simulated per-layer LRU) vs plain routing
cd "$(dirname "$0")/../.."
O=${OUT:-out/quality}; mkdir -p "$O"   # needs $O/q2xl-top8.f32: the plain top-8 reference logits (FM_CACHE_ROUTE unset)
M=models/q235b-q2xl/Qwen3-235B-A22B-Instruct-2507-UD-Q2_K_XL-00001-of-00002.gguf
SLOTS=${SLOTS:-9}
for g in "$@"; do
  echo "== gamma $g slots $SLOTS $(date +%T)"
  env FM_CPU_ONLY=1 FM_NO_REPACK=1 FM_CACHE_ROUTE=$g FM_CACHE_ROUTE_SLOTS=$SLOTS nice -n 10 build-tpu/fm-logits -m $M -f data/wikitext-2-raw/wiki.test.raw -o $O/cr-g$g-s$SLOTS.f32 --n 512 -t 8 2>&1 | grep -E "fm-logits:|fm-cache-route|error|abort"
  python3 tools/logits/compare_logits.py $O/q2xl-top8.f32 $O/cr-g$g-s$SLOTS.f32
done
echo DONE
