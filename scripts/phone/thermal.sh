#!/bin/sh
# tok/s vs battery heating at CPU caps: NAME:X1_KHZ:A78_KHZ args; 235B UD-Q2_K_XL, TPU predictor, -t 4 on cpus 4-7, 96 tokens
# run as root under guard.sh; each run starts at battery <= 38 C; a sampler stops fm-gen at 46 C
ROOT=${FM_ROOT:-/userdata/frontier-model}; SRC=$(cd "$(dirname "$0")/../.." && pwd); cd "$ROOT"
P=/sys/devices/system/cpu/cpufreq
batt() { t=0; for c in /sys/class/power_supply/maxfg_base/temp /sys/class/power_supply/maxfg_secondary/temp; do v=$(cat $c); [ $v -gt $t ] && t=$v; done; echo $t; }
restore() { echo 2348000 > $P/policy4/scaling_max_freq; echo 2850000 > $P/policy6/scaling_max_freq; }
trap "restore; exit 1" INT TERM; trap restore EXIT
M=models/q235b-q2xl-sparse/Qwen3-235B-A22B-Instruct-2507-UD-Q2_K_XL-00001-of-00002.gguf
for cfg in "$@"; do
  name=${cfg%%:*}; r=${cfg#*:}; x1=${r%%:*}; a78=${r#*:}
  echo 2048000 > $P/policy6/scaling_max_freq   # never above 2.05 GHz, even while cooling
  while [ $(batt) -gt 380 ]; do sleep 10; done
  [ -e /run/frontier-yield ] && { echo "STOP: yield requested"; exit 1; }
  echo $x1 > $P/policy6/scaling_max_freq; echo $a78 > $P/policy4/scaling_max_freq
  t0=$(date +%s); b0=$(batt)
  env FM_ESTORE_STATS=1 FM_ESTORE_PACK=models/q235b-q2xl.pack FM_ESTORE_POOL_GB=6.2 FM_CPU_ONLY=1 \
    FM_ESTORE_PRED_INT8=tpu/q235b-tpu-chain-102k-pkg FM_ESTORE_PRED_P=0.7 FM_ESTORE_PREFETCH_QD=2 FM_ESTORE_PRED_TPU=1 \
    FM_FINCH_CPUS=0-3 FM_ESTORE_PRED_M=8 FM_ESTORE_PF_BUFFER=64 \
    taskset -c 4-7 build/fm-gen -m $M -f data/wikitext-2-raw/wiki.valid.raw --prompt-n 16 --n 96 -t 4 -o th-$name.ids > th-$name.log 2>&1 &
  g=$!; hot=""
  while kill -0 $g 2>/dev/null; do
    echo "$(( $(date +%s) - t0 )) $(batt)" >> th-$name.temp
    [ $(batt) -ge 460 ] && { hot=" (STOPPED at 46 C)"; kill $g; }
    sleep 5
  done
  t1=$(date +%s); b1=$(batt)
  echo "== $name X1 $x1 A78 $a78: batt $b0 -> $b1 (0.1 C) in $((t1 - t0)) s$hot"
  grep -E "fm-gen:" th-$name.log
done
