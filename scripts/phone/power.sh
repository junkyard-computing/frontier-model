#!/bin/sh
# energy per generated token on battery: CPU-only vs CPU+GPU split at decode. Samples both fuel gauges (V x I) every 0.5 s.
# Each config runs with N=32 and N=8 decoded tokens; E(32) - E(8) = energy of 24 decoded tokens (load and prompt cancel).
ROOT=${FM_ROOT:-/userdata/frontier-model}; SRC=$(cd "$(dirname "$0")/../.." && pwd); cd "$ROOT"
P=/sys/devices/system/cpu/cpufreq; G=$(ls -d /sys/class/devfreq/*gpu* | head -1)
restore() { echo 2850000 > $P/policy6/scaling_max_freq; echo simple_ondemand > $G/governor; kill $SP 2>/dev/null; }
trap "restore; exit 1" INT TERM; trap restore EXIT
echo 2048000 > $P/policy6/scaling_max_freq
batt() { t=0; for c in /sys/class/power_supply/maxfg_base/temp /sys/class/power_supply/maxfg_secondary/temp; do v=$(( $(cat $c) / 10 )); [ $v -gt $t ] && t=$v; done; echo $t; }
big() { for z in /sys/class/thermal/thermal_zone*; do [ "$(cat $z/type)" = big-thermal ] && echo $(( $(cat $z/temp) / 1000 )); done; }
F=/sys/class/power_supply
( while :; do
    echo "$(date +%s.%N | cut -c1-14) $(cat $F/maxfg_base/voltage_now) $(cat $F/maxfg_base/current_now) $(cat $F/maxfg_secondary/voltage_now) $(cat $F/maxfg_secondary/current_now) $(cat $F/max77759-charger/status)"
    sleep 0.5
  done > /tmp/pw-samples.txt ) &
SP=$!
mark() { echo "$(date +%s.%N | cut -c1-14) $1" >> /tmp/pw-marks.txt; }
: > /tmp/pw-marks.txt
mark idle-start; sleep 30; mark idle-end
M=models/q235b-q2xl-sparse/Qwen3-235B-A22B-Instruct-2507-UD-Q2_K_XL-00001-of-00002.gguf
BASE="FM_ESTORE_STATS=1 FM_ESTORE_PACK=models/q235b-q2xl.pack FM_ESTORE_POOL_GB=6.2 FM_ESTORE_PRED_INT8=tpu/q235b-tpu-chain-102k-pkg FM_ESTORE_PRED_P=0.7 FM_ESTORE_PREFETCH_QD=2 FM_ESTORE_PRED_M=8 FM_ESTORE_PF_BUFFER=64 FM_ESTORE_PRED_TPU=1 FM_FINCH_CPUS=0-3 FM_CACHE_ROUTE=1"
for cfg in cpu32:32:FM_CPU_ONLY=1 cpu8:8:FM_CPU_ONLY=1 gpu32:32:FM_ESTORE_SPLIT=4,FM_ESTORE_SPLIT_MIN_TOKENS=1,FM_SCHED_OVERLAP=1,FM_NO_HOST=1 gpu8:8:FM_ESTORE_SPLIT=4,FM_ESTORE_SPLIT_MIN_TOKENS=1,FM_SCHED_OVERLAP=1,FM_NO_HOST=1; do
  name=${cfg%%:*}; r=${cfg#*:}; n=${r%%:*}; spec=$(echo "${r#*:}" | tr ',' ' ')
  while [ $(batt) -gt 39 ] || [ $(big) -gt 55 ]; do sleep 5; done
  case " $spec " in *" FM_ESTORE_SPLIT="*) echo userspace > $G/governor && echo 848000000 > $G/userspace/set_freq ;; esac
  mark $name-start
  env $BASE $spec taskset -c 4-7 build/fm-gen -m $M -f data/wikitext-2-raw/wiki.valid.raw --prompt-n 16 --n $n -t 4 -o /tmp/pw-$name.ids > /tmp/pw-$name.log 2>&1
  mark $name-end
  echo simple_ondemand > $G/governor
  echo "== $name: $(grep fm-gen: /tmp/pw-$name.log) (batt $(batt) C, big $(big) C)"
done
