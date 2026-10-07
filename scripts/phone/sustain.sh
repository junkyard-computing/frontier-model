#!/bin/sh
# sustained decode: 235B UD-Q2_K_XL, 16-token prompt + N decoded tokens with the thermal pause loop (battery 44/41 C,
# big cluster 85/65 C, resume on the sensor that paused). Args NAME:X1_KHZ:A78_KHZ:ENV,ENV. Root; each run starts cool.
ROOT=${FM_ROOT:-/userdata/frontier-model}; SRC=$(cd "$(dirname "$0")/../.." && pwd); cd "$ROOT"
P=/sys/devices/system/cpu/cpufreq
restore() { echo 2348000 > $P/policy4/scaling_max_freq; echo 2850000 > $P/policy6/scaling_max_freq; }
trap "restore; exit 1" INT TERM; trap restore EXIT
batt() { t=0; for c in /sys/class/power_supply/maxfg_base/temp /sys/class/power_supply/maxfg_secondary/temp; do v=$(( $(cat $c) / 10 )); [ $v -gt $t ] && t=$v; done; echo $t; }
big() { for z in /sys/class/thermal/thermal_zone*; do [ "$(cat $z/type)" = big-thermal ] && echo $(( $(cat $z/temp) / 1000 )); done; }
throttle() {   # $1 = pid, $2 = log
  st=run; why=""
  while kill -0 $1 2>/dev/null; do
    t=$(batt); b=$(big)
    if [ $st = run ]; then
      if [ $t -ge 44 ]; then why=batt; elif [ $b -ge 85 ]; then why=soc; else why=""; fi
      [ -n "$why" ] && { kill -STOP $1; st=stop; s0=$(date +%s); echo "pause ($why) ${t}C/${b}C" >> $2; }
    else
      ok=0
      case $why in batt) [ $t -le 41 ] && [ $b -lt 85 ] && ok=1 ;; soc) [ $b -le 65 ] && [ $t -lt 44 ] && ok=1 ;; esac
      [ $ok = 1 ] && { kill -CONT $1; st=run; echo "resume after $(( $(date +%s) - s0 )) s" >> $2; }
    fi
    sleep 1
  done
}
M=models/q235b-q2xl-sparse/Qwen3-235B-A22B-Instruct-2507-UD-Q2_K_XL-00001-of-00002.gguf
BASE="FM_ESTORE_STATS=1 FM_ESTORE_PACK=models/q235b-q2xl.pack FM_ESTORE_POOL_GB=6.2 FM_CPU_ONLY=1 FM_ESTORE_PRED_INT8=tpu/q235b-tpu-chain-102k-pkg FM_ESTORE_PRED_P=0.7 FM_ESTORE_PREFETCH_QD=2 FM_ESTORE_PRED_M=8 FM_ESTORE_PF_BUFFER=64 FM_ESTORE_PRED_TPU=1 FM_FINCH_CPUS=0-3"
for cfg in "$@"; do
  name=${cfg%%:*}; r=${cfg#*:}; x1=${r%%:*}; r=${r#*:}; a78=${r%%:*}; spec=$(echo "${r#*:}" | tr ',' ' ')
  echo 2048000 > $P/policy6/scaling_max_freq; echo 2348000 > $P/policy4/scaling_max_freq
  while [ $(batt) -gt 38 ] || [ $(big) -gt 50 ]; do sleep 10; done
  echo $x1 > $P/policy6/scaling_max_freq; echo $a78 > $P/policy4/scaling_max_freq
  : > /tmp/su-$name.pauses; b0=$(batt)
  env $BASE $spec taskset -c 4-7 build/fm-gen -m $M -f data/wikitext-2-raw/wiki.valid.raw --prompt-n 16 --n ${N:-200} -t 4 -o /tmp/su-$name.ids > /tmp/su-$name.log 2>&1 &
  g=$!; throttle $g /tmp/su-$name.pauses; wait $g
  np=$(grep -c pause /tmp/su-$name.pauses); tp=$(awk '/resume after/ {s += $3} END {print s + 0}' /tmp/su-$name.pauses)
  echo "== $name (X1 $x1, A78 $a78, $spec): batt $b0 -> $(batt) C; $np pauses, $tp s paused"
  grep -E "fm-gen:|^estore: [0-9]" /tmp/su-$name.log
done
