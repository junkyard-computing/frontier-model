#!/bin/sh
# live-routing quality: fm-logits one token per decode step (FM_UBATCH=1), 235B UD-Q2_K_XL, TPU predictor; NAME:ENV,ENV args; logits in /tmp
ROOT=${FM_ROOT:-/userdata/frontier-model}; SRC=$(cd "$(dirname "$0")/../.." && pwd); cd "$ROOT"
P=/sys/devices/system/cpu/cpufreq
restore() { echo 2850000 > $P/policy6/scaling_max_freq; }
trap "restore; exit 1" INT TERM; trap restore EXIT
echo 2048000 > $P/policy6/scaling_max_freq
batt() { t=0; for c in /sys/class/power_supply/maxfg_base/temp /sys/class/power_supply/maxfg_secondary/temp; do v=$(( $(cat $c) / 10 )); [ $v -gt $t ] && t=$v; done; echo $t; }
big() { for z in /sys/class/thermal/thermal_zone*; do [ "$(cat $z/type)" = big-thermal ] && echo $(( $(cat $z/temp) / 1000 )); done; }
# SoC duty cycle: pause the run at >= 85 C on the big cluster, resume at <= 65 C (guard.sh still stops at 95)
throttle() { st=run; while kill -0 $1 2>/dev/null; do b=$(big); if [ $st = run ] && [ $b -ge 85 ]; then kill -STOP $1; st=stop; echo "   pause at ${b} C $(date +%T)"; elif [ $st = stop ] && [ $b -le 65 ]; then kill -CONT $1; st=run; fi; sleep 1; done; }
M=models/q235b-q2xl-sparse/Qwen3-235B-A22B-Instruct-2507-UD-Q2_K_XL-00001-of-00002.gguf
ENVS="FM_ESTORE_STATS=1 FM_ESTORE_PACK=models/q235b-q2xl.pack FM_ESTORE_POOL_GB=6.2 FM_CPU_ONLY=1 FM_NO_REPACK=1 FM_EXPS_CPU=1 FM_UBATCH=1 FM_ESTORE_PRED_INT8=tpu/q235b-tpu-chain-102k-pkg FM_ESTORE_PRED_P=0.7 FM_ESTORE_PREFETCH_QD=2 FM_ESTORE_PRED_M=8 FM_ESTORE_PF_BUFFER=64 FM_ESTORE_PRED_TPU=1 FM_FINCH_CPUS=0-3"
for cfg in "$@"; do
  name=${cfg%%:*}; spec=$(echo "${cfg#*:}" | tr ',' ' ')
  while [ $(batt) -gt 38 ] || [ $(big) -gt 50 ]; do sleep 10; done
  t0=$(date +%s)
  env $ENVS $spec taskset -c 4-7 build/fm-logits -m $M -f data/wikitext-2-raw/wiki.test.raw -o /tmp/ql-$name.f32 --n ${N:-128} -t 4 > /tmp/ql-$name.log 2>&1 &
  g=$!; throttle $g; wait $g
  echo "== $name: $(( $(date +%s) - t0 )) s, batt $(batt) C"
  grep -E "fm-logits:|fm-cache-route|^estore: [0-9]|error|abort" /tmp/ql-$name.log
done
