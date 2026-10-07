#!/bin/sh
# prompt processing with the GPU mat-vec id kernel: correctness, then 235B UD-Q2_K_XL 128-token prompt, CPU vs CPU+GPU expert split
# root; X1 capped; GPU pinned 848 MHz via the userspace governor (never min/max); cool to batt <= 39 C and big <= 50 C between runs
ROOT=${FM_ROOT:-/userdata/frontier-model}; SRC=$(cd "$(dirname "$0")/../.." && pwd); cd "$ROOT"
P=/sys/devices/system/cpu/cpufreq
G=$(ls -d /sys/class/devfreq/*gpu* | head -1)
batt() { t=0; for c in /sys/class/power_supply/maxfg_base/temp /sys/class/power_supply/maxfg_secondary/temp; do v=$(( $(cat $c) / 10 )); [ $v -gt $t ] && t=$v; done; echo $t; }
big() { for z in /sys/class/thermal/thermal_zone*; do [ "$(cat $z/type)" = big-thermal ] && echo $(( $(cat $z/temp) / 1000 )); done; }
cool() { while [ $(batt) -gt 39 ] || [ $(big) -gt 50 ]; do sleep 10; done; }
restore() { echo 2850000 > $P/policy6/scaling_max_freq; echo simple_ondemand > $G/governor; }
trap "restore; exit 1" INT TERM; trap restore EXIT
echo 2048000 > $P/policy6/scaling_max_freq
echo userspace > $G/governor && echo 848000000 > $G/userspace/set_freq
export GGML_VK_MMV_ID_MAX_TOKENS=512
[ -n "$SKIPTEST" ] || echo "== correctness: MUL_MAT_ID 128 experts top-8, GPU vs CPU, mat-vec id up to 512 tokens"
[ -n "$SKIPTEST" ] || taskset -c 4-7 build/bin/test-backend-ops test -o MUL_MAT_ID -b Vulkan0 -p "n_mats=128,n_used=8" 2>&1 | grep -E "FAIL|passed|tests passed" | tail -6
M=models/q235b-q2xl-sparse/Qwen3-235B-A22B-Instruct-2507-UD-Q2_K_XL-00001-of-00002.gguf
ES="FM_ESTORE_STATS=1 FM_ESTORE_PACK=models/q235b-q2xl.pack FM_ESTORE_POOL_GB=${POOL:-6.2}"
for cfg in "$@"; do
  name=${cfg%%:*}; spec=$(echo "${cfg#*:}" | tr ',' ' ')
  case " $spec " in *" FM_CPU_ONLY=1 "*|*" FM_NO_HOST=1 "*) ;; *) spec="$spec FM_NO_HOST=1" ;; esac
  cool
  echo "== $name [$spec] (batt $(batt) C, big $(big) C)"
  env $ES $spec taskset -c 4-7 build/fm-gen -m $M -f data/wikitext-2-raw/wiki.valid.raw --prompt-n ${PN:-128} --n ${NG:-4} -t 4 -o /tmp/pp-$name.ids > /tmp/pp-$name.log 2>&1
  grep -E "fm-gen:|^estore: [0-9]|split experts|imported|abort|DeviceLost|error" /tmp/pp-$name.log | grep -v set_abort | head -5
  echo "   after: batt $(batt) C, big $(big) C"
done
