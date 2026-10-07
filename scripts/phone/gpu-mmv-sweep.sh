#!/bin/sh
# GPU MUL_MAT_ID: mat-vec id kernel vs tiled, by GGML_VK_MMV_ID_MAX_TOKENS, UD-Q2_K_XL expert types at 16 and 128 tokens
# root; X1 capped; GPU pinned 848 MHz via the userspace governor (never min/max); waits for battery <= 39 C and big <= 55 C
ROOT=${FM_ROOT:-/userdata/frontier-model}; SRC=$(cd "$(dirname "$0")/../.." && pwd); cd "$ROOT"
P=/sys/devices/system/cpu/cpufreq
G=$(ls -d /sys/class/devfreq/*gpu* | head -1)
batt() { t=0; for c in /sys/class/power_supply/maxfg_base/temp /sys/class/power_supply/maxfg_secondary/temp; do v=$(( $(cat $c) / 10 )); [ $v -gt $t ] && t=$v; done; echo $t; }
big() { for z in /sys/class/thermal/thermal_zone*; do [ "$(cat $z/type)" = big-thermal ] && echo $(( $(cat $z/temp) / 1000 )); done; }
restore() { echo 2850000 > $P/policy6/scaling_max_freq; echo simple_ondemand > $G/governor; }
trap "restore; exit 1" INT TERM; trap restore EXIT
echo 2048000 > $P/policy6/scaling_max_freq
echo userspace > $G/governor && echo 848000000 > $G/userspace/set_freq
echo "GPU $(cat $G/cur_freq) Hz; Mesa bd0debe9 (system), no Mesa knobs"
for lim in ${LIMITS:-16 32 64 128}; do
  while [ $(batt) -gt 39 ] || [ $(big) -gt 55 ]; do sleep 10; done
  for n in 16 128; do
    echo "== GGML_VK_MMV_ID_MAX_TOKENS=$lim n=$n (batt $(batt) C, big $(big) C)"
    for shape in "type_a=q2_K,type_b=f32,n_mats=128,n_used=8,b=0,m=1536,n=$n,k=4096" "type_a=q3_K,type_b=f32,n_mats=128,n_used=8,b=0,m=4096,n=$n,k=1536"; do
      GGML_VK_MMV_ID_MAX_TOKENS=$lim taskset -c 4-7 build/bin/test-backend-ops perf -o MUL_MAT_ID -b Vulkan0 -p "$shape" 2>&1 | grep -E "MUL_MAT_ID\(" | sed 's/\x1b\[[0-9;]*m//g; s/,amax=1.000000//'
    done
  done
done
