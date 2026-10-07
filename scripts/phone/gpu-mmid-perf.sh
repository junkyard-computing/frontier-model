#!/bin/sh
# prompt-sized expert matmuls, GPU vs CPU: 235B MUL_MAT_ID shapes at batch 16 and 128 (root; X1 capped; GPU pinned 848 MHz)
ROOT=${FM_ROOT:-/userdata/frontier-model}; SRC=$(cd "$(dirname "$0")/../.." && pwd); cd "$ROOT"
P=/sys/devices/system/cpu/cpufreq
G=$(ls -d /sys/class/devfreq/*gpu* | head -1)
restore() { echo 2850000 > $P/policy6/scaling_max_freq; echo simple_ondemand > $G/governor; }
trap "restore; exit 1" INT TERM; trap restore EXIT
echo 2048000 > $P/policy6/scaling_max_freq
# pin the GPU clock with the userspace governor; never write min_freq/max_freq (deadlocks this kernel)
echo userspace > $G/governor && echo 848000000 > $G/userspace/set_freq
echo "GPU $(cat $G/cur_freq) Hz, governor $(cat $G/governor); Mesa $(vulkaninfo --summary 2>/dev/null | grep -m1 driverInfo); knobs: none (GGML_VK_SG16_WM_TILES default)"
for bs in 16 128; do
  for b in Vulkan0 CPU; do
    echo "== $b n=$bs"
    taskset -c 4-7 build/bin/test-backend-ops perf -o MUL_MAT_ID -b $b -p "n_mats=128,n_used=8,b=0,m=1536,n=$bs,k=4096" 2>&1 | grep -E "MUL_MAT_ID\(" 
    taskset -c 4-7 build/bin/test-backend-ops perf -o MUL_MAT_ID -b $b -p "n_mats=128,n_used=8,b=0,m=4096,n=$bs,k=1536" 2>&1 | grep -E "MUL_MAT_ID\("
  done
done
