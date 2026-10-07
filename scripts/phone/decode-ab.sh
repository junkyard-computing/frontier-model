#!/bin/sh
# read-only UFS DMA A/B: 235B UD-Q2_K_XL decode (NAME:ENV,ENV args), logs in /tmp only; X1 capped; reports system CPU and swiotlb
ROOT=${FM_ROOT:-/userdata/frontier-model}; SRC=$(cd "$(dirname "$0")/../.." && pwd); cd "$ROOT"
P=/sys/devices/system/cpu/cpufreq
restore() { echo 2850000 > $P/policy6/scaling_max_freq; echo simple_ondemand > $(ls -d /sys/class/devfreq/*gpu* | head -1)/governor; }
trap "restore; exit 1" INT TERM; trap restore EXIT
echo 2048000 > $P/policy6/scaling_max_freq
batt() { t=0; for c in /sys/class/power_supply/maxfg_base/temp /sys/class/power_supply/maxfg_secondary/temp; do v=$(( $(cat $c) / 10 )); [ $v -gt $t ] && t=$v; done; echo $t; }
M=models/q235b-q2xl-sparse/Qwen3-235B-A22B-Instruct-2507-UD-Q2_K_XL-00001-of-00002.gguf
ES="FM_ESTORE_STATS=1 FM_ESTORE_PACK=models/q235b-q2xl.pack FM_ESTORE_POOL_GB=6.2"
PRED="FM_ESTORE_PRED_INT8=tpu/q235b-tpu-chain-102k-pkg FM_ESTORE_PRED_P=0.7 FM_ESTORE_PREFETCH_QD=2 FM_ESTORE_PRED_M=8 FM_ESTORE_PF_BUFFER=64"
SW=$(ls /sys/kernel/debug/swiotlb/io_tlb_used 2>/dev/null)
G=$(ls -d /sys/class/devfreq/*gpu* | head -1)
for cfg in "$@"; do
  name=${cfg%%:*}; spec=$(echo "${cfg#*:}" | tr ',' ' ' | sed "s|PRED|$PRED|"); n=${N:-32}; pn=${PN:-16}
  while [ $(batt) -gt 39 ]; do sleep 10; done
  case " $spec " in *" FM_CPU_ONLY=1 "*|*" FM_NO_HOST=1 "*) ;; *) echo "SKIP $name: neither FM_CPU_ONLY nor FM_NO_HOST"; continue ;; esac
  # GPU runs: pin 848 MHz via the userspace governor (never min/max), back to simple_ondemand after
  case " $spec " in *" FM_ESTORE_SPLIT="*) echo userspace > $G/governor && echo 848000000 > $G/userspace/set_freq ;; esac
  s0=$(head -1 /proc/stat); w0=$(date +%s)
  env $ES $spec taskset -c 4-7 build/fm-gen -m $M -f data/wikitext-2-raw/wiki.valid.raw --prompt-n $pn --n $n -t 4 -o /tmp/ab-$name.ids > /tmp/ab-$name.log 2>&1
  s1=$(head -1 /proc/stat); w1=$(date +%s)
  echo simple_ondemand > $G/governor
  echo "== $name ($(date +%T), batt $(batt) C, $((w1 - w0)) s wall) swiotlb_used=$( [ -n "$SW" ] && cat $SW || echo n/a)"
  echo "$s0
$s1" | awk 'NR==1{for(i=2;i<=NF;i++)a[i]=$i} NR==2{t=0;for(i=2;i<=NF;i++){d[i]=$i-a[i];t+=d[i]} printf "   cpu: user %.1f%% sys %.1f%% iowait %.1f%% irq+soft %.1f%% idle %.1f%%\n",100*d[2]/t,100*d[4]/t,100*d[6]/t,100*(d[7]+d[8])/t,100*d[5]/t}'
  grep -E "fm-gen:|^estore: [0-9]" /tmp/ab-$name.log
done
