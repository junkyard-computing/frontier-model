#!/bin/sh
# root wrapper: cap X1, pin the GPU at 848 MHz (userspace governor, never min/max), run serve.sh under guard.sh
# (battery 47 C) with a pause loop (battery 44/41 C, big cluster 85/65 C); restore everything on exit.
# Stop it with: kill -CONT <llama-server>; kill <llama-server>; kill -9 after ~10 s if it hangs.
ROOT=${FM_ROOT:-/userdata/frontier-model}; SRC=$(cd "$(dirname "$0")/../.." && pwd); cd "$ROOT"
P=/sys/devices/system/cpu/cpufreq; G=$(ls -d /sys/class/devfreq/*gpu* | head -1)
restore() { echo 2850000 > $P/policy6/scaling_max_freq; echo 2348000 > $P/policy4/scaling_max_freq; echo simple_ondemand > $G/governor; }
trap "restore; exit 1" INT TERM; trap restore EXIT
# sustained-decode caps (2026-10-06 benchmark): the big cluster stays under the 85 C pause point
echo 1459000 > $P/policy6/scaling_max_freq; echo 1606000 > $P/policy4/scaling_max_freq
echo userspace > $G/governor && echo 848000000 > $G/userspace/set_freq
(sleep 5; HI=44 LO=41 SOC_HI=85 SOC_LO=65 exec "$SRC"/scripts/phone/serve-throttle.sh) &
"$SRC"/phase0/guard.sh -b 47 -l serve-guard.csv -- "$SRC"/scripts/phone/serve.sh > serve.log 2>&1
echo "server exited $? at $(date +%T)" >> serve.log
