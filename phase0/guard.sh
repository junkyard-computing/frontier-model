#!/bin/sh
# guard.sh — run a benchmark under a battery/SoC temperature guard.
#
#   guard.sh [-b MAX_BATT_C] [-s MAX_SOC_C] [-l LOG.csv] -- command args...
#
# Polls every second; logs battery + every thermal zone to CSV; kills the
# command's process group and exits 3 if the battery reaches MAX_BATT_C
# (default 50) or any SoC zone reaches MAX_SOC_C (default 95). Stock thermal
# trips are never touched.
#
# Fail-safe: if a battery is present but its temperature is unreadable
# (e.g. mainline without the max1720x fix), refuse to start unless
# GUARD_NO_BATT_TEMP_OK=1 is set.

MAX_BATT=50   # approved 2026-09-30 (battery idles ~43 C on the charger)
MAX_SOC=95
LOG=guard-$(date +%Y%m%d-%H%M%S).csv
while [ $# -gt 0 ]; do
    case "$1" in
        -b) MAX_BATT=$2; shift 2 ;;
        -s) MAX_SOC=$2; shift 2 ;;
        -l) LOG=$2; shift 2 ;;
        --) shift; break ;;
        *) echo "usage: $0 [-b C] [-s C] [-l log] -- cmd..." >&2; exit 2 ;;
    esac
done
[ $# -gt 0 ] || { echo "guard: no command" >&2; exit 2; }

# every battery cell (the Fold has two fuel gauges: maxfg_base, maxfg_secondary)
batt_dirs=
for d in /sys/class/power_supply/*; do
    [ "$(cat "$d/type" 2>/dev/null)" = Battery ] && batt_dirs="$batt_dirs $d"
done
batt_dir=${batt_dirs# }

# hottest cell in whole degrees C (power_supply temp is in tenths of a degree);
# fails if any cell is unreadable
batt_temp() {
    [ -n "$batt_dirs" ] || return 1
    max=
    for d in $batt_dirs; do
        t=$(cat "$d/temp" 2>/dev/null) || return 1
        [ -n "$t" ] || return 1
        t=$((t / 10))
        { [ -z "$max" ] || [ "$t" -gt "$max" ]; } && max=$t
    done
    echo "$max"
}

if [ -n "$batt_dir" ]; then
    if ! batt_temp >/dev/null; then
        if [ "${GUARD_NO_BATT_TEMP_OK:-0}" != 1 ]; then
            echo "guard: battery at$batt_dirs but temperature unreadable; refusing (set GUARD_NO_BATT_TEMP_OK=1 to override)" >&2
            exit 4
        fi
        echo "guard: WARNING battery temperature unreadable; guarding SoC zones only" >&2
    fi
else
    echo "guard: no battery found; guarding SoC zones only" >&2
fi

zones=$(ls -d /sys/class/thermal/thermal_zone* 2>/dev/null)
{
    printf 'time,batt_c'
    for z in $zones; do printf ',%s' "$(cat "$z/type")"; done
    printf '\n'
} > "$LOG"

setsid "$@" &
pid=$!
trap 'kill -TERM -$pid 2>/dev/null; exit 130' INT TERM

status=0
while kill -0 $pid 2>/dev/null; do
    b=$(batt_temp 2>/dev/null || echo NA)
    line="$(date +%s),$b"
    hot=
    for z in $zones; do
        t=$(( $(cat "$z/temp" 2>/dev/null || echo 0) / 1000 ))
        line="$line,$t"
        [ "$t" -ge "$MAX_SOC" ] && hot="$(cat "$z/type")=${t}C"
    done
    echo "$line" >> "$LOG"
    if [ "$b" != NA ] && [ "$b" -ge "$MAX_BATT" ]; then hot="battery=${b}C"; fi
    if [ -n "$hot" ]; then
        echo "guard: $hot over limit; stopping benchmark" >&2
        kill -TERM -$pid 2>/dev/null; sleep 2; kill -KILL -$pid 2>/dev/null
        status=3
        break
    fi
    sleep 1
done
wait $pid 2>/dev/null
rc=$?
[ $status -ne 0 ] && exit $status
exit $rc
