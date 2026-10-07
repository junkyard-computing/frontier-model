#!/bin/sh
# sysinfo.sh — the Phase 0 facts we don't have yet. Read-only.
echo "== kernel";        uname -a
echo "== memory";        grep -E "MemTotal|MemAvailable|HugePages_Total|Hugepagesize|CmaTotal|CmaFree" /proc/meminfo
echo "== thp";           cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null
echo "== storage";       df -h / /userdata 2>/dev/null
for q in /sys/block/sd*/queue; do
    [ -r "$q/scheduler" ] && echo "$q: sched=$(cat $q/scheduler) nr_requests=$(cat $q/nr_requests) max_sectors_kb=$(cat $q/max_sectors_kb) read_ahead_kb=$(cat $q/read_ahead_kb)"
done
echo "== udmabuf / dma-heaps"; ls -l /dev/udmabuf /dev/dma_heap 2>&1
echo "== io_uring";      grep -q io_uring /proc/kallsyms && echo present || echo absent
echo "== battery"
for d in /sys/class/power_supply/*; do
    echo "$d type=$(cat $d/type 2>/dev/null) temp=$(cat $d/temp 2>/dev/null) capacity=$(cat $d/capacity 2>/dev/null) status=$(cat $d/status 2>/dev/null)"
done
echo "== thermal zones"
for z in /sys/class/thermal/thermal_zone*; do echo "$(cat $z/type) $(( $(cat $z/temp) / 1000 ))C"; done
echo "== cpufreq"
for p in /sys/devices/system/cpu/cpufreq/policy*; do
    echo "$p cpus=$(cat $p/related_cpus) gov=$(cat $p/scaling_governor) max=$(cat $p/scaling_max_freq)"
done
echo "== accel / gpu";   ls -l /dev/accel /dev/dri 2>&1
echo "== vulkan";        command -v vulkaninfo >/dev/null && vulkaninfo --summary 2>/dev/null | grep -E "deviceName|driverName|apiVersion"
