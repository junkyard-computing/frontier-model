#!/bin/sh
# poll the phone's llama-server every 30 s; log a status line per poll and per finished request; exit (to wake Claude) on trouble
PHONE=${FM_PHONE_SSH:-ssh fold}; LOG=${LOG:-serve-watch.log}; seen=-1; sshfail=0; hfail=0
while :; do
  out=$($PHONE 'cd /userdata/frontier-model
    p=$(pgrep -x llama-server >/dev/null && echo up || echo DOWN)
    t=0; for c in /sys/class/power_supply/maxfg_base/temp /sys/class/power_supply/maxfg_secondary/temp; do v=$(( $(cat $c) / 10 )); [ $v -gt $t ] && t=$v; done
    m=$(awk "/MemAvailable/{print int(\$2/1024)}" /proc/meminfo)
    h=$(curl -s -o /dev/null -m 5 -w %{http_code} http://127.0.0.1:8080/health)
    n=$(grep -c "total time" serve.log)
    echo "$p $t $m $h $n"; grep "        eval time" serve.log | tail -1 | sed "s/.*eval time/LAST eval time/"' 2>/dev/null)
  if [ -z "$out" ]; then sshfail=$((sshfail+1)); echo "$(date +%T) ssh failed ($sshfail)" >> $LOG
    [ $sshfail -ge 4 ] && { echo "ALERT phone unreachable for 2 min"; exit 1; }; sleep 30; continue; fi
  sshfail=0
  set -- $(echo "$out" | head -1); p=$1 t=$2 m=$3 h=$4 n=$5
  echo "$(date +%T) $p batt=${t}C memavail=${m}MB health=$h requests=$n $(cat throttle.log 2>/dev/null | tail -1)" >> $LOG
  if [ "$n" != "$seen" ] && [ $seen -ge 0 ]; then echo "$(date +%T) request done: $(echo "$out" | grep LAST)" >> $LOG; fi; seen=$n
  [ "$p" = DOWN ] && { echo "ALERT server process gone"; $PHONE 'tail -5 /userdata/frontier-model/serve.log'; exit 1; }
  [ "$t" -ge 47 ] && { echo "ALERT battery ${t}C (guard kill threshold)"; exit 1; }
  [ "$m" -lt 400 ] && { echo "ALERT MemAvailable ${m} MB"; exit 1; }
  if [ "$h" != 200 ]; then hfail=$((hfail+1)); [ $hfail -ge 10 ] && { echo "ALERT /health=$h for 5 min"; exit 1; }; else hfail=0; fi
  sleep 30
done
