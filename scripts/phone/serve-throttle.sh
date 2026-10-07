#!/bin/sh
# thermal duty-cycle for llama-server: SIGSTOP when the battery reaches HI C or the big cluster SOC_HI C; SIGCONT once the
# sensor that caused the pause has cooled (battery <= LO, big <= SOC_LO) and the other is below its pause point.
# guard.sh still kills at battery 47 C.
HI=${HI:-44}; LO=${LO:-41}; SOC_HI=${SOC_HI:-85}; SOC_LO=${SOC_LO:-65}
big() { for z in /sys/class/thermal/thermal_zone*; do [ "$(cat $z/type)" = big-thermal ] && echo $(( $(cat $z/temp) / 1000 )); done; }
cd "${FM_ROOT:-/userdata/frontier-model}"; st=run; why=""
while pid=$(pgrep -x llama-server); do
  t=0; for c in /sys/class/power_supply/maxfg_base/temp /sys/class/power_supply/maxfg_secondary/temp; do v=$(( $(cat $c) / 10 )); [ $v -gt $t ] && t=$v; done
  b=$(big)
  if [ $st = run ]; then
    if [ $t -ge $HI ]; then why=batt; elif [ $b -ge $SOC_HI ]; then why=soc; else why=""; fi
    [ -n "$why" ] && { kill -STOP $pid; st=stop; echo "$(date +%T) pause ($why) at ${t}C batt / ${b}C big" >> throttle.log; }
  else
    ok=0
    case $why in
      batt) [ $t -le $LO ] && [ $b -lt $SOC_HI ] && ok=1 ;;
      soc)  [ $b -le $SOC_LO ] && [ $t -lt $HI ] && ok=1 ;;
    esac
    [ $ok = 1 ] && { kill -CONT $pid; st=run; echo "$(date +%T) resume at ${t}C batt / ${b}C big" >> throttle.log; }
  fi
  sleep 3
done
