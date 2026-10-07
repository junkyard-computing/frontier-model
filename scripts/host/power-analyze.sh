#!/bin/sh
# integrate battery power (both fuel gauges, -V*I) between the marks in pw-marks.txt; then decode energy per token = (E32 - E8) / 24
MARKS=${1:-pw-marks.txt}; SAMPLES=${2:-pw-samples.txt}   # copied from the phone (/tmp/pw-*.txt)
awk '
  FNR == NR { split($2, a, "-"); key = a[1]; if (a[2] == "start") st[key] = $1; else en[key] = $1; next }
  { t[n] = $1; p[n] = -($2 * $3 + $4 * $5) * 1e-12; n++ }
  END {
    for (k in st) {
      e = 0; tsec = en[k] - st[k]
      for (i = 1; i < n; i++) if (t[i] > st[k] && t[i] <= en[k]) e += (p[i] + p[i-1]) / 2 * (t[i] - t[i-1])
      E[k] = e; T[k] = tsec
      printf "%-6s %6.1f s  %7.1f J  %5.2f W avg\n", k, tsec, e, e / tsec
    }
    if (("cpu32" in E) && ("cpu8" in E)) printf "CPU decode: %.1f J per token (24 tokens: %.1f J in %.1f s)\n", (E["cpu32"] - E["cpu8"]) / 24, E["cpu32"] - E["cpu8"], T["cpu32"] - T["cpu8"]
    if (("gpu32" in E) && ("gpu8" in E)) printf "GPU split decode: %.1f J per token (24 tokens: %.1f J in %.1f s)\n", (E["gpu32"] - E["gpu8"]) / 24, E["gpu32"] - E["gpu8"], T["gpu32"] - T["gpu8"]
  }' "$MARKS" "$SAMPLES"
