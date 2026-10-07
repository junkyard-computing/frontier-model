#!/bin/sh
# fio-experts.sh PACK [OUTDIR] — UFS read throughput for expert-shaped reads.
#
# Reads the real expert pack (pack_experts.py output) with io_uring + O_DIRECT:
#   * random reads of whole bundles (bs = bundle stride, bundle-aligned)
#   * random 1 MiB / 128 KiB reads (sub-expert granularity)
#   * sequential 16 MiB reads (the ceiling)
# at queue depths 1, 4, 16, 32. Needs fio (apt install fio). Read-only.
# Run under guard.sh.
set -e
PACK=$1
OUT=${2:-fio-$(date +%Y%m%d-%H%M%S)}
[ -r "$PACK" ] && [ -r "$PACK.json" ] || { echo "usage: $0 PACK [OUTDIR] (needs PACK.json)" >&2; exit 2; }
command -v fio >/dev/null || { echo "fio not installed" >&2; exit 2; }
mkdir -p "$OUT"

STRIDE=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['layers'][0]['bundle_stride'])" "$PACK.json")
SIZE=$(stat -c %s "$PACK")
echo "pack $PACK: $SIZE bytes, bundle stride $STRIDE"

run() {  # name rw bs align qd
    fio --name="$1-qd$5" --filename="$PACK" --readonly --rw="$2" --bs="$3" --blockalign="$4" \
        --ioengine=io_uring --direct=1 --iodepth="$5" --size="$SIZE" \
        --time_based --runtime=20 --ramp_time=2 --norandommap --randrepeat=0 \
        --output-format=json --output="$OUT/$1-qd$5.json"
    python3 - "$OUT/$1-qd$5.json" <<'EOF'
import json, sys
j = json.load(open(sys.argv[1]))["jobs"][0]["read"]
print(f"{sys.argv[1].rsplit('/', 1)[-1]:28s} {j['bw_bytes'] / 1e9:6.3f} GB/s  "
      f"{j['iops']:9.1f} IOPS  p50 {j['clat_ns']['percentile']['50.000000'] / 1e3:8.0f} us  "
      f"p99 {j['clat_ns']['percentile']['99.000000'] / 1e3:8.0f} us")
EOF
}

for qd in 1 4 16 32; do
    run bundle   randread "$STRIDE" "$STRIDE" $qd
    run rand1m   randread 1M        4k        $qd
    run rand128k randread 128k      4k        $qd
    run seq16m   read     16M       4k        $qd
done | tee "$OUT/summary.txt"
