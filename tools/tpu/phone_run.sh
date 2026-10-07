#!/bin/sh
# phone_run.sh DIR [FINCH] [SMOKE] — verify and time the prefetch predictors on the Edge TPU.
#
# DIR: tools/tpu/make_predictors.py output copied to the phone (pred_<l>.bin are already
#      compiled by finch on the host; no closed compiler needed).
# FINCH: the finch CLI (default: finch on PATH). SMOKE: tensor-tpu crates/ffi/examples/smoke
#      built for the phone (optional; the C API check the finch team asked us to send back).
# Needs root for the TPU device node. Run under phase0/guard.sh.
#
# Per predictor: `finch vii-run` on each test vector, compared with the host int8 reference
# (exact and within 1 LSB), then `finch bench` (register once, back-to-back dispatches).
set -e
DIR=$1
FINCH=${2:-finch}
SMOKE=$3
[ -f "$DIR/manifest.txt" ] || { echo "usage: $0 DIR [FINCH] [SMOKE]" >&2; exit 2; }
# mainline kernels expose the TPU as /dev/accel/accel0 (clean-room accel driver)
MAINLINE=; [ -e /dev/accel/accel0 ] && MAINLINE=--mainline

first=; second=
tail -n +2 "$DIR/manifest.txt" | while read -r l ns rest; do
    set -- $rest
    din=$1
    eval "dout=\${$((ns + 1))}"
    n=$(( $(stat -c %s "$DIR/pred_$l.in.i8") / din ))
    exact=0; near=0; i=0
    while [ $i -lt $n ]; do
        dd if="$DIR/pred_$l.in.i8" of=/tmp/fm_in.i8 bs=$din skip=$i count=1 status=none
        sudo "$FINCH" vii-run "$DIR/pred_$l.bin" $MAINLINE --input /tmp/fm_in.i8 --out-file /tmp/fm_out.i8 >/dev/null
        dd if="$DIR/pred_$l.expect.i8" of=/tmp/fm_exp.i8 bs=$dout skip=$i count=1 status=none
        r=$(python3 -c "import numpy as np; a=np.fromfile('/tmp/fm_out.i8',np.int8)[:$dout].astype(int); b=np.fromfile('/tmp/fm_exp.i8',np.int8).astype(int); d=np.abs(a-b); print(int((d==0).all()), int((d<=1).all()))")
        exact=$((exact + ${r% *})); near=$((near + ${r#* })); i=$((i + 1))
    done
    t=$(sudo "$FINCH" bench "$DIR/pred_$l.bin" $MAINLINE --insize $din --outsize $dout --iters 500 2>&1 | grep -oE "[0-9.]+ ms/dispatch" | tail -1)
    echo "layer $l (${ns}-stage $din->$dout): $exact/$n exact, $near/$n within 1 LSB; bench: $t"
done

# the finch team's C API check (finch_open/register/dispatch, then finch_update to the second,
# same-shape predictor under the same input); they asked for its output either way
if [ -n "$SMOKE" ]; then
    set -- $(tail -n +2 "$DIR/manifest.txt" | awk 'NR<=2 {print $1}')
    a=$1; b=$2
    set -- $(tail -n +2 "$DIR/manifest.txt" | head -1)
    ns=$2; din=$3; eval "dout=\${$((ns + 3))}"
    head -c $din "$DIR/pred_$a.in.i8" > /tmp/fm_sa.in
    head -c $dout "$DIR/pred_$a.expect.i8" > /tmp/fm_sa.exp
    head -c $dout "$DIR/pred_$b.expect_on_$a.i8" > /tmp/fm_sb.exp
    echo "== smoke: C API, layer $a predictor"
    sudo "$SMOKE" "$DIR/pred_$a.bin" $din $dout /tmp/fm_sa.in /tmp/fm_sa.exp || true
    echo "== smoke: C API + finch_update to layer $b's predictor"
    sudo "$SMOKE" "$DIR/pred_$a.bin" $din $dout /tmp/fm_sa.in /tmp/fm_sa.exp 0 "$DIR/pred_$b.bin" /tmp/fm_sb.exp || true
fi
