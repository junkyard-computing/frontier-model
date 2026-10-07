# finch: replies to finch-requests.md

From the tensor-tpu team, 2026-10-01; updated 2026-10-03. All three asks are done and device-checked on
both of our phones: the AOSP janeiro driver and the mainline accel driver (7.3-rc5). See the end for what
changed on mainline.

## 1. Single-dispatch from-scratch FC with trained weights and scales: done

Our compiler now builds `2048 → 512` (and `in` 1024..4096 → `out` 256 / 512 / 1024) as **one executable**
at any per-tensor scales. You no longer need the closed compiler for predictors.

This was a real bug, not just a missing feature. Our FC used to be correct only at input scale 1/32, and
it refused weight scales other than 1/127. At trained scales it ran and gave wrong outputs. Fixed in
tensor-tpu `de70d5b`. Our executables give **byte-identical outputs** to the closed compiler's on 51 test
FCs (41 scale combinations at 2048 → 512, plus 1024 → 1024 / 512 / 256, 2048 → 1024 and 4096 → 512).

In `tools/tpu/phone_run.sh`, replace the closed-compiler step (the `linker64 comp …` line) with:

```sh
finch compile pred_$l.tflite --out-dir pred_$l.d && cp pred_$l.d/tile_00.bin pred_$l.bin
```

`finch compile` runs offline (host or phone). It writes the FC to `tile_00.bin` plus a `manifest.txt`; an
FC that is one executable has exactly one tile. Everything after that step (`vii-run` against your
`int8_fc` vectors, `bench`) is unchanged.

### 1b. The fused 2048 → 1024 → 512 chain: done, one executable

Shapes 2048 → 1024 → 512 and 4096 → 1024 → 512 (`e0af797`). Our chains give byte-identical outputs to the
closed compiler's (3 models × 2 shapes) at 0.5–0.9 ms per run. Build the graph and compile it:

```sh
finch fc-chain-tflite --dims 2048,1024,512 \
    --weights fc1.w.i8,fc2.w.i8 --bias fc1.b.i32,fc2.b.i32 \
    --act-scales IN,MID,OUT --w-scales W1,W2 pred_$l.tflite
finch compile pred_$l.tflite --out-bin pred_$l.bin
```

The weight and bias files have the same formats as for `fc-tflite`. `MID` is the scale of the 1024-wide
intermediate: fc1's output scale and fc2's input scale. A 2048-wide bottleneck (`--rank 2048`) is not
supported. Our FC programs go up to 1024 outputs per executable, so that would take two dispatches.

## 2. C API: done

`crates/ffi` in tensor-tpu (`dda88b8`). Full contract in `crates/ffi/include/finch.h`; overview in
`docs/c-api.md`; working example in `crates/ffi/examples/smoke.c`.

```sh
nix build .#ffi -o finch-ffi    # in tensor-tpu: aarch64 glibc lib/libfinch_ffi.{a,so}, include/finch.h
gcc -O2 app.c -Ifinch-ffi/include finch-ffi/lib/libfinch_ffi.a -lpthread -ldl -lm
```

| call | |
|---|---|
| `finch_open(opts, &ctx)` | auto-detects mainline or janeiro; needs root for the device node |
| `finch_compile(tflite, len, &exe, &n)` | `.tflite` → one executable in-process (FC or chain) |
| `finch_compile_fc(in, out, w, b, in_s, w_s, out_s, &exe, &n)` | an FC straight from int8 weights and scales |
| `finch_register(ctx, exe, n, in_sizes, n_in, out_sizes, n_out, &id)` | once per executable |
| `finch_dispatch(ctx, id, inputs, outputs)` | synchronous |
| `finch_submit(ctx, id, inputs)` / `finch_wait(ctx, outputs)` | async, up to 16 in flight, collected in order |
| `finch_update(ctx, id, exe, n)` | see 3 |
| `finch_last_error`, `finch_close` | |

Buffers are yours: inputs are copied at submit, so you can reuse them immediately.

**Timing:**
* Back-to-back synchronous runs of a 2048 → 512 FC cost 0.37–0.55 ms.
* A predictor that runs after a long idle period (yours, between decoding steps) pays a TPU wake-up of
  roughly 0.3–0.7 ms. Submit it as early as you can and wait when you need the result: the run overlaps
  your CPU work. On the device, 0.2 ms of CPU work between submit and wait was fully hidden.

The router input is still copied into our buffer. The mainline driver can now import a dma-buf
(`DRM_EDGETPU_IMPORT_BO`, from a udmabuf, a dma-heap buffer or a GPU export), so zero-copy input is
possible on the driver side. finch's C API does not use it yet; ask if you want it.

## 3. Swappable weights: done

`finch_update(ctx, id, exe, n)` replaces a registered executable in place, with no re-registration.

* **Why it works:** the TPU firmware reads the executable from memory on every run. On both phones the
  next run after an update gave exactly a freshly registered executable's outputs, for new weights, new
  scales, and a whole chain.
* **Constraint:** the new executable must have the same layout (same shape and I/O, e.g. the same FC
  shape). Anything else returns `FINCH_EINVAL`. Updates are refused while runs are submitted
  (`FINCH_EBUSY`).
* **Cost:** 1–7 ms for a 1–2.6 MB executable on janeiro, 2.4–2.7 ms on mainline.
* **Expert matmuls:** register one FC per expert shape, then `finch_compile_fc` + `finch_update` per
  expert. The swap costs a few ms against ~0.3–1 ms of compute per FC, so it pays off only when an
  expert's weights are reused across several runs.

## Mainline (2026-10-03)

**Validated on our mainline phone**, outputs byte-identical to janeiro:
* the C API: dispatch, submit / wait and `finch_update` for FC weights, FC scales and a whole chain;
* FC and chain executables.

Timings at the TPU's 1066 MHz: a 2048 → 512 FC runs in 0.48 ms per synchronous run and the 2048 → 1024 → 512
chain in 0.57 ms. Submit / wait hides 0.2 ms of CPU work.

**Driver behaviour to know** (the kernel team's 2026-10-03 driver):
* Buffer objects go up to 256 MiB (4 MiB before), so large executables register.
* If the TPU firmware crashes or locks up, the driver now restarts it itself. Registrations don't survive
  a restart, so that context's runs fail with `FINCH_ERR`, and `finch_last_error` says the firmware
  restarted. Close the context and open a new one, then register again.
* `finch_options.tpu_khz` must stay 0: that driver already runs the TPU at 1066 MHz while a client is
  attached.

**On the Fold:** if it runs the same mainline driver, the quick check is `smoke.c` with a predictor and its
expected output: `sudo ./smoke pred.bin 2048 512 pred.in.i8 pred.expect.i8 0 pred2.bin pred2.expect.i8`. The
last two arguments are a second same-shape executable and its expected output, which exercise
`finch_update`. Please send us its output if anything differs.
