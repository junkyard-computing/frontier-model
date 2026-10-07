# finch on the Fold (mainline): results for the tensor-tpu team

From frontier-model, 2026-10-01, in reply to the "What to check on the Fold" section of
finch-replies.md. Phone: felix, mainline 7.3.0-rc4 (`#1 SMP PREEMPT Thu Oct 1 18:02:01 PDT 2026`),
`/dev/accel/accel0`. finch `0.1.0` from tensor-tpu `f9c6332`, built with `nix build .#default`
(CLI) and `.#ffi` (C API). smoke.c was compiled on the phone with its gcc against libfinch_ffi.a.

## Our predictors: 12 fused chains, 2048 -> 1024 -> 512, compiled by `finch compile` on the host

`finch vii-run --mainline` on 16 held-out router inputs each, compared with our host int8 reference:

* 191 / 192 outputs byte-exact, 192 / 192 within 1 LSB (the one difference is the 16-bit
  requant multiplier you mentioned).
* `finch bench --mainline --insize 2048 --outsize 512 --iters 500`: **0.270 ms per dispatch**
  (3703/s), with or without `--resident`.

## smoke.c (C API) on mainline

```
== smoke: C API, layer 0 predictor
finch 0.1.0 opened
dispatch: 0 of 512 output bytes differ from expected
sync 0.577 ms/run; submit + 0.2 ms CPU + wait 0.291 ms/run; outputs still match
wait with nothing submitted: -4 (finch_wait: nothing submitted)
== smoke: C API + finch_update to layer 4's predictor
finch 0.1.0 opened
dispatch: 0 of 512 output bytes differ from expected
sync 0.567 ms/run; submit + 0.2 ms CPU + wait 0.522 ms/run; outputs still match
finch_update (2.75 ms): 0 of 512 outputs differ from exe2's
wait with nothing submitted: -4 (finch_wait: nothing submitted)
```

The second run uses `pred.bin 2048 512 pred.in.i8 pred.expect.i8 0 pred2.bin pred2.expect.i8`,
where pred2's expected output is computed on pred's input. The C API, `finch_update`
and chain executables all work on mainline as they did on janeiro.
