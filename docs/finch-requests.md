# Requests for finch (from frontier-model)

frontier-model runs Qwen3-30B-A3B on the Fold with experts streamed from UFS
([results.md](results.md)). The TPU's job is the **expert-prefetch predictor**:
at layers 0, 4, ..., 44 it reads the router input (2048 int8) and predicts the
experts of the next 4 layers (4 x 128 = 512 int8 logits). That is 12 FC
dispatches per decoded token. On the phone, decode runs at 5.2 tok/s and is
CPU-bound, so moving this off the CPU matters.

Predictor shapes are already chosen to fit finch's limits: int8 symmetric,
per-tensor scales, FC in-width 2048, out-width 512. The application side is ready:
`tools/predictor/train.py --tpu-export`, `tools/tpu/make_predictors.py` (finch
`fc-tflite` + bit-exact int8 test vectors), and `tools/tpu/phone_run.sh` (closed
compiler, `vii-run` check, `bench`).

## Asks, most valuable first

1. **Single-dispatch from-scratch FC, 2048 -> 512, trained int8 weights + scales.**
   `fc-wide --single` is validated only to out <= 32 at in = 1024. Beyond that it
   is out/8 tiles, which means 64 dispatches per predictor and 768 per token, past the
   ~158 registrations per context. With this we no longer need the closed compiler
   for predictors. Even better: a fused 2048 -> 1024 -> 512 chain in one executable
   (our most accurate predictor: recall@8 0.771 vs 0.759).
2. **A C API for applications** (the `finch-ffi` in `docs/ggml-finch-backend.md`):
   open, register an executable once, dispatch with caller-owned input/output
   buffers, and close. Asynchronous submit + wait would let the CPU keep decoding
   while the TPU predicts. Budget: 12 dispatches per token at ~0.24 ms is fine.
3. **Later: weights in a separate, swappable buffer** for a registered
   executable, so the TPU could take expert matmuls during prefill without
   re-registering (prefill touches nearly every expert).

For the TPU kernel-driver owner rather than finch: dma-buf import (PRIME) in the
mainline accel driver, so the router input can reach the TPU from a shared buffer
without a copy. finch only calls it.

**Status (2026-10-01, from the TPU driver owner): built and verified, not yet committed**
(`0001-accel-edgetpu-dma-buf-import.diff`). `DRM_IOCTL_EDGETPU_IMPORT_BO` takes a dma-buf fd and
returns a handle plus the TPU-visible address. The driver maps the dma-buf through the SysMMU
(`dma_buf_map_attachment`), which places scattered pages into one contiguous TPU address range.
Its limits:

* the mapping must come back as a single TPU address range (SysMMU gives that for scattered
  pages, so physical contiguity is **not** required);
* it must fit the TPU's address window (0x80000000-0xfffff000, about 2 GiB, shared with
  firmware and driver allocations; set by the TPU's address map, so treat it as fixed);
* the CPU accesses it through the dma-buf's own mmap (BO read/write is refused for
  imported buffers).

It needs a samsung-iommu off-by-one fix that let a mapping land one page outside the
window. It was verified with a test whose output depends on the input.

How frontier-model will use it: the predictor's input (2 KB int8) and output (512 B)
can be any small dma-buf, e.g. a slice of the store's memfd through udmabuf. The CPU writes
through its own mapping, bracketed by `DMA_BUF_IOCTL_SYNC`.

Optional, later (prefill on the TPU): the expert pools are about 7 GB, larger than the window,
so at most a window-sized slice can be imported at a time. If we pursue it, the ask for the
TPU driver owner is fast import and release of window-sized slices of the pools.

## Contact points in our code

* Predictor training/export: `tools/predictor/train.py` (`int8_fc` is the reference math).
* Where predictions are consumed: `third_party/llama.cpp/src/llama-estore.cpp`
  (`estore_residency_op`, today a CPU matvec).
