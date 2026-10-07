# Expert-streaming runtime: design

Status: store, llama.cpp integration and CPU predictor prefetch working on the x86 host; verified bitwise. Numbers: [results.md](results.md). See [plan.md](plan.md) for the why.

## Data layout

**Flash — the expert pack** (`tools/relayout/pack_experts.py`). One bundle per
(layer, expert), `[gate | up | down]`. Each bundle starts on a 4 KiB boundary
and each projection on a 64 B boundary. Bytes are verbatim from the GGUF.
Fetching an expert is one read.

**Memory — three packed slot pools per layer group**, one each for gate, up
and down. Each pool is `[ne00, ne01, n_slots]`, packed exactly like a stock
`ffn_*_exps` tensor, so the unmodified CPU and Vulkan `MUL_MAT_ID` kernels
consume it directly.

* CPU `mul_mat_id` indexes experts by `nb02` and would accept a strided
  view of a bundle pool. Vulkan would not: it sizes the binding and the
  expert stride from element counts (`ggml_vk_mul_mat_id_q_f16`,
  `ggml_vk_mul_mat_vec_id_q_f16`: `qx_sz = type_size * nelements / blck`).
  Hence packed per-projection pools.
* A bundle is fetched with **one `preadv`** whose three iovecs point at its
  slots in the three pools (flash stays one contiguous read; memory is
  three packed tensors). O_DIRECT requires each iovec length to be a
  multiple of 4 KiB. The pack index sets `scatter_ok` when that holds
  (Qwen3-30B-A3B Q4_K/Q6_K: gate/up 884,736 B = 216 × 4 KiB, down
  1,290,240 B = 315 × 4 KiB). Where it doesn't hold, fall back to a bounce
  read of the tail.
* Slot offsets are `slot * proj_size`, so 64 B alignment holds whenever
  `proj_size % 64 == 0`. The pack index records this; the runtime asserts it.

**Pools are udmabuf-backed.** A hugetlb memfd is exported through udmabuf.
The CPU mmaps it, io_uring O_DIRECT reads into it, Vulkan imports it
(`EXT_external_memory_dma_buf`), and later finch imports it (PRIME). One
allocation, no copies.

## Graph integration (llama.cpp)

In `build_moe_ffn`:

```
selected = argsort_top_k(router(x))           # as today
slots    = expert_residency(selected, il)      # NEW: CPU custom op
out      = mul_mat_id(pool_{gate,up,down}[il], x, slots)   # pools replace *_exps
```

`expert_residency` is a CPU op. It reads the ids, looks each (layer, expert)
up in the cache, issues/awaits reads for misses (evicting via the policy,
never evicting this layer's working set), and outputs slot indices. On the
GPU it forces a graph split at every MoE layer. That split is inherent:
nothing can fetch an expert before the router has chosen it. Prefetch
hides the wait.

Resident tensors (attention, norms, router, shared experts, embeddings,
output) come from the original GGUF as today. Its `*_exps` tensors are
never touched, so their mmap costs no RAM.

## Components

| Component | Where | Status |
|---|---|---|
| Expert pack + index | `tools/relayout` | done; verified byte-identical (Qwen3-30B-A3B, 6144 bundles, 0% padding) |
| Expert store: udmabuf pools, io_uring scatter reads, per-group LRU, capped prefetch queue (demand priority) | `runtime/estore.{h,c}` | done; memfd→udmabuf→O_DIRECT path works (host udmabuf caps at 64 MB; mainline default is INT_MAX) |
| Replay harness (real I/O, modelled compute) | `runtime/replay.c` → `fm-replay` | done |
| `expert_residency` op + pool tensors in llama.cpp | `patches/llama.cpp-estore.patch` (`FM_ESTORE_PACK=...`) | done, CPU; logits **bitwise identical** to stock with repacking off (64-token prefill, 3 GB pool, eviction exercised) |
| Predictor hook → prefetch queue | llama.cpp estore op (`FM_ESTORE_PRED`, `_P`, `_M`, `FM_ESTORE_PREFETCH_QD`) | done on CPU; greedy output identical; +80% tok/s in real-I/O replay ([results.md](results.md)) |
| Predictor on the TPU (finch executables for the int8 2048->1024->512 nets) | tensor-tpu + estore op | next |
| Vulkan: import the pool dma-bufs as ggml buffers (`FM_ESTORE_VK=<dev>`) | llama.cpp ggml-vulkan (app side: `ggml_backend_vk_buffer_from_dmabuf`) | done; on the Intel iGPU (ANV) logits are bitwise identical between 16- and 48-slot pools; vs CPU: argmax 94%, KL 0.024, in line with GPU numerics (stock GPU attention alone gives KL 0.013) |
| GPU L2 invalidate on slot refill (non-coherent G710) | runtime + Vulkan barrier | phone only |

### Findings so far

* **Vulkan import (2026-10-01).** Each pool is exported as its own udmabuf over its memfd
  range, because Vulkan imports a dma-buf from offset 0 and the GPU has a per-buffer size limit.
  ANV can import but refuses `vkMapMemory` on imported dma-bufs, so the import treats the buffer as
  device-only when mapping fails; the CPU already has the memfd mapping. Pool tensors live in the
  imported buffers, so ggml's scheduler runs `MUL_MAT_ID` on the GPU. The residency op stays on the
  CPU, which moves ids and slots across the split. On the phone, check the non-coherent G710 path
  (the GPU must see DMA-written slots) and the 64-byte alignment fast path.
* **Phone CPU baseline (2026-09-30, mainline 7.3-rc4 #5, CPUs 4-7, X1 capped 2.05 GHz).** Streamed:
  5.17 tok/s decode; naive mmap 3.70; streamed + predictor 4.71. Only ~2 s of ~19 s were I/O
  stall, so the phone is compute-bound and the next gain is moving compute to the GPU/TPU.
* **Phone memory.** With the Vulkan backend built in, CPU weights go to the GPU's pinned host
  buffers (the whole model in RAM) and get OOM-killed. CPU-only runs need no GPU devices
  (`FM_CPU_ONLY`), and GPU runs need `-ot exps=CPU` (`FM_EXPS_CPU`) so the model's experts stay mmapped.

* **Layer groups.** Q4_K_M mixes quant types per layer (24 layers with Q6_K `down`, 24 with Q4_K).
  So pools are per shape group; each group caches the same fraction of its experts.
* **CPU repack.** On x86, llama.cpp repacks Q4_K weights at load into interleaved layouts with their
  own kernels. Pool tensors are raw GGUF layout, so they take the generic kernel: the math is the
  same, but accumulation differs (KL 0.005 vs the repacked stock run; bitwise equal to stock with
  `use_extra_bufts=false`). ARM has repacked kernels too. For speed on the phone, either repack
  on load (CPU cost per miss) or store a pre-repacked layout in the pack; measure first.
* **Queue depth.** Demand misses come about one per layer, so they are issued at QD 1. The replay
  on this host ran 0.55 of the time stalled, against the simulator's 0.32, because QD-1 3 MB reads
  on dm-crypt NVMe under load reach only 0.34 GB/s. Prefetch matters for keeping the queue deep,
  not only for hiding latency. Phase 0 fio at QD 1 sets the phone number.

## Open questions

* Tune `prefetch_qd` on UFS (Phase 0 fio at QD 1-4 decides it).
* Train predictors for miss recall (weight experts that are rarely cached).
* Pool sizing per layer: uniform, or weighted by miss rate from traces?
* Quant types with sizes that are not a multiple of 4 KiB (IQ2/IQ3 for
  DeepSeek-class models) need the bounce-tail path.
