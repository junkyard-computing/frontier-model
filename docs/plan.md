# Plan

Written 2026-09-30. Numbers are cited to the sibling repos; "est." marks
back-of-envelope figures that measurement should replace.

## Scope

This project owns application work only: the frontier-model runtime and tools, and our llama.cpp patches (ggml-vulkan as a Vulkan application). Kernel, Mesa/panvk, finch-compiler and system-image changes belong to their teams; we design around what they expose and file requests.

## Goal

1. One phone: a >100B-parameter Chinese MoE model (Qwen3-235B-A22B), weights
   streamed from UFS, with CPU, GPU and TPU all doing real work, zero-copy.
2. Later: the same runtime across the k3s cluster (41 phones), where a
   remote phone is just another expert store.

## Per-phone budget

| Resource | Value | Source |
|---|---|---|
| DRAM | 12 GB spec, ~10 GB usable (est.; carveouts + ~1.1 GB CMA pools). **Measure MemTotal.** | junkyard-boot-img dts |
| DRAM bandwidth | ~34 GB/s (clpeak, open 33.9 / closed 34.7) | jboot-mainline `gpu-perf-investigation.md` |
| Flash | UFS 3.1, 2.0 GB/s sequential on mainline; random unmeasured | jboot-mainline `mainline-port-audit.md:112` |
| GPU | G710 MC7: fp16, int8 dot (4x8 packed), no coop-matrix, 32 KB shared, ≤4 GiB/buffer, **non-coherent** | mesa `panvk_vX_physical_device.c` |
| GPU speed (Qwen2.5-0.5B Q4_K_M, vs libmali 161.6–162.4 pp512 / 32.0 tg64) | defaults (`exp/wls-offsets`, no knobs, simple_ondemand): pp512 ≈136, tg64 ≈27.6 (84–86%). Full TEMP knob set + GPU pinned 848 MHz + `MESA_SHADER_CACHE_DISABLE=true`: pp512 ≈159–161 (~99%), tg64 ≈38.6–39.3 (**1.2×**) | GPU team, 2026-09-30; phone `/run/usb-evidence/gpuwork/bench.log` (defaults) |
| TPU | int8 only; FC in-width 1024–4096; FC out-width 64/128 refused; ≤~158 graph regs/context; weights baked into executables; ~0.24 ms/dispatch @1066 MHz | tensor-tpu `RESULTS.md`, `research/compiler-re.md` |
| Thermal | fanless; battery present on dev units — stop runs at ~45 °C battery; stock trips untouched | |

Decode is flash-bound: est. ~0.3–0.4 tok/s for Qwen3-235B at Q2–Q3 (~5–6 GB
of expert reads/token at 2 GB/s, before cache hits).

### GPU benchmarking rules (from the GPU team)

* Always report the Mesa branch, knobs, GPU clock and governor with a number.
* Knob set for best results: `PAN_SSBO_ROUND`/`PAN_SSBO_ALIGN64` (64-byte descriptor rounding, decode 28 -> 36.5),
  `PAN_LOAD_MLP` + `PAN_LOAD_HOIST` (decode -> 38.5), `PAN_TASK_INC1_MIN_THREADS=128` (+ `_BALANCE`) (prefill -> 159),
  `PAN_WLS_WRAP` (prefill -> 161). Pin 848 MHz with `set_freq` (userspace governor; writing devfreq min/max deadlocks).
* An intermittent GPU fault ends runs longer than about 25 minutes, possibly linked to frequency switching. Keep GPU runs
  under ~20 min or pinned.
* Gaps that remain with every knob on: pp2048 ~1.09x (Br=8 flash-attention spills), q6_K matmul 1.10-1.17x.
  Qwen3 Q4_K_M experts mix Q4_K and **Q6_K**, so the q6_K gap hits streamed experts directly.
* Streamed-expert pools already give 64-byte aligned address and size, the property the SSBO knob enforces.

## Division of labour

| Unit | Decode | Prefill |
|---|---|---|
| TPU | expert predictor (int8 low-rank nets, weights baked in — the shape finch supports today) → prefetch lookahead | stretch: expert matmuls via weight-patchable executables |
| GPU | attention, KV cache, resident/shared weights | main compute engine |
| CPU | io_uring O_DIRECT reads, router, cache, expert matmuls on arriving weights | assist; keep flash busy |

Qwen3-235B chosen over DeepSeek because hidden = 4096 matches finch's max FC
input width; experts are 4096→1536→4096.

Predictor shape constraint: finch refuses FC output widths 64/128, so no
rank-128 bottleneck and no bare 128-expert head — use rank ≥256 and predict
several layers per head (k × 128 outputs).

## Zero-copy memory

One hugetlb memfd → udmabuf → a single dma-buf seen by all three:

* io_uring O_DIRECT from UFS writes into it (system dma-heap mmap is PFN-map
  and can't be an O_DIRECT target — to verify).
* GPU imports it via `EXT_external_memory_dma_buf` (panvk supports it).
* TPU imports it via PRIME. That is driver work owned by the TPU/kernel team (request it); AOSP
  janeiro already has `EDGETPU_MAP_DMABUF`). TPU IOVA window ~1.75 GiB.

Kernel: `CONFIG_UDMABUF` is now on in the phone's slot-A kernel (kernel team, 2026-09-30). Access to `/dev/udmabuf` (root:kvm) is the image owners' call.

### Coherency (treat the GPU as non-coherent, permanently)

* The GPU is non-coherent in both the mainline and AOSP DTs. Our DT notes say
  the G710 reaches DRAM on a path that bypasses the CCI, and in June,
  coherent mode gave stale/torn reads. Making it coherent is **not** in the plan.
* This design needs little from coherency: experts are written by UFS DMA,
  never by the CPU. Refilling a slot needs a GPU L2 invalidate (host→shader
  barrier) before first use. CPU-written data (resident load, KB-sized
  activations) is flushed explicitly.
* TPU coherency is configured (SysReg shareability + `IOMMU_CACHE`, no cache
  maintenance) but has never been tested directly. Until a targeted test passes,
  handoff buffers use `DMA_BUF_IOCTL_SYNC`.

### 64-byte alignment (GPU decode fast path)

Every GPU buffer descriptor must be 64-byte aligned in address and size; a
misaligned descriptor falls onto the slow load path (up to 1.6× slower
decode kernel).
* Expert slots are page-aligned; the relaid model file puts each expert
  bundle on a 4 KiB boundary, with gate/up/down 64-byte-aligned and padded
  (stock GGUF only guarantees 32).
* (GPU team's call, not ours) panvk could report `minStorageBufferOffsetAlignment = 64` (graduate
  `PAN_SSBO_ALIGN64`), and all allocations, imports included, get
  padding for size rounding (`PAN_SSBO_ROUND`).
* Assert on misaligned descriptors in the allocator/import path.

## Phases

0. **Measure on the phone.**
   * MemTotal and free space on /userdata.
   * fio at 1–16 MB blocks, QD 1–32, io_uring O_DIRECT.
   * CPU vs panvk baselines.
   * flash→GPU slot-reuse test, with and without the barrier.
   * TPU coherency test, with the shareability-cleared control.
   * `GPU_COHERENCY_FEATURES` readout.
   * aligned vs misaligned imported-buffer benchmark.
   * battery-temperature logger that stops the run at ~45 °C.
1. Qwen3-30B-A3B baseline (CPU, panvk).
2. Qwen3-235B naive mmap baseline.
3. Expert streaming. 3a: train predictors from router traces. 3b: run them on the TPU.
4. CPU/GPU/TPU pipeline with overlap; stretch: TPU prefill experts.
5. Cluster: fleet to mainline, llama.cpp RPC pipeline first, then a remote expert store.

## Off-phone work (status 2026-09-30; see [results.md](results.md))

1. Router traces + predictor training — `tools/trace`, `tools/predictor`. **Done** for Qwen3-30B-A3B.
2. Cache simulator — `tools/sim`. **Done.**
3. GGUF relayout tool (4 KiB bundles, 64 B sub-alignment) — `tools/relayout`. **Done**, verified.
4. Streaming runtime in llama.cpp — `runtime/`, `patches/`. **CPU path done** (bitwise-identical output); Vulkan import and cgroup phone emulation still to do. Emulate the phone with a ~10 GB memory
   cgroup and 2 GB/s io.max on NVMe; test zero-copy on the Intel iGPU
   (UMA, but coherent, so it can't reproduce the Mali coherency problems).
5. Phase-0 test kit — `phase0/`. **Scripts done** (guard, sysinfo, fio, baseline bench); aarch64 cross-build and coherency/alignment tests still to do.
6. Driver-side needs (owned by the kernel, GPU and TPU teams; we only request them): accel PRIME import + mmap, `/dev/udmabuf` access, panvk alignment defaults.

## Far future: architectures that would suit phones (parked 2026-10-06)

Not pursued now (needs training); kept as a record of what tonight's measurements point to:

* **Pre-gated, locality-trained routing:** routers that decide a layer ahead and are trained to reuse experts
  across neighbouring tokens. Post-hoc cache-aware routing already gave +33% decode on an untrained model.
  A small MoE trained with and without a locality penalty would test it.
* **Multi-token prediction heads** (DeepSeek-V3 style): trained acceptance (~85-90%) clears the ~75-80%
  break-even our speculative-decoding simulation found, amortising each expert load over 2-3 tokens.
* **Ternary / ~1.6-bit weights trained from scratch:** fewer flash bytes and cheaper CPU math.
* **Shallower, wider models with parallel attention/FFN blocks:** fewer sequential steps and fewer network
  round trips per token in an expert-parallel cluster.
* **Memory layers** (small reads from a large table in flash) instead of 7 MB experts.

Rough ceiling with all of these: ~5-10 tok/s for one user on one phone, 10-20+ on a wired cluster, for a
~200B-class model. Today's 235B: 0.67 tok/s on one Fold, ~3-5 tok/s estimated for a wired cluster.
