# Results (off-phone, 2026-09-30)

Model: Qwen3-30B-A3B Q4_K_M (48 layers, 128 experts, top-8, 3.06 MB/expert,
17.6 GB of experts, 1.0 GB resident). Trace: 20,480 wikitext-2 test tokens
(`traces/q30b-wiki`). Predictors are trained on the first 16,384 tokens; all
prefetch numbers are on the held-out last 4,096.

**Host numbers are not phone numbers.** The replay ran on an i7-11800H with
dm-crypt NVMe under heavy unrelated load. The simulator uses phone
parameters (2 GB/s flash, 25 GB/s DRAM, 9 GB RAM budget), which Phase 0 must
calibrate.

## Correctness

| Check | Result |
|---|---|
| Expert pack vs GGUF | 6,144 bundles byte-identical, 0% padding |
| Logits: stock vs streamed (64-token prefill, 3 GB pool, eviction exercised) | **bitwise identical** with `use_extra_bufts=false` in both; with x86 Q4_K repacking in stock only: KL 0.005, argmax agreement 98.4% (different kernels) |
| Greedy decode, 32 tokens: stock / streamed / streamed + smoke predictor / streamed + trained predictor | identical token ids |

## Routing locality

* Expert reuse between consecutive tokens: 49% (fraction of a token's
  experts, at a given layer, used by the previous token).
* Belady (optimal) eviction reads 2.7x less flash than LRU at a 7 GB cache.
  This is the headroom that prediction and eviction policy can win back.

## Cache policy (simulator, phone parameters, 16,384 tokens)

| Policy | 9 GB RAM (7 GB cache, 40% of experts) | 7 GB RAM (5 GB cache, 28%) |
|---|---|---|
| LRU | 92.4% hit, 83 MB/token, 7.9 tok/s | 83.4%, 184 MB/token, 5.7 tok/s |
| pinned-hot + LRU | 91.4%, 95 MB/token, 7.6 tok/s | 81.8%, 202 MB/token, 5.4 tok/s |
| Belady (bound) | 97.2%, 31 MB/token, 10.1 tok/s | 93.2%, 75 MB/token, 8.2 tok/s |

A static hot set loses to LRU: expert popularity drifts across the corpus.

## Predictors (span 4: layer l predicts layers l+1..l+4; one per 4 layers)

| | recall@8 | p>0.5 prec/recall | p>0.7 prec/recall | size (int8) |
|---|---|---|---|---|
| previous token's experts (baseline) | 0.488 | | | |
| single FC 2048->512 (float) | 0.759 | 0.876 / 0.597 | 0.944 / 0.437 | 1.0 MB each |
| FC 2048->1024->512, int8 QAT (finch-compilable) | **0.771** | 0.849 / 0.673 | 0.917 / 0.554 | 2.6 MB each, 31.5 MB total |

Caveat: the predictor is best at popular experts, which are usually cached
already. The misses left under LRU are the hard-to-predict experts. Train
for miss recall next (weight rarely-cached experts).

## Prefetch

Simulator (held-out tokens, 7 GB cache):

| Prefetch | FIFO queue | demand priority |
|---|---|---|
| none | 8.26 tok/s | 8.26 |
| top-8 (single FC) | 8.21 | 9.10 (102 MB/token) |
| int8 predictor, p>0.5 | | **9.14** (86 MB/token) |
| int8 predictor, p>0.7 | | 9.00 (79 MB/token) |

Real I/O (`fm-replay`, host, 1,200 held-out tokens, 7 GB cache):

| Prefetch | tok/s | stall | flash MB/token | prefetches used |
|---|---|---|---|---|
| none | 3.65 | 0.61 | 86 | - |
| top-8, uncapped | 4.06 | 0.49 | 107 | 88% |
| top-8, <=2 prefetches in flight | 5.42 | 0.40 | 104 | 88% |
| **int8 predictor, p>0.7, <=2 in flight** | **6.55 (+80%)** | 0.38 | 89 | 96% |

Takeaways:

1. **Demand reads must never queue behind prefetches.** The kernel can't
   reorder an io_uring queue, so estore caps prefetches in flight
   (`prefetch_qd`, default 2) and holds the rest in a software queue,
   dropping stale entries. On real I/O this is worth more than the
   predictor itself.
2. **Threshold on confidence, not top-k.** p>0.7 gives the best real
   speed with the least flash traffic, so less power and heat on the phone.
3. The winning predictor is already the finch/TPU shape: int8,
   FC in-widths 2048 and 1024, out-widths 1024 and 512.

## TPU predictor (int8, finch-ready), 2026-10-01

Single FC 2048->512 per predictor layer (12 per token), symmetric int8 with
per-tensor scales and a static input scale, as `finch fc-tflite` builds it.
Packaged by `tools/tpu/make_predictors.py` into `traces/q30b-tpu-pkg`. `finch plan`
reports all 12 as FULLY_CONNECTED and lowerable; each `.tflite` is about 1 MB.

| | recall@8 | p>0.7 precision / recall |
|---|---|---|
| float single FC (earlier) | 0.759 | 0.944 / 0.437 |
| **int8, bit-exact int8 arithmetic** | **0.758** | **0.944 / 0.435** |

* Quantization costs nothing measurable.
* The C++ int8 loop in llama-estore (`FM_ESTORE_PRED_INT8`) matches the Python reference
  on 98,304 of 98,304 test outputs. Greedy decode with it is identical to stock.
* Simulator (phone parameters, demand priority): 8.76 tok/s vs 8.26 with no prefetch (+6%).
  The rank-1024 float predictor gives 9.00; a fused 2048->1024->512 finch executable would
  close the gap (finch-requests ask 1).
* Real-I/O replay on a heavily loaded host (load ~30; absolute numbers not comparable
  with the earlier table): none 2.23 tok/s; int8 TPU predictor 3.66 (+64%, 98% of
  prefetches used); rank-1024 float 3.01 (+35%). The difference between the two predictors
  is within this host's run-to-run noise. Both are clearly better than no prefetch.

Not yet run on the TPU: the phone script `tools/tpu/phone_run.sh` compiles each predictor
with the closed compiler, checks `vii-run` output against the int8 reference (exact and
within 1 LSB), and times `finch bench`. It needs tensor-tpu's staged tree on the phone.

### Fused chain on finch (2026-10-01, after finch-replies.md)

finch now compiles our predictors itself (`finch compile`, no closed compiler), including the
fused 2048 -> 1024 -> 512 chain as one executable.

| int8 predictor | recall@8 | p>0.7 precision / recall | sim (phone params, demand priority) |
|---|---|---|---|
| single FC 2048->512 | 0.758 | 0.944 / 0.435 | 8.76 tok/s |
| **fused chain 2048->1024->512** | **0.774** | 0.921 / 0.556 | **9.00 tok/s** (= the float rank-1024 predictor) |

* All 12 chains compile to one 2.6 MB executable each; the C++ int8 path matches the Python
  reference on 98,304 / 98,304 outputs.
* llama.cpp: `FM_ESTORE_PRED_INT8=<pkg>` runs them on the CPU with TPU arithmetic;
  adding `FM_ESTORE_PRED_TPU=1` (build with `-DLLAMA_FM_FINCH_DIR`) runs them on the TPU through
  the finch C API. It submits at the predictor's layer and collects at the next layer's
  residency op. Linked and checked here up to `finch_open` (no TPU on this host).
* Phone checklist (`tools/tpu/phone_run.sh`): vii-run vs the int8 reference, `bench`, and the
  finch team's `smoke` C-API check, plain and with `finch_update` (they asked for its output).

## Qwen3-235B-A22B-Instruct-2507 Q3_K_M (the target), 2026-10-01

Model: 94 layers x 128 experts, top-8, n_embd 4096. Experts 108.1 GB (8.95-9.73 MB each: gate/up
Q3_K, down Q4_K, or Q5_K in 5 layers), resident 4.3 GB. Expert data per token: 752 experts, ~6.8 GB.
Download SHA-256 verified; expert pack 108.1 GB, 0% padding, scatter_ok, 12,032 bundles verified.
Trace: 12,288 wikitext tokens (2048-token windows), recorded through estore itself (O_DIRECT
streaming, 543 GB read) in a 16 GB / no-swap cgroup.

### Phone projection (simulator: 10.9 GB RAM, 1.9 GB/s UFS, CPU compute 13 GB/s effective)

Cache = 10.9 - 1.0 reserve - 4.3 resident = 5.6 GB = 5.2% of the experts.

| Eviction | hit rate | flash MB/token | tok/s |
|---|---|---|---|
| global LRU | 0% | 6,759 | 0.22 |
| pinned popular + LRU | 24% | 5,123 | 0.28 |
| **estore LAYER (default)** | **41%** | 3,985 | **0.33** |
| Belady (optimal bound) | 60% | 2,680 | 0.43 |
| LAYER + int8 chain predictor (p>0.7) | 34% + 14% prefetched | 5,284 | 0.31 |

* **Global LRU fails completely.** A token walks the 94 layers in order and its working set
  (6.8 GB) exceeds the cache, so LRU always evicts what the next token needs first.
* **LAYER policy:** each layer keeps an equal share (6 experts here, after a 16-slot reserve),
  inserts in reverse router order (top-ranked newest), and is trimmed back to its share once
  computed. Real store, real I/O: 43.2% hits on the same 60 tokens where the simulator says 42.0%.
  A first version that took overflow slots from the globally oldest experts gave 12%: in a cyclic
  walk those belong to the next layers.
* **The predictor does not help at this scale.** recall@8 is ~0.45-0.65 per layer, about the same
  as the previous token's experts; there are only ~9.8k training tokens for 4.7M-parameter
  4096-wide predictors. In 6-slot shares, wrong prefetches evict hits.
* **What would move 235B on one phone**, in order: more RAM for the cache (every GB is ~1% of the
  experts); eviction closer to Belady (60%); fewer bits per expert (Q2_K is about 25% fewer bytes
  than Q3_K_M); then the cluster. Faster compute helps little: 72% of the time is flash stall.

### Store fixes found along the way (all in runtime/estore and the llama.cpp patch)

* **In-flight cap.** Reads in flight could exceed the CQ ring (2 x qd) during big prefills, so
  the kernel dropped completions (EBADR). Now at most qd reads are in flight.
* **Minimum slots per layer group.** Every group gets at least its widest layer (prefill can need
  all 128 experts of one layer at once).
* **No repacking or prefetch of streamed experts.** With `FM_ESTORE_PACK` set, the tools disable CPU
  weight repacking (it copied all 108 GB into RAM, causing an OOM), and the patch marks `*_exps`
  lazy, so llama.cpp no longer prefetches their file ranges at load (load read 1 GB instead of
  58+ GB). Both are bitwise identical to stock on the 30B model.

## Phone session 2026-10-01 (evening): TPU and GPU on the Fold

Kernel 7.3.0-rc5-00170 (`#1 SMP PREEMPT Thu Oct 1 18:02:01`); Vulkan: system Mesa 26.3.0-devel
`bd0debe9d5` (the GPU team's fixed build, now the default), no knobs, simple_ondemand. CPU runs:
CPUs 4-7, 4 threads, X1 capped at 2.05 GHz. Battery stayed at or below 39 C.

**TPU (finch 0.1.0, mainline accel driver).** Our 12 fused-chain predictors: 191/192 byte-exact vs
the int8 reference, 192/192 within 1 LSB; 0.270 ms per dispatch. C API (`smoke.c`): sync 0.57 ms,
submit/0.2 ms CPU/wait 0.29-0.52 ms; `finch_update` 2.75 ms with correct outputs. Sent to the
tensor-tpu team as docs/finch-mainline-results.md.

**30B decode (32-token prompt + 64 generated, 7 GB pool):**

| | prefill | decode | correct |
|---|---|---|---|
| **CPU only** | 4.80 tok/s | **4.78 tok/s** | yes |
| GPU everything, experts zero-copy (dma-buf), slots cleaned | 1.06 | 3.45 | yes: KL 0.024 vs CPU, argmax 93.8% |
| GPU attention + CPU experts | 3.35 | 1.70 | yes (tokens == CPU) |

* **Coherency bug, found and fixed.** Without cleaning, GPU-expert runs were wrong and
  nondeterministic (KL 1-3 between identical runs, differing from position 0). UFS DMA writes can
  sit in CPU-side caches, but the G710 reads DRAM directly. estore now cleans each landed slot to
  DRAM (`DC CVAC`, automatic with `FM_ESTORE_VK`). Afterwards: bitwise identical across a 0.8 GB and
  a 7 GB pool and across repeats. An earlier "4.85 tok/s" GPU figure came from the broken path and is
  withdrawn.
* Earlier device-lost faults (INSTR_BARRIER_FAULT) came from the old `/opt/mesa-g710` build; the
  current default driver runs clean (GPU team's diagnosis; no faults in these runs).
* Scheduler fixes needed for GPU runs: the pool buffers are flagged `WEIGHTS`, and op offload is
  off when streaming. Without them ggml tried to copy the 7 GB pool to the GPU (OOM; VM_BIND err=12).
* For 30B on this phone the CPU is fastest. GPU-everything pays one CPU<->GPU handoff per MoE layer
  for the residency op (the CPU must see the router's choice to fetch from flash), plus the slot
  cleaning, and its prefill is slow.

## All three units on the Fold (2026-10-02)

**235B, GPU everything:** does not fit. `PANTHOR_VM_BIND` ENOMEM even with a 5 GB pool: panvk has no
host-pointer import, so ggml-vulkan copies the 4.3 GB of resident weights into GPU buffers on top
of the pool. Making room means shrinking the expert cache, which costs more than GPU compute could
save on a flash-bound model. **235B CPU:** 0.29 tok/s decode at a 5 GB pool (0.27 at 6 GB earlier).

**30B with the TPU predictor through llama.cpp (finch C API, async):** works end to end, with tokens
identical to no predictor and to the CPU predictor. Decode from a cooled start (battery 36 C, big 45 C):

| | decode |
|---|---|
| no predictor | 4.39 tok/s |
| predictor on CPU (TPU arithmetic) | 4.31 / 4.27 |
| predictor on TPU (finch, async) | 4.15 |

The predictor raises hits (80.8% to ~83-84%) and cuts I/O stall, but 30B decode on the phone is
compute-bound, so there is nothing to win. The TPU path's per-token submit/collect, plus finch's worker
thread sharing the 4 pinned cores, make it slightly slower. The TPU predictor pays off only on a
flash-bound model (235B) with a predictor that beats previous-token reuse (needs far more 235B
training data).

Fixes found:
* The mainline TPU driver backs each buffer with physically contiguous pages (`alloc_pages(order)`),
  so registering 2.6 MB executables failed with ENOMEM once the 7 GB pool was allocated.
  llama-estore now registers the predictors before allocating the pools. (For the TPU driver
  owner: buffers could use scattered pages through the SysMMU.)
* Benchmarks must cool on SoC temperature too: big cores at 88 C throttled runs to ~30-60% speed.
  On the charger the battery idles at 37-38 C, so a battery-only cool-down target below that never ends.

### finch worker pinned to the little cores (2026-10-02)

llama-estore now calls `finch_open` from a helper thread pinned to `FM_FINCH_CPUS` (default `0-3`,
the A55s), so finch's worker thread inherits that mask and doesn't share the decode cores. 30B decode,
cooled start, two rounds in alternating order (repeats agree within 0.02 tok/s):

| | decode | vs none |
|---|---|---|
| no predictor | 4.41 / 4.40 | - |
| TPU predictor, worker on cores 4-7 | 4.17 / 4.17 | -5.4% |
| **TPU predictor, worker on cores 0-3** | **4.34 / 4.32** | **-1.7%** |

Thread masks during the runs confirm the worker on 0-3. The remaining 1.7% is per-token work on the
main thread (quantize, submit, collect) plus extra flash reads for wrong guesses. 30B decode is
compute-bound on the phone, so the predictor has nothing to win back here.

## 235B predictor on 102k tokens, prefetch buffer (2026-10-03)

**Predictor.** Retrained the 24 int8 chains (span 4, rank 1024, QAT) on a 102,400-token 235B trace
(traces/q235b-train-all, recorded in two parts and merged), evaluated on the separate 12,288-token
wiki trace. Mean recall@8 is **0.801** against **0.500** for reusing the previous token's experts (it
was ~0.45-0.65 with 9.8k training tokens). At p > 0.7: precision 0.911, recall 0.654. Packaged with
`finch compile` (24 chains, 4.7 MB each); the CPU fallback is bit-exact with the Python int8
reference (tools/tpu/int8check: 0 / 196,608 outputs differ).

**Prefetch buffer** (`FM_ESTORE_PF_BUFFER`, `--pf-buffer`): prefetched experts go into a separate
slot pool instead of the layer shares, so a wrong guess doesn't evict a hit; a used one moves into
its layer's share. Greedy 30B decode with predictor + 64-slot buffer gives the same tokens as no prefetch.

Simulator, phone parameters (11.67 GB, 0.6 GB reserve, 1.9 GB/s, prefetch QD 2 with demand priority):

| | tok/s |
|---|---|
| LAYER, no prefetch | 0.351 |
| prefetch into the shares, top-4, p > 0.7 | 0.384 |
| **64-slot buffer, top-8, p > 0.7, full span** | **0.438 (+25%)** |
| Belady bound | 0.47 |

Host replay with real I/O (fm-replay, O_DIRECT from the pack, 6.8 GB pool, 50 measured wiki tokens):

| | tok/s | hits | in-flight hits | notes |
|---|---|---|---|---|
| no prefetch | 0.751 | 48.4% | - | |
| shares, top-4 | 0.884 | 53.3% | 11.1% | 98% of prefetches used |
| buffer 64, top-8 | 0.861 | 51.4% | 18.1% | 612 prefetches dropped (queue) |

The simulator gives 0.359 / 0.394 / 0.462 on the same tokens. On real I/O both prefetch modes give
+15-18%, and the buffer doesn't beat the shares there. The host's flash is faster (2.6-3.1 GB/s), and
with prefetch QD 2 the buffer's larger top-8 set gets dropped from the queue before it is issued.

### On the Fold (2026-10-03): 235B prefetch with the 102k predictor

CPU-only decode, 6 GB pool, prefetch QD 2, X1 capped at 2.05 GHz, cooled start (battery ≤39 C, big ≤50 C),
16-token prompt, 32 decoded tokens, two rounds in alternating order. The predictors ran on the CPU (int8,
bit-identical to the TPU executables): each 235B chain is 4,743,168 bytes and the edgetpu driver then
capped CREATE_BO at 4 MiB (contiguous BOs). The kernel team lifted the cap the same day; TPU runs below.
All prefetch runs produced the same tokens as no prefetch.

| | decode (s) | tok/s | hits | flash stall | GB read |
|---|---|---|---|---|---|
| no prefetch | 123.6 / 105.7 | 0.26 / 0.30 | 35.1% | 20.7 s | 163.6 |
| shares, top-4 | 98.3 / 98.3 | 0.33 | 52.0% | 16.5 s | 165.6 |
| buffer 64, top-8 | 96.7 / 96.7 | 0.33 | 56.0% | 14.4 s | 166.2 |

* Prefetch gives **+7.5% (shares) / +9.3% (buffer)** against the second no-prefetch run (the first was
  slower with the same stall, most likely the resident weights still paging in after the phone sat idle).
  95.5-95.8% of prefetches are used; 50 (shares) and ~555 (buffer) are dropped as stale from the queue.
* ~~The phone is not flash-bound; flash stall is only ~20% of decode.~~ **Wrong, corrected the same
  evening (see "Flash I/O on the Fold" below):** the stall counter only timed waits after submission,
  and on this UFS most of the wait happens *inside* `io_uring_submit` (the block queue holds fewer than
  two expert reads). Counting both, 235B decode without prefetch is mostly flash-bound.

**With the TPU** (same afternoon, after the edgetpu update; all 24 predictors register; same settings,
finch worker on cores 0-3; every run gave the same tokens as no prefetch):

| | decode (s) | tok/s | hits | flash stall | prefetches |
|---|---|---|---|---|---|
| no prefetch | 105.7 / 105.8 | 0.30 | 35.1% | 20.8 s | - |
| TPU async, shares top-4 | 106.4 / 103.5 | 0.30 | 48.3% | 17.9 s | 3,900 |
| TPU async, buffer 64 top-8 | 101.4 / 101.4 | 0.32 | 52.8% | 15.4 s | 5,290 |
| **TPU sync, shares top-4** | 98.1 / 97.6 | 0.33 | 51.9% | 17.3 s | 4,920 |
| **TPU sync, buffer 64 top-8** | **96.3 / 96.0** | **0.33** | 55.6% | 15.3 s | 6,025 |

* **Sync mode (`FM_ESTORE_TPU_SYNC=1`) is the right one for 235B: +9.1% over no prefetch with the buffer**,
  the same as the CPU predictor (96.7 s). The 24 blocking TPU calls per token cost nothing measurable
  against ~3 s/token.
* **Async mode loses a layer of lead time.** It submits layer l's prediction at l's residency op and
  collects it at l+1's, by which point layer l+1's prefetches are skipped as already acquired: 1 of the
  4 span layers is lost (3,900 / 4,950 = 0.79 of the prefetches), and the other three start one layer
  later. Async was built for 30B, where decode is fast and the predictor is on the critical path; a
  collector thread that issues the prefetches as soon as the TPU finishes would give both.
* What the TPU buys on 235B is moving the predictor off the decode cores, which at this model size is
  small: the CPU predictor is equally fast. (An earlier version of this bullet said decode is mostly CPU
  compute; it is mostly flash, see below.)

## Flash I/O on the Fold (2026-10-03 evening)

**The TPU collector thread** (async predictions issue their prefetches as soon as the TPU returns) first
made decode *slower*: 118-119 s against 96-98 s for sync. Lock timers showed the collector holding the
store lock for 53.6 s of the run and the decode thread waiting 42.9 s for it; the TPU itself took 0.67 s
over ~1,000 predictions (~0.7 ms each).

**Why: submitting a read blocks.** A submit timer in estore: **3.1 ms per expert read inside
`io_uring_submit`** (fm-replay, 69% of wall time). The UFS queue explains it: `max_sectors_kb` 256 splits
a 9 MB expert read into ~36 requests, `nr_requests` 64 holds fewer than two reads, and every further
submit sleeps until requests drain. So "submit time" is flash wait under another name, and the collector
was sleeping in submit while holding the lock. With submit time counted, no-prefetch decode spends
~84-92 s of its ~105-116 s waiting on flash (163.6 GB read at ~2 GB/s, the link's limit: HS-G4 x2).

What didn't help (235B, 32 tokens, no prefetch):
* **2 MiB hugetlb pool:** submit 3.01 vs 3.13 ms/read. The UFS host caps DMA segments at 4 KiB
  (`max_segment_size` 4096), so page size doesn't change the I/O path. No visible compute gain either.
* **UFS IRQ on the little cores (1-3) instead of 4-5:** decode unchanged, prefill 11% slower (38.0 vs
  34.3 s). The image's placement on 4-5 stays.
* **`nr_requests` 512:** decode unchanged; the same ~92 s of flash wait moves from submit to stall.

What helps: prefetch with the I/O path fixed. `FM_ESTORE_ASYNC_SUBMIT=1` (IOSQE_ASYNC: io_uring's
workers wait for block requests, submit drops to ~0.6 s per run) plus `FM_ESTORE_PF_IDLE=1` (prefetch
reads in the idle I/O class so mq-deadline dispatches demand reads first), with `nr_requests` 512:

| (hot room, X1 capped at 2.05 GHz) | decode (s) | tok/s | vs no prefetch |
|---|---|---|---|
| no prefetch, defaults (end of the batch) | 105.4 | 0.30 | - |
| **TPU sync, buffer 64 top-8, async submit, idle prefetch** | **86.9** | **0.37** | **+21%** |
| TPU collector, same | 88.5 | 0.36 | +19% |

The collector now holds the store lock for 0.04 s in all, and matches sync. Every run gave the same
tokens. Caveats: the room was hot (mid cluster thermally throttled, cooling state 3/16), those two runs
also had the UFS IRQ on 1-3, and decode times drifted between runs (115.9 s for the first baseline,
105.4 s for the last).

**Follow-up, UFS IRQ on 4-5 throughout** (same evening, same tokens in every run, defaults restored):

| | decode (s) | tok/s | vs defaults |
|---|---|---|---|
| no prefetch, defaults | 105.5 | 0.30 | - |
| no prefetch, async submit | 108.2 | 0.30 | -2.6% |
| **TPU sync prefetch, async submit, idle prefetch, `nr_requests` 64** | **87.4** | **0.37** | **+21%** |
| same, `nr_requests` 512 | 87.1 | 0.37 | +21% |

* **The gain is all app-side.** `nr_requests` 512 adds nothing over the default 64, so the kernel/image
  needs no change for it. Async submit costs ~3% without prefetch and is what makes prefetch (and the
  collector) work, so llama-estore now turns on `FM_ESTORE_ASYNC_SUBMIT` and `FM_ESTORE_PF_IDLE` by
  default (`=0` disables either). Without them, prefetch gave +9%.
* The 124-131 s runs in the first batch combined async submit with the UFS IRQ on the little cores (and
  the heat); with the IRQ on 4-5, async submit alone is near neutral.
* The collector matches sync once submits don't block (88.5 vs 86.9 s in the first batch), but doesn't
  beat it; see the 30B check below for which mode is the default.

**30B, collector vs sync** (2026-10-04 morning, cool room, X1 capped, two rounds alternating, identical
tokens; async submit and idle prefetch on):

| | round 1 | round 2 | vs none |
|---|---|---|---|
| no prefetch | 4.37 | 4.38 | - |
| TPU sync | 4.33 | 4.32 | -1.1% |
| TPU collector | 4.29 | 4.29 | -1.9% |

Sync is slightly faster than the collector on both models (30B: 0.8%, 235B: 1.8%) and simpler, so it
is now the default TPU mode; `FM_ESTORE_TPU_ASYNC=1` selects the collector. 30B decode stays
compute-bound, so the predictor still costs it 1-2%.

**Incident:** during this work the phone panicked out of memory and rebooted (15:57). My test script
had omitted `FM_CPU_ONLY`/`FM_ESTORE_PACK`, so fm-gen loaded 235B onto Vulkan and panthor pinned memory,
while 6.6 GB of huge pages were also reserved. Scripts now refuse to start a 235B run without both, and
huge pages are reserved only around the run that uses them. Separately, after that reboot
`ufs-irq-affinity.service` reported success but left the UFS IRQ on CPU 0 (its script discards write
errors).

## Expert matmul: CPU vs GPU on the Fold (2026-10-03 night)

`test-backend-ops perf -o MUL_MAT_ID` with 235B's expert shapes added to the perf cases (128 experts,
top-8): gate/up [4096 -> 1536] in Q3_K, down [1536 -> 4096] in Q4_K/Q5_K. CPU: 4 threads pinned to cores
4-7, X1 capped at 2.05 GHz. GPU: Vulkan0 (G710, panvk, Mesa 26.3.0-devel bd0debe9d5, system default,
no Mesa/panvk env knobs), devfreq simple_ondemand, 848 MHz during the run. Hot room; under guard.sh.
Time per call = one layer's projection for 8 experts, decode (batch 1):

| | CPU | GPU |
|---|---|---|
| gate/up Q3_K (4096 -> 1536) | 2.50 ms | 2.42 ms |
| same shape in Q4_K | 1.43 ms | 1.47 ms |
| down Q4_K (1536 -> 4096) | 1.54 ms | 1.61 ms |
| down Q5_K | 2.00 ms | 2.24 ms |

* **GPU = CPU at decode.** Moving experts to the GPU alone wouldn't speed decode; splitting each layer's
  experts between CPU and GPU concurrently could nearly halve expert-matmul time if memory bandwidth
  holds.
* **Expert matmuls are ~0.6 s of a ~3 s decode token** (2 x 2.50 + 1.54 ms per layer x 94), so that split
  is worth at most ~10%. Flash is still the bigger lever.
* **Q3_K is expensive to compute:** at the same shape it takes 1.75x Q4_K's time on the CPU (and 1.6x on
  the GPU) despite 25% fewer bytes; its kernels are compute-bound (~8.6 GB/s of weights vs ~20 for Q4_K).
  Re-quantizing gate/up to Q4_K would cut that compute but add ~30% to flash bytes, a bad trade while
  decode is flash-bound.
* Batch 16/128 (prefill) not measured: the phone stopped cooling below 45 C at idle in the heat.

The first attempt (21:05) ran without guard.sh or the X1 cap and the phone hard-reset with no kernel
log. Cause unexplained: the kernel team first inferred a sub-PMIC UVLO on VSYS, then retracted it (2026-10-05)
because the same PMIC OFFSRC values latch on a panic reboot. No kernel record rules out a panic; UVLO vs a
hardware watchdog/hard lockup is open. Capping policy6 at 2.05 GHz stays as a precaution; every later run used it.

## GPU batched MUL_MAT_ID fixed; per-layer cache shares don't help (2026-10-04)

**Batched expert matmul on the G710 (ggml-vulkan, app-side).** Diagnosed independently by the kernel
instance on the device and by me from the source: batches of more than 8 tokens take the tiled
integer-dot kernel, whose small-tile config is wrong on subgroup-16 GPUs (`s_warptile_wm` = 32 with
WARP 16: 2 warps share one 32x32 warp tile, the second reads past the shared-memory tile and the
accumulator count doubles, so 280 spills). Fixes in our ggml-vulkan: `WM = 32 / (BLOCK_SIZE / WARP)` for
the small tiles on subgroup-16 devices without coopmat, and an opt-in `GGML_VK_MMV_ID_MAX_TOKENS`
to use the mat-vec id kernel for small token counts. Batch 16, 128 experts top-8, X1 capped
(libggml-vulkan.so sha256 f73d3c3cada971f6):

| | original | WM fix | + mat-vec id up to 16 tokens | CPU |
|---|---|---|---|---|
| q3_K gate/up | 1217 ms | 321 ms | **37.2 ms** | 39.5 ms |
| q4_K gate/up | 808 ms | 107 ms | **22.9 ms** | 24.2 ms |
| q4_K down | 745 ms | 105 ms | **25.0 ms** | 24.2 ms |
| q5_K down | - | 158 ms | **34.4 ms** | 33.2 ms |

Correctness with the WM fix, GPU vs CPU: MUL_MAT_ID q3_K/q4_K/q5_K 90/90, MUL_MAT q3_K/q4_K/q5_K/f16
342/342. The GPU now matches the CPU at batch 1 and batch 16, which a CPU+GPU split needs.

**Per-layer cache shares: no gain (negative result).** `tools/sim/policy_lab.py` first claimed fitted
per-layer shares would lift hits from 49.7% to 58.5%. That was a modelling bug: the reuse distance gave
a token's 8 experts the same timestamp, so a 1-slot share was credited with holding the whole previous
token. On the phone fitted shares were worse (235B, 48 tokens: 28.7% hits / 118.5 s vs 38.3% / 103.0 s
equal), and cachesim with estore's LAYER semantics agreed (steady state 33.9% vs 46.2%). With the fix:
oracle shares 49.6% vs equal 49.4%, and even per-layer Belady eviction reaches only 49.7%. At 760 slots a
layer's share (~8) is one token's top-8, so any per-layer policy reduces to reusing the previous token's
experts. The global Belady bound (~60%) needs moving memory between layers with future knowledge, which
is what prefetch approximates. The estore `FM_ESTORE_SHARES` / `--shares` support stays (default equal).
Lesson: validate an analytic model against cachesim before spending phone time.

## CPU+GPU on the Fold, night of 2026-10-05 (30B decode, 64 tokens, X1 capped)

**Expert split** (`FM_ESTORE_SPLIT=k`: per token, the first k of the 8 experts on the GPU via the zero-copy
Vulkan view of the estore pools, the rest on the CPU, outputs concatenated in router order;
`FM_SCHED_OVERLAP` lets the CPU chain run while the GPU chain computes; `FM_NO_HOST` keeps CPU weights out
of pinned memory). CPU-only: 4.36-4.44 tok/s throughout.

| | decode tok/s |
|---|---|
| split k=4, first version (CPU chain waited for the GPU) | 1.29 |
| + concat pinned to the GPU (CPU chain overlaps) | 1.85 |
| + router pinned to the CPU | 1.53 (noise or slightly worse) |
| + GPU clock pinned at 848 MHz | **2.44-2.45** (overlapped and serialized alike) |

* The GPU kernels themselves are slow at 30B's expert size: a 4-expert mat-vec (768x2048 Q4_K) takes ~0.9
  ms (~14 GFLOPS) under the default `simple_ondemand` governor, which leaves the G710 near 202 MHz under
  bursty per-layer work. Pinning 848 MHz (governor `userspace` + `userspace/set_freq`; never min_freq,
  see below) was worth +60%.
* The rest is CPU<->GPU handoff cost, ~1 ms each, about three per layer. A 30B layer is ~4.7 ms in all
  (experts ~2 ms), so per-layer splitting cannot win for 30B; overlap made no difference once the clock
  was pinned. For 235B (experts ~6.5 ms/layer) the same arithmetic leaves at most a few percent.
* **All on the GPU** (only the estore residency op on the CPU, one round trip per layer): 2.76-2.85 tok/s
  at default clocks. It device-lost (`CS_BUS_FAULT`) with my subgroup-16 WM change applied to all small
  tiles; with upstream WM (`GGML_VK_SG16_WM_OLD=1`) it runs clean. The WM change is now limited to the
  MUL_MAT_ID quant tiles (verified 90/90, and where the batch-16 gain was) pending the kernel team's
  check on the pre-regression Mesa.
* **q6_K mat-vec with exactly 5 columns faults the G710** (`CS_BUS_FAULT`; 4 and 6-8 columns fine). The
  kernel team reproduced it with stock llama.cpp and traced it to a regression in their experimental Mesa
  compiler commits (`/opt/mesa-next` passes); bisecting. App-side workaround: q6_K n==5 takes the tiled
  matmul on ARM (`GGML_VK_Q6K_N5_WORKAROUND=0` disables it).
* **Incident:** writing the GPU devfreq `min_freq` deadlocks this kernel (devfreq PM-QoS -> panthor ->
  interconnect -> nested dev_pm_qos); my script did, and the hung-task panic rebooted the phone (01:21).
  It came back by itself.

Conclusion so far: on gs201 the G710 decodes at or below CPU speed, and fine-grained CPU/GPU splitting
loses to ~1 ms handoffs. The GPU is a real gain only for work that is batched or coarse-grained.

## Fewer experts per token on 235B (2026-10-05)

Routing each token to fewer than the model's 8 experts (`FM_EXPERT_USED=k`, a GGUF KV override of
`qwen3moe.expert_used_count`; top-k of the router, weights renormalized as usual) cuts expert bytes
directly. Quality on the host (`fm-logits` + `tools/logits/compare_logits.py`, 512 tokens of wikitext-2
test, vs top-8); speed on the Fold (TPU prefetch, 64-slot buffer, X1 capped, same batch, warm page cache):

| experts per token | perplexity | mean KL vs top-8 | top-1 agree | decode (32 tok) | tok/s |
|---|---|---|---|---|---|
| 8 (model default) | 4.283 | - | - | 85.8 s | 0.37 |
| 7 | 4.325 (+1.0%) | 0.029 | 91.7% | 73.7 s | 0.43 (+16%) |
| 6 | 4.438 (+3.6%) | 0.070 | 90.3% | 63.0 s | 0.51 (+36%) |

The speedup beats the byte cut (12% / 25%) at top-6 because the smaller working set also hits the cache
more (54.6% -> 58.7% -> 62.8%). Chris chose to try top-7. Caveat: the first run after a rebuild is cold
(the build evicts the resident weights from the page cache) and reads 98 s for top-8; the table uses the
warm repeat.

**Subgroup-16 WM change, per tile family** (`GGML_VK_SG16_WM_TILES` bitmask, all-on-GPU 30B decode): only the
`mmq_int_k` family (int-dot K-quant plain matmul) device-losts with WM=16; s_warptile, mmq, mmq_int, the fp
id tile and the three mmqid families run clean (2.84-2.93 tok/s). Upstream WM on the plain tiles gives
wrong results on Mali (MUL_MAT q6_K 12/13, q5_K 28/29), so the plan is WM fix on every family except
mmq_int_k, and K-quant plain matmuls off the int-dot tiled path. The kernel team confirmed this fault also
happens on the pre-regression Mesa (not their bug).

**Kernel team's independent check (2026-10-05, test-backend-ops in their chroot, fixed Mesa):** confirmed as
a llama.cpp bug, not a GPU bug: on subgroup 16 the small int-dot tiles put two warps on one 32x32 warp tile,
so warp 1 writes 32 columns past the tile. With WM = min(32, 32*WARP/BLOCK) (the same thing `s_wm` computes)
the plain tiles pass: q4_1 29/30 (NaN) -> 30/30, q5_K 28/29 -> 29/29, q4_K 46/47 -> 47/47, q3_K and q2_K
12/13 -> 13/13, with no device-lost on any of them, including the mmq_int_k family. So the mmq_int_k WM=16
fault in my all-GPU 30B run is not reproduced in isolation and is probably the q6_K mat-vec bug instead.
Worth upstreaming as `s_warptile_wm = min(32, 32*WARP/BLOCK)` for the int tiles. Still to test on 30B:
`GGML_VK_SG16_WM_TILES=255` (every family) on bd0debe9 with the q6_K n==5 workaround in place.

Second, pre-existing PanVK bug (theirs, open): q6_K mat-vec n=7 (m=16, k=256) device-losts 3/3 on
`/opt/mesa-next` and on their fixed build (TRANSLATION_FAULT), and only passes on bd0debe9 because that
build's misaligned-load merging hides it, so their n=5 fix currently trades n=5 for n=7. Until it is fixed:
stay on bd0debe9 and keep the app-side q6_K n==5 workaround.

## Smaller experts: UD-Q2_K_XL on the Fold (2026-10-05 evening)

unsloth Qwen3-235B-A22B-Instruct-2507 **UD-Q2_K_XL** (88.8 GB, SHA-256 verified): gate/up Q2_K everywhere,
down Q3_K in 79 layers and Q4_K in 15 (read from the GGUF header with `tools/relayout/gguf_remote_types.py`
before downloading). Experts 6.97 MB mean (was 8.99), always-loaded weights 4.95 GB (was 4.30), so the
estore pool was set to 6.2 GB (6.9 before) to keep total RAM the same. Pack 83.8 GB, 12,032 bundles
verified byte-identical. Why Q2_K: on the phone CPU it is also *faster* to compute than Q3_K (8 experts,
1536x4096: 1.83 vs 2.79 ms), unlike the IQ2/IQ3 formats (2.5-5.2 ms).

Quality on the host (512 tokens wikitext-2 test, vs Q3_K_M top-8): UD-Q2_K_XL top-8 ppl 4.283 -> 4.369
(+2.0%), mean KL 0.214, top-1 agree 83.0%; top-7 4.522 (+5.6%), KL 0.239, 84.8%.

Phone (TPU prefetch, 64-slot buffer, X1 capped, warm page cache, 32 tokens):

| config | ppl vs Q3_K_M top-8 | decode | tok/s | flash read | hits |
|---|---|---|---|---|---|
| Q3_K_M top-8 | - | 85.8 s | 0.37 | 162.7 GB | 54.6% |
| Q3_K_M top-7 | +1.0% | 73.7 s | 0.43 | 140.0 GB | 58.7% |
| **UD-Q2_K_XL top-8** | **+2.0%** | **66.9 s** | **0.48 (+28%)** | 115.6 GB | 59.2% |
| Q3_K_M top-6 | +3.6% | 63.0 s | 0.51 | 119.0 GB | 62.8% |
| UD-Q2_K_XL top-7 | +5.6% | 59.4 s | 0.54 (+44%) | 101.3 GB | 61.7% |

UD-Q2_K_XL without prefetch: 78.1 s (0.41 tok/s, vs 103.9 s for Q3_K_M), already faster than Q3_K_M with
prefetch. Each row trades quality for speed; UD-Q2_K_XL top-8 is the best of the trades measured. The
phone now holds only the UD-Q2_K_XL pack (the Q3_K_M pack is kept on the host).

## Web UI on the Fold, and what limits it (2026-10-05 night)

`llama-server` (built with `-DFM_LLAMA_SERVER=ON -DLLAMA_USE_PREBUILT_UI=ON`) served UD-Q2_K_XL top-8 on
:8080 for a test session: `-c 2048 -np 1 -ctk q8_0 -ctv q8_0 -t 4`, pool 5.8 GB, CPU-only, run under
guard.sh at 47 C and a pause loop (SIGSTOP at 44 C, SIGCONT at 41 C). A chat with the TPU predictor:
212-token prompt 384 s (0.55 tok/s, including a 167 s thermal pause), 41-token reply 94 s (0.43 tok/s).
Two things limit it, neither of them flash:

* **Prompt processing is CPU compute.** CPU expert matmuls cost the same per token at batch 16 as at batch 1
  (q3_K gate/up 39.5 ms for 16 tokens vs 2.50 ms for 1, from the table above), so a 212-token prompt costs
  ~212 decode tokens of compute, while its flash traffic is bounded by the whole pack (84 GB, ~47 s at
  1.8 GB/s). Sequential layer streaming would not help much; a faster batched matmul (the GPU at batch
  >= 32) would.
* **Heat.** The battery climbs ~1 C per 30 s while generating; the session lost ~40% of wall-clock to
  pauses, and an earlier server was killed by guard.sh at 47 C because the battery kept heating for a few
  seconds after a pause at 46 C. Stopping a SIGSTOPped server also hangs its shutdown (resume it first).

Prompt cache: llama-server already reuses the KV of the longest common prefix in its slot
(`cache_prompt`), so follow-up turns and new chats with the same system prompt skip it; the 1 cached token
seen was the first request after a restart.

## Speculative decoding: simulated, not worth it here (2026-10-05 night)

`tools/sim/spec_overlap.py` on the Q3_K_M wikitext trace (4096 tokens, per-layer LRU of 9 slots, the
phone's share). A verify pass over K drafts loads the union of the K+1 positions' experts once per layer.
Adjacent tokens do share experts (distinct experts over 4 tokens: 18.1 of 32), but the per-layer cache
already captures most of that, and a rejected draft is a different token whose experts are mostly wasted
reads (modelled with a random position's routing). Flash misses per generated token, relative to plain
decode (355.5):

| drafts K | accept 0.3 | 0.5 | 0.7 | 0.9 |
|---|---|---|---|---|
| 1 | 1.55x | 1.27x | 1.06x | 0.88x |
| 2 | 1.96x | 1.43x | 1.10x | 0.79x |
| 4 | 2.39x | 1.70x | 1.15x | 0.70x |
| 6 | 2.63x | 1.84x | 1.14x | 0.63x |

Break-even needs per-draft acceptance around 0.75-0.8; n-gram drafts on prose accept far less and a
0.6B draft model would also take ~0.6 GB from the pool. (Counting a rejected draft's routing as the true
next token's instead, which wrongly turns it into a perfect prefetch, gives 0.80-0.41x: that bound is not
reachable.) Dropped unless a workload with very predictable text (code edits, quoting) shows up.

## CPU caps vs heat (2026-10-05 night, partial)

UD-Q2_K_XL top-8, TPU predictor, -t 4 on cores 4-7, 16-token prompt + 96 decoded tokens, each run started
at battery <= 38 C, charger reporting Full (no charge current). Hotter cell temperature:

| caps (X1 / A78) | decode tok/s | battery | rise | rate |
|---|---|---|---|---|
| 1.10 / 1.20 GHz | 0.37 | 35.8 -> 41.1 C in 312 s | 5.3 C | 1.0 C/min |
| 1.46 / 1.61 GHz | 0.43 | 37.9 -> 42.9 C in 266 s | 5.0 C | 1.1 C/min |
| 2.05 / 2.35 GHz | not run (phone handed to the kernel team for the 64-bit UFS DMA test) | | | |

Heat per decoded token is about the same at both caps (~0.05 C/token) for the battery, but see "Sustained
decode" below: lower caps keep the big cluster under the 85 C pause point, which makes them the best sustained setting. Decode is not purely flash-bound
at these clocks (0.37 -> 0.43 tok/s with +33% clock). The top-cap run is still needed to compare; the
earlier web-UI session at full caps heated ~1.2 C/min while generating.

## Cache-aware routing (2026-10-05 night, host simulation)

`FM_CACHE_ROUTE=gamma` (llama-graph.cpp, a map_custom1 op on the router probabilities before top-k):
experts already in the cache get their selection score multiplied by (1 + gamma), so a cached expert
takes the place of an uncached one whose probability is less than (1 + gamma) times its own. Expert
weights still come from the unbiased probabilities (renormalised over the chosen 8, as Qwen3 does).
Host mode simulates the phone's cache: a per-layer LRU of 9 slots (`FM_CACHE_ROUTE_SLOTS`), walking each
batch's tokens in order. Because the LRU depends only on its own layer's history, one batched pass gives
exactly the cache state token-by-token decode would, so a 512-token logits run measures both quality and
misses. With `FM_ESTORE_PACK` set it asks the real store instead (`estore_resident`: resident or in
flight, prefetches included); not yet run on the phone.

UD-Q2_K_XL, 512 tokens of wikitext-2 test, against plain top-8 routing of the same model:

| routing | flash misses/token | ppl | KL mean | KL p99 | top-1 agree |
|---|---|---|---|---|---|
| top-8 (gamma 0, check: identical logits) | 364.8 | 4.369 | 0 | 0 | 100% |
| gamma 0.25 | 329.7 (-9.6%) | 4.382 (+0.3%) | 0.016 | 0.22 | 94.3% |
| gamma 0.5 | 303.4 (-16.8%) | 4.437 (+1.6%) | 0.021 | 0.25 | 92.7% |
| gamma 1 | 264.9 (-27.4%) | 4.462 (+2.1%) | 0.033 | 0.49 | 93.1% |
| gamma 2 | 218.0 (-40.2%) | 4.504 (+3.1%) | 0.058 | 0.81 | 90.5% |
| top-7 (for comparison) | 311.3 (-14.7%) | 4.522 (+3.5%) | 0.030 | 0.43 | 90.1% |

Cache-aware routing beats dropping to top-7 everywhere: gamma 0.5 saves more misses than top-7 at under
half the perplexity cost, and gamma 1 saves nearly twice as many at the same mean KL. Router mass kept is
0.998 / 0.994 / 0.984 / 0.963 for gamma 0.25 / 0.5 / 1 / 2: the swapped-in experts are close seconds.
Caveats: 512 tokens (ppl differences under ~1% are noise), and the phone's cache also holds prefetched
experts, so the live gain needs measuring there.

## 64-bit UFS DMA (kernel team's slot-B test kernel), 2026-10-05 night

Mainline ran gs201 UFS with a 32-bit DMA mask (a stale bring-up workaround; AOSP's UFS core overrides it to
64-bit), so with all 10 GB of RAM above 4 GB every expert read was bounced through swiotlb and memcpy'd,
and requests were capped at 256 KB. Test kernel: mask dropped, max_hw_sectors_kb 1024, average read
419 KB, swiotlb flat, 0 UFS errors; per-GiB SHA-256 of the 84 GB pack (O_DIRECT) matched the host 79/79.

Same runs as the UD-Q2_K_XL table (16-token prompt + 32 decoded, X1 capped, warm, read-only, logs in tmpfs):

| | slot A (32-bit, bouncing) | slot B (64-bit) | |
|---|---|---|---|
| TPU prefetch (the real config) | 66.9 s, 0.48 tok/s, stall 30.2 s | **63.4 s, 0.51 tok/s**, stall 29.9 s | **+6%** |
| CPU-predictor prefetch | - | 64.3 s, 0.50 tok/s | |
| no prefetch | 78.1 s, 0.41 tok/s, stall 46.6 s | 91.7 s, 0.35 tok/s, stall 46.5 s | -17% |
| prefill (16 tokens) | 27.2 s | 26.7 s | |

Flash wait is unchanged: decode was already at the link's ~2 GB/s and the bounce was not the limit. With
prefetch the gain is ~3 s less CPU time per run (the memcpy gone). Without prefetch decode got slower on
the CPU side (identical stall, +13.6 s compute, both slot-B runs agree, slot A had one run): UFS DMA on gs201 is
cache-coherent (IOCC, per the kernel team), so there is no cache maintenance: the likely cause is that a
direct DMA write lands cold in DRAM, where the bounce memcpy used to leave part of the expert cache-hot; prefetch hides it because the read
lands long before use. Net: a modest win for the configuration we run.
Write test (kernel team): 12 GiB written (8 O_DIRECT + 4 buffered) + sync, all per-GiB hashes match after
drop_caches, 0 errors. Slot B awaits Chris's decision to commit.

### Cache-aware routing on the Fold (live residency), 2026-10-05 night

Slot B kernel, UD-Q2_K_XL top-8, TPU predictor, buffer 64, pool 6.2 GB, X1 capped, 16-token prompt + 32
decoded, each run from battery <= 39 C. `FM_CACHE_ROUTE=gamma` with the store's live residency (resident
or in flight, prefetches included):

| gamma | decode | tok/s | hits | flash read | stall | prefetches used | router mass kept |
|---|---|---|---|---|---|---|---|
| off | 63.9 s | 0.50 | 59.0% | 115.6 GB | 30.1 s | 94.2% | 1 |
| 0.5 | 52.9 s | 0.60 (+21%) | 66.3% | 100.8 GB | 23.0 s | 97.6% | 0.996 |
| 1 | 47.9 / 48.3 s | 0.67 (+33%) | 69.7% | 93.9 GB | 19.3 s | 98.2% | 0.991 |
| 2 | 42.9 s | 0.75 (+49%) | 73.8% | 85.1 GB | 15.2 s | 99.2% | 0.980 |

Live routing swaps fewer experts than the 9-slot simulation at the same gamma and keeps more router
mass (0.991 vs 0.984 at gamma 1), because the real cache also holds prefetched experts, so the quality
cost should be at or below the simulated one. It also makes the TPU predictor's prefetches more useful:
a predicted expert that is close in the router gets chosen. The speedup is larger than the byte cut
(-19% bytes, +33% tok/s at gamma 1) because hits land on experts that are already there, and stall
time falls faster than reads (30.1 -> 19.3 s). Output is no longer deterministic: two gamma 1 runs
diverged at token ~20, since what is resident depends on when prefetches complete. All continuations
were fluent and on topic.

Quality estimate for the live runs: the simulation does not reproduce live behaviour by just growing the
cache (gamma 1 at 12 / 16 slots: still ~100 swaps/token and router mass kept 0.983; ppl 4.424 / 4.491,
KL 0.036 / 0.040), while live gamma 1 swaps 62/token. Matching on swap count instead, live gamma 1 is
closest to simulated gamma 0.5 (59 swaps: +1.6% ppl, KL 0.021) and live gamma 2 (91) to simulated gamma 1
(97: +2.1%, KL 0.033). Both are under top-7 (+3.5%, KL 0.030 at 14.7% fewer misses). 512-token ppl moves
by ~1% between runs, so KL is the steadier number. To measure live quality directly: fm-logits one token
per decode on the phone (n_ubatch 1, ~17 min per setting) against the host reference logits.

### Not pursued (estimates), 2026-10-05 night

* **Lower-bit experts (IQ2_XXS / IQ1_M):** IQ2_XXS saves ~21% of gate/up bytes vs Q2_K, but the phone CPU
  runs IQ2/IQ3 expert matmuls in 2.5-5.2 ms vs 1.83 ms for Q2_K (8 experts, 1536x4096). With cache-aware
  routing at gamma 1 a 32-token decode is ~19 s flash stall and ~29 s compute, so at most ~4 s of stall
  saved against >= 10 s of added compute, before the quality cost. Skipped (would need a ~70 GB download).
* **Expert pruning (drop the least-used ~25% per layer):** cache-aware routing already narrows each
  layer's working set to what is resident, adaptively and per context, without removing experts for good.
  Deprioritised.

**Live quality, measured on the phone** (fm-logits one token per step, `FM_UBATCH=1`, 128 tokens of
wikitext-2 test, TPU predictor, SoC pause at 85 C; each run compared with the routing-off run on the same
tokens, 111 scored positions):

| live routing | ppl | KL mean | KL p99 | top-1 agree | hits | swaps/token | mass kept |
|---|---|---|---|---|---|---|---|
| off | 4.690 | - | - | - | 70.1% | 0 | 1 |
| gamma 1 | 4.673 (-0.4%, noise) | 0.049 | 0.35 | 87.4% | 81.4% | 87 | 0.987 |
| gamma 2 | 4.880 (+4.1%) | 0.083 | 0.70 | 85.6% | 85.9% | 124 | 0.972 |

These runs swapped more than the speed runs (87 vs 62/token at gamma 1) because the thermal pauses let
prefetches land, so they are a pessimistic case for quality. gamma 1 shifts the output distribution
(KL 0.049, more than top-7's 0.030 on the host) without a measurable perplexity cost over 111 positions;
gamma 2 costs a clear +4%. **gamma 1 is the setting to use** (0.67 tok/s vs 0.50).

Heat: continuous one-token decode holds the big/mid/little clusters at 70-88 C, and an unpaused gamma 1
run reached 95 C (guard.sh's SoC limit) after ~70 s: with fewer flash stalls the CPU idles less. Long
runs now gate on big-thermal <= 50 C at the start and pause at 85 C.

## Prompt-sized expert matmuls, GPU vs CPU (2026-10-05 night)

`test-backend-ops perf -o MUL_MAT_ID`, 235B expert shapes (128 experts, top-8), one layer's projection.
GPU: G710 MC7, Mesa bd0debe9 (system), no knobs (`GGML_VK_SG16_WM_TILES` default, `GGML_VK_MMV_ID_MAX_TOKENS`
default 8, so 16 and 128 tokens take the tiled mmqid kernel), pinned 848 MHz (userspace governor).
CPU: 4 threads on cores 4-7, X1 capped. Slot B kernel.

| shape | 16 tok GPU | 16 tok CPU | 128 tok GPU | 128 tok CPU |
|---|---|---|---|---|
| gate/up Q2_K | 292 ms (5.5 GFLOPS) | 27.1 ms (59) | 4604 ms (2.8) | 226 ms (57) |
| gate/up Q3_K | 322 ms (5.0) | 42.4 ms (38) | 1188 ms (10.8) | 334 ms (39) |
| gate/up Q4_K | 98 ms (16.4) | 25.1 ms (64) | 1189 ms (10.8) | 196 ms (66) |
| down Q3_K | 306 ms (5.3) | 40.3 ms (40) | 1184 ms (10.9) | stopped (battery 47 C) |
| down Q4_K | 114 ms (14.1) | 25.8 ms (62) | 1190 ms (10.8) | stopped |

* The CPU's throughput is flat in batch size (57-66 GFLOPS for Q2_K/Q4_K), confirming that prompt processing
  is CPU-compute-bound at about one decode token's compute per prompt token.
* The GPU's tiled MUL_MAT_ID runs at ~11 GFLOPS (2.8 for Q2_K), roughly 1-2% of what the G710 should do:
  the kernel, not the hardware. The mat-vec id path matched the CPU at 16 tokens on 2026-10-04 (37 ms for
  q3_K with `GGML_VK_MMV_ID_MAX_TOKENS=16`). Next steps proposed: sweep that limit to 128, read PanVK's
  shader stats (spills, occupancy) for the tiled kernel and tune tiles for Mali, ask the GPU team for a
  libmali comparison and the int8-dot peak, and measure energy per matmul on GPU vs CPU (heat is now the
  limit).

**The fix that exists today: use the mat-vec id kernel for prompts.** `GGML_VK_MMV_ID_MAX_TOKENS` sweep (same
setup; the limit only matters when it covers the batch):

| shape | GPU tiled (default) | GPU mat-vec id (limit >= n) | CPU |
|---|---|---|---|
| gate/up Q2_K, 16 tokens | 292 ms (5.5 GFLOPS) | **16.3 ms (98.6)** | 27.1 ms (59) |
| down Q3_K, 16 tokens | 306 ms (5.3) | 40.3 ms (40.0) | 40.3 ms (40) |
| gate/up Q2_K, 128 tokens | 4612 ms (2.8) | **130 ms (98.9)** | 226 ms (57) |
| down Q3_K, 128 tokens | 1147 ms (11.2) | **322 ms (40.1)** | ~334 ms (Q3_K gate/up shape: 39) |

The mat-vec id kernel processes each (token, expert) pair separately and so re-reads weights per token,
but on the G710 it holds ~99 GFLOPS for Q2_K and ~40 for Q3_K at any batch size, beating the CPU (and the
tiled kernel by 9-35x). Per layer at 128 tokens: GPU ~580 ms vs CPU ~780 ms for gate+up+down, so a
CPU+GPU split of the experts could roughly halve prompt time.

Why the tiled kernel is slow (kernel team, shader stats from the drm-shim with BIFROST_MESA_DEBUG=shaderdb):
load/store-bound (q4_K n=128: 1129 LS cycles vs 22 FMA), 64 registers (half occupancy) and heavy spilling
(q2_K n=128: 368 spills / 1437 fills). It is PanVK's register allocation on these tiles, as with the q5_0
l-tile before; the kernel team is chasing it in Mesa. G710 MC7 at 848 MHz (libmali clpeak): FP32/FP16
731 GFLOPS, INT32 189 GIOPS, DRAM ~34 GB/s.

**End to end, 128-token prompt** (235B UD-Q2_K_XL, CPU 4 threads + G710 via `FM_ESTORE_SPLIT=k`,
`GGML_VK_MMV_ID_MAX_TOKENS=512`, pool 4 GB because the GPU path copies the 4.7 GB of resident weights into
anonymous memory instead of mapping them: at 6.2 GB the OOM killer stopped fm-gen, no panic):

| config | prefill 128 tokens |
|---|---|
| CPU only (pool 6.2 / 4 GB) | 123.6 / 124.3 s (1.04 tok/s) |
| 4 of 8 experts on the GPU (+ `FM_SCHED_OVERLAP`) | 128.6 s (128.5 s) |
| 5 of 8 on the GPU (+ overlap) | 126.5 s (126.3 s) |
| all 8 on the GPU | 120.2 s (1.06 tok/s) |

The GPU's half runs at benchmark speed inside the graph (`GGML_VK_PERF_LOGGER`: 4 experts x 128 tokens,
q2_K gate/up 135 ms at 95 GFLOPS, q3_K down 163 ms at 39, 299 ms per layer), but the CPU and GPU halves do
not overlap: a split is never faster than either alone. Moving all experts to the GPU saves only 4 s, so
expert matmuls are ~60 s of the 124 s; the rest is the dense part (attention projections, router,
output) plus 10 s of flash stall. To speed prompts up: make the split concurrent (up to ~30 s), and run
the dense matmuls on the GPU, which needs the tiled kernels fixed (the Mesa spilling + Mali tile sizes).
Also to fix: the GPU path's anonymous copy of the resident weights (`CPU model buffer 4724 MiB` instead of
`CPU_Mapped`). (The perf logger asserts `compute_ctx.expired()` under `FM_SCHED_OVERLAP`.)

**The split now overlaps (2026-10-06).** The CPU chain's expert ids (a `ggml_cont` of a view of the slot
ids) were built after the GPU chain in graph order, so the scheduler expanded the GPU assignment down onto
them; the CPU chain then had a GPU input and waited for the whole GPU chain. Building them before the GPU
chain and pinning `ffn_moe_ids_cpu` to the CPU (llama-graph.cpp, llama-context.cpp) fixes it. 128-token
prompt, pool 4 GB, `FM_SCHED_OVERLAP=1`: 4 of 8 experts on the GPU 102.5 s, **5 of 8: 100.2 s (1.28 tok/s,
+24% vs 124.3 s CPU-only)**.

The anonymous copy of the resident weights: in AUTO load mode llama.cpp turns mmap off for every weight when
any device lacks `mmap_support`, and panvk's device doesn't report it. fm-gen and fm-logits now request
`LLAMA_LOAD_MODE_MMAP` with `FM_NO_HOST` (the GPU only reads the estore pools); llama-server needs
`--load-mode mmap`.

**Skipping weak uncached experts: no better than a larger gamma (host simulation, 2026-10-06).**
`FM_CACHE_SKIP=tau` (with `FM_CACHE_ROUTE`): a chosen expert that is not cached and holds less than tau of the
chosen router mass is swapped for the best cached expert outside the top-k, whose weight is then zeroed
before normalisation (the token runs on 7 experts, no flash read). UD-Q2_K_XL, 512 tokens, 9-slot LRU,
vs plain top-8:

| routing | misses/token | ppl | KL mean | top-1 agree |
|---|---|---|---|---|
| gamma 1 | 264.9 | 4.462 (+2.1%) | 0.033 | 93.1% |
| gamma 1, skip 0.05 | 246.2 | 4.498 (+3.0%) | 0.038 | 92.9% |
| gamma 2 | 218.0 | 4.504 (+3.1%) | 0.058 | 90.5% |
| gamma 1, skip 0.1 | 156.1 | 4.847 (+10.9%) | 0.130 | 87.7% |

tau 0.05 sits on the same quality/bytes curve as raising gamma; tau 0.1 drops 1.3 experts per layer and
costs far more quality. Kept as an option (default off).

**CPU work in GPU host memory (2026-10-06).** With a Vulkan device present, llama.cpp allocates the CPU
backend's compute buffer (and the output buffer) from the GPU's host buffer type; `no_host` only covered
the weights. On panvk that memory is slow for CPU compute: 235B decode took 76.7 s instead of 50.3 s with
every op on the CPU (one CPU split per graph, `GGML_SCHED_DEBUG`). llama-context now honours `no_host` for
those buffers too (`llama_model::no_host()`). With it, 128-token prompt + 32 decoded, gamma 1, TPU
prefetch, split 5 of 8 on the GPU for batches >= 32 tokens: **prefill 90.2 s (1.42 tok/s, +37% vs 123.5 s
CPU-only)**, decode 56.9 s (vs 50.3 s CPU-only: a remaining ~13% cost of having the GPU backend loaded).
Lazy slot cleaning (`estore_set_gpu_batch`: slots that land outside a GPU batch are cleaned only when a GPU
batch acquires them) is in, but was not the cause.

After the buffer fix, with the GPU backend loaded but no split, decode matches CPU-only (48.96 vs 49.36 s for
32 tokens), and the split config with a 16-token prompt (below the 32-token threshold) decodes in 48.27 s
(0.66 tok/s). The ~6 s slower decode after a split 128-token prompt is heat: the CPU+GPU prompt leaves the
big cluster at ~87 C, near the 90 C CPU trip, as decode starts. Net for chat it is still a clear gain (~33 s
saved per 128 prompt tokens). Server config: `scripts/phone/serve.sh` / `scripts/phone/serve-run.sh` (gamma 1,
TPU prefetch, split 5 for batches >= 32, `--no-host -lm mmap`, GPU pinned 848 MHz, pause at battery 44 C or
big cluster 85 C).

Kernel team's device A/B of their q6_K alignment fix (Mesa 97fe3e3a) on mat-vec id, 235B gate shape, 848 MHz:
q2_K unchanged (90 / 98.5 GFLOPS at n=1 / 16); q3_K drops 41.7 -> 24.9 GFLOPS (-40%) because 2-byte-aligned
data is forced into 16-bit loads. So: stay on the system Mesa (bd0debe9) with the app-side q6_K n==5
workaround. Their next Mesa item is aligned wide loads plus shifts, which should keep q6_K safe and make
q3_K faster than today's ~42 GFLOPS.

**Correction (kernel team, 2026-10-06):** the q6_K mat-vec fault is not misaligned loads but a bug in Mesa's
Bifrost/Valhall SSA register allocator under high register pressure (the "alignment fix" only moved it from
n=5 to n=7, and was reverted). Workaround: `BIFROST_MESA_DEBUG=nossara` (legacy RA), measured neutral on
mat-vec id (q2_K 89.5 vs 89.6, q3_K 41.5 vs 41.6 GFLOPS). Not needed for the split configuration, where the
GPU only runs q2_K/q3_K/q4_K expert matmuls; it would matter if dense layers (q6_K) moved to the GPU, and
could then replace the app-side q6_K n==5 routing after a whole-model A/B.

**llama-server smoke test with the best config (2026-10-06).** Needed beyond fm-gen's settings: `-ngl 0` (the
server offloads all layers by default, which put the 4.9 GB of dense weights in pinned GPU memory and left
283 MB available) and `-ub 128` (the default 512-token ubatch asked for a 949 MB GPU compute buffer, which
failed with PANTHOR_VM_BIND ENOMEM). With those: 196-token chat prompt 163.9 s (1.20 tok/s, split on the
GPU), correct answer; a follow-up turn reused 211 cached tokens and processed 25 new ones, then generated
29 tokens at 0.46 tok/s with short SoC pauses (85 C pause / 65 C resume, ~3-7 s each). The first turn's
generation lost 191 s to a pause bug (an SoC-triggered pause waited for the battery to reach 41 C);
serve-throttle.sh now resumes on the sensor that caused the pause.

## Energy per token on battery (2026-10-06)

Phone unplugged (83% -> 80%), power = -(V x I) summed over both fuel gauges, sampled every 0.5 s. 235B
UD-Q2_K_XL, gamma 1, TPU predictor, X1 capped. Each config ran with 32 and with 8 decoded tokens (16-token
prompt); the difference is the decode energy of 24 tokens, with load and prompt cancelling.

| decode | 24 tokens | time | per token | net of idle (2.52 W) |
|---|---|---|---|---|
| CPU only | 328 J | 33.6 s (~9.8 W) | **13.7 J** | 10.1 J |
| 4 of 8 experts on the GPU (split forced at batch 1) | 384 J | 38.5 s (~10.0 W) | 16.0 J | 12.0 J |

The GPU is not more efficient for decode: the split costs ~17% more energy per token and is ~15% slower,
so it would make the phone pause for heat sooner, not later. Decode stays on the CPU; the GPU stays a
prompt-only accelerator. (The 16-token prompts also took 48 s with the split forced on vs 27 s CPU-only:
at small batches the per-layer handoff dominates, which is why the split threshold is 32 tokens.)

## Sustained decode: lower clocks win (2026-10-06)

200 decoded tokens per run (16-token prompt), gamma as listed, TPU predictor, CPU only, with the pause loop
(battery 44/41 C, big cluster 85/65 C, resume on the sensor that paused); decode time is wall clock, so it
includes pauses. Each run started at battery <= 38 C and big <= 50 C; phone on the charger (charging).

| config | pauses | paused | decode 200 tokens | sustained |
|---|---|---|---|---|
| gamma 1, X1 2.05 / A78 2.35 GHz | 33 | 250 s | 547 s | 0.37 tok/s |
| gamma 2, X1 2.05 / A78 2.35 GHz | 51 | 137 s | 404 s | 0.50 tok/s |
| **gamma 1, X1 1.46 / A78 1.61 GHz** | **0** | **0 s** | **350 s** | **0.57 tok/s** |

At full clocks the big cluster hits 85 C every few seconds (all pauses were SoC-triggered, 2-4 s each, many
of them); at the lower caps it never reaches 85 C, and the lost compute speed is smaller than the pauses
it avoids (flash stall also fell, 116 -> 93 s, with fewer pause-interrupted reads). The sustained limit is the
SoC sensor, not battery heat per token. Server default is now these caps (scripts/phone/serve-run.sh).
