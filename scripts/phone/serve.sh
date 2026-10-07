#!/bin/sh
# llama-server, 235B UD-Q2_K_XL, best config (2026-10-06): cache-aware routing gamma 1, TPU predictor prefetch,
# CPU+GPU expert split (5 of 8 on the G710) for prompt batches >= 32 tokens only; generation stays on the CPU. Needs root.
ROOT=${FM_ROOT:-/userdata/frontier-model}; SRC=$(cd "$(dirname "$0")/../.." && pwd); cd "$ROOT"
exec env FM_ESTORE_STATS=1 FM_ESTORE_PACK=models/q235b-q2xl.pack FM_ESTORE_POOL_GB=${POOL:-5.8} \
  FM_ESTORE_PRED_INT8=tpu/q235b-tpu-chain-102k-pkg FM_ESTORE_PRED_P=0.7 FM_ESTORE_PREFETCH_QD=2 FM_ESTORE_PRED_M=8 FM_ESTORE_PF_BUFFER=64 \
  FM_ESTORE_PRED_TPU=1 FM_FINCH_CPUS=0-3 FM_CACHE_ROUTE=1 \
  FM_ESTORE_SPLIT=5 FM_ESTORE_SPLIT_MIN_TOKENS=32 FM_SCHED_OVERLAP=1 GGML_VK_MMV_ID_MAX_TOKENS=512 \
  taskset -c 4-7 build/bin/llama-server \
  -m models/q235b-q2xl-sparse/Qwen3-235B-A22B-Instruct-2507-UD-Q2_K_XL-00001-of-00002.gguf \
  -ot "_exps=CPU" -nr --no-op-offload --no-host -lm mmap -ngl 0 -c 2048 -ub 128 -np 1 -ctk q8_0 -ctv q8_0 -t 4 -tb 4 \
  --host 0.0.0.0 --port 8080 --api-key "${FM_API_KEY:?set FM_API_KEY}" --alias qwen3-235b-a22b-q2xl
