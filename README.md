# frontier-model

Run large Chinese open-weight MoE LLMs on one Pixel Fold (gs201), streaming
experts from UFS in the style of *LLM in a Flash* (arXiv 2312.11514), with the
CPU, the Mali-G710 (panvk) and the Edge TPU (finch) all doing real work;
then scale out to the junkyard cluster. The plan and the hardware constraints
behind it are in [docs/plan.md](docs/plan.md).

Target: Qwen3-235B-A22B. Development model: Qwen3-30B-A3B (same architecture).

## Layout

| Path | What |
|---|---|
| `tools/trace/` | `fm-trace`: records per-token MoE routing (+ router-input hidden states) via llama.cpp's eval callback |
| `tools/sim/` | `profile_model.py` (expert vs resident bytes), `cachesim.py` (cache/prefetch simulator) |
| `tools/predictor/train.py` | expert-prefetch predictors, shaped for finch (int8 FC, widths 1024–4096, no 64/128 outputs) |
| `tools/relayout/pack_experts.py` | GGUF → expert pack (4 KiB bundles, 64 B projections) + index; `--verify` |
| `tools/logits/` | `fm-logits`: dump logits to compare stock vs streamed runs |
| `runtime/` | `estore` (udmabuf pools, io_uring O_DIRECT scatter reads, LRU, prefetch) and `fm-replay` |
| `patches/llama.cpp-estore.patch` | llama.cpp integration (`FM_ESTORE_PACK=model.pack`) |
| `phase0/` | on-phone kit: `guard.sh` (battery/SoC temperature stop), `sysinfo.sh`, `fio-experts.sh`, `bench-baseline.sh` |
| `docs/` | [plan.md](docs/plan.md), [runtime.md](docs/runtime.md), [results.md](docs/results.md) |
| `models/`, `data/`, `traces/`, `third_party/` | downloads, outputs, llama.cpp clone (git-ignored) |

## Build

```sh
nix develop
# tools against stock llama.cpp (a clone at ../llama.cpp, unmodified, built out-of-tree)
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON
ninja -C build

# llama.cpp with expert streaming
# the patch is against upstream llama.cpp 4da6337767f973e2b4d0797e5b323d77d8565e4a (2026-09-27)
git clone https://github.com/ggml-org/llama.cpp third_party/llama.cpp
git -C third_party/llama.cpp checkout 4da6337767f973e2b4d0797e5b323d77d8565e4a
git -C third_party/llama.cpp apply ../../patches/llama.cpp-estore.patch
cmake -S . -B build-es -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON \
  -DLLAMA_CPP_DIR=$PWD/third_party/llama.cpp -DLLAMA_FM_ESTORE_DIR=$PWD/runtime
ninja -C build-es fm-logits
```

The dev shell sets `NIX_ENFORCE_NO_NATIVE=0`. Without it the Nix cc wrapper
silently drops `-march=native` and llama.cpp builds for baseline x86-64.

## Collect a trace and simulate

```sh
curl -L -o models/Qwen3-30B-A3B-Q4_K_M.gguf \
  https://huggingface.co/Qwen/Qwen3-30B-A3B-GGUF/resolve/main/Qwen3-30B-A3B-Q4_K_M.gguf
curl -L -o data/wikitext-2-raw-v1.zip \
  https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip
(cd data && unzip wikitext-2-raw-v1.zip)

build/fm-trace -m models/Qwen3-30B-A3B-Q4_K_M.gguf -f data/wikitext-2-raw/wiki.test.raw \
  -o traces/q30b-wiki --max-tokens 50000 --hidden-stride 4 -t 12
python3 tools/sim/profile_model.py models/Qwen3-30B-A3B-Q4_K_M.gguf > traces/q30b-profile.json
python3 tools/sim/cachesim.py traces/q30b-wiki traces/q30b-profile.json --policy hot
```

Trace format: see the header of `tools/trace/trace.cpp`.

## Stream experts from flash

```sh
python3 tools/relayout/pack_experts.py models/Qwen3-30B-A3B-Q4_K_M.gguf models/q30b.pack
python3 tools/relayout/pack_experts.py --verify models/Qwen3-30B-A3B-Q4_K_M.gguf models/q30b.pack
FM_ESTORE_PACK=models/q30b.pack FM_ESTORE_POOL_GB=7 build-es/fm-logits -m models/Qwen3-30B-A3B-Q4_K_M.gguf ...
build/fm-replay --pack models/q30b.pack --index models/q30b.pack.idx --trace traces/q30b-wiki
```
