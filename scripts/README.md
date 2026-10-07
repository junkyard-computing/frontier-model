# Scripts

Helpers used for the Pixel Fold runs in [docs/results.md](../docs/results.md).

## On the phone (`phone/`)

They run as root from the phone's work directory `FM_ROOT` (default `/userdata/frontier-model`), which holds
`build/` (fm-gen, fm-logits, llama-server, test-backend-ops), `models/` (sparse GGUF + expert pack), `data/`
(wikitext-2), and `tpu/` (predictor packages). This repo sits next to them (e.g. `$FM_ROOT/src`); the scripts
find `phase0/guard.sh` relative to themselves. Every run caps the X1 at 2.05 GHz or lower, pins the GPU with the
userspace devfreq governor (never min/max_freq), runs under `guard.sh`, and restores clocks on exit.

| script | what |
|---|---|
| `serve-run.sh` | llama-server with the best config: cache-aware routing (gamma 1), TPU prefetch, CPU+GPU split for prompts >= 32 tokens, sustained caps (X1 1.46 / A78 1.61 GHz), thermal pause loop. Needs `FM_API_KEY`. |
| `serve.sh`, `serve-throttle.sh` | the server command, and the pause loop (battery 44/41 C, big cluster 85/65 C, resumes on the sensor that paused). Stop the server with `kill -CONT` then `kill`. |
| `decode-ab.sh` | decode A/B: `NAME:ENV,ENV ...` configs, 16-token prompt + `N` tokens (`PN`, `N`), CPU %, estore stats |
| `quality-live.sh` | fm-logits one token per step (live residency) for routing quality, with an SoC pause |
| `prompt-split.sh` | prompt processing CPU vs CPU+GPU split |
| `sustain.sh` | 200-token sustained decode with the pause loop, per clock cap |
| `power.sh` | energy per decoded token on battery (fuel gauges), CPU vs GPU split; analyse with `host/power-analyze.sh` |
| `thermal.sh` | decode tok/s and battery heating at CPU caps |
| `gpu-mmid-perf.sh`, `gpu-mmv-sweep.sh` | `test-backend-ops` MUL_MAT_ID perf, GPU vs CPU, and the mat-vec id token-limit sweep |
| `pack-hash.sh` | per-GiB SHA-256 of a file with O_DIRECT (verify the pack after storage/kernel changes) |

## On the host (`host/`)

| script | what |
|---|---|
| `cache-route.sh`, `cache-skip.sh` | cache-aware routing quality (simulated LRU) vs plain top-8; outputs in `OUT` (default `out/quality`), needs the reference `q2xl-top8.f32` there |
| `serve-watch.sh` | polls the phone's server (via `FM_PHONE_SSH`, default `ssh fold`) and exits on trouble |
| `power-analyze.sh` | integrates `pw-samples.txt` between the marks in `pw-marks.txt` |
