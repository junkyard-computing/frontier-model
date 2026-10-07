// fm-logits: decode the first --n tokens of a text file in one batch and
// write every position's logits (f32, [n, n_vocab]) to --out. Run once with
// and once without FM_ESTORE_PACK and compare the files: expert streaming
// feeds the same bytes to the same kernels, so the logits must match exactly.
//
//   fm-logits -m model.gguf -f text.txt -o logits.f32 [--n 128] [-t 8]

#include "llama.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    std::string model, file, out;
    int n = 128, threads = 8;
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string k = argv[i];
        if      (k == "-m")  model = argv[i + 1];
        else if (k == "-f")  file = argv[i + 1];
        else if (k == "-o")  out = argv[i + 1];
        else if (k == "--n") n = atoi(argv[i + 1]);
        else if (k == "-t")  threads = atoi(argv[i + 1]);
        else { fprintf(stderr, "unknown arg %s\n", k.c_str()); return 2; }
    }
    if (model.empty() || file.empty() || out.empty()) {
        fprintf(stderr, "usage: %s -m model.gguf -f text.txt -o logits.f32 [--n 128] [-t 8]\n", argv[0]);
        return 2;
    }
    std::ifstream in(file);
    std::stringstream ss;
    ss << in.rdbuf();
    const std::string text = ss.str().substr(0, (size_t) n * 16);

    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    // FM_NO_REPACK=1: keep weights in GGUF layout (as the estore pools are), so
    // stock and streamed runs use the same CPU kernels and must match bitwise
    if (getenv("FM_NO_REPACK")) mp.use_extra_bufts = false;
    // FM_CPU_ONLY=1: no GPU devices, so CPU weights stay mmapped instead of going into
    // the GPU backend's pinned host buffers (which would hold the whole model in RAM)
    static ggml_backend_dev_t no_devices[1] = { nullptr };
    if (getenv("FM_CPU_ONLY")) mp.devices = no_devices;
    // FM_NO_HOST=1: keep CPU weights in plain (mmapped) memory even with a GPU present, instead of the GPU
    // backend's pinned host buffers; with FM_ESTORE_SPLIT the GPU only runs experts from the estore pools
    // and mmap them explicitly: in AUTO load mode llama.cpp turns mmap off for every weight when any device
    // (here the Mali Vulkan device) lacks mmap support, copying the ~5 GB of resident weights into RAM
    if (getenv("FM_NO_HOST")) { mp.no_host = true; mp.load_mode = LLAMA_LOAD_MODE_MMAP; }
    // FM_NGL=n: offload n layers to the GPU; FM_EXPS_CPU=1: keep the model's own *_exps tensors in
    // mmapped CPU memory (with FM_ESTORE_PACK they are never read; the pools stand in for them)
    if (getenv("FM_NGL")) mp.n_gpu_layers = atoi(getenv("FM_NGL"));
    static llama_model_tensor_buft_override exps_cpu[2] = { { "_exps", nullptr }, { nullptr, nullptr } };
    // with FM_ESTORE_PACK the experts stream into pools, so the model's own copies must stay
    // mmapped and untouched (never repacked into RAM or uploaded to a GPU)
    if (getenv("FM_EXPS_CPU") || getenv("FM_ESTORE_PACK")) {
        exps_cpu[0].buft = ggml_backend_cpu_buffer_type();
        mp.tensor_buft_overrides = exps_cpu;
        // a CPU override still goes through the CPU repack buffers, which copy every expert into
        // RAM at load; repacking must be off entirely (resident weights then use the generic kernels)
        mp.use_extra_bufts = false;
        // and never prefetch their file ranges at load (the patch marks *_exps lazy when streaming)
        mp.lazy_mode = LLAMA_LAZY_MODE_ON;
    }
    // FM_EXPERT_USED=k: route each token to k experts instead of the model's n_expert_used (quality
    // experiments: fewer experts per token means fewer expert bytes to stream)
    static llama_model_kv_override kvo[2] = {};
    if (const char * eu = getenv("FM_EXPERT_USED")) {
        kvo[0].tag = LLAMA_KV_OVERRIDE_TYPE_INT;
        snprintf(kvo[0].key, sizeof(kvo[0].key), "qwen3moe.expert_used_count");
        kvo[0].val_i64 = atoll(eu);
        mp.kv_overrides = kvo;
    }
    llama_model * m = llama_model_load_from_file(model.c_str(), mp);
    if (!m) return 1;
    const llama_vocab * vocab = llama_model_get_vocab(m);

    std::vector<llama_token> toks(text.size() + 8);
    int nt = llama_tokenize(vocab, text.data(), (int) text.size(), toks.data(), (int) toks.size(), true, false);
    if (nt < n) { fprintf(stderr, "text too short: %d tokens\n", nt); return 1; }
    toks.resize(n);

    auto cp = llama_context_default_params();
    cp.n_ctx = cp.n_batch = n;
    cp.n_ubatch = getenv("FM_UBATCH") ? atoi(getenv("FM_UBATCH")) : n;   // small ubatches keep each layer's expert union small
    cp.n_threads = cp.n_threads_batch = threads;
    // with streamed experts the pools ARE the expert weights: never let the scheduler copy them to
    // the GPU for large batches (op offload would duplicate GBs in shared RAM)
    if (getenv("FM_ESTORE_PACK")) cp.op_offload = false;
    if (getenv("FM_NO_FA")) cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;   // FM_NO_FA=1: no flash attention
    llama_context * ctx = llama_init_from_model(m, cp);

    llama_batch b = llama_batch_init(n, 0, 1);
    b.n_tokens = n;
    for (int i = 0; i < n; ++i) {
        b.token[i] = toks[i]; b.pos[i] = i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = 1;
    }
    if (llama_decode(ctx, b) != 0) { fprintf(stderr, "decode failed\n"); return 1; }

    const int nv = llama_vocab_n_tokens(vocab);
    FILE * f = fopen(out.c_str(), "wb");
    for (int i = 0; i < n; ++i) fwrite(llama_get_logits_ith(ctx, i), sizeof(float), nv, f);
    fclose(f);
    fprintf(stderr, "fm-logits: wrote %d x %d logits to %s\n", n, nv, out.c_str());
    // the token ids too, so perplexity can be computed from the logits (logits[i] predicts toks[i + 1])
    FILE * ft = fopen((out + ".tok").c_str(), "wb");
    fwrite(toks.data(), sizeof(llama_token), toks.size(), ft);
    fclose(ft);

    llama_batch_free(b);
    llama_free(ctx);
    llama_model_free(m);
    return 0;
}
