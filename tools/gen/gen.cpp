// fm-gen: greedy decode benchmark. Prefills --prompt-n tokens of a text file,
// then generates --n tokens one at a time (argmax), printing the token ids
// and decode tok/s. Greedy output must be identical with and without expert
// streaming (and with any prefetch predictor): they change timing, not math.
//
//   fm-gen -m model.gguf -f text.txt [--prompt-n 64] [--n 64] [-t 8] [-o ids.txt]

#include "llama.h"
#include "ggml-backend.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    std::string model, file, out;
    int prompt_n = 64, n = 64, threads = 8;
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string k = argv[i];
        if      (k == "-m")         model = argv[i + 1];
        else if (k == "-f")         file = argv[i + 1];
        else if (k == "-o")         out = argv[i + 1];
        else if (k == "--prompt-n") prompt_n = atoi(argv[i + 1]);
        else if (k == "--n")        n = atoi(argv[i + 1]);
        else if (k == "-t")         threads = atoi(argv[i + 1]);
        else { fprintf(stderr, "unknown arg %s\n", k.c_str()); return 2; }
    }
    if (model.empty() || file.empty()) {
        fprintf(stderr, "usage: %s -m model.gguf -f text.txt [--prompt-n 64] [--n 64] [-t 8] [-o ids.txt]\n", argv[0]);
        return 2;
    }
    std::ifstream in(file);
    std::stringstream ss;
    ss << in.rdbuf();
    const std::string text = prompt_n > 0 ? ss.str().substr(0, (size_t) prompt_n * 16) : ss.str();   // --prompt-n 0: whole file

    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
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
    // FM_EXPERT_USED=k: route each token to k experts instead of the model's n_expert_used (quality/speed
    // trade experiments: fewer experts per token means fewer expert bytes to stream)
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
    // --prompt-n 0: the whole file is the prompt, special tokens parsed (e.g. a chat-template prompt)
    const bool whole = prompt_n <= 0;
    int nt = llama_tokenize(vocab, text.data(), (int) text.size(), toks.data(), (int) toks.size(), true, whole);
    if (whole) prompt_n = nt;
    if (nt < prompt_n) { fprintf(stderr, "text too short: %d tokens\n", nt); return 1; }
    toks.resize(prompt_n);

    auto cp = llama_context_default_params();
    cp.n_ctx = prompt_n + n + 8;
    cp.n_batch = prompt_n;
    cp.n_ubatch = getenv("FM_UBATCH") ? atoi(getenv("FM_UBATCH")) : prompt_n;   // small ubatches keep each layer's expert union small
    cp.n_threads = cp.n_threads_batch = threads;
    // with streamed experts the pools ARE the expert weights: never let the scheduler copy them to
    // the GPU for large batches (op offload would duplicate GBs in shared RAM)
    if (getenv("FM_ESTORE_PACK")) cp.op_offload = false;
    if (getenv("FM_NO_FA")) cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;   // FM_NO_FA=1: no flash attention
    llama_context * ctx = llama_init_from_model(m, cp);
    const int nv = llama_vocab_n_tokens(vocab);

    auto argmax = [&](const float * lg) {
        int best = 0;
        for (int i = 1; i < nv; ++i) if (lg[i] > lg[best]) best = i;
        return best;
    };

    auto t0 = std::chrono::steady_clock::now();
    if (llama_decode(ctx, llama_batch_get_one(toks.data(), prompt_n)) != 0) return 1;
    auto t1 = std::chrono::steady_clock::now();
    std::vector<llama_token> gen;
    llama_token tok = argmax(llama_get_logits_ith(ctx, -1));
    for (int i = 0; i < n; ++i) {
        gen.push_back(tok);
        if (llama_decode(ctx, llama_batch_get_one(&tok, 1)) != 0) return 1;
        tok = argmax(llama_get_logits_ith(ctx, -1));
    }
    auto t2 = std::chrono::steady_clock::now();
    const double pp = std::chrono::duration<double>(t1 - t0).count();
    const double tg = std::chrono::duration<double>(t2 - t1).count();
    fprintf(stderr, "fm-gen: prefill %d tok in %.2f s (%.2f tok/s), decode %d tok in %.2f s (%.2f tok/s)\n",
            prompt_n, pp, prompt_n / pp, n, tg, n / tg);

    std::string ids;
    for (llama_token t : gen) ids += std::to_string(t) + " ";
    if (!out.empty()) {
        std::ofstream(out) << ids << "\n";
        // and the generated text, for reading the output
        std::string txt;
        char piece[256];
        for (llama_token t : gen) {
            const int np = llama_token_to_piece(vocab, t, piece, sizeof(piece), 0, false);
            if (np > 0) txt.append(piece, np);
        }
        std::ofstream(out + ".txt") << txt << "\n";
    }
    llama_free(ctx);
    llama_model_free(m);
    return 0;
}
