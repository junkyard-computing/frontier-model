// fm-trace: record MoE routing decisions (and optionally router-input hidden
// states) for every token of a corpus, for predictor training and the
// expert-cache simulator.
//
// Hooks llama.cpp's graph eval callback and captures, per layer:
//   ffn_moe_topk-<il>  [n_expert_used, n_tokens] i32 (strided view)
//   ffn_norm-<il>      [n_embd,        n_tokens] f32 (router input)
//   ffn_moe_probs-<il> [n_expert,      n_tokens] f32 (router probabilities; with --probs-k)
//
// Output directory:
//   meta.json            shapes and provenance
//   tokens.i32           [N]
//   topk.u16             [N, n_layer, n_expert_used]  router order (best first);
//                        0xFFFF for layers without a MoE block
//   hidden.f16           [N, n_hidden_layers, n_embd] layers in meta.hidden_layers
//   probs_ids.u16        [N, n_layer, probs_k]  the probs_k most probable experts, best first (--probs-k)
//   probs.f16            [N, n_layer, probs_k]  their router probabilities (pre-normalisation)
//
// The corpus is split into independent windows of --ctx tokens; each window is
// one llama_decode (one ubatch), so every captured tensor covers the whole window.

#include "ggml.h"
#include "ggml-backend.h"
#include "llama.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

struct args {
    std::string model;
    std::string corpus;
    std::string out;
    int ctx           = 512;
    long max_tokens   = 50000;
    int hidden_stride = 4;   // record hidden state every N layers (0 = none)
    int threads       = 8;
    int ngl           = 0;
    long start_window = 0;   // skip this many windows (to continue an interrupted trace)
    int probs_k       = 0;   // record the top-K router probabilities per layer (0 = none)
};

struct trace_state {
    int n_layer = 0, n_embd = 0, n_expert = 0, n_used = 0;
    std::vector<int> hidden_layers;      // layer ids with hidden capture
    std::vector<int> hidden_slot;        // layer id -> slot, or -1

    // per-window buffers, filled by the callback
    int n_tok = 0;
    std::vector<uint16_t> topk;          // [n_tok, n_layer, n_used]
    std::vector<ggml_fp16_t> hidden;     // [n_tok, n_hidden, n_embd]
    std::vector<char> seen;              // [n_layer] topk captured this window
    int probs_k = 0;
    std::vector<uint16_t> probs_ids;     // [n_tok, n_layer, probs_k]
    std::vector<ggml_fp16_t> probs;      // [n_tok, n_layer, probs_k]
};

static bool parse_layer(const char * name, const char * prefix, int * il) {
    size_t n = strlen(prefix);
    if (strncmp(name, prefix, n) != 0) {
        return false;
    }
    char * end = nullptr;
    long v = strtol(name + n, &end, 10);
    if (end == name + n || *end != '\0') {
        return false;
    }
    *il = (int) v;
    return true;
}

// Copy row i1 of a (possibly strided) 2-D tensor into dst.
static void get_row(trace_state & st, const ggml_tensor * t, int64_t i1, void * dst, size_t nbytes) {
    if (ggml_backend_buffer_is_host(t->buffer)) {
        memcpy(dst, (const char *) t->data + i1 * t->nb[1], nbytes);
    } else {
        ggml_backend_tensor_get(t, dst, i1 * t->nb[1], nbytes);
    }
}

static bool eval_cb(ggml_tensor * t, bool ask, void * ud) {
    auto & st = *(trace_state *) ud;
    int il;
    bool is_topk = parse_layer(t->name, "ffn_moe_topk-", &il);
    bool is_norm = !is_topk && parse_layer(t->name, "ffn_norm-", &il);
    bool is_prob = !is_topk && !is_norm && st.probs_k > 0 && parse_layer(t->name, "ffn_moe_probs-", &il);
    if (!is_topk && !is_norm && !is_prob) {
        return !ask;   // not interested; keep computing
    }
    if (il < 0 || il >= st.n_layer) {
        return !ask;
    }
    if (is_norm && st.hidden_slot[il] < 0) {
        return !ask;
    }
    if (ask) {
        return true;
    }

    if (t->ne[1] != st.n_tok) {
        fprintf(stderr, "fm-trace: %s has %lld tokens, expected %d (ubatch split?)\n",
                t->name, (long long) t->ne[1], st.n_tok);
        exit(1);
    }

    if (is_topk) {
        GGML_ASSERT(t->type == GGML_TYPE_I32 && t->ne[0] == st.n_used);
        std::vector<int32_t> ids(st.n_used);
        for (int i = 0; i < st.n_tok; ++i) {
            get_row(st, t, i, ids.data(), st.n_used * sizeof(int32_t));
            uint16_t * dst = &st.topk[((size_t) i * st.n_layer + il) * st.n_used];
            for (int k = 0; k < st.n_used; ++k) {
                dst[k] = (uint16_t) ids[k];
            }
        }
        st.seen[il] = 1;
    } else if (is_prob) {
        GGML_ASSERT(t->type == GGML_TYPE_F32 && t->ne[0] == st.n_expert);
        std::vector<float> p(st.n_expert);
        std::vector<int> ord(st.n_expert);
        for (int i = 0; i < st.n_tok; ++i) {
            get_row(st, t, i, p.data(), st.n_expert * sizeof(float));
            for (int e = 0; e < st.n_expert; ++e) ord[e] = e;
            std::partial_sort(ord.begin(), ord.begin() + st.probs_k, ord.end(), [&](int x, int y) { return p[x] > p[y]; });
            const size_t o = ((size_t) i * st.n_layer + il) * st.probs_k;
            for (int k = 0; k < st.probs_k; ++k) {
                st.probs_ids[o + k] = (uint16_t) ord[k];
                st.probs[o + k]     = ggml_fp32_to_fp16(p[ord[k]]);
            }
        }
    } else {
        GGML_ASSERT(t->type == GGML_TYPE_F32 && t->ne[0] == st.n_embd);
        const int slot = st.hidden_slot[il];
        const size_t n_hidden = st.hidden_layers.size();
        std::vector<float> h(st.n_embd);
        for (int i = 0; i < st.n_tok; ++i) {
            get_row(st, t, i, h.data(), st.n_embd * sizeof(float));
            ggml_fp32_to_fp16_row(h.data(), &st.hidden[((size_t) i * n_hidden + slot) * st.n_embd], st.n_embd);
        }
    }
    return true;
}

static std::string meta_str(const llama_model * m, const std::string & key) {
    char buf[256];
    if (llama_model_meta_val_str(m, key.c_str(), buf, sizeof(buf)) < 0) {
        return "";
    }
    return buf;
}

static void usage(const char * argv0) {
    fprintf(stderr,
            "usage: %s -m model.gguf -f corpus.txt -o outdir [--ctx 512] [--max-tokens 50000]\n"
            "          [--hidden-stride 4] [-t threads] [-ngl layers] [--start-window N] [--probs-k K]\n", argv0);
    exit(1);
}

static args parse_args(int argc, char ** argv) {
    args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto next = [&]() -> const char * {
            if (i + 1 >= argc) usage(argv[0]);
            return argv[++i];
        };
        if      (k == "-m")              a.model = next();
        else if (k == "-f")              a.corpus = next();
        else if (k == "-o")              a.out = next();
        else if (k == "--ctx")           a.ctx = atoi(next());
        else if (k == "--max-tokens")    a.max_tokens = atol(next());
        else if (k == "--hidden-stride") a.hidden_stride = atoi(next());
        else if (k == "-t")              a.threads = atoi(next());
        else if (k == "-ngl")            a.ngl = atoi(next());
        else if (k == "--start-window")  a.start_window = atol(next());
        else if (k == "--probs-k")       a.probs_k = atoi(next());
        else usage(argv[0]);
    }
    if (a.model.empty() || a.corpus.empty() || a.out.empty() || a.ctx <= 0) {
        usage(argv[0]);
    }
    return a;
}

int main(int argc, char ** argv) {
    args a = parse_args(argc, argv);

    std::ifstream in(a.corpus, std::ios::binary);
    if (!in) {
        fprintf(stderr, "fm-trace: cannot read %s\n", a.corpus.c_str());
        return 1;
    }
    std::stringstream ss;
    ss << in.rdbuf();
    const std::string text = ss.str();

    std::error_code ec;
    std::filesystem::create_directories(a.out, ec);
    if (ec) {
        fprintf(stderr, "fm-trace: cannot create %s\n", a.out.c_str());
        return 1;
    }

    llama_backend_init();

    auto mp = llama_model_default_params();
    mp.n_gpu_layers = a.ngl;
    // with FM_ESTORE_PACK (expert streaming) the model's own *_exps tensors are never read: keep
    // them mmapped (no repacking into RAM, which for a 235B model means copying ~108 GB)
    static llama_model_tensor_buft_override exps_cpu[2] = { { "_exps", nullptr }, { nullptr, nullptr } };
    if (getenv("FM_ESTORE_PACK")) {
        exps_cpu[0].buft = ggml_backend_cpu_buffer_type();
        mp.tensor_buft_overrides = exps_cpu;
        // a CPU override still goes through the CPU repack buffers (they copy every expert into RAM)
        mp.use_extra_bufts = false;
        // and never prefetch their file ranges at load (the patch marks *_exps lazy when streaming)
        mp.lazy_mode = LLAMA_LAZY_MODE_ON;
    }
    llama_model * model = llama_model_load_from_file(a.model.c_str(), mp);
    if (!model) {
        fprintf(stderr, "fm-trace: failed to load %s\n", a.model.c_str());
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    trace_state st;
    st.n_layer = llama_model_n_layer(model);
    st.n_embd  = llama_model_n_embd(model);
    const std::string arch = meta_str(model, "general.architecture");
    st.n_expert = atoi(meta_str(model, arch + ".expert_count").c_str());
    st.n_used   = atoi(meta_str(model, arch + ".expert_used_count").c_str());
    if (st.n_expert <= 0 || st.n_used <= 0) {
        fprintf(stderr, "fm-trace: %s is not a MoE model\n", a.model.c_str());
        return 1;
    }
    st.probs_k = std::min(a.probs_k, st.n_expert);
    st.hidden_slot.assign(st.n_layer, -1);
    if (a.hidden_stride > 0) {
        for (int il = 0; il < st.n_layer; il += a.hidden_stride) {
            st.hidden_slot[il] = (int) st.hidden_layers.size();
            st.hidden_layers.push_back(il);
        }
    }

    // tokenize the whole corpus once
    std::vector<llama_token> toks(text.size() + 16);
    int n = llama_tokenize(vocab, text.data(), (int32_t) text.size(), toks.data(), (int32_t) toks.size(), false, false);
    if (n < 0) {
        fprintf(stderr, "fm-trace: tokenization failed\n");
        return 1;
    }
    toks.resize(n);
    const long n_windows = std::min<long>(n / a.ctx, (a.max_tokens + a.ctx - 1) / a.ctx);
    if (a.start_window >= n_windows) {
        fprintf(stderr, "fm-trace: --start-window %ld but only %ld windows\n", a.start_window, n_windows);
        return 1;
    }
    if (n_windows <= 0) {
        fprintf(stderr, "fm-trace: corpus has %d tokens, fewer than --ctx %d\n", n, a.ctx);
        return 1;
    }
    fprintf(stderr, "fm-trace: %s, %d layers, %d/%d experts, n_embd %d; %d corpus tokens -> %ld windows of %d\n",
            arch.c_str(), st.n_layer, st.n_used, st.n_expert, st.n_embd, n, n_windows, a.ctx);

    auto cp = llama_context_default_params();
    cp.n_ctx = a.ctx;
    cp.n_batch = a.ctx;
    cp.n_ubatch = a.ctx;
    cp.n_threads = a.threads;
    cp.n_threads_batch = a.threads;
    cp.cb_eval = eval_cb;
    cp.cb_eval_user_data = &st;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        fprintf(stderr, "fm-trace: failed to create context\n");
        return 1;
    }

    FILE * f_tok = fopen((a.out + "/tokens.i32").c_str(), "wb");
    FILE * f_top = fopen((a.out + "/topk.u16").c_str(), "wb");
    FILE * f_hid = st.hidden_layers.empty() ? nullptr : fopen((a.out + "/hidden.f16").c_str(), "wb");
    FILE * f_pid = st.probs_k ? fopen((a.out + "/probs_ids.u16").c_str(), "wb") : nullptr;
    FILE * f_prb = st.probs_k ? fopen((a.out + "/probs.f16").c_str(), "wb") : nullptr;
    if (!f_tok || !f_top || (!st.hidden_layers.empty() && !f_hid) || (st.probs_k && (!f_pid || !f_prb))) {
        fprintf(stderr, "fm-trace: cannot open outputs in %s: %s\n", a.out.c_str(), strerror(errno));
        return 1;
    }

    // every token requests logits so the last layer is not pruned to the output rows
    llama_batch batch = llama_batch_init(a.ctx, 0, 1);
    std::vector<char> moe_layer(st.n_layer, 0);
    long n_done = 0;

    for (long w = a.start_window; w < n_windows; ++w) {
        const llama_token * wt = toks.data() + w * a.ctx;
        batch.n_tokens = a.ctx;
        for (int i = 0; i < a.ctx; ++i) {
            batch.token[i]     = wt[i];
            batch.pos[i]       = i;
            batch.n_seq_id[i]  = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i]    = 1;
        }

        st.n_tok = a.ctx;
        st.topk.assign((size_t) a.ctx * st.n_layer * st.n_used, 0xFFFF);
        st.hidden.assign((size_t) a.ctx * st.hidden_layers.size() * st.n_embd, 0);
        st.seen.assign(st.n_layer, 0);
        st.probs_ids.assign((size_t) a.ctx * st.n_layer * st.probs_k, 0xFFFF);
        st.probs.assign((size_t) a.ctx * st.n_layer * st.probs_k, ggml_fp32_to_fp16(0.0f));

        llama_memory_clear(llama_get_memory(ctx), true);
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "fm-trace: decode failed at window %ld\n", w);
            return 1;
        }

        int n_moe = 0;
        for (int il = 0; il < st.n_layer; ++il) {
            moe_layer[il] |= st.seen[il];
            n_moe += st.seen[il];
        }
        if (n_moe == 0) {
            fprintf(stderr, "fm-trace: no ffn_moe_topk tensors seen; unsupported graph?\n");
            return 1;
        }

        fwrite(wt, sizeof(llama_token), a.ctx, f_tok);
        fwrite(st.topk.data(), sizeof(uint16_t), st.topk.size(), f_top);
        if (f_hid) {
            fwrite(st.hidden.data(), sizeof(ggml_fp16_t), st.hidden.size(), f_hid);
        }
        if (f_pid) {
            fwrite(st.probs_ids.data(), sizeof(uint16_t), st.probs_ids.size(), f_pid);
            fwrite(st.probs.data(), sizeof(ggml_fp16_t), st.probs.size(), f_prb);
        }
        n_done += a.ctx;
        fprintf(stderr, "\rfm-trace: %ld/%ld windows (%ld tokens this run)", w + 1, n_windows, n_done);
    }
    fprintf(stderr, "\n");
    fclose(f_tok);
    fclose(f_top);
    if (f_hid) fclose(f_hid);
    if (f_pid) { fclose(f_pid); fclose(f_prb); }

    FILE * fm = fopen((a.out + "/meta.json").c_str(), "w");
    fprintf(fm, "{\n");
    fprintf(fm, "  \"model\": \"%s\",\n", a.model.c_str());
    fprintf(fm, "  \"arch\": \"%s\",\n", arch.c_str());
    fprintf(fm, "  \"corpus\": \"%s\",\n", a.corpus.c_str());
    fprintf(fm, "  \"n_tokens\": %ld,\n", n_done);
    fprintf(fm, "  \"start_token\": %ld,\n", a.start_window * a.ctx);
    fprintf(fm, "  \"ctx\": %d,\n", a.ctx);
    fprintf(fm, "  \"n_layer\": %d,\n", st.n_layer);
    fprintf(fm, "  \"n_embd\": %d,\n", st.n_embd);
    fprintf(fm, "  \"n_expert\": %d,\n", st.n_expert);
    fprintf(fm, "  \"n_expert_used\": %d,\n", st.n_used);
    fprintf(fm, "  \"probs_k\": %d,\n", st.probs_k);
    fprintf(fm, "  \"moe_layers\": [");
    for (int il = 0, first = 1; il < st.n_layer; ++il) {
        if (moe_layer[il]) { fprintf(fm, "%s%d", first ? "" : ", ", il); first = 0; }
    }
    fprintf(fm, "],\n  \"hidden_tensor\": \"ffn_norm\",\n  \"hidden_layers\": [");
    for (size_t i = 0; i < st.hidden_layers.size(); ++i) {
        fprintf(fm, "%s%d", i ? ", " : "", st.hidden_layers[i]);
    }
    fprintf(fm, "]\n}\n");
    fclose(fm);

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
