// fm-replay: drive estore with a routing trace, real O_DIRECT reads against
// the expert pack, and modelled compute (sleep) between layers. Measures
// what cachesim.py estimates: hit rate, flash MB/token, stall share, tok/s.
//
//   fm-replay --pack P --index P.idx --trace DIR [--pool-gb 7] [--qd 16]
//             [--layer-us 1450] [--token-us 17000] [--tokens 4000] [--warm 800]
//             [--preds PREDS] [--pred-m 8]
//
// Compute model: --layer-us per MoE layer, --token-us per token (output
// head etc.); defaults are cachesim's 25 GB/s DRAM model for Qwen3-30B-A3B.
// --preds is train.py --emit output (reads PREDS.u16 + PREDS.txt); tokens
// then start at the prediction split.
#define _GNU_SOURCE
#include "estore.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t) ts.tv_sec * 1000000000ull + ts.tv_nsec;
}

static void sleep_us(double us) {
    if (us <= 0) return;
    struct timespec ts = { (time_t) (us / 1e6), (long) ((us - (long) (us / 1e6) * 1e6) * 1e3) };
    nanosleep(&ts, NULL);
}

static char * slurp(const char * path, size_t * n) {
    FILE * f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    fseek(f, 0, SEEK_END);
    *n = (size_t) ftell(f);
    fseek(f, 0, SEEK_SET);
    char * b = malloc(*n + 1);
    if (fread(b, 1, *n, f) != *n) { perror(path); exit(1); }
    b[*n] = 0;
    fclose(f);
    return b;
}

static long json_int(const char * js, const char * key) {
    char pat[64];
    snprintf(pat, sizeof(pat), "\"%s\":", key);
    const char * p = strstr(js, pat);
    if (!p) { fprintf(stderr, "fm-replay: meta.json lacks %s\n", key); exit(1); }
    return strtol(p + strlen(pat), NULL, 10);
}

static void usage(void) {
    fprintf(stderr, "usage: fm-replay --pack P --index I --trace DIR [--pool-gb G] [--qd N] [--layer-us U]\n"
                    "                 [--token-us U] [--tokens N] [--warm N] [--preds F] [--pred-m M] [--prefetch-qd N] [--policy layer|lru] [--pf-buffer N] [--hugetlb] [--async-submit] [--pf-idle] [--shares FILE]\n");
    exit(2);
}

int main(int argc, char ** argv) {
    const char *pack = NULL, *index = NULL, *trace = NULL, *preds_path = NULL;
    double pool_gb = 7.0, layer_us = 1450, token_us = 17000;
    long tokens = 4000, warm = 800;
    int qd = 16, pred_m = 8, hugetlb = 0, prefetch_qd = 0, policy = ESTORE_POLICY_LAYER, pf_buffer = 0;
    int async_submit = 0, pf_idle = 0;
    const char * shares_path = NULL;
    for (int i = 1; i < argc; ++i) {
        const char * a = argv[i];
#define NEXT (i + 1 < argc ? argv[++i] : (usage(), (char *) NULL))
        if      (!strcmp(a, "--pack"))     pack = NEXT;
        else if (!strcmp(a, "--index"))    index = NEXT;
        else if (!strcmp(a, "--trace"))    trace = NEXT;
        else if (!strcmp(a, "--pool-gb"))  pool_gb = atof(NEXT);
        else if (!strcmp(a, "--qd"))       qd = atoi(NEXT);
        else if (!strcmp(a, "--layer-us")) layer_us = atof(NEXT);
        else if (!strcmp(a, "--token-us")) token_us = atof(NEXT);
        else if (!strcmp(a, "--tokens"))   tokens = atol(NEXT);
        else if (!strcmp(a, "--warm"))     warm = atol(NEXT);
        else if (!strcmp(a, "--preds"))    preds_path = NEXT;
        else if (!strcmp(a, "--pred-m"))   pred_m = atoi(NEXT);
        else if (!strcmp(a, "--hugetlb"))  hugetlb = 1;
        else if (!strcmp(a, "--async-submit")) async_submit = 1;
        else if (!strcmp(a, "--pf-idle"))  pf_idle = 1;
        else if (!strcmp(a, "--shares"))   shares_path = NEXT;
        else if (!strcmp(a, "--prefetch-qd")) prefetch_qd = atoi(NEXT);
        else if (!strcmp(a, "--pf-buffer")) pf_buffer = atoi(NEXT);
        else if (!strcmp(a, "--policy"))   policy = !strcmp(NEXT, "lru") ? ESTORE_POLICY_LRU : ESTORE_POLICY_LAYER;
        else usage();
    }
    if (!pack || !index || !trace) usage();

    char path[4096];
    size_t n;
    snprintf(path, sizeof(path), "%s/meta.json", trace);
    char * meta = slurp(path, &n);
    const int L = (int) json_int(meta, "n_layer");
    const int K = (int) json_int(meta, "n_expert_used");
    snprintf(path, sizeof(path), "%s/topk.u16", trace);
    uint16_t * topk = (uint16_t *) slurp(path, &n);
    const long n_trace = (long) (n / (sizeof(uint16_t) * L * K));

    // predictions: [n_pred, L, M] u16 starting at trace token eval_start
    uint16_t * preds = NULL;
    long eval_start = 0, n_pred = 0;
    int pred_M = 0, pred_span = 0, n_pl = 0, pred_layers[256];
    if (preds_path) {
        snprintf(path, sizeof(path), "%s.txt", preds_path);
        FILE * f = fopen(path, "r");
        if (!f || fscanf(f, "%ld %ld %*d %d %d %d", &eval_start, &n_pred, &pred_M, &pred_span, &n_pl) != 5) {
            fprintf(stderr, "fm-replay: bad %s\n", path);
            return 1;
        }
        for (int i = 0; i < n_pl; ++i) if (fscanf(f, "%d", &pred_layers[i]) != 1) return 1;
        fclose(f);
        snprintf(path, sizeof(path), "%s.u16", preds_path);
        preds = (uint16_t *) slurp(path, &n);
        if (pred_m > pred_M) pred_m = pred_M;
    }
    const long t_begin = preds ? eval_start : 0;
    const long t_avail = preds ? n_pred : n_trace;
    if (tokens > t_avail) tokens = t_avail;
    if (warm >= tokens) warm = tokens / 5;

    struct estore_cfg cfg = { .pack = pack, .index = index, .pool_bytes = (uint64_t) (pool_gb * 1e9),
                              .qd = qd, .hugetlb = hugetlb, .prefetch_qd = prefetch_qd, .policy = policy, .prefetch_buffer = pf_buffer,
                              .async_submit = async_submit, .pf_ioprio_idle = pf_idle };
    int * weights = NULL;
    if (shares_path) {
        cfg.n_layer_weights = estore_read_layer_weights(shares_path, &weights);
        if (cfg.n_layer_weights < 0) return 1;
        cfg.layer_weights = weights;
    }
    struct estore * st = estore_open(&cfg);
    if (!st) return 1;
    if (estore_n_layer(st) != L) { fprintf(stderr, "fm-replay: trace/pack layer mismatch\n"); return 1; }
    int n_slots = 0;
    for (int g = 0; g < estore_n_groups(st); ++g) n_slots += estore_n_slots(st, g);
    fprintf(stderr, "fm-replay: %d layer groups, %d slots (%.2f GB), dma-bufs %s, %ld tokens (%ld warm-up) from %ld, preds %s (m=%d)\n",
            estore_n_groups(st), n_slots, pool_gb, estore_dmabuf_ok(st) ? "yes" : "no", tokens, warm, t_begin,
            preds ? preds_path : "none", pred_m);

    int experts[64], slots[64];
    char is_pl[256] = {0};
    for (int i = 0; i < n_pl; ++i) is_pl[pred_layers[i]] = 1;
    uint64_t t0 = now_ns();
    for (long ti = 0; ti < tokens; ++ti) {
        if (ti == warm) { estore_reset_stats(st); t0 = now_ns(); }
        const long t = t_begin + ti;
        for (int l = 0; l < L; ++l) {
            const uint16_t * tk = &topk[((size_t) t * L + l) * K];
            if (tk[0] != 0xFFFF) {   // skip dense layers
                for (int k = 0; k < K; ++k) experts[k] = tk[k];
                if (estore_acquire(st, l, K, experts, slots) < 0) return 1;
            }
            // as in llama.cpp: predict after this layer's experts are pinned, overlap with its compute
            if (preds && is_pl[l]) {
                for (int lp = l + 1; lp < l + 1 + pred_span && lp < L; ++lp) {
                    const uint16_t * pr = &preds[((size_t) ti * L + lp) * pred_M];
                    for (int m = 0; m < pred_m; ++m) {
                        if (pr[m] != 0xFFFF) estore_prefetch(st, lp, pr[m]);
                    }
                }
            }
            if (tk[0] != 0xFFFF) sleep_us(layer_us);
        }
        sleep_us(token_us);
        if ((ti + 1) % 200 == 0) fprintf(stderr, "\rfm-replay: %ld/%ld", ti + 1, tokens);
    }
    fprintf(stderr, "\n");
    const double elapsed = (now_ns() - t0) / 1e9;
    struct estore_stats s;
    estore_get_stats(st, &s);
    const long measured = tokens - warm;
    const double acc = (double) (s.hits + s.inflight_hits + s.misses);
    printf("{\"tokens\": %ld, \"pool_gb\": %.2f, \"slots\": %d, \"hit_rate\": %.4f, \"inflight_hit_rate\": %.4f, "
           "\"flash_mb_per_token\": %.1f, \"flash_gbps\": %.3f, \"stall_frac\": %.3f, \"prefetches\": %lu, \"prefetch_dropped\": %lu, "
           "\"prefetch_useful_frac\": %s%.3f, \"submit_ms_per_read\": %.3f, \"submit_frac\": %.3f, \"tok_per_s\": %.3f}\n",
           measured, pool_gb, n_slots, s.hits / acc, s.inflight_hits / acc,
           s.bytes_read / 1e6 / measured, s.bytes_read / 1e9 / elapsed, s.stall_ns / 1e9 / elapsed, s.prefetches, s.prefetch_dropped,
           "", s.prefetches ? (double) s.prefetch_used / s.prefetches : 0.0,
           (s.misses + s.prefetches) ? s.submit_ns / 1e6 / (double) (s.misses + s.prefetches) : 0.0, s.submit_ns / 1e9 / elapsed, measured / elapsed);
    estore_close(st);
    free(weights);
    return 0;
}
