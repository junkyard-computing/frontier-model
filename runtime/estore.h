// estore: flash-streamed MoE expert cache.
//
// Experts live on flash in an expert pack (tools/relayout/pack_experts.py):
// one contiguous bundle [gate | up | down] per (layer, expert). In memory
// they live in packed per-projection slot pools, shaped like stock
// ffn_{gate,up,down}_exps tensors ([ne00, ne01, n_slots]), so the unmodified
// CPU and Vulkan MUL_MAT_ID kernels can index them by slot. A bundle is
// fetched with one O_DIRECT preadv whose iovecs scatter it into its slots.
// All pools share one memfd; each pool is also exported as its own dma-buf
// (udmabuf over that memfd range) when available, so the GPU and TPU can
// import the same pages (zero copy).
//
// Layers are grouped by projection signature (types and sizes; mixed quants
// such as Q4_K_M give several groups). Each group has its own pools, slots
// and LRU, shared by all of its layers; each group caches the same fraction
// of its experts.
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ESTORE_MAX_PROJ   3
#define ESTORE_MAX_GROUPS 16

struct estore_cfg {
    const char * pack;          // expert pack
    const char * index;         // pack .idx (text, fmpack 2)
    uint64_t     pool_bytes;    // total bytes for all slots of all groups
    int          qd;            // io_uring queue depth
    int          hugetlb;       // back the pools with hugetlb pages
    int          prefetch_qd;   // max prefetch reads in flight (0 = no cap); the rest wait
                                // in a software queue so demand reads never queue behind them
    int          policy;        // ESTORE_POLICY_LAYER (default) or ESTORE_POLICY_LRU
    int          prefetch_buffer; // LAYER policy: slots reserved for prefetched experts (0: prefetches go
                                  // into the layer shares). Wrong guesses then never evict cached experts
    int          clean_slots;   // clean each slot to DRAM when its read lands (needed when a device
                                // that does not snoop the CPU caches, e.g. the G710, reads the pools)
    int          async_submit;  // IOSQE_ASYNC on every read: io_uring's workers wait for block-layer request
                                // slots instead of the submitting thread (a 9 MB read is ~36 requests on
                                // UFS with max_sectors_kb 256, and nr_requests 64 holds fewer than 2 reads)
    int          pf_ioprio_idle; // prefetch reads in the idle I/O class, so the scheduler (mq-deadline)
                                 // dispatches demand reads ahead of queued prefetch requests
    const int   *layer_weights; // LAYER policy: relative share of each layer (n_layer_weights entries, NULL =
    int          n_layer_weights; // equal shares). Each group's slots are split in proportion; min 1 per layer
};

// Eviction policies. LRU: one LRU per layer group. LAYER (default): every layer gets an equal
// share of its group's slots, less a small reserve for the layer being computed. A miss takes a
// free slot, else the layer's own oldest expert. When the next layer is acquired, the previous one
// is trimmed back to its share (oldest first), returning its extra slots. Experts are inserted in
// reverse router order, so a layer keeps its top-ranked experts. A token walks the layers in order,
// so once a token's working set (layers x top-k) exceeds the cache, plain LRU evicts exactly what
// the next token needs first: 0% hits on Qwen3-235B with a 5.6 GB cache, vs 40% with LAYER.
#define ESTORE_POLICY_LAYER 0
#define ESTORE_POLICY_LRU   1

struct estore_stats {
    uint64_t hits;              // resident at acquire time
    uint64_t inflight_hits;     // prefetch issued earlier, still loading
    uint64_t misses;            // demand reads
    uint64_t prefetches;        // prefetch reads issued
    uint64_t prefetch_dropped;  // queued prefetches dropped as stale (layer already run)
    uint64_t prefetch_used;     // prefetched slots later acquired
    uint64_t bytes_read;
    uint64_t stall_ns;          // time acquire spent waiting on I/O
    uint64_t submit_ns;         // time in io_uring_submit (the kernel pins and maps the destination pages)
};

struct estore;

struct estore * estore_open(const struct estore_cfg * cfg);
void            estore_close(struct estore * st);

int estore_n_layer(const struct estore * st);
int estore_n_expert(const struct estore * st, int layer);
int estore_layer_group(const struct estore * st, int layer);
int estore_n_groups(const struct estore * st);
int estore_n_slots(const struct estore * st, int group);
int estore_n_proj(const struct estore * st, int group);

const char * estore_proj_name(const struct estore * st, int group, int p);
const char * estore_proj_type(const struct estore * st, int group, int p);
uint64_t     estore_proj_size(const struct estore * st, int group, int p);   // bytes per slot
void *       estore_pool(const struct estore * st, int group, int p);        // CPU view
uint64_t     estore_pool_offset(const struct estore * st, int group, int p); // offset in the memfd
uint64_t     estore_pool_bytes(const struct estore * st, int group, int p);  // page-rounded pool size
// dma-buf over exactly pool (group, p), for zero-copy import by the GPU/TPU (offset 0 = slot 0);
// -1 without udmabuf. Owned by the store.
int          estore_pool_dmabuf_fd(const struct estore * st, int group, int p);
int          estore_dmabuf_ok(const struct estore * st);                     // every pool has a dma-buf

// Start loading (layer, expert) if it is not resident or loading. With
// prefetch_qd, reads beyond the cap are queued and issued as earlier
// prefetches complete; queued requests for layers at or before the last
// acquired layer are dropped. Returns 1 if already resident/in flight, 0 if a
// read was issued or queued, -1 if no slot could be freed without waiting.
int estore_prefetch(struct estore * st, int layer, int expert);

// Make experts[0..n) of `layer` resident and return their slots (in the
// layer's group). Blocks on I/O. The slots stay pinned until the next
// estore_acquire call. Returns 0, or -1 on error.
int estore_acquire(struct estore * st, int layer, int n, const int * experts, int * slots);

// Whether the coming acquires feed a GPU batch. With clean_slots, slots that landed outside a GPU batch are
// cleaned to DRAM only when a GPU batch acquires them (decode on the CPU then pays nothing for it).
void estore_set_gpu_batch(struct estore * st, int on);

// Is (layer, expert) in the cache? 2 = resident, 1 = read in flight, 0 = not. No I/O, no LRU update.
int estore_resident(const struct estore * st, int layer, int expert);

// Read whitespace-separated layer weights (one integer per layer) from a text file, e.g. the shares fitted by
// tools/sim/policy_lab.py alloc --save. Returns the count and a malloc'd array in *out, or -1 on error.
int estore_read_layer_weights(const char * path, int ** out);

void estore_get_stats(const struct estore * st, struct estore_stats * out);
void estore_reset_stats(struct estore * st);

#ifdef __cplusplus
}
#endif
