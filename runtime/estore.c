// estore: flash-streamed MoE expert cache. See estore.h.
#define _GNU_SOURCE
#include "estore.h"

#include <errno.h>
#include <fcntl.h>
#include <liburing.h>
#include <linux/udmabuf.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/uio.h>
#include <time.h>
#include <unistd.h>

enum { SLOT_FREE, SLOT_LOADING, SLOT_READY };

struct slot {
    int32_t      key;          // global expert key, -1 if free
    int32_t      layer;        // layer of key
    uint8_t      state;
    uint8_t      prefetched;   // loaded by prefetch, not yet acquired
    uint8_t      pf_io;        // a prefetch read is in flight into this slot
    uint8_t      in_buf;       // holds a prefetched expert not yet used (prefetch buffer, not a layer share)
    uint8_t      dirty;        // landed while no GPU batch was running: not yet cleaned to DRAM (clean_slots)
    uint32_t     pin_epoch;    // == st->epoch while pinned by the current acquire
    int32_t      prev, next;   // group LRU list (head = most recent)
    struct iovec iov[ESTORE_MAX_PROJ];
};

struct proj {
    char     name[16];
    char     type[16];
    uint64_t size;             // bytes per expert (= per slot)
    uint64_t bundle_off;       // offset inside the bundle
    uint64_t pool_off;         // offset of this pool inside the memfd
    uint64_t pool_bytes;       // page-rounded size of this pool
    int      dmabuf_fd;        // udmabuf over [pool_off, pool_off + pool_bytes), or -1
};

struct group {
    int          n_proj;
    struct proj  proj[ESTORE_MAX_PROJ];
    uint64_t     bundle_bytes; // sum of projection sizes (one preadv)
    uint64_t     all_bytes;    // bundle_bytes * experts of all member layers
    int          n_slots;
    struct slot *slots;
    int32_t      lru_head, lru_tail;
    int32_t     *free_list;
    int          n_free;
    int          pf_cap;       // prefetch buffer slots (0: prefetches go into the layer shares)
    int          pf_count;     // slots currently holding unused prefetches
};

struct estore {
    int       n_layer;
    int      *n_expert;        // [n_layer]
    uint64_t *layer_base;      // [n_layer] pack offset of expert 0
    uint64_t *layer_stride;    // [n_layer] bundle stride
    int      *layer_group;     // [n_layer]
    int      *key_base;        // [n_layer] first key of the layer
    int       n_keys;
    int32_t  *key_slot;        // [n_keys] slot in the layer's group, or -1

    int          n_groups;
    struct group groups[ESTORE_MAX_GROUPS];
    uint32_t     epoch;
    int          inflight;
    int          cur_layer;    // layer of the last acquire

    int          prefetch_qd;  // cap on prefetch reads in flight (0 = none)
    int          pf_inflight;
    int32_t     *pfq;          // software prefetch queue of keys (ring)
    int          pfq_cap, pfq_head, pfq_len;
    int         *key_layer;    // [n_keys]
    int          policy;
    int          clean_slots;  // clean each landed slot to DRAM (GPU reads on non-coherent paths)
    int          gpu_batch;    // the current batch reads the pools on the GPU (estore_set_gpu_batch)
    int          async_submit, pf_ioprio_idle;
    int         *layer_count;  // [n_layer] slots held by each layer
    int         *layer_quota;  // [n_layer] its share of its group's slots (LAYER policy)
    int          last_layer;   // layer of the previous acquire (trimmed at the next one)

    int      pack_fd;
    int      memfd;
    uint8_t *mem;
    uint64_t mem_bytes;

    struct io_uring ring;
    int             qd;

    struct estore_stats stats;
};

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t) ts.tv_sec * 1000000000ull + ts.tv_nsec;
}

static uint64_t align_up(uint64_t x, uint64_t a) { return (x + a - 1) / a * a; }

// Clean [p, p+n) from the CPU caches to the point of coherency (DRAM). On gs201 UFS DMA is coherent
// with the CPU cluster, so its writes can sit in CPU-side caches, while the G710 reads DRAM directly
// (it bypasses the CPU interconnect). A slot the GPU reads must be cleaned after its read lands.
// arm64 Linux allows DC CVAC from user space (SCTLR_EL1.UCI).
static void clean_to_poc(void * p, size_t n) {
#if defined(__aarch64__)
    uint64_t ctr;
    __asm__ volatile("mrs %0, ctr_el0" : "=r"(ctr));
    const size_t line = (size_t) 4 << ((ctr >> 16) & 0xf);   // DminLine
    for (uintptr_t a = (uintptr_t) p & ~(uintptr_t) (line - 1); a < (uintptr_t) p + n; a += line) {
        __asm__ volatile("dc cvac, %0" : : "r"(a) : "memory");
    }
    __asm__ volatile("dsb sy" : : : "memory");
#else
    (void) p; (void) n;
#endif
}

static void clean_slot(struct group * g, int s) {
    for (int p = 0; p < g->n_proj; ++p) clean_to_poc(g->slots[s].iov[p].iov_base, g->slots[s].iov[p].iov_len);
    g->slots[s].dirty = 0;
}

static int same_signature(const struct group * g, int n_proj, const struct proj * p) {
    if (g->n_proj != n_proj) return 0;
    for (int i = 0; i < n_proj; ++i) {
        if (strncmp(g->proj[i].name, p[i].name, sizeof(p[i].name)) || strncmp(g->proj[i].type, p[i].type, sizeof(p[i].type)) ||
            g->proj[i].size != p[i].size || g->proj[i].bundle_off != p[i].bundle_off) return 0;
    }
    return 1;
}

// ---- index ----------------------------------------------------------------

static int load_index(struct estore * st, const char * path) {
    FILE * f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "estore: cannot open index %s: %s\n", path, strerror(errno));
        return -1;
    }
    char tag[32] = "";
    int version = 0, layers_seen = 0;
    uint64_t bundle_align = 0;
    while (fscanf(f, "%31s", tag) == 1) {
        if (!strcmp(tag, "fmpack")) {
            if (fscanf(f, "%d", &version) != 1 || version != 2) goto bad;
        } else if (!strcmp(tag, "bundle_align")) {
            if (fscanf(f, "%lu", &bundle_align) != 1 || !bundle_align) goto bad;
        } else if (!strcmp(tag, "n_layer")) {
            if (fscanf(f, "%d", &st->n_layer) != 1 || st->n_layer <= 0) goto bad;
            st->n_expert     = calloc(st->n_layer, sizeof(int));
            st->layer_base   = calloc(st->n_layer, sizeof(uint64_t));
            st->layer_stride = calloc(st->n_layer, sizeof(uint64_t));
            st->layer_group  = calloc(st->n_layer, sizeof(int));
            st->key_base     = calloc(st->n_layer, sizeof(int));
        } else if (!strcmp(tag, "layer")) {
            int il, ne, np;
            uint64_t base, stride;
            if (!st->n_expert || !bundle_align ||
                fscanf(f, "%d %d %lu %lu %d", &il, &ne, &base, &stride, &np) != 5) goto bad;
            if (il < 0 || il >= st->n_layer || np < 1 || np > ESTORE_MAX_PROJ) goto bad;
            struct proj p[ESTORE_MAX_PROJ];
            uint64_t off = 0;
            for (int i = 0; i < np; ++i) {
                memset(&p[i], 0, sizeof(p[i]));
                if (fscanf(f, "%31s %15s %15s %lu %lu", tag, p[i].name, p[i].type, &p[i].size, &p[i].bundle_off) != 5 ||
                    strcmp(tag, "proj")) goto bad;
                // one O_DIRECT preadv must be able to scatter the bundle into the pools
                if (p[i].bundle_off != off || p[i].size % bundle_align) {
                    fprintf(stderr, "estore: layer %d %s is not scatter-readable (offset %lu, size %lu); "
                                    "bounce path not implemented\n", il, p[i].name, p[i].bundle_off, p[i].size);
                    fclose(f);
                    return -1;
                }
                off += p[i].size;
            }
            int g = 0;
            while (g < st->n_groups && g < ESTORE_MAX_GROUPS && !same_signature(&st->groups[g], np, p)) g++;
            if (g == st->n_groups) {
                if (g == ESTORE_MAX_GROUPS) { fprintf(stderr, "estore: too many layer groups\n"); fclose(f); return -1; }
                st->groups[g].n_proj = np;
                memcpy(st->groups[g].proj, p, sizeof(p[0]) * np);
                for (int i = 0; i < np; ++i) st->groups[g].proj[i].dmabuf_fd = -1;
                st->groups[g].bundle_bytes = off;
                st->n_groups++;
            }
            st->groups[g].all_bytes += off * (uint64_t) ne;
            st->layer_group[il] = g;
            st->n_expert[il] = ne;
            st->layer_base[il] = base;
            st->layer_stride[il] = stride;
            layers_seen++;
        } else {
            goto bad;
        }
    }
    fclose(f);
    if (layers_seen != st->n_layer) {
        fprintf(stderr, "estore: incomplete index %s (fmpack 2 required)\n", path);
        return -1;
    }
    for (int il = 0; il < st->n_layer; ++il) {
        st->key_base[il] = st->n_keys;
        st->n_keys += st->n_expert[il];
    }
    return 0;
bad:
    fprintf(stderr, "estore: malformed index %s near '%s'\n", path, tag);
    fclose(f);
    return -1;
}

// ---- LRU ------------------------------------------------------------------

static void lru_remove(struct group * g, int s) {
    struct slot * x = &g->slots[s];
    if (x->prev >= 0) g->slots[x->prev].next = x->next; else g->lru_head = x->next;
    if (x->next >= 0) g->slots[x->next].prev = x->prev; else g->lru_tail = x->prev;
    x->prev = x->next = -1;
}

static void lru_push_front(struct group * g, int s) {
    struct slot * x = &g->slots[s];
    x->prev = -1;
    x->next = g->lru_head;
    if (g->lru_head >= 0) g->slots[g->lru_head].prev = s;
    g->lru_head = s;
    if (g->lru_tail < 0) g->lru_tail = s;
}

// ---- I/O ------------------------------------------------------------------

static int reap(struct estore * st, int wait) {
    int n = 0;
    struct io_uring_cqe * cqe;
    for (;;) {
        int rc = (wait && n == 0) ? io_uring_wait_cqe(&st->ring, &cqe) : io_uring_peek_cqe(&st->ring, &cqe);
        if (rc == -EAGAIN || (rc == 0 && !cqe)) break;
        if (rc == -EINTR) continue;
        if (rc < 0) {
            fprintf(stderr, "estore: io_uring wait: %s\n", strerror(-rc));
            return -1;
        }
        const uint64_t ud = io_uring_cqe_get_data64(cqe);
        struct group * g = &st->groups[ud >> 32];
        const int s = (int) (uint32_t) ud;
        const int res = cqe->res;
        io_uring_cqe_seen(&st->ring, cqe);
        if (res != (int) g->bundle_bytes) {
            fprintf(stderr, "estore: read into group %d slot %d returned %d (want %lu): %s\n", (int) (ud >> 32), s,
                    res, g->bundle_bytes, res < 0 ? strerror(-res) : "short read");
            return -1;
        }
        if (st->clean_slots) {
            // clean now only if a GPU batch is running; otherwise defer to the GPU batch that acquires it
            if (st->gpu_batch) clean_slot(g, s); else g->slots[s].dirty = 1;
        }
        g->slots[s].state = SLOT_READY;
        if (g->slots[s].pf_io) {
            g->slots[s].pf_io = 0;
            st->pf_inflight--;
        }
        st->inflight--;
        st->stats.bytes_read += g->bundle_bytes;
        n++;
    }
    return n;
}

// I/O priority class values from linux/ioprio.h
#define ESTORE_IOPRIO_CLASS_IDLE 3
#define ESTORE_IOPRIO_CLASS_SHIFT 13

static int submit_read(struct estore * st, int gi, int s, int layer, int expert, int prefetch) {
    struct group * g = &st->groups[gi];
    struct slot * x = &g->slots[s];
    for (int p = 0; p < g->n_proj; ++p) {
        x->iov[p].iov_base = st->mem + g->proj[p].pool_off + (uint64_t) s * g->proj[p].size;
        x->iov[p].iov_len  = g->proj[p].size;
    }
    // never have more reads in flight than the queue depth: SQ slots free up at submit time, so
    // without this cap completions can outrun the CQ ring (2 x qd) and the kernel drops them (EBADR)
    while (st->inflight >= st->qd) {
        if (reap(st, 1) < 0) return -1;
    }
    struct io_uring_sqe * sqe;
    while (!(sqe = io_uring_get_sqe(&st->ring))) {
        io_uring_submit(&st->ring);
        if (reap(st, 1) < 0) return -1;
    }
    const uint64_t off = st->layer_base[layer] + (uint64_t) expert * st->layer_stride[layer];
    io_uring_prep_readv(sqe, st->pack_fd, x->iov, g->n_proj, off);
    io_uring_sqe_set_data64(sqe, ((uint64_t) gi << 32) | (uint32_t) s);
    if (st->async_submit) sqe->flags |= IOSQE_ASYNC;
    if (prefetch && st->pf_ioprio_idle) sqe->ioprio = ESTORE_IOPRIO_CLASS_IDLE << ESTORE_IOPRIO_CLASS_SHIFT;
    x->state = SLOT_LOADING;
    st->inflight++;
    const uint64_t t0 = now_ns();
    const int rc = io_uring_submit(&st->ring);
    st->stats.submit_ns += now_ns() - t0;
    return rc < 0 ? -1 : 0;
}

// Find a slot in group g: a free one, else the least recently used READY
// slot not pinned by the current acquire. With `wait`, blocks on in-flight
// reads when everything evictable is still loading.
static void evict_slot(struct estore * st, struct group * g, int s) {
    struct slot * x = &g->slots[s];
    st->key_slot[x->key] = -1;
    if (x->in_buf) {
        x->in_buf = 0;
        g->pf_count--;
    } else {
        st->layer_count[x->layer]--;
    }
    lru_remove(g, s);
    x->key = -1;
    x->state = SLOT_FREE;
    x->prefetched = 0;
}

static int evictable(const struct estore * st, const struct slot * x) {
    return x->state == SLOT_READY && x->pin_epoch != st->epoch;
}

// LAYER policy: shrink a finished layer back to its share, oldest first, freeing the extra slots
static void trim_layer(struct estore * st, int layer) {
    struct group * g = &st->groups[st->layer_group[layer]];
    for (int s = g->lru_tail; s >= 0 && st->layer_count[layer] > st->layer_quota[layer];) {
        const int prev = g->slots[s].prev;
        if (g->slots[s].layer == layer && !g->slots[s].in_buf && evictable(st, &g->slots[s])) {
            evict_slot(st, g, s);
            g->free_list[g->n_free++] = s;
        }
        s = prev;
    }
}

// Find a slot in group g for layer `layer`: a free one; else (LAYER) the layer's own oldest expert,
// else the newest expert of another layer (in a layer-by-layer walk it is needed furthest ahead);
// (LRU) the oldest evictable expert. With `wait`, blocks on in-flight reads when nothing is evictable.
static int alloc_slot(struct estore * st, struct group * g, int layer, int wait) {
    for (;;) {
        if (g->n_free > 0) return g->free_list[--g->n_free];
        int victim = -1;
        if (st->policy == ESTORE_POLICY_LAYER) {
            for (int s = g->lru_tail; s >= 0 && victim < 0; s = g->slots[s].prev) {
                if (g->slots[s].layer == layer && !g->slots[s].in_buf && evictable(st, &g->slots[s])) victim = s;
            }
            for (int s = g->lru_tail; s >= 0 && victim < 0; s = g->slots[s].prev) {   // oldest unused prefetch
                if (g->slots[s].in_buf && evictable(st, &g->slots[s])) victim = s;
            }
            for (int s = g->lru_head; s >= 0 && victim < 0; s = g->slots[s].next) {
                if (g->slots[s].layer != layer && evictable(st, &g->slots[s])) victim = s;
            }
        } else {
            for (int s = g->lru_tail; s >= 0 && victim < 0; s = g->slots[s].prev) {
                if (evictable(st, &g->slots[s])) victim = s;
            }
        }
        if (victim >= 0) {
            evict_slot(st, g, victim);
            return victim;
        }
        if (!wait || st->inflight == 0) return -1;
        if (reap(st, 1) < 0) return -1;
    }
}

// prefetch buffer slot: a free slot, else the oldest unused prefetch; never a layer's cached expert
static int alloc_buf_slot(struct estore * st, struct group * g) {
    if (g->pf_count < g->pf_cap && g->n_free > 0) return g->free_list[--g->n_free];
    for (int s = g->lru_tail; s >= 0; s = g->slots[s].prev) {
        if (g->slots[s].in_buf && evictable(st, &g->slots[s])) {
            evict_slot(st, g, s);
            return s;
        }
    }
    return -1;
}

static int claim(struct estore * st, int key, int layer, int expert, int wait, int buffered) {
    const int gi = st->layer_group[layer];
    struct group * g = &st->groups[gi];
    int s = buffered ? alloc_buf_slot(st, g) : alloc_slot(st, g, layer, wait);
    if (s < 0) return -1;
    g->slots[s].key = key;
    g->slots[s].layer = layer;
    if (buffered) {
        g->slots[s].in_buf = 1;
        g->pf_count++;
    } else {
        st->layer_count[layer]++;
    }
    st->key_slot[key] = s;
    lru_push_front(g, s);
    if (submit_read(st, gi, s, layer, expert, !wait) < 0) return -2;   // only demand reads wait
    return s;
}

// ---- API ------------------------------------------------------------------

struct estore * estore_open(const struct estore_cfg * cfg) {
    struct estore * st = calloc(1, sizeof(*st));
    st->memfd = st->pack_fd = -1;
    if (load_index(st, cfg->index) < 0) goto fail;

    // every group caches the same fraction of its experts
    uint64_t all = 0;
    for (int g = 0; g < st->n_groups; ++g) all += st->groups[g].all_bytes;
    const double frac = (double) cfg->pool_bytes / (double) all;
    const uint64_t page = cfg->hugetlb ? (2u << 20) : 4096;
    for (int gi = 0; gi < st->n_groups; ++gi) {
        struct group * g = &st->groups[gi];
        g->n_slots = (int) (frac * (double) (g->all_bytes / g->bundle_bytes));
        const int all_slots = (int) (g->all_bytes / g->bundle_bytes);
        // a prefill ubatch can need every expert of one layer at once: never give a group less
        // than its widest layer, even if that exceeds its proportional share of pool_bytes
        int widest = 0;
        for (int il = 0; il < st->n_layer; ++il) {
            if (st->layer_group[il] == gi && st->n_expert[il] > widest) widest = st->n_expert[il];
        }
        if (g->n_slots < widest) g->n_slots = widest;
        if (g->n_slots > all_slots) g->n_slots = all_slots;
        if (g->n_slots < 1) {
            fprintf(stderr, "estore: pool_bytes %lu leaves group %d without slots\n", cfg->pool_bytes, gi);
            goto fail;
        }
        for (int p = 0; p < g->n_proj; ++p) {
            g->proj[p].pool_off = st->mem_bytes;
            g->proj[p].pool_bytes = align_up((uint64_t) g->n_slots * g->proj[p].size, page);
            g->proj[p].dmabuf_fd = -1;
            st->mem_bytes += g->proj[p].pool_bytes;
        }
    }

    st->memfd = memfd_create("estore-pools", MFD_ALLOW_SEALING | MFD_CLOEXEC | (cfg->hugetlb ? MFD_HUGETLB : 0));
    if (st->memfd < 0 || ftruncate(st->memfd, (off_t) st->mem_bytes) < 0) {
        fprintf(stderr, "estore: memfd (%lu bytes%s): %s\n", st->mem_bytes, cfg->hugetlb ? ", hugetlb" : "", strerror(errno));
        goto fail;
    }
    st->mem = mmap(NULL, st->mem_bytes, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, st->memfd, 0);
    if (st->mem == MAP_FAILED) {
        st->mem = NULL;
        fprintf(stderr, "estore: mmap pools (%lu bytes%s): %s\n", st->mem_bytes, cfg->hugetlb ? ", hugetlb" : "", strerror(errno));
        goto fail;
    }

    // export each pool's pages as its own dma-buf for GPU/TPU import (optional)
    if (fcntl(st->memfd, F_ADD_SEALS, F_SEAL_SHRINK) == 0) {
        int dev = open("/dev/udmabuf", O_RDWR | O_CLOEXEC);
        int n_fail = 0, err = 0;
        uint64_t max_failed = 0;
        for (int gi = 0; dev >= 0 && gi < st->n_groups; ++gi) {
            struct group * g = &st->groups[gi];
            for (int p = 0; p < g->n_proj; ++p) {
                struct udmabuf_create c = { .memfd = (uint32_t) st->memfd, .flags = UDMABUF_FLAGS_CLOEXEC,
                                            .offset = g->proj[p].pool_off, .size = g->proj[p].pool_bytes };
                g->proj[p].dmabuf_fd = ioctl(dev, UDMABUF_CREATE, &c);
                if (g->proj[p].dmabuf_fd < 0) {
                    n_fail++;
                    err = errno;
                    if (g->proj[p].pool_bytes > max_failed) max_failed = g->proj[p].pool_bytes;
                }
            }
        }
        if (n_fail) {
            // EINVAL here is usually udmabuf's size_limit_mb (64 MB on older kernels)
            fprintf(stderr, "estore: no dma-buf for %d pools (largest %lu MB): %s; streaming continues without zero-copy export\n",
                    n_fail, max_failed >> 20, strerror(err));
        }
        if (dev >= 0) close(dev);
    }

    st->pack_fd = open(cfg->pack, O_RDONLY | O_DIRECT | O_CLOEXEC);
    if (st->pack_fd < 0) {
        fprintf(stderr, "estore: open %s (O_DIRECT): %s\n", cfg->pack, strerror(errno));
        goto fail;
    }
    st->qd = cfg->qd > 0 ? cfg->qd : 16;
    int rc = io_uring_queue_init((unsigned) st->qd, &st->ring, 0);
    if (rc < 0) {
        fprintf(stderr, "estore: io_uring_queue_init: %s\n", strerror(-rc));
        st->qd = 0;
        goto fail;
    }

    st->key_slot = malloc(sizeof(int32_t) * st->n_keys);
    for (int k = 0; k < st->n_keys; ++k) st->key_slot[k] = -1;
    st->key_layer = malloc(sizeof(int) * st->n_keys);
    for (int il = 0; il < st->n_layer; ++il) {
        for (int e = 0; e < st->n_expert[il]; ++e) st->key_layer[st->key_base[il] + e] = il;
    }
    st->prefetch_qd = cfg->prefetch_qd;
    st->policy = cfg->policy;
    {   // prefetch buffer, split across groups in proportion to their slots
        int total = 0;
        for (int gi = 0; gi < st->n_groups; ++gi) total += st->groups[gi].n_slots;
        for (int gi = 0; gi < st->n_groups; ++gi) {
            st->groups[gi].pf_cap = total ? (int) ((int64_t) cfg->prefetch_buffer * st->groups[gi].n_slots / total) : 0;
            st->groups[gi].pf_count = 0;
        }
    }
    st->clean_slots = cfg->clean_slots;
    st->async_submit = cfg->async_submit;
    st->pf_ioprio_idle = cfg->pf_ioprio_idle;
    if (st->n_layer <= 0) goto fail;   // load_index guarantees this; tells the compiler too
    st->layer_count = calloc((size_t) st->n_layer, sizeof(int));
    st->layer_quota = calloc((size_t) st->n_layer, sizeof(int));
    if (cfg->layer_weights && cfg->n_layer_weights != st->n_layer) {
        fprintf(stderr, "estore: %d layer weights for %d layers\n", cfg->n_layer_weights, st->n_layer);
        goto fail;
    }
    for (int gi = 0; gi < st->n_groups; ++gi) {
        // keep a reserve for the layer being computed (top-k in decode; prefill can overflow it, then
        // alloc_slot takes other layers' newest experts)
        const int reserve = 16;
        const int budget = st->groups[gi].n_slots - reserve - st->groups[gi].pf_cap;
        // split the group's budget in proportion to the layer weights (equal without them); the
        // slots left after rounding down go to the largest remainders
        int64_t wsum = 0;
        for (int il = 0; il < st->n_layer; ++il) {
            if (st->layer_group[il] == gi) wsum += cfg->layer_weights ? cfg->layer_weights[il] : 1;
        }
        if (wsum <= 0) wsum = 1;
        int used = 0;
        for (int il = 0; il < st->n_layer; ++il) {
            if (st->layer_group[il] != gi) continue;
            const int64_t w = cfg->layer_weights ? cfg->layer_weights[il] : 1;
            st->layer_quota[il] = budget > 0 ? (int) (w * budget / wsum) : 0;
            used += st->layer_quota[il];
        }
        while (budget > 0 && used < budget) {
            int best = -1;
            int64_t best_rem = -1;
            for (int il = 0; il < st->n_layer; ++il) {
                if (st->layer_group[il] != gi) continue;
                const int64_t w = cfg->layer_weights ? cfg->layer_weights[il] : 1;
                const int64_t rem = w * budget - (int64_t) st->layer_quota[il] * wsum;
                if (rem > best_rem) { best_rem = rem; best = il; }
            }
            if (best < 0) break;
            st->layer_quota[best]++;
            used++;
        }
        for (int il = 0; il < st->n_layer; ++il) {
            if (st->layer_group[il] == gi && st->layer_quota[il] < 1) st->layer_quota[il] = 1;
        }
    }
    st->last_layer = -1;
    st->pfq_cap = 1024;
    st->pfq = malloc(sizeof(int32_t) * st->pfq_cap);
    st->cur_layer = -1;
    for (int gi = 0; gi < st->n_groups; ++gi) {
        struct group * g = &st->groups[gi];
        g->slots = calloc(g->n_slots, sizeof(struct slot));
        g->free_list = malloc(sizeof(int32_t) * g->n_slots);
        for (int s = 0; s < g->n_slots; ++s) {
            g->slots[s].key = -1;
            g->slots[s].prev = g->slots[s].next = -1;
            g->free_list[g->n_free++] = g->n_slots - 1 - s;   // hand out slot 0 first
        }
        g->lru_head = g->lru_tail = -1;
    }
    st->epoch = 1;
    return st;
fail:
    estore_close(st);
    return NULL;
}

void estore_close(struct estore * st) {
    if (!st) return;
    if (st->qd) {
        while (st->inflight > 0 && reap(st, 1) >= 0) {}
        io_uring_queue_exit(&st->ring);
    }
    if (st->pack_fd >= 0) close(st->pack_fd);
    for (int g = 0; g < st->n_groups; ++g) {
        for (int p = 0; p < st->groups[g].n_proj; ++p) {
            if (st->groups[g].proj[p].dmabuf_fd >= 0) close(st->groups[g].proj[p].dmabuf_fd);
        }
    }
    if (st->mem) munmap(st->mem, st->mem_bytes);
    if (st->memfd >= 0) close(st->memfd);
    for (int g = 0; g < st->n_groups; ++g) {
        free(st->groups[g].slots);
        free(st->groups[g].free_list);
    }
    free(st->n_expert); free(st->layer_base); free(st->layer_stride); free(st->layer_group);
    free(st->key_base); free(st->key_slot); free(st->key_layer); free(st->pfq);
    free(st->layer_count); free(st->layer_quota);
    free(st);
}

int estore_n_layer(const struct estore * st) { return st->n_layer; }
int estore_n_expert(const struct estore * st, int layer) { return st->n_expert[layer]; }
int estore_layer_group(const struct estore * st, int layer) { return st->layer_group[layer]; }
int estore_n_groups(const struct estore * st) { return st->n_groups; }
int estore_n_slots(const struct estore * st, int g) { return st->groups[g].n_slots; }
int estore_n_proj(const struct estore * st, int g) { return st->groups[g].n_proj; }
const char * estore_proj_name(const struct estore * st, int g, int p) { return st->groups[g].proj[p].name; }
const char * estore_proj_type(const struct estore * st, int g, int p) { return st->groups[g].proj[p].type; }
uint64_t     estore_proj_size(const struct estore * st, int g, int p) { return st->groups[g].proj[p].size; }
void *       estore_pool(const struct estore * st, int g, int p) { return st->mem + st->groups[g].proj[p].pool_off; }
uint64_t     estore_pool_offset(const struct estore * st, int g, int p) { return st->groups[g].proj[p].pool_off; }
uint64_t     estore_pool_bytes(const struct estore * st, int g, int p) { return st->groups[g].proj[p].pool_bytes; }
int          estore_pool_dmabuf_fd(const struct estore * st, int g, int p) { return st->groups[g].proj[p].dmabuf_fd; }
int estore_dmabuf_ok(const struct estore * st) {
    for (int g = 0; g < st->n_groups; ++g) {
        for (int p = 0; p < st->groups[g].n_proj; ++p) {
            if (st->groups[g].proj[p].dmabuf_fd < 0) return 0;
        }
    }
    return st->n_groups > 0;
}

static int issue_prefetch(struct estore * st, int key, int layer, int expert) {
    const int buffered = st->groups[st->layer_group[layer]].pf_cap > 0;
    const int s = claim(st, key, layer, expert, 0, buffered);
    if (s < 0) return -1;
    struct slot * x = &st->groups[st->layer_group[layer]].slots[s];
    x->prefetched = 1;
    x->pf_io = 1;
    st->pf_inflight++;
    st->stats.prefetches++;
    return 0;
}

// issue queued prefetches while under the cap; drop stale ones
static void pump_prefetch(struct estore * st) {
    while (st->pfq_len > 0 && (!st->prefetch_qd || st->pf_inflight < st->prefetch_qd)) {
        const int key = st->pfq[st->pfq_head];
        st->pfq_head = (st->pfq_head + 1) % st->pfq_cap;
        st->pfq_len--;
        const int layer = st->key_layer[key];
        if (st->key_slot[key] >= 0) continue;
        if (layer <= st->cur_layer) { st->stats.prefetch_dropped++; continue; }
        if (issue_prefetch(st, key, layer, key - st->key_base[layer]) < 0) break;
    }
}

int estore_prefetch(struct estore * st, int layer, int expert) {
    if (reap(st, 0) < 0) return -1;
    pump_prefetch(st);
    const int key = st->key_base[layer] + expert;
    if (st->key_slot[key] >= 0) return 1;
    if (!st->prefetch_qd || st->pf_inflight < st->prefetch_qd) {
        return issue_prefetch(st, key, layer, expert);
    }
    if (st->pfq_len == st->pfq_cap) {      // full: drop the oldest request
        st->pfq_head = (st->pfq_head + 1) % st->pfq_cap;
        st->pfq_len--;
        st->stats.prefetch_dropped++;
    }
    st->pfq[(st->pfq_head + st->pfq_len++) % st->pfq_cap] = key;
    return 0;
}

void estore_set_gpu_batch(struct estore * st, int on) { st->gpu_batch = on; }

int estore_resident(const struct estore * st, int layer, int expert) {
    const int s = st->key_slot[st->key_base[layer] + expert];
    if (s < 0) return 0;
    return st->groups[st->layer_group[layer]].slots[s].state == SLOT_READY ? 2 : 1;
}

int estore_acquire(struct estore * st, int layer, int n, const int * experts, int * slots) {
    if (reap(st, 0) < 0) return -1;
    struct group * g = &st->groups[st->layer_group[layer]];
    st->epoch++;                 // unpins the previous acquire's slots
    if (st->policy == ESTORE_POLICY_LAYER && st->last_layer >= 0 && st->last_layer != layer) {
        trim_layer(st, st->last_layer);
    }
    st->last_layer = layer;
    st->cur_layer = layer;
    // reverse router order: the top-ranked expert is inserted last, so it is the layer's newest
    for (int i = n - 1; i >= 0; --i) {
        const int key = st->key_base[layer] + experts[i];
        int s = st->key_slot[key];
        if (s >= 0) {
            struct slot * x = &g->slots[s];
            if (x->state == SLOT_READY) st->stats.hits++; else st->stats.inflight_hits++;
            if (x->prefetched) { st->stats.prefetch_used++; x->prefetched = 0; }
            if (x->in_buf) {        // a used prefetch joins its layer's share
                x->in_buf = 0;
                g->pf_count--;
                st->layer_count[layer]++;
            }
            lru_remove(g, s);
            lru_push_front(g, s);
        } else {
            s = claim(st, key, layer, experts[i], 1, 0);
            if (s < 0) {
                fprintf(stderr, "estore: no slot for layer %d expert %d (%d slots in group < working set?)\n",
                        layer, experts[i], g->n_slots);
                return -1;
            }
            st->stats.misses++;
        }
        g->slots[s].pin_epoch = st->epoch;
        slots[i] = s;
    }
    uint64_t t0 = 0;
    for (int i = 0; i < n; ++i) {
        while (g->slots[slots[i]].state != SLOT_READY) {
            if (!t0) t0 = now_ns();
            if (reap(st, 1) < 0) return -1;
        }
    }
    if (t0) st->stats.stall_ns += now_ns() - t0;
    if (st->clean_slots && st->gpu_batch) {
        for (int i = 0; i < n; ++i) if (g->slots[slots[i]].dirty) clean_slot(g, slots[i]);
    }
    pump_prefetch(st);
    return 0;
}

void estore_get_stats(const struct estore * st, struct estore_stats * out) { *out = st->stats; }
void estore_reset_stats(struct estore * st) { memset(&st->stats, 0, sizeof(st->stats)); }

int estore_read_layer_weights(const char * path, int ** out) {
    FILE * f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "estore: cannot read layer weights %s: %s\n", path, strerror(errno));
        return -1;
    }
    int n = 0, cap = 128, v;
    int * w = malloc(sizeof(int) * (size_t) cap);
    while (fscanf(f, "%d", &v) == 1) {
        if (n == cap) w = realloc(w, sizeof(int) * (size_t) (cap *= 2));
        w[n++] = v < 0 ? 0 : v;
    }
    const int bad = !feof(f);
    fclose(f);
    if (bad || n == 0) {
        fprintf(stderr, "estore: %s: expected one integer per layer\n", path);
        free(w);
        return -1;
    }
    *out = w;
    return n;
}
