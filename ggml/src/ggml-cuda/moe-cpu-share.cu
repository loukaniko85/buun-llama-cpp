#include "moe-cpu-share.cuh"
#include "moe-weighted-reduction.cuh"

#if defined(GGML_USE_MUSA)

void ggml_moe_cpu_share_note(const ggml_tensor *, const void *, const void *) {}

ggml_moe_cpu_share_args ggml_moe_cpu_share_begin(const ggml_tensor *, const ggml_tensor *, const ggml_tensor *,
        const ggml_tensor *, const ggml_cuda_mm_fusion_args_host *, cudaStream_t) {
    return {};
}

ggml_moe_cpu_share_args ggml_moe_cpu_share_down(const ggml_tensor *, const ggml_tensor *, const ggml_tensor *) {
    return {};
}

bool ggml_moe_cpu_share_hoist(int64_t) {
    return false;
}

ggml_moe_cpu_share_args ggml_moe_cpu_share_merge(const ggml_tensor *) {
    return {};
}

#else

#include "moe-cache.cuh"
#include "ggml-cpu.h"
#include "../ggml-backend-moe-cache.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define SHARE_PAUSE() _mm_pause()
#else
#define SHARE_PAUSE() std::this_thread::yield()
#endif

static constexpr int share_max_entries = 64;   // routed (token, expert) pairs per layer
static constexpr int share_max_tokens  = 8;
static constexpr int share_max_experts = 16;   // host experts per layer
static constexpr int share_max_routes  = 8192; // ids index space: row stride * tokens
static constexpr int share_max_layers  = 512;
static constexpr int share_max_staged  = 8;    // GPU-streamed experts staged in VRAM per layer
// Each device's tickets stay in their own residue class, so devices that take turns on the
// one mailbox never repeat each other's ticket.
static constexpr uint32_t share_ticket_step = GGML_CUDA_MAX_DEVICES;
static_assert((share_ticket_step & (share_ticket_step - 1)) == 0, "ticket step must divide 2^32");

// Mapped host memory. The GPU writes the job then seq; the host writes y then done.
struct share_mailbox {
    volatile uint32_t seq;
    uint32_t pad0[31];
    volatile uint32_t done;
    uint32_t pad1[31];
    volatile float stage_cost;                  // GPU-set: timer ticks per byte of the last staging copy
    volatile int16_t host_num[GGML_TYPE_COUNT]; // host-set: host share of a layer's misses by up type, in 1/256
    int32_t layer;
    int32_t n_tok;
    int32_t n_expert;
    int32_t n_entry;
    float glu_limit; // zero for plain SwiGLU, positive for SWIGLU_CLAMP
    int32_t expert[share_max_experts];
    int32_t entry_expert[share_max_entries]; // index into expert
    int32_t entry_tok[share_max_entries];
};

struct share_copy {
    const uint4 * src;
    uint4 * dst;
    size_t n16;
};

// Device memory, written by the doorbell and read by the stage copy and the matvecs.
struct share_state {
    uint32_t ticket;
    uint8_t skip[share_max_routes];
    // [up, gate, down][route]: staged weights of a GPU-streamed miss, else null
    const void * stage[3][share_max_routes];
    int n_copy;
    share_copy copy[3*share_max_staged];
    uint64_t t_ring;            // timer when the copies were set up
    unsigned int n_copy_blocks; // stage copy blocks finished
    uint32_t n_rings;           // layers with misses
};

static __device__ __forceinline__ uint64_t share_ticks() {
#if defined(GGML_USE_HIP)
    return wall_clock64();
#else
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
#endif
}

struct share_tensor {
    const char * host = nullptr;
    const char * device = nullptr;
    size_t span = 0;
    size_t nb1 = 0;
    size_t nb2 = 0;
    int64_t ne0 = 0;
    int64_t ne1 = 0;
    ggml_type type = GGML_TYPE_COUNT;
};

struct share_layer {
    share_tensor up;
    share_tensor gate;
    share_tensor down;
    // the session whose block this is, and one of its routed tensors to tell whether it still lives
    const void * owner = nullptr;
    const char * owner_host = nullptr;
};

struct share_doorbell_args {
    const int32_t * ids;
    int n_k;
    int n_tok;
    int ids_stride;
    const float * x;
    int64_t x_stride;
    int n_embd;
    const void * const * table[3];
    const char * base[3];
    size_t span[3];
    size_t nb2[3];
    char * stage;          // share_max_staged slots of 3 tensors x stage_stride bytes, or null
    size_t stage_stride;
    int layer;
    int up_type;
    bool probe;            // the share is balanced, not pinned
    float glu_limit;
    share_mailbox * mb;
    float * mb_x;
    share_state * st;
};

static __device__ __forceinline__ bool share_is_host(const void * const * table, const char * base, size_t span, int expert) {
    return (size_t)((const char *) table[expert] - base) < span;
}

static __device__ __forceinline__ const void * share_stage_ptr(const share_doorbell_args & a, int i, int expert, int slot) {
    return slot >= 0 && a.nb2[i] <= a.stage_stride && share_is_host(a.table[i], a.base[i], a.span[i], expert)
        ? a.stage + ((size_t) slot*3 + i)*a.stage_stride : nullptr;
}

static __global__ void share_doorbell(const share_doorbell_args a) {
    __shared__ int s_expert[share_max_entries];
    __shared__ int s_score[share_max_entries];
    __shared__ int s_pick[share_max_entries];
    __shared__ int s_stage[share_max_entries];
    __shared__ int s_n_entry;

    const int n = a.n_k*a.n_tok;
    const int r = threadIdx.x;
    int score = 0;
    if (r < n) {
        const int expert = a.ids[r % a.n_k + (r / a.n_k)*a.ids_stride];
        for (int i = 0; i < 3 && expert >= 0; ++i) {
            score += share_is_host(a.table[i], a.base[i], a.span[i], expert);
        }
        s_expert[r] = expert;
        s_score[r] = score;
    }
    if (!__syncthreads_or(score)) {
        // all hits, the common case once the cache is warm
        if (r < n) {
            const int route = r % a.n_k + (r / a.n_k)*a.ids_stride;
            a.st->skip[route] = 0;
            for (int i = 0; i < 3; ++i) {
                a.st->stage[i][route] = nullptr;
            }
        }
        if (r == 0) {
            a.st->n_copy = 0;
        }
        return;
    }

    // The previous job is finished: its weighted reduction waited for the host.
    if (threadIdx.x == 0) {
        share_mailbox * mb = a.mb;
        share_state * st = a.st;
        const int host_num = mb->host_num[a.up_type]; // a bus read, issued before the dedup to overlap it
        int miss_expert[share_max_entries];
        int miss_score[share_max_entries];
        int n_miss = 0;
        for (int i = 0; i < n; ++i) {
            if (!s_score[i]) {
                continue;
            }
            bool seen = false;
            for (int j = 0; j < n_miss && !seen; ++j) {
                seen = miss_expert[j] == s_expert[i];
            }
            if (!seen) {
                miss_expert[n_miss] = s_expert[i];
                miss_score[n_miss] = s_score[i];
                n_miss++;
            }
        }
        // The host takes the experts with the most bytes on the bus first.
        int n_host = min((n_miss*host_num + 128) >> 8, share_max_experts);
        // A side that gets no misses is never measured again, so now and then it gets one.
        if (a.probe && (++st->n_rings & 63) == 0) {
            if (n_host == 0) {
                n_host = 1;
            } else if (n_host == n_miss) {
                n_host--;
            }
        }
        int picked[share_max_experts];
        int n_pick = 0;
        for (int score = 3; score > 0 && n_pick < n_host; --score) {
            for (int j = 0; j < n_miss && n_pick < n_host; ++j) {
                if (miss_score[j] == score) {
                    mb->expert[n_pick] = miss_expert[j];
                    picked[n_pick++] = miss_expert[j];
                }
            }
        }
        int n_entry = 0;
        for (int i = 0; i < n; ++i) {
            int pick = -1;
            for (int j = 0; j < n_pick; ++j) {
                if (picked[j] == s_expert[i]) {
                    pick = j;
                }
            }
            s_pick[i] = pick < 0 ? -1 : n_entry;
            if (pick >= 0) {
                mb->entry_expert[n_entry] = pick;
                mb->entry_tok[n_entry] = i / a.n_k;
                n_entry++;
            }
        }
        if (n_entry) {
            st->ticket += share_ticket_step;
            mb->layer = a.layer;
            mb->n_tok = a.n_tok;
            mb->n_expert = n_pick;
            mb->n_entry = n_entry;
            mb->glu_limit = a.glu_limit;
        }
        s_n_entry = n_entry;

        // The GPU's misses are copied once each into VRAM slots with wide loads, instead of
        // every routed matvec block reading host memory through narrow zero-copy loads.
        int staged[share_max_staged];
        int n_staged = 0;
        int n_copy = 0;
        for (int j = 0; j < n_miss && a.stage && n_staged < share_max_staged; ++j) {
            bool host = false;
            for (int p = 0; p < n_pick && !host; ++p) {
                host = picked[p] == miss_expert[j];
            }
            if (host) {
                continue;
            }
            for (int i = 0; i < 3; ++i) {
                const void * dst = share_stage_ptr(a, i, miss_expert[j], n_staged);
                if (dst) {
                    st->copy[n_copy++] = { (const uint4 *) a.table[i][miss_expert[j]], (uint4 *) dst, a.nb2[i]/16 };
                }
            }
            staged[n_staged++] = miss_expert[j];
        }
        st->n_copy = n_copy;
        st->t_ring = share_ticks();
        for (int i = 0; i < n; ++i) {
            s_stage[i] = -1;
            for (int j = 0; j < n_staged && s_pick[i] < 0; ++j) {
                if (staged[j] == s_expert[i]) {
                    s_stage[i] = j;
                }
            }
        }
    }
    __syncthreads();

    if (r < n) {
        const int route = r % a.n_k + (r / a.n_k)*a.ids_stride;
        a.st->skip[route] = s_pick[r] + 1;
        for (int i = 0; i < 3; ++i) {
            a.st->stage[i][route] = share_stage_ptr(a, i, s_expert[r], s_stage[r]);
        }
    }
    if (!s_n_entry) {
        return;
    }
    for (int i = threadIdx.x; i < a.n_tok*a.n_embd; i += blockDim.x) {
        a.mb_x[i] = a.x[(i / a.n_embd)*a.x_stride + i % a.n_embd];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence_system();
        a.mb->seq = a.st->ticket;
    }
}

// Copies the doorbell's staged experts; a no-op launch when the layer streams nothing.
static __global__ void share_stage_copy(share_state * st, share_mailbox * mb) {
    const int n_copy = st->n_copy;
    if (!n_copy) {
        return;
    }
    const size_t step = (size_t) gridDim.x*blockDim.x;
    for (int c = 0; c < n_copy; ++c) {
        const uint4 * __restrict__ src = st->copy[c].src;
        uint4 * __restrict__ dst = st->copy[c].dst;
        const size_t n16 = st->copy[c].n16;
        for (size_t i = blockIdx.x*blockDim.x + threadIdx.x; i < n16; i += step) {
            dst[i] = src[i];
        }
    }
    // the last block to finish times the copy, the bus speed the host balances its share against
    if (ggml_cuda_block_ticket(&st->n_copy_blocks) == gridDim.x - 1 && threadIdx.x == 0) {
        size_t n16 = 0;
        for (int c = 0; c < n_copy; ++c) {
            n16 += st->copy[c].n16;
        }
        mb->stage_cost = (float) (share_ticks() - st->t_ring)/(16.0f*n16);
    }
}

// Per-device state: devices of a layer split take turns on the host share, one layer at a time.
struct share_dev {
    share_state * st = nullptr;
    char * stage = nullptr;
    size_t stage_stride = 0;
};

struct share_host {
    std::mutex mu;
    std::unordered_map<const void *, std::pair<int, share_tensor>> tensors; // host base -> (layer, tensor)
    share_layer layers[share_max_layers];

    share_mailbox * mb = nullptr;   // host view
    share_mailbox * d_mb = nullptr; // device view
    float * x = nullptr;
    float * y = nullptr;
    float * d_x = nullptr;
    float * d_y = nullptr;
    share_dev devs[GGML_CUDA_MAX_DEVICES];
    int n_embd = 0;
    int n_threads = 0;

    double ns_per_tick = 1.0; // of the GPU timer
    // leader-only: measured ns per byte of the staging copy and of a host job by up type, 0 = unknown
    double bus_cost = 0.0;
    double host_cost[GGML_TYPE_COUNT] = {};

    // launch-time pairing of a layer's gate/up and down
    const void * pending_ids = nullptr;
    int pending_device = -1;
    int pending_layer = -1;
    int pending_ids_stride = 0;
    // launch-time pairing of a layer's down and the weighted reduction that merges it
    const void * merge_experts = nullptr;

    // The running job, published by the leader through job. Each phase hands out chunks
    // tagged with the job (job << 32 | chunk), so a worker that wakes late takes fewer
    // chunks and never touches a newer job's state.
    std::atomic<uint32_t> job{0};
    std::atomic<uint64_t> next[3];
    std::atomic<uint64_t> done[3];
    int n_chunk[3] = {};
    std::mutex wake_mu;
    std::condition_variable wake_cv;
    std::atomic<int> sleepers{0};
    share_layer cur;
    const ggml_type_traits_cpu * t_up = nullptr;
    const ggml_type_traits_cpu * t_down = nullptr;
    int n_tok = 0;
    int n_expert = 0;
    int n_entry = 0;
    float glu_limit = 0.0f;
    int expert[share_max_experts];
    int entry_tok[share_max_entries];
    std::vector<int> expert_entries[share_max_experts];
    size_t xq_row = 0;
    size_t hq_row = 0;
    std::vector<char> xq;
    std::vector<float> gu;   // [entry][2][n_ff]
    std::vector<char> hq;
};

// Never destroyed: detached workers may still wait on its condvar at exit, and
// destroying a waited-on condvar blocks forever in glibc.
static share_host & g_share = *new share_host;

// Fraction of a layer's routed misses the GPU streams itself: GGML_CUDA_MOE_CPU_SHARE pins it,
// else it starts at 0.4 and the leader balances it by measured speeds; -1 when the share is off
// (a fraction of 1 or more).
static bool share_pinned() {
    static const bool pinned = getenv("GGML_CUDA_MOE_CPU_SHARE") != nullptr;
    return pinned;
}

static float share_gpu_fraction() {
    static const float frac = [] {
        const char * value = getenv("GGML_CUDA_MOE_CPU_SHARE");
        const float f = value ? fmaxf((float) atof(value), 0.0f) : 0.4f;
        return f >= 1.0f ? -1.0f : f;
    }();
    return frac;
}

static constexpr int64_t share_up_chunk = 64;    // up/gate rows per chunk
static constexpr int64_t share_down_chunk = 256; // down rows per chunk

// Rows [row, row + n) of host expert ei's matrix t, through the CPU row kernel, up to
// 4 entries at a time: dst(entry) returns the entry's output row and act(entry) its activation.
template <typename D, typename A>
static void share_rows(share_host & h, const share_tensor & t, int ei, int64_t row, int64_t n, D dst, A act) {
    const std::vector<int> & entries = h.expert_entries[ei];
    for (size_t first = 0; first < entries.size(); first += 4) {
        const int nr = (int) std::min<size_t>(4, entries.size() - first);
        float * out[4];
        const void * in[4];
        for (int r = 0; r < nr; ++r) {
            out[r] = dst(entries[first + r]) + row;
            in[r] = act(entries[first + r]);
        }
        ggml_moe_cache_cpu_rows(t.type, (int) t.ne0, out,
            t.host + h.expert[ei]*t.nb2 + row*t.nb1, t.nb1, n, in, nr);
    }
}

static void share_chunk(share_host & h, int phase, int c) {
    const share_layer & l = h.cur;
    const int64_t n_ff = l.up.ne1;
    if (phase == 0) {
        // up then gate, per expert, share_up_chunk rows at a time
        const int per = (int) ((n_ff + share_up_chunk - 1)/share_up_chunk);
        const bool gate = c >= h.n_expert*per;
        const int ei = (c % (h.n_expert*per))/per;
        const int64_t row = (c % per)*share_up_chunk;
        share_rows(h, gate ? l.gate : l.up, ei, row, std::min(share_up_chunk, n_ff - row),
            [&](int j) { return h.gu.data() + (size_t) j*2*n_ff + (gate ? n_ff : 0); },
            [&](int j) { return (const void *) (h.xq.data() + h.entry_tok[j]*h.xq_row); });
    } else if (phase == 1) {
        float * gu = h.gu.data() + (size_t) c*2*n_ff;
        // SWIGLU_CLAMP clamps the gate BEFORE SiLU, not its result.
        const float lim = h.glu_limit > 0.0f ? h.glu_limit : INFINITY;
        for (int64_t i = 0; i < n_ff; ++i) {
            const float g = std::min(gu[n_ff + i], lim);
            gu[i] = std::clamp(gu[i], -lim, lim) * (g/(1.0f + expf(-g)));
        }
        ggml_moe_cache_cpu_traits(h.t_down->vec_dot_type)->from_float(gu, h.hq.data() + c*h.hq_row, n_ff);
    } else {
        const int64_t n_embd = l.down.ne1;
        const int per = (int) ((n_embd + share_down_chunk - 1)/share_down_chunk);
        const int64_t row = (c % per)*share_down_chunk;
        share_rows(h, l.down, c/per, row, std::min(share_down_chunk, n_embd - row),
            [&](int j) { return h.y + (size_t) j*n_embd; },
            [&](int j) { return (const void *) (h.hq.data() + j*h.hq_row); });
    }
}

// Runs chunks of job gen until none are left; returns false once gen is stale.
static bool share_work(share_host & h, uint32_t gen) {
    const uint64_t tag = (uint64_t) gen << 32;
    for (int phase = 0; phase < 3; ++phase) {
        if (phase > 0) {
            for (uint64_t d; (d = h.done[phase - 1].load(std::memory_order_acquire)) != tag + h.n_chunk[phase - 1]; ) {
                if ((d >> 32) != gen) {
                    return false;
                }
                SHARE_PAUSE();
            }
        }
        for (uint64_t v = h.next[phase].load(std::memory_order_acquire); ; ) {
            if ((v >> 32) != gen) {
                return false;
            }
            if ((int) (uint32_t) v >= h.n_chunk[phase]) {
                break;
            }
            if (h.next[phase].compare_exchange_weak(v, v + 1, std::memory_order_acq_rel)) {
                share_chunk(h, phase, (int) (uint32_t) v);
                h.done[phase].fetch_add(1, std::memory_order_release);
                v = h.next[phase].load(std::memory_order_acquire);
            }
        }
    }
    return true;
}

// Spins while jobs arrive; backs off to short sleeps after a quiet period.
template <typename F>
static void share_wait(F ready) {
    auto quiet = std::chrono::steady_clock::now();
    for (uint32_t spins = 0; !ready(); ++spins) {
        if ((spins & 1023) == 0 && std::chrono::steady_clock::now() - quiet > std::chrono::milliseconds(20)) {
            std::this_thread::sleep_for(std::chrono::microseconds(20));
        } else {
            SHARE_PAUSE();
        }
    }
}

// After a host job of ns: gives the host the share of each up type's misses that finishes with
// the bus, cost_bus/(cost_bus + cost_host) of them, from the measured ns per byte of each.
// The speed of the host's kernels differs by type far more than the bus does.
static void share_balance(share_host & h, double ns) {
    const auto ema = [](double & cost, double sample) {
        cost = cost > 0.0 ? cost + (sample - cost)/8 : sample;
    };
    ema(h.host_cost[h.cur.up.type], ns/(h.n_expert*(double) (h.cur.up.nb2 + h.cur.gate.nb2 + h.cur.down.nb2)));
    const float stage_cost = h.mb->stage_cost;
    if (stage_cost > 0.0f) {
        ema(h.bus_cost, stage_cost*h.ns_per_tick);
    }
    if (h.bus_cost <= 0.0) {
        return;
    }
    for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
        if (h.host_cost[t] > 0.0) {
            h.mb->host_num[t] = (int16_t) lround(256.0*h.bus_cost/(h.bus_cost + h.host_cost[t]));
        }
    }
}

static void share_leader(share_host & h) {
    uint32_t seen = 0;
    for (;;) {
        share_wait([&] { return h.mb->seq != seen; });
        const auto t0 = std::chrono::steady_clock::now();
        seen = h.mb->seq;
        std::atomic_thread_fence(std::memory_order_acquire);

        {
            std::lock_guard<std::mutex> lock(h.mu);
            h.cur = h.layers[h.mb->layer];
        }
        h.t_up = ggml_moe_cache_cpu_traits(h.cur.up.type);
        h.t_down = ggml_moe_cache_cpu_traits(h.cur.down.type);
        h.n_tok = h.mb->n_tok;
        h.n_expert = h.mb->n_expert;
        h.n_entry = h.mb->n_entry;
        h.glu_limit = h.mb->glu_limit;
        for (int i = 0; i < h.n_expert; ++i) {
            h.expert[i] = h.mb->expert[i];
            h.expert_entries[i].clear();
        }
        for (int j = 0; j < h.n_entry; ++j) {
            h.entry_tok[j] = h.mb->entry_tok[j];
            h.expert_entries[h.mb->entry_expert[j]].push_back(j);
        }
        const int64_t n_embd = h.cur.up.ne0;
        const int64_t n_ff = h.cur.up.ne1;
        const ggml_type_traits_cpu * t_xq = ggml_moe_cache_cpu_traits(h.t_up->vec_dot_type);
        h.xq_row = ggml_row_size(h.t_up->vec_dot_type, n_embd);
        h.hq_row = ggml_row_size(h.t_down->vec_dot_type, n_ff);
        h.xq.resize(h.n_tok*h.xq_row);
        h.gu.resize((size_t) h.n_entry*2*n_ff);
        h.hq.resize(h.n_entry*h.hq_row);
        for (int t = 0; t < h.n_tok; ++t) {
            t_xq->from_float(h.x + t*n_embd, h.xq.data() + t*h.xq_row, n_embd);
        }

        const int up_per = (int) ((n_ff + share_up_chunk - 1)/share_up_chunk);
        const int down_per = (int) ((h.cur.down.ne1 + share_down_chunk - 1)/share_down_chunk);
        h.n_chunk[0] = 2*h.n_expert*up_per;
        h.n_chunk[1] = h.n_entry;
        h.n_chunk[2] = h.n_expert*down_per;
        const uint64_t tag = (uint64_t) seen << 32;
        for (int phase = 0; phase < 3; ++phase) {
            h.done[phase].store(tag, std::memory_order_relaxed);
            h.next[phase].store(tag, std::memory_order_release);
        }
        h.job.store(seen);
        if (h.sleepers.load() > 0) {
            std::lock_guard<std::mutex> lock(h.wake_mu);
            h.wake_cv.notify_all();
        }
        share_work(h, seen);
        while (h.done[2].load(std::memory_order_acquire) != tag + h.n_chunk[2]) {
            SHARE_PAUSE();
        }
        std::atomic_thread_fence(std::memory_order_release);
        h.mb->done = seen;
        if (!share_pinned()) {
            share_balance(h, std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count());
        }
    }
}

// Spins briefly after a job (layers come in bursts), then sleeps until the leader wakes it.
static void share_worker(share_host & h) {
    uint32_t seen = 0;
    for (;;) {
        const auto t0 = std::chrono::steady_clock::now();
        for (uint32_t spins = 0; h.job.load() == seen; ++spins) {
            if ((spins & 255) == 0 && std::chrono::steady_clock::now() - t0 > std::chrono::microseconds(200)) {
                std::unique_lock<std::mutex> lock(h.wake_mu);
                h.sleepers++;
                h.wake_cv.wait(lock, [&] { return h.job.load() != seen; });
                h.sleepers--;
                break;
            }
            SHARE_PAUSE();
        }
        seen = h.job.load();
        share_work(h, seen);
    }
}

// Called with h.mu held, outside any graph capture (route registration).
static bool share_init(share_host & h, int n_embd) {
    if (h.mb) {
        return n_embd <= h.n_embd;
    }
    const size_t header = GGML_PAD(sizeof(share_mailbox), 256);
    const size_t bytes = header + (size_t)(share_max_tokens + share_max_entries)*n_embd*sizeof(float);
    char * host = nullptr;
    char * device = nullptr;
    // portable: under unified addressing every device sees the mailbox at the same address
    if (cudaHostAlloc((void **) &host, bytes, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess ||
        cudaHostGetDevicePointer((void **) &device, host, 0) != cudaSuccess) {
        (void) cudaGetLastError();
        GGML_LOG_WARN("%s: moe cpu share disabled: mailbox allocation failed\n", __func__);
        return false;
    }
    memset(host, 0, bytes);
    h.mb = (share_mailbox *) host;
    h.d_mb = (share_mailbox *) device;
    h.x = (float *)(host + header);
    h.d_x = (float *)(device + header);
    h.y = h.x + (size_t) share_max_tokens*n_embd;
    h.d_y = h.d_x + (size_t) share_max_tokens*n_embd;
    h.n_embd = n_embd;
    for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
        h.mb->host_num[t] = (int16_t) lroundf((1.0f - share_gpu_fraction())*256.0f);
    }
#if defined(GGML_USE_HIP)
    int khz = 0;
    if (cudaDeviceGetAttribute(&khz, hipDeviceAttributeWallClockRate, ggml_cuda_get_device()) == cudaSuccess && khz > 0) {
        h.ns_per_tick = 1e6/khz;
    }
#endif

    const char * threads = getenv("GGML_CUDA_MOE_CPU_SHARE_THREADS");
    // a sixth of the cores (6..32), never more than half: 6 starves the share on large hosts
    const unsigned hw = std::max(std::thread::hardware_concurrency(), 2u);
    h.n_threads = threads ? std::max(1, atoi(threads)) : (int) std::min(hw/2, std::clamp(hw/6, 6u, 32u));
    std::thread(share_leader, std::ref(h)).detach();
    for (int tid = 1; tid < h.n_threads; ++tid) {
        std::thread(share_worker, std::ref(h)).detach();
    }
    GGML_LOG_INFO("%s: moe cpu share: GPU keeps %.0f%% of routed misses%s, %d host threads\n",
        __func__, 100.0f*share_gpu_fraction(), share_pinned() ? "" : " until it measures host and bus speed", h.n_threads);
    return true;
}

// Called with h.mu held, outside any graph capture, on the device whose route is registered.
static bool share_dev_init(share_host & h) {
    const int id = ggml_cuda_get_device();
    share_dev & d = h.devs[id];
    if (d.st) {
        return true;
    }
    const uint32_t ticket = (uint32_t) id;
    if (cudaMalloc((void **) &d.st, sizeof(share_state)) != cudaSuccess ||
        cudaMemset(d.st, 0, sizeof(share_state)) != cudaSuccess ||
        cudaMemcpy(&d.st->ticket, &ticket, sizeof(ticket), cudaMemcpyHostToDevice) != cudaSuccess) {
        (void) cudaGetLastError();
        cudaFree(d.st);
        d.st = nullptr;
        GGML_LOG_WARN("%s: moe cpu share disabled on device %d: state allocation failed\n", __func__, id);
        return false;
    }
    return true;
}

void ggml_moe_cpu_share_note(const ggml_tensor * weights, const void * device_base, const void * owner) {
    int layer = -1;
    if (share_gpu_fraction() < 0.0f || !ggml_moe_cache_cpu_traits || !ggml_moe_cache_cpu_rows ||
        sscanf(weights->name, "blk.%d.", &layer) != 1 || layer < 0 || layer >= share_max_layers) {
        return;
    }
    share_tensor t;
    t.host = (const char *) weights->data;
    t.device = (const char *) device_base;
    t.span = weights->nb[2]*weights->ne[2];
    t.nb1 = weights->nb[1];
    t.nb2 = weights->nb[2];
    t.ne0 = weights->ne[0];
    t.ne1 = weights->ne[1];
    t.type = weights->type;

    share_host & h = g_share;
    std::lock_guard<std::mutex> lock(h.mu);
    // layers are keyed by block index: while one session routes a block, another session's
    // tensors of the same index stay out of the share
    share_layer & l = h.layers[layer];
    ggml_moe_cache_route_table live;
    if (l.owner != owner) {
        if (l.owner && ggml_moe_cache_route_find(l.owner_host, live)) {
            return;
        }
        for (auto it = h.tensors.begin(); it != h.tensors.end();) {
            it = it->second.first == layer ? h.tensors.erase(it) : std::next(it);
        }
        l = {};
        l.owner = owner;
        l.owner_host = t.host;
    }
    h.tensors[weights->data] = { layer, t };
    if (strstr(weights->name, "ffn_down_exps") && share_init(h, (int) weights->ne[1]) && share_dev_init(h)) {
        l.down = t;
    }
}

static bool share_traits_ok(ggml_type type, int64_t n) {
    const ggml_type_traits_cpu * t = ggml_moe_cache_cpu_traits(type);
    if (!t || !t->vec_dot || t->nrows != 1 || n % ggml_blck_size(type)) {
        return false;
    }
    const ggml_type_traits_cpu * q = ggml_moe_cache_cpu_traits(t->vec_dot_type);
    return q && q->from_float && n % ggml_blck_size(t->vec_dot_type) == 0;
}

ggml_moe_cpu_share_args ggml_moe_cpu_share_begin(
        const ggml_tensor * up, const ggml_tensor * gate, const ggml_tensor * ids,
        const ggml_tensor * x, const ggml_cuda_mm_fusion_args_host * fusion, cudaStream_t stream) {
    share_host & h = g_share;
    const int device = ggml_cuda_get_device();
    share_dev & d = h.devs[device];
    if (share_gpu_fraction() < 0.0f || !h.mb || !d.st) {
        return {};
    }
    // the next doorbell relies on the previous shared layer's merge having waited for the host
    GGML_ASSERT(!h.pending_ids && "moe cpu share: a shared layer never ran its down");
    GGML_ASSERT(!h.merge_experts && "moe cpu share: a shared down was never merged");
    const int64_t n_k = ids->ne[0];
    const int64_t n_tok = ids->ne[1];
    const int64_t ids_stride = ids->nb[1]/sizeof(int32_t);
    // only the fused weighted reduction merges the host's rows, so the share needs its width
    if (n_k < 2 || n_k > MOE_WEIGHTED_REDUCTION_MAX_EXPERTS ||
        !fusion || fusion->gate != gate || fusion->x_bias || fusion->gate_bias || fusion->x_scale ||
        fusion->gate_scale || fusion->residual ||
        !((fusion->glu_op == GGML_GLU_OP_SWIGLU && fusion->glu_limit == 0.0f) ||
          (fusion->glu_op == GGML_GLU_OP_SWIGLU_CLAMP && fusion->glu_limit > 0.0f && std::isfinite(fusion->glu_limit))) ||
        x->type != GGML_TYPE_F32 || x->ne[1] != 1 ||
        x->ne[0] != up->ne[0] || n_tok > share_max_tokens || n_k*n_tok > share_max_entries ||
        (n_tok - 1)*ids_stride + n_k > share_max_routes || ids->nb[0] != sizeof(int32_t)) {
        return {};
    }

    int layer;
    share_layer l;
    {
        std::lock_guard<std::mutex> lock(h.mu);
        auto found_up = h.tensors.find(up->data);
        auto found_gate = h.tensors.find(gate->data);
        if (found_up == h.tensors.end() || found_gate == h.tensors.end() ||
            found_up->second.first != found_gate->second.first) {
            return {};
        }
        layer = found_up->second.first;
        share_layer & reg = h.layers[layer];
        if (!reg.down.host) {
            return {};
        }
        // the host job reads the fused op's own operands, recorded before its doorbell rings
        reg.up = found_up->second.second;
        reg.gate = found_gate->second.second;
        l = reg;
        // sized once by the largest routed expert; graph capture is relaxed, so this may run inside one
        if (!d.stage && getenv("GGML_CUDA_MOE_STAGE_OFF") == nullptr) {
            size_t stride = 0;
            for (const auto & t : h.tensors) {
                stride = std::max(stride, GGML_PAD(t.second.second.nb2, 256));
            }
            if (cudaMalloc((void **) &d.stage, (size_t) share_max_staged*3*stride) == cudaSuccess) {
                d.stage_stride = stride;
                GGML_LOG_INFO("%s: moe miss staging: %d experts x %.2f MiB per layer\n",
                    __func__, share_max_staged, 3.0*stride/(1024.0*1024.0));
            } else {
                (void) cudaGetLastError();
                d.stage = nullptr;
            }
        }
    }
    ggml_moe_cache_route_table up_route, gate_route, down_route;
    if (l.up.ne0 != h.n_embd || l.down.ne1 != h.n_embd || l.gate.ne0 != l.up.ne0 ||
        l.gate.ne1 != l.up.ne1 || l.down.ne0 != l.up.ne1 || l.up.ne1 % 4 ||
        !share_traits_ok(l.up.type, l.up.ne0) || !share_traits_ok(l.gate.type, l.gate.ne0) ||
        !share_traits_ok(l.down.type, l.down.ne0) ||
        ggml_moe_cache_cpu_traits(l.up.type)->vec_dot_type != ggml_moe_cache_cpu_traits(l.gate.type)->vec_dot_type ||
        !ggml_moe_cache_route_find(l.up.host, up_route) ||
        !ggml_moe_cache_route_find(l.gate.host, gate_route) ||
        !ggml_moe_cache_route_find(l.down.host, down_route)) {
        return {};
    }

    share_doorbell_args a;
    a.ids = (const int32_t *) ids->data;
    a.n_k = (int) n_k;
    a.n_tok = (int) n_tok;
    a.ids_stride = (int) ids_stride;
    a.x = (const float *) x->data;
    a.x_stride = x->nb[2]/sizeof(float);
    a.n_embd = h.n_embd;
    const share_tensor * tensors[3] = { &l.up, &l.gate, &l.down };
    const ggml_moe_cache_route_table * routes[3] = { &up_route, &gate_route, &down_route };
    for (int i = 0; i < 3; ++i) {
        a.table[i] = routes[i]->table;
        a.base[i] = tensors[i]->device;
        a.span[i] = tensors[i]->span;
        // the copy moves 16-byte words: an unaligned tensor stays zero-copy
        const bool aligned = tensors[i]->nb2 % 16 == 0 && (uintptr_t) tensors[i]->device % 16 == 0;
        a.nb2[i] = aligned ? tensors[i]->nb2 : SIZE_MAX;
    }
    a.stage = d.stage;
    a.stage_stride = d.stage_stride;
    a.layer = layer;
    a.up_type = l.up.type;
    a.probe = !share_pinned();
    // Pass the activation with the GPU job, so captured graphs replay their own
    // limit rather than observing mutable launch-time state from another layer.
    a.glu_limit = fusion->glu_limit;
    a.mb = h.d_mb;
    a.mb_x = h.d_x;
    a.st = d.st;
    share_doorbell<<<1, 256, 0, stream>>>(a);
    CUDA_CHECK(cudaGetLastError());
    if (d.stage) {
        share_stage_copy<<<2*ggml_cuda_info().devices[device].nsm, 256, 0, stream>>>(d.st, h.d_mb);
        CUDA_CHECK(cudaGetLastError());
    }

    h.pending_ids = ids->data;
    h.pending_device = device;
    h.pending_layer = layer;
    h.pending_ids_stride = (int) ids_stride;
    ggml_moe_cpu_share_args args;
    args.skip = d.st->skip;
    if (d.stage) {
        args.stage_x = d.st->stage[0];
        args.stage_gate = d.st->stage[1];
    }
    return args;
}

ggml_moe_cpu_share_args ggml_moe_cpu_share_down(const ggml_tensor * down, const ggml_tensor * ids, const ggml_tensor * dst) {
    share_host & h = g_share;
    if (!h.pending_ids) {
        return {};
    }
    {
        std::lock_guard<std::mutex> lock(h.mu);
        if (h.layers[h.pending_layer].down.host != down->data) {
            return {};
        }
    }
    GGML_ASSERT(ids->data == h.pending_ids && "moe cpu share: down routed by different ids");
    h.pending_ids = nullptr;
    h.merge_experts = dst->data;
    const share_dev & d = h.devs[h.pending_device];
    ggml_moe_cpu_share_args args;
    args.skip = d.st->skip;
    if (d.stage) {
        args.stage_x = d.st->stage[2];
    }
    return args;
}

bool ggml_moe_cpu_share_hoist(int64_t n_tok) {
    static const bool off = getenv("GGML_CUDA_MOE_SHARE_HOIST") && atoi(getenv("GGML_CUDA_MOE_SHARE_HOIST")) == 0;
    return !off && share_gpu_fraction() >= 0.0f && n_tok <= share_max_tokens;
}

ggml_moe_cpu_share_args ggml_moe_cpu_share_merge(const ggml_tensor * experts) {
    share_host & h = g_share;
    if (!h.merge_experts || experts->data != h.merge_experts) {
        return {};
    }
    h.merge_experts = nullptr;
    const share_dev & d = h.devs[h.pending_device];
    ggml_moe_cpu_share_args args;
    args.skip = d.st->skip;
    args.ids_stride = h.pending_ids_stride;
    args.y = h.d_y;
    args.done = (const uint32_t *) &h.d_mb->done;
    args.ticket = &d.st->ticket;
    return args;
}

#endif
