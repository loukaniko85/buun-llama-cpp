#include "common.cuh"
#include "mmid.cuh"

#if defined(GGML_CUDA_USE_CUB) && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#include <cub/cub.cuh>

static __global__ void mm_ids_radix_prepare(
        const int32_t * ids, int32_t * keys, int32_t * slots,
        int pairs, int used, int stride, int experts) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= pairs) {
        return;
    }
    const int expert = ids[(i/used)*stride + i%used];
    keys[i] = expert >= 0 && expert < experts ? expert : experts;
    slots[i] = i;
}

static __global__ void mm_ids_radix_finish(
        const int32_t * keys, const int32_t * slots,
        int32_t * src, int32_t * dst, int32_t * bounds,
        int pairs, int experts, int used, int channels, int source_stride, bool inverse) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i <= experts) {
        int lo = 0, hi = pairs;
        while (lo < hi) {
            const int mid = lo + (hi - lo)/2;
            if (keys[mid] < i) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        bounds[i] = lo;
    }
    if (i < pairs) {
        if (keys[i] < experts) {
            const int slot = slots[i];
            dst[i] = slot;
            if (inverse) {
                src[slot] = i;
            } else {
                src[i] = (slot/used)*source_stride + (slot%used)%channels;
            }
        } else if (!inverse) {
            src[i] = 0;
        }
    }
}
#endif

// To reduce shared memory use, store "it" and "iex_used" with 22/10 bits each.
struct mm_ids_helper_store {
    uint32_t data;

    __device__ mm_ids_helper_store(const uint32_t it, const uint32_t iex_used) {
        data = (it & 0x003FFFFF) | (iex_used << 22);
    }

    __device__ uint32_t it() const {
        return data & 0x003FFFFF;
    }

    __device__ uint32_t iex_used() const {
        return data >> 22;
    }
};
static_assert(sizeof(mm_ids_helper_store) == 4, "unexpected size for mm_ids_helper_store");

// the generic path passes 0, which needs no padding since it never groups lanes by token
template <int n> struct mm_ids_pow2 { static constexpr int value = 2*mm_ids_pow2<(n + 1)/2>::value; };
template <>      struct mm_ids_pow2<1> { static constexpr int value = 1; };
template <>      struct mm_ids_pow2<0> { static constexpr int value = 1; };

// Helper function for mul_mat_id, converts ids to a more convenient format.
// ids_src1 describes how to permute the flattened column indices of src1 in order to get a compact src1 tensor sorted by expert.
// ids_dst describes the same mapping but for the dst tensor.
// The upper and lower bounds for the ith expert in the compact src1 tensor are stored in expert_bounds[i:i+1].
template <int n_expert_used_template>
__launch_bounds__(ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_tokens, const int n_expert_used_var, const int nchannels_y, const int si1, const int sis1, const bool write_inverse) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const int n_expert_used = n_expert_used_template == 0 ? n_expert_used_var : n_expert_used_template;
    const int expert = blockIdx.x;

    // token slots per warp lane group, padded to a power of 2 so a warp divides evenly
    constexpr int neu_padded = mm_ids_pow2<n_expert_used_template>::value;

    extern __shared__ char data_mm_ids_helper[];
    mm_ids_helper_store * store = (mm_ids_helper_store *) data_mm_ids_helper;

    int nex_prev   = 0; // Number of columns for experts with a lower index.
    int it_compact = 0; // Running index for the compact slice of this expert.

    if constexpr (n_expert_used_template == 0) {
        // Generic implementation:
        for (int it = 0; it < n_tokens; ++it) {
            int iex_used = -1; // The index at which the expert is used, if any.
            for (int iex = threadIdx.x; iex < n_expert_used; iex += warp_size) {
                int expert_used = ids[it*si1 + iex];
                if (expert_used < 0) {
                    expert_used = INT_MAX; // expert-parallel window: routed elsewhere, never matches
                }
                nex_prev += expert_used < expert;
                if (expert_used == expert) {
                    iex_used = iex;
                }
            }

            if (iex_used != -1) {
                store[it_compact] = mm_ids_helper_store(it, iex_used);
            }

            if (warp_reduce_any<warp_size>(iex_used != -1)) {
                it_compact++;
            }
        }
    } else {
        // Implementation optimized for specific numbers of experts used:
        // a warp holds a whole number of token slots, so the slot count is padded to a power of 2
        static_assert(neu_padded <= warp_size && warp_size % neu_padded == 0, "bad n_expert_used");
        for (int it0 = 0; it0 < n_tokens; it0 += warp_size/neu_padded) {
            const int it = it0 + threadIdx.x / neu_padded;

            const int iex = threadIdx.x % neu_padded; // The index at which the expert is used, if any.
            int expert_used = (neu_padded == n_expert_used || iex < n_expert_used) && it < n_tokens ?
                ids[it*si1 + iex] : INT_MAX;
            if (expert_used < 0) {
                expert_used = INT_MAX; // expert-parallel window: routed elsewhere, never matches
            }
            const int iex_used = expert_used == expert ? iex : -1;
            nex_prev += expert_used < expert;

            // Whether the threads at this token position have used the expert:
            const int it_compact_add_self = warp_reduce_any<neu_padded>(iex_used != -1);

            // Do a scan over threads at lower token positions in warp to get the correct index for writing data:
            int it_compact_add_lower = 0;
#pragma unroll
            for (int offset = neu_padded; offset < warp_size; offset += neu_padded) {
                const int tmp = __shfl_up_sync(0xFFFFFFFFULL, it_compact_add_self, offset, warp_size);
                if (threadIdx.x >= static_cast<unsigned int>(offset)) {
                    it_compact_add_lower += tmp;
                }
            }

            if (iex_used != -1) {
                store[it_compact + it_compact_add_lower] = mm_ids_helper_store(it, iex_used);
            }

            // The thread with the highest index in the warp always has the sum over the whole warp, use it to increment all threads:
            it_compact += __shfl_sync(0xFFFFFFFFULL, it_compact_add_lower + it_compact_add_self, warp_size - 1, warp_size);
        }
    }
    nex_prev = warp_reduce_sum<warp_size>(nex_prev);
    ggml_cuda_syncwarp();

    for (int itc = threadIdx.x; itc < it_compact; itc += warp_size) {
        const mm_ids_helper_store store_it = store[itc];
        const int it       = store_it.it();
        const int iex_used = store_it.iex_used();
        ids_dst[nex_prev + itc] = it*n_expert_used + iex_used;
        // ids_src1 holds the forward map, or the inverse map (token slot -> compact row) for quant dedup
        if (write_inverse) {
            ids_src1[it*n_expert_used + iex_used] = nex_prev + itc;
        } else {
            ids_src1[nex_prev + itc] = it*sis1 + iex_used % nchannels_y;
        }
    }

    if (threadIdx.x != 0) {
        return;
    }

    expert_bounds[expert] = nex_prev;

    if (expert < static_cast<int>(gridDim.x) - 1) {
        return;
    }

    expert_bounds[gridDim.x] = nex_prev + it_compact;
}

template <int n_expert_used_template>
static void launch_mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_experts, const int n_tokens, const int n_expert_used_var, const int nchannels_y, const int si1, const int sis1, const bool write_inverse, cudaStream_t stream) {
    GGML_ASSERT(n_tokens          < (1 << 22) && "too few bits in mm_ids_helper_store");
    GGML_ASSERT(n_expert_used_var < (1 << 10) && "too few bits in mm_ids_helper_store");

    const int id = ggml_cuda_get_device();
    const int warp_size = ggml_cuda_info().devices[id].warp_size;
    const size_t smpbo = ggml_cuda_info().devices[id].smpbo;
    CUDA_SET_SHARED_MEMORY_LIMIT(mm_ids_helper<n_expert_used_template>, smpbo);

    const dim3 num_blocks(n_experts, 1, 1);
    const dim3 block_size(warp_size, 1, 1);
    const size_t nbytes_shared = n_tokens*sizeof(mm_ids_helper_store);
    GGML_ASSERT(nbytes_shared <= smpbo);
    // Slots routed to another device (expert-parallel window) produce no compact row, so the maps are not fully
    // written: the inverse map marks such slots -1, and the forward map's unused tail points at row 0 so the
    // consumers that walk all n_tokens*n_expert_used rows (the activation quantizers) gather a valid row.
    CUDA_CHECK(cudaMemsetAsync(ids_src1, write_inverse ? 0xFF : 0, (size_t) n_tokens*n_expert_used_var*sizeof(int32_t), stream));
    mm_ids_helper<n_expert_used_template><<<num_blocks, block_size, nbytes_shared, stream>>>
        (ids, ids_src1, ids_dst, expert_bounds, n_tokens, n_expert_used_var, nchannels_y, si1, sis1, write_inverse);
}

void ggml_cuda_launch_mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_experts, const int n_tokens, const int n_expert_used, const int nchannels_y, const int si1, const int sis1, const bool write_inverse, cudaStream_t stream) {
    switch (n_expert_used) {
        case  2:
            launch_mm_ids_helper< 2>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case  4:
            launch_mm_ids_helper< 4>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case  6:
            launch_mm_ids_helper< 6>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case  8:
            launch_mm_ids_helper< 8>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case 10:
            launch_mm_ids_helper<10>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case 16:
            launch_mm_ids_helper<16>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case 32:
            launch_mm_ids_helper<32>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        default:
            launch_mm_ids_helper< 0>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
    }
}

void ggml_cuda_launch_mm_ids_helper(
        ggml_cuda_pool & pool,
        const int32_t * ids, int32_t * ids_src1, int32_t * ids_dst, int32_t * expert_bounds,
        int n_experts, int n_tokens, int n_expert_used, int nchannels_y, int si1, int sis1, bool write_inverse, cudaStream_t stream) {
#if defined(GGML_CUDA_USE_CUB) && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    // Long 512-expert batches otherwise scan every route once per expert.
    // Stable radix sorting preserves token/slot order within each expert.
    // Short batches keep the faster scan; this first admission is measured SM86.
    if (ggml_cuda_info().devices[ggml_cuda_get_device()].cc == 860 &&
            n_experts == 512 && n_expert_used == 10 && n_tokens >= 2048 && n_tokens <= 8192 &&
            (nchannels_y == 1 || nchannels_y == n_expert_used)) {
        const int pairs = n_tokens*n_expert_used;
        ggml_cuda_pool_alloc<int32_t> keys_in(pool, pairs), keys_out(pool, pairs);
        ggml_cuda_pool_alloc<int32_t> slots_in(pool, pairs), slots_out(pool, pairs);
        size_t scratch_bytes = 0;
        // Ten bits include the invalid-route sentinel 512 after valid experts.
        CUDA_CHECK(cub::DeviceRadixSort::SortPairs(nullptr, scratch_bytes,
                keys_in.get(), keys_out.get(), slots_in.get(), slots_out.get(), pairs, 0, 10, stream));
        ggml_cuda_pool_alloc<char> scratch(pool, scratch_bytes);
        mm_ids_radix_prepare<<<(pairs + 255)/256, 256, 0, stream>>>(
                ids, keys_in.get(), slots_in.get(), pairs, n_expert_used, si1, n_experts);
        CUDA_CHECK(cub::DeviceRadixSort::SortPairs(scratch.get(), scratch_bytes,
                keys_in.get(), keys_out.get(), slots_in.get(), slots_out.get(), pairs, 0, 10, stream));
        if (write_inverse) {
            CUDA_CHECK(cudaMemsetAsync(ids_src1, 0xff, size_t(pairs)*sizeof(int32_t), stream));
        }
        mm_ids_radix_finish<<<(pairs + 255)/256, 256, 0, stream>>>(
                keys_out.get(), slots_out.get(), ids_src1, ids_dst, expert_bounds,
                pairs, n_experts, n_expert_used, nchannels_y, sis1, write_inverse);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
#else
    GGML_UNUSED(pool);
#endif
    ggml_cuda_launch_mm_ids_helper(ids, ids_src1, ids_dst, expert_bounds,
            n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
}
