#include "dsa-topk.cuh"

static constexpr int DSA_TOPK_THREADS = 1024;
static constexpr int DSA_TOPK_BINS    = 256;

// order-preserving map of a float onto an unsigned key
static __device__ __forceinline__ uint32_t dsa_topk_key(float x) {
    const uint32_t b = __float_as_uint(x);
    return b ^ ((b >> 31) ? 0xffffffffu : 0x80000000u);
}

// key above, inverted pool below: distinct, and larger = higher score, then lower pool
static __device__ __forceinline__ uint64_t dsa_topk_pack(uint32_t key, int i) {
    return uint64_t(key) << 32 | ~uint32_t(i);
}

// one block per row: radix-select the k-th largest key (all of them when k == n), collect the k survivors in
// shared memory, place each at its rank by descending score (ties by ascending pool) and expand each pool to
// its member cells
static __global__ void __launch_bounds__(DSA_TOPK_THREADS) dsa_topk(
        const float * __restrict__ score, const int * __restrict__ pool_idxs, int * __restrict__ dst,
        const int n, const int k, const int kpool, const int64_t s_score, const int64_t s_pidx) {
    __shared__ int      scan[DSA_TOPK_BINS];
    __shared__ int      wsum[DSA_TOPK_BINS/WARP_SIZE];
    __shared__ uint64_t skey[DSA_TOPK_MAX_K];
    __shared__ int      sord[DSA_TOPK_MAX_K];
    __shared__ uint32_t prefix;
    __shared__ int      rank;
    __shared__ int      n_gt;
    __shared__ int      n_eq;

    const int t = threadIdx.x;
    const float * x = score + blockIdx.x*s_score;

    if (k == n) {
        for (int i = t; i < n; i += blockDim.x) {
            skey[i] = dsa_topk_pack(dsa_topk_key(x[i]), i);
        }
    } else {
        if (t == 0) {
            prefix = 0;
            rank   = k;
            n_gt   = 0;
            n_eq   = 0;
        }
        uint32_t mask = 0;
        for (int shift = 24; shift >= 0; shift -= 8) {
            if (t < DSA_TOPK_BINS) {
                scan[t] = 0;
            }
            __syncthreads();
            const uint32_t pre  = prefix;
            const int      need = rank;
            for (int i = t; i < n; i += blockDim.x) {
                const uint32_t key = dsa_topk_key(x[i]);
                if ((key & mask) == pre) {
                    // slot 255 - digit, so the scan below runs from the top digit down
                    atomicAdd(&scan[255 - ((key >> shift) & 255)], 1);
                }
            }
            __syncthreads();
            const int count = t < DSA_TOPK_BINS ? scan[t] : 0;
            int incl = 0;
            if (t < DSA_TOPK_BINS) {
                incl = warp_prefix_inclusive_sum(count);
                if (t % WARP_SIZE == WARP_SIZE - 1) {
                    wsum[t / WARP_SIZE] = incl;
                }
            }
            __syncthreads();
            if (t < DSA_TOPK_BINS) {
                for (int w = 0; w < t / WARP_SIZE; ++w) {
                    incl += wsum[w];
                }
                const int above = incl - count;
                if (above < need && need <= incl) {
                    prefix = pre | uint32_t(255 - t) << shift;
                    rank   = need - above;
                }
            }
            mask |= 0xffu << shift;
            __syncthreads();
        }

        // exactly k - rank keys exceed the prefix; the first rank keys equal to it fill the rest
        for (int i = t; i < n; i += blockDim.x) {
            const uint32_t key = dsa_topk_key(x[i]);
            int pos = -1;
            if (key > prefix) {
                pos = atomicAdd(&n_gt, 1);
            } else if (key == prefix) {
                pos = atomicAdd(&n_eq, 1);
                pos = pos < rank ? k - rank + pos : -1;
            }
            if (pos >= 0) {
                skey[pos] = dsa_topk_pack(key, i);
            }
        }
    }
    __syncthreads();

    // the packed keys are distinct, so each survivor's rank is the count of larger ones
    for (int s = t; s < k; s += blockDim.x) {
        const uint64_t c = skey[s];
        int r = 0;
#pragma unroll 8
        for (int j = 0; j < k; ++j) {
            r += skey[j] > c;
        }
        sord[r] = ~uint32_t(c);
    }
    __syncthreads();

    int * d = dst + int64_t(blockIdx.x)*k*kpool;
    for (int e = t; e < k*kpool; e += blockDim.x) {
        d[e] = pool_idxs[sord[e / kpool]*s_pidx + e % kpool];
    }
}

void ggml_cuda_op_dsa_topk(ggml_backend_cuda_context & ctx, const ggml_tensor * score, const ggml_tensor * pool_idxs,
        int k, ggml_tensor * dst) {
    GGML_ASSERT(k >= 1 && k <= DSA_TOPK_MAX_K && k <= score->ne[0]);
    dsa_topk<<<(unsigned) ggml_nrows(score), DSA_TOPK_THREADS, 0, ctx.stream()>>>(
        (const float *) score->data, (const int *) pool_idxs->data, (int *) dst->data, (int) score->ne[0], k,
        (int) pool_idxs->ne[0], score->nb[1]/sizeof(float), pool_idxs->nb[1]/sizeof(int));
}
