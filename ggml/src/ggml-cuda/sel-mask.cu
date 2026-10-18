#include "sel-mask.cuh"
#include "convert.cuh"

// One block per token. dst may be mask itself, so the mask under each scattered column is read before the fill
// overwrites the row. -inf + mask is -inf for any mask value but +inf, so the fill skips reading mask.
// Scatters run in order with a block barrier between them, matching the SET_ROWS sequence.
template <typename T>
static __global__ void sel_mask_f(T * dst, const T * mask, const int64_t n_kv, const int64_t s_dst, const int64_t s_mask,
        const sel_mask_scatter sc0, const sel_mask_scatter sc1) {
    extern __shared__ float smask[]; // [sc0.n + sc1.n]
    const int64_t t = blockIdx.x;
    T       * d = dst  + t*s_dst;
    const T * m = mask + t*s_mask;
    const sel_mask_scatter scs[2] = { sc0, sc1 };
    const int              off[2] = { 0, sc0.n };

    for (int k = 0; k < 2; ++k) {
        const sel_mask_scatter & sc = scs[k];
        float * sm = smask + off[k];
        for (int s = threadIdx.x; s < sc.n; s += blockDim.x) {
            const int64_t c = sc.idx[s*sc.s_idx0 + t*sc.s_idx1];
            sm[s] = c < n_kv ? ggml_cuda_cast<float>(m[c]) : 0.0f;
        }
    }
    __syncthreads();
    for (int64_t c = threadIdx.x; c < n_kv; c += blockDim.x) {
        d[c] = ggml_cuda_cast<T>(-INFINITY);
    }
    for (int k = 0; k < 2; ++k) {
        const sel_mask_scatter & sc = scs[k];
        const float * sm = smask + off[k];
        __syncthreads();
        for (int s = threadIdx.x; s < sc.n; s += blockDim.x) {
            const int64_t c = sc.idx[s*sc.s_idx0 + t*sc.s_idx1];
            if (c < n_kv) {
                // SET_ROWS rounds the value to T, then ADD sums in F32
                const float v = ggml_cuda_cast<float>(ggml_cuda_cast<T>(sc.val[s*sc.s_val0 + t*sc.s_val1]));
                d[c] = ggml_cuda_cast<T>(v + sm[s]);
            }
        }
    }
}

void ggml_cuda_op_sel_mask(ggml_backend_cuda_context & ctx, const ggml_tensor * mask,
        const sel_mask_scatter * scatters, int n_scatters, ggml_tensor * dst) {
    GGML_ASSERT(n_scatters >= 1 && n_scatters <= 2);
    const sel_mask_scatter none = {};
    const sel_mask_scatter & sc1 = n_scatters > 1 ? scatters[1] : none;
    GGML_ASSERT(scatters[0].n + sc1.n <= SEL_MASK_MAX_SLOTS);

    const int64_t n_kv     = dst->ne[0];
    const int64_t n_tokens = dst->ne[1];
    const size_t  ts       = ggml_type_size(dst->type);
    const int     block    = 256;
    const size_t  smem     = (scatters[0].n + sc1.n)*sizeof(float);
    if (dst->type == GGML_TYPE_F16) {
        sel_mask_f<<<n_tokens, block, smem, ctx.stream()>>>((half *) dst->data, (const half *) mask->data,
            n_kv, dst->nb[1]/ts, mask->nb[1]/ts, scatters[0], sc1);
    } else {
        sel_mask_f<<<n_tokens, block, smem, ctx.stream()>>>((float *) dst->data, (const float *) mask->data,
            n_kv, dst->nb[1]/ts, mask->nb[1]/ts, scatters[0], sc1);
    }
}
