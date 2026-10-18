#include "kpool-pool.cuh"
#include "convert.cuh"

template <typename dst_t>
static __global__ void kpool_pool(
        const float * __restrict__ gate, const float * __restrict__ ape, const float * __restrict__ key,
        dst_t * __restrict__ dst, const int64_t * __restrict__ rows, const int n_embd, const int kpool,
        const int64_t n_pool, const int64_t s_g1, const int64_t s_g2, const int64_t s_a1, const int64_t s_k1,
        const int64_t s_k2, const int64_t s_d) {
    const int d = blockIdx.x*blockDim.x + threadIdx.x;
    if (d >= n_embd) {
        return;
    }

    for (int64_t n = blockIdx.y; n < n_pool; n += gridDim.y) {
        const int64_t row = rows ? rows[n] : n;
        if (row < 0) {
            continue; // as in SET_ROWS
        }
        const float * g = gate + n*s_g2 + d;
        const float * k = key  + n*s_k2 + d;
        const float * a = ape  + d;

        float vmax = -INFINITY;
        for (int i = 0; i < kpool; ++i) {
            vmax = fmaxf(vmax, g[i*s_g1] + a[i*s_a1]);
        }

        float sum = 0.0f;
        float acc = 0.0f;
        for (int i = 0; i < kpool; ++i) {
            const float e = expf(g[i*s_g1] + a[i*s_a1] - vmax);
            sum += e;
            acc += e*k[i*s_k1];
        }

        dst[row*s_d + d] =ggml_cuda_cast<dst_t>(acc/sum);
    }
}

template <typename dst_t>
static void kpool_pool_cuda(ggml_backend_cuda_context & ctx, const ggml_tensor * gate, const ggml_tensor * ape,
        const ggml_tensor * key, dst_t * dst, const int64_t * rows, int64_t s_d) {
    const int     n_embd = gate->ne[0];
    const int     kpool  = gate->ne[1];
    const int64_t n_pool = gate->ne[2];
    const size_t  ts     = sizeof(float);

    const int block_size = 128;
    const dim3 grid((n_embd + block_size - 1)/block_size, (unsigned) std::min<int64_t>(n_pool, 65535), 1);
    kpool_pool<<<grid, block_size, 0, ctx.stream()>>>(
        (const float *) gate->data, (const float *) ape->data, (const float *) key->data, dst, rows,
        n_embd, kpool, n_pool, gate->nb[1]/ts, gate->nb[2]/ts, ape->nb[1]/ts, key->nb[1]/ts, key->nb[2]/ts, s_d);
}

void ggml_cuda_op_kpool_pool(ggml_backend_cuda_context & ctx, const ggml_tensor * gate, const ggml_tensor * ape,
        const ggml_tensor * key, ggml_tensor * dst, const ggml_tensor * rows) {
    const int64_t * rows_d = rows ? (const int64_t *) rows->data : nullptr;
    const int64_t   s_d    = rows ? dst->nb[1]/ggml_type_size(dst->type) : gate->ne[0];
    switch (dst->type) {
        case GGML_TYPE_F32:  kpool_pool_cuda(ctx, gate, ape, key, (float *) dst->data, rows_d, s_d); break;
        case GGML_TYPE_F16:  kpool_pool_cuda(ctx, gate, ape, key, (half *) dst->data, rows_d, s_d); break;
        case GGML_TYPE_BF16: kpool_pool_cuda(ctx, gate, ape, key, (nv_bfloat16 *) dst->data, rows_d, s_d); break;
        default: GGML_ABORT("unsupported type");
    }
}
