#include "set-rows-concat.cuh"
#include "convert.cuh"

// one block per row; negative rows are skipped, as in SET_ROWS
template <typename idx_t, typename dst_t>
static __global__ void set_rows_concat(const float * __restrict__ a, const float * __restrict__ b,
        const idx_t * __restrict__ idx, dst_t * __restrict__ dst, const int na, const int nb, const int ne0,
        const int64_t s_a, const int64_t s_b, const int64_t s_idx, const int64_t s_dst) {
    const int64_t r   = blockIdx.x;
    const int64_t row = idx[r*s_idx];
    if (row < 0) {
        return;
    }
    dst_t * d = dst + row*s_dst;
    for (int c = threadIdx.x; c < ne0; c += blockDim.x) {
        const float v = c < na ? a[r*s_a + c] : c < na + nb ? b[r*s_b + c - na] : 0.0f;
        d[c] = ggml_cuda_cast<dst_t>(v);
    }
}

template <typename idx_t>
static void set_rows_concat_cuda(ggml_backend_cuda_context & ctx, const ggml_tensor * a, const ggml_tensor * b,
        const ggml_tensor * idx, ggml_tensor * dst) {
    const int64_t n_rows = a->ne[1];
    const int     ne0    = dst->ne[0];
    const int     block  = std::min(256, (ne0 + WARP_SIZE - 1)/WARP_SIZE*WARP_SIZE);
    const size_t  ts     = ggml_type_size(dst->type);
    const float * a_d = (const float *) a->data;
    const float * b_d = (const float *) b->data;
    const idx_t * i_d = (const idx_t *) idx->data;
    const int64_t s_a = a->nb[1]/sizeof(float), s_b = b->nb[1]/sizeof(float);
    const int64_t s_i = idx->nb[0]/sizeof(idx_t), s_d = dst->nb[1]/ts;
    const auto launch = [&](auto * d) {
        set_rows_concat<<<n_rows, block, 0, ctx.stream()>>>(a_d, b_d, i_d, d, a->ne[0], b->ne[0], ne0, s_a, s_b, s_i, s_d);
    };
    switch (dst->type) {
        case GGML_TYPE_F32:  launch((float *)       dst->data); break;
        case GGML_TYPE_F16:  launch((half *)        dst->data); break;
        case GGML_TYPE_BF16: launch((nv_bfloat16 *) dst->data); break;
        default:
            GGML_ABORT("unsupported type");
    }
}

void ggml_cuda_op_set_rows_concat(ggml_backend_cuda_context & ctx, const ggml_tensor * a, const ggml_tensor * b,
        const ggml_tensor * idx, ggml_tensor * set_rows) {
    if (idx->type == GGML_TYPE_I64) {
        set_rows_concat_cuda<int64_t>(ctx, a, b, idx, set_rows);
    } else {
        set_rows_concat_cuda<int32_t>(ctx, a, b, idx, set_rows);
    }
}
