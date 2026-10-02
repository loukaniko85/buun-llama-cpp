#include "common.cuh"
#include "convert.cuh"
#include "fwht.cuh"
#include "dequantize.cuh"

template <int N>
static __device__ __forceinline__ void fwht_registers(float * reg, const int lane) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int el_w = N / warp_size;
#pragma unroll
    for (int h = 1; h < warp_size; h *= 2) {
#pragma unroll
        for (int j = 0; j < el_w; ++j) {
            const float x = reg[j];
            const float y = __shfl_xor_sync(0xFFFFFFFF, x, h, warp_size);
            reg[j] = (lane & h) == 0 ? x + y : y - x;
        }
    }
#pragma unroll
    for (int h = warp_size; h < N; h *= 2) {
        const int step = h / warp_size;
#pragma unroll
        for (int j = 0; j < el_w; j += 2 * step) {
#pragma unroll
            for (int k = 0; k < step; ++k) {
                const float x = reg[j + k];
                const float y = reg[j + k + step];
                reg[j + k] = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }
}

template <typename T>
__device__ __forceinline__ float fwht_load(const T value) {
    return value;
}

template <>
__device__ __forceinline__ float fwht_load<half>(const half value) {
    return __half2float(value);
}

template <int N, typename T, bool has_signs>
__launch_bounds__(4*ggml_cuda_get_physical_warp_size(), 1)
__global__ void fwht_cuda(const T * src, float * dst, const int64_t n_rows, const float scale,
                          const float * signs, const int n_blk, const int src_blocks,
                          const int64_t src_row_stride) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    const int64_t r = (int64_t) blockIdx.x * blockDim.y + threadIdx.y;

    if (r >= n_rows) {
        return;
    }

    src += src_row_stride == int64_t(src_blocks) * N ? r * N :
        (r / src_blocks) * src_row_stride + (r % src_blocks) * N;
    dst += r * N;

    static constexpr int el_w = N / warp_size;
    float     reg[el_w];
    const int lane = threadIdx.x;

    ggml_cuda_pdl_sync();
    const float * signs_row = has_signs ? signs + (r % n_blk) * N : nullptr;
#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        float value = fwht_load(src[i * warp_size + lane]);
        if (has_signs) {
            value *= signs_row[i * warp_size + lane];
            if constexpr (sizeof(T) == sizeof(half)) {
                // Preserve the F16 intermediate of the unfused MUL.
                value = __half2float(__float2half_rn(value));
            }
        }
        reg[i] = value * scale;
    }

    fwht_registers<N>(reg, lane);

#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        dst[i * warp_size + lane] = reg[i];
    }
}

template <typename T>
static bool fwht_launch(ggml_backend_cuda_context & ctx, const T * src_d, float * dst_d,
                        const int n, const int64_t rows, const float scale,
                        const float * signs, const int n_blk, const int src_blocks,
                        const int64_t src_row_stride) {
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int rows_per_block = 4;
    const int64_t num_blocks = (rows + rows_per_block - 1) / rows_per_block;
    cudaStream_t stream = ctx.stream();
    dim3 grid_dims(num_blocks, 1, 1);
    dim3 block_dims(warp_size, rows_per_block, 1);
    const ggml_cuda_kernel_launch_params launch_params =
        ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);

    switch (n) {
#define FWHT_CASE(NN) \
        case NN: \
            if (signs) { \
                ggml_cuda_kernel_launch(fwht_cuda<NN, T, true>,  launch_params, src_d, dst_d, rows, scale, signs, n_blk, src_blocks, src_row_stride); \
            } else { \
                ggml_cuda_kernel_launch(fwht_cuda<NN, T, false>, launch_params, src_d, dst_d, rows, scale, nullptr, 1, src_blocks, src_row_stride); \
            } \
            return true;
        FWHT_CASE(64)
        FWHT_CASE(128)
        FWHT_CASE(256)
        FWHT_CASE(512)
        FWHT_CASE(1024)
        FWHT_CASE(2048)
#undef FWHT_CASE
        default:
            return false;
    }
}

static bool fwht_dispatch(ggml_backend_cuda_context & ctx, const ggml_tensor * src, ggml_tensor * dst,
                          const ggml_tensor * signs_t) {
    GGML_ASSERT(ggml_nelements(src) == ggml_nelements(dst));
    // Segment views have contiguous elements but skip the other segments
    // between rows. Higher-dimensional strided layouts keep the fallback.
    if ((!ggml_is_contiguous(src) &&
            (src->ne[2] != 1 || src->ne[3] != 1 || src->nb[0] != ggml_type_size(src->type))) ||
        !ggml_is_contiguous(dst)) {
        return false;
    }
    const int     n    = dst->ne[0];
    const int64_t rows = ggml_nelements(dst) / n;

    if (src->ne[0] % n != 0) {
        return false;
    }
    const int src_blocks = src->ne[0] / n;
    const int64_t src_row_stride = src->nb[1] / ggml_type_size(src->type);

    if ((src->type != GGML_TYPE_F32 && src->type != GGML_TYPE_F16) || dst->type != GGML_TYPE_F32) {
        return false;
    }

    const float * signs = nullptr;
    int n_blk = 1;
    if (signs_t) {
        if (signs_t->type != GGML_TYPE_F32 || !ggml_is_contiguous(signs_t) || signs_t->ne[0] % n != 0) {
            return false;
        }
        signs = (const float *) signs_t->data;
        n_blk = signs_t->ne[0] / n;
    }

    float * dst_d = (float *) dst->data;
    const float scale = 1 / sqrtf(n);

    if (src->type == GGML_TYPE_F32) {
        return fwht_launch<float>(ctx, (const float *) src->data, dst_d, n, rows, scale, signs, n_blk,
                                  src_blocks, src_row_stride);
    }
    return fwht_launch<half>(ctx, (const half *) src->data, dst_d, n, rows, scale, signs, n_blk,
                             src_blocks, src_row_stride);
}

bool ggml_cuda_op_fwht(ggml_backend_cuda_context & ctx, const ggml_tensor * src, ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_shape(src, dst));
    return fwht_dispatch(ctx, src, dst, nullptr);
}

bool ggml_cuda_op_fwht_signed(ggml_backend_cuda_context & ctx, const ggml_tensor * src,
                              const ggml_tensor * signs, ggml_tensor * dst) {
    return fwht_dispatch(ctx, src, dst, signs);
}

struct fwht_segments_args {
    const float * src[3];
    const float * signs[3];
    int64_t stride[3];
    int64_t rows;
};

template<int N>
static __device__ __forceinline__ void fwht_segment_store(const float * src, const float * signs, float * dst) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    float reg[N / warp_size];
    const int lane = threadIdx.x;
    const float scale = 1.0f / sqrtf(float(N));
#pragma unroll
    for (int j = 0; j < N / warp_size; ++j) reg[j] = (src[j*warp_size+lane] * signs[j*warp_size+lane]) * scale;
    fwht_registers<N>(reg, lane);
#pragma unroll
    for (int j = 0; j < N / warp_size; ++j) dst[j*warp_size+lane] = reg[j];
}

template<int segments>
__launch_bounds__(4*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void fwht_segments_cuda(fwht_segments_args args, float * dst) {
    const int64_t group = int64_t(blockIdx.x)*4 + threadIdx.y;
    const int64_t row = group / segments;
    const int segment = group % segments;
    if (row >= args.rows) return;
    const float * src = args.src[segment] + row*args.stride[segment];
    const float * signs = args.signs[segment];
    ggml_cuda_pdl_sync();
    if constexpr (segments == 3) {
        dst += row*2560 + segment*1024;
        if (segment < 2) fwht_segment_store<1024>(src, signs, dst);
        else fwht_segment_store<512>(src, signs, dst);
    } else {
        dst += row*640 + segment*512;
        if (segment == 0) fwht_segment_store<512>(src, signs, dst);
        else fwht_segment_store<128>(src, signs, dst);
    }
}

bool ggml_cuda_op_fwht_segments(ggml_backend_cuda_context & ctx,
                                const ggml_tensor * const * transforms, int count, ggml_tensor * dst) {
    if ((count != 2 && count != 3) || dst->type != GGML_TYPE_F32 || !ggml_is_contiguous(dst) ||
            dst->ne[2] != 1 || dst->ne[3] != 1 || dst->ne[0] != (count == 3 ? 2560 : 640)) return false;
    fwht_segments_args args{};
    args.rows = dst->ne[1];
    for (int i = 0; i < count; ++i) {
        const auto * mm = transforms[i];
        const auto * mul = mm->src[1];
        const auto * src = mul->src[0];
        const auto * signs = mul->src[1];
        const int width = count == 3 ? (i < 2 ? 1024 : 512) : (i == 0 ? 512 : 128);
        if (ggml_get_op_params_i32(mm, 1) != GGML_HINT_SRC0_IS_HADAMARD ||
                mul->type != GGML_TYPE_F32 || src->type != GGML_TYPE_F32 || signs->type != GGML_TYPE_F32 ||
                !ggml_are_same_shape(src, mm) || src->ne[0] != width || src->ne[1] != args.rows ||
                src->ne[2] != 1 || src->ne[3] != 1 || src->nb[0] != sizeof(float) ||
                signs->ne[0] != width || ggml_nelements(signs) != width || !ggml_is_contiguous(signs)) return false;
        args.src[i] = (const float *) src->data;
        args.signs[i] = (const float *) signs->data;
        args.stride[i] = src->nb[1] / sizeof(float);
    }
    const int warp_size = ggml_cuda_info().devices[ctx.device].warp_size;
    const ggml_cuda_kernel_launch_params params(dim3((args.rows*count+3)/4), dim3(warp_size,4), 0, ctx.stream());
    if (count == 3) ggml_cuda_kernel_launch(fwht_segments_cuda<3>, params, args, (float *) dst->data);
    else ggml_cuda_kernel_launch(fwht_segments_cuda<2>, params, args, (float *) dst->data);
    return true;
}

// A CTA cooperates on one 1024-element group. Unlike the ordinary FWHT,
// the Q8 epilogue needs reductions and rounding for every 32-element block;
// four warps avoid serializing all 32 blocks through one warp.
static __device__ __forceinline__ void fwht_cta_1024(float * reg) {
    constexpr int n = 1024;
    constexpr int threads = 128;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const int tid = threadIdx.x;
    const int lane = tid % warp_size;
#pragma unroll
    for (int h = 1; h < warp_size; h *= 2) {
#pragma unroll
        for (int j = 0; j < n / threads; ++j) {
            const float x = reg[j];
            const float y = __shfl_xor_sync(0xFFFFFFFF, x, h, warp_size);
            reg[j] = (lane & h) == 0 ? x + y : y - x;
        }
    }
    __shared__ float buf[n];
#pragma unroll
    for (int j = 0; j < n / threads; ++j) {
        buf[j * threads + tid] = reg[j];
    }
    __syncthreads();
#pragma unroll
    for (int j = 0; j < n / threads; ++j) {
        // Complete the inter-warp stages from the read-only shared tile.
        const int base = j * threads + lane;
        if constexpr (warp_size == 32) {
            const float a = buf[base], b = buf[base + 32];
            const float c = buf[base + 64], d = buf[base + 96];
            const float lo = (tid & 32) ? a - b : a + b;
            const float hi = (tid & 32) ? c - d : c + d;
            reg[j] = (tid & 64) ? lo - hi : lo + hi;
        } else {
            const float a = buf[base], b = buf[base + 64];
            reg[j] = (tid & 64) ? a - b : a + b;
        }
    }
#pragma unroll
    for (int step = 1; step < n / threads; step *= 2) {
#pragma unroll
        for (int j = 0; j < n / threads; j += 2 * step) {
#pragma unroll
            for (int k = 0; k < step; ++k) {
                const float x = reg[j + k], y = reg[j + k + step];
                reg[j + k] = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }
}

static __global__ void fwht_q8_1_cuda(const float * src, const float * signs,
        block_q8_1 * dst, int blocks_per_row) {
    constexpr int n = 1024;
    constexpr int threads = 128;
    const int tid = threadIdx.x;
    const int sign_offset = (blockIdx.x % blocks_per_row) * n;
    const int64_t offset = int64_t(blockIdx.x) * n;
    float reg[n / threads];
    ggml_cuda_pdl_sync();
#pragma unroll
    for (int j = 0; j < n / threads; ++j) {
        reg[j] = (src[offset + j * threads + tid] * signs[sign_offset + j * threads + tid]) * (1.0f / 32.0f);
    }
    fwht_cta_1024(reg);
#pragma unroll
    for (int j = 0; j < n / threads; ++j) {
        const float x = reg[j];
        const float amax = warp_reduce_max<QK8_1>(fabsf(x));
        const float sum = warp_reduce_sum<QK8_1>(x);
        const float d = amax / 127.0f;
        block_q8_1 & b = dst[(offset + j * threads + tid) / QK8_1];
        b.qs[tid % QK8_1] = amax == 0.0f ? 0 : roundf(x / d);
        if (tid % QK8_1 == 0) {
            b.ds = make_half2(d, sum);
        }
    }
}

void ggml_cuda_fwht_q8_1(ggml_backend_cuda_context & ctx, const ggml_tensor * src,
                         const ggml_tensor * signs, void * dst) {
    constexpr int n = 1024;
    GGML_ASSERT(src->type == GGML_TYPE_F32 && ggml_is_contiguous(src) && src->ne[0] % n == 0);
    GGML_ASSERT(signs->type == GGML_TYPE_F32 && ggml_is_contiguous(signs) &&
                ggml_nelements(signs) == src->ne[0]);
    const int64_t rows = ggml_nelements(src) / n;
    const ggml_cuda_kernel_launch_params params(dim3(rows, 1, 1), dim3(128, 1, 1), 0, ctx.stream());
    ggml_cuda_kernel_launch(fwht_q8_1_cuda, params, (const float *) src->data,
        (const float *) signs->data, (block_q8_1 *) dst, int(src->ne[0] / n));
}

template <dequantize_kernel_t dequantize>
static __global__ void get_rows_fwht_cuda(const char * weights, const int32_t * ids,
        const float * signs, float * dst, int blocks_per_row, size_t row_stride) {
    constexpr int n = 1024;
    constexpr int threads = 128;
    const int64_t r = blockIdx.x;
    ggml_cuda_pdl_sync();
    const int lane = threadIdx.x;
    const int offset = (r % blocks_per_row) * n;
    const char * row = weights + int64_t(ids[r / blocks_per_row]) * row_stride;
    float reg[n / threads];
#pragma unroll
    for (int j = 0; j < n / threads; ++j) {
        const int col = offset + j * threads + lane;
        float2 v;
        dequantize(row, col / 128, (col % 128) & ~1, v);
        reg[j] = ((col & 1) ? v.y : v.x) * (1.0f / 32.0f);
    }
    fwht_cta_1024(reg);
#pragma unroll
    for (int j = 0; j < n / threads; ++j) {
        dst[r * n + j * threads + lane] = reg[j] * signs[offset + j * threads + lane];
    }
}

bool ggml_cuda_get_rows_fwht(ggml_backend_cuda_context & ctx, const ggml_tensor * rows,
                             const ggml_tensor * signs, ggml_tensor * dst) {
    const ggml_tensor * w = rows->src[0];
    const ggml_tensor * ids = rows->src[1];
    if ((w->type != GGML_TYPE_PTQ1_0 && w->type != GGML_TYPE_Q2_0_G128) ||
        w->ne[0] % 1024 != 0 || w->ne[2] != 1 || w->ne[3] != 1 ||
        ids->type != GGML_TYPE_I32 || ids->ne[1] != 1 || ids->ne[2] != 1 || ids->ne[3] != 1 ||
        signs->type != GGML_TYPE_F32 || signs->ne[0] != w->ne[0] || ggml_nelements(signs) != w->ne[0] ||
        rows->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || !ggml_are_same_shape(rows, dst) ||
        !ggml_is_contiguous(w) || !ggml_is_contiguous(ids) ||
        !ggml_is_contiguous(signs) || !ggml_is_contiguous(dst)) {
        return false;
    }
    const int64_t nr = ggml_nelements(dst) / 1024;
    const ggml_cuda_kernel_launch_params params(dim3(nr, 1, 1), dim3(128, 1, 1), 0, ctx.stream());
#define LAUNCH_ROWS_FWHT(dequantize) \
    ggml_cuda_kernel_launch(get_rows_fwht_cuda<dequantize>, params, \
        (const char *) w->data, (const int32_t *) ids->data, (const float *) signs->data, \
        (float *) dst->data, int(w->ne[0] / 1024), w->nb[1])
    if (w->type == GGML_TYPE_PTQ1_0) {
        LAUNCH_ROWS_FWHT(dequantize_ptq1_0);
    } else {
        LAUNCH_ROWS_FWHT(dequantize_q2_0_g128);
    }
#undef LAUNCH_ROWS_FWHT
    return true;
}
