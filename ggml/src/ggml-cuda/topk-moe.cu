#include "ggml-cuda/common.cuh"
#include "ggml.h"
#include "mmvf.cuh"
#include "topk-moe.cuh"

#include <cmath>
#include <initializer_list>

// Kernel config struct - passed by value to CUDA kernel
struct topk_moe_config {
    bool use_sigmoid;
    bool use_sqrt_softplus;
    bool with_norm;
    bool delayed_softmax;
};

// Warp-local softmax used for both the pre-top-k logits and the post-top-k delayed path.
template <int experts_per_thread, bool use_limit>
__device__ void softmax_warp_inplace(float (&vals)[experts_per_thread], const int limit, const int lane) {
    float max_val = -INFINITY;

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        if (active) {
            max_val = max(max_val, vals[i]);
        }
    }

    max_val = warp_reduce_max(max_val);

    float sum = 0.f;

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        if (active) {
            const float val = expf(vals[i] - max_val);
            vals[i]         = val;
            sum += val;
        } else {
            vals[i] = 0.f;
        }
    }

    sum = warp_reduce_sum(sum);

    const float inv_sum = 1.0f / sum;

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        if (active) {
            vals[i] *= inv_sum;
        }
    }
}

template <int experts_per_thread, bool use_limit>
__device__ void sigmoid_warp_inplace(float (&vals)[experts_per_thread], const int limit, const int lane) {
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        vals[i]           = active ? 1.f / (1.f + expf(-vals[i])) : -INFINITY;
    }
}

template <int experts_per_thread, bool use_limit>
__device__ void sqrt_softplus_warp_inplace(float (&vals)[experts_per_thread], const int limit, const int lane) {
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        vals[i]           = active ? sqrtf(vals[i] > 20.0f ? vals[i] : logf(1.0f + expf(vals[i]))) : -INFINITY;
    }
}

static constexpr __host__ __device__ int topk_moe_experts_per_thread(int n_experts) {
    return n_experts > WARP_SIZE ? n_experts / WARP_SIZE : 1;
}

// steps 1-4 of topk_moe_cuda for one row whose logits are in wt, lane by lane across a warp
template <int n_experts, bool has_bias>
static __device__ __forceinline__ void topk_moe_select(float (&wt)[topk_moe_experts_per_thread(n_experts)],
        float * weights, int32_t * ids, const float * bias, bool valid_row, int n_expert_used, float clamp_val,
        float scale_val, const topk_moe_config & config, int lane) {
    constexpr int experts_per_thread = topk_moe_experts_per_thread(n_experts);

    if (!config.delayed_softmax) {
        if (config.use_sigmoid) {
           sigmoid_warp_inplace<experts_per_thread, false>(wt, n_experts, lane);
        } else if (config.use_sqrt_softplus) {
           sqrt_softplus_warp_inplace<experts_per_thread, false>(wt, n_experts, lane);
        } else {
           softmax_warp_inplace<experts_per_thread, false>(wt, n_experts, lane);
        }
    }

    // Sanitize NaN to -FLT_MAX so the iterative argmax produces unique expert IDs.
    // NaN comparisons always return false, which would cause the same expert to be
    // selected repeatedly. -FLT_MAX compares normally and is still excluded by the
    // -INFINITY sentinel used after each selection round.
    // More relevant for the cuBLAS path. See https://github.com/ggml-org/llama.cpp/issues/19659
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        if (__isnanf(wt[i])) {
            wt[i] = -FLT_MAX;
        }
    }

    // selection_wt is only needed when bias is present (selection uses wt + bias)
    // when no bias, we use wt directly for both selection and weight values
    [[maybe_unused]] float selection_wt[has_bias ? experts_per_thread : 1];

    if constexpr (has_bias) {
#pragma unroll
        for (int i = 0; i < experts_per_thread; i++) {
            selection_wt[i] = -INFINITY;
        }
#pragma unroll
        for (int i = 0; i < n_experts; i += WARP_SIZE) {
            const int expert = i + lane;
            selection_wt[i / WARP_SIZE] =
                (n_experts % WARP_SIZE == 0 || expert < n_experts) ? wt[i / WARP_SIZE] + bias[expert] : -INFINITY;
        }
    }

    //at this point, each thread holds either a portion of the softmax distribution
    //or the raw logits. We do the argmax reduce over n_expert_used, each time marking
    //the expert weight as -inf to exclude from the next iteration

    float wt_sum = 0.f;

    float output_weights[experts_per_thread];

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        output_weights[i] = 0.f;
    }

    ggml_cuda_pdl_lc();
    for (int k = 0; k < n_expert_used; k++) {
        float max_val    = wt[0];
        int   max_expert = lane;

        if constexpr (has_bias) {
            float max_val_s = selection_wt[0];

#pragma unroll
            for (int i = 1; i < experts_per_thread; i++) {
                const int expert = lane + i * WARP_SIZE;
                if ((n_experts % WARP_SIZE == 0 || expert < n_experts) && selection_wt[i] > max_val_s) {
                    max_val    = wt[i];
                    max_val_s  = selection_wt[i];
                    max_expert = expert;
                }
            }

#pragma unroll
            for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
                const float val    = __shfl_xor_sync(0xFFFFFFFFULL, max_val, mask, WARP_SIZE);
                const float val_s  = __shfl_xor_sync(0xFFFFFFFFULL, max_val_s, mask, WARP_SIZE);
                const int   expert = __shfl_xor_sync(0xFFFFFFFFULL, max_expert, mask, WARP_SIZE);
                if (val_s > max_val_s || (val_s == max_val_s && expert < max_expert)) {
                    max_val    = val;
                    max_val_s  = val_s;
                    max_expert = expert;
                }
            }

            if ((max_expert & (WARP_SIZE - 1)) == lane) {
                selection_wt[max_expert / WARP_SIZE] = -INFINITY;
            }
        } else {
#pragma unroll
            for (int i = 1; i < experts_per_thread; i++) {
                const int expert = lane + i * WARP_SIZE;
                if ((n_experts % WARP_SIZE == 0 || expert < n_experts) && wt[i] > max_val) {
                    max_val    = wt[i];
                    max_expert = expert;
                }
            }

#pragma unroll
            for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
                const float val    = __shfl_xor_sync(0xFFFFFFFFULL, max_val, mask, WARP_SIZE);
                const int   expert = __shfl_xor_sync(0xFFFFFFFFULL, max_expert, mask, WARP_SIZE);
                if (val > max_val || (val == max_val && expert < max_expert)) {
                    max_val    = val;
                    max_expert = expert;
                }
            }

            if ((max_expert & (WARP_SIZE - 1)) == lane) {
                wt[max_expert / WARP_SIZE] = -INFINITY;
            }
        }

        if ((k & (WARP_SIZE - 1)) == lane) {
            output_weights[k / WARP_SIZE] = max_val;
        }

        if ((max_expert & (WARP_SIZE - 1)) == lane) {
            if (valid_row) {
                ids[k] = max_expert;
            }
            if (config.with_norm) {
                wt_sum += max_val;
            }
        }
    }

    if (config.with_norm) {
        wt_sum              = warp_reduce_sum(wt_sum);
        wt_sum              = max(wt_sum, clamp_val);
        const float inv_sum = 1.0f / wt_sum;

        for (int i = 0; i < experts_per_thread; i++) {
            output_weights[i] *= inv_sum;
        }
    }

    if (config.delayed_softmax) {
        softmax_warp_inplace<experts_per_thread, true>(output_weights, n_expert_used, lane);
    }

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int idx = i * WARP_SIZE + lane;
        if (valid_row && idx < n_expert_used) {
            weights[idx] = output_weights[i] * scale_val;
        }
    }
}

/*
    This kernel does the following:
    1. optionally softmax over the logits per token [n_experts, n_tokens]
    2. argmax reduce over the top-k (n_experts_used) logits
    3. write weights + ids to global memory
    4. optionally normalize the weights or apply softmax over the selected logits

    It is intended as fusion of softmax->top-k->get_rows pipeline for MoE models
*/
template <int n_experts, bool has_bias>
__launch_bounds__(TOPK_MOE_ROWS_PER_BLOCK * WARP_SIZE, 1)
__global__ void topk_moe_cuda(const float *         logits,
                              float *               weights,
                              int32_t *             ids,
                              float *               bias,
                              const int             n_rows,
                              const int             n_expert_used,
                              const float           clamp_val,
                              const float           scale_val,
                              const topk_moe_config config) {
    const bool valid_row = blockIdx.x * blockDim.y + threadIdx.y < n_rows;
#if defined(GGML_USE_MUSA)
    // MUSA: every warp of a partially filled block must reach the barrier below.
    const int row = MIN(blockIdx.x * blockDim.y + threadIdx.y, n_rows - 1);
#else
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
#endif // defined(GGML_USE_MUSA)
    if (row >= n_rows) {
        return;
    }

    logits += n_experts * row;
    weights += n_expert_used * row;
    ids += n_experts * row;

    constexpr int experts_per_thread = topk_moe_experts_per_thread(n_experts);

    float wt[experts_per_thread];

    // Initialize all slots to -INFINITY
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        wt[i] = -INFINITY;
    }

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int i = 0; i < n_experts; i += WARP_SIZE) {
        const int expert  = i + threadIdx.x;
        wt[i / WARP_SIZE] = (n_experts % WARP_SIZE == 0 || expert < n_experts) ? logits[expert] : -INFINITY;
    }

    // Weights and IDs can alias logits, so wait until every row in the block reads its logits.
    __syncthreads();

    topk_moe_select<n_experts, has_bias>(wt, weights, ids, bias, valid_row, n_expert_used, clamp_val, scale_val,
        config, threadIdx.x);
}

// One warp per token runs the selection in the last block, so a block has at least that many warps.
#define ROUTER_TOPK_MOE_WARPS MMVF_F16_MULTI_MAX_TOKENS

// Router matmul + top-k at decode: one warp per expert computes the f16 router logits of
// every token, then the last block to finish runs the selection, one warp per token.
template <int n_experts, bool has_bias, int ncols>
__launch_bounds__(ROUTER_TOPK_MOE_WARPS * WARP_SIZE, 1)
static __global__ void router_topk_moe_cuda(const half * __restrict__ w, const float * __restrict__ x,
        float * logits, float * weights, int32_t * ids, const float * bias, unsigned int * ticket_counter,
        const int k, const int64_t s_w, const int64_t s_x, const int n_expert_used, const float clamp_val,
        const float scale_val, const topk_moe_config config) {
    const int lane = threadIdx.x;
    const int row  = blockIdx.x*ROUTER_TOPK_MOE_WARPS + threadIdx.y;

    ggml_cuda_pdl_sync();
    if (row < n_experts) {
        float acc[ncols] = { 0.0f };
        mmvf_f16_row_dot<ncols>(w + row*s_w, x, s_x, k, lane, acc);
#pragma unroll
        for (int t = 0; t < ncols; ++t) {
            acc[t] = warp_reduce_sum(acc[t]);
            if (lane == 0) {
                logits[t*n_experts + row] = acc[t];
            }
        }
    }

    if (ggml_cuda_block_ticket(ticket_counter) != gridDim.x - 1) {
        return;
    }

    const int t = threadIdx.y;
    float wt[topk_moe_experts_per_thread(n_experts)];
    if (t < ncols) {
#pragma unroll
        for (int i = 0; i < n_experts; i += WARP_SIZE) {
            const int expert  = i + lane;
            wt[i / WARP_SIZE] = (n_experts % WARP_SIZE == 0 || expert < n_experts) ?
                __ldcg(logits + t*n_experts + expert) : -INFINITY;
        }
    }
    // weights and ids can alias the logits of other tokens
    __syncthreads();
    if (t >= ncols) {
        return;
    }
    topk_moe_select<n_experts, has_bias>(wt, weights + t*n_expert_used, ids + t*n_experts, bias, true,
        n_expert_used, clamp_val, scale_val, config, lane);
}

// Large softmax routers must not change their reduction tree merely because
// allocator placement makes the fusion possible. Match soft_max_f32<512,512>
// first, then reduce selected weights in rank order like SUM_ROWS (not in
// expert-lane order). Small/decode routers retain the existing warp path.
static __global__ void topk_moe_softmax512_prefill(
        const float * logits, float * weights, int32_t * ids, int used, float clamp_val) {
    __shared__ float scratch[32];
    __shared__ float probs[512];
    __shared__ int order[512];
    const int lane = threadIdx.x;
    const int64_t row = blockIdx.x;
    const float value = logits[row*512 + lane] + 0.0f;
    const float maximum = block_reduce<block_reduce_method::MAX, 512>(value, scratch);
    const float exponential = expf(value - maximum);
    __syncthreads();
    const float total = block_reduce<block_reduce_method::SUM, 512>(exponential, scratch);
    probs[lane] = exponential * (1.0f / total);
    order[lane] = lane;
    __syncthreads();
    // Use the ordinary bitonic network, including its equal-score ordering.
    // Choosing the smallest expert ID on ties is NOT equivalent to argsort.
    for (int k = 2; k <= 512; k *= 2) {
        for (int j = k/2; j > 0; j /= 2) {
            const int partner = lane ^ j;
            if (partner > lane && ((lane & k) == 0 ?
                    probs[order[lane]] < probs[order[partner]] :
                    probs[order[lane]] > probs[order[partner]])) {
                const int previous = order[lane];
                order[lane] = order[partner];
                order[partner] = previous;
            }
            __syncthreads();
        }
    }
    if (lane >= 32) return;
    const float selected = lane < used ? probs[order[lane]] : 0.0f;
    const float denominator = fmaxf(warp_reduce_sum(selected), clamp_val);
    if (lane < used) {
        ids[row*512 + lane] = order[lane];
        weights[row*used + lane] = selected / denominator;
    }
}

template<bool has_bias>
static void launch_topk_moe_cuda(ggml_backend_cuda_context & ctx,
                                 const float *               logits,
                                 float *                     weights,
                                 int32_t *                   ids,
                                 float *                     bias,
                                 const int                   n_rows,
                                 const int                   n_expert,
                                 const int                   n_expert_used,
                                 const float                 clamp_val,
                                 const float                 scale_val,
                                 const topk_moe_config       config) {
    const int    rows_per_block = TOPK_MOE_ROWS_PER_BLOCK;
    dim3         grid_dims((n_rows + rows_per_block - 1) / rows_per_block, 1, 1);
    dim3         block_dims(WARP_SIZE, rows_per_block, 1);
    cudaStream_t stream = ctx.stream();
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);

    switch (n_expert) {
        case 1:
            ggml_cuda_kernel_launch(topk_moe_cuda<1, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 2:
            ggml_cuda_kernel_launch(topk_moe_cuda<2, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 4:
            ggml_cuda_kernel_launch(topk_moe_cuda<4, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 8:
            ggml_cuda_kernel_launch(topk_moe_cuda<8, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 16:
            ggml_cuda_kernel_launch(topk_moe_cuda<16, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 32:
            ggml_cuda_kernel_launch(topk_moe_cuda<32, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 64:
            ggml_cuda_kernel_launch(topk_moe_cuda<64, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 128:
            ggml_cuda_kernel_launch(topk_moe_cuda<128, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 256:
            ggml_cuda_kernel_launch(topk_moe_cuda<256, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 288: // StepFun 3.7
            ggml_cuda_kernel_launch(topk_moe_cuda<288, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 512:
            ggml_cuda_kernel_launch(topk_moe_cuda<512, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 576:
            ggml_cuda_kernel_launch(topk_moe_cuda<576, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        default:
            GGML_ASSERT(false && "fatal error");
            break;
    }
}

static topk_moe_config topk_moe_make_config(const ggml_cuda_topk_moe_args & args, const ggml_tensor * clamp,
        const ggml_tensor * scale, float & clamp_val, float & scale_val) {
    clamp_val = clamp ? ggml_get_op_params_f32(clamp, 0) : -INFINITY;
    scale_val = scale ? ggml_get_op_params_f32(scale, 0) : 1.0f;

    topk_moe_config config;
    config.use_sigmoid       = args.sigmoid;
    config.use_sqrt_softplus = args.sqrt_softplus;
    config.with_norm         = clamp != nullptr;
    config.delayed_softmax   = args.delayed_softmax;
    GGML_ASSERT(!(config.with_norm && config.delayed_softmax) &&
                "delayed softmax is not supported with weight normalization");
    return config;
}

bool ggml_cuda_router_topk_moe_supported(const ggml_tensor * mm) {
    const int64_t n_expert = mm->src[0]->ne[1];
    return ggml_cuda_mul_mat_vec_f16_multi_supported(mm, mm->src[1]) && ggml_is_contiguous(mm) &&
        (n_expert == 128 || n_expert == 256 || n_expert == 288 || n_expert == 512);
}

template <int n_experts>
static void launch_router_topk_moe_cuda(ggml_backend_cuda_context & ctx, const ggml_tensor * mm, float * weights,
        int32_t * ids, const float * bias, int n_expert_used, float clamp_val, float scale_val,
        const topk_moe_config & config) {
    static_assert(MMVF_F16_MULTI_MAX_TOKENS == 4, "update the token count switch");
    const ggml_tensor * w   = mm->src[0];
    const ggml_tensor * x   = mm->src[1];
    const int64_t       s_w = w->nb[1] / sizeof(half);
    const int64_t       s_x = x->nb[1] / sizeof(float);
    const dim3 grid((n_experts + ROUTER_TOPK_MOE_WARPS - 1) / ROUTER_TOPK_MOE_WARPS, 1, 1);
    const dim3 block(WARP_SIZE, ROUTER_TOPK_MOE_WARPS, 1);
    unsigned int * ticket_counter = ggml_cuda_ticket_counter(ctx, GGML_CUDA_TICKET_ROUTER_TOPK_MOE);
    const ggml_cuda_kernel_launch_params params(grid, block, 0, ctx.stream());
    const auto launch = [&](auto kernel) {
        ggml_cuda_kernel_launch(kernel, params, (const half *) w->data, (const float *) x->data, (float *) mm->data,
            weights, ids, bias, ticket_counter, (int) w->ne[0], s_w, s_x, n_expert_used, clamp_val, scale_val,
            config);
    };
    const auto launch_ncols = [&](auto has_bias_tag) {
        constexpr bool has_bias = decltype(has_bias_tag)::value;
        switch (x->ne[1]) {
            case 1: launch(router_topk_moe_cuda<n_experts, has_bias, 1>); break;
            case 2: launch(router_topk_moe_cuda<n_experts, has_bias, 2>); break;
            case 3: launch(router_topk_moe_cuda<n_experts, has_bias, 3>); break;
            case 4: launch(router_topk_moe_cuda<n_experts, has_bias, 4>); break;
            default: GGML_ABORT("unsupported token count");
        }
    };
    if (bias) {
        launch_ncols(std::true_type{});
    } else {
        launch_ncols(std::false_type{});
    }
}

void ggml_cuda_op_router_topk_moe(ggml_backend_cuda_context &     ctx,
                                  const ggml_tensor *             mm,
                                  ggml_tensor *                   weights,
                                  ggml_tensor *                   ids,
                                  const ggml_tensor *             clamp,
                                  const ggml_tensor *             scale,
                                  const ggml_tensor *             bias,
                                  const ggml_cuda_topk_moe_args & args) {
    GGML_ASSERT(weights->type == GGML_TYPE_F32 && ids->type == GGML_TYPE_I32);
    GGML_ASSERT(ids->nb[1] / ggml_type_size(ids->type) == (size_t) mm->ne[0]);

    float clamp_val;
    float scale_val;
    const topk_moe_config config = topk_moe_make_config(args, clamp, scale, clamp_val, scale_val);

    float *       weights_d     = (float *) weights->data;
    int32_t *     ids_d         = (int32_t *) ids->data;
    const float * bias_d        = bias ? (const float *) bias->data : nullptr;
    const int     n_expert_used = weights->ne[1];

    switch (mm->ne[0]) {
        case 128: launch_router_topk_moe_cuda<128>(ctx, mm, weights_d, ids_d, bias_d, n_expert_used, clamp_val, scale_val, config); break;
        case 256: launch_router_topk_moe_cuda<256>(ctx, mm, weights_d, ids_d, bias_d, n_expert_used, clamp_val, scale_val, config); break;
        case 288: launch_router_topk_moe_cuda<288>(ctx, mm, weights_d, ids_d, bias_d, n_expert_used, clamp_val, scale_val, config); break;
        case 512: launch_router_topk_moe_cuda<512>(ctx, mm, weights_d, ids_d, bias_d, n_expert_used, clamp_val, scale_val, config); break;
        default: GGML_ABORT("unsupported expert count");
    }
}

void ggml_cuda_op_topk_moe(ggml_backend_cuda_context &     ctx,
                           const ggml_tensor *             logits,
                           ggml_tensor *                   weights,
                           ggml_tensor *                   ids,
                           const ggml_tensor *             clamp,
                           const ggml_tensor *             scale,
                           const ggml_tensor *             bias,
                           const ggml_cuda_topk_moe_args & args) {
    GGML_ASSERT(logits->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(ids->type == GGML_TYPE_I32);

    const int n_experts = logits->ne[0];
    const int n_rows    = logits->ne[1];

    const float * logits_d  = (const float *) logits->data;
    float *       weights_d = (float *) weights->data;
    int32_t *     ids_d     = (int32_t *) ids->data;
    float *       bias_d    = bias ? (float *) bias->data : nullptr;

    GGML_ASSERT(ids->nb[1] / ggml_type_size(ids->type) == (size_t) n_experts);

    const int n_expert_used = weights->ne[1];

    float clamp_val;
    float scale_val;
    const topk_moe_config config    = topk_moe_make_config(args, clamp, scale, clamp_val, scale_val);
    const bool            with_norm = config.with_norm;

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    // Compact routing IDs expose this fusion from 256-row prefill batches
    // on both SM75 and SM86; preserve the unfused router's ordered arithmetic.
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if ((cc == 860 || cc == 750) && n_rows >= 256 &&
            n_experts == 512 && n_expert_used > 0 && n_expert_used <= 32 &&
            args.softmax && with_norm && !args.sigmoid && !args.sqrt_softplus &&
            !args.delayed_softmax && !bias && !scale) {
        topk_moe_softmax512_prefill<<<n_rows, 512, 0, ctx.stream()>>>(
                logits_d, weights_d, ids_d, n_expert_used, clamp_val);
        return;
    }
#endif

    if (bias) {
        launch_topk_moe_cuda<true>(ctx, logits_d, weights_d, ids_d, bias_d, n_rows, n_experts, n_expert_used, clamp_val,
                             scale_val, config);
    } else {
        launch_topk_moe_cuda<false>(ctx, logits_d, weights_d, ids_d, bias_d, n_rows, n_experts, n_expert_used, clamp_val,
                             scale_val, config);
    }
}

bool ggml_cuda_should_use_topk_moe(const ggml_tensor * gating_op,
                                   const ggml_tensor * weights,
                                   const ggml_tensor * logits,
                                   const ggml_tensor * ids) {
    // must match an instantiation of launch_topk_moe_cuda: a power of 2 up to 512,
    // or one of the non-power-of-2 expert counts of supported models
    const int n_expert = ids->nb[1] / ids->nb[0];
    if (((n_expert & (n_expert - 1)) != 0 || n_expert > 512) && n_expert != 288 && n_expert != 576) {
        return false;
    }

    if (!ggml_is_contiguous(weights) || !ggml_is_contiguous(logits)) {
        return false;
    }

    if (gating_op->op == GGML_OP_SOFT_MAX) {
        const ggml_tensor * softmax  = gating_op;
        float               scale    = 1.0f;
        float               max_bias = 0.0f;

        memcpy(&scale, (const float *) softmax->op_params + 0, sizeof(float));
        memcpy(&max_bias, (const float *) softmax->op_params + 1, sizeof(float));

        if (!ggml_is_contiguous(softmax->src[0])) {
            return false;
        }

        if (scale != 1.0f || max_bias != 0.0f) {
            return false;
        }

        // don't fuse when masks or sinks are present
        if (softmax->src[1] || softmax->src[2]) {
            return false;
        }
    } else if (gating_op->op == GGML_OP_UNARY) {
        ggml_unary_op op = ggml_get_unary_op(gating_op);

        if (op != GGML_UNARY_OP_SIGMOID && op != GGML_UNARY_OP_SOFTPLUS) {
            return false;
        }
    }

    return true;
}
