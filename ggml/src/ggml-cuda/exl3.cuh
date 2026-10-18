#pragma once

#include "common.cuh"

// EXL3 (exllamav3) trellis-coded weights.  See docs/development/exl3-format-plan.md.
// Tensor: [k, n] with type GGML_TYPE_EXL3_{K}; data = 16x16 tiles in n-tile-major order
// [n/16][k/16][32*K bytes].  MUL_MAT sources: src[2] = svh (F16 [n]), src[3] = suh (F16 [k]).

static inline bool ggml_cuda_is_exl3(ggml_type type) {
    return ggml_type_is_exl3(type);
}

static inline int ggml_cuda_exl3_bits(ggml_type type) {
    return ggml_exl3_bits(type);
}

// 2 = mul1, 1 = mcg, 0 = 3inst
static inline int ggml_cuda_exl3_codebook(ggml_type type) {
    return ggml_exl3_codebook(type);
}

bool ggml_cuda_exl3_supports_mul_mat(const ggml_tensor * dst);

// Only admit the measured Turing image: older PTX may have no tiled GEMM body.
static inline bool ggml_cuda_exl3_turing_gemm_supported(int cc, int compiled_cc, int m, int n, int k) {
    return cc == GGML_CUDA_CC_TURING && compiled_cc == GGML_CUDA_CC_TURING &&
        m >= 9 && (m <= 32 || (m <= 128 && int64_t(n) >= 2 * int64_t(k)));
}

void ggml_cuda_mul_mat_exl3(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

// dst = MUL_MAT(exl3, glu) with glu a SWIGLU_CLAMP of separate gate/up rows, computed inside the
// down projection's input load. With sum, writes residual + dst there instead of dst (sum may
// alias residual). False (nothing launched) outside its single-token int8 shapes.
bool ggml_cuda_mul_mat_exl3_glu(ggml_backend_cuda_context & ctx, const ggml_tensor * glu, ggml_tensor * dst,
        const ggml_tensor * residual = nullptr, ggml_tensor * sum = nullptr);

// Pair independent dense projections sharing an input, preserving both outputs.
// Caller validates graph dependencies and memory ranges; false retains ordinary execution.
bool ggml_cuda_exl3_bundle(ggml_backend_cuda_context & ctx, ggml_tensor * a, ggml_tensor * b);

// MUL_MAT_ID with EXL3 experts: grouped kernel (no host sync) for decode shapes; src[3] = svh
// [n, n_expert], src[4] = suh [k, n_expert].
bool ggml_cuda_exl3_mul_mat_id_fast(const ggml_tensor * dst);
void ggml_cuda_mul_mat_id_exl3(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// Whole decode-sized mul1 MoE layer in one launch: up/gate MUL_MAT_ID -> swiglu (clamped when limit > 0)
// -> down MUL_MAT_ID -> router-weighted expert sum into dst [k, n_tokens]. sh_up/sh_gate/sh_down (all or
// none): a dense shared expert MUL_MAT FFN on the same rows with the same swiglu, added to dst. dst_aliases_src:
// dst overlaps a source, so the result goes through scratch. False (nothing launched) when the shape, routing
// or arch is unsupported.
bool ggml_cuda_exl3_moe_fused(ggml_backend_cuda_context & ctx, const ggml_tensor * up, const ggml_tensor * gate,
    const ggml_tensor * down, float limit, const ggml_tensor * weights, const ggml_tensor * sh_up,
    const ggml_tensor * sh_gate, const ggml_tensor * sh_down, ggml_tensor * dst, bool dst_aliases_src);

// CPU MoE-cache bridge: dot products of packed weights and pre-transformed
// F16-rounded F32 activations. The caller owns both Hadamard transforms.
void ggml_cuda_exl3_cache_mmv(const void * weights, ggml_type type,
    const float * activations, const int32_t * slots, const int32_t * activation_ids,
    float * output, int k, int n, size_t expert_bytes, int rows, int activation_rows,
    cudaStream_t stream);

// Dequantize the whole tensor to F16 rows W[n][k] (row n contiguous over k); rows [n0, n1).
void ggml_cuda_exl3_reconstruct_rows(const ggml_tensor * src0, int64_t n0, int64_t n1, half * dst, cudaStream_t stream);
