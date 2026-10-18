#include "common.cuh"
#include "ggml.h"

void ggml_cuda_op_dsv4_hc_params(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_dsv4_hc_comb(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_dsv4_hc_pre(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_dsv4_hc_post(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_dsv4_hc_mix(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
bool ggml_cuda_dsv4_hc_mix_supported(const ggml_tensor * op);
bool ggml_cuda_dsv4_hc_pre_fused_supported(const ggml_tensor * rms, const ggml_tensor * mm, const ggml_tensor * params);
bool ggml_cuda_dsv4_hc_pre_apply_supported(const ggml_tensor * params, const ggml_tensor * hc_pre);
// hc_post feeding rms: a DSV4_HC_POST with comb whose output rms normalizes, computed by the same pass
bool ggml_cuda_dsv4_hc_post_fold_supported(const ggml_tensor * hc_post, const ggml_tensor * rms);
// rms -> mul: RMS_NORM of the DSV4_HC_PRE output times a weight row; x is the stream the hc_pre RMS_NORM reads
bool ggml_cuda_dsv4_hc_pre_norm_supported(const ggml_tensor * x, const ggml_tensor * hc_pre, const ggml_tensor * rms,
        const ggml_tensor * mul);
// hc_pre (optional): an ungated DSV4_HC_PRE reading pre from params, computed in the same launch
// hc_post (optional): a DSV4_HC_POST passing ggml_cuda_dsv4_hc_post_fold_supported, computed in the same launch
// norm (optional, needs hc_pre): the MUL passing ggml_cuda_dsv4_hc_pre_norm_supported, computed in the same launch
void ggml_cuda_op_dsv4_hc_pre_fused(ggml_backend_cuda_context & ctx,
        const ggml_tensor * rms, const ggml_tensor * mm, ggml_tensor * params, ggml_tensor * hc_pre,
        ggml_tensor * hc_post = nullptr, ggml_tensor * norm = nullptr);

void ggml_cuda_op_hc_combine_fused(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * residual, const ggml_tensor * block,
        const ggml_tensor * repeated, const ggml_tensor * inject,
        ggml_tensor * dst, bool use_repeated_block);
