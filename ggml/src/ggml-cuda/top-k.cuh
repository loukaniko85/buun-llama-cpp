#include "common.cuh"

void ggml_cuda_op_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_top_k_qsa(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
        const ggml_tensor * scores, const ggml_tensor * cell_blocks, const ggml_tensor * mask);
