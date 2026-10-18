#include "common.cuh"

// pooled[d, n] = sum_k softmax_k(gate[d, k, n] + ape[d, k]) * key[d, k, n]
// gate, key: [n_embd, kpool, n] F32 with unit stride in d; ape: [n_embd, kpool] F32.
// Without rows, dst is the contiguous F32 [1, n_embd, n] result. With rows (I64 [n]), dst is a SET_ROWS
// target (F32, F16 or BF16, unit stride in d) and pool n goes to its row rows[n].
void ggml_cuda_op_kpool_pool(ggml_backend_cuda_context & ctx, const ggml_tensor * gate, const ggml_tensor * ape,
        const ggml_tensor * key, ggml_tensor * dst, const ggml_tensor * rows = nullptr);
