#include "common.cuh"

// SET_ROWS of PAD(CONCAT(a, b, dim 0)) with zero right padding in dim 0: row r of dst at idx[r] receives
// a[:, r] | b[:, r] | 0 up to dst->ne[0]. a, b: F32 [ne0, n_rows] with unit stride in dim 0;
// idx: I32 or I64 [n_rows]; set_rows: the SET_ROWS node, F32, F16 or BF16.
void ggml_cuda_op_set_rows_concat(ggml_backend_cuda_context & ctx, const ggml_tensor * a, const ggml_tensor * b,
        const ggml_tensor * idx, ggml_tensor * set_rows);
