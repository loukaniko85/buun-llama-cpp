#include "common.cuh"

#define DSA_TOPK_MAX_K 1024

// The DSA indexer selection: per row of score [n_pool, n_rows] F32, the k best pools ordered by descending
// score, expanded to their member cells: dst[c + kpool*s, r] = pool_idxs[c, top(s, r)]. pool_idxs is I32
// [kpool, n_pool], dst I32 contiguous. Replaces TOP_K -> GET_ROWS -> ARGSORT(DESC) -> GET_ROWS -> GET_ROWS.
void ggml_cuda_op_dsa_topk(ggml_backend_cuda_context & ctx, const ggml_tensor * score, const ggml_tensor * pool_idxs,
        int k, ggml_tensor * dst);
