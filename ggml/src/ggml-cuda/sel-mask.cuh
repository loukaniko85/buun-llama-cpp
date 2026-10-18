#include "common.cuh"

#define SEL_MASK_MAX_SLOTS 8192

// One scatter of F32 values into a row of the mask: row t receives val[s, t] at column idx[s, t].
struct sel_mask_scatter {
    const float   * val;
    const int32_t * idx;
    int             n;
    int64_t         s_val0, s_val1; // elements between slots, tokens
    int64_t         s_idx0, s_idx1;
};

// dst[c, t] = mask[c, t] + (the last value scattered to column c of row t, else -inf), for a -inf fill
// followed by in-order SET_ROWS scatters and an ADD of mask. Columns >= dst->ne[0] are dropped.
// dst, mask: [n_kv, n_tokens] F16 or F32 with unit stride in c; dst may be mask itself (same rows).
// At most SEL_MASK_MAX_SLOTS slots over both scatters.
void ggml_cuda_op_sel_mask(ggml_backend_cuda_context & ctx, const ggml_tensor * mask,
        const sel_mask_scatter * scatters, int n_scatters, ggml_tensor * dst);
