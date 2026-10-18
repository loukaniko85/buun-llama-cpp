#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "test-exl3-gpu-common.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

// A clamped SwiGLU feeding an EXL3 projection may be computed inside the projection's
// input load. The same SwiGLU kept as a graph output cannot be fused, so the two
// projections must agree bit for bit, also with a following residual ADD.
static bool check_glu(ggml_backend_t backend, int bits, int k, int n, bool add) {
    constexpr float limit = 7.0f;
    auto * ctx = ggml_init({1024*1024, nullptr, true});
    auto * compute = ggml_init({1024*1024, nullptr, true});
    auto * w = ggml_new_tensor_2d(ctx, ggml_exl3_type(bits, 2), k, n);
    auto * gate = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, 1);
    auto * up = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, 1);
    auto * svh = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, n);
    auto * suh = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, k);
    auto * residual = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, 1);
    auto * graph = ggml_new_graph_custom(compute, 128, false);
    ggml_tensor * out[2];
    for (int i = 0; i < 2; ++i) {
        auto * glu = ggml_swiglu_clamp(compute, gate, up, limit);
        if (i == 1) ggml_set_output(glu);
        out[i] = ggml_mul_mat(compute, w, glu);
        out[i]->src[2] = svh;
        out[i]->src[3] = suh;
        if (add) out[i] = ggml_add(compute, residual, out[i]);
        ggml_set_output(out[i]);
        ggml_build_forward_expand(graph, out[i]);
    }
    auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    auto * alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    GGML_ASSERT(buffer && ggml_gallocr_alloc_graph(alloc, graph));
    unsigned rng = 7654321;
    auto next = [&]() { rng = rng*1664525u + 1013904223u; return rng; };
    std::vector<unsigned char> weights(ggml_nbytes(w));
    for (auto & v : weights) v = next() >> 24;
    ggml_backend_tensor_set(w, weights.data(), 0, weights.size());
    // wide enough that both clamps engage
    std::vector<float> input(std::max(k, n));
    for (auto * t : { gate, up, residual }) {
        for (auto & v : input) v = (float(next() >> 8)/16777216.0f - 0.5f)*24.0f;
        ggml_backend_tensor_set(t, input.data(), 0, ggml_nbytes(t));
    }
    std::vector<ggml_fp16_t> signs(k), scales(n);
    for (int i = 0; i < k; ++i) signs[i] = ggml_fp32_to_fp16(i%3 ? 1.0f : -1.0f);
    for (int i = 0; i < n; ++i) scales[i] = ggml_fp32_to_fp16((i%7 ? 1.0f : -1.0f)*(0.01f + float(i%13)*0.002f));
    ggml_backend_tensor_set(suh, signs.data(), 0, ggml_nbytes(suh));
    ggml_backend_tensor_set(svh, scales.data(), 0, ggml_nbytes(svh));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    std::vector<float> fused(n), expected(n);
    ggml_backend_tensor_get(out[0], fused.data(), 0, ggml_nbytes(out[0]));
    ggml_backend_tensor_get(out[1], expected.data(), 0, ggml_nbytes(out[1]));
    bool ok = true;
    for (float v : expected) ok &= std::isfinite(v);
    ok &= memcmp(fused.data(), expected.data(), n*sizeof(float)) == 0;
    ggml_gallocr_free(alloc);
    ggml_free(compute);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    printf("K%d k=%d n=%d add=%d: %s\n", bits, k, n, add, ok ? "PASS" : "FAIL");
    return ok;
}

int main() {
    auto * reg = ggml_backend_cuda_reg();
    if (ggml_backend_reg_dev_count(reg) == 0) return 77;
    auto * backend = ggml_backend_dev_init(ggml_backend_reg_dev_get(reg, 0), nullptr);
    GGML_ASSERT(backend);
    if (!exl3_gpu_supported(backend)) {
        ggml_backend_free(backend);
        return 77;
    }
    bool ok = true;
    for (int bits = 1; bits <= 8; ++bits) {
        for (bool add : { false, true }) {
            ok &= check_glu(backend, bits, 1536, 4096, add);
            ok &= check_glu(backend, bits, 4096, 512, add);
        }
    }
    ggml_backend_free(backend);
    return ok ? 0 : 1;
}
