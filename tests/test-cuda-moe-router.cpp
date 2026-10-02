#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

// Unlike the CPU-reference TOPK_MOE check, compare ordered IDs AND weights on
// the same device. Sorting the outputs would hide row/rank/ID-weight mismatches.
static bool check(ggml_backend_t backend, int rows, int used, int seed) {
    auto * ctx = ggml_init({ggml_tensor_overhead()*64 + ggml_graph_overhead_custom(64, false), nullptr, true});
    GGML_ASSERT(ctx);
    auto * logits = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 512, rows);
    auto * probs = ggml_soft_max(ctx, logits);
    auto * selected = ggml_argsort_top_k(ctx, probs, used);
    auto * picked = ggml_get_rows(ctx, ggml_reshape_3d(ctx, probs, 1, 512, rows), selected);
    auto * ranks = ggml_reshape_2d(ctx, picked, used, rows);
    auto * denominator = ggml_clamp(ctx, ggml_sum_rows(ctx, ranks), 6.103515625e-5f, INFINITY);
    auto * weights = ggml_reshape_3d(ctx, ggml_div(ctx, ranks, denominator), 1, used, rows);
    auto * graph = ggml_new_graph_custom(ctx, 64, false);
    ggml_build_forward_expand(graph, weights);
    auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    GGML_ASSERT(buffer);

    std::vector<float> input(size_t(rows)*512);
    uint32_t rng = 913 + seed;
    for (auto & value : input) {
        rng = 1664525*rng + 1013904223;
        // Include many exact ties, not just random unique scores.
        value = seed == 0 ? float(int(rng%17) - 8) : float(int(rng >> 8) - 8388608)*0.000003f;
    }
    ggml_backend_tensor_set(logits, input.data(), 0, input.size()*sizeof(float));
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        ggml_set_output(ggml_graph_node(graph, i));
    }
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    std::vector<float> expected(size_t(rows)*used), actual(expected.size());
    std::vector<int32_t> ids(expected.size()), actual_ids(ids.size());
    ggml_backend_tensor_get(weights, expected.data(), 0, expected.size()*sizeof(float));
    const size_t id_bytes = used*sizeof(int32_t);
    ggml_backend_tensor_get_2d(selected, ids.data(), 0, id_bytes, rows, selected->nb[1], id_bytes);
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        ggml_graph_node(graph, i)->flags &= ~GGML_TENSOR_FLAG_OUTPUT;
    }
    // Prove fusion actually ran: it elides probs, while ordinary SOFT_MAX
    // would overwrite this poison. Poison outputs too so stale/no-op writes
    // cannot pass by reusing the reference values still in their buffers.
    const float poison = std::numeric_limits<float>::quiet_NaN();
    std::fill(input.begin(), input.end(), poison);
    std::fill(actual.begin(), actual.end(), poison);
    std::fill(actual_ids.begin(), actual_ids.end(), -1);
    ggml_backend_tensor_set(probs, input.data(), 0, input.size()*sizeof(float));
    ggml_backend_tensor_set(weights, actual.data(), 0, actual.size()*sizeof(float));
    ggml_backend_tensor_set_2d(selected, actual_ids.data(), 0, id_bytes, rows, selected->nb[1], id_bytes);
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_tensor_get(weights, actual.data(), 0, actual.size()*sizeof(float));
    ggml_backend_tensor_get_2d(selected, actual_ids.data(), 0, id_bytes, rows, selected->nb[1], id_bytes);
    ggml_backend_tensor_get(probs, input.data(), 0, input.size()*sizeof(float));
    const bool fused = std::all_of(input.begin(), input.end(), [](float value) { return std::isnan(value); });
    size_t changed = 0, changed_ids = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        changed += std::memcmp(&actual[i], &expected[i], sizeof(float)) != 0 ||
                   !std::isfinite(actual[i]) || !std::isfinite(expected[i]);
        changed_ids += ids[i] != actual_ids[i];
    }
    std::printf("router rows=%d used=%d seed=%d fused=%d changed_weights=%zu changed_ids=%zu\n",
                rows, used, seed, int(fused), changed, changed_ids);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return fused && changed == 0 && changed_ids == 0;
}

int main() {
    // Re-evaluate fusion eligibility after changing observable intermediates.
#ifdef _WIN32
    GGML_ASSERT(_putenv_s("GGML_CUDA_DISABLE_GRAPHS", "1") == 0);
#else
    GGML_ASSERT(setenv("GGML_CUDA_DISABLE_GRAPHS", "1", 1) == 0);
#endif
    const char * disabled = std::getenv("GGML_CUDA_DISABLE_FUSION");
    if (disabled && std::atoi(disabled)) {
        std::fprintf(stderr, "SKIP: router fusion disabled\n");
        return 77;
    }
    cudaDeviceProp props{};
    if (cudaGetDeviceProperties(&props, 0) != cudaSuccess ||
            !((props.major == 7 && props.minor == 5) || (props.major == 8 && props.minor == 6))) {
        std::fprintf(stderr, "SKIP: exact-router qualification requires SM75 or SM86\n");
        return 77;
    }
    auto backend = ggml_backend_cuda_init(0);
    GGML_ASSERT(backend);
    bool ok = true;
    for (int rows : {256, 512, 1024, 2048, 4097}) {
        for (int used : {1, 10, 16, 32}) {
            for (int seed = 0; seed < 3; ++seed) {
                ok = check(backend, rows, used, seed) && ok;
            }
        }
    }
    ggml_backend_free(backend);
    return ok ? 0 : 1;
}
