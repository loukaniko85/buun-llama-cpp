#include "../src/llama-graph.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

#include <cmath>
#include <cstdio>
#include <vector>

// Independent butterfly oracle: runtime graphs multiply by explicit normalized
// matrices (or backend FWHT fusions). Cover the actual 2560/640 Mooney axes,
// multiple expert rows and noncontiguous activations, with per-weight signs.
static bool check(ggml_backend_t backend, const std::vector<int> & blocks, int rows, bool strided) {
    int width = 0;
    for (int n : blocks) width += n;
    auto * ctx = ggml_init({ 2*1024*1024, nullptr, true });
    auto * storage = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width + (strided ? 32 : 0), rows);
    ggml_set_input(storage);
    auto * input = ggml_view_2d(ctx, storage, width, rows, storage->nb[1], 0);
    std::vector<float> data(ggml_nelements(storage)), reference(width*rows);
    for (size_t i = 0; i < data.size(); ++i) data[i] = std::sin(float(i)*0.017f);
    llama_hadamard_transform transform {};
    std::vector<std::vector<float>> matrices, signs;
    int offset = 0;
    for (int n : blocks) {
        auto * rotation = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, n);
        auto * sign = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
        std::vector<float> h(n*n), s(n);
        const float scale = 1.0f/std::sqrt(float(n));
        for (int i = 0; i < n; ++i) {
            s[i] = (i*17 + offset) % 7 < 3 ? -1.0f : 1.0f;
            for (int j = 0; j < n; ++j) {
                unsigned bits = unsigned(i & j), parity = 0;
                while (bits) { parity ^= bits & 1; bits >>= 1; }
                h[i*n+j] = parity ? -scale : scale;
            }
        }
        for (int row = 0; row < rows; ++row) {
            std::vector<float> x(n);
            for (int i = 0; i < n; ++i) x[i] = data[row*storage->ne[0]+offset+i]*s[i];
            for (int step = 1; step < n; step *= 2) {
                for (int base = 0; base < n; base += 2*step) {
                    for (int i = 0; i < step; ++i) {
                        const float a = x[base+i], b = x[base+i+step];
                        x[base+i] = a+b;
                        x[base+i+step] = a-b;
                    }
                }
            }
            for (int i = 0; i < n; ++i) reference[row*width+offset+i] = x[i]*scale;
        }
        transform.segments.push_back({rotation, sign, offset});
        matrices.push_back(std::move(h));
        signs.push_back(std::move(s));
        offset += n;
    }
    auto * output = llama_hadamard_segments_apply(ctx, input, transform);
    ggml_set_output(output);
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    GGML_ASSERT(buffer);
    ggml_backend_tensor_set(storage, data.data(), 0, data.size()*sizeof(float));
    for (size_t i = 0; i < blocks.size(); ++i) {
        ggml_backend_tensor_set(transform.segments[i].rot, matrices[i].data(), 0, matrices[i].size()*sizeof(float));
        ggml_backend_tensor_set(transform.segments[i].signs, signs[i].data(), 0, signs[i].size()*sizeof(float));
    }
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    std::vector<float> actual(reference.size());
    ggml_backend_tensor_get(output, actual.data(), 0, actual.size()*sizeof(float));
    double delta = 0, magnitude = 0;
    bool finite = true;
    for (size_t i = 0; i < actual.size(); ++i) {
        finite &= std::isfinite(actual[i]);
        delta += double(actual[i]-reference[i])*(actual[i]-reference[i]);
        magnitude += double(reference[i])*reference[i];
    }
    const bool ok = finite && delta/(magnitude+1e-20) < 1e-10;
    std::printf("%s width=%d rows=%d strided=%d nmse=%.9g %s\n", ggml_backend_name(backend),
                width, rows, strided, delta/(magnitude+1e-20), ok ? "PASS" : "FAIL");
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return ok;
}

int main() {
    ggml_backend_load_all();
    bool ok = true;
    for (size_t d = 0; d < ggml_backend_dev_count(); ++d) {
        auto * backend = ggml_backend_dev_init(ggml_backend_dev_get(d), nullptr);
        GGML_ASSERT(backend);
        // The 384-wide layout cannot use the specialized segmented fusion:
        // it also pins the row stride of the individual signed FWHT fallback.
        for (const auto & blocks : {std::vector<int>{1024,1024,512}, std::vector<int>{512,128},
                                   std::vector<int>{256,128}, std::vector<int>{128}}) {
            for (int rows : {1,3,4,20}) {
                for (bool strided : {false,true}) ok &= check(backend, blocks, rows, strided);
            }
        }
        ggml_backend_free(backend);
    }
    return ok ? 0 : 1;
}
