#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml.h"

#include <cstdio>
#include <vector>

static ggml_backend_meta_split_state mirrored(const ggml_tensor *, void *) {
    return { GGML_BACKEND_SPLIT_AXIS_MIRRORED, { 0 }, { 1 }, 1 };
}

int main() {
    ggml_backend_load_all();
    std::vector<ggml_backend_dev_t> devices;
    for (size_t i = 0; i < ggml_backend_dev_count() && devices.size() < 2; ++i) {
        auto dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            devices.push_back(dev);
        }
    }
    if (devices.size() < 2) {
        fprintf(stderr, "SKIP: requires two GPUs\n");
        return 77;
    }
    auto             dev = ggml_backend_meta_device(devices.data(), devices.size(), mirrored, nullptr);
    ggml_backend_ptr backend(ggml_backend_dev_init(dev, nullptr));
    auto             buft = ggml_backend_dev_buffer_type(dev);
    // More than the old 16-record limit: a sweep of CPU-separated layer
    // fragments must keep selecting the correct storage when graphs recur.
    constexpr int n_weights = 32;
    ggml_context_ptr weights_ctx(ggml_init({ ggml_tensor_overhead() * (n_weights + 2), nullptr, true }));
    ggml_tensor *    weights[n_weights];
    for (int i = 0; i < n_weights; ++i) {
        weights[i] = ggml_new_tensor_2d(weights_ctx.get(), GGML_TYPE_F32, 128, 16);
        ggml_format_name(weights[i], "weight-%d", i);
    }
    ggml_backend_buffer_ptr weights_buf(ggml_backend_alloc_ctx_tensors_from_buft(weights_ctx.get(), buft));
    GGML_ASSERT(weights_buf);
    ggml_backend_buffer_set_usage(weights_buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    // Static meta tensors deliberately share a placeholder data address. Their
    // actual device storage differs and must distinguish recorded graph keys.
    GGML_ASSERT(weights[0]->data == weights[1]->data);
    for (int i = 0; i < n_weights; ++i) {
        std::vector<float> values(128 * 16, float(i + 1));
        ggml_backend_tensor_set(weights[i], values.data(), 0, values.size() * sizeof(float));
    }
    ggml_backend_ptr           cpu(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr));
    ggml_backend_t             backends[] = { backend.get(), cpu.get() };
    ggml_backend_buffer_type_t bufts[]    = { buft, ggml_backend_get_default_buffer_type(cpu.get()) };
    ggml_backend_sched_ptr     sched(ggml_backend_sched_new(backends, bufts, 2, 128, false, true));
    bool                       passed = true;
    for (int iteration = 0; iteration < 3 * n_weights; ++iteration) {
        const int wi = iteration % n_weights;
        ggml_backend_sched_reset(sched.get());
        ggml_context_ptr ctx(
            ggml_init({ ggml_tensor_overhead() * 8 + ggml_graph_overhead_custom(128, false), nullptr, true }));
        auto * x = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 128, 1);
        ggml_set_input(x);
        auto * y = ggml_mul_mat(ctx.get(), weights[wi], x);
        ggml_set_output(y);
        auto * graph = ggml_new_graph_custom(ctx.get(), 128, false);
        ggml_build_forward_expand(graph, y);
        GGML_ASSERT(ggml_backend_sched_alloc_graph(sched.get(), graph));
        std::vector<float> input(128, 1.f), output(16);
        ggml_backend_tensor_set(x, input.data(), 0, input.size() * sizeof(float));
        for (int repeat = 0; repeat < 3; ++repeat) {
            GGML_ASSERT(ggml_backend_sched_graph_compute(sched.get(), graph) == GGML_STATUS_SUCCESS);
            ggml_backend_tensor_get(y, output.data(), 0, output.size() * sizeof(float));
            const float expected = 128.f * (wi + 1);
            for (float value : output) {
                passed = passed && value == expected;
            }
            printf("weight=%d repeat=%d output=%g expected=%g\n", wi, repeat, output[0], expected);
        }
        ggml_backend_sched_reset(sched.get());
    }

    // Scheduler graph views omit their input leaves. Emulate a staging buffer
    // reused for different weight formats: identical addresses and output
    // shapes must not let F16 replay an executable recorded for F32.
    auto * half_weights = ggml_new_tensor_2d(weights_ctx.get(), GGML_TYPE_F16, 128, 16);
    ggml_set_name(half_weights, "staged-half-weights");
    GGML_ASSERT(ggml_backend_tensor_alloc(weights_buf.get(), half_weights,
                ggml_backend_buffer_get_base(weights_buf.get())) == GGML_STATUS_SUCCESS);
    for (size_t rank = 0; rank < devices.size(); ++rank) {
        GGML_ASSERT(ggml_backend_meta_buffer_simple_tensor(half_weights, rank)->data ==
                    ggml_backend_meta_buffer_simple_tensor(weights[0], rank)->data);
    }
    for (bool half : { false, true, false, true }) {
        ggml_context_ptr ctx(
            ggml_init({ ggml_tensor_overhead() * 8 + ggml_graph_overhead_custom(128, false), nullptr, true }));
        auto * x = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 128, 1);
        ggml_set_input(x);
        auto * w = half ? half_weights : weights[0];
        auto * y = ggml_mul_mat(ctx.get(), w, x);
        ggml_set_output(y);
        auto * graph = ggml_new_graph_custom(ctx.get(), 128, false);
        ggml_build_forward_expand(graph, y);
        GGML_ASSERT(ggml_backend_sched_alloc_graph(sched.get(), graph));
        if (half) {
            std::vector<ggml_fp16_t> values(128 * 16, ggml_fp32_to_fp16(2.f));
            ggml_backend_tensor_set(w, values.data(), 0, values.size() * sizeof(ggml_fp16_t));
        } else {
            std::vector<float> values(128 * 16, 1.f);
            ggml_backend_tensor_set(w, values.data(), 0, values.size() * sizeof(float));
        }
        std::vector<float> input(128, 1.f), output(16);
        ggml_backend_tensor_set(x, input.data(), 0, input.size() * sizeof(float));
        for (int repeat = 0; repeat < 4; ++repeat) {
            GGML_ASSERT(ggml_backend_sched_graph_compute(sched.get(), graph) == GGML_STATUS_SUCCESS);
            ggml_backend_tensor_get(y, output.data(), 0, output.size() * sizeof(float));
            const float expected = half ? 256.f : 128.f;
            for (float value : output) {
                passed = passed && value == expected;
            }
            printf("staged=%s repeat=%d output=%g expected=%g\n", ggml_type_name(w->type), repeat, output[0], expected);
        }
        ggml_backend_sched_reset(sched.get());
    }
    return passed ? 0 : 1;
}
