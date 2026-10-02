// Unit tests for quantization specific functions - quantize, dequantize and dot product

#include "ggml.h"
#include "ggml-cpu.h"
#include "gguf.h"

#undef NDEBUG
#include <assert.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <math.h>
#include <stdio.h>
#include <string>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

constexpr float MAX_QUANTIZATION_REFERENCE_ERROR = 0.0001f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR = 0.002f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_BINARY = 0.025f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_TERNARY = 0.01f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_2BITS = 0.0075f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_3BITS = 0.0040f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_3BITS_XXS = 0.0050f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_FP4 = 0.0030f;
constexpr float MAX_DOT_PRODUCT_ERROR = 0.02f;
constexpr float MAX_DOT_PRODUCT_ERROR_LOWBIT = 0.04f;
constexpr float MAX_DOT_PRODUCT_ERROR_FP4 = 0.03f;
constexpr float MAX_DOT_PRODUCT_ERROR_BINARY = 0.40f;
constexpr float MAX_DOT_PRODUCT_ERROR_TERNARY = 0.15f;

static const char* RESULT_STR[] = {"ok", "FAILED"};


// Generate synthetic data
static void generate_data(float offset, size_t n, float * dst, float amplitude = 2.0f) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = 0.1 + amplitude*cosf(i + offset);
    }
}

// Calculate RMSE between two float arrays
static float array_rmse(const float * a1, const float * a2, size_t n) {
    double sum = 0;
    for (size_t i = 0; i < n; i++) {
        double diff = a1[i] - a2[i];
        sum += diff * diff;
    }
    return sqrtf(sum) / n;
}

// Total quantization error on test data
static float total_quantization_error(const ggml_type_traits * qfns, const ggml_type_traits_cpu * qfns_cpu, size_t test_size, const float * test_data) {
    std::vector<uint8_t> tmp_q(2*test_size);
    std::vector<float> tmp_out(test_size);

    qfns_cpu->from_float(test_data, tmp_q.data(), test_size);
    qfns->to_float(tmp_q.data(), tmp_out.data(), test_size);
    return array_rmse(test_data, tmp_out.data(), test_size);
}

// Total quantization error on test data
static float reference_quantization_error(const ggml_type_traits * qfns, const ggml_type_traits_cpu * qfns_cpu, size_t test_size, const float * test_data) {
    std::vector<uint8_t> tmp_q(2*test_size);
    std::vector<float> tmp_out(test_size);
    std::vector<float> tmp_out_ref(test_size);

    // FIXME: why is done twice?
    qfns_cpu->from_float(test_data, tmp_q.data(), test_size);
    qfns->to_float(tmp_q.data(), tmp_out.data(), test_size);

    qfns->from_float_ref(test_data, tmp_q.data(), test_size);
    qfns->to_float(tmp_q.data(), tmp_out_ref.data(), test_size);

    return array_rmse(tmp_out.data(), tmp_out_ref.data(), test_size);
}

static float dot_product(const float * a1, const float * a2, size_t test_size) {
    double sum = 0;
    for (size_t i = 0; i < test_size; i++) {
        sum += a1[i] * a2[i];
    }
    return sum;
}

// Total dot product error
static float dot_product_error(const ggml_type_traits_cpu * qfns_cpu, ggml_type src0_type, size_t test_size,
                               const float * test_data1, const float * test_data2,
                               const float * test_data3, const float * test_data4,
                               const int nrc) {
    const auto * vdot = ggml_get_type_traits_cpu(qfns_cpu->vec_dot_type);
    const size_t pad  = 64;
    const size_t bx   = ggml_row_size(src0_type, test_size) + pad;
    const size_t by   = ggml_row_size(qfns_cpu->vec_dot_type, test_size) + pad;

    std::vector<uint8_t> tmp_q1(bx * nrc);
    std::vector<uint8_t> tmp_q2(by * nrc);

    qfns_cpu->from_float(test_data1, tmp_q1.data(), test_size);
    vdot->from_float(test_data2, tmp_q2.data(), test_size);

    if (nrc == 1) {
        float result = INFINITY;
        qfns_cpu->vec_dot(test_size, &result, 0, tmp_q1.data(), 0, tmp_q2.data(), 0, 1);

        const float dot_ref = dot_product(test_data1, test_data2, test_size);
        return fabsf(result - dot_ref) / test_size;
    }

    // nrc == 2: kernel computes a 2x2 dot product matrix
    // Output layout: s[0]=dot(vx0,vy0), s[1]=dot(vx1,vy0), s[bs]=dot(vx0,vy1), s[bs+1]=dot(vx1,vy1)
    // row and output strides are padded, same as in the mul_mat path
    qfns_cpu->from_float(test_data3, tmp_q1.data() + bx, test_size);
    vdot->from_float(test_data4, tmp_q2.data() + by, test_size);

    const size_t bs = 16;
    std::vector<float> result(bs + 2, INFINITY);
    qfns_cpu->vec_dot(test_size, result.data(), bs, tmp_q1.data(), bx, tmp_q2.data(), by, 2);

    const float ref00 = dot_product(test_data1, test_data2, test_size);
    const float ref10 = dot_product(test_data3, test_data2, test_size);
    const float ref01 = dot_product(test_data1, test_data4, test_size);
    const float ref11 = dot_product(test_data3, test_data4, test_size);

    const auto err = [test_size](float val, float ref) {
        const float e = fabsf(val - ref) / test_size;
        return std::isfinite(e) ? e : INFINITY;
    };

    return std::max({err(result[0], ref00), err(result[1], ref10), err(result[bs], ref01), err(result[bs + 1], ref11)});
}

static int test_vec_dot_f32(bool verbose) {
    const auto * f32 = ggml_get_type_traits_cpu(GGML_TYPE_F32);
    int num_failed = 0;
    for (int n : {1, 2, 3, 5, 7, 8, 15, 16, 17, 31, 33, 63, 67, 127, 129, 193, 255, 1023}) {
        std::vector<float> a(n);
        std::vector<float> b(n);
        generate_data(0.0, n, a.data());
        generate_data(1.0, n, b.data());

        float result = 0.0f;
        f32->vec_dot(n, &result, 0, a.data(), 0, b.data(), 0, 1);
        const float ref = dot_product(a.data(), b.data(), n);
        const float error = fabsf(result - ref) / n;

        const bool failed = !(error < MAX_QUANTIZATION_REFERENCE_ERROR);
        num_failed += failed;
        if (failed || verbose) {
            printf(" f32 vec_dot n=%4d:                 %s (ref=%f got=%f err=%f)\n",
                   n, RESULT_STR[failed], ref, result, error);
        }
    }
    return num_failed;
}

static int test_f8_e4m3_known_codes(bool verbose) {
    const auto * traits = ggml_get_type_traits(GGML_TYPE_F8_E4M3);
    const uint8_t codes[] = { 0x00, 0x01, 0x07, 0x08, 0x38, 0x77, 0x78, 0x7e, 0x80, 0xb8, 0xfe };
    const float expected[] = {
        0.0f, 0.001953125f, 0.013671875f, 0.015625f, 1.0f, 240.0f,
        256.0f, 448.0f, -0.0f, -1.0f, -448.0f,
    };
    float actual[sizeof(codes)] = {};
    traits->to_float(codes, actual, sizeof(codes));

    int num_failed = 0;
    for (size_t i = 0; i < sizeof(codes); ++i) {
        const bool failed = actual[i] != expected[i] || (codes[i] == 0x80 && !signbit(actual[i]));
        num_failed += failed;
        if (failed || verbose) {
            printf(" f8_e4m3 code 0x%02x:              %s (expected=%g got=%g)\n",
                   codes[i], RESULT_STR[failed], expected[i], actual[i]);
        }
    }

    const float infinities[] = { INFINITY, -INFINITY };
    uint8_t     saturated[2] = {};
    traits->from_float_ref(infinities, saturated, 2);
    for (size_t i = 0; i < 2; ++i) {
        const uint8_t expected_code = i == 0 ? 0x7e : 0xfe;
        const bool    failed        = saturated[i] != expected_code;
        num_failed += failed;
        if (failed || verbose) {
            printf(" f8_e4m3 infinity %zu:               %s (expected=0x%02x got=0x%02x)\n", i,
                   RESULT_STR[failed], expected_code, saturated[i]);
        }
    }
    return num_failed;
}

static int test_vec_dot_q(bool verbose) {
    int num_failed = 0;

    const size_t test_size = 32 * 128;

    std::vector<float> test_data(test_size);
    std::vector<float> test_data2(test_size);
    std::vector<float> test_data3(test_size);
    std::vector<float> test_data4(test_size);

    generate_data(0.0, test_data.size(), test_data.data());
    generate_data(1.0, test_data2.size(), test_data2.data());
    generate_data(3.0, test_data3.size(), test_data3.data(), 1.0f);
    generate_data(4.0, test_data4.size(), test_data4.data(), 1.5f);

    for (int i = 0; i < GGML_TYPE_COUNT; i++) {
        ggml_type type = (ggml_type) i;
        const auto * qfns = ggml_get_type_traits(type);
        const auto * qfns_cpu = ggml_get_type_traits_cpu(type);

        // deprecated - skip
        if (qfns->blck_size == 0) {
            continue;
        }

        // fork KV-cache codecs (turbo/TCQ) are not weight-quant types: they have no
        // vec_dot by design (decode is fused inside flash-attention) and their error
        // profile is defined on FWHT-rotated distributions, not this test's raw data.
        // They are gated by the KLD suites; running them here null-calls vec_dot.
        if (ggml_is_turbo_kv_type(type)) {
            continue;
        }

        const ggml_type ei = (ggml_type)i;

        printf("Testing %s\n", ggml_type_name((ggml_type) i));
        ggml_quantize_init(ei);

        if (qfns_cpu->from_float && qfns->to_float) {
            const float total_error = total_quantization_error(qfns, qfns_cpu, test_size, test_data.data());
            const float max_quantization_error =
                type == GGML_TYPE_Q1_0    ? MAX_QUANTIZATION_TOTAL_ERROR_BINARY :
                type == GGML_TYPE_TQ1_0   ? MAX_QUANTIZATION_TOTAL_ERROR_TERNARY :
                type == GGML_TYPE_TQ2_0   ? MAX_QUANTIZATION_TOTAL_ERROR_TERNARY :
                type == GGML_TYPE_Q2_0    ? MAX_QUANTIZATION_TOTAL_ERROR_TERNARY :
                type == GGML_TYPE_Q2_0_G128 ? MAX_QUANTIZATION_TOTAL_ERROR_TERNARY :
                type == GGML_TYPE_PTQ1_0 ? MAX_QUANTIZATION_TOTAL_ERROR_TERNARY :
                type == GGML_TYPE_Q2_K    ? MAX_QUANTIZATION_TOTAL_ERROR_2BITS :
                type == GGML_TYPE_IQ2_S   ? MAX_QUANTIZATION_TOTAL_ERROR_2BITS :
                type == GGML_TYPE_Q3_K    ? MAX_QUANTIZATION_TOTAL_ERROR_3BITS :
                type == GGML_TYPE_IQ3_S   ? MAX_QUANTIZATION_TOTAL_ERROR_3BITS :
                type == GGML_TYPE_IQ3_XXS ? MAX_QUANTIZATION_TOTAL_ERROR_3BITS_XXS :
                type == GGML_TYPE_NVFP4   ? MAX_QUANTIZATION_TOTAL_ERROR_FP4 : MAX_QUANTIZATION_TOTAL_ERROR;
            bool failed = !(total_error < max_quantization_error);
            num_failed += failed;
            if (failed || verbose) {
                printf("%5s absolute quantization error:    %s (%f)\n", ggml_type_name(type), RESULT_STR[failed], total_error);
            }

            const float reference_error = reference_quantization_error(qfns, qfns_cpu, test_size, test_data.data());
            failed = !(reference_error < MAX_QUANTIZATION_REFERENCE_ERROR);
            num_failed += failed;
            if (failed || verbose) {
                printf("%5s reference implementation error: %s (%f)\n", ggml_type_name(type), RESULT_STR[failed], reference_error);
            }

            const float vec_dot_error = dot_product_error(qfns_cpu, type, test_size, test_data.data(), test_data2.data(), nullptr, nullptr, 1);
            const float max_allowed_error = type == GGML_TYPE_Q2_K || type == GGML_TYPE_IQ2_XS || type == GGML_TYPE_IQ2_XXS ||
                type == GGML_TYPE_IQ3_XXS || type == GGML_TYPE_IQ3_S || type == GGML_TYPE_IQ2_S
                ? MAX_DOT_PRODUCT_ERROR_LOWBIT
                : type == GGML_TYPE_Q1_0
                ? MAX_DOT_PRODUCT_ERROR_BINARY
                : type == GGML_TYPE_TQ1_0 || type == GGML_TYPE_TQ2_0 || type == GGML_TYPE_Q2_0 || type == GGML_TYPE_Q2_0_G128 || type == GGML_TYPE_PTQ1_0
                ? MAX_DOT_PRODUCT_ERROR_TERNARY
                : type == GGML_TYPE_NVFP4
                ? MAX_DOT_PRODUCT_ERROR_FP4
                : MAX_DOT_PRODUCT_ERROR;
            failed = !(vec_dot_error < max_allowed_error);
            num_failed += failed;
            if (failed || verbose) {
                printf("%5s dot product error:              %s (%f)\n", ggml_type_name(type), RESULT_STR[failed], vec_dot_error);
            }

            // Test nrc=2 path for types that support it
            if (qfns_cpu->nrows == 2) {
                const float vec_dot_error_nrc2 = dot_product_error(qfns_cpu, type, test_size, test_data.data(), test_data2.data(), test_data3.data(), test_data4.data(), 2);
                failed = !(vec_dot_error_nrc2 < max_allowed_error);
                num_failed += failed;
                if (failed || verbose) {
                    printf("%5s dot product error (nrc=2):    %s (%f)\n", ggml_type_name(type), RESULT_STR[failed], vec_dot_error_nrc2);
                }
            }
        }
    }

    return num_failed;
}

static void test_bonsai_codecs() {
    constexpr int width = 384, rows = 3, count = width * rows;
    std::vector<float> input(count), pq(count), ptq(count);
    for (int i = 0; i < count; ++i) {
        input[i] = (i / 128) * 0.125f * ((i * 17 % 3) - 1);
    }
    for (const auto type : {GGML_TYPE_Q2_0_G128, GGML_TYPE_PTQ1_0}) {
        std::vector<uint8_t> packed(ggml_row_size(type, width) * rows);
        const size_t written = ggml_quantize_chunk(type, input.data(), packed.data(), 0, rows, width, nullptr);
        assert(written == packed.size());
        assert(ggml_validate_row_data(type, packed.data(), packed.size()));
        auto & output = type == GGML_TYPE_PTQ1_0 ? ptq : pq;
        ggml_get_type_traits(type)->to_float(packed.data(), output.data(), count);
        assert(output == input);

        ggml_context * ctx = ggml_init({ggml_tensor_overhead(), nullptr, true});
        assert(ctx);
        ggml_tensor * tensor = ggml_new_tensor_1d(ctx, type, width);
        ggml_set_name(tensor, "bonsai");
        gguf_context * file = gguf_init_empty();
        gguf_add_tensor(file, tensor);
        std::vector<uint8_t> metadata(gguf_get_meta_size(file));
        gguf_get_meta_data(file, metadata.data());
        // Header, tensor name length/name, dimension count and one dimension.
        uint32_t wire_type = 0;
        memcpy(&wire_type, metadata.data() + 24 + 8 + strlen("bonsai") + 4 + 8, sizeof(wire_type));
        assert(wire_type == (type == GGML_TYPE_PTQ1_0 ? 143u : 142u));
        gguf_context * restored = gguf_init_from_buffer(metadata.data(), metadata.size(), {true, nullptr});
        assert(restored && gguf_get_tensor_type(restored, 0) == type);
        gguf_free(restored);
        gguf_free(file);
        ggml_free(ctx);
    }
    assert(pq == ptq);
}

// Raw blocks cover every packed code byte and the entire signed activation
// range, including -128 (not normally emitted by the float quantizer).
static void test_q2_0_packed_dot(ggml_type type) {
    const auto * traits = ggml_get_type_traits_cpu(type);
    const int qk = ggml_blck_size(type);
    const size_t stride = ggml_type_size(type);
    assert(qk == (type == GGML_TYPE_Q2_0 ? 64 : 128));
    assert(stride == size_t(2 + qk / 4));
    assert(ggml_type_size(GGML_TYPE_Q8_0) == 34);
    for (const int n : {64, 128, 192, 256, 320, 640, 2560, 32768}) {
        if (n % qk) continue;
        std::vector<uint8_t> x(ggml_row_size(type, n));
        std::vector<uint8_t> y(ggml_row_size(GGML_TYPE_Q8_0, n));
        for (int pattern = 0; pattern < 256; ++pattern) {
            float expected = 0.0f;
            for (int block = 0; block < n / qk; ++block) {
                const float dx = (block % 5 == 4) ? 0.0f : 0.25f;
                const ggml_fp16_t dxh = ggml_fp32_to_fp16(dx);
                memcpy(x.data() + block * stride, &dxh, 2);
                float partial = 0.0f;
                for (int chunk = 0; chunk < qk / 32; ++chunk) {
                    const float dy = chunk == 0 ? 0.5f : -2.0f;
                    const ggml_fp16_t dyh = ggml_fp32_to_fp16(dy);
                    uint8_t * qy = y.data() + (block * (qk / 32) + chunk) * 34;
                    memcpy(qy, &dyh, 2);
                    int dot = 0;
                    for (int b = 0; b < 8; ++b) {
                        const uint8_t packed = uint8_t(pattern + block * 13 + b * 17);
                        x[block * stride + 2 + chunk * 8 + b] = packed;
                        for (int j = 0; j < 4; ++j) {
                            const int value = ((pattern + block * 31 + chunk * 97 + b * 4 + j) & 255) - 128;
                            qy[2 + b * 4 + j] = uint8_t(value);
                            dot += (((packed >> (2 * j)) & 3) - 1) * value;
                        }
                    }
                    partial += dy * dot;
                }
                expected += dx * partial;
            }
            float actual = NAN;
            traits->vec_dot(n, &actual, 0, x.data(), 0, y.data(), 0, 1);
            assert(actual == expected);
        }
    }
}

// Repeated expert routes exercise multi-row reuse, including incomplete groups.
// Compare with ordinary dots on the same packed bytes, not a second graph that
// could take the same optimized route and hide an indexing or rounding error.
static void test_q2_0_repeated_experts() {
    const auto * traits = ggml_get_type_traits_cpu(GGML_TYPE_Q2_0);
    for (int n : {64, 320, 640, 2560, 16384, 16448}) {
        for (int columns : {7, 33, 257}) for (int rows : {1, 2, 3, 4, 5, 8, 9}) for (int lanes : {1, 2}) for (bool from_f32 : {false, true}) {
            // Span several output-column work chunks and leave a partial tail.
            if (columns == 257 && n != 640 && n != 2560) continue;
            ggml_context * ctx = ggml_init({4*1024*1024, nullptr, false});
            assert(ctx);
            ggml_tensor * weights = ggml_new_tensor_3d(ctx, GGML_TYPE_Q2_0, n, columns, 2);
            ggml_tensor * quantized_acts = ggml_new_tensor_3d(ctx, GGML_TYPE_Q8_0, n, lanes, rows);
            ggml_tensor * acts = quantized_acts;
            ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, rows);
            auto * x = static_cast<uint8_t *>(weights->data);
            auto * y = static_cast<uint8_t *>(acts->data);
            for (size_t i = 0; i < ggml_nbytes(weights); i += 18) {
                const ggml_fp16_t scale = ggml_fp32_to_fp16(float(int(i % 97) - 48)*0.00317f);
                memcpy(x + i, &scale, 2);
                for (int j = 2; j < 18; ++j) x[i + j] = uint8_t(i*17 + j*29);
            }
            for (size_t i = 0; i < ggml_nbytes(acts); i += 34) {
                const ggml_fp16_t scale = ggml_fp32_to_fp16(float(int(i % 43) - 21)*0.00291f);
                memcpy(y + i, &scale, 2);
                for (int j = 2; j < 34; ++j) y[i + j] = uint8_t(i*13 + j*19);
            }
            if (from_f32) {
                acts = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n, lanes, rows);
                ggml_get_type_traits(GGML_TYPE_Q8_0)->to_float(y, static_cast<float *>(acts->data), ggml_nelements(acts));
                // The reference uses the same quantized bytes as the worker
                // conversion, independently of how its blocks are partitioned.
                ggml_get_type_traits_cpu(GGML_TYPE_Q8_0)->from_float(static_cast<float *>(acts->data), y, ggml_nelements(acts));
            }
            auto * routes = static_cast<int32_t *>(ids->data);
            for (int i = 0; i < 2*rows; ++i) routes[i] = (i + i/2) % 2;
            ggml_tensor * out = ggml_mul_mat_id(ctx, weights, acts, ids);
            ggml_cgraph * graph = ggml_new_graph(ctx);
            ggml_build_forward_expand(graph, out);
            for (int threads : {1, 3, 6}) {
                assert(ggml_graph_compute_with_ctx(ctx, graph, threads) == GGML_STATUS_SUCCESS);
                for (int token = 0; token < rows; ++token) {
                    for (int route = 0; route < 2; ++route) {
                        for (int row = 0; row < columns; ++row) {
                            float expected = NAN;
                            traits->vec_dot(n, &expected, 0,
                                x + routes[2*token + route]*weights->nb[2] + row*weights->nb[1], 0,
                                y + (route % lanes)*quantized_acts->nb[1] + token*quantized_acts->nb[2], 0, 1);
                            const float actual = static_cast<float *>(out->data)[(2*token + route)*columns + row];
                            assert(std::isfinite(actual) && actual == expected);
                        }
                    }
                }
            }
            ggml_free(ctx);
        }
    }
}

int main(int argc, char * argv[]) {
    bool verbose = false;

    std::string arg;
    for (int i = 1; i < argc; i++) {
        arg = argv[i];

        if (arg == "-v") {
            verbose = true;
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            return 1;
        }
    }

    ggml_cpu_init();
    test_bonsai_codecs();
    test_q2_0_packed_dot(GGML_TYPE_Q2_0);
    test_q2_0_packed_dot(GGML_TYPE_Q2_0_G128);
    test_q2_0_repeated_experts();

    int num_failed = 0;

    num_failed += test_vec_dot_f32(verbose);
    num_failed += test_f8_e4m3_known_codes(verbose);
    num_failed += test_vec_dot_q(verbose);

    if (num_failed || verbose) {
        printf("%d tests failed\n", num_failed);
    }

    return num_failed > 0;
}
