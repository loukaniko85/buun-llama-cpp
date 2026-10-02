// Byte ranges of a split meta tensor that cut its rows, as a state writer streaming in fixed-size
// pieces asks for them: every get (sync or async) must return the bytes a whole-tensor copy holds,
// and every set or memset must change exactly its own bytes.

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <random>
#include <utility>
#include <vector>

static size_t n_devs = 0;

// "rows": split along axis 1; "fused": two segments along the row, each split across the devices
static bool is_rows(const ggml_tensor * tensor) {
    return strcmp(tensor->name, "rows") == 0;
}

static std::vector<int64_t> segments_of(const ggml_tensor * tensor) {
    return strcmp(tensor->name, "fused") == 0 ? std::vector<int64_t>{ 64, 32 } : std::vector<int64_t>{ tensor->ne[is_rows(tensor)] };
}

static ggml_backend_meta_split_state split_state(const ggml_tensor * tensor, void *) {
    ggml_backend_meta_split_state state = {};
    state.axis = is_rows(tensor) ? GGML_BACKEND_SPLIT_AXIS_1 : GGML_BACKEND_SPLIT_AXIS_0;
    const std::vector<int64_t> segments = segments_of(tensor);
    for (size_t s = 0; s < segments.size(); ++s) {
        for (size_t j = 0; j < n_devs; ++j) {
            state.ne[s*n_devs + j] = segments[s] / int64_t(n_devs);
        }
        state.nr[s] = 1;
    }
    state.n_segments = uint32_t(segments.size());
    return state;
}

// Device j's shard, written out independently of the backend: stripe by stripe, its share of each segment.
static std::vector<uint8_t> expected_shard(const ggml_tensor * tensor, size_t j, const std::vector<uint8_t> & full) {
    const size_t unit   = tensor->nb[is_rows(tensor)];
    const size_t stripe = tensor->nb[is_rows(tensor) + 1];
    std::vector<uint8_t> shard;
    for (size_t base = 0; base < full.size(); base += stripe) {
        int64_t seg_start = 0;
        for (const int64_t seg : segments_of(tensor)) {
            const int64_t share = seg / int64_t(n_devs);
            const size_t  begin = base + size_t(seg_start + int64_t(j)*share) * unit;
            shard.insert(shard.end(), full.begin() + begin, full.begin() + begin + size_t(share) * unit);
            seg_start += seg;
        }
    }
    return shard;
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
    if (devices.empty()) {
        fprintf(stderr, "SKIP: requires a GPU\n");
        return 77;
    }
    if (devices.size() == 1) {
        // one GPU stands in for two devices: each still gets its own shard buffer
        devices.push_back(devices[0]);
    }
    n_devs = devices.size();

    auto             dev = ggml_backend_meta_device(devices.data(), devices.size(), split_state, nullptr);
    auto             buft = ggml_backend_dev_buffer_type(dev);
    ggml_backend_ptr backend(ggml_backend_dev_init(dev, nullptr));
    GGML_ASSERT(backend);
    ggml_context_ptr ctx(ggml_init({ ggml_tensor_overhead() * 4, nullptr, true }));
    ggml_tensor *    tensors[3] = {
        ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 96, 24),
        ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 96, 24),
        ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 32, 24, 3),
    };
    ggml_set_name(tensors[0], "single");
    ggml_set_name(tensors[1], "fused");
    ggml_set_name(tensors[2], "rows");
    ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft));
    GGML_ASSERT(buf);

    std::mt19937 rng(42);
    const auto   random_bytes = [&](size_t n) {
        std::vector<uint8_t> bytes(n);
        for (auto & b : bytes) {
            b = uint8_t(rng());
        }
        return bytes;
    };

    bool passed = true;
    for (ggml_tensor * tensor : tensors) {
        const size_t         nbytes = ggml_nbytes(tensor);
        const size_t         row    = tensor->nb[1];
        std::vector<uint8_t> mirror = random_bytes(nbytes);
        ggml_backend_tensor_set(tensor, mirror.data(), 0, nbytes);

        // Masked prefill can leave an empty cross-backend input without storage
        // or a resolvable shard layout. Like synchronous I/O, async zero-length
        // transfers must not dispatch to the backend or require allocated data.
        ggml_tensor empty = *tensor;
        empty.ne[0] = 0;
        empty.data = nullptr;
        uint8_t untouched = 0x5a;
        ggml_backend_tensor_get_async(backend.get(), &empty, &untouched, 0, 0);
        ggml_backend_tensor_set_async(backend.get(), &empty, &untouched, 0, 0);
        ggml_backend_synchronize(backend.get());
        GGML_ASSERT(untouched == 0x5a);

        const std::pair<size_t, size_t> ranges[] = {
            { 0, 1 }, { 7, 1001 }, { row - 3, 6 }, { row, row }, { 5*row + 11, 3*row }, { nbytes - 13, 13 },
            { tensor->nb[2] < nbytes ? tensor->nb[2] - 5 : 3, 10 }, { 0, nbytes },
        };
        for (const auto & range : ranges) {
            // not a structured binding: C++17 lambdas cannot capture those
            const size_t offset = range.first, size = range.second;
            const auto expect = [&](const char * op, bool ok) {
                if (!ok) {
                    fprintf(stderr, "FAIL: %s %s [%zu, +%zu)\n", tensor->name, op, offset, size);
                    passed = false;
                }
            };
            const auto matches_mirror = [&]() {
                std::vector<uint8_t> whole(nbytes);
                ggml_backend_tensor_get(tensor, whole.data(), 0, nbytes);
                return whole == mirror;
            };

            std::vector<uint8_t> got(size);
            ggml_backend_tensor_get(tensor, got.data(), offset, size);
            expect("get", memcmp(got.data(), mirror.data() + offset, size) == 0);

            std::fill(got.begin(), got.end(), 0);
            ggml_backend_tensor_get_async(backend.get(), tensor, got.data(), offset, size);
            ggml_backend_synchronize(backend.get());
            expect("get_async", memcmp(got.data(), mirror.data() + offset, size) == 0);

            std::vector<uint8_t> put = random_bytes(size);
            ggml_backend_tensor_set(tensor, put.data(), offset, size);
            memcpy(mirror.data() + offset, put.data(), size);
            expect("set", matches_mirror());

            put = random_bytes(size);
            ggml_backend_tensor_set_async(backend.get(), tensor, put.data(), offset, size);
            ggml_backend_synchronize(backend.get());
            memcpy(mirror.data() + offset, put.data(), size);
            expect("set_async", matches_mirror());

            const uint8_t value = uint8_t(rng());
            ggml_backend_tensor_memset(tensor, value, offset, size);
            memset(mirror.data() + offset, value, size);
            expect("memset", matches_mirror());
        }

        // the last range memsets the whole tensor, which would hide any permutation of the pieces
        mirror = random_bytes(nbytes);
        ggml_backend_tensor_set(tensor, mirror.data(), 0, nbytes);
        for (size_t j = 0; j < n_devs; j++) {
            const ggml_tensor *  shard = ggml_backend_meta_buffer_simple_tensor(tensor, j);
            std::vector<uint8_t> got(ggml_nbytes(shard));
            ggml_backend_tensor_get(shard, got.data(), 0, got.size());
            if (got != expected_shard(tensor, j, mirror)) {
                fprintf(stderr, "FAIL: %s shard %zu layout\n", tensor->name, j);
                passed = false;
            }
        }
    }

    printf("%s: partial-row meta get/set on %zu device(s)\n", passed ? "OK" : "FAIL", n_devs);
    return passed ? 0 : 1;
}
