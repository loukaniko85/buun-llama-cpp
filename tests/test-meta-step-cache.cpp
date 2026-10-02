#include "../ggml/src/ggml-backend-meta-step-cache.h"

#include <algorithm>
#include <cstdio>
#include <vector>

static int failures = 0;

static void expect(bool value, const char * message) {
    if (!value) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        ++failures;
    }
}

// Exercise the same LRU admission predicate with layer fragments, without a
// GPU. Device graph capture/replay is separately tested with tensor sharding.
static size_t misses(size_t segments, size_t nodes, size_t override_limit) {
    std::vector<size_t> lru;
    size_t count = 0;
    for (int token = 0; token < 3; ++token) {
        for (size_t segment = 0; segment < segments; ++segment) {
            auto found = std::find(lru.begin(), lru.end(), segment);
            if (found != lru.end()) {
                lru.erase(found);
            } else {
                ++count;
                while (ggml_backend_meta_step_cache_over_budget(
                        lru.size() + 1, (lru.size() + 1) * nodes, override_limit)) {
                    lru.erase(lru.begin());
                }
            }
            lru.push_back(segment);
        }
    }
    return count;
}

int main() {
    expect(!ggml_backend_meta_step_cache_over_budget(16, 1000000), "preserve 16 large graph records");
    expect(ggml_backend_meta_step_cache_over_budget(17, 1000000), "do not expand large graph retention");
    expect(!ggml_backend_meta_step_cache_over_budget(256, 32768), "small graphs share the node budget");
    expect(ggml_backend_meta_step_cache_over_budget(256, 32769), "node budget boundary");
    expect(!ggml_backend_meta_step_cache_over_budget(512, 0), "failed captures may fill count budget");
    expect(ggml_backend_meta_step_cache_over_budget(513, 0), "failed captures cannot grow without bound");
    expect(!ggml_backend_meta_step_cache_over_budget(1, 1000000, 1), "one-record override");
    expect(ggml_backend_meta_step_cache_over_budget(2, 0, 1), "one-record override eviction");
    expect(!ggml_backend_meta_step_cache_over_budget(512, 1000000, 512), "explicit cap retains legacy semantics");
    expect(misses(192, 128, 0) == 192, "layer fragments capture only on first sweep");
    expect(misses(192, 128, 16) == 576, "old count limit reproduces thrashing");
    expect(misses(16, 4096, 0) == 16, "full graph working set unchanged");
    expect(misses(17, 4096, 0) == 51, "large graphs remain bounded to old capacity");
    if (failures == 0) {
        std::puts("meta step cache budget: PASS");
    }
    return failures != 0;
}
