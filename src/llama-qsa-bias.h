#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>

// Direct QSA layouts use natural position-block IDs. A prefix of complete
// blocks, optionally one incomplete tail, can fill query bias by intervals.
struct llama_qsa_bias_prefix {
    int64_t full;
    int64_t active;
};

inline llama_qsa_bias_prefix llama_qsa_classify_bias_prefix(
        const int32_t * representatives, const int32_t * complete, int64_t blocks) {
    int64_t full = 0;
    while (full < blocks && representatives[full] >= 0 && complete[full] != 0) {
        ++full;
    }
    int64_t active = full;
    if (active < blocks && representatives[active] >= 0) {
        ++active;
    }
    for (int64_t b = active; b < blocks; ++b) {
        if (representatives[b] >= 0) {
            return { -1, 0 };
        }
    }
    return { full, active };
}

// Caller established direct, same-sequence queries and nonnegative positions.
inline void llama_qsa_fill_bias_prefix(float * bias, int64_t blocks,
        llama_qsa_bias_prefix prefix, int64_t query, int64_t ratio, bool causal) {
    const int64_t tail = (query + 1)/ratio;
    const int64_t zero_end = causal ? std::min(tail, prefix.full) : prefix.full;
    const int64_t force_begin = causal ? std::min(tail, prefix.active) : prefix.full;
    const float hidden = -std::numeric_limits<float>::infinity();
    std::fill(bias, bias + zero_end, 0.0f);
    std::fill(bias + zero_end, bias + force_begin, hidden);
    std::fill(bias + force_begin, bias + prefix.active, 1e9f);
    std::fill(bias + prefix.active, bias + blocks, hidden);
}
