#pragma once

#include "ggml.h"
#include "../ggml-impl.h"

#include <cstdint>

// CUDA graph instances hard-code tensor shapes. This bounded O(1) key keeps
// alternating speculative verify widths in independent warmup/cache entries.
// Any residual collision remains safe because the existing graph update check
// compares every node property before capture or replay.
static inline uint64_t ggml_cuda_graph_shape_key(const ggml_cgraph * cgraph) {
    uint64_t key = (uint64_t) (uintptr_t) cgraph->nodes[0];
    const auto mix = [&key](uint64_t value) {
        key = (key ^ value) * 0x100000001b3ull;
    };

    mix(cgraph->n_nodes);
    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        mix(cgraph->nodes[0]->ne[d]);
        mix(cgraph->nodes[cgraph->n_nodes - 1]->ne[d]);
    }
    // pipeline parallelism rotates a reused graph through several input copies: keep one entry
    // per copy instead of recapturing on every rotation (a split usually opens on its input)
    for (int i = 0; i < cgraph->n_nodes && i < 8; ++i) {
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            const ggml_tensor * src = cgraph->nodes[i]->src[j];
            if (src && (src->flags & GGML_TENSOR_FLAG_INPUT)) {
                mix((uint64_t) (uintptr_t) src->data);
                return key;
            }
        }
    }
    return key;
}
