#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <initializer_list>

// Wide top-k views otherwise download the full expert-row pitch. Pack only
// cache-prefill routes with host weights and an admitted cache provider.
// The loader gives no_alloc weights typed dummy buffers, so fit uses the same
// placement predicate without requiring any weight data to be allocated.
inline bool llama_moe_ids_need_compaction(
        const ggml_tensor * ids, int64_t n_tokens, bool cache_enabled,
        std::initializer_list<const ggml_tensor *> experts) {
    if (!cache_enabled || n_tokens < 256 ||
            ids->type != GGML_TYPE_I32 || ids->ne[0] < 1 || ids->ne[0] > 16 ||
            ids->ne[2] != 1 || ids->ne[3] != 1 || ids->nb[0] != sizeof(int32_t) ||
            ids->nb[1] / (ids->ne[0] * sizeof(int32_t)) < 4) {
        return false;
    }
    for (const auto * weight : experts) {
        if (weight) {
            const auto * base = weight->view_src ? weight->view_src : weight;
            if (base->buffer && ggml_backend_buffer_is_host(base->buffer)) {
                return true;
            }
        }
    }
    return false;
}
