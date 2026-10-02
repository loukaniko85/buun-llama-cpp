#pragma once

#include <cstddef>

// CPU-offloaded experts split one model step into many small device graphs.
// Count-only eviction at 16 records can evict an entire layer sweep before its
// next token. Allow more small graphs, without retaining 512 full-model graphs.
// GGML nodes are a storage-cost proxy, not a byte-accurate CUDA memory limit.
inline bool ggml_backend_meta_step_cache_over_budget(
        size_t records, size_t nodes, size_t records_override = 0) {
    if (records_override != 0) {
        return records > records_override;
    }
    return records > 512 || (records > 16 && nodes > 32768);
}
