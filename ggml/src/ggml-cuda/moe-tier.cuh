#pragma once

// RAM tier for device-routed experts whose host bytes are a file mapping: a pinned
// arena holds the recently routed experts, demand-read from the file with direct I/O,
// so a model larger than RAM streams from disk at drive speed instead of faulting.

#include "common.cuh"
#include "moe-cache.cuh"

struct ggml_moe_tier;

// Before a launch that reads the experts ids selects through the routes' tables:
// points every entry outside VRAM at a resident RAM copy, waiting for its read.
// gate may be null; untiered routes are skipped. input is the launch's activation: at
// decode it also hints the next layer's experts by that layer's router.
void ggml_moe_tier_resolve(const ggml_moe_cache_route_table & x, const ggml_moe_cache_route_table * gate,
        const ggml_tensor * ids, const ggml_tensor * input, cudaStream_t stream);

// Finds the device's MoE routers (blk.N.ffn_gate_inp) in a graph about to run.
void ggml_moe_tier_scan_graph(const ggml_cgraph * cgraph, int device);
// Drops the routers in a device allocation about to be freed.
void ggml_moe_tier_forget(const void * base, size_t size);

// GGML_CUDA_MOE_TIER (off | auto | MiB) enables the tier.
bool ggml_moe_tier_enabled();
// Null when disabled. Routed kernels must run on stream. The arena is sized at the
// first resolve for the largest expert added by then.
ggml_moe_tier * ggml_moe_tier_create(int device, cudaStream_t stream);
// The caller has synchronized every stream that read the arena.
void ggml_moe_tier_free(ggml_moe_tier * tier);
void ggml_moe_tier_log_stats(const ggml_moe_tier * tier, int device);

// table is the route's device table (entries outside VRAM null); null when RAM cannot
// hold the minimum arena of these experts, when they outgrow the slots of an arena
// already sized, or when they lie outside a registered mapping.
ggml_moe_tier_route * ggml_moe_tier_route_add(ggml_moe_tier * tier, const void * host_base,
        int64_t n_expert, size_t expert_size, const void ** table);
// Once the route's table is live: routes of one group (a layer) share ids, so a
// resolve of one also prefetches the others.
void ggml_moe_tier_route_link(ggml_moe_tier_route * route, int group);

// The expert's RAM copy, held until unpin, or null when it is not resident.
const void * ggml_moe_tier_pin(ggml_moe_tier_route * route, int32_t expert);
void ggml_moe_tier_unpin(ggml_moe_tier_route * route, int32_t expert);
// VRAM-resident experts are the tier's first eviction choice.
void ggml_moe_tier_set_vram(ggml_moe_tier_route * route, int32_t expert, bool vram);

// Copies experts want marks into dst, the route's tensor, on stream after its earlier
// work, and returns once they landed: RAM copies directly, the rest read from the store
// through a pinned ring. Clears the bits of the experts it copied; the rest (all of
// them without a ring, or after a failed read) stay the caller's. ahead also reads the
// next routes' experts.
void ggml_moe_tier_prefill(ggml_moe_tier_route * route, uint32_t * want, bool ahead, char * dst, cudaStream_t stream);
