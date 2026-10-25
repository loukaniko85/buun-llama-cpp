#pragma once

// Host share of routed MoE misses (GGML_CUDA_MOE_CPU_SHARE=<GPU fraction of misses> pins it; by default the
// host takes the share that finishes with the bus, from measured host and bus speed per quant type).
// A doorbell kernel publishes the layer's routed experts and activations to mapped
// host memory; host workers compute a share of the missing experts end to end
// (up/gate, GLU, down) from the host weights while the GPU streams the rest over
// the bus. The routed matvecs skip the host entries; the weighted reduction that
// follows the down waits for the host and reads its rows in their place.
// The doorbell / host-computed-miss design follows Strata (github.com/Niko1221/Strata, MIT).

#include "common.cuh"

// Device view of the share handed to a routed matvec or to the weighted reduction.
struct ggml_moe_cpu_share_args {
    const uint8_t * skip = nullptr;     // per route (indexed like ids): 0 = GPU, else 1 + host entry
    // per route: the staged VRAM copy of a GPU-streamed miss's weights, or null to read the table
    const void * const * stage_x = nullptr;
    const void * const * stage_gate = nullptr;
    int ids_stride = 0;                 // reduction only: ids row stride, in elements
    const float * y = nullptr;          // reduction only: host output rows, one per entry
    const uint32_t * done = nullptr;    // mapped: last ticket the host finished
    const uint32_t * ticket = nullptr;  // current ticket
};

// Records a device-routed expert tensor of an owner (moe-cache session) when its route is registered.
void ggml_moe_cpu_share_note(const ggml_tensor * weights, const void * device_base, const void * owner);

// Fused up * GLU(gate) launch: rings the doorbell; skip is null when this layer is not shared.
ggml_moe_cpu_share_args ggml_moe_cpu_share_begin(
        const ggml_tensor * up, const ggml_tensor * gate, const ggml_tensor * ids,
        const ggml_tensor * x, const ggml_cuda_mm_fusion_args_host * fusion, cudaStream_t stream);

// Down launch of the layer opened by begin; skip is null otherwise.
ggml_moe_cpu_share_args ggml_moe_cpu_share_down(const ggml_tensor * down, const ggml_tensor * ids, const ggml_tensor * dst);

// Whether graph optimization should run the nodes after a weighted reduction of n_tok tokens before it.
bool ggml_moe_cpu_share_hoist(int64_t n_tok);

// Weighted reduction over experts: the merge view when experts is a shared down's output.
ggml_moe_cpu_share_args ggml_moe_cpu_share_merge(const ggml_tensor * experts);
