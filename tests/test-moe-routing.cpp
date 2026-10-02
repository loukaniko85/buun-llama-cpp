#include "llama-moe-routing.h"
#include "ggml-backend-impl.h"
#include "ggml-backend-moe-cache.h"

#include <cstdio>
#include <cstdlib>

static void check(bool value) {
    if (!value) {
        std::fputs("MoE routing admission check failed\n", stderr);
        std::abort();
    }
}

static void test_resolved_session() {
    ggml_backend_load_all();
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    check(cpu != nullptr);
    const auto saved_api = ggml_moe_cache;
    ggml_moe_cache = {};
    ggml_backend_t backends[] = { cpu, cpu };
    for (int count : { 1, 2 }) {
        auto * sched = ggml_backend_sched_new(backends, nullptr, count, 32, false, true);
        // Backend count is deliberately not eligibility: no provider means no
        // packing even for a multi-backend CPU/accelerator configuration.
        check(!ggml_backend_sched_has_moe_cache(sched));
        ggml_backend_sched_set_moe_cache(sched, GGML_MOE_CACHE_MODE_AUTO, 0, 0, -2, nullptr);
        check(!ggml_backend_sched_has_moe_cache(sched));
        ggml_moe_cache.query_config = [](int, size_t, ggml_moe_cache_config *) { return 1; };
        ggml_moe_cache.session_create = [](void * const *, int, const ggml_moe_cache_config *) -> void * {
            static int session;
            return &session;
        };
        ggml_moe_cache.session_destroy = [](void *) {};
        ggml_backend_sched_set_moe_cache(sched, GGML_MOE_CACHE_MODE_AUTO, 0, 0, -2, nullptr);
        check(ggml_backend_sched_has_moe_cache(sched));
        ggml_backend_sched_set_moe_cache(sched, GGML_MOE_CACHE_MODE_OFF, 0, 0, -2, nullptr);
        check(!ggml_backend_sched_has_moe_cache(sched));
        ggml_moe_cache.query_config = [](int, size_t, ggml_moe_cache_config *) { return 0; };
        ggml_backend_sched_set_moe_cache(sched, GGML_MOE_CACHE_MODE_AUTO, 0, 0, -2, nullptr);
        check(!ggml_backend_sched_has_moe_cache(sched));
        ggml_moe_cache.query_config = [](int, size_t, ggml_moe_cache_config *) { return 1; };
        ggml_moe_cache.session_create = [](void * const *, int, const ggml_moe_cache_config *) -> void * {
            return nullptr;
        };
        ggml_backend_sched_set_moe_cache(sched, GGML_MOE_CACHE_MODE_AUTO, 0, 0, -2, nullptr);
        check(!ggml_backend_sched_has_moe_cache(sched));
        ggml_backend_sched_free(sched);
        ggml_moe_cache = {};
    }
    ggml_moe_cache = saved_api;
    ggml_backend_free(cpu);
}

int main() {
    test_resolved_session();
    // Metadata-only fake buffers model the loader's zero-byte fit buffers.
    // Nothing in this admission test reads or allocates weight data.
    ggml_backend_buffer_type host_type = {};
    host_type.iface.is_host = [](ggml_backend_buffer_type_t) { return true; };
    ggml_backend_buffer_type device_type = {};
    ggml_backend_buffer host_buffer = {};
    ggml_backend_buffer device_buffer = {};
    host_buffer.buft = &host_type;
    device_buffer.buft = &device_type;
    ggml_tensor host = {};
    ggml_tensor device = {};
    ggml_tensor view = {};
    host.buffer = &host_buffer;
    device.buffer = &device_buffer;
    view.view_src = &host;
    ggml_tensor ids = {};
    ids.type = GGML_TYPE_I32;
    ids.ne[0] = 10;
    ids.ne[1] = 256;
    ids.ne[2] = ids.ne[3] = 1;
    ids.nb[0] = sizeof(int32_t);
    ids.nb[1] = 512*sizeof(int32_t);
    const auto eligible = [&] {
        return llama_moe_ids_need_compaction(&ids, ids.ne[1], true, { &host });
    };
    check(eligible());
    check(!llama_moe_ids_need_compaction(&ids, 256, true, { &device, nullptr, &device }));
    check(!llama_moe_ids_need_compaction(&ids, 256, false, { &host }));
    check(!llama_moe_ids_need_compaction(&ids, 256, true, { nullptr }));
    check(llama_moe_ids_need_compaction(&ids, 256, true, { &device, &host }));
    check(llama_moe_ids_need_compaction(&ids, 256, true, { nullptr, nullptr, &host }));
    check(llama_moe_ids_need_compaction(&ids, 256, true, { nullptr, nullptr, nullptr, &host }));
    check(llama_moe_ids_need_compaction(&ids, 256, true, { &view }));
    host.buffer = nullptr;
    check(!eligible());
    check(!llama_moe_ids_need_compaction(&ids, 256, true, { &view }));
    host.buffer = &host_buffer;

    for (int64_t k : { 0, 1, 10, 16, 17 }) {
        ids.ne[0] = k;
        for (int64_t rows : { 1, 255, 256, 512 }) {
            ids.ne[1] = rows;
            check(eligible() == (k >= 1 && k <= 16 && rows >= 256));
        }
    }
    ids.ne[0] = 10;
    ids.ne[1] = 256;
    for (size_t pitch : { size_t(40), size_t(156), size_t(160), size_t(2048) }) {
        ids.nb[1] = pitch;
        check(eligible() == (pitch >= 160));
    }
    ids.type = GGML_TYPE_F32;
    check(!eligible());
    ids.type = GGML_TYPE_I32;
    ids.nb[0] = 8;
    check(!eligible());
    ids.nb[0] = sizeof(int32_t);
    ids.ne[2] = 2;
    check(!eligible());
    ids.ne[2] = 1;
    ids.ne[3] = 2;
    check(!eligible());
    ids.ne[3] = 1;
    check(eligible());
    std::puts("MoE routing admission: host/view/fit and exclusion checks pass");
}
