#include "llama-memory-hybrid-idx.h"

#include <algorithm>
#include <cmath>
#include <type_traits>

#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-io.h"
#include "llama-model.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iterator>
#include <stdexcept>

//
// llama_memory_hybrid_idx
//

namespace {

// One grouping rule for the live incremental layout and pre-edit validation.
// Return the first cell that could begin a pool after a later append.
template<typename F>
size_t kpool_visit(const std::vector<std::pair<llama_pos, uint32_t>> & cells,
        size_t j, uint32_t kpool, bool by_order, F visit) {
    if (cells.empty()) {
        return j;
    }
    const llama_pos first = cells.front().first;
    while (j + kpool <= cells.size()) {
        bool complete = by_order;
        if (!by_order && (cells[j].first - first) % (llama_pos) kpool == 0) {
            complete = true;
            for (uint32_t k = 1; k < kpool; ++k) {
                if (cells[j + k].first != cells[j].first + (llama_pos) k) {
                    complete = false;
                    break;
                }
            }
        }
        if (complete) {
            visit(j);
            j += kpool;
        } else {
            ++j;
        }
    }
    return j;
}

}

llama_memory_hybrid_idx::llama_memory_hybrid_idx(
        const llama_model & model,
                            /* attn */
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                 uint32_t   kv_size,
                 uint32_t   n_pad,
                 uint32_t   n_swa,
           llama_swa_type   swa_type,
                            /* recurrent */
                ggml_type   type_r,
                ggml_type   type_s,
                 uint32_t   rs_size,
                            /* common */
                 uint32_t   n_seq_max,
                 uint32_t   n_rs_seq,
                     bool   offload,
                     bool   unified,
                            /* layer filters */
    const layer_filter_cb & filter_attn,
    const layer_filter_cb & filter_recr,
    const layer_filter_cb & filter_idx,
    const llama_memory_vbr_params & vbr) :
    llama_memory_hybrid(
        model,
        type_k, type_v, v_trans, kv_size, n_pad, n_swa, swa_type,
        type_r, type_s, rs_size,
        n_seq_max, n_rs_seq, offload, unified,
        filter_attn, filter_recr, vbr),
    hparams_idx(model.hparams),
    mem_idx(filter_idx == nullptr ? nullptr : [&] {
        // MQA with a single key head of indexer_head_size, as llama_kv_cache_dsa shapes its own
        std::fill(hparams_idx.n_head_kv_arr.begin(), hparams_idx.n_head_kv_arr.end(), 1);
        // a k-pool indexer caches its per-token rows and the pooled key side by side
        // (glm5-next: key | gate | pooled, qwen4exp: key | pooled)
        hparams_idx.n_embd_head_k_full = model.hparams.indexer_head_size * (model.hparams.indexer_kpool > 0 ? model.hparams.indexer_kpool_row : 1);

        // the cached indexer keys are raw, rotation happens after pooling at read time, so a
        // K-shift must not rotate them while the stream copies in the same update still apply
        hparams_idx.rope_type = LLAMA_ROPE_TYPE_NONE;

        // fool llama_kv_cache into thinking this is a MLA cache, so it won't cache V tensors
        hparams_idx.n_embd_head_k_mla_impl = model.hparams.indexer_head_size;
        hparams_idx.n_embd_head_v_mla_impl = model.hparams.indexer_head_size;

        LLAMA_LOG_INFO("%s: creating indexer KV cache, size = %u cells\n", __func__, kv_size);

        // QSA index keys come from a separate learned projection and are consumed by ordinary
        // selection matmuls. They are fixed F16 state, never attention Turbo/VBR storage.
        return new llama_kv_cache(
            model, hparams_idx, GGML_TYPE_F16, GGML_TYPE_F16, v_trans, offload, unified,
            kv_size, n_seq_max, n_pad, n_swa, swa_type,
            nullptr, filter_idx, nullptr, nullptr, "idx_");
    }()) {
}

void llama_memory_hybrid_idx::index_state_restored() {
    // the restore rewrote the cells behind the pool layout's back
    mem_idx_stale_set(-1, 0);
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) {
    // note: repeats llama_memory_hybrid::init_batch, as the indexer needs the attention slot infos that the base context hides
    do {
        balloc.split_reset();

        // follow the recurrent pattern for creating the ubatch splits
        std::vector<llama_ubatch> ubatches;

        while (true) {
            llama_ubatch ubatch;

            if (embd_all) {
                // if all tokens are output, split by sequence
                ubatch = balloc.split_seq(n_ubatch);
            } else {
                // Use non-sequential split when KV cache is unified (needed for hellaswag/winogrande/multiple-choice)
                const bool unified = (get_mem_attn()->get_n_stream() == 1);

                // [TAG_RECURRENT_ROLLBACK_SPLITS]
                // the trailing (1 + n_rs_seq) tokens of each seq must stay in the same ubatch
                //   so that the rollback snapshots remain valid
                const uint32_t n_rs_seq = get_mem_recr()->n_rs_seq;

                ubatch = balloc.split_equal(n_ubatch, !unified, n_rs_seq > 0 ? n_rs_seq + 1 : 0);
            }

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        // prepare the recurrent batches first
        if (!get_mem_recr()->prepare(ubatches)) {
            // TODO: will the recurrent cache be in an undefined context at this point?
            LLAMA_LOG_ERROR("%s: failed to prepare recurrent ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        // prepare the attention cache
        auto heads_attn = get_mem_attn()->prepare(ubatches);
        if (heads_attn.empty()) {
            LLAMA_LOG_ERROR("%s: failed to prepare attention ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        // the indexer uses the attention cache's slot layout; a separate one can drift from it
        llama_kv_cache::slot_info_vec_t heads_idx;
        if (mem_idx) {
            heads_idx = heads_attn;
        }

        return std::make_unique<llama_memory_hybrid_idx_context>(
                this, std::move(heads_attn), std::move(heads_idx), std::move(ubatches));
    } while(false);

    return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_full() {
    return std::make_unique<llama_memory_hybrid_idx_context>(this);
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_update(llama_context * lctx, bool optimize) {
    return std::make_unique<llama_memory_hybrid_idx_context>(this, lctx, optimize);
}

void llama_memory_hybrid_idx::clear(bool data) {
    llama_memory_hybrid::clear(data);

    if (mem_idx) {
        mem_idx->clear(data);
        mem_idx_stale_set(-1, 0);
    }
}

// A pooled key is only valid while the grouping that produced it holds. Grouping is sequence relative,
// so an edit at p0 leaves every pool that ends before p0 alone.
void llama_memory_hybrid_idx::mem_idx_stale_set(llama_seq_id seq_id, llama_pos p0) {
    p0 = std::max<llama_pos>(p0, 0);

    if (seq_id < 0) {
        for (auto & p : mem_idx_stale) {
            p = std::min(p, p0);
        }

        return;
    }

    GGML_ASSERT(seq_id < (llama_seq_id) LLAMA_MAX_SEQ);

    mem_idx_stale[seq_id] = std::min(mem_idx_stale[seq_id], p0);
}

// An edit at or below the first position moves pos_min, which regroups the whole sequence.
llama_pos llama_memory_hybrid_idx::mem_idx_stale_pos(llama_seq_id seq_id, llama_pos p0) const {
    if (seq_id < 0 || p0 <= mem_idx->seq_pos_min(seq_id)) {
        return 0;
    }

    return p0;
}

bool llama_memory_hybrid_idx::kpool_can_remove(llama_seq_id seq_id, llama_pos p0, llama_pos p1) const {
    // Separate streams never share a representative. Suffix/full removal
    // cannot regroup surviving pools; keep speculative rollback constant-time.
    if (!mem_idx || get_kpool() <= 1 || mem_idx->get_n_stream() != 1 ||
            seq_id >= (llama_seq_id) LLAMA_MAX_SEQ ||
            p1 < 0 || p0 >= p1 ||
            (seq_id >= 0 && p1 > mem_idx->seq_pos_max(seq_id))) {
        return true;
    }
    p0 = std::max<llama_pos>(p0, 0);
    const auto & cells = mem_idx->get_cells(0);
    const uint32_t kpool = get_kpool();
    std::map<uint32_t, std::vector<uint32_t>> representatives;
    bool compatible = true;
    for (llama_seq_id s = 0; s < LLAMA_MAX_SEQ; ++s) {
        const auto & positions = cells.seq_pos_get(s);
        std::vector<std::pair<llama_pos, uint32_t>> remaining;
        remaining.reserve(positions.size());
        for (const auto & pc : positions) {
            if ((seq_id < 0 || seq_id == s) && p0 <= pc.first && pc.first < p1) {
                continue;
            }
            remaining.push_back(pc);
        }
        kpool_visit(remaining, 0, kpool, get_kpool_by_order(), [&](size_t j) {
            const uint32_t rep = remaining[j + kpool - 1].second;
            if (cells.seq_count(rep) <= 1) {
                return;
            }
            std::vector<uint32_t> members;
            members.reserve(kpool);
            for (uint32_t k = 0; k < kpool; ++k) {
                members.push_back(remaining[j + k].second);
            }
            const auto inserted = representatives.emplace(rep, members);
            compatible &= inserted.second || inserted.first->second == members;
        });
        if (!compatible) {
            LLAMA_LOG_WARN("%s: refusing edit that aliases different pools at a shared representative\n", __func__);
            return false;
        }
    }
    return true;
}

bool llama_memory_hybrid_idx::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    if (!kpool_can_remove(seq_id, p0, p1)) {
        return false;
    }
    // same order as llama_memory_hybrid::seq_rm: the recurrent cache can refuse, so try it first
    if (!get_mem_recr()->seq_rm(seq_id, p0, p1)) {
        return false;
    }

    if (mem_idx) {
        const llama_pos stale = mem_idx_stale_pos(seq_id, p0);
        const bool removed_idx = mem_idx->seq_rm(seq_id, p0, p1);
        GGML_ASSERT(removed_idx);
        GGML_UNUSED(removed_idx);
        mem_idx_stale_set(seq_id, stale);
    }

    return get_mem_attn()->seq_rm(seq_id, p0, p1);
}

bool llama_memory_hybrid_idx::seq_rm_attn(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    if (!kpool_can_remove(seq_id, p0, p1)) {
        return false;
    }
    // The indexer is auxiliary attention state. Every attention-only edit must retain the same
    // cell membership in both children, while deliberately leaving recurrent state untouched.
    if (mem_idx) {
        const llama_pos stale = mem_idx_stale_pos(seq_id, p0);
        const bool removed_idx = mem_idx->seq_rm(seq_id, p0, p1);
        GGML_ASSERT(removed_idx);
        GGML_UNUSED(removed_idx);
        mem_idx_stale_set(seq_id, stale);
    }
    return get_mem_attn()->seq_rm(seq_id, p0, p1);
}

bool llama_memory_hybrid_idx::seq_rm_transient(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    if (!kpool_can_remove(seq_id, p0, p1)) {
        return false;
    }
    if (!get_mem_recr()->seq_rm(seq_id, p0, p1)) {
        return false;
    }
    if (mem_idx) {
        const llama_pos stale = mem_idx_stale_pos(seq_id, p0);
        const bool removed_idx = mem_idx->seq_rm_transient(seq_id, p0, p1);
        GGML_ASSERT(removed_idx);
        GGML_UNUSED(removed_idx);
        mem_idx_stale_set(seq_id, stale);
    }
    return get_mem_attn()->seq_rm_transient(seq_id, p0, p1);
}

bool llama_memory_hybrid_idx::seq_rm_attn_transient(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    if (!kpool_can_remove(seq_id, p0, p1)) {
        return false;
    }
    if (mem_idx) {
        const llama_pos stale = mem_idx_stale_pos(seq_id, p0);
        const bool removed_idx = mem_idx->seq_rm_attn_transient(seq_id, p0, p1);
        GGML_ASSERT(removed_idx);
        GGML_UNUSED(removed_idx);
        mem_idx_stale_set(seq_id, stale);
    }
    return get_mem_attn()->seq_rm_attn_transient(seq_id, p0, p1);
}

void llama_memory_hybrid_idx::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    (void) try_seq_cp(seq_id_src, seq_id_dst, p0, p1);
}

bool llama_memory_hybrid_idx::try_seq_cp(
        llama_seq_id seq_id_src,
        llama_seq_id seq_id_dst,
        llama_pos p0,
        llama_pos p1) {
    // A shared representative row cannot hold two different pool groupings.
    // Refuse before any child mutates; the transactional caller may restore
    // a physical snapshot or replay instead of aliasing a partial sequence.
    if (get_kpool() > 0 && (p0 > 0 || p1 >= 0)) {
        return false;
    }
    const bool dst_empty = mem_idx && mem_idx->seq_pos_max(seq_id_dst) < 0;
    if (!llama_memory_hybrid::try_seq_cp(seq_id_src, seq_id_dst, p0, p1)) {
        // The base has already invalidated attention and recurrent state.
        const bool removed_idx = !mem_idx || mem_idx->seq_rm(seq_id_dst, -1, -1);
        GGML_ASSERT(removed_idx);
        GGML_UNUSED(removed_idx);
        mem_idx_stale_set(seq_id_dst, 0);
        return false;
    }

    if (!mem_idx || mem_idx->try_seq_cp(seq_id_src, seq_id_dst, p0, p1)) {
        mem_idx_stale_copy(seq_id_src, seq_id_dst, dst_empty);
        return true;
    }

    // A failed index copy after the base succeeded invalidates the destination rather than
    // exposing children from different timelines.
    state_drop(seq_id_dst);
    return false;
}

bool llama_memory_hybrid_idx::try_seq_cp_transient(
        llama_seq_id seq_id_src,
        llama_seq_id seq_id_dst,
        llama_pos p0,
        llama_pos p1) {
    if (get_kpool() > 0 && (p0 > 0 || p1 >= 0)) {
        return false;
    }
    const bool dst_empty = mem_idx && mem_idx->seq_pos_max(seq_id_dst) < 0;
    if (!llama_memory_hybrid::try_seq_cp_transient(seq_id_src, seq_id_dst, p0, p1)) {
        const bool removed_idx = !mem_idx || mem_idx->seq_rm_transient(seq_id_dst, -1, -1);
        GGML_ASSERT(removed_idx);
        GGML_UNUSED(removed_idx);
        mem_idx_stale_set(seq_id_dst, 0);
        return false;
    }

    if (!mem_idx || mem_idx->try_seq_cp_transient(seq_id_src, seq_id_dst, p0, p1)) {
        mem_idx_stale_copy(seq_id_src, seq_id_dst, dst_empty);
        return true;
    }

    const bool removed_recr = get_mem_recr()->seq_rm(seq_id_dst, -1, -1);
    const bool removed_attn = get_mem_attn()->seq_rm_transient(seq_id_dst, -1, -1);
    const bool removed_idx = !mem_idx || mem_idx->seq_rm_transient(seq_id_dst, -1, -1);
    GGML_ASSERT(removed_recr && removed_attn && removed_idx);
    GGML_UNUSED(removed_recr);
    GGML_UNUSED(removed_attn);
    GGML_UNUSED(removed_idx);
    mem_idx_stale_set(seq_id_dst, 0);
    return false;
}

void llama_memory_hybrid_idx::seq_keep(llama_seq_id seq_id) {
    llama_memory_hybrid::seq_keep(seq_id);

    if (mem_idx) {
        mem_idx->seq_keep(seq_id);
        // every other sequence loses its cells, so their layouts must rebuild
        mem_idx_stale_set(-1, 0);
    }
}

void llama_memory_hybrid_idx::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    if (shift == 0) {
        return;
    }

    // Prove every affected active position is [t,t,t,0] before any child
    // mutates.  The indexer mirrors the attention cells, but checking both
    // also turns an already-drifted cache into a fail-closed error.
    if (get_kpool_by_order()) {
        GGML_ASSERT(get_mem_attn()->can_shift_qwen4_text_range(seq_id, p0, p1));
        GGML_ASSERT(!mem_idx || mem_idx->can_shift_qwen4_text_range(seq_id, p0, p1));
    }

    llama_memory_hybrid::seq_add(seq_id, p0, p1, shift);

    if (mem_idx) {
        // QSA caches the unnormalised, pre-RoPE index projection.  Only its
        // block-position metadata follows the edit; rotating these raw keys
        // would make the next QSA selection disagree with an unshifted decode.
        const llama_pos stale = mem_idx_stale_pos(seq_id, shift < 0 ? p0 + shift : p0);
        if (get_kpool_by_order()) {
            mem_idx->seq_add_raw_mrope(seq_id, p0, p1, shift);
        } else {
            mem_idx->seq_add(seq_id, p0, p1, shift);
        }
        mem_idx_stale_set(seq_id, stale);
    }
    // the shift moved keys in bulk: every sequence re-pools
    mem_idx_stale_set(-1, 0);
}

void llama_memory_hybrid_idx::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    if (d == 1) {
        return;
    }
    if (!get_kpool_by_order()) {
        llama_memory_hybrid::seq_div(seq_id, p0, p1, d);
        if (mem_idx) {
            mem_idx->seq_div(seq_id, p0, p1, d);
            mem_idx_stale_set(seq_id, 0);
        }
        return;
    }

    GGML_UNUSED(seq_id);
    GGML_UNUSED(p0);
    GGML_UNUSED(p1);
    GGML_UNUSED(d);

    // A division gives the active M-RoPE sections different, position-dependent
    // deltas and cannot be represented by the single K-shift input.  Refuse
    // before mutating any of the three children.
    GGML_ABORT("Qwen4 indexed memory does not support seq_div / Self-Extend");
}

std::map<ggml_backend_buffer_type_t, size_t> llama_memory_hybrid_idx::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> mb = llama_memory_hybrid::memory_breakdown();

    if (mem_idx) {
        for (const auto & buft_size : mem_idx->memory_breakdown()) {
            mb[buft_size.first] += buft_size.second;
        }
    }

    return mb;
}

void llama_memory_hybrid_idx::state_write_range(llama_io_write_i & io, llama_seq_id seq_id, llama_pos p0, llama_pos p1) const {
    // The inherited attention-only range omits index keys, while PARTIAL_ONLY
    // carries only recurrent state. Do not publish an incomplete base snapshot.
    if (mem_idx) {
        throw std::runtime_error("indexed memory does not support position-range state");
    }
    llama_memory_hybrid::state_write_range(io, seq_id, p0, p1);
}

void llama_memory_hybrid_idx::state_append_range(llama_io_read_i & io, llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos p_limit) {
    // Refuse before any child mutates: an attention-only append would leave
    // index membership and pooled-key history inconsistent with attention.
    if (mem_idx) {
        throw std::runtime_error("indexed memory does not support position-range state");
    }
    llama_memory_hybrid::state_append_range(io, seq_id, p0, p1, p_limit);
}

void llama_memory_hybrid_idx::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    llama_memory_hybrid::state_write(io, seq_id, flags);

    // [TAG_HYBRID_IDX_STATE] the indexer section goes last, so it is a pure suffix: an old reader stops early instead of misparsing it
    // The indexer mirrors the attention cache, so it uses the same PARTIAL_ONLY gate.
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        if (mem_idx) {
            mem_idx->state_write(io, seq_id, flags);
        }
    }

}

void llama_memory_hybrid_idx::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    // note: repeats llama_memory_hybrid::state_read
    // the indexer needs the attention cache's cells, and a half-failed restore must leave all three caches alike

    // [TAG_HYBRID_IDX_SINFO]
    // the indexer restore adopts the attention cache's layout instead of searching for cells of its own
    // two find_slot calls agree only while both caches see the same occupancy, which a restore cannot promise
    llama_kv_cache::slot_info_vec_t sinfos_attn;

    // restored keys bypass the ubatch path that tracks the pool layout
    mem_idx_stale_set(-1, 0);

    try {
        if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
            get_mem_attn()->state_read_sinfo(io, seq_id, flags, mem_idx ? &sinfos_attn : nullptr, nullptr);
        }

        get_mem_recr()->state_read(io, seq_id, flags);

        // [TAG_HYBRID_IDX_STATE] must mirror the write order in state_write
        if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
            if (mem_idx) {
                mem_idx->state_read_sinfo(io, seq_id, flags, nullptr, &sinfos_attn);
                // the restore rewrites the cells behind the pool layout's back
                mem_idx_stale_set(seq_id, 0);
            }
        }

    } catch (...) {
        // a half-restored context is the one state the indexer cannot fix by itself: attention holds new cells, the indexer old ones
        // drop what was being restored from all of them, which is a state they do agree on.
        io.discard();
        if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
            get_mem_attn()->state_clear(seq_id);
            if (mem_idx) {
                mem_idx->state_clear(seq_id);
            }
        }
        state_drop(seq_id);

        throw;
    }
}

void llama_memory_hybrid_idx::state_drop(llama_seq_id seq_id) {
    // dropped directly, not via seq_rm: the recurrent cache may refuse it and then only the other two get cleared
    if (seq_id < 0) {
        clear(true);

        return;
    }

    get_mem_attn()->state_clear(seq_id);
    get_mem_recr()->seq_rm(seq_id, -1, -1);

    if (mem_idx) {
        mem_idx->state_clear(seq_id);
        mem_idx_stale_set(seq_id, 0);
    }
}

llama_kv_cache * llama_memory_hybrid_idx::get_mem_idx() const {
    return mem_idx.get();
}


//
// llama_memory_hybrid_idx_context
//

// streams in each ubatch's slot info, matching get_k/get_v's `ns`
static std::vector<uint32_t> llama_memory_hybrid_idx_ns(const llama_kv_cache::slot_info_vec_t & sinfos) {
    std::vector<uint32_t> res;
    res.reserve(sinfos.size());

    for (const auto & sinfo : sinfos) {
        res.push_back(sinfo.s1 - sinfo.s0 + 1);
    }

    return res;
}

// Which cells of a sequence make up which pool, for the whole cache.
struct llama_memory_hybrid_idx::kpool_layout {
    struct seq {
        llama_pos pos_min = 0;
        uint32_t  strm    = 0; // Stream holding this sequence's cells
        std::vector<std::pair<llama_pos, uint32_t>> cells; // Position and stream local cell pairs, sorted by position.
        std::vector<uint32_t> pools;

        // Where the pool scan stopped, so an append resumes instead of starting over.
        size_t j_next = 0;
    };

    std::array<seq, LLAMA_MAX_SEQ> seqs;

    uint32_t n_pool_real = 0;
};

// A whole sequence copied into an empty one shares the source's pools, members and pooled rep rows alike,
// so it inherits the source's staleness instead of re-pooling everything. Only its layout entry is rebuilt.
// A copy into a non-empty sequence regroups it, so it stales everything.
void llama_memory_hybrid_idx::mem_idx_stale_copy(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, bool dst_empty) {
    if (!dst_empty) {
        mem_idx_stale_set(seq_id_dst, 0);
        return;
    }
    GGML_ASSERT(seq_id_dst >= 0 && seq_id_dst < (llama_seq_id) LLAMA_MAX_SEQ);

    mem_idx_stale[seq_id_dst] = mem_idx_stale[seq_id_src];
    if (kpool_lay) {
        kpool_lay->seqs[seq_id_dst] = kpool_layout::seq();
    }
}

// Which pools of the layout the current ubatch must re-pool, in the layout's pool order.
struct llama_memory_hybrid_idx_context::kpool_state {
    std::vector<uint32_t> is_new;
    // Sequences sharing cells (a seq_cp, or tokens decoded for several sequences) list the same pool once each.
    // canon maps every pool to the first one with its rep, the only entry ever re-pooled or made visible.
    std::vector<uint32_t> canon;
    std::vector<uint32_t> rep_gen;   // per global cell, the generation that last saw a pool with that rep
    std::vector<uint32_t> rep_first; // per global cell, that pool, valid when rep_gen is current
    uint32_t generation = 0;

    // Sequences that branched mid-pool each list a different pool over their shared cells, so a pool invisible
    // to a token can overlap a visible one. The graph then routes dead selection slots away from their cells.
    bool partial_share = false;

    uint32_t n_pool_real = 0;
    uint32_t n_pool_g    = 0; // graph size of the pool list, kpool_pad of at least n_pool_real
    uint32_t n_new       = 0;
    uint32_t n_new_g     = 1; // graph size of the new pool list, stable across decode steps
};

namespace {

// The last padded pool is always unused.
uint32_t kpool_pad(uint32_t n_pool) {
    return std::max<uint32_t>(64u, GGML_PAD(n_pool + 1, 64u));
}

// Rank of (pos, cell) in a sequence's cells sorted by position then cell, or -1 when absent.
// In order mode the rank alone places a token: cells sharing a position (M-RoPE images) have distinct ranks.
int64_t kpool_rank(const std::vector<std::pair<llama_pos, uint32_t>> & cells, llama_pos pos, uint32_t cell) {
    auto it = std::lower_bound(cells.begin(), cells.end(), std::make_pair(pos, cell));
    return it != cells.end() && it->second == cell && it->first == pos ? it - cells.begin() : -1;
}

}

llama_memory_hybrid_idx::~llama_memory_hybrid_idx() = default;

const llama_memory_hybrid_idx::kpool_layout & llama_memory_hybrid_idx::kpool_layout_get() const {
    GGML_ASSERT(kpool_lay != nullptr);

    return *kpool_lay;
}

// Pools are fixed by the positions relative to the sequence's first one, so the layout survives a plain
// append. A sequence edit can regroup them, and mem_idx_stale tells us it happened.
const llama_memory_hybrid_idx::kpool_layout & llama_memory_hybrid_idx::kpool_layout_update() {
    GGML_ASSERT(mem_idx != nullptr);

    if (!kpool_lay) {
        kpool_lay = std::make_unique<kpool_layout>();
    }

    auto & lay = *kpool_lay;

    const uint32_t kpool       = get_kpool();
    const uint32_t n_stream_kv = mem_idx->get_n_stream();
    const bool     unified     = n_stream_kv == 1;

    lay.n_pool_real = 0;

    for (llama_seq_id s = 0; s < LLAMA_MAX_SEQ; ++s) {
        auto & sq = lay.seqs[s];

        // a non unified cache gives each sequence its own stream, with stream local cell indices
        if (!unified && s >= (llama_seq_id) n_stream_kv) {
            sq = kpool_layout::seq();
            continue;
        }

        const auto & cells = mem_idx->get_cells(unified ? 0 : s);
        const auto & sp    = cells.seq_pos_get(s);

        sq.strm = unified ? 0 : mem_idx->get_stream(s);

        if (mem_idx_stale[s] == POS_CLEAN && !sq.cells.empty() && !sp.empty() &&
                sq.pos_min == sp.begin()->first) {
            for (auto it = sp.upper_bound(sq.cells.back()); it != sp.end(); ++it) {
                sq.cells.push_back(*it);
            }
        }

        // the appended tail accounts for every cell only if nothing before it was dropped, but an edit can
        // regroup a sequence without changing its cell count, so a stale sequence must rebuild regardless
        if (sq.cells.size() != sp.size() || mem_idx_stale[s] != POS_CLEAN) {
            sq.cells.assign(sp.begin(), sp.end());
            sq.pools.clear();
            sq.j_next  = 0;
            sq.pos_min = sp.empty() ? 0 : sp.begin()->first;
        }

        // Pools start at the first valid token
        sq.j_next = kpool_visit(sq.cells, sq.j_next, kpool, hparams_idx.indexer_kpool_by_order,
                [&](size_t j) { sq.pools.push_back((uint32_t) j); });

        lay.n_pool_real += (uint32_t) sq.pools.size();
    }

    return lay;
}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(llama_memory_status status) :
    llama_memory_hybrid_context(status) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(llama_memory_hybrid_idx * mem) :
    llama_memory_hybrid_context(mem),
    mem(mem),
    // graph reservation walks a full context, and qwen4exp builds the sparse attention only when this is set
    // without it the reserved worst case is the dense graph, so ggml-alloc must grow the buffer on the first decode
    ns_ubatch(mem->get_mem_idx() == nullptr ?
        std::vector<uint32_t>() : std::vector<uint32_t>{ mem->get_mem_idx()->get_n_stream() }),
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        new llama_kv_cache_context(mem->get_mem_idx())) {
    if (kpool_track()) {
        mem->kpool_layout_update();
        auto st = kpool_build_sizes();
        const auto * idx = mem->get_mem_idx();
        const uint64_t n_pool_max = uint64_t(idx->get_size() / mem->get_kpool()) * idx->get_n_seq_max();
        GGML_ASSERT(n_pool_max <= UINT32_MAX - 64);
        st.n_pool_real = std::max(st.n_pool_real, uint32_t(n_pool_max));
        st.n_pool_g    = kpool_pad(st.n_pool_real);
        st.n_new   = st.n_pool_real;
        st.n_new_g = std::max(st.n_new, 1u);
        st.partial_share = true; // reserve the graph with dead-slot routing, the larger one
        kpool_st = std::make_unique<kpool_state>(std::move(st));
        i_kpool  = 0;
    }
}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(
        llama_memory_hybrid_idx * mem,
                  llama_context * lctx,
                           bool   optimize) :
    llama_memory_hybrid_context(mem, lctx, optimize),
    mem(mem),
    is_update(true),
    // update() applies a pending cross-stream seq_cp, else the copy keeps stale indexer keys
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        mem->get_mem_idx()->init_update(lctx, optimize)) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(
        llama_memory_hybrid_idx * mem,
                slot_info_vec_t   sinfos_attn,
                slot_info_vec_t   sinfos_idx,
      std::vector<llama_ubatch>   ubatches) :
    // note: the base copies the ubatches; ctx_idx gets a copy of its own
    llama_memory_hybrid_context(mem, std::move(sinfos_attn), ubatches),
    mem(mem),
    ns_ubatch(llama_memory_hybrid_idx_ns(sinfos_idx)),
    sinfos_kpool(mem->get_mem_idx() != nullptr && mem->get_kpool() > 0 && mem->get_kpool_by_order() ? sinfos_idx : slot_info_vec_t()),
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        // the indexer shares the attention KQ mask, so its n_kv follows the attention cache's
        new llama_kv_cache_context(mem->get_mem_idx(), std::move(sinfos_idx), ubatches, get_attn()->batch_stable())) {
    for (size_t i = 1; i < ubatches.size() && get_attn()->batch_stable(); ++i) {
        n_tokens_rest += ubatches[i].n_tokens;
    }
    // Sequence edits force the touched positions to re-pool.
    mem_idx_stale_batch = mem->mem_idx_stale_get();
}

llama_memory_hybrid_idx_context::~llama_memory_hybrid_idx_context() = default;

bool llama_memory_hybrid_idx_context::next() {
    // Clear only after a successful ubatch.
    if (i_cur == 0 && mem != nullptr) {
        mem->mem_idx_stale_clear();
    }

    if (ctx_idx) {
        ctx_idx->next();
    }

    ++i_cur;

    return llama_memory_hybrid_context::next();
}

bool llama_memory_hybrid_idx_context::apply() {
    bool res = llama_memory_hybrid_context::apply();

    if (ctx_idx) {
        res = res & ctx_idx->apply();
    }

    if (is_update) {
        // a K-shift, defrag or stream copy moves indexer keys between cells
        mem->mem_idx_stale_set(-1, 0);
    }

    if (res && kpool_track()) {
        mem->kpool_layout_update();
        if (!kpool_st) {
            kpool_st = std::make_unique<kpool_state>();
        }
        kpool_build_state(get_ubatch());
        i_kpool = i_cur;
    }

    return res;
}

bool llama_memory_hybrid_idx_context::kpool_track() const {
    // Derived from mem instead of being cached.
    return mem != nullptr && mem->get_mem_idx() != nullptr && mem->get_kpool() > 0 && !ns_ubatch.empty();
}

const llama_kv_cache_context * llama_memory_hybrid_idx_context::get_idx() const {
    return static_cast<const llama_kv_cache_context *>(ctx_idx.get());
}

llama_memory_hybrid_idx_context::kpool_access::kpool_access(ggml_context * ctx, ggml_tensor * k, int64_t n_embd) : ctx(ctx) {
    // rows are the per-token part (glm5-next: key | gate, qwen4exp: key), then the pooled key
    const int64_t n_tok = k->ne[0] - n_embd;
    GGML_ASSERT(n_tok > 0 && n_tok % n_embd == 0);

    const int64_t n_cells = k->ne[1]*k->ne[2];

    // Pool indices can refer to other streams. Revisit these full-storage views if that changes:
    // https://github.com/ggml-org/llama.cpp/pull/27773#discussion_r4130905603
    key_gate = ggml_view_2d(ctx, k, n_tok,  n_cells, k->nb[1], 0);
    pooled   = ggml_view_2d(ctx, k, n_embd, n_cells, k->nb[1], ggml_row_size(k->type, n_tok));
}

ggml_tensor * llama_memory_hybrid_idx_context::kpool_access::gather_key_gate(ggml_tensor * idxs) const {
    return ggml_get_rows(ctx, key_gate, idxs);
}

ggml_tensor * llama_memory_hybrid_idx_context::kpool_access::scatter_pooled(ggml_tensor * values, ggml_tensor * idxs) const {
    return ggml_set_rows(ctx, pooled, values, idxs);
}

ggml_tensor * llama_memory_hybrid_idx_context::kpool_access::gather_pooled(ggml_tensor * idxs) const {
    return ggml_get_rows(ctx, pooled, idxs);
}

llama_memory_hybrid_idx_context::kpool_access llama_memory_hybrid_idx_context::get_kpool_access(
        ggml_context * ctx, int32_t il, int64_t n_embd) const {
    GGML_ASSERT(mem != nullptr && mem->get_mem_idx() != nullptr);

    return kpool_access(ctx, mem->get_mem_idx()->get_k_storage(il), n_embd);
}

// k-pool DSA indexer (glm5-next, qwen4exp QSA)

// Sizes only, used by the full cache context so get_n_kpool() works during graph reserve.
llama_memory_hybrid_idx_context::kpool_state llama_memory_hybrid_idx_context::kpool_build_sizes() const {
    const auto & lay = mem->kpool_layout_get();

    kpool_state st;
    st.n_pool_real = lay.n_pool_real;

    return st;
}

// Which pools this ubatch must re-pool.
// Pool cache lifecycle:
// 1. cpy_k writes each token's key | gate into its idx cache row, pooled slot are zeroed.
// 2. This marks the pools the ubatch touches or completes as new, during decode that's one pool every kpool tokens, zero elsewise.
// 3. The graph pools only the new pools and set_rows each result into the pooled slot of the pool's last member row.
// 4. All pools are gathered in one get_rows via pool_cells, fresh ones just written, older ones from whatever batch last wrote them.
// A seq_* edit regroups the pools from the edited position on, so it stales them and the first ubatch of the next batch
// rebuilds them from the still-valid key | gate rows, rewriting the (possibly different) rep rows.
// Orphaned pooled slots are never cleared, a slot is only ever read through pool_cells, which follows the current grouping.
void llama_memory_hybrid_idx_context::kpool_build_state(const llama_ubatch & ubatch) {
    const auto & lay = mem->kpool_layout_get();
    auto & st = *kpool_st;

    const auto *   idx     = mem->get_mem_idx();
    const uint32_t kv_size = idx->get_size();
    const uint32_t kpool   = mem->get_kpool();

    st.n_pool_real = lay.n_pool_real;
    st.n_new       = 0;
    if (++st.generation == 0) {
        std::fill(st.is_new.begin(),  st.is_new.end(),  0);
        std::fill(st.rep_gen.begin(), st.rep_gen.end(), 0);
        st.generation = 1;
    }
    st.is_new.resize(lay.n_pool_real, 0);
    st.canon.resize(lay.n_pool_real);
    st.rep_gen.resize((size_t) kv_size*idx->get_n_stream(), 0);
    st.rep_first.resize(st.rep_gen.size());

    std::array<uint32_t, LLAMA_MAX_SEQ> pool_start;

    uint32_t ip = 0;
    uint32_t n_seq_canon = 0; // sequences owning a canonical pool, the pools of one sequence are disjoint
    for (llama_seq_id s = 0; s < LLAMA_MAX_SEQ; ++s) {
        const auto & sq = lay.seqs[s];
        pool_start[s] = ip;
        bool owns = false;
        for (const uint32_t j : sq.pools) {
            const size_t rep = (size_t) sq.strm*kv_size + sq.cells[j + kpool - 1].second;
            if (st.rep_gen[rep] != st.generation) {
                st.rep_gen[rep]   = st.generation;
                st.rep_first[rep] = ip;
                owns = true;
            }
            st.canon[ip++] = st.rep_first[rep];
        }
        n_seq_canon += owns;
    }
    GGML_ASSERT(ip == st.is_new.size());

    // two canonical pools overlap where they both claim a cell; the stamps now also mark claimed members,
    // a rep cell already holds its own canonical pool
    auto any_partial = [&]() {
        for (llama_seq_id s = 0; s < LLAMA_MAX_SEQ; ++s) {
            const auto & sq = lay.seqs[s];
            for (size_t k = 0; k < sq.pools.size(); ++k) {
                const uint32_t p = pool_start[s] + (uint32_t) k;
                if (st.canon[p] != p) {
                    continue;
                }
                for (uint32_t m = 0; m < kpool; ++m) {
                    const size_t c = (size_t) sq.strm*kv_size + sq.cells[sq.pools[k] + m].second;
                    if (st.rep_gen[c] == st.generation && st.rep_first[c] != p) {
                        return true;
                    }
                    st.rep_gen[c]   = st.generation;
                    st.rep_first[c] = p;
                }
            }
        }
        return false;
    };
    // the scatter takes stream-local cells, so the pools of sequences in different streams of one ubatch collide too
    st.partial_share = (idx->get_n_stream() > 1 && ubatch.n_seqs_unq > 1) || (n_seq_canon > 1 && any_partial());

    // a pool is marked once, through its canonical entry, so the scatter rows stay unique
    auto mark = [&](llama_seq_id s, size_t k) {
        const uint32_t c = st.canon[pool_start[s] + k];
        if (st.is_new[c] != st.generation) {
            st.is_new[c] = st.generation;
            ++st.n_new;
        }
    };

    for (llama_seq_id s = 0; s < LLAMA_MAX_SEQ; ++s) {
        const auto & sq = lay.seqs[s];

        // A sequence edit invalidates only pools ending after the edited position.
        const llama_pos stale_from = i_cur == 0 ?
            mem_idx_stale_batch[s] : llama_memory_hybrid_idx::POS_CLEAN;
        if (stale_from == llama_memory_hybrid_idx::POS_CLEAN) {
            continue;
        }

        auto first = std::lower_bound(sq.pools.begin(), sq.pools.end(), stale_from,
                [&](uint32_t j, llama_pos p) { return sq.cells[j + kpool - 1].first < p; });
        for (auto it = first; it != sq.pools.end(); ++it) {
            mark(s, it - sq.pools.begin());
        }
    }

    // in order mode a token's cell gives its rank, and the rank its pool: positions cannot, as an image shares one
    const bool by_order = mem->get_kpool_by_order();
    const auto *   sinfo = by_order ? &sinfos_kpool[i_cur] : nullptr;
    const uint32_t n_tps = by_order ? (uint32_t) sinfo->size() : 0;

    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        const llama_pos p = ubatch.pos[i];
        for (int32_t k = 0; k < ubatch.n_seq_id[i]; ++k) {
            const llama_seq_id s = ubatch.seq_id[i][k];
            const auto & sq = lay.seqs[s];
            if (by_order) {
                const int64_t r = kpool_rank(sq.cells, p, sinfo->idxs[i / n_tps][i % n_tps]);
                GGML_ASSERT(r >= 0);
                if ((size_t) r / kpool < sq.pools.size()) {
                    mark(s, (size_t) r / kpool);
                }
                continue;
            }
            auto it = std::upper_bound(sq.pools.begin(), sq.pools.end(), p,
                    [&](llama_pos pos, uint32_t j) { return pos < sq.cells[j].first; });
            if (it == sq.pools.begin()) {
                continue;
            }
            --it;
            if (p <= sq.cells[*it + kpool - 1].first) {
                mark(s, it - sq.pools.begin());
            }
        }
    }

    // a ubatch touches at most t_s/kpool + 1 pools per sequence, pad to that bound so the graph keeps its shape
    // as the count moves; reserve sizes the list for every pool the cache can hold, so never pad past n_pool_max
    const uint32_t n_pool_max = kv_size / kpool * idx->get_n_seq_max();
    const uint32_t bound = ubatch.n_tokens/kpool + ubatch.n_seqs_unq;

    // the later ubatches of a batch add at most t_s/kpool + 1 pools per sequence; the first one sizes the pool
    // list for all of them, so their graphs keep one shape and are reused (pipeline parallelism needs that)
    if (i_cur == 0) {
        n_pool_batch = n_tokens_rest == 0 ? 0 :
            uint32_t(std::min<uint64_t>(n_pool_max, uint64_t(st.n_pool_real) + n_tokens_rest/kpool + idx->get_n_seq_max()));
    }
    st.n_pool_g = kpool_pad(std::max(st.n_pool_real, n_pool_batch));

    // Singleton pools can occupy every cell: there is no non-representative
    // scratch row for scatter padding. Every nonempty ubatch marks a pool.
    st.n_new_g = kpool == 1 ? std::max(st.n_new, 1u) :
        std::max({st.n_new, 1u, std::min({bound, st.n_pool_g - 1, n_pool_max})});
}

const llama_memory_hybrid_idx_context::kpool_state & llama_memory_hybrid_idx_context::kpool_cur() const {
    GGML_ASSERT(kpool_st != nullptr && i_kpool == i_cur && "k-pool state read before apply()");

    return *kpool_st;
}

uint32_t llama_memory_hybrid_idx_context::get_n_kpool() const {
    return kpool_cur().n_pool_g;
}

uint32_t llama_memory_hybrid_idx_context::get_n_kpool_new() const {
    return kpool_cur().n_new_g;
}

bool llama_memory_hybrid_idx_context::get_kpool_partial_share() const {
    return kpool_cur().partial_share;
}

void llama_memory_hybrid_idx_context::set_input_kpool(ggml_tensor * pool_cells, ggml_tensor * pool_idxs, ggml_tensor * pool_mask, ggml_tensor * tail_idxs,
        ggml_tensor * sel_mask, ggml_tensor * new_pool_idxs, ggml_tensor * new_pool_rep,
        const llama_ubatch * ubatch, ggml_tensor * new_pool_pos) const {
    GGML_ASSERT(mem != nullptr && mem->get_mem_idx() != nullptr);
    GGML_ASSERT(ggml_backend_buffer_is_host(pool_cells->buffer));
    GGML_ASSERT(ggml_backend_buffer_is_host(pool_idxs->buffer));
    GGML_ASSERT(ggml_backend_buffer_is_host(pool_mask->buffer));
    GGML_ASSERT(tail_idxs == nullptr || ggml_backend_buffer_is_host(tail_idxs->buffer));

    const uint32_t kpool = mem->get_kpool();
    const uint32_t n_kv  = get_idx()->get_n_kv();

    const auto & st  = kpool_cur();
    const auto & lay = mem->kpool_layout_get();

    const uint32_t n_tokens = ubatch->n_tokens;
    const uint32_t n_pool   = (uint32_t) pool_cells->ne[0];
    const uint32_t n_new    = st.n_new;
    // the graph always pools at least one entry, padded to a stable bound, see kpool_build_state
    const uint32_t n_new_g  = st.n_new_g;

    const bool by_order = mem->get_kpool_by_order();

    GGML_ASSERT(n_pool == st.n_pool_g);
    GGML_ASSERT(st.is_new.size() == st.n_pool_real);
    GGML_ASSERT(pool_mask->ne[0] == (int64_t) n_pool && pool_mask->ne[1] == (int64_t) n_tokens);
    GGML_ASSERT(tail_idxs ? (tail_idxs->ne[0] == (int64_t) kpool - 1 && tail_idxs->ne[1] == (int64_t) n_tokens) : kpool == 1);
    GGML_ASSERT(pool_idxs->ne[0] == (int64_t) kpool && pool_idxs->ne[1] == (int64_t) n_pool);
    GGML_ASSERT(ggml_backend_buffer_is_host(new_pool_idxs->buffer));
    GGML_ASSERT(new_pool_idxs->ne[0] == (int64_t) kpool && new_pool_idxs->ne[1] == (int64_t) n_new_g);
    // the graph always scatters the fresh pooled keys back into the cache, see build_qsa_sel
    GGML_ASSERT(new_pool_rep != nullptr && ggml_backend_buffer_is_host(new_pool_rep->buffer));
    GGML_ASSERT(new_pool_rep->ne[0] == (int64_t) n_new_g);
    if (new_pool_pos != nullptr) {
        GGML_ASSERT(ggml_backend_buffer_is_host(new_pool_pos->buffer));
        GGML_ASSERT(new_pool_pos->ne[0] == 4*(int64_t) n_new_g);
    }

    const uint32_t kv_size = mem->get_mem_idx()->get_size();
    const uint32_t n_stream_kv = mem->get_mem_idx()->get_n_stream();

    auto gcell = [&](const llama_memory_hybrid_idx::kpool_layout::seq & sq, uint32_t cell) {
        return (int64_t) sq.strm*kv_size + cell;
    };

    // Sequences present in this ubatch, pools of absent sequences must fall on the scatter sentinel row.
    std::vector<uint8_t> seq_in_ub(LLAMA_MAX_SEQ, 0);
    for (uint32_t i = 0; i < n_tokens; ++i) {
        for (int32_t k = 0; k < ubatch->n_seq_id[i]; ++k) {
            seq_in_ub[ubatch->seq_id[i][k]] = 1;
        }
    }

    // a cell of this ubatch, written before any read, so the padded pools read a finite K row
    int64_t dummy_cell = 0;
    {
        const llama_seq_id s = ubatch->seq_id[0][0];
        const auto & sq = lay.seqs[s];
        auto it = std::lower_bound(sq.cells.begin(), sq.cells.end(), std::make_pair(ubatch->pos[0], 0u));
        GGML_ASSERT(it != sq.cells.end() && it->first == ubatch->pos[0]);
        dummy_cell = gcell(sq, it->second);
    }

    // in order mode a token sees the pools and the tail up to its own rank in the sequence, which its cell pins down
    std::vector<int64_t> rank;
    if (by_order) {
        const auto &   sinfo = sinfos_kpool[i_cur];
        const uint32_t n_tps = (uint32_t) sinfo.size();

        rank.resize(n_tokens);
        for (uint32_t i = 0; i < n_tokens; ++i) {
            rank[i] = kpool_rank(lay.seqs[ubatch->seq_id[i][0]].cells, ubatch->pos[i], sinfo.idxs[i / n_tps][i % n_tps]);
            GGML_ASSERT(rank[i] >= 0);
        }
    }

    // padding and absent cells point at the n_kv sentinel row, one past the live cells
    const int32_t sentinel = (int32_t) n_kv;

    float *  gm    = nullptr;
    uint32_t n_sel = 0;
    uint32_t n_top = 0; // Pools per token in the selection.
    if (sel_mask != nullptr) {
        GGML_ASSERT(ggml_backend_buffer_is_host(sel_mask->buffer));
        GGML_ASSERT(sel_mask->type == GGML_TYPE_F32);
        GGML_ASSERT(sel_mask->ne[3] == (int64_t) n_tokens && sel_mask->ne[1] == 1 && sel_mask->ne[2] == 1);
        n_sel = (uint32_t) sel_mask->ne[0];
        // The tail slots, when selected, are the n_sel % kpool != 0 remainder.
        n_top = n_sel / kpool;
        GGML_ASSERT(n_sel % kpool == 0 || n_sel % kpool == kpool - 1);
        gm = (float *) sel_mask->data;
    }

    // pools are laid out per sequence
    std::vector<uint32_t>  seq_pool_start(LLAMA_MAX_SEQ, 0);
    std::vector<llama_pos> pool_end;
    pool_end.reserve(n_pool);

    int32_t * pcell = (int32_t *) pool_cells->data;
    int32_t * pidx  = (int32_t *) pool_idxs->data;
    int32_t * nidx  = (int32_t *) new_pool_idxs->data;
    int64_t * nrep  = (int64_t *) new_pool_rep->data;
    int32_t * npos  = new_pool_pos != nullptr ? (int32_t *) new_pool_pos->data : nullptr;

    if (npos != nullptr) {
        std::fill(npos, npos + 4*n_new_g, 0);
    }

    uint32_t i_new = 0;
    for (llama_seq_id s = 0; s < LLAMA_MAX_SEQ; ++s) {
        const auto & sq = lay.seqs[s];
        seq_pool_start[s] = (uint32_t) pool_end.size();

        const bool inert = n_stream_kv > 1 && !seq_in_ub[s];

        for (size_t pi = 0; pi < sq.pools.size(); ++pi) {
            const uint32_t j  = sq.pools[pi];
            const uint32_t ip = (uint32_t) pool_end.size();
            GGML_ASSERT(ip + 1 < n_pool);

            // The pooled key lives in the last member's row.
            const uint32_t rep = sq.cells[j + kpool - 1].second;
            pcell[ip] = (int32_t) gcell(sq, rep);

            // A copy of a shared pool is dead to every token, its -inf scatter must not land on the cells
            // the canonical entry makes live.
            const bool copy = st.canon[ip] != ip;

            for (uint32_t k = 0; k < kpool; ++k) {
                pidx[(size_t) ip*kpool + k] = inert || copy ? sentinel : (int32_t) sq.cells[j + k].second;
            }

            if (st.is_new[ip] == st.generation) {
                GGML_ASSERT(i_new < n_new);
                for (uint32_t k = 0; k < kpool; ++k) {
                    nidx[(size_t) i_new*kpool + k] = (int32_t) gcell(sq, sq.cells[j + k].second);
                }
                nrep[i_new] = gcell(sq, rep);
                if (npos != nullptr) {
                    // a pooled key is rotated to the M-RoPE position of its first member
                    const uint32_t c = sq.cells[j].second;
                    const auto &   e = mem->get_mem_idx()->get_cells(s).ext_get(c);
                    npos[0*n_new_g + i_new] = sq.cells[j].first;
                    npos[1*n_new_g + i_new] = e.y;
                    npos[2*n_new_g + i_new] = e.x;
                    npos[3*n_new_g + i_new] = sq.cells[j].first;
                }
                ++i_new;
            }

            pool_end.push_back(sq.cells[j + kpool - 1].first);
        }
    }
    GGML_ASSERT(i_new == n_new);

    // Padded entries re-pool cells whose pooled slot is never read: only the reps of complete pools are read.
    // Each entry takes its own cell, entries sharing one would write it from several threads in the scatter.
    if (n_new_g > n_new) {
        std::vector<int64_t> reps(pcell, pcell + pool_end.size());
        std::sort(reps.begin(), reps.end());

        int64_t pad_cell = 0;
        for (uint32_t i = n_new; i < n_new_g; ++i, ++pad_cell) {
            while (std::binary_search(reps.begin(), reps.end(), pad_cell)) {
                ++pad_cell;
            }
            GGML_ASSERT(pad_cell < (int64_t) kv_size*n_stream_kv);
            for (uint32_t k = 0; k < kpool; ++k) {
                nidx[(size_t) i*kpool + k] = (int32_t) pad_cell;
            }
            nrep[i] = pad_cell;
        }
    }

    const uint32_t n_pool_real = (uint32_t) pool_end.size();
    for (uint32_t ip = n_pool_real; ip < n_pool; ++ip) {
        pcell[ip] = (int32_t) dummy_cell; // pool_cells always addresses the K storage
        for (uint32_t k = 0; k < kpool; ++k) {
            pidx[(size_t) ip*kpool + k] = sentinel;
        }
    }

    // a pool is visible when it belongs to the token's sequence and ends at or before it
    auto fill_mask = [&](auto * data) {
        using T = std::remove_pointer_t<decltype(data)>;
        const T keep = llama_cast<T>(0.0f);
        const T drop = llama_cast<T>(-INFINITY);

        for (uint32_t i = 0; i < n_tokens; ++i) {
            const llama_seq_id s = ubatch->seq_id[i][0];
            const llama_pos    p = ubatch->pos[i];

            T * row = data + (size_t) i*n_pool;
            std::fill(row, row + n_pool, drop);

            const uint32_t p0 = seq_pool_start[s];
            const uint32_t p1 = p0 + (uint32_t) lay.seqs[s].pools.size();
            const uint32_t nv = by_order ? std::min(p1 - p0, (uint32_t) ((rank[i] + 1)/kpool)) :
                (uint32_t) (std::upper_bound(pool_end.begin() + p0, pool_end.begin() + p1, p) - (pool_end.begin() + p0));
            for (uint32_t ip = p0; ip < p0 + nv; ++ip) {
                row[st.canon[ip]] = keep;
            }

            // Finite visible pools occupy the first min(nv, n_top) ranked slots.
            if (gm != nullptr) {
                const uint32_t nvc = std::min(nv, n_top);
                float * grow = gm + (size_t) i*n_sel;
                std::fill(grow,                        grow + (size_t) nvc*kpool,  0.0f);
                std::fill(grow + (size_t) nvc*kpool,   grow + (size_t) n_top*kpool, -INFINITY);
            }
        }
    };
    if (pool_mask->type == GGML_TYPE_F16) {
        fill_mask((ggml_fp16_t *) pool_mask->data);
    } else {
        fill_mask((float *) pool_mask->data);
    }

    int32_t * tidx = tail_idxs ? (int32_t *) tail_idxs->data : nullptr;
    for (uint32_t i = 0; i < n_tokens; ++i) {
        const llama_seq_id s = ubatch->seq_id[i][0];
        const llama_pos    p = ubatch->pos[i];
        const auto & sq = lay.seqs[s];

        const uint32_t n_tail = by_order ?
            (uint32_t) ((rank[i] + 1) % kpool) :
            (uint32_t) ((p - sq.pos_min + 1) % (llama_pos) kpool);

        for (uint32_t k = 0; k < kpool - 1; ++k) {
            int32_t cell = sentinel;
            bool    real = false;
            if (k < n_tail && by_order) {
                const uint32_t c = sq.cells[rank[i] - k].second;
                cell = (int32_t) c;
                real = true;
            } else if (k < n_tail) {
                const llama_pos pt = p - (llama_pos) k;
                auto it = std::lower_bound(sq.cells.begin(), sq.cells.end(), std::make_pair(pt, 0u));
                if (it != sq.cells.end() && it->first == pt) {
                    cell = (int32_t) it->second;
                    real = true;
                }
            }
            tidx[(size_t) i*(kpool - 1) + k] = cell;

            if (gm != nullptr && n_sel % kpool != 0) {
                gm[(size_t) i*n_sel + (size_t) n_top*kpool + k] = real ? 0.0f : -INFINITY;
            }
        }
    }
}
