#pragma once

// Internal mask algorithms shared with the direct correctness tests.
#include "llama-batch.h"
#include "llama-hparams.h"
#include "llama-kv-cells.h"
#include "llama-impl.h"

#include <algorithm>
#include <cmath>
#include <climits>
#include <unordered_map>

struct args_set_input_kq_mask {
    const llama_hparams & hparams;
    const llama_ubatch  * ubatch;

    const std::vector<llama_kv_cells> & v_cells;
    const std::vector<uint32_t>       & seq_to_stream;

    uint32_t       n_swa;
    llama_swa_type swa_type;

    int64_t n_kv;
    int64_t n_stream;
    int64_t n_tps;
};

template<typename T, bool causal, bool swa, bool is_2d, bool alibi>
static void set_input_kq_mask_impl(const args_set_input_kq_mask & args, T * data) {
  //const auto & hparams = args.hparams;
    const auto & ubatch  = args.ubatch;

    const auto & v_cells       = args.v_cells;
    const auto & seq_to_stream = args.seq_to_stream;

    const uint32_t       n_swa    = args.n_swa;
    const llama_swa_type swa_type = args.swa_type;

    const int64_t n_kv     = args.n_kv;
    const int64_t n_stream = args.n_stream;
    const int64_t n_tps    = args.n_tps;

    const T mask_keep = llama_cast<T>(0.0f);
    const T mask_drop = llama_cast<T>(-INFINITY);

    // the min position in the batch for each sequence
    llama_pos seq_pos_min[LLAMA_MAX_SEQ];
    std::fill(seq_pos_min, seq_pos_min + LLAMA_MAX_SEQ, INT32_MAX);

    for (uint32_t i = 0; i < ubatch->n_tokens; ++i) {
        const llama_seq_id seq_id = ubatch->seq_id[i][0];

        seq_pos_min[seq_id] = std::min(seq_pos_min[seq_id], ubatch->pos[i]);
    }

    for (uint32_t s = 0; s < n_stream; ++s) {
        // bookkeeping of the KQ mask cells that could change for other tokens of the same sequence
        std::unordered_map<llama_seq_id, uint32_t>              seq_srct;
        std::unordered_map<llama_seq_id, std::vector<uint32_t>> seq_idxs;

        for (uint32_t ii = 0; ii < n_tps; ++ii) {
            const uint32_t i = s*n_tps + ii;

            const llama_seq_id seq_id = ubatch->seq_id[i][0];

            const auto & cells = v_cells.at(seq_to_stream[seq_id]);

                  llama_pos p0 = -1;
            const llama_pos p1 = ubatch->pos[i];

            // for M-RoPE
            const llama_pos p1_x = is_2d ? ubatch->pos[i + ubatch->n_tokens*2] : 0;
            const llama_pos p1_y = is_2d ? ubatch->pos[i + ubatch->n_tokens]   : 0;

            const uint64_t idst = n_kv*i;

            // for tokens of the same sequence, the mask is mostly the same, so we can reuse it
            // the only cells that could change are the ones that are with similar positions as the
            //   ones in the batch (i.e. due to causal masking, SWA, etc.)
            // keep track of those cells and shortcut the loop to save time
            // note: this optimization is not compatible with Alibi position encoding
            // ref:  https://github.com/ggml-org/llama.cpp/pull/18842
            bool prev = false;

            auto & idxs = seq_idxs[seq_id];

            if (!alibi) {
                if (seq_srct.find(seq_id) != seq_srct.end()) {
                    const uint32_t srct = seq_srct[seq_id];

                    const uint64_t idst_prev = n_kv*srct;

                    std::copy(data + idst_prev, data + idst_prev + n_kv, data + idst);

                    prev = true;
                } else {
                    idxs.clear();
                    idxs.reserve(ubatch->n_tokens + n_swa + 32);

                    seq_srct[seq_id] = i;
                }
            }

            for (uint32_t jj = 0; jj < n_kv; ++jj) {
                uint32_t j = jj;

                // we have an exiting mask for this sequence -> update just seq_idxs
                if (!alibi) {
                    if (prev) {
                        if (jj >= idxs.size()) {
                            break;
                        }

                        j = idxs[jj];
                    }
                }

                if (cells.is_empty(j)) {
                    goto skip;
                }

                // mask the token if not the same sequence
                if (!cells.seq_has(j, seq_id)) {
                    goto skip;
                }

                p0 = cells.pos_get(j);

                if (!alibi) {
                    if (!prev) {
                        // record all cells for which: p0 >= seq_pos_min[seq_id] - n_swa - 32
                        if (p0 + (int32_t) (n_swa + 32) >= seq_pos_min[seq_id]) {
                            idxs.push_back(j);
                        }
                    }
                }

                if (causal) {
                    // mask future tokens
                    if (p0 > p1) {
                        goto skip;
                    }

                    // M-RoPE causal mask
                    if (is_2d) {
                        if (p0 == p1) {
                            const auto & p0_ext = cells.ext_get(j);

                            if (p0_ext.is_2d_gt(p1_x, p1_y)) {
                                goto skip;
                            }
                        }
                    }
                }

                // apply SWA if any
                if (swa) {
                    // see llama_non_causal_type
                    const bool in_span = !causal && args.hparams.non_causal_type == LLAMA_NON_CAUSAL_TYPE_SWA_FULL && p0 >= seq_pos_min[seq_id];
                    if (!in_span && llama_hparams::is_masked_swa(n_swa, swa_type, p0, p1)) {
                        goto skip;
                    }
                }

                if (alibi) {
                    data[idst + j] = llama_cast<T>(static_cast<float>(-std::abs(p0 - p1)));
                } else {
                    data[idst + j] = mask_keep;
                }

                continue;
skip:
                data[idst + j] = mask_drop;
            }
        }
    }
}

// For an ordinary contiguous text prefix, every row is just a visible interval.
// Validate the cells rather than assuming physical placement survives restore,
// sequence sharing or compaction. Other layouts keep the general mask builder.
template<typename T>
static bool set_input_kq_mask_contiguous(const args_set_input_kq_mask & args, T * data) {
    if (args.n_tps < 32 || args.hparams.use_alibi || args.n_swa > INT32_MAX) {
        return false;
    }
    for (int64_t s = 0; s < args.n_stream; ++s) {
        const auto seq = args.ubatch->seq_id[s*args.n_tps][0];
        const auto & cells = args.v_cells.at(args.seq_to_stream[seq]);
        for (int64_t i = s*args.n_tps; i < (s + 1)*args.n_tps; ++i) {
            if (args.ubatch->seq_id[i][0] != seq || args.ubatch->pos[i] < 0) {
                return false;
            }
        }
        if (!args.n_kv || cells.is_empty(0)) {
            return false;
        }
        const int64_t first = cells.pos_get(0);
        if (first < 0) {
            return false;
        }
        int64_t used = 0;
        while (used < args.n_kv && !cells.is_empty(used)) {
            if (!cells.seq_has(used, seq) || cells.pos_get(used) != first + used) {
                return false;
            }
            ++used;
        }
        for (int64_t j = used; j < args.n_kv; ++j) {
            if (!cells.is_empty(j)) {
                return false;
            }
        }
        for (int64_t i = s*args.n_tps; i < (s + 1)*args.n_tps; ++i) {
            const int64_t q = args.ubatch->pos[i];
            int64_t low = 0;
            switch (args.swa_type) {
                case LLAMA_SWA_TYPE_NONE: break;
                case LLAMA_SWA_TYPE_STANDARD: low = q - int64_t(args.n_swa) + 1; break;
                case LLAMA_SWA_TYPE_CHUNKED:
                    if (!args.n_swa) return false;
                    low = (q/args.n_swa)*args.n_swa;
                    break;
                case LLAMA_SWA_TYPE_SYMMETRIC: low = q - int64_t(args.n_swa)/2; break;
                default: return false;
            }
            int64_t end = std::clamp(q + 1 - first, int64_t(0), used);
            // Strictly consecutive temporal positions leave at most one cell
            // tied with this query. Preserve the M-RoPE spatial tie-break at
            // that boundary; repeated temporal positions took the fallback.
            if (args.ubatch->is_pos_2d() && q >= first && q < first + used) {
                const auto x = args.ubatch->pos[i + 2*args.ubatch->n_tokens];
                const auto y = args.ubatch->pos[i + args.ubatch->n_tokens];
                if (cells.ext_get(q - first).is_2d_gt(x, y)) {
                    --end;
                }
            }
            const int64_t begin = std::clamp(low - first, int64_t(0), end);
            T * row = data + i*args.n_kv;
            std::fill(row, row + begin, llama_cast<T>(-INFINITY));
            std::fill(row + begin, row + end, llama_cast<T>(0.0f));
            std::fill(row + end, row + args.n_kv, llama_cast<T>(-INFINITY));
        }
    }
    return true;
}

template<typename T, bool causal, bool swa, bool is_2d>
static void set_input_kq_mask_impl(const args_set_input_kq_mask & args, T * data) {
    const bool alibi = args.hparams.use_alibi;
    if (alibi) {
        set_input_kq_mask_impl<T, causal, swa, is_2d, true> (args, data);
    } else {
        set_input_kq_mask_impl<T, causal, swa, is_2d, false>(args, data);
    }
}

template<typename T, bool causal, bool swa>
static void set_input_kq_mask_impl(const args_set_input_kq_mask & args, T * data) {
    const bool is_2d = args.ubatch->is_pos_2d();
    if (is_2d) {
        set_input_kq_mask_impl<T, causal, swa, true> (args, data);
    } else {
        set_input_kq_mask_impl<T, causal, swa, false>(args, data);
    }
}

template<typename T, bool causal>
static void set_input_kq_mask_impl(const args_set_input_kq_mask & args, T * data) {
    const bool swa = args.swa_type != LLAMA_SWA_TYPE_NONE;
    if (swa) {
        set_input_kq_mask_impl<T, causal, true> (args, data);
    } else {
        set_input_kq_mask_impl<T, causal, false>(args, data);
    }
}

template<typename T>
static void set_input_kq_mask_impl(const args_set_input_kq_mask & args, T * data, bool causal_attn) {
    if (causal_attn && set_input_kq_mask_contiguous(args, data)) {
        return;
    }
    if (causal_attn) {
        set_input_kq_mask_impl<T, true> (args, data);
    } else {
        set_input_kq_mask_impl<T, false>(args, data);
    }
}
