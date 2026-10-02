#include "llama-kv-cache-mask.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
template<class T>
static void check_mask(unsigned seed, int variant, bool causal) {
    std::mt19937 rng(seed);
    const int row_counts[] = {1, 31, 32, 33, 64};
    const int streams = 1 + seed%2, rows = row_counts[(seed/16)%5], tokens = streams*rows;
    const int used = 40 + seed%41, capacity = used + seed%9;
    const int first = seed%3 == 0 ? 0 : seed%81;
    llama_hparams hp{};
    hp.use_alibi = variant == 6;
    std::vector<llama_kv_cells> cells(streams);
    std::vector<uint32_t> mapping(streams);
    std::vector<llama_pos> positions(3*tokens);
    std::vector<llama_seq_id> sequences(tokens);
    std::vector<llama_seq_id *> seq_ptrs(tokens);
    for (int s = 0; s < streams; ++s) {
        mapping[s] = streams - s - 1;
        auto & c = cells[mapping[s]];
        c.resize(capacity);
        for (int j = 0; j < used; ++j) {
            if (variant == 1 && j == used/2) continue; // interior hole
            c.pos_set(j, first+j-(variant == 2 && j == used/2)); // duplicate position
            c.seq_add(j, variant == 3 && j == used/2 ? streams : s);
            if (variant == 7) c.seq_add(j, streams); // shared but valid membership
            llama_kv_cell_ext ext;
            ext.x = rng()%160; ext.y = rng()%160;
            c.ext_set(j, ext);
        }
        for (int i = s*rows; i < (s+1)*rows; ++i) {
            sequences[i] = s;
            seq_ptrs[i] = &sequences[i];
            positions[i] = rng()%180;
            positions[i+tokens] = rng()%160;
            positions[i+2*tokens] = rng()%160;
        }
    }
    if (variant == 4) positions[tokens-1] = -1;
    if (variant == 5 && streams == 2) sequences[tokens-1] = 0;
    llama_ubatch batch{};
    batch.n_tokens = tokens; batch.n_pos = (seed/8)%2 ? 3 : 1;
    batch.pos = positions.data(); batch.seq_id = seq_ptrs.data();
    const llama_swa_type types[] = {LLAMA_SWA_TYPE_NONE, LLAMA_SWA_TYPE_STANDARD,
        LLAMA_SWA_TYPE_CHUNKED, LLAMA_SWA_TYPE_SYMMETRIC};
    const args_set_input_kq_mask args{hp, &batch, cells, mapping,
        unsigned(1+seed%33), types[(seed/2)%4], capacity, streams, rows};
    std::vector<T> original(tokens*capacity), candidate(tokens*capacity);
    if (causal) set_input_kq_mask_impl<T, true>(args, original.data());
    else set_input_kq_mask_impl<T, false>(args, original.data());
    const bool eligible = set_input_kq_mask_contiguous(args, candidate.data());
    const bool expected = rows >= 32 &&
            (variant == 0 || variant == 7 || (variant == 5 && streams == 1));
    if (eligible != expected) {
        fprintf(stderr, "admission mismatch seed=%u variant=%d actual=%d\n", seed, variant, eligible);
        exit(3);
    }
    set_input_kq_mask_impl<T>(args, candidate.data(), causal);
    if (memcmp(original.data(), candidate.data(), candidate.size()*sizeof(T))) {
        fprintf(stderr, "mask mismatch seed=%u variant=%d causal=%d bytes=%zu\n", seed, variant, causal, sizeof(T));
        exit(4);
    }
}
int main() {
    // Initializes the FP16 lookup table used by the ordinary cast helpers.
    ggml_init_params params{1024*1024, nullptr, true};
    auto * ctx = ggml_init(params);
    for (unsigned seed = 0; seed < 200; ++seed) {
        for (int variant = 0; variant < 8; ++variant) {
            for (bool causal : {false, true}) {
                // Negative positions violate the original SWA precondition;
                // test rejection separately through the no-SWA seeds only.
                if (variant == 4 && (seed/2)%4 != 0) continue;
                check_mask<float>(seed, variant, causal);
                check_mask<ggml_fp16_t>(seed, variant, causal);
            }
        }
    }
    ggml_free(ctx);
    puts("contiguous and general KV masks match");
}
