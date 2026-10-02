#include "llama-qsa-bias.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void check(bool ok) {
    if (!ok) {
        std::fprintf(stderr, "QSA bias prefix mismatch\n");
        std::abort();
    }
}

int main() {
    size_t cases = 0;
    for (int64_t blocks : { 1, 2, 8, 1024 }) {
        for (int64_t full = 0; full <= blocks; ++full) {
            for (int partial = 0; partial <= int(full < blocks); ++partial) {
                const int64_t active = full + partial;
                std::vector<int32_t> rep(blocks, -1), complete(blocks, 0);
                for (int64_t b = 0; b < active; ++b) {
                    rep[b] = int32_t(7*(active - b)); // physical order need not match positions
                    complete[b] = b < full;
                }
                const auto prefix = llama_qsa_classify_bias_prefix(rep.data(), complete.data(), blocks);
                check(prefix.full == full && prefix.active == active);
                for (int64_t ratio : { 1, 4, 16, 64 }) {
                    for (int64_t query : { int64_t(0), ratio - 1, ratio, blocks*ratio - 1, int64_t(INT32_MAX) }) {
                        for (bool causal : { false, true }) {
                            std::vector<float> expected(blocks), actual(blocks, 123.0f);
                            const int64_t tail_start = (query + 1)/ratio*ratio;
                            for (int64_t b = 0; b < blocks; ++b) {
                                if (rep[b] < 0) {
                                    expected[b] = -std::numeric_limits<float>::infinity();
                                } else {
                                    const bool incomplete = complete[b] == 0;
                                    expected[b] = !causal ? (incomplete ? 1e9f : 0.0f) :
                                        (b*ratio >= tail_start ? 1e9f :
                                         (incomplete ? -std::numeric_limits<float>::infinity() : 0.0f));
                                }
                            }
                            llama_qsa_fill_bias_prefix(actual.data(), blocks, prefix, query, ratio, causal);
                            check(std::memcmp(actual.data(), expected.data(), blocks*sizeof(float)) == 0);
                            ++cases;
                        }
                    }
                }
            }
        }
    }
    // Holes and old incomplete blocks before later live blocks must fall back.
    const int32_t hole_rep[] = { 0, -1, 2, -1 };
    const int32_t old_tail_rep[] = { 0, 1, 2, -1 };
    const int32_t flags[] = { 1, 0, 1, 0 };
    check(llama_qsa_classify_bias_prefix(hole_rep, flags, 4).full < 0);
    check(llama_qsa_classify_bias_prefix(old_tail_rep, flags, 4).full < 0);
    std::printf("QSA block bias: %zu exact rows and fallback cases pass\n", cases);
}
