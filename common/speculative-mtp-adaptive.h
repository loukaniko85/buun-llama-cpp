#pragma once

#include <algorithm>

// Keep the established conservative full-depth -> 2 decision, but make it reversible
// within a request. Depth 1 did not earn its exploration cost in serving tests.
class common_speculative_mtp_adaptive {
public:
    explicit common_speculative_mtp_adaptive(int minimum = 1, int maximum = 3)
        : maximum_depth(std::max(2, maximum)),
          minimum_depth(std::max(2, std::min(maximum_depth, minimum))), cap(maximum_depth) {}

    int depth() const { return cap; }

    void reset() { *this = common_speculative_mtp_adaptive(minimum_depth, maximum_depth); }

    // Keep a learned depth across requests, but treat the new prefix as an
    // opportunity to recover. Restarting every request at full depth repeats
    // the losing probe on a stream of low-match prose requests.
    void begin() {
        attempts = full = prefix_full = full_streak = 0;
        // Preserve the periodic countdown too: many short requests must not
        // postpone a full-depth probe indefinitely.
    }

    void accept(int drafted, int accepted, bool other) {
        // Clipped/failed drafts and another implementation's proposals do not
        // measure our selected depth. Duplicate carry refreshes have no draft.
        if (other || drafted != cap || accepted < 0 || accepted > drafted || minimum_depth == maximum_depth) {
            return;
        }

        if (cap == minimum_depth) {
            full_streak = accepted == cap ? full_streak + 1 : 0;
            // A matching streak signals a phase change only if the retained
            // positions were not already near-perfect in the full-depth probe.
            // Periodic probes also recover when no such streak is observed.
            if (--hold == 0 || (prefix_full < 12 && full_streak >= 8)) {
                reset();
            }
            return;
        }

        full += accepted == cap;
        prefix_full += accepted >= minimum_depth;
        if (++attempts == 16) {
            if (full < 8) {
                cap = minimum_depth;
                hold = 256;
            } else {
                prefix_full = 0;
            }
            attempts = full = 0;
        }
    }

private:
    int maximum_depth;
    int minimum_depth;
    int cap;
    int attempts = 0;
    int full = 0;
    int prefix_full = 0;
    int full_streak = 0;
    int hold = 0;
};
