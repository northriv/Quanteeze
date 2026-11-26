// SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
// Lanes: pinned mailbox rows per request thread for cpu_experts::Server. Each calling thread claims
// a free lane once via CAS on KAME's atomic<int> and releases it on thread exit (~LanesHeld).
// Both Server and participating threads hold std::shared_ptr<Lanes>, so a freed Server's address
// reused later is never confused with an earlier one (`gone` drops stale thread entries).
// A 1-lane server has no exclusive owner and serves every caller on lane 0.
#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "atomic.h"

namespace {

struct Lanes {
    explicit Lanes(int n) : taken(std::make_unique<atomic<int>[]>(n)), n(n) {
        for (int k = 0; k < n; ++k) taken[k] = 0;
    }
    std::unique_ptr<atomic<int>[]> taken;  // 1 while owned by a live thread
    int n;
    atomic<int> gone{0};
};

struct LanesHeld {
    std::vector<std::pair<std::shared_ptr<Lanes>, int>> held;
    ~LanesHeld() {
        for (auto& h : held) h.first->taken[h.second] = 0;
    }
};

// Return calling thread's lane for `lanes`, claiming one if free, or -1 if all lanes are held.
inline int lane_of(const std::shared_ptr<Lanes>& lanes) {
    if (!lanes) return -1;
    if (lanes->n == 1) return 0;
    static thread_local LanesHeld mine;
    for (size_t k = 0; k < mine.held.size();) {
        if (mine.held[k].first == lanes) return mine.held[k].second;
        if (mine.held[k].first->gone) {
            mine.held[k] = std::move(mine.held.back());
            mine.held.pop_back();
        } else {
            ++k;
        }
    }
    for (int k = 0; k < lanes->n; ++k) {
        if (lanes->taken[k].compare_set_strong(0, 1)) {
            mine.held.emplace_back(lanes, k);
            return k;
        }
    }
    return -1;
}

}  // namespace
