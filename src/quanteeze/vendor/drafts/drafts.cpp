// SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
// Process-wide speculative drafting round counters shared across request threads (`quanteeze.mtp.counted`).
// With per-request threads, each request runs a shallow copy of the drafter (`request_threads._drafter`), so
// Python attributes updated on the copy are lost when the request finishes. Three KAME `atomic<int64_t>` counters
// accumulate rounds, drafted tokens, and accepted tokens (in thousandths) lock-free across all threads.
#include <nanobind/nanobind.h>
#include <nanobind/stl/tuple.h>

#include <cmath>
#include <cstdint>
#include <tuple>

#include "atomic.h"

namespace nb = nanobind;

namespace {

constexpr double MILLI = 1000.0;

atomic<int64_t> rounds{0}, drafted{0}, accepted_milli{0};

void count(int64_t n, int64_t draft, double accepted) {
    rounds.fetch_add(n, std::memory_order_relaxed);
    drafted.fetch_add(draft, std::memory_order_relaxed);
    accepted_milli.fetch_add(static_cast<int64_t>(std::llround(accepted * MILLI)), std::memory_order_relaxed);
}

std::tuple<int64_t, int64_t, double> counts() {
    return {
        rounds.load(std::memory_order_relaxed),
        drafted.load(std::memory_order_relaxed),
        accepted_milli.load(std::memory_order_relaxed) / MILLI,
    };
}

void from_zero() {
    rounds.store(0, std::memory_order_relaxed);
    drafted.store(0, std::memory_order_relaxed);
    accepted_milli.store(0, std::memory_order_relaxed);
}

}  // namespace

NB_MODULE(_drafts, m) {
    m.doc() = "quanteeze: cross-thread speculative drafting round counters (drafts.cpp)";
    m.def("add", &count, nb::arg("n"), nb::arg("draft"), nb::arg("accepted"));
    m.def("read", &counts);
    m.def("reset", &from_zero);
}
