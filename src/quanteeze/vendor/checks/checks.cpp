// SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
// quanteeze.native._checks: per-shape kernel check counters and sticky choices in one atomic word
// (quanteeze.kernel_checks.tables). A shape's choice and count sit in a single 64-bit word updated
// via compare-and-swap so concurrent request threads cannot overwrite an earlier mismatch. Mismatches
// stick: once failing, the word records the reference choice with all checks counted, and no later
// match overwrites it.
//
// Each shape gets a permanent Cell keyed by table name and key repr. The lookup table itself is
// copied and atomically swapped via KAME's atomic_shared_ptr; cells live for the process lifetime so
// Python holds raw references safely without calling back into Python or spawning threads.
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <cstdint>
#include <map>
#include <memory>
#include <string>

#include "atomic.h"
#include "atomic_smart_ptr.h"

namespace nb = nanobind;

namespace {

// Word layout: bit 61 = has choice, bits 32..60 = choice code (0 False, 1 True, 2.. code()'s),
// bits 0..31 = checks counted.
constexpr int64_t HAS = int64_t(1) << 61, COUNT = (int64_t(1) << 32) - 1;
constexpr int CODE_SHIFT = 32;
constexpr int64_t CODE = ((int64_t(1) << 29) - 1) << CODE_SHIFT;

struct Cell {
    atomic<int64_t> word{0};

    int64_t load() const { return word.load(std::memory_order_acquire); }

    // Count one check. If the word already records `reference` with at least `checks` counted,
    // leave it sticky and unchanged. Otherwise a failing check pins `reference` with `checks`,
    // and a matching check records `code` and increments the count. Returns the resulting word.
    int64_t count(int64_t code, int64_t reference, int64_t checks, bool failing) {
        const int64_t coded = (code << CODE_SHIFT) & CODE;
        const int64_t referred = (reference << CODE_SHIFT) & CODE;
        for (;;) {
            const int64_t was = load();
            const int64_t seen = was & COUNT;
            if ((was & HAS) && (was & CODE) == referred && seen >= checks) return was;
            const int64_t now = failing ? (HAS | referred | (checks & COUNT))
                                        : (HAS | coded | (seen < COUNT ? seen + 1 : seen));
            if (word.compare_set_strong(was, now)) return now;
        }
    }

    // Update the bits selected by `mask` to `bits`, preserving all other bits.
    int64_t put(int64_t mask, int64_t bits) {
        for (;;) {
            const int64_t was = load();
            const int64_t now = (was & ~mask) | (bits & mask);
            if (word.compare_set_strong(was, now)) return now;
        }
    }
};

struct Names {
    std::map<std::string, std::shared_ptr<Cell>> cells;  // shared across table copies
    std::map<std::string, int> codes;                    // choice repr -> integer code (2..)
};

atomic_shared_ptr<Names> names{make_local_shared<Names>()};

// Return the permanent Cell for `name`, creating one via CAS table copy if absent.
Cell &cell(const std::string &name) {
    for (;;) {
        local_shared_ptr<Names> old(names);
        auto found = old->cells.find(name);
        if (found != old->cells.end()) return *found->second;
        local_shared_ptr<Names> made(make_local_shared<Names>(*old));
        made->cells.emplace(name, std::make_shared<Cell>());
        if (names.compareAndSet(old, made)) return *made->cells.find(name)->second;
    }
}

// Return the persistent integer code (>= 2) for choice repr `value`, or -1 if codes overflow.
int code(const std::string &value) {
    for (;;) {
        local_shared_ptr<Names> old(names);
        auto found = old->codes.find(value);
        if (found != old->codes.end()) return found->second;
        const int next = static_cast<int>(old->codes.size()) + 2;
        if (int64_t(next) > (CODE >> CODE_SHIFT)) return -1;
        local_shared_ptr<Names> made(make_local_shared<Names>(*old));
        made->codes.emplace(value, next);
        if (names.compareAndSet(old, made)) return next;
    }
}

}  // namespace

NB_MODULE(_checks, m) {
    m.doc() = "quanteeze.native: per-shape atomic kernel check counters (checks.cpp)";
    m.attr("HAS") = HAS;
    m.attr("COUNT") = COUNT;
    m.attr("CODE") = CODE;
    m.attr("CODE_SHIFT") = CODE_SHIFT;
    nb::class_<Cell>(m, "Cell")
        .def("load", &Cell::load)
        .def("count", &Cell::count, nb::arg("code"), nb::arg("reference"), nb::arg("checks"), nb::arg("failing"))
        .def("put", &Cell::put, nb::arg("mask"), nb::arg("bits"));
    m.def("cell", &cell, nb::arg("name"), nb::rv_policy::reference);
    m.def("code", &code, nb::arg("value"));
}
