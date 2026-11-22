// SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
// Zero-GIL memmapped row page-touch helper for quanteeze.ngram_prefetch.
// Reads the first and last element of each requested row with the GIL released
// so concurrent SSD page faults do not block other Python threads.

#include <nanobind/nanobind.h>

#include <cstdint>

namespace nb = nanobind;

namespace {

int64_t qz_touch(uintptr_t base_addr, int64_t stride, int64_t last, int64_t elem, int64_t count,
                 uintptr_t rows_addr, int64_t n, uintptr_t out_addr) {
    if (n <= 0) {
        return 0;
    }
    if (base_addr == 0 || rows_addr == 0 || stride <= 0 || elem <= 0 || count <= 0 || last < 0) {
        return -1;
    }
    nb::gil_scoped_release release;
    const volatile uint8_t* base = reinterpret_cast<const volatile uint8_t*>(base_addr);
    const int64_t* rows = reinterpret_cast<const int64_t*>(rows_addr);
    uint8_t* out = reinterpret_cast<uint8_t*>(out_addr);

    for (int64_t i = 0; i < n; ++i) {
        if (rows[i] < 0 || rows[i] >= count) {
            return -1;
        }
    }

    uint64_t sink = 0;
    for (int64_t i = 0; i < n; ++i) {
        const volatile uint8_t* row = base + rows[i] * stride;
        for (int64_t b = 0; b < elem; ++b) {
            const uint8_t first = row[b];
            const uint8_t end = row[last + b];
            if (out != nullptr) {
                out[(2 * i) * elem + b] = first;
                out[(2 * i + 1) * elem + b] = end;
            }
            sink ^= static_cast<uint64_t>(first ^ end);
        }
    }
#if defined(__GNUC__) || defined(__clang__)
    __asm__ volatile("" : : "r"(sink));
#endif
    return n;
}

}  // namespace

NB_MODULE(_touch, m) {
    m.doc() = "Zero-GIL memmapped row page-touch extension for QuantEEze";
    m.def("touch", &qz_touch,
          nb::arg("base"), nb::arg("stride"), nb::arg("last"), nb::arg("elem"),
          nb::arg("count"), nb::arg("rows"), nb::arg("n"), nb::arg("out") = 0);
}
