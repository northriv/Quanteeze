// SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
// Host-side C++20 Engram multi-head rolling XOR-prime hash computation for DeepSeek V4.1
// (quanteeze.engram_host, QUANTEEZE_ENGRAM_HASHES_ON_HOST).
// Matches mlx-vlm's NgramHashState bit for bit without stalling on GPU->host synchronization.

#include <nanobind/nanobind.h>

#include <algorithm>
#include <cstdint>

namespace nb = nanobind;

namespace {

constexpr int64_t DEAD = -1;
constexpr int MAX_NGRAM = 16;

int qz_hashes(uintptr_t token_map_addr, int64_t vocab, uintptr_t mult_addr, uintptr_t primes_addr,
              uintptr_t offsets_addr, int64_t pad, int64_t n, int64_t layers, int64_t heads,
              uintptr_t history_addr, int64_t hist_len, uintptr_t ids_addr, int64_t L,
              int64_t start, uintptr_t mask_addr, uintptr_t out_addr) {
    if (L < 0 || start < 0 || n < 2 || n > MAX_NGRAM || layers < 0 || heads < 0 || start + L > hist_len) {
        return -1;
    }
    if (L == 0) {
        return 0;
    }
    if (!token_map_addr || !mult_addr || !primes_addr || !offsets_addr || !history_addr || !ids_addr || !out_addr) {
        return -1;
    }
    nb::gil_scoped_release release;
    const int64_t* token_map = reinterpret_cast<const int64_t*>(token_map_addr);
    const int64_t* mult = reinterpret_cast<const int64_t*>(mult_addr);
    const int64_t* primes = reinterpret_cast<const int64_t*>(primes_addr);
    const int64_t* offsets = reinterpret_cast<const int64_t*>(offsets_addr);
    int64_t* history = reinterpret_cast<int64_t*>(history_addr);
    const int64_t* ids = reinterpret_cast<const int64_t*>(ids_addr);
    const uint8_t* mask = reinterpret_cast<const uint8_t*>(mask_addr);
    int64_t* out = reinterpret_cast<int64_t*>(out_addr);

    for (int64_t j = 0; j < L; ++j) {
        const int64_t id = ids[j];
        if (id < 0 || id >= vocab) {
            return -1;
        }
        history[start + j] = (mask && !mask[j]) ? DEAD : token_map[id];
    }

    const int64_t cols = (n - 1) * heads;
    for (int64_t j = 0; j < L; ++j) {
        const int64_t pos = start + j;
        int64_t tokens[MAX_NGRAM];
        bool blocked = false;
        for (int64_t shift = 0; shift < n; ++shift) {
            const int64_t at = std::min(std::max<int64_t>(pos - shift, 0), hist_len - 1);
            const int64_t source = history[at];
            blocked = blocked || pos < shift || source == DEAD;
            tokens[shift] = blocked ? pad : source;
        }
        for (int64_t l = 0; l < layers; ++l) {
            const int64_t* m = mult + l * n;
            int64_t* o = out + (j * layers + l) * cols;
            uint64_t rolling = static_cast<uint64_t>(tokens[0]) * static_cast<uint64_t>(m[0]);
            for (int64_t i = 1; i < n; ++i) {
                rolling ^= static_cast<uint64_t>(tokens[i]) * static_cast<uint64_t>(m[i]);
                const int64_t r = static_cast<int64_t>(rolling);
                const int64_t* p = primes + (l * (n - 1) + (i - 1)) * heads;
                const int64_t* off = offsets + l * cols + (i - 1) * heads;
                for (int64_t h = 0; h < heads; ++h) {
                    int64_t mod = r % p[h];
                    if (mod != 0 && ((mod < 0) != (p[h] < 0))) {
                        mod += p[h];
                    }
                    o[(i - 1) * heads + h] = mod + off[h];
                }
            }
        }
    }
    return 0;
}

}  // namespace

NB_MODULE(_engram, m) {
    m.doc() = "Host-side C++20 DeepSeek V4.1 Engram hash helper for QuantEEze";
    m.def("hashes", &qz_hashes,
          nb::arg("token_map"), nb::arg("vocab"), nb::arg("mult"), nb::arg("primes"),
          nb::arg("offsets"), nb::arg("pad"), nb::arg("n"), nb::arg("layers"), nb::arg("heads"),
          nb::arg("history"), nb::arg("hist_len"), nb::arg("ids"), nb::arg("L"),
          nb::arg("start"), nb::arg("mask") = 0, nb::arg("out") = 0);
}
