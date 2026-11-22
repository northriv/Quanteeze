// SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
// Routed MoE expert projections computed on the CPU from HostCopy rows in host memory (quanteeze.cpu_experts):
// rows `row0` to `row1` of W x read directly from host arrays without copying:
//
// - affine (MLX): weight [rows, cols * bits / 32] uint32, `32 / bits` values per word, lowest bits first;
//   scales and biases [rows, cols / group] bf16; w = scale * q + bias.
//   So y[r] = sum_g scale[r, g] * (sum_{j in g} q[r, j] * x[j]) + bias[r, g] * xsum[g], where xsum holds the sums
//   of x over each group. Supports bits 2, 4, or 8.
// - mxfp4 (DeepSeek V4.1 FP4): weight [rows, cols / 8] uint32 of E2M1 codes, 8 per word, lowest bits first;
//   scales [rows, cols / 32] uint8 E8M0: w = 2^(scale - 127) * e2m1(q).
//
// AVX-512 (16 lanes) and AVX2 with FMA (8 lanes) are selected at runtime when supported (isa 2, 1), else plain C++ (0).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define QZ_X86 1
#endif

#define API extern "C" __attribute__((visibility("default")))

static inline float bf16(uint16_t v) {
    uint32_t bits = uint32_t(v) << 16;
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

static const float E2M1[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                               -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

static inline float e8m0(uint8_t s) {
    // 2^(s - 127): exponent bits (s == 0 is subnormal 2^-127)
    uint32_t bits = s ? uint32_t(s) << 23 : 0x00400000u;
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

// ---------------------------------------------------------------- plain C++

static void affine_plain(const uint32_t* w, const uint16_t* sc, const uint16_t* bi, int cols, int bits, int group,
                         const float* x, const float* xsum, float* y, int row0, int row1) {
    const int per = 32 / bits, words = cols / per, groups = cols / group;
    const uint32_t mask = (1u << bits) - 1u;
    for (int r = row0; r < row1; ++r) {
        const uint32_t* wr = w + size_t(r) * words;
        float acc = 0.0f;
        for (int g = 0; g < groups; ++g) {
            float dot = 0.0f;
            for (int j = g * group; j < (g + 1) * group; ++j)
                dot += float((wr[j / per] >> ((j % per) * bits)) & mask) * x[j];
            acc += bf16(sc[size_t(r) * groups + g]) * dot + bf16(bi[size_t(r) * groups + g]) * xsum[g];
        }
        y[r] = acc;
    }
}

static void mxfp4_plain(const uint32_t* w, const uint8_t* sc, int cols, const float* x, float* y, int row0,
                        int row1) {
    const int words = cols / 8, groups = cols / 32;
    for (int r = row0; r < row1; ++r) {
        const uint32_t* wr = w + size_t(r) * words;
        float acc = 0.0f;
        for (int g = 0; g < groups; ++g) {
            float dot = 0.0f;
            for (int j = g * 32; j < (g + 1) * 32; ++j) dot += E2M1[(wr[j / 8] >> ((j % 8) * 4)) & 15u] * x[j];
            acc += e8m0(sc[size_t(r) * groups + g]) * dot;
        }
        y[r] = acc;
    }
}

static void quantize_q8(const float* x, int cols, int8_t* xq, float* xd) {
    for (int b = 0; b < cols / 32; ++b) {
        float m = 0.0f;
        for (int j = b * 32; j < (b + 1) * 32; ++j) m = std::max(m, std::fabs(x[j]));
        const float d = m / 127.0f, inv = m > 0.0f ? 127.0f / m : 0.0f;
        xd[b] = d;
        for (int j = b * 32; j < (b + 1) * 32; ++j) xq[j] = static_cast<int8_t>(std::lrintf(x[j] * inv));
    }
}

#ifdef QZ_X86
// ---------------------------------------------------------------- AVX2 + FMA: 8 values per step

__attribute__((target("avx2,fma"))) static inline float hsum8(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    return _mm_cvtss_f32(s);
}

// 8 consecutive values' words, one per lane: broadcast one word (per >= 8: bits 2, 4) or spread two (per 4: 8 bits)
__attribute__((target("avx2,fma"))) static inline __m256i lanes8(const uint32_t* p, int per, __m256i spread) {
    if (per >= 8) return _mm256_set1_epi32(int(p[0]));
    return _mm256_permutevar8x32_epi32(
        _mm256_castsi128_si256(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(p))), spread);
}

template <int bits>
__attribute__((target("avx2,fma"))) static void affine_avx2_t(const uint32_t* w, const uint16_t* sc,
                                                              const uint16_t* bi, int cols, int group, const float* x,
                                                              const float* xsum, float* y, int row0, int row1) {
    constexpr int per = 32 / bits, kinds = per >= 8 ? per / 8 : 1;
    const int words = cols / per, groups = cols / group;
    const __m256i mask = _mm256_set1_epi32((1 << bits) - 1);
    const __m256i spread = _mm256_setr_epi32(0, 0, 0, 0, 1, 1, 1, 1);
    __m256i shift[4];
    for (int k = 0; k < kinds; ++k) {
        alignas(32) int s[8];
        for (int l = 0; l < 8; ++l) s[l] = ((k * 8 + l) % per) * bits;
        shift[k] = _mm256_load_si256(reinterpret_cast<const __m256i*>(s));
    }
    for (int r = row0; r < row1; ++r) {
        const uint32_t* wr = w + size_t(r) * words;
        __m256 total = _mm256_setzero_ps();
        float bias = 0.0f;
        for (int g = 0; g < groups; ++g) {
            __m256 d[4] = {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()};
            const int j0 = g * group;
            for (int j = j0; j < j0 + group; j += 8) {
                const int k = (j >> 3) & (kinds - 1);
                __m256i q = _mm256_and_si256(_mm256_srlv_epi32(lanes8(wr + j / per, per, spread), shift[k]), mask);
                d[(j >> 3) & 3] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(q), _mm256_loadu_ps(x + j), d[(j >> 3) & 3]);
            }
            __m256 dot = _mm256_add_ps(_mm256_add_ps(d[0], d[1]), _mm256_add_ps(d[2], d[3]));
            total = _mm256_fmadd_ps(dot, _mm256_set1_ps(bf16(sc[size_t(r) * groups + g])), total);
            bias += bf16(bi[size_t(r) * groups + g]) * xsum[g];
        }
        y[r] = hsum8(total) + bias;
    }
}

static void affine_avx2(const uint32_t* w, const uint16_t* sc, const uint16_t* bi, int cols, int bits, int group,
                        const float* x, const float* xsum, float* y, int row0, int row1) {
    if (bits == 2) affine_avx2_t<2>(w, sc, bi, cols, group, x, xsum, y, row0, row1);
    else if (bits == 4) affine_avx2_t<4>(w, sc, bi, cols, group, x, xsum, y, row0, row1);
    else affine_avx2_t<8>(w, sc, bi, cols, group, x, xsum, y, row0, row1);
}

__attribute__((target("avx2,fma"))) static void mxfp4_avx2(const uint32_t* w, const uint8_t* sc, int cols,
                                                           const float* x, float* y, int row0, int row1) {
    const int words = cols / 8, groups = cols / 32;
    const __m256i shift = _mm256_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28), mask = _mm256_set1_epi32(15);
    const __m256 mag = _mm256_setr_ps(0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f);
    for (int r = row0; r < row1; ++r) {
        const uint32_t* wr = w + size_t(r) * words;
        __m256 total = _mm256_setzero_ps();
        for (int g = 0; g < groups; ++g) {
            __m256 d[4];
            for (int k = 0; k < 4; ++k) {
                const int j = g * 32 + k * 8;
                __m256i q = _mm256_and_si256(_mm256_srlv_epi32(_mm256_set1_epi32(int(wr[j / 8])), shift), mask);
                __m256 v = _mm256_permutevar8x32_ps(mag, q);
                __m256i sign = _mm256_slli_epi32(_mm256_srli_epi32(q, 3), 31);
                v = _mm256_castsi256_ps(_mm256_xor_si256(_mm256_castps_si256(v), sign));
                d[k] = _mm256_mul_ps(v, _mm256_loadu_ps(x + j));
            }
            __m256 dot = _mm256_add_ps(_mm256_add_ps(d[0], d[1]), _mm256_add_ps(d[2], d[3]));
            total = _mm256_fmadd_ps(dot, _mm256_set1_ps(e8m0(sc[size_t(r) * groups + g])), total);
        }
        y[r] = hsum8(total);
    }
}

// ---------------------------------------------------------------- AVX2 + Q8 inputs: 32 values per step

__attribute__((target("avx2,fma"))) static inline __m256i nibbles32(const uint32_t* w) {
    const __m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w));
    const __m128i m = _mm_set1_epi8(15);
    const __m128i lo = _mm_and_si128(b, m), hi = _mm_and_si128(_mm_srli_epi16(b, 4), m);
    return _mm256_set_m128i(_mm_unpackhi_epi8(lo, hi), _mm_unpacklo_epi8(lo, hi));
}

__attribute__((target("avx2,fma"))) static inline __m256 dot32(__m256i wu8, __m256i xs8) {
    const __m256i p16 = _mm256_maddubs_epi16(wu8, xs8);
    return _mm256_cvtepi32_ps(_mm256_madd_epi16(p16, _mm256_set1_epi16(1)));
}

__attribute__((target("avx2,fma"))) static void mxfp4_q8(const uint32_t* w, const uint8_t* sc, int cols,
                                                         const int8_t* xq, const float* xd, float* y, int row0,
                                                         int row1) {
    const int words = cols / 8, groups = cols / 32;
    const __m256i lut = _mm256_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12,
                                         0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12);
    for (int r = row0; r < row1; ++r) {
        const uint32_t* wr = w + size_t(r) * words;
        __m256 total = _mm256_setzero_ps();
        for (int g = 0; g < groups; ++g) {
            const __m256i v = _mm256_shuffle_epi8(lut, nibbles32(wr + g * 4));
            const __m256i x8 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(xq + g * 32));
            const __m256 dot = dot32(_mm256_sign_epi8(v, v), _mm256_sign_epi8(x8, v));
            total = _mm256_fmadd_ps(dot, _mm256_set1_ps(0.5f * e8m0(sc[size_t(r) * groups + g]) * xd[g]), total);
        }
        y[r] = hsum8(total);
    }
}

__attribute__((target("avx2,fma"))) static void affine4_q8(const uint32_t* w, const uint16_t* sc, const uint16_t* bi,
                                                           int cols, int group, const int8_t* xq, const float* xd,
                                                           const float* xsum, float* y, int row0, int row1) {
    const int words = cols / 8, groups = cols / group, per = group / 32;
    for (int r = row0; r < row1; ++r) {
        const uint32_t* wr = w + size_t(r) * words;
        __m256 total = _mm256_setzero_ps();
        float bias = 0.0f;
        for (int g = 0; g < groups; ++g) {
            const float s = bf16(sc[size_t(r) * groups + g]);
            for (int k = 0; k < per; ++k) {
                const int b = g * per + k;
                const __m256i x8 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(xq + b * 32));
                total = _mm256_fmadd_ps(dot32(nibbles32(wr + b * 4), x8), _mm256_set1_ps(s * xd[b]), total);
            }
            bias += bf16(bi[size_t(r) * groups + g]) * xsum[g];
        }
        y[r] = hsum8(total) + bias;
    }
}

// ---------------------------------------------------------------- AVX-512: 16 values per step

__attribute__((target("avx512f"))) static inline __m512i lanes16(const uint32_t* p, int per, __m512i spread) {
    __m128i v = per >= 16 ? _mm_cvtsi32_si128(int(p[0]))
                          : per == 8 ? _mm_loadl_epi64(reinterpret_cast<const __m128i*>(p))
                                     : _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
    return _mm512_permutexvar_epi32(spread, _mm512_castsi128_si512(v));
}

template <int bits>
__attribute__((target("avx512f"))) static void affine_avx512_t(const uint32_t* w, const uint16_t* sc,
                                                               const uint16_t* bi, int cols, int group, const float* x,
                                                               const float* xsum, float* y, int row0, int row1) {
    constexpr int per = 32 / bits, kinds = per >= 16 ? per / 16 : 1;
    const int words = cols / per, groups = cols / group;
    const __m512i mask = _mm512_set1_epi32((1 << bits) - 1);
    alignas(64) int idx[16];
    for (int l = 0; l < 16; ++l) idx[l] = per >= 16 ? 0 : l / per;
    const __m512i spread = _mm512_load_si512(idx);
    __m512i shift[2];
    for (int k = 0; k < kinds; ++k) {
        alignas(64) int s[16];
        for (int l = 0; l < 16; ++l) s[l] = ((k * 16 + l) % per) * bits;
        shift[k] = _mm512_load_si512(s);
    }
    for (int r = row0; r < row1; ++r) {
        const uint32_t* wr = w + size_t(r) * words;
        __m512 total = _mm512_setzero_ps();
        float bias = 0.0f;
        for (int g = 0; g < groups; ++g) {
            __m512 d[2] = {_mm512_setzero_ps(), _mm512_setzero_ps()};
            const int j0 = g * group;
            for (int j = j0; j < j0 + group; j += 16) {
                const int k = (j >> 4) & (kinds - 1);
                __m512i q = _mm512_and_si512(_mm512_srlv_epi32(lanes16(wr + j / per, per, spread), shift[k]), mask);
                d[(j >> 4) & 1] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(q), _mm512_loadu_ps(x + j), d[(j >> 4) & 1]);
            }
            total = _mm512_fmadd_ps(_mm512_add_ps(d[0], d[1]), _mm512_set1_ps(bf16(sc[size_t(r) * groups + g])),
                                    total);
            bias += bf16(bi[size_t(r) * groups + g]) * xsum[g];
        }
        y[r] = _mm512_reduce_add_ps(total) + bias;
    }
}

static void affine_avx512(const uint32_t* w, const uint16_t* sc, const uint16_t* bi, int cols, int bits, int group,
                          const float* x, const float* xsum, float* y, int row0, int row1) {
    if (bits == 2) affine_avx512_t<2>(w, sc, bi, cols, group, x, xsum, y, row0, row1);
    else if (bits == 4) affine_avx512_t<4>(w, sc, bi, cols, group, x, xsum, y, row0, row1);
    else affine_avx512_t<8>(w, sc, bi, cols, group, x, xsum, y, row0, row1);
}

__attribute__((target("avx512f"))) static void mxfp4_avx512(const uint32_t* w, const uint8_t* sc, int cols,
                                                            const float* x, float* y, int row0, int row1) {
    const int words = cols / 8, groups = cols / 32;
    const __m512i shift = _mm512_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28, 0, 4, 8, 12, 16, 20, 24, 28);
    const __m512i mask = _mm512_set1_epi32(15);
    const __m512 lut = _mm512_loadu_ps(E2M1);
    const __m512i two = _mm512_setr_epi32(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1);
    for (int r = row0; r < row1; ++r) {
        const uint32_t* wr = w + size_t(r) * words;
        __m512 total = _mm512_setzero_ps();
        for (int g = 0; g < groups; ++g) {
            __m512 d[2];
            for (int k = 0; k < 2; ++k) {
                const int j = g * 32 + k * 16;
                __m512i pair = _mm512_permutexvar_epi32(
                    two, _mm512_castsi128_si512(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(wr + j / 8))));
                __m512i q = _mm512_and_si512(_mm512_srlv_epi32(pair, shift), mask);
                d[k] = _mm512_mul_ps(_mm512_permutexvar_ps(q, lut), _mm512_loadu_ps(x + j));
            }
            total = _mm512_fmadd_ps(_mm512_add_ps(d[0], d[1]), _mm512_set1_ps(e8m0(sc[size_t(r) * groups + g])),
                                    total);
        }
        y[r] = _mm512_reduce_add_ps(total);
    }
}
#endif

// ---------------------------------------------------------------- C ABI entrypoints

API int qz_isa() {
    // 2: AVX-512, 1: AVX2 with FMA, 0: plain C++
#ifdef QZ_X86
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx512f")) return 2;
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) return 1;
#endif
    return 0;
}

API int qz_affine(const uint32_t* w, const uint16_t* sc, const uint16_t* bi, int cols, int bits, int group,
                  const float* x, const float* xsum, float* y, int row0, int row1, int isa) {
    // 0: done; -1: unsupported shape (bits not 2/4/8 or group not dividing cols)
    if (bits != 2 && bits != 4 && bits != 8) return -1;
    if (group <= 0 || cols % group) return -1;
#ifdef QZ_X86
    if (isa >= 2 && group % 16 == 0) {
        affine_avx512(w, sc, bi, cols, bits, group, x, xsum, y, row0, row1);
        return 0;
    }
    if (isa >= 1 && group % 8 == 0) {
        affine_avx2(w, sc, bi, cols, bits, group, x, xsum, y, row0, row1);
        return 0;
    }
#endif
    affine_plain(w, sc, bi, cols, bits, group, x, xsum, y, row0, row1);
    return 0;
}

API int qz_mxfp4(const uint32_t* w, const uint8_t* sc, int cols, const float* x, float* y, int row0, int row1,
                 int isa) {
    if (cols % 32) return -1;
#ifdef QZ_X86
    if (isa >= 2) {
        mxfp4_avx512(w, sc, cols, x, y, row0, row1);
        return 0;
    }
    if (isa >= 1) {
        mxfp4_avx2(w, sc, cols, x, y, row0, row1);
        return 0;
    }
#endif
    mxfp4_plain(w, sc, cols, x, y, row0, row1);
    return 0;
}

// ---------------------------------------------------------------- lock-free worker pool

#include <atomic>
#include <chrono>
#include <memory>
#include <thread>
#include <vector>
#include "atomic_smart_ptr.h"
#include "xwaitcell.h"
#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#include <sys/prctl.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

struct Proj {
    int kind;                 // 0: affine, 1: mxfp4
    const void* w;
    const void* sc;
    const void* bi;
    int rows, cols, bits, group;
};

static bool q8_takes(const Proj& p, int isa) {
#ifdef QZ_X86
    if (isa < 1) return false;
    if (p.kind == 1) return p.cols % 32 == 0;
    return p.bits == 4 && p.group % 32 == 0 && p.cols % p.group == 0;
#else
    (void)p;
    (void)isa;
    return false;
#endif
}

static int project_q8(const Proj& p, const int8_t* xq, const float* xd, const float* xsum, float* y, int r0,
                      int r1) {
#ifdef QZ_X86
    if (p.kind == 1)
        mxfp4_q8(static_cast<const uint32_t*>(p.w), static_cast<const uint8_t*>(p.sc), p.cols, xq, xd, y, r0, r1);
    else
        affine4_q8(static_cast<const uint32_t*>(p.w), static_cast<const uint16_t*>(p.sc),
                   static_cast<const uint16_t*>(p.bi), p.cols, p.group, xq, xd, xsum, y, r0, r1);
    return 0;
#else
    (void)p;
    (void)xq;
    (void)xd;
    (void)xsum;
    (void)y;
    (void)r0;
    (void)r1;
    return -1;
#endif
}

static int project(const Proj& p, const float* x, const float* xsum, float* y, int r0, int r1, int isa) {
    if (p.kind == 1)
        return qz_mxfp4(static_cast<const uint32_t*>(p.w), static_cast<const uint8_t*>(p.sc), p.cols, x, y, r0, r1,
                        isa);
    return qz_affine(static_cast<const uint32_t*>(p.w), static_cast<const uint16_t*>(p.sc),
                     static_cast<const uint16_t*>(p.bi), p.cols, p.bits, p.group, x, xsum, y, r0, r1, isa);
}

// Between projections: SwiGLU (gate <= limit, |up| <= limit when limit > 0) * router weight.
// ACTIVATION_PLAIN operates in float32 throughout; ACTIVATION_DEEPSEEK_V41 rounds gate, up, and the weighted product
// via bf16 and block-32 E4M3/UE8M0 FP8 roundtrip (matching DeepseekV41SwiGLU on GPU), with bf16 output rounding.
enum { ACTIVATION_PLAIN = 0, ACTIVATION_DEEPSEEK_V41 = 1 };

static inline float as_float(uint32_t bits) {
    float out;
    std::memcpy(&out, &bits, 4);
    return out;
}

static inline uint32_t bits_of(float v) {
    uint32_t out;
    std::memcpy(&out, &v, 4);
    return out;
}

static inline float to_bf16(float v) {
    const uint32_t b = bits_of(v);
    return as_float(((b + 0x7fffu + ((b >> 16) & 1u)) >> 16) << 16);
}

static void fp8_roundtrip(float* v, int n) {
    for (int b0 = 0; b0 < n; b0 += 32) {
        const int b1 = std::min(n, b0 + 32);
        float amax = 0.0f;
        for (int i = b0; i < b1; ++i) amax = std::max(amax, std::fabs(v[i]));
        amax = std::max(amax, 1e-4f);
        const uint32_t sb = bits_of(amax * (1.0f / 448.0f));
        const float scale = as_float(((sb >> 23) + ((sb & 0x7fffffu) != 0)) << 23);
        for (int i = b0; i < b1; ++i) {
            float q = std::min(448.0f, std::max(-448.0f, v[i] / scale));
            const int e = int(bits_of(std::fabs(q)) >> 23) - 3;
            const float step = as_float(uint32_t(std::max(e, 118)) << 23);
            v[i] = std::nearbyint(q / step) * step * scale;
        }
    }
}

static void activate(const float* g, const float* u, float* h, int n, float limit, float weight, int style) {
    for (int i = 0; i < n; ++i) {
        float a = g[i], b = u[i];
        if (style == ACTIVATION_DEEPSEEK_V41) {
            a = to_bf16(a);
            b = to_bf16(b);
        }
        if (limit > 0) {
            a = a < limit ? a : limit;
            b = b < -limit ? -limit : (b > limit ? limit : b);
        }
        h[i] = a / (1.0f + std::exp(-a)) * b * weight;
        if (style == ACTIVATION_DEEPSEEK_V41) h[i] = to_bf16(h[i]);
    }
    if (style == ACTIVATION_DEEPSEEK_V41) fp8_roundtrip(h, n);
}

static void finish(float* out, int n, int style) {
    if (style == ACTIVATION_DEEPSEEK_V41)
        for (int i = 0; i < n; ++i) out[i] = to_bf16(out[i]);
}

API void qz_activate(const float* g, const float* u, float* h, int n, float limit, float weight, int style) {
    activate(g, u, h, n, limit, weight, style);
}

// Single-threaded BLAS-free float64 reference used for runtime sanity checks (avoids OpenBLAS spinning worker threads).
static float bf16_value(uint16_t v) {
    uint32_t bits = uint32_t(v) << 16;
    float out;
    std::memcpy(&out, &bits, 4);
    return out;
}

static void reference_rows(const Proj& p, const float* x, float* y) {
    static const double E2M1_VALUES[16] = {0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6};
    std::vector<double> xd(p.cols);
    for (int j = 0; j < p.cols; ++j) xd[j] = double(x[j]);
    const uint32_t* w = static_cast<const uint32_t*>(p.w);
    const int bits = p.kind == 1 ? 4 : p.bits;
    const int per = 32 / bits;
    const int words = p.cols / per;
    for (int r = 0; r < p.rows; ++r) {
        double acc[4] = {0.0, 0.0, 0.0, 0.0};
        for (int j = 0; j < p.cols; ++j) {
            const uint32_t q = (w[size_t(r) * words + j / per] >> ((j % per) * bits)) & ((1u << bits) - 1u);
            double wval;
            if (p.kind == 1) {
                const int e = static_cast<const uint8_t*>(p.sc)[size_t(r) * (p.cols / 32) + j / 32];
                wval = E2M1_VALUES[q] * std::ldexp(1.0, e - 127);
            } else {
                const size_t g = size_t(r) * (p.cols / p.group) + j / p.group;
                wval = double(q) * bf16_value(static_cast<const uint16_t*>(p.sc)[g]) +
                       bf16_value(static_cast<const uint16_t*>(p.bi)[g]);
            }
            acc[j & 3] += wval * xd[j];
        }
        y[r] = float((acc[0] + acc[1]) + (acc[2] + acc[3]));
    }
}

API void qz_reference(const Proj* p, const float* x, float* y) { reference_rows(*p, x, y); }

static void as_q8(const float* x, int cols, std::vector<float>& out) {
    std::vector<int8_t> q(cols);
    std::vector<float> d(cols / 32);
    quantize_q8(x, cols, q.data(), d.data());
    out.resize(cols);
    for (int j = 0; j < cols; ++j) out[j] = d[j / 32] * static_cast<float>(q[j]);
}

static void bias_back(const Proj& p, const float* x, const float* xhat, float* y) {
    if (p.kind == 1) return;
    const int groups = p.cols / p.group;
    const uint16_t* bi = static_cast<const uint16_t*>(p.bi);
    for (int g = 0; g < groups; ++g) {
        double d = 0.0;
        for (int j = g * p.group; j < (g + 1) * p.group; ++j) d += double(x[j]) - double(xhat[j]);
        for (int r = 0; r < p.rows; ++r) y[r] += static_cast<float>(double(bf16_value(bi[size_t(r) * groups + g])) * d);
    }
}

API void qz_expert_reference(const Proj* gate, const Proj* up, const Proj* down, const float* x, float* out,
                             float limit, float weight, int style, int q8) {
    std::vector<float> g(gate->rows), u(up->rows), h(gate->rows), xin, hin;
    const float* a = x;
    if (q8 && q8_takes(*gate, 1) && q8_takes(*up, 1)) {
        as_q8(x, gate->cols, xin);
        a = xin.data();
    }
    reference_rows(*gate, a, g.data());
    reference_rows(*up, a, u.data());
    if (a != x) {
        bias_back(*gate, x, a, g.data());
        bias_back(*up, x, a, u.data());
    }
    activate(g.data(), u.data(), h.data(), gate->rows, limit, weight, style);
    const float* b = h.data();
    if (q8 && q8_takes(*down, 1)) {
        as_q8(h.data(), down->cols, hin);
        b = hin.data();
    }
    reference_rows(*down, b, out);
    if (b != h.data()) bias_back(*down, h.data(), b, out);
    finish(out, down->rows, style);
}

static const int CHUNK_ROWS = 32;

static inline int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Monotonic telemetry counters and test stall injection scalars; KAME pointer/wait-cell primitives
// coordinate phase publication and wakeup, whereas these require lock-free arithmetic/bitmask updates.
struct Counts {
    std::atomic<int64_t> rung{0}, woke{0}, wake_ns{0}, wake_ns_max{0}, by_workers{0}, by_caller{0}, stolen{0};
    std::atomic<int64_t> workers_ns{0}, caller_ns{0};
    std::atomic<int64_t> stall{0};
    std::atomic<int64_t> stall_before{0};
    std::atomic<uint64_t> caller_cpus{0}, worker_cpus{0};
    std::atomic<int64_t> moved{0};
    std::atomic<int> computing{0};
};

// A chunk transitions FREE -> TAKEN -> DONE; only the first winner copies out of thread-local scratch.
// If a low-priority worker is preempted (either between next.fetch_add and TAKEN or while computing a chunk),
// the caller steals and recomputes the chunk after STEAL_NS nanoseconds so latency never stalls on an idle worker.
enum { FREE = 0, TAKEN = 1, DONE = 2 };
static const int64_t STEAL_NS = 200000;

struct Phase : atomic_countable {
    Proj proj[2];
    const Proj* p[2];
    std::vector<float> x_own, xsum_own[2];
    std::vector<int8_t> xq_own;
    std::vector<float> xd_own;
    bool q8 = false;
    const float* x;
    const float* xsum[2];
    float* y[2];
    int isa;
    int parts;
    int chunks[2];
    int total;
    int64_t rung_at = 0;
    // Per-chunk CAS states and timestamps plus fetch_add index/done counters have no KAME multi-slot equivalent;
    // thread sleep/wakeup on phase completion uses KAME XWaitCell (`done`).
    std::unique_ptr<std::atomic<int>[]> state;
    std::unique_ptr<std::atomic<int64_t>[]> taken_at;
    int capacity = 0;
    std::atomic<int> next{0}, failed{0}, finished{0};
    XWaitCell done;
    Counts* counts = nullptr;

    void compute(int c, std::vector<float>& scratch) {
        const int k = c < chunks[0] ? 0 : 1;
        const int local = k == 0 ? c : c - chunks[0];
        const Proj& q = *p[k];
        const int r0 = local * CHUNK_ROWS, r1 = std::min(q.rows, r0 + CHUNK_ROWS);
        if (int(scratch.size()) < q.rows) scratch.resize(q.rows);
        const int got = q8 ? project_q8(q, xq_own.data(), xd_own.data(), xsum[k], scratch.data(), r0, r1)
                           : project(q, x, xsum[k], scratch.data(), r0, r1, isa);
        if (got != 0) failed = 1;
        int taken = TAKEN;
        if (!state[c].compare_exchange_strong(taken, DONE, std::memory_order_acq_rel)) return;
        std::copy(scratch.data() + r0, scratch.data() + r1, y[k] + r0);
        if (finished.fetch_add(1, std::memory_order_acq_rel) + 1 == total) done.wake_one();
    }

    bool step(std::vector<float>& scratch, std::atomic<int64_t>* stall = nullptr,
              std::atomic<int64_t>* stall_before = nullptr) {
        const int c = next.fetch_add(1, std::memory_order_relaxed);
        if (c >= total) return false;
        if (stall_before && stall_before->load(std::memory_order_relaxed) > 0) {
            const int64_t ns = stall_before->exchange(0);
            if (ns > 0) std::this_thread::sleep_for(std::chrono::nanoseconds(ns));
        }
        int free_ = FREE;
        if (!state[c].compare_exchange_strong(free_, TAKEN, std::memory_order_acq_rel)) return true;
        taken_at[c].store(now_ns(), std::memory_order_relaxed);
        if (stall && stall->load(std::memory_order_relaxed) > 0) {
            const int64_t ns = stall->exchange(0);
            if (ns > 0) std::this_thread::sleep_for(std::chrono::nanoseconds(ns));
        }
        compute(c, scratch);
        return true;
    }

    int steal(std::vector<float>& scratch) {
        int again = 0;
        const int64_t now = now_ns();
        const int taken = std::min(next.load(std::memory_order_acquire), total);
        for (int c = 0; c < taken; ++c) {
            const int st = state[c].load(std::memory_order_acquire);
            if (st == FREE) {
                if (now - rung_at < STEAL_NS) continue;
                int free_ = FREE;
                if (!state[c].compare_exchange_strong(free_, TAKEN, std::memory_order_acq_rel)) continue;
                taken_at[c].store(now, std::memory_order_relaxed);
            } else if (st == TAKEN) {
                const int64_t at = taken_at[c].load(std::memory_order_relaxed);
                if (at == 0 || now - at < STEAL_NS) continue;
            } else {
                continue;
            }
            compute(c, scratch);
            ++again;
        }
        return again;
    }
};

static thread_local std::vector<float> chunk_scratch;

static inline int current_cpu() {
#if defined(__linux__)
    return sched_getcpu();
#else
    return -1;
#endif
}

static inline void seen_on(std::atomic<uint64_t>& mask, int cpu) {
    if (cpu >= 0 && cpu < 64) mask.fetch_or(uint64_t(1) << cpu, std::memory_order_relaxed);
}

struct Bell {
    alignas(64) XWaitCell rung;
    atomic_shared_ptr<Phase> slot;
};

struct Pool {
    std::vector<std::thread> threads;
    std::unique_ptr<Bell[]> bells;
    std::vector<int> cpus;
    // Non-blocking shutdown flag checked in spin loops alongside KAME XWaitCell bell wakeups.
    std::atomic<int> stop{0};
    bool q8 = false;
    // Non-blocking single-caller try-claim: returns -3 immediately when another caller is active
    // instead of blocking on a mutex (KAME primitives do not model non-blocking try-claim return).
    std::atomic<int> calling{0};
    std::vector<local_shared_ptr<Phase>> spare;

    local_shared_ptr<Phase> phase() {
        for (auto& p : spare)
            if (p.unique()) return p;
        spare.emplace_back(make_local_shared<Phase>());
        return spare.back();
    }

    int priority = 0;         // 0: SCHED_IDLE (else nice 19), 1: nice 19, 2: normal, 3: wake normal -> SCHED_IDLE
    int spin_us = 0;
    Counts counts;

    void work(int index) {
#if defined(__linux__)
        if (index < int(cpus.size()) && cpus[index] >= 0) {
            cpu_set_t set;
            CPU_ZERO(&set);
            CPU_SET(cpus[index], &set);
            pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
        }
        sched_param sp{};
        if (priority == 0 && pthread_setschedparam(pthread_self(), SCHED_IDLE, &sp) == 0) {
        } else if (priority <= 1) {
            setpriority(PRIO_PROCESS, static_cast<id_t>(syscall(SYS_gettid)), 19);
        }
#endif
        auto idle = [&](bool on) {
#if defined(__linux__)
            if (priority != 3) return;
            sched_param none{};
            pthread_setschedparam(pthread_self(), on ? SCHED_IDLE : SCHED_OTHER, &none);
#else
            (void)on;
#endif
        };
        XWaitCell& bell = bells[index].rung;
        uint32_t seen = bell.gen();
        for (;;) {
            const auto until = std::chrono::steady_clock::now() + std::chrono::microseconds(spin_us);
            while (spin_us > 0 && bell.gen() == seen && !stop.load() && std::chrono::steady_clock::now() < until)
                pause4spin();
            if (bell.gen() == seen) idle(false);
            while (bell.gen() == seen && !stop.load(std::memory_order_acquire)) bell.wait(seen, 1000000u);
            if (stop.load(std::memory_order_acquire)) return;
            idle(true);
            seen = bell.gen();
            local_shared_ptr<Phase> mine(bells[index].slot);
            if (mine) {
                bool first = true;
                for (;;) {
                    counts.computing.fetch_add(1, std::memory_order_acq_rel);
                    const int64_t t0 = now_ns();
                    const bool took = mine->step(chunk_scratch, &counts.stall, &counts.stall_before);
                    if (took) {
                        counts.workers_ns.fetch_add(now_ns() - t0, std::memory_order_relaxed);
                        counts.by_workers.fetch_add(1, std::memory_order_relaxed);
                        const int cpu = current_cpu();
                        seen_on(counts.worker_cpus, cpu);
                        if (index < int(cpus.size()) && cpus[index] >= 0 && cpu >= 0 && cpu != cpus[index])
                            counts.moved.fetch_add(1, std::memory_order_relaxed);
                        if (first) {
                            const int64_t late = t0 - mine->rung_at;
                            counts.woke.fetch_add(1, std::memory_order_relaxed);
                            counts.wake_ns.fetch_add(late, std::memory_order_relaxed);
                            int64_t top = counts.wake_ns_max.load(std::memory_order_relaxed);
                            while (late > top && !counts.wake_ns_max.compare_exchange_weak(top, late)) {
                            }
                            first = false;
                        }
                    }
                    counts.computing.fetch_sub(1, std::memory_order_acq_rel);
                    if (!took) break;
                }
            }
        }
    }

    int run(const local_shared_ptr<Phase>& p) {
        p->counts = &counts;
        p->rung_at = now_ns();
        for (int i = 0; i < p->parts; ++i) {
            bells[i].slot = p;
            bells[i].rung.wake_one();
        }
        counts.rung.fetch_add(p->parts, std::memory_order_relaxed);
        for (;;) {
            const int64_t t0 = now_ns();
            if (!p->step(chunk_scratch)) break;
            counts.caller_ns.fetch_add(now_ns() - t0, std::memory_order_relaxed);
            counts.by_caller.fetch_add(1, std::memory_order_relaxed);
            seen_on(counts.caller_cpus, current_cpu());
        }
        for (;;) {
            const uint32_t g = p->done.gen();
            if (p->finished.load(std::memory_order_acquire) == p->total) break;
            counts.stolen.fetch_add(p->steal(chunk_scratch), std::memory_order_relaxed);
            if (p->finished.load(std::memory_order_acquire) == p->total) break;
            p->done.wait(g, unsigned(STEAL_NS / 2000));
        }
        for (int i = 0; i < p->parts; ++i) bells[i].slot = local_shared_ptr<Phase>();
        return p->failed.load();
    }
};

static local_shared_ptr<Phase> make_phase(Pool* pool, const Proj* a, const Proj* b, const float* x,
                                          const float* xa, const float* xb, float* ya, float* yb, int isa, int parts) {
    local_shared_ptr<Phase> p = pool->phase();
    p->proj[0] = *a;
    p->p[0] = &p->proj[0];
    p->p[1] = nullptr;
    if (b) {
        p->proj[1] = *b;
        p->p[1] = &p->proj[1];
    }
    p->x_own.assign(x, x + a->cols);
    p->x = p->x_own.data();
    p->q8 = pool->q8 && q8_takes(*a, isa) && (!b || q8_takes(*b, isa));
    if (p->q8) {
        p->xq_own.resize(a->cols);
        p->xd_own.resize(a->cols / 32);
        quantize_q8(p->x, a->cols, p->xq_own.data(), p->xd_own.data());
    }
    const float* given[2] = {xa, xb};
    for (int k = 0; k < 2; ++k) {
        p->xsum[k] = nullptr;
        if (given[k] && p->p[k] && p->p[k]->kind == 0) {
            p->xsum_own[k].assign(given[k], given[k] + p->p[k]->cols / p->p[k]->group);
            p->xsum[k] = p->xsum_own[k].data();
        }
    }
    p->y[0] = ya;
    p->y[1] = yb;
    p->isa = isa;
    p->parts = parts;
    p->chunks[0] = (a->rows + CHUNK_ROWS - 1) / CHUNK_ROWS;
    p->chunks[1] = b ? (b->rows + CHUNK_ROWS - 1) / CHUNK_ROWS : 0;
    p->total = p->chunks[0] + p->chunks[1];
    if (p->total > p->capacity) {
        p->state = std::make_unique<std::atomic<int>[]>(p->total);
        p->taken_at = std::make_unique<std::atomic<int64_t>[]>(p->total);
        p->capacity = p->total;
    }
    p->next.store(0, std::memory_order_relaxed);
    p->failed.store(0, std::memory_order_relaxed);
    p->finished.store(0, std::memory_order_relaxed);
    p->counts = nullptr;
    for (int c = 0; c < p->total; ++c) {
        p->state[c].store(FREE, std::memory_order_relaxed);
        p->taken_at[c].store(0, std::memory_order_relaxed);
    }
    return p;
}

API int qz_pool_busy(void* handle) {
    return static_cast<Pool*>(handle)->counts.computing.load();
}

API void qz_pool_stall(void* handle, int64_t ns) {
    static_cast<Pool*>(handle)->counts.stall.store(ns);
}

API void qz_pool_q8(void* handle, int on) {
    static_cast<Pool*>(handle)->q8 = on != 0;
}

API void qz_pool_stall_before(void* handle, int64_t ns) {
    static_cast<Pool*>(handle)->counts.stall_before.store(ns);
}

API void qz_pool_stats(void* handle, int64_t* out) {
    const Counts& c = static_cast<Pool*>(handle)->counts;
    out[0] = c.rung.load();
    out[1] = c.woke.load();
    out[2] = c.wake_ns.load();
    out[3] = c.wake_ns_max.load();
    out[4] = c.by_workers.load();
    out[5] = c.by_caller.load();
    out[6] = c.stolen.load();
    out[7] = c.workers_ns.load();
    out[8] = c.caller_ns.load();
    out[9] = int64_t(c.caller_cpus.load());
    out[10] = int64_t(c.worker_cpus.load());
    out[11] = c.moved.load();
}

static void sums(const Proj& p, const float* x, std::vector<float>& out) {
    if (p.kind == 1) return;
    const int groups = p.cols / p.group;
    out.assign(groups, 0.0f);
    for (int g = 0; g < groups; ++g)
        for (int j = g * p.group; j < (g + 1) * p.group; ++j) out[g] += x[j];
}

API void* qz_pool_new(int threads, int priority, int spin_us, const int* cpus, int ncpus) {
    auto owned = std::make_unique<Pool>();
    Pool* pool = owned.get();
    pool->priority = priority;
    pool->spin_us = spin_us;
    pool->bells = std::make_unique<Bell[]>(threads > 0 ? threads : 1);
    for (int i = 0; i < threads; ++i) pool->cpus.push_back(ncpus > 0 ? cpus[i % ncpus] : -1);
    for (int i = 0; i < threads; ++i) pool->threads.emplace_back([pool, i] { pool->work(i); });
    return owned.release();
}

API void qz_pool_free(void* handle) {
    std::unique_ptr<Pool> pool(static_cast<Pool*>(handle));
    pool->stop.store(1, std::memory_order_release);
    for (size_t i = 0; i < pool->threads.size(); ++i) pool->bells[i].rung.wake_one();
    for (auto& t : pool->threads) t.join();
}

API int qz_expert(void* handle, int active, const Proj* gate, const Proj* up, const Proj* down, const float* x,
                  float* out, float* hidden, float limit, float weight, int style, int isa) {
    Pool* pool = static_cast<Pool*>(handle);
    const int parts = active < int(pool->threads.size()) ? active : int(pool->threads.size());
    if (parts <= 0) return -2;
    if (style == ACTIVATION_DEEPSEEK_V41 && gate->rows % 32 != 0) return -1;
    if (pool->calling.exchange(1, std::memory_order_acquire)) return -3;
    struct Release {
        Pool* p;
        ~Release() { p->calling.store(0, std::memory_order_release); }
    } release{pool};
    const int inner = gate->rows;
    static thread_local std::vector<float> g, u, xs[3];
    if (int(g.size()) < inner) {
        g.resize(inner);
        u.resize(inner);
    }
    sums(*gate, x, xs[0]);
    sums(*up, x, xs[1]);
    if (pool->run(make_phase(pool, gate, up, x, xs[0].data(), xs[1].data(), g.data(), u.data(), isa, parts)))
        return -1;
    activate(g.data(), u.data(), hidden, inner, limit, weight, style);
    sums(*down, hidden, xs[2]);
    if (pool->run(make_phase(pool, down, nullptr, hidden, xs[2].data(), nullptr, out, nullptr, isa, parts))) return -1;
    finish(out, down->rows, style);
    return 0;
}

// ---------------------------------------------------------------- zero-GIL CUDA handshake server

// Request row layout (int32 words): [stamp, n, slots[MMAX] (-1: none), weights[MMAX] (float), xrow[MMAX], B],
// then x [B, D] float at HEAD.
// Answer row layout (int32 words): [stamp, n, done[MMAX] (1: computed)], then outputs [MMAX, D] float at HEAD.
static const int SERVE_MMAX = 16;
static const int SERVE_BMAX = 8;
static const int SERVE_HEAD = 64;

struct ServeLayer {
    // Release-acquire publication flag read unlocked by the polling server loop once layer descriptors are written.
    std::atomic<bool> on{false};
    const uint8_t* base[3][3] = {};
    int64_t rowb[3][3] = {};
    Proj like[3];
    float limit = 0.0f;
    int style = ACTIVATION_PLAIN;
};

struct Server {
    Pool* pool = nullptr;
    int isa = 0, D = 0;
    int64_t box_words = 0, ans_words = 0;
    int32_t* box = nullptr;
    int32_t* ans = nullptr;
    std::unique_ptr<ServeLayer[]> layers;
    int n_layers = 0;
    std::vector<int64_t> done;
    std::vector<int> cpus;
    // Non-blocking active count, shutdown flag, and scalar telemetry for the pinned-mailbox polling loop
    // (the polling loop cannot block on KAME XWaitCell while scanning GPU mailboxes across layers).
    std::atomic<int> active{1};
    std::atomic<int> stop{0};
    std::atomic<int64_t> requests{0}, experts{0}, failed{0}, compute_ns{0}, serving{0};
    int spin_us = 100;
    std::thread thread;

    void answer(int i) {
        ServeLayer& L = layers[i];
        volatile int32_t* b = box + int64_t(i) * box_words;
        volatile int32_t* a = ans + int64_t(i) * ans_words;
        const int32_t stamp = b[0];
        std::atomic_thread_fence(std::memory_order_acquire);
        int n = b[1];
        n = n < 0 ? 0 : (n > SERVE_MMAX ? SERVE_MMAX : n);
        const float* x = reinterpret_cast<const float*>(const_cast<int32_t*>(b) + SERVE_HEAD);
        float* out = reinterpret_cast<float*>(const_cast<int32_t*>(a) + SERVE_HEAD);
        const int64_t t0 = now_ns();
        for (int m = 0; m < n; ++m) {
            const int slot = b[2 + m];
            int ok = 0;
            if (slot >= 0) {
                int32_t wbits = b[2 + SERVE_MMAX + m];
                float weight;
                std::memcpy(&weight, &wbits, 4);
                int xr = b[2 + 2 * SERVE_MMAX + m];
                if (xr < 0 || xr >= SERVE_BMAX) xr = 0;
                Proj p[3];
                for (int q = 0; q < 3; ++q) {
                    p[q] = L.like[q];
                    p[q].w = L.base[q][0] + slot * L.rowb[q][0];
                    p[q].sc = L.base[q][1] + slot * L.rowb[q][1];
                    p[q].bi = L.base[q][2] ? L.base[q][2] + slot * L.rowb[q][2] : nullptr;
                }
                static thread_local std::vector<float> hidden;
                if (int(hidden.size()) < p[0].rows) hidden.resize(p[0].rows);
                const int act = active.load(std::memory_order_relaxed);
                const int r = qz_expert(pool, act > 0 ? act : 1, &p[0], &p[1], &p[2], x + int64_t(xr) * D,
                                        out + int64_t(m) * D, hidden.data(), L.limit, weight, L.style, isa);
                ok = r == 0;
                if (ok) experts.fetch_add(1, std::memory_order_relaxed);
                else failed.fetch_add(1, std::memory_order_relaxed);
            }
            a[2 + m] = ok;
        }
        a[1] = n;
        compute_ns.fetch_add(now_ns() - t0, std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_release);
        a[0] = stamp;
        done[i] = stamp;
        requests.fetch_add(1, std::memory_order_relaxed);
    }

    void loop() {
#if defined(__linux__)
        prctl(PR_SET_TIMERSLACK, 1000UL, 0, 0, 0);
        if (!cpus.empty()) {
            cpu_set_t set;
            CPU_ZERO(&set);
            for (int c : cpus) CPU_SET(c, &set);
            pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
        }
#endif
        int64_t last = now_ns();
        while (!stop.load(std::memory_order_acquire)) {
            bool any = false;
            for (int i = 0; i < n_layers; ++i) {
                if (!layers[i].on.load(std::memory_order_acquire)) continue;
                const int32_t s = reinterpret_cast<volatile int32_t*>(box)[int64_t(i) * box_words];
                if (s > done[i]) {
                    serving.store(1, std::memory_order_release);
                    answer(i);
                    serving.store(0, std::memory_order_release);
                    any = true;
                }
            }
            if (any) {
                last = now_ns();
            } else if (now_ns() - last < int64_t(spin_us) * 1000) {
                pause4spin();
            } else {
                std::this_thread::sleep_for(std::chrono::microseconds(10));
            }
        }
    }
};

API void* qz_server_new(void* pool, int isa, int D, int n_layers, int32_t* box, int64_t box_words, int32_t* ans,
                        int64_t ans_words, int spin_us) {
    auto owned = std::make_unique<Server>();
    Server* s = owned.get();
    s->pool = static_cast<Pool*>(pool);
    s->isa = isa;
    s->D = D;
    s->box = box;
    s->ans = ans;
    s->box_words = box_words;
    s->ans_words = ans_words;
    s->spin_us = spin_us;
    s->layers = std::make_unique<ServeLayer[]>(n_layers);
    s->n_layers = n_layers;
    s->done.assign(n_layers, 0);
    for (int i = 0; i < n_layers; ++i) s->done[i] = box[int64_t(i) * box_words];
    return owned.release();
}

API void qz_server_layer(void* handle, int i, const Proj* like, const void* const* bases, const int64_t* rowb,
                         float limit, int style, int inner) {
    Server* s = static_cast<Server*>(handle);
    ServeLayer& L = s->layers[i];
    for (int q = 0; q < 3; ++q) {
        L.like[q] = like[q];
        for (int k = 0; k < 3; ++k) {
            L.base[q][k] = static_cast<const uint8_t*>(bases[q * 3 + k]);
            L.rowb[q][k] = rowb[q * 3 + k];
        }
    }
    L.limit = limit;
    L.style = style;
    (void)inner;
    L.on.store(true, std::memory_order_release);
}

API void qz_server_cpus(void* handle, const int* cpus, int n) {
    Server* s = static_cast<Server*>(handle);
    s->cpus.assign(cpus, cpus + n);
}

API void qz_server_start(void* handle) {
    Server* s = static_cast<Server*>(handle);
    s->thread = std::thread([s] { s->loop(); });
}

API void qz_server_active(void* handle, int active) {
    static_cast<Server*>(handle)->active.store(active, std::memory_order_relaxed);
}

API int qz_server_busy(void* handle) {
    Server* s = static_cast<Server*>(handle);
    return s->serving.load(std::memory_order_acquire) + qz_pool_busy(s->pool);
}

API void qz_server_counts(void* handle, int64_t* out) {
    Server* s = static_cast<Server*>(handle);
    out[0] = s->requests.load();
    out[1] = s->experts.load();
    out[2] = s->failed.load();
    out[3] = s->compute_ns.load();
}

API void qz_server_free(void* handle) {
    std::unique_ptr<Server> s(static_cast<Server*>(handle));
    s->stop.store(1, std::memory_order_release);
    if (s->thread.joinable()) s->thread.join();
}
