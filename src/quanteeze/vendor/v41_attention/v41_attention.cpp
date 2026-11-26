// SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
// DeepSeek V4.1 short-pass full attention in C++20 on MLX (v41_attention.cpp,
// QUANTEEZE_V41_ATTENTION_NATIVE).
// Builds the exact same MLX op graph as Python's front_python/back_python in two calls
// per layer on Metal and CUDA (compressor and indexer run in Python between front and back).

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <mlx/fast.h>
#include <mlx/mlx.h>

#include <algorithm>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "atomic_smart_ptr.h"

namespace nb = nanobind;
namespace mx = mlx::core;

namespace {

template <class K, class V>
class Made {
public:
    struct Entry : atomic_emplaced {
        Entry(K k, V v) : key(std::move(k)), value(std::move(v)) {}
        K key;
        V value;
    };
    using EntryP = local_shared_ptr<const Entry>;

    template <class F>
    EntryP get(const K& key, F make) {
        if (EntryP found = find(local_shared_ptr<const List>(list_), key)) {
            return found;
        }
        const EntryP made(make_local_shared<Entry>(key, make()));
        for (;;) {
            local_shared_ptr<const List> was(list_);
            if (EntryP found = find(was, key)) {
                return found;
            }
            auto now = make_local_shared<List>();
            if (was) {
                now->all = was->all;
            }
            now->all.push_back(made);
            if (list_.compareAndSet(was, local_shared_ptr<const List>(std::move(now)))) {
                return made;
            }
        }
    }

private:
    struct List : atomic_emplaced {
        std::vector<EntryP> all;
    };
    static EntryP find(const local_shared_ptr<const List>& list, const K& key) {
        if (list) {
            for (const auto& e : list->all) {
                if (e->key == key) {
                    return e;
                }
            }
        }
        return EntryP();
    }
    atomic_shared_ptr<const List> list_;
};

// Projection descriptor:
//   kind 0: nn.Linear
//   kind 1: nn.QuantizedLinear
//   kind 2: mla.MultiLinear
//   kind 3: mla.QuantizedMultiLinear
struct Lin {
    int kind = 0;
    mx::array weight = mx::array(0.0f);
    std::optional<mx::array> scales, biases, bias;
    int group_size = 0, bits = 0;
    std::string mode;
};

Lin lin(int kind, mx::array weight, std::optional<mx::array> scales, std::optional<mx::array> biases,
        std::optional<mx::array> bias, int group_size, int bits, std::string mode) {
    if (kind < 0 || kind > 3) {
        throw std::invalid_argument("lin: kind must be 0..3");
    }
    if ((kind == 1 || kind == 3) && !scales) {
        throw std::invalid_argument("lin: quantized projection requires scales");
    }
    return Lin{
        kind, std::move(weight), std::move(scales), std::move(biases),
        std::move(bias), group_size, bits, std::move(mode),
    };
}

mx::array apply(const Lin& l, const mx::array& x) {
    switch (l.kind) {
    case 0:
        return l.bias ? mx::addmm(*l.bias, x, mx::transpose(l.weight)) : mx::matmul(x, mx::transpose(l.weight));
    case 1: {
        auto y = mx::quantized_matmul(x, l.weight, *l.scales, l.biases, true, l.group_size, l.bits, l.mode);
        return l.bias ? mx::add(y, *l.bias) : y;
    }
    case 2:
        return mx::matmul(x, mx::swapaxes(l.weight, -1, -2));
    default:
        return mx::quantized_matmul(x, l.weight, *l.scales, l.biases, true, l.group_size, l.bits, l.mode);
    }
}

const char* FP8_SOURCE = R"(
            uint i = thread_position_in_grid.x;
            float value = float(x[i]);
            float amax = metal::max(metal::simd_max(metal::abs(value)), 1e-4f);
            uint bits = as_type<uint>(amax * (1.0f / 448.0f));
            uint exponent = (bits >> 23) + ((bits & 0x7fffff) != 0);
            float scale = as_type<float>(exponent << 23);
            float v = metal::clamp(value / scale, -448.0f, 448.0f);
            uint vbits = as_type<uint>(metal::abs(v));
            uint qexp = metal::max(int(vbits >> 23) - 3, 118);
            float step = as_type<float>(qexp << 23);
            out[i] = T((metal::rint(v / step) * step) * scale);
        )";

mx::array fq8(const mx::array& x) {
    static const mx::fast::CustomKernelFunction kernel =
        mx::fast::metal_kernel("deepseek_v41_fp8_roundtrip", {"x"}, {"out"}, FP8_SOURCE, "", true, false);
    if (x.size() == 0 || x.shape().back() % 32) {
        throw std::invalid_argument("fq8: last dimension must be a non-empty multiple of 32");
    }
    return kernel(
        {x},
        {x.shape()},
        {x.dtype()},
        {static_cast<int>(x.size()), 1, 1},
        {256, 1, 1},
        {{"T", x.dtype()}},
        std::nullopt,
        false,
        {}
    )[0];
}

struct CudaKernel {
    std::string name, header, source;
    std::vector<std::string> inputs, outputs;
    int shared = 0;
};

CudaKernel cuda_kernel_spec(std::string name, std::vector<std::string> inputs, std::vector<std::string> outputs,
                            std::string header, std::string source, int shared) {
    return CudaKernel{
        std::move(name), std::move(header), std::move(source),
        std::move(inputs), std::move(outputs), shared,
    };
}

const mx::fast::CustomKernelFunction& cuda_kernel_of(const CudaKernel& k) {
    static Made<std::string, mx::fast::CustomKernelFunction> made;
    return made.get(k.name, [&k] {
        return mx::fast::cuda_kernel(k.name, k.inputs, k.outputs, k.source, k.header, true, k.shared);
    })->value;
}

mx::array fq8(const mx::array& x, const std::optional<CudaKernel>& cuda) {
    if (!cuda) {
        return fq8(x);
    }
    if (x.size() == 0 || x.shape().back() % 32) {
        throw std::invalid_argument("fq8: last dimension must be a non-empty multiple of 32");
    }
    return cuda_kernel_of(*cuda)(
        {x},
        {x.shape()},
        {x.dtype()},
        {static_cast<int>(x.size()), 1, 1},
        {32, 1, 1},
        {{"T", x.dtype()}},
        std::nullopt,
        false,
        {}
    )[0];
}

mx::array rope(const mx::array& x, int offset, const mx::array& freqs) {
    return mx::fast::rope(x, x.shape().back(), true, std::nullopt, 1.0f, offset, freqs);
}

std::vector<mx::array> split_softmax_fn(const std::vector<mx::array>& in) {
    auto normalizer = mx::logaddexp(in[0], in[3]);
    return {mx::exp(mx::subtract(in[1], normalizer)), mx::exp(mx::subtract(in[2], normalizer))};
}

std::vector<mx::array> split_softmax(const std::vector<mx::array>& in) {
    static const auto compiled = mx::compile(&split_softmax_fn, true);
    return compiled(in);
}

mx::array score_mask(const mx::array& scores, const mx::array& mask) {
    return mx::where(mask, scores, mx::array(std::numeric_limits<float>::lowest(), scores.dtype()));
}

mx::array gather_pooled(const mx::array& pooled, const mx::array& topk) {
    const int B = pooled.shape(0), n = pooled.shape(1), D = pooled.shape(2), L = topk.shape(1), k = topk.shape(2);
    if (B == 1) {
        return mx::reshape(mx::take(pooled, mx::reshape(topk, {-1}), 1), {B, L, k, D});
    }
    auto rows = mx::where(mx::less(topk, mx::array(0, topk.dtype())), mx::add(topk, mx::array(n, topk.dtype())), topk);
    rows = mx::add(rows, mx::reshape(mx::multiply(mx::arange(B, topk.dtype()), mx::array(n, topk.dtype())), {B, 1, 1}));
    return mx::reshape(mx::take(mx::reshape(pooled, {B * n, D}), mx::reshape(rows, {-1}), 0), {B, L, k, D});
}

int lse_threads(int n) {
    const int need = (n + 3) / 4;
    int block = 32;
    while (block < need && block < 1024) {
        block *= 2;
    }
    return block;
}

bool takes_glue(const mx::array& q, const mx::array& local_kv, const mx::array& topk,
                const mx::array& local_mask, const mx::array& pooled_mask) {
    return q.ndim() == 4 && q.shape(0) == 1 && q.shape(2) == 1 && q.dtype() == mx::float32
        && topk.ndim() == 3 && topk.shape(0) == 1 && topk.shape(1) == 1 && topk.shape(2) > 0 && topk.shape(2) <= 4096
        && local_kv.shape(-2) > 0 && local_kv.shape(-2) <= 4096
        && local_mask.dtype() == mx::bool_ && static_cast<int64_t>(local_mask.size()) == local_kv.shape(-2)
        && pooled_mask.dtype() == mx::bool_ && static_cast<int64_t>(pooled_mask.size()) == topk.shape(2);
}

mx::array sparse(const mx::array& q, const mx::array& window, const mx::array& pool,
                 const mx::array& indices, const mx::array& window_mask, float scale, const mx::array& sinks,
                 const std::optional<CudaKernel>& glue = std::nullopt) {
    const auto qf = mx::astype(q, mx::float32);
    const auto local_kv = mx::expand_dims(window, 1);
    const auto pooled_mask = mx::expand_dims(mx::not_equal(indices, mx::array(-1, indices.dtype())), 1);
    const auto sinks32 = mx::astype(sinks, mx::float32);
    const auto pooled_sq = gather_pooled(pool, indices);
    const auto q_scaled = mx::multiply(qf, mx::array(scale, qf.dtype()));
    if (glue && takes_glue(qf, local_kv, indices, window_mask, pooled_mask)) {
        const auto local_scores = mx::matmul(q_scaled, mx::swapaxes(local_kv, -1, -2));
        const auto pooled_scores = mx::transpose(
            mx::matmul(mx::transpose(q_scaled, {0, 2, 1, 3}), mx::swapaxes(pooled_sq, -1, -2)),
            {0, 2, 1, 3}
        );
        const int H = local_scores.shape(1), W = local_scores.shape(-1), P = pooled_scores.shape(-1);
        const int block = std::max(lse_threads(W), lse_threads(P));
        const auto w = cuda_kernel_of(*glue)(
            {
                local_scores,
                mx::reshape(window_mask, {-1}),
                pooled_scores,
                mx::reshape(pooled_mask, {-1}),
                mx::astype(sinks32, mx::float32),
            },
            {local_scores.shape(), pooled_scores.shape()},
            {mx::float32, mx::float32},
            {H * block, 1, 1},
            {block, 1, 1},
            {},
            std::nullopt,
            false,
            {}
        );
        auto out = mx::matmul(w[0], local_kv);
        out = mx::add(out, mx::transpose(mx::matmul(mx::transpose(w[1], {0, 2, 1, 3}), pooled_sq), {0, 2, 1, 3}));
        return mx::astype(mx::astype(out, qf.dtype()), q.dtype());
    }
    auto local_scores = score_mask(mx::matmul(q_scaled, mx::swapaxes(local_kv, -1, -2)), window_mask);
    auto normalizer = mx::logsumexp(local_scores, -1, true);
    const auto q_bl = mx::transpose(q_scaled, {0, 2, 1, 3});
    auto pooled_scores = mx::transpose(mx::matmul(q_bl, mx::swapaxes(pooled_sq, -1, -2)), {0, 2, 1, 3});
    pooled_scores = score_mask(pooled_scores, pooled_mask);
    normalizer = mx::logaddexp(normalizer, mx::logsumexp(pooled_scores, -1, true));
    const auto w = split_softmax(
        {normalizer, local_scores, pooled_scores, mx::reshape(sinks32, {1, sinks32.shape(0), 1, 1})}
    );
    auto out = mx::matmul(w[0], local_kv);
    out = mx::add(out, mx::transpose(mx::matmul(mx::transpose(w[1], {0, 2, 1, 3}), pooled_sq), {0, 2, 1, 3}));
    return mx::astype(mx::astype(out, qf.dtype()), q.dtype());
}

struct Layer {
    Lin wq_a, wq_b, wkv, wo_a, wo_b;
    mx::array q_norm = mx::array(0.0f), kv_norm = mx::array(0.0f), freqs = mx::array(0.0f),
              inverse = mx::array(0.0f), sinks = mx::array(0.0f);
    float q_eps = 0.0f, kv_eps = 0.0f, scale = 0.0f;
    int heads = 0, head_dim = 0, o_groups = 0, window = 0;
    std::optional<CudaKernel> fp8, glue;
};

Layer layer(Lin wq_a, mx::array q_norm, float q_eps, Lin wq_b, Lin wkv, mx::array kv_norm, float kv_eps,
            Lin wo_a, Lin wo_b, mx::array freqs, mx::array inverse, mx::array sinks, float scale,
            int heads, int head_dim, int o_groups, int window) {
    Layer l;
    l.wq_a = std::move(wq_a);
    l.wq_b = std::move(wq_b);
    l.wkv = std::move(wkv);
    l.wo_a = std::move(wo_a);
    l.wo_b = std::move(wo_b);
    l.q_norm = std::move(q_norm);
    l.kv_norm = std::move(kv_norm);
    l.freqs = std::move(freqs);
    l.inverse = std::move(inverse);
    l.sinks = std::move(sinks);
    l.q_eps = q_eps;
    l.kv_eps = kv_eps;
    l.scale = scale;
    l.heads = heads;
    l.head_dim = head_dim;
    l.o_groups = o_groups;
    l.window = window;
    return l;
}

Layer layer_on_cuda(Layer l, std::optional<CudaKernel> fp8, std::optional<CudaKernel> glue) {
    l.fp8 = std::move(fp8);
    l.glue = std::move(glue);
    return l;
}

using Write = std::tuple<int, int, int, int>;

std::tuple<mx::array, mx::array, mx::array, mx::array>
front(const Layer& l, const mx::array& x, int offset, std::optional<mx::array> buffer,
      const std::optional<mx::array>& history, const std::vector<Write>& writes) {
    const int B = x.shape(0), L = x.shape(1), D = l.head_dim, win = l.window;
    const auto fx = fq8(x, l.fp8);
    const auto qr = mx::fast::rms_norm(apply(l.wq_a, fx), l.q_norm, l.q_eps);
    auto q = mx::transpose(mx::reshape(apply(l.wq_b, fq8(qr, l.fp8)), {B, L, l.heads, D}), {0, 2, 1, 3});
    q = rope(q, offset, l.freqs);
    auto kv = mx::fast::rms_norm(apply(l.wkv, fx), l.kv_norm, l.kv_eps);
    kv = fq8(mx::reshape(rope(mx::expand_dims(kv, 1), offset, l.freqs), {B, L, D}), l.fp8);
    mx::array ring = buffer ? *buffer : mx::zeros({B, win, D}, kv.dtype());
    if (ring.shape(0) < B) {
        ring = mx::pad(ring, {{0, B - ring.shape(0)}, {0, 0}, {0, 0}});
    }
    const auto part = history
        ? mx::concatenate({mx::take(mx::slice(ring, {0, 0, 0}, {B, ring.shape(1), D}), *history, 1), kv}, 1)
        : kv;
    const int kept = std::min(L, win);
    const auto tail = mx::slice(kv, {0, L - kept, 0}, {B, L, D});
    for (const auto& [a, b, c, d] : writes) {
        ring = mx::slice_update(ring, mx::slice(tail, {0, c, 0}, {B, d, D}), mx::Shape{0, a, 0}, mx::Shape{B, b, D});
    }
    return {q, qr, part, ring};
}

mx::array back(const Layer& l, const mx::array& q, const mx::array& part, const mx::array& valid,
               const std::optional<mx::array>& pool, const std::optional<mx::array>& indices, int offset) {
    const int B = q.shape(0), L = q.shape(2), D = l.head_dim;
    mx::array out = (pool && indices && pool->shape(1))
        ? sparse(q, part, *pool, *indices, valid, l.scale, l.sinks, l.glue)
        : [&] {
              const auto kv = mx::astype(mx::expand_dims(part, 1), q.dtype());
              return mx::fast::scaled_dot_product_attention(
                  q, kv, kv, l.scale, "array", valid, mx::astype(l.sinks, q.dtype())
              );
          }();
    out = rope(out, offset, l.inverse);
    out = mx::flatten(mx::transpose(mx::reshape(out, {B, l.o_groups, -1, L, D}), {0, 1, 3, 2, 4}), -2, -1);
    out = mx::flatten(mx::transpose(apply(l.wo_a, out), {0, 2, 1, 3}), -2, -1);
    return apply(l.wo_b, fq8(out, l.fp8));
}

}  // namespace

NB_MODULE(_v41_attention, m) {
    nb::module_::import_("mlx.core");
    m.doc() = "DeepSeek V4.1 attention pass in C++20 on MLX for QuantEEze";
    m.def("probe", [](const mx::array& a) { return mx::add(a, mx::array(1.0f, a.dtype())); });
    nb::class_<Lin>(m, "Lin");
    nb::class_<Layer>(m, "Layer");
    nb::class_<CudaKernel>(m, "CudaKernel");
    m.def("lin", &lin, nb::arg("kind"), nb::arg("weight"), nb::arg("scales").none(), nb::arg("biases").none(),
          nb::arg("bias").none(), nb::arg("group_size"), nb::arg("bits"), nb::arg("mode"));
    m.def("layer", &layer, nb::arg("wq_a"), nb::arg("q_norm"), nb::arg("q_eps"), nb::arg("wq_b"), nb::arg("wkv"),
          nb::arg("kv_norm"), nb::arg("kv_eps"), nb::arg("wo_a"), nb::arg("wo_b"), nb::arg("freqs"), nb::arg("inverse"),
          nb::arg("sinks"), nb::arg("scale"), nb::arg("heads"), nb::arg("head_dim"), nb::arg("o_groups"),
          nb::arg("window"));
    m.def("cuda_kernel_spec", &cuda_kernel_spec, nb::arg("name"), nb::arg("inputs"), nb::arg("outputs"),
          nb::arg("header"), nb::arg("source"), nb::arg("shared"));
    m.def("layer_on_cuda", &layer_on_cuda, nb::arg("layer"), nb::arg("fp8").none(), nb::arg("glue").none());
    m.def("fq8", [](const mx::array& x, const std::optional<CudaKernel>& cuda) { return fq8(x, cuda); },
          nb::arg("x"), nb::arg("cuda").none() = nb::none());
    m.def("front", &front, nb::arg("layer"), nb::arg("x"), nb::arg("offset"), nb::arg("buffer").none(),
          nb::arg("history").none(), nb::arg("writes"));
    m.def("back", &back, nb::arg("layer"), nb::arg("q"), nb::arg("part"), nb::arg("valid"), nb::arg("pool").none(),
          nb::arg("indices").none(), nb::arg("offset"));
}
