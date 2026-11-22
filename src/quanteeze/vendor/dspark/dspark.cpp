// SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
// DSpark Markov proposal loop and confidence gate in C++20 on MLX (dspark.cpp,
// QUANTEEZE_DSPARK_NATIVE).

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <mlx/mlx.h>

#include <cmath>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace nb = nanobind;
namespace mx = mlx::core;

namespace {

struct Proj {
    int kind = 0;
    mx::array weight = mx::array(0.0f);
    std::optional<mx::array> scales, biases, bias;
    int group_size = 0, bits = 0;
    std::string mode;
};

Proj proj(int kind, mx::array weight, std::optional<mx::array> scales, std::optional<mx::array> biases,
          int group_size, int bits, std::string mode, std::optional<mx::array> bias) {
    Proj p;
    p.kind = kind;
    p.weight = std::move(weight);
    p.scales = std::move(scales);
    p.biases = std::move(biases);
    p.group_size = group_size;
    p.bits = bits;
    p.mode = std::move(mode);
    p.bias = std::move(bias);
    return p;
}

mx::array embed(const Proj& p, const mx::array& ids) {
    if (p.kind == 0) {
        return mx::take(p.weight, ids, 0);
    }
    std::optional<mx::array> b;
    if (p.biases) {
        b = mx::take(*p.biases, ids, 0);
    }
    return mx::dequantize(
        mx::take(p.weight, ids, 0), mx::take(*p.scales, ids, 0), b,
        p.group_size, p.bits, p.mode
    );
}

mx::array linear(const Proj& p, const mx::array& x) {
    if (p.kind == 0) {
        if (p.bias) {
            return mx::addmm(*p.bias, x, mx::transpose(p.weight));
        }
        return mx::matmul(x, mx::transpose(p.weight));
    }
    mx::array y = mx::quantized_matmul(x, p.weight, *p.scales, p.biases, true, p.group_size, p.bits, p.mode);
    if (p.bias) {
        y = mx::add(y, *p.bias);
    }
    return y;
}

mx::array logprobs(const mx::array& logits, double temperature, double top_p, int top_k) {
    mx::array lp = mx::astype(logits, mx::float32);
    lp = mx::subtract(lp, mx::logsumexp(lp, -1, true));
    const int vocab = lp.shape(-1);
    if (0 < top_k && top_k < vocab) {
        mx::array idx = mx::argpartition(mx::negative(lp), top_k - 1, -1);
        mx::Shape start(idx.ndim(), 0), stop = idx.shape();
        start.back() = top_k;
        idx = mx::slice(idx, start, stop);
        lp = mx::put_along_axis(lp, idx, mx::array(-INFINITY, lp.dtype()), -1);
    }
    lp = mx::divide(lp, mx::array(static_cast<float>(temperature), mx::float32));
    lp = mx::subtract(lp, mx::logsumexp(lp, -1, true));
    if (0 < top_p && top_p < 1) {
        mx::array sorted = mx::sort(lp, -1);
        mx::array probs = mx::exp(mx::astype(sorted, mx::float32));
        mx::array above = mx::cumsum(probs, -1, true, false);
        mx::Shape start(above.ndim(), 0), stop = above.shape();
        stop.back() = 1;
        mx::array total = mx::add(mx::slice(above, start, stop), mx::slice(probs, start, stop));
        mx::array dropped = mx::sum(
            mx::greater_equal(above, mx::multiply(total, mx::array(static_cast<float>(top_p), mx::float32))),
            std::vector<int>{-1},
            true
        );
        mx::array threshold = mx::take_along_axis(sorted, dropped, -1);
        lp = mx::where(mx::less(lp, threshold), mx::array(-INFINITY, mx::float32), lp);
        lp = mx::subtract(lp, mx::logsumexp(lp, -1, true));
    }
    return lp;
}

std::tuple<mx::array, std::optional<mx::array>, mx::array>
markov(const mx::array& logits, int anchor, const Proj& embed_p, const Proj& head_p,
       std::optional<std::tuple<double, double, int>> sampling, std::vector<mx::array> keys) {
    const int batch = logits.shape(0), block = logits.shape(1);
    if (sampling && static_cast<int>(keys.size()) < block) {
        throw std::invalid_argument("dspark native: requires one PRNG key per block position");
    }
    mx::array token = mx::array(std::vector<int32_t>(batch, anchor).data(), {batch}, mx::int32);
    std::vector<mx::array> out, qs, embeds;
    for (int i = 0; i < block; ++i) {
        mx::array e = embed(embed_p, token);
        embeds.push_back(e);
        mx::array bias = mx::astype(linear(head_p, e), mx::float32);
        mx::array at = mx::squeeze(mx::slice(logits, {0, i, 0}, {batch, i + 1, logits.shape(2)}), 1);
        mx::array scores = mx::add(at, bias);
        if (!sampling) {
            token = mx::astype(mx::argmax(scores, -1), mx::int32);
        } else {
            auto [temperature, top_p, top_k] = *sampling;
            mx::array q = logprobs(scores, temperature, top_p, top_k);
            token = mx::astype(mx::random::categorical(q, -1, keys[i]), mx::int32);
            qs.push_back(q);
        }
        out.push_back(token);
    }
    std::optional<mx::array> q;
    if (!qs.empty()) {
        q = mx::stack(qs, 1);
    }
    return {mx::stack(out, 1), q, mx::stack(embeds, 1)};
}

int gated(const mx::array& conf, int k, double threshold) {
    mx::array c = mx::slice(conf, {0, 0}, {1, k});
    c = mx::squeeze(c, 0);
    mx::array kept = mx::greater_equal(
        mx::cumprod(mx::sigmoid(c), 0),
        mx::array(static_cast<float>(threshold), mx::float32)
    );
    mx::array n = mx::sum(mx::cumprod(mx::astype(kept, mx::int32), 0));
    nb::gil_scoped_release release;
    n.eval();
    return n.item<int>();
}

}  // namespace

NB_MODULE(_dspark, m) {
    nb::module_::import_("mlx.core");
    m.doc() = "DSpark Markov loop and confidence gate in C++20 on MLX for QuantEEze";
    m.def("probe", [](const mx::array& a) { return mx::add(a, mx::array(1.0f, a.dtype())); });
    nb::class_<Proj>(m, "Proj");
    m.def("proj", &proj, nb::arg("kind"), nb::arg("weight"), nb::arg("scales").none(), nb::arg("biases").none(),
          nb::arg("group_size"), nb::arg("bits"), nb::arg("mode"), nb::arg("bias").none());
    m.def("markov", &markov, nb::arg("logits"), nb::arg("anchor"), nb::arg("embed"), nb::arg("head"),
          nb::arg("sampling").none(), nb::arg("keys"));
    m.def("gated", &gated, nb::arg("conf"), nb::arg("k"), nb::arg("threshold"));
}
