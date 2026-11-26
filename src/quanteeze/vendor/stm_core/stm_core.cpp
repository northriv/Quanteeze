// SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
// Portable C++20 STM core for QuantEEze (KAME STM + nanobind + MLX).
// Holds per-layer routing placements, active GPU pass leases, and an asynchronous
// landing thread backed by C native readers and CUDA / unified-memory copies.

#include <support.h>
#include <transaction.h>
#include <transaction_impl.h>

#include <mlx/mlx.h>

#include <Python.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace nb = nanobind;
namespace mx = mlx::core;

namespace {

// ---------------------------------------------------------------------------
// KAME STM nodes
// ---------------------------------------------------------------------------

class CountNode : public Transactional::Node<CountNode> {
public:
    struct Payload : public Transactional::Node<CountNode>::Payload {
        int64_t count{0};
        int64_t sum{0};
    };
};

using Vec64 = std::vector<int64_t>;
using VecU8 = std::vector<uint8_t>;

class LayerNode : public Transactional::Node<LayerNode> {
public:
    struct Payload : public Transactional::Node<LayerNode>::Payload {
        int64_t version{0};
        uint64_t lease_seq{0};
        std::shared_ptr<const Vec64> route;
        std::shared_ptr<const VecU8> absent;
        std::shared_ptr<const Vec64> experts;
        std::shared_ptr<const Vec64> in_use;
        std::shared_ptr<const Vec64> inflight;
        bool writing{false};
    };
};

using LayerPtr = std::shared_ptr<LayerNode>;
using LayerTx = Transactional::Transaction<LayerNode>;
using LayerSnap = Transactional::Snapshot<LayerNode>;

constexpr int TURN_PROMPT = 0;
constexpr int TURN_DECODE = 1;
constexpr size_t TURN_SLOTS = 32;

class TurnNode : public Transactional::Node<TurnNode> {
public:
    struct Payload : public Transactional::Node<TurnNode>::Payload {
        bool prompt{false};
        int64_t prompt_wait{0};
        int32_t decodes{0};
        std::array<int64_t, TURN_SLOTS> waits{};
    };
};

using TurnPtr = std::shared_ptr<TurnNode>;
using TurnTx = Transactional::Transaction<TurnNode>;
using TurnSnap = Transactional::Snapshot<TurnNode>;

class RoomNode : public Transactional::Node<RoomNode> {
public:
    struct Payload : public Transactional::Node<RoomNode>::Payload {
        int64_t room{0};
    };
};

using RoomPtr = std::shared_ptr<RoomNode>;
using RoomTx = Transactional::Transaction<RoomNode>;
using RoomSnap = Transactional::Snapshot<RoomNode>;

struct TurnHeld {
    uint64_t owner{0};
    int kind{0};
    int64_t birth{0};
    double timeout_ms{0.0};
    bool holding{false};
    bool paused{false};
};

std::atomic<uint64_t> next_model_id{1};
thread_local TurnHeld tl_turn;

template <typename T>
nb::ndarray<nb::numpy, T> make_1d(std::vector<T> vec) {
    size_t n = vec.size();
    auto heap = std::make_unique<std::vector<T>>(std::move(vec));
    T* ptr = n ? heap->data() : nullptr;
    nb::capsule owner(heap.release(), [](void* p) noexcept {
        std::unique_ptr<std::vector<T>> owned(static_cast<std::vector<T>*>(p));
    });
    size_t shape[1] = {n};
    return nb::ndarray<nb::numpy, T>(ptr, 1, shape, owner);
}

nb::ndarray<nb::numpy, bool> make_bool_1d(const VecU8& src) {
    size_t n = src.size();
    auto buf = n ? std::make_unique<bool[]>(n) : std::unique_ptr<bool[]>{};
    for (size_t i = 0; i < n; ++i) {
        buf[i] = src[i] != 0;
    }
    bool* ptr = buf.release();
    nb::capsule owner(ptr, [](void* p) noexcept {
        std::unique_ptr<bool[]> owned(static_cast<bool*>(p));
    });
    size_t shape[1] = {n};
    return nb::ndarray<nb::numpy, bool>(ptr, 1, shape, owner);
}

inline int64_t monotonic_ns() noexcept {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

template <typename Out, typename Read>
void host_strided(const mx::array& a, std::vector<Out>& out, Read read) {
    if (!a.is_available()) {
        throw std::logic_error("stm_core: array not evaluated");
    }
    out.resize(a.size());
    if (a.size() == 0) {
        return;
    }
    const size_t nd = a.ndim();
    if (nd == 0) {
        out[0] = read(0);
        return;
    }
    std::vector<int64_t> at(nd, 0);
    int64_t off = 0;
    for (size_t q = 0; q < out.size(); ++q) {
        out[q] = read(off);
        for (size_t d = nd; d-- > 0;) {
            off += a.strides()[d];
            if (++at[d] < a.shape()[d]) {
                break;
            }
            off -= a.strides()[d] * a.shape()[d];
            at[d] = 0;
        }
    }
}

void host_ints(const mx::array& a, std::vector<int64_t>& out) {
    switch (a.dtype()) {
        case mx::int64: { auto p = a.data<int64_t>(); host_strided(a, out, [p](int64_t o) { return p[o]; }); return; }
        case mx::int32: { auto p = a.data<int32_t>(); host_strided(a, out, [p](int64_t o) { return static_cast<int64_t>(p[o]); }); return; }
        case mx::uint32: { auto p = a.data<uint32_t>(); host_strided(a, out, [p](int64_t o) { return static_cast<int64_t>(p[o]); }); return; }
        case mx::int16: { auto p = a.data<int16_t>(); host_strided(a, out, [p](int64_t o) { return static_cast<int64_t>(p[o]); }); return; }
        case mx::uint16: { auto p = a.data<uint16_t>(); host_strided(a, out, [p](int64_t o) { return static_cast<int64_t>(p[o]); }); return; }
        case mx::uint8: { auto p = a.data<uint8_t>(); host_strided(a, out, [p](int64_t o) { return static_cast<int64_t>(p[o]); }); return; }
        default: throw std::invalid_argument("stm_core: expected integer array");
    }
}

void host_doubles(const mx::array& a, std::vector<double>& out) {
    switch (a.dtype()) {
        case mx::float32: { auto p = a.data<float>(); host_strided(a, out, [p](int64_t o) { return static_cast<double>(p[o]); }); return; }
        case mx::float64: { auto p = a.data<double>(); host_strided(a, out, [p](int64_t o) { return p[o]; }); return; }
        case mx::bfloat16: {
            auto p = a.data<uint16_t>();
            host_strided(a, out, [p](int64_t o) {
                const uint32_t bits = static_cast<uint32_t>(p[o]) << 16;
                float f;
                std::memcpy(&f, &bits, sizeof(f));
                return static_cast<double>(f);
            });
            return;
        }
        case mx::float16: {
            auto p = a.data<mx::float16_t>();
            host_strided(a, out, [p](int64_t o) { return static_cast<double>(static_cast<float>(p[o])); });
            return;
        }
        default: throw std::invalid_argument("stm_core: expected floating-point array");
    }
}

// ---------------------------------------------------------------------------
// Host allocator + optional CUDA driver pinning / async copy tables
// ---------------------------------------------------------------------------

struct CudaFns {
    uintptr_t ctx{0};
    int (*push_ctx)(uintptr_t){nullptr};
    int (*pop_ctx)(uintptr_t*){nullptr};
    int (*stream_create)(uintptr_t*, unsigned int){nullptr};
    int (*memcpy_async)(uintptr_t, uintptr_t, size_t, uintptr_t){nullptr};
    int (*stream_sync)(uintptr_t){nullptr};
    int (*stream_destroy)(uintptr_t){nullptr};

    bool attached() const noexcept {
        return ctx && push_ctx && pop_ctx && stream_create && memcpy_async && stream_sync && stream_destroy;
    }
};

struct HostFns {
    void* (*alloc)(size_t, size_t){nullptr};
    void (*free_fn)(void*){nullptr};
    int (*host_reg)(void*, size_t, unsigned int){nullptr};
    int (*host_unreg)(void*){nullptr};
    int (*host_get_dev)(uint64_t*, void*, unsigned int){nullptr};

    bool has_alloc() const noexcept { return alloc && free_fn; }
    bool has_pin() const noexcept { return host_reg && host_unreg && host_get_dev; }
};

struct HostBlock {
    void* ptr{nullptr};
    size_t size{0};
    bool pinned{false};
    uint64_t dev_ptr{0};
    HostFns fns{};

    HostBlock() = default;
    HostBlock(const HostBlock&) = delete;
    HostBlock& operator=(const HostBlock&) = delete;
    HostBlock(HostBlock&& other) noexcept
        : ptr(other.ptr), size(other.size), pinned(other.pinned), dev_ptr(other.dev_ptr), fns(other.fns) {
        other.ptr = nullptr;
        other.size = 0;
        other.pinned = false;
        other.dev_ptr = 0;
    }
    HostBlock& operator=(HostBlock&& other) noexcept {
        if (this != &other) {
            release();
            ptr = other.ptr;
            size = other.size;
            pinned = other.pinned;
            dev_ptr = other.dev_ptr;
            fns = other.fns;
            other.ptr = nullptr;
            other.size = 0;
            other.pinned = false;
            other.dev_ptr = 0;
        }
        return *this;
    }
    ~HostBlock() { release(); }

    void release() noexcept {
        if (!ptr) return;
        if (pinned && fns.host_unreg) {
            fns.host_unreg(ptr);
        }
        if (fns.free_fn) {
            fns.free_fn(ptr);
        } else {
            std::free(ptr);
        }
        ptr = nullptr;
        size = 0;
        pinned = false;
        dev_ptr = 0;
    }

    static HostBlock allocate(size_t bytes, const HostFns& h, bool pin) {
        HostBlock b;
        b.fns = h;
        b.size = bytes;
        size_t alloc_bytes = bytes ? bytes : 64;
        if (h.has_alloc()) {
            b.ptr = h.alloc(64, alloc_bytes);
        } else {
            size_t rounded = ((alloc_bytes + 63) / 64) * 64;
#if defined(_WIN32)
            b.ptr = _aligned_malloc(rounded, 64);
#else
            b.ptr = std::aligned_alloc(64, rounded);
#endif
        }
        if (!b.ptr) {
            throw std::bad_alloc();
        }
        if (pin && bytes > 0 && h.has_pin()) {
            if (h.host_reg(b.ptr, bytes, 2 /* CU_MEMHOSTREGISTER_DEVICEMAP */) == 0) {
                uint64_t dp = 0;
                if (h.host_get_dev(&dp, b.ptr, 0) == 0) {
                    b.pinned = true;
                    b.dev_ptr = dp;
                } else {
                    h.host_unreg(b.ptr);
                }
            }
        }
        return b;
    }
};

// ---------------------------------------------------------------------------
// Native pread reader interface (readers.cpp)
// ---------------------------------------------------------------------------

struct ReaderFns {
    void* handle{nullptr};
    int64_t (*submit)(void*, int, int, int, const int32_t*, const int64_t*, const int64_t*, const uint64_t*){nullptr};
    int32_t (*status)(void*, int64_t){nullptr};
    void (*release)(void*, int64_t){nullptr};
    void (*times)(void*, int64_t, int64_t*){nullptr};
    int64_t (*backlog)(void*){nullptr};
    int workers{0};

    bool attached() const noexcept {
        return handle && submit && status && release;
    }
};

constexpr int32_t RD_QUEUED = 0;
constexpr int32_t RD_TAKEN = 1;
constexpr int32_t RD_DONE = 2;
constexpr int32_t RD_FAILED = 3;
constexpr int32_t RD_CANCELLED = 4;

// ---------------------------------------------------------------------------
// Pass token & active lease hub
// KAME XWaitCell models fixed wait-on-address scalars; condvars here handle
// unbounded std::deque queues and multi-predicate timed condvar waits.
// ---------------------------------------------------------------------------

struct PassesHub {
    std::mutex mu;
    std::condition_variable cv;
    std::unordered_map<int64_t, int> live_tokens;
    std::deque<int64_t> retired;
    std::atomic<int64_t> waiters{0};

    void token_created(int64_t pass_num) {
        std::lock_guard<std::mutex> lk(mu);
        live_tokens[pass_num] += 1;
    }

    void token_deleted(int64_t pass_num) noexcept {
        std::lock_guard<std::mutex> lk(mu);
        auto it = live_tokens.find(pass_num);
        if (it != live_tokens.end()) {
            it->second -= 1;
            if (it->second <= 0) {
                live_tokens.erase(it);
                retired.push_back(pass_num);
            }
        } else {
            retired.push_back(pass_num);
        }
        cv.notify_all();
    }

    Vec64 drain_retired() {
        std::lock_guard<std::mutex> lk(mu);
        Vec64 out(retired.begin(), retired.end());
        retired.clear();
        return out;
    }

    bool has_live_token(int64_t pass_num) {
        std::lock_guard<std::mutex> lk(mu);
        auto it = live_tokens.find(pass_num);
        return it != live_tokens.end() && it->second > 0;
    }
};

// ---------------------------------------------------------------------------
// Quant & tensor metadata per layer
// ---------------------------------------------------------------------------

struct QuantSlot {
    bool valid{false};
    int bits{0};
    int group_size{0};
    std::string mode;
    std::optional<mx::array> arrays[3];   // weight, scales, biases
    int tensor_idx[3]{-1, -1, -1};
    size_t row_bytes[3]{0, 0, 0};
    int64_t num_slots[3]{0, 0, 0};
    uintptr_t base_ptr[3]{0, 0, 0};
};

struct TensorSpec {
    bool valid{false};
    std::vector<int32_t> files;
    Vec64 offsets;
    int64_t row_bytes{0};
};

struct LandingJob {
    int k{0};
    Vec64 slots;
    Vec64 experts;
    int priority{1};
};

struct LandedRecord {
    int k{0};
    int64_t slot{0};
    int64_t expert{0};
    int status{0};     // 0=landed, 1=not_fit, 2=failed
    int64_t read_ns{0};
    int64_t copy_ns{0};
};

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

class Model {
public:
    explicit Model(int nlayers)
        : nlayers_(nlayers),
          hints_(nlayers, 0),
          layers_(nlayers),
          turns_(Transactional::Node<TurnNode>::create<TurnNode>()),
          room_(Transactional::Node<RoomNode>::create<RoomNode>()),
          layer_quants_(nlayers),
          layer_tensors_(nlayers),
          hub_(std::make_shared<PassesHub>()) {
        if (nlayers < 0) {
            throw std::invalid_argument("nlayers must be non-negative");
        }
        auto empty_experts = std::make_shared<Vec64>();
        auto empty_in_use = std::make_shared<Vec64>();
        auto empty_inflight = std::make_shared<Vec64>();
        for (int i = 0; i < nlayers; ++i) {
            layers_[i].reset(Transactional::Node<LayerNode>::create<LayerNode>());
            layers_[i]->iterate_commit([&](LayerTx& tr) {
                auto& p = tr[*layers_[i]];
                p.experts = empty_experts;
                p.in_use = empty_in_use;
                p.inflight = empty_inflight;
            });
            layer_tensors_[i].resize(64);
        }
    }

    ~Model() {
        stop_lander();
        destroy_sync_stream();
        if (tl_turn.owner == id_) {
            tl_turn = TurnHeld{};
        }
    }

    int nlayers() const noexcept { return nlayers_; }

    nb::ndarray<nb::numpy, const int64_t, nb::ndim<1>> hints(nb::handle owner) const {
        size_t shape[1] = {static_cast<size_t>(nlayers_)};
        const int64_t* ptr = nlayers_ ? hints_.data() : nullptr;
        return nb::ndarray<nb::numpy, const int64_t, nb::ndim<1>>(ptr, 1, shape, owner);
    }

    int64_t put_placement(
        int k,
        nb::ndarray<const int64_t, nb::c_contig, nb::device::cpu> route_arr,
        nb::ndarray<const bool, nb::c_contig, nb::device::cpu> absent_arr,
        nb::ndarray<const int64_t, nb::c_contig, nb::device::cpu> experts_arr) {
        check_layer(k);
        auto route = std::make_shared<Vec64>(
            route_arr.data(), route_arr.data() + route_arr.size());
        auto absent = std::make_shared<VecU8>(absent_arr.size());
        const bool* ab_p = absent_arr.data();
        for (size_t i = 0; i < absent_arr.size(); ++i) {
            (*absent)[i] = ab_p[i] ? 1 : 0;
        }
        auto experts = std::make_shared<Vec64>(
            experts_arr.data(), experts_arr.data() + experts_arr.size());
        int64_t ver = 0;
        layers_[k]->iterate_commit([&](LayerTx& tr) {
            auto& p = tr[*layers_[k]];
            p.route = route;
            p.absent = absent;
            p.experts = experts;
            p.version += 1;
            ver = p.version;
        });
        __atomic_store_n(&hints_[k], ver, __ATOMIC_RELEASE);
        return ver;
    }

    std::optional<nb::tuple> placement(int k) const {
        check_layer(k);
        int64_t ver = 0;
        std::shared_ptr<const Vec64> route;
        std::shared_ptr<const VecU8> absent;
        std::shared_ptr<const Vec64> experts;
        {
            nb::gil_scoped_release release_gil;
            LayerSnap snap(*layers_[k]);
            const auto& p = snap[*layers_[k]];
            ver = p.version;
            route = p.route;
            absent = p.absent;
            experts = p.experts;
        }
        if (!route || !absent || !experts) {
            return std::nullopt;
        }
        return nb::make_tuple(
            ver,
            make_1d(*route),
            make_bool_1d(*absent),
            make_1d(*experts));
    }

    std::optional<nb::tuple> tables(int k) const {
        check_layer(k);
        int64_t ver = 0;
        std::shared_ptr<const Vec64> route;
        std::shared_ptr<const VecU8> absent;
        std::shared_ptr<const Vec64> experts;
        {
            nb::gil_scoped_release release_gil;
            LayerSnap snap(*layers_[k]);
            const auto& p = snap[*layers_[k]];
            ver = p.version;
            route = p.route;
            absent = p.absent;
            experts = p.experts;
        }
        if (!route || !absent || !experts) {
            return std::nullopt;
        }
        size_t E = route->size();
        Vec64 slot_in(E, 0);
        const auto& exps = *experts;
        for (size_t s = 0; s < exps.size(); ++s) {
            int64_t e = exps[s];
            if (e >= 0 && static_cast<size_t>(e) < E) {
                slot_in[static_cast<size_t>(e)] = static_cast<int64_t>(s);
            }
        }
        return nb::make_tuple(
            ver,
            make_1d(*route),
            make_1d(*absent),
            make_1d(exps),
            make_1d(std::move(slot_in)));
    }

    int check(
        int k,
        nb::ndarray<const int64_t, nb::c_contig, nb::device::cpu> route_arr,
        nb::ndarray<const bool, nb::c_contig, nb::device::cpu> absent_arr,
        nb::ndarray<const int64_t, nb::c_contig, nb::device::cpu> experts_arr) const {
        check_layer(k);
        LayerSnap snap(*layers_[k]);
        const auto& p = snap[*layers_[k]];
        int bits = 0;
        if (!p.route || p.route->size() != route_arr.size() ||
            std::memcmp(p.route->data(), route_arr.data(), route_arr.size() * sizeof(int64_t)) != 0) {
            bits |= 1;
        }
        bool ab_same = p.absent && p.absent->size() == absent_arr.size();
        if (ab_same) {
            const bool* ap = absent_arr.data();
            for (size_t i = 0; i < absent_arr.size(); ++i) {
                if (((*p.absent)[i] != 0) != ap[i]) {
                    ab_same = false;
                    break;
                }
            }
        }
        if (!ab_same) {
            bits |= 2;
        }
        if (!p.experts || p.experts->size() != experts_arr.size() ||
            std::memcmp(p.experts->data(), experts_arr.data(), experts_arr.size() * sizeof(int64_t)) != 0) {
            bits |= 4;
        }
        return bits;
    }

    // GIL released by binding call_guard. Zero heap allocations inside STM closures.
    int64_t begin_layer(int k, int64_t pass_num) {
        check_layer(k);
        for (;;) {
            uint64_t base_seq = 0;
            bool writing_now = false;
            std::shared_ptr<const Vec64> base_in_use;
            {
                LayerSnap snap(*layers_[k]);
                const auto& p = snap[*layers_[k]];
                writing_now = p.writing;
                base_seq = p.lease_seq;
                base_in_use = p.in_use;
                if (!writing_now && base_in_use &&
                    std::find(base_in_use->begin(), base_in_use->end(), pass_num) != base_in_use->end()) {
                    return p.version;
                }
            }
            if (writing_now) {
                std::unique_lock<std::mutex> lk(hub_->mu);
                hub_->cv.wait_for(lk, std::chrono::microseconds(200));
                continue;
            }
            auto next_in_use = std::make_shared<Vec64>(base_in_use ? *base_in_use : Vec64{});
            next_in_use->push_back(pass_num);
            int64_t ver = 0;
            bool committed = false;
            layers_[k]->iterate_commit_while([&](LayerTx& tr) {
                committed = false;
                auto& p = tr[*layers_[k]];
                if (p.writing || p.lease_seq != base_seq) {
                    return false;
                }
                p.in_use = next_in_use;
                p.lease_seq += 1;
                ver = p.version;
                committed = true;
                return true;
            });
            if (committed) {
                return ver;
            }
        }
    }

    void release_layer(int k, int64_t pass_num) {
        check_layer(k);
        bool became_empty = false;
        for (;;) {
            uint64_t base_seq = 0;
            std::shared_ptr<const Vec64> base_in_use;
            {
                LayerSnap snap(*layers_[k]);
                const auto& p = snap[*layers_[k]];
                base_seq = p.lease_seq;
                base_in_use = p.in_use;
            }
            if (!base_in_use ||
                std::find(base_in_use->begin(), base_in_use->end(), pass_num) == base_in_use->end()) {
                return;
            }
            auto next_in_use = std::make_shared<Vec64>();
            next_in_use->reserve(base_in_use->size());
            for (int64_t x : *base_in_use) {
                if (x != pass_num) next_in_use->push_back(x);
            }
            bool committed = false;
            layers_[k]->iterate_commit_while([&](LayerTx& tr) {
                committed = false;
                became_empty = false;
                auto& p = tr[*layers_[k]];
                if (p.lease_seq != base_seq) {
                    return false;
                }
                p.in_use = next_in_use;
                p.lease_seq += 1;
                became_empty = next_in_use->empty();
                committed = true;
                return true;
            });
            if (committed) {
                if (became_empty) {
                    std::lock_guard<std::mutex> lk(hub_->mu);
                    hub_->cv.notify_all();
                }
                return;
            }
        }
    }

    void release(int64_t pass_num, int keep_k = -1) {
        for (int k = 0; k < nlayers_; ++k) {
            if (k == keep_k) continue;
            release_layer(k, pass_num);
        }
    }

    void release_all() {
        auto empty_vec = std::make_shared<Vec64>();
        bool woke = false;
        for (int k = 0; k < nlayers_; ++k) {
            bool cleared = false;
            layers_[k]->iterate_commit_while([&](LayerTx& tr) {
                cleared = false;
                auto& p = tr[*layers_[k]];
                if (!p.in_use || p.in_use->empty()) {
                    return false;
                }
                p.in_use = empty_vec;
                p.lease_seq += 1;
                cleared = true;
                return true;
            });
            if (cleared) woke = true;
        }
        if (woke) {
            std::lock_guard<std::mutex> lk(hub_->mu);
            hub_->cv.notify_all();
        }
    }

    mx::array pass_token(int64_t pass_num) {
        hub_->token_created(pass_num);
        void* buf = std::malloc(1);
        if (!buf) {
            hub_->token_deleted(pass_num);
            throw std::bad_alloc();
        }
        *static_cast<uint8_t*>(buf) = 1;
        std::weak_ptr<PassesHub> weak_hub = hub_;
        return mx::array(buf, {1}, mx::uint8, [weak_hub, pass_num](void* p) noexcept {
            std::free(p);
            if (auto h = weak_hub.lock()) {
                h->token_deleted(pass_num);
            }
        });
    }

    int completed() {
        auto passes = hub_->drain_retired();
        for (int64_t pnum : passes) {
            release(pnum, -1);
        }
        return static_cast<int>(passes.size());
    }

    bool is_completed(int64_t pass_num) const {
        if (hub_->has_live_token(pass_num)) {
            return false;
        }
        for (int k = 0; k < nlayers_; ++k) {
            LayerSnap snap(*layers_[k]);
            const auto& in_use = snap[*layers_[k]].in_use;
            if (in_use && std::find(in_use->begin(), in_use->end(), pass_num) != in_use->end()) {
                return false;
            }
        }
        return true;
    }

    int64_t held_back() const noexcept {
        return hub_->waiters.load(std::memory_order_acquire);
    }

    Vec64 active_passes(int k) const {
        check_layer(k);
        LayerSnap snap(*layers_[k]);
        const auto& iu = snap[*layers_[k]].in_use;
        return iu ? *iu : Vec64{};
    }

    void set_now(int64_t pass_num) noexcept {
        now_.store(pass_num, std::memory_order_release);
    }

    // -----------------------------------------------------------------------
    // Hardware / reader / memory hooks
    // -----------------------------------------------------------------------

    void attach_cuda(
        uintptr_t ctx, uintptr_t push_ctx, uintptr_t pop_ctx,
        uintptr_t stream_create, uintptr_t memcpy_async,
        uintptr_t stream_sync, uintptr_t stream_destroy) {
        std::lock_guard<std::mutex> lk(hw_mu_);
        cuda_.ctx = ctx;
        cuda_.push_ctx = reinterpret_cast<int (*)(uintptr_t)>(push_ctx);
        cuda_.pop_ctx = reinterpret_cast<int (*)(uintptr_t*)>(pop_ctx);
        cuda_.stream_create = reinterpret_cast<int (*)(uintptr_t*, unsigned int)>(stream_create);
        cuda_.memcpy_async = reinterpret_cast<int (*)(uintptr_t, uintptr_t, size_t, uintptr_t)>(memcpy_async);
        cuda_.stream_sync = reinterpret_cast<int (*)(uintptr_t)>(stream_sync);
        cuda_.stream_destroy = reinterpret_cast<int (*)(uintptr_t)>(stream_destroy);
    }

    void attach_host(
        uintptr_t alloc_fn, uintptr_t free_fn,
        uintptr_t reg_fn, uintptr_t unreg_fn, uintptr_t get_dev_fn) {
        std::lock_guard<std::mutex> lk(hw_mu_);
        host_.alloc = reinterpret_cast<void* (*)(size_t, size_t)>(alloc_fn);
        host_.free_fn = reinterpret_cast<void (*)(void*)>(free_fn);
        host_.host_reg = reinterpret_cast<int (*)(void*, size_t, unsigned int)>(reg_fn);
        host_.host_unreg = reinterpret_cast<int (*)(void*)>(unreg_fn);
        host_.host_get_dev = reinterpret_cast<int (*)(uint64_t*, void*, unsigned int)>(get_dev_fn);
    }

    void attach_readers(
        uintptr_t handle, uintptr_t submit, uintptr_t status,
        uintptr_t release, uintptr_t times, uintptr_t backlog, int workers) {
        std::lock_guard<std::mutex> lk(hw_mu_);
        readers_.handle = reinterpret_cast<void*>(handle);
        readers_.submit = reinterpret_cast<int64_t (*)(void*, int, int, int, const int32_t*, const int64_t*, const int64_t*, const uint64_t*)>(submit);
        readers_.status = reinterpret_cast<int32_t (*)(void*, int64_t)>(status);
        readers_.release = reinterpret_cast<void (*)(void*, int64_t)>(release);
        readers_.times = reinterpret_cast<void (*)(void*, int64_t, int64_t*)>(times);
        readers_.backlog = reinterpret_cast<int64_t (*)(void*)>(backlog);
        readers_.workers = workers;
    }

    void register_quant(
        int k, int q,
        std::optional<mx::array> weight,
        std::optional<mx::array> scales,
        std::optional<mx::array> biases,
        int bits, int group_size, const std::string& mode,
        std::tuple<int, int, int> indices) {
        check_layer(k);
        if (q < 0 || q >= 3) {
            throw std::out_of_range("invalid projection index");
        }
        std::optional<mx::array> arrs[3] = {weight, scales, biases};
        for (int m = 0; m < 3; ++m) {
            if (arrs[m].has_value()) {
                if (arrs[m]->status() == mx::array::Status::unscheduled) {
                    arrs[m]->eval();
                }
                arrs[m]->wait();
                if (arrs[m]->shape(0) > 0 && !arrs[m]->flags().row_contiguous) {
                    throw std::invalid_argument("register_quant: array must be row-contiguous");
                }
            }
        }
        std::lock_guard<std::mutex> lk(hw_mu_);
        auto& qs = layer_quants_[k][q];
        qs.valid = true;
        qs.bits = bits;
        qs.group_size = group_size;
        qs.mode = mode;
        qs.tensor_idx[0] = std::get<0>(indices);
        qs.tensor_idx[1] = std::get<1>(indices);
        qs.tensor_idx[2] = std::get<2>(indices);
        for (int m = 0; m < 3; ++m) {
            qs.arrays[m] = arrs[m];
            if (arrs[m].has_value() && arrs[m]->shape(0) > 0) {
                int64_t rows = arrs[m]->shape(0);
                qs.num_slots[m] = rows;
                qs.row_bytes[m] = arrs[m]->nbytes() / static_cast<size_t>(rows);
                qs.base_ptr[m] = reinterpret_cast<uintptr_t>(arrs[m]->data<void>());
            } else {
                qs.num_slots[m] = 0;
                qs.row_bytes[m] = 0;
                qs.base_ptr[m] = 0;
            }
        }
    }

    void drop_quant(int k) {
        check_layer(k);
        drop_layer_arrays(k);
    }

    void register_tensor(
        int k, int tidx,
        nb::ndarray<const int32_t, nb::c_contig, nb::device::cpu> files,
        nb::ndarray<const int64_t, nb::c_contig, nb::device::cpu> offsets,
        int64_t row_bytes) {
        check_layer(k);
        if (tidx < 0 || files.size() != offsets.size()) {
            throw std::invalid_argument("invalid tensor index or file/offset length mismatch");
        }
        std::lock_guard<std::mutex> lk(hw_mu_);
        auto& ts = layer_tensors_[k];
        if (static_cast<size_t>(tidx) >= ts.size()) {
            ts.resize(tidx + 1);
        }
        ts[tidx].valid = true;
        ts[tidx].files.assign(files.data(), files.data() + files.size());
        ts[tidx].offsets.assign(offsets.data(), offsets.data() + offsets.size());
        ts[tidx].row_bytes = row_bytes;
    }

    // -----------------------------------------------------------------------
    // GPU work-room budget for concurrent request leases (RoomNode)
    // -----------------------------------------------------------------------

    void room_put(int64_t bytes) {
        room_->iterate_commit([&](RoomTx& tr) {
            tr[*room_].room += bytes;
        });
        {
            std::lock_guard<std::mutex> lk(room_mu_);
            room_seq_.fetch_add(1, std::memory_order_release);
        }
        room_cv_.notify_all();
    }

    bool room_take(int64_t bytes, bool overdraw = false, double timeout_ms = 0.0) {
        const auto deadline = (timeout_ms > 0.0)
            ? (std::chrono::steady_clock::now() + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                std::chrono::duration<double, std::milli>(timeout_ms)))
            : std::chrono::steady_clock::time_point::min();
        for (;;) {
            uint64_t seen = room_seq_.load(std::memory_order_acquire);
            bool ok = false;
            room_->iterate_commit_while([&](RoomTx& tr) {
                ok = false;
                auto& p = tr[*room_];
                if (!overdraw && p.room < bytes) {
                    return false;
                }
                p.room -= bytes;
                ok = true;
                return true;
            });
            if (ok) {
                {
                    std::lock_guard<std::mutex> lk(room_mu_);
                    room_seq_.fetch_add(1, std::memory_order_release);
                }
                room_cv_.notify_one();
                return true;
            }
            if (timeout_ms <= 0.0 || stop_requested_.load(std::memory_order_acquire)
                || std::chrono::steady_clock::now() >= deadline) {
                return false;
            }
            std::unique_lock<std::mutex> lk(room_mu_);
            auto wake_at = std::min(deadline, std::chrono::steady_clock::now() + std::chrono::milliseconds(5));
            room_cv_.wait_until(lk, wake_at, [&] {
                return room_seq_.load(std::memory_order_acquire) != seen
                    || stop_requested_.load(std::memory_order_acquire);
            });
        }
    }

    void room_give(int64_t bytes) {
        room_->iterate_commit([&](RoomTx& tr) {
            tr[*room_].room += bytes;
        });
        {
            std::lock_guard<std::mutex> lk(room_mu_);
            room_seq_.fetch_add(1, std::memory_order_release);
        }
        room_cv_.notify_all();
    }

    int64_t room_free() const {
        RoomSnap snap(*room_);
        return snap[*room_].room;
    }

    nb::list arrays_of(int k) const {
        check_layer(k);
        std::lock_guard<std::mutex> lk(hw_mu_);
        nb::list out;
        for (int q = 0; q < 3; ++q) {
            for (int m = 0; m < 3; ++m) {
                const auto& opt = layer_quants_[k][q].arrays[m];
                if (opt.has_value()) {
                    out.append(nb::cast(*opt));
                } else {
                    out.append(nb::none());
                }
            }
        }
        return out;
    }

    // -----------------------------------------------------------------------
    // Asynchronous landing engine
    // -----------------------------------------------------------------------

    void start_lander(double wait_timeout_ms = 5000.0) {
        std::lock_guard<std::mutex> lk(lander_mu_);
        if (running_.load(std::memory_order_acquire)) {
            return;
        }
        wait_timeout_ms_ = wait_timeout_ms;
        stop_requested_.store(false, std::memory_order_release);
        running_.store(true, std::memory_order_release);
        lander_thread_ = std::thread([this]() { lander_loop(); });
    }

    void stop_lander() {
        std::thread to_join;
        {
            std::lock_guard<std::mutex> lk(lander_mu_);
            if (running_.load(std::memory_order_acquire) || lander_thread_.joinable()) {
                stop_requested_.store(true, std::memory_order_release);
                running_.store(false, std::memory_order_release);
                lander_cv_.notify_all();
                {
                    std::lock_guard<std::mutex> hlk(hub_->mu);
                    hub_->cv.notify_all();
                }
                {
                    std::lock_guard<std::mutex> tlk(turn_mu_);
                    turn_cv_.notify_all();
                }
                {
                    std::lock_guard<std::mutex> rlk(room_mu_);
                    room_seq_.fetch_add(1, std::memory_order_release);
                }
                room_cv_.notify_all();
                to_join = std::move(lander_thread_);
            }
        }
        if (to_join.joinable()) {
            if (PyGILState_Check()) {
                nb::gil_scoped_release release_gil;
                to_join.join();
            } else {
                to_join.join();
            }
        }
        destroy_sync_stream();
        for (int k = 0; k < nlayers_; ++k) {
            drop_layer_arrays(k);
        }
    }

    bool submit_landing(
        int k,
        const Vec64& slots,
        const Vec64& experts,
        int priority = 1) {
        check_layer(k);
        if (slots.size() != experts.size()) {
            throw std::invalid_argument("slots and experts must have equal length");
        }
        if (slots.empty() || !running_.load(std::memory_order_acquire)) {
            return false;
        }
        Vec64 accepted_slots;
        Vec64 accepted_experts;
        for (;;) {
            accepted_slots.clear();
            accepted_experts.clear();
            uint64_t base_seq = 0;
            std::shared_ptr<const Vec64> base_inflight;
            {
                LayerSnap snap(*layers_[k]);
                const auto& p = snap[*layers_[k]];
                base_seq = p.lease_seq;
                base_inflight = p.inflight;
            }
            for (size_t i = 0; i < experts.size(); ++i) {
                int64_t e = experts[i];
                if (e < 0) continue;
                if (base_inflight &&
                    std::find(base_inflight->begin(), base_inflight->end(), e) != base_inflight->end()) {
                    continue;
                }
                if (std::find(accepted_experts.begin(), accepted_experts.end(), e) != accepted_experts.end()) {
                    continue;
                }
                accepted_slots.push_back(slots[i]);
                accepted_experts.push_back(e);
            }
            if (accepted_experts.empty()) {
                return false;
            }
            auto next_inflight = std::make_shared<Vec64>(base_inflight ? *base_inflight : Vec64{});
            next_inflight->insert(next_inflight->end(), accepted_experts.begin(), accepted_experts.end());
            bool committed = false;
            layers_[k]->iterate_commit_while([&](LayerTx& tr) {
                committed = false;
                auto& p = tr[*layers_[k]];
                if (p.lease_seq != base_seq) {
                    return false;
                }
                p.inflight = next_inflight;
                p.lease_seq += 1;
                committed = true;
                return true;
            });
            if (committed) {
                break;
            }
        }
        {
            std::lock_guard<std::mutex> lk(lander_mu_);
            submits_count_.fetch_add(1, std::memory_order_relaxed);
            land_queue_.push_back(LandingJob{k, std::move(accepted_slots), std::move(accepted_experts), priority});
        }
        lander_cv_.notify_one();
        return true;
    }

    std::tuple<int64_t, int64_t, int64_t, int64_t> landings() const {
        std::lock_guard<std::mutex> lk(lander_mu_);
        return {
            cnt_landed_.load(std::memory_order_acquire),
            cnt_not_fit_.load(std::memory_order_acquire),
            cnt_failed_.load(std::memory_order_acquire),
            static_cast<int64_t>(land_queue_.size()),
        };
    }

    std::vector<std::tuple<int, int64_t, int64_t, int, int64_t, int64_t>> landed() {
        std::lock_guard<std::mutex> lk(records_mu_);
        std::vector<std::tuple<int, int64_t, int64_t, int, int64_t, int64_t>> out;
        out.reserve(landed_records_.size());
        for (const auto& r : landed_records_) {
            out.emplace_back(r.k, r.slot, r.expert, r.status, r.read_ns, r.copy_ns);
        }
        landed_records_.clear();
        return out;
    }

    int64_t write_failed() const noexcept {
        return cnt_write_failed_.load(std::memory_order_acquire);
    }

    nb::dict stats() const {
        nb::dict d;
        d["submits"] = submits_count_.load(std::memory_order_acquire);
        d["landed"] = cnt_landed_.load(std::memory_order_acquire);
        d["not_fit"] = cnt_not_fit_.load(std::memory_order_acquire);
        d["failed"] = cnt_failed_.load(std::memory_order_acquire);
        d["write_failed"] = cnt_write_failed_.load(std::memory_order_acquire);
        d["read_ns"] = total_read_ns_.load(std::memory_order_acquire);
        d["copy_ns"] = total_copy_ns_.load(std::memory_order_acquire);
        d["wait_ns"] = total_wait_ns_.load(std::memory_order_acquire);
        d["tx_commits"] = total_commits();
        return d;
    }

    uint64_t total_commits() const noexcept {
        uint64_t total = 0;
        for (const auto& l : layers_) {
            total += l->numTransactionsCommitted();
        }
        return total;
    }

    // Synchronous parallel C++ landing for `slots` <- `experts` on layer `k`.
    // Temporarily releases `pass_num`'s own lease on layer `k` while waiting for all
    // other passes to finish reading, reads all experts concurrently through `readers_`,
    // copies into registered slot buffers, commits STM metadata, and re-acquires `pass_num`.
    // Returns (ok, version, read_ns, copy_ns, count).
    std::tuple<bool, int64_t, int64_t, int64_t, int64_t> land_now(
        int k, int64_t pass_num,
        const Vec64& slots,
        const Vec64& experts,
        double timeout_ms = 5000.0) {
        check_layer(k);
        if (slots.size() != experts.size()) {
            throw std::invalid_argument("slots and experts must have equal length");
        }
        const size_t M = slots.size();
        if (M == 0) {
            return {true, __atomic_load_n(&hints_[k], __ATOMIC_ACQUIRE), 0, 0, 0};
        }

        if (pass_num >= 0) {
            release_layer(k, pass_num);
        }
        auto reacquire = [&]() -> int64_t {
            if (pass_num >= 0) {
                return begin_layer(k, pass_num);
            }
            return -1;
        };

        ReaderFns r;
        HostFns h;
        bool use_cuda = false;
        std::vector<std::vector<int32_t>> all_files(M);
        std::vector<Vec64> all_offsets(M);
        std::vector<Vec64> all_lengths(M);
        std::vector<std::vector<size_t>> seg_offsets(M);
        std::vector<std::vector<size_t>> seg_sizes(M);
        std::vector<std::vector<std::pair<int, int>>> seg_targets(M);
        std::vector<size_t> total_bytes(M, 0);
        {
            std::lock_guard<std::mutex> lk(hw_mu_);
            r = readers_;
            h = host_;
            use_cuda = cuda_.attached();
            if (!r.attached()) {
                cnt_failed_.fetch_add(static_cast<int64_t>(M), std::memory_order_release);
                reacquire();
                return {false, -1, 0, 0, 0};
            }
            for (size_t b = 0; b < M; ++b) {
                if (!prepare_expert_segments_locked(
                        k, experts[b], all_files[b], all_offsets[b], all_lengths[b],
                        seg_offsets[b], seg_sizes[b], seg_targets[b], total_bytes[b])) {
                    cnt_failed_.fetch_add(static_cast<int64_t>(M), std::memory_order_release);
                    reacquire();
                    return {false, -1, 0, 0, 0};
                }
            }
        }

        std::vector<HostBlock> blocks(M);
        std::vector<int64_t> tickets(M, -1);
        struct BatchTicketGuard {
            ReaderFns* r;
            std::vector<int64_t>* tickets;
            ~BatchTicketGuard() {
                if (!r || !r->attached() || !tickets) return;
                for (int64_t& tok : *tickets) {
                    if (tok < 0) continue;
                    unsigned sp = 10;
                    for (;;) {
                        int32_t st = r->status(r->handle, tok);
                        if (st == RD_DONE || st == RD_FAILED || st == RD_CANCELLED) break;
                        std::this_thread::sleep_for(std::chrono::microseconds(sp));
                        if (sp < 250) sp *= 2;
                    }
                    r->release(r->handle, tok);
                    tok = -1;
                }
            }
        } ticket_guard{&r, &tickets};

        int64_t t_read0 = monotonic_ns();
        bool submit_ok = true;
        for (size_t b = 0; b < M; ++b) {
            blocks[b] = HostBlock::allocate(total_bytes[b], h, use_cuda);
            std::vector<uint64_t> dsts(seg_offsets[b].size());
            uint8_t* base = static_cast<uint8_t*>(blocks[b].ptr);
            for (size_t i = 0; i < seg_offsets[b].size(); ++i) {
                dsts[i] = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(base + seg_offsets[b][i]));
            }
            // ticket == 0 is VALID; < 0 indicates error or refusal.
            int64_t tok = r.submit(
                r.handle, 0 /* waited */, 1 /* cache */,
                static_cast<int>(all_files[b].size()),
                all_files[b].data(), all_offsets[b].data(), all_lengths[b].data(), dsts.data());
            if (tok < 0) {
                submit_ok = false;
                break;
            }
            tickets[b] = tok;
        }

        bool read_ok = submit_ok;
        unsigned sleep_us = 25;
        for (;;) {
            bool all_done = true;
            for (size_t b = 0; b < M; ++b) {
                if (tickets[b] < 0) continue;
                int32_t st = r.status(r.handle, tickets[b]);
                if (st == RD_DONE) {
                    continue;
                }
                if (st == RD_FAILED || st == RD_CANCELLED) {
                    read_ok = false;
                    continue;
                }
                all_done = false;
            }
            if (all_done) {
                break;
            }
            if (stop_requested_.load(std::memory_order_acquire)) {
                read_ok = false;
            }
            std::this_thread::sleep_for(std::chrono::microseconds(sleep_us));
            if (sleep_us < 250) sleep_us *= 2;
        }
        for (int64_t& tok : tickets) {
            if (tok >= 0) {
                r.release(r.handle, tok);
                tok = -1;
            }
        }

        int64_t read_ns = std::max<int64_t>(0, monotonic_ns() - t_read0);
        total_read_ns_.fetch_add(read_ns, std::memory_order_relaxed);
        if (!read_ok) {
            cnt_failed_.fetch_add(static_cast<int64_t>(M), std::memory_order_release);
            reacquire();
            return {false, -1, read_ns, 0, 0};
        }

        int64_t wait_ns = 0;
        if (!wait_and_enter_write_batch(k, slots, timeout_ms, wait_ns)) {
            cnt_not_fit_.fetch_add(static_cast<int64_t>(M), std::memory_order_release);
            reacquire();
            return {false, -1, read_ns, 0, 0};
        }

        int64_t copy_ns = 0;
        if (!copy_segments_batch(k, slots, blocks, seg_offsets, seg_sizes, seg_targets, 0, true, copy_ns)) {
            clear_writing(k);
            cnt_failed_.fetch_add(static_cast<int64_t>(M), std::memory_order_release);
            reacquire();
            return {false, -1, read_ns, copy_ns, 0};
        }

        int64_t new_ver = commit_slots(k, slots, experts, pass_num);
        cnt_landed_.fetch_add(static_cast<int64_t>(M), std::memory_order_release);
        return {true, new_ver, read_ns, copy_ns, static_cast<int64_t>(M)};
    }

    // Zero-GIL per-layer decode routing + LRU weight update + synchronous STM miss landing.
    // Returns (status, needed_ids, hits, misses, version, read_s):
    //   status 0: ready for decode compute (`hits`, `misses`, `version`, `read_s` populated)
    //   status 1: needed.size() > num_slots -> caller runs chunked prefill path (weights updated)
    //   status 2: landing fell back -> caller runs Python miss path (weights updated, layer held)
    std::tuple<int, Vec64, int64_t, int64_t, int64_t, double> demand_step(
        int k, int64_t pass_num,
        mx::array indices,
        std::optional<mx::array> scores,
        nb::ndarray<double, nb::c_contig, nb::device::cpu> weight_arr,
        double decay,
        double timeout_ms = 5000.0,
        std::optional<mx::array> prev_out = std::nullopt) {
        check_layer(k);
        double* w_ptr = weight_arr.data();
        const size_t w_len = weight_arr.size();

        std::vector<mx::array> evals;
        evals.reserve(3);
        evals.push_back(indices);
        if (scores.has_value()) {
            if (scores->size() != indices.size()) {
                throw std::invalid_argument("demand_step: indices and scores shapes differ");
            }
            evals.push_back(*scores);
        }
        if (prev_out.has_value()) {
            evals.push_back(*prev_out);
        }
        mx::eval(evals);
        indices.wait();
        if (scores.has_value()) {
            scores->wait();
        }
        if (prev_out.has_value()) {
            prev_out->wait();
        }
        if (pass_num >= 0) {
            release(pass_num, -1);
        }

        std::vector<int64_t> ids;
        std::vector<double> sc;
        host_ints(indices, ids);
        if (scores.has_value()) {
            host_doubles(*scores, sc);
        } else {
            sc.assign(ids.size(), 1.0);
        }

        for (int64_t id : ids) {
            if (id < 0 || static_cast<size_t>(id) >= w_len) {
                throw std::out_of_range("demand_step: expert index out of range");
            }
        }
        for (size_t q = 0; q < w_len; ++q) {
            w_ptr[q] *= decay;
        }
        for (size_t q = 0; q < ids.size(); ++q) {
            w_ptr[static_cast<size_t>(ids[q])] += sc[q];
        }

        Vec64 needed = ids;
        std::sort(needed.begin(), needed.end());
        needed.erase(std::unique(needed.begin(), needed.end()), needed.end());

        size_t num_slots = 0;
        {
            LayerSnap snap(*layers_[k]);
            const auto& p = snap[*layers_[k]];
            if (p.experts) {
                num_slots = p.experts->size();
            }
        }
        if (needed.size() > num_slots) {
            return {1, std::move(needed), 0, 0, -1, 0.0};
        }

        int64_t ver = begin_layer(k, pass_num);
        std::shared_ptr<const Vec64> route;
        std::shared_ptr<const Vec64> experts_ptr;
        {
            LayerSnap snap(*layers_[k]);
            const auto& p = snap[*layers_[k]];
            ver = p.version;
            route = p.route;
            experts_ptr = p.experts;
        }
        if (!route || !experts_ptr) {
            return {2, std::move(needed), 0, 0, ver, 0.0};
        }

        Vec64 present, missing;
        present.reserve(needed.size());
        missing.reserve(needed.size());
        for (int64_t e : needed) {
            if (e >= 0 && static_cast<size_t>(e) < route->size() && (*route)[static_cast<size_t>(e)] >= 0) {
                present.push_back(e);
            } else {
                missing.push_back(e);
            }
        }
        const int64_t hits = static_cast<int64_t>(present.size());
        const int64_t misses = static_cast<int64_t>(missing.size());
        if (missing.empty()) {
            return {0, Vec64{}, hits, 0, ver, 0.0};
        }

        int64_t t0 = monotonic_ns();
        const auto& experts = *experts_ptr;
        const size_t n_slots = experts.size();
        std::vector<uint8_t> free_mask(n_slots, 1);
        for (int64_t e : present) {
            int64_t slot = (*route)[static_cast<size_t>(e)];
            if (slot >= 0 && static_cast<size_t>(slot) < n_slots) {
                free_mask[static_cast<size_t>(slot)] = 0;
            }
        }
        Vec64 cand;
        cand.reserve(n_slots);
        for (size_t s = 0; s < n_slots; ++s) {
            if (free_mask[s]) {
                cand.push_back(static_cast<int64_t>(s));
            }
        }
        if (cand.size() < missing.size()) {
            return {2, std::move(needed), 0, 0, ver, 0.0};
        }
        std::stable_sort(cand.begin(), cand.end(), [&](int64_t a, int64_t b) {
            int64_t ea = experts[static_cast<size_t>(a)];
            int64_t eb = experts[static_cast<size_t>(b)];
            double wa = (ea >= 0 && static_cast<size_t>(ea) < w_len) ? w_ptr[ea] : 0.0;
            double wb = (eb >= 0 && static_cast<size_t>(eb) < w_len) ? w_ptr[eb] : 0.0;
            return wa < wb;
        });
        Vec64 victims(cand.begin(), cand.begin() + static_cast<ptrdiff_t>(missing.size()));
        auto [ok, new_ver, read_ns, copy_ns, landed_n] = land_now(k, pass_num, victims, missing, timeout_ms);
        double elapsed_s = static_cast<double>(std::max<int64_t>(0, monotonic_ns() - t0)) * 1e-9;
        if (!ok) {
            int64_t v = begin_layer(k, pass_num);
            return {2, std::move(needed), 0, 0, v, elapsed_s};
        }
        return {0, Vec64{}, hits, misses, new_ver, elapsed_s};
    }

    bool wait_idle(int k, double timeout_ms = 5000.0) {
        check_layer(k);
        auto deadline = std::chrono::steady_clock::now() + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
            std::chrono::duration<double, std::milli>(timeout_ms));
        TurnPauseScope pause_scope;
        for (;;) {
            completed();
            bool busy = false;
            {
                LayerSnap snap(*layers_[k]);
                const auto& p = snap[*layers_[k]];
                busy = (p.in_use && !p.in_use->empty()) || p.writing;
            }
            if (!busy) {
                return true;
            }
            if (stop_requested_.load(std::memory_order_acquire) ||
                std::chrono::steady_clock::now() >= deadline) {
                return false;
            }
            pause_scope.pause(this);
            struct WaitScope {
                std::atomic<int64_t>& w;
                explicit WaitScope(std::atomic<int64_t>& a) : w(a) { w.fetch_add(1, std::memory_order_release); }
                ~WaitScope() { w.fetch_sub(1, std::memory_order_release); }
            } wait_scope{hub_->waiters};

            std::unique_lock<std::mutex> lk(hub_->mu);
            hub_->cv.wait_until(lk, std::min(deadline, std::chrono::steady_clock::now() + std::chrono::milliseconds(5)));
        }
    }

    static int64_t now_ns() noexcept {
        return monotonic_ns();
    }

    int64_t turn_take(int kind, int64_t birth = 0, double timeout_ms = 2000.0) {
        if (birth <= 0) {
            birth = monotonic_ns();
        }
        if (tl_turn.owner == id_ && tl_turn.holding) {
            turn_give();
        }
        const int64_t start = monotonic_ns();
        const auto deadline = (timeout_ms > 0.0)
            ? (std::chrono::steady_clock::now() + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                std::chrono::duration<double, std::milli>(timeout_ms)))
            : std::chrono::steady_clock::time_point::max();
        int slot = -1;
        bool forced = false;
        bool waited = false;
        if (!turn_attempt(kind, birth, slot, true, false)) {
            waited = true;
            std::unique_lock<std::mutex> lk(turn_mu_);
            for (;;) {
                const bool force = stop_requested_.load(std::memory_order_acquire)
                    || (timeout_ms > 0.0 && std::chrono::steady_clock::now() >= deadline);
                if (turn_attempt(kind, birth, slot, true, force)) {
                    forced = force;
                    break;
                }
                const auto wake_at = (timeout_ms > 0.0)
                    ? std::min(deadline, std::chrono::steady_clock::now() + std::chrono::milliseconds(1))
                    : (std::chrono::steady_clock::now() + std::chrono::milliseconds(1));
                turn_cv_.wait_until(lk, wake_at);
            }
        }
        tl_turn = TurnHeld{id_, kind, birth, timeout_ms, true, false};
        const int64_t took = std::max<int64_t>(0, monotonic_ns() - start);
        turn_counts_[kind == TURN_PROMPT ? 0 : 1].fetch_add(1, std::memory_order_relaxed);
        if (waited) {
            turn_counts_[kind == TURN_PROMPT ? 2 : 3].fetch_add(1, std::memory_order_relaxed);
            turn_counts_[4].fetch_add(took, std::memory_order_relaxed);
        }
        if (forced) {
            turn_counts_[5].fetch_add(1, std::memory_order_relaxed);
        }
        return forced ? -1 : took;
    }

    bool turn_try(int kind, int64_t birth = 0) {
        if (birth <= 0) {
            birth = monotonic_ns();
        }
        if (tl_turn.owner == id_ && tl_turn.holding) {
            turn_give();
        }
        int slot = -1;
        if (!turn_attempt(kind, birth, slot, false, false)) {
            return false;
        }
        tl_turn = TurnHeld{id_, kind, birth, 0.0, true, false};
        turn_counts_[kind == TURN_PROMPT ? 0 : 1].fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    void turn_give() {
        if (tl_turn.owner != id_ || !(tl_turn.holding || tl_turn.paused)) {
            return;
        }
        if (tl_turn.holding) {
            turn_let_go(tl_turn.kind);
        }
        tl_turn = TurnHeld{};
    }

    Vec64 turn_stats() const {
        Vec64 out;
        out.reserve(turn_counts_.size());
        for (const auto& c : turn_counts_) {
            out.push_back(c.load(std::memory_order_relaxed));
        }
        return out;
    }

    std::tuple<bool, bool, int32_t, int32_t> turn_state() const {
        TurnSnap snap(*turns_);
        const auto& p = snap[*turns_];
        int32_t waiting = 0;
        for (int64_t w : p.waits) {
            if (w > 0) waiting += 1;
        }
        return {p.prompt, p.prompt_wait > 0, p.decodes, waiting};
    }

private:
    void check_layer(int k) const {
        if (k < 0 || k >= nlayers_) {
            throw std::out_of_range("layer index out of range");
        }
    }

    void drop_layer_arrays(int k) {
        std::lock_guard<std::mutex> lk(hw_mu_);
        for (int q = 0; q < 3; ++q) {
            for (int m = 0; m < 3; ++m) {
                layer_quants_[k][q].arrays[m].reset();
                layer_quants_[k][q].base_ptr[m] = 0;
            }
        }
    }

    void clear_inflight(int k, int64_t expert) {
        for (;;) {
            uint64_t base_seq = 0;
            bool was_writing = false;
            std::shared_ptr<const Vec64> base_inflight;
            {
                LayerSnap snap(*layers_[k]);
                const auto& p = snap[*layers_[k]];
                base_seq = p.lease_seq;
                was_writing = p.writing;
                base_inflight = p.inflight;
            }
            bool has_expert = base_inflight &&
                std::find(base_inflight->begin(), base_inflight->end(), expert) != base_inflight->end();
            if (!has_expert && !was_writing) {
                return;
            }
            auto next_inflight = std::make_shared<Vec64>();
            if (base_inflight) {
                next_inflight->reserve(base_inflight->size());
                for (int64_t x : *base_inflight) {
                    if (x != expert) next_inflight->push_back(x);
                }
            }
            bool committed = false;
            layers_[k]->iterate_commit_while([&](LayerTx& tr) {
                committed = false;
                auto& p = tr[*layers_[k]];
                if (p.lease_seq != base_seq) {
                    return false;
                }
                p.inflight = next_inflight;
                p.writing = false;
                p.lease_seq += 1;
                committed = true;
                return true;
            });
            if (committed) {
                std::lock_guard<std::mutex> hlk(hub_->mu);
                hub_->cv.notify_all();
                return;
            }
        }
    }

    void record_outcome(int k, int64_t slot, int64_t expert, int status, int64_t read_ns, int64_t copy_ns) {
        if (status == 0) cnt_landed_.fetch_add(1, std::memory_order_release);
        else if (status == 1) cnt_not_fit_.fetch_add(1, std::memory_order_release);
        else cnt_failed_.fetch_add(1, std::memory_order_release);

        std::lock_guard<std::mutex> lk(records_mu_);
        if (landed_records_.size() < 4096) {
            landed_records_.push_back(LandedRecord{k, slot, expert, status, read_ns, copy_ns});
        }
    }

    void clear_writing(int k) {
        layers_[k]->iterate_commit([&](LayerTx& tr) {
            auto& p = tr[*layers_[k]];
            p.writing = false;
            p.lease_seq += 1;
        });
        std::lock_guard<std::mutex> hlk(hub_->mu);
        hub_->cv.notify_all();
    }

    void destroy_sync_stream() noexcept {
        CudaFns cu;
        {
            std::lock_guard<std::mutex> lk(hw_mu_);
            cu = cuda_;
        }
        std::lock_guard<std::mutex> slk(sync_cu_mu_);
        if (sync_cuda_stream_ && cu.attached()) {
            if (cu.push_ctx(cu.ctx) == 0) {
                cu.stream_destroy(sync_cuda_stream_);
                uintptr_t popped = 0;
                cu.pop_ctx(&popped);
            }
            sync_cuda_stream_ = 0;
        }
    }

    bool prepare_expert_segments_locked(
        int k, int64_t expert,
        std::vector<int32_t>& files, Vec64& offsets, Vec64& lengths,
        std::vector<size_t>& seg_offsets, std::vector<size_t>& seg_sizes,
        std::vector<std::pair<int, int>>& seg_targets, size_t& total_bytes) const {
        files.clear();
        offsets.clear();
        lengths.clear();
        seg_offsets.clear();
        seg_sizes.clear();
        seg_targets.clear();
        total_bytes = 0;
        const auto& gqs = layer_quants_[k];
        const auto& ts = layer_tensors_[k];
        bool any_tensor = false;
        for (int q = 0; q < 3; ++q) {
            if (!gqs[q].valid) continue;
            for (int m = 0; m < 3; ++m) {
                int tidx = gqs[q].tensor_idx[m];
                if (tidx < 0) continue;
                if (static_cast<size_t>(tidx) >= ts.size() || !ts[tidx].valid) {
                    return false;
                }
                if (expert < 0 || static_cast<size_t>(expert) >= ts[tidx].files.size()) {
                    return false;
                }
                size_t nbytes = static_cast<size_t>(ts[tidx].row_bytes);
                if (nbytes == 0 || nbytes != gqs[q].row_bytes[m]) {
                    return false;
                }
                any_tensor = true;
                files.push_back(ts[tidx].files[expert]);
                offsets.push_back(ts[tidx].offsets[expert]);
                lengths.push_back(ts[tidx].row_bytes);
                seg_offsets.push_back(total_bytes);
                seg_sizes.push_back(nbytes);
                seg_targets.emplace_back(q, m);
                total_bytes += ((nbytes + 63) / 64) * 64;
            }
        }
        return any_tensor;
    }

    bool read_expert_segments(
        int k, int64_t expert, int priority,
        HostBlock& block, std::vector<size_t>& seg_offsets,
        std::vector<size_t>& seg_sizes, std::vector<std::pair<int, int>>& seg_targets,
        int64_t& read_ns_out) {
        ReaderFns r;
        HostFns h;
        bool use_cuda = false;
        std::vector<int32_t> files;
        Vec64 offsets;
        Vec64 lengths;
        size_t total_bytes = 0;
        {
            std::lock_guard<std::mutex> lk(hw_mu_);
            r = readers_;
            h = host_;
            use_cuda = cuda_.attached();
            if (!prepare_expert_segments_locked(
                    k, expert, files, offsets, lengths, seg_offsets, seg_sizes, seg_targets, total_bytes)) {
                return false;
            }
        }
        if (!r.attached()) {
            return false;
        }
        block = HostBlock::allocate(total_bytes, h, use_cuda);
        std::vector<uint64_t> dsts(seg_offsets.size());
        uint8_t* base = static_cast<uint8_t*>(block.ptr);
        for (size_t i = 0; i < seg_offsets.size(); ++i) {
            dsts[i] = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(base + seg_offsets[i]));
        }

        int64_t t0 = monotonic_ns();
        // IMPORTANT: ticket == 0 is a VALID token; < 0 indicates error or refusal.
        int64_t ticket = r.submit(
            r.handle, priority, 1 /* cache */,
            static_cast<int>(files.size()), files.data(), offsets.data(), lengths.data(), dsts.data());
        if (ticket < 0) {
            return false;
        }
        struct TicketGuard {
            ReaderFns* r;
            int64_t tok;
            ~TicketGuard() {
                if (tok >= 0 && r && r->release) {
                    r->release(r->handle, tok);
                }
            }
        } guard{&r, ticket};

        unsigned sleep_us = 25;
        int32_t st = RD_QUEUED;
        for (;;) {
            st = r.status(r.handle, ticket);
            if (st == RD_DONE || st == RD_FAILED || st == RD_CANCELLED) {
                break;
            }
            if (stop_requested_.load(std::memory_order_acquire)) {
                // Drain before returning so HostBlock is never freed while worker writes.
                unsigned sp = 25;
                while (st != RD_DONE && st != RD_FAILED && st != RD_CANCELLED) {
                    std::this_thread::sleep_for(std::chrono::microseconds(sp));
                    if (sp < 250) sp *= 2;
                    st = r.status(r.handle, ticket);
                }
                return false;
            }
            std::this_thread::sleep_for(std::chrono::microseconds(sleep_us));
            if (sleep_us < 250) sleep_us *= 2;
        }
        if (r.times) {
            int64_t times_out[4] = {0, 0, 0, 0};
            r.times(r.handle, ticket, times_out);
            if (times_out[1] > times_out[0] && times_out[0] > 0) {
                read_ns_out = times_out[1] - times_out[0];
            } else {
                read_ns_out = std::max<int64_t>(0, monotonic_ns() - t0);
            }
        } else {
            read_ns_out = std::max<int64_t>(0, monotonic_ns() - t0);
        }
        total_read_ns_.fetch_add(read_ns_out, std::memory_order_relaxed);
        return st == RD_DONE;
    }

    bool wait_and_enter_write_batch(int k, const Vec64& slots, double timeout_ms, int64_t& wait_ns_out) {
        int64_t t0 = monotonic_ns();
        auto deadline = std::chrono::steady_clock::now() + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
            std::chrono::duration<double, std::milli>(timeout_ms));
        TurnPauseScope pause_scope;
        for (;;) {
            completed();
            bool entered = false;
            bool slot_valid = true;
            layers_[k]->iterate_commit_while([&](LayerTx& tr) {
                entered = false;
                slot_valid = true;
                auto& p = tr[*layers_[k]];
                if (!p.experts) {
                    slot_valid = false;
                    return false;
                }
                const size_t n_slots = p.experts->size();
                for (int64_t slot : slots) {
                    if (slot < 0 || static_cast<size_t>(slot) >= n_slots) {
                        slot_valid = false;
                        return false;
                    }
                }
                if ((p.in_use && !p.in_use->empty()) || p.writing) {
                    return false;
                }
                p.writing = true;
                entered = true;
                return true;
            });
            if (!slot_valid) {
                wait_ns_out = std::max<int64_t>(0, monotonic_ns() - t0);
                total_wait_ns_.fetch_add(wait_ns_out, std::memory_order_relaxed);
                return false;
            }
            if (entered) {
                wait_ns_out = std::max<int64_t>(0, monotonic_ns() - t0);
                total_wait_ns_.fetch_add(wait_ns_out, std::memory_order_relaxed);
                return true;
            }
            if (stop_requested_.load(std::memory_order_acquire) ||
                std::chrono::steady_clock::now() >= deadline) {
                wait_ns_out = std::max<int64_t>(0, monotonic_ns() - t0);
                total_wait_ns_.fetch_add(wait_ns_out, std::memory_order_relaxed);
                return false;
            }
            pause_scope.pause(this);
            struct WaitScope {
                std::atomic<int64_t>& w;
                explicit WaitScope(std::atomic<int64_t>& a) : w(a) { w.fetch_add(1, std::memory_order_release); }
                ~WaitScope() { w.fetch_sub(1, std::memory_order_release); }
            } wait_scope{hub_->waiters};

            std::unique_lock<std::mutex> lk(hub_->mu);
            hub_->cv.wait_until(lk, std::min(deadline, std::chrono::steady_clock::now() + std::chrono::milliseconds(5)));
        }
    }

    bool wait_and_enter_write(int k, int64_t slot, int64_t& wait_ns_out) {
        return wait_and_enter_write_batch(k, Vec64{slot}, wait_timeout_ms_, wait_ns_out);
    }

    bool copy_segments_batch(
        int k, const Vec64& slots,
        const std::vector<HostBlock>& blocks,
        const std::vector<std::vector<size_t>>& seg_offsets,
        const std::vector<std::vector<size_t>>& seg_sizes,
        const std::vector<std::vector<std::pair<int, int>>>& seg_targets,
        uintptr_t cuda_stream, bool use_sync_stream, int64_t& copy_ns_out) {
        bool any_target = false;
        for (const auto& t : seg_targets) {
            if (!t.empty()) {
                any_target = true;
                break;
            }
        }
        if (!any_target) {
            copy_ns_out = 0;
            return true;
        }
        int64_t t0 = monotonic_ns();
        CudaFns cu;
        std::array<QuantSlot, 3> gqs_copy;
        {
            std::lock_guard<std::mutex> lk(hw_mu_);
            cu = cuda_;
            gqs_copy = layer_quants_[k];
        }
        if (cu.attached() && (cuda_stream || use_sync_stream)) {
            std::unique_lock<std::mutex> slk(sync_cu_mu_, std::defer_lock);
            if (use_sync_stream) {
                slk.lock();
            }
            if (cu.push_ctx(cu.ctx) != 0) {
                return false;
            }
            struct CtxGuard {
                CudaFns* cu;
                ~CtxGuard() {
                    uintptr_t popped = 0;
                    cu->pop_ctx(&popped);
                }
            } pop_guard{&cu};
            uintptr_t stream = cuda_stream;
            if (use_sync_stream) {
                if (!sync_cuda_stream_ && cu.stream_create(&sync_cuda_stream_, 1 /* CU_STREAM_NON_BLOCKING */) != 0) {
                    return false;
                }
                stream = sync_cuda_stream_;
            }
            bool copied_any = false;
            for (size_t b = 0; b < slots.size(); ++b) {
                int64_t slot = slots[b];
                const uint8_t* base = static_cast<const uint8_t*>(blocks[b].ptr);
                for (size_t i = 0; i < seg_targets[b].size(); ++i) {
                    int q = seg_targets[b][i].first;
                    int m = seg_targets[b][i].second;
                    uintptr_t dst_base = gqs_copy[q].base_ptr[m];
                    size_t rb = gqs_copy[q].row_bytes[m];
                    if (!dst_base || slot < 0 || slot >= gqs_copy[q].num_slots[m] || rb != seg_sizes[b][i]) {
                        if (copied_any) {
                            cu.stream_sync(stream);
                            cnt_write_failed_.fetch_add(1, std::memory_order_release);
                        }
                        return false;
                    }
                    uintptr_t dst = dst_base + static_cast<size_t>(slot) * rb;
                    uintptr_t src = blocks[b].pinned
                        ? static_cast<uintptr_t>(blocks[b].dev_ptr + seg_offsets[b][i])
                        : reinterpret_cast<uintptr_t>(base + seg_offsets[b][i]);
                    if (cu.memcpy_async(dst, src, rb, stream) != 0) {
                        if (copied_any) {
                            cu.stream_sync(stream);
                            cnt_write_failed_.fetch_add(1, std::memory_order_release);
                        }
                        return false;
                    }
                    copied_any = true;
                }
            }
            if (cu.stream_sync(stream) != 0) {
                cnt_write_failed_.fetch_add(1, std::memory_order_release);
                return false;
            }
        } else {
            for (size_t b = 0; b < slots.size(); ++b) {
                int64_t slot = slots[b];
                const uint8_t* base = static_cast<const uint8_t*>(blocks[b].ptr);
                for (size_t i = 0; i < seg_targets[b].size(); ++i) {
                    int q = seg_targets[b][i].first;
                    int m = seg_targets[b][i].second;
                    uintptr_t dst_base = gqs_copy[q].base_ptr[m];
                    size_t rb = gqs_copy[q].row_bytes[m];
                    if (!dst_base || slot < 0 || slot >= gqs_copy[q].num_slots[m] || rb != seg_sizes[b][i]) {
                        return false;
                    }
                    uint8_t* dst = reinterpret_cast<uint8_t*>(dst_base) + static_cast<size_t>(slot) * rb;
                    std::memcpy(dst, base + seg_offsets[b][i], rb);
                }
            }
        }
        copy_ns_out = std::max<int64_t>(0, monotonic_ns() - t0);
        total_copy_ns_.fetch_add(copy_ns_out, std::memory_order_relaxed);
        return true;
    }

    // Build all updated shared_ptrs outside KAME's iterate_commit_while closure so zero
    // heap allocations occur inside the STM retry lambda.
    int64_t commit_slots(int k, const Vec64& slots, const Vec64& experts, int64_t pass_num) {
        for (;;) {
            int64_t base_ver = 0;
            uint64_t base_seq = 0;
            std::shared_ptr<const Vec64> cur_experts;
            std::shared_ptr<const Vec64> cur_route;
            std::shared_ptr<const VecU8> cur_absent;
            std::shared_ptr<const Vec64> cur_inflight;
            std::shared_ptr<const Vec64> cur_in_use;
            {
                LayerSnap snap(*layers_[k]);
                const auto& p = snap[*layers_[k]];
                base_ver = p.version;
                base_seq = p.lease_seq;
                cur_experts = p.experts;
                cur_route = p.route;
                cur_absent = p.absent;
                cur_inflight = p.inflight;
                cur_in_use = p.in_use;
            }

            auto new_experts = std::make_shared<Vec64>(cur_experts ? *cur_experts : Vec64{});
            auto new_route = std::make_shared<Vec64>(cur_route ? *cur_route : Vec64{});
            auto new_absent = std::make_shared<VecU8>(cur_absent ? *cur_absent : VecU8{});
            auto next_inflight = std::make_shared<Vec64>();
            if (cur_inflight) {
                next_inflight->reserve(cur_inflight->size());
                for (int64_t x : *cur_inflight) {
                    if (std::find(experts.begin(), experts.end(), x) == experts.end()) {
                        next_inflight->push_back(x);
                    }
                }
            }
            std::shared_ptr<Vec64> next_in_use;
            if (pass_num >= 0) {
                next_in_use = std::make_shared<Vec64>(cur_in_use ? *cur_in_use : Vec64{});
                if (std::find(next_in_use->begin(), next_in_use->end(), pass_num) == next_in_use->end()) {
                    next_in_use->push_back(pass_num);
                }
            }

            Vec64 evicted;
            evicted.reserve(slots.size());
            for (size_t idx = 0; idx < slots.size(); ++idx) {
                int64_t slot = slots[idx];
                int64_t expert = experts[idx];
                if (slot >= 0 && static_cast<size_t>(slot) < new_experts->size()) {
                    int64_t old = (*new_experts)[slot];
                    if (old >= 0) {
                        evicted.push_back(old);
                    }
                    (*new_experts)[slot] = expert;
                }
            }
            size_t E = new_route->size();
            for (int64_t ev : evicted) {
                if (ev >= 0 && static_cast<size_t>(ev) < E) {
                    (*new_route)[ev] = -1;
                    if (static_cast<size_t>(ev) < new_absent->size()) {
                        (*new_absent)[ev] = 1;
                    }
                }
            }
            for (size_t idx = 0; idx < slots.size(); ++idx) {
                int64_t slot = slots[idx];
                int64_t expert = experts[idx];
                if (expert >= 0 && static_cast<size_t>(expert) < E) {
                    (*new_route)[expert] = slot;
                    if (static_cast<size_t>(expert) < new_absent->size()) {
                        (*new_absent)[expert] = 0;
                    }
                }
            }

            int64_t new_version = 0;
            bool committed = false;
            layers_[k]->iterate_commit_while([&](LayerTx& tr) {
                committed = false;
                new_version = 0;
                auto& p = tr[*layers_[k]];
                if (p.version != base_ver || p.lease_seq != base_seq) {
                    return false;
                }
                p.experts = new_experts;
                p.route = new_route;
                p.absent = new_absent;
                p.inflight = next_inflight;
                if (next_in_use) {
                    p.in_use = next_in_use;
                }
                p.writing = false;
                p.lease_seq += 1;
                p.version += 1;
                new_version = p.version;
                committed = true;
                return true;
            });
            if (committed) {
                __atomic_store_n(&hints_[k], new_version, __ATOMIC_RELEASE);
                std::lock_guard<std::mutex> hlk(hub_->mu);
                hub_->cv.notify_all();
                return new_version;
            }
        }
    }

    void lander_loop() {
        uintptr_t cuda_stream = 0;
        CudaFns cu_init;
        {
            std::lock_guard<std::mutex> lk(hw_mu_);
            cu_init = cuda_;
        }
        if (cu_init.attached()) {
            if (cu_init.push_ctx(cu_init.ctx) == 0) {
                cu_init.stream_create(&cuda_stream, 1 /* CU_STREAM_NON_BLOCKING */);
                uintptr_t popped = 0;
                cu_init.pop_ctx(&popped);
            }
        }

        for (;;) {
            LandingJob job;
            {
                std::unique_lock<std::mutex> lk(lander_mu_);
                lander_cv_.wait(lk, [&] {
                    return stop_requested_.load(std::memory_order_acquire) || !land_queue_.empty();
                });
                if (stop_requested_.load(std::memory_order_acquire) && land_queue_.empty()) {
                    break;
                }
                job = std::move(land_queue_.front());
                land_queue_.pop_front();
            }

            for (size_t i = 0; i < job.experts.size(); ++i) {
                int64_t slot = job.slots[i];
                int64_t expert = job.experts[i];
                if (stop_requested_.load(std::memory_order_acquire)) {
                    clear_inflight(job.k, expert);
                    record_outcome(job.k, slot, expert, 2, 0, 0);
                    continue;
                }
                HostBlock block;
                std::vector<size_t> seg_offsets, seg_sizes;
                std::vector<std::pair<int, int>> seg_targets;
                int64_t read_ns = 0, copy_ns = 0, wait_ns = 0;
                if (!read_expert_segments(
                        job.k, expert, job.priority,
                        block, seg_offsets, seg_sizes, seg_targets, read_ns)) {
                    clear_inflight(job.k, expert);
                    record_outcome(job.k, slot, expert, 2, read_ns, 0);
                    continue;
                }
                if (!wait_and_enter_write(job.k, slot, wait_ns)) {
                    clear_inflight(job.k, expert);
                    record_outcome(job.k, slot, expert, 1, read_ns, 0);
                    continue;
                }
                std::vector<HostBlock> one_block;
                one_block.push_back(std::move(block));
                std::vector<std::vector<size_t>> one_off{std::move(seg_offsets)};
                std::vector<std::vector<size_t>> one_sz{std::move(seg_sizes)};
                std::vector<std::vector<std::pair<int, int>>> one_tgt{std::move(seg_targets)};
                if (!copy_segments_batch(
                        job.k, Vec64{slot}, one_block, one_off, one_sz, one_tgt,
                        cuda_stream, false, copy_ns)) {
                    clear_inflight(job.k, expert);
                    record_outcome(job.k, slot, expert, 2, read_ns, copy_ns);
                    continue;
                }
                commit_slots(job.k, Vec64{slot}, Vec64{expert}, -1);
                record_outcome(job.k, slot, expert, 0, read_ns, copy_ns);
            }
        }

        if (cuda_stream && cu_init.attached()) {
            if (cu_init.push_ctx(cu_init.ctx) == 0) {
                cu_init.stream_destroy(cuda_stream);
                uintptr_t popped = 0;
                cu_init.pop_ctx(&popped);
            }
        }
    }

    bool turn_attempt(int kind, int64_t birth, int& slot, bool record, bool force) {
        bool ok = false;
        int tried = slot;
        turns_->iterate_commit_while([&](TurnTx& tr) {
            auto& p = tr[*turns_];
            ok = false;
            tried = slot;
            if (kind == TURN_PROMPT) {
                bool older = false;
                for (int64_t w : p.waits) {
                    if (w > 0 && w < birth) {
                        older = true;
                        break;
                    }
                }
                if (force || (p.decodes == 0 && !older)) {
                    p.prompt = true;
                    p.prompt_wait = 0;
                    ok = true;
                    return true;
                }
                if (!record || p.prompt_wait == birth) {
                    return false;
                }
                p.prompt_wait = birth;
                return true;
            }
            if (force || (!p.prompt && !(p.prompt_wait > 0 && p.prompt_wait < birth))) {
                p.decodes += 1;
                if (tried >= 0 && static_cast<size_t>(tried) < TURN_SLOTS) {
                    p.waits[static_cast<size_t>(tried)] = 0;
                }
                ok = true;
                return true;
            }
            if (!record) {
                return false;
            }
            if (tried < 0) {
                for (size_t k = 0; k < TURN_SLOTS; ++k) {
                    if (p.waits[k] == 0) {
                        tried = static_cast<int>(k);
                        break;
                    }
                }
            }
            if (tried < 0 || p.waits[static_cast<size_t>(tried)] == birth) {
                return false;
            }
            p.waits[static_cast<size_t>(tried)] = birth;
            return true;
        });
        slot = ok ? -1 : tried;
        return ok;
    }

    void turn_let_go(int kind) {
        turns_->iterate_commit([&](TurnTx& tr) {
            auto& p = tr[*turns_];
            if (kind == TURN_PROMPT) {
                p.prompt = false;
            } else {
                p.decodes = std::max(0, p.decodes - 1);
            }
        });
        std::lock_guard<std::mutex> lk(turn_mu_);
        turn_cv_.notify_all();
    }

    void turn_pause() {
        if (tl_turn.owner != id_ || !tl_turn.holding) {
            return;
        }
        turn_let_go(tl_turn.kind);
        tl_turn.holding = false;
        tl_turn.paused = true;
    }

    void turn_resume() {
        if (tl_turn.owner != id_ || !tl_turn.paused) {
            return;
        }
        const TurnHeld was = tl_turn;
        tl_turn = TurnHeld{};
        turn_take(was.kind, was.birth, was.timeout_ms);
    }

    struct TurnPauseScope {
        Model* self{nullptr};
        bool paused{false};
        void pause(Model* m) {
            if (!paused && m && tl_turn.owner == m->id_ && tl_turn.holding) {
                self = m;
                paused = true;
                self->turn_pause();
            }
        }
        ~TurnPauseScope() {
            if (paused && self) {
                self->turn_resume();
            }
        }
    };

    const uint64_t id_{next_model_id.fetch_add(1, std::memory_order_relaxed)};
    int nlayers_{0};
    Vec64 hints_;
    std::vector<LayerPtr> layers_;
    TurnPtr turns_;
    RoomPtr room_;
    mutable std::mutex turn_mu_;
    std::condition_variable turn_cv_;
    mutable std::mutex room_mu_;
    std::condition_variable room_cv_;
    std::atomic<uint64_t> room_seq_{0};
    std::array<std::atomic<int64_t>, 6> turn_counts_{};

    mutable std::mutex hw_mu_;
    mutable std::mutex sync_cu_mu_;
    uintptr_t sync_cuda_stream_{0};
    CudaFns cuda_{};
    HostFns host_{};
    ReaderFns readers_{};
    std::vector<std::array<QuantSlot, 3>> layer_quants_;
    std::vector<std::vector<TensorSpec>> layer_tensors_;

    std::shared_ptr<PassesHub> hub_;
    std::atomic<int64_t> now_{0};

    mutable std::mutex lander_mu_;
    std::condition_variable lander_cv_;
    std::thread lander_thread_;
    std::deque<LandingJob> land_queue_;
    double wait_timeout_ms_{5000.0};
    std::atomic<bool> running_{false};
    std::atomic<bool> stop_requested_{false};

    mutable std::mutex records_mu_;
    std::vector<LandedRecord> landed_records_;

    std::atomic<int64_t> submits_count_{0};
    std::atomic<int64_t> cnt_landed_{0};
    std::atomic<int64_t> cnt_not_fit_{0};
    std::atomic<int64_t> cnt_failed_{0};
    std::atomic<int64_t> cnt_write_failed_{0};
    std::atomic<int64_t> total_read_ns_{0};
    std::atomic<int64_t> total_copy_ns_{0};
    std::atomic<int64_t> total_wait_ns_{0};
};

// ---------------------------------------------------------------------------
// Module self-test: concurrent writers + snapshot readers (zero torn reads)
// ---------------------------------------------------------------------------

std::tuple<int64_t, int64_t, int64_t, int64_t> selftest(int writers = 4, int readers = 2, int commits = 200) {
    if (writers <= 0 || readers < 0 || commits <= 0) {
        throw std::invalid_argument("writers and commits must be positive");
    }
    std::shared_ptr<CountNode> node(Transactional::Node<CountNode>::create<CountNode>());
    std::atomic<bool> writing{true};
    std::atomic<int64_t> torn{0}, shots{0};
    std::vector<std::thread> pool;
    pool.reserve(static_cast<size_t>(writers + readers));
    for (int w = 0; w < writers; ++w) {
        pool.emplace_back([&node, w, commits]() {
            for (int c = 0; c < commits; ++c) {
                node->iterate_commit([&node, w](Transactional::Transaction<CountNode>& tr) {
                    auto& p = tr[*node];
                    p.count += 1;
                    p.sum += w + 1;
                });
            }
        });
    }
    for (int r = 0; r < readers; ++r) {
        pool.emplace_back([&node, &writing, &torn, &shots, writers]() {
            int64_t last = 0;
            while (writing.load(std::memory_order_acquire)) {
                Transactional::Snapshot<CountNode> snap(*node);
                const auto& p = snap[*node];
                if (p.sum < p.count || p.sum > p.count * writers || p.count < last) {
                    torn.fetch_add(1, std::memory_order_relaxed);
                }
                last = p.count;
                shots.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }
    for (int w = 0; w < writers; ++w) {
        pool[static_cast<size_t>(w)].join();
    }
    writing.store(false, std::memory_order_release);
    for (size_t t = static_cast<size_t>(writers); t < pool.size(); ++t) {
        pool[t].join();
    }
    Transactional::Snapshot<CountNode> snap(*node);
    const auto& p = snap[*node];
    return {p.count, p.sum, torn.load(), shots.load()};
}

}  // namespace

NB_MODULE(_stm_core, m) {
    nb::module_::import_("mlx.core");
    m.doc() = "C++20 KAME STM core for QuantEEze";

    m.attr("MISMATCH") = nb::make_tuple("route", "absent", "experts");
    m.attr("LANDING") = nb::make_tuple("landed", "not_fit", "failed");
    m.attr("TURN_PROMPT") = TURN_PROMPT;
    m.attr("TURN_DECODE") = TURN_DECODE;

    using G = nb::call_guard<nb::gil_scoped_release>;

    m.def("selftest", &selftest,
          nb::arg("writers") = 4, nb::arg("readers") = 2, nb::arg("commits") = 200, G(),
          "Verify concurrent KAME STM writers and snapshot readers see zero torn reads");

    nb::class_<Model>(m, "Model")
        .def(nb::init<int>(), nb::arg("nlayers"))
        .def_prop_ro("nlayers", &Model::nlayers)
        .def_prop_ro("hints", [](nb::handle self) { return nb::cast<Model&>(self).hints(self); },
                     nb::rv_policy::reference_internal)
        .def("put_placement", &Model::put_placement,
             nb::arg("k"), nb::arg("route"), nb::arg("absent"), nb::arg("experts"), G())
        .def("placement", &Model::placement, nb::arg("k"))
        .def("tables", &Model::tables, nb::arg("k"))
        .def("check", &Model::check,
             nb::arg("k"), nb::arg("route"), nb::arg("absent"), nb::arg("experts"), G())
        .def("begin_layer", &Model::begin_layer, nb::arg("k"), nb::arg("pass_num"), G())
        .def("release_layer", &Model::release_layer, nb::arg("k"), nb::arg("pass_num"), G())
        .def("release", &Model::release, nb::arg("pass_num"), nb::arg("keep_k") = -1, G())
        .def("release_all", &Model::release_all, G())
        .def("pass_token", &Model::pass_token, nb::arg("pass_num"))
        .def("completed", &Model::completed, G())
        .def("is_completed", &Model::is_completed, nb::arg("pass_num"), G())
        .def("held_back", &Model::held_back)
        .def("active_passes", &Model::active_passes, nb::arg("k"), G())
        .def("set_now", &Model::set_now, nb::arg("pass_num"))
        .def("attach_cuda", &Model::attach_cuda,
             nb::arg("ctx"), nb::arg("push_ctx"), nb::arg("pop_ctx"),
             nb::arg("stream_create"), nb::arg("memcpy_async"),
             nb::arg("stream_sync"), nb::arg("stream_destroy"))
        .def("attach_host", &Model::attach_host,
             nb::arg("alloc_fn"), nb::arg("free_fn"),
             nb::arg("reg_fn") = 0, nb::arg("unreg_fn") = 0, nb::arg("get_dev_fn") = 0)
        .def("attach_readers", &Model::attach_readers,
             nb::arg("handle"), nb::arg("submit"), nb::arg("status"),
             nb::arg("release"), nb::arg("times") = 0, nb::arg("backlog") = 0, nb::arg("workers") = 1)
        .def("register_quant", &Model::register_quant,
             nb::arg("k"), nb::arg("q"),
             nb::arg("weight"), nb::arg("scales"), nb::arg("biases"),
             nb::arg("bits"), nb::arg("group_size"), nb::arg("mode"), nb::arg("indices"))
        .def("drop_quant", &Model::drop_quant, nb::arg("k"))
        .def("register_tensor", &Model::register_tensor,
             nb::arg("k"), nb::arg("tidx"), nb::arg("files"), nb::arg("offsets"), nb::arg("row_bytes"))
        .def("land_now", &Model::land_now,
             nb::arg("k"), nb::arg("pass_num"),
             nb::arg("slots"), nb::arg("experts"), nb::arg("timeout_ms") = 5000.0, G())
        .def("demand_step", &Model::demand_step,
             nb::arg("k"), nb::arg("pass_num"),
             nb::arg("indices"), nb::arg("scores"),
             nb::arg("weight_arr"), nb::arg("decay"),
             nb::arg("timeout_ms") = 5000.0, nb::arg("prev_out") = nb::none(), G())
        .def("wait_idle", &Model::wait_idle,
             nb::arg("k"), nb::arg("timeout_ms") = 5000.0, G())
        .def_static("now_ns", &Model::now_ns)
        .def("turn_take", &Model::turn_take,
             nb::arg("kind"), nb::arg("birth") = 0, nb::arg("timeout_ms") = 2000.0, G())
        .def("turn_try", &Model::turn_try, nb::arg("kind"), nb::arg("birth") = 0, G())
        .def("turn_give", &Model::turn_give, G())
        .def("turn_stats", &Model::turn_stats)
        .def("turn_state", &Model::turn_state)
        .def("room_put", &Model::room_put, nb::arg("bytes"), G())
        .def("room_take", &Model::room_take,
             nb::arg("bytes"), nb::arg("overdraw") = false, nb::arg("timeout_ms") = 0.0, G())
        .def("room_give", &Model::room_give, nb::arg("bytes"), G())
        .def("room_free", &Model::room_free, G())
        .def("arrays_of", &Model::arrays_of, nb::arg("k"))
        .def("start_lander", &Model::start_lander, nb::arg("wait_timeout_ms") = 5000.0)
        .def("stop_lander", &Model::stop_lander)
        .def("submit_landing", &Model::submit_landing,
             nb::arg("k"), nb::arg("slots"), nb::arg("experts"), nb::arg("priority") = 1, G())
        .def("landings", &Model::landings)
        .def("landed", &Model::landed)
        .def("write_failed", &Model::write_failed)
        .def("stats", &Model::stats)
        .def("total_commits", &Model::total_commits);
}
