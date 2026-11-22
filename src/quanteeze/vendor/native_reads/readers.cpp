// SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
// Lock-free, zero-GIL C++ pread worker pool for Quanteeze Demand SSD streaming.
// Uses KAME atomic<T>, atomic_bounded_ring, and XWaitCell without mutexes, condvars, or raw new/delete.
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

#include "atomic.h"
#include "atomic_queue.h"
#include "atomic_smart_ptr.h"
#include "xwaitcell.h"

#define API extern "C" __attribute__((visibility("default")))

namespace {

enum State : int { QUEUED = 0, TAKEN = 1, DONE = 2, FAILED = 3, CANCELLED = 4 };
constexpr int MAX_SEGMENTS = 12;
constexpr uint32_t SLAB = 4096;
constexpr uint32_t SLABS = 4;
constexpr unsigned TICKET_CAP = SLAB * SLABS;
constexpr unsigned RING_CAP = TICKET_CAP * 2;
constexpr int MAX_FILES = 4096;
constexpr int64_t CHUNK = int64_t(1) << 30;
constexpr unsigned NAP_US = 200000;

inline int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct Segment {
    int32_t file = -1;
    int64_t offset = 0;
    int64_t length = 0;
    uint8_t* dst = nullptr;
};

struct Ticket {
    atomic<uint64_t> tag{0};         // (gen << 1) | taken
    atomic<int> refs{0};
    atomic<int> caller_held{0};
    atomic<int> state{QUEUED};
    atomic<int> waiters{0};
    int cache = 1;
    int nseg = 0;
    int error = 0;
    int32_t failed_file = -1;
    int64_t taken_ns = 0;
    int64_t done_ns = 0;
    Segment seg[MAX_SEGMENTS];
    XWaitCell cell;

    uint32_t gen() const noexcept {
        return static_cast<uint32_t>(tag.load(std::memory_order_acquire) >> 1);
    }

    bool take(uint32_t g) noexcept {
        return tag.compare_set_strong(uint64_t(g) << 1, (uint64_t(g) << 1) | 1);
    }

    void settle(int s) noexcept {
        state.store(s, std::memory_order_release);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        const int n = waiters.load(std::memory_order_acquire);
        for (int k = 0; k <= n; ++k) {
            cell.wake_one();
        }
    }

    int wait(int64_t usec, uint32_t g) noexcept {
        if (gen() != g) {
            return CANCELLED;
        }
        int st = state.load(std::memory_order_acquire);
        if (gen() != g) {
            return CANCELLED;
        }
        if (st >= DONE) {
            return st;
        }
        const int64_t until = usec > 0 ? now_ns() + usec * 1000 : 0;
        waiters.fetch_add(1, std::memory_order_acq_rel);
        for (;;) {
            const uint32_t cg = cell.gen();
            std::atomic_thread_fence(std::memory_order_seq_cst);
            st = state.load(std::memory_order_acquire);
            if (gen() != g) {
                st = CANCELLED;
                break;
            }
            if (st >= DONE) {
                break;
            }
            if (usec == 0) {
                st = -1;
                break;
            }
            unsigned nap = NAP_US;
            if (until > 0) {
                const int64_t left = (until - now_ns()) / 1000;
                if (left <= 0) {
                    st = -1;
                    break;
                }
                nap = static_cast<unsigned>(std::min<int64_t>(left, NAP_US));
            }
            cell.wait(cg, nap);
        }
        waiters.fetch_sub(1, std::memory_order_acq_rel);
        return st;
    }
};

struct WorkItem {
    uint32_t gen = 0;
    uint32_t index = 0;
};

struct FileSlot {
    atomic<int> ready{0};
    std::string path;
    atomic<int> fd_cached{-1};
    atomic<int> fd_nocache{-1};
};

struct Readers {
    atomic<int64_t> live_refs{1};
    atomic<int> closed_once{0};
    atomic<int> stop{0};
    long page_size = 4096;
    atomic<int> slab_state[SLABS];   // 0=empty, 1=allocating, 2=ready
    std::unique_ptr<Ticket[]> slab_mem[SLABS];
    atomic<Ticket*> slabs[SLABS];
    atomic_bounded_ring<uint32_t, RING_CAP> free_ring;
    atomic_bounded_ring<WorkItem, RING_CAP> ring_waited;
    atomic_bounded_ring<WorkItem, RING_CAP> ring_ahead;
    XWaitCell work_cell;
    atomic<int> n_files{0};
    std::unique_ptr<FileSlot[]> files;
    atomic<int64_t> submitted{0};
    atomic<int64_t> completed{0};
    atomic<int64_t> cancelled{0};
    atomic<int64_t> failed{0};
    atomic<int64_t> bytes_read{0};
    atomic<int64_t> refused{0};
    atomic<int64_t> queued_count{0};
    std::vector<std::thread> threads;

    Readers() {
        const long ps = ::sysconf(_SC_PAGESIZE);
        page_size = ps > 0 ? ps : 4096;
        for (uint32_t s = 0; s < SLABS; ++s) {
            slab_state[s].store(0, std::memory_order_relaxed);
            slabs[s].store(nullptr, std::memory_order_relaxed);
        }
        files = std::make_unique<FileSlot[]>(MAX_FILES);
        init_slab(0);
    }

    void retain() noexcept {
        live_refs.fetch_add(1, std::memory_order_relaxed);
    }

    static void release_self(Readers* r) noexcept {
        if (r && r->live_refs.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            std::unique_ptr<Readers> bye(r);
        }
    }

    bool init_slab(uint32_t s) {
        if (s >= SLABS) {
            return false;
        }
        if (slab_state[s].compare_set_strong(0, 1)) {
            slab_mem[s] = std::make_unique<Ticket[]>(SLAB);
            slabs[s].store(slab_mem[s].get(), std::memory_order_release);
            for (uint32_t i = 0; i < SLAB; ++i) {
                while (!free_ring.push(s * SLAB + i)) {
                    pause4spin();
                }
            }
            slab_state[s].store(2, std::memory_order_release);
            return true;
        }
        while (slab_state[s].load(std::memory_order_acquire) == 1) {
            std::this_thread::yield();
        }
        return slab_state[s].load(std::memory_order_acquire) == 2;
    }

    Ticket* ticket_at(uint32_t idx) const noexcept {
        Ticket* base = slabs[idx / SLAB].load(std::memory_order_acquire);
        return base ? base + (idx % SLAB) : nullptr;
    }

    bool alloc_ticket(uint32_t& idx) {
        if (free_ring.pop(idx)) {
            return true;
        }
        for (uint32_t s = 1; s < SLABS; ++s) {
            if (slab_state[s].load(std::memory_order_acquire) != 2) {
                if (init_slab(s) && free_ring.pop(idx)) {
                    return true;
                }
            }
        }
        for (int spin = 0; spin < 64; ++spin) {
            if (free_ring.pop(idx)) {
                return true;
            }
            pause4spin();
        }
        return false;
    }

    void drop_ref(uint32_t idx) noexcept {
        Ticket* t = ticket_at(idx);
        if (!t) {
            return;
        }
        if (t->refs.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            const uint32_t next_gen = (static_cast<uint32_t>(t->tag.load(std::memory_order_relaxed) >> 1) + 1u) & 0x7fffffffu;
            t->tag.store(uint64_t(next_gen) << 1, std::memory_order_release);
            while (!free_ring.push(idx)) {
                pause4spin();
            }
            release_self(this);
        }
    }

    int open_fd(int32_t file_id, int cache) noexcept {
        if (file_id < 0 || file_id >= n_files.load(std::memory_order_acquire)) {
            return -EINVAL;
        }
        FileSlot& slot = files[file_id];
        while (!slot.ready.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        atomic<int>& fd_ref = cache ? slot.fd_cached : slot.fd_nocache;
        const int existing = fd_ref.load(std::memory_order_acquire);
        if (existing >= 0) {
            return existing;
        }
        const int nfd = ::open(slot.path.c_str(), O_RDONLY);
        if (nfd < 0) {
            return -errno;
        }
        if (!cache) {
#if defined(F_NOCACHE)
            ::fcntl(nfd, F_NOCACHE, 1);
#endif
        }
        int expected = -1;
        if (!fd_ref.compare_exchange_strong(expected, nfd, std::memory_order_acq_rel, std::memory_order_acquire)) {
            ::close(nfd);
            return expected;
        }
        return nfd;
    }

    void run_item(const WorkItem& item) noexcept {
        Ticket* t = ticket_at(item.index);
        if (!t || !t->take(item.gen)) {
            return;
        }
        queued_count.fetch_sub(1, std::memory_order_relaxed);
        t->taken_ns = now_ns();
        t->state.store(TAKEN, std::memory_order_release);
        int err = 0;
        int32_t bad_file = -1;
        int64_t read_total = 0;
        for (int s = 0; s < t->nseg; ++s) {
            const Segment& sg = t->seg[s];
            const int fd = open_fd(sg.file, t->cache);
            if (fd < 0) {
                err = -fd;
                bad_file = sg.file;
                break;
            }
            int64_t done = 0;
            while (done < sg.length) {
                const size_t want = static_cast<size_t>(std::min<int64_t>(sg.length - done, CHUNK));
                const ssize_t got = ::pread(fd, sg.dst + done, want, static_cast<off_t>(sg.offset + done));
                if (got < 0) {
                    if (errno == EINTR) {
                        continue;
                    }
                    err = errno;
                    bad_file = sg.file;
                    break;
                }
                if (got == 0) {
                    err = -1;
                    bad_file = sg.file;
                    break;
                }
                done += got;
            }
            if (err != 0) {
                break;
            }
            read_total += done;
            if (!t->cache) {
#if defined(__linux__) && defined(POSIX_FADV_DONTNEED)
                const int64_t ps = page_size;
                const int64_t first = (sg.offset / ps) * ps;
                const int64_t end = ((sg.offset + sg.length + ps - 1) / ps) * ps;
                if (end > first) {
                    ::posix_fadvise(fd, static_cast<off_t>(first), static_cast<off_t>(end - first), POSIX_FADV_DONTNEED);
                }
#endif
            }
        }
        if (read_total > 0) {
            bytes_read.fetch_add(read_total, std::memory_order_relaxed);
        }
        t->done_ns = now_ns();
        if (err == 0) {
            completed.fetch_add(1, std::memory_order_relaxed);
            t->settle(DONE);
        } else {
            t->error = err;
            t->failed_file = bad_file;
            failed.fetch_add(1, std::memory_order_relaxed);
            t->settle(FAILED);
        }
        drop_ref(item.index);
    }

    void worker_loop() noexcept {
        WorkItem item;
        for (;;) {
            if (ring_waited.pop(item) || ring_ahead.pop(item)) {
                run_item(item);
                continue;
            }
            const uint32_t cg = work_cell.gen();
            std::atomic_thread_fence(std::memory_order_seq_cst);
            if (ring_waited.pop(item) || ring_ahead.pop(item)) {
                run_item(item);
                continue;
            }
            if (stop.load(std::memory_order_acquire)) {
                break;
            }
            work_cell.wait(cg, NAP_US);
        }
    }

    void drain_remaining(atomic_bounded_ring<WorkItem, RING_CAP>& ring) noexcept {
        WorkItem item;
        while (ring.pop(item)) {
            Ticket* t = ticket_at(item.index);
            if (t && t->take(item.gen)) {
                queued_count.fetch_sub(1, std::memory_order_relaxed);
                cancelled.fetch_add(1, std::memory_order_relaxed);
                t->settle(CANCELLED);
                drop_ref(item.index);
            }
        }
    }

    void shutdown() noexcept {
        stop.store(1, std::memory_order_release);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        const size_t wakes = threads.size() * 2 + 8;
        for (size_t i = 0; i < wakes; ++i) {
            work_cell.wake_one();
        }
        for (auto& th : threads) {
            if (th.joinable()) {
                th.join();
            }
        }
        drain_remaining(ring_waited);
        drain_remaining(ring_ahead);
        const int nf = std::min(MAX_FILES, n_files.load(std::memory_order_acquire));
        for (int i = 0; i < nf; ++i) {
            const int fc = files[i].fd_cached.exchange(-1, std::memory_order_acq_rel);
            if (fc >= 0) {
                ::close(fc);
            }
            const int fn = files[i].fd_nocache.exchange(-1, std::memory_order_acq_rel);
            if (fn >= 0) {
                ::close(fn);
            }
        }
    }
};

inline bool valid_tok(Readers* r, int64_t tok, uint32_t& gen, uint32_t& idx, Ticket*& t) noexcept {
    if (!r || tok < 0) {
        return false;
    }
    gen = static_cast<uint32_t>(static_cast<uint64_t>(tok) >> 32);
    idx = static_cast<uint32_t>(static_cast<uint64_t>(tok));
    if (idx >= TICKET_CAP) {
        return false;
    }
    t = r->ticket_at(idx);
    return t != nullptr;
}

}  // namespace

API void* qz_rd_new(int workers) {
    auto owned = std::make_unique<Readers>();
    Readers* r = owned.get();
    const int n = workers >= 0 ? workers : 1;
    r->threads.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        r->threads.emplace_back([r] { r->worker_loop(); });
    }
    return owned.release();
}

API void qz_rd_free(void* h) {
    Readers* r = static_cast<Readers*>(h);
    if (!r) {
        return;
    }
    if (r->closed_once.compare_set_strong(0, 1)) {
        r->shutdown();
        Readers::release_self(r);
    }
}

API int32_t qz_rd_file(void* h, const char* path) {
    Readers* r = static_cast<Readers*>(h);
    if (!r || !path || r->stop.load(std::memory_order_acquire)) {
        return -1;
    }
    int idx = r->n_files.load(std::memory_order_acquire);
    for (;;) {
        if (idx >= MAX_FILES) {
            return -1;
        }
        if (r->n_files.compare_exchange_weak(idx, idx + 1, std::memory_order_acq_rel, std::memory_order_acquire)) {
            break;
        }
    }
    r->files[idx].path = path;
    r->files[idx].ready.store(1, std::memory_order_release);
    return static_cast<int32_t>(idx);
}

API int64_t qz_rd_submit(void* h, int priority, int cache, int nseg, const int32_t* files, const int64_t* offsets,
                         const int64_t* lengths, const uint64_t* dsts) {
    Readers* r = static_cast<Readers*>(h);
    if (!r || r->stop.load(std::memory_order_acquire) || nseg <= 0 || nseg > MAX_SEGMENTS
        || !files || !offsets || !lengths || !dsts) {
        return -2;
    }
    const int nf = r->n_files.load(std::memory_order_acquire);
    for (int s = 0; s < nseg; ++s) {
        if (files[s] < 0 || files[s] >= nf || offsets[s] < 0 || lengths[s] < 0 || (lengths[s] > 0 && dsts[s] == 0)) {
            return -2;
        }
    }
    uint32_t idx = 0;
    if (!r->alloc_ticket(idx)) {
        r->refused.fetch_add(1, std::memory_order_relaxed);
        return -1;
    }
    r->retain();
    Ticket* t = r->ticket_at(idx);
    const uint32_t g = t->gen();
    t->cache = cache ? 1 : 0;
    t->nseg = nseg;
    t->error = 0;
    t->failed_file = -1;
    t->taken_ns = 0;
    t->done_ns = 0;
    for (int s = 0; s < nseg; ++s) {
        t->seg[s] = Segment{files[s], offsets[s], lengths[s], reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(dsts[s]))};
    }
    t->state.store(QUEUED, std::memory_order_release);
    t->caller_held.store(1, std::memory_order_release);
    t->refs.store(2, std::memory_order_release);
    r->queued_count.fetch_add(1, std::memory_order_relaxed);
    const WorkItem item{g, idx};
    const bool pushed = (priority == 0) ? r->ring_waited.push(item) : r->ring_ahead.push(item);
    if (!pushed) {
        r->queued_count.fetch_sub(1, std::memory_order_relaxed);
        t->caller_held.store(0, std::memory_order_relaxed);
        t->refs.store(0, std::memory_order_release);
        while (!r->free_ring.push(idx)) {
            pause4spin();
        }
        r->refused.fetch_add(1, std::memory_order_relaxed);
        Readers::release_self(r);
        return -1;
    }
    std::atomic_thread_fence(std::memory_order_seq_cst);
    if (r->stop.load(std::memory_order_acquire)) {
        r->drain_remaining(r->ring_waited);
        r->drain_remaining(r->ring_ahead);
    } else {
        r->submitted.fetch_add(1, std::memory_order_relaxed);
        r->work_cell.wake_one();
    }
    return static_cast<int64_t>((uint64_t(g) << 32) | idx);
}

API int32_t qz_rd_wait(void* h, int64_t tok, int64_t timeout_us, int32_t* err_out, int32_t* file_out) {
    Readers* r = static_cast<Readers*>(h);
    uint32_t gen = 0, idx = 0;
    Ticket* t = nullptr;
    if (!valid_tok(r, tok, gen, idx, t)) {
        return -1;
    }
    const int st = t->wait(timeout_us, gen);
    if (st == FAILED && t->gen() == gen) {
        if (err_out) *err_out = t->error;
        if (file_out) *file_out = t->failed_file;
    }
    return static_cast<int32_t>(st);
}

API int32_t qz_rd_status(void* h, int64_t tok) {
    Readers* r = static_cast<Readers*>(h);
    uint32_t gen = 0, idx = 0;
    Ticket* t = nullptr;
    if (!valid_tok(r, tok, gen, idx, t) || t->gen() != gen) {
        return CANCELLED;
    }
    const int st = t->state.load(std::memory_order_acquire);
    if (t->gen() != gen) {
        return CANCELLED;
    }
    return static_cast<int32_t>(st);
}

API int32_t qz_rd_cancel(void* h, int64_t tok) {
    Readers* r = static_cast<Readers*>(h);
    uint32_t gen = 0, idx = 0;
    Ticket* t = nullptr;
    if (!valid_tok(r, tok, gen, idx, t) || t->gen() != gen) {
        return 0;
    }
    if (t->take(gen)) {
        r->queued_count.fetch_sub(1, std::memory_order_relaxed);
        r->cancelled.fetch_add(1, std::memory_order_relaxed);
        t->settle(CANCELLED);
        r->drop_ref(idx);
        return 1;
    }
    return 0;
}

API void qz_rd_times(void* h, int64_t tok, int64_t* out) {
    if (!out) {
        return;
    }
    out[0] = 0;
    out[1] = 0;
    Readers* r = static_cast<Readers*>(h);
    uint32_t gen = 0, idx = 0;
    Ticket* t = nullptr;
    if (!valid_tok(r, tok, gen, idx, t) || t->gen() != gen) {
        return;
    }
    const int64_t taken = t->taken_ns;
    const int64_t done = t->done_ns;
    if (t->gen() == gen) {
        out[0] = taken;
        out[1] = done;
    }
}

API int64_t qz_rd_backlog(void* h) {
    Readers* r = static_cast<Readers*>(h);
    if (!r) {
        return 0;
    }
    const int64_t q = r->queued_count.load(std::memory_order_relaxed);
    return q > 0 ? q : 0;
}

API void qz_rd_release(void* h, int64_t tok) {
    Readers* r = static_cast<Readers*>(h);
    uint32_t gen = 0, idx = 0;
    Ticket* t = nullptr;
    if (!valid_tok(r, tok, gen, idx, t) || t->gen() != gen) {
        return;
    }
    if (t->caller_held.exchange(0, std::memory_order_acq_rel) == 1) {
        r->drop_ref(idx);
    }
}

API void qz_rd_counts(void* h, int64_t* out) {
    Readers* r = static_cast<Readers*>(h);
    if (!r || !out) {
        return;
    }
    out[0] = r->submitted.load(std::memory_order_relaxed);
    out[1] = r->completed.load(std::memory_order_relaxed);
    out[2] = r->cancelled.load(std::memory_order_relaxed);
    out[3] = r->failed.load(std::memory_order_relaxed);
    out[4] = r->bytes_read.load(std::memory_order_relaxed);
    out[5] = r->refused.load(std::memory_order_relaxed);
}
