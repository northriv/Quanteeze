// SPDX-License-Identifier: GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0
// NVRTC automatic precompiled headers for MLX custom CUDA kernels (QUANTEEZE_NVRTC_PCH; quanteeze/__init__.py).
//
// Loaded with RTLD_GLOBAL before MLX is imported, this library interposes nvrtcCreateProgram and
// nvrtcCompileProgram and delegates to the libnvrtc.so.<major> this process has loaded (CUDA 12's or 13's: found
// among the loaded objects by name, opened with RTLD_NOLOAD):
// - "--pch" and "--pch-dir=<dir>" are added only for mx.fast.cuda_kernel programs (named "custom_kernel_*")
//   whose compile follows on the same thread; MLX's own PTX kernels (which are cached on disk and can
//   require >268 MB PCH heaps) compile unmodified.
// - PCH files go to a per-process directory set once via qz_nvrtc_pch_dir (never cwd), skipped in forked
//   children, skipped if NVRTC < 12.8 lacks nvrtcGetPCHCreateStatus, and capped at MOST_MADE files.
#include <dlfcn.h>
#include <link.h>
#include <unistd.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "atomic.h"
#include "atomic_smart_ptr.h"

namespace {

using Result = int;                                  // nvrtcResult
using Program = void *;                              // nvrtcProgram
using Compile = Result (*)(Program, int, const char *const *);
using Create = Result (*)(Program *, const char *, const char *, int, const char *const *, const char *const *);
using Status = Result (*)(Program);                  // nvrtcGetPCHCreateStatus
using Needed = Result (*)(Program, size_t *);        // nvrtcGetPCHHeapSizeRequired
using Resize = Result (*)(size_t);                   // nvrtcSetPCHHeapSize
constexpr Result INTERNAL_ERROR = 11;                // NVRTC_ERROR_INTERNAL_ERROR
constexpr Result HEAP_EXHAUSTED = 14;                // NVRTC_ERROR_PCH_CREATE_HEAP_EXHAUSTED
constexpr size_t HEAP_MOST = size_t(1) << 30;        // the PCH heap is never grown past this
constexpr char CUSTOM[] = "custom_kernel_";
constexpr uint64_t MOST_MADE = 8;
constexpr size_t PATH_MOST = 4096;                   // a loaded library's path, with its end

atomic<Compile> real_compile{nullptr};
atomic<Create> real_create{nullptr};
atomic<Status> real_status{nullptr};
atomic<Needed> real_needed{nullptr};
atomic<Resize> real_resize{nullptr};
atomic<uint64_t> grown{0};
atomic<uint64_t> compiles{0}, added{0}, made{0};

struct Place {
    std::string option;                              // "--pch-dir=<directory>"
    pid_t pid;
};
atomic_shared_ptr<const Place> place;

thread_local Program custom = nullptr;

constexpr char NVRTC[] = "libnvrtc.so";              // libnvrtc.so.12, libnvrtc.so.13, ...: whichever MLX was built for

// dl_iterate_phdr callback: the first loaded object whose file is libnvrtc.so[.<version>] (not libnvrtc-builtins,
// not this library), its path copied to the caller's buffer.
int find_nvrtc(struct dl_phdr_info *info, size_t, void *data) noexcept {
    const char *path = info->dlpi_name;
    if(!path || !*path)
        return 0;
    const char *base = std::strrchr(path, '/');
    base = base ? base + 1 : path;
    const size_t n = sizeof(NVRTC) - 1;
    if(std::strncmp(base, NVRTC, n) != 0 || (base[n] != '\0' && base[n] != '.'))
        return 0;
    char *out = static_cast<char *>(data);
    if(std::strlen(path) >= PATH_MOST)
        return 0;
    std::strcpy(out, path);
    return 1;
}

// A handle on the NVRTC already loaded here (RTLD_NOLOAD: never loads one), or nullptr.
void *loaded_nvrtc() noexcept {
    char path[PATH_MOST] = {0};
    if(!dl_iterate_phdr(find_nvrtc, path))
        return nullptr;
    return dlopen(path, RTLD_NOW | RTLD_NOLOAD);
}

template<typename F>
F own(atomic<F> &real, const char *name) noexcept {
    F f = real.load(std::memory_order_acquire);
    if(f)
        return f;
    void *lib = loaded_nvrtc();
    if(!lib)
        return nullptr;
    f = reinterpret_cast<F>(dlsym(lib, name));
    dlclose(lib);
    if(f)
        real.store(f, std::memory_order_release);
    return f;
}

bool asks_pch(int n, const char *const *options) noexcept {
    for(int i = 0; i < n; ++i)
        if(options[i] && (std::strncmp(options[i], "--pch", 5) == 0 || std::strncmp(options[i], "-pch", 4) == 0 ||
                          std::strncmp(options[i], "--create-pch", 12) == 0 ||
                          std::strncmp(options[i], "--use-pch", 9) == 0))
            return true;
    return false;
}

}  // namespace

extern "C" __attribute__((visibility("default"))) Result nvrtcCreateProgram(Program *program, const char *src,
                                                                             const char *name, int headers,
                                                                             const char *const *contents,
                                                                             const char *const *names) noexcept {
    Create f = own(real_create, "nvrtcCreateProgram");
    if(!f)
        return INTERNAL_ERROR;
    const Result r = f(program, src, name, headers, contents, names);
    custom = (r == 0 && program && name && std::strncmp(name, CUSTOM, sizeof(CUSTOM) - 1) == 0) ? *program : nullptr;
    return r;
}

extern "C" __attribute__((visibility("default"))) Result nvrtcCompileProgram(Program program, int n,
                                                                              const char *const *options) noexcept {
    Compile f = own(real_compile, "nvrtcCompileProgram");
    if(!f)
        return INTERNAL_ERROR;
    compiles.fetch_add(1, std::memory_order_relaxed);
    const bool mine = program && program == custom;
    custom = nullptr;
    if(n < 0 || (n > 0 && !options))
        return f(program, n, options);
    if(!mine || asks_pch(n, options))
        return f(program, n, options);
    const local_shared_ptr<const Place> at(place);
    Status status = own(real_status, "nvrtcGetPCHCreateStatus");
    if(!at || at->pid != getpid() || !status || made.load(std::memory_order_relaxed) >= MOST_MADE)
        return f(program, n, options);
    std::vector<const char *> with;
    try {
        with.reserve(static_cast<size_t>(n) + 2);
        if(options && n > 0)
            with.assign(options, options + n);
        with.push_back("--pch");
        with.push_back(at->option.c_str());
    } catch(...) {
        return f(program, n, options);
    }
    added.fetch_add(1, std::memory_order_relaxed);
    const Result r = f(program, static_cast<int>(with.size()), with.data());
    const Result created = status(program);
    if(created == 0)
        made.fetch_add(1, std::memory_order_relaxed);
    else if(created == HEAP_EXHAUSTED) {             // the heap NVRTC set aside was too small for this program's
        Needed needed = own(real_needed, "nvrtcGetPCHHeapSizeRequired");     // headers: grown to what it asks, for
        Resize resize = own(real_resize, "nvrtcSetPCHHeapSize");             // the custom kernels compiled after
        size_t bytes = 0;
        if(needed && resize && needed(program, &bytes) == 0 && bytes > 0 && bytes <= HEAP_MOST &&
           resize(bytes) == 0)
            grown.fetch_add(1, std::memory_order_relaxed);
    }
    return r;
}

// Set the per-process directory where NVRTC writes PCH files (once per process). Returns 1 if set, 0 otherwise.
extern "C" __attribute__((visibility("default"))) int qz_nvrtc_pch_dir(const char *directory) noexcept {
    if(!directory || !*directory)
        return 0;
    try {
        const local_shared_ptr<const Place> mine(
            make_local_shared<Place>(Place{std::string("--pch-dir=") + directory, getpid()}));
        return place.compareAndSet(local_shared_ptr<const Place>(), mine) ? 1 : 0;
    } catch(...) {
        return 0;
    }
}

// Total nvrtcCompileProgram calls, custom kernel calls given --pch, and compiles that created a PCH file.
extern "C" __attribute__((visibility("default"))) void qz_nvrtc_pch_counts(uint64_t *out) noexcept {
    out[0] = compiles.load(std::memory_order_relaxed);
    out[1] = added.load(std::memory_order_relaxed);
    out[2] = made.load(std::memory_order_relaxed);
}

// How many times the PCH heap was grown to what a compile asked for.
extern "C" __attribute__((visibility("default"))) uint64_t qz_nvrtc_pch_grown() noexcept {
    return grown.load(std::memory_order_relaxed);
}
