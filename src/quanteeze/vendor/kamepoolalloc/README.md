# KAME's lock-free STM, pool allocator, smart pointer, queue, and wait cell

Taken from [KAME](https://github.com/northriv/KAME) and [kamepoolalloc](https://github.com/northriv/kamepoolalloc)
(Kentaro Kitagawa), dual-licensed Apache-2.0 OR GPL-2.0-or-later (`LICENSE-APACHE-2.0`, `LICENSE-GPL-2.0`).

From `kamepoolalloc/` (`v1.4.0` at KAME master `39eb1ce08`):

- `atomic.h`, `atomic_mfence.h`, `atomic_smart_ptr.h`: lock-free primitives (`atomic_shared_ptr<T>` / `local_shared_ptr<T>`).
- `allocator.cpp`, `allocator.h`, `allocator_prv.h`, `kame_pool.h`: lock-free per-thread pool allocator,
  preloaded via `DYLD_INSERT_LIBRARIES` / `LD_PRELOAD` and exposing `kame_pool_aligned_alloc` / `kame_pool_free`.

From `kamestm/` (`39eb1ce08`):

Paper: K. Kitagawa, *"Formally Verified Lock-Free Software Transactional Memory for Scientific Measurement"*, [arXiv:2608.12024](https://arxiv.org/abs/2608.12024) (2026), DOI: [`10.48550/arXiv.2608.12024`](https://doi.org/10.48550/arXiv.2608.12024).

- `transaction.h`, `transaction_definitions.h`, `transaction_detail.h`, `transaction_impl.h`,
  `transaction_journal.h`, `transaction_negotiation.h`, `transaction_neg_impl.h`, `transaction_signal.h`:
  lock-free Software Transactional Memory (`Transactional::Node`, `Snapshot`, `Transaction`).
- `threadlocal.h/.cpp`, `xthread.h/.cpp`, `xtime.h/.cpp`: translation units for `kamestm`.
- `support.h`, `fast_vector.h`: minimal Qt-free support types (`XString`, `XKameError`).
- `atomic_queue.h`: lock-free bounded MPMC ring queues.
- `xwaitcell.h`: futex-style timed wait-on-address primitive.
