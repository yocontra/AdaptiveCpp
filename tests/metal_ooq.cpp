/*
 * This file is part of AdaptiveCpp, an implementation of SYCL and C++ standard
 * parallelism for CPUs and GPUs.
 *
 * Copyright The AdaptiveCpp Contributors
 *
 * AdaptiveCpp is released under the BSD 2-Clause "Simplified" License.
 * See file LICENSE in the project root for full license details.
 */
// SPDX-License-Identifier: BSD-2-Clause

// Metal out-of-order queue correctness. A SYCL queue that is NOT constructed
// with `sycl::property::queue::in_order{}` must still respect explicit event
// dependencies passed via `cgh.depends_on(events)` / `q.submit(..., evs)`.
//
// Today on Metal this hangs (the event wiring on the out-of-order path is
// the open item the queue-agent is addressing). The test asserts that it
// completes within 10 seconds — a timeout is treated as a hard failure so
// a regression here is visible to CI rather than silently hanging indefinitely.
//
// If the Metal backend has decided to downgrade all queues to in-order mode
// (queue-agent's Option B), this test skips itself — probed by checking the
// queue's `is_in_order()` property after construction without the flag.

#include <sycl/sycl.hpp>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>

namespace {

bool is_metal(const sycl::device &dev) {
  const auto plat = dev.get_platform().get_info<sycl::info::platform::name>();
  return plat.find("Metal") != std::string::npos ||
         plat.find("metal") != std::string::npos;
}

constexpr std::size_t N = 4096;
constexpr auto kTimeout = std::chrono::seconds(10);

} // namespace

int main() {
  sycl::queue probe;
  if (!is_metal(probe.get_device())) {
    std::printf("metal_ooq: not a Metal device — skipping\n");
    return 0;
  }

  // Construct a queue WITHOUT the in_order property.
  sycl::queue q{probe.get_context(), probe.get_device()};
  if (q.is_in_order()) {
    std::printf(
        "metal_ooq: backend downgraded all queues to in-order — skipping\n");
    return 0;
  }

  int *a = sycl::malloc_shared<int>(N, q);
  int *b = sycl::malloc_shared<int>(N, q);
  int *c = sycl::malloc_shared<int>(N, q);
  if (!a || !b || !c) {
    std::fprintf(stderr, "metal_ooq: USM alloc failed\n");
    return 1;
  }
  for (std::size_t i = 0; i < N; ++i) {
    a[i] = 0;
    b[i] = 0;
    c[i] = 0;
  }

  // Run the work + wait on a worker thread so the main thread can enforce
  // the 10-second deadline. If the wait() hangs (the bug this test is the
  // gate on), we exit non-zero instead of letting CTest's own timeout fire.
  std::atomic<bool> done{false};
  std::atomic<int> worker_rc{0};
  std::thread worker([&] {
    try {
      sycl::event e1 = q.submit([&](sycl::handler &cgh) {
        cgh.parallel_for(sycl::range<1>{N},
                         [=](sycl::id<1> i) { a[i] = static_cast<int>(i[0]) + 1; });
      });
      sycl::event e2 = q.submit([&](sycl::handler &cgh) {
        cgh.parallel_for(sycl::range<1>{N}, [=](sycl::id<1> i) {
          b[i] = static_cast<int>(i[0]) * 2;
        });
      });
      // Third kernel depends on both. If event wiring is broken, this never
      // enqueues or never signals completion.
      q.submit([&](sycl::handler &cgh) {
         cgh.depends_on({e1, e2});
         cgh.parallel_for(sycl::range<1>{N},
                          [=](sycl::id<1> i) { c[i] = a[i] + b[i]; });
       }).wait();
      q.wait();
      done.store(true, std::memory_order_release);
    } catch (const sycl::exception &e) {
      std::fprintf(stderr, "metal_ooq: SYCL exception: %s\n", e.what());
      worker_rc.store(1, std::memory_order_release);
      done.store(true, std::memory_order_release);
    }
  });

  const auto deadline = std::chrono::steady_clock::now() + kTimeout;
  while (!done.load(std::memory_order_acquire)) {
    if (std::chrono::steady_clock::now() > deadline) {
      std::fprintf(stderr,
                   "metal_ooq: q.wait() hung past 10s — out-of-order event "
                   "dependency wiring is broken\n");
      // Detach: we cannot safely join a hung worker, but the process is
      // going to exit non-zero immediately so the OS will clean it up.
      worker.detach();
      return 1;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  worker.join();

  if (worker_rc.load(std::memory_order_acquire) != 0) {
    return 1;
  }

  int failures = 0;
  for (std::size_t i = 0; i < N; ++i) {
    const int expected = (static_cast<int>(i) + 1) + (static_cast<int>(i) * 2);
    if (c[i] != expected) {
      if (failures < 8) {
        std::fprintf(stderr,
                     "metal_ooq: c[%zu] = %d expected %d (a=%d b=%d)\n", i,
                     c[i], expected, a[i], b[i]);
      }
      ++failures;
    }
  }

  sycl::free(a, q);
  sycl::free(b, q);
  sycl::free(c, q);

  if (failures != 0) {
    std::fprintf(stderr, "metal_ooq: %d element mismatches\n", failures);
    return 1;
  }
  std::printf("metal_ooq: OK\n");
  return 0;
}
