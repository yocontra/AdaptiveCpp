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

// Metal atomic64 correctness. Verifies that any Metal device advertising
// aspect::atomic64 preserves the semantics required by atomic_ref<uint64_t>:
//   1. fetch_add counts exactly across millions of concurrent workitems.
//   2. fetch_max converges on the true maximum across all workitems.
//   3. 64-bit arithmetic does not silently wrap at 2^32 - the value must
//      be representable above UINT32_MAX.
//
// Skipped on devices without aspect::atomic64 so this test can live in the
// common test binary without becoming noise on backends that don't expose
// 64-bit atomics.

#include <sycl/sycl.hpp>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

bool is_metal(const sycl::device &dev) {
  const auto plat = dev.get_platform().get_info<sycl::info::platform::name>();
  return plat.find("Metal") != std::string::npos ||
         plat.find("metal") != std::string::npos;
}

constexpr std::size_t N_WORKITEMS = 10'000'000;

int test_fetch_add(sycl::queue &q) {
  std::uint64_t *counter = sycl::malloc_shared<std::uint64_t>(1, q);
  if (!counter) {
    std::fprintf(stderr, "atomic64: failed to allocate shared counter\n");
    return 1;
  }
  *counter = 0;
  q.parallel_for(sycl::range<1>{N_WORKITEMS}, [=](sycl::id<1>) {
     sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                      sycl::memory_scope::device>
         r{*counter};
     r.fetch_add(1ull);
   }).wait();

  const std::uint64_t observed = *counter;
  sycl::free(counter, q);

  if (observed != static_cast<std::uint64_t>(N_WORKITEMS)) {
    std::fprintf(stderr,
                 "atomic64: fetch_add mismatch: got %llu expected %zu\n",
                 static_cast<unsigned long long>(observed), N_WORKITEMS);
    return 1;
  }
  std::printf("atomic64: fetch_add OK (%llu)\n",
              static_cast<unsigned long long>(observed));
  return 0;
}

int test_fetch_max(sycl::queue &q) {
  std::uint64_t *m = sycl::malloc_shared<std::uint64_t>(1, q);
  if (!m) {
    std::fprintf(stderr, "atomic64: failed to allocate shared max\n");
    return 1;
  }
  *m = 0;
  q.parallel_for(sycl::range<1>{N_WORKITEMS}, [=](sycl::id<1> i) {
     sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                      sycl::memory_scope::device>
         r{*m};
     r.fetch_max(static_cast<std::uint64_t>(i[0]));
   }).wait();

  const std::uint64_t observed = *m;
  sycl::free(m, q);

  const std::uint64_t expected = static_cast<std::uint64_t>(N_WORKITEMS - 1);
  if (observed != expected) {
    std::fprintf(stderr,
                 "atomic64: fetch_max mismatch: got %llu expected %llu\n",
                 static_cast<unsigned long long>(observed),
                 static_cast<unsigned long long>(expected));
    return 1;
  }
  std::printf("atomic64: fetch_max OK (%llu)\n",
              static_cast<unsigned long long>(observed));
  return 0;
}

int test_overflow_above_uint32(sycl::queue &q) {
  // Start near 2^32 and add 10M. The final value exceeds 2^32, so any
  // backend that reduces the emulated 64-bit atomic to a 32-bit primitive
  // will wrap and fail the assertion.
  std::uint64_t *counter = sycl::malloc_shared<std::uint64_t>(1, q);
  if (!counter) {
    std::fprintf(stderr, "atomic64: failed to allocate shared counter\n");
    return 1;
  }
  constexpr std::uint64_t kBase = (static_cast<std::uint64_t>(1) << 32) - 100ull;
  *counter = kBase;
  q.parallel_for(sycl::range<1>{N_WORKITEMS}, [=](sycl::id<1>) {
     sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed,
                      sycl::memory_scope::device>
         r{*counter};
     r.fetch_add(1ull);
   }).wait();

  const std::uint64_t observed = *counter;
  sycl::free(counter, q);

  const std::uint64_t expected = kBase + static_cast<std::uint64_t>(N_WORKITEMS);
  if (observed != expected) {
    std::fprintf(stderr,
                 "atomic64: overflow mismatch: got %llu expected %llu\n",
                 static_cast<unsigned long long>(observed),
                 static_cast<unsigned long long>(expected));
    return 1;
  }
  if (observed <= static_cast<std::uint64_t>(UINT32_MAX)) {
    std::fprintf(stderr,
                 "atomic64: overflow did not exceed UINT32_MAX (%llu)\n",
                 static_cast<unsigned long long>(observed));
    return 1;
  }
  std::printf("atomic64: overflow OK (%llu)\n",
              static_cast<unsigned long long>(observed));
  return 0;
}

} // namespace

int main() {
  sycl::queue q;
  const sycl::device dev = q.get_device();

  if (!is_metal(dev)) {
    std::printf("metal_atomic64: not a Metal device - skipping\n");
    return 0;
  }
  if (!dev.has(sycl::aspect::atomic64)) {
    std::printf("metal_atomic64: device has no atomic64 aspect - skipping\n");
    return 0;
  }

  if (test_fetch_add(q) != 0) return 1;
  if (test_fetch_max(q) != 0) return 1;
  if (test_overflow_above_uint32(q) != 0) return 1;

  std::printf("metal_atomic64: OK\n");
  return 0;
}
