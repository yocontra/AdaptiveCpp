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

// Metal fmin/fmax NaN propagation under -ffast-math. IEEE 754-2008 requires
// that fmin(NaN, x) == x and fmin(x, NaN) == x (NaN acts as a "missing
// operand"). When built with -ffast-math, LLVM is free to replace fmin/fmax
// with the faster llvm.minnum/llvm.maxnum (or even a simple `x < y ? x : y`)
// which on Metal can forward the NaN. Apple's Metal intrinsics fmin/fmax
// must keep the IEEE semantics. This test checks that contract.
//
// The whole file is intentionally compiled with -ffast-math so the behaviour
// under aggressive FP optimizations is exercised.

#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace {

bool is_metal(const sycl::device &dev) {
  const auto plat = dev.get_platform().get_info<sycl::info::platform::name>();
  return plat.find("Metal") != std::string::npos ||
         plat.find("metal") != std::string::npos;
}

bool soft_fp64_enabled() {
  const char *v = std::getenv("ACPP_METAL_ENABLE_SOFT_FP64");
  return v != nullptr && std::strcmp(v, "1") == 0;
}

template <class T> bool bitwise_equal(T a, T b) {
  static_assert(std::is_floating_point_v<T>);
  using U = std::conditional_t<sizeof(T) == 4, std::uint32_t, std::uint64_t>;
  U ua = 0;
  U ub = 0;
  std::memcpy(&ua, &a, sizeof a);
  std::memcpy(&ub, &b, sizeof b);
  return ua == ub;
}

// Kernel writes fmin/fmax of (a[i], b[i]) into out[i]. Compiled with
// -ffast-math at the target level, so the compiler's intrinsic lowering
// path is exercised.
template <class T> int run_case(sycl::queue &q) {
  const T nan = std::numeric_limits<T>::quiet_NaN();
  const T finite = static_cast<T>(3.5);

  // Pairs: {a, b, expected fmin, expected fmax}
  struct Triple {
    T a;
    T b;
    T min_ref;
    T max_ref;
  };
  const std::vector<Triple> cases = {
      {finite, nan, finite, finite},
      {nan, finite, finite, finite},
      {nan, nan, nan, nan},
      // IEEE 754-2008 signed-zero preservation: fmin(-0,+0)=-0, fmax=+0
      // in both operand orderings.
      {static_cast<T>(-0.0), static_cast<T>(0.0), static_cast<T>(-0.0),
       static_cast<T>(0.0)},
      {static_cast<T>(0.0), static_cast<T>(-0.0), static_cast<T>(-0.0),
       static_cast<T>(0.0)},
      {static_cast<T>(-1.5), static_cast<T>(2.25), static_cast<T>(-1.5),
       static_cast<T>(2.25)},
      {std::numeric_limits<T>::infinity(), finite, finite,
       std::numeric_limits<T>::infinity()},
      {-std::numeric_limits<T>::infinity(), finite,
       -std::numeric_limits<T>::infinity(), finite},
  };

  const std::size_t N = cases.size();
  T *a = sycl::malloc_shared<T>(N, q);
  T *b = sycl::malloc_shared<T>(N, q);
  T *omin = sycl::malloc_shared<T>(N, q);
  T *omax = sycl::malloc_shared<T>(N, q);
  if (!a || !b || !omin || !omax) {
    std::fprintf(stderr, "minmax: USM alloc failed\n");
    return 1;
  }
  for (std::size_t i = 0; i < N; ++i) {
    a[i] = cases[i].a;
    b[i] = cases[i].b;
  }

  q.parallel_for(sycl::range<1>{N}, [=](sycl::id<1> i) {
     omin[i] = sycl::fmin(a[i], b[i]);
     omax[i] = sycl::fmax(a[i], b[i]);
   }).wait();

  int failures = 0;
  for (std::size_t i = 0; i < N; ++i) {
    const bool min_ok =
        std::isnan(cases[i].min_ref) ? std::isnan(omin[i])
                                     : bitwise_equal(omin[i], cases[i].min_ref);
    const bool max_ok =
        std::isnan(cases[i].max_ref) ? std::isnan(omax[i])
                                     : bitwise_equal(omax[i], cases[i].max_ref);
    if (!min_ok) {
      std::fprintf(stderr, "minmax: fmin(case %zu) wrong (size=%zu)\n", i,
                   sizeof(T));
      ++failures;
    }
    if (!max_ok) {
      std::fprintf(stderr, "minmax: fmax(case %zu) wrong (size=%zu)\n", i,
                   sizeof(T));
      ++failures;
    }
  }

  sycl::free(a, q);
  sycl::free(b, q);
  sycl::free(omin, q);
  sycl::free(omax, q);
  return failures;
}

} // namespace

int main() {
  sycl::queue q;
  const sycl::device dev = q.get_device();

  if (!is_metal(dev)) {
    std::printf("metal_minnum_maxnum: not a Metal device — skipping\n");
    return 0;
  }

  int failures = 0;
  failures += run_case<float>(q);

  if (dev.has(sycl::aspect::fp64) && soft_fp64_enabled()) {
    failures += run_case<double>(q);
  } else {
    std::printf(
        "metal_minnum_maxnum: fp64/soft-fp64 not active — double path skipped\n");
  }

  if (failures != 0) {
    std::fprintf(stderr, "metal_minnum_maxnum: %d failures\n", failures);
    return 1;
  }
  std::printf("metal_minnum_maxnum: OK\n");
  return 0;
}
