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

// Metal fmin/fmax NaN handling under -ffast-math. IEEE 754-2008 requires
// that fmin(NaN, x) == x and fmin(x, NaN) == x (NaN acts as a "missing
// operand"). When built with -ffast-math, LLVM is free to replace fmin/fmax
// with the faster llvm.minnum/llvm.maxnum (or even a simple `x < y ? x : y`)
// which on Metal can forward the NaN. Apple's Metal intrinsics fmin/fmax
// must keep the IEEE semantics. This test checks that contract.
//
// The whole file is intentionally compiled with -ffast-math so the behavior
// under aggressive FP optimizations is exercised.

#include <sycl/sycl.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <string>
#include <type_traits>
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

template <class T> using bits_t =
    std::conditional_t<sizeof(T) == 4, std::uint32_t, std::uint64_t>;

template <class T> bits_t<T> bits_of(T value) {
  static_assert(std::is_floating_point_v<T>);
  bits_t<T> bits = 0;
  std::memcpy(&bits, &value, sizeof bits);
  return bits;
}

template <class T> void store_bits(T& value, bits_t<T> bits) {
  static_assert(std::is_floating_point_v<T>);
  std::memcpy(&value, &bits, sizeof value);
}

template <class T> bits_t<T> bits_3_5() {
  if constexpr (sizeof(T) == 4) {
    return 0x40600000u;
  } else {
    return 0x400c000000000000ull;
  }
}

template <class T> bits_t<T> bits_minus_1_5() {
  if constexpr (sizeof(T) == 4) {
    return 0xbfc00000u;
  } else {
    return 0xbff8000000000000ull;
  }
}

template <class T> bits_t<T> bits_2_25() {
  if constexpr (sizeof(T) == 4) {
    return 0x40100000u;
  } else {
    return 0x4002000000000000ull;
  }
}

template <class T> bits_t<T> bits_quiet_nan() {
  if constexpr (sizeof(T) == 4) {
    return 0x7fc00000u;
  } else {
    return 0x7ff8000000000000ull;
  }
}

template <class T> bits_t<T> bits_positive_infinity() {
  if constexpr (sizeof(T) == 4) {
    return 0x7f800000u;
  } else {
    return 0x7ff0000000000000ull;
  }
}

template <class T> bits_t<T> bits_negative_infinity() {
  if constexpr (sizeof(T) == 4) {
    return 0xff800000u;
  } else {
    return 0xfff0000000000000ull;
  }
}

template <class T> bool bits_is_nan(bits_t<T> bits) {
  static_assert(std::is_floating_point_v<T>);
  if constexpr (sizeof(T) == 4) {
    return (bits & 0x7f800000u) == 0x7f800000u &&
           (bits & 0x007fffffu) != 0;
  } else {
    return (bits & 0x7ff0000000000000ull) == 0x7ff0000000000000ull &&
           (bits & 0x000fffffffffffffull) != 0;
  }
}

// Kernel writes fmin/fmax of (a[i], b[i]) into out[i]. Compiled with
// -ffast-math at the target level, so the compiler's intrinsic lowering
// path is exercised.
template <class T> int run_case(sycl::queue &q) {
  using U = bits_t<T>;
  const U nan = bits_quiet_nan<T>();
  const U finite = bits_3_5<T>();

  // Pairs: {a, b, expected fmin, expected fmax}
  struct Triple {
    U a;
    U b;
    U min_ref;
    U max_ref;
  };
  const std::vector<Triple> cases = {
      {finite, nan, finite, finite},
      {nan, finite, finite, finite},
      {nan, nan, nan, nan},
      // IEEE 754-2008 signed-zero preservation: fmin(-0,+0)=-0, fmax=+0
      // in both operand orderings.
      {U(1) << (sizeof(T) * 8 - 1), U{0}, U(1) << (sizeof(T) * 8 - 1), U{0}},
      {U{0}, U(1) << (sizeof(T) * 8 - 1), U(1) << (sizeof(T) * 8 - 1), U{0}},
      {bits_minus_1_5<T>(), bits_2_25<T>(), bits_minus_1_5<T>(),
       bits_2_25<T>()},
      {bits_positive_infinity<T>(), finite, finite, bits_positive_infinity<T>()},
      {bits_negative_infinity<T>(), finite, bits_negative_infinity<T>(), finite},
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
    store_bits(a[i], cases[i].a);
    store_bits(b[i], cases[i].b);
  }

  q.parallel_for(sycl::range<1>{N}, [=](sycl::id<1> i) {
     omin[i] = sycl::fmin(a[i], b[i]);
     omax[i] = sycl::fmax(a[i], b[i]);
   }).wait();

  int failures = 0;
  for (std::size_t i = 0; i < N; ++i) {
    const U got_min = bits_of(omin[i]);
    const U got_max = bits_of(omax[i]);
    const bool min_ok = bits_is_nan<T>(cases[i].min_ref)
                            ? bits_is_nan<T>(got_min)
                            : got_min == cases[i].min_ref;
    const bool max_ok = bits_is_nan<T>(cases[i].max_ref)
                            ? bits_is_nan<T>(got_max)
                            : got_max == cases[i].max_ref;
    if (!min_ok) {
      std::fprintf(stderr,
                   "minmax: fmin(case %zu) wrong (size=%zu, got=0x%llx, "
                   "expected=0x%llx, a=0x%llx, b=0x%llx)\n",
                   i, sizeof(T), static_cast<unsigned long long>(got_min),
                   static_cast<unsigned long long>(cases[i].min_ref),
                   static_cast<unsigned long long>(cases[i].a),
                   static_cast<unsigned long long>(cases[i].b));
      ++failures;
    }
    if (!max_ok) {
      std::fprintf(stderr,
                   "minmax: fmax(case %zu) wrong (size=%zu, got=0x%llx, "
                   "expected=0x%llx, a=0x%llx, b=0x%llx)\n",
                   i, sizeof(T), static_cast<unsigned long long>(got_max),
                   static_cast<unsigned long long>(cases[i].max_ref),
                   static_cast<unsigned long long>(cases[i].a),
                   static_cast<unsigned long long>(cases[i].b));
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
    std::printf("metal_minnum_maxnum: not a Metal device - skipping\n");
    return 0;
  }

  int failures = 0;
  failures += run_case<float>(q);

  if (dev.has(sycl::aspect::fp64) && soft_fp64_enabled()) {
    failures += run_case<double>(q);
  } else {
    std::printf(
        "metal_minnum_maxnum: fp64/soft-fp64 not active - double path skipped\n");
  }

  if (failures != 0) {
    std::fprintf(stderr, "metal_minnum_maxnum: %d failures\n", failures);
    return 1;
  }
  std::printf("metal_minnum_maxnum: OK\n");
  return 0;
}
