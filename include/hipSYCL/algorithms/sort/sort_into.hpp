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

#ifndef ACPP_ALGORITHMS_SORT_INTO
#define ACPP_ALGORITHMS_SORT_INTO

// acpp::sort_into
// ---------------
// Thin facade over hipsycl::algorithms::sorting::bitonic_sort that exposes a
// pointer/size API for callers that already own raw device pointers.
//
// Properties inherited from the underlying bitonic kernel:
//   * Not stable.
//   * Power-of-two problem sizes recommended: the underlying implementation
//     already guards index comparisons with ``can_compare`` so non-po2 sizes
//     are accepted, but callers that want the classical O(n log^2 n) bound
//     should pad to a power of two with sentinels.
//   * The sort runs entirely on the device pointer ``keys`` points at; caller
//     is responsible for USM/device/shared allocation semantics.
//
// The key-value overload performs a co-sort: ``values[i]`` is moved with
// ``keys[i]`` so that the value array reflects the same permutation. It is
// implemented by re-running the bitonic stage loop and swapping both arrays
// in the same compare-exchange step (the existing ``bitonic_sort`` template
// only moves a single iterator, so we can't simply chain it).

#include <cstddef>
#include <functional>

#include "hipSYCL/sycl/event.hpp"
#include "hipSYCL/sycl/queue.hpp"
#include "hipSYCL/algorithms/sort/bitonic_sort.hpp"

namespace acpp {

namespace detail {

inline bool sort_into_can_compare(std::size_t left_id, std::size_t right_id,
                                  std::size_t problem_size) {
  return (left_id < right_id) && (left_id < problem_size) &&
         (right_id < problem_size);
}

} // namespace detail

// Keys-only sort. Delegates to the existing in-tree bitonic sort, which works
// on any SYCL backend including Metal via the host-side queue::parallel_for
// submission pattern (one kernel launch per bitonic stage).
template <typename Key>
sycl::event sort_into(sycl::queue &q, Key *keys, std::size_t n) {
  if (n < 2 || keys == nullptr) {
    return sycl::event{};
  }
  return hipsycl::algorithms::sorting::bitonic_sort(
      q, keys, keys + n, std::less<Key>{});
}

// Key-value co-sort. Mirrors the bitonic stage loop in sort/bitonic_sort.hpp
// but swaps both ``keys`` and ``values`` in the same compare-exchange.
template <typename Key, typename Value>
sycl::event sort_into(sycl::queue &q, Key *keys, Value *values,
                      std::size_t n) {
  if (n < 2 || keys == nullptr || values == nullptr) {
    return sycl::event{};
  }

  const std::size_t problem_size = n;
  sycl::event most_recent;
  bool first = true;

  auto launch = [&](std::size_t j) {
    auto kernel = [=](sycl::id<1> idx) {
      const std::size_t a = idx.get(0);
      const std::size_t b = a ^ j;
      if (detail::sort_into_can_compare(a, b, problem_size)) {
        const Key ka = keys[a];
        const Key kb = keys[b];
        if (kb < ka) {
          keys[a] = kb;
          keys[b] = ka;
          const Value va = values[a];
          values[a] = values[b];
          values[b] = va;
        }
      }
    };
    if (first || q.is_in_order()) {
      most_recent = q.parallel_for(problem_size, kernel);
    } else {
      most_recent = q.parallel_for(problem_size, most_recent, kernel);
    }
    first = false;
  };

  for (std::size_t k = 2; (k >> 1) < problem_size; k *= 2) {
    launch(k - 1);
    for (std::size_t j = k >> 1; j > 0; j >>= 1) {
      launch(j);
    }
  }
  return most_recent;
}

} // namespace acpp

#endif
