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

// Metal bitonic sort
// ------------------
//
// The actual bitonic sort kernel for the Metal backend is NOT implemented as
// an SSCP libkernel builtin in this directory. Two architectural reasons:
//
//  1. src/libkernel/sscp/metal/*.cpp in this repo is built as the ``metal``
//     bitcode target (see CMakeLists.txt: libkernel_generate_bitcode_target
//     TRIPLE "arm64-apple-macosx"). TUs here define device-side intrinsics
//     (``__acpp_sscp_*``) that are linked into user kernels. They cannot call
//     ``sycl::queue::parallel_for`` or submit kernels -- there is no host
//     queue at that layer.
//
//  2. Bitonic sort over arbitrary problem sizes needs O(log^2 n) kernel
//     launches stepping through the bitonic stages. That is inherently a
//     host-side queue-submission algorithm. The existing in-tree
//     implementation (``include/hipSYCL/algorithms/sort/bitonic_sort.hpp``,
//     introduced in commit ecb9d136 "[stdpar] Add simple bitonic sort") works
//     on every AdaptiveCpp backend -- including Metal -- because it submits
//     through the generic SYCL runtime.
//
// The public pointer/size API lives at
// ``include/hipSYCL/algorithms/sort/sort_into.hpp`` and delegates to the
// existing ``hipsycl::algorithms::sorting::bitonic_sort`` for the keys-only
// case. The key-value overload in that header re-runs the bitonic stage loop
// so the compare-exchange moves both keys and values in lockstep.
//
// This file is intentionally an empty translation unit: it exists to satisfy
// the per-backend file layout expected by callers that grep for Metal sort
// support, and to document the above. It is NOT added to the bitcode target
// in CMakeLists.txt; compiling it with the host toolchain produces no
// symbols.
//
// If a future change wants to specialize the Metal path (e.g. fuse several
// bitonic stages into a single dispatch that cooperates through threadgroup
// memory and ``sycl::group_barrier``), that optimization would live in
// ``sort_into.hpp`` guarded on ``SYCL_DEVICE_COPYABLE`` / ``__ACPP_BACKEND``,
// or as a new algorithm in ``include/hipSYCL/algorithms/sort/``. It should
// NOT live in this directory.

namespace hipsycl::sycl::detail::metal_builtins {
// Intentionally empty. See comment above.
} // namespace hipsycl::sycl::detail::metal_builtins
