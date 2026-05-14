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

// Metal bitonic sort explicit instantiations
// ------------------------------------------
//
// See ``sort.cpp`` in this directory for why the bitonic sort implementation
// is NOT built into the SSCP Metal bitcode target. The public API lives at
// ``include/hipSYCL/algorithms/sort/sort_into.hpp``.
//
// Normally this file would hold ``template sycl::event acpp::sort_into<...>``
// explicit instantiations to avoid re-codegen per TU. That optimization is
// inapplicable here: the ``bitonic_sort`` template delegated to by
// ``sort_into`` takes an arbitrary comparator lambda and iterator type, so
// each call site produces a unique instantiation regardless of whether the
// key instantiation is externed. Additionally, the kernel bodies are emitted
// by the SYCL compiler into per-kernel device bitcode -- a host-side extern
// template would only deduplicate the trivial submission loop, not the
// per-stage kernel objects themselves.
//
// If a future version of ``sort_into`` grows a non-lambda, non-iterator
// implementation path (e.g. a custom device functor with stable mangling),
// this file is the right place for explicit instantiations of:
//
//     template sycl::event acpp::sort_into<int32_t>(sycl::queue&, int32_t*, size_t);
//     template sycl::event acpp::sort_into<int64_t>(sycl::queue&, int64_t*, size_t);
//     template sycl::event acpp::sort_into<uint32_t>(sycl::queue&, uint32_t*, size_t);
//     template sycl::event acpp::sort_into<uint64_t>(sycl::queue&, uint64_t*, size_t);
//     template sycl::event acpp::sort_into<float>(sycl::queue&, float*, size_t);
//     // NOTE: ``double`` is deferred until the soft-fp64 agent lands a
//     // working double-precision path on Metal. Today Apple GPUs do not
//     // expose IEEE-754 double precision; a ``sort_into<double>`` would
//     // have to either call into a soft-double emulation layer or trap.
//
// Until that refactor, this file is an empty translation unit. It is NOT
// added to the bitcode target in CMakeLists.txt.

namespace hipsycl::sycl::detail::metal_builtins {
// Intentionally empty. See comment above.
} // namespace hipsycl::sycl::detail::metal_builtins
