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
//
// Forwarder TU: AdaptiveCpp Metal SSCP fp64 primitives -> soft-fp64 sf64_*.
//
// Every symbol below is one of the "Required primitives" listed in
// ./README.md (the ABI contract). Each body is a one-line forward to the
// matching `sf64_*` entry point - no local arithmetic, no host FPU
// dependency, no SYCL-isms.
//
// soft-fp64 is consumed via `-DACPP_SOFT_FP64_SRC_DIR=<path>` (see the
// parent ../CMakeLists.txt). At libkernel-bitcode link time, these
// forwarders combine with soft-fp64's `src/*.cpp` and `src/sleef/*.cpp`
// to provide the complete fp64 surface; the `__builtin_trap()` block in
// ../math.cpp is elided via `-DACPP_HAS_EXTERNAL_SOFT_FP64`.

#include "hipSYCL/sycl/libkernel/sscp/builtins/builtin_config.hpp"

#include "soft_fp64/soft_f64.h"

#include <cstdint>

// ---- Arithmetic ------------------------------------------------------------

HIPSYCL_SSCP_BUILTIN double __acpp_sscp_soft_f64_add(double a, double b) {
    return sf64_add(a, b);
}
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_soft_f64_sub(double a, double b) {
    return sf64_sub(a, b);
}
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_soft_f64_mul(double a, double b) {
    return sf64_mul(a, b);
}
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_soft_f64_div(double a, double b) {
    return sf64_div(a, b);
}
// LLVM `frem` (fmod semantics: sign of result = sign of dividend); maps to
// sf64_rem. sf64_fmod is documented as identical semantics - pick one.
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_soft_f64_rem(double a, double b) {
    return sf64_rem(a, b);
}
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_soft_f64_neg(double a) {
    return sf64_neg(a);
}

// ---- Min / max (IEEE 754-2008 precise: NaN-propagating) --------------------

HIPSYCL_SSCP_BUILTIN double __acpp_sscp_soft_f64_fmin_precise(double a, double b) {
    return sf64_fmin_precise(a, b);
}
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_soft_f64_fmax_precise(double a, double b) {
    return sf64_fmax_precise(a, b);
}

// ---- Conversions: f64 <-> f32 ---------------------------------------------

HIPSYCL_SSCP_BUILTIN double __acpp_sscp_soft_f64_from_f32(float a) {
    return sf64_from_f32(a);
}
HIPSYCL_SSCP_BUILTIN float __acpp_sscp_soft_f64_to_f32(double a) {
    return sf64_to_f32(a);
}

// ---- Conversions: f64 <-> signed integer ----------------------------------

HIPSYCL_SSCP_BUILTIN double __acpp_sscp_soft_f64_from_i32(int a) {
    return sf64_from_i32(static_cast<int32_t>(a));
}
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_soft_f64_from_i64(long a) {
    return sf64_from_i64(static_cast<int64_t>(a));
}
HIPSYCL_SSCP_BUILTIN int __acpp_sscp_soft_f64_to_i32(double a) {
    return static_cast<int>(sf64_to_i32(a));
}
HIPSYCL_SSCP_BUILTIN long __acpp_sscp_soft_f64_to_i64(double a) {
    return static_cast<long>(sf64_to_i64(a));
}

// ---- Conversions: f64 <-> unsigned integer --------------------------------

HIPSYCL_SSCP_BUILTIN double __acpp_sscp_soft_f64_from_u32(unsigned int a) {
    return sf64_from_u32(static_cast<uint32_t>(a));
}
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_soft_f64_from_u64(unsigned long a) {
    return sf64_from_u64(static_cast<uint64_t>(a));
}
HIPSYCL_SSCP_BUILTIN unsigned int __acpp_sscp_soft_f64_to_u32(double a) {
    return static_cast<unsigned int>(sf64_to_u32(a));
}
HIPSYCL_SSCP_BUILTIN unsigned long __acpp_sscp_soft_f64_to_u64(double a) {
    return static_cast<unsigned long>(sf64_to_u64(a));
}

// ---- Narrow integer conversions --------------------------------------------
// Metal SSCP emitter Emitter.cpp:988-1007 emits _to_i16/_to_i8/_to_u16/_to_u8
// for narrow integer types. Signatures return signed/unsigned N-bit ints.

HIPSYCL_SSCP_BUILTIN signed char __acpp_sscp_soft_f64_to_i8(double a) {
    return static_cast<signed char>(sf64_to_i8(a));
}
HIPSYCL_SSCP_BUILTIN short __acpp_sscp_soft_f64_to_i16(double a) {
    return static_cast<short>(sf64_to_i16(a));
}
HIPSYCL_SSCP_BUILTIN unsigned char __acpp_sscp_soft_f64_to_u8(double a) {
    return static_cast<unsigned char>(sf64_to_u8(a));
}
HIPSYCL_SSCP_BUILTIN unsigned short __acpp_sscp_soft_f64_to_u16(double a) {
    return static_cast<unsigned short>(sf64_to_u16(a));
}

// ---- Compare ---------------------------------------------------------------
// sf64_fcmp takes the same LLVM FCmpInst::Predicate encoding (0..15) that
// the Metal SSCP emitter uses - see Emitter.cpp:1321. Pass-through forward.

HIPSYCL_SSCP_BUILTIN int __acpp_sscp_soft_f64_fcmp(double lhs, double rhs, int pred) {
    return sf64_fcmp(lhs, rhs, pred);
}
