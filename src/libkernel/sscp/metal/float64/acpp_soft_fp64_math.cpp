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
// Forwarder TU: AdaptiveCpp Metal SSCP fp64 math library -> soft-fp64 sf64_*.
//
// Optional surface — when present, these preempt the `__builtin_trap()`
// stubs in ../math.cpp at bitcode link time. The canonical symbol list
// lives in ./README.md. Every entry below is either a 1:1 forward to an
// `sf64_*` entry point or a documented short composite.
//
// soft-fp64 is consumed via `-DACPP_SOFT_FP64_SRC_DIR=<path>` (see the
// parent ../CMakeLists.txt).

#include "hipSYCL/sycl/libkernel/sscp/builtins/builtin_config.hpp"

#include "soft_fp64/soft_f64.h"

// ---- Unary double -> double -----------------------------------------------

#define SF64_UNARY_FWD(name)                                                                       \
    HIPSYCL_SSCP_BUILTIN double __acpp_sscp_##name##_f64(double x) { return sf64_##name(x); }

SF64_UNARY_FWD(acos)
SF64_UNARY_FWD(acosh)
SF64_UNARY_FWD(acospi)
SF64_UNARY_FWD(asin)
SF64_UNARY_FWD(asinh)
SF64_UNARY_FWD(asinpi)
SF64_UNARY_FWD(atan)
SF64_UNARY_FWD(atanh)
SF64_UNARY_FWD(atanpi)
SF64_UNARY_FWD(cbrt)
SF64_UNARY_FWD(ceil)
SF64_UNARY_FWD(cos)
SF64_UNARY_FWD(cosh)
SF64_UNARY_FWD(cospi)
SF64_UNARY_FWD(erf)
SF64_UNARY_FWD(erfc)
SF64_UNARY_FWD(exp)
SF64_UNARY_FWD(exp2)
SF64_UNARY_FWD(exp10)
SF64_UNARY_FWD(expm1)
SF64_UNARY_FWD(fabs)
SF64_UNARY_FWD(floor)
SF64_UNARY_FWD(lgamma)
SF64_UNARY_FWD(log)
SF64_UNARY_FWD(log2)
SF64_UNARY_FWD(log10)
SF64_UNARY_FWD(log1p)
SF64_UNARY_FWD(logb)
SF64_UNARY_FWD(rint)
SF64_UNARY_FWD(round)
SF64_UNARY_FWD(rsqrt)
SF64_UNARY_FWD(sin)
SF64_UNARY_FWD(sinh)
SF64_UNARY_FWD(sinpi)
SF64_UNARY_FWD(sqrt)
SF64_UNARY_FWD(tan)
SF64_UNARY_FWD(tanh)
SF64_UNARY_FWD(tanpi)
SF64_UNARY_FWD(tgamma)
SF64_UNARY_FWD(trunc)

#undef SF64_UNARY_FWD

// ---- Binary double, double -> double --------------------------------------

#define SF64_BINARY_FWD(name)                                                                      \
    HIPSYCL_SSCP_BUILTIN double __acpp_sscp_##name##_f64(double x, double y) {                     \
        return sf64_##name(x, y);                                                                  \
    }

SF64_BINARY_FWD(atan2)
SF64_BINARY_FWD(atan2pi)
SF64_BINARY_FWD(copysign)
SF64_BINARY_FWD(fdim)
SF64_BINARY_FWD(fmax)
SF64_BINARY_FWD(fmin)
SF64_BINARY_FWD(fmod)
SF64_BINARY_FWD(hypot)
SF64_BINARY_FWD(maxmag)
SF64_BINARY_FWD(minmag)
SF64_BINARY_FWD(nextafter)
SF64_BINARY_FWD(pow)
SF64_BINARY_FWD(powr)
SF64_BINARY_FWD(remainder)

#undef SF64_BINARY_FWD

// ---- Ternary double, double, double -> double -----------------------------

HIPSYCL_SSCP_BUILTIN double __acpp_sscp_fma_f64(double a, double b, double c) {
    return sf64_fma(a, b, c);
}
// OpenCL `mad`: a*b + c with implementation-defined precision. Forwarding to
// sf64_fma gives deterministic, correctly-rounded semantics (stricter than
// OpenCL requires — legal).
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_mad_f64(double a, double b, double c) {
    return sf64_fma(a, b, c);
}

// ---- Mixed signatures ------------------------------------------------------

// OpenCL fract(x, iptr): *iptr = floor(x); return min(x - floor(x), nextafter(1,0)).
// sf64_fract already returns x - floor(x) in [0, 1); we forward directly.
// The OpenCL clamp to "largest double < 1" is a defensive measure against
// FP rounding pushing the result to 1.0; sf64_fract's integer-only
// implementation keeps the result strictly in [0, 1), so no clamp is needed.
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_fract_f64(double x, double* iptr) {
    *iptr = sf64_floor(x);
    return sf64_fract(x);
}

HIPSYCL_SSCP_BUILTIN double __acpp_sscp_frexp_f64(double x, int* exp) {
    return sf64_frexp(x, exp);
}

HIPSYCL_SSCP_BUILTIN int __acpp_sscp_ilogb_f64(double x) {
    return sf64_ilogb(x);
}

HIPSYCL_SSCP_BUILTIN double __acpp_sscp_ldexp_f64(double x, int k) {
    return sf64_ldexp(x, k);
}

// `__acpp_int32` is `int` (AdaptiveCpp's `include/hipSYCL/sycl/libkernel/
// detail/int_types.hpp`), so signature matches `sf64_lgamma_r(double, int*)`
// exactly. 1:1 forward.
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_lgamma_r_f64(double x, int* signp) {
    return sf64_lgamma_r(x, signp);
}

HIPSYCL_SSCP_BUILTIN double __acpp_sscp_modf_f64(double x, double* iptr) {
    return sf64_modf(x, iptr);
}

HIPSYCL_SSCP_BUILTIN double __acpp_sscp_pown_f64(double x, int n) {
    return sf64_pown(x, n);
}

HIPSYCL_SSCP_BUILTIN double __acpp_sscp_rootn_f64(double x, int n) {
    return sf64_rootn(x, n);
}

// ---- Classification --------------------------------------------------------

#define SF64_CLASSIFY_FWD(name)                                                                    \
    HIPSYCL_SSCP_BUILTIN int __acpp_sscp_##name##_f64(double x) { return sf64_##name(x); }

SF64_CLASSIFY_FWD(isnan)
SF64_CLASSIFY_FWD(isinf)
SF64_CLASSIFY_FWD(isfinite)
SF64_CLASSIFY_FWD(isnormal)
SF64_CLASSIFY_FWD(signbit)

#undef SF64_CLASSIFY_FWD
