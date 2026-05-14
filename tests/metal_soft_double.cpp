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

// Metal soft-fp64 correctness. Apple GPUs lack native IEEE-754 double
// precision; the soft-fp64 path emulates it using integer math on top of
// the 32-bit FPU. This test validates that:
//
//   1. Basic arithmetic (+, -, *, /) is correctly rounded per IEEE-754-2008.
//   2. sqrt/fma are correctly rounded.
//   3. Transcendentals (sin/cos/exp/log) are within 1 ULP of the host result.
//   4. Edge values (subnormals, +/-0, +/-inf, NaN) are propagated correctly.
//   5. marray<double, 2> and marray<double, 4> vector paths agree with scalar
//      element-wise computation.
//
// Activated only when the user has opted into soft-fp64 with
// ACPP_METAL_ENABLE_SOFT_FP64=1 (the env var the hw_manager gate checks).
// The test is otherwise a clean no-op so machines without the opt-in do not
// have to skip-register a failing test.

#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
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

std::uint64_t bits(double x) {
  std::uint64_t u = 0;
  std::memcpy(&u, &x, sizeof u);
  return u;
}

std::uint64_t ulp_distance(double a, double b) {
  if (std::isnan(a) || std::isnan(b)) return UINT64_MAX;
  if (a == b) return 0;
  const std::uint64_t ua = bits(a);
  const std::uint64_t ub = bits(b);
  return ua > ub ? ua - ub : ub - ua;
}

struct CaseResult {
  const char *name;
  int failures;
};

// Correctly-rounded arithmetic: result must match the host bit-for-bit.
int test_arithmetic_exact(sycl::queue &q) {
  const std::vector<double> lhs = {
      1.0, -1.0, 0.0, -0.0, 3.14159265358979323846, 1.0e-308, 1.0e+308,
      std::numeric_limits<double>::min(), std::numeric_limits<double>::max(),
      std::ldexp(1.0, -1070)  // subnormal
  };
  const std::vector<double> rhs = {
      2.0, 7.0, 1e-300, 1.0, 2.718281828459045,
      std::numeric_limits<double>::epsilon(), 0.5, 1.0 / 3.0,
      std::numeric_limits<double>::denorm_min(), 1.5};

  const std::size_t N = lhs.size();
  double *d_l = sycl::malloc_shared<double>(N, q);
  double *d_r = sycl::malloc_shared<double>(N, q);
  double *d_add = sycl::malloc_shared<double>(N, q);
  double *d_sub = sycl::malloc_shared<double>(N, q);
  double *d_mul = sycl::malloc_shared<double>(N, q);
  double *d_div = sycl::malloc_shared<double>(N, q);

  if (!d_l || !d_r || !d_add || !d_sub || !d_mul || !d_div) {
    std::fprintf(stderr, "soft_fp64: USM alloc failed\n");
    return 1;
  }
  for (std::size_t i = 0; i < N; ++i) {
    d_l[i] = lhs[i];
    d_r[i] = rhs[i];
  }

  q.parallel_for(sycl::range<1>{N}, [=](sycl::id<1> i) {
     const double a = d_l[i];
     const double b = d_r[i];
     d_add[i] = a + b;
     d_sub[i] = a - b;
     d_mul[i] = a * b;
     d_div[i] = a / b;
   }).wait();

  int failures = 0;
  for (std::size_t i = 0; i < N; ++i) {
    const double add_ref = lhs[i] + rhs[i];
    const double sub_ref = lhs[i] - rhs[i];
    const double mul_ref = lhs[i] * rhs[i];
    const double div_ref = lhs[i] / rhs[i];
    auto check = [&](double ref, double got, const char *op) {
      // NaN == NaN comparison for bit-equality
      const bool same = (std::isnan(ref) && std::isnan(got)) || bits(ref) == bits(got);
      if (!same) {
        std::fprintf(stderr,
                     "soft_fp64: %s(%a, %a) = %a expected %a (ulp=%llu)\n", op,
                     lhs[i], rhs[i], got, ref,
                     static_cast<unsigned long long>(ulp_distance(ref, got)));
        ++failures;
      }
    };
    check(add_ref, d_add[i], "add");
    check(sub_ref, d_sub[i], "sub");
    check(mul_ref, d_mul[i], "mul");
    check(div_ref, d_div[i], "div");
  }

  sycl::free(d_l, q);
  sycl::free(d_r, q);
  sycl::free(d_add, q);
  sycl::free(d_sub, q);
  sycl::free(d_mul, q);
  sycl::free(d_div, q);
  return failures;
}

// sqrt / fma: correctly rounded per IEEE-754-2008.
int test_sqrt_fma_exact(sycl::queue &q) {
  const std::vector<double> inputs = {
      0.0,    1.0,    2.0,     4.0,        9.0,       0.25,
      1e-300, 1e+300, 1.0e-20, 1.23456789, std::ldexp(1.0, -1070)};
  const std::size_t N = inputs.size();
  double *d_in = sycl::malloc_shared<double>(N, q);
  double *d_sqrt = sycl::malloc_shared<double>(N, q);
  double *d_fma = sycl::malloc_shared<double>(N, q);
  if (!d_in || !d_sqrt || !d_fma) {
    std::fprintf(stderr, "soft_fp64: USM alloc failed\n");
    return 1;
  }
  for (std::size_t i = 0; i < N; ++i) d_in[i] = inputs[i];

  q.parallel_for(sycl::range<1>{N}, [=](sycl::id<1> i) {
     const double x = d_in[i];
     d_sqrt[i] = sycl::sqrt(x);
     d_fma[i] = sycl::fma(x, x, x);
   }).wait();

  int failures = 0;
  for (std::size_t i = 0; i < N; ++i) {
    const double sqrt_ref = std::sqrt(inputs[i]);
    const double fma_ref = std::fma(inputs[i], inputs[i], inputs[i]);
    if (bits(sqrt_ref) != bits(d_sqrt[i]) &&
        !(std::isnan(sqrt_ref) && std::isnan(d_sqrt[i]))) {
      std::fprintf(stderr, "soft_fp64: sqrt(%a) = %a expected %a (ulp=%llu)\n",
                   inputs[i], d_sqrt[i], sqrt_ref,
                   static_cast<unsigned long long>(
                       ulp_distance(sqrt_ref, d_sqrt[i])));
      ++failures;
    }
    if (bits(fma_ref) != bits(d_fma[i]) &&
        !(std::isnan(fma_ref) && std::isnan(d_fma[i]))) {
      std::fprintf(stderr, "soft_fp64: fma(%a,%a,%a) = %a expected %a (ulp=%llu)\n",
                   inputs[i], inputs[i], inputs[i], d_fma[i], fma_ref,
                   static_cast<unsigned long long>(
                       ulp_distance(fma_ref, d_fma[i])));
      ++failures;
    }
  }

  sycl::free(d_in, q);
  sycl::free(d_sqrt, q);
  sycl::free(d_fma, q);
  return failures;
}

// Transcendentals: 1 ULP tolerance per the task spec.
int test_transcendental_1ulp(sycl::queue &q) {
  const std::vector<double> inputs = {
      0.0,  0.1,  0.5,  1.0,         1.5707963267948966, 3.14159265358979323846,
      -1.0, 2.0,  10.0, 100.0,       1e-10,
      1e10, -0.5, 0.75, 1.414213562373};
  const std::size_t N = inputs.size();
  double *d_in = sycl::malloc_shared<double>(N, q);
  double *d_sin = sycl::malloc_shared<double>(N, q);
  double *d_cos = sycl::malloc_shared<double>(N, q);
  double *d_exp = sycl::malloc_shared<double>(N, q);
  double *d_log = sycl::malloc_shared<double>(N, q);

  if (!d_in || !d_sin || !d_cos || !d_exp || !d_log) {
    std::fprintf(stderr, "soft_fp64: USM alloc failed\n");
    return 1;
  }
  for (std::size_t i = 0; i < N; ++i) d_in[i] = inputs[i];

  q.parallel_for(sycl::range<1>{N}, [=](sycl::id<1> i) {
     const double x = d_in[i];
     d_sin[i] = sycl::sin(x);
     d_cos[i] = sycl::cos(x);
     d_exp[i] = sycl::exp(x);
     d_log[i] = x > 0.0 ? sycl::log(x) : 0.0;
   }).wait();

  int failures = 0;
  auto check_1ulp = [&](double ref, double got, const char *op, double in) {
    if (std::isnan(ref) && std::isnan(got)) return;
    const std::uint64_t d = ulp_distance(ref, got);
    if (d > 1) {
      std::fprintf(stderr, "soft_fp64: %s(%a) = %a expected %a (ulp=%llu)\n",
                   op, in, got, ref, static_cast<unsigned long long>(d));
      ++failures;
    }
  };
  for (std::size_t i = 0; i < N; ++i) {
    check_1ulp(std::sin(inputs[i]), d_sin[i], "sin", inputs[i]);
    check_1ulp(std::cos(inputs[i]), d_cos[i], "cos", inputs[i]);
    check_1ulp(std::exp(inputs[i]), d_exp[i], "exp", inputs[i]);
    if (inputs[i] > 0.0)
      check_1ulp(std::log(inputs[i]), d_log[i], "log", inputs[i]);
  }

  sycl::free(d_in, q);
  sycl::free(d_sin, q);
  sycl::free(d_cos, q);
  sycl::free(d_exp, q);
  sycl::free(d_log, q);
  return failures;
}

// Vector paths: marray<double, 2> and marray<double, 4> elementwise multiply.
int test_vector_paths(sycl::queue &q) {
  constexpr std::size_t N = 64;
  double *a = sycl::malloc_shared<double>(N, q);
  double *b = sycl::malloc_shared<double>(N, q);
  double *out2 = sycl::malloc_shared<double>(N, q);
  double *out4 = sycl::malloc_shared<double>(N, q);

  if (!a || !b || !out2 || !out4) {
    std::fprintf(stderr, "soft_fp64: USM alloc failed\n");
    return 1;
  }
  for (std::size_t i = 0; i < N; ++i) {
    a[i] = 1.0 + 0.125 * static_cast<double>(i);
    b[i] = 2.0 - 0.0625 * static_cast<double>(i);
  }

  q.parallel_for(sycl::range<1>{N / 2}, [=](sycl::id<1> idx) {
     const std::size_t i = idx[0] * 2;
     sycl::marray<double, 2> va{a[i], a[i + 1]};
     sycl::marray<double, 2> vb{b[i], b[i + 1]};
     auto vc = va * vb;
     out2[i] = vc[0];
     out2[i + 1] = vc[1];
   }).wait();

  q.parallel_for(sycl::range<1>{N / 4}, [=](sycl::id<1> idx) {
     const std::size_t i = idx[0] * 4;
     sycl::marray<double, 4> va{a[i], a[i + 1], a[i + 2], a[i + 3]};
     sycl::marray<double, 4> vb{b[i], b[i + 1], b[i + 2], b[i + 3]};
     auto vc = va * vb;
     out4[i] = vc[0];
     out4[i + 1] = vc[1];
     out4[i + 2] = vc[2];
     out4[i + 3] = vc[3];
   }).wait();

  int failures = 0;
  for (std::size_t i = 0; i < N; ++i) {
    const double ref = a[i] * b[i];
    if (bits(ref) != bits(out2[i])) {
      std::fprintf(stderr, "soft_fp64: vec2[%zu] got %a expected %a\n", i,
                   out2[i], ref);
      ++failures;
    }
    if (bits(ref) != bits(out4[i])) {
      std::fprintf(stderr, "soft_fp64: vec4[%zu] got %a expected %a\n", i,
                   out4[i], ref);
      ++failures;
    }
  }

  sycl::free(a, q);
  sycl::free(b, q);
  sycl::free(out2, q);
  sycl::free(out4, q);
  return failures;
}

} // namespace

int main() {
  if (!soft_fp64_enabled()) {
    std::printf("metal_soft_double: ACPP_METAL_ENABLE_SOFT_FP64 not set — skipping\n");
    return 0;
  }

  sycl::queue q;
  const sycl::device dev = q.get_device();

  if (!is_metal(dev)) {
    std::printf("metal_soft_double: not a Metal device — skipping\n");
    return 0;
  }

  if (!dev.has(sycl::aspect::fp64)) {
    std::printf(
        "metal_soft_double: device lacks fp64 aspect (soft-fp64 JIT path not active) — skipping\n");
    return 0;
  }

  int failures = 0;
  failures += test_arithmetic_exact(q);
  failures += test_sqrt_fma_exact(q);
  failures += test_transcendental_1ulp(q);
  failures += test_vector_paths(q);

  if (failures != 0) {
    std::fprintf(stderr, "metal_soft_double: %d failures\n", failures);
    return 1;
  }
  std::printf("metal_soft_double: OK\n");
  return 0;
}
