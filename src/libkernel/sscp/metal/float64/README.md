# External soft-fp64 ABI contract (Metal SSCP libkernel)

This document is the **integration contract** between the AdaptiveCpp Metal
libkernel and an external soft-fp64 implementation. The external
implementation lives in a separate repository and is consumed by AdaptiveCpp
via the `ACPP_METAL_EXTERNAL_FP64_DIR` CMake cache variable (see
`../CMakeLists.txt`).

> Historical context: the vendored skeleton in this directory (Defines.h,
> Double.h, Vector.h, MetalFloat64.h) was pulled from the abandoned upstream
> `philipturner/metal-float64` repo and contains no math bodies. The
> trap-stubs in `../math.cpp` are the current placeholder. See
> `./MAINTENANCE.md` for background.

## How the integration works

1. The external repository provides a set of `.cpp` files implementing the
   symbols listed below, plus any header files they depend on.
2. The consumer configures AdaptiveCpp with
   `-DACPP_METAL_EXTERNAL_FP64_DIR=/path/to/external/src`.
3. The CMake rule in `../CMakeLists.txt` globs `*.cpp` from that directory
   and appends them to `METAL_LIBKERNEL_BITCODE_SOURCES`, so they are
   compiled into the Metal libkernel bitcode alongside `math.cpp`.
4. At bitcode-link time, the external definitions supersede the trap-stubs
   via linker symbol preemption (or, if both sides provide the same symbol,
   the fp64 agent rewires `math.cpp` trap bodies to call external
   primitives — see "Rewiring math.cpp" below).

## Required symbols (primitives)

These are emitted by the Metal SSCP compiler
(`src/compiler/llvm-to-backend/metal/Emitter.cpp`) for every LLVM fp64
arithmetic / conversion / comparison / min-max / unary-negation op. The
external dep **must** provide all of them or kernel bitcode will fail to
link.

### Arithmetic

```c
double __acpp_sscp_soft_f64_add(double a, double b);
double __acpp_sscp_soft_f64_sub(double a, double b);
double __acpp_sscp_soft_f64_mul(double a, double b);
double __acpp_sscp_soft_f64_div(double a, double b);
double __acpp_sscp_soft_f64_rem(double a, double b);   // frem: IEEE remainder
double __acpp_sscp_soft_f64_neg(double a);             // -a with IEEE sign flip
```

### Min / max (IEEE 754-2008 minNum / maxNum semantics — NaN propagate, signed-zero preserved)

```c
double __acpp_sscp_soft_f64_fmin_precise(double a, double b);
double __acpp_sscp_soft_f64_fmax_precise(double a, double b);
```

### Conversions — f64 ↔ f32

```c
double __acpp_sscp_soft_f64_from_f32(float a);
float  __acpp_sscp_soft_f64_to_f32(double a);
```

### Conversions — f64 ↔ signed integer

```c
double  __acpp_sscp_soft_f64_from_i32(int    a);
double  __acpp_sscp_soft_f64_from_i64(long   a);
int     __acpp_sscp_soft_f64_to_i32(double a);
long    __acpp_sscp_soft_f64_to_i64(double a);
```

### Conversions — f64 ↔ unsigned integer

```c
double   __acpp_sscp_soft_f64_from_u32(unsigned int   a);
double   __acpp_sscp_soft_f64_from_u64(unsigned long  a);
unsigned int  __acpp_sscp_soft_f64_to_u32(double a);
unsigned long __acpp_sscp_soft_f64_to_u64(double a);
```

Narrower conversions (`_to_i16`, `_to_i8`, `_to_u16`, `_to_u8`) are emitted
by the SSCP compiler for narrow integer types — see `Emitter.cpp:988-1007`.
They follow the same naming pattern: `__acpp_sscp_soft_f64_to_i<N>(double)`
returning a signed `N`-bit int, `__acpp_sscp_soft_f64_to_u<N>(double)`
returning unsigned.

### Comparison

```c
// Returns 0 or 1 for the IEEE predicate encoded in `pred`.
// pred values follow LLVM's FCmp::Predicate enum order (see Emitter.cpp:1321).
int __acpp_sscp_soft_f64_fcmp(double lhs, double rhs, int pred);
```

## Optional symbols (math library)

The external dep **may** also provide the higher-level IEEE 754 math
functions. When provided, these replace the `__builtin_trap()` bodies in
`../math.cpp`. The rewiring is not automatic — the fp64 agent owns
`math.cpp` and will replace each trap body with a real call when the
external dep is available.

Symbol list (see `../math.cpp` lines 368-481 for the canonical set; this
README reflects that snapshot):

```c
// Unary double -> double
double __acpp_sscp_{acos,acosh,acospi,asin,asinh,asinpi,atan,atanh,atanpi,
                   cbrt,ceil,cos,cosh,cospi,erf,erfc,exp,exp2,exp10,expm1,
                   fabs,floor,lgamma,log,log2,log10,log1p,logb,rint,round,
                   rsqrt,sin,sinh,sinpi,sqrt,tan,tanh,tanpi,tgamma,trunc}_f64(double);

// Binary
double __acpp_sscp_{atan2,atan2pi,copysign,fdim,fmax,fmin,fmod,hypot,maxmag,
                    minmag,nextafter,pow,powr,remainder}_f64(double, double);

// Ternary
double __acpp_sscp_{fma,mad}_f64(double, double, double);

// Mixed signatures
double __acpp_sscp_fract_f64(double, double* iptr);
double __acpp_sscp_frexp_f64(double, int* exp);
int    __acpp_sscp_ilogb_f64(double);
double __acpp_sscp_ldexp_f64(double, int k);
double __acpp_sscp_lgamma_r_f64(double, int* signp);
double __acpp_sscp_modf_f64(double, double* iptr);
double __acpp_sscp_pown_f64(double, int n);
double __acpp_sscp_rootn_f64(double, int n);
int    __acpp_sscp_{isnan,isinf,isfinite,isnormal,signbit}_f64(double);
```

The special-case `fmin`/`fmax` math-library entry points are IEEE-correct;
the Metal SSCP emitter routes them to `__acpp_sscp_soft_f64_fmin_precise`
/ `__acpp_sscp_soft_f64_fmax_precise` rather than calling
`__acpp_sscp_fmin_f64` / `__acpp_sscp_fmax_f64` directly (see
`Emitter.cpp:1485-1503`).

## Calling convention

- Standard C linkage. Use `extern "C"` in C++ sources.
- No SYCL-isms. Bodies run in libkernel bitcode, which is pre-SYCL IR
  (Clang + libkernel headers only).
- Every symbol must have `HIPSYCL_SSCP_BUILTIN` visibility (see
  `../math.cpp` for the macro). Practically: declare with
  `__attribute__((__visibility__("default")))` and do not mark `static`.
- Pointer arguments (e.g. `double* iptr` in `fract_f64`) are SSCP
  generic-AS pointers — no `__device` / `__global` qualifiers.

## Rewiring math.cpp (fp64 agent responsibility)

When the external dep provides math-library implementations, the fp64
agent owns this step:

1. Replace each `__builtin_trap()` body in `math.cpp` with a call into the
   external impl (direct call, or composite of primitives).
2. Keep the trap-stub as the fallback when
   `ACPP_METAL_EXTERNAL_FP64_DIR` is unset — guard with a preprocessor
   macro set by the CMake integration (e.g. add
   `target_compile_definitions(... PRIVATE ACPP_METAL_EXTERNAL_FP64=1)` to
   the glue, and `#ifdef ACPP_METAL_EXTERNAL_FP64` in math.cpp around each
   body).

This README documents the contract; it does not edit math.cpp.

## CMake knob

```cmake
# In AdaptiveCpp build config:
-DACPP_METAL_EXTERNAL_FP64_DIR=/absolute/path/to/fp64-repo/src
```

When set, the directory must contain at least one `.cpp` file. All `.cpp`
files in the directory are compiled into the Metal libkernel bitcode.
`CONFIGURE_DEPENDS` is used so adding / removing `.cpp` files in the
external dir re-triggers CMake.

## Verification

After configuring with the external dep:

```bash
# Sanity: the external sources show up in the bitcode target's source list.
cmake --build <build_dir> --target libkernel-metal-bitcode 2>&1 | grep -i fp64

# End-to-end: pg_accel benches exercise fp64 reduce / sort / agg.
cd /Users/contra/Projects/pg_accel
just gpu-build && just gpu-test && just test && just bench
```

Absent the external dep (default), the trap stubs remain active and fp64
kernels silently trap on dispatch — this is the canary mode. pg_accel's
`device_has_fp64_cached()` gate keeps fp64 dispatch off on Metal in that
case, so nothing crashes; cast-down-to-fp32 fallbacks execute instead.
