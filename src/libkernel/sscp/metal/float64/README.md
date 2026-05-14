# Metal SSCP fp64: AdaptiveCpp <-> soft-fp64 integration

Apple GPUs do not implement IEEE 754 fp64 in hardware. The Metal SSCP
emitter rewrites every `llvm.<op>.f64` arithmetic / conversion / comparison
op to a call to `__acpp_sscp_*_f64` or `__acpp_sscp_soft_f64_*`. Those
symbols are provided by the two forwarder TUs in this directory, which
forward to soft-fp64's `sf64_*` ABI.

## Layout

| File | Role |
|------|------|
| `acpp_soft_fp64_primitives.cpp` | Required primitives: arith / conv / cmp / min-max. Direct forwards to `sf64_*`. |
| `acpp_soft_fp64_math.cpp`       | Math library: trig / exp-log / pow / classify. Direct forwards to `sf64_*`. |
| `Defines.h`, `Double.h`, `MetalFloat64.h`, `Vector.h` | Historic skeleton from the abandoned `philipturner/metal-float64` repo. Kept for git-blame continuity; not compiled. See `MAINTENANCE.md`. |
| `MAINTENANCE.md` | Background on why the local skeleton exists and the migration to soft-fp64. |

## Consumption: ACPP_SOFT_FP64_SRC_DIR

soft-fp64 is a **mandatory build dependency** of the Metal libkernel. The
build is wired up via a single CMake cache variable that points at a
soft-fp64 source checkout:

```
cmake -S /path/to/AdaptiveCpp -B build \
      -DWITH_LLVM_TO_METAL=ON \
      -DACPP_SOFT_FP64_SRC_DIR=/abs/path/to/soft-fp64
```

Resolution order (in `../CMakeLists.txt`):
1. `-DACPP_SOFT_FP64_SRC_DIR=<abs path>` (explicit override)
2. `ENV{ACPP_SOFT_FP64_SRC_DIR}`
3. `$ENV{HOME}/Projects/soft-fp64` (common local checkout)

Hard-fails the configure if the path is unset, missing, or doesn't look
like a soft-fp64 checkout (no `include/soft_fp64/soft_f64.h`). Required
soft-fp64 tag: `>=v1.3.0`.

soft-fp64's `src/*.cpp` and `src/sleef/*.cpp` are compiled in-place from
the checkout - no copying, no header rewrites. The build adds three
`-I` paths (`include/`, `src/`, `src/sleef/`) so soft-fp64's relative
includes (`"internal.h"`, `"soft_fp64/foo.h"`, `"sleef_internal.h"`,
`"../../include/soft_fp64/foo.h"`) all resolve naturally.

### Compile flags applied to the libkernel TUs

The Metal libkernel target adds the following flags (covers both
soft-fp64 and AdaptiveCpp's own libkernel TUs):

| Flag | Reason |
|------|--------|
| `-DACPP_HAS_EXTERNAL_SOFT_FP64` | Elides the `__builtin_trap()` fp64 stub block in `../math.cpp` so the forwarders are the sole definers of every `__acpp_sscp_*_f64` symbol. |
| `-fno-vectorize -fno-slp-vectorize -fno-unroll-loops` | The Metal source emitter cannot translate `<N x T>` element access or large element-wise shifts on packed vectors to MSL. Keeping libkernel TUs scalar prevents surprise vectorised bodies leaking into MSL. |
| `-DSOFT_FP64_FENV_MODE=0` | MSL has no `thread_local` storage class. Mode 0 ("disabled") compiles `SF64_FE_RAISE` to a no-op and removes the TLS variable. Host-side IEEE flag observability is unaffected - flags are surfaced from outside the kernel anyway. |
| `-DSOFT_FP64_SNAN_PROPAGATE=0` | Matches soft-fp64's default `quiet` sNaN policy. |
| `-DSOFT_FP64_OCL_ENABLED=1 -DSOFT_FP64_FTZ_MODE=0` | Builds soft-fp64's additive OpenCL compatibility/native symbol surface from `src/ocl.cpp`, with FTZ disabled so Metal's opt-in soft-fp64 path remains subnormal-preserving. |

## Required ABI symbols

These are emitted by the Metal SSCP compiler
(`src/compiler/llvm-to-backend/metal/Emitter.cpp`) for every LLVM fp64 op.
The forwarder TUs **must** provide all of them or kernel bitcode will fail
to link.

### Arithmetic

```c
double __acpp_sscp_soft_f64_add(double a, double b);
double __acpp_sscp_soft_f64_sub(double a, double b);
double __acpp_sscp_soft_f64_mul(double a, double b);
double __acpp_sscp_soft_f64_div(double a, double b);
double __acpp_sscp_soft_f64_rem(double a, double b);   // frem: IEEE remainder
double __acpp_sscp_soft_f64_neg(double a);             // -a with IEEE sign flip
```

### Min / max (IEEE 754-2008 minNum / maxNum semantics)

```c
double __acpp_sscp_soft_f64_fmin_precise(double a, double b);
double __acpp_sscp_soft_f64_fmax_precise(double a, double b);
```

### Conversions

```c
double __acpp_sscp_soft_f64_from_f32(float a);
float  __acpp_sscp_soft_f64_to_f32(double a);

double  __acpp_sscp_soft_f64_from_i32(int    a);
double  __acpp_sscp_soft_f64_from_i64(long   a);
int     __acpp_sscp_soft_f64_to_i32(double a);
long    __acpp_sscp_soft_f64_to_i64(double a);

double   __acpp_sscp_soft_f64_from_u32(unsigned int   a);
double   __acpp_sscp_soft_f64_from_u64(unsigned long  a);
unsigned int  __acpp_sscp_soft_f64_to_u32(double a);
unsigned long __acpp_sscp_soft_f64_to_u64(double a);
```

Narrower conversions (`_to_i16`, `_to_i8`, `_to_u16`, `_to_u8`) are emitted
by the SSCP compiler for narrow integer types - see `Emitter.cpp:988-1007`.

### Comparison

```c
// Returns 0 or 1 for the IEEE predicate encoded in `pred`.
// pred values follow LLVM's FCmp::Predicate enum order (see Emitter.cpp:1321).
int __acpp_sscp_soft_f64_fcmp(double lhs, double rhs, int pred);
```

## Math-library symbols

`acpp_soft_fp64_math.cpp` provides the higher-level IEEE 754 math
functions (trig, exp/log, hyperbolic, classification, fract/frexp/modf,
etc.). The complete list lives in that file; it covers every entry the
Metal SSCP emitter routes for `llvm.<name>.f64` intrinsics and OpenCL
math-library calls.

The special-case `fmin` / `fmax` math-library entry points are
IEEE-correct; the Metal SSCP emitter routes them to
`__acpp_sscp_soft_f64_fmin_precise` /
`__acpp_sscp_soft_f64_fmax_precise` rather than calling
`__acpp_sscp_fmin_f64` / `__acpp_sscp_fmax_f64` directly (see
`Emitter.cpp:1485-1503`).

## Calling convention

- Standard C linkage. All forwarders use `HIPSYCL_SSCP_BUILTIN` (default
  visibility, `extern "C"`) - see
  `include/hipSYCL/sycl/libkernel/sscp/builtins/builtin_config.hpp`.
- No SYCL-isms. Bodies run in libkernel bitcode, which is pre-SYCL IR
  (Clang + libkernel headers + soft-fp64 headers only).
- Pointer arguments (e.g. `double* iptr` in `fract_f64`) are SSCP
  generic-AS pointers - no `__device` / `__global` qualifiers.

## Verification

```bash
# Sanity: external sources show up in the bitcode target's source list.
cmake --build <build_dir> --target libkernel-sscp-metal --verbose 2>&1 | \
  grep -E "soft_fp64|acpp_soft_fp64"
```
