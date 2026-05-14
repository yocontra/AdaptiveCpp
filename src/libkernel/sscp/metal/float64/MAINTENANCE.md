# Vendored metal-float64 — maintenance notes

## Upstream snapshot

- Repository:  https://github.com/philipturner/metal-float64
- Commit SHA:  `01306dec8f44f4315ad018b2f7b24c162320ffeb`
- Commit date: 2023-05-16 ("Correct for rounding error")
- Pulled on:   2026-04-20
- License:     MIT (see ./LICENSE, ./NOTICE)

## What was vendored

Only the four public headers from `Sources/MetalFloat64/include/MetalFloat64/`:

    Defines.h        attribute macros
    Double.h         float64_t skeleton class (no math yet)
    Vector.h         vec<T, N> skeleton class
    MetalFloat64.h   umbrella header (metal-only in upstream)

The following upstream directories/files were deliberately **not** vendored:

    Sources/MetalFloat64/src/Atomic.metal       — stub (`increment(x) { return x+1; }`)
    Sources/MetalFloat64/tests/*.metal           — test harnesses, not production
    Sources/MetalAtomic64/                       — separate sub-package, not f64 math
    Package.swift, build.sh, build.swift        — Swift Package Manager driver
    README.md                                    — unnecessary; use this file

## State of upstream (as of 2026-04-20)

**Upstream is effectively abandoned / never-finished.**

- `Double.h` contains `class float64_t { ulong data; }` and two siblings (`float59_t`,
  `float43_t`), **with no arithmetic, comparison, conversion, or math function
  implementations whatsoever**.
- `Vector.h` is a non-trivial (583-line) template skeleton for `vec<T, N>` types
  but contains no float64_t specializations wired to soft-double operations.
- `src/Atomic.metal` is a 16-line stub whose only function is `increment(uint)`
  and is annotated `// TODO: Remove this entire file.`
- Last upstream commit: 2023-05-16. No soft-double IEEE 754 arithmetic was ever
  shipped.

## Consequence for AdaptiveCpp

Because there is no usable soft-double math in the upstream source, every
`__acpp_sscp_*_f64` stub in `../math.cpp` is currently implemented as
`__builtin_trap()` and tagged with `// TODO(acpp-soft-fp64): wire to
metal-float64 impl`. Kernel code that actually calls an f64 math builtin at
runtime will abort the shader. The stubs exist to resolve link-time undefined
symbols only.

## When a real implementation is available

To wire a real implementation into a given stub:

1. Add the implementation file (e.g. `Double.cpp`, adapted from a future
   upstream or hand-rolled) to this directory.
2. Add the file to `src/libkernel/sscp/metal/CMakeLists.txt` under
   `METAL_LIBKERNEL_BITCODE_SOURCES`.
3. Replace the matching `__builtin_trap()` body in `../math.cpp` with the call.
4. Update this file's "State of upstream" section accordingly.

## Adaptation guidance (when implementations arrive)

Upstream files assume Metal Shading Language. When porting to AdaptiveCpp
libkernel:

- Strip `device` / `thread` / `constant` / `threadgroup` address-space qualifiers
  (AdaptiveCpp libkernel runs on SSCP generic-AS pointers).
- Replace `metal::bit_cast<T>(x)` with `__builtin_bit_cast(T, x)`.
- Replace `metal::isnan` / `metal::isinf` / etc. with the already-wired
  `__acpp_sscp_isnan_f32` etc. where applicable, or `__builtin_isnan`.
- Replace `metal::select(a, b, cond)` with a plain ternary.
- Keep `ALWAYS_INLINE` / `NOINLINE` / `EXPORT` from `Defines.h` — they map to
  standard GCC attributes and work under the libkernel Clang front-end.
