if(WITH_LLVM_TO_METAL)
  set(METAL_LIBKERNEL_BITCODE_SOURCES
    assert.cpp
    atomic.cpp
    barrier.cpp
    broadcast.cpp
    collpredicate.cpp
    core.cpp
    half.cpp
    integer.cpp
    localmem.cpp
    math.cpp
    native.cpp
    print.cpp
    reduction.cpp
    scan_inclusive.cpp
    scan_exclusive.cpp
    shuffle.cpp
    subgroup.cpp
    # Forwarder TUs that adapt soft-fp64's `sf64_*` ABI to the
    # `__acpp_sscp_*_f64` / `__acpp_sscp_soft_f64_*` symbols the Metal
    # SSCP emitter calls. These two files live in this tree (they ARE
    # AdaptiveCpp's ABI; they have nothing to do with soft-fp64's
    # public surface). The actual sf64_* implementation comes from
    # soft-fp64 - see ACPP_SOFT_FP64_SRC_DIR below.
    float64/acpp_soft_fp64_primitives.cpp
    float64/acpp_soft_fp64_math.cpp)

  # ============================================================================
  # soft-fp64 integration (mandatory for Metal libkernel).
  # ----------------------------------------------------------------------------
  # Apple GPUs do not implement IEEE 754 fp64 in hardware. The Metal SSCP
  # emitter rewrites every `llvm.<op>.f64` and fp64 arith / cmp / convert op
  # to a call to `__acpp_sscp_*_f64` or `__acpp_sscp_soft_f64_*`. Those
  # symbols MUST resolve at libkernel-bitcode link time - soft-fp64 is the
  # implementation that provides them via the two forwarder TUs above.
  #
  # Resolution order for ACPP_SOFT_FP64_SRC_DIR:
  #   1. -DACPP_SOFT_FP64_SRC_DIR=<abs path>      (explicit override)
  #   2. ENV{ACPP_SOFT_FP64_SRC_DIR}              (CI / multi-checkout)
  #   3. $ENV{HOME}/Projects/soft-fp64            (common local checkout)
  #
  # Whatever path resolves must point at the *root* of a soft-fp64 checkout
  # (the dir containing `include/`, `src/`, `src/sleef/`). soft-fp64's
  # `.cpp` files are compiled in-place from that tree - no copying, no
  # rewriting of `#include "../../include/..."` lines, no flattening. The
  # required tag is `v1.2.0`; soft-fp64 maintains semver, so newer
  # `v1.x` tags are also accepted.
  # ============================================================================
  set(_ACPP_SOFT_FP64_DEFAULT "")
  if(DEFINED ENV{ACPP_SOFT_FP64_SRC_DIR})
    set(_ACPP_SOFT_FP64_DEFAULT "$ENV{ACPP_SOFT_FP64_SRC_DIR}")
  elseif(EXISTS "$ENV{HOME}/Projects/soft-fp64/include/soft_fp64/soft_f64.h")
    set(_ACPP_SOFT_FP64_DEFAULT "$ENV{HOME}/Projects/soft-fp64")
  endif()

  set(ACPP_SOFT_FP64_SRC_DIR "${_ACPP_SOFT_FP64_DEFAULT}" CACHE PATH
      "Path to the soft-fp64 source tree (dir containing include/, src/, src/sleef/). Required when WITH_LLVM_TO_METAL=ON.")

  if(NOT ACPP_SOFT_FP64_SRC_DIR)
    message(FATAL_ERROR
      "WITH_LLVM_TO_METAL requires ACPP_SOFT_FP64_SRC_DIR to point at a "
      "soft-fp64 checkout (tag >=v1.2.0). Set "
      "-DACPP_SOFT_FP64_SRC_DIR=<abs path> or env ACPP_SOFT_FP64_SRC_DIR. "
      "AdaptiveCpp's Metal libkernel cannot link without an fp64 "
      "implementation - Apple GPUs lack hardware fp64.")
  endif()
  if(NOT IS_DIRECTORY "${ACPP_SOFT_FP64_SRC_DIR}")
    message(FATAL_ERROR
      "ACPP_SOFT_FP64_SRC_DIR='${ACPP_SOFT_FP64_SRC_DIR}' is not a directory.")
  endif()
  if(NOT EXISTS "${ACPP_SOFT_FP64_SRC_DIR}/include/soft_fp64/soft_f64.h")
    message(FATAL_ERROR
      "ACPP_SOFT_FP64_SRC_DIR='${ACPP_SOFT_FP64_SRC_DIR}' does not look like "
      "a soft-fp64 checkout (missing include/soft_fp64/soft_f64.h).")
  endif()

  # soft-fp64 source files compiled into the Metal libkernel bitcode.
  # The list is fixed (matches soft-fp64 v1.2.0 / target_sources in its
  # top-level CMakeLists.txt for the `soft_fp64` target). The fenv TU is
  # included even though SOFT_FP64_FENV_MODE=0 below makes its bodies
  # collapse to no-ops - keeps the symbol surface complete.
  set(_SOFT_FP64_CORE_SOURCES
    src/arithmetic.cpp
    src/classify.cpp
    src/compare.cpp
    src/convert.cpp
    src/fenv.cpp
    src/rounding.cpp
    src/sqrt_fma.cpp
    src/sleef/sleef_exp_log.cpp
    src/sleef/sleef_inv_hyp_pow.cpp
    src/sleef/sleef_stubs.cpp
    src/sleef/sleef_trig.cpp)

  set(_SOFT_FP64_ABS_SOURCES "")
  foreach(_src ${_SOFT_FP64_CORE_SOURCES})
    set(_abs "${ACPP_SOFT_FP64_SRC_DIR}/${_src}")
    if(NOT EXISTS "${_abs}")
      message(FATAL_ERROR
        "soft-fp64 source missing: ${_abs}. Is your soft-fp64 checkout "
        "at the expected tag (>=v1.2.0)?")
    endif()
    list(APPEND _SOFT_FP64_ABS_SOURCES "${_abs}")
  endforeach()

  list(APPEND METAL_LIBKERNEL_BITCODE_SOURCES ${_SOFT_FP64_ABS_SOURCES})

  set(METAL_LIBKERNEL_EXTRA_ARGS "")

  # `ACPP_HAS_EXTERNAL_SOFT_FP64` elides the f64 trap-stub block in
  # math.cpp so the soft-fp64 forwarders are the sole definers of every
  # `__acpp_sscp_*_f64` symbol. Always on now (soft-fp64 is mandatory).
  list(APPEND METAL_LIBKERNEL_EXTRA_ARGS -DACPP_HAS_EXTERNAL_SOFT_FP64)

  # Vectorisation / unrolling control - APPLIES TO ALL libkernel TUs in
  # this list (including the AdaptiveCpp-native ones), not just soft-fp64.
  # This is broader than strictly required, but the Metal source emitter
  # (`compiler/llvm-to-backend/metal/Emitter.cpp`) cannot translate
  # `<N x T>` element access or large element-wise shifts on packed
  # vectors to MSL, so keeping every TU scalar avoids surprise vectorised
  # code paths leaking into MSL via libkernel inlining. The cost is small
  # - these are mostly cold thunks.
  list(APPEND METAL_LIBKERNEL_EXTRA_ARGS -fno-vectorize -fno-slp-vectorize -fno-unroll-loops)

  # soft-fp64's `tls` fenv mode uses `thread_local` for IEEE flag
  # accumulation. MSL has no TLS - the resulting TLS wrapper is
  # unresolvable. Mode 0 ("disabled") compiles SF64_FE_RAISE to a no-op
  # and removes the thread_local variable. Host-side fenv observability
  # is unaffected (it's surfaced from outside the kernel anyway).
  list(APPEND METAL_LIBKERNEL_EXTRA_ARGS -DSOFT_FP64_FENV_MODE=0)

  # Include paths for soft-fp64's relative #includes:
  #   - src/*.cpp              -> "internal*.h"               (sibling, src/)
  #                               "soft_fp64/foo.h"           (include/)
  #   - src/sleef/*.cpp        -> "sleef_internal.h"          (sibling, src/sleef/)
  #                               "../internal_fenv.h"        (relative, OK as-is)
  #                               "sleef_fe_macros.h"         (sibling)
  #   - src/sleef/sleef_*.h    -> "../../include/soft_fp64/foo.h" (resolves naturally
  #                                from src/sleef/ - soft-fp64 root has include/)
  list(APPEND METAL_LIBKERNEL_EXTRA_ARGS
       -I "${ACPP_SOFT_FP64_SRC_DIR}/include"
       -I "${ACPP_SOFT_FP64_SRC_DIR}/src"
       -I "${ACPP_SOFT_FP64_SRC_DIR}/src/sleef")

  message(STATUS "Metal libkernel: linking soft-fp64 from ${ACPP_SOFT_FP64_SRC_DIR}")

  libkernel_generate_bitcode_target(
      TARGETNAME metal
      TRIPLE "arm64-apple-macosx"
      SOURCES ${METAL_LIBKERNEL_BITCODE_SOURCES}
      ADDITIONAL_ARGS ${METAL_LIBKERNEL_EXTRA_ARGS})
endif()
