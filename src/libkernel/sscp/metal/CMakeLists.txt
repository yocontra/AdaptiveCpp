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
    subgroup.cpp)

  # External soft-fp64 integration.
  # When ACPP_METAL_EXTERNAL_FP64_DIR points at a directory containing the
  # external soft-fp64 implementation (see ./float64/README.md for the ABI
  # contract), the .cpp files there are compiled into the Metal libkernel
  # bitcode alongside math.cpp. The external dep's bodies replace the
  # __builtin_trap() stubs in math.cpp by linker symbol preemption at
  # bitcode-link time (external definitions win over the trap stubs).
  set(ACPP_METAL_EXTERNAL_FP64_DIR "" CACHE PATH
      "Path to external soft-fp64 implementation source dir (see src/libkernel/sscp/metal/float64/README.md)")
  set(METAL_LIBKERNEL_EXTRA_ARGS "")
  if(ACPP_METAL_EXTERNAL_FP64_DIR)
    if(NOT IS_DIRECTORY "${ACPP_METAL_EXTERNAL_FP64_DIR}")
      message(FATAL_ERROR
        "ACPP_METAL_EXTERNAL_FP64_DIR='${ACPP_METAL_EXTERNAL_FP64_DIR}' is not a directory")
    endif()
    file(GLOB ACPP_METAL_EXTERNAL_FP64_SOURCES
         CONFIGURE_DEPENDS
         "${ACPP_METAL_EXTERNAL_FP64_DIR}/*.cpp")
    if(NOT ACPP_METAL_EXTERNAL_FP64_SOURCES)
      message(FATAL_ERROR
        "ACPP_METAL_EXTERNAL_FP64_DIR='${ACPP_METAL_EXTERNAL_FP64_DIR}' contains no .cpp files")
    endif()
    list(APPEND METAL_LIBKERNEL_BITCODE_SOURCES ${ACPP_METAL_EXTERNAL_FP64_SOURCES})
    # Elide the f64 trap-stub block in math.cpp so the external TUs are the
    # sole definers of `__acpp_sscp_*_f64`. Without this guard `llvm-link`
    # errors on "symbol multiply defined" — bitcode link (unlike ELF) has
    # no weak/preempt semantics.
    list(APPEND METAL_LIBKERNEL_EXTRA_ARGS -DACPP_HAS_EXTERNAL_SOFT_FP64)
    # Disable LLVM's loop + SLP vectorizers and loop unrolling in the
    # bitcode compile when external soft-fp64 is active. At `-O3` these
    # passes turn straightforward scalar soft-fp64 code (e.g. mantissa
    # normalisation) into `insertelement <2 x double>` and `<4 x i32>`
    # shifts with amounts >= 32, which the Metal source emitter in
    # `compiler/llvm-to-backend/metal/Emitter.cpp` cannot translate to
    # MSL (no `<N x T>` element access emission, no element-wise large
    # shifts on packed vectors). Keeping the soft-fp64 bodies scalar
    # makes them emittable as ordinary Metal helper functions that
    # precede each kernel.
    list(APPEND METAL_LIBKERNEL_EXTRA_ARGS -fno-vectorize -fno-slp-vectorize -fno-unroll-loops)
    # soft-fp64's default fenv mode uses `thread_local` for IEEE flag
    # accumulation. Metal shaders don't support TLS — the resulting
    # `_ZTW...sf64_internal_fe_flagsE` TLS wrapper is unresolvable in MSL.
    # Mode 0 ("disabled") compiles SF64_FE_RAISE to a no-op, removes the
    # thread_local variable, and the stack-local flag-accumulator class
    # becomes empty (its methods DCE to nothing). IEEE exception-flag
    # observability via `sf64_fe_*` is surfaced back to the host anyway,
    # not from inside a GPU kernel, so losing in-kernel accumulation is
    # inherent to any Metal deployment.
    list(APPEND METAL_LIBKERNEL_EXTRA_ARGS -DSOFT_FP64_FENV_MODE=0)
    message(STATUS "Metal libkernel: external soft-fp64 impl from "
                   "${ACPP_METAL_EXTERNAL_FP64_DIR} "
                   "(${ACPP_METAL_EXTERNAL_FP64_SOURCES})")
  else()
    message(STATUS "Metal libkernel: soft-fp64 trap-stub mode (math.cpp) — "
                   "set ACPP_METAL_EXTERNAL_FP64_DIR to link an implementation")
  endif()

  libkernel_generate_bitcode_target(
      TARGETNAME metal
      TRIPLE "arm64-apple-macosx"
      SOURCES ${METAL_LIBKERNEL_BITCODE_SOURCES}
      ADDITIONAL_ARGS ${METAL_LIBKERNEL_EXTRA_ARGS})
endif()
