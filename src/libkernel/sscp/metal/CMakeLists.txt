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
      SOURCES ${METAL_LIBKERNEL_BITCODE_SOURCES})
endif()
