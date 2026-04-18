cmake_minimum_required(VERSION 3.12)

# Fork-safety tests exercise each backend's per-process reset path at its
# dispatch chokepoint. Each backend has its own expected outcome:
#   Metal, HIP/ROCm  — transparent recovery
#   CUDA, Level Zero — loud refusal with clear error
#   HIP over CUDA    — loud refusal (delegates to CUDA's answer)
# Each test is skipped if its backend wasn't built.

if(WITH_ACCELERATED_CPU OR APPLE)
  # Metal is only built on Apple; the test binary links SYCL unconditionally
  # and decides at runtime whether a Metal device is present.
  if(APPLE)
    add_executable(fork_safety_metal metal.cpp)
    target_include_directories(fork_safety_metal PRIVATE ${Boost_INCLUDE_DIRS})
    target_link_libraries(fork_safety_metal PRIVATE Threads::Threads)
    add_sycl_to_target(TARGET fork_safety_metal)
    add_test(NAME fork_safety_metal COMMAND fork_safety_metal)
    set_tests_properties(fork_safety_metal PROPERTIES LABELS "fork_safety")
  endif()
endif()

if(WITH_ROCM_BACKEND OR WITH_CUDA_BACKEND)
  # HIP backend target: the test compiles regardless and picks its assertion
  # at runtime based on the HIP target macro.
  add_executable(fork_safety_hip hip.cpp)
  target_include_directories(fork_safety_hip PRIVATE ${Boost_INCLUDE_DIRS})
  target_link_libraries(fork_safety_hip PRIVATE Threads::Threads)
  add_sycl_to_target(TARGET fork_safety_hip)
  add_test(NAME fork_safety_hip COMMAND fork_safety_hip)
  set_tests_properties(fork_safety_hip PROPERTIES LABELS "fork_safety")
endif()

if(WITH_CUDA_BACKEND)
  add_executable(fork_safety_cuda cuda.cpp)
  target_include_directories(fork_safety_cuda PRIVATE ${Boost_INCLUDE_DIRS})
  target_link_libraries(fork_safety_cuda PRIVATE Threads::Threads)
  add_sycl_to_target(TARGET fork_safety_cuda)
  add_test(NAME fork_safety_cuda COMMAND fork_safety_cuda)
  set_tests_properties(fork_safety_cuda PROPERTIES LABELS "fork_safety")
endif()
