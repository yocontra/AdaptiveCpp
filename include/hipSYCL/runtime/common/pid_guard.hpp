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
#ifndef HIPSYCL_RUNTIME_COMMON_PID_GUARD_HPP
#define HIPSYCL_RUNTIME_COMMON_PID_GUARD_HPP

#include <atomic>

#if !defined(_WIN32)
#include <unistd.h>
#endif

namespace hipsycl {
namespace rt {

// Detect fork-without-exec by comparing the creator's pid to the current pid.
// After fork() the child inherits the creator's backend handles (device
// contexts, command queues, compiled code objects, XPC connections) that are
// not valid to use or release from the child. Backends embed a pid_guard
// member and query forked() at their dispatch chokepoints; on mismatch they
// run their internal reset path, then call rearm() to arm for the next fork.
//
// Internal runtime helper; not part of the stable SYCL ABI.
class pid_guard {
public:
#if defined(_WIN32)
  // Windows has no fork(); treat every process as the creator.
  pid_guard() noexcept : _creator_pid(0) {}
  bool forked() const noexcept { return false; }
  void rearm() noexcept {}
#else
  pid_guard() noexcept : _creator_pid(static_cast<long>(::getpid())) {}

  bool forked() const noexcept {
    return static_cast<long>(::getpid()) !=
           _creator_pid.load(std::memory_order_acquire);
  }

  // Call after the backend's internal reset has rebuilt state. The next
  // post-fork dispatch from the post-rearm process sees forked() == false.
  void rearm() noexcept {
    _creator_pid.store(static_cast<long>(::getpid()),
                       std::memory_order_release);
  }
#endif

private:
  // long fits any reasonable pid_t (POSIX defines pid_t as a signed integer
  // type; Linux/macOS/BSD all use int or long). Stored as atomic so the
  // rearm-after-reset store is visible to other threads that re-enter the
  // backend before the parent thread's next dispatch completes.
  std::atomic<long> _creator_pid;
};

} // namespace rt
} // namespace hipsycl

#endif
