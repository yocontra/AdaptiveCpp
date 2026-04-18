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
#include "hipSYCL/runtime/backend_loader.hpp"

#include "hipSYCL/runtime/hip/hip_backend.hpp"
#include "hipSYCL/runtime/hip/hip_event.hpp"
#include "hipSYCL/runtime/hip/hip_target.hpp"
#include "hipSYCL/runtime/hip/hip_queue.hpp"
#include "hipSYCL/runtime/kernel_cache.hpp"
#include "hipSYCL/runtime/multi_queue_executor.hpp"

HIPSYCL_PLUGIN_API_EXPORT
hipsycl::rt::backend *hipsycl_backend_plugin_create() {
  return new hipsycl::rt::hip_backend();
}

static const char *backend_name = "hip";

HIPSYCL_PLUGIN_API_EXPORT
const char *hipsycl_backend_plugin_get_name() {
  return backend_name;
}


namespace hipsycl {
namespace rt {


namespace {

std::unique_ptr<multi_queue_executor>
create_multi_queue_executor(hip_backend *b) {
  return std::make_unique<multi_queue_executor>(
      *b, [b](device_id dev) { return std::make_unique<hip_queue>(b, dev); });
}

}

hip_backend::hip_backend()
    : _hw_manager{hip_backend::get_hardware_platform()},
      _executor{[this]() {
        return create_multi_queue_executor(this);
      }} {}

api_platform hip_backend::get_api_platform() const {
  return api_platform::hip;
}

hardware_platform hip_backend::get_hardware_platform() const {
#ifdef HIPSYCL_RT_HIP_TARGET_CUDA
  return hardware_platform::cuda;
#elif defined(HIPSYCL_RT_HIP_TARGET_ROCM)
  return hardware_platform::rocm;
#elif defined(HIPSYCL_RT_HIP_TARGET_HIPCPU)
  return hardware_platform::cpu;
#else
  #error HIP Backend: Not HIP Target specified
#endif
}

backend_id hip_backend::get_unique_backend_id() const {
  return backend_id::hip;
}

backend_hardware_manager *hip_backend::get_hardware_manager() const {
  maybe_reset_after_fork();
  return &_hw_manager;
}

backend_executor *hip_backend::get_executor(device_id dev) const {
  if (dev.get_full_backend_descriptor().sw_platform != api_platform::hip) {
    register_error(
        __acpp_here(),
        error_info{"hip_backend: Passed device id from other backend to HIP backend"});
    return nullptr;
  }

  maybe_reset_after_fork();
  return _executor.get();
}

backend_allocator* hip_backend::get_allocator(device_id dev) const {
  assert(dev.get_backend() == this->get_unique_backend_id());
  return static_cast<hip_hardware_context *>(
             get_hardware_manager()->get_device(dev.get_id()))
      ->get_allocator();
}

hip_event_pool* hip_backend::get_event_pool(device_id dev) const {
  assert(dev.get_backend() == this->get_unique_backend_id());
  return static_cast<hip_hardware_context *>(
             get_hardware_manager()->get_device(dev.get_id()))
      ->get_event_pool();
}

void hip_backend::maybe_reset_after_fork() const {
  if (_fork_guard.forked()) {
    reset_after_fork_internal();
    _fork_guard.rearm();
  }
}

void hip_backend::reset_after_fork_internal() const {
#if defined(HIPSYCL_RT_HIP_TARGET_ROCM)
  // ROCm / libhsakmt supports fork-without-exec: the KFD thunk's
  // hsakmt_is_forked_child() + clear_after_fork() path re-attaches
  // /dev/kfd, rebuilds doorbells/events, and restores the VM aperture on
  // the first post-fork ioctl. Cooperate by dropping inherited state and
  // rebuilding.
  if (auto cache = kernel_cache::get())
    cache->drop_code_objects_for_backend(backend_id::hip);
  _executor.abandon_after_fork();
  _hw_manager.reset_after_fork();
#elif defined(HIPSYCL_RT_HIP_TARGET_CUDA)
  // HIP-over-CUDA: the underlying driver forbids fork-without-exec (see
  // NVIDIA CUDA C Programming Guide). Match the native CUDA backend and
  // refuse loudly rather than silently corrupt.
  register_error(
      __acpp_here(),
      error_info{
          "hip_backend (HIP-over-CUDA): the CUDA driver does not support "
          "fork() without exec() and NVIDIA documents any use of CUDA in a "
          "forked child as undefined behavior. Use fork+exec, the Python "
          "'spawn' start method, MPI, or NVIDIA MPS for multi-tenant GPU "
          "sharing. See "
          "https://docs.nvidia.com/cuda/cuda-c-programming-guide/"
          "#cuda-and-fork",
          error_type::runtime_error});
#else
  // HIP-CPU: no device handles, nothing to rebuild.
  if (auto cache = kernel_cache::get())
    cache->drop_code_objects_for_backend(backend_id::hip);
  _executor.abandon_after_fork();
#endif
}

std::string hip_backend::get_name() const {
  return "HIP";
}

std::unique_ptr<backend_executor>
hip_backend::create_inorder_executor(device_id dev, int priority) {
  std::unique_ptr<inorder_queue> q =
      std::make_unique<hip_queue>(this, dev, priority);

  return std::make_unique<inorder_executor>(std::move(q));
}
  
}
}