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
#include <vector>

#include "../backend.hpp"
#include "../multi_queue_executor.hpp"
#include "../common/pid_guard.hpp"

#include "cuda_allocator.hpp"
#include "cuda_queue.hpp"
#include "cuda_hardware_manager.hpp"
#include "cuda_event_pool.hpp"

#ifndef HIPSYCL_CUDA_BACKEND_HPP
#define HIPSYCL_CUDA_BACKEND_HPP

namespace hipsycl {
namespace rt {


class cuda_backend : public backend
{
public:
  cuda_backend();
  virtual api_platform get_api_platform() const override;
  virtual hardware_platform get_hardware_platform() const override;
  virtual backend_id get_unique_backend_id() const override;
  
  virtual backend_hardware_manager* get_hardware_manager() const override;
  virtual backend_executor* get_executor(device_id dev) const override;
  virtual backend_allocator *get_allocator(device_id dev) const override;

  virtual std::string get_name() const override;
  
  virtual ~cuda_backend(){}

  cuda_event_pool* get_event_pool(device_id dev) const;

  virtual std::unique_ptr<backend_executor>
  create_inorder_executor(device_id dev, int priority) override;
private:
  // Detect fork-without-exec at dispatch entry. Unlike Metal and HIP, the
  // CUDA driver documents any use of CUDA after fork() without exec() as
  // undefined behavior (CUDA C Programming Guide: "CUDA does not duplicate
  // any of its internal data structures"); in practice a forked child sees
  // CUDA_ERROR_NOT_INITIALIZED plus corrupted context state. Every
  // mainstream CUDA framework (PyTorch, TensorFlow, JAX, RAPIDS, cuPy)
  // refuses loudly instead of attempting recovery. This backend does the
  // same: register_error with a clear pointer at the supported alternatives
  // (fork+exec, spawn, MPI, MPS).
  void fail_if_forked() const;

  mutable cuda_hardware_manager _hw_manager;
  mutable lazily_constructed_executor<multi_queue_executor> _executor;
  mutable pid_guard _fork_guard;
};

}
}


#endif
