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
#ifndef HIPSYCL_ZE_BACKEND_HPP
#define HIPSYCL_ZE_BACKEND_HPP


#include <vector>
#include <memory>
#include <level_zero/ze_api.h>

#include "../backend.hpp"
#include "../multi_queue_executor.hpp"
#include "../common/pid_guard.hpp"

#include "ze_allocator.hpp"
#include "ze_hardware_manager.hpp"

namespace hipsycl {
namespace rt {


class ze_backend : public backend
{
public:
  ze_backend();
  virtual api_platform get_api_platform() const override;
  virtual hardware_platform get_hardware_platform() const override;
  virtual backend_id get_unique_backend_id() const override;
  
  virtual backend_hardware_manager* get_hardware_manager() const override;
  virtual backend_executor* get_executor(device_id dev) const override;
  virtual backend_allocator *get_allocator(device_id dev) const override;

  virtual std::string get_name() const override;
  
  std::unique_ptr<backend_executor>
  create_inorder_executor(device_id dev, int priority) override;

  virtual ~ze_backend(){}

private:
  // Detect fork-without-exec at dispatch entry. The Level Zero spec requires
  // zeInit() to run per-process; inherited driver/context/device handles are
  // invalid in a forked child and L0 provides no re-attach path. Match the
  // CUDA backend: refuse loudly and point at the supported alternatives
  // (fork+exec, spawn, MPI, per-process initialization).
  void fail_if_forked() const;

  std::unique_ptr<ze_hardware_manager> _hardware_manager;
  std::unique_ptr<lazily_constructed_executor<multi_queue_executor>> _executor;
  mutable std::vector<ze_allocator> _allocators;
  mutable pid_guard _fork_guard;
};


}
}

#endif

