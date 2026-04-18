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
#ifndef HIPSYCL_METAL_CODE_OBJECT_HPP
#define HIPSYCL_METAL_CODE_OBJECT_HPP

#include <vector>
#include <string>

#include "hipSYCL/runtime/kernel_configuration.hpp"
#include "hipSYCL/runtime/error.hpp"
#include "hipSYCL/runtime/device_id.hpp"
#include "hipSYCL/runtime/kernel_cache.hpp"

namespace MTL {
class Device;
class Library;
class Function;
class ComputePipelineState;
class BinaryArchive;
} // namespace MTL

namespace hipsycl {
namespace rt {

class metal_executable_object : public code_object {
public:
  virtual ~metal_executable_object() {}

  virtual MTL::Library* get_library() const = 0;

  virtual result get_build_result() const = 0;

  virtual MTL::Device* get_device() const = 0;

  // Per-library MTLBinaryArchive containing pre-compiled AGX machine code
  // for every exported kernel. Owned by the executable object; may be null
  // if archive build failed (e.g. forked child without MTLCompilerService
  // access and no cached .metalar on disk). Used at pipeline-state creation
  // with MTLPipelineOptionFailOnBinaryArchiveMiss to avoid post-fork XPC
  // lookups to MTLCompilerService.
  virtual MTL::BinaryArchive* get_binary_archive() const = 0;
};

class metal_sscp_executable_object : public metal_executable_object {
public:
  metal_sscp_executable_object(const std::string &metal_source,
                               const std::string &target_arch,
                               hcf_object_id hcf_source,
                               const std::vector<std::string> &kernel_names,
                               MTL::Device* device,
                               const kernel_configuration &config);

  // Non-copyable and non-movable: owns raw MTL::Library / MTL::BinaryArchive
  // pointers that are released in the destructor.
  metal_sscp_executable_object(const metal_sscp_executable_object&) = delete;
  metal_sscp_executable_object&
  operator=(const metal_sscp_executable_object&) = delete;
  metal_sscp_executable_object(metal_sscp_executable_object&&) = delete;
  metal_sscp_executable_object&
  operator=(metal_sscp_executable_object&&) = delete;

  virtual ~metal_sscp_executable_object();

  virtual result get_build_result() const override;

  virtual code_object_state state() const override;
  virtual code_format format() const override;
  virtual backend_id managing_backend() const override;
  virtual hcf_object_id hcf_source() const override;
  virtual std::string target_arch() const override;
  virtual compilation_flow source_compilation_flow() const override;

  virtual std::vector<std::string>
  supported_backend_kernel_names() const override;
  virtual bool contains(const std::string &backend_kernel_name) const override;

  virtual MTL::Library* get_library() const override;
  virtual MTL::Device* get_device() const override;
  virtual MTL::BinaryArchive* get_binary_archive() const override;

  const std::string& get_msl_source() const { return _msl_source; }

private:
  result build(const std::string& source);

  std::string _target_arch;
  hcf_object_id _hcf;
  std::vector<std::string> _kernel_names;
  result _build_result;
  kernel_configuration::id_type _id;
  MTL::Device* _device;
  MTL::Library* _library;
  MTL::BinaryArchive* _archive;

  // Keep the MSL source for potential debugging
  std::string _msl_source;
};

} // namespace rt
} // namespace hipsycl

#endif // HIPSYCL_METAL_CODE_OBJECT_HPP
