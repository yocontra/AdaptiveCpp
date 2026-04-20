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

// acpp-metal-archive-build: subprocess helper that produces a serialized
// MTLBinaryArchive (.metalar) from a compiled .metallib + a list of kernel
// names. Called by the Metal backend from metal_sscp_executable_object::build
// when it needs to materialize an archive for a forked child to load via
// newBinaryArchive(url) without re-entering MTLCompilerService.
//
// This tool runs in a fresh process (and therefore has its own live
// MTLCompilerService XPC connection), so AIR->AGX compilation via
// addComputePipelineFunctions succeeds even when the parent process that
// spawned us no longer has a usable XPC connection.

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

void log_err(const char* msg, NS::Error* error = nullptr) {
  std::fprintf(stderr, "acpp-metal-archive-build: %s", msg);
  if (error && error->localizedDescription()) {
    std::fprintf(stderr, ": %s", error->localizedDescription()->utf8String());
  }
  std::fputc('\n', stderr);
}

NS::String* ns(const char* s) {
  return NS::String::string(s, NS::UTF8StringEncoding);
}

NS::URL* file_url(const char* path) {
  return NS::URL::fileURLWithPath(ns(path));
}

} // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr,
        "usage: %s <input.metallib> <output.metalar> <kernel1> [kernel2 ...]\n",
        argv[0]);
    return 2;
  }

  const char* metallib_path = argv[1];
  const char* metalar_path = argv[2];

  NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();

  // Use MTLCopyAllDevices (IOKit-backed, WindowServer-free) for consistency
  // with the runtime's metal_hardware_manager and to keep this helper
  // usable in environments without a WindowServer (headless CI, sandboxed
  // spawns). On Apple Silicon there is exactly one integrated GPU, so [0]
  // is equivalent to the system default. Retain the device so it survives
  // the array release below.
  NS::Array* all_devices = MTL::CopyAllDevices();
  if (!all_devices || all_devices->count() == 0) {
    log_err("MTLCopyAllDevices returned no devices");
    if (all_devices) all_devices->release();
    pool->release();
    return 3;
  }
  MTL::Device* device =
      static_cast<MTL::Device*>(all_devices->object(0));
  device->retain();
  all_devices->release();

  NS::Error* err = nullptr;
  MTL::Library* library = device->newLibrary(file_url(metallib_path), &err);
  if (!library) {
    log_err("newLibrary failed", err);
    device->release();
    pool->release();
    return 4;
  }

  MTL::BinaryArchiveDescriptor* archive_desc =
      MTL::BinaryArchiveDescriptor::alloc()->init();
  // Leave url unset to create an empty archive.
  MTL::BinaryArchive* archive = device->newBinaryArchive(archive_desc, &err);
  archive_desc->release();
  if (!archive) {
    log_err("newBinaryArchive failed", err);
    library->release();
    device->release();
    pool->release();
    return 5;
  }

  int kernels_added = 0;
  int kernels_skipped = 0;
  for (int i = 3; i < argc; ++i) {
    const char* kernel_name = argv[i];
    MTL::Function* function = library->newFunction(ns(kernel_name));
    if (!function) {
      std::fprintf(stderr,
          "acpp-metal-archive-build: kernel '%s' not found in library; skipping\n",
          kernel_name);
      ++kernels_skipped;
      continue;
    }

    MTL::ComputePipelineDescriptor* pipe_desc =
        MTL::ComputePipelineDescriptor::alloc()->init();
    pipe_desc->setComputeFunction(function);

    NS::Error* add_err = nullptr;
    if (!archive->addComputePipelineFunctions(pipe_desc, &add_err)) {
      std::fprintf(stderr,
          "acpp-metal-archive-build: addComputePipelineFunctions('%s') failed",
          kernel_name);
      if (add_err && add_err->localizedDescription()) {
        std::fprintf(stderr, ": %s", add_err->localizedDescription()->utf8String());
      }
      std::fputc('\n', stderr);
      pipe_desc->release();
      function->release();
      archive->release();
      library->release();
      device->release();
      pool->release();
      return 6;
    }

    pipe_desc->release();
    function->release();
    ++kernels_added;
  }

  if (kernels_added == 0) {
    log_err("no kernels were added to the archive");
    archive->release();
    library->release();
    device->release();
    pool->release();
    return 7;
  }

  NS::Error* ser_err = nullptr;
  if (!archive->serializeToURL(file_url(metalar_path), &ser_err)) {
    log_err("serializeToURL failed", ser_err);
    archive->release();
    library->release();
    device->release();
    pool->release();
    return 8;
  }

  std::fprintf(stdout,
      "acpp-metal-archive-build: wrote %s (kernels_added=%d skipped=%d)\n",
      metalar_path, kernels_added, kernels_skipped);

  archive->release();
  library->release();
  device->release();
  pool->release();
  return 0;
}
