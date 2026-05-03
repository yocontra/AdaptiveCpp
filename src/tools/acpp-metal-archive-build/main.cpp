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
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/stat.h>

namespace {

// Skip-when-too-large size threshold. Soft-fp64 metallibs (e.g. sphere_distance
// + st_length under SLEEF lowering) emit 1.1+ MB metallibs whose AGX pipeline
// states OOM the helper at addComputePipelineFunctions / serializeToURL with
// SIGKILL exit 137. When that happens the parent runtime sees a generic crash
// and falls back to the next-startup MTLCompilerService path — which crashes
// in forked children. This threshold makes the helper exit gracefully (exit 9)
// before the OOM, signalling to the runtime that pipeline-state archival was
// skipped on purpose so it can either accept slower in-process JIT or gate the
// kernel out for the current backend. Override with ACPP_METAL_ARCHIVE_MAX_BYTES.
constexpr long long DEFAULT_MAX_METALLIB_BYTES = 900 * 1024;  // 900 KiB

long long get_metallib_size_limit() {
  if (const char* env = std::getenv("ACPP_METAL_ARCHIVE_MAX_BYTES")) {
    char* end = nullptr;
    long long parsed = std::strtoll(env, &end, 10);
    if (end != env && parsed > 0) {
      return parsed;
    }
  }
  return DEFAULT_MAX_METALLIB_BYTES;
}

long long file_size_bytes(const char* path) {
  struct stat st;
  if (::stat(path, &st) != 0) {
    return -1;
  }
  return static_cast<long long>(st.st_size);
}

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

  // Skip-when-too-large size threshold (added to address OOM SIGKILL exit 137
  // on soft-fp64 metallibs such as sphere_distance / st_length). See
  // namespace-level DEFAULT_MAX_METALLIB_BYTES for full rationale. Exit 9 is
  // distinct from any of the existing failure exit codes so the runtime can
  // distinguish "intentionally skipped, archive will not exist for this lib"
  // from "helper crashed, retry pending".
  long long metallib_size = file_size_bytes(metallib_path);
  long long size_limit = get_metallib_size_limit();
  if (metallib_size < 0) {
    std::fprintf(stderr,
        "acpp-metal-archive-build: cannot stat metallib %s\n", metallib_path);
    return 4;
  }
  if (metallib_size > size_limit) {
    std::fprintf(stderr,
        "acpp-metal-archive-build: skipping archive build for %s "
        "(size=%lld bytes exceeds limit=%lld bytes; soft-fp64 OOM guard)\n",
        metallib_path, metallib_size, size_limit);
    return 9;
  }

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
