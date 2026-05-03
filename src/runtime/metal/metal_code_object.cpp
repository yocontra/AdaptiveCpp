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
#include "hipSYCL/runtime/metal/metal_code_object.hpp"

#include <Metal/Metal.hpp>

#undef nil

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

#include "hipSYCL/common/debug.hpp"
#include "hipSYCL/common/filesystem.hpp"
#include "hipSYCL/runtime/kernel_configuration.hpp"

extern char** environ;

namespace hipsycl {
namespace rt {

namespace {

// Wait for `pid`, retrying on EINTR. Returns the process exit status (0..255)
// on normal exit, or -1 on abnormal termination / waitpid failure.
int wait_for_child_status(pid_t pid, const char *tag) {
  int status = 0;
  while (waitpid(pid, &status, 0) < 0) {
    if (errno == EINTR) continue;
    HIPSYCL_DEBUG_WARNING << "metal_code_object: waitpid failed for " << tag
                          << ": " << std::strerror(errno) << std::endl;
    return -1;
  }
  if (!WIFEXITED(status)) {
    HIPSYCL_DEBUG_WARNING
        << "metal_code_object: " << tag << " terminated abnormally"
        << std::endl;
    return -1;
  }
  return WEXITSTATUS(status);
}

// Wait for `pid`, retrying on EINTR. Returns true iff the process exited
// normally with status 0. Logs non-zero exits at WARNING level.
bool wait_for_child(pid_t pid, const char *tag) {
  int rc = wait_for_child_status(pid, tag);
  if (rc != 0) {
    HIPSYCL_DEBUG_WARNING
        << "metal_code_object: " << tag << " exited non-zero (" << rc << ")"
        << std::endl;
    return false;
  }
  return true;
}

// posix_spawnp `xcrun` with the given trailing arguments. Avoids the shell,
// so paths with spaces or other metacharacters are passed safely.
bool spawn_xcrun(const std::vector<std::string> &args) {
  std::vector<char *> argv;
  argv.reserve(args.size() + 1);
  // posix_spawn takes char* const argv[]; data() gives us a writable pointer
  // to each string's buffer, which is sufficient (the child does not mutate).
  for (const auto &a : args) argv.push_back(const_cast<char *>(a.c_str()));
  argv.push_back(nullptr);

  pid_t pid = 0;
  int rc = posix_spawnp(&pid, argv[0], nullptr, nullptr, argv.data(), environ);
  if (rc != 0) {
    HIPSYCL_DEBUG_WARNING << "metal_code_object: posix_spawnp(" << argv[0]
                          << ") failed: " << std::strerror(rc) << std::endl;
    return false;
  }
  return wait_for_child(pid, argv[0]);
}

// Pre-compile MSL to a `.metallib` file via `xcrun metal` + `xcrun metallib`.
// Loading a `.metallib` via `device->newLibrary(url, ...)` does NOT route
// through `MTLCompilerService`, which makes the resulting library usable in
// a forked child (where the parent's MTLCompilerService XPC connection is
// dead). Returns the absolute path on success, empty string on failure.
std::string compile_msl_to_metallib(const std::string &source,
                                    const std::string &id_str) {
  using namespace common::filesystem;
  const std::string cache_dir = persistent_storage::get().get_jit_cache_dir();
  const std::string metal_path = join_path(cache_dir, id_str + ".metal");
  const std::string air_path = join_path(cache_dir, id_str + ".air");
  const std::string metallib_path = join_path(cache_dir, id_str + ".metallib");

  if (!atomic_write(metal_path, source)) {
    HIPSYCL_DEBUG_WARNING
        << "metal_code_object: could not write MSL source to " << metal_path
        << std::endl;
    return {};
  }

  if (!spawn_xcrun({"xcrun", "metal", "-c", metal_path, "-o", air_path,
                    "-Wno-unused-function"})) {
    HIPSYCL_DEBUG_WARNING
        << "metal_code_object: xcrun metal failed for " << metal_path
        << std::endl;
    return {};
  }

  if (!spawn_xcrun({"xcrun", "metallib", air_path, "-o", metallib_path})) {
    HIPSYCL_DEBUG_WARNING
        << "metal_code_object: xcrun metallib failed for " << air_path
        << std::endl;
    return {};
  }

  if (std::getenv("ACPP_METAL_KEEP_SOURCE") == nullptr) {
    std::remove(metal_path.c_str());
  }
  std::remove(air_path.c_str());

  HIPSYCL_DEBUG_INFO
      << "metal_code_object: produced fork-safe metallib at " << metallib_path
      << std::endl;
  return metallib_path;
}

result load_metal_library_from_url(MTL::Library *&library, MTL::Device *device,
                                   const std::string &metallib_path) {
  if (!device) {
    return make_error(__acpp_here(),
                      error_info{"metal_code_object: Device is null"});
  }

  NS::String *path_str =
      NS::String::string(metallib_path.c_str(), NS::UTF8StringEncoding);
  NS::URL *url = NS::URL::fileURLWithPath(path_str);

  NS::Error *error = nullptr;
  library = device->newLibrary(url, &error);

  if (error || !library) {
    std::string msg = "metal_code_object: newLibrary(url) failed";
    if (error && error->localizedDescription()) {
      msg += ": ";
      msg += error->localizedDescription()->utf8String();
    }
    return make_error(__acpp_here(), error_info{msg});
  }

  HIPSYCL_DEBUG_INFO
      << "metal_code_object: loaded precompiled metallib from "
      << metallib_path << std::endl;
  return make_success();
}

// In-process MSL source compile via device->newLibrary. Only safe on the
// original (pre-fork) process, since this routes through MTLCompilerService.
result build_metal_library_from_source(MTL::Library*& library,
                                       MTL::Device* device,
                                       const std::string& source) {
  if (!device) {
    return make_error(__acpp_here(),
                      error_info{"metal_code_object: Device is null"});
  }

  NS::Error* error = nullptr;

  NS::SharedPtr<MTL::CompileOptions> options = NS::TransferPtr(MTL::CompileOptions::alloc()->init());
  options->setLanguageVersion(MTL::LanguageVersion4_0);
  options->setOptimizationLevel(MTL::LibraryOptimizationLevel::LibraryOptimizationLevelSize);

  NS::String* sourceString = NS::String::string(source.c_str(),
                                                NS::UTF8StringEncoding);

  library = device->newLibrary(sourceString, options.get(), &error);

  if (error) {
    std::string error_msg = "metal_code_object: Shader compilation failed";
    if (error->localizedDescription()) {
      error_msg += ": ";
      error_msg += error->localizedDescription()->utf8String();
    }
    return make_error(__acpp_here(), error_info{error_msg});
  }

  if (!library) {
    return make_error(__acpp_here(),
                      error_info{"metal_code_object: Library creation failed "
                                 "without error message"});
  }

  HIPSYCL_DEBUG_INFO << "metal_code_object: Successfully compiled Metal shader"
                     << std::endl;

  return make_success();
}

// Locate the acpp-metal-archive-build helper binary. We check, in order:
//   1) ACPP_METAL_ARCHIVE_BUILD env override
//   2) <dir of libacpp-rt.dylib>/../bin/acpp-metal-archive-build
//   3) PATH via execvp-style bare name (posix_spawnp handles this)
// Returns empty string to mean "spawn by bare name via posix_spawnp".
std::string locate_archive_builder() {
  if (const char* override_path = std::getenv("ACPP_METAL_ARCHIVE_BUILD")) {
    if (*override_path) return override_path;
  }
  // Try to derive from the runtime lib location. Install layouts we need
  // to support:
  //   <prefix>/lib/libacpp-rt.dylib          -> ../bin/
  //   <prefix>/lib/hipSYCL/librt-backend-metal.dylib -> ../../bin/
  Dl_info info;
  if (dladdr(reinterpret_cast<void*>(&locate_archive_builder), &info) && info.dli_fname) {
    std::filesystem::path lib{info.dli_fname};
    std::error_code ec;
    for (auto dir = lib.parent_path(); !dir.empty() && dir.has_parent_path();
         dir = dir.parent_path()) {
      auto candidate = dir.parent_path() / "bin" / "acpp-metal-archive-build";
      if (std::filesystem::exists(candidate, ec) && !ec) {
        return candidate.string();
      }
      // Stop after climbing past the install prefix.
      if (dir.filename() == "bin" || dir.filename() == "") break;
    }
  }
  return {};
}

// Result of spawning `acpp-metal-archive-build`. The runtime distinguishes
// "success" (load the archive) from "intentionally skipped" (no archive on
// disk, but not a failure — kernel will JIT in-process at first dispatch in
// each backend) from "failure" (something actually went wrong).
enum class archive_builder_result {
  // Archive was written to `metalar_path` and should be loaded.
  built,
  // Helper deliberately skipped this lib (e.g. metallib too large, exit 9).
  // No archive on disk; do not treat as an error.
  skipped,
  // Real failure: helper crashed, exited with an unexpected non-zero status,
  // posix_spawn failed, etc.
  failed,
};

// Spawn `acpp-metal-archive-build <metallib> <metalar> <kernel>...` and wait.
// Distinguishes built / skipped / failed via the helper's exit status.
//
// Helper exit-code contract (see src/tools/acpp-metal-archive-build/main.cpp):
//   0 -> archive written
//   9 -> intentionally skipped (metallib exceeded ACPP_METAL_ARCHIVE_MAX_BYTES);
//        no archive produced. Kernel is still functional via in-process JIT in
//        the original (pre-fork) parent backend; forked children that hit the
//        same kernel cold will still fail at MTLCompilerService — caller is
//        responsible for documenting that limitation for skipped libs.
//   anything else -> real failure
archive_builder_result spawn_archive_builder(
    const std::string& metallib_path,
    const std::string& metalar_path,
    const std::vector<std::string>& kernel_names) {
  if (kernel_names.empty()) {
    HIPSYCL_DEBUG_WARNING
        << "metal_code_object: no kernel names; skipping archive build"
        << std::endl;
    return archive_builder_result::failed;
  }

  std::string helper = locate_archive_builder();
  const char* argv0 = helper.empty() ? "acpp-metal-archive-build" : helper.c_str();

  std::vector<std::string> owned;
  owned.reserve(kernel_names.size() + 3);
  owned.emplace_back(argv0);
  owned.emplace_back(metallib_path);
  owned.emplace_back(metalar_path);
  for (const auto& k : kernel_names) owned.emplace_back(k);

  std::vector<char*> argv;
  argv.reserve(owned.size() + 1);
  for (auto& s : owned) argv.push_back(s.data());
  argv.push_back(nullptr);

  pid_t pid = 0;
  int rc = helper.empty()
               ? posix_spawnp(&pid, argv0, nullptr, nullptr, argv.data(), environ)
               : posix_spawn(&pid, argv0, nullptr, nullptr, argv.data(), environ);
  if (rc != 0) {
    HIPSYCL_DEBUG_WARNING
        << "metal_code_object: posix_spawn(" << argv0 << ") failed: "
        << std::strerror(rc) << std::endl;
    return archive_builder_result::failed;
  }

  int exit_status = wait_for_child_status(pid, "acpp-metal-archive-build");
  if (exit_status == 0) {
    return archive_builder_result::built;
  }
  if (exit_status == 9) {
    // Helper deliberately skipped this metallib. Surface a single warning
    // naming the metallib + the override knob so the user can opt back in if
    // they have headroom. The kernels will still work in the parent backend
    // via in-process JIT at first dispatch; forked workers that hit a skipped
    // kernel cold will still crash at MTLCompilerService — that's a known
    // tradeoff documented in CLAUDE.md (MTLBinaryArchive cache).
    std::error_code ec;
    auto sz = std::filesystem::file_size(metallib_path, ec);
    HIPSYCL_DEBUG_WARNING
        << "metal_code_object: archive build skipped for " << metallib_path
        << " (size=" << (ec ? 0 : sz)
        << " bytes; raise ACPP_METAL_ARCHIVE_MAX_BYTES to opt in). Kernel will "
           "JIT at first dispatch."
        << std::endl;
    return archive_builder_result::skipped;
  }
  HIPSYCL_DEBUG_WARNING
      << "metal_code_object: acpp-metal-archive-build exited with status "
      << exit_status << " for " << metallib_path << std::endl;
  return archive_builder_result::failed;
}

// Load an already-serialized `.metalar` file into an MTL::BinaryArchive.
// Safe in a forked child: deserialization does not route through
// MTLCompilerService.
MTL::BinaryArchive* load_binary_archive_from_url(MTL::Device* device,
                                                 const std::string& metalar_path) {
  if (!device) return nullptr;
  NS::String* path_str =
      NS::String::string(metalar_path.c_str(), NS::UTF8StringEncoding);
  NS::URL* url = NS::URL::fileURLWithPath(path_str);

  MTL::BinaryArchiveDescriptor* desc =
      MTL::BinaryArchiveDescriptor::alloc()->init();
  desc->setUrl(url);

  NS::Error* err = nullptr;
  MTL::BinaryArchive* archive = device->newBinaryArchive(desc, &err);
  desc->release();

  if (!archive) {
    std::string msg = "metal_code_object: newBinaryArchive(url) failed";
    if (err && err->localizedDescription()) {
      msg += ": ";
      msg += err->localizedDescription()->utf8String();
    }
    HIPSYCL_DEBUG_WARNING << msg << std::endl;
    return nullptr;
  }
  HIPSYCL_DEBUG_INFO
      << "metal_code_object: loaded binary archive from " << metalar_path
      << std::endl;
  return archive;
}

} // anonymous namespace

metal_sscp_executable_object::metal_sscp_executable_object(
    const std::string &metal_source, const std::string &target_arch,
    hcf_object_id hcf_source, const std::vector<std::string> &kernel_names,
    MTL::Device* device, const kernel_configuration &config)
    : _target_arch{target_arch}, _hcf{hcf_source}, _kernel_names{kernel_names},
      _id{config.generate_id()}, _device{device}, _library{nullptr},
      _archive{nullptr}, _msl_source{metal_source} {
  _build_result = build(metal_source);
}

metal_sscp_executable_object::~metal_sscp_executable_object() {
  if (_archive) {
    _archive->release();
  }
  if (_library) {
    _library->release();
  }
}

result metal_sscp_executable_object::get_build_result() const {
  return _build_result;
}

code_object_state metal_sscp_executable_object::state() const {
  return _library ? code_object_state::executable : code_object_state::invalid;
}

code_format metal_sscp_executable_object::format() const {
  // Metal uses its own shading language format
  return code_format::native_isa;
}

backend_id metal_sscp_executable_object::managing_backend() const {
  return backend_id::metal;
}

hcf_object_id metal_sscp_executable_object::hcf_source() const {
  return _hcf;
}

std::string metal_sscp_executable_object::target_arch() const {
  return _target_arch;
}

compilation_flow metal_sscp_executable_object::source_compilation_flow() const {
  return compilation_flow::sscp;
}

std::vector<std::string>
metal_sscp_executable_object::supported_backend_kernel_names() const {
  return _kernel_names;
}

MTL::Library* metal_sscp_executable_object::get_library() const {
  return _library;
}

MTL::BinaryArchive* metal_sscp_executable_object::get_binary_archive() const {
  return _archive;
}

MTL::Device* metal_sscp_executable_object::get_device() const {
  return _device;
}

result metal_sscp_executable_object::build(const std::string& source) {
  if (_library != nullptr)
    return make_success();

  using namespace common::filesystem;
  const std::string id_str = kernel_configuration::to_string(_id);
  const std::string cache_dir = persistent_storage::get().get_jit_cache_dir();
  const std::string metallib_path = join_path(cache_dir, id_str + ".metallib");
  const std::string metalar_path = join_path(cache_dir, id_str + ".metalar");

  auto populate_archive = [&](const std::string& produced_metallib) {
    // Load a previously-serialized archive if present; otherwise spawn the
    // helper subprocess to produce one and load the result. Failure is
    // non-fatal: _archive stays nullptr and the pipeline-state path falls
    // through to in-process compile (which will fail on a forked child, but
    // is useful for diagnostics on the parent).
    std::error_code ec;
    if (std::filesystem::exists(metalar_path, ec) && !ec) {
      _archive = load_binary_archive_from_url(_device, metalar_path);
      if (_archive) return;
      HIPSYCL_DEBUG_WARNING
          << "metal_code_object: cached metalar at " << metalar_path
          << " failed to load; rebuilding" << std::endl;
      std::remove(metalar_path.c_str());
    }
    archive_builder_result rc =
        spawn_archive_builder(produced_metallib, metalar_path, _kernel_names);
    switch (rc) {
      case archive_builder_result::built:
        _archive = load_binary_archive_from_url(_device, metalar_path);
        break;
      case archive_builder_result::skipped:
        // Intentional skip (e.g. metallib too large). _archive stays nullptr;
        // the kernel-launch path falls back to in-process pipeline-state
        // creation (works in parent backend, will fail on forked children for
        // this specific kernel — known tradeoff, logged once above).
        break;
      case archive_builder_result::failed:
        // Real failure. _archive stays nullptr; the warning was already
        // emitted by spawn_archive_builder. The fast-path metallib load
        // still succeeded so the parent backend can run; forked children
        // will hit MTLCompilerService.
        break;
    }
  };

  // Fast path: previously produced .metallib on disk. newLibrary(url) does
  // not require MTLCompilerService, so this path is safe on the forked child
  // side where the parent's XPC connection is no longer reachable.
  std::error_code ec;
  if (std::filesystem::exists(metallib_path, ec) && !ec) {
    auto r = load_metal_library_from_url(_library, _device, metallib_path);
    if (r.is_success()) {
      populate_archive(metallib_path);
      return r;
    }
    HIPSYCL_DEBUG_WARNING
        << "metal_code_object: cached metallib at " << metallib_path
        << " failed to load; removing and falling back to compile"
        << std::endl;
    std::remove(metallib_path.c_str());
    _library = nullptr;
  }

  // Slow path: no cached metallib. Compile out-of-process via `xcrun metal`
  // and `xcrun metallib`. The subprocess has its own MTLCompilerService
  // connection, so this works in both the pre-fork parent and a forked child
  // that has already lost access to the inherited connection.
  std::string produced = compile_msl_to_metallib(source, id_str);
  if (!produced.empty()) {
    auto r = load_metal_library_from_url(_library, _device, produced);
    if (r.is_success()) {
      populate_archive(produced);
      return r;
    }
    _library = nullptr;
  }

  // Last-resort: direct in-process source compile. Only expected to succeed
  // on the pre-fork parent; retained for environments without an Xcode
  // metal toolchain on PATH.
  return build_metal_library_from_source(_library, _device, source);
}

bool metal_sscp_executable_object::contains(
    const std::string &backend_kernel_name) const {
  for (const auto& kernel_name : _kernel_names) {
    if (kernel_name == backend_kernel_name)
      return true;
  }
  return false;
}

} // namespace rt
} // namespace hipsycl
