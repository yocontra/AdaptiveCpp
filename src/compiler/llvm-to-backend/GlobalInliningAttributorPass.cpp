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
#include "hipSYCL/compiler/llvm-to-backend/GlobalInliningAttributorPass.hpp"

#include <llvm/IR/Module.h>

namespace hipsycl {
namespace compiler {

GlobalInliningAttributorPass::GlobalInliningAttributorPass(const std::vector<std::string> &KN)
    : KernelNames{KN} {}

llvm::PreservedAnalyses GlobalInliningAttributorPass::run(llvm::Module &M,
                                                          llvm::ModuleAnalysisManager &MAM) {

  llvm::SmallPtrSet<llvm::Function*, 16> Kernels;

  for(auto& KN : KernelNames)
    if(auto* F = M.getFunction(KN))
      Kernels.insert(F);

  for(auto &F : M) {
    // Ignore kernels and intrinsics
    if (!F.isIntrinsic() && !Kernels.contains(&F)) {
      // Ignore undefined functions
      if(!F.empty()) {
        // Respect an explicit `noinline` attribute. Backends mark
        // functions that must survive to source emission (e.g. Metal
        // soft-fp64 primitives - the Metal emitter translates IR-level
        // fp64 ops to source-level call strings that reference these
        // symbols, so their bodies must remain in the module and be
        // emitted as separate helpers rather than inlined away.)
        if (F.hasFnAttribute(llvm::Attribute::NoInline))
          continue;
        F.setLinkage(llvm::GlobalValue::InternalLinkage);
        if(!F.hasFnAttribute(llvm::Attribute::AlwaysInline))
          F.addFnAttr(llvm::Attribute::AlwaysInline);
      }
    }
  }
  return llvm::PreservedAnalyses::none();
}
}
}


