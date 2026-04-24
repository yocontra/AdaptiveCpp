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
#include "hipSYCL/compiler/llvm-to-backend/metal/LLVMToMetal.hpp"
#include "hipSYCL/compiler/llvm-to-backend/metal/PointerTranslationAnnotationPass.hpp"
#include "hipSYCL/compiler/llvm-to-backend/metal/PointerTranslationPass.hpp"
#include "hipSYCL/compiler/llvm-to-backend/AddressSpaceInferencePass.hpp"
#include "hipSYCL/compiler/llvm-to-backend/AddressSpaceMap.hpp"
#include "hipSYCL/compiler/llvm-to-backend/LLVMToBackend.hpp"
#include "hipSYCL/compiler/llvm-to-backend/Utils.hpp"
#include "hipSYCL/compiler/sscp/IRConstantReplacer.hpp"
#include "hipSYCL/compiler/utils/LLVMUtils.hpp"
#include "hipSYCL/glue/llvm-sscp/jit-reflection/queries.hpp"
#include "hipSYCL/common/filesystem.hpp"
#include "hipSYCL/common/debug.hpp"
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/IR/Attributes.h>
#include <llvm/IR/CallingConv.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Metadata.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/DebugInfo.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/Program.h>


#include <llvm/Transforms/InstCombine/InstCombine.h>
#include <llvm/Transforms/Scalar/EarlyCSE.h>
#include <llvm/Transforms/Scalar/LoopRotation.h>
#include <llvm/Transforms/Scalar/LoopSimplifyCFG.h>
#include <llvm/Transforms/Scalar/SROA.h>
#include <llvm/Transforms/Scalar/SimplifyCFG.h>
#include <llvm/Transforms/Scalar/StructurizeCFG.h>

#include <llvm/Transforms/Utils/LowerMemIntrinsics.h>
#include <llvm/Transforms/Utils/LowerSwitch.h>
#include "llvm/Transforms/Utils/Mem2Reg.h"
#include "llvm/Transforms/Scalar/SimplifyCFG.h"
#include "llvm/Transforms/Scalar/DCE.h"
#include "llvm/Transforms/Scalar/ADCE.h"
#include <llvm/Transforms/IPO/AlwaysInliner.h>
#include <llvm/Transforms/Utils/ModuleUtils.h>

#include <memory>
#include <cassert>
#include <string>
#include <system_error>
#include <vector>

#include <unistd.h>

#include "Emitter.hpp"

namespace hipsycl {
namespace compiler {

namespace {

// these are remapped for f32 and f64
static constexpr std::array remapped_llvm_math_builtins = {
  "sin", "cos", "tan", "sqrt",
  "asin", "acos", "atan", "atan2",
  "sinh", "cosh", "tanh",
  "log", "log2", "log10",
  "exp", "exp2", "exp10",
  "ldexp",
  "fabs", "floor", "ceil",
  "copysign"
};

using builtin_mapping = std::tuple<const char*, const char*, int>; // llvm name, acpp name, arg count (for count intrinsics)

// LLVM math intrinsics where the ACPP name differs from the LLVM name
// Format: {llvm_name, acpp_name}
// llvm.<llvm_name>.f32 -> __acpp_sscp_<acpp_name>_f32
static constexpr std::array remapped_llvm_math_builtins_renamed = {
  builtin_mapping{"maxnum", "fmax", -1},
  builtin_mapping{"minnum", "fmin", -1},
  builtin_mapping{"fmuladd", "fma", -1},
  builtin_mapping{"pow", "powr", -1},
};

static constexpr std::array remapped_llvm_count_builtins = {
  builtin_mapping{"ctlz", "clz", 1},
  builtin_mapping{"cttz", "ctz", 1},
  builtin_mapping{"ctpop", "popcount", 1}
};

struct ReplaceIntrinsics : llvm::PassInfoMixin<ReplaceIntrinsics> {
  std::unordered_map<std::string, std::pair<std::string, int>> Replacement;

  ReplaceIntrinsics() {
    std::string llvm_prefix = "llvm.";
    std::string acpp_prefix = "__acpp_sscp_";
    for (const auto& Name : remapped_llvm_math_builtins) {
      Replacement[llvm_prefix + Name + ".f32"] = {acpp_prefix + Name + "_f32", -1};
      Replacement[llvm_prefix + Name + ".f64"] = {acpp_prefix + Name + "_f64", -1};
    }
    for (const auto& [Name, Mapping, ArgCount] : remapped_llvm_math_builtins_renamed) {
      Replacement[llvm_prefix + Name + ".f32"] = {acpp_prefix + Mapping + "_f32", ArgCount};
      Replacement[llvm_prefix + Name + ".f64"] = {acpp_prefix + Mapping + "_f64", ArgCount};
    }
    Replacement["llvm.powi.f32.i32"] = {acpp_prefix + "pown_f32", -1};
    Replacement["llvm.powi.f32.i64"] = {acpp_prefix + "pown_f32", -1};
    Replacement["__assert_rtn"] = {"__acpp_sscp_assert_fail", -1};
    for (const auto& [Name, Mapping, ArgCount] : remapped_llvm_count_builtins) {
      Replacement[llvm_prefix + Name + ".i8"] = {acpp_prefix + Mapping + "_u8", ArgCount};
      Replacement[llvm_prefix + Name + ".i16"] = {acpp_prefix + Mapping + "_u16", ArgCount};
      Replacement[llvm_prefix + Name + ".i32"] = {acpp_prefix + Mapping + "_u32", ArgCount};
      Replacement[llvm_prefix + Name + ".i64"] = {acpp_prefix + Mapping + "_u64", ArgCount};
    }
  }

  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &MAM) {
    for(const auto& [Name, Value] : Replacement) {
      const auto& [ReplacementName, ArgCount] = Value;
      if(llvm::Function* F = M.getFunction(Name)) {
        llvm::Function* Replacement = M.getFunction(ReplacementName);
        if(!Replacement) {
          if (ArgCount == -1) {
            Replacement = llvm::Function::Create(F->getFunctionType(), F->getLinkage(), ReplacementName, M);
          } else {
            llvm::Type* RetTy = F->getReturnType();
            llvm::SmallVector<llvm::Type*, 8> ArgTys;
            for (unsigned i = 0; i < ArgCount; ++i) {
              ArgTys.push_back(F->getArg(i)->getType());
            }
            llvm::FunctionType* FT = llvm::FunctionType::get(RetTy, ArgTys, false);
            Replacement = llvm::Function::Create(FT, F->getLinkage(), ReplacementName, M);
          }
          Replacement->setLinkage(llvm::GlobalValue::ExternalLinkage);
        }

        HIPSYCL_DEBUG_INFO << "Metal: ReplaceIntrinsics: Remapping calls from " << Name << " to "
                           << ReplacementName << "\n";
        if (F->getFunctionType() == Replacement->getFunctionType()) {
          F->replaceAllUsesWith(Replacement);
        } else {
          // Signatures differ (e.g. llvm.ctlz has an extra i1 is_zero_undef arg)
          llvm::SmallVector<llvm::CallInst*, 16> Calls;
          for (auto* U : F->users()) {
            if (auto* CI = llvm::dyn_cast<llvm::CallInst>(U)) {
              Calls.push_back(CI);
            }
          }
          for (auto* CI : Calls) {
            llvm::SmallVector<llvm::Value*, 4> Args;
            for (unsigned i = 0; i < (unsigned)ArgCount; ++i) {
              Args.push_back(CI->getArgOperand(i));
            }
            llvm::CallInst* NewCI = llvm::CallInst::Create(Replacement->getFunctionType(), Replacement, Args, "", CI->getIterator());
            NewCI->takeName(CI);
            CI->replaceAllUsesWith(NewCI);
            CI->eraseFromParent();
          }
        }
      }
    }

    return llvm::PreservedAnalyses::none();
  }
};

struct ExpandIntrinsics : llvm::PassInfoMixin<ExpandIntrinsics> {
  llvm::PreservedAnalyses run(llvm::Function& F, llvm::FunctionAnalysisManager& FAM) {
    llvm::SmallVector<llvm::IntrinsicInst*, 16> Work;

    // expand can change CFG, so we need to collect intrinsics first and then expand them in a separate loop
    for (auto& BB : F) {
      for (auto& I : BB) {
        auto* II = llvm::dyn_cast<llvm::IntrinsicInst>(&I);
        if (!II) {
          continue;
        }
        Work.push_back(II);
      }
    }

    const llvm::TargetTransformInfo& TTI = FAM.getResult<llvm::TargetIRAnalysis>(F);
    llvm::ScalarEvolution* SE = nullptr;
    if (FAM.getCachedResult<llvm::ScalarEvolutionAnalysis>(F)) {
      SE = &FAM.getResult<llvm::ScalarEvolutionAnalysis>(F);
    }

    bool Changed = false;

    for (auto* II : Work) {
      auto ID = II->getIntrinsicID();
      if (auto* MC = llvm::dyn_cast<llvm::MemCpyInst>(II)) {
        llvm::expandMemCpyAsLoop(MC, TTI, SE);
        II->eraseFromParent();
        Changed = true;
      } else if (auto* MM = llvm::dyn_cast<llvm::MemMoveInst>(II)) {
        bool lowered = llvm::expandMemMoveAsLoop(MM, TTI);
        if (lowered) {
          II->eraseFromParent();
          Changed = true;
        }
      } else if (auto* MS = llvm::dyn_cast<llvm::MemSetInst>(II)) {
        llvm::expandMemSetAsLoop(MS);
        II->eraseFromParent();
        Changed = true;
      } else if (auto* MSP = llvm::dyn_cast<llvm::MemSetPatternInst>(II)) {
        llvm::expandMemSetPatternAsLoop(MSP);
        II->eraseFromParent();
        Changed = true;
#if LLVM_VERSION_MAJOR >= 21
      } else if (auto* AMC = llvm::dyn_cast<llvm::AnyMemCpyInst>(II)) {
#else
      } else if (auto* AMC = llvm::dyn_cast<llvm::AtomicMemCpyInst>(II)) {
#endif
        llvm::expandAtomicMemCpyAsLoop(AMC, TTI, SE);
        II->eraseFromParent();
        Changed = true;
      } else if (ID == llvm::Intrinsic::uadd_with_overflow || ID == llvm::Intrinsic::usub_with_overflow) {
        expandUaddUsubWithOverflow(II);
        II->eraseFromParent();
        Changed = true;
      } else if (ID == llvm::Intrinsic::usub_sat || ID == llvm::Intrinsic::uadd_sat) {
        expandUaddUsubSat(II);
        II->eraseFromParent();
        Changed = true;
      } else if (ID == llvm::Intrinsic::fshl || ID == llvm::Intrinsic::fshr) {
        expandFunnelShift(II);
        II->eraseFromParent();
        Changed = true;
      } else if (ID == llvm::Intrinsic::smax || ID == llvm::Intrinsic::smin) {
        expandSminSmax(II);
        II->eraseFromParent();
        Changed = true;
      } else if (ID == llvm::Intrinsic::umax || ID == llvm::Intrinsic::umin) {
        expandUminUmax(II);
        II->eraseFromParent();
        Changed = true;
      } else if (ID == llvm::Intrinsic::abs) {
        expandAbs(II);
        II->eraseFromParent();
        Changed = true;
      } else if (ID == llvm::Intrinsic::scmp || ID == llvm::Intrinsic::ucmp) {
        expandCmpIntrinsic(II);
        II->eraseFromParent();
        Changed = true;
      } else if (ID == llvm::Intrinsic::lifetime_start ||
         ID == llvm::Intrinsic::lifetime_end ||
         ID == llvm::Intrinsic::assume ||
         ID == llvm::Intrinsic::invariant_start ||
         ID == llvm::Intrinsic::invariant_end ||
         ID == llvm::Intrinsic::experimental_noalias_scope_decl)
      {
        II->eraseFromParent();
        Changed = true;
      }
    }

    return Changed ? llvm::PreservedAnalyses::none()
                   : llvm::PreservedAnalyses::all();
  }

  void expandAbs(llvm::IntrinsicInst* II) {
    llvm::IRBuilder<> B(II);

    llvm::Value* X = II->getArgOperand(0);
    llvm::Value* IsPoison = II->getArgOperand(1);

    auto* Ty = llvm::cast<llvm::IntegerType>(X->getType());

    llvm::Value* Zero = llvm::ConstantInt::get(Ty, 0);
    llvm::Value* IsNeg = B.CreateICmpSLT(X, Zero, "isneg");
    llvm::Value* Neg = B.CreateNeg(X, "neg");
    llvm::Value* Abs = B.CreateSelect(IsNeg, Neg, X, "abs");

    if (auto* CI = llvm::dyn_cast<llvm::ConstantInt>(IsPoison)) {
      if (CI->isOne()) {
        // poison on INT_MIN
        II->replaceAllUsesWith(Abs);
        return;
      }
    }

    llvm::Value* IntMin = llvm::ConstantInt::getSigned(
        Ty, -(1LL << (Ty->getBitWidth() - 1)));
    llvm::Value* IsMin = B.CreateICmpEQ(X, IntMin, "ismin");
    llvm::Value* Res = B.CreateSelect(IsMin, IntMin, Abs, "abs_safe");

    II->replaceAllUsesWith(Res);
  }

  void expandCmpIntrinsic(llvm::IntrinsicInst* II) {
    auto* CI = llvm::cast<llvm::CmpIntrinsic>(II);
    llvm::IRBuilder<> B(II);

    llvm::Value* A  = CI->getLHS();
    llvm::Value* Bv = CI->getRHS();
    auto* RetTy = II->getType();

    llvm::Value* GT = B.CreateICmp(CI->getGTPredicate(), A, Bv, "gt");
    llvm::Value* LT = B.CreateICmp(CI->getLTPredicate(), A, Bv, "lt");

    auto* One    = llvm::ConstantInt::get(RetTy, 1);
    auto* Zero   = llvm::ConstantInt::get(RetTy, 0);
    auto* MinOne = llvm::ConstantInt::getSigned(RetTy, -1);

    llvm::Value* Res = B.CreateSelect(GT, One, B.CreateSelect(LT, MinOne, Zero));
    II->replaceAllUsesWith(Res);
  }

  void expandSminSmax(llvm::IntrinsicInst* II) {
    llvm::IRBuilder<> B(II);

    llvm::Value* A  = II->getArgOperand(0);
    llvm::Value* Bv = II->getArgOperand(1);

    auto* Ty = A->getType();
    llvm::Value* Cmp = B.CreateICmpSGT(A, Bv, "scmp");

    llvm::Value* Res;
    if (II->getIntrinsicID() == llvm::Intrinsic::smax) {
      Res = B.CreateSelect(Cmp, A, Bv, "smax");
    } else {
      Res = B.CreateSelect(Cmp, Bv, A, "smin");
    }

    II->replaceAllUsesWith(Res);
  }

  void expandUminUmax(llvm::IntrinsicInst* II) {
    llvm::IRBuilder<> B(II);

    llvm::Value* A  = II->getArgOperand(0);
    llvm::Value* Bv = II->getArgOperand(1);

    llvm::Value* Cmp = B.CreateICmpUGT(A, Bv, "ucmp");

    llvm::Value* Res;
    if (II->getIntrinsicID() == llvm::Intrinsic::umax) {
      Res = B.CreateSelect(Cmp, A, Bv, "umax");
    } else {
      Res = B.CreateSelect(Cmp, Bv, A, "umin");
    }

    II->replaceAllUsesWith(Res);
  }

  void expandFunnelShift(llvm::IntrinsicInst* II) {
    llvm::IRBuilder<> B(II);

    auto* A = II->getArgOperand(0);
    auto* Bv = II->getArgOperand(1);
    auto* S = II->getArgOperand(2);

    auto* Ty = llvm::cast<llvm::IntegerType>(A->getType());
    unsigned W = Ty->getBitWidth();

    auto* WConst = llvm::ConstantInt::get(Ty, W);
    auto* Zero   = llvm::ConstantInt::get(Ty, 0);

    llvm::Value* Shift = S;
    if (Shift->getType() != Ty) {
      Shift = B.CreateZExtOrTrunc(Shift, Ty);
    }

    llvm::Value* Sh = B.CreateURem(Shift, WConst, "sh");

    llvm::Value* IsZero = B.CreateICmpEQ(Sh, Zero, "sh_is_zero");

    llvm::Value* WmSh = B.CreateSub(WConst, Sh, "w_minus_sh");

    llvm::Value* ResShifted = nullptr;
    llvm::Value* ZeroRes = nullptr;
    if (II->getIntrinsicID() == llvm::Intrinsic::fshl) {
      // fshl(A, B, S) = msb_extract({A:B} << S) = (A << S) | (B >> (W-S))
      llvm::Value* L = B.CreateShl(A, Sh, "l");
      llvm::Value* R = B.CreateLShr(Bv, WmSh, "r");
      ResShifted = B.CreateOr(L, R, "fshl");
      ZeroRes = A; // fshl(A, B, 0) = A
    } else { // fshr
      // fshr(A, B, S) = lsb_extract({A:B} >> S) = (B >> S) | (A << (W-S))
      llvm::Value* L = B.CreateLShr(Bv, Sh, "l");
      llvm::Value* R = B.CreateShl(A, WmSh, "r");
      ResShifted = B.CreateOr(L, R, "fshr");
      ZeroRes = Bv; // fshr(A, B, 0) = B
    }

    llvm::Value* Res = B.CreateSelect(IsZero, ZeroRes, ResShifted, "fsh");
    II->replaceAllUsesWith(Res);
  }

  void expandUaddUsubSat(llvm::IntrinsicInst* II) {
    llvm::IRBuilder<> B(II);

    auto* A = II->getArgOperand(0);
    auto* Bv = II->getArgOperand(1);
    auto* Ty = A->getType();
    llvm::Value* Res;
    if (II->getIntrinsicID() == llvm::Intrinsic::usub_sat) {
      llvm::Value* Diff = B.CreateSub(A, Bv, "diff");
      llvm::Value* Under = B.CreateICmpULT(A, Bv, "under");
      llvm::Value* Zero = llvm::ConstantInt::get(Ty, 0);
      Res = B.CreateSelect(Under, Zero, Diff, "usub.sat");
    } else {
      llvm::Value* Sum = B.CreateAdd(A, Bv, "sum");
      llvm::Value* Over = B.CreateICmpULT(Sum, A, "over");
      llvm::Value* Max = llvm::ConstantInt::getAllOnesValue(Ty);
      Res = B.CreateSelect(Over, Max, Sum, "uadd.sat");
    }
    II->replaceAllUsesWith(Res);
  }

  void expandUaddUsubWithOverflow(llvm::IntrinsicInst* II) {
    llvm::IRBuilder<> B(II);

    llvm::Value* A = II->getArgOperand(0);
    llvm::Value* Bv = II->getArgOperand(1);

    llvm::Value* Result;
    llvm::Value* Overflow;

    if (II->getIntrinsicID() == llvm::Intrinsic::uadd_with_overflow) {
      Result = B.CreateAdd(A, Bv, "sum");
      Overflow = B.CreateICmpULT(Result, A, "overflow");
    } else { // usub
      Result = B.CreateSub(A, Bv, "diff");
      Overflow = B.CreateICmpULT(A, Bv, "overflow");
    }

    llvm::Type* RetTy = II->getType();
    llvm::Value* Agg = llvm::UndefValue::get(RetTy);
    Agg = B.CreateInsertValue(Agg, Result, 0);
    Agg = B.CreateInsertValue(Agg, Overflow, 1);

    II->replaceAllUsesWith(Agg);
  }
};

} // namespace


LLVMToMetalTranslator::LLVMToMetalTranslator(const std::vector<std::string>& KernelNames)
  : LLVMToBackendTranslator{static_cast<int>(sycl::AdaptiveCpp_jit::compiler_backend::metal), KernelNames, KernelNames}
  , KernelNames(KernelNames)
  , ActualKernelNames(KernelNames.begin(), KernelNames.end())
{ }

LLVMToMetalTranslator::~LLVMToMetalTranslator() = default;

AddressSpaceMap LLVMToMetalTranslator::getAddressSpaceMap() const
{
  AddressSpaceMap ASMap;

  ASMap[AddressSpace::Generic] = 0;
  ASMap[AddressSpace::Global] = 1;
  ASMap[AddressSpace::Local] = 3;
  ASMap[AddressSpace::Private] = 5;
  ASMap[AddressSpace::Constant] = 4;
  ASMap[AddressSpace::AllocaDefault] = 5;
  ASMap[AddressSpace::GlobalVariableDefault] = 1;
  ASMap[AddressSpace::ConstantGlobalVariableDefault] = 4;

  return ASMap;
}

bool LLVMToMetalTranslator::isKernelAfterFlavoring(llvm::Function& F) {
  return ActualKernelNames.count(F.getName().str()) > 0;
}

bool LLVMToMetalTranslator::prepareBackendFlavor(llvm::Module& M) {
  return true;
}

bool LLVMToMetalTranslator::toBackendFlavor(llvm::Module &M, PassHandler& PH) {

  AddressSpaceMap ASMap = getAddressSpaceMap();

  KernelFunctionParameterRewriter ParamRewriter{
      KernelFunctionParameterRewriter::ByValueArgAttribute::ByVal,
      ASMap[AddressSpace::Generic],
      ASMap[AddressSpace::Global]};

  ParamRewriter.run(M, KernelNames, *PH.ModuleAnalysisManager);

  // First linking: provides __acpp_sscp_* definitions so that the base class inliner
  // (which runs after toBackendFlavor) can inline them. The inlined bodies then go through
  // the base class O3 optimization pipeline, which may re-introduce LLVM intrinsics such as
  // llvm.minnum / llvm.maxnum / llvm.fmuladd via InstCombine. Those are handled in
  // translateToBackendFormat with a second ReplaceIntrinsics + link pass.
  // Pre-link soft-fp64 anchor. The MetalEmitter translates IR-level fp64
  // instructions (fadd/fsub/fmul/fdiv/frem/fneg/fcmp, fp-int conversions)
  // to source-level call strings like `__acpp_sscp_soft_f64_add(...)`.
  // Those emissions bypass the LLVM callgraph: the IR has no `call`
  // instruction referencing those symbols. Consequences without
  // intervention:
  //   * `LinkOnlyNeeded=true` (the default) doesn't pull the soft-fp64
  //     bodies into M — nothing "needs" them from M's perspective.
  //   * Even if we force-link, GlobalInliningAttributorPass marks every
  //     non-kernel function as `alwaysinline`+InternalLinkage; the
  //     inliner finds no call sites; and O3 DCE removes the bodies.
  // Either way, the Emitter emits calls to symbols with no definitions
  // and Metal compilation fails with "undeclared identifier".
  //
  // Fix (two-step):
  //   (a) BEFORE the link, declare every `__acpp_sscp_soft_f64_*`
  //       primitive the Emitter may reference and append each to
  //       `@llvm.compiler.used`. That anchors the symbols as "needed"
  //       so `LinkOnlyNeeded=true` pulls in their bodies.
  //   (b) AFTER the link, mark every `__acpp_sscp_soft_f64_*` and
  //       `__acpp_sscp_*_f64` function `noinline` + strip
  //       `alwaysinline`, re-anchor in `@llvm.compiler.used`, and keep
  //       external linkage. This survives GlobalInliningAttributorPass
  //       (which now respects `noinline`) and O3 DCE. The MetalEmitter's
  //       topological-sort loop then emits each body as a Metal helper
  //       function ahead of the kernel.
  //
  // Pairs with CMake change: `-fno-vectorize -fno-slp-vectorize
  // -fno-unroll-loops` on the libkernel bitcode compile when the
  // external soft-fp64 is active, so the resulting bodies don't contain
  // `<N x double>` or `<4 x i32>` ops the Emitter can't translate.
  {
    static const struct { const char* name; const char* sig; } kSoftF64Primitives[] = {
      {"__acpp_sscp_soft_f64_add", "ddd"},
      {"__acpp_sscp_soft_f64_sub", "ddd"},
      {"__acpp_sscp_soft_f64_mul", "ddd"},
      {"__acpp_sscp_soft_f64_div", "ddd"},
      {"__acpp_sscp_soft_f64_rem", "ddd"},
      {"__acpp_sscp_soft_f64_neg", "dd"},
      {"__acpp_sscp_soft_f64_fcmp", "bddi"},
      {"__acpp_sscp_soft_f64_fmin_precise", "ddd"},
      {"__acpp_sscp_soft_f64_fmax_precise", "ddd"},
      {"__acpp_sscp_soft_f64_from_f32", "df"},
      {"__acpp_sscp_soft_f64_to_f32",   "fd"},
      {"__acpp_sscp_soft_f64_from_i32", "di"},
      {"__acpp_sscp_soft_f64_from_i64", "dI"},
      {"__acpp_sscp_soft_f64_from_u32", "du"},
      {"__acpp_sscp_soft_f64_from_u64", "dU"},
      {"__acpp_sscp_soft_f64_to_i32",   "id"},
      {"__acpp_sscp_soft_f64_to_i64",   "Id"},
      {"__acpp_sscp_soft_f64_to_u32",   "ud"},
      {"__acpp_sscp_soft_f64_to_u64",   "Ud"},
    };
    auto typeFromCode = [&](char c) -> llvm::Type* {
      auto& Ctx = M.getContext();
      switch (c) {
        case 'd': return llvm::Type::getDoubleTy(Ctx);
        case 'f': return llvm::Type::getFloatTy(Ctx);
        case 'i': case 'u': return llvm::Type::getInt32Ty(Ctx);
        case 'I': case 'U': return llvm::Type::getInt64Ty(Ctx);
        case 'b': return llvm::Type::getInt1Ty(Ctx);
        default: return nullptr;
      }
    };
    llvm::SmallVector<llvm::GlobalValue*, 64> PreLinkUsed;
    for (const auto& prim : kSoftF64Primitives) {
      llvm::Type* retTy = typeFromCode(prim.sig[0]);
      if (!retTy) continue;
      llvm::SmallVector<llvm::Type*, 4> argTys;
      for (const char* p = prim.sig + 1; *p; ++p)
        if (auto* t = typeFromCode(*p)) argTys.push_back(t);
      auto* FT = llvm::FunctionType::get(retTy, argTys, false);
      auto Callee = M.getOrInsertFunction(prim.name, FT);
      if (auto* F = llvm::dyn_cast<llvm::Function>(Callee.getCallee()))
        PreLinkUsed.push_back(F);
    }

    // Also anchor every `__acpp_sscp_<name>_f64` math forwarder that
    // `ReplaceIntrinsics` can produce. Without this, an `llvm.fabs.f64`
    // (or `llvm.sqrt.f64`, `llvm.copysign.f64`, etc.) introduced by
    // InstCombine pattern-matching AFTER the first link pass gets remapped
    // to `__acpp_sscp_fabs_f64` — a symbol the linker never imported
    // because the call site didn't exist at link time. The result is a
    // use-of-undeclared-identifier MSL compile failure downstream.
    //
    // Signatures derive from the `llvm.<name>.f64` intrinsic each forwarder
    // replaces: all entries in `remapped_llvm_math_builtins` except
    // `atan2`, `ldexp`, and `copysign` are unary (double → double); those
    // three plus `remapped_llvm_math_builtins_renamed` (minnum/maxnum/pow
    // → fmin/fmax/powr) are binary with the noted special cases.
    static const struct {
      const char* llvm_name;
      bool is_binary;
      bool second_is_int;
    } kMathForwarders[] = {
        // unary f64 forwarders (double → double)
        {"sin",    false, false}, {"cos",    false, false},
        {"tan",    false, false}, {"sqrt",   false, false},
        {"asin",   false, false}, {"acos",   false, false},
        {"atan",   false, false},
        {"sinh",   false, false}, {"cosh",   false, false},
        {"tanh",   false, false},
        {"log",    false, false}, {"log2",   false, false},
        {"log10",  false, false},
        {"exp",    false, false}, {"exp2",   false, false},
        {"exp10",  false, false},
        {"fabs",   false, false}, {"floor",  false, false},
        {"ceil",   false, false},
        // binary f64 forwarders (double, double) → double
        {"atan2",    true, false},
        {"copysign", true, false},
        {"fmax",     true, false}, // from llvm.maxnum.f64
        {"fmin",     true, false}, // from llvm.minnum.f64
        {"powr",     true, false}, // from llvm.pow.f64
        // binary (double, int) → double
        {"ldexp", true, true},
    };
    llvm::Type* dTy = llvm::Type::getDoubleTy(M.getContext());
    llvm::Type* iTy = llvm::Type::getInt32Ty(M.getContext());
    for (const auto& fwd : kMathForwarders) {
      std::string acppName =
          std::string{"__acpp_sscp_"} + fwd.llvm_name + "_f64";
      llvm::SmallVector<llvm::Type*, 3> argTys;
      argTys.push_back(dTy);
      if (fwd.is_binary) argTys.push_back(fwd.second_is_int ? iTy : dTy);
      auto* FT = llvm::FunctionType::get(dTy, argTys, false);
      auto Callee = M.getOrInsertFunction(acppName, FT);
      if (auto* F = llvm::dyn_cast<llvm::Function>(Callee.getCallee()))
        PreLinkUsed.push_back(F);
    }
    // `llvm.fmuladd.f64` → `__acpp_sscp_fma_f64` (double, double, double).
    {
      auto* FT = llvm::FunctionType::get(dTy, {dTy, dTy, dTy}, false);
      auto Callee = M.getOrInsertFunction("__acpp_sscp_fma_f64", FT);
      if (auto* F = llvm::dyn_cast<llvm::Function>(Callee.getCallee()))
        PreLinkUsed.push_back(F);
    }

    if (!PreLinkUsed.empty())
      llvm::appendToCompilerUsed(M, PreLinkUsed);
  }

  std::string BuiltinBitcodeFile =
      common::filesystem::join_path(getBitcodePath(), "libkernel-sscp-metal-full.bc");
  if (!this->linkBitcodeFile(M, BuiltinBitcodeFile))
    return false;

  // Post-link preservation. The bodies pulled in above must survive
  // `GlobalInliningAttributorPass` (which now skips `noinline` per the
  // same commit) and O3 DCE, so the MetalEmitter sees them.
  llvm::SmallVector<llvm::GlobalValue*, 64> SoftF64Funcs;
  for (llvm::Function& F : M) {
    if (F.isDeclaration()) continue;
    llvm::StringRef Name = F.getName();
    bool isSoftF64Primitive = Name.find("__acpp_sscp_soft_f64_") == 0;
    bool isF64MathForwarder =
        Name.find("__acpp_sscp_") == 0 &&
        (Name.ends_with("_f64") || Name.ends_with("_f64_precise"));
    // `sf64_*` are the core soft-fp64 bodies these forwarders call into.
    // Preserve them too: O3's InstCombine pattern-matches their bit-
    // twiddle implementations back into LLVM intrinsics (e.g. the
    // `(x & ~sign) | (y & sign)` copysign pattern → `llvm.copysign.f64`
    // → `__acpp_sscp_copysign_f64`), creating a self-call loop.
    bool isSoftF64Core = Name.find("sf64_") == 0;
    if (!isSoftF64Primitive && !isF64MathForwarder && !isSoftF64Core) continue;
    if (!F.hasFnAttribute(llvm::Attribute::NoInline))
      F.addFnAttr(llvm::Attribute::NoInline);
    F.removeFnAttr(llvm::Attribute::AlwaysInline);
    // `optnone` stops the O3 pipeline from pattern-matching bit-twiddle
    // soft-fp64 bodies back into LLVM intrinsics. Specifically: without
    // this, InstCombine recognises `sf64_copysign`'s `(x & ~sign) |
    // (y & sign)` pattern as `llvm.copysign.f64`, ReplaceIntrinsics
    // then remaps that to `__acpp_sscp_copysign_f64`, and the body ends
    // up calling its own forwarder. The soft-fp64 bodies are already
    // hand-optimised for correctness; further optimisation is unwanted.
    if (!F.hasFnAttribute(llvm::Attribute::OptimizeNone))
      F.addFnAttr(llvm::Attribute::OptimizeNone);
    if (F.getLinkage() == llvm::GlobalValue::InternalLinkage)
      F.setLinkage(llvm::GlobalValue::ExternalLinkage);
    SoftF64Funcs.push_back(&F);
  }
  if (!SoftF64Funcs.empty()) {
    llvm::appendToCompilerUsed(M, SoftF64Funcs);
    HIPSYCL_DEBUG_INFO
        << "LLVMToMetal: preserved " << SoftF64Funcs.size()
        << " soft-fp64 function bodies for MetalEmitter source emission\n";
  }

  AddressSpaceInferencePass ASIPass{ASMap};
  ASIPass.run(M, *PH.ModuleAnalysisManager);

  llvm::StripDebugInfo(M);

  return true;
}

bool LLVMToMetalTranslator::optimizeFlavoredIR(llvm::Module& M, PassHandler& PH) {
  // Metal-specific override: run the O3 pipeline with loop + SLP
  // vectorizers disabled, and interleaving turned off. The default
  // pipeline introduces `<N x T>` vector ops (especially `<4 x i32>`
  // shifts with non-{32,64,96} amounts, `<2 x double>` SLP bundles, and
  // `insertelement`/`shufflevector`) that the MetalEmitter cannot lower
  // to MSL. Keeping the soft-fp64 bodies scalar through the whole
  // optimization pipeline is what makes the external-fp64 integration
  // actually work end-to-end.
  HIPSYCL_DEBUG_INFO << "LLVMToMetal::optimizeFlavoredIR: O3 with vectorization disabled\n";
  llvm::PipelineTuningOptions PTO;
  PTO.LoopVectorization = false;
  PTO.SLPVectorization = false;
  PTO.LoopInterleaving = false;

  llvm::LoopAnalysisManager LAM;
  llvm::FunctionAnalysisManager FAM;
  llvm::CGSCCAnalysisManager CGAM;
  llvm::ModuleAnalysisManager MAM;
  llvm::PassBuilder PB(nullptr, PTO);
  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  llvm::ModulePassManager MPM =
      PB.buildPerModuleDefaultPipeline(llvm::OptimizationLevel::O3);
  MPM.run(M, MAM);
  return true;
}

bool LLVMToMetalTranslator::translateToBackendFormat(llvm::Module& FlavoredModule, std::string& out) {
  AddressSpaceMap ASMap = getAddressSpaceMap();

  auto ok = withPassBuilder([&](auto& PB, auto& LAM, auto& FAM, auto& CGAM, auto& MAM) {
    // Second ReplaceIntrinsics + link pass: the base class O3 pipeline (InstCombine etc.) may
    // have re-introduced LLVM intrinsics (llvm.minnum, llvm.maxnum, llvm.fmuladd) from the
    // inlined builtin bodies. We remap them to __acpp_sscp_* builtins here, then re-link the
    // Metal bitcode to supply their definitions. The subsequent inliner pass (AlwaysInlinerPass
    // inside withPassBuilder) inlines those definitions so MetalEmitter can see the
    // __acpp_sscp_metal_math_* calls it needs to emit native Metal code.
    // Any LLVM intrinsics that remain after linking have no __acpp_sscp_* counterpart and are
    // lowered to plain IR by ExpandIntrinsics below.
    ReplaceIntrinsics{}.run(FlavoredModule, MAM);

    std::string BuiltinBitcodeFile =
      common::filesystem::join_path(getBitcodePath(), "libkernel-sscp-metal-full.bc");

    if (!linkBitcodeFile(FlavoredModule, BuiltinBitcodeFile))
      return false;

    // Re-run AS inference so GlobalVariables newly imported by the link
    // above land in their canonical address spaces. Soft-fp64's SLEEF
    // polynomial coefficient tables (`@__const.*kLogkCoef`, etc.) are
    // `private unnamed_addr constant` in the source, which the frontend
    // emits in AS 0. Without a second inference pass they stay in AS 0
    // and MetalEmitter's emitGlobalConstants (which emits AS 4 constants
    // with MSL's `constexpr constant` keyword) skips them, producing
    // use-of-undeclared-identifier errors downstream.
    AddressSpaceInferencePass{getAddressSpaceMap()}.run(FlavoredModule, MAM);

    llvm::AlwaysInlinerPass{}.run(FlavoredModule, MAM);

    // Third ReplaceIntrinsics pass: the AlwaysInliner above inlines
    // soft-fp64 bodies from the just-linked libkernel. Those bodies use
    // LLVM intrinsics like `llvm.copysign.f64`, `llvm.fabs.f64`, etc.
    // that would otherwise survive to MetalEmitter and be written as
    // literal `llvm.copysign.f64(...)` calls, which Metal compilation
    // rejects ("undeclared identifier 'llvm'"). Re-run the remap so the
    // inlined intrinsic call sites point at `__acpp_sscp_*_f64` names,
    // and the bodies of those `__acpp_sscp_*_f64` forwarders (which the
    // soft-fp64 preservation pass keeps alive) get emitted.
    ReplaceIntrinsics{}.run(FlavoredModule, MAM);

    llvm::FunctionPassManager FPM;
    FPM.addPass(llvm::PromotePass());
    FPM.addPass(ExpandIntrinsics());
    FPM.addPass(llvm::LowerSwitchPass());
    FPM.addPass(llvm::LoopSimplifyPass());
    FPM.addPass(llvm::LCSSAPass());
    FPM.addPass(llvm::DCEPass());
    FPM.addPass(llvm::ADCEPass());
    FPM.addPass(llvm::StructurizeCFGPass());
    FPM.addPass(llvm::SimplifyCFGPass());
    AddressSpaceInferencePass ASIPass{ASMap};
    llvm::ModulePassManager MPM;
    MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));
    MPM.addPass(std::move(ASIPass));
    MPM.addPass(PointerTranslationAnnotationPass(ASMap[AddressSpace::Global]));
    MPM.addPass(PointerTranslationPass(ASMap[AddressSpace::Global]));
    MPM.run(FlavoredModule, MAM);
    return true;
  });

  if (!ok) {
    registerError("LLVMToMetal: Failed to prepare module for Metal translation");
    return false;
  }

  std::unordered_set<std::string> kernelNames(KernelNames.begin(), KernelNames.end());

  if (getenv("__ACPP_PRINT_IR_BEFORE_EMIT")) {
    FlavoredModule.print(llvm::errs(), nullptr);
  }
#ifdef ACPP_PRINT_IR_BEFORE_EMIT
  FlavoredModule.print(llvm::errs(), nullptr);
#endif
  if (const char* dump = std::getenv("ACPP_METAL_DUMP_IR")) {
    std::error_code ec;
    llvm::raw_fd_ostream f(dump, ec);
    if (!ec) FlavoredModule.print(f, nullptr);
  }

  MetalEmitterOptions emitterOpts;
  if (MaxArgsForFlatMode.has_value()) {
    emitterOpts.maxArgsForFlatMode = MaxArgsForFlatMode.value();
  }
  MetalEmitter emitter(FlavoredModule, kernelNames, emitterOpts);
  bool success = emitter.emit(out);
  if (!success) {
    registerError("LLVMToMetal: MetalEmitter failed: " +
                  emitter.errorMessage().value_or("unknown error"));
    return false;
  }

  if (getenv("__ACPP_PRINT_METAL_CODE")) {
    llvm::errs() << "Generated Metal code:\n" << out << "\n";
  }
  return true;
}

bool LLVMToMetalTranslator::applyBuildOption(const std::string &Option, const std::string &Value) {
  if (Option == "metal-max-args-for-flat-mode") {
    MaxArgsForFlatMode = std::stoi(Value);
    return true;
  }
  return false;
}

void LLVMToMetalTranslator::migrateKernelProperties(llvm::Function* From, llvm::Function* To) {
  ActualKernelNames.erase(From->getName().str());
  ActualKernelNames.insert(To->getName().str());
}


std::unique_ptr<LLVMToBackendTranslator>
createLLVMToMetalTranslator(const std::vector<std::string> &KernelNames) {
  return std::make_unique<LLVMToMetalTranslator>(KernelNames);
}

} // namespace compiler
} // namespace hipsycl
