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
#include <llvm/IR/Dominators.h>
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/Analysis/PostDominators.h>

#include <llvm/IR/CFG.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>

#include <llvm/IRReader/IRReader.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/raw_ostream.h>

#include <llvm/Passes/PassBuilder.h>
#include <llvm/Passes/PassPlugin.h>

#include <llvm/Transforms/InstCombine/InstCombine.h>
#include <llvm/Transforms/Scalar/EarlyCSE.h>
#include <llvm/Transforms/Scalar/LoopRotation.h>
#include <llvm/Transforms/Scalar/LoopSimplifyCFG.h>
#include <llvm/Transforms/Scalar/SROA.h>
#include <llvm/Transforms/Scalar/SimplifyCFG.h>
#include <llvm/Transforms/Scalar/StructurizeCFG.h>
#include <llvm/Transforms/Scalar/Reg2Mem.h>
#include <llvm/Transforms/Utils/LowerSwitch.h>
#include "llvm/Transforms/Utils/Mem2Reg.h"
#include "llvm/Transforms/Scalar/DCE.h"
#include "llvm/Transforms/Scalar/ADCE.h"

#include "Emitter.hpp"
#include "HLTree.hpp"
#include "HLExtractionPass.hpp"

using namespace llvm;

namespace hipsycl {
namespace compiler {

using namespace hl;

namespace {

std::string indent(int level) {
  return std::string(level * 2, ' ');
}

std::string instToString(const llvm::Instruction& I) {
  std::string str;
  llvm::raw_string_ostream rso(str);
  I.print(rso);
  return rso.str();
}

std::optional<uint64_t> getConstU64(llvm::Value* V) {
  if (auto* C = llvm::dyn_cast<llvm::ConstantInt>(V)) {
    return C->getZExtValue();
  }
  return std::nullopt;
}

std::optional<std::string> extractStringConstant(llvm::Value* V, std::string& errorStr) {
  llvm::GlobalVariable* GV = nullptr;

  // Handle either direct GlobalVariable or ConstantExpr that refers to one
  if (auto* gv = llvm::dyn_cast<llvm::GlobalVariable>(V)) {
    GV = gv;
  } else if (auto* CE = llvm::dyn_cast<llvm::ConstantExpr>(V)) {
    if (CE->getOpcode() == llvm::Instruction::AddrSpaceCast) {
      return extractStringConstant(CE->getOperand(0), errorStr);
    } else if (CE->getOpcode() == llvm::Instruction::GetElementPtr) {
      GV = llvm::dyn_cast<llvm::GlobalVariable>(CE->getOperand(0));
    }
  }

  if (!GV || !GV->hasInitializer()) {
    errorStr = "Argument must be a string constant";
    return std::nullopt;
  }

  auto* CDA = llvm::dyn_cast<llvm::ConstantDataArray>(GV->getInitializer());
  if (!CDA || !CDA->isString()) {
    errorStr = "Argument must be a string constant";
    return std::nullopt;
  }

  std::string result = CDA->getAsString().str();
  if (!result.empty() && result.back() == '\0') {
    result.pop_back();
  }
  return result;
}

struct EmitContext {
  const llvm::Function* F;
  std::string name;
};

std::optional<EmitContext> initEmitContext(const llvm::CallInst* CI, std::string& errorStr) {
  errorStr.clear();
  const llvm::Function* F = CI ? CI->getCalledFunction() : nullptr;
  if (!F) {
    errorStr = "CallInst has no called function";
    return std::nullopt;
  }
  return EmitContext{F, F->getName().str()};
}

} // namespace

MetalEmitter::MetalEmitter(Module& M, const std::unordered_set<std::string>& kernelNames, const MetalEmitterOptions& opt)
  : M(M), kernelNames(kernelNames), opt(opt)
  , addressSpaceMap{opt.addressSpaceMap}
{
}

// NOTE: this destructor is user-provided out-of-line *by design*.
//
// Prior to this, MetalEmitter had no user-declared destructor, so the compiler
// generated an implicit inline destructor in every translation unit that
// instantiated MetalEmitter. In libllvm-to-metal.dylib on macOS that implicit
// destructor aggressively inlined and reordered the destruction of the
// `std::ostringstream os` member (including its virtual-base `basic_ios`
// subobject, which holds an `ios_base::__loc_` locale). On the fork-child
// side of postmaster-style fork()-without-exec (observed in forked Postgres
// backends and in tests/fork_safety/metal.cpp cold-fork), the generated path
// ended up invoking `std::locale::~locale()` against an `__loc_` whose
// `_Impl*` backing pointer was zero, faulting at offset 0x8 (the refcount
// word inside `locale::_Impl`) on the very first kernel JIT dispatch.
//
// Providing an explicit, out-of-line destructor forces clang to emit the
// ostringstream subobject destruction through the normal virtual-base VTT
// path in a single, non-inlined function body. That eliminates the crash
// without changing behavior -- the destructor still has no other work to do,
// the string output has already been flushed via `out = os.str()` in
// `emit()` before MetalEmitter goes out of scope, so `~ostringstream()`
// now runs against a fully-initialized locale and succeeds.
MetalEmitter::~MetalEmitter() = default;

bool MetalEmitter::emit(std::string& out) {
  PassBuilder PB;
  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;

  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  analyzeCallInsts();
  analyzeAtomicI64Storage();

  os << R"__(
#include <metal_stdlib>
using namespace metal;

// These structs are used for void** dereferencing
struct __struct_ptr_to_device {
  device void* value;
};

struct __struct_ptr_to_threadgroup {
  threadgroup void* value;
};

struct __struct_ptr_to_constant {
  constant void* value;
};

struct __struct_ptr_to_thread {
  thread void* value;
};

template<typename To, typename From>
struct __atomic_pointer_type {};

template<typename To>
struct __atomic_pointer_type<To, device void*> {
  using type = device atomic<To>*;
};

template<typename To>
struct __atomic_pointer_type<To, threadgroup void*> {
  using type = threadgroup atomic<To>*;
};

template<typename To>
struct __atomic_pointer_type<To, thread void*> {
  using type = thread atomic<To>*;
};

template<typename To, typename From>
inline typename __atomic_pointer_type<To, From>::type __atomic_pointer_cast(From ptr) {
  return (typename __atomic_pointer_type<To, From>::type)(ptr);
}

template<typename To, typename From>
struct __pointer_type {};

template<typename To>
struct __pointer_type<To, device void*> {
  using type = device To*;
};

template<typename To>
struct __pointer_type<To, threadgroup void*> {
  using type = threadgroup To*;
};

template<typename To>
struct __pointer_type<To, thread void*> {
  using type = thread To*;
};

template<typename To, typename From>
inline typename __pointer_type<To, From>::type __pointer_cast(From ptr) {
  return (typename __pointer_type<To, From>::type)(ptr);
}

uint __simd_size [[threads_per_simdgroup]];
uint __simd_group_id [[simdgroup_index_in_threadgroup]];
uint __simd_lane_id [[thread_index_in_simdgroup]];

uint3 __acpp_sscp_metal_group_id [[threadgroup_position_in_grid]];
uint3 __acpp_sscp_metal_num_groups [[threadgroups_per_grid]];
uint3 __acpp_sscp_metal_local_id [[thread_position_in_threadgroup]];
uint3 __acpp_sscp_metal_local_size [[threads_per_threadgroup]];

threadgroup void * constant __acpp_sscp_metal_dynamic_local_memory [[threadgroup(0)]];
constant uint& constant __acpp_sscp_metal_dynamic_local_memory_size [[buffer(0)]];
constant long& constant __acpp_sscp_metal_gpu_to_host_addr_diff [[buffer(1)]];

)__";

  // Emit the soft-fp64 helper types (acpp_f64, i48u) BEFORE LLVM struct
  // emission. LLVM aggregate types that contain an fp64 field (e.g.
  // `{ double, double, i64 }`) get emitted with `acpp_f64` as the field
  // type by `emitTypes()`, so the struct must be declared first or Metal
  // compilation rejects the shader with "unknown type name 'acpp_f64'".
  emitEarlyFp64Helpers();
  emitTypes();
  emitIntrinsicHelpers();
  emitGlobalConstants();

  auto callGraph = buildCallGraph();
  auto sortedFunctions = topologicalSort(callGraph);

  // Emit forward declarations for every non-kernel function before any
  // bodies. InstCombine (during the libkernel bitcode build at -O3) can
  // pattern-match soft-fp64 bit-twiddle bodies into calls to LLVM
  // intrinsics, which ReplaceIntrinsics then remaps to `__acpp_sscp_*_f64`
  // — producing mutual references like `sf64_copysign` calling
  // `__acpp_sscp_copysign_f64` calling `sf64_copysign`. Topological sort
  // can't pick an order that satisfies such cycles, so MSL refuses to
  // compile on "use before declaration". Forward decls break the cycle.
  for (Function* F : sortedFunctions) {
    if (kernelNames.count(F->getName().str()) > 0) continue;
    os << mapType(F->getReturnType()) << " " << F->getName().str() << " (";
    bool first = true;
    for (Argument& A : F->args()) {
      if (!first) os << ", ";
      first = false;
      os << mapType(A.getType()) << " " << valueName(&A);
    }
    os << ");\n";
  }
  os << "\n";

  for (Function* F : sortedFunctions) {
    HLExtractionPass hlPass;
    hlPass.run(*F, FAM);
    if (!hlPass.tree) {
      errorMsg = "Failed to extract HL tree for " + F->getName().str();
      return false;
    }

    if (!emitFunction(*F, *hlPass.tree)) {
      return false;
    }
  }

  if (errorMsg.has_value()) {
    return false;
  }

  out = os.str();
  return true;
}

bool MetalEmitter::emitFunction(Function& F, const Node& node) {
  collectVariablesInfo(F);

  bool success = emitArgStruct(F) &&
    emitSignature(F) &&
    emitDeclarations() &&
    emitNode(node, 1);

  if (!success) {
    return false;
  }

  os << "}\n\n";
  return true;
}

std::string MetalEmitter::emitConstantInitializer(const Constant* C) {
  if (isa<ConstantAggregateZero>(C)) {
    return "{}";
  }
  if (auto *CDA = dyn_cast<ConstantDataArray>(C)) {
    std::string result = "{";
    for (unsigned i = 0; i < CDA->getNumElements(); ++i) {
      if (i > 0) result += ", ";
      result += emitExpr(CDA->getElementAsConstant(i));
    }
    result += "}";
    return result;
  }
  if (auto *CA = dyn_cast<ConstantArray>(C)) {
    std::string result = "{";
    for (unsigned i = 0; i < CA->getNumOperands(); ++i) {
      if (i > 0) result += ", ";
      result += emitConstantInitializer(cast<Constant>(CA->getOperand(i)));
    }
    result += "}";
    return result;
  }
  if (auto *CS = dyn_cast<ConstantStruct>(C)) {
    std::string result = "{";
    for (unsigned i = 0; i < CS->getNumOperands(); ++i) {
      if (i > 0) result += ", ";
      result += emitConstantInitializer(cast<Constant>(CS->getOperand(i)));
    }
    result += "}";
    return result;
  }
  // Scalars: reuse emitExpr (handles ConstantInt, ConstantFP, etc.)
  return emitExpr(C);
}

void MetalEmitter::emitGlobalConstants() {
  for (const GlobalVariable& GV : M.globals()) {
    if (!GV.isConstant() || !GV.hasInitializer()) continue;
    if (GV.getAddressSpace() != 4) continue;

    std::string name = valueName(&GV);
    std::string init = emitConstantInitializer(GV.getInitializer());
    Type* valTy = GV.getValueType();

    if (auto* AT = dyn_cast<ArrayType>(valTy)) {
      std::string elemType = mapType(AT->getElementType());
      os << "constexpr constant " << elemType << " " << name
         << "[" << AT->getNumElements() << "] = " << init << ";\n\n";
    } else {
      os << "constexpr constant " << mapType(valTy) << " " << name << " = " << init << ";\n\n";
    }
  }
}

void MetalEmitter::emitTypes() {
  std::vector<const StructType*> structs;
  for (const StructType *ST : M.getIdentifiedStructTypes()) {
    structs.push_back(ST);
  }
  for (auto& [T, _] : anonStructs) {
    if (auto* ST = dyn_cast<StructType>(T)) {
      structs.push_back(ST);
    }
  }

  std::unordered_map<const StructType*, std::unordered_set<const StructType*>> deps;

  std::function<const StructType*(Type*)> structType = [&](Type *T) -> const StructType* {
    if (auto *ST = dyn_cast<StructType>(T)) {
      return ST;
    }
    if (auto *AT = dyn_cast<ArrayType>(T)) {
      return structType(AT->getElementType());
    }
    return nullptr;
  };

  auto addDeps = [&](const StructType *ST) {
    for (unsigned i = 0; i < ST->getNumElements(); ++i) {
      Type *elemTy = ST->getElementType(i);
      if (auto *elemST = structType(elemTy)) {
        deps[ST].insert(elemST);
      }
    }
  };

  for (const StructType *ST : structs) {
    addDeps(ST);
  }

  std::unordered_set<const StructType*> emitted;
  std::function<void(const StructType*)> emitType = [&](const StructType *ST) {
    if (!ST) return;
    if (emitted.count(ST)) return;

    for (const StructType *dep : deps[ST]) {
      emitType(dep);
    }

    emitted.insert(ST);

    os << "struct " << mapType(ST) << " { \n";
    for (unsigned i = 0; i < ST->getNumElements(); ++i) {
      const Type *elemTy = ST->getElementType(i);
      auto typeName = mapType(elemTy);
      os << indent(1) << typeName << " field" << i << ";\n";
    }
    os << "};\n\n";
  };

  for (const StructType *ST : structs) {
    emitType(ST);
  }
}

void MetalEmitter::emitEarlyFp64Helpers() {
  // Soft-double fp64 representation. Metal GPUs don't natively support
  // fp64, so we lower every LLVM `double` op to a call into the
  // __acpp_sscp_soft_f64_* library. Storage: two uint halves to sidestep
  // MSL alignment quirks on ulong and keep the struct trivially
  // memcpy-able.
  //
  // Emitted BEFORE emitTypes() so that LLVM struct types containing an
  // fp64 field (emitted with `acpp_f64` field type) can reference this
  // declaration. Also emits the `i48u` helper (used by the soft-fp64
  // library bodies themselves) so it's available before the Emitter's
  // topological-sort loop emits the soft-fp64 function definitions.
  os << R"__(
struct acpp_f64 {
  uint lo;
  uint hi;
};

struct i48u {
  packed_ushort3 w;
  i48u() : w(packed_ushort3(0,0,0)) {}
  explicit i48u(packed_ushort3 ww) : w(ww) {}
  explicit i48u(ushort x) : w(packed_ushort3(x, 0, 0)) {}
  explicit i48u(uint x)
  : w(packed_ushort3((ushort)(x & 0xffffu),
                     (ushort)((x >> 16) & 0xffffu),
                     0))
  {}
  explicit i48u(ulong x)
  : w(packed_ushort3((ushort)(x & 0xfffful),
                     (ushort)((x >> 16) & 0xfffful),
                     (ushort)((x >> 32) & 0xfffful)))
  {}

  friend inline i48u operator|(i48u a, i48u b) {
    return i48u(packed_ushort3((ushort)(a.w[0] | b.w[0]),
                               (ushort)(a.w[1] | b.w[1]),
                               (ushort)(a.w[2] | b.w[2])));
  }

  friend inline i48u operator<<(i48u a, uint bits) {
    uint s = bits >> 4; // /16
    if ((bits & 0xFu) != 0) {
      ulong x = a.to_ulong();
      x = (x << bits) & 0x0000FFFFFFFFFFFFul;
      return i48u(x);
    }
    if (s == 0) return a;
    if (s == 1) return i48u(packed_ushort3(0, a.w[0], a.w[1]));
    if (s == 2) return i48u(packed_ushort3(0, 0, a.w[0]));
    return i48u(); // >=48 => 0
  }

  friend inline i48u operator>>(i48u a, uint bits) {
    uint s = bits >> 4; // /16
    if ((bits & 0xFu) != 0) {
      ulong x = a.to_ulong();
      x = (x >> bits);
      return i48u(x);
    }
    if (s == 0) return a;
    if (s == 1) return i48u(packed_ushort3(a.w[1], a.w[2], 0));
    if (s == 2) return i48u(packed_ushort3(a.w[2], 0, 0));
    return i48u(); // >=48 => 0
  }

  inline ulong to_ulong() const {
    return (ulong)w[0] | ((ulong)w[1] << 16) | ((ulong)w[2] << 32);
  }
};
)__";
  os << "\n";
}

void MetalEmitter::emitIntrinsicHelpers() {
  for (Function& F : M) {
    if (F.isDeclaration()) {
      continue;
    }
    for (const BasicBlock& BB : F) {
      for (const Instruction& I : BB) {
        if (!I.getType()->isVoidTy()) {
          // mark used type-helpers
          mapType(I.getType());
        }
      }
    }
  }

  os << R"__(
  inline char __as_signed(uchar value) {
    return as_type<char>(value);
  }

  inline short __as_signed(ushort value) {
    return as_type<short>(value);
  }

  inline int __as_signed(uint value) {
    return as_type<int>(value);
  }

  inline long __as_signed(ulong value) {
    return as_type<long>(value);
  }

  // LLVM sext i1 true to iN yields all-ones (-1), so replicate that semantic
  inline int __as_signed(bool value) {
    return value ? -1 : 0;
  }
)__";

  os << "\n";
}

bool MetalEmitter::emitArgStruct(Function& F) {
  bool isKernel = kernelNames.count(F.getName().str()) > 0;
  if (!isKernel) {
    return true;
  }
  bool useArgStruct = isKernel && F.arg_size() > opt.maxArgsForFlatMode;
  if (!useArgStruct) {
    return true;
  }

  inputStructName = "Input_" + std::to_string(inputStructCounter++);
  os << "struct " << inputStructName << " {\n";
  int fieldIdx = 0;
  for (Argument& A : F.args()) {
    auto typeName = mapType(A.getType());
    os << indent(1) << typeName << " " << valueName(&A) << " [[id(" << fieldIdx++ << ")]];\n";
  }
  os << "};\n\n";
  return true;
}

bool MetalEmitter::emitSignature(Function& F) {
  bool isKernel = kernelNames.count(F.getName().str()) > 0;

  bool useArgStruct = isKernel && F.arg_size() > opt.maxArgsForFlatMode;
  if (isKernel) {
    os << "[[kernel]] ";
  }

  std::string returnType = isKernel ? "void" : mapType(F.getReturnType());
  os << returnType << " " << F.getName().str() << " (";

  bool first = true;
  int bufIdx = 2; // index=0 is reserved for dynamic local memory size, index=1 for host-to-device address difference, so start from 2
  if (useArgStruct) {
    first = false;
    os << "device " << inputStructName << "& __args [[buffer(" << bufIdx++ << ")]]";
  } else {
    for (Argument &A : F.args()) {
      if (!first) os << ", ";
      first = false;
      auto typeName = mapType(A.getType());

      if (isKernel) {
        if (!A.getType()->isPointerTy()) {
          os << "constant " << typeName << "& " << valueName(&A) << " [[buffer(" << bufIdx++ << ")]]";
        } else {
          os << typeName << " " << valueName(&A) << " [[buffer(" << bufIdx++ << ")]]";
        }

      } else {
        os << typeName << " " << valueName(&A);
      }
    }
  }

  os << ")\n{\n";

  if (useArgStruct) {
    for (Argument &A : F.args()) {
      auto typeName = mapType(A.getType());
      os << indent(1) << typeName << " " << valueName(&A) << " = __args." << valueName(&A) << ";\n";
    }
    os << "\n";
  }
  return true;
}

bool MetalEmitter::emitDeclarations() {
  os << indent(1) << "// Locals\n";
  std::vector<std::pair<const Instruction*, int>> sortedAllocas(allocaIndex.begin(), allocaIndex.end());
  std::sort(sortedAllocas.begin(), sortedAllocas.end(),
            [](const auto &a, const auto &b) { return a.second < b.second; });
  for (auto &[AI, idx] : sortedAllocas) {
    Type *allocTy = dyn_cast<AllocaInst>(AI)->getAllocatedType();
    auto typeName = mapType(allocTy);
    // Change 1: if this alloca backs an atomic i64 (SYCL atomic_ref<uint64_t>
    // or LLVM atomicrmw on i64), promote storage to `atomic_ulong` so the
    // libkernel helper's `__atomic_pointer_cast<long>` cast lands on
    // layout-compatible storage.
    if (atomicI64Values.count(AI) && allocTy->isIntegerTy() &&
        allocTy->getIntegerBitWidth() == 64) {
      typeName = "atomic_ulong";
    }
    os << indent(1) << typeName << " local" << idx << ";\n";
  }
  os << "\n";

  // PHI input variables - sorted by name
  if (!phiNodes.empty()) {
    os << indent(1) << "// PHI input variables\n";
    std::vector<const PHINode*> sortedPhis(phiNodes.begin(), phiNodes.end());
    std::unordered_map<const Value*, std::string> phiNameCache;
    for (const PHINode *PHI : sortedPhis) {
      phiNameCache[PHI] = valueName(PHI);
    }
    std::sort(sortedPhis.begin(), sortedPhis.end(),
              [&phiNameCache](const PHINode *a, const PHINode *b) { return phiNameCache[a] < phiNameCache[b]; });
    for (const PHINode *PHI : sortedPhis) {
      auto typeName = mapType(PHI);
      os << indent(1) << typeName << " " << phiNameCache[PHI] << "_in;\n";
    }
    os << "\n";
  }

  // All other variables - sorted by name
  os << indent(1) << "// Variables\n";
  std::vector<std::pair<const Value*, std::string>> sortedVars(valuesToDeclare.begin(), valuesToDeclare.end());
  std::unordered_map<const Value*, std::string> varNameCache;
  for (const auto &[V, ty] : sortedVars) {
    varNameCache[V] = valueName(V);
  }
  std::sort(sortedVars.begin(), sortedVars.end(),
            [&varNameCache](const auto &a, const auto &b) { return varNameCache[a.first] < varNameCache[b.first]; });
  for (auto &[V, ty] : sortedVars) {
    const std::string &name = varNameCache[V];
    // Skip allocas (handled above as pointers)
    if (isa<AllocaInst>(V)) {
      os << indent(1) << "thread void* " << name << ";\n";
      continue;
    }
    // Skip PHIs (handled separately)
    if (isa<PHINode>(V)) {
      os << indent(1) << mapType(V) << " " << name << ";\n";
      continue;
    }
    os << indent(1) << ty << " " << name << ";\n";
  }
  os << "\n";
  return true;
}

bool MetalEmitter::emitNode(const Node& node, int level) {
  bool success = true;
  switch (node.kind) {
    case NodeKind::List: {
      auto& listNode = static_cast<const ListNode&>(node);
      for (const auto& item : listNode.items) {
        if (item && success) {
          success &= emitNode(*item, level);
        }
      }
      break;
    }

    case NodeKind::Block: {
      auto& blockNode = static_cast<const BlockNode&>(node);
      if (blockNode.bb) {
        os << indent(level) << "// block " << basicBlockName(blockNode.bb) << "\n";
        success = emitBasicBlock(blockNode.bb, level);
      }
      break;
    }

    case NodeKind::If: {
      auto& ifNode = static_cast<const IfNode&>(node);

      os << indent(level) << "if (" << emitExpr(ifNode.cond) << ") {\n";
      if (ifNode.then_branch) {
        success &= emitNode(*ifNode.then_branch, level + 1);
      }
      os << indent(level) << "}";
      if (ifNode.else_branch) {
        os << " else {\n";
        success &= emitNode(*ifNode.else_branch, level + 1);
        os << indent(level) << "}";
      }
      os << "\n";
      break;
    }

    case NodeKind::Loop: {
      auto& loopNode = static_cast<const LoopNode&>(node);

      os << indent(level) << "while (true) {\n";
      if (loopNode.body) {
        success = emitNode(*loopNode.body, level + 1);
      }
      os << indent(level) << "}\n";

      break;
    }

    case NodeKind::Break: {
      os << indent(level) << "break;\n";
      break;
    }

    case NodeKind::Continue: {
      os << indent(level) << "continue;\n";
      break;
    }

    case NodeKind::Return: {
      auto& returnNode = static_cast<const ReturnNode&>(node);
      if (returnNode.value) {
        os << indent(level) << "return " << emitExpr(returnNode.value) << ";\n";
      } else {
        os << indent(level) << "return;\n";
      }
      break;
    }

    default: {
      std::ostringstream ss;
      ss << "ERROR: Unknown HL node kind: " << static_cast<int>(node.kind);
      errorMsg = ss.str();
      return false;
    }
  }
  return success;
}

bool MetalEmitter::emitBasicBlock(const BasicBlock* BB, int level) {
  const Instruction *TI = BB->getTerminator();
  bool success = true;
  for (const Instruction& I : *BB) {
    if (&I == TI) {
      continue;
    }
    if (!success) {
      return false;
    }
    success &= emitInstruction(I, level);
  }

  for (const PHINode *PHI : phiNodes) {
    auto it = phiIncomingFromBlock.find({BB, PHI});
    if (it != phiIncomingFromBlock.end()) {
      const Value *V = it->second;
      if (!isa<UndefValue>(V) && !isa<PoisonValue>(V)) {
        os << indent(level) << valueName(PHI) << "_in = " << emitExpr(V) << ";\n";
      } else {
        // For aggregate / non-scalar types the integer literal `0` is
        // not assignable — MSL needs a type-compatible placeholder.
        // Use `{}` for structs/vectors, `0` for scalar integers,
        // `0.0` for float/acpp_f64 is wrong too; any constant is fine
        // because "undef" means we must not depend on the value.
        std::string ty = mapType(PHI->getType());
        if (ty == "acpp_f64") {
          os << indent(level) << valueName(PHI) << "_in = acpp_f64{0u, 0u};\n";
        } else if (PHI->getType()->isAggregateType() ||
                   PHI->getType()->isVectorTy()) {
          os << indent(level) << valueName(PHI) << "_in = " << ty << "{};\n";
        } else {
          os << indent(level) << valueName(PHI) << "_in = /* undef */ 0;\n";
        }
      }
    }
  }
  return success;
}

bool MetalEmitter::emitInstruction(const Instruction& I, int level) {
  std::string name = valueName(&I);

  if (auto *AI = dyn_cast<AllocaInst>(&I)) {
    os << indent(level) << name << " = &local" << allocaIndex[&I] << "; " << "// " << instToString(I) << "\n";
    return true;
  }

  if (auto *LI = dyn_cast<LoadInst>(&I)) {
    std::string ptr = emitExpr(LI->getPointerOperand());
    std::string elemType = mapType(LI->getType());
    std::string addrSpace;

    if (LI->getPointerOperandType()->isPointerTy()) {
      // Do NOT trust pointer operand type if it went through addrspacecast-to-generic.
      unsigned physAS = getPhysicalPointerAddressSpace(LI->getPointerOperand());
      addrSpace = getAddressSpaceKeyword(physAS);
    } else {
      errorMsg = "Error: Load from non-pointer type";
      return false;
    }

    if (LI->getType()->isPointerTy()) {
      // ** type, need to use struct slot hack to dereference pointer to pointer
      auto elemPhysAS = getPhysicalPointerAddressSpace(LI);
      auto elemAddrSpace = getAddressSpaceKeyword(elemPhysAS);
      auto structName = "__struct_ptr_to_" + elemAddrSpace;
      os << indent(level) << name << " = ((" << addrSpace << " " << structName << "*)" << ptr << ")->value; " << "// " << instToString(I) << "\n";
    } else {
      os << indent(level) << name << " = *((" << addrSpace << " " << elemType << "*)" << ptr << "); " << "// " << instToString(I) << "\n";
    }
    return true;
  }

  if (auto *SI = dyn_cast<StoreInst>(&I)) {
    std::string ptr = emitExpr(SI->getPointerOperand());
    std::string val = emitExpr(SI->getValueOperand());
    std::string elemType = mapType(SI->getValueOperand()->getType());
    unsigned physAS = getPhysicalPointerAddressSpace(SI->getPointerOperand());
    std::string addrSpace = getAddressSpaceKeyword(physAS);

    if (SI->getValueOperand()->getType()->isPointerTy()) {
      // ** type, need to use struct slot hack to dereference pointer to pointer
      auto elemPhysAS = getPhysicalPointerAddressSpace(SI->getValueOperand());
      auto elemAddrSpace = getAddressSpaceKeyword(elemPhysAS);
      auto structName = "__struct_ptr_to_" + elemAddrSpace;
      os << indent(level) << "((" << addrSpace << " " << structName << "*)" << ptr << ")->value = " << val << "; " << "// " << instToString(I) << "\n";
    } else {
      os << indent(level) << "*((" << addrSpace << " " << elemType << "*)" << ptr << ") = " << val << "; " << "// " << instToString(I) << "\n";
    }
    return true;
  }

  if (auto *PHI = dyn_cast<PHINode>(&I)) {
    os << indent(level) << name << " = " << name << "_in; " << "// " << instToString(I) << "\n";
    return true;
  }

  if (auto *UO = dyn_cast<UnaryOperator>(&I)) {
    emitUnaryOperator(UO, name, level);
    return true;
  }

  if (auto *BO = dyn_cast<BinaryOperator>(&I)) {
    emitBinaryOperator(BO, name, level);
    return true;
  }

  if (auto *IC = dyn_cast<ICmpInst>(&I)) {
    emitICmpInstruction(IC, name, level);
    return true;
  }

  if (auto *FC = dyn_cast<FCmpInst>(&I)) {
    emitFCmpInstruction(FC, name, level);
    return true;
  }

  if (auto *SI = dyn_cast<SelectInst>(&I)) {
    std::string cond = emitExpr(SI->getCondition());
    std::string trueVal = emitExpr(SI->getTrueValue());
    std::string falseVal = emitExpr(SI->getFalseValue());
    os << indent(level) << name << " = (" << cond << ") ? (" << trueVal << ") : (" << falseVal << "); " << "// " << instToString(I) << "\n";
    return true;
  }

  if (auto *CI = dyn_cast<CastInst>(&I)) {
    return emitCastInstruction(CI, name, level);
  }

  if (auto *GEP = dyn_cast<GetElementPtrInst>(&I)) {
    emitGEPInstruction(GEP, name, level);
    return true;
  }

  if (auto *CI = dyn_cast<CallInst>(&I)) {
    return emitCallInstruction(CI, name, level);
  }

  if (auto *EV = dyn_cast<ExtractValueInst>(&I)) {
    std::string aggr = emitExpr(EV->getAggregateOperand());
    os << indent(level) << name << " = " << aggr;
    Type* aggrType = EV->getAggregateOperand()->getType();
    for (unsigned idx : EV->getIndices()) {
      if (aggrType->isStructTy()) {
        os << ".field" << idx;
        aggrType = aggrType->getStructElementType(idx);
      } else if (aggrType->isArrayTy()) {
        os << "[" << idx << "]";
        aggrType = aggrType->getArrayElementType();
      } else {
        errorMsg = "Error: ExtractValue from unsupported aggregate type";
        return false;
      }
    }
    os << "; " << "// " << instToString(I) << "\n";
    return true;
  }

  if (auto *IV = dyn_cast<InsertValueInst>(&I)) {
    std::string res = emitExpr(IV);
    std::string aggr = emitExpr(IV->getAggregateOperand());
    std::string val = emitExpr(IV->getInsertedValueOperand());
    Type* aggrType = IV->getAggregateOperand()->getType();
    auto typeName = mapType(aggrType);
    os << indent(level) << res << " = " << aggr << "; " << "// " << instToString(I) << "\n";
    os << indent(level) << res;
    for (unsigned idx : IV->getIndices()) {
      if (aggrType->isStructTy()) {
        os << ".field" << idx;
        aggrType = aggrType->getStructElementType(idx);
      } else if (aggrType->isArrayTy()) {
        os << "[" << idx << "]";
        aggrType = aggrType->getArrayElementType();
      } else {
        errorMsg = "Error: InsertValue into unsupported aggregate type";
        return false;
      }
      os << " = " << val << "; " << "// " << instToString(I) << "\n";
    }
    return true;
  }

  if (auto *RI = dyn_cast<ReturnInst>(&I)) {
    if (Value *RV = RI->getReturnValue()) {
      os << indent(level) << "return " << emitExpr(RV) << "; " << "// " << instToString(I) << "\n";
    } else {
      os << indent(level) << "return; " << "// " << instToString(I) << "\n";
    }
    return true;
  }

  if (auto *FI = dyn_cast<FreezeInst>(&I)) {
    std::string operand = emitExpr(FI->getOperand(0));
    os << indent(level) << name << " = " << operand << "; " << "// " << instToString(I) << "\n";
    return true;
  }

  errorMsg = "ERROR: Unknown LLVM instruction: " + std::string(I.getOpcodeName());
  if (I.hasName()) {
    *errorMsg += " (name: " + I.getName().str() + ")";
  }
  *errorMsg += "\n";
  *errorMsg += "  " + instToString(I) + "\n";
  return false;
}

void MetalEmitter::emitUnaryOperator(const UnaryOperator* UO, const std::string& name, int level) {
  switch (UO->getOpcode()) {
    case Instruction::FNeg: {
      std::string operand = emitExpr(UO->getOperand(0));
      // Change 2: soft-double fneg. Cannot use `-x` on acpp_f64.
      if (UO->getType()->isDoubleTy()) {
        // TODO(acpp-soft-fp64): __acpp_sscp_soft_f64_neg
        os << indent(level) << name << " = __acpp_sscp_soft_f64_neg("
           << operand << "); // " << instToString(*UO) << "\n";
        break;
      }
      if (auto* VT = dyn_cast<FixedVectorType>(UO->getType())) {
        if (VT->getElementType()->isDoubleTy()) {
          // TODO(acpp-soft-fp64): element-wise unroll for <N x double> fneg
          unsigned N = VT->getNumElements();
          for (unsigned i = 0; i < N; ++i) {
            os << indent(level) << name << "[" << i
               << "] = __acpp_sscp_soft_f64_neg(" << operand << "[" << i
               << "]); // " << instToString(*UO) << "\n";
          }
          break;
        }
      }
      os << indent(level) << name << " = -" << operand << "; " << "// " << instToString(*UO) << "\n";
      break;
    }

    default:
      errorMsg = "ERROR: Unknown unary operator: " + std::string(UO->getOpcodeName());
      return;
  }
}

bool MetalEmitter::emitCastInstruction(const CastInst* CI, const std::string& name, int level) {
  auto destType = mapType(CI->getDestTy());
  std::string src = emitExpr(CI->getOperand(0));
  auto srcType = mapType(CI->getSrcTy());

  // Change 2: soft-double conversions. Any cast touching fp64 (source or
  // destination) is lowered to a libkernel helper. Integer widths 32/64 are
  // first-class; narrower integers round-trip via i32.
  // TODO(acpp-soft-fp64): symbols referenced below provided by libkernel.
  auto isF64Ty = [](llvm::Type* T) { return T->isDoubleTy(); };
  auto intBits = [](llvm::Type* T) -> unsigned {
    return T->isIntegerTy() ? T->getIntegerBitWidth() : 0u;
  };

  if (isF64Ty(CI->getDestTy()) || isF64Ty(CI->getSrcTy())) {
    llvm::Type* DT = CI->getDestTy();
    llvm::Type* ST = CI->getSrcTy();
    switch (CI->getOpcode()) {
      case Instruction::FPExt: {
        // float -> double
        os << indent(level) << name << " = __acpp_sscp_soft_f64_from_f32("
           << src << "); // " << instToString(*CI) << "\n";
        return true;
      }
      case Instruction::FPTrunc: {
        // double -> float
        os << indent(level) << name << " = __acpp_sscp_soft_f64_to_f32("
           << src << "); // " << instToString(*CI) << "\n";
        return true;
      }
      case Instruction::FPToSI: {
        // double -> iN
        unsigned bits = intBits(DT);
        if (bits == 32 || bits == 64) {
          os << indent(level) << name << " = __acpp_sscp_soft_f64_to_i"
             << bits << "(" << src << "); // " << instToString(*CI) << "\n";
        } else {
          os << indent(level) << name
             << " = (" << destType
             << ") __acpp_sscp_soft_f64_to_i32(" << src << "); // "
             << instToString(*CI) << "\n";
        }
        return true;
      }
      case Instruction::FPToUI: {
        unsigned bits = intBits(DT);
        if (bits == 32 || bits == 64) {
          os << indent(level) << name << " = __acpp_sscp_soft_f64_to_u"
             << bits << "(" << src << "); // " << instToString(*CI) << "\n";
        } else {
          os << indent(level) << name
             << " = (" << destType
             << ") __acpp_sscp_soft_f64_to_u32(" << src << "); // "
             << instToString(*CI) << "\n";
        }
        return true;
      }
      case Instruction::SIToFP: {
        unsigned bits = intBits(ST);
        if (bits == 32 || bits == 64) {
          os << indent(level) << name << " = __acpp_sscp_soft_f64_from_i"
             << bits << "(__as_signed(" << src << ")); // "
             << instToString(*CI) << "\n";
        } else {
          os << indent(level) << name
             << " = __acpp_sscp_soft_f64_from_i32((int)__as_signed(" << src
             << ")); // " << instToString(*CI) << "\n";
        }
        return true;
      }
      case Instruction::UIToFP: {
        unsigned bits = intBits(ST);
        if (bits == 32 || bits == 64) {
          os << indent(level) << name << " = __acpp_sscp_soft_f64_from_u"
             << bits << "(" << src << "); // " << instToString(*CI) << "\n";
        } else {
          os << indent(level) << name
             << " = __acpp_sscp_soft_f64_from_u32((uint)" << src
             << "); // " << instToString(*CI) << "\n";
        }
        return true;
      }
      case Instruction::BitCast: {
        // i64 <-> double: rebuild via lo/hi halves. Use as_type<uint2>(ulong)
        // to split, and pack back with ((ulong)hi << 32) | lo.
        if (isF64Ty(DT) && intBits(ST) == 64) {
          os << indent(level) << name
             << " = acpp_f64 { as_type<uint2>(" << src << ").x, as_type<uint2>("
             << src << ").y }; // " << instToString(*CI) << "\n";
          return true;
        }
        if (isF64Ty(ST) && intBits(DT) == 64) {
          os << indent(level) << name << " = (((ulong)" << src << ".hi) << 32) | (ulong)"
             << src << ".lo; // " << instToString(*CI) << "\n";
          return true;
        }
        // Fall through to generic BitCast handling for other mixes.
        break;
      }
      default: break;
    }
  }

  auto emitUint4Cast = [&]() -> bool {
    struct CastEntry {
      const char* src;
      const char* dst;
      const char* fmt;
    };
    CastEntry table[] = {
      // trunc i128 -> i32
      {"uint4", "uint",  "{src}.x"},
      // truct i128 -> i48u
      {"uint4", "i48u", "i48u(packed_ushort3({src}.x, {src}.y, {src}.z))"},
      // trunc i128 -> i64
      {"uint4", "ulong", "as_type<ulong>({src}.xy)"},
      // zext i32 -> i128
      {"uint",  "uint4", "uint4({src}, 0u, 0u, 0u)"},
      // zext i48u -> i128
      {"i48u",  "uint4", "uint4({src}.w[0], {src}.w[1], {src}.w[2], 0u)"},
      // zext i64 -> i128
      {"ulong", "uint4", "uint4(as_type<uint2>({src}), 0u, 0u)"},
    };
    for (auto& entry : table) {
      if (srcType == entry.src && destType == entry.dst) {
        std::string expr = entry.fmt;
        auto pos = expr.find("{src}");
        while (pos != std::string::npos) {
          expr.replace(pos, 5, src);
          pos = expr.find("{src}", pos + src.size());
        }
        os << indent(level) << name << " = " << expr << "; // " << instToString(*CI) << "\n";
        return true;
      }
    }
    errorMsg = "Error: Unsupported cast involving uint4: " + srcType + " -> " + destType;
    return false;
  };

  // Special-case: addrspacecast to generic loses the information; in MSL we treat it as no-op.
  if (dyn_cast<AddrSpaceCastInst>(CI)) {
    os << indent(level) << name << " = " << src << "; // " << instToString(*CI) << "\n";
  } else if (dyn_cast<BitCastInst>(CI)) {
    os << indent(level) << name << " = as_type<" << destType << ">(" << src << "); // " << instToString(*CI) << "\n";
  } else if (dyn_cast<SExtInst>(CI)) {
    auto destSigned = getSignedType(CI->getDestTy());
    os << indent(level) << name << " = as_type<" << destType << ">((" << destSigned << ")__as_signed(" << src << ")); // " << instToString(*CI) << "\n";
  } else if (CI->getOpcode() == Instruction::SIToFP) {
    os << indent(level) << name << " = (" << destType << ") __as_signed(" << src << "); // " << instToString(*CI) << "\n";
  } else if (CI->getOpcode() == Instruction::FPToSI) {
    auto destSigned = getSignedType(CI->getDestTy());
    os << indent(level) << name << " = (" << destSigned << ") " << src << "; // " << instToString(*CI) << "\n";
  } else if (srcType == "uint4" || destType == "uint4") {
    return emitUint4Cast();
  } else {
    os << indent(level) << name << " = (" << destType << ") " << src << "; // " << instToString(*CI) << "\n";
  }
  return true;
}

void MetalEmitter::emitBinaryOperator(const BinaryOperator* BO, const std::string& name, int level) {
  std::string lhs = emitExpr(BO->getOperand(0));
  std::string rhs = emitExpr(BO->getOperand(1));

  std::string resultType = mapType(BO->getType());

  // Change 2: soft-double lowering. Any BinaryOperator whose result type is
  // double (scalar or <N x double>) is lowered to a call into the
  // __acpp_sscp_soft_f64_* library. The libkernel agent provides these
  // symbols in parallel — linker will complain until they land (expected).
  auto softF64Sym = [](unsigned Opcode) -> const char* {
    switch (Opcode) {
      case Instruction::FAdd: return "__acpp_sscp_soft_f64_add";
      case Instruction::FSub: return "__acpp_sscp_soft_f64_sub";
      case Instruction::FMul: return "__acpp_sscp_soft_f64_mul";
      case Instruction::FDiv: return "__acpp_sscp_soft_f64_div";
      case Instruction::FRem: return "__acpp_sscp_soft_f64_rem";
      default: return nullptr;
    }
  };

  if (BO->getType()->isDoubleTy()) {
    if (const char* sym = softF64Sym(BO->getOpcode())) {
      // TODO(acpp-soft-fp64): __acpp_sscp_soft_f64_{add,sub,mul,div,rem}
      os << indent(level) << name << " = " << sym << "(" << lhs << ", "
         << rhs << "); // " << instToString(*BO) << "\n";
      return;
    }
  }
  if (auto* VT = dyn_cast<FixedVectorType>(BO->getType())) {
    if (VT->getElementType()->isDoubleTy()) {
      if (const char* sym = softF64Sym(BO->getOpcode())) {
        // TODO(acpp-soft-fp64): element-wise unroll for <N x double>
        unsigned N = VT->getNumElements();
        for (unsigned i = 0; i < N; ++i) {
          os << indent(level) << name << "[" << i << "] = " << sym << "("
             << lhs << "[" << i << "], " << rhs << "[" << i << "]); // "
             << instToString(*BO) << "\n";
        }
        return;
      }
    }
  }

  switch (BO->getOpcode()) {
    case Instruction::FAdd:
    case Instruction::Add:
      os << indent(level) << name << " = " << lhs << " + " << rhs << "; " << "// " << instToString(*BO) << "\n";
      break;

    case Instruction::FSub:
    case Instruction::Sub:
      os << indent(level) << name << " = " << lhs << " - " << rhs << "; " << "// " << instToString(*BO) << "\n";
      break;

    case Instruction::FMul:
    case Instruction::Mul:
      os << indent(level) << name << " = " << lhs << " * " << rhs << "; " << "// " << instToString(*BO) << "\n";
      break;

    case Instruction::FDiv:
    case Instruction::UDiv:
      os << indent(level) << name << " = " << lhs << " / " << rhs << "; " << "// " << instToString(*BO) << "\n";
      break;
    case Instruction::URem:
      os << indent(level) << name << " = " << lhs << " % " << rhs << "; " << "// " << instToString(*BO) << "\n";
      break;

    case Instruction::SDiv:
      os << indent(level) << name << " = as_type<" << resultType << ">((__as_signed(" << lhs << ")) / (__as_signed(" << rhs << ")));\n";
      break;
    case Instruction::SRem:
      os << indent(level) << name << " = as_type<" << resultType << ">((__as_signed(" << lhs << ")) % (__as_signed(" << rhs << ")));\n";
      break;
    case Instruction::FRem:
      os << indent(level) << name << " = fmod(" << lhs << ", " << rhs << ");\n";
      break;

    case Instruction::Shl:
      if (resultType == "uint4") {
        // uint4 is the Metal lowering of LLVM `i128`. Lanes hold the
        // 128-bit value in little-endian 32-bit words: x = bits [0..31],
        // y = [32..63], z = [64..95], w = [96..127].
        // For constant shift amounts we emit a specialised expression
        // (the soft-fp64 library shifts by 9/23/55/etc. inside fma).
        if (auto* CIrhs = llvm::dyn_cast<llvm::ConstantInt>(BO->getOperand(1))) {
          uint64_t shift = CIrhs->getZExtValue() & 127u;
          uint64_t wordShift = shift / 32;
          uint64_t bitShift  = shift % 32;
          auto lane = [&](int i) -> std::string {
            static const char* kNames[4] = {".x", ".y", ".z", ".w"};
            return lhs + kNames[i];
          };
          auto zero = std::string{"0u"};
          std::string lanes[4];
          for (int i = 0; i < 4; ++i) {
            int srcHi = i - static_cast<int>(wordShift);
            int srcLo = srcHi - 1;
            std::string hi = (srcHi >= 0 && srcHi < 4) ? lane(srcHi) : zero;
            std::string lo = (srcLo >= 0 && srcLo < 4) ? lane(srcLo) : zero;
            if (bitShift == 0) {
              lanes[i] = hi;
            } else {
              // (src[hi] << bitShift) | (src[lo] >> (32 - bitShift))
              lanes[i] = "((" + hi + " << " + std::to_string(bitShift) +
                         "u) | (" + lo + " >> " +
                         std::to_string(32u - bitShift) + "u))";
            }
          }
          os << indent(level) << name << " = uint4(" << lanes[0] << ","
             << lanes[1] << "," << lanes[2] << "," << lanes[3] << "); // "
             << instToString(*BO) << "\n";
        } else {
          errorMsg = "ERROR: uint4 Shl with non-constant shift amount: " + rhs;
        }
      } else {
        os << indent(level) << name << " = " << lhs << " << " << rhs << ";\n";
      }
      break;
    case Instruction::AShr:
      os << indent(level) << name << " = as_type<" << resultType << ">((__as_signed(" << lhs << ")) >> (__as_signed(" << rhs << ")));\n";
      break;

    case Instruction::LShr:
      if (resultType == "uint4") {
        // i128 logical right shift. Lanes are little-endian 32-bit
        // words (see Shl above). For lane `i` the result is built from
        // `src[i + wordShift]` (low word) and `src[i + wordShift + 1]`
        // (high word), combined via `bitShift`.
        if (auto* CIrhs = llvm::dyn_cast<llvm::ConstantInt>(BO->getOperand(1))) {
          uint64_t shift = CIrhs->getZExtValue() & 127u;
          uint64_t wordShift = shift / 32;
          uint64_t bitShift  = shift % 32;
          auto lane = [&](int i) -> std::string {
            static const char* kNames[4] = {".x", ".y", ".z", ".w"};
            return lhs + kNames[i];
          };
          auto zero = std::string{"0u"};
          std::string lanes[4];
          for (int i = 0; i < 4; ++i) {
            int srcLo = i + static_cast<int>(wordShift);
            int srcHi = srcLo + 1;
            std::string lo = (srcLo >= 0 && srcLo < 4) ? lane(srcLo) : zero;
            std::string hi = (srcHi >= 0 && srcHi < 4) ? lane(srcHi) : zero;
            if (bitShift == 0) {
              lanes[i] = lo;
            } else {
              // (src[lo] >> bitShift) | (src[hi] << (32 - bitShift))
              lanes[i] = "((" + lo + " >> " + std::to_string(bitShift) +
                         "u) | (" + hi + " << " +
                         std::to_string(32u - bitShift) + "u))";
            }
          }
          os << indent(level) << name << " = uint4(" << lanes[0] << ","
             << lanes[1] << "," << lanes[2] << "," << lanes[3] << "); // "
             << instToString(*BO) << "\n";
        } else {
          errorMsg = "ERROR: uint4 LShr with non-constant shift amount: " + rhs;
        }
      } else {
        os << indent(level) << name << " = " << lhs << " >> " << rhs << ";\n";
      }
      break;

    // bit logical ops
    case Instruction::And:
      os << indent(level) << name << " = " << lhs << " & " << rhs << ";\n";
      break;
    case Instruction::Or:
      os << indent(level) << name << " = " << lhs << " | " << rhs << ";\n";
      break;
    case Instruction::Xor:
      os << indent(level) << name << " = " << lhs << " ^ " << rhs << ";\n";
      break;


    default: {
      std::ostringstream ss;
      ss << "ERROR: Unknown binary operator: " << BO->getOpcodeName() << std::endl;
      errorMsg = ss.str();
      return;
    }
  }
}

void MetalEmitter::emitICmpInstruction(const ICmpInst* IC, const std::string& name, int level) {
  std::string lhs = emitExpr(IC->getOperand(0));
  std::string rhs = emitExpr(IC->getOperand(1));

  auto resultType = mapType(IC->getType());

  // Change 2: i128 lowers to `uint4` via mapType (4x uint32). MSL vector-
  // scalar comparison is component-wise and returns `bool4`, but LLVM's
  // icmp mandates a scalar `i1` result — reduce via all()/any() for eq/ne.
  // Ordered (unsigned/signed lt/gt/le/ge) comparisons on i128 require a
  // proper big-integer comparator and are not emitted by the soft-fp64
  // prelude in practice; report as unsupported rather than silently emit
  // a component-wise compare that would collapse to `(bool4,bool4)→bool4`.
  Type *opTy = IC->getOperand(0)->getType();
  bool isI128 = opTy->isIntegerTy(128);

  switch (IC->getPredicate()) {
    case ICmpInst::ICMP_EQ:
      if (isI128) {
        os << indent(level) << name << " = all(" << lhs << " == " << rhs << ");\n";
      } else {
        os << indent(level) << name << " = (" << lhs << " == " << rhs << ");\n";
      }
      break;
    case ICmpInst::ICMP_NE:
      if (isI128) {
        os << indent(level) << name << " = any(" << lhs << " != " << rhs << ");\n";
      } else {
        os << indent(level) << name << " = (" << lhs << " != " << rhs << ");\n";
      }
      break;
    case ICmpInst::ICMP_UGT:
    case ICmpInst::ICMP_UGE:
    case ICmpInst::ICMP_ULT:
    case ICmpInst::ICMP_ULE:
    case ICmpInst::ICMP_SGT:
    case ICmpInst::ICMP_SGE:
    case ICmpInst::ICMP_SLT:
    case ICmpInst::ICMP_SLE: {
      if (isI128) {
        errorMsg = "Ordered icmp on i128 is not supported by the Metal "
                   "emitter (would need a big-integer comparator).";
        return;
      }
      const char *op = nullptr;
      bool isSigned = false;
      switch (IC->getPredicate()) {
        case ICmpInst::ICMP_UGT: op = ">"; break;
        case ICmpInst::ICMP_UGE: op = ">="; break;
        case ICmpInst::ICMP_ULT: op = "<"; break;
        case ICmpInst::ICMP_ULE: op = "<="; break;
        case ICmpInst::ICMP_SGT: op = ">"; isSigned = true; break;
        case ICmpInst::ICMP_SGE: op = ">="; isSigned = true; break;
        case ICmpInst::ICMP_SLT: op = "<"; isSigned = true; break;
        case ICmpInst::ICMP_SLE: op = "<="; isSigned = true; break;
        default: break;
      }
      if (isSigned) {
        os << indent(level) << name << " = ((__as_signed(" << lhs << ")) "
           << op << " (__as_signed(" << rhs << ")));\n";
      } else {
        os << indent(level) << name << " = (" << lhs << " " << op << " "
           << rhs << ");\n";
      }
      break;
    }
    default: {
      std::ostringstream ss;
      ss << "ERROR: Unknown ICmp predicate: " << IC->getPredicate() << std::endl;
      errorMsg = ss.str();
      return;
    }
  }
}

void MetalEmitter::emitFCmpInstruction(const FCmpInst* FC, const std::string& name, int level) {
  std::string lhs = emitExpr(FC->getOperand(0));
  std::string rhs = emitExpr(FC->getOperand(1));

  // Change 2: soft-double comparison. Route through a single soft-float
  // fcmp helper keyed by the LLVM FCmpInst::Predicate enum value.
  // The libkernel agent provides __acpp_sscp_soft_f64_fcmp(lhs, rhs, pred)
  // returning bool with the same semantics as LLVM's fcmp for predicate
  // 0..15 (from FCMP_FALSE to FCMP_TRUE).
  if (FC->getOperand(0)->getType()->isDoubleTy()) {
    // TODO(acpp-soft-fp64): __acpp_sscp_soft_f64_fcmp(a, b, predicate)
    //   predicate values follow llvm::FCmpInst::Predicate (uchar 0..15).
    unsigned pred = static_cast<unsigned>(FC->getPredicate());
    os << indent(level) << name << " = __acpp_sscp_soft_f64_fcmp(" << lhs
       << ", " << rhs << ", " << pred << "u); // " << instToString(*FC)
       << "\n";
    return;
  }
  if (auto* VT = dyn_cast<FixedVectorType>(FC->getOperand(0)->getType())) {
    if (VT->getElementType()->isDoubleTy()) {
      // TODO(acpp-soft-fp64): element-wise fcmp unroll for <N x double>
      unsigned N = VT->getNumElements();
      unsigned pred = static_cast<unsigned>(FC->getPredicate());
      for (unsigned i = 0; i < N; ++i) {
        os << indent(level) << name << "[" << i
           << "] = __acpp_sscp_soft_f64_fcmp(" << lhs << "[" << i << "], "
           << rhs << "[" << i << "], " << pred << "u); // "
           << instToString(*FC) << "\n";
      }
      return;
    }
  }

  switch (FC->getPredicate()) {
    case FCmpInst::FCMP_OEQ:
      os << indent(level) << name << " = (" << lhs << " == " << rhs << ");\n";
      break;
    case FCmpInst::FCMP_ONE:
      os << indent(level) << name << " = (isordered(" << lhs << ", " << rhs << ") && (" << lhs << " != " << rhs << "));\n";
      break;
    case FCmpInst::FCMP_OGT:
      os << indent(level) << name << " = (" << lhs << " > " << rhs << ");\n";
      break;
    case FCmpInst::FCMP_OGE:
      os << indent(level) << name << " = (" << lhs << " >= " << rhs << ");\n";
      break;
    case FCmpInst::FCMP_OLT:
      os << indent(level) << name << " = (" << lhs << " < " << rhs << ");\n";
      break;
    case FCmpInst::FCMP_OLE:
      os << indent(level) << name << " = (" << lhs << " <= " << rhs << ");\n";
      break;
    case FCmpInst::FCMP_ORD:
      os << indent(level) << name << " = isordered(" << lhs << ", " << rhs << ");\n";
      break;
    case FCmpInst::FCMP_UNO:
      os << indent(level) << name << " = isunordered(" << lhs << ", " << rhs << ");\n";
      break;

    case FCmpInst::FCMP_UEQ:
      os << indent(level) << name << " = (isunordered(" << lhs << ", " << rhs << ") || (" << lhs << " == " << rhs << "));\n";
      break;
    case FCmpInst::FCMP_UGT:
      os << indent(level) << name << " = (isunordered(" << lhs << ", " << rhs << ") || (" << lhs << " > " << rhs << "));\n";
      break;
    case FCmpInst::FCMP_UGE:
      os << indent(level) << name << " = (isunordered(" << lhs << ", " << rhs << ") || (" << lhs << " >= " << rhs << "));\n";
      break;
    case FCmpInst::FCMP_ULT:
      os << indent(level) << name << " = (isunordered(" << lhs << ", " << rhs << ") || (" << lhs << " < " << rhs << "));\n";
      break;
    case FCmpInst::FCMP_ULE:
      os << indent(level) << name << " = (isunordered(" << lhs << ", " << rhs << ") || (" << lhs << " <= " << rhs << "));\n";
      break;
    case FCmpInst::FCMP_UNE:
      os << indent(level) << name << " = (isunordered(" << lhs << ", " << rhs << ") || (" << lhs << " != " << rhs << "));\n";
      break;

    default: {
      std::ostringstream ss;
      ss << "ERROR: Unknown FCmp predicate: " << FC->getPredicate() << std::endl;
      errorMsg = ss.str();
      return;
    }
  }
}

void MetalEmitter::emitGEPInstruction(const GetElementPtrInst* GEP, const std::string& name, int level) {
  std::string base = emitExpr(GEP->getPointerOperand());
  Type* srcElemType = GEP->getSourceElementType();
  std::string typeName = mapType(srcElemType);
  // Pointer-typed elements (e.g., ptr[]) cannot be expressed in MSL as "device void**"
  if (srcElemType->isPointerTy()) {
    auto pointerAS = srcElemType->getPointerAddressSpace();
    auto elemAddrSpace = getAddressSpaceKeyword(pointerAS);
    auto structName = "__struct_ptr_to_" + elemAddrSpace;
    typeName = structName;
  }

  unsigned physAS = getPhysicalPointerAddressSpace(GEP->getPointerOperand());
  std::string addrSpace = getAddressSpaceKeyword(physAS);

  std::string accessExpr;
  Type *currentType = srcElemType;

  auto idxIt = GEP->idx_begin();
  if (idxIt != GEP->idx_end()) {
    Value *firstIdx = *idxIt;
    std::string castBase = "((" + addrSpace + " " + typeName + "*)" + base + ")";

    if (auto *CI = dyn_cast<ConstantInt>(firstIdx)) {
      accessExpr = castBase + "[" + std::to_string(CI->getSExtValue()) + "]";
    } else {
      accessExpr = castBase + "[" + emitExpr(firstIdx) + "]";
    }
    ++idxIt;
  } else {
    // No indices - just cast
    accessExpr = "((" + addrSpace + " " + typeName + "*)" + base + ")";
  }

  for (; idxIt != GEP->idx_end(); ++idxIt) {
    Value *idx = *idxIt;

    if (auto *ST = dyn_cast<StructType>(currentType)) {
      if (auto *CI = dyn_cast<ConstantInt>(idx)) {
        unsigned fieldIdx = CI->getZExtValue();
        accessExpr += ".field" + std::to_string(fieldIdx);
        currentType = ST->getElementType(fieldIdx);
      } else {
        errorMsg = "Error: Non-constant struct index in GEP\n";
        return;
      }
    } else if (auto *AT = dyn_cast<ArrayType>(currentType)) {
      if (auto *CI = dyn_cast<ConstantInt>(idx)) {
        accessExpr += "[" + std::to_string(CI->getZExtValue()) + "]";
      } else {
        accessExpr += "[" + emitExpr(idx) + "]";
      }
      currentType = AT->getElementType();
    } else {
      // Pointer arithmetic for other types
      if (auto *CI = dyn_cast<ConstantInt>(idx)) {
        if (!CI->isZero()) {
          accessExpr = "(" + accessExpr + " + " + std::to_string(CI->getSExtValue()) + ")";
        }
      } else {
        accessExpr = "(" + accessExpr + " + " + emitExpr(idx) + ")";
      }
    }
  }

  os << indent(level) << name << " = &(" << accessExpr << "); " << "// " << instToString(*GEP) << "\n";
}

bool MetalEmitter::emitCallInstruction(const CallInst* CI, const std::string& name, int level) {
  Function *callee = CI->getCalledFunction();
  if (!callee) {
    errorMsg = "Error: Indirect call not supported: " + instToString(*CI);
    return false;
  }

  std::string calleeName = callee->getName().str();

  if (calleeName.find("__acpp_sscp_metal") == 0) {
    return emitMetalInlineCall(CI, name, level);
  }

  // Change 3: IEEE-correct NaN-propagating fmin/fmax to kill the pg_accel
  // `-ffast-math` workaround. LLVM's llvm.minnum.f32 / llvm.maxnum.f32 are
  // renamed by ReplaceIntrinsics (LLVMToMetal.cpp) to __acpp_sscp_fmin_f32 /
  // __acpp_sscp_fmax_f32. Under -ffast-math, InstCombine may drop the
  // `nnan` contract, and Metal's `metal::fmin` is undefined on NaN in
  // fast-math mode. We emit an explicit NaN-propagating sequence inline:
  //   (isnan(a) ? b : (isnan(b) ? a : metal::fmin(a, b)))
  // For f64 we can't use metal::fmin directly (no native f64 on Metal —
  // see Change 2 for soft-fp64 lowering). Route to a libkernel precise
  // variant that propagates NaN in soft-float.
  if (CI->arg_size() == 2) {
    const char* msl_op = nullptr;
    bool is_f32 = false;
    bool is_f64 = false;
    if (calleeName == "__acpp_sscp_fmin_f32") { msl_op = "metal::fmin"; is_f32 = true; }
    else if (calleeName == "__acpp_sscp_fmax_f32") { msl_op = "metal::fmax"; is_f32 = true; }
    else if (calleeName == "__acpp_sscp_fmin_f64") { is_f64 = true; }
    else if (calleeName == "__acpp_sscp_fmax_f64") { is_f64 = true; }

    if (is_f32) {
      // Defensive NaN-propagating sequence for any __acpp_sscp_fmin_f32 /
      // __acpp_sscp_fmax_f32 call that survives inlining (the libkernel body
      // in math.cpp already supplies IEEE 754-2008 NaN + signed-zero
      // semantics and is always_inline, so in practice this branch is
      // unreachable — kept as a safety net).
      std::string a = emitExpr(CI->getArgOperand(0));
      std::string b = emitExpr(CI->getArgOperand(1));
      os << indent(level) << name << " = ("
         << "isnan(" << a << ") ? (" << b << ") : "
         << "isnan(" << b << ") ? (" << a << ") : " << msl_op << "(" << a
         << ", " << b << ")"
         << "); // " << instToString(*CI) << "\n";
      return true;
    }
    if (is_f64) {
      // TODO(acpp-soft-fp64): libkernel agent provides
      //   __acpp_sscp_soft_f64_fmin_precise / __acpp_sscp_soft_f64_fmax_precise
      // (NaN-propagating soft-float fmin/fmax). Symbol names stable; this
      // call may show as an unresolved extern until that lands.
      const char* soft_sym = (calleeName == "__acpp_sscp_fmin_f64")
                                 ? "__acpp_sscp_soft_f64_fmin_precise"
                                 : "__acpp_sscp_soft_f64_fmax_precise";
      std::string a = emitExpr(CI->getArgOperand(0));
      std::string b = emitExpr(CI->getArgOperand(1));
      os << indent(level) << name << " = " << soft_sym << "(" << a << ", "
         << b << "); // " << instToString(*CI) << "\n";
      return true;
    }
  }

  int argsSize = CI->arg_size();
  if (callee->isDeclaration()) {
    auto returnType = callee->getReturnType();
    if (returnType->isStructTy()) {
      calleeName += "<" + mapType(returnType) + ">";
    }
  }

  std::string callExpr = calleeName + "(";
  for (unsigned i = 0; i < argsSize; ++i) {
    if (i > 0) {
      callExpr += ", ";
    }
    callExpr += emitExpr(CI->getArgOperand(i));
  }
  callExpr += ")";

  if (!CI->getType()->isVoidTy()) {
    os << indent(level) << name << " = " << callExpr << ";\n";
  } else {
    os << indent(level) << callExpr << ";\n";
  }
  return true;
}

std::string MetalEmitter::emitExpr(const Value* V) {
  if (auto *CI = dyn_cast<ConstantInt>(V)) {
    if (CI->getBitWidth() == 1) {
      return CI->isZero() ? "false" : "true";
    }
    uint64_t val = CI->getZExtValue();
    std::ostringstream hex;
    hex << "0x" << std::hex << val << "u";
    if (CI->getBitWidth() == 64) {
      hex << "l";
    }
    return hex.str();
  }

  if (auto *CF = dyn_cast<ConstantFP>(V)) {
    // Change 2: fp64 literal -> acpp_f64 struct initializer. Bit-pattern
    // via the APFloat raw bits so denormals/NaN/Inf round-trip exactly.
    if (CF->getType()->isDoubleTy()) {
      double dv = CF->getValue().convertToDouble();
      uint64_t bits; memcpy(&bits, &dv, sizeof(bits));
      uint32_t lo = static_cast<uint32_t>(bits);
      uint32_t hi = static_cast<uint32_t>(bits >> 32);
      std::ostringstream hex;
      hex << "acpp_f64 { 0x" << std::hex << lo << "u, 0x" << hi << "u }";
      return hex.str();
    }
    float floatVal = CF->getValue().convertToFloat();
    uint32_t val; memcpy(&val, &floatVal, sizeof(float));
    std::ostringstream hex;
    hex << "as_type<float>(" << "0x" << std::hex << val << "u)";
    return hex.str();
  }

  if (isa<ConstantPointerNull>(V)) {
    return "nullptr";
  }

  if (isa<UndefValue>(V) || isa<PoisonValue>(V)) {
    if (V->getType()->isPointerTy()) {
      return "/* undef */ nullptr";
    }
    if (V->getType()->isStructTy()) {
      return "/* undef */ {}";
    }
    // Change 2: soft-double undef/poison. MSL has no implicit int->acpp_f64
    // conversion, so `int` literal zero cannot stand in for a double undef
    // in an fcmp, select, or binary op. Must emit an explicit acpp_f64.
    if (V->getType()->isDoubleTy()) {
      return "/* undef */ acpp_f64 { 0u, 0u }";
    }
    // <N x double> lowered as array<acpp_f64, N> (see mapType). Fill lanes
    // with soft-fp64 zero so brace-init matches element type.
    if (auto *VT = dyn_cast<FixedVectorType>(V->getType())) {
      if (VT->getElementType()->isDoubleTy()) {
        std::string lanes;
        for (unsigned i = 0; i < VT->getNumElements(); ++i) {
          if (i) lanes += ", ";
          lanes += "acpp_f64 { 0u, 0u }";
        }
        return "/* undef */ { " + lanes + " }";
      }
    }
    return "/* undef */ 0";
  }

  // Strip pointer casts (addrspacecast, bitcast)
  if (auto *CE = dyn_cast<ConstantExpr>(V)) {
    if (CE->getOpcode() == Instruction::AddrSpaceCast ||
        CE->getOpcode() == Instruction::BitCast) {
      return emitExpr(CE->getOperand(0));
    }
  }

  // In LLVM IR, globals are pointers, but in MSL we emit them as values.
  // Return the address so that all uses (GEP, load, store) work correctly.
  if (isa<GlobalVariable>(V))
    return "(&" + valueName(V) + ")";

  return valueName(V);
}

std::string MetalEmitter::valueName(const Value* V) {
  std::string s;
  raw_string_ostream rso(s);
  V->printAsOperand(rso, false);
  std::string name = rso.str();
  if (name[0] == '%') {
    name = name.substr(1);
  }
  for (char& c : name) {
    if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '_')) {
      c = '_';
    }
  }
  return "t" + name;
}

std::string MetalEmitter::basicBlockName(const BasicBlock* BB) {
  if (BB->getName().str().empty()) {
    std::string s;
    raw_string_ostream rso(s);
    BB->printAsOperand(rso, false);
    return rso.str();
  } else {
    return "%" + BB->getName().str();
  }
}

std::string MetalEmitter::getSignedType(Type* T) {
  if (T->isIntegerTy()) {
    unsigned bitWidth = T->getIntegerBitWidth();
    if (bitWidth == 1) {
      return "bool";
    } else if (bitWidth == 8) {
      return "char";
    } else if (bitWidth == 16) {
      return "short";
    } else if (bitWidth == 32) {
      return "int";
    } else if (bitWidth == 64) {
      return "long";
    } else if (bitWidth == 128) {
      return "int4";
    } else {
      std::ostringstream ss;
      ss << "Error: Unsupported integer bit width: " << bitWidth << "\n";
      errorMsg = ss.str();
      return "";
    }
  } else {
    errorMsg = "Error: getSignedType called on non-integer type\n";
    return "";
  }
}

std::string MetalEmitter::mapType(const Type* T) {
  auto it = typeCache.find(T);
  if (it != typeCache.end()) {
    return it->second;
  }

  if (T->isPointerTy()) {
    unsigned AS = T->getPointerAddressSpace();
    auto keyword = getAddressSpaceKeyword(AS);

    return typeCache[T] = keyword + " void*";
  }

  if (auto *AT = dyn_cast<ArrayType>(T)) {
    auto elemType = mapType(AT->getElementType());
    return typeCache[T] = "array<" + elemType + ", " + std::to_string(AT->getNumElements()) + ">";
  }

  // Change 2: auto-vectorized fp64 kernels may produce <N x double>. Lower
  // as a fixed-length array so that the element-wise op lowering in the
  // emit*Operator paths can index it directly. Restricted to fp64 element
  // types only; full generic vector lowering is a separate effort.
  // TODO(acpp-vector-lowering): generalize to non-fp64 vectors.
  if (auto *VT = dyn_cast<FixedVectorType>(T)) {
    if (VT->getElementType()->isDoubleTy()) {
      auto elemType = mapType(VT->getElementType());
      return typeCache[T] = "array<" + elemType + ", " +
                            std::to_string(VT->getNumElements()) + ">";
    }
  }

  if (auto *ST = dyn_cast<StructType>(T)) {
    if (ST->hasName()) {
      auto str = ST->getName().str();
      for (char& c : str) {
        if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '_')) {
          c = '_';
        }
      }
      return typeCache[T] = str;
    }

    auto structName = "anon_struct_" + std::to_string(nextAnonymousStructId++);
    anonStructs[T] = structName;
    return typeCache[T] = structName;
  }

  if (T->isVoidTy()) {
    return typeCache[T] = "void";
  } else if (T->isIntegerTy()) {
    unsigned bitWidth = T->getIntegerBitWidth();
    if (bitWidth == 1) {
      return typeCache[T] = "bool";
    } else if (bitWidth == 8) {
      return typeCache[T] = "uchar";
    } else if (bitWidth == 16) {
      return typeCache[T] = "ushort";
    } else if (bitWidth == 32) {
      return typeCache[T] = "uint";
    } else if (bitWidth == 48) {
      return typeCache[T] = "i48u";
    } else if (bitWidth == 64) {
      return typeCache[T] = "ulong";
    } else if (bitWidth == 128) {
      return typeCache[T] = "uint4";
    } else {
      std::ostringstream ss;
      ss << "Error: Unsupported integer bit width: " << bitWidth << "\n";
      errorMsg = ss.str();
      return "";
    }
  } else if (T->isFloatTy()) {
    return typeCache[T] = "float";
  } else if (T->isDoubleTy()) {
    // Change 2: soft-double lowering. See acpp_f64 helper at emitIntrinsicHelpers.
    return typeCache[T] = "acpp_f64";
  } else if (T->isHalfTy()) {
    return typeCache[T] = "half";
  } else {
    std::string typeStr;
    llvm::raw_string_ostream rso(typeStr);
    T->print(rso);
    rso.flush();
    errorMsg = "Error: Unsupported type encountered : " + typeStr;
    return "";
  }
}

std::string MetalEmitter::mapType(const llvm::Value* V) {
  if (V->getType()->isPointerTy()) {
    unsigned AS = getPhysicalPointerAddressSpace(V);
    auto keyword = getAddressSpaceKeyword(AS);
    return keyword + " void*";
  } else {
    return mapType(V->getType());
  }
}

std::string MetalEmitter::getAddressSpaceKeyword(unsigned AS) {
  auto it = addressSpaceMap.find(AS);
  if (it != addressSpaceMap.end()) {
    return it->second;
  } else {
    errorMsg = "Error: Unknown address space " + std::to_string(AS) + "\n";
    return "";
  }
}

const Value* MetalEmitter::stripToRootObject(const Value* V) {
  const Value* Cur = V;
  while (true) {
    if (auto* ASC = dyn_cast<AddrSpaceCastInst>(Cur)) {
      Cur = ASC->getPointerOperand();
      continue;
    }
    if (auto* BC = dyn_cast<BitCastInst>(Cur)) {
      Cur = BC->getOperand(0);
      continue;
    }
    if (auto* GEP = dyn_cast<GetElementPtrInst>(Cur)) {
      // Accept only "trivial" GEPs (all indices are zero) for minimal patch
      bool allZero = true;
      for (auto it = GEP->idx_begin(); it != GEP->idx_end(); ++it) {
        if (auto* CI = dyn_cast<ConstantInt>(*it)) {
          if (!CI->isZero()) { allZero = false; break; }
        } else {
          allZero = false;
          break;
        }
      }
      if (allZero) {
        Cur = GEP->getPointerOperand();
        continue;
      }
    }
    break;
  }
  return Cur;
}

unsigned MetalEmitter::getPhysicalPointerAddressSpace(const Value* V) {
  Type* T = V->getType();
  if (!T->isPointerTy()) {
    return 0; // not a pointer
  }
  unsigned AS = T->getPointerAddressSpace();
  if (AS != 0) {
    return AS;
  }
  auto it = inferredPtrAS.find(V);
  if (it != inferredPtrAS.end()) {
    return it->second;
  }

  inferredPtrAS[V] = 0; // to break cycles in PHIs and Loads

  if (auto* Alloca = dyn_cast<AllocaInst>(V)) {
    return (inferredPtrAS[V] = 5 /* private */);
  }
  if (auto* ASC = dyn_cast<AddrSpaceCastInst>(V)) {
    return (inferredPtrAS[V] = getPhysicalPointerAddressSpace(ASC->getOperand(0)));
  }
  if (auto* GEP = dyn_cast<GetElementPtrInst>(V)) {
    return (inferredPtrAS[V] = getPhysicalPointerAddressSpace(GEP->getPointerOperand()));
  }
  if (auto* PHI = dyn_cast<PHINode>(V)) {
    unsigned deducedAS = 0;
    for (unsigned i = 0; i < PHI->getNumIncomingValues(); ++i) {
      Value* incomingV = PHI->getIncomingValue(i);
      unsigned incomingAS = getPhysicalPointerAddressSpace(incomingV);
      if (incomingAS != 0) {
        if (deducedAS == 0) {
          deducedAS = incomingAS;
        } else if (deducedAS != incomingAS) {
          errorMsg = "Error: Conflicting address space information in PHI node\n";
          return 0;
        }
      }
    }
    if (deducedAS != 0) {
      return (inferredPtrAS[V] = deducedAS);
    }
  }
  if (auto* Select = dyn_cast<SelectInst>(V)) {
    unsigned trueAS = getPhysicalPointerAddressSpace(Select->getTrueValue());
    unsigned falseAS = getPhysicalPointerAddressSpace(Select->getFalseValue());
    if (trueAS != 0 && trueAS == falseAS) {
      return (inferredPtrAS[V] = trueAS);
    }
  }
  if (auto* LI = dyn_cast<LoadInst>(V)) {
    const Value* addr = LI->getPointerOperand();       // address of the slot
    const Value* root = stripToRootObject(addr);       // e.g. %.sroa.xxx.i alloca in AS5

    unsigned deducedAS = 0;

    // Scan stores that write into the same root object
    for (const User* U : root->users()) {
      const StoreInst* SI = dyn_cast<StoreInst>(U);
      if (!SI) continue;

      // Does this store target the same root (allow casts/trivial GEP)?
      const Value* stRoot = stripToRootObject(SI->getPointerOperand());
      if (stRoot != root) continue;

      unsigned storedAS = getPhysicalPointerAddressSpace(SI->getValueOperand());
      if (storedAS == 0) continue;

      if (deducedAS == 0) deducedAS = storedAS;
      else if (deducedAS != storedAS) {
        // Conflicting stores into same slot
        return 0;
      }
    }

    if (deducedAS != 0) {
      return (inferredPtrAS[V] = deducedAS);
    }
  }

  return 0;
}

void MetalEmitter::analyzeCallInsts() {
  for (auto& F : M) {
    if (F.isDeclaration()) {
      continue;
    }

    for (BasicBlock& BB : F) {
      for (Instruction& I : BB) {
        mapType(I.getType()); // for cache all types and emit anon structs declarations
        auto *CI = dyn_cast<CallInst>(&I);
        if (!CI) {
          continue;
        }
        Function *Callee = CI->getCalledFunction();
        if (!Callee) {
          continue;
        }
        // add return type and args to cache
        auto returnType = Callee->getReturnType();
        mapType(returnType);
        for (auto& Arg : Callee->args()) {
          mapType(Arg.getType());
        }
      }
    }
  }
}

// Change 1: scan the module for i64 storage that backs an atomic op
// (LLVM atomic instructions OR libkernel `__acpp_sscp_atomic_*_{i64,u64}`
// helpers — the SYCL atomic_ref<uint64_t> surface lowers to the helper
// calls, not to atomicrmw). For every such pointer operand, walk back via
// stripToRootObject and mark each SSA value along the way as atomic-i64 so
// that mapType emits `atomic_ulong` for the storage declaration.
void MetalEmitter::analyzeAtomicI64Storage() {
  auto markChain = [this](const llvm::Value* V) {
    // Walk bitcasts / addrspacecasts / trivial GEPs back to the root object,
    // marking every intermediate SSA value so valuesToDeclare picks up the
    // atomic type when the value is declared. stripToRootObject already
    // handles this traversal; we replicate its walk here so we can mark
    // every step, not just the final root.
    const llvm::Value* Cur = V;
    while (true) {
      atomicI64Values.insert(Cur);
      if (auto* ASC = llvm::dyn_cast<llvm::AddrSpaceCastInst>(Cur)) {
        Cur = ASC->getPointerOperand();
        continue;
      }
      if (auto* BC = llvm::dyn_cast<llvm::BitCastInst>(Cur)) {
        Cur = BC->getOperand(0);
        continue;
      }
      if (auto* GEP = llvm::dyn_cast<llvm::GetElementPtrInst>(Cur)) {
        bool allZero = true;
        for (auto it = GEP->idx_begin(); it != GEP->idx_end(); ++it) {
          if (auto* CI = llvm::dyn_cast<llvm::ConstantInt>(*it)) {
            if (!CI->isZero()) { allZero = false; break; }
          } else {
            allZero = false;
            break;
          }
        }
        if (allZero) {
          Cur = GEP->getPointerOperand();
          continue;
        }
      }
      break;
    }
  };

  auto isI64 = [](llvm::Type* T) {
    return T->isIntegerTy() && T->getIntegerBitWidth() == 64;
  };

  for (auto& F : M) {
    if (F.isDeclaration()) continue;
    for (auto& BB : F) {
      for (auto& I : BB) {
        // Native LLVM atomic ops.
        if (auto* RMW = llvm::dyn_cast<llvm::AtomicRMWInst>(&I)) {
          if (isI64(RMW->getValOperand()->getType())) {
            markChain(RMW->getPointerOperand());
          }
          continue;
        }
        if (auto* CX = llvm::dyn_cast<llvm::AtomicCmpXchgInst>(&I)) {
          if (isI64(CX->getNewValOperand()->getType())) {
            markChain(CX->getPointerOperand());
          }
          continue;
        }
        if (auto* LI = llvm::dyn_cast<llvm::LoadInst>(&I)) {
          if (LI->isAtomic() && isI64(LI->getType())) {
            markChain(LI->getPointerOperand());
          }
          continue;
        }
        if (auto* SI = llvm::dyn_cast<llvm::StoreInst>(&I)) {
          if (SI->isAtomic() && isI64(SI->getValueOperand()->getType())) {
            markChain(SI->getPointerOperand());
          }
          continue;
        }
        // SYCL atomic surface: libkernel helpers.
        // Naming convention (see src/libkernel/sscp/*/atomic.cpp):
        //   __acpp_sscp_atomic_<op>_i64  / _u64
        //   __acpp_sscp_cmp_exch_{weak,strong}_i64 / _u64
        //   __acpp_sscp_atomic_fetch_<op>_i64 / _u64
        if (auto* CI = llvm::dyn_cast<llvm::CallInst>(&I)) {
          llvm::Function* Callee = CI->getCalledFunction();
          if (!Callee) continue;
          llvm::StringRef Name = Callee->getName();
          bool isAtomicHelper =
              (Name.starts_with("__acpp_sscp_atomic_") ||
               Name.starts_with("__acpp_sscp_cmp_exch_"));
          if (!isAtomicHelper) continue;
          bool i64Suffix = Name.ends_with("_i64") || Name.ends_with("_u64");
          if (!i64Suffix) continue;
          // The pointer argument is conventionally the first `T*` argument.
          // Mark every pointer-typed argument to be safe (cmpxchg has two).
          for (unsigned i = 0; i < CI->arg_size(); ++i) {
            llvm::Value* Arg = CI->getArgOperand(i);
            if (Arg->getType()->isPointerTy()) {
              markChain(Arg);
            }
          }
        }
      }
    }
  }
}

void MetalEmitter::collectVariablesInfo(const Function& F) {
  allocaIndex.clear();
  phiSources.clear();
  sourceToPhis.clear();
  phiIncomingFromBlock.clear();
  phiNodes.clear();
  valuesToDeclare.clear();

  for (const BasicBlock& BB : F) {
    for (const Instruction& I : BB) {
      if (auto* AI = dyn_cast<AllocaInst>(&I)) {
        int index = static_cast<int>(allocaIndex.size());
        allocaIndex[&I] = index;
      }

      if (auto* PHI = dyn_cast<PHINode>(&I)) {
        phiNodes.insert(PHI);
        for (unsigned i = 0; i < PHI->getNumIncomingValues(); ++i) {
          Value* V = PHI->getIncomingValue(i);
          BasicBlock* incomingBB = PHI->getIncomingBlock(i);

          phiSources[PHI].push_back(V);
          phiIncomingFromBlock[{incomingBB, PHI}] = V;

          if (!isa<Constant>(V)) {
            sourceToPhis[V].push_back(PHI);
          }
        }
      }

      if (!I.getType()->isVoidTy()) {
        if (auto* ASC = dyn_cast<AddrSpaceCastInst>(&I)) {
          valuesToDeclare[&I] = mapType(ASC->getOperand(0));
        } else if (auto* GEP = dyn_cast<GetElementPtrInst>(&I)) {
          valuesToDeclare[&I] = mapType(GEP->getPointerOperand());
        } else {
          valuesToDeclare[&I] = mapType(&I);
        }
      }
    }
  }
}

std::unordered_map<Function*, std::vector<Function*>> MetalEmitter::buildCallGraph() {
  std::unordered_map<Function*, std::vector<Function*>> callGraph;

  for (Function& F : M) {
    if (F.isDeclaration()) {
      continue;
    }

    for (BasicBlock& BB : F) {
      for (Instruction& I : BB) {
        auto* CI = dyn_cast<CallInst>(&I);
        if (!CI) {
          continue;
        }

        Function* Callee = CI->getCalledFunction();
        if (!Callee || Callee->isDeclaration()) {
          continue;
        }

        callGraph[&F].push_back(Callee);
      }
    }
  }

  return callGraph;
}

std::vector<Function*> MetalEmitter::topologicalSort(const std::unordered_map<Function*, std::vector<Function*>>& callGraph) {
  std::vector<Function*> result;
  std::unordered_set<Function*> visited;
  std::unordered_set<Function*> inStack;

  std::function<bool(Function*)> visit = [&](Function* F) -> bool {
    if (inStack.count(F)) {
      return true;
    }
    if (visited.count(F)) {
      return true;
    }

    inStack.insert(F);

    auto it = callGraph.find(F);
    if (it != callGraph.end()) {
      for (Function* Callee : it->second) {
        if (!visit(Callee)) {
          return false;
        }
      }
    }

    inStack.erase(F);
    visited.insert(F);
    result.push_back(F);

    return true;
  };

  for (Function& F : M) {
    if (F.isDeclaration()) {
      continue;
    }
    visit(&F);
  }

  return result;
}

bool MetalEmitter::emitMetalInlineCall(const llvm::CallInst* CI, const std::string& name, int level) {
  std::string errorStr;
  auto ctx = initEmitContext(CI, errorStr);
  if (!ctx) {
    errorMsg = errorStr;
    return false;
  }

  if (ctx->name.find("__acpp_sscp_metal") == std::string::npos) {
    errorMsg = "Not a metal function";
    return false;
  }

  bool is_symbol = ctx->name.find("__acpp_sscp_metal_symbol") == 0;

  if (CI->arg_size() < 1) {
    errorMsg = "__acpp_sscp_metal: expected at least 1 argument (function name / constant)";
    return false;
  }

  if (is_symbol && CI->arg_size() != 1) {
    errorMsg = "__acpp_sscp_metal_symbol: expected at least 1 arguments (symbol name constant)";
    return false;
  }

  // Extract first argument as string constant (function name)
  auto funcName = extractStringConstant(CI->getArgOperand(0), errorStr);
  if (!funcName) {
    errorMsg = "__acpp_sscp_metal: " + errorStr;
    return false;
  }

  if (is_symbol) {
    if (!CI->getType()->isVoidTy()) {
      os << indent(level) << name << " = " << *funcName << ";\n";
    } else {
      os << indent(level) << *funcName << ";\n";
    }
    return true;
  }

  std::string result;
  if (funcName->find("%s") != std::string::npos) {
    // expand printf-style format string with arguments
    size_t pos = 0;
    int arg = 1;
    while (pos != std::string::npos && arg < CI->arg_size()) {
      auto next = funcName->find("%s", pos);
      result += funcName->substr(pos, next == std::string::npos ? std::string::npos : next - pos);
      llvm::Value* argValue = CI->getArgOperand(arg++);
      result += emitExpr(argValue);
      pos = next == std::string::npos ? std::string::npos : next + 2;
    }
    result += funcName->substr(pos);
  } else {
    result = *funcName + "(";

    for (unsigned i = 1; i < CI->arg_size(); ++i) {
      if (i > 1) {
        result += ", ";
      }
      llvm::Value* arg = CI->getArgOperand(i);
      result += emitExpr(arg);
    }

    result += ")";
  }

  if (!CI->getType()->isVoidTy()) {
    os << indent(level) << name << " = " << result << ";\n";
  } else {
    os << indent(level) << result << ";\n";
  }
  return true;
}

} // namespace compiler
} // namespace hipsycl
