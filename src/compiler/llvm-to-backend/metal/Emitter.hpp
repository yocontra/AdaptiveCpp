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
#ifndef EMITTER_HPP
#define EMITTER_HPP

#include <string>
#include <unordered_set>
#include <sstream>
#include <map>

#include "HLTree.hpp"

#include <memory>

namespace llvm {
class Module;
class ModuleSlotTracker;
class Function;
class Value;
class Constant;
class BasicBlock;
class Type;
} // namespace llvm

namespace hipsycl {
namespace compiler {

struct MetalEmitterOptions {
  // Metal supports at most 31 [[buffer(N)]] arguments in flat mode.
  // When a kernel has more than maxArgsForFlatMode parameters, all arguments
  // are packed into a single argument buffer struct instead.
  // Controlled at runtime via kernel_build_option::metal_max_args_for_flat_mode.
  int maxArgsForFlatMode = 6;
  std::unordered_map<unsigned, std::string> addressSpaceMap = {
    {0, "device"},
    {1, "device"},
    {3, "threadgroup"},
    {4, "constant"},
    {5, "thread"},
  };
};

class MetalEmitter {
public:
  MetalEmitter(llvm::Module& M, const std::unordered_set<std::string>& kernelNames, const MetalEmitterOptions& opt = {});
  ~MetalEmitter();
  bool emit(std::string& out);
  std::optional<std::string> errorMessage() const {
    return errorMsg;
  }

private:
  bool emitFunction(llvm::Function& F, const hl::Node& node);
  void emitEarlyFp64Helpers();
  void emitTypes();
  void emitIntrinsicHelpers();
  void emitGlobalConstants();
  std::string emitConstantInitializer(const llvm::Constant* C);
  bool emitArgStruct(llvm::Function& F);
  bool emitSignature(llvm::Function& F);
  bool emitDeclarations();
  bool emitNode(const hl::Node& node, int level);
  bool emitBasicBlock(const llvm::BasicBlock* BB, int level);
  bool emitInstruction(const llvm::Instruction& I, int level);
  void emitUnaryOperator(const llvm::UnaryOperator* UO, const std::string& name, int level);
  void emitBinaryOperator(const llvm::BinaryOperator* BO, const std::string& name, int level);
  bool emitCastInstruction(const llvm::CastInst* CI, const std::string& name, int level);
  void emitICmpInstruction(const llvm::ICmpInst* IC, const std::string& name, int level);
  void emitFCmpInstruction(const llvm::FCmpInst* FC, const std::string& name, int level);
  void emitGEPInstruction(const llvm::GetElementPtrInst* GEP, const std::string& name, int level);
  bool emitCallInstruction(const llvm::CallInst* CI, const std::string& name, int level);
  std::string emitExpr(const llvm::Value* V);
  std::string valueName(const llvm::Value* V);
  std::string basicBlockName(const llvm::BasicBlock* BB);
  std::string getSignedType(llvm::Type* T);
  std::string mapType(const llvm::Type* T);
  std::string mapType(const llvm::Value* V);
  std::string getAddressSpaceKeyword(unsigned AS);
  void analyzeCallInsts();
  void analyzeAtomicI64Storage();
  void collectVariablesInfo(const llvm::Function& F);
  unsigned getPhysicalPointerAddressSpace(const llvm::Value* V);
  const llvm::Value* stripToRootObject(const llvm::Value* V);
  std::unordered_map<llvm::Function*, std::vector<llvm::Function*>> buildCallGraph();
  std::vector<llvm::Function*> topologicalSort(const std::unordered_map<llvm::Function*, std::vector<llvm::Function*>>& callGraph);
  bool emitMetalInlineCall(const llvm::CallInst* CI, const std::string& name, int level);

  MetalEmitterOptions opt;

  llvm::Module& M;
  std::unordered_set<std::string> kernelNames;

  // types and per-module info
  std::unordered_map<unsigned, std::string> addressSpaceMap;
  std::unordered_map<const llvm::Type*, std::string> typeCache;
  std::unordered_map<const llvm::Type*, std::string> anonStructs;
  int nextAnonymousStructId = 0;

  // local variables info
  std::unordered_map<const llvm::Instruction*, int> allocaIndex;
  std::unordered_map<const llvm::PHINode*, std::vector<const llvm::Value*>> phiSources;
  std::unordered_map<const llvm::Value*, std::vector<const llvm::PHINode*>> sourceToPhis;
  std::map<std::pair<const llvm::BasicBlock*, const llvm::PHINode*>, const llvm::Value*> phiIncomingFromBlock;
  std::unordered_set<const llvm::PHINode*> phiNodes;
  std::unordered_map<const llvm::Value*, std::string> valuesToDeclare;
  //
  std::unordered_map<const llvm::Value*, unsigned> inferredPtrAS;

  // Change 1: set of SSA values whose i64 storage backs an atomic_ref<uint64_t>
  // (or equivalent SYCL atomic i64 op). mapType(Value*) consults this to emit
  // `atomic_ulong` instead of `ulong` for the storage declaration so the
  // `__atomic_pointer_cast<long>` pointer reinterpret lands on MSL
  // `atomic<long>`-compatible storage.
  std::unordered_set<const llvm::Value*> atomicI64Values;

  // Cache for valueName(). With the per-function ModuleSlotTracker below,
  // each first-time slot lookup is O(1) amortised - but identical-Value
  // queries from emit/declare/HL-tree paths still benefit from skipping
  // the printAsOperand + string-cleaning round trip. Cleared at the
  // start of each emitFunction.
  std::unordered_map<const llvm::Value*, std::string> valueNameCache;

  // Per-function slot tracker reused by valueName(). Without this,
  // llvm::Value::printAsOperand constructs a fresh SlotTracker on every
  // call that walks the whole enclosing Function to assign slot numbers
  // - O(N^2) over a single function emit, which on soft-fp64's larger
  // helper bodies (sf64_fcmp ~400 IR lines, sf64_add ~520) chews CPU
  // for minutes per kernel JIT. Recreated in collectVariablesInfo for
  // the function being emitted; the unique_ptr breaks the include
  // dependency on ModuleSlotTracker.h from this header.
  std::unique_ptr<llvm::ModuleSlotTracker> currentSlotTracker;

  int inputStructCounter = 0;
  std::string inputStructName;

  std::ostringstream os;
  std::optional<std::string> errorMsg;
};

} // namespace compiler
} // namespace hipsycl

#endif // EMITTER_HPP