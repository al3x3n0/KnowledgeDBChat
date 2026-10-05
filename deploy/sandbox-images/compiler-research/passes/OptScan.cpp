//===- OptScan.cpp --------------------------------------------------------===//
//
// Report optimisation opportunities in a module without changing it. Two jobs,
// deliberately separate:
//
//   KNOWN  -- sites an existing pass here can already exploit, so the question
//             is only how many and where.
//   SHAPE  -- the operand shape of every expensive operation, tallied. A shape
//             that is frequent and has no pass is a candidate for one; this is
//             the front end of writing the next pass rather than guessing at
//             what it should match.
//
// WHAT THIS DOES NOT MEASURE. Every count here is static: how often a pattern
// is *written*, not how often it *runs*. That distinction has already produced
// a wrong answer in this project -- candidates were ranked by source
// occurrence and the ranking inverted once dynamic counts replaced it. A
// scanner output is therefore a list of places to look, not a ranking, and it
// labels itself so at the top. Ranking needs a profile.
//
// Output is one tab-separated record per line, prefixed OPTSCAN, so the
// aggregation and the ranking live in Python where they are easy to change.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {

/// Mirrors SqrtErrnoElision's analysis. Kept in step by being the same shape;
/// if they drift, the scanner reports opportunities the pass will decline.
static bool provablyNonNegative(const Value *V, unsigned Depth = 0) {
  if (Depth > 6)
    return false;
  if (const auto *C = dyn_cast<ConstantFP>(V))
    return !C->getValueAPF().isNegative() || C->getValueAPF().isZero();
  const auto *I = dyn_cast<Instruction>(V);
  if (!I)
    return false;
  switch (I->getOpcode()) {
  case Instruction::FMul:
    return I->getOperand(0) == I->getOperand(1);
  case Instruction::FAdd:
    return provablyNonNegative(I->getOperand(0), Depth + 1) &&
           provablyNonNegative(I->getOperand(1), Depth + 1);
  case Instruction::Call: {
    const auto *CI = cast<CallInst>(I);
    switch (CI->getIntrinsicID()) {
    case Intrinsic::fmuladd:
    case Intrinsic::fma:
      return CI->getArgOperand(0) == CI->getArgOperand(1) &&
             provablyNonNegative(CI->getArgOperand(2), Depth + 1);
    case Intrinsic::sqrt:
    case Intrinsic::fabs:
      return true;
    default:
      return false;
    }
  }
  default:
    return false;
  }
}

static bool isLibmSqrt(const CallInst *CI) {
  const Function *F = CI->getCalledFunction();
  if (!F || !F->isDeclaration() ||
      CI->getIntrinsicID() != Intrinsic::not_intrinsic)
    return false;
  StringRef N = F->getName();
  return (N == "sqrt" || N == "sqrtf") && CI->arg_size() == 1;
}

/// A one-level description of where an operand came from, which is what a
/// future pass would have to match on.
static std::string shapeOf(const Value *V) {
  if (isa<ConstantFP>(V))
    return "const";
  if (isa<Argument>(V))
    return "arg";
  const auto *I = dyn_cast<Instruction>(V);
  if (!I)
    return "other";
  if (const auto *CI = dyn_cast<CallInst>(I)) {
    if (CI->getIntrinsicID() != Intrinsic::not_intrinsic)
      return ("intrinsic." + Intrinsic::getBaseName(CI->getIntrinsicID())).str();
    return "call";
  }
  return I->getOpcodeName();
}

struct OptScan : PassInfoMixin<OptScan> {
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM) {
    auto &FAM =
        MAM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
    errs() << "OPTSCAN\tmeta\tcounts_are_static\t"
              "how often a pattern is written, not how often it runs\n";

    for (Function &F : M) {
      if (F.isDeclaration())
        continue;

      // KNOWN: sqrt whose errno path is dead.
      unsigned SqrtElidable = 0, SqrtNot = 0;
      // KNOWN: divisions grouped by denominator, per block.
      DenseMap<unsigned, unsigned> DivGroupSizes; // size -> how many groups
      // SHAPE: what feeds each expensive operation.
      StringMap<unsigned> DivNumerator, SqrtArg;
      // A division inside a loop whose denominator does not change across
      // iterations: one hoisted reciprocal could replace all of them. The
      // block-local grouping above cannot see these, so this is the measure
      // that decides whether a loop-invariant-divisor pass is worth writing.
      unsigned HoistableDiv = 0, LoopDiv = 0;
      LoopInfo &LI = FAM.getResult<LoopAnalysis>(F);

      for (BasicBlock &BB : F) {
        DenseMap<const Value *, unsigned> ByDenominator;
        for (Instruction &I : BB) {
          if (auto *CI = dyn_cast<CallInst>(&I)) {
            if (isLibmSqrt(CI)) {
              (provablyNonNegative(CI->getArgOperand(0)) ? SqrtElidable : SqrtNot)++;
              SqrtArg[shapeOf(CI->getArgOperand(0))]++;
            }
            continue;
          }
          auto *BO = dyn_cast<BinaryOperator>(&I);
          if (!BO || BO->getOpcode() != Instruction::FDiv)
            continue;
          DivNumerator[shapeOf(BO->getOperand(0))]++;
          if (!isa<Constant>(BO->getOperand(1)))
            ByDenominator[BO->getOperand(1)]++;
          if (const Loop *L = LI.getLoopFor(&BB)) {
            ++LoopDiv;
            if (L->isLoopInvariant(BO->getOperand(1)) &&
                !L->isLoopInvariant(BO->getOperand(0)))
              ++HoistableDiv;
          }
        }
        for (auto &E : ByDenominator)
          if (E.second >= 2)
            DivGroupSizes[E.second]++;
      }

      if (SqrtElidable)
        errs() << "OPTSCAN\tknown\tsqrt-errno-elision\t" << F.getName() << "\t"
               << SqrtElidable << "\n";
      if (SqrtNot)
        errs() << "OPTSCAN\tdeclined\tsqrt-errno-elision\t" << F.getName()
               << "\t" << SqrtNot << "\n";
      for (auto &E : DivGroupSizes)
        errs() << "OPTSCAN\tknown\tcommon-divisor-reciprocal\t" << F.getName()
               << "\t" << E.second << "\tgroup_size=" << E.first << "\n";
      if (LoopDiv)
        errs() << "OPTSCAN\tcandidate\tloop-invariant-divisor\t" << F.getName()
               << "\t" << HoistableDiv << "\tof_loop_divisions=" << LoopDiv
               << "\n";
      for (auto &E : DivNumerator)
        errs() << "OPTSCAN\tshape\tfdiv.numerator\t" << E.first() << "\t"
               << E.second << "\n";
      for (auto &E : SqrtArg)
        errs() << "OPTSCAN\tshape\tsqrt.argument\t" << E.first() << "\t"
               << E.second << "\n";
    }
    return PreservedAnalyses::all();
  }
};

} // namespace

extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "OptScan", "0.1", [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, ModulePassManager &MPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name == "opt-scan") {
                    MPM.addPass(OptScan());
                    return true;
                  }
                  return false;
                });
          }};
}
