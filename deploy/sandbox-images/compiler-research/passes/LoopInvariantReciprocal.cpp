//===- LoopInvariantReciprocal.cpp ----------------------------------------===//
//
// Hoist 1/d out of a loop when d does not change across iterations, turning
// every division by it into a multiply.
//
// This exists because a scan said so. OptScan over raylib found 419 divisions
// inside loops, 179 of them (42.7%) with a loop-invariant denominator --
// against 25 sites for the block-local reciprocal pass, so seven times the
// reach. Whole functions are wholly hoistable: ma_pcm_f32_to_s16 at 42 of 42,
// m3d_load at 17 of 17, ImageFormat at 14 of 14. The opportunity is real and
// untaken: clang -O2 leaves the fdiv in the loop and only -ffast-math hoists
// the reciprocal.
//
// The threshold differs from the block-local pass, and the reason is the whole
// point. There, two divisions are needed before a reciprocal pays for itself,
// because the reciprocal is a division too and sits in the same block. Here
// the reciprocal leaves the loop entirely, so it is amortised over every
// iteration and a single division is already worth it.
//
// NOT value-preserving, the same way and for the same reason as the
// block-local pass: x/d rounds once, x*(1/d) rounds twice. Measured on a
// normalise kernel, that trade moves 38.18% of results by up to 3 ulp.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {

struct LoopInvariantReciprocal : PassInfoMixin<LoopInvariantReciprocal> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &FAM) {
    LoopInfo &LI = FAM.getResult<LoopAnalysis>(F);
    unsigned Hoisted = 0, Replaced = 0;

    // Innermost first, so a division is claimed by the tightest loop whose
    // denominator is invariant -- hoisting to an outer loop it also happens to
    // be invariant in would run the reciprocal fewer times but is not what the
    // division's own loop asked for.
    SmallVector<Loop *, 8> Loops(LI.begin(), LI.end());
    for (unsigned i = 0; i < Loops.size(); ++i)
      Loops.append(Loops[i]->begin(), Loops[i]->end());

    for (Loop *L : reverse(Loops)) {
      BasicBlock *Pre = L->getLoopPreheader();
      if (!Pre)
        continue; // nowhere safe to put the reciprocal

      DenseMap<Value *, SmallVector<BinaryOperator *, 4>> ByDenominator;
      for (BasicBlock *BB : L->blocks()) {
        if (LI.getLoopFor(BB) != L)
          continue; // belongs to a nested loop, which was handled first
        for (Instruction &I : *BB) {
          auto *Div = dyn_cast<BinaryOperator>(&I);
          if (!Div || Div->getOpcode() != Instruction::FDiv)
            continue;
          Value *D = Div->getOperand(1);
          if (isa<Constant>(D))
            continue; // already folded to a multiply
          if (!L->isLoopInvariant(D))
            continue;
          if (L->isLoopInvariant(Div->getOperand(0)))
            continue; // the whole division is invariant; LICM hoists it
          ByDenominator[D].push_back(Div);
        }
      }

      for (auto &Entry : ByDenominator) {
        IRBuilder<> B(Pre->getTerminator());
        Value *One = ConstantFP::get(Entry.second.front()->getType(), 1.0);
        Value *Recip = B.CreateFDiv(One, Entry.first, "inv.recip");
        if (auto *RI = dyn_cast<Instruction>(Recip))
          RI->copyFastMathFlags(Entry.second.front());
        ++Hoisted;
        for (BinaryOperator *Div : Entry.second) {
          IRBuilder<> M(Div);
          Value *Mul = M.CreateFMul(Div->getOperand(0), Recip);
          if (auto *MI = dyn_cast<Instruction>(Mul))
            MI->copyFastMathFlags(Div);
          Div->replaceAllUsesWith(Mul);
          Div->eraseFromParent();
          ++Replaced;
        }
      }
    }

    if (Hoisted)
      errs() << "loop-invariant-reciprocal: " << F.getName() << ": hoisted "
             << Hoisted << " reciprocal(s), replaced " << Replaced
             << " division(s)\n";

    return Hoisted ? PreservedAnalyses::none() : PreservedAnalyses::all();
  }
};

} // namespace

extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "LoopInvariantReciprocal", "0.1",
          [](PassBuilder &PB) {
            PB.registerScalarOptimizerLateEPCallback(
                [](FunctionPassManager &FPM, OptimizationLevel Level) {
                  if (Level != OptimizationLevel::O0)
                    FPM.addPass(LoopInvariantReciprocal());
                });
            PB.registerPipelineParsingCallback(
                [](StringRef Name, FunctionPassManager &FPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name == "loop-invariant-reciprocal") {
                    FPM.addPass(LoopInvariantReciprocal());
                    return true;
                  }
                  return false;
                });
          }};
}
