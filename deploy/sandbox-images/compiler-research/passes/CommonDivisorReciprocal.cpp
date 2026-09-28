//===- CommonDivisorReciprocal.cpp ----------------------------------------===//
//
// Replace N divisions by a shared denominator with one reciprocal and N
// multiplies: x/d, y/d, z/d becomes r = 1/d; x*r, y*r, z*r.
//
// This is NOT value-preserving, and that is the point of measuring it. x/d
// rounds once; x*(1/d) rounds twice, so results differ by an ulp on many
// inputs. -ffast-math permits the substitution, but -ffast-math permits a
// dozen other things at the same time -- reassociation, finite-math
// assumptions, dropping errno -- so measuring that flag prices the bundle and
// not the transformation. This isolates one member of it.
//
// Why it should pay. On the gem5 ARM O3 default the FP divider is two
// unpipelined units at opLat 12, giving one divide per six cycles, while
// FloatMult is opLat 4 and pipelined. Three divides per element is eighteen
// cycles of divider occupancy in a normalise loop measured at 31.8
// cycles/element -- 57% of it. Trading two of the three for pipelined
// multiplies should therefore be worth roughly a third of the loop, and unlike
// the vectorisation route this stays scalar, so this gem5 configuration prices
// it correctly.
//
// The threshold is an empirical question, not a guess: at one division the
// trade is a loss (a divide plus a multiply costs more than a divide), at two
// it is near break-even, at three it should win. The pass fires at two or more
// and the study varies the kernel rather than the pass.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
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

/// Fewer than this many divisions by the same value and the reciprocal is not
/// worth its own instruction.
static const unsigned MinDivisions = 2;

struct CommonDivisorReciprocal : PassInfoMixin<CommonDivisorReciprocal> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &) {
    unsigned Groups = 0, Replaced = 0;

    for (BasicBlock &BB : F) {
      // Group by denominator, within a block so dominance is free.
      DenseMap<Value *, SmallVector<BinaryOperator *, 4>> ByDenominator;
      for (Instruction &I : BB) {
        auto *Div = dyn_cast<BinaryOperator>(&I);
        if (!Div || Div->getOpcode() != Instruction::FDiv)
          continue;
        // A constant denominator is already folded to a multiply by InstCombine.
        if (isa<Constant>(Div->getOperand(1)))
          continue;
        ByDenominator[Div->getOperand(1)].push_back(Div);
      }

      for (auto &Entry : ByDenominator) {
        auto &Divs = Entry.second;
        if (Divs.size() < MinDivisions)
          continue;

        // The reciprocal goes at the first division, which every other use in
        // this block follows and where the denominator is certainly defined.
        IRBuilder<> B(Divs.front());
        Value *One = ConstantFP::get(Divs.front()->getType(), 1.0);
        Value *Recip = B.CreateFDiv(One, Entry.first, "recip");
        if (auto *RI = dyn_cast<Instruction>(Recip))
          RI->copyFastMathFlags(Divs.front());

        for (BinaryOperator *Div : Divs) {
          IRBuilder<> M(Div);
          Value *Mul = M.CreateFMul(Div->getOperand(0), Recip);
          if (auto *MI = dyn_cast<Instruction>(Mul))
            MI->copyFastMathFlags(Div);
          Div->replaceAllUsesWith(Mul);
          Div->eraseFromParent();
          ++Replaced;
        }
        ++Groups;
      }
    }

    if (Groups)
      errs() << "common-divisor-reciprocal: " << F.getName() << ": " << Groups
             << " group(s), " << Replaced << " divisions -> " << Groups
             << " reciprocal(s) + " << Replaced << " multiplies\n";

    return Groups ? PreservedAnalyses::none() : PreservedAnalyses::all();
  }
};

} // namespace

extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "CommonDivisorReciprocal", "0.1",
          [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, FunctionPassManager &FPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name == "common-divisor-reciprocal") {
                    FPM.addPass(CommonDivisorReciprocal());
                    return true;
                  }
                  return false;
                });
          }};
}
