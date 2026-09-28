//===- SqrtErrnoElision.cpp -----------------------------------------------===//
//
// Replace a libm sqrt call with llvm.sqrt when the argument is provably not
// negative, which is the only input for which sqrt sets errno.
//
// Why this is worth a pass. Without -fno-math-errno a compiler must assume
// sqrt can set errno, so the call stays a call: it cannot be sunk, reordered
// freely, or -- the expensive part -- vectorised. Measured on a vector
// normalise kernel, that single call is what keeps gcc's inner loop scalar,
// and -fno-math-errno produces bit-identical output while unblocking the
// vectoriser. But -fno-math-errno is a global promise about every libm call in
// the translation unit, including the ones whose argument really can be
// negative. This makes the same promise only where it is proved.
//
// The proof. errno is set for sqrt only on a domain error, i.e. argument < 0.
// A product a*a is either non-negative or NaN; a sum of such values is
// non-negative, NaN, or +inf. sqrt of NaN returns NaN and raises no domain
// error, and sqrt(+0.0) and sqrt(+inf) are likewise errno-free. So for a sum
// of squares the errno path is dead code, and llvm.sqrt -- which carries no
// errno contract -- computes the identical value.
//
// What is deliberately NOT proved: a bare argument, a load, a call result, or
// anything reached through fneg or fsub. Those can be negative, and the pass
// leaves them alone. The test kernel carries one of each so the pass is
// checked for what it declines as well as what it rewrites.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {

/// Whether `V` can be shown non-negative (NaN permitted) without assuming
/// anything the caller did not write. Depth-limited: a proof that needs to
/// walk far is a proof this pass should not be making.
static bool provablyNonNegative(const Value *V, unsigned Depth = 0) {
  if (Depth > 6)
    return false;

  if (const auto *C = dyn_cast<ConstantFP>(V))
    return !C->getValueAPF().isNegative() || C->getValueAPF().isZero();

  const auto *I = dyn_cast<Instruction>(V);
  if (!I)
    return false; // arguments, loads through opaque pointers, anything unknown

  switch (I->getOpcode()) {
  case Instruction::FMul:
    // a*a only. a*b with a != b is negative whenever the signs differ.
    return I->getOperand(0) == I->getOperand(1);

  case Instruction::FAdd:
    return provablyNonNegative(I->getOperand(0), Depth + 1) &&
           provablyNonNegative(I->getOperand(1), Depth + 1);

  case Instruction::Call: {
    const auto *CI = cast<CallInst>(I);
    switch (CI->getIntrinsicID()) {
    case Intrinsic::fmuladd:
    case Intrinsic::fma:
      // a*a + c, which is how clang emits a sum of squares at -O2.
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
  if (!F || !F->isDeclaration() || CI->getIntrinsicID() != Intrinsic::not_intrinsic)
    return false;
  StringRef N = F->getName();
  return (N == "sqrt" || N == "sqrtf") && CI->arg_size() == 1 &&
         CI->getType() == CI->getArgOperand(0)->getType();
}

struct SqrtErrnoElision : PassInfoMixin<SqrtErrnoElision> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &) {
    SmallVector<CallInst *, 8> Rewrite;
    unsigned Declined = 0;

    for (Instruction &I : instructions(F)) {
      auto *CI = dyn_cast<CallInst>(&I);
      if (!CI || !isLibmSqrt(CI))
        continue;
      if (provablyNonNegative(CI->getArgOperand(0))) {
        Rewrite.push_back(CI);
      } else {
        ++Declined;
      }
    }

    for (CallInst *CI : Rewrite) {
      IRBuilder<> B(CI);
      Function *Decl = Intrinsic::getDeclaration(F.getParent(), Intrinsic::sqrt,
                                                 {CI->getType()});
      CallInst *New = B.CreateCall(Decl, {CI->getArgOperand(0)});
      New->copyFastMathFlags(CI);
      New->setDebugLoc(CI->getDebugLoc());
      CI->replaceAllUsesWith(New);
      CI->eraseFromParent();
    }

    if (!Rewrite.empty() || Declined)
      errs() << "sqrt-errno-elision: " << F.getName() << ": rewrote "
             << Rewrite.size() << ", declined " << Declined << "\n";

    return Rewrite.empty() ? PreservedAnalyses::all() : PreservedAnalyses::none();
  }
};

} // namespace

extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "SqrtErrnoElision", "0.1",
          [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, FunctionPassManager &FPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name == "sqrt-errno-elision") {
                    FPM.addPass(SqrtErrnoElision());
                    return true;
                  }
                  return false;
                });
          }};
}
