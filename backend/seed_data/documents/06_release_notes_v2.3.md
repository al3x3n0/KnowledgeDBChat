# OptimizeIR v2.3.0 Release Notes

**Release Date:** January 15, 2026

## Release Overview

OptimizeIR v2.3.0 is a major release that introduces ARM SVE2 support, a new machine learning-guided optimization framework, and significant performance improvements. This release represents 6 months of development with contributions from 23 developers.

## New Features

### ARM SVE2 Support

Full support for ARM Scalable Vector Extension 2 (SVE2):

- **Automatic vectorization**: Loops are automatically vectorized using SVE2 predicated instructions
- **Vector length agnostic code**: Generated code adapts to any SVE vector length (128-2048 bits)
- **New intrinsics**: Support for SVE2-specific operations (complex arithmetic, cryptography)

```cpp
// Example: SVE2 vectorized reduction
int sum_sve2(int* arr, int n) {
    svint32_t acc = svdup_s32(0);
    for (int i = 0; i < n; i += svcntw()) {
        svbool_t pg = svwhilelt_b32(i, n);
        svint32_t v = svld1_s32(pg, &arr[i]);
        acc = svadd_s32_m(pg, acc, v);
    }
    return svaddv_s32(svptrue_b32(), acc);
}
```

### ML-Guided Optimization Framework

New infrastructure for machine learning-guided optimization decisions:

- **Pluggable models**: Support for custom ML models via ONNX runtime
- **Built-in models**: Pre-trained models for inlining, vectorization, and unrolling decisions
- **Training pipeline**: Tools to train custom models on your codebase

Performance impact:
- 12% improvement in optimization quality on SPEC CPU 2017
- 8% reduction in code size with similar performance

### Enhanced Loop Tiling

Improved automatic loop tiling for better cache utilization:

- **Multi-level tiling**: Automatic selection of tile sizes for L1/L2/L3 caches
- **Rectangular tiling**: Support for non-square tile shapes
- **Parametric tiles**: Runtime-adjustable tile sizes for different hardware

### Profile-Guided Optimization Improvements

Enhanced PGO workflow:

- **CSIR profiles**: Context-sensitive profiles for better inlining decisions
- **Stale profile handling**: Graceful degradation when profiles are outdated
- **Profile debugging**: New tools to visualize and debug profile data

## Improvements

### Performance Improvements

- **15% faster compilation** through improved pass scheduling
- **Loop vectorization**: 23% more loops vectorized on average
- **Alias analysis**: 40% faster with new caching strategy
- **Inlining**: Better cost model reduces code bloat by 8%

### Usability Improvements

- **Better diagnostics**: Optimization remarks now include actionable suggestions
- **Pass pipeline visualization**: New tool to visualize pass execution order
- **Configuration validation**: Early detection of invalid configuration combinations

### Platform Support

- **RISC-V Vector Extension**: Initial support for RVV 1.0
- **Windows on ARM**: Full support for Windows 11 on ARM64
- **macOS Sonoma**: Verified compatibility with macOS 14

## Bug Fixes

### Critical Fixes

- **[#1234]** Fixed incorrect vectorization of loops with wrap-around indices
- **[#1256]** Fixed crash when optimizing functions with over 10,000 basic blocks
- **[#1278]** Fixed memory corruption in parallel pass execution

### Important Fixes

- **[#1189]** Fixed suboptimal register allocation in hot loops
- **[#1201]** Fixed incorrect constant folding of floating-point operations
- **[#1215]** Fixed debug info corruption after loop unrolling
- **[#1223]** Fixed incorrect code generation for atomic operations on ARM
- **[#1245]** Fixed performance regression in string operations

### Minor Fixes

- **[#1112]** Fixed typos in diagnostic messages
- **[#1134]** Fixed incorrect statistics reporting
- **[#1156]** Fixed build warnings with GCC 13

## Breaking Changes

### API Changes

1. **PassManager API**: The `addPass(StringRef)` method now throws on unknown pass names instead of silently failing

```cpp
// Old behavior (silent failure)
PM.addPass("nonexistent-pass"); // No error

// New behavior (exception)
PM.addPass("nonexistent-pass"); // Throws UnknownPassException
```

2. **Analysis API**: `AnalysisManager::getResult()` now returns a reference instead of a pointer

```cpp
// Old API
auto* result = AM.getResult<LoopAnalysis>(F);
if (result) { ... }

// New API
auto& result = AM.getResult<LoopAnalysis>(F); // Throws if not available
// Or use tryGetResult for optional access
if (auto* result = AM.tryGetResult<LoopAnalysis>(F)) { ... }
```

### Configuration Changes

- Removed deprecated `--enable-experimental-vectorizer` flag (now always enabled)
- Renamed `--ml-model-path` to `--optimization-model-path` for clarity

## Known Issues

1. **SVE2 + LTO**: Combining SVE2 code generation with LTO may cause linker errors on older toolchains. Workaround: Use LLVM's lld linker.

2. **PGO on Windows**: Profile-guided optimization shows reduced effectiveness on Windows due to sampling limitations. We're working with Microsoft on a fix.

3. **Debug info for tiled loops**: Stepping through tiled loops may show incorrect source locations. Fix planned for v2.3.1.

## Upgrade Instructions

### From v2.2.x

1. Update your build scripts to use the new API:
```cmake
# Old
target_link_libraries(myapp optimizeir::core)

# New (unchanged, but verify CMake config)
target_link_libraries(myapp optimizeir::core)
```

2. Update any custom passes using the old AnalysisManager API

3. Review and update configuration files for renamed options

### From v2.1.x or Earlier

We recommend upgrading to v2.2.x first, then to v2.3.x to minimize migration effort.

## Dependencies

- LLVM 17.0.0 or later (17.0.6 recommended)
- CMake 3.20 or later
- C++17 compatible compiler
- Python 3.8+ for Python bindings
- ONNX Runtime 1.15+ for ML features (optional)

## Acknowledgments

Thanks to all contributors who made this release possible:
- Core team: Alice Chen, Bob Smith, Carol Williams, David Kim
- Community contributors: 19 individuals from 8 organizations
- Special thanks to the LLVM community for the excellent upstream infrastructure

## Next Release

OptimizeIR v2.4.0 is planned for July 2026 with focus on:
- Automatic parallelization (OpenMP code generation)
- Enhanced GPU offloading support
- Improved whole-program optimization
