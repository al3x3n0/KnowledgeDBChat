# OptimizeIR API Reference

## Overview

OptimizeIR provides both C++ and Python APIs for integration with existing toolchains and workflows.

## C++ API

### Core Classes

#### `OptimizeIRContext`

The main entry point for using OptimizeIR programmatically.

```cpp
#include <optimizeir/Context.h>

class OptimizeIRContext {
public:
    // Create a new context with default configuration
    OptimizeIRContext();

    // Create with custom configuration
    explicit OptimizeIRContext(const Config& config);

    // Load an LLVM module from file
    std::unique_ptr<Module> loadModule(StringRef path);

    // Load from bitcode buffer
    std::unique_ptr<Module> loadModuleFromBuffer(MemoryBufferRef buffer);

    // Get the pass manager for this context
    PassManager& getPassManager();

    // Run optimization pipeline on module
    OptimizationResult optimize(Module& module, OptLevel level);

    // Write optimized module to file
    void writeModule(Module& module, StringRef outputPath);
};
```

#### `PassManager`

Manages the execution of optimization passes.

```cpp
class PassManager {
public:
    // Add a pass to the pipeline
    void addPass(std::unique_ptr<Pass> pass);

    // Add a pass by name
    void addPass(StringRef passName);

    // Remove a pass from the pipeline
    void removePass(StringRef passName);

    // Set optimization level (-O0 to -O3)
    void setOptLevel(OptLevel level);

    // Enable/disable specific optimization categories
    void enableLoopOptimizations(bool enable);
    void enableVectorization(bool enable);
    void enableMemoryOptimizations(bool enable);

    // Run the pipeline on a module
    bool run(Module& module);

    // Get statistics from last run
    const PassStatistics& getStatistics() const;
};
```

#### `AnalysisManager`

Provides access to analysis results.

```cpp
class AnalysisManager {
public:
    // Get analysis result for a function
    template<typename AnalysisT>
    typename AnalysisT::Result& getResult(Function& F);

    // Check if analysis is cached
    template<typename AnalysisT>
    bool isCached(Function& F) const;

    // Invalidate cached analysis
    void invalidate(Function& F);

    // Clear all cached analyses
    void clear();
};
```

### Utility Functions

```cpp
namespace optimizeir {

// Estimate the cost of a transformation
int estimateTransformationCost(const Instruction& I, const TargetInfo& TI);

// Check if a loop is vectorizable
bool isVectorizable(const Loop& L, const AnalysisManager& AM);

// Get recommended vector width for target
unsigned getRecommendedVectorWidth(const TargetInfo& TI, Type* elementType);

// Dump IR to string for debugging
std::string dumpIR(const Value& V);

}
```

## Python API

### Basic Usage

```python
import optimizeir as oir

# Create context and load module
ctx = oir.Context()
module = ctx.load_module("input.bc")

# Configure optimization
config = oir.OptimizationConfig()
config.opt_level = oir.OptLevel.O3
config.enable_vectorization = True
config.target_cpu = "skylake"

# Run optimization
result = ctx.optimize(module, config)

# Check results
print(f"Optimized {result.functions_modified} functions")
print(f"Estimated speedup: {result.estimated_speedup:.2f}x")

# Save output
ctx.write_module(module, "output.bc")
```

### Analysis API

```python
import optimizeir as oir
from optimizeir.analysis import LoopAnalysis, MemoryAnalysis

ctx = oir.Context()
module = ctx.load_module("input.bc")

# Analyze loops in a function
for func in module.functions:
    loop_info = LoopAnalysis.analyze(func)
    for loop in loop_info.loops:
        print(f"Loop at {loop.location}:")
        print(f"  Trip count: {loop.trip_count}")
        print(f"  Vectorizable: {loop.is_vectorizable}")
        print(f"  Nesting depth: {loop.nesting_depth}")
```

### Custom Pass Development

```python
import optimizeir as oir
from optimizeir.passes import FunctionPass

class MyCustomPass(FunctionPass):
    name = "my-custom-pass"

    def run(self, function, analysis_manager):
        modified = False
        for block in function.basic_blocks:
            for inst in block.instructions:
                if self.should_transform(inst):
                    self.transform(inst)
                    modified = True
        return modified

    def should_transform(self, inst):
        # Custom logic
        return inst.opcode == oir.Opcode.ADD

    def transform(self, inst):
        # Custom transformation
        pass

# Register and use custom pass
oir.register_pass(MyCustomPass)
ctx = oir.Context()
ctx.pass_manager.add_pass("my-custom-pass")
```

## REST API

OptimizeIR also provides a REST API for remote optimization services.

### Endpoints

#### POST /api/v1/optimize

Optimize LLVM IR remotely.

**Request:**
```json
{
    "module_base64": "<base64-encoded LLVM bitcode>",
    "config": {
        "opt_level": "O3",
        "target_triple": "x86_64-unknown-linux-gnu",
        "target_cpu": "skylake",
        "enable_vectorization": true,
        "enable_loop_unrolling": true
    }
}
```

**Response:**
```json
{
    "status": "success",
    "optimized_module_base64": "<base64-encoded optimized bitcode>",
    "statistics": {
        "passes_run": 45,
        "functions_modified": 12,
        "instructions_deleted": 234,
        "instructions_created": 189,
        "estimated_speedup": 1.35
    }
}
```

#### GET /api/v1/passes

List available optimization passes.

#### POST /api/v1/analyze

Run analysis passes without transformation.

## Error Handling

```cpp
try {
    auto result = ctx.optimize(module, config);
} catch (const OptimizeIRException& e) {
    std::cerr << "Optimization failed: " << e.what() << "\n";
    std::cerr << "Error code: " << e.code() << "\n";
}
```

Error codes:
- `E_INVALID_MODULE`: Input module is malformed
- `E_UNSUPPORTED_TARGET`: Target architecture not supported
- `E_PASS_FAILED`: A pass failed during execution
- `E_VERIFICATION_FAILED`: Output verification failed
