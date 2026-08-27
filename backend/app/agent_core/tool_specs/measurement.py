"""Measurement and sandbox tools: the ones this project's research runs on.

The first family to declare itself once. These are also the tools whose
registration went wrong most often, which is why they moved first.
"""

from __future__ import annotations

from app.agent_core.tool_specs.spec import ToolSpec

SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="compile_c_snippet",
        description="Compile a C snippet in the compiler research sandbox and return "
        "the generated assembly plus codegen counts (vector instructions, "
        "conditional branches, calls). Use this to check what the compiler "
        "actually emitted before drawing conclusions from any timing: a "
        "loop may be vectorized or if-converted, leaving no branch to "
        "measure. Prefer this over execute_python for anything involving a "
        "compiler.",
        parameters={
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "C source to compile"},
                "flags": {
                    "type": "string",
                    "description": (
                        "Compiler flags, e.g. '-O2' or '-O3 -ffast-math'. The "
                        "sandbox targets aarch64: use '-mcpu=native' to tune "
                        "for the host, as clang there rejects '-march=native'."
                    ),
                },
                "emit": {
                    "type": "string",
                    "description": "'asm' (default) returns assembly, 'ir' returns LLVM IR",
                },
                "label": {
                    "type": "string",
                    "description": (
                        "Short name for what this snippet is, e.g. 'float sum "
                        "reduction'. Recorded with the measurement; without it "
                        "several measurements cannot be told apart afterwards."
                    ),
                },
            },
            "required": ["code"],
        },
        effects="write",
        cost_tier="high",
        pii_risk="medium",
        produces=("codegen_measurement",),
        typical_seconds=10,
        consumes="C source; returns the assembly the compiler really emitted.",
    ),
    ToolSpec(
        name="profile_c_workload",
        description="Compile a self-contained C program, run it under callgrind, and "
        "report what actually executed: exact dynamic instruction counts "
        "per function, and the hottest straight-line blocks with their "
        "disassembly. Use this to find where the time really goes before "
        "proposing anything -- source occurrence is not execution "
        "frequency, and a sequence appearing often in cold code is worth "
        "less than one appearing twice in an inner loop. Instrumented "
        "execution is ~50x slower than native, so give the program a "
        "bounded input. It counts instructions; it does not time them.",
        parameters={
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Self-contained C program including main()",
                },
                "flags": {
                    "type": "string",
                    "description": (
                        "Compiler flags (default '-O3 -g'). Debug info is added "
                        "if absent, since without it nothing can be attributed "
                        "to a function. This sandbox targets aarch64: use "
                        "'-mcpu=native', not '-march=native'."
                    ),
                },
                "run_args": {
                    "type": "string",
                    "description": "Arguments passed to the program (optional)",
                },
                "label": {
                    "type": "string",
                    "description": (
                        "Short name for this workload, recorded with the "
                        "profile so several can be told apart afterwards."
                    ),
                },
                "top_functions": {
                    "type": "integer",
                    "description": "How many functions to rank (default 8)",
                },
                "top_blocks": {
                    "type": "integer",
                    "description": "How many hot blocks to return (default 5)",
                },
            },
            "required": ["code"],
        },
        effects="write",
        cost_tier="high",
        pii_risk="medium",
        produces=("dynamic_profile",),
        typical_seconds=60,
        consumes="a self-contained program; runs it and counts what executed.",
    ),
    ToolSpec(
        name="find_fusion_candidates",
        description="Mine hot blocks for instruction sequences that could become one "
        "instruction. Builds the data-flow graph of each block, finds the "
        "connected groups a single opcode could encode -- convex, and "
        "within the operand budget -- and ranks them by how often the "
        "containing block actually executed. This is the step between "
        "profiling and proposing: it answers 'which sequences recur on "
        "the hot path', which reading disassembly by hand does not scale "
        "to. Feed it the blocks from profile_c_workload. A result is a "
        "claim that a shape is frequent, not that fusing it pays; cost it "
        "with analyze_snippet_cycles before proposing anything.",
        parameters={
            "type": "object",
            "properties": {
                "blocks": {
                    # Either shape: the schema check is what refused a model
                    # that described its blocks in prose instead of omitting
                    # the field, before the handler could fall back to the
                    # profile this run had already taken.
                    "type": ["array", "string"],
                    "items": {"type": "object"},
                    "description": (
                        "Optional. If you have already run profile_c_workload "
                        "in this job, leave this out and its hot blocks are "
                        "used automatically -- do not copy the disassembly "
                        "across, since a truncated copy mines a different "
                        "program than the one profiled. Otherwise pass objects "
                        "with an `instructions` list of assembly lines and an "
                        "`executions` count."
                    ),
                },
                "max_instructions": {
                    "type": "integer",
                    "description": (
                        "Largest group to consider (default 3). Bigger groups "
                        "are harder to encode and rarer."
                    ),
                },
                "max_inputs": {
                    "type": "integer",
                    "description": (
                        "External registers the fused instruction may read "
                        "(default 2, the usual budget for a 32-bit encoding)."
                    ),
                },
                "max_outputs": {
                    "type": "integer",
                    "description": "Results it may write (default 1).",
                },
                "min_executions": {
                    "type": "integer",
                    "description": (
                        "Drop candidates whose blocks ran fewer times than " "this."
                    ),
                },
            },
            "required": [],
        },
        effects="write",
        cost_tier="high",
        pii_risk="medium",
        produces=("fusion_candidate",),
        requires=("profile_c_workload",),
        consumes="the hot blocks of a profile taken in this run -- leave `blocks` out and the most recent profile is used.",
    ),
    ToolSpec(
        name="cost_fusion_candidate",
        description="Cost a mined fusion candidate and bound what fusing it could "
        "save, per occurrence, on a named core. Costs the sequence as it "
        "stands and each operation in it alone, then reports the saving as "
        "a range: at best the sequence's cost minus the slowest operation "
        "the fused form still has to perform, at worst nothing. It does "
        "not ask you to name an instruction to stand for the fused one, "
        "because the answer would then depend on that choice -- picking a "
        "slow stand-in manufactures a regression. Multiply the range by "
        "the candidate's dynamic_occurrences for the benefit.",
        parameters={
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": (
                        "The candidate, as find_fusion_candidates spells it, "
                        "e.g. 'fsqrt fdiv | 0>1'."
                    ),
                },
                "cpu": {
                    "type": "string",
                    "description": (
                        "The core model to cost against, e.g. neoverse-n1. "
                        "Required: a cycle count is a property of a core."
                    ),
                },
                "mode": {
                    "type": "string",
                    "description": (
                        "'dependent' (default) measures the chain's latency, "
                        "what a loop-carried computation meets; 'independent' "
                        "measures throughput, what an unrolled loop meets. "
                        "They disagree by a lot."
                    ),
                },
                "copies": {
                    "type": "integer",
                    "description": "Repetitions inside the region (default 20).",
                },
                "label": {"type": "string", "description": "A name for the run."},
            },
            "required": ["pattern", "cpu"],
        },
        effects="write",
        cost_tier="high",
        pii_risk="medium",
        produces=("fusion_cost_bound",),
        requires=("find_fusion_candidates",),
        consumes="the `pattern` string of a candidate, e.g. 'fsqrt fdiv | 0>1' -- the shape, not the instructions it was found in.",
    ),
    ToolSpec(
        name="analyze_snippet_cycles",
        description="Cost a code sequence against a named core's scheduling model with "
        "llvm-mca, without running it: cycles per iteration, IPC, uops and "
        "block reciprocal throughput. Use this to compare two sequences "
        "that do the same work, and to cost a sequence no hardware here "
        "can run -- a proposed instruction, or a target that is not this "
        "host. These are modelled estimates, not measurements: mca assumes "
        "a warm front end and no cache misses.",
        parameters={
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": (
                        "C source to compile and analyse. A function is enough; "
                        "no main() is needed."
                    ),
                },
                "asm": {
                    "type": "string",
                    "description": (
                        "Assembly to analyse directly, instead of code. Use this "
                        "to cost a hypothetical sequence, such as one where an "
                        "idiom is replaced by the instruction being proposed. "
                        "If you intend to compare the estimate against a "
                        "measurement of a compiled program, this must be the "
                        "compiler's own output with your edit applied, never "
                        "assembly you wrote by hand: a run that hand-wrote a "
                        "plausible-looking loop estimated 36.26 cycles per "
                        "iteration where the code that actually ran cost 59.05, "
                        "and blamed the estimate rather than the substitution. "
                        "To cost a loop rather than a whole function, fence it "
                        "with '# LLVM-MCA-BEGIN name' and a bare '# LLVM-MCA-END' "
                        "(no name after END, or llvm-mca rejects it): the same "
                        "kernel measures 24.14 cycles as a function and 7.18 as "
                        "its inner loop. These markers are assembly comments and "
                        "must go in 'asm', never in 'code'."
                    ),
                },
                "cpu": {
                    "type": "string",
                    "description": (
                        "Required. The core model to cost against, e.g. "
                        "'neoverse-n1', 'cortex-a78', 'cortex-x2'. A cycle "
                        "count without a named model cannot be compared."
                    ),
                },
                "flags": {
                    "type": "string",
                    "description": "Compiler flags used when code is given (default -O3)",
                },
                "target": {
                    "type": "string",
                    "description": (
                        "Target TRIPLE, e.g. aarch64-linux-gnu (the default). "
                        "Not a core model and not a name for the run: the core "
                        "goes in 'cpu' and a name goes in 'label'. Cross-target "
                        "analysis works, since the code is never executed."
                    ),
                },
                "iterations": {
                    "type": "integer",
                    "description": "Iterations to simulate (default 100)",
                },
                "label": {
                    "type": "string",
                    "description": (
                        "Short name for what this sequence is. Recorded with the "
                        "estimate; without it several estimates cannot be told "
                        "apart afterwards."
                    ),
                },
            },
            "required": ["cpu"],
        },
        effects="write",
        cost_tier="high",
        pii_risk="medium",
        produces=("cycle_model_measurement",),
        typical_seconds=10,
        consumes="assembly fenced with # LLVM-MCA-BEGIN / # LLVM-MCA-END.",
    ),
    ToolSpec(
        name="benchmark_c_snippet",
        description="Compile and run a self-contained C program in the compiler "
        "research sandbox, returning its stdout and wall-clock time over "
        "repeated trials (minimum reported). The program must print its "
        "own measurements; there are no performance counters in the "
        "sandbox, and there is no network.",
        parameters={
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Self-contained C program including main()",
                },
                "flags": {
                    "type": "string",
                    "description": (
                        "Compiler flags. The sandbox targets aarch64: use "
                        "'-mcpu=native', not '-march=native'."
                    ),
                },
                "repeat": {
                    "type": "integer",
                    "description": "Trials to run, 1-10 (default 3); the fastest is reported",
                },
                "label": {
                    "type": "string",
                    "description": (
                        "Short name for what this snippet is, e.g. 'float sum "
                        "reduction'. Recorded with the measurement; without it "
                        "several measurements cannot be told apart afterwards."
                    ),
                },
            },
            "required": ["code"],
        },
        effects="write",
        cost_tier="high",
        pii_risk="medium",
        produces=("benchmark_measurement",),
        typical_seconds=30,
        consumes="a program that times itself; runs it on the real host.",
    ),
    ToolSpec(
        name="simulate_c_workload",
        description="Run a self-contained C program in a simulated out-of-order core "
        "with caches and a branch predictor, and report the cycles it "
        "took. This is the referee for a performance claim: "
        "analyze_snippet_cycles estimates how a sequence issues assuming a "
        "warm front end and no cache misses, while this executes it and "
        "measures. Simulation runs on the order of 100k instructions a "
        "second, so bring a kernel with a bounded input, and screen "
        "candidates with analyze_snippet_cycles first. Compare runs by "
        "cycles rather than sim_seconds.",
        parameters={
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Self-contained C program including main()",
                },
                "flags": {
                    "type": "string",
                    "description": (
                        "Compiler flags (default '-O3 -static'). Static linking "
                        "is added if absent: syscall-emulation mode has no "
                        "dynamic loader."
                    ),
                },
                "cpu_type": {
                    "type": "string",
                    "description": (
                        "Which core to model. Generic: O3CPU (out-of-order, the "
                        "one a timing claim needs), MinorCPU (in-order), "
                        "TimingSimpleCPU, AtomicSimpleCPU (no timing model at "
                        "all). Named ARM cores: NeoverseV2, O3_ARM_v7a_3, HPI, "
                        "ex5_big, ex5_LITTLE. The generic models carry gem5's "
                        "default latencies, which match no shipped silicon -- "
                        "measured against an Apple M3 host, O3CPU is 40% off "
                        "per instruction and NeoverseV2 77%, so name the core a "
                        "claim is about and calibrate it. NeoverseV2, ex5_big "
                        "and ex5_LITTLE have no functional unit for scalar "
                        "fused multiply-add: this tool refuses such workloads "
                        "rather than hanging, and -ffp-contract=off avoids it."
                    ),
                },
                "param_overrides": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Tune the core model for this run: full parameter "
                        "assignments such as "
                        "'system.cpu[0].instQueues[0].fuPool.FUList[3]"
                        ".opList[4].opLat=10' or 'system.cpu[0].issueWidth=6'. "
                        "Call describe_model_parameters to get the exact paths "
                        "a model exposes -- they are not guessable, and the "
                        "flattened names in gem5's config.ini cannot be "
                        "assigned to. This is how a model is calibrated "
                        "against measured silicon without forking a config."
                    ),
                },
                "run_args": {
                    "type": "string",
                    "description": "Arguments passed to the program (optional)",
                },
                "label": {
                    "type": "string",
                    "description": (
                        "Short name for this run, recorded with the "
                        "measurement so variants can be told apart."
                    ),
                },
            },
            "required": ["code"],
        },
        effects="write",
        cost_tier="high",
        pii_risk="medium",
        produces=("simulated_measurement",),
        typical_seconds=60,
        consumes="a self-contained program; runs it in a modelled core.",
    ),
    ToolSpec(
        name="describe_model_parameters",
        description="List the tunable parameters of a simulated core model and the "
        "exact paths that set them. Tuning a model is impossible without "
        "this: the per-op-class latencies live in a functional-unit pool "
        "whose layout differs per model, and the paths gem5 prints in its "
        "own config.ini are not the paths that can be assigned to. Returns "
        "op-class latencies (what a single-instruction benchmark "
        "constrains) separately from widths and queue depths (which only "
        "whole-kernel behaviour can pin down). Feed the `parameter` "
        "strings to simulate_c_workload's param_overrides with =<value> "
        "appended. Costs one short simulation, so call it once per model "
        "rather than per candidate.",
        parameters={
            "type": "object",
            "properties": {
                "cpu_type": {
                    "type": "string",
                    "description": (
                        "Which model to inspect: O3CPU, MinorCPU, "
                        "TimingSimpleCPU, AtomicSimpleCPU, NeoverseV2, "
                        "O3_ARM_v7a_3, HPI, ex5_big or ex5_LITTLE."
                    ),
                },
                "op_classes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional filter, e.g. ['FloatSqrt','FloatDiv']. "
                        "Omit to see every op class the model defines."
                    ),
                },
            },
            "required": [],
        },
        effects="write",
        cost_tier="high",
        pii_risk="medium",
        produces=("model_parameters",),
        consumes="a core model name; returns its tunable parameters and paths.",
    ),
    ToolSpec(
        name="describe_gem5_mechanisms",
        description="List the microarchitectural mechanisms this gem5 build carries -- "
        "prefetchers, cache replacement policies, branch direction "
        "predictors -- asked of the build rather than remembered. These "
        "move between gem5 releases: on this build every direction "
        "predictor was moved under a new base class, so gem5's own "
        "--bp-type flag offers one of the dozen that are installed. Feed a "
        "class name to simulate_mechanism. Call it once per study.",
        parameters={
            "type": "object",
            "properties": {
                "kind": {
                    "type": "string",
                    "description": (
                        "Optional filter: prefetcher, replacement_policy, "
                        "conditional_predictor or cpu_type. Omit for all."
                    ),
                },
            },
            "required": [],
        },
        effects="write",
        cost_tier="low",
        pii_risk="medium",
        produces=("mechanism_catalog",),
        typical_seconds=15,
        consumes="nothing; reports what the simulator can be asked for.",
    ),
    ToolSpec(
        name="simulate_mechanism",
        description="Measure what one microarchitectural mechanism is worth. Runs the "
        "workload TWICE in the simulated core -- once with the mechanism, "
        "once without -- and reports the cycle ratio between them. This is "
        "the tool for a prefetcher, a cache replacement policy or a branch "
        "predictor; simulate_c_workload tunes the values on a model and "
        "cannot install one of these at all. The pair is the unit on "
        "purpose: a prefetcher added alongside a wider MSHR file measured "
        "2.59x here, and the MSHRs alone measured the same 2.59x, so a "
        "single run cannot say which change it is reporting. Arms that "
        "differ in anything but the mechanism are refused, and a mechanism "
        "that never fires is reported as a failure rather than as no "
        "benefit.",
        parameters={
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": (
                        "Self-contained C program including main(). Give it a "
                        "bounded input: two full simulations run, at roughly "
                        "100k instructions a second each."
                    ),
                },
                "variant": {
                    "type": "object",
                    "description": (
                        "The configuration carrying the mechanism, e.g. "
                        '{"caches": {"l2": {"prefetcher": "StridePrefetcher"}}} '
                        'or with parameters {"caches": {"l2": {"prefetcher": '
                        '{"class": "StridePrefetcher", "params": {"degree": 8}}}}}. '
                        'Also takes "cpu_type", cache geometry (size, assoc, '
                        'mshrs) per level, "replacement_policy", and '
                        '"branch_pred": {"conditional": "LTAGE"}. Attach a '
                        "prefetcher to L2 first: at gem5's default L1 mshrs=4 "
                        "one has no spare capacity to issue into and measures "
                        "as nothing at all."
                    ),
                },
                "baseline": {
                    "type": "object",
                    "description": (
                        "The configuration to compare against. Omit it and the "
                        "variant with its mechanisms stripped out is used, "
                        "which is the safe default -- a hand-written baseline "
                        "is where the geometry silently drifts apart."
                    ),
                },
                "flags": {
                    "type": "string",
                    "description": (
                        "Compiler flags (default '-O2 -static'). Static linking "
                        "is added if absent; syscall-emulation mode has no "
                        "dynamic loader. Both arms run the same binary."
                    ),
                },
                "run_args": {
                    "type": "string",
                    "description": "Arguments passed to the program (optional)",
                },
                "label": {
                    "type": "string",
                    "description": (
                        "Short name for what this comparison is, e.g. 'L2 "
                        "stride prefetch on strided scan'. Recorded with the "
                        "result so several comparisons can be told apart."
                    ),
                },
                "plugin_source": {
                    "type": "string",
                    "description": (
                        "C++ for a mechanism the simulator does NOT ship -- "
                        "this is how one gets invented rather than chosen. It "
                        "is compiled to /work/plugin.so before the runs and "
                        "loaded by naming that path. Two kinds are supported, "
                        "each with its own header, already on the include "
                        "path. Include one and nothing else: no gem5 headers "
                        "exist here and none are needed. "
                        "A CACHE REPLACEMENT POLICY: include "
                        '<gem5_rp_plugin_abi.h>, export `extern "C" const '
                        "Gem5RpApiV1 *gem5_rp_api_v1(void)`, select it with "
                        '{"class": "PluginRP", "params": {"library": '
                        '"/work/plugin.so", "config": "your=params"}}. You get '
                        "a Gem5RpEntry per cache line (last_touch_tick, "
                        "touches, scratch[4]); get_victim returns the INDEX of "
                        "the entry to evict. "
                        "A PREFETCHER: include <gem5_pf_plugin_abi.h>, export "
                        '`extern "C" const Gem5PfApiV1 '
                        "*gem5_pf_api_v1(void)`, select it with "
                        '{"class": "PluginPrefetcher", "params": {"library": '
                        '"/work/plugin.so", "config": "your=params"}}. You are '
                        "asked only the algorithm -- given one access, which "
                        "addresses would you fetch -- because gem5 keeps the "
                        "queue, the throttling and the page-crossing checks; a "
                        "prefetcher that reimplements those measures the "
                        "harness rather than the idea. "
                        "Mirror a shipped mechanism first and check the two "
                        "agree: a novel algorithm as the first plugin leaves "
                        "nothing to check the boundary against but an opinion."
                    ),
                },
            },
            "required": ["code", "variant"],
        },
        effects="write",
        cost_tier="high",
        pii_risk="medium",
        produces=("mechanism_comparison",),
        typical_seconds=180,
        consumes="a program and a mechanism; runs both arms in one model.",
    ),
    ToolSpec(
        name="explain_bottleneck",
        description="Run a C kernel in the simulated core once and say what its cycles "
        "were spent waiting on: which structure backpressures the pipeline, "
        "how much miss latency it paid for, how often it mispredicted. A "
        "cycle count says how slow a kernel is and nothing about why, so "
        "this is what turns a measurement into a next step. It ranks "
        "suspects and names the limit studies that would settle them -- it "
        "does not conclude, because the signals overlap and the ranking has "
        "already been wrong once. Call measure_headroom on the targets it "
        "names, and take more than the first.",
        parameters={
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Self-contained C program including main(), with a bounded input",
                },
                "config": {
                    "type": "object",
                    "description": (
                        "Optional core configuration, same shape as "
                        "simulate_mechanism's variant. Omit for the default "
                        "out-of-order core."
                    ),
                },
                "flags": {
                    "type": "string",
                    "description": "Compiler flags (default '-O2 -static')",
                },
                "run_args": {
                    "type": "string",
                    "description": "Arguments passed to the program (optional)",
                },
                "label": {
                    "type": "string",
                    "description": "Short name for the kernel, recorded with the attribution",
                },
            },
            "required": ["code"],
        },
        effects="write",
        cost_tier="high",
        pii_risk="medium",
        produces=("bottleneck_attribution",),
        typical_seconds=90,
        consumes="a program; reports what limited it.",
    ),
    ToolSpec(
        name="measure_headroom",
        description="Bound what a microarchitectural idea could be worth BEFORE designing "
        "one. Makes a named structure effectively infinite -- the issue "
        "queue, the reorder buffer, a cache -- and reports the cycles that "
        "recovers, plus what becomes the limit once it is removed. An "
        "idealised structure is not buildable, so each number is a ceiling "
        "on any mechanism aimed there: a target with near-zero headroom "
        "cannot pay however it is implemented. Measured on one kernel: "
        "idealising the L1 recovered 84% and the issue queue 3.4%, and on "
        "another 29% and 12% -- neither is guessable from the cycle count. "
        "Targets: issue_queue, reorder_buffer, load_queue, store_queue, "
        "physical_registers, pipeline_width, l1d_capacity, l1i_capacity, "
        "l2_capacity, branch_prediction.",
        parameters={
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Self-contained C program including main()",
                },
                "targets": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Which structures to idealise, one simulation each. "
                        "explain_bottleneck names the ones worth trying; take "
                        "two or three, since the strongest signal is not "
                        "reliably the one worth the most."
                    ),
                },
                "config": {
                    "type": "object",
                    "description": "Optional core configuration to idealise against",
                },
                "flags": {
                    "type": "string",
                    "description": "Compiler flags (default '-O2 -static')",
                },
                "run_args": {
                    "type": "string",
                    "description": "Arguments passed to the program (optional)",
                },
                "label": {"type": "string", "description": "Short name for the kernel"},
            },
            "required": ["code", "targets"],
        },
        effects="write",
        cost_tier="high",
        pii_risk="medium",
        produces=("headroom_bound",),
        typical_seconds=240,
        consumes="a program and a list of structures; bounds each one's payoff.",
    ),
    ToolSpec(
        name="sweep_mechanism",
        description="Measure a mechanism at several settings and return the curve, not two "
        "points. Says where a setting stops paying and whether it ever "
        "turns around -- a mechanism reported at its best setting alone is "
        "a best case, not a result. All points share one baseline, so the "
        "sweep costs one simulation per value plus one.",
        parameters={
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Self-contained C program including main()",
                },
                "variant": {
                    "type": "object",
                    "description": (
                        "The configuration carrying the mechanism, e.g. "
                        '{"caches": {"l2": {"prefetcher": "StridePrefetcher"}}}'
                    ),
                },
                "vary": {
                    "type": "string",
                    "description": (
                        "Dotted path to sweep within the variant, e.g. "
                        "'caches.l2.prefetcher.params.degree' or "
                        "'caches.l2.size'. Intermediate levels are created as "
                        "needed, so a prefetcher named as a bare string works."
                    ),
                },
                "values": {
                    "type": "array",
                    "description": "The settings to try, 2 to 8 of them, in order",
                },
                "baseline": {
                    "type": "object",
                    "description": (
                        "What every point is measured against. Omit for the "
                        "default core with no mechanism."
                    ),
                },
                "flags": {
                    "type": "string",
                    "description": "Compiler flags (default '-O2 -static')",
                },
                "run_args": {
                    "type": "string",
                    "description": "Arguments passed to the program (optional)",
                },
                "label": {"type": "string", "description": "Short name for the sweep"},
            },
            "required": ["code", "variant", "vary", "values"],
        },
        effects="write",
        cost_tier="high",
        pii_risk="medium",
        produces=("mechanism_sweep",),
        typical_seconds=300,
        consumes="a mechanism and a range; returns its curve.",
    ),
    ToolSpec(
        name="evaluate_across_kernels",
        description="Measure one mechanism on several kernels and report the distribution: "
        "geometric mean, worst case, and which kernels regressed. A "
        "single-workload result is the one that gets overturned, and the "
        "worst case is what decides whether a mechanism ships. Each kernel "
        "is measured against its own baseline with the same arms, and arms "
        "differing in anything but the mechanism are refused.",
        parameters={
            "type": "object",
            "properties": {
                "kernels": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": (
                        'Two to six kernels, each {"name": "...", "code": '
                        '"...", "run_args": "..."}. Give them different '
                        "access and control-flow shapes -- three variations "
                        "of one loop measure one workload three times."
                    ),
                },
                "variant": {
                    "type": "object",
                    "description": "The configuration carrying the mechanism",
                },
                "baseline": {
                    "type": "object",
                    "description": (
                        "What to compare against. Omit and the variant with "
                        "its mechanisms stripped out is used."
                    ),
                },
                "flags": {
                    "type": "string",
                    "description": "Compiler flags (default '-O2 -static')",
                },
                "label": {
                    "type": "string",
                    "description": "Short name for the evaluation",
                },
            },
            "required": ["kernels", "variant"],
        },
        effects="write",
        cost_tier="high",
        pii_risk="medium",
        produces=("mechanism_evaluation",),
        typical_seconds=600,
        consumes="a mechanism and several kernels; returns its distribution.",
    ),
    ToolSpec(
        name="sample_hardware_counters",
        description="Run a C workload in a simulated core and return its hardware "
        "counters SAMPLED OVER TIME, rather than as run totals. Call "
        "M5_SAMPLE() in the program wherever a sample should be taken -- "
        "typically once per outer-loop iteration or per phase -- and each "
        "interval reports what happened since the previous call. This is "
        "the shape a hardware predictor reads: run totals cannot train or "
        "evaluate one. The macro is injected; do not define it. Only "
        "counters that vary across the trace are returned, because a "
        "counter that never changes cannot predict anything that does.",
        parameters={
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": (
                        "Self-contained C program calling M5_SAMPLE() at each "
                        "sampling point. Fewer than 4 samples is a total, not "
                        "a trace, and the result says so."
                    ),
                },
                "cpu_type": {
                    "type": "string",
                    "description": "Core model to simulate; see describe_model_parameters",
                },
                "flags": {"type": "string", "description": "Compiler flags"},
                "label": {
                    "type": "string",
                    "description": "Short name for what this workload is",
                },
                "max_counters": {
                    "type": "integer",
                    "description": "How many varying counters to return (default 60)",
                },
                "co_runner": {
                    "type": "string",
                    "description": (
                        "A second C program to run on the same core under SMT. "
                        "This is how a contention experiment is built: the "
                        "workload above is the thread being predicted, and "
                        "this is what competes with it. Give the primary "
                        "steady work so that any variation in its progress is "
                        "the co-runner's doing. The register-file overrides "
                        "SMT needs are applied automatically."
                    ),
                },
                "language": {
                    "type": "string",
                    "description": (
                        "'c' (default) or 'c++'. C++ needs a toolchain in the "
                        "image; the tool checks and says so rather than "
                        "failing at the compiler."
                    ),
                },
                "extra_files": {
                    "type": "object",
                    "description": (
                        "Headers or sources to stage alongside the workload, "
                        "as {relative path: contents}. This is how a real "
                        "corpus is measured rather than a workload written "
                        "for the study -- stage the header closure and include "
                        "the corpus from the workload."
                    ),
                },
                "include_dirs": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Directories added to the compiler's include path, "
                        "relative to the staged files."
                    ),
                },
                "intends_alternating_phases": {
                    "type": "boolean",
                    "description": (
                        "Set true when the workload is meant to ALTERNATE "
                        "between phases interval by interval. The design is "
                        "measured in a cheap simulator before the real run, "
                        "and running all of one phase and then all of the "
                        "other produces two regimes rather than an "
                        "alternation -- refused when you declare the intent, "
                        "and only warned about when you do not, because a run "
                        "with a legitimate startup phase should not be told "
                        "its design is wrong."
                    ),
                },
            },
            "required": ["code"],
        },
        effects="write",
        cost_tier="high",
        pii_risk="medium",
        produces=("counter_trace",),
        typical_seconds=2400,
        consumes="a self-contained program that calls M5_SAMPLE() wherever a sample should be taken. Without those calls it returns one total, which is not a trace: predictability is a property of counters over time and cannot be read from a run's totals.",
    ),
    ToolSpec(
        name="measure_predictability",
        description="Measure how much signal the sampled hardware counters carry about "
        "a target counter's NEXT interval -- the ceiling on any predictor "
        "tapping them, established before designing one. Reads the trace "
        "from the sample_hardware_counters call this run already made; do "
        "not paste it back. The number that decides anything is "
        "'information beyond persistence': what a counter adds over simply "
        "predicting the same value as last interval. Programs run in "
        "phases, so almost every counter looks predictive until you ask "
        "what it contributes, and a predictor that cannot beat last-value "
        "is not worth building in hardware. A trace too short to estimate "
        "on is refused rather than answered.",
        parameters={
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "description": (
                        "Counter to predict, e.g. system.cpu.numCycles. Must be "
                        "present in the trace."
                    ),
                },
                "bins": {
                    "type": "integer",
                    "description": "Discretisation levels, default 3. More bins need a longer trace",
                },
                "from_interval": {
                    "type": "integer",
                    "description": (
                        "Ignore intervals before this one. Use it when "
                        "sample_hardware_counters warned that the trace "
                        "changes regime -- the intervals before the break "
                        "describe a machine that does not recur (a co-runner "
                        "still initialising, a cache still cold), and a number "
                        "taken across it largely measures the break rather "
                        "than the workload. Pass the interval the warning "
                        "names to study the steady side."
                    ),
                },
            },
            "required": ["target"],
        },
        effects="write",
        cost_tier="high",
        pii_risk="medium",
        produces=("predictability_ceiling",),
        requires=("sample_hardware_counters",),
        consumes="the name of the counter to predict. The trace this run already sampled is read from the run -- do not paste it back.",
    ),
    ToolSpec(
        name="select_counter_taps",
        description="Which counters, TOGETHER, a predictor should tap -- and how many "
        "of them are worth the wires. Reads the trace this run already "
        "sampled. measure_predictability scores counters one at a time; "
        "this does greedy forward selection from persistence, stopping at "
        "the depth the trace can support, because conditioning on one more "
        "counter multiplies the cells the estimate needs by the bin count "
        "and a trace is hundreds of intervals, not millions. Every tap is "
        "placed against a null that runs the SAME selection on permuted "
        "counters, so the threshold contains the advantage that picking "
        "the best of fifty confers -- and each tap is judged on what IT "
        "added, not on the running total, because its own increment is "
        "what its area is being bought with. Use this once a single "
        "counter has shown signal, to decide whether a second tap is a "
        "design or a coincidence.",
        parameters={
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "description": (
                        "Counter to predict, e.g. derived.thread0_ipc. Must be "
                        "present in the trace."
                    ),
                },
                "bins": {
                    "type": "integer",
                    "description": (
                        "Discretisation levels, default 3. More bins cost "
                        "supported taps: the depth this trace allows falls as "
                        "the bin count rises."
                    ),
                },
                "from_interval": {
                    "type": "integer",
                    "description": (
                        "Ignore intervals before this one. Use it when "
                        "sample_hardware_counters warned that the trace "
                        "changes regime -- the intervals before the break "
                        "describe a machine that does not recur (a co-runner "
                        "still initialising, a cache still cold), and a number "
                        "taken across it largely measures the break rather "
                        "than the workload. Pass the interval the warning "
                        "names to study the steady side."
                    ),
                },
            },
            "required": ["target"],
        },
        effects="write",
        cost_tier="high",
        pii_risk="medium",
        produces=("counter_tap_selection",),
        requires=("measure_predictability",),
        consumes="the counter to predict; returns which taps survive their own null.",
    ),
    ToolSpec(
        name="evaluate_predictor_design",
        description="Run the predictor a measurement indicated and score it against "
        "its own ceiling, on intervals it was not trained on. Use after "
        "select_counter_taps has named a tap. An information ceiling says "
        "what is available; this says what a table of saturating counters "
        "actually reaches, which is the number that decides whether to "
        "stop or to spend more. Reports a design's cost in bits of state "
        "next to what it gains OVER PREDICTING THE SAME AS LAST INTERVAL "
        "-- never over chance, which flatters every predictor on an "
        "autocorrelated target. The split is contiguous, never random, "
        "because adjacent intervals are near-identical and a random split "
        "puts each scored row's twin in the warm-up. If the cheap design "
        "captures most of its ceiling, a learned model is competing for "
        "the remainder and no training corpus needs generating to find "
        "that out.",
        parameters={
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "description": "Counter to predict, e.g. derived.thread0_ipc.",
                },
                "tap": {
                    "type": "string",
                    "description": (
                        "The counter the predictor reads alongside the target's "
                        "own last value. Normally the tap select_counter_taps "
                        "recommended."
                    ),
                },
                "bins": {
                    "type": "integer",
                    "description": "Discretisation levels, default 3.",
                },
                "split": {
                    "type": "number",
                    "description": (
                        "Fraction of the trace that warms the tables, default "
                        "0.5. The rest is scored and never trained on."
                    ),
                },
                "from_interval": {
                    "type": "integer",
                    "description": (
                        "Ignore intervals before this one. Use it when "
                        "sample_hardware_counters warned that the trace "
                        "changes regime -- the intervals before the break "
                        "describe a machine that does not recur (a co-runner "
                        "still initialising, a cache still cold), and a number "
                        "taken across it largely measures the break rather "
                        "than the workload. Pass the interval the warning "
                        "names to study the steady side."
                    ),
                },
            },
            "required": ["target", "tap"],
        },
        effects="write",
        cost_tier="high",
        pii_risk="medium",
        produces=("predictor_design_result",),
        requires=("select_counter_taps",),
        consumes="the counter to predict and the tap to read alongside its own last value -- normally the one select_counter_taps recommended. An information ceiling says what is available; this says what a table of counters reaches.",
    ),
    ToolSpec(
        name="record_prediction",
        description="State what you expect a measurement to show, and how you reached "
        "that, BEFORE running the thing that measures it. This is what "
        "makes a methodology scoreable: the error between this number and "
        "what is measured is the score, and a prediction written after the "
        "outcome is known scores perfectly while teaching nothing. Returns "
        "a prediction_id to settle later with record_measurement.",
        parameters={
            "type": "object",
            "properties": {
                "subject": {
                    "type": "string",
                    "description": "What this is about, e.g. 'fused ldp+fmla in saxpy loop'",
                },
                "metric": {
                    "type": "string",
                    "description": (
                        "The quantity, e.g. 'speedup' or 'cycles_per_iteration'. "
                        "Errors on different quantities cannot be compared."
                    ),
                },
                "predicted_value": {
                    "type": "number",
                    "description": "The number you expect the measurement to produce",
                },
                "methodology": {
                    "type": "string",
                    "description": (
                        "How you arrived at it. This is what is being scored, "
                        "so describe the approach, not just the answer."
                    ),
                },
                "methodology_tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Short tags for the approach, e.g. ['mca', 'sampled']. "
                        "Later runs group errors by these to see which "
                        "approach predicts well."
                    ),
                },
                "prediction_basis": {
                    "type": "string",
                    "description": "The evidence behind the number (optional)",
                },
                "derived_from": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Required. The finding types this prediction is "
                        "computed from, e.g. ['cycle_model_measurement']. Each "
                        "must already exist in this run or the call is "
                        "refused: a prediction citing a measurement it never "
                        "obtained is worse than no prediction. If this is a "
                        "judgement with no measurement behind it, pass "
                        "['none'] and explain in the methodology -- that is "
                        "recorded as such, and is honest where a silent guess "
                        "is not."
                    ),
                },
            },
            "required": [
                "subject",
                "metric",
                "predicted_value",
                "methodology",
                "derived_from",
            ],
        },
        effects="write",
        cost_tier="high",
        pii_risk="medium",
        produces=("prediction_recorded",),
        consumes="a number, and `derived_from` naming finding types this run has already produced.",
    ),
    ToolSpec(
        name="record_measurement",
        description="Settle a prediction with what was actually measured, naming the "
        "referee that produced it. A prediction can only be settled once, "
        "so the flattering measurement cannot be the one kept.",
        parameters={
            "type": "object",
            "properties": {
                "prediction_id": {
                    "type": "string",
                    "description": "The UUID returned by record_prediction",
                },
                "measured_value": {
                    "type": "number",
                    "description": "What the measurement produced",
                },
                "measurement_source": {
                    "type": "string",
                    "description": (
                        "What produced it, e.g. 'gem5 O3 neoverse-n1' or "
                        "'wall clock, 5 trials'. A number without its source "
                        "cannot be compared with another."
                    ),
                },
                "notes": {"type": "string", "description": "Optional context"},
            },
            "required": ["prediction_id", "measured_value", "measurement_source"],
        },
        effects="write",
        cost_tier="high",
        pii_risk="medium",
        produces=("prediction_settled",),
        requires=("record_prediction",),
        consumes="the prediction_id returned by record_prediction, and the measured value.",
    ),
    ToolSpec(
        name="record_method",
        description="Record how to investigate something, with the evidence that it "
        "works, so later jobs inherit it. Findings say what you learned "
        "about the subject; this is for what you learned about method -- "
        "the part that transfers to a different subject. Record one when "
        "an approach turned out to be necessary, when an obvious approach "
        "produced a wrong answer, or when you found a check that catches a "
        "class of mistake. You must name the finding types in this run "
        "that establish it: a method claimed without evidence can only be "
        "stored by passing derived_from=['none'], and is marked unvalidated.",
        parameters={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": (
                        "Short name, e.g. 'measure instruction latency with "
                        "inline-asm dependent chains'."
                    ),
                },
                "procedure": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "The steps in order, concrete enough that another run "
                        "can follow them without rediscovering the method."
                    ),
                },
                "prevents": {
                    "type": "string",
                    "description": (
                        "The wrong answer this exists to stop, stated "
                        "specifically. This is what tells a future reader "
                        "whether their situation is the same one, so 'improves "
                        "accuracy' is not an answer -- 'the compiler vectorises "
                        "a C loop so the measurement is of different code' is."
                    ),
                },
                "derived_from": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Finding types produced in THIS run that establish the "
                        "method. Refused if no such finding exists. Pass "
                        "['none'] to record an untested method, which is stored "
                        "as unvalidated."
                    ),
                },
                "applies_to": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "What it applies to -- tools, domains or workload kinds "
                        "-- so it is recalled by jobs that need it."
                    ),
                },
                "limits": {
                    "type": "string",
                    "description": (
                        "Where it stops working, and what would falsify it."
                    ),
                },
                "builds_on": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Names of methods recalled into this run that you "
                        "actually followed. Saying so is what lets a method "
                        "earn a track record: a method merely present in your "
                        "context is weak evidence about this run, and one you "
                        "name is strong."
                    ),
                },
            },
            "required": ["name", "procedure", "prevents", "derived_from"],
        },
        effects="write",
        produces=("method_recorded",),
        consumes="the procedure, what it prevents, and the findings establishing it.",
    ),
    ToolSpec(
        name="axis_check",
        description="Validate an AXIS architecture description (.axisl) of an "
        "instruction-set extension. AXIS is the source of truth for a "
        "proposal: one description elaborates into encoder, decoder, "
        "semantics, compiler patterns and SMT-LIB semantics, so a proposal "
        "is checkable and its artifacts are regenerable rather than "
        "hand-written per candidate. Run this before emitting anything.",
        parameters={
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "The AXIS description source (.axisl text)",
                },
            },
            "required": ["source"],
        },
        effects="write",
        cost_tier="high",
        pii_risk="medium",
        produces=("axis_description",),
        consumes="an .axisl description of an instruction.",
    ),
    ToolSpec(
        name="axis_prove",
        description="Prove a claim about a proposed instruction against the formal "
        "semantics AXIS emits for it, using an SMT solver. This is the "
        "strongest gate available: a cycle count says a sequence is "
        "faster, a proof says the replacement computes the same thing for "
        "every input. Use it to show a fused instruction is equivalent to "
        "the sequence it replaces before spending simulation on it.",
        parameters={
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "The AXIS description source (.axisl text)",
                },
                "obligation": {
                    "type": "string",
                    "description": (
                        "SMT-LIB appended to the emitted semantics, so it can "
                        "call the generated functions by name. Assert the "
                        "NEGATION of your claim and end with (check-sat): "
                        "'unsat' then means no counterexample exists and the "
                        "claim holds for all inputs, while 'sat' returns a "
                        "counterexample showing the candidate is wrong."
                    ),
                },
            },
            "required": ["source", "obligation"],
        },
        effects="write",
        cost_tier="high",
        pii_risk="medium",
        produces=("equivalence_proof",),
        requires=("axis_check",),
        typical_seconds=30,
        consumes="an .axisl description and the sequence it should be equivalent to.",
    ),
    ToolSpec(
        name="axis_emit",
        description="Generate one artifact from an AXIS description: 'smt2' for formal "
        "semantics, 'decode-c'/'encode-c' to get the instruction into and "
        "out of a binary, 'semantics-c'/'sim-c' for a simulator, "
        "'golden-python' for a reference model, 'llvm-patterns'/'tablegen' "
        "for compiler support, 'pyrtl' for hardware. Note that the TableGen "
        "backend emits RISC-V instruction formats; check it before relying "
        "on it for another target.",
        parameters={
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "The AXIS description source (.axisl text)",
                },
                "target": {
                    "type": "string",
                    "description": (
                        "What to emit: json, bundle-manifest, legality-json, "
                        "encode-c, encode-json, decode-c, decode-json, "
                        "roundtrip-json, asm-disasm-json, semantics-c, "
                        "semantics-rust, semantics-json, sim-c, exec-c, "
                        "exec-python, golden-python, smt2, tablegen, llvm-ir, "
                        "llvm-patterns, llvm-intrinsics, intrinsics, pyrtl"
                    ),
                },
            },
            "required": ["source", "target"],
        },
        effects="write",
        cost_tier="high",
        pii_risk="medium",
    ),
    ToolSpec(
        name="verify_run_bundle",
        description="Check this run's evidence bundle, which is written as the run "
        "goes: every measurement call, its parameters, its result and the "
        "image it ran in. Without replay it confirms the artifacts are the "
        "ones this run produced. With replay=true it re-runs each recorded "
        "call and reports whether the same results come back, judging "
        "nothing that reports wall clock, since two honest runs of a "
        "benchmark disagree. Use it before claiming a result is "
        "reproducible.",
        parameters={
            "type": "object",
            "properties": {
                "replay": {
                    "type": "boolean",
                    "description": (
                        "Re-run the recorded calls and compare (slow: it "
                        "repeats every measurement). Default false, which "
                        "checks integrity only."
                    ),
                },
            },
            "required": [],
        },
        effects="write",
        cost_tier="high",
        pii_risk="medium",
    ),
)
