"""Measurement and sandbox tools: the ones this project's research runs on.

The first family to declare itself once. These are also the tools whose
registration went wrong most often, which is why they moved first.
"""

from __future__ import annotations

from app.agent_core.tool_specs.spec import ToolSpec
from app.services.agent_toolchains import (
    SUPPORTED,
    describe_llvm_passes,
    describe_rust_crates,
)

#: Named once and used in both tool descriptions, so the crate set a model is
#: told about cannot differ between the tool that checks code and the tool that
#: times it -- and so neither can drift from the image, which derives its
#: manifest from the same list.
_RUST_CRATES_HINT = describe_rust_crates()

#: Read from the same table, for the same reason. A pass that exists in the
#: image and is named in no tool description is reachable and undiscoverable,
#: which for a model is the same as absent.
_LLVM_PASSES_HINT = describe_llvm_passes()

#: Read from the toolchain table, never restated. A language the table can
#: build but the schema does not offer is unreachable -- the model is refused
#: for naming it -- and a language the schema offers but the table cannot build
#: fails at compile time with an error that blames the code.
_LANGUAGES = list(SUPPORTED)


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
                        "for the host, as clang there rejects '-march=native'. "
                        "Research LLVM passes built into the image, added with "
                        "the flag shown: " + _LLVM_PASSES_HINT
                    ),
                },
                "emit": {
                    "type": "string",
                    # Enumerated rather than described. The accepted set lived
                    # only in the validator, so a caller could read the
                    # description, pick a word that was not in it, and spend an
                    # iteration finding out -- which is what happened with
                    # 'counts', three times in one stage.
                    "enum": ["asm", "ir"],
                    "description": (
                        "Which listing to return: 'asm' (default) for assembly, "
                        "'ir' for LLVM IR. The codegen COUNTS -- vector ops, "
                        "branches, conditional selects, calls -- come back on "
                        "every call regardless; this only chooses the listing "
                        "beside them."
                    ),
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
                    "description": (
                        "Compiler flags used when code is given (default -O3). "
                        "Research LLVM passes in this image: " + _LLVM_PASSES_HINT
                    ),
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
        description="Compile and run a self-contained C, Rust or Python program "
        "in the compiler research sandbox, returning its stdout and "
        "wall-clock time over repeated trials (minimum reported). Set "
        "`language`; it defaults to C, and the name of this tool is "
        "historical. Python is timed as a whole process, so its result "
        "includes interpreter startup, reported as interpreter_startup_ms "
        "and warned about when it dominates. The program must print its own measurements; there "
        "are no performance counters in the sandbox, and there is no "
        "network. Benchmark the SAME source you passed to "
        "check_implementation -- a timing of code that was never checked "
        "is an accurate number for work nobody looked at.",
        parameters={
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Self-contained C program including main()",
                },
                "language": {
                    "type": "string",
                    "enum": _LANGUAGES,
                    "description": (
                        "Which toolchain builds it (default 'c'). Rust is one "
                        "self-contained .rs file compiled with rustc at "
                        "edition 2021 -- no Cargo.toml, no modules. A fixed "
                        "set of crates is prebuilt into the image and already "
                        "in scope, so `use rand::Rng;` works with no "
                        "declaration; nothing outside that set can be fetched, "
                        "because the sandbox has no network. Available: "
                        + _RUST_CRATES_HINT
                        + ". Python is one self-contained .py file run by "
                        "CPython with only the standard library -- there is no "
                        "network, so nothing can be pip-installed. Give a "
                        "Python program real work to do: a run that finishes "
                        "near the interpreter's own startup time has measured "
                        "startup, not the algorithm."
                    ),
                },
                "flags": {
                    "type": "string",
                    "description": (
                        "Compiler flags, defaulting per language ('-O2' for C, "
                        "'-O' for Rust). The sandbox targets aarch64: for C use "
                        "'-mcpu=native', not '-march=native'. rustc REJECTS "
                        "'-O2' -- use '-O' or '-C opt-level=3', and note that "
                        "its default build is unoptimised, which times the "
                        "debug binary rather than the algorithm."
                        "C only, and only in this image: " + _LLVM_PASSES_HINT
                    ),
                },
                "repeat": {
                    "type": "integer",
                    "description": "Trials to run, 1-10 (default 5); the fastest is reported. The slowest quarter are discarded before the spread is measured, so more trials buy a better precision estimate; the loop stops early if the trials would eat the timeout.",
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
        name="retract_finding",
        description=(
            "Withdraw a finding from an earlier run that this run has shown to be "
            "wrong, so later runs stop recalling it. Requires the evidence that "
            "overturns it, named among the findings THIS run produced -- a "
            "retraction that cites nothing is an opinion with the power to delete "
            "evidence. The finding is withdrawn, not deleted, and the reason is "
            "kept so a reader can tell a number withdrawn for a harness defect "
            "from one withdrawn because the question changed."
        ),
        parameters={
            "type": "object",
            "properties": {
                "ref": {
                    "type": "string",
                    "description": (
                        "The finding to withdraw, as the `ref` field that "
                        "recall_prior_findings reports on every finding it returns "
                        "(`<job_id>#<index>`)."
                    ),
                },
                "reason": {
                    "type": "string",
                    "description": "What is wrong with it, specifically enough to act on.",
                },
                "contradicted_by": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Findings this run produced that overturn the claim, by type "
                        "or title."
                    ),
                },
            },
            "required": ["ref", "reason", "contradicted_by"],
        },
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
        name="verify_run_bundle",
        description="Check an evidence bundle -- this run's by default, or an earlier run's by passing its job_id. Bundles are written as the run "
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
                "job_id": {
                    "type": "string",
                    "description": (
                        "Verify an earlier run's bundle instead of this one's. "
                        "Use the `recalled_from_job` value that "
                        "recall_prior_findings returns with each finding, to "
                        "check the evidence under a number before building on "
                        "it. Only runs belonging to you can be verified."
                    ),
                },
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
    ToolSpec(
        name="check_implementation",
        description=(
            "Run an implementation against reference cases with known outputs, "
            "before timing it. The fastest implementation of any algorithm is "
            "one that returns garbage, so a benchmark of unverified code is an "
            "accurate number for something nobody checked. Pass the SAME "
            "source you will hand to benchmark_c_snippet: verifying one "
            "program and timing another proves nothing about the timing. A "
            "check with no cases is reported as unverified rather than "
            "passing vacuously. The strongest check is an ORACLE: give a "
            "`reference` (the obvious, slow, plainly correct way -- `%`, a "
            "scalar loop, the definition transcribed) and a `generator` that "
            "prints one input per index, and every generated input is run "
            "through both and the outputs compared, with no expected output "
            "written by hand."
        ),
        parameters={
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": (
                        "Self-contained program including main(). It should "
                        "read its input from stdin and print its result to "
                        "stdout."
                    ),
                },
                "language": {
                    "type": "string",
                    "enum": _LANGUAGES,
                    "description": (
                        "Which toolchain builds it (default 'c'). Use the same "
                        "language and the same source when you benchmark it, "
                        "or the program that was checked is not the program "
                        "that was timed. Rust is a single .rs file at edition "
                        "2021 with these crates prebuilt and in scope: "
                        + _RUST_CRATES_HINT
                        + ". Python is a single .py file, standard library "
                        "only; it is syntax-checked before the cases run, so a "
                        "parse error reports as a build failure rather than as "
                        "every case failing."
                    ),
                },
                "cases": {
                    "type": "array",
                    "description": (
                        "Reference cases. Take them from the paper's own "
                        "worked examples where it gives them: a case invented "
                        "to match the implementation checks nothing. Most "
                        "algorithm papers give none -- then pass `reference` "
                        "and `generator` instead of cases: a plainly correct "
                        "implementation and a program printing one input per "
                        "index, compared on every input. Where no plainly "
                        "correct reference exists, check the PROPERTIES the "
                        "paper states (an output range, an invariant, an "
                        "equivalence with the baseline it replaces) over many "
                        "inputs: deriving a property from the paper is reading "
                        "it. A program that only checks itself and prints PASS "
                        "is the weakest form -- a live one printed PASS "
                        "whatever its input -- and a case invented to match "
                        "your code checks nothing. An implementation with no "
                        "way to be checked is one nobody may time."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "input": {
                                "type": "string",
                                "description": (
                                    "Fed to the program on stdin VERBATIM, so "
                                    "write it in the form your program reads "
                                    "-- '1000 7', not a JSON object. A paper's "
                                    "worked example is usually structured; "
                                    "converting it to your program's input "
                                    "format is part of implementing it."
                                ),
                            },
                            "expected_output": {
                                "type": "string",
                                "description": (
                                    "Exactly what a correct program prints, as "
                                    "text. Numbers are compared numerically "
                                    "with a tolerance, so '6' matches '6.0'."
                                ),
                            },
                        },
                        "required": ["expected_output"],
                    },
                },
                "flags": {
                    "type": "string",
                    "description": (
                        "Compiler flags, defaulting per language ('-O2' for C, "
                        "'-O' for Rust). rustc rejects '-O2'."
                    ),
                },
                "tolerance": {
                    "type": "number",
                    "description": (
                        "Relative tolerance for numeric outputs (default "
                        "1e-6). An algorithm rebuilt from prose will not "
                        "match a reference bit for bit."
                    ),
                },
                "reference": {
                    "type": "object",
                    "description": (
                        "The oracle: a second self-contained program reading "
                        "the same stdin and printing what a correct answer "
                        "prints. Write it the obvious way, however slow; it "
                        "must not be the implementation itself. Needs "
                        "`generator` too."
                    ),
                    "properties": {
                        "code": {"type": "string"},
                        "language": {"type": "string", "enum": _LANGUAGES},
                    },
                    "required": ["code"],
                },
                "generator": {
                    "type": "object",
                    "description": (
                        "A program that reads an index (0, 1, 2, ...) on stdin "
                        "and prints ONE input, deterministically from that "
                        "index. Make the first indices the edge cases (0, 1, "
                        "the largest values, powers of two) and seed a random "
                        "generator from the index for the rest. It must vary "
                        "with the index: one input repeated is one case."
                    ),
                    "properties": {
                        "code": {"type": "string"},
                        "language": {"type": "string", "enum": _LANGUAGES},
                        "count": {
                            "type": "integer",
                            "description": "Inputs to generate (default 100, at most 1000).",
                        },
                    },
                    "required": ["code"],
                },
            },
            "required": ["code"],
        },
        effects="write",
        cost_tier="high",
        pii_risk="medium",
        produces=("implementation_verified",),
        requires=("extract_algorithm_spec",),
        typical_seconds=30,
        # Perishable for the same reason a test run is: the implement stage
        # loops, and a check that passed against the code as it stood two
        # rounds ago describes a program that no longer exists. Inheriting it
        # would let an edited implementation carry forward a certificate it
        # earned before the edit -- a gate that inherits its own answer is not
        # a gate.
        perishable=True,
        consumes=(
            "the source to be timed, and reference cases from the paper or a "
            "reference program and an input generator."
        ),
    ),
    ToolSpec(
        name="compare_to_claim",
        description=(
            "Score a measurement against a number the paper claimed, and say "
            "whether the two are comparable at all. Three verdicts, not two: "
            "'incomparable' is a first-class answer, because a 3.0x claimed on "
            "a Xeon at one input size and a 2.1x measured on aarch64 at "
            "another has not failed to reproduce -- those numbers never tested "
            "each other, and calling that a refutation claims a finding the "
            "run did not earn. State the conditions on both sides so this can "
            "tell the cases apart."
        ),
        parameters={
            "type": "object",
            "properties": {
                "subject": {
                    "type": "string",
                    "description": "What was compared, e.g. 'blocked GEMM speedup over naive'",
                },
                "metric": {
                    "type": "string",
                    "description": "The quantity, e.g. 'speedup' or 'runtime'",
                },
                "claimed_value": {
                    "type": "number",
                    "description": "The number the paper reports",
                },
                "claimed_unit": {
                    "type": "string",
                    "description": "Its unit, e.g. 'x', 'percent', 'ms', 'cycles_per_element'",
                },
                "measured_value": {
                    "type": "number",
                    "description": "What this run measured",
                },
                "measured_unit": {
                    "type": "string",
                    "description": "Its unit; a mismatch with the claim is refused, not converted",
                },
                "measurement_source": {
                    "type": "string",
                    "description": (
                        "What produced the measured number, e.g. 'wall clock, "
                        "5 trials, benchmark_c_snippet'. Required: a number "
                        "whose origin is not stated cannot settle a claim."
                    ),
                },
                "claimed_conditions": {
                    "type": "object",
                    "description": (
                        "The conditions the paper's number was taken under: "
                        "{'hardware': 'Xeon 8280', 'input_size': '1e7', "
                        "'baseline': 'naive triple loop'}"
                    ),
                },
                "measured_conditions": {
                    "type": "object",
                    "description": "The same fields for this run's measurement",
                },
                "tolerance": {
                    "type": "number",
                    "description": (
                        "Relative tolerance, default 0.2. Papers report to two "
                        "significant figures on a machine you do not have."
                    ),
                },
            },
            "required": [
                "subject",
                "metric",
                "claimed_value",
                "measured_value",
                "measurement_source",
            ],
        },
        effects="write",
        cost_tier="low",
        produces=("reproduction_verdict",),
        requires=("check_implementation",),
        typical_seconds=5,
        consumes="a claimed number with its conditions, and a measured one with its source.",
    ),
)
