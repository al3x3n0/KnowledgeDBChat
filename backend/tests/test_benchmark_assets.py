from pathlib import Path

from app.services.benchmark_harness_service import BUILTIN_BENCHMARK_SUITES

BACKEND_ROOT = Path(__file__).resolve().parents[1]


def test_builtin_benchmark_source_assets_exist():
    missing = []
    for definition in BUILTIN_BENCHMARK_SUITES:
        for case in definition["cases"]:
            source_ref = case.get("source_ref")
            if source_ref and not (BACKEND_ROOT / source_ref).is_file():
                missing.append(source_ref)

    assert missing == []


def test_compiler_benchmark_builds_the_binary_it_runs():
    compiler_suite = next(
        definition
        for definition in BUILTIN_BENCHMARK_SUITES
        if definition["suite"]["id"] == "compiler-llvm-regression-core"
    )
    runtime_case = next(
        case
        for case in compiler_suite["cases"]
        if case["id"] == "case-loop-vectorize-reduction"
    )

    assert "/tmp/loop_vectorize_reduction" in runtime_case["compile_command_template"]
    assert (
        runtime_case["run_command_template"]
        == "/tmp/loop_vectorize_reduction --iters=50"
    )
