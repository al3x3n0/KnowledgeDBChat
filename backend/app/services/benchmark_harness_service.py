from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.benchmark import BenchmarkBaseline, BenchmarkCase, BenchmarkSuite


BUILTIN_BENCHMARK_SUITES: List[Dict[str, Any]] = [
    {
        "suite": {
            "id": "compiler-llvm-regression-core",
            "name": "LLVM Regression Core",
            "description": "Core compile-regression harness for validating compiler hypotheses against representative LLVM-style kernels.",
            "track_type": "compiler",
            "benchmark_family": "compiler_regression",
            "suite_version": 1,
            "tags": ["compiler", "regression", "llvm"],
            "metadata": {"default_baseline_id": "baseline-llvm-main"},
            "enabled": True,
            "system_managed": True,
        },
        "cases": [
            {
                "id": "case-instcombine-sroa",
                "name": "InstCombine + SROA stress",
                "description": "Compile-focused kernel for scalar replacement and canonicalization regressions.",
                "rank": 1,
                "source_ref": "fixtures/llvm/instcombine_sroa.c",
                "benchmark_query": "instcombine scalar replacement compile regression",
                "compile_command_template": "clang -O3 -S -emit-llvm fixtures/llvm/instcombine_sroa.c -o /tmp/instcombine_sroa.ll",
                "run_command_template": None,
                "expected_artifacts": ["compiler_logs", "ir_or_codegen_artifacts"],
                "metrics": [
                    {"name": "compile_time_ms", "direction": "lower_better"},
                    {"name": "artifact_diff_score", "direction": "lower_better"},
                ],
                "metadata": {
                    "focus": ["instcombine", "sroa", "compile_time"],
                    "observability": {
                        "capture_ir": True,
                        "capture_asm": False,
                        "capture_remarks": True,
                        "capture_compile_logs": True,
                        "capture_perf_stat": False,
                        "repeat_count": 2,
                        "pass_signals": ["instcombine", "sroa"],
                    },
                },
            },
            {
                "id": "case-loop-vectorize-reduction",
                "name": "Loop vectorize reduction",
                "description": "Vectorization-sensitive reduction kernel for compile/codegen comparison.",
                "rank": 2,
                "source_ref": "fixtures/llvm/loop_vectorize_reduction.c",
                "benchmark_query": "loop vectorize reduction codegen quality",
                "compile_command_template": "clang -O3 -Rpass=loop-vectorize fixtures/llvm/loop_vectorize_reduction.c -S -o /tmp/loop_vectorize_reduction.s",
                "run_command_template": "./bench/loop_vectorize_reduction --iters=50",
                "expected_artifacts": ["compiler_logs", "ir_or_codegen_artifacts", "benchmark_output"],
                "metrics": [
                    {"name": "compile_time_ms", "direction": "lower_better"},
                    {"name": "runtime_ms", "direction": "lower_better"},
                    {"name": "binary_size_bytes", "direction": "lower_better"},
                ],
                "metadata": {
                    "focus": ["vectorization", "asm", "runtime"],
                    "observability": {
                        "capture_ir": False,
                        "capture_asm": True,
                        "capture_remarks": True,
                        "capture_compile_logs": True,
                        "capture_perf_stat": True,
                        "repeat_count": 3,
                        "pass_signals": ["loop-vectorize"],
                    },
                },
            },
        ],
        "baselines": [
            {
                "id": "baseline-llvm-main",
                "name": "LLVM main baseline",
                "description": "Mainline compiler baseline for regression comparison.",
                "compiler_revision": "llvm-main",
                "toolchain_id": "clang-18",
                "sandbox_profile_id": "scientific-compiler-sandbox",
                "measurements": {
                    "compile_time_ms": 1280,
                    "runtime_ms": 24.8,
                    "binary_size_bytes": 16384,
                },
                "environment_snapshot": {"os": "ubuntu-22.04", "cpu": "x86_64", "profile": "scientific-compiler-sandbox"},
                "enabled": True,
                "system_managed": True,
            }
        ],
    },
    {
        "suite": {
            "id": "compiler-codegen-quality-pack",
            "name": "Codegen Quality Pack",
            "description": "Assembly and binary-quality oriented suite for validating instruction selection and register-pressure hypotheses.",
            "track_type": "compiler",
            "benchmark_family": "codegen_quality",
            "suite_version": 1,
            "tags": ["compiler", "codegen", "quality"],
            "metadata": {"default_baseline_id": "baseline-codegen-quality"},
            "enabled": True,
            "system_managed": True,
        },
        "cases": [
            {
                "id": "case-regalloc-pressure",
                "name": "Register pressure kernel",
                "description": "Codegen stress case for spills and allocator decisions.",
                "rank": 1,
                "source_ref": "fixtures/compiler/regalloc_pressure.cpp",
                "benchmark_query": "register allocation spill pressure codegen quality",
                "compile_command_template": "clang++ -O3 fixtures/compiler/regalloc_pressure.cpp -S -o /tmp/regalloc_pressure.s",
                "run_command_template": None,
                "expected_artifacts": ["compiler_logs", "ir_or_codegen_artifacts"],
                "metrics": [
                    {"name": "compile_time_ms", "direction": "lower_better"},
                    {"name": "binary_size_bytes", "direction": "lower_better"},
                    {"name": "artifact_diff_score", "direction": "lower_better"},
                ],
                "metadata": {
                    "focus": ["register_allocation", "spills"],
                    "observability": {
                        "capture_ir": False,
                        "capture_asm": True,
                        "capture_remarks": True,
                        "capture_compile_logs": True,
                        "capture_perf_stat": False,
                        "repeat_count": 2,
                        "pass_signals": ["regalloc", "spill-analysis"],
                    },
                },
            },
            {
                "id": "case-isel-simd-hotloop",
                "name": "SIMD hot loop isel",
                "description": "Instruction-selection quality case for tight SIMD loops.",
                "rank": 2,
                "source_ref": "fixtures/compiler/simd_hotloop.c",
                "benchmark_query": "simd hot loop instruction selection quality",
                "compile_command_template": "clang -O3 fixtures/compiler/simd_hotloop.c -S -o /tmp/simd_hotloop.s",
                "run_command_template": "./bench/simd_hotloop --repeat=100",
                "expected_artifacts": ["compiler_logs", "ir_or_codegen_artifacts", "benchmark_output"],
                "metrics": [
                    {"name": "runtime_ms", "direction": "lower_better"},
                    {"name": "binary_size_bytes", "direction": "lower_better"},
                    {"name": "artifact_diff_score", "direction": "lower_better"},
                ],
                "metadata": {
                    "focus": ["isel", "simd", "throughput"],
                    "observability": {
                        "capture_ir": False,
                        "capture_asm": True,
                        "capture_remarks": True,
                        "capture_compile_logs": True,
                        "capture_perf_stat": True,
                        "repeat_count": 3,
                        "pass_signals": ["isel", "simd-hotloop"],
                    },
                },
            },
        ],
        "baselines": [
            {
                "id": "baseline-codegen-quality",
                "name": "Codegen quality baseline",
                "description": "Reference codegen outputs for the quality pack.",
                "compiler_revision": "clang-stable",
                "toolchain_id": "clang-18",
                "sandbox_profile_id": "scientific-compiler-sandbox",
                "measurements": {"binary_size_bytes": 14080, "runtime_ms": 19.4},
                "environment_snapshot": {"profile": "scientific-compiler-sandbox"},
                "enabled": True,
                "system_managed": True,
            }
        ],
    },
    {
        "suite": {
            "id": "compiler-kernel-compile-smoke",
            "name": "Kernel Compile Smoke",
            "description": "Compile-only suite for quickly validating buildability and artifact generation across representative kernels.",
            "track_type": "compiler",
            "benchmark_family": "kernel_compile",
            "suite_version": 1,
            "tags": ["compiler", "kernel", "compile"],
            "metadata": {"default_baseline_id": "baseline-kernel-compile"},
            "enabled": True,
            "system_managed": True,
        },
        "cases": [
            {
                "id": "case-kernel-stencil-2d",
                "name": "Stencil 2D compile",
                "description": "Compile smoke test for cache-aware stencil kernels.",
                "rank": 1,
                "source_ref": "fixtures/compiler/stencil2d_kernel.c",
                "benchmark_query": "stencil kernel compile smoke",
                "compile_command_template": "clang -O3 -c fixtures/compiler/stencil2d_kernel.c -o /tmp/stencil2d_kernel.o",
                "run_command_template": None,
                "expected_artifacts": ["compiler_logs", "benchmark_output"],
                "metrics": [{"name": "compile_time_ms", "direction": "lower_better"}],
                "metadata": {
                    "focus": ["kernel_compile", "locality"],
                    "observability": {
                        "capture_ir": False,
                        "capture_asm": False,
                        "capture_remarks": False,
                        "capture_compile_logs": True,
                        "capture_perf_stat": False,
                        "repeat_count": 1,
                        "pass_signals": ["kernel-compile"],
                    },
                },
            },
            {
                "id": "case-kernel-matmul",
                "name": "Matmul compile",
                "description": "Compile smoke test for tiled matmul kernels.",
                "rank": 2,
                "source_ref": "fixtures/compiler/matmul_kernel.c",
                "benchmark_query": "matmul kernel compile smoke",
                "compile_command_template": "clang -O3 -c fixtures/compiler/matmul_kernel.c -o /tmp/matmul_kernel.o",
                "run_command_template": None,
                "expected_artifacts": ["compiler_logs", "benchmark_output"],
                "metrics": [{"name": "compile_time_ms", "direction": "lower_better"}],
                "metadata": {
                    "focus": ["kernel_compile", "tiling"],
                    "observability": {
                        "capture_ir": False,
                        "capture_asm": False,
                        "capture_remarks": False,
                        "capture_compile_logs": True,
                        "capture_perf_stat": False,
                        "repeat_count": 1,
                        "pass_signals": ["kernel-compile", "tiling"],
                    },
                },
            },
        ],
        "baselines": [
            {
                "id": "baseline-kernel-compile",
                "name": "Kernel compile baseline",
                "description": "Reference compile-time baseline for smoke kernels.",
                "compiler_revision": "clang-stable",
                "toolchain_id": "clang-18",
                "sandbox_profile_id": "scientific-compiler-sandbox",
                "measurements": {"compile_time_ms": 960},
                "environment_snapshot": {"profile": "scientific-compiler-sandbox"},
                "enabled": True,
                "system_managed": True,
            }
        ],
    },
]


def _suite_payload(raw: Dict[str, Any]) -> Dict[str, Any]:
    payload = deepcopy(raw)
    payload["metadata_json"] = payload.pop("metadata", {}) if isinstance(payload.get("metadata"), dict) else {}
    return payload


def _case_payload(raw: Dict[str, Any], *, suite_id: str) -> Dict[str, Any]:
    payload = deepcopy(raw)
    payload["suite_id"] = suite_id
    payload["metadata_json"] = payload.pop("metadata", {}) if isinstance(payload.get("metadata"), dict) else {}
    return payload


def _baseline_payload(raw: Dict[str, Any], *, suite_id: str) -> Dict[str, Any]:
    payload = deepcopy(raw)
    payload["suite_id"] = suite_id
    return payload


async def ensure_builtin_benchmark_suites(db: AsyncSession) -> None:
    for definition in BUILTIN_BENCHMARK_SUITES:
        suite_data = _suite_payload(definition["suite"])
        suite = await db.get(BenchmarkSuite, suite_data["id"])
        if suite is None:
            suite = BenchmarkSuite(**suite_data)
            db.add(suite)
        else:
            for key, value in suite_data.items():
                setattr(suite, key, value)
        await db.flush()

        for case_definition in definition.get("cases", []):
            case_data = _case_payload(case_definition, suite_id=suite.id)
            case = await db.get(BenchmarkCase, case_data["id"])
            if case is None:
                case = BenchmarkCase(**case_data)
                db.add(case)
            else:
                for key, value in case_data.items():
                    setattr(case, key, value)

        for baseline_definition in definition.get("baselines", []):
            baseline_data = _baseline_payload(baseline_definition, suite_id=suite.id)
            baseline = await db.get(BenchmarkBaseline, baseline_data["id"])
            if baseline is None:
                baseline = BenchmarkBaseline(**baseline_data)
                db.add(baseline)
            else:
                for key, value in baseline_data.items():
                    setattr(baseline, key, value)

    await db.flush()


async def list_benchmark_suites(
    db: AsyncSession,
    *,
    track_type: Optional[str] = None,
    include_disabled: bool = False,
) -> List[Dict[str, Any]]:
    await ensure_builtin_benchmark_suites(db)
    stmt = select(BenchmarkSuite).options(
        selectinload(BenchmarkSuite.cases),
        selectinload(BenchmarkSuite.baselines),
    )
    if track_type:
        stmt = stmt.where(BenchmarkSuite.track_type == str(track_type).strip().lower())
    if not include_disabled:
        stmt = stmt.where(BenchmarkSuite.enabled.is_(True))
    stmt = stmt.order_by(BenchmarkSuite.system_managed.desc(), BenchmarkSuite.name.asc())
    rows = list((await db.execute(stmt)).scalars().all())
    return [row.to_dict() for row in rows]


async def get_benchmark_suite(
    db: AsyncSession,
    suite_id: Optional[str],
    *,
    include_disabled: bool = False,
) -> Optional[Dict[str, Any]]:
    await ensure_builtin_benchmark_suites(db)
    requested = str(suite_id or "").strip()
    if not requested:
        return None
    stmt = (
        select(BenchmarkSuite)
        .options(selectinload(BenchmarkSuite.cases), selectinload(BenchmarkSuite.baselines))
        .where(BenchmarkSuite.id == requested)
    )
    row = (await db.execute(stmt)).scalars().first()
    if row is None:
        return None
    if not include_disabled and not bool(row.enabled):
        return None
    return row.to_dict()
