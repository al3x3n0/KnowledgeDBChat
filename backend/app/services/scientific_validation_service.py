from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.experiment import ExperimentRun
from app.models.research_note import ResearchNote
from app.models.scientific_sandbox_profile import ScientificSandboxProfile
from app.models.synthesis_job import SynthesisJob

DEFAULT_VALIDATION_BACKOFF_POLICY: Dict[str, Any] = {
    "max_consecutive_failures": 2,
    "cooldown_minutes": 180,
}


DEFAULT_VALIDATION_POLICY: Dict[str, Any] = {
    "confidence_threshold": 0.72,
    "experiment_readiness_threshold": 0.8,
    "max_auto_follow_up_launches": 2,
    "auto_create_experiment_plans": True,
    "auto_launch_follow_up": True,
    "auto_execute_validation_runs": False,
    "max_concurrent_validation_runs": 1,
    "max_validation_runtime_minutes": 20,
    "max_validation_budget_per_run": 25.0,
    "validation_backoff_policy": DEFAULT_VALIDATION_BACKOFF_POLICY,
}


DEFAULT_PORTFOLIO_AUTOMATION_POLICY: Dict[str, Any] = {
    **DEFAULT_VALIDATION_POLICY,
    "duplicate_window_items": 60,
    "auto_launch_experiment_runs": False,
    "follow_up_review_mode": "queue_for_approval",
}


DEFAULT_MAX_AUTONOMY_PORTFOLIO_AUTOMATION_POLICY: Dict[str, Any] = {
    **DEFAULT_VALIDATION_POLICY,
    "confidence_threshold": 0.68,
    "experiment_readiness_threshold": 0.72,
    "max_auto_follow_up_launches": 4,
    "auto_execute_validation_runs": True,
    "max_concurrent_validation_runs": 2,
    "max_validation_runtime_minutes": 30,
    "max_validation_budget_per_run": 50.0,
    "duplicate_window_items": 120,
    "auto_launch_experiment_runs": True,
    "follow_up_review_mode": "auto_launch_safe",
}


BUILTIN_SCIENTIFIC_SANDBOX_PROFILES: Dict[str, Dict[str, Any]] = {
    "scientific-compiler-sandbox": {
        "id": "scientific-compiler-sandbox",
        "name": "Compiler Validation Sandbox",
        "description": "Docker-isolated compiler research sandbox for compile/codegen/regression validation.",
        "track_type": "compiler",
        "backend": "docker",
        "docker_image": "ghcr.io/al3x3n0/kdbc-compiler-research:latest",
        "timeout_seconds": 1200,
        "resource_caps": {"memory_mb": 4096, "cpus": 2.0, "pids_limit": 256},
        "allowed_benchmark_families": [
            "compiler_regression",
            "codegen_quality",
            "kernel_compile",
        ],
        "allowed_perf_collectors": [
            "benchmark_output",
            "compile_time",
            "artifact_diff",
            "perf_stat",
        ],
        "required_capabilities": ["repo_reconstruction"],
        "toolchains": ["clang", "llvm-opt", "cmake", "ninja", "pytest"],
        "budget_limit_default": 35.0,
        "enabled": True,
        "system_managed": True,
        "is_default": True,
    },
    "scientific-microarchitecture-sandbox": {
        "id": "scientific-microarchitecture-sandbox",
        "name": "Microarchitecture Validation Sandbox",
        "description": "Docker-isolated sandbox for perf-counter and benchmark-based microarchitecture validation.",
        "track_type": "microarchitecture",
        "backend": "docker",
        "docker_image": "ghcr.io/al3x3n0/kdbc-microarch-research:latest",
        "timeout_seconds": 1200,
        "resource_caps": {"memory_mb": 4096, "cpus": 2.0, "pids_limit": 256},
        "allowed_benchmark_families": [
            "perf_counter_regression",
            "cache_branch_analysis",
            "throughput_latency",
        ],
        "allowed_perf_collectors": [
            "perf_stat",
            "cache_miss",
            "branch_miss",
            "benchmark_output",
        ],
        "required_capabilities": ["repo_reconstruction", "perf_counters"],
        "toolchains": ["python", "pytest", "perf"],
        "budget_limit_default": 40.0,
        "enabled": True,
        "system_managed": True,
        "is_default": True,
    },
    "scientific-generic-sandbox": {
        "id": "scientific-generic-sandbox",
        "name": "Scientific Validation Sandbox",
        "description": "Default docker-isolated sandbox for bounded technical validation runs.",
        "track_type": "generic",
        "backend": "docker",
        "docker_image": "python:3.11-slim",
        "timeout_seconds": 900,
        "resource_caps": {"memory_mb": 2048, "cpus": 1.5, "pids_limit": 192},
        "allowed_benchmark_families": ["generic_validation"],
        "allowed_perf_collectors": ["benchmark_output"],
        "required_capabilities": ["repo_reconstruction"],
        "toolchains": ["python", "pytest"],
        "budget_limit_default": 25.0,
        "enabled": True,
        "system_managed": True,
        "is_default": True,
    },
}


SCIENTIFIC_VALIDATION_RECIPE_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "compiler": {
        "recipe_id": "compiler_validation_v1",
        "recipe_version": 1,
        "recipe_family": "compiler_validation",
        "benchmark_family": "compiler_regression",
        "allowed_perf_collectors": [
            "benchmark_output",
            "compile_time",
            "artifact_diff",
            "perf_stat",
        ],
        "required_capabilities": ["repo_reconstruction"],
        "artifact_collection_rules": [
            "compiler_logs",
            "benchmark_output",
            "ir_or_codegen_artifacts",
            "compiler_remarks",
            "perf_counter_summary",
        ],
        "compiler_observability_defaults": {
            "capture_ir": True,
            "capture_asm": True,
            "capture_remarks": True,
            "capture_compile_logs": True,
            "capture_perf_stat": False,
            "repeat_count": 1,
            "artifact_kinds": [
                "compiler_logs",
                "ir_or_codegen_artifacts",
                "compiler_remarks",
            ],
        },
        "verification_prefixes": [
            "pytest",
            "python -m pytest",
            "cmake",
            "ninja",
            "ctest",
            "make",
            "cargo test",
            "cargo bench",
            "clang",
            "clang++",
            "llvm-lit",
            "opt",
            "llc",
        ],
        "bootstrap_prefixes": [
            "pip install",
            "python -m pip install",
            "npm ci",
            "npm install",
            "pnpm install",
            "yarn install",
            "cargo fetch",
            "cargo build",
            "cmake",
            "ninja",
            "make",
        ],
        "fallback_prefixes": [
            "pytest",
            "python -m pytest",
            "ctest",
            "make test",
            "cargo test",
            "cargo bench",
            "ninja test",
        ],
        "success_criteria": [
            "Compile or test commands complete successfully in the sandbox.",
            "Compile-time, benchmark, or artifact deltas are measurable against baseline.",
            "Collected artifacts support or falsify the hypothesis without mutating customer code.",
        ],
        "baseline_comparison": {
            "type": "compile_and_benchmark",
            "focus": ["compile_time", "codegen_quality", "regression_surface"],
        },
    },
    "microarchitecture": {
        "recipe_id": "microarchitecture_validation_v1",
        "recipe_version": 1,
        "recipe_family": "microarchitecture_validation",
        "benchmark_family": "perf_counter_regression",
        "allowed_perf_collectors": [
            "perf_stat",
            "cache_miss",
            "branch_miss",
            "benchmark_output",
        ],
        "required_capabilities": ["repo_reconstruction", "perf_counters"],
        "artifact_collection_rules": [
            "benchmark_output",
            "perf_counter_samples",
            "latency_throughput_summary",
        ],
        "verification_prefixes": [
            "perf stat",
            "pytest",
            "python -m pytest",
            "cargo test",
            "cargo bench",
            "ctest",
            "make",
            "python",
        ],
        "bootstrap_prefixes": [
            "pip install",
            "python -m pip install",
            "npm ci",
            "npm install",
            "cargo fetch",
            "cargo build",
            "cmake",
            "ninja",
            "make",
        ],
        "fallback_prefixes": [
            "perf stat",
            "pytest",
            "python -m pytest",
            "cargo bench",
            "ctest",
            "make",
            "python",
        ],
        "success_criteria": [
            "Compile or test commands complete successfully in the sandbox.",
            "Perf-counter or benchmark samples are collected for the target workload.",
            "Cache, branch, SIMD, or throughput deltas are measurable against baseline.",
        ],
        "baseline_comparison": {
            "type": "benchmark_and_perf_counter",
            "focus": ["ipc", "cache_behavior", "branch_behavior", "throughput_latency"],
        },
    },
    "generic": {
        "recipe_id": "generic_validation_v1",
        "recipe_version": 1,
        "recipe_family": "generic_validation",
        "benchmark_family": "generic_validation",
        "allowed_perf_collectors": ["benchmark_output"],
        "required_capabilities": ["repo_reconstruction"],
        "artifact_collection_rules": ["benchmark_output"],
        "verification_prefixes": [
            "pytest",
            "python -m pytest",
            "python",
            "make",
            "ctest",
            "cargo test",
        ],
        "bootstrap_prefixes": [
            "pip install",
            "python -m pip install",
            "npm ci",
            "npm install",
            "cargo fetch",
            "cargo build",
            "make",
        ],
        "fallback_prefixes": [
            "pytest",
            "python -m pytest",
            "python",
            "make",
            "ctest",
            "cargo test",
        ],
        "success_criteria": [
            "Validation commands complete successfully in the sandbox.",
            "Observed outputs provide bounded evidence for or against the hypothesis.",
        ],
        "baseline_comparison": {
            "type": "generic_validation",
            "focus": ["command_success", "benchmark_output"],
        },
    },
}


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _clean_string(value: Any) -> str:
    return str(value or "").strip()


def _clean_string_list(value: Any, *, limit: int = 24) -> List[str]:
    if not isinstance(value, list):
        return []
    out: List[str] = []
    seen: set[str] = set()
    for item in value:
        text = _clean_string(item)
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
        if len(out) >= limit:
            break
    return out


def _parse_csv_setting(value: str) -> List[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def get_scientific_validation_runtime_limits() -> Dict[str, Any]:
    return {
        "allowed_docker_images": _parse_csv_setting(
            settings.SCIENTIFIC_VALIDATION_ALLOWED_DOCKER_IMAGES
        ),
        "allowed_capabilities": _parse_csv_setting(
            settings.SCIENTIFIC_VALIDATION_ALLOWED_CAPABILITIES
        ),
        "allowed_benchmark_families": _parse_csv_setting(
            settings.SCIENTIFIC_VALIDATION_ALLOWED_BENCHMARK_FAMILIES
        ),
        "allowed_perf_collectors": _parse_csv_setting(
            settings.SCIENTIFIC_VALIDATION_ALLOWED_PERF_COLLECTORS
        ),
        "max_timeout_seconds": max(
            30,
            int(
                getattr(settings, "SCIENTIFIC_VALIDATION_MAX_TIMEOUT_SECONDS", 1800)
                or 1800
            ),
        ),
        "max_memory_mb": max(
            128,
            int(getattr(settings, "SCIENTIFIC_VALIDATION_MAX_MEMORY_MB", 8192) or 8192),
        ),
        "max_cpus": max(
            0.25, float(getattr(settings, "SCIENTIFIC_VALIDATION_MAX_CPUS", 8.0) or 8.0)
        ),
        "max_pids_limit": max(
            32,
            int(
                getattr(settings, "SCIENTIFIC_VALIDATION_MAX_PIDS_LIMIT", 1024) or 1024
            ),
        ),
        "max_budget_per_run": max(
            1.0,
            float(
                getattr(settings, "SCIENTIFIC_VALIDATION_MAX_BUDGET_PER_RUN", 10000.0)
                or 10000.0
            ),
        ),
    }


def normalize_validation_backoff_policy(value: Any) -> Dict[str, Any]:
    raw = value if isinstance(value, dict) else {}
    return {
        "max_consecutive_failures": max(
            1, min(_safe_int(raw.get("max_consecutive_failures"), 2), 10)
        ),
        "cooldown_minutes": max(
            5, min(_safe_int(raw.get("cooldown_minutes"), 180), 10080)
        ),
    }


def normalize_validation_policy(value: Any) -> Dict[str, Any]:
    raw = value if isinstance(value, dict) else {}
    return {
        "confidence_threshold": max(
            0.0, min(_safe_float(raw.get("confidence_threshold"), 0.72), 1.0)
        ),
        "experiment_readiness_threshold": max(
            0.0, min(_safe_float(raw.get("experiment_readiness_threshold"), 0.8), 1.0)
        ),
        "max_auto_follow_up_launches": max(
            0, min(_safe_int(raw.get("max_auto_follow_up_launches"), 2), 10)
        ),
        "auto_create_experiment_plans": bool(
            raw.get("auto_create_experiment_plans", True)
        ),
        "auto_launch_follow_up": bool(raw.get("auto_launch_follow_up", True)),
        "auto_execute_validation_runs": bool(
            raw.get(
                "auto_execute_validation_runs",
                raw.get("auto_launch_experiment_runs", False),
            )
        ),
        "max_concurrent_validation_runs": max(
            1, min(_safe_int(raw.get("max_concurrent_validation_runs"), 1), 8)
        ),
        "max_validation_runtime_minutes": max(
            5, min(_safe_int(raw.get("max_validation_runtime_minutes"), 20), 240)
        ),
        "max_validation_budget_per_run": round(
            max(
                1.0,
                min(
                    _safe_float(raw.get("max_validation_budget_per_run"), 25.0), 10000.0
                ),
            ),
            2,
        ),
        "validation_backoff_policy": normalize_validation_backoff_policy(
            raw.get("validation_backoff_policy")
        ),
    }


def normalize_portfolio_automation_policy(value: Any) -> Dict[str, Any]:
    raw = value if isinstance(value, dict) else {}
    normalized = normalize_validation_policy(raw)
    normalized["duplicate_window_items"] = max(
        1, min(_safe_int(raw.get("duplicate_window_items"), 60), 500)
    )
    normalized["auto_launch_experiment_runs"] = bool(
        raw.get(
            "auto_launch_experiment_runs",
            normalized.get("auto_execute_validation_runs", False),
        )
    )
    normalized["auto_execute_validation_runs"] = normalized[
        "auto_launch_experiment_runs"
    ]
    follow_up_review_mode = (
        _clean_string(raw.get("follow_up_review_mode"))
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )
    if follow_up_review_mode not in {
        "auto_launch_safe",
        "queue_for_approval",
        "manual_only",
    }:
        follow_up_review_mode = "queue_for_approval"
    normalized["follow_up_review_mode"] = follow_up_review_mode
    return normalized


def normalize_portfolio_automation_profile(
    value: Any, *, default: str = "balanced"
) -> str:
    text = _clean_string(value).lower()
    if text in {"max_autonomy", "max-autonomy", "max"}:
        return "max_autonomy"
    if text in {"balanced", "default", "standard"}:
        return "balanced"
    return default


def resolve_portfolio_automation_policy(
    automation_profile: Any,
    value: Any,
) -> Dict[str, Any]:
    profile = normalize_portfolio_automation_profile(automation_profile)
    base_policy = (
        DEFAULT_MAX_AUTONOMY_PORTFOLIO_AUTOMATION_POLICY
        if profile == "max_autonomy"
        else DEFAULT_PORTFOLIO_AUTOMATION_POLICY
    )
    raw = value if isinstance(value, dict) else {}
    return normalize_portfolio_automation_policy(
        {
            **deepcopy(base_policy),
            **raw,
        }
    )


def _normalize_scientific_sandbox_profile_payload(
    value: Dict[str, Any]
) -> Dict[str, Any]:
    raw = value if isinstance(value, dict) else {}
    resource_caps = (
        raw.get("resource_caps") if isinstance(raw.get("resource_caps"), dict) else {}
    )
    return {
        "id": _clean_string(raw.get("id")),
        "name": _clean_string(raw.get("name")),
        "description": _clean_string(raw.get("description")) or None,
        "track_type": _clean_string(raw.get("track_type") or "generic").lower()
        or "generic",
        "backend": _clean_string(raw.get("backend") or "docker").lower() or "docker",
        "docker_image": _clean_string(raw.get("docker_image")) or None,
        "timeout_seconds": max(
            30, min(_safe_int(raw.get("timeout_seconds"), 900), 7200)
        ),
        "resource_caps": {
            "memory_mb": max(
                128, min(_safe_int(resource_caps.get("memory_mb"), 2048), 65536)
            ),
            "cpus": max(0.25, min(_safe_float(resource_caps.get("cpus"), 1.0), 64.0)),
            "pids_limit": max(
                32, min(_safe_int(resource_caps.get("pids_limit"), 128), 8192)
            ),
        },
        "allowed_benchmark_families": _clean_string_list(
            raw.get("allowed_benchmark_families"), limit=16
        ),
        "allowed_perf_collectors": _clean_string_list(
            raw.get("allowed_perf_collectors"), limit=16
        ),
        "required_capabilities": _clean_string_list(
            raw.get("required_capabilities"), limit=16
        ),
        "toolchains": _clean_string_list(raw.get("toolchains"), limit=24),
        "budget_limit_default": round(
            max(1.0, min(_safe_float(raw.get("budget_limit_default"), 25.0), 10000.0)),
            2,
        ),
        "enabled": bool(raw.get("enabled", True)),
        "system_managed": bool(raw.get("system_managed", False)),
        "is_default": bool(raw.get("is_default", False)),
    }


def validate_scientific_sandbox_profile_payload(
    value: Dict[str, Any],
    *,
    allow_system_managed: bool = False,
) -> Dict[str, Any]:
    normalized = _normalize_scientific_sandbox_profile_payload(value)
    limits = get_scientific_validation_runtime_limits()
    allowed_docker_images = set(limits["allowed_docker_images"])
    allowed_capabilities = set(limits["allowed_capabilities"])
    allowed_benchmark_families = set(limits["allowed_benchmark_families"])
    allowed_perf_collectors = set(limits["allowed_perf_collectors"])

    if not normalized["id"]:
        raise ValueError("Profile id is required")
    if not normalized["name"]:
        raise ValueError("Profile name is required")
    if normalized["backend"] not in {"docker", "subprocess"}:
        raise ValueError("Unsupported backend")
    if normalized["system_managed"] and not allow_system_managed:
        raise ValueError(
            "System-managed profiles cannot be created or modified via this API"
        )
    if normalized["backend"] == "docker":
        if not normalized["docker_image"]:
            raise ValueError("Docker image is required for docker-backed profiles")
        if normalized["docker_image"] not in allowed_docker_images:
            raise ValueError(
                "Docker image is not in the scientific validation allowlist"
            )
    else:
        normalized["docker_image"] = None

    if normalized["resource_caps"]["memory_mb"] > limits["max_memory_mb"]:
        raise ValueError("Memory cap exceeds the scientific validation runtime ceiling")
    if normalized["resource_caps"]["cpus"] > limits["max_cpus"]:
        raise ValueError("CPU cap exceeds the scientific validation runtime ceiling")
    if normalized["resource_caps"]["pids_limit"] > limits["max_pids_limit"]:
        raise ValueError("PID cap exceeds the scientific validation runtime ceiling")
    if normalized["timeout_seconds"] > limits["max_timeout_seconds"]:
        raise ValueError("Timeout exceeds the scientific validation runtime ceiling")
    if normalized["budget_limit_default"] > limits["max_budget_per_run"]:
        raise ValueError("Budget exceeds the scientific validation runtime ceiling")

    unsupported_capabilities = sorted(
        set(normalized["required_capabilities"]) - allowed_capabilities
    )
    if unsupported_capabilities:
        raise ValueError(
            f"Unsupported required capabilities: {', '.join(unsupported_capabilities)}"
        )
    unsupported_benchmarks = sorted(
        set(normalized["allowed_benchmark_families"]) - allowed_benchmark_families
    )
    if unsupported_benchmarks:
        raise ValueError(
            f"Unsupported benchmark families: {', '.join(unsupported_benchmarks)}"
        )
    unsupported_collectors = sorted(
        set(normalized["allowed_perf_collectors"]) - allowed_perf_collectors
    )
    if unsupported_collectors:
        raise ValueError(
            f"Unsupported perf collectors: {', '.join(unsupported_collectors)}"
        )

    return normalized


async def ensure_builtin_scientific_sandbox_profiles(db: AsyncSession) -> None:
    for profile_id, definition in BUILTIN_SCIENTIFIC_SANDBOX_PROFILES.items():
        existing = await db.get(ScientificSandboxProfile, profile_id)
        if existing is not None:
            continue
        normalized = validate_scientific_sandbox_profile_payload(
            definition, allow_system_managed=True
        )
        db.add(ScientificSandboxProfile(**normalized))
    await db.flush()


async def list_scientific_sandbox_profiles(
    db: AsyncSession,
    *,
    include_disabled: bool = False,
) -> List[Dict[str, Any]]:
    await ensure_builtin_scientific_sandbox_profiles(db)
    stmt = select(ScientificSandboxProfile)
    if not include_disabled:
        stmt = stmt.where(ScientificSandboxProfile.enabled.is_(True))
    stmt = stmt.order_by(
        ScientificSandboxProfile.system_managed.desc(),
        ScientificSandboxProfile.name.asc(),
    )
    rows = list((await db.execute(stmt)).scalars().all())
    return [row.to_dict() for row in rows]


async def get_scientific_sandbox_profile(
    db: AsyncSession,
    profile_id: Optional[str],
    *,
    track_type: str = "generic",
    include_disabled: bool = False,
) -> Optional[Dict[str, Any]]:
    await ensure_builtin_scientific_sandbox_profiles(db)
    requested = _clean_string(profile_id)
    if requested:
        row = await db.get(ScientificSandboxProfile, requested)
        if row is None:
            return None
        if not include_disabled and not bool(row.enabled):
            return None
        return row.to_dict()

    normalized_track = _clean_string(track_type or "generic").lower() or "generic"
    stmt = (
        select(ScientificSandboxProfile)
        .where(
            ScientificSandboxProfile.track_type == normalized_track,
            ScientificSandboxProfile.is_default.is_(True),
        )
        .order_by(
            ScientificSandboxProfile.system_managed.desc(),
            ScientificSandboxProfile.name.asc(),
        )
    )
    if not include_disabled:
        stmt = stmt.where(ScientificSandboxProfile.enabled.is_(True))
    row = (await db.execute(stmt)).scalars().first()
    if row is not None:
        return row.to_dict()

    fallback = await db.get(ScientificSandboxProfile, "scientific-generic-sandbox")
    if fallback is not None and (include_disabled or bool(fallback.enabled)):
        return fallback.to_dict()
    return None


def _normalize_command_for_match(command: str) -> str:
    text = _clean_string(command)
    while text and "=" in text.split(" ", 1)[0]:
        parts = text.split(" ", 1)
        if len(parts) == 1:
            return ""
        text = parts[1].strip()
    return text.lower()


def _filter_commands_by_prefixes(
    commands: List[str], allowed_prefixes: List[str], *, limit: int
) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for command in commands:
        text = _clean_string(command)
        if not text or text in seen:
            continue
        normalized = _normalize_command_for_match(text)
        if not normalized:
            continue
        if not any(
            normalized == prefix or normalized.startswith(f"{prefix} ")
            for prefix in allowed_prefixes
        ):
            continue
        seen.add(text)
        out.append(text)
        if len(out) >= limit:
            break
    return out


def evaluate_scientific_validation_capabilities(
    *,
    source_id: Optional[str],
    sandbox_profile: Dict[str, Any],
    recipe: Dict[str, Any],
) -> Dict[str, Any]:
    required = _clean_string_list(recipe.get("required_capabilities"), limit=16)
    satisfied: List[str] = []
    missing: List[str] = []
    profile_caps = set(
        _clean_string_list(sandbox_profile.get("required_capabilities"), limit=16)
    )
    toolchains = set(_clean_string_list(sandbox_profile.get("toolchains"), limit=24))
    backend = _clean_string(sandbox_profile.get("backend")).lower() or "docker"

    for capability in required:
        if capability == "repo_reconstruction":
            if _clean_string(source_id):
                satisfied.append(capability)
            else:
                missing.append(capability)
            continue
        if capability == "perf_counters":
            if (
                capability in profile_caps
                and backend == "docker"
                and (
                    "perf" in toolchains
                    or "perf_stat"
                    in set(
                        _clean_string_list(
                            sandbox_profile.get("allowed_perf_collectors"), limit=16
                        )
                    )
                )
            ):
                satisfied.append(capability)
            else:
                missing.append(capability)
            continue
        missing.append(capability)

    return {
        "ok": not missing,
        "required": required,
        "satisfied": satisfied,
        "missing": missing,
    }


def build_scientific_validation_recipe(
    *,
    track_type: str,
    objective: str,
    hypothesis_title: str,
    hypothesis_text: str,
    benchmark_queries: Optional[List[str]] = None,
    verification_commands: Optional[List[str]] = None,
    bootstrap_commands: Optional[List[str]] = None,
    fallback_commands: Optional[List[str]] = None,
    supporting_evidence: Optional[List[str]] = None,
    supporting_sources: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    normalized_track = _clean_string(track_type or "generic").lower() or "generic"
    definition = deepcopy(
        SCIENTIFIC_VALIDATION_RECIPE_DEFINITIONS.get(normalized_track)
        or SCIENTIFIC_VALIDATION_RECIPE_DEFINITIONS["generic"]
    )
    benchmarks = _clean_string_list(benchmark_queries, limit=8)
    evidence = _clean_string_list(supporting_evidence, limit=8)
    sources = [
        dict(item) for item in (supporting_sources or []) if isinstance(item, dict)
    ][:8]

    compiled_commands = _filter_commands_by_prefixes(
        [
            _clean_string(item)
            for item in (verification_commands or [])
            if _clean_string(item)
        ],
        definition.get("verification_prefixes", []),
        limit=4,
    )
    compiled_bootstrap_commands = _filter_commands_by_prefixes(
        [
            _clean_string(item)
            for item in (bootstrap_commands or [])
            if _clean_string(item)
        ],
        definition.get("bootstrap_prefixes", []),
        limit=4,
    )
    compiled_fallback_commands = _filter_commands_by_prefixes(
        [
            _clean_string(item)
            for item in (fallback_commands or [])
            if _clean_string(item)
        ],
        definition.get("fallback_prefixes", []),
        limit=4,
    )

    decision_summary = (
        f"Validate {normalized_track} hypothesis '{hypothesis_title}' in a bounded scientific sandbox. "
        f"Objective: {objective}"
    ).strip()

    return {
        "recipe_id": definition["recipe_id"],
        "recipe_version": int(definition["recipe_version"]),
        "recipe_family": definition["recipe_family"],
        "benchmark_family": definition["benchmark_family"],
        "allowed_perf_collectors": definition.get("allowed_perf_collectors", []),
        "required_capabilities": definition.get("required_capabilities", []),
        "commands": compiled_commands,
        "bootstrap_commands": compiled_bootstrap_commands,
        "fallback_commands": compiled_fallback_commands,
        "benchmark_queries": benchmarks,
        "artifact_collection_rules": definition.get("artifact_collection_rules", []),
        "success_criteria": definition.get("success_criteria", []),
        "decision_summary": decision_summary[:2000],
        "baseline_comparison": definition.get("baseline_comparison", {}),
        "hypothesis": {
            "title": hypothesis_title[:240],
            "statement": hypothesis_text[:2000],
        },
        "supporting_evidence": evidence,
        "supporting_sources": sources,
        "command_policy": {
            "verification_prefixes": definition.get("verification_prefixes", []),
            "bootstrap_prefixes": definition.get("bootstrap_prefixes", []),
            "fallback_prefixes": definition.get("fallback_prefixes", []),
        },
    }


def build_scientific_validation_run_summary(run: ExperimentRun) -> Dict[str, Any]:
    config = run.config if isinstance(run.config, dict) else {}
    scientific_validation = (
        config.get("scientific_validation")
        if isinstance(config.get("scientific_validation"), dict)
        else {}
    )
    execution_handoff = (
        config.get("execution_handoff")
        if isinstance(config.get("execution_handoff"), dict)
        else {}
    )
    operator_actions = (
        scientific_validation.get("operator_actions")
        if isinstance(scientific_validation.get("operator_actions"), list)
        else []
    )
    latest_operator_action = next(
        (item for item in reversed(operator_actions) if isinstance(item, dict)),
        {},
    )
    profile_snapshot = (
        scientific_validation.get("profile_snapshot")
        if isinstance(scientific_validation.get("profile_snapshot"), dict)
        else {}
    )
    return {
        "id": run.id,
        "agent_job_id": run.agent_job_id,
        "name": run.name,
        "status": run.status,
        "progress": int(run.progress or 0),
        "validation_kind": _clean_string(scientific_validation.get("validation_kind"))
        or None,
        "sandbox_profile_id": _clean_string(
            scientific_validation.get("sandbox_profile_id")
        )
        or None,
        "sandbox_profile_name": _clean_string(profile_snapshot.get("name")) or None,
        "recipe_family": _clean_string(scientific_validation.get("recipe_family"))
        or None,
        "recipe_id": _clean_string(scientific_validation.get("recipe_id")) or None,
        "benchmark_family": _clean_string(
            scientific_validation.get("benchmark_family")
            or execution_handoff.get("benchmark_family")
        )
        or None,
        "benchmark_suite_id": _clean_string(
            scientific_validation.get("benchmark_suite_id")
            or execution_handoff.get("benchmark_suite_id")
        )
        or None,
        "benchmark_case_ids": _clean_string_list(
            scientific_validation.get("benchmark_case_ids")
            if isinstance(scientific_validation.get("benchmark_case_ids"), list)
            else execution_handoff.get("benchmark_case_ids")
        ),
        "blocked_reason_code": _clean_string(
            scientific_validation.get("blocked_reason_code")
            or scientific_validation.get("blocked_reason")
        )
        or None,
        "hypothesis_id": _clean_string(scientific_validation.get("hypothesis_id"))
        or None,
        "track_type": _clean_string(scientific_validation.get("track_type")) or None,
        "domain_research_profile_id": _clean_string(
            scientific_validation.get("domain_research_profile_id")
        )
        or None,
        "research_portfolio_id": _clean_string(
            scientific_validation.get("research_portfolio_id")
        )
        or None,
        "parent_run_id": run.parent_run_id,
        "latest_child_run_id": run.latest_child_run_id,
        "retry_count": int(run.retry_count or 0),
        "latest_operator_action": _clean_string(latest_operator_action.get("action"))
        or None,
        "latest_operator_outcome_status": _clean_string(
            latest_operator_action.get("outcome_status")
        )
        or None,
        "created_at": run.created_at,
        "started_at": run.started_at,
        "completed_at": run.completed_at,
    }


def _compiler_source_run_ids_for_run(
    run: ExperimentRun,
) -> tuple[list[str], Optional[str], Optional[str]]:
    config = run.config if isinstance(run.config, dict) else {}
    scientific_validation = (
        config.get("scientific_validation")
        if isinstance(config.get("scientific_validation"), dict)
        else {}
    )
    execution_handoff = (
        config.get("execution_handoff")
        if isinstance(config.get("execution_handoff"), dict)
        else {}
    )
    source_run_ids = _clean_string_list(
        scientific_validation.get("source_run_ids")
        if isinstance(scientific_validation.get("source_run_ids"), list)
        else execution_handoff.get("source_run_ids")
    )
    primary_run_id = (
        _clean_string(
            scientific_validation.get("primary_run_id")
            or execution_handoff.get("primary_run_id")
            or run.id
        )
        or None
    )
    comparison_run_id = (
        _clean_string(
            scientific_validation.get("comparison_run_id")
            or execution_handoff.get("comparison_run_id")
            or run.parent_run_id
            or run.latest_child_run_id
        )
        or None
    )
    deduped: list[str] = []
    for value in [*(source_run_ids or []), primary_run_id, comparison_run_id]:
        text = _clean_string(value)
        if not text or text in deduped:
            continue
        deduped.append(text)
    if primary_run_id and comparison_run_id:
        pair = [primary_run_id, comparison_run_id]
    else:
        pair = deduped[:2]
        if pair:
            primary_run_id = pair[0]
        if len(pair) > 1:
            comparison_run_id = pair[1]
    return pair[:2], primary_run_id, comparison_run_id


async def _attach_compiler_artifact_summaries(
    db: AsyncSession,
    *,
    user_id: Any,
    runs: list[ExperimentRun],
    summaries: list[Dict[str, Any]],
) -> list[Dict[str, Any]]:
    if not runs or not summaries:
        return summaries

    notes = list(
        (
            await db.execute(
                select(ResearchNote)
                .where(ResearchNote.user_id == user_id)
                .order_by(ResearchNote.updated_at.desc())
                .limit(400)
            )
        )
        .scalars()
        .all()
    )
    jobs = list(
        (
            await db.execute(
                select(SynthesisJob)
                .where(
                    SynthesisJob.user_id == user_id,
                    SynthesisJob.job_type.in_(
                        [
                            "compiler_regression_explanation",
                            "compiler_patch_proposal",
                            "compiler_patch_draft",
                        ]
                    ),
                )
                .order_by(SynthesisJob.created_at.desc())
                .limit(400)
            )
        )
        .scalars()
        .all()
    )

    note_payloads: list[tuple[ResearchNote, dict[str, Any]]] = [
        (
            note,
            note.structured_payload
            if isinstance(note.structured_payload, dict)
            else {},
        )
        for note in notes
    ]

    for run, summary in zip(runs, summaries):
        track_type = _clean_string(summary.get("track_type")).lower()
        if track_type != "compiler":
            continue
        (
            source_run_ids,
            primary_run_id,
            comparison_run_id,
        ) = _compiler_source_run_ids_for_run(run)
        benchmark_family = _clean_string(summary.get("benchmark_family"))
        benchmark_suite_id = _clean_string(summary.get("benchmark_suite_id"))

        explanation_note: Optional[ResearchNote] = None
        proposal_note: Optional[ResearchNote] = None
        patch_draft_note: Optional[ResearchNote] = None
        explanation_note_ids: list[str] = []
        proposal_note_ids: list[str] = []

        for note, payload in note_payloads:
            artifact_type = _clean_string(payload.get("artifact_type"))
            if artifact_type == "compiler_regression_explanation":
                payload_source_run_ids = _clean_string_list(
                    payload.get("source_run_ids")
                )
                if set(source_run_ids).intersection(set(payload_source_run_ids or [])):
                    if explanation_note is None:
                        explanation_note = note
                    explanation_note_ids.append(str(note.id))
        for note, payload in note_payloads:
            artifact_type = _clean_string(payload.get("artifact_type"))
            if artifact_type == "compiler_patch_proposal":
                source_explanation_note_id = _clean_string(
                    payload.get("source_explanation_note_id")
                )
                if (
                    explanation_note_ids
                    and source_explanation_note_id in explanation_note_ids
                ):
                    if proposal_note is None:
                        proposal_note = note
                    proposal_note_ids.append(str(note.id))
        for note, payload in note_payloads:
            artifact_type = _clean_string(payload.get("artifact_type"))
            if artifact_type == "compiler_patch_draft":
                source_proposal_note_id = _clean_string(
                    payload.get("source_proposal_note_id")
                )
                if (
                    proposal_note_ids
                    and source_proposal_note_id in proposal_note_ids
                    and patch_draft_note is None
                ):
                    patch_draft_note = note

        explanation_job: Optional[SynthesisJob] = None
        proposal_job: Optional[SynthesisJob] = None
        patch_draft_job: Optional[SynthesisJob] = None
        for job in jobs:
            options = job.options if isinstance(job.options, dict) else {}
            metadata = (
                job.result_metadata if isinstance(job.result_metadata, dict) else {}
            )
            if str(job.job_type or "").strip() == "compiler_regression_explanation":
                job_run_ids = _clean_string_list(
                    options.get("experiment_run_ids")
                ) or _clean_string_list(metadata.get("source_run_ids"))
                if set(source_run_ids).intersection(set(job_run_ids or [])):
                    explanation_job = explanation_job or job
            elif str(job.job_type or "").strip() == "compiler_patch_proposal":
                if (
                    explanation_note_ids
                    and _clean_string(job.research_note_id) in explanation_note_ids
                ):
                    proposal_job = proposal_job or job
            elif str(job.job_type or "").strip() == "compiler_patch_draft":
                if (
                    proposal_note_ids
                    and _clean_string(job.research_note_id) in proposal_note_ids
                ):
                    patch_draft_job = patch_draft_job or job

        available_actions: list[str] = []
        if (
            not explanation_note
            and (
                explanation_job is None
                or str(explanation_job.status or "").strip().lower()
                not in {"pending", "analyzing", "synthesizing", "generating"}
            )
            and len(source_run_ids) == 2
            and primary_run_id
            and comparison_run_id
            and benchmark_family
            and benchmark_suite_id
        ):
            available_actions.append("create_regression_explanation")
        if (
            explanation_note
            and not proposal_note
            and (
                proposal_job is None
                or str(proposal_job.status or "").strip().lower()
                not in {"pending", "analyzing", "synthesizing", "generating"}
            )
        ):
            available_actions.append("create_patch_proposal")
        if (
            proposal_note
            and not patch_draft_note
            and (
                patch_draft_job is None
                or str(patch_draft_job.status or "").strip().lower()
                not in {"pending", "analyzing", "synthesizing", "generating"}
            )
        ):
            available_actions.append("create_patch_draft")

        patch_payload = (
            patch_draft_note.structured_payload
            if patch_draft_note
            and isinstance(patch_draft_note.structured_payload, dict)
            else {}
        )
        proposal_payload = (
            proposal_note.structured_payload
            if proposal_note and isinstance(proposal_note.structured_payload, dict)
            else {}
        )
        summary["compiler_artifact_summary"] = {
            "source_run_ids": source_run_ids,
            "primary_run_id": primary_run_id,
            "comparison_run_id": comparison_run_id,
            "explanation_note_id": str(explanation_note.id)
            if explanation_note
            else None,
            "explanation_synthesis_job_id": str(explanation_job.id)
            if explanation_job
            else None,
            "explanation_synthesis_status": (
                _clean_string(explanation_job.status) or None
            )
            if explanation_job
            else None,
            "proposal_note_id": str(proposal_note.id) if proposal_note else None,
            "proposal_synthesis_job_id": str(proposal_job.id) if proposal_job else None,
            "proposal_synthesis_status": (_clean_string(proposal_job.status) or None)
            if proposal_job
            else None,
            "patch_draft_note_id": str(patch_draft_note.id)
            if patch_draft_note
            else None,
            "patch_draft_synthesis_job_id": str(patch_draft_job.id)
            if patch_draft_job
            else None,
            "patch_draft_synthesis_status": (
                _clean_string(patch_draft_job.status) or None
            )
            if patch_draft_job
            else None,
            "source_explanation_note_id": _clean_string(
                proposal_payload.get("source_explanation_note_id")
            )
            or None,
            "source_proposal_note_id": _clean_string(
                patch_payload.get("source_proposal_note_id")
            )
            or None,
            "source_id": _clean_string(
                patch_payload.get("source_id")
                or (
                    patch_draft_job.options
                    if patch_draft_job and isinstance(patch_draft_job.options, dict)
                    else {}
                ).get("source_id")
            )
            or None,
            "source_name": _clean_string(patch_payload.get("source_name")) or None,
            "available_actions": available_actions,
        }
    return summaries


async def list_scientific_validation_run_summaries(
    db: AsyncSession,
    *,
    user_id: Any,
    run_ids: Optional[List[str]],
    limit: int = 5,
) -> List[Dict[str, Any]]:
    ordered_ids: List[str] = []
    seen: set[str] = set()
    for value in run_ids or []:
        text = _clean_string(value)
        if not text or text in seen:
            continue
        seen.add(text)
        ordered_ids.append(text)
    if not ordered_ids:
        return []

    recent_ids = ordered_ids[-max(1, min(limit, 20)) :]
    query_ids: List[UUID] = []
    for value in recent_ids:
        try:
            query_ids.append(UUID(value))
        except Exception:
            continue
    if not query_ids:
        return []

    rows = list(
        (
            await db.execute(
                select(ExperimentRun).where(
                    ExperimentRun.user_id == user_id,
                    ExperimentRun.id.in_(query_ids),
                )
            )
        )
        .scalars()
        .all()
    )
    runs_by_id = {str(row.id): row for row in rows}
    summaries: List[Dict[str, Any]] = []
    ordered_runs: List[ExperimentRun] = []
    for run_id in reversed(recent_ids):
        run = runs_by_id.get(run_id)
        if run is None:
            continue
        summaries.append(build_scientific_validation_run_summary(run))
        ordered_runs.append(run)
    return await _attach_compiler_artifact_summaries(
        db, user_id=user_id, runs=ordered_runs, summaries=summaries
    )
