"""Deterministic, trial-based evaluation for autonomous R&D agents.

The harness deliberately grades observable outcomes rather than an agent's
self-reported progress. Agent/model execution is injected through a callback,
which keeps the evaluation format useful for local models, hosted models, and
replayed production trajectories.
"""

from __future__ import annotations

import inspect
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Optional

BUILTIN_AUTONOMOUS_RND_EVAL_SUITES = {
    "compiler_research_v1": (
        Path(__file__).resolve().parents[2]
        / "evals"
        / "autonomous_rnd"
        / "compiler_research_v1.json"
    )
}


class EvalDefinitionError(ValueError):
    """Raised when an autonomous R&D evaluation suite is malformed."""


@dataclass(frozen=True)
class AutonomousRnDEvalTask:
    id: str
    name: str
    category: str
    prompt: str
    graders: List[Dict[str, Any]]
    trials: int = 3
    pass_threshold: float = 1.0

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AutonomousRnDEvalTask":
        task_id = str(value.get("id") or "").strip()
        if not task_id:
            raise EvalDefinitionError("Evaluation task requires a non-empty id")
        graders = value.get("graders")
        if not isinstance(graders, list) or not graders:
            raise EvalDefinitionError(f"Evaluation task '{task_id}' requires graders")
        if any(not isinstance(item, dict) for item in graders):
            raise EvalDefinitionError(
                f"Evaluation task '{task_id}' graders must be JSON objects"
            )
        trials = _bounded_int(value.get("trials"), default=3, minimum=1, maximum=100)
        threshold = _bounded_float(
            value.get("pass_threshold"), default=1.0, minimum=0.0, maximum=1.0
        )
        return cls(
            id=task_id,
            name=str(value.get("name") or task_id).strip() or task_id,
            category=str(value.get("category") or "general").strip() or "general",
            prompt=str(value.get("prompt") or ""),
            graders=[dict(item) for item in graders],
            trials=trials,
            pass_threshold=threshold,
        )


@dataclass(frozen=True)
class AutonomousRnDEvalSuite:
    id: str
    name: str
    version: int
    tasks: List[AutonomousRnDEvalTask]
    seed: int = 1729
    description: str = ""

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AutonomousRnDEvalSuite":
        suite_id = str(value.get("id") or "").strip()
        if not suite_id:
            raise EvalDefinitionError("Evaluation suite requires a non-empty id")
        raw_tasks = value.get("tasks")
        if not isinstance(raw_tasks, list) or not raw_tasks:
            raise EvalDefinitionError(f"Evaluation suite '{suite_id}' requires tasks")
        tasks = [
            AutonomousRnDEvalTask.from_dict(item)
            for item in raw_tasks
            if isinstance(item, dict)
        ]
        if not tasks:
            raise EvalDefinitionError(
                f"Evaluation suite '{suite_id}' has no valid tasks"
            )
        task_ids = [task.id for task in tasks]
        if len(task_ids) != len(set(task_ids)):
            raise EvalDefinitionError(
                f"Evaluation suite '{suite_id}' contains duplicate task ids"
            )
        return cls(
            id=suite_id,
            name=str(value.get("name") or suite_id).strip() or suite_id,
            description=str(value.get("description") or ""),
            version=_bounded_int(
                value.get("version"), default=1, minimum=1, maximum=1_000_000
            ),
            seed=_bounded_int(
                value.get("seed"), default=1729, minimum=0, maximum=2**31 - 1
            ),
            tasks=tasks,
        )


TrialExecutor = Callable[
    [AutonomousRnDEvalTask, int, int],
    Awaitable[Mapping[str, Any]] | Mapping[str, Any],
]


class AutonomousRnDEvalHarness:
    """Load, execute, grade, and aggregate autonomous R&D evaluations."""

    def load_suite(self, path: str | Path) -> AutonomousRnDEvalSuite:
        source = Path(path)
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise EvalDefinitionError(
                f"Unable to load evaluation suite {source}: {exc}"
            )
        if not isinstance(payload, dict):
            raise EvalDefinitionError("Evaluation suite root must be a JSON object")
        return AutonomousRnDEvalSuite.from_dict(payload)

    def load_builtin_suite(self, suite_id: str) -> AutonomousRnDEvalSuite:
        normalized = str(suite_id or "").strip()
        source = BUILTIN_AUTONOMOUS_RND_EVAL_SUITES.get(normalized)
        if source is None:
            raise EvalDefinitionError(
                f"Unknown built-in autonomous R&D evaluation suite: {normalized}"
            )
        return self.load_suite(source)

    def list_builtin_suites(self) -> List[Dict[str, Any]]:
        suites = []
        for suite_id in sorted(BUILTIN_AUTONOMOUS_RND_EVAL_SUITES):
            suite = self.load_builtin_suite(suite_id)
            suites.append(
                {
                    "id": suite.id,
                    "name": suite.name,
                    "description": suite.description,
                    "version": suite.version,
                    "task_count": len(suite.tasks),
                    "default_trial_count": sum(task.trials for task in suite.tasks),
                }
            )
        return suites

    async def run_suite(
        self,
        suite: AutonomousRnDEvalSuite,
        executor: TrialExecutor,
        *,
        trials_override: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run every task and return an outcome-based aggregate report."""
        outcomes: Dict[str, List[Dict[str, Any]]] = {}
        for task_index, task in enumerate(suite.tasks):
            trial_count = (
                _bounded_int(
                    trials_override, default=task.trials, minimum=1, maximum=100
                )
                if trials_override is not None
                else task.trials
            )
            task_outcomes: List[Dict[str, Any]] = []
            for trial_index in range(trial_count):
                seed = suite.seed + (task_index * 10_000) + trial_index
                started = time.perf_counter()
                try:
                    result = executor(task, trial_index, seed)
                    if inspect.isawaitable(result):
                        result = await result
                    outcome = dict(result) if isinstance(result, Mapping) else {}
                except Exception as exc:  # noqa: BLE001 - failures are eval outcomes
                    outcome = {
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}"[:1000],
                    }
                outcome.setdefault("eval_seed", seed)
                outcome.setdefault(
                    "eval_duration_ms",
                    int((time.perf_counter() - started) * 1000),
                )
                task_outcomes.append(outcome)
            outcomes[task.id] = task_outcomes
        return self.grade_suite_outcomes(suite, outcomes)

    def grade_suite_outcomes(
        self,
        suite: AutonomousRnDEvalSuite,
        outcomes: Mapping[str, List[Mapping[str, Any]]],
    ) -> Dict[str, Any]:
        """Grade already-recorded outcomes, enabling deterministic replay."""
        task_reports: List[Dict[str, Any]] = []
        for task in suite.tasks:
            raw_trials = outcomes.get(task.id) or []
            trials = [
                self.grade_trial(task, dict(outcome))
                for outcome in raw_trials
                if isinstance(outcome, Mapping)
            ]
            task_reports.append(self._aggregate_task(task, trials))

        task_count = len(task_reports)
        pass_at_k = (
            sum(1 for report in task_reports if report["pass_at_k"]) / task_count
            if task_count
            else 0.0
        )
        pass_pow_k = (
            sum(1 for report in task_reports if report["pass_pow_k"]) / task_count
            if task_count
            else 0.0
        )
        mean_score = (
            sum(float(report["mean_score"]) for report in task_reports) / task_count
            if task_count
            else 0.0
        )
        total_trials = sum(int(report["trial_count"]) for report in task_reports)
        return {
            "suite_id": suite.id,
            "suite_name": suite.name,
            "suite_version": suite.version,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "task_count": task_count,
            "trial_count": total_trials,
            "mean_score": round(mean_score, 6),
            "pass_at_k": round(pass_at_k, 6),
            "pass_pow_k": round(pass_pow_k, 6),
            "tasks": task_reports,
        }

    def grade_trial(
        self, task: AutonomousRnDEvalTask, outcome: Dict[str, Any]
    ) -> Dict[str, Any]:
        checks = [self._grade_spec(spec, outcome) for spec in task.graders]
        total_weight = sum(float(check["weight"]) for check in checks)
        weighted_score = sum(
            float(check["score"]) * float(check["weight"]) for check in checks
        )
        score = weighted_score / total_weight if total_weight else 0.0
        required_passed = all(
            bool(check["passed"]) for check in checks if bool(check["required"])
        )
        passed = required_passed and score >= task.pass_threshold
        return {
            "passed": passed,
            "score": round(score, 6),
            "required_passed": required_passed,
            "checks": checks,
            "outcome": outcome,
        }

    def _grade_spec(
        self, spec: Mapping[str, Any], outcome: Mapping[str, Any]
    ) -> Dict[str, Any]:
        grader_type = str(spec.get("type") or "").strip()
        grader_id = str(spec.get("id") or grader_type or "grader").strip()
        weight = _bounded_float(
            spec.get("weight"), default=1.0, minimum=0.0, maximum=1000.0
        )
        required = bool(spec.get("required", True))
        if not grader_type:
            return _check(
                grader_id,
                "unknown",
                False,
                weight,
                required,
                "Grader type is missing",
            )

        handler = getattr(self, f"_grade_{grader_type}", None)
        if handler is None:
            return _check(
                grader_id,
                grader_type,
                False,
                weight,
                required,
                f"Unsupported grader type: {grader_type}",
            )
        passed, details = handler(spec, outcome)
        return _check(
            grader_id,
            grader_type,
            bool(passed),
            weight,
            required,
            details,
        )

    @staticmethod
    def _grade_status_equals(spec, outcome):
        expected = str(spec.get("expected") or "completed")
        actual = str(outcome.get("status") or "")
        return actual == expected, f"expected={expected!r}, actual={actual!r}"

    @staticmethod
    def _grade_field_equals(spec, outcome):
        path = str(spec.get("path") or "").strip()
        expected = spec.get("expected")
        exists, actual = _lookup(outcome, path)
        return exists and actual == expected, (
            f"path={path!r}, expected={expected!r}, actual={actual!r}"
        )

    @staticmethod
    def _grade_required_output_keys(spec, outcome):
        keys = [str(item).strip() for item in spec.get("keys", []) if str(item).strip()]
        missing = [path for path in keys if not _lookup(outcome, path)[0]]
        return not missing, f"missing={missing}"

    @staticmethod
    def _grade_required_artifacts(spec, outcome):
        required_kinds = {
            str(item).strip() for item in spec.get("kinds", []) if str(item).strip()
        }
        artifacts = outcome.get("artifacts")
        actual_kinds = {
            str(item.get("kind") or "").strip()
            for item in artifacts or []
            if isinstance(item, dict)
        }
        missing = sorted(required_kinds - actual_kinds)
        return not missing, f"missing={missing}, actual={sorted(actual_kinds)}"

    @staticmethod
    def _grade_traceable_claims(spec, outcome):
        claims = outcome.get("claims")
        evidence = outcome.get("evidence")
        if not isinstance(claims, list) or not claims:
            return False, "No claims were recorded"
        allow_unverified_evidence = bool(spec.get("allow_unverified_evidence", False))
        evidence_ids = set()
        for item in evidence or []:
            if not isinstance(item, dict):
                continue
            evidence_id = str(item.get("id") or "").strip()
            if not evidence_id:
                continue
            verification_status = (
                str(item.get("verification_status") or "").strip().lower()
            )
            if verification_status == "rejected":
                continue
            if verification_status == "unverified" and not allow_unverified_evidence:
                continue
            evidence_ids.add(evidence_id)
        minimum = _bounded_int(
            spec.get("min_evidence_per_claim"), default=1, minimum=1, maximum=100
        )
        allow_unverified = bool(spec.get("allow_explicit_unverified", False))
        invalid: List[str] = []
        for index, claim in enumerate(claims):
            if not isinstance(claim, dict):
                invalid.append(f"claim[{index}]:invalid")
                continue
            status = str(claim.get("verification_status") or "").strip().lower()
            if allow_unverified and status == "unverified":
                continue
            refs = {
                str(item).strip()
                for item in claim.get("evidence_ids", [])
                if str(item).strip()
            }
            valid_refs = refs & evidence_ids
            if len(valid_refs) < minimum or valid_refs != refs:
                invalid.append(str(claim.get("id") or f"claim[{index}]"))
        return not invalid, f"invalid_or_under_supported={invalid}"

    @staticmethod
    def _grade_forbidden_tools(spec, outcome):
        forbidden = {
            str(item).strip() for item in spec.get("tools", []) if str(item).strip()
        }
        actions = outcome.get("actions")
        used = {
            str(item.get("tool") or "").strip()
            for item in actions or []
            if isinstance(item, dict)
        }
        violations = sorted(forbidden & used)
        return not violations, f"violations={violations}"

    @staticmethod
    def _grade_verification_plan_coverage(spec, outcome):
        del spec
        unresolved = {
            str(item.get("id") or "").strip()
            for item in outcome.get("evidence") or []
            if isinstance(item, dict)
            and str(item.get("kind") or "").strip()
            in {"external_agent_response", "external_system_response"}
            and str(item.get("verification_status") or "").strip().lower()
            in {"unverified", "corroborated"}
            and str(item.get("id") or "").strip()
        }
        plan = outcome.get("verification_plan")
        tasks = plan.get("tasks") if isinstance(plan, dict) else []
        covered = {
            str(item.get("evidence_id") or "").strip()
            for item in tasks or []
            if isinstance(item, dict) and str(item.get("evidence_id") or "").strip()
        }
        missing = sorted(unresolved - covered)
        return not missing, f"missing_plans={missing}, unresolved={sorted(unresolved)}"

    @staticmethod
    def _grade_metric_bounds(spec, outcome):
        path = str(spec.get("path") or "").strip()
        exists, raw_value = _lookup(outcome, path)
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            return False, f"path={path!r} is not numeric: {raw_value!r}"
        minimum = spec.get("min")
        maximum = spec.get("max")
        passed = exists and math.isfinite(value)
        if minimum is not None:
            passed = passed and value >= float(minimum)
        if maximum is not None:
            passed = passed and value <= float(maximum)
        return passed, f"path={path!r}, value={value}, min={minimum}, max={maximum}"

    @staticmethod
    def _grade_experiment_execution(spec, outcome):
        experiment = outcome.get("experiment")
        if not isinstance(experiment, dict):
            return False, "Experiment outcome is missing"
        required_repeats = _bounded_int(
            spec.get("min_repeat_count"), default=1, minimum=1, maximum=100_000
        )
        repeat_count = _bounded_int(
            experiment.get("repeat_count"), default=0, minimum=0, maximum=100_000
        )
        require_commands_ok = bool(spec.get("require_commands_ok", True))
        commands_ok = bool(experiment.get("all_commands_ok"))
        passed = repeat_count >= required_repeats and (
            commands_ok or not require_commands_ok
        )
        return passed, (
            f"repeat_count={repeat_count}, required={required_repeats}, "
            f"all_commands_ok={commands_ok}"
        )

    @staticmethod
    def _aggregate_task(
        task: AutonomousRnDEvalTask, trials: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        passed_count = sum(1 for trial in trials if bool(trial.get("passed")))
        trial_count = len(trials)
        return {
            "task_id": task.id,
            "task_name": task.name,
            "category": task.category,
            "expected_trial_count": task.trials,
            "trial_count": trial_count,
            "passed_count": passed_count,
            "pass_rate": round(passed_count / trial_count, 6) if trial_count else 0.0,
            "pass_at_k": passed_count > 0,
            "pass_pow_k": trial_count >= task.trials and passed_count == trial_count,
            "mean_score": round(
                sum(float(trial.get("score") or 0.0) for trial in trials) / trial_count,
                6,
            )
            if trial_count
            else 0.0,
            "trials": trials,
        }


def _lookup(value: Mapping[str, Any], path: str) -> tuple[bool, Any]:
    if not path:
        return False, None
    current: Any = value
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return False, None
        current = current[part]
    return True, current


def _check(
    grader_id: str,
    grader_type: str,
    passed: bool,
    weight: float,
    required: bool,
    details: str,
) -> Dict[str, Any]:
    return {
        "id": grader_id,
        "type": grader_type,
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "weight": weight,
        "required": required,
        "details": details,
    }


def _bounded_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(parsed, maximum))


def _bounded_float(
    value: Any, *, default: float, minimum: float, maximum: float
) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    if not math.isfinite(parsed):
        parsed = default
    return max(minimum, min(parsed, maximum))


autonomous_rnd_eval_harness = AutonomousRnDEvalHarness()
