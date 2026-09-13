"""Persist graded autonomous R&D evaluation runs and compare them to baselines.

The harness in ``autonomous_rnd_eval_service`` grades outcomes but keeps no
history, so a change to a model, prompt, tool, or orchestration path cannot be
told apart from noise. This service stores each graded report as a run, lets an
operator promote one run per suite to the baseline, and diffs a candidate run
against it. ``pass_pow_k`` is the reliability signal for unattended operation,
so it drives the regression verdict.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional
from uuid import UUID, uuid4

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.autonomous_rnd_eval_run import (
    EVAL_RUN_SOURCE_GRADE_JOBS,
    AutonomousRndEvalRun,
)

MAX_RUN_LIST_LIMIT = 100

# Metric deltas below this are treated as float noise rather than movement.
METRIC_EPSILON = 1e-9


class EvalRunError(ValueError):
    """Raised when a run cannot be recorded, promoted, or compared."""


def _metric(report: Mapping[str, Any], key: str) -> float:
    try:
        return round(float(report.get(key) or 0.0), 6)
    except (TypeError, ValueError):
        return 0.0


def _task_index(report: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    tasks = report.get("tasks")
    if not isinstance(tasks, list):
        return {}
    indexed: Dict[str, Mapping[str, Any]] = {}
    for task in tasks:
        if not isinstance(task, Mapping):
            continue
        task_id = str(task.get("task_id") or task.get("id") or "").strip()
        if task_id:
            indexed[task_id] = task
    return indexed


def _delta(candidate: float, baseline: float) -> float:
    difference = candidate - baseline
    return 0.0 if abs(difference) < METRIC_EPSILON else round(difference, 6)


class AutonomousRnDEvalRunService:
    """Store, list, promote, and diff graded evaluation runs."""

    async def record_run(
        self,
        db: AsyncSession,
        *,
        user_id: UUID,
        report: Mapping[str, Any],
        task_bindings: Optional[Mapping[str, List[str]]] = None,
        source: str = EVAL_RUN_SOURCE_GRADE_JOBS,
        label: Optional[str] = None,
    ) -> AutonomousRndEvalRun:
        suite_id = str(report.get("suite_id") or "").strip()
        if not suite_id:
            raise EvalRunError("Graded report is missing a suite_id")

        run = AutonomousRndEvalRun(
            id=uuid4(),
            user_id=user_id,
            suite_id=suite_id,
            suite_name=str(report.get("suite_name") or suite_id)[:300],
            suite_version=int(report.get("suite_version") or 1),
            label=(label.strip()[:200] or None) if label else None,
            source=str(source or EVAL_RUN_SOURCE_GRADE_JOBS)[:32],
            is_baseline=False,
            task_count=int(report.get("task_count") or 0),
            trial_count=int(report.get("trial_count") or 0),
            mean_score=_metric(report, "mean_score"),
            pass_at_k=_metric(report, "pass_at_k"),
            pass_pow_k=_metric(report, "pass_pow_k"),
            report=dict(report),
            task_bindings=(
                {key: list(value) for key, value in task_bindings.items()}
                if task_bindings
                else None
            ),
        )
        db.add(run)
        await db.flush()
        return run

    async def get_run(
        self, db: AsyncSession, *, user_id: UUID, run_id: UUID
    ) -> Optional[AutonomousRndEvalRun]:
        return (
            await db.execute(
                select(AutonomousRndEvalRun).where(
                    AutonomousRndEvalRun.id == run_id,
                    AutonomousRndEvalRun.user_id == user_id,
                )
            )
        ).scalar_one_or_none()

    async def list_runs(
        self,
        db: AsyncSession,
        *,
        user_id: UUID,
        suite_id: Optional[str] = None,
        limit: int = 20,
    ) -> List[AutonomousRndEvalRun]:
        bounded = max(1, min(int(limit or 20), MAX_RUN_LIST_LIMIT))
        query = select(AutonomousRndEvalRun).where(
            AutonomousRndEvalRun.user_id == user_id
        )
        if suite_id:
            query = query.where(AutonomousRndEvalRun.suite_id == suite_id.strip())
        query = query.order_by(AutonomousRndEvalRun.created_at.desc()).limit(bounded)
        return list((await db.execute(query)).scalars().all())

    async def get_baseline(
        self, db: AsyncSession, *, user_id: UUID, suite_id: str
    ) -> Optional[AutonomousRndEvalRun]:
        return (
            await db.execute(
                select(AutonomousRndEvalRun).where(
                    AutonomousRndEvalRun.user_id == user_id,
                    AutonomousRndEvalRun.suite_id == suite_id.strip(),
                    AutonomousRndEvalRun.is_baseline.is_(True),
                )
            )
        ).scalar_one_or_none()

    async def set_baseline(
        self, db: AsyncSession, *, user_id: UUID, run_id: UUID
    ) -> AutonomousRndEvalRun:
        run = await self.get_run(db, user_id=user_id, run_id=run_id)
        if run is None:
            raise EvalRunError("Evaluation run was not found")
        # Demote first so the partial unique index never sees two baselines.
        await db.execute(
            update(AutonomousRndEvalRun)
            .where(
                AutonomousRndEvalRun.user_id == user_id,
                AutonomousRndEvalRun.suite_id == run.suite_id,
                AutonomousRndEvalRun.id != run.id,
                AutonomousRndEvalRun.is_baseline.is_(True),
            )
            .values(is_baseline=False)
        )
        run.is_baseline = True
        await db.flush()
        return run

    def compare(
        self,
        *,
        baseline: AutonomousRndEvalRun,
        candidate: AutonomousRndEvalRun,
    ) -> Dict[str, Any]:
        """Diff two runs of the same suite. Pure over the persisted reports."""
        if baseline.suite_id != candidate.suite_id:
            raise EvalRunError("Runs belong to different evaluation suites")

        baseline_report = baseline.report or {}
        candidate_report = candidate.report or {}
        baseline_tasks = _task_index(baseline_report)
        candidate_tasks = _task_index(candidate_report)

        task_diffs: List[Dict[str, Any]] = []
        regressed: List[str] = []
        improved: List[str] = []
        for task_id in sorted(set(baseline_tasks) | set(candidate_tasks)):
            before = baseline_tasks.get(task_id)
            after = candidate_tasks.get(task_id)
            if before is None:
                status = "added"
            elif after is None:
                status = "removed"
            else:
                was = bool(before.get("pass_pow_k"))
                now = bool(after.get("pass_pow_k"))
                if was and not now:
                    status = "regressed"
                    regressed.append(task_id)
                elif now and not was:
                    status = "improved"
                    improved.append(task_id)
                else:
                    status = "unchanged"
            task_diffs.append(
                {
                    "task_id": task_id,
                    "status": status,
                    "baseline_pass_pow_k": (
                        bool(before.get("pass_pow_k")) if before else None
                    ),
                    "candidate_pass_pow_k": (
                        bool(after.get("pass_pow_k")) if after else None
                    ),
                    "baseline_pass_at_k": (
                        bool(before.get("pass_at_k")) if before else None
                    ),
                    "candidate_pass_at_k": (
                        bool(after.get("pass_at_k")) if after else None
                    ),
                    "mean_score_delta": (
                        _delta(
                            _metric(after, "mean_score"),
                            _metric(before, "mean_score"),
                        )
                        if before and after
                        else None
                    ),
                }
            )

        metrics = {
            key: {
                "baseline": getattr(baseline, key),
                "candidate": getattr(candidate, key),
                "delta": _delta(getattr(candidate, key), getattr(baseline, key)),
            }
            for key in ("mean_score", "pass_at_k", "pass_pow_k")
        }
        # A suite regresses when any task loses all-trial reliability, or when
        # aggregate pass_pow_k drops even with no individual task flipping.
        has_regression = bool(regressed) or metrics["pass_pow_k"]["delta"] < 0
        return {
            "suite_id": candidate.suite_id,
            "baseline_run_id": str(baseline.id),
            "candidate_run_id": str(candidate.id),
            "baseline_suite_version": baseline.suite_version,
            "candidate_suite_version": candidate.suite_version,
            "suite_version_changed": (
                baseline.suite_version != candidate.suite_version
            ),
            "metrics": metrics,
            "regressed_task_ids": regressed,
            "improved_task_ids": improved,
            "has_regression": has_regression,
            "tasks": task_diffs,
        }


autonomous_rnd_eval_run_service = AutonomousRnDEvalRunService()
