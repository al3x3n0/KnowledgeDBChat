"""Read model for autonomous R&D evidence-verification lifecycle state."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Mapping

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.agent_job import AgentJob
from app.models.experiment import ExperimentPlan, ExperimentRun
from app.models.tool_audit import ToolExecutionAudit
from app.services.autonomous_rnd_verification_launch_service import (
    autonomous_rnd_verification_launch_service,
)


class AutonomousRnDVerificationStatusService:
    """Combine proposed, launched, and reconciled verification state."""

    async def build(
        self,
        *,
        parent_job: AgentJob,
        outcome: Mapping[str, Any],
        db: AsyncSession,
    ) -> Dict[str, Any]:
        plan = (
            outcome.get("verification_plan")
            if isinstance(outcome.get("verification_plan"), Mapping)
            else {}
        )
        tasks_by_id = {
            str(item.get("id") or "").strip(): dict(item)
            for item in plan.get("tasks", [])
            if isinstance(item, Mapping) and str(item.get("id") or "").strip()
        }
        results = parent_job.results if isinstance(parent_job.results, dict) else {}
        reconciliations = {
            str(item.get("verification_task_id") or "").strip(): dict(item)
            for item in results.get("verification_reconciliations", [])
            if isinstance(item, Mapping)
            and str(item.get("verification_task_id") or "").strip()
        }
        evidence_status = {
            str(item.get("id") or "")
            .strip(): str(item.get("verification_status") or "")
            .strip()
            for item in outcome.get("evidence", [])
            if isinstance(item, Mapping) and str(item.get("id") or "").strip()
        }
        for task_id, reconciliation in reconciliations.items():
            tasks_by_id.setdefault(
                task_id,
                {
                    "id": task_id,
                    "evidence_id": reconciliation.get("external_evidence_id"),
                    "required_checks": [],
                    "priority": "historical",
                },
            )

        launch_ids = {
            task_id: autonomous_rnd_verification_launch_service.launch_ids(
                parent_job.user_id,
                parent_job.id,
                task_id,
            )
            for task_id in tasks_by_id
        }
        plans = await self._by_id(
            db,
            ExperimentPlan,
            [ids["experiment_plan_id"] for ids in launch_ids.values()],
        )
        runs = await self._by_id(
            db,
            ExperimentRun,
            [ids["experiment_run_id"] for ids in launch_ids.values()],
        )
        jobs = await self._by_id(
            db,
            AgentJob,
            [ids["agent_job_id"] for ids in launch_ids.values()],
        )
        audits = await self._by_id(
            db,
            ToolExecutionAudit,
            [ids["audit_id"] for ids in launch_ids.values()],
        )

        rows = []
        timeline = []
        for task_id, task in tasks_by_id.items():
            ids = launch_ids[task_id]
            launch_plan = plans.get(ids["experiment_plan_id"])
            run = runs.get(ids["experiment_run_id"])
            job = jobs.get(ids["agent_job_id"])
            audit = audits.get(ids["audit_id"])
            evidence_id = str(task.get("evidence_id") or "").strip()
            reconciliation = reconciliations.get(task_id, {})
            launch_status = (
                str(run.status or "").strip().lower()
                if run is not None
                else "not_launched"
            )
            budget = (
                {
                    key: audit.tool_input.get(key)
                    for key in (
                        "repeat_count",
                        "timeout_seconds",
                        "max_runtime_minutes",
                        "budget_limit",
                    )
                    if audit.tool_input.get(key) is not None
                }
                if audit is not None and isinstance(audit.tool_input, dict)
                else {}
            )
            reconciliation_at = reconciliation.get("recorded_at")
            timeline.extend(
                self._timeline_events(
                    parent_job=parent_job,
                    task_id=task_id,
                    launch_plan=launch_plan,
                    run=run,
                    job=job,
                    audit=audit,
                    reconciliation_status=reconciliation.get("status"),
                    reconciliation_at=reconciliation_at,
                )
            )
            rows.append(
                {
                    "task_id": task_id,
                    "evidence_id": evidence_id,
                    "evidence_status": evidence_status.get(evidence_id)
                    or task.get("current_status"),
                    "priority": task.get("priority"),
                    "priority_score": task.get("priority_score"),
                    "required_checks": list(task.get("required_checks") or []),
                    "launch_status": launch_status,
                    "job_status": str(job.status or "").strip().lower()
                    if job is not None
                    else None,
                    "approval_status": str(audit.approval_status or "").strip()
                    if audit is not None
                    else None,
                    "reconciliation_status": reconciliation.get("status"),
                    "reconciliation_recorded_at": reconciliation_at,
                    "experiment_plan_id": str(launch_plan.id)
                    if launch_plan is not None
                    else None,
                    "experiment_run_id": str(run.id) if run is not None else None,
                    "agent_job_id": str(job.id) if job is not None else None,
                    "audit_id": str(audit.id) if audit is not None else None,
                    "budget": budget,
                }
            )
        rows.sort(
            key=lambda row: (
                -int(row.get("priority_score") or 0),
                str(row.get("task_id") or ""),
            )
        )
        counts = Counter(str(row["launch_status"]) for row in rows)
        evidence_counts = Counter(
            str(row.get("evidence_status") or "unknown") for row in rows
        )
        timeline.sort(
            key=lambda event: (
                str(event.get("at") or ""),
                int(event.get("sequence") or 0),
                str(event.get("task_id") or ""),
            )
        )
        for event in timeline:
            event.pop("sequence", None)
        return {
            "task_count": len(rows),
            "launch_status_counts": dict(sorted(counts.items())),
            "evidence_status_counts": dict(sorted(evidence_counts.items())),
            "tasks": rows,
            "timeline": timeline,
        }

    @classmethod
    def _timeline_events(
        cls,
        *,
        parent_job: AgentJob,
        task_id: str,
        launch_plan: ExperimentPlan | None,
        run: ExperimentRun | None,
        job: AgentJob | None,
        audit: ToolExecutionAudit | None,
        reconciliation_status: Any,
        reconciliation_at: Any,
    ) -> list[Dict[str, Any]]:
        events = []

        def add(
            event_type: str,
            *,
            at: Any,
            sequence: int,
            actor: str,
            label: str,
            status: Any = None,
            entity_type: str | None = None,
            entity_id: Any = None,
        ) -> None:
            timestamp = cls._iso(at)
            if not timestamp:
                return
            events.append(
                {
                    "event_id": f"{task_id}:{event_type}",
                    "task_id": task_id,
                    "event_type": event_type,
                    "at": timestamp,
                    "sequence": sequence,
                    "actor": actor,
                    "label": label,
                    "status": str(status or "").strip().lower() or None,
                    "entity_type": entity_type,
                    "entity_id": str(entity_id) if entity_id else None,
                }
            )

        add(
            "proposal_created",
            at=parent_job.completed_at or parent_job.created_at,
            sequence=10,
            actor="planner",
            label="Verification proposed",
            status="approval_required",
            entity_type="agent_job",
            entity_id=parent_job.id,
        )
        if audit is not None:
            add(
                "approval_recorded",
                at=audit.approved_at or audit.created_at,
                sequence=20,
                actor="operator",
                label="Verification approved",
                status=audit.approval_status,
                entity_type="tool_audit",
                entity_id=audit.id,
            )
        if launch_plan is not None:
            add(
                "verification_launched",
                at=launch_plan.created_at,
                sequence=30,
                actor="system",
                label="Bounded experiment created",
                status=getattr(run, "status", None) or "planned",
                entity_type="experiment_plan",
                entity_id=launch_plan.id,
            )
        if run is not None or job is not None:
            started_at = getattr(run, "started_at", None) or getattr(
                job, "started_at", None
            )
            add(
                "execution_started",
                at=started_at,
                sequence=40,
                actor="runner",
                label="Verification execution started",
                status=getattr(job, "status", None) or getattr(run, "status", None),
                entity_type="agent_job",
                entity_id=getattr(job, "id", None),
            )
            terminal = str(
                getattr(run, "status", None) or getattr(job, "status", None) or ""
            ).lower() in {"succeeded", "failed", "blocked", "completed", "cancelled"}
            if terminal:
                add(
                    "execution_completed",
                    at=getattr(run, "completed_at", None)
                    or getattr(job, "completed_at", None)
                    or reconciliation_at,
                    sequence=50,
                    actor="runner",
                    label="Verification execution completed",
                    status=getattr(run, "status", None) or getattr(job, "status", None),
                    entity_type="experiment_run",
                    entity_id=getattr(run, "id", None),
                )
        if reconciliation_status:
            add(
                "reconciliation_recorded",
                at=reconciliation_at
                or getattr(run, "completed_at", None)
                or getattr(job, "completed_at", None),
                sequence=60,
                actor="reconciler",
                label="Evidence reconciliation recorded",
                status=reconciliation_status,
                entity_type="experiment_run",
                entity_id=getattr(run, "id", None),
            )
        return events

    @staticmethod
    def _iso(value: Any) -> str | None:
        if value is None:
            return None
        isoformat = getattr(value, "isoformat", None)
        if callable(isoformat):
            return str(isoformat())
        text = str(value).strip()
        return text or None

    @staticmethod
    async def _by_id(
        db: AsyncSession,
        model: type,
        ids: list[Any],
    ) -> Dict[Any, Any]:
        if not ids:
            return {}
        rows = list(
            (await db.execute(select(model).where(model.id.in_(ids)))).scalars().all()
        )
        return {row.id: row for row in rows}


autonomous_rnd_verification_status_service = AutonomousRnDVerificationStatusService()
