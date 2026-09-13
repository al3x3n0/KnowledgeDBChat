"""Project completed local verification runs back into originating R&D jobs."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, Mapping
from urllib.parse import quote
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.attributes import flag_modified

from app.models.agent_job import AgentJob
from app.models.experiment import ExperimentRun
from app.models.notification import Notification, NotificationType
from app.services.autonomous_rnd_trajectory_service import (
    autonomous_rnd_trajectory_adapter,
)
from app.services.notification_service import notification_service


class AutonomousRnDVerificationReconciliationService:
    """Record replayable verification evidence without copying command output."""

    async def reconcile(
        self,
        *,
        verification_job: AgentJob,
        db: AsyncSession,
    ) -> bool:
        config = (
            verification_job.config if isinstance(verification_job.config, dict) else {}
        )
        origin = (
            config.get("verification_origin")
            if isinstance(config.get("verification_origin"), dict)
            else {}
        )
        parent_id = _uuid(origin.get("parent_job_id"))
        run_id = _uuid(config.get("experiment_run_id"))
        task_id = str(origin.get("verification_task_id") or "").strip()
        external_evidence_id = str(origin.get("external_evidence_id") or "").strip()
        if not parent_id or not run_id or not task_id or not external_evidence_id:
            return False

        parent = await db.get(AgentJob, parent_id)
        run = await db.get(ExperimentRun, run_id)
        if (
            parent is None
            or run is None
            or parent.user_id != verification_job.user_id
            or run.user_id != verification_job.user_id
            or run.agent_job_id != verification_job.id
        ):
            return False

        job_results = (
            verification_job.results
            if isinstance(verification_job.results, dict)
            else {}
        )
        experiment = (
            job_results.get("experiment_run")
            if isinstance(job_results.get("experiment_run"), dict)
            else {}
        )
        runs = (
            experiment.get("runs") if isinstance(experiment.get("runs"), list) else []
        )
        command_count = len(
            {
                str(item.get("command") or "").strip()
                for item in runs
                if isinstance(item, Mapping) and str(item.get("command") or "").strip()
            }
        )
        repeat_count = max(
            [
                int(item.get("repeat_index") or 0)
                for item in runs
                if isinstance(item, Mapping)
            ]
            or [int(experiment.get("repeat_count") or 0)]
        )
        all_commands_ok = bool(runs) and all(
            isinstance(item, Mapping) and item.get("ok") is True for item in runs
        )
        ran = bool(runs)
        local_evidence_id = f"local-experiment:{run.id}"
        artifact_id = str(run.id)

        parent_results = (
            deepcopy(parent.results) if isinstance(parent.results, dict) else {}
        )
        local_evidence = {
            "id": local_evidence_id,
            "kind": "controlled_experiment_result",
            "record_origin": "local_experiment",
            "source": f"experiment_run:{run.id}",
            "experiment_run_id": str(run.id),
            "verification_job_id": str(verification_job.id),
            "command_count": command_count,
            "repeat_count": repeat_count,
            "all_commands_ok": all_commands_ok,
        }
        _upsert(parent_results, "evidence", local_evidence)

        verification_experiment = {
            "id": str(run.id),
            "run_id": str(run.id),
            "job_id": str(verification_job.id),
            "ran": ran,
            "repeat_count": repeat_count,
            "all_commands_ok": all_commands_ok,
            "status": str(run.status or "").strip().lower(),
        }
        _upsert(
            parent_results,
            "verification_experiments",
            verification_experiment,
            id_field="run_id",
        )
        link = {
            "external_evidence_id": external_evidence_id,
            "verdict": "supports" if all_commands_ok else "inconclusive",
            "local_evidence_ids": [local_evidence_id],
            "artifact_ids": [artifact_id],
            "experiment_run_id": str(run.id),
            "min_repeat_count": int(
                config.get("repeat_count")
                or experiment.get("repeat_count")
                or repeat_count
                or 2
            ),
        }
        _upsert(
            parent_results,
            "verification_links",
            link,
            id_field="external_evidence_id",
            secondary_field="experiment_run_id",
        )
        existing_reconciliation = next(
            (
                item
                for item in parent_results.get("verification_reconciliations", [])
                if isinstance(item, Mapping)
                and str(item.get("id") or "").strip() == str(run.id)
            ),
            {},
        )
        reconciliation = {
            "id": str(run.id),
            "verification_task_id": task_id,
            "external_evidence_id": external_evidence_id,
            "verification_job_id": str(verification_job.id),
            "experiment_run_id": str(run.id),
            "status": "support_recorded" if all_commands_ok else "inconclusive",
            "recorded_at": existing_reconciliation.get("recorded_at")
            or datetime.now(timezone.utc).isoformat(),
        }
        _upsert(parent_results, "verification_reconciliations", reconciliation)

        artifacts = (
            list(parent.output_artifacts)
            if isinstance(parent.output_artifacts, list)
            else []
        )
        artifact = {
            "id": artifact_id,
            "type": "experiment_run",
            "kind": "experiment_run",
            "record_origin": "local_experiment",
            "verification_task_id": task_id,
        }
        artifacts = [
            item
            for item in artifacts
            if not (
                isinstance(item, Mapping)
                and str(item.get("id") or "").strip() == artifact_id
            )
        ]
        artifacts.append(artifact)
        parent.output_artifacts = artifacts[-200:]
        parent.results = parent_results
        parent.results[
            "evaluation_outcome"
        ] = autonomous_rnd_trajectory_adapter.build_outcome(parent)
        flag_modified(parent, "results")
        flag_modified(parent, "output_artifacts")
        await db.flush()
        await self._notify_reconciliation(
            parent=parent,
            verification_job=verification_job,
            run=run,
            task_id=task_id,
            external_evidence_id=external_evidence_id,
            command_count=command_count,
            repeat_count=repeat_count,
            all_commands_ok=all_commands_ok,
            db=db,
        )
        return True

    @staticmethod
    async def _notify_reconciliation(
        *,
        parent: AgentJob,
        verification_job: AgentJob,
        run: ExperimentRun,
        task_id: str,
        external_evidence_id: str,
        command_count: int,
        repeat_count: int,
        all_commands_ok: bool,
        db: AsyncSession,
    ) -> None:
        existing = list(
            (
                await db.execute(
                    select(Notification)
                    .where(
                        Notification.user_id == parent.user_id,
                        Notification.notification_type
                        == NotificationType.AUTONOMOUS_RND_VERIFICATION_UPDATE,
                        Notification.related_entity_id == parent.id,
                    )
                    .order_by(Notification.created_at.desc())
                    .limit(50)
                )
            )
            .scalars()
            .all()
        )
        if any(
            str((item.data or {}).get("experiment_run_id") or "") == str(run.id)
            and str((item.data or {}).get("verification_task_id") or "") == task_id
            for item in existing
            if isinstance(item.data, dict)
        ):
            return

        status = "verified" if all_commands_ok else "inconclusive"
        reconciliation_status = (
            "support_recorded" if all_commands_ok else "inconclusive"
        )
        await notification_service.create_notification(
            db=db,
            user_id=parent.user_id,
            notification_type=NotificationType.AUTONOMOUS_RND_VERIFICATION_UPDATE,
            title=(
                "R&D evidence verified"
                if all_commands_ok
                else "R&D verification inconclusive"
            ),
            message=(
                f"Local verification completed with {repeat_count} controlled "
                f"repetition{'s' if repeat_count != 1 else ''}; evidence is {status}."
            ),
            priority="normal" if all_commands_ok else "high",
            related_entity_type="agent_job",
            related_entity_id=parent.id,
            data={
                "parent_job_id": str(parent.id),
                "verification_job_id": str(verification_job.id),
                "verification_task_id": task_id,
                "external_evidence_id": external_evidence_id,
                "experiment_run_id": str(run.id),
                "verification_status": status,
                "reconciliation_status": reconciliation_status,
                "all_commands_ok": all_commands_ok,
                "command_count": command_count,
                "repeat_count": repeat_count,
            },
            action_url=(
                f"/autonomous-agents?job={parent.id}"
                f"&verification_task={quote(task_id, safe='')}"
            ),
            commit=False,
        )


def _uuid(value: Any) -> UUID | None:
    try:
        return UUID(str(value))
    except (TypeError, ValueError, AttributeError):
        return None


def _upsert(
    payload: Dict[str, Any],
    field: str,
    row: Dict[str, Any],
    *,
    id_field: str = "id",
    secondary_field: str | None = None,
) -> None:
    records = (
        [dict(item) for item in payload.get(field, []) if isinstance(item, Mapping)]
        if isinstance(payload.get(field), list)
        else []
    )
    records = [
        item
        for item in records
        if not (
            str(item.get(id_field) or "").strip()
            == str(row.get(id_field) or "").strip()
            and (
                secondary_field is None
                or str(item.get(secondary_field) or "").strip()
                == str(row.get(secondary_field) or "").strip()
            )
        )
    ]
    records.append(row)
    payload[field] = records[-200:]


autonomous_rnd_verification_reconciliation_service = (
    AutonomousRnDVerificationReconciliationService()
)
