"""Bulk job-backed checkpoint queue action boundary."""

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.api.endpoints.auth import get_current_active_user
from app.core.database import get_db
from app.models.agent_job import AgentJob, AgentJobStatus
from app.models.user import User
from app.schemas.agent_job import (
    AgentCheckpointQueueBulkActionRequest,
    AgentCheckpointQueueBulkActionResponse,
    AgentCheckpointQueueBulkActionResultResponse,
    AgentJobActionRequest,
)

ApprovalPayloadExtractor = Callable[..., tuple[dict, dict, dict | None]]
JobActionPerformer = Callable[..., Awaitable[AgentJob]]
OperatorEventRecorder = Callable[..., Awaitable[Any]]
SchedulerStateExtractor = Callable[[AgentJob | None], dict[str, Any] | None]


@dataclass(frozen=True)
class CheckpointJobActionApi:
    router: APIRouter
    checkpoint_queue_bulk_action: Callable[..., Any]
    validate_bulk_queue_action: Callable[..., Any]
    job_matches_bulk_queue_item_type: Callable[..., Any]


def build_checkpoint_job_action_api(
    *,
    router: APIRouter,
    allowed_actions: dict[str, set[str]],
    extract_approval_payload: ApprovalPayloadExtractor,
    perform_job_action: JobActionPerformer,
    record_operator_event: OperatorEventRecorder,
    extract_scheduler_state: SchedulerStateExtractor,
) -> CheckpointJobActionApi:
    """Register safe homogeneous bulk actions for job-backed queue rows."""

    def validate_bulk_queue_action(item_type: str, action: str) -> None:
        normalized_item_type = str(item_type or "").strip().lower()
        normalized_action = str(action or "").strip().lower()
        allowed = allowed_actions.get(normalized_item_type)
        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "Bulk actions are only supported for approval_checkpoint "
                    "and job_recovery items"
                ),
            )
        if normalized_action not in allowed:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Action {normalized_action} is not allowed for queue item "
                    f"type {normalized_item_type}"
                ),
            )

    def job_matches_bulk_queue_item_type(
        job: AgentJob,
        item_type: str,
    ) -> tuple[bool, Optional[str]]:
        normalized_item_type = str(item_type or "").strip().lower()
        if normalized_item_type == "approval_checkpoint":
            _, _, pending_checkpoint = extract_approval_payload(job.results)
            if job.status != AgentJobStatus.PAUSED.value or not isinstance(
                pending_checkpoint,
                dict,
            ):
                return (
                    False,
                    "Job is not currently paused on an approval checkpoint",
                )
            return True, None
        if normalized_item_type == "job_recovery":
            scheduler_state = (
                ((job.results or {}).get("execution_strategy") or {}).get(
                    "scheduler_state"
                )
                if isinstance(job.results, dict)
                else None
            )
            queue_reason = (
                str((scheduler_state or {}).get("queue_reason") or "").strip().lower()
                if isinstance(scheduler_state, dict)
                else ""
            )
            if queue_reason not in {
                "execution_failure",
                "stalled_run",
                "scheduler_backoff",
            }:
                return (
                    False,
                    "Job is not currently represented as a recovery queue item",
                )
            return True, None
        return False, "Unsupported queue item type"

    @router.post(
        "/checkpoint-queue/bulk-action",
        response_model=AgentCheckpointQueueBulkActionResponse,
    )
    async def checkpoint_queue_bulk_action(
        request: AgentCheckpointQueueBulkActionRequest,
        db: AsyncSession = Depends(get_db),
        current_user: User = Depends(get_current_active_user),
    ):
        item_type = str(request.item_type or "").strip().lower()
        action = str(request.action or "").strip().lower()
        validate_bulk_queue_action(item_type, action)

        ordered_raw_ids: list[str] = []
        id_map: dict[str, UUID | None] = {}
        for raw_id in request.job_ids:
            key = str(raw_id).strip()
            if not key or key in id_map:
                continue
            ordered_raw_ids.append(key)
            try:
                id_map[key] = UUID(key)
            except (ValueError, AttributeError, TypeError):
                id_map[key] = None

        valid_ids = [value for value in id_map.values() if value is not None]
        jobs_result = (
            await db.execute(
                select(AgentJob)
                .options(selectinload(AgentJob.agent_definition))
                .where(
                    and_(
                        AgentJob.user_id == current_user.id,
                        AgentJob.id.in_(valid_ids),
                    )
                )
            )
            if valid_ids
            else None
        )
        jobs = list(jobs_result.scalars().all()) if jobs_result is not None else []
        jobs_by_id = {job.id: job for job in jobs}

        results = []
        for raw_id in ordered_raw_ids:
            job_uuid = id_map.get(raw_id)
            job = jobs_by_id.get(job_uuid) if job_uuid is not None else None
            response_job_id = job_uuid if job_uuid is not None else raw_id
            if job is None:
                results.append(
                    AgentCheckpointQueueBulkActionResultResponse(
                        job_id=response_job_id,
                        ok=False,
                        error="Agent job not found",
                    )
                )
                continue

            matches, mismatch_reason = job_matches_bulk_queue_item_type(
                job,
                item_type,
            )
            if not matches:
                results.append(
                    AgentCheckpointQueueBulkActionResultResponse(
                        job_id=response_job_id,
                        ok=False,
                        status=str(job.status or ""),
                        queue_key=f"{item_type}:{job.id}",
                        error=(
                            mismatch_reason
                            or "Job does not match selected queue item type"
                        ),
                    )
                )
                continue

            try:
                async with db.begin_nested():
                    previous_status = str(job.status or "")
                    updated_job = await perform_job_action(
                        job,
                        AgentJobActionRequest(
                            action=action,
                            checkpoint_note=request.checkpoint_note,
                        ),
                        db=db,
                        current_user=current_user,
                    )
                    await db.flush()
                    await record_operator_event(
                        db=db,
                        job=job,
                        current_user=current_user,
                        action=action,
                        note=request.checkpoint_note,
                        previous_status=previous_status,
                        next_status=str(updated_job.status or ""),
                        scheduler_state=extract_scheduler_state(job),
                        metadata={
                            "queue_item_type": item_type,
                            "bulk_action": True,
                        },
                        summary=(
                            f"{str(job.name or 'Agent job').strip()}: bulk "
                            f"{action.replace('_', ' ')}"
                        ),
                    )
                    results.append(
                        AgentCheckpointQueueBulkActionResultResponse(
                            job_id=response_job_id,
                            ok=True,
                            status=str(updated_job.status or ""),
                            queue_key=f"{item_type}:{updated_job.id}",
                        )
                    )
            except HTTPException as exc:
                results.append(
                    AgentCheckpointQueueBulkActionResultResponse(
                        job_id=response_job_id,
                        ok=False,
                        status=str(job.status or ""),
                        queue_key=f"{item_type}:{job.id}",
                        error=str(exc.detail),
                    )
                )

        await db.commit()
        applied = sum(row.ok for row in results)
        return AgentCheckpointQueueBulkActionResponse(
            requested_count=len(ordered_raw_ids),
            applied=applied,
            failed=len(results) - applied,
            results=results,
        )

    return CheckpointJobActionApi(
        router=router,
        checkpoint_queue_bulk_action=checkpoint_queue_bulk_action,
        validate_bulk_queue_action=validate_bulk_queue_action,
        job_matches_bulk_queue_item_type=job_matches_bulk_queue_item_type,
    )
