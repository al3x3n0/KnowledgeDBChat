"""Correlate delivered external responses with agent checkpoints and resume jobs."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional
from uuid import uuid4

from sqlalchemy import or_, select, update

from app.models.agent_external_call_outbox import AgentExternalCallOutbox
from app.models.agent_job import AgentJob, AgentJobStatus
from app.services.agent_checkpoint_service import AgentCheckpointService
from app.services.agent_runtime_state_service import initialize_runtime_state


class AgentExternalResponseCorrelationService:
    """Claim completed calls, merge results once, and enqueue a job resume."""

    CLAIM_TTL_SECONDS = 120
    MAX_CHECKPOINT_RESPONSE_BYTES = 20 * 1024

    def __init__(self) -> None:
        self.checkpoint_service = AgentCheckpointService()

    async def claim_next(
        self,
        *,
        db: Any,
        owner_id: str,
        now: Optional[datetime] = None,
    ) -> Optional[AgentExternalCallOutbox]:
        claimed_at = now or datetime.now(timezone.utc)
        row = (
            await db.execute(
                select(AgentExternalCallOutbox)
                .where(
                    AgentExternalCallOutbox.status == "succeeded",
                    AgentExternalCallOutbox.job_id.is_not(None),
                    AgentExternalCallOutbox.correlation.is_not(None),
                    AgentExternalCallOutbox.resume_enqueued_at.is_(None),
                    or_(
                        AgentExternalCallOutbox.resume_claim_expires_at.is_(None),
                        AgentExternalCallOutbox.resume_claim_expires_at <= claimed_at,
                    ),
                )
                .order_by(
                    AgentExternalCallOutbox.delivered_at.asc(),
                    AgentExternalCallOutbox.created_at.asc(),
                )
                .limit(1)
                .with_for_update(skip_locked=True)
            )
        ).scalar_one_or_none()
        if row is None:
            return None
        row.resume_claim_owner = str(owner_id)[:200]
        row.resume_claim_token = str(uuid4())
        row.resume_claim_expires_at = claimed_at + timedelta(
            seconds=self.CLAIM_TTL_SECONDS
        )
        await db.commit()
        await db.refresh(row)
        return row

    async def correlate_and_dispatch(
        self,
        *,
        db: Any,
        row: AgentExternalCallOutbox,
    ) -> Dict[str, Any]:
        """Merge a claimed result, then dispatch behind a fenced marker."""
        claim_token = str(row.resume_claim_token or "")
        if not claim_token:
            return {"status": "claim_conflict", "outbox_id": str(row.id)}

        job = await db.get(AgentJob, row.job_id)
        if job is None:
            return await self._finish_without_resume(
                db=db,
                row=row,
                claim_token=claim_token,
                status="job_missing",
            )
        terminal = {
            AgentJobStatus.COMPLETED.value,
            AgentJobStatus.FAILED.value,
            AgentJobStatus.CANCELLED.value,
        }
        if str(job.status) in terminal:
            return await self._finish_without_resume(
                db=db,
                row=row,
                claim_token=claim_token,
                status="job_terminal",
            )

        checkpoint = await self.checkpoint_service.load_latest_checkpoint(
            job_id=job.id,
            db=db,
        )
        checkpoint_state = checkpoint.state if checkpoint is not None else {}
        state = initialize_runtime_state(checkpoint_state)
        result_key = str(row.id)
        results = state.setdefault("external_call_results", {})
        already_correlated = result_key in results
        if not already_correlated:
            correlation = (
                dict(row.correlation) if isinstance(row.correlation, dict) else {}
            )
            results[result_key] = {
                "outbox_id": result_key,
                "capability": row.capability,
                "status": row.status,
                "delivered_at": (
                    row.delivered_at.isoformat() if row.delivered_at else None
                ),
                "correlation": correlation,
                "response": self._compact_response(row.response),
            }
            state.setdefault("external_calls_pending", {}).pop(result_key, None)
            state.setdefault("findings", []).append(
                {
                    "type": "external_agent_response",
                    "outbox_id": result_key,
                    "capability": row.capability,
                    "summary": f"External capability {row.capability} completed.",
                }
            )
            state.setdefault("artifacts", []).append(
                {
                    "type": "external_call_response",
                    "id": result_key,
                    "capability": row.capability,
                }
            )
            self._complete_waiting_step(
                state=state,
                row=row,
                correlation=correlation,
                iteration=int(job.iteration or 0),
            )
            job.current_phase = "external_response_ready"
            job.phase_details = f"External capability completed: {row.capability}"[:280]
            await self.checkpoint_service.save_checkpoint(
                job=job,
                state=state,
                db=db,
                reason="external_call_response",
            )

        should_resume = (
            str(job.status)
            in {AgentJobStatus.PAUSED.value, AgentJobStatus.PENDING.value}
            and str(job.current_phase) == "external_response_ready"
        )
        if not should_resume:
            return await self._finish_without_resume(
                db=db,
                row=row,
                claim_token=claim_token,
                status="resume_not_applicable",
            )

        job.status = AgentJobStatus.PENDING.value
        await db.commit()
        try:
            from app.tasks.agent_job_tasks import execute_agent_job_task

            execute_agent_job_task.delay(str(job.id), str(job.user_id))
        except Exception:
            await db.execute(
                update(AgentExternalCallOutbox)
                .where(
                    AgentExternalCallOutbox.id == row.id,
                    AgentExternalCallOutbox.resume_claim_token == claim_token,
                )
                .values(
                    resume_claim_owner=None,
                    resume_claim_token=None,
                    resume_claim_expires_at=None,
                )
            )
            job.status = AgentJobStatus.PAUSED.value
            await db.commit()
            return {"status": "resume_retry", "outbox_id": result_key}

        finished_at = datetime.now(timezone.utc)
        updated = await db.execute(
            update(AgentExternalCallOutbox)
            .where(
                AgentExternalCallOutbox.id == row.id,
                AgentExternalCallOutbox.resume_claim_token == claim_token,
                AgentExternalCallOutbox.resume_enqueued_at.is_(None),
            )
            .values(
                correlated_at=finished_at,
                resume_enqueued_at=finished_at,
                resume_claim_owner=None,
                resume_claim_token=None,
                resume_claim_expires_at=None,
            )
        )
        await db.commit()
        return {
            "status": "resume_enqueued"
            if int(updated.rowcount or 0) == 1
            else "claim_conflict",
            "outbox_id": result_key,
            "already_correlated": already_correlated,
        }

    async def _finish_without_resume(
        self,
        *,
        db: Any,
        row: AgentExternalCallOutbox,
        claim_token: str,
        status: str,
    ) -> Dict[str, Any]:
        finished_at = datetime.now(timezone.utc)
        await db.execute(
            update(AgentExternalCallOutbox)
            .where(
                AgentExternalCallOutbox.id == row.id,
                AgentExternalCallOutbox.resume_claim_token == claim_token,
            )
            .values(
                correlated_at=finished_at,
                resume_enqueued_at=finished_at,
                resume_claim_owner=None,
                resume_claim_token=None,
                resume_claim_expires_at=None,
            )
        )
        await db.commit()
        return {"status": status, "outbox_id": str(row.id)}

    def _compact_response(self, response: Any) -> Any:
        try:
            encoded = json.dumps(response, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return {"preview": str(response)[: self.MAX_CHECKPOINT_RESPONSE_BYTES]}
        if len(encoded.encode("utf-8")) <= self.MAX_CHECKPOINT_RESPONSE_BYTES:
            return response
        return {
            "truncated": True,
            "preview": encoded[: self.MAX_CHECKPOINT_RESPONSE_BYTES],
        }

    @staticmethod
    def _complete_waiting_step(
        *,
        state: Dict[str, Any],
        row: AgentExternalCallOutbox,
        correlation: Dict[str, Any],
        iteration: int,
    ) -> None:
        plan = state.get("execution_plan")
        if not isinstance(plan, list) or not plan:
            return
        try:
            index = int(correlation.get("plan_step_index"))
        except (TypeError, ValueError):
            return
        if index < 0 or index >= len(plan) or not isinstance(plan[index], dict):
            return
        step = plan[index]
        expected_id = str(correlation.get("plan_step_id") or "")
        actual_id = str(step.get("step_id") or f"step_{index + 1}")
        if expected_id and expected_id != actual_id:
            return
        if str(step.get("status")) != "waiting_external" or str(
            step.get("external_outbox_id") or ""
        ) != str(row.id):
            return
        step["status"] = "done"
        step["external_response_received"] = True
        step["completed_iteration"] = iteration
        events = state.setdefault("step_events", [])
        events.extend(
            [
                {
                    "type": "external_response_received",
                    "iteration": iteration,
                    "plan_step_id": actual_id,
                    "plan_step_index": index,
                    "outbox_id": str(row.id),
                    "capability": row.capability,
                },
                {
                    "type": "step_completed",
                    "iteration": iteration,
                    "plan_step_id": actual_id,
                    "plan_step_index": index,
                    "tool": "external_agent_response",
                },
            ]
        )
        next_index = min(len(plan) - 1, index + 1)
        state["plan_step_index"] = next_index
        if next_index != index and isinstance(plan[next_index], dict):
            if plan[next_index].get("status") != "done":
                plan[next_index]["status"] = "in_progress"
                events.append(
                    {
                        "type": "step_started",
                        "iteration": iteration,
                        "plan_step_id": str(
                            plan[next_index].get("step_id") or f"step_{next_index + 1}"
                        ),
                        "plan_step_index": next_index,
                        "triggered_by_step_id": actual_id,
                    }
                )
        elif next_index == index:
            state["plan_completed"] = True
        state["step_events"] = events[-500:]


agent_external_response_correlation_service = AgentExternalResponseCorrelationService()
