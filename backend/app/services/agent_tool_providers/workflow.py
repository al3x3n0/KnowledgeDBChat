"""Autonomous-job tools: the ``workflow`` provider.

Split out of ``agent_tool_dispatch``, which re-exports every name here.
"""

from __future__ import annotations

import json
from typing import Any, Dict

from sqlalchemy import select

from app.services.agent_tool_providers.base import (
    AgentToolExecutionContext,
    FunctionToolProvider,
)


def build_autonomous_workflow_provider(executor: Any) -> FunctionToolProvider:
    """Workflow orchestration tools for AutonomousAgentExecutor."""

    async def _list_available_workflows(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from sqlalchemy import select as _select
        from sqlalchemy.orm import selectinload as _selectinload

        from app.models.workflow import Workflow

        job = ctx.job
        try:
            is_active = params.get("is_active", True)
            wf_query = _select(Workflow).where(Workflow.user_id == job.user_id)
            if is_active is not None:
                wf_query = wf_query.where(Workflow.is_active == bool(is_active))
            wf_query = (
                wf_query.options(_selectinload(Workflow.nodes))
                .order_by(Workflow.updated_at.desc())
                .limit(20)
            )
            wf_result = await ctx.db.execute(wf_query)
            workflows = wf_result.scalars().all()
            return {
                "success": True,
                "data": {
                    "workflows": [
                        {
                            "id": str(wf.id),
                            "name": wf.name,
                            "description": wf.description or "",
                            "is_active": wf.is_active,
                            "node_count": len(wf.nodes) if wf.nodes else 0,
                        }
                        for wf in workflows
                    ],
                    "count": len(workflows),
                },
            }
        except Exception as exc:
            return {"error": f"Failed to list workflows: {exc}"}

    def _bounded_json(value: Any, limit: int = 20000) -> Any:
        """`value` if it serialises within `limit` characters, else a note
        saying how large it was and its top-level keys."""
        if not value:
            return {}
        try:
            size = len(json.dumps(value, default=str))
        except Exception:
            return {"_unreadable": True}
        if size <= limit:
            return value
        keys = sorted(value) if isinstance(value, dict) else []
        return {"_truncated": True, "_size": size, "_keys": keys[:100]}

    async def _execute_workflow(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from uuid import UUID as _UUID

        from app.models.user import User as _User
        from app.services.workflow_engine import WorkflowEngine

        job = ctx.job
        wf_id_str = str(params.get("workflow_id", "")).strip()
        if not wf_id_str:
            return {"error": "workflow_id is required"}
        try:
            user_obj = await ctx.db.get(_User, job.user_id)
            if not user_obj:
                return {"error": "Could not load user for workflow execution"}
            engine = WorkflowEngine(ctx.db, user_obj)
            # Queued, not run here: the spec says this launches, and running
            # inline held the whole agent turn on every node of the graph.
            execution = await engine.queue_workflow(
                workflow_id=_UUID(wf_id_str),
                trigger_type="agent_job",
                trigger_data=params.get("trigger_data")
                or {"source_job_id": str(job.id)},
                initial_context=params.get("inputs"),
            )
            return {
                "success": True,
                "data": {
                    "execution_id": str(execution.id),
                    "status": execution.status,
                    "workflow_id": wf_id_str,
                },
            }
        except Exception as exc:
            # The engine commits a failed execution before it raises. Name
            # it, or the run cannot ask what went wrong and a retry simply
            # makes another.
            failed_id = None
            try:
                from app.models.workflow import WorkflowExecution

                await ctx.db.rollback()
                failed_id = (
                    await ctx.db.execute(
                        select(WorkflowExecution.id)
                        .where(
                            WorkflowExecution.workflow_id == _UUID(wf_id_str),
                            WorkflowExecution.user_id == job.user_id,
                            WorkflowExecution.status == "failed",
                        )
                        .order_by(WorkflowExecution.created_at.desc())
                        .limit(1)
                    )
                ).scalar_one_or_none()
            except Exception:
                failed_id = None
            if failed_id is not None:
                return {
                    "error": f"Workflow execution failed: {exc} "
                    f"(execution_id {failed_id})",
                    "execution_id": str(failed_id),
                    "status": "failed",
                }
            return {"error": f"Workflow execution failed: {exc}"}

    async def _get_workflow_status(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from uuid import UUID as _UUID

        from sqlalchemy import select as _select

        from app.models.workflow import WorkflowExecution

        exec_id_str = str(params.get("execution_id", "")).strip()
        if not exec_id_str:
            return {"error": "execution_id is required"}
        try:
            # The caller's own executions only; a stranger's reads as absent.
            exec_result = await ctx.db.execute(
                _select(WorkflowExecution).where(
                    WorkflowExecution.id == _UUID(exec_id_str),
                    WorkflowExecution.user_id == ctx.job.user_id,
                )
            )
            execution = exec_result.scalar_one_or_none()
            if not execution:
                return {"error": f"Workflow execution {exec_id_str} not found"}
            return {
                "success": True,
                "data": {
                    "execution_id": str(execution.id),
                    "workflow_id": str(execution.workflow_id),
                    "status": execution.status,
                    "progress": execution.progress,
                    "error": execution.error,
                    "started_at": str(execution.started_at)
                    if execution.started_at
                    else None,
                    "completed_at": str(execution.completed_at)
                    if execution.completed_at
                    else None,
                    # What the workflow produced: without it a run could
                    # start a workflow and never read its result.
                    "output": _bounded_json(execution.context),
                },
            }
        except Exception as exc:
            return {"error": f"Failed to get workflow status: {exc}"}

    async def _enqueue_external_agent_call(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        import hashlib as _hashlib
        import json as _json
        from uuid import UUID as _UUID

        from app.models.agent_job import AgentJobStatus as _AgentJobStatus
        from app.models.user import User as _User
        from app.models.workflow import UserTool as _UserTool
        from app.services.agent_external_call_outbox_service import (
            AgentExternalCallOutboxError,
            agent_external_call_outbox_service,
        )
        from app.services.external_agent_gateway_service import (
            external_agent_gateway_service,
        )
        from app.services.tool_policy_engine import evaluate_tool_policy

        job = ctx.job
        try:
            tool_id = _UUID(str(params.get("tool_id") or "").strip())
        except (TypeError, ValueError):
            return {"error": "tool_id must be a valid external-agent connection ID"}
        capability = str(params.get("capability") or "").strip().lower()
        payload = params.get("payload")
        if not capability:
            return {"error": "capability is required"}
        if not isinstance(payload, dict):
            return {"error": "payload must be an object"}
        user = await ctx.db.get(_User, job.user_id)
        tool = await ctx.db.get(_UserTool, tool_id)
        if (
            user is None
            or tool is None
            or tool.user_id != job.user_id
            or tool.tool_type != "external_agent"
            or not bool(tool.is_enabled)
        ):
            return {"error": "Enabled external-agent connection was not found"}
        try:
            gateway_config = external_agent_gateway_service.validate_config(
                tool.config if isinstance(tool.config, dict) else {}
            )
        except Exception as exc:
            return {"error": f"External-agent connection is invalid: {exc}"}
        if capability not in set(gateway_config.get("capabilities") or []):
            return {"error": "Capability is not allowed by this connection"}
        decision = await evaluate_tool_policy(
            db=ctx.db,
            tool_name=f"user_tool:{tool.id}",
            tool_args={
                "capability": capability,
                "payload": payload,
                "agent_job_id": str(job.id),
                "delivery_mode": "transactional_outbox",
            },
            user=user,
        )
        if not decision.allowed:
            return {
                "error": decision.denied_reason
                or "External-agent call was denied by tool policy"
            }
        if decision.require_approval:
            return {
                "error": (
                    "External-agent call requires approval before it can be " "enqueued"
                ),
                "approval_required": True,
            }
        idempotency_key = str(
            params.get("idempotency_key") or ctx.idempotency_key or ""
        ).strip()
        if not idempotency_key:
            fingerprint = _json.dumps(
                {
                    "job_id": str(job.id),
                    "iteration": int(job.iteration or 0),
                    "tool_id": str(tool.id),
                    "capability": capability,
                    "payload": payload,
                },
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
            idempotency_key = _hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()
        state = ctx.state if isinstance(ctx.state, dict) else {}
        plan = state.get("execution_plan")
        plan_step_index = int(state.get("plan_step_index", 0) or 0)
        plan_step = None
        if isinstance(plan, list) and plan:
            plan_step_index = max(0, min(plan_step_index, len(plan) - 1))
            plan_step = plan[plan_step_index]
        plan_step_id = (
            str(plan_step.get("step_id") or f"step_{plan_step_index + 1}")
            if isinstance(plan_step, dict)
            else None
        )
        correlation = {
            "job_id": str(job.id),
            "iteration": int(job.iteration or 0),
            "plan_step_id": plan_step_id,
            "plan_step_index": plan_step_index if plan_step_id else None,
            "journal_idempotency_key": idempotency_key,
        }
        try:
            row, created = await agent_external_call_outbox_service.enqueue(
                db=ctx.db,
                job_id=job.id,
                user_id=job.user_id,
                tool_id=tool.id,
                capability=capability,
                payload=payload,
                idempotency_key=idempotency_key,
                max_attempts=params.get("max_attempts", 5),
                correlation=correlation,
            )
        except AgentExternalCallOutboxError as exc:
            return {"error": str(exc)}
        deferred = str(row.status) != "succeeded"
        if deferred:
            pending = state.setdefault("external_calls_pending", {})
            pending[str(row.id)] = {
                **correlation,
                "capability": capability,
                "status": str(row.status),
            }
            if isinstance(plan_step, dict):
                plan_step["status"] = "waiting_external"
                plan_step["external_outbox_id"] = str(row.id)
                plan_step["external_capability"] = capability
                plan_step["waiting_since_iteration"] = int(job.iteration or 0)
            job.status = _AgentJobStatus.PAUSED.value
            job.current_phase = "awaiting_external"
            job.phase_details = f"Waiting for external capability: {capability}"[:280]
        return {
            "success": True,
            "deferred_external": deferred,
            "correlation": correlation,
            "data": {
                "outbox_id": str(row.id),
                "status": str(row.status),
                "created": created,
                "idempotency_key": row.idempotency_key,
                "request_id": row.request_id,
                "response": row.response if not deferred else None,
            },
            "artifacts": [
                {
                    "type": "external_call_outbox",
                    "id": str(row.id),
                    "status": str(row.status),
                }
            ],
        }

    async def _get_external_call_status(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from uuid import UUID as _UUID

        from app.models.agent_external_call_outbox import AgentExternalCallOutbox

        try:
            outbox_id = _UUID(str(params.get("outbox_id") or "").strip())
        except (TypeError, ValueError):
            return {"error": "outbox_id must be a valid UUID"}
        row = await ctx.db.get(AgentExternalCallOutbox, outbox_id)
        if (
            row is None
            or row.user_id != ctx.job.user_id
            or (row.job_id is not None and row.job_id != ctx.job.id)
        ):
            return {"error": "External-call outbox row was not found"}
        return {
            "success": True,
            "data": {
                "outbox_id": str(row.id),
                "status": str(row.status),
                "attempts": int(row.attempts or 0),
                "max_attempts": int(row.max_attempts or 0),
                "next_attempt_at": (
                    row.next_attempt_at.isoformat()
                    if row.next_attempt_at is not None
                    else None
                ),
                "delivered_at": (
                    row.delivered_at.isoformat()
                    if row.delivered_at is not None
                    else None
                ),
                "correlated_at": (
                    row.correlated_at.isoformat()
                    if row.correlated_at is not None
                    else None
                ),
                "resume_enqueued_at": (
                    row.resume_enqueued_at.isoformat()
                    if row.resume_enqueued_at is not None
                    else None
                ),
                "error": str(row.error or "")[:1000] or None,
                "response": (
                    row.response
                    if row.status == "succeeded" and isinstance(row.response, dict)
                    else None
                ),
            },
        }

    return FunctionToolProvider(
        name="autonomous_workflow_tools",
        modes={"autonomous"},
        handlers={
            "list_available_workflows": _list_available_workflows,
            "execute_workflow": _execute_workflow,
            "get_workflow_status": _get_workflow_status,
            "enqueue_external_agent_call": _enqueue_external_agent_call,
            "get_external_call_status": _get_external_call_status,
        },
    )
