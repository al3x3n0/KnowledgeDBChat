"""Policy-aware synchronization of bounded CompOps evidence subscriptions."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Mapping, Tuple
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.agent_job import AgentJob
from app.models.compops_evidence_subscription import CompOpsEvidenceSubscription
from app.models.tool_audit import ToolExecutionAudit
from app.models.user import User
from app.models.workflow import UserTool
from app.services.external_agent_gateway_service import (
    ExternalAgentGatewayError,
    external_agent_gateway_service,
)
from app.services.external_system_evidence_link_service import (
    ExternalSystemEvidenceLinkError,
    external_system_evidence_link_service,
)
from app.services.tool_policy_engine import evaluate_tool_policy


class CompOpsEvidenceSyncError(RuntimeError):
    """Raised when a subscription or synchronization request is unsafe."""


class CompOpsEvidenceSyncService:
    SYNC_CAPABILITIES = {
        "compops.studies.report": ("study_id", {"study_id", "metric", "order"}),
        "compops.studies.gates.evaluate": ("study_id", {"study_id"}),
        "compops.runs.get": ("run_id", {"run_id"}),
        "compops.artifacts.get": ("artifact_id", {"artifact_id"}),
        "compops.artifacts.lineage": (
            "artifact_id",
            {"artifact_id", "direction", "depth"},
        ),
    }
    MIN_INTERVAL_MINUTES = 5
    MAX_INTERVAL_MINUTES = 1440

    async def validate_definition(
        self,
        *,
        user: User,
        job_id: UUID,
        tool_id: UUID,
        capability: str,
        payload: Mapping[str, Any],
        db: AsyncSession,
    ) -> Tuple[AgentJob, UserTool, Dict[str, Any], str]:
        normalized_capability = str(capability or "").strip().lower()
        definition = self.SYNC_CAPABILITIES.get(normalized_capability)
        if definition is None:
            raise CompOpsEvidenceSyncError(
                "Only bounded CompOps study, run, and artifact reads can be synchronized"
            )
        job = (
            await db.execute(
                select(AgentJob).where(
                    AgentJob.id == job_id,
                    AgentJob.user_id == user.id,
                )
            )
        ).scalar_one_or_none()
        if job is None:
            raise CompOpsEvidenceSyncError("Agent job was not found")
        if not isinstance(job.results, dict) or not isinstance(
            job.results.get("evaluation_outcome"), dict
        ):
            raise CompOpsEvidenceSyncError(
                "Agent job does not have a canonical R&D outcome"
            )
        tool = (
            await db.execute(
                select(UserTool).where(
                    UserTool.id == tool_id,
                    UserTool.user_id == user.id,
                    UserTool.tool_type == "external_agent",
                )
            )
        ).scalar_one_or_none()
        if tool is None or not bool(tool.is_enabled):
            raise CompOpsEvidenceSyncError(
                "Enabled external-agent connection was not found"
            )
        config = external_agent_gateway_service.validate_config(
            tool.config if isinstance(tool.config, dict) else {}
        )
        if config["provider_type"] != "compops":
            raise CompOpsEvidenceSyncError(
                "Evidence synchronization requires a CompOps connection"
            )
        if normalized_capability not in set(config["capabilities"]):
            raise CompOpsEvidenceSyncError(
                "CompOps connection does not allow this synchronization capability"
            )

        id_field, allowed_fields = definition
        unexpected = sorted(set(payload) - allowed_fields)
        if unexpected:
            raise CompOpsEvidenceSyncError(
                f"Unsupported synchronization fields: {unexpected}"
            )
        remote_id = str(payload.get(id_field) or "").strip()
        if not external_agent_gateway_service.REMOTE_ID_PATTERN.fullmatch(remote_id):
            raise CompOpsEvidenceSyncError(f"{id_field} is invalid")
        normalized_payload: Dict[str, Any] = {id_field: remote_id}
        if normalized_capability == "compops.studies.report":
            for field in ("metric", "order"):
                value = str(payload.get(field) or "").strip()
                if value:
                    normalized_payload[field] = value[:120]
        if normalized_capability == "compops.artifacts.lineage":
            direction = str(payload.get("direction") or "both").strip().lower()
            if direction not in {"upstream", "downstream", "both"}:
                raise CompOpsEvidenceSyncError(
                    "Artifact lineage direction must be upstream, downstream, or both"
                )
            try:
                depth = int(payload.get("depth") or 3)
            except (TypeError, ValueError) as exc:
                raise CompOpsEvidenceSyncError(
                    "Artifact lineage depth must be an integer"
                ) from exc
            if depth < 1 or depth > 20:
                raise CompOpsEvidenceSyncError(
                    "Artifact lineage depth must be between 1 and 20"
                )
            normalized_payload.update({"direction": direction, "depth": depth})

        decision = await evaluate_tool_policy(
            db=db,
            tool_name=f"user_tool:{tool.id}",
            tool_args={
                "capability": normalized_capability,
                "payload": normalized_payload,
                "agent_job_id": str(job.id),
                "scheduled_sync": True,
            },
            user=user,
        )
        if not decision.allowed:
            raise CompOpsEvidenceSyncError(
                decision.denied_reason
                or "CompOps synchronization was denied by tool policy"
            )
        if decision.require_approval:
            raise CompOpsEvidenceSyncError(
                "Scheduled synchronization requires a policy that does not request per-call approval"
            )
        return job, tool, normalized_payload, remote_id

    async def sync(
        self,
        *,
        subscription: CompOpsEvidenceSubscription,
        db: AsyncSession,
        trigger: str = "poll",
        trigger_event_id: str | None = None,
    ) -> bool:
        now = datetime.now(timezone.utc)
        user = await db.get(User, subscription.user_id)
        tool = await db.get(UserTool, subscription.tool_id)
        job = await db.get(AgentJob, subscription.job_id)
        if user is None or tool is None or job is None:
            subscription.is_enabled = False
            subscription.status = "invalid"
            subscription.next_sync_at = None
            subscription.last_error = (
                "Subscription owner, job, or connection is missing"
            )
            subscription.last_attempt_at = now
            await db.commit()
            return False
        if not subscription.is_enabled:
            raise CompOpsEvidenceSyncError("CompOps evidence subscription is disabled")

        tool_args = {
            "capability": subscription.capability,
            "payload": dict(subscription.payload or {}),
            "agent_job_id": str(subscription.job_id),
            "scheduled_sync": True,
            "sync_trigger": str(trigger or "poll")[:32],
        }
        if trigger_event_id:
            tool_args["trigger_event_id"] = str(trigger_event_id)[:200]
        decision = await evaluate_tool_policy(
            db=db,
            tool_name=f"user_tool:{tool.id}",
            tool_args=tool_args,
            user=user,
        )
        subscription.last_attempt_at = now
        if not decision.allowed or decision.require_approval:
            policy_error = (
                "Tool policy now requires approval; re-enable after updating policy"
                if decision.require_approval
                else decision.denied_reason or "Tool policy denied synchronization"
            )
            audit = ToolExecutionAudit(
                user_id=user.id,
                tool_name=f"user_tool:{tool.id}",
                tool_input={
                    **tool_args,
                    "sync_subscription_id": str(subscription.id),
                },
                policy_decision={
                    "allowed": bool(decision.allowed),
                    "require_approval": bool(decision.require_approval),
                    "denied_reason": decision.denied_reason,
                    "matched_policies": decision.matched_policies,
                },
                status="failed",
                error=policy_error,
                approval_required=False,
            )
            db.add(audit)
            await db.flush()
            subscription.is_enabled = False
            subscription.status = (
                "approval_required" if decision.require_approval else "policy_blocked"
            )
            subscription.next_sync_at = None
            subscription.last_error = policy_error
            subscription.last_audit_id = audit.id
            await db.commit()
            return False

        request_id = f"compops-sync-{subscription.id}-{uuid4()}"
        audit = ToolExecutionAudit(
            user_id=user.id,
            tool_name=f"user_tool:{tool.id}",
            tool_input={
                **tool_args,
                "request_id": request_id,
                "sync_subscription_id": str(subscription.id),
            },
            policy_decision={
                "allowed": True,
                "require_approval": False,
                "matched_policies": decision.matched_policies,
            },
            status="running",
            approval_required=False,
        )
        db.add(audit)
        await db.flush()
        try:
            result = await external_agent_gateway_service.invoke(
                tool=tool,
                user=user,
                db=db,
                capability=subscription.capability,
                payload=dict(subscription.payload or {}),
                request_id=request_id,
            )
            audit.status = "completed"
            audit.tool_output = result
            provenance = (
                result.get("provenance")
                if isinstance(result.get("provenance"), Mapping)
                else {}
            )
            audit.execution_time_ms = int(provenance.get("execution_time_ms") or 0)
            changed = await external_system_evidence_link_service.link(
                job_id=subscription.job_id,
                user_id=subscription.user_id,
                tool=tool,
                gateway_result=result,
                audit_id=audit.id,
                db=db,
                evidence_key=str(subscription.id),
            )
            subscription.status = "active"
            subscription.last_error = None
            subscription.last_success_at = now
            subscription.last_response_sha256 = (
                str(provenance.get("response_sha256") or "") or None
            )
            subscription.last_audit_id = audit.id
            subscription.next_sync_at = now + timedelta(
                minutes=subscription.interval_minutes
            )
            await db.commit()
            return changed
        except (ExternalAgentGatewayError, ExternalSystemEvidenceLinkError) as exc:
            audit.status = "failed"
            audit.error = str(exc)[:4000]
            subscription.status = "error"
            subscription.last_error = str(exc)[:4000]
            subscription.last_audit_id = audit.id
            subscription.next_sync_at = now + timedelta(
                minutes=subscription.interval_minutes
            )
            await db.commit()
            return False


compops_evidence_sync_service = CompOpsEvidenceSyncService()
