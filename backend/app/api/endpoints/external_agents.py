"""Registry and guarded invocation API for external agents."""

import secrets
from datetime import datetime, timezone
from typing import Any, Dict
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.endpoints.auth import get_current_active_user
from app.core.database import get_db
from app.models.agent_job import AgentJob
from app.models.compops_evidence_subscription import CompOpsEvidenceSubscription
from app.models.secret import UserSecret
from app.models.tool_audit import ToolExecutionAudit
from app.models.user import User
from app.models.workflow import UserTool
from app.schemas.external_agent import (
    CompOpsEvidenceSubscriptionCreateRequest,
    CompOpsEvidenceSubscriptionListResponse,
    CompOpsEvidenceSubscriptionResponse,
    CompOpsEvidenceSubscriptionUpdateRequest,
    CompOpsEvidenceSyncResponse,
    CompOpsWebhookReceiptResponse,
    CompOpsWebhookSetupResponse,
    ExternalAgentCreateRequest,
    ExternalAgentInvokeRequest,
    ExternalAgentInvokeResponse,
    ExternalAgentListResponse,
    ExternalAgentResponse,
)
from app.services.compops_evidence_sync_service import (
    CompOpsEvidenceSyncError,
    compops_evidence_sync_service,
)
from app.services.compops_webhook_service import (
    CompOpsWebhookAuthError,
    CompOpsWebhookConflictError,
    compops_webhook_service,
)
from app.services.custom_tool_service import CustomToolService, ToolExecutionError
from app.services.external_agent_gateway_service import (
    ExternalAgentGatewayError,
    external_agent_gateway_service,
)
from app.services.external_system_evidence_link_service import (
    ExternalSystemEvidenceLinkError,
    external_system_evidence_link_service,
)
from app.services.secret_service import SecretService
from app.services.tool_policy_engine import evaluate_tool_policy

router = APIRouter()


async def _owned_subscription(
    *,
    subscription_id: UUID,
    job_id: UUID,
    user_id: UUID,
    db: AsyncSession,
) -> CompOpsEvidenceSubscription:
    row = (
        await db.execute(
            select(CompOpsEvidenceSubscription).where(
                CompOpsEvidenceSubscription.id == subscription_id,
                CompOpsEvidenceSubscription.job_id == job_id,
                CompOpsEvidenceSubscription.user_id == user_id,
            )
        )
    ).scalar_one_or_none()
    if row is None:
        raise HTTPException(
            status_code=404,
            detail="CompOps evidence subscription was not found",
        )
    return row


def _to_response(tool: UserTool) -> ExternalAgentResponse:
    config = external_agent_gateway_service.validate_config(
        tool.config if isinstance(tool.config, dict) else {}
    )
    return ExternalAgentResponse(
        id=tool.id,
        name=tool.name,
        description=tool.description,
        provider_type=config["provider_type"],
        endpoint_url=config["endpoint_url"],
        capabilities=config["capabilities"],
        auth_type=config["auth_type"],
        secret_id=UUID(config["secret_id"]) if config["secret_id"] else None,
        auth_header_name=config["auth_header_name"],
        timeout_seconds=config["timeout_seconds"],
        is_enabled=bool(tool.is_enabled),
        version=int(tool.version or 1),
        created_at=tool.created_at,
        updated_at=tool.updated_at,
    )


@router.get(
    "/jobs/{job_id}/compops-sync-subscriptions",
    response_model=CompOpsEvidenceSubscriptionListResponse,
)
async def list_compops_evidence_subscriptions(
    job_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    job = (
        await db.execute(
            select(AgentJob.id).where(
                AgentJob.id == job_id,
                AgentJob.user_id == current_user.id,
            )
        )
    ).scalar_one_or_none()
    if job is None:
        raise HTTPException(status_code=404, detail="Agent job not found")
    rows = list(
        (
            await db.execute(
                select(CompOpsEvidenceSubscription)
                .where(
                    CompOpsEvidenceSubscription.job_id == job_id,
                    CompOpsEvidenceSubscription.user_id == current_user.id,
                )
                .order_by(CompOpsEvidenceSubscription.created_at.asc())
            )
        )
        .scalars()
        .all()
    )
    return {"subscriptions": rows, "total": len(rows)}


@router.post(
    "/jobs/{job_id}/compops-sync-subscriptions",
    response_model=CompOpsEvidenceSyncResponse,
    status_code=201,
)
async def create_compops_evidence_subscription(
    job_id: UUID,
    request: CompOpsEvidenceSubscriptionCreateRequest,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    try:
        (
            _,
            _,
            payload,
            remote_id,
        ) = await compops_evidence_sync_service.validate_definition(
            user=current_user,
            job_id=job_id,
            tool_id=request.tool_id,
            capability=request.capability,
            payload=request.payload,
            db=db,
        )
    except (CompOpsEvidenceSyncError, ExternalAgentGatewayError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    existing = (
        await db.execute(
            select(CompOpsEvidenceSubscription).where(
                CompOpsEvidenceSubscription.job_id == job_id,
                CompOpsEvidenceSubscription.tool_id == request.tool_id,
                CompOpsEvidenceSubscription.capability == request.capability,
                CompOpsEvidenceSubscription.remote_id == remote_id,
            )
        )
    ).scalar_one_or_none()
    if existing is not None:
        raise HTTPException(
            status_code=409,
            detail="This CompOps evidence target is already synchronized",
        )
    row = CompOpsEvidenceSubscription(
        user_id=current_user.id,
        job_id=job_id,
        tool_id=request.tool_id,
        capability=request.capability,
        remote_id=remote_id,
        payload=payload,
        interval_minutes=request.interval_minutes,
        is_enabled=True,
        status="active",
        next_sync_at=datetime.now(timezone.utc),
    )
    db.add(row)
    await db.commit()
    await db.refresh(row)
    changed = False
    if request.sync_immediately:
        changed = await compops_evidence_sync_service.sync(
            subscription=row,
            db=db,
        )
        await db.refresh(row)
    return {"subscription": row, "evidence_changed": changed}


@router.patch(
    "/jobs/{job_id}/compops-sync-subscriptions/{subscription_id}",
    response_model=CompOpsEvidenceSubscriptionResponse,
)
async def update_compops_evidence_subscription(
    job_id: UUID,
    subscription_id: UUID,
    request: CompOpsEvidenceSubscriptionUpdateRequest,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    row = await _owned_subscription(
        subscription_id=subscription_id,
        job_id=job_id,
        user_id=current_user.id,
        db=db,
    )
    if request.interval_minutes is not None:
        row.interval_minutes = request.interval_minutes
    if request.is_enabled is not None:
        if request.is_enabled:
            try:
                await compops_evidence_sync_service.validate_definition(
                    user=current_user,
                    job_id=job_id,
                    tool_id=row.tool_id,
                    capability=row.capability,
                    payload=row.payload or {},
                    db=db,
                )
            except (CompOpsEvidenceSyncError, ExternalAgentGatewayError) as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            row.is_enabled = True
            row.status = "active"
            row.last_error = None
            row.next_sync_at = datetime.now(timezone.utc)
        else:
            row.is_enabled = False
            row.status = "paused"
            row.next_sync_at = None
    await db.commit()
    await db.refresh(row)
    return row


@router.post(
    "/jobs/{job_id}/compops-sync-subscriptions/{subscription_id}/sync",
    response_model=CompOpsEvidenceSyncResponse,
)
async def sync_compops_evidence_subscription(
    job_id: UUID,
    subscription_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    row = await _owned_subscription(
        subscription_id=subscription_id,
        job_id=job_id,
        user_id=current_user.id,
        db=db,
    )
    try:
        changed = await compops_evidence_sync_service.sync(
            subscription=row,
            db=db,
        )
    except CompOpsEvidenceSyncError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    await db.refresh(row)
    return {"subscription": row, "evidence_changed": changed}


@router.post(
    "/jobs/{job_id}/compops-sync-subscriptions/{subscription_id}/webhook",
    response_model=CompOpsWebhookSetupResponse,
)
async def enable_compops_subscription_webhook(
    job_id: UUID,
    subscription_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    row = await _owned_subscription(
        subscription_id=subscription_id,
        job_id=job_id,
        user_id=current_user.id,
        db=db,
    )
    plaintext = secrets.token_urlsafe(32)
    secret_name = f"compops-webhook-{row.id}"
    secret = (
        await db.get(UserSecret, row.webhook_secret_id)
        if row.webhook_secret_id is not None
        else None
    )
    if secret is None:
        secret = (
            await db.execute(
                select(UserSecret).where(
                    UserSecret.user_id == current_user.id,
                    UserSecret.name == secret_name,
                )
            )
        ).scalar_one_or_none()
    if secret is None:
        secret = UserSecret(
            user_id=current_user.id,
            name=secret_name,
            encrypted_value=SecretService().encrypt(plaintext),
        )
        db.add(secret)
        await db.flush()
    else:
        if secret.user_id != current_user.id:
            raise HTTPException(status_code=404, detail="Webhook secret not found")
        secret.encrypted_value = SecretService().encrypt(plaintext)
    row.webhook_secret_id = secret.id
    row.webhook_enabled = True
    await db.commit()
    await db.refresh(row)
    return {
        "subscription": row,
        "callback_path": (f"/api/v1/external-agents/compops-webhooks/{row.id}"),
        "signing_secret": plaintext,
    }


@router.delete(
    "/jobs/{job_id}/compops-sync-subscriptions/{subscription_id}/webhook",
    response_model=CompOpsEvidenceSubscriptionResponse,
)
async def disable_compops_subscription_webhook(
    job_id: UUID,
    subscription_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    row = await _owned_subscription(
        subscription_id=subscription_id,
        job_id=job_id,
        user_id=current_user.id,
        db=db,
    )
    row.webhook_enabled = False
    await db.commit()
    await db.refresh(row)
    return row


@router.post(
    "/compops-webhooks/{subscription_id}",
    response_model=CompOpsWebhookReceiptResponse,
    status_code=202,
)
async def receive_compops_subscription_webhook(
    subscription_id: UUID,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > compops_webhook_service.MAX_BODY_BYTES:
            raise HTTPException(status_code=413, detail="Webhook body is too large")
    try:
        event, duplicate = await compops_webhook_service.verify_and_record(
            subscription_id=subscription_id,
            raw_body=bytes(body),
            timestamp_value=request.headers.get("X-CompOps-Timestamp", ""),
            event_id=request.headers.get("X-CompOps-Event-ID", ""),
            signature=request.headers.get("X-CompOps-Signature", ""),
            event_type=request.headers.get("X-CompOps-Event-Type", ""),
            db=db,
        )
    except CompOpsWebhookAuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    except CompOpsWebhookConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if not duplicate:
        try:
            from app.tasks.compops_sync_tasks import sync_compops_webhook_event

            sync_compops_webhook_event.delay(
                str(subscription_id),
                str(event.id),
            )
        except Exception:
            # The committed next_sync_at timestamp preserves Beat reconciliation
            # when the broker is temporarily unavailable.
            pass
    return {
        "accepted": True,
        "duplicate": duplicate,
        "event_id": event.event_id,
    }


@router.get("", response_model=ExternalAgentListResponse)
async def list_external_agents(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    rows = list(
        (
            await db.execute(
                select(UserTool)
                .where(
                    UserTool.user_id == current_user.id,
                    UserTool.tool_type == "external_agent",
                )
                .order_by(UserTool.name.asc())
            )
        )
        .scalars()
        .all()
    )
    return ExternalAgentListResponse(
        agents=[_to_response(row) for row in rows],
        total=len(rows),
    )


@router.post("", response_model=ExternalAgentResponse, status_code=201)
async def create_external_agent(
    request: ExternalAgentCreateRequest,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    if request.secret_id is not None:
        secret = await db.get(UserSecret, request.secret_id)
        if secret is None or secret.user_id != current_user.id:
            raise HTTPException(status_code=404, detail="Secret not found")

    existing = (
        await db.execute(
            select(UserTool).where(
                UserTool.user_id == current_user.id,
                func.lower(UserTool.name) == request.name.lower(),
            )
        )
    ).scalar_one_or_none()
    if existing is not None:
        raise HTTPException(
            status_code=409,
            detail="A user tool with this name already exists",
        )

    raw_config: Dict[str, Any] = {
        "provider_type": request.provider_type,
        "endpoint_url": request.endpoint_url,
        "capabilities": request.capabilities,
        "auth_type": request.auth_type,
        "secret_id": str(request.secret_id) if request.secret_id else None,
        "auth_header_name": request.auth_header_name,
        "timeout_seconds": request.timeout_seconds,
    }
    try:
        config = external_agent_gateway_service.validate_config(raw_config)
    except ExternalAgentGatewayError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    tool = UserTool(
        user_id=current_user.id,
        name=request.name,
        description=request.description,
        tool_type="external_agent",
        parameters_schema={
            "type": "object",
            "required": ["capability", "payload"],
            "properties": {
                "capability": {"type": "string"},
                "payload": {"type": "object"},
                "request_id": {"type": "string"},
            },
        },
        config=config,
        is_enabled=request.is_enabled,
    )
    db.add(tool)
    await db.commit()
    await db.refresh(tool)
    return _to_response(tool)


@router.post(
    "/{agent_id}/invoke",
    response_model=ExternalAgentInvokeResponse,
)
async def invoke_external_agent(
    agent_id: UUID,
    request: ExternalAgentInvokeRequest,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    tool = await db.get(UserTool, agent_id)
    if (
        tool is None
        or tool.user_id != current_user.id
        or tool.tool_type != "external_agent"
    ):
        raise HTTPException(status_code=404, detail="External agent not found")
    if not bool(tool.is_enabled):
        raise HTTPException(status_code=409, detail="External agent is disabled")
    if request.agent_job_id is not None:
        config = external_agent_gateway_service.validate_config(
            tool.config if isinstance(tool.config, dict) else {}
        )
        if config["provider_type"] not in {"compops", "mlflow"}:
            raise HTTPException(
                status_code=422,
                detail=(
                    "Only typed CompOps or MLflow calls can be linked "
                    "to R&D evidence"
                ),
            )
        job = (
            await db.execute(
                select(AgentJob).where(
                    AgentJob.id == request.agent_job_id,
                    AgentJob.user_id == current_user.id,
                )
            )
        ).scalar_one_or_none()
        if job is None:
            raise HTTPException(status_code=404, detail="Agent job not found")
        if not isinstance(job.results, dict) or not isinstance(
            job.results.get("evaluation_outcome"), dict
        ):
            raise HTTPException(
                status_code=409,
                detail="Agent job does not have a canonical R&D outcome",
            )

    tool_name = f"user_tool:{tool.id}"
    tool_input = request.model_dump(mode="json", exclude_none=True)
    decision = await evaluate_tool_policy(
        db=db,
        tool_name=tool_name,
        tool_args=tool_input,
        user=current_user,
    )
    if not decision.allowed:
        raise HTTPException(
            status_code=403,
            detail=decision.denied_reason or "External agent denied by policy",
        )

    audit = ToolExecutionAudit(
        user_id=current_user.id,
        tool_name=tool_name,
        tool_input=tool_input,
        policy_decision={
            "allowed": bool(decision.allowed),
            "require_approval": bool(decision.require_approval),
            "denied_reason": decision.denied_reason,
            "matched_policies": decision.matched_policies,
        },
        status="requires_approval" if decision.require_approval else "running",
        approval_required=bool(decision.require_approval),
        approval_mode="owner_and_admin" if decision.require_approval else None,
        approval_status="pending_owner" if decision.require_approval else None,
    )
    db.add(audit)
    await db.commit()
    await db.refresh(audit)
    if decision.require_approval:
        return ExternalAgentInvokeResponse(
            status="requires_approval",
            audit_id=audit.id,
        )

    try:
        result = await CustomToolService().execute_tool(
            tool=tool,
            inputs=tool_input,
            user=current_user,
            db=db,
            bypass_approval_gate=True,
        )
        audit.status = "completed"
        audit.tool_output = result
        audit.execution_time_ms = int(result.get("execution_time_ms") or 0)
        evidence_linked = False
        if request.agent_job_id is not None:
            evidence_linked = await external_system_evidence_link_service.link(
                job_id=request.agent_job_id,
                user_id=current_user.id,
                tool=tool,
                gateway_result=result.get("output") or {},
                audit_id=audit.id,
                db=db,
            )
        await db.commit()
        return ExternalAgentInvokeResponse(
            status="completed",
            audit_id=audit.id,
            output=None if request.agent_job_id is not None else result.get("output"),
            evidence_linked=evidence_linked,
        )
    except (
        ExternalAgentGatewayError,
        ExternalSystemEvidenceLinkError,
        ToolExecutionError,
    ) as exc:
        audit.status = "failed"
        audit.error = str(exc)[:4000]
        await db.commit()
        return ExternalAgentInvokeResponse(
            status="failed",
            audit_id=audit.id,
            error=str(exc),
        )
