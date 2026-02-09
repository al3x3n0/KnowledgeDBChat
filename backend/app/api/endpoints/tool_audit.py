"""
Tool audit & approvals endpoints.
"""

from datetime import datetime
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.tool_audit import ToolExecutionAudit
from app.models.user import User
from app.schemas.agent import AgentToolCall
from app.schemas.tool_audit import ToolAuditResponse, ToolApprovalRequest
from app.services.agent_service import AgentService
from app.services.auth_service import get_current_user

router = APIRouter()

def _approval_mode(row: ToolExecutionAudit) -> str:
    mode = str(row.approval_mode or "").strip().lower()
    if mode in {"owner_or_admin", "owner_and_admin"}:
        return mode
    # Back-compat with old rows
    return "owner_or_admin"


def _recompute_approval_status(row: ToolExecutionAudit) -> None:
    mode = _approval_mode(row)
    if mode == "owner_and_admin":
        has_owner = bool(row.owner_approved_at)
        has_admin = bool(row.admin_approved_at)
        if has_owner and has_admin:
            row.approval_status = "approved"
        elif has_owner and not has_admin:
            row.approval_status = "pending_admin"
        elif has_admin and not has_owner:
            row.approval_status = "pending_owner"
        else:
            # If legacy "pending" slipped in, normalize.
            row.approval_status = row.approval_status or "pending_owner"
    else:
        # Legacy mode uses a single approval.
        row.approval_status = row.approval_status or "pending"


def _to_response(row: ToolExecutionAudit) -> ToolAuditResponse:
    return ToolAuditResponse(
        id=row.id,
        user_id=row.user_id,
        agent_definition_id=row.agent_definition_id,
        conversation_id=row.conversation_id,
        tool_name=row.tool_name,
        tool_input=row.tool_input,
        tool_output=row.tool_output,
        policy_decision=row.policy_decision if isinstance(row.policy_decision, dict) else (row.policy_decision or None),
        status=row.status,
        error=row.error,
        execution_time_ms=row.execution_time_ms,
        approval_required=row.approval_required,
        approval_mode=row.approval_mode,
        approval_status=row.approval_status,
        approved_by=row.approved_by,
        approved_at=row.approved_at,
        approval_note=row.approval_note,
        owner_approved_by=row.owner_approved_by,
        owner_approved_at=row.owner_approved_at,
        admin_approved_by=row.admin_approved_by,
        admin_approved_at=row.admin_approved_at,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


@router.get("/tools", response_model=List[ToolAuditResponse])
async def list_tool_audit(
    status: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    query = select(ToolExecutionAudit).order_by(desc(ToolExecutionAudit.created_at)).limit(limit)
    if status:
        query = query.where(ToolExecutionAudit.status == status)
    if not current_user.is_admin():
        query = query.where(ToolExecutionAudit.user_id == current_user.id)
    result = await db.execute(query)
    rows = result.scalars().all()
    return [_to_response(r) for r in rows]


@router.post("/tools/{audit_id}/approve", response_model=ToolAuditResponse)
async def approve_tool(
    audit_id: UUID,
    payload: ToolApprovalRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(ToolExecutionAudit).where(ToolExecutionAudit.id == audit_id))
    row = result.scalar_one_or_none()
    if not row:
        raise HTTPException(status_code=404, detail="Audit record not found")
    if not (current_user.is_admin() or row.user_id == current_user.id):
        raise HTTPException(status_code=403, detail="Forbidden")
    if not row.approval_required:
        raise HTTPException(status_code=400, detail="No approval required for this record")

    # Normalize status for backward compatibility.
    _recompute_approval_status(row)
    if row.approval_status in {"approved", "rejected"}:
        raise HTTPException(status_code=400, detail="No pending approval for this record")

    now = datetime.utcnow()
    mode = _approval_mode(row)
    is_owner = row.user_id == current_user.id
    is_admin = current_user.is_admin()

    if mode == "owner_and_admin":
        # If the owner is also an admin, a single approval satisfies both.
        if is_owner:
            row.owner_approved_by = current_user.id
            row.owner_approved_at = now
        if is_admin:
            row.admin_approved_by = current_user.id
            row.admin_approved_at = now

        _recompute_approval_status(row)
        # Keep legacy fields populated (last approver / last note).
        row.approved_by = current_user.id
        row.approved_at = now
        row.approval_note = payload.note
    else:
        row.approval_status = "approved"
        row.approved_by = current_user.id
        row.approved_at = now
        row.approval_note = payload.note

    await db.commit()
    await db.refresh(row)
    return _to_response(row)


@router.post("/tools/{audit_id}/reject", response_model=ToolAuditResponse)
async def reject_tool(
    audit_id: UUID,
    payload: ToolApprovalRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(ToolExecutionAudit).where(ToolExecutionAudit.id == audit_id))
    row = result.scalar_one_or_none()
    if not row:
        raise HTTPException(status_code=404, detail="Audit record not found")
    if not (current_user.is_admin() or row.user_id == current_user.id):
        raise HTTPException(status_code=403, detail="Forbidden")
    if not row.approval_required:
        raise HTTPException(status_code=400, detail="No approval required for this record")

    _recompute_approval_status(row)
    if row.approval_status in {"approved", "rejected"}:
        raise HTTPException(status_code=400, detail="No pending approval for this record")

    row.approval_status = "rejected"
    row.approved_by = current_user.id
    row.approved_at = datetime.utcnow()
    row.approval_note = payload.note
    row.status = "failed"
    row.error = "Rejected"
    await db.commit()
    await db.refresh(row)
    return _to_response(row)


@router.post("/tools/{audit_id}/run", response_model=ToolAuditResponse)
async def run_tool(
    audit_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(ToolExecutionAudit).where(ToolExecutionAudit.id == audit_id))
    row = result.scalar_one_or_none()
    if not row:
        raise HTTPException(status_code=404, detail="Audit record not found")
    if not (current_user.is_admin() or row.user_id == current_user.id):
        raise HTTPException(status_code=403, detail="Forbidden")

    if row.status in {"running", "completed"}:
        raise HTTPException(status_code=400, detail=f"Tool is already {row.status}")

    if row.approval_required:
        _recompute_approval_status(row)
        if row.approval_status != "approved":
            raise HTTPException(status_code=400, detail="Tool approval required (not approved yet)")

    # Persist any normalization changes (best-effort).
    await db.commit()

    tool_name = str(row.tool_name or "").strip()
    if tool_name.startswith("mcp:"):
        # Execute MCP tool (by name) using the captured api_key_id.
        import time
        from uuid import UUID as _UUID

        from app.mcp.auth import MCPAuthContext
        from app.mcp.executor import execute_mcp_tool
        from app.models.api_key import APIKey

        short_name = tool_name.split("mcp:", 1)[1].strip()
        payload = row.tool_input if isinstance(row.tool_input, dict) else {}
        args = payload.get("arguments") if isinstance(payload.get("arguments"), dict) else (row.tool_input or {})
        api_key_id_raw = payload.get("api_key_id")

        if not api_key_id_raw:
            raise HTTPException(status_code=422, detail="Missing api_key_id for MCP audit record")

        try:
            api_key_id = _UUID(str(api_key_id_raw))
        except Exception:
            raise HTTPException(status_code=422, detail="Invalid api_key_id for MCP audit record")

        api_key = await db.get(APIKey, api_key_id)
        if not api_key or api_key.user_id != row.user_id:
            raise HTTPException(status_code=404, detail="API key not found")

        owner = await db.get(User, row.user_id)
        if owner is None:
            raise HTTPException(status_code=404, detail="Tool owner not found")

        auth = MCPAuthContext(user=owner, api_key=api_key, scopes=api_key.scopes or [])

        row.status = "running"
        await db.commit()

        try:
            started = time.time()
            result = await execute_mcp_tool(tool_name=short_name, args=args or {}, auth=auth, db=db)
            row.status = "completed"
            row.tool_output = result
            row.execution_time_ms = int((time.time() - started) * 1000)
            await db.commit()
        except Exception as exc:
            row.status = "failed"
            row.error = str(exc)
            await db.commit()

    elif tool_name.startswith("user_tool:"):
        # Execute user-defined tool (by ID)
        from uuid import UUID as _UUID

        from app.models.workflow import UserTool
        from app.services.custom_tool_service import CustomToolService, ToolExecutionError

        try:
            tool_id = _UUID(tool_name.split("user_tool:", 1)[1].strip())
        except Exception:
            raise HTTPException(status_code=422, detail="Invalid user_tool id")

        tool = await db.get(UserTool, tool_id)
        if not tool or tool.user_id != row.user_id:
            raise HTTPException(status_code=404, detail="User tool not found")

        owner = await db.get(User, row.user_id)
        if owner is None:
            raise HTTPException(status_code=404, detail="Tool owner not found")

        service = CustomToolService()
        row.status = "running"
        await db.commit()

        try:
            result = await service.execute_tool(
                tool=tool,
                inputs=row.tool_input or {},
                user=owner,
                db=db,
                bypass_approval_gate=True,
            )
            row.status = "completed"
            row.tool_output = result.get("output")
            row.execution_time_ms = int(result.get("execution_time_ms") or 0)
            await db.commit()
        except ToolExecutionError as exc:
            row.status = "failed"
            row.error = str(exc)
            await db.commit()
    else:
        tool_call = AgentToolCall(
            tool_name=row.tool_name,
            tool_input=row.tool_input or {},
            status="pending",
        )
        agent_service = AgentService()
        await agent_service._execute_tool(
            tool_call,
            user_id=row.user_id,
            db=db,
            conversation_id=row.conversation_id,
            agent_definition_id=row.agent_definition_id,
            audit_row=row,
            bypass_approval_gate=True,
        )

    await db.refresh(row)
    return _to_response(row)
