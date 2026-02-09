"""
Tool registry and per-user tool policies.
"""

from __future__ import annotations

from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import and_, delete, desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.tool_policy import ToolPolicy
from app.models.user import User
from app.schemas.tool_policy import ToolPolicyCreate, ToolPolicyEvaluateRequest, ToolPolicyEvaluateResponse, ToolPolicyResponse
from app.services.auth_service import get_current_user
from app.services.tool_registry import iter_builtin_tools


router = APIRouter()


@router.get("/registry")
async def list_tool_registry(
    include_custom_tools: bool = Query(True),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    tools = []
    for t in iter_builtin_tools():
        tools.append(
            {
                "name": t.name,
                "display_name": t.name,
                "kind": "builtin",
                "description": t.description,
                "input_schema": t.input_schema,
                "effects": t.effects,
                "network": t.network,
                "cost_tier": t.cost_tier,
                "pii_risk": t.pii_risk,
            }
        )
    # Add MCP-only fallbacks (ensure discoverable)
    from app.services.tool_registry import get_tool_metadata

    for extra in ["search", "list_documents", "get_document", "list_sources", "chat", "create_presentation", "create_repo_report", "get_job_status", "list_jobs"]:
        m = get_tool_metadata(f"mcp:{extra}")
        if m and not any(x["name"] == m.name for x in tools):
            tools.append(
                {
                    "name": m.name,
                    "display_name": extra,
                    "kind": "mcp",
                    "description": m.description,
                    "input_schema": m.input_schema,
                    "effects": m.effects,
                    "network": m.network,
                    "cost_tier": m.cost_tier,
                    "pii_risk": m.pii_risk,
                }
            )

    if include_custom_tools:
        from app.models.workflow import UserTool

        res = await db.execute(
            select(UserTool).where(UserTool.user_id == current_user.id, UserTool.is_enabled == True).order_by(UserTool.name.asc())
        )
        for ut in res.scalars().all():
            schema = ut.parameters_schema if isinstance(ut.parameters_schema, dict) else {}
            tools.append(
                {
                    "name": f"user_tool:{ut.id}",
                    "display_name": ut.name,
                    "kind": "custom_tool",
                    "description": ut.description or "",
                    "input_schema": schema,
                    "effects": "write" if ut.tool_type in {"webhook", "docker_container"} else "read",
                    "network": "egress" if ut.tool_type in {"webhook", "docker_container"} else "none",
                    "cost_tier": "low",
                    "pii_risk": "medium",
                    "tool_type": ut.tool_type,
                    "tool_id": str(ut.id),
                }
            )

    # Attach per-user policy decision snapshot for UI
    try:
        from app.services.tool_policy_engine import evaluate_tool_policy

        for tool in tools:
            decision = await evaluate_tool_policy(db=db, tool_name=tool["name"], tool_args=None, user=current_user)
            tool["allowed"] = bool(decision.allowed)
            tool["require_approval"] = bool(decision.require_approval)
    except Exception:
        for tool in tools:
            tool["allowed"] = True
            tool["require_approval"] = False

    tools.sort(key=lambda x: x["name"])
    return {"tools": tools, "total": len(tools)}

@router.post("/evaluate", response_model=ToolPolicyEvaluateResponse)
async def evaluate_tool_policy_endpoint(
    payload: ToolPolicyEvaluateRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    # For safety, only admins can evaluate policies in other security contexts.
    if (payload.agent_definition_id or payload.api_key_id) and str(current_user.role).lower() != "admin":
        raise HTTPException(status_code=403, detail="Only admins can evaluate with agent_definition_id/api_key_id")

    from app.services.tool_policy_engine import evaluate_tool_policy

    decision = await evaluate_tool_policy(
        db=db,
        tool_name=payload.tool_name,
        tool_args=payload.tool_args,
        user=current_user,
        agent_definition_id=payload.agent_definition_id,
        api_key_id=payload.api_key_id,
    )
    return ToolPolicyEvaluateResponse(
        tool_name=payload.tool_name,
        allowed=bool(decision.allowed),
        require_approval=bool(decision.require_approval),
        denied_reason=decision.denied_reason,
        matched_policies=decision.matched_policies,
    )


@router.get("/policies", response_model=List[ToolPolicyResponse])
async def list_my_tool_policies(
    include_global: bool = Query(True),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    clauses = [(ToolPolicy.subject_type == "user") & (ToolPolicy.subject_id == current_user.id)]
    if include_global:
        clauses.append(ToolPolicy.subject_type == "global")

    combined = clauses[0]
    for c in clauses[1:]:
        combined = combined | c
    stmt = select(ToolPolicy).where(combined).order_by(desc(ToolPolicy.created_at))
    res = await db.execute(stmt)
    return [ToolPolicyResponse.model_validate(p) for p in res.scalars().all()]


@router.post("/policies", response_model=ToolPolicyResponse, status_code=201)
async def create_my_tool_policy(
    payload: ToolPolicyCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    tool_name = payload.tool_name.strip()
    if not tool_name:
        raise HTTPException(status_code=422, detail="tool_name required")

    pol = ToolPolicy(
        subject_type="user",
        subject_id=current_user.id,
        tool_name=tool_name,
        effect=payload.effect,
        require_approval=bool(payload.require_approval),
        constraints=payload.constraints,
    )
    db.add(pol)
    await db.commit()
    await db.refresh(pol)
    return ToolPolicyResponse.model_validate(pol)


@router.delete("/policies/{policy_id}", status_code=204)
async def delete_my_tool_policy(
    policy_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    stmt = delete(ToolPolicy).where(and_(ToolPolicy.id == policy_id, ToolPolicy.subject_type == "user", ToolPolicy.subject_id == current_user.id))
    res = await db.execute(stmt)
    if res.rowcount == 0:
        raise HTTPException(status_code=404, detail="Not found")
    await db.commit()
