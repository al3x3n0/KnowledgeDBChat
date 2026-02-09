"""
AI Hub evaluation endpoints (pluggable templates).
"""

from datetime import datetime
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.endpoints.auth import get_current_active_user
from app.core.config import settings
from app.core.database import get_db
from app.models.model_registry import ModelAdapter
from app.models.user import User
from app.schemas.ai_hub_eval import EvalTemplatesResponse, EvalTemplateInfo, RunEvalRequest, RunEvalResponse
from app.core.feature_flags import get_str as get_feature_str
from app.services.ai_hub_eval_service import ai_hub_eval_service
from app.services.model_registry_service import model_registry_service

router = APIRouter()


@router.get("/templates", response_model=EvalTemplatesResponse)
async def list_eval_templates(
    current_user: User = Depends(get_current_active_user),
):
    templates = ai_hub_eval_service.list_templates()
    return EvalTemplatesResponse(
        templates=[
            EvalTemplateInfo(
                id=t.id,
                name=t.name,
                description=t.description,
                version=t.version,
                rubric=t.rubric or {},
                case_count=len(t.cases or []),
            )
            for t in templates
        ]
    )


def _parse_enabled_ids(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [x.strip() for x in raw.split(",") if x and x.strip()]


@router.get("/templates/enabled", response_model=EvalTemplatesResponse)
async def list_enabled_eval_templates(
    current_user: User = Depends(get_current_active_user),
):
    """
    List eval templates available to this deployment/customer.

    Priority:
    1) Redis feature flag `ai_hub_enabled_eval_templates` (CSV)
    2) Env/config `AI_HUB_EVAL_ENABLED_TEMPLATE_IDS` (CSV)
    3) Default: all templates
    """
    override = await get_feature_str("ai_hub_enabled_eval_templates")
    # If an admin has explicitly set the override (even to empty), do not fall back to env.
    if override is None:
        enabled_ids = _parse_enabled_ids(getattr(settings, "AI_HUB_EVAL_ENABLED_TEMPLATE_IDS", None))
    else:
        enabled_ids = _parse_enabled_ids(override)

    templates = ai_hub_eval_service.list_templates()
    if enabled_ids:
        templates = [t for t in templates if t.id in set(enabled_ids)]

    return EvalTemplatesResponse(
        templates=[
            EvalTemplateInfo(
                id=t.id,
                name=t.name,
                description=t.description,
                version=t.version,
                rubric=t.rubric or {},
                case_count=len(t.cases or []),
            )
            for t in templates
        ]
    )


@router.post("/run", response_model=RunEvalResponse)
async def run_eval(
    payload: RunEvalRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    template = ai_hub_eval_service.get_template(payload.template_id)
    if not template:
        raise HTTPException(status_code=404, detail="Eval template not found")

    try:
        adapter_uuid = UUID(payload.adapter_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid adapter_id")

    adapter = await model_registry_service.get_adapter(db, adapter_uuid, current_user.id)
    if not adapter:
        raise HTTPException(status_code=404, detail="Adapter not found")

    if not adapter.is_deployed or not adapter.get_ollama_model_name():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Adapter must be deployed to run eval. Deploy first.",
        )

    base_model = adapter.base_model
    candidate_model = adapter.get_ollama_model_name()
    judge_model = payload.judge_model or base_model

    report = await ai_hub_eval_service.run_eval(
        template=template,
        base_model=base_model,
        candidate_model=candidate_model,
        judge_model=judge_model,
    )

    # Store a lightweight summary on the adapter for "last eval" display.
    if adapter.training_metrics is None:
        adapter.training_metrics = {}
    eval_runs = adapter.training_metrics.get("eval_runs") or []
    if not isinstance(eval_runs, list):
        eval_runs = []
    eval_runs.append(
        {
            "template_id": report["template_id"],
            "template_version": report["template_version"],
            "avg_score": report["avg_score"],
            "num_cases": report["num_cases"],
            "judge_model": report["judge_model"],
            "candidate_model": report["candidate_model"],
            "base_model": report["base_model"],
            "ran_at": datetime.utcnow().isoformat(),
        }
    )
    adapter.training_metrics["eval_runs"] = eval_runs[-20:]
    await db.commit()

    return RunEvalResponse(**report)
