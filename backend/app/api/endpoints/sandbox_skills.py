"""Sandbox skills over HTTP.

Three ways in -- write one by hand, draft one from a description, or review one
a run proposed -- and they all arrive at the same place: a **draft**. Nothing
here makes a skill active except `activate`, and `activate` refuses a skill
whose control has not passed against the content it has now. The endpoints are
arranged so that there is no other path.

Images are deployment state rather than a user's, so they sit under `/images`
with a different rule: anyone who may author skills may *propose* one, and only
an administrator may have it built.
"""

from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.endpoints.auth import get_current_active_user
from app.core.config import settings
from app.core.database import get_db
from app.models.sandbox_skill import SandboxSkill
from app.models.user import User
from app.schemas.sandbox_skill import (
    SandboxSkillDraftQueued,
    SandboxSkillDraftRequest,
    SandboxSkillDraftStatus,
    SandboxSkillDryRunResponse,
    SandboxSkillImageListResponse,
    SandboxSkillImagePropose,
    SandboxSkillImageResponse,
    SandboxSkillListResponse,
    SandboxSkillResponse,
    SandboxSkillWrite,
)
from app.services import (
    agent_sandbox_runtime,
    sandbox_skill_image_service,
    sandbox_skill_manifest,
    sandbox_skill_service,
)
from app.services.sandbox_skill_image_service import ImageError
from app.services.sandbox_skill_manifest import SkillError

router = APIRouter()


def _authoring_enabled() -> bool:
    return bool(getattr(settings, "SANDBOX_SKILLS_AUTHORING_ENABLED", True))


def _require_authoring() -> None:
    if not _authoring_enabled():
        raise HTTPException(
            status_code=403,
            detail=(
                "Authoring sandbox skills is disabled on this deployment "
                "(SANDBOX_SKILLS_AUTHORING_ENABLED=false)."
            ),
        )


def _respond(skill: SandboxSkill) -> SandboxSkillResponse:
    return SandboxSkillResponse(
        id=skill.id,
        slug=skill.slug,
        name=skill.name,
        description=skill.description or "",
        manifest=skill.manifest if isinstance(skill.manifest, dict) else {},
        status=skill.status,
        origin=skill.origin,
        origin_job_id=skill.origin_job_id,
        produces=sandbox_skill_manifest.evidence_type(skill.slug),
        verified=sandbox_skill_service.is_verified(skill),
        last_dry_run=skill.last_dry_run
        if isinstance(skill.last_dry_run, dict)
        else None,
        notes=[str(n) for n in (skill.notes or [])],
        created_at=skill.created_at,
        updated_at=skill.updated_at,
    )


async def _owned(db: AsyncSession, user: User, skill_id: UUID) -> SandboxSkill:
    skill = await sandbox_skill_service.get_owned(db, user.id, skill_id)
    if skill is None:
        # 404 for someone else's skill as well: confirming it exists is the
        # thing being withheld.
        raise HTTPException(status_code=404, detail="No such skill")
    return skill


@router.get("", response_model=SandboxSkillListResponse)
async def list_skills(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """This user's skills, and what a new one could be built from."""
    skills = await sandbox_skill_service.list_skills(db, current_user.id)
    return SandboxSkillListResponse(
        items=[_respond(s) for s in skills],
        images=await sandbox_skill_service.known_images(db),
        execution_enabled=agent_sandbox_runtime.execution_enabled(),
        authoring_enabled=_authoring_enabled(),
        image_build_enabled=sandbox_skill_image_service.build_enabled(),
    )


@router.post(
    "", response_model=SandboxSkillResponse, status_code=status.HTTP_201_CREATED
)
async def create_skill(
    payload: SandboxSkillWrite,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Store a skill as a draft. It is never active on creation."""
    _require_authoring()
    try:
        skill = await sandbox_skill_service.create_skill(
            db, user_id=current_user.id, raw=payload.manifest, origin="manual"
        )
    except SkillError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return _respond(skill)


@router.post(
    "/draft",
    response_model=SandboxSkillDraftQueued,
    status_code=status.HTTP_202_ACCEPTED,
)
async def draft_skill(
    payload: SandboxSkillDraftRequest,
    current_user: User = Depends(get_current_active_user),
):
    """Start drafting a skill. Returns immediately; poll for the result.

    Drafting validates what it wrote and runs its control in the sandbox, and
    asks the model again when either refuses. It stores nothing: the skill
    comes back for review, and creating it is a separate, deliberate request.
    """
    _require_authoring()

    from app.tasks.sandbox_skill_tasks import draft_sandbox_skill

    task = draft_sandbox_skill.delay(
        payload.description, str(current_user.id), payload.current
    )
    return SandboxSkillDraftQueued(
        task_id=task.id, poll_url=f"/api/v1/sandbox-skills/draft/{task.id}"
    )


@router.get("/draft/{task_id}", response_model=SandboxSkillDraftStatus)
async def get_skill_draft(
    task_id: str,
    current_user: User = Depends(get_current_active_user),
):
    """Where a draft has got to, and its result once there is one."""
    from celery.result import AsyncResult

    from app.core.celery import celery_app

    result = AsyncResult(task_id, app=celery_app)
    state = str(result.state or "PENDING")
    info = result.info if isinstance(result.info, dict) else {}

    owner = str(info.get("user_id") or "")
    if owner and owner != str(current_user.id):
        raise HTTPException(status_code=404, detail="No such draft")

    if state == "SUCCESS":
        return SandboxSkillDraftStatus(
            state=state,
            stage="done",
            attempt=int(info.get("attempts") or 0),
            notes=[str(n) for n in (info.get("notes") or [])],
            manifest=info.get("manifest"),
            dry_run=info.get("dry_run"),
            attempts=int(info.get("attempts") or 0),
            pending=False,
        )
    if state == "FAILURE":
        # The task returns its failures rather than raising, so arriving here
        # means the worker itself died.
        return SandboxSkillDraftStatus(
            state=state,
            stage="done",
            notes=["The worker drafting this stopped before it finished."],
            pending=False,
        )
    return SandboxSkillDraftStatus(
        state=state,
        stage=str(info.get("stage") or "") or None,
        attempt=int(info.get("attempt") or 0),
        notes=[str(n) for n in (info.get("notes") or [])],
        pending=True,
    )


# ---------------------------------------------------------------- images


@router.get("/images", response_model=SandboxSkillImageListResponse)
async def list_images(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Built images, your own proposals, and -- for an admin -- everything."""
    images = await sandbox_skill_image_service.list_images(
        db, user_id=current_user.id, is_admin=current_user.is_admin()
    )
    return SandboxSkillImageListResponse(
        items=[SandboxSkillImageResponse.model_validate(i) for i in images],
        build_enabled=sandbox_skill_image_service.build_enabled(),
    )


@router.post(
    "/images",
    response_model=SandboxSkillImageResponse,
    status_code=status.HTTP_201_CREATED,
)
async def propose_image(
    payload: SandboxSkillImagePropose,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Propose an image. Builds nothing; an administrator decides that."""
    _require_authoring()
    try:
        image = await sandbox_skill_image_service.propose_image(
            db,
            user_id=current_user.id,
            slug=payload.slug,
            dockerfile=payload.dockerfile,
            description=payload.description,
        )
    except ImageError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return SandboxSkillImageResponse.model_validate(image)


def _require_admin(user: User) -> None:
    if not user.is_admin():
        raise HTTPException(
            status_code=403,
            detail="Only an administrator may build or reject a skill image.",
        )


@router.post("/images/{image_id}/build", response_model=SandboxSkillImageResponse)
async def build_image(
    image_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Approve an image and queue its build.

    Approval and build are one action on purpose: an approved image that was
    never built is a state with no use, and a skill naming it would validate
    and then fail to start.
    """
    _require_admin(current_user)
    image = await sandbox_skill_image_service.get_image(db, image_id)
    if image is None:
        raise HTTPException(status_code=404, detail="No such image")
    try:
        await sandbox_skill_image_service.mark_building(db, image, current_user.id)
    except ImageError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    from app.tasks.sandbox_skill_tasks import build_sandbox_skill_image

    build_sandbox_skill_image.delay(str(image.id))
    await db.refresh(image)
    return SandboxSkillImageResponse.model_validate(image)


@router.post("/images/{image_id}/reject", response_model=SandboxSkillImageResponse)
async def reject_image(
    image_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    _require_admin(current_user)
    image = await sandbox_skill_image_service.get_image(db, image_id)
    if image is None:
        raise HTTPException(status_code=404, detail="No such image")
    try:
        await sandbox_skill_image_service.reject_image(db, image, current_user.id)
    except ImageError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    await db.refresh(image)
    return SandboxSkillImageResponse.model_validate(image)


# ---------------------------------------------------------------- one skill


@router.get("/{skill_id}", response_model=SandboxSkillResponse)
async def get_skill(
    skill_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    return _respond(await _owned(db, current_user, skill_id))


@router.put("/{skill_id}", response_model=SandboxSkillResponse)
async def update_skill(
    skill_id: UUID,
    payload: SandboxSkillWrite,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Replace a skill's content. An active skill that changes is a draft again."""
    _require_authoring()
    skill = await _owned(db, current_user, skill_id)
    try:
        skill = await sandbox_skill_service.update_skill(db, skill, payload.manifest)
    except SkillError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return _respond(skill)


@router.delete("/{skill_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_skill(
    skill_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    skill = await _owned(db, current_user, skill_id)
    await db.delete(skill)
    await db.commit()


@router.post("/{skill_id}/dry-run", response_model=SandboxSkillDryRunResponse)
async def dry_run_skill(
    skill_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Run the skill's control in the sandbox and record the verdict.

    Held on the request because a control is meant to be the smallest thing
    that works; it is cut off at two minutes, and one that needs longer is
    told that it is not a control.
    """
    skill = await _owned(db, current_user, skill_id)
    outcome = await sandbox_skill_service.dry_run_skill(db, skill)
    return SandboxSkillDryRunResponse(
        skill=_respond(skill),
        ok=bool(outcome.get("ok")),
        ran=bool(outcome.get("ran")),
        detail=str(outcome.get("detail") or ""),
        returncode=outcome.get("returncode"),
        stdout=str(outcome.get("stdout") or ""),
        stderr=str(outcome.get("stderr") or ""),
        result=outcome.get("result"),
    )


@router.post("/{skill_id}/activate", response_model=SandboxSkillResponse)
async def activate_skill(
    skill_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Offer the skill to this user's runs. Refused until its control passes."""
    skill = await _owned(db, current_user, skill_id)
    try:
        skill = await sandbox_skill_service.activate_skill(db, skill)
    except SkillError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return _respond(skill)


@router.post("/{skill_id}/disable", response_model=SandboxSkillResponse)
async def disable_skill(
    skill_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Stop offering the skill, keeping it and its verdict."""
    skill = await _owned(db, current_user, skill_id)
    return _respond(await sandbox_skill_service.disable_skill(db, skill))


__all__: List[str] = ["router"]
