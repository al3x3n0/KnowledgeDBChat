"""Storing skills, and the one rule about offering them to a run.

A skill is offered to an autonomous job only when it is **active**, and it can
only become active while the hash that passed its control run is the hash it
has now. That is the whole lifecycle:

    draft --dry run passes--> verified --activate--> active
      ^                                                 |
      +------------------ any edit ---------------------+

The edit arrow is the one worth stating. A verified skill that is edited is a
different skill, and leaving it active would offer a run something whose
control last passed against other content -- the same shape as a test suite
that was green before the patch.

Skills belong to a user. Images do not: see :mod:`sandbox_skill_image_service`.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.sandbox_skill import SandboxSkill, SandboxSkillImage
from app.services import (
    agent_sandbox_runtime,
    sandbox_skill_manifest,
    sandbox_skill_runtime,
)
from app.services.sandbox_skill_manifest import SkillError

DRAFT = "draft"
ACTIVE = "active"
DISABLED = "disabled"

ORIGINS = ("manual", "drafted", "agent")

#: How many drafts one run may leave behind. A run that proposes a skill per
#: iteration is not contributing knowledge, it is filling somebody's review
#: queue; three is room for a genuine second thought.
MAX_AGENT_DRAFTS_PER_JOB = 3


async def known_images(db: AsyncSession) -> List[str]:
    """Every image a skill may name right now.

    The server's allowlist, plus authored images that have actually been
    built. An image merely proposed -- or approved and not yet built -- is not
    here: a skill naming it would validate and then fail to start.
    """
    allowed = list(agent_sandbox_runtime.allowed_images())
    built = (
        (
            await db.execute(
                select(SandboxSkillImage.tag).where(SandboxSkillImage.status == "built")
            )
        )
        .scalars()
        .all()
    )
    return list(dict.fromkeys([*allowed, *built]))


async def _slug_taken(
    db: AsyncSession, user_id: Any, slug: str, *, excluding: Any = None
) -> bool:
    query = select(func.count(SandboxSkill.id)).where(
        SandboxSkill.user_id == user_id, SandboxSkill.slug == slug
    )
    if excluding is not None:
        query = query.where(SandboxSkill.id != excluding)
    return bool((await db.execute(query)).scalar_one())


async def create_skill(
    db: AsyncSession,
    *,
    user_id: Any,
    raw: Any,
    origin: str = "manual",
    origin_job_id: Any = None,
    notes: Optional[Sequence[str]] = None,
) -> SandboxSkill:
    """Store a skill as a draft. Never active on creation, whoever wrote it."""
    manifest = sandbox_skill_manifest.validate_skill(
        raw, known_images=await known_images(db)
    )
    if await _slug_taken(db, user_id, manifest["id"]):
        raise SkillError(
            f"you already have a skill with id {manifest['id']!r}. Edit that "
            "one, or choose another id."
        )
    skill = SandboxSkill(
        user_id=user_id,
        slug=manifest["id"],
        name=manifest["name"],
        description=manifest["description"],
        manifest=manifest,
        status=DRAFT,
        origin=origin if origin in ORIGINS else "manual",
        origin_job_id=origin_job_id,
        content_hash=sandbox_skill_manifest.content_hash(manifest),
        notes=[str(n) for n in (notes or [])],
    )
    db.add(skill)
    await db.commit()
    await db.refresh(skill)
    return skill


async def update_skill(db: AsyncSession, skill: SandboxSkill, raw: Any) -> SandboxSkill:
    """Replace a skill's content, returning it to unverified if it changed."""
    manifest = sandbox_skill_manifest.validate_skill(
        raw, known_images=await known_images(db)
    )
    if manifest["id"] != skill.slug:
        raise SkillError(
            f"id cannot change from {skill.slug!r} to {manifest['id']!r}. The "
            "evidence a skill yields is named after its id, so renaming it "
            "would leave every contract requiring "
            f"{sandbox_skill_manifest.evidence_type(skill.slug)} with nothing "
            "that produces it. Create a new skill instead."
        )
    new_hash = sandbox_skill_manifest.content_hash(manifest)
    changed = new_hash != skill.content_hash
    skill.name = manifest["name"]
    skill.description = manifest["description"]
    skill.manifest = manifest
    skill.content_hash = new_hash
    if changed and skill.status == ACTIVE:
        skill.status = DRAFT
        skill.notes = [
            *list(skill.notes or []),
            "Edited while active, so it is a draft again until its control "
            "passes against the new content.",
        ]
    await db.commit()
    await db.refresh(skill)
    return skill


def is_verified(skill: SandboxSkill) -> bool:
    return bool(skill.verified_hash) and skill.verified_hash == skill.content_hash


async def dry_run_skill(db: AsyncSession, skill: SandboxSkill) -> Dict[str, Any]:
    """Run the control and record the verdict against the current content."""
    manifest = skill.manifest if isinstance(skill.manifest, dict) else {}
    allowed = str(manifest.get("image") or "") in await known_images(db)
    outcome = await sandbox_skill_runtime.dry_run(manifest, image_allowed=allowed)
    outcome["at"] = datetime.now(timezone.utc).isoformat()
    # A new object, not a mutation: an in-place edit of a JSON column is not
    # seen as a change and is silently not written.
    skill.last_dry_run = dict(outcome)
    if outcome["ok"]:
        skill.verified_hash = skill.content_hash
    elif outcome.get("ran"):
        # It ran and failed, so the old verdict no longer describes this
        # skill. When nothing could run at all the old verdict is untouched:
        # an unreachable daemon says nothing about the skill either way.
        skill.verified_hash = None
        if skill.status == ACTIVE:
            skill.status = DRAFT
    await db.commit()
    await db.refresh(skill)
    return outcome


async def activate_skill(db: AsyncSession, skill: SandboxSkill) -> SandboxSkill:
    if not is_verified(skill):
        raise SkillError(
            f"{skill.slug!r} has not passed its control run"
            + (
                " since it was last edited"
                if skill.verified_hash and skill.verified_hash != skill.content_hash
                else ""
            )
            + ". Run the control first: a skill is offered to a run only once "
            "it has been shown to work."
        )
    image = str((skill.manifest or {}).get("image") or "")
    if image not in await known_images(db):
        raise SkillError(
            f"{skill.slug!r} names image {image!r}, which is no longer allowed "
            "on this server."
        )
    skill.status = ACTIVE
    await db.commit()
    await db.refresh(skill)
    return skill


async def disable_skill(db: AsyncSession, skill: SandboxSkill) -> SandboxSkill:
    skill.status = DISABLED
    await db.commit()
    await db.refresh(skill)
    return skill


async def list_skills(db: AsyncSession, user_id: Any) -> List[SandboxSkill]:
    return list(
        (
            await db.execute(
                select(SandboxSkill)
                .where(SandboxSkill.user_id == user_id)
                .order_by(SandboxSkill.name, SandboxSkill.slug)
            )
        )
        .scalars()
        .all()
    )


async def active_skills(db: AsyncSession, user_id: Any) -> List[SandboxSkill]:
    """What a run is actually offered: active, and still verified."""
    return [
        skill
        for skill in await list_skills(db, user_id)
        if skill.status == ACTIVE and is_verified(skill)
    ]


async def get_active(
    db: AsyncSession, user_id: Any, slug: str
) -> Optional[SandboxSkill]:
    wanted = str(slug or "").strip()
    # A run that read the evidence type off its contract will often pass that
    # rather than the id; both name the same skill.
    wanted = sandbox_skill_manifest.slug_of_evidence(wanted) or wanted
    for skill in await active_skills(db, user_id):
        if skill.slug == wanted:
            return skill
    return None


async def get_owned(
    db: AsyncSession, user_id: Any, skill_id: UUID
) -> Optional[SandboxSkill]:
    return (
        await db.execute(
            select(SandboxSkill).where(
                SandboxSkill.id == skill_id, SandboxSkill.user_id == user_id
            )
        )
    ).scalar_one_or_none()


async def agent_drafts_for_job(db: AsyncSession, job_id: Any) -> int:
    if not job_id:
        return 0
    return int(
        (
            await db.execute(
                select(func.count(SandboxSkill.id)).where(
                    SandboxSkill.origin_job_id == job_id
                )
            )
        ).scalar_one()
        or 0
    )


async def unmet_skill_evidence(
    db: AsyncSession, user_id: Any, finding_types: Iterable[str]
) -> List[str]:
    """Skill evidence in this list that no active skill of this user yields.

    The static pipeline checks cannot answer this -- they have no user -- so
    they accept any `skill_*` type as producible in principle. This is the
    half that knows whose skills exist.
    """
    wanted = [
        str(t).strip()
        for t in finding_types
        if sandbox_skill_manifest.slug_of_evidence(str(t))
    ]
    if not wanted:
        return []
    have = {
        sandbox_skill_manifest.evidence_type(skill.slug)
        for skill in await active_skills(db, user_id)
    }
    return [name for name in dict.fromkeys(wanted) if name not in have]


async def evidence_types_for_user(db: AsyncSession, user_id: Any) -> List[Any]:
    """This user's active skills, as entries in the pipeline vocabulary.

    The vocabulary a pipeline author picks from is derived from the tool specs
    and knows nothing of users. These are the entries it cannot derive: one
    per active skill, so the studio offers `skill_<id>` in its list and the
    drafter is told it exists rather than left to invent a name.
    """
    from app.services import agent_evidence_map
    from app.services.agent_pipeline_vocabulary import EvidenceType

    out = []
    for skill in await active_skills(db, user_id):
        name = sandbox_skill_manifest.evidence_type(skill.slug)
        out.append(
            EvidenceType(
                name=name,
                producers=tuple(agent_evidence_map.producers_of(name)),
                typical_seconds=agent_evidence_map.estimate_chain_seconds([name]),
                consumes=f"the sandbox skill {skill.name!r}: {skill.description}",
            )
        )
    return out


def describe_for_run(skill: SandboxSkill) -> Dict[str, Any]:
    """The one-line view of a skill a run chooses from."""
    manifest: Mapping[str, Any] = skill.manifest or {}
    return {
        "skill": skill.slug,
        "name": skill.name,
        "description": skill.description,
        "produces": sandbox_skill_manifest.evidence_type(skill.slug),
        "image": manifest.get("image"),
    }
