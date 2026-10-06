"""Images authored for skills: proposed by anyone, built by an administrator.

The server's image allowlist is a setting, and the right answer to "this skill
needs a toolchain none of those images has" used to be a pull request against
`deploy/sandbox-images/`. This is the in-application route to the same place,
and it is deliberately narrower than a Dockerfile.

**Proposing and building are separate people.** Running a confined command is
one grant; running an arbitrary Dockerfile on the host daemon, with the
network available, is a much larger one. So anyone who may author skills may
*propose* an image, and nothing is built until an administrator reads it and
says so -- and then only where `SANDBOX_SKILL_IMAGE_BUILD_ENABLED` is on, which
it is not by default.

**An authored image extends an allowed one.** Exactly one `FROM`, naming an
image already on the server's allowlist. That keeps the sandbox's assumptions
-- a shell at `/bin/sh`, something that works as uid 65534 -- true by
inheritance rather than by hoping the author knew about them.

**No build context.** `COPY` and `ADD` are refused, because there is nothing
to copy from: the build is given the Dockerfile and an empty directory. A file
an image needs is written by a `RUN`.

**No `ENTRYPOINT`.** The sandbox invokes `/bin/sh -lc <script>` as the
container's command. An entrypoint would receive that as its arguments, and
every skill in the image would fail in a way that reads as the skill's fault.

The tag carries the Dockerfile's hash, so an edited proposal is a new image to
approve rather than a silent replacement of one already approved.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional, Tuple

from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.sandbox_skill import SandboxSkillImage
from app.services import agent_sandbox_runtime

SLUG_PATTERN = re.compile(r"^[a-z][a-z0-9-]{1,47}$")
MAX_DOCKERFILE_CHARS = 20_000
BUILD_LOG_TAIL_CHARS = 8_000

#: Repository authored images are tagged under. Local: they are built on the
#: daemon that runs them and never pushed.
TAG_REPOSITORY = "kdbc-skill"

ALLOWED_INSTRUCTIONS = (
    "FROM",
    "RUN",
    "ENV",
    "ARG",
    "WORKDIR",
    "USER",
    "LABEL",
    "SHELL",
)

PROPOSED = "proposed"
BUILDING = "building"
BUILT = "built"
FAILED = "failed"
REJECTED = "rejected"


class ImageError(ValueError):
    """An image proposal that cannot be accepted, with the reason."""


def build_enabled() -> bool:
    from app.core.config import settings

    return bool(getattr(settings, "SANDBOX_SKILL_IMAGE_BUILD_ENABLED", False))


def _instructions(dockerfile: str) -> List[Tuple[str, str]]:
    """(INSTRUCTION, arguments) per logical line, continuations joined."""
    out: List[Tuple[str, str]] = []
    pending = ""
    for raw in dockerfile.splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if not pending and (not stripped or stripped.startswith("#")):
            continue
        if stripped.endswith("\\"):
            pending += stripped[:-1] + " "
            continue
        logical = (pending + stripped).strip()
        pending = ""
        if not logical:
            continue
        head, _, rest = logical.partition(" ")
        out.append((head.upper(), rest.strip()))
    if pending.strip():
        head, _, rest = pending.strip().partition(" ")
        out.append((head.upper(), rest.strip()))
    return out


def validate_dockerfile(dockerfile: Any) -> str:
    """The base image this Dockerfile extends, or an :class:`ImageError`."""
    text = dockerfile if isinstance(dockerfile, str) else ""
    if not text.strip():
        raise ImageError("dockerfile is required")
    if len(text) > MAX_DOCKERFILE_CHARS:
        raise ImageError(
            f"dockerfile is {len(text)} characters; at most {MAX_DOCKERFILE_CHARS}"
        )

    instructions = _instructions(text)
    refused = sorted({name for name, _ in instructions} - set(ALLOWED_INSTRUCTIONS))
    if refused:
        why = []
        if {"COPY", "ADD"} & set(refused):
            why.append(
                "COPY and ADD have nothing to copy from -- the build is given "
                "this Dockerfile and an empty directory, so write files with RUN"
            )
        if {"ENTRYPOINT", "CMD"} & set(refused):
            why.append(
                "ENTRYPOINT and CMD are not allowed because the sandbox runs "
                "/bin/sh -lc as the container's command"
            )
        raise ImageError(
            f"instruction(s) {', '.join(refused)} may not be used. Allowed: "
            f"{', '.join(ALLOWED_INSTRUCTIONS)}"
            + ("; " + "; ".join(why) if why else "")
        )

    froms = [args for name, args in instructions if name == "FROM"]
    if len(froms) != 1:
        raise ImageError(
            f"a skill image has exactly one FROM, and this has {len(froms)}. "
            "Multi-stage builds are not supported."
        )
    if not instructions or instructions[0][0] not in ("FROM", "ARG"):
        raise ImageError("the Dockerfile must begin with FROM")

    parts = froms[0].split()
    base = parts[0] if parts else ""
    if len(parts) > 1 or base.startswith("--"):
        raise ImageError(
            f"FROM {froms[0]!r} must name one image and nothing else (no AS, "
            "no --platform)"
        )
    allowed = agent_sandbox_runtime.allowed_images()
    if base not in allowed:
        raise ImageError(
            f"FROM {base!r} is not an image this server allows. A skill image "
            "extends one that is"
            + (f": {', '.join(allowed)}" if allowed else ", and none is")
            + ". Spell it exactly, including the registry and tag."
        )
    return base


def tag_for(slug: str, dockerfile: str) -> str:
    digest = hashlib.sha256(dockerfile.encode("utf-8")).hexdigest()[:12]
    return f"{TAG_REPOSITORY}/{slug}:{digest}"


async def propose_image(
    db: AsyncSession,
    *,
    user_id: Any,
    slug: Any,
    dockerfile: Any,
    description: Any = None,
) -> SandboxSkillImage:
    name = str(slug or "").strip()
    if not SLUG_PATTERN.match(name):
        raise ImageError(
            f"slug {name!r} is not usable: 2 to 48 characters, lowercase "
            "letters, digits and hyphen, starting with a letter"
        )
    base = validate_dockerfile(dockerfile)
    tag = tag_for(name, dockerfile)
    existing = (
        await db.execute(select(SandboxSkillImage).where(SandboxSkillImage.tag == tag))
    ).scalar_one_or_none()
    if existing is not None:
        raise ImageError(
            f"this exact image has already been proposed as {tag} "
            f"(status: {existing.status})"
        )
    image = SandboxSkillImage(
        slug=name,
        description=str(description or "").strip() or None,
        dockerfile=dockerfile,
        base_image=base,
        tag=tag,
        status=PROPOSED,
        proposed_by=user_id,
    )
    db.add(image)
    await db.commit()
    await db.refresh(image)
    return image


async def list_images(
    db: AsyncSession, *, user_id: Any, is_admin: bool
) -> List[SandboxSkillImage]:
    """Built images for everyone; a person's own proposals; all for an admin."""
    rows = (
        (
            await db.execute(
                select(SandboxSkillImage).order_by(SandboxSkillImage.created_at.desc())
            )
        )
        .scalars()
        .all()
    )
    if is_admin:
        return list(rows)
    return [r for r in rows if r.status == BUILT or r.proposed_by == user_id]


async def get_image(db: AsyncSession, image_id: Any) -> Optional[SandboxSkillImage]:
    return (
        await db.execute(
            select(SandboxSkillImage).where(SandboxSkillImage.id == image_id)
        )
    ).scalar_one_or_none()


async def reject_image(
    db: AsyncSession, image: SandboxSkillImage, admin_id: Any
) -> None:
    if image.status == BUILT:
        raise ImageError(
            "this image is already built; skills may be using it. Remove it "
            "from the daemon by hand if it must go."
        )
    image.status = REJECTED
    image.approved_by = admin_id
    await db.commit()


async def mark_building(
    db: AsyncSession, image: SandboxSkillImage, admin_id: Any
) -> None:
    """Record the approval, before the build is queued."""
    if not build_enabled():
        raise ImageError(
            "Building skill images is disabled on this deployment "
            "(SANDBOX_SKILL_IMAGE_BUILD_ENABLED=false)."
        )
    if image.status == BUILDING:
        raise ImageError("this image is already being built")
    # Re-checked at approval, not only at proposal: the allowlist is a setting
    # and may have changed since.
    validate_dockerfile(image.dockerfile)
    image.status = BUILDING
    image.approved_by = admin_id
    image.build_log = None
    await db.commit()


async def _docker_build(tag: str, dockerfile: str, timeout: int) -> Tuple[int, str]:
    context = Path(tempfile.mkdtemp(prefix="skill_image_"))
    try:
        (context / "Dockerfile").write_text(dockerfile, encoding="utf-8")
        process = await asyncio.create_subprocess_exec(
            "docker",
            "build",
            "--tag",
            tag,
            str(context),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            output, _ = await asyncio.wait_for(process.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            process.kill()
            return 124, f"The build did not finish within {timeout} seconds."
        return int(process.returncode or 0), (output or b"").decode("utf-8", "replace")
    finally:
        shutil.rmtree(context, ignore_errors=True)


async def build_image(db: AsyncSession, image: SandboxSkillImage) -> SandboxSkillImage:
    """Build an approved image and record how it went. Never raises."""
    from app.core.config import settings

    timeout = int(getattr(settings, "SANDBOX_SKILL_IMAGE_BUILD_TIMEOUT_SECONDS", 1800))
    try:
        if not build_enabled():
            raise ImageError(
                "Building skill images is disabled on this deployment "
                "(SANDBOX_SKILL_IMAGE_BUILD_ENABLED=false)."
            )
        validate_dockerfile(image.dockerfile)
        returncode, log = await _docker_build(image.tag, image.dockerfile, timeout)
    except FileNotFoundError:
        returncode, log = 127, (
            "The docker client is not installed where the build runs. Rebuild "
            "backend and celery with WITH_DOCKER_CLI=true, as docker-compose.yml "
            "does."
        )
    except Exception as exc:
        logger.warning(f"Skill image build for {image.tag} failed: {exc}")
        returncode, log = 1, str(exc)

    image.build_log = log[-BUILD_LOG_TAIL_CHARS:]
    if returncode == 0:
        image.status = BUILT
        image.built_at = datetime.now(timezone.utc)
    else:
        image.status = FAILED
    await db.commit()
    await db.refresh(image)
    return image
