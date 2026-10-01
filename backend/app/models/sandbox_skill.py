"""A sandbox skill: a packaged way of doing one kind of sandboxed work.

Every sandbox-backed capability used to be a Python handler plus a
``ToolSpec`` -- compile, benchmark, simulate, each written in this repository.
A new kind of sandboxed work therefore meant a code change, and a pipeline
could only require evidence one of those handlers produced.

A skill is the same capability as *data*. It carries a procedure the agent
reads, helper files placed in the sandbox, the image it runs in, and the shape
of the result it must leave behind. The agent decides how to carry the
procedure out; the platform decides whether what came back counts as evidence.

Two tables. ``sandbox_skills`` holds one user's skills. ``sandbox_skill_images``
holds images authored for skills, which are deployment state rather than user
state: an image, once built, runs code on the host daemon for everybody, so a
person proposes one and an administrator approves and builds it.
"""

from datetime import datetime
from uuid import uuid4

from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Index,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PostgresUUID

from app.core.database import Base


class SandboxSkill(Base):
    """One packaged procedure, and whether it has been shown to work."""

    __tablename__ = "sandbox_skills"

    id = Column(PostgresUUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(
        PostgresUUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )

    #: Stable identifier. The evidence a skill yields is named after it
    #: (``skill_<slug>``), so it is short and never changes once created.
    slug = Column(String(32), nullable=False)
    name = Column(String(200), nullable=False)
    #: When to reach for this skill. It is the only part an agent sees before
    #: loading one, so it is a column rather than a manifest field.
    description = Column(Text, nullable=False, default="")

    #: The skill as validated: procedure, image, files, result shape, control.
    #: Kept whole for the reason a plugin's manifest is -- it is the author's
    #: document, and what it may carry will grow.
    manifest = Column(JSONB, nullable=False, default=dict)

    #: 'draft' (not offered to any run), 'active', or 'disabled'.
    status = Column(String(16), nullable=False, default="draft")
    #: 'manual', 'drafted' (written from a description) or 'agent' (proposed by
    #: a run). Shown beside the skill: who wrote it is part of how hard a
    #: person should look before activating it.
    origin = Column(String(16), nullable=False, default="manual")
    origin_job_id = Column(
        PostgresUUID(as_uuid=True),
        ForeignKey("agent_jobs.id", ondelete="SET NULL"),
        nullable=True,
    )

    #: Hash of the manifest, and the hash that last passed its control run.
    #: Activation requires the two to match, so editing a verified skill
    #: returns it to unverified rather than leaving the old verdict standing.
    content_hash = Column(String(64), nullable=False, default="")
    verified_hash = Column(String(64), nullable=True)
    #: What the last control run did: ok, when, and the detail when not.
    last_dry_run = Column(JSONB, nullable=True)

    #: What had to be repaired while drafting, or what a run said in proposing.
    notes = Column(JSONB, nullable=False, default=list)

    created_at = Column(
        DateTime(timezone=True), default=datetime.utcnow, nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    __table_args__ = (
        UniqueConstraint("user_id", "slug", name="uq_sandbox_skills_user_slug"),
        Index("ix_sandbox_skills_user_id", "user_id"),
        Index("ix_sandbox_skills_status", "status"),
    )


class SandboxSkillImage(Base):
    """An image authored for skills: proposed by a person, built by an admin."""

    __tablename__ = "sandbox_skill_images"

    id = Column(PostgresUUID(as_uuid=True), primary_key=True, default=uuid4)

    slug = Column(String(48), nullable=False)
    description = Column(Text, nullable=True)
    dockerfile = Column(Text, nullable=False)
    #: The image the Dockerfile builds FROM. Recorded rather than re-parsed so
    #: a reviewer sees what the new image inherits without reading the file.
    base_image = Column(String(255), nullable=False)
    #: What the built image is tagged as, and what a skill names to use it.
    #: Derived from the slug and the Dockerfile's hash, so an edited Dockerfile
    #: is a different image rather than a silent replacement of an approved one.
    tag = Column(String(255), nullable=False)

    #: 'proposed', 'building', 'built', 'failed' or 'rejected'.
    status = Column(String(16), nullable=False, default="proposed")
    proposed_by = Column(
        PostgresUUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )
    approved_by = Column(
        PostgresUUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )
    #: The tail of the build output. The reason a build failed is the only
    #: thing its author can act on.
    build_log = Column(Text, nullable=True)
    built_at = Column(DateTime(timezone=True), nullable=True)

    created_at = Column(
        DateTime(timezone=True), default=datetime.utcnow, nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    __table_args__ = (
        UniqueConstraint("tag", name="uq_sandbox_skill_images_tag"),
        Index("ix_sandbox_skill_images_status", "status"),
    )
