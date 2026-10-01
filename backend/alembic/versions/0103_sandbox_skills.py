"""Sandbox skills: a sandboxed capability as data rather than as a handler.

Two tables. `sandbox_skills` holds a user's packaged procedures -- the manifest
kept whole, as a plugin's is, with the hash that last passed its control run
beside it so an edit returns a verified skill to unverified. `sandbox_skill_images`
holds images authored for skills; those are deployment state, proposed by a
person and built by an administrator.

Revision ID: 0103_sandbox_skills
Revises: 0102_inbox_rejection_reason
"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0103_sandbox_skills"
down_revision = "0102_inbox_rejection_reason"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "sandbox_skills",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        # Short and immutable: the evidence a skill yields is named after it.
        sa.Column("slug", sa.String(length=32), nullable=False),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("description", sa.Text(), nullable=False, server_default=""),
        sa.Column(
            "manifest",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default="{}",
        ),
        sa.Column(
            "status", sa.String(length=16), nullable=False, server_default="draft"
        ),
        sa.Column(
            "origin", sa.String(length=16), nullable=False, server_default="manual"
        ),
        sa.Column(
            "origin_job_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("agent_jobs.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "content_hash", sa.String(length=64), nullable=False, server_default=""
        ),
        sa.Column("verified_hash", sa.String(length=64), nullable=True),
        sa.Column(
            "last_dry_run", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.Column(
            "notes",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default="[]",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.UniqueConstraint("user_id", "slug", name="uq_sandbox_skills_user_slug"),
    )
    op.create_index("ix_sandbox_skills_user_id", "sandbox_skills", ["user_id"])
    op.create_index("ix_sandbox_skills_status", "sandbox_skills", ["status"])

    op.create_table(
        "sandbox_skill_images",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("slug", sa.String(length=48), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("dockerfile", sa.Text(), nullable=False),
        sa.Column("base_image", sa.String(length=255), nullable=False),
        sa.Column("tag", sa.String(length=255), nullable=False),
        sa.Column(
            "status", sa.String(length=16), nullable=False, server_default="proposed"
        ),
        sa.Column(
            "proposed_by",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "approved_by",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("build_log", sa.Text(), nullable=True),
        sa.Column("built_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.UniqueConstraint("tag", name="uq_sandbox_skill_images_tag"),
    )
    op.create_index(
        "ix_sandbox_skill_images_status", "sandbox_skill_images", ["status"]
    )


def downgrade() -> None:
    op.drop_index("ix_sandbox_skill_images_status", table_name="sandbox_skill_images")
    op.drop_table("sandbox_skill_images")
    op.drop_index("ix_sandbox_skills_status", table_name="sandbox_skills")
    op.drop_index("ix_sandbox_skills_user_id", table_name="sandbox_skills")
    op.drop_table("sandbox_skills")
