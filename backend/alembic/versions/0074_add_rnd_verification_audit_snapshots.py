"""add immutable R&D verification audit snapshots

Revision ID: 0074_add_rnd_verification_audit_snapshots
Revises: 0073_add_llm_call_snapshots
Create Date: 2026-07-28 00:00:00.000000
"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0074_add_rnd_verification_audit_snapshots"
down_revision = "0073_add_llm_call_snapshots"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "autonomous_rnd_verification_audit_snapshots",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "job_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("agent_jobs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("schema_version", sa.Integer(), nullable=False),
        sa.Column("snapshot", postgresql.JSONB(), nullable=False),
        sa.Column("canonicalization", sa.String(length=64), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column("signature_algorithm", sa.String(length=32), nullable=False),
        sa.Column("signature_encoding", sa.String(length=16), nullable=False),
        sa.Column("signature", sa.String(length=128), nullable=False),
        sa.Column("key_id", sa.String(length=120), nullable=False),
        sa.Column("public_key", sa.String(length=64), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index(
        "ix_autonomous_rnd_verification_audit_snapshots_user_id",
        "autonomous_rnd_verification_audit_snapshots",
        ["user_id"],
    )
    op.create_index(
        "ix_autonomous_rnd_verification_audit_snapshots_job_id",
        "autonomous_rnd_verification_audit_snapshots",
        ["job_id"],
    )
    op.create_index(
        "ix_autonomous_rnd_verification_audit_snapshots_sha256",
        "autonomous_rnd_verification_audit_snapshots",
        ["sha256"],
    )
    op.create_index(
        "ix_autonomous_rnd_verification_audit_snapshots_key_id",
        "autonomous_rnd_verification_audit_snapshots",
        ["key_id"],
    )
    op.create_index(
        "ix_autonomous_rnd_verification_audit_snapshots_created_at",
        "autonomous_rnd_verification_audit_snapshots",
        ["created_at"],
    )
    op.create_index(
        "ix_rnd_verification_audit_job_created",
        "autonomous_rnd_verification_audit_snapshots",
        ["job_id", "created_at"],
    )
    op.execute(
        """
        CREATE FUNCTION reject_rnd_verification_audit_snapshot_update()
        RETURNS trigger AS $$
        BEGIN
            RAISE EXCEPTION 'verification audit snapshots are immutable';
        END;
        $$ LANGUAGE plpgsql
        """
    )
    op.execute(
        """
        CREATE TRIGGER trg_reject_rnd_verification_audit_snapshot_update
        BEFORE UPDATE ON autonomous_rnd_verification_audit_snapshots
        FOR EACH ROW EXECUTE FUNCTION reject_rnd_verification_audit_snapshot_update()
        """
    )


def downgrade() -> None:
    op.execute(
        """
        DROP TRIGGER IF EXISTS trg_reject_rnd_verification_audit_snapshot_update
        ON autonomous_rnd_verification_audit_snapshots
        """
    )
    op.execute("DROP FUNCTION IF EXISTS reject_rnd_verification_audit_snapshot_update")
    op.drop_index(
        "ix_rnd_verification_audit_job_created",
        table_name="autonomous_rnd_verification_audit_snapshots",
    )
    op.drop_index(
        "ix_autonomous_rnd_verification_audit_snapshots_created_at",
        table_name="autonomous_rnd_verification_audit_snapshots",
    )
    op.drop_index(
        "ix_autonomous_rnd_verification_audit_snapshots_key_id",
        table_name="autonomous_rnd_verification_audit_snapshots",
    )
    op.drop_index(
        "ix_autonomous_rnd_verification_audit_snapshots_sha256",
        table_name="autonomous_rnd_verification_audit_snapshots",
    )
    op.drop_index(
        "ix_autonomous_rnd_verification_audit_snapshots_job_id",
        table_name="autonomous_rnd_verification_audit_snapshots",
    )
    op.drop_index(
        "ix_autonomous_rnd_verification_audit_snapshots_user_id",
        table_name="autonomous_rnd_verification_audit_snapshots",
    )
    op.drop_table("autonomous_rnd_verification_audit_snapshots")
