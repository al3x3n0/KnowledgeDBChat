"""Add artifact drafts and retrieval traces.

Revision ID: 0049_add_artifact_drafts_and_retrieval_traces
Revises: 0048_add_tool_audit_dual_approvals
Create Date: 2026-02-05 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa


revision = "0049_add_artifact_drafts_and_retrieval_traces"
down_revision = "0048_add_tool_audit_dual_approvals"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "retrieval_traces",
        sa.Column("id", sa.dialects.postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", sa.dialects.postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("session_id", sa.dialects.postgresql.UUID(as_uuid=True), sa.ForeignKey("chat_sessions.id", ondelete="SET NULL"), nullable=True),
        sa.Column("chat_message_id", sa.dialects.postgresql.UUID(as_uuid=True), sa.ForeignKey("chat_messages.id", ondelete="SET NULL"), nullable=True),
        sa.Column("trace_type", sa.String(length=32), nullable=False, server_default="chat"),
        sa.Column("query", sa.Text(), nullable=False),
        sa.Column("processed_query", sa.Text(), nullable=True),
        sa.Column("provider", sa.String(length=32), nullable=True),
        sa.Column("settings_snapshot", sa.JSON(), nullable=True),
        sa.Column("trace", sa.JSON(), nullable=False, server_default=sa.text("'{}'::json")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.create_index("ix_retrieval_traces_user_created", "retrieval_traces", ["user_id", "created_at"])
    op.create_index("ix_retrieval_traces_session_created", "retrieval_traces", ["session_id", "created_at"])
    op.create_index("ix_retrieval_traces_user_id", "retrieval_traces", ["user_id"])
    op.create_index("ix_retrieval_traces_session_id", "retrieval_traces", ["session_id"])
    op.create_index("ix_retrieval_traces_chat_message_id", "retrieval_traces", ["chat_message_id"])

    op.add_column(
        "chat_messages",
        sa.Column(
            "retrieval_trace_id",
            sa.dialects.postgresql.UUID(as_uuid=True),
            sa.ForeignKey("retrieval_traces.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    op.create_index("ix_chat_messages_retrieval_trace_id", "chat_messages", ["retrieval_trace_id"])

    op.create_table(
        "artifact_drafts",
        sa.Column("id", sa.dialects.postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", sa.dialects.postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("artifact_type", sa.String(length=32), nullable=False),
        sa.Column("source_id", sa.dialects.postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("title", sa.String(length=500), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("status", sa.String(length=24), nullable=False, server_default="draft"),
        sa.Column("draft_payload", sa.JSON(), nullable=False, server_default=sa.text("'{}'::json")),
        sa.Column("published_payload", sa.JSON(), nullable=True),
        sa.Column("approvals", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_artifact_drafts_user_id", "artifact_drafts", ["user_id"])
    op.create_index("ix_artifact_drafts_artifact_type", "artifact_drafts", ["artifact_type"])
    op.create_index("ix_artifact_drafts_source_id", "artifact_drafts", ["source_id"])
    op.create_index("ix_artifact_drafts_user_status", "artifact_drafts", ["user_id", "status"])
    op.create_index("ix_artifact_drafts_user_created", "artifact_drafts", ["user_id", "created_at"])


def downgrade() -> None:
    op.drop_index("ix_artifact_drafts_user_created", table_name="artifact_drafts")
    op.drop_index("ix_artifact_drafts_user_status", table_name="artifact_drafts")
    op.drop_index("ix_artifact_drafts_source_id", table_name="artifact_drafts")
    op.drop_index("ix_artifact_drafts_artifact_type", table_name="artifact_drafts")
    op.drop_index("ix_artifact_drafts_user_id", table_name="artifact_drafts")
    op.drop_table("artifact_drafts")

    op.drop_index("ix_chat_messages_retrieval_trace_id", table_name="chat_messages")
    op.drop_column("chat_messages", "retrieval_trace_id")

    op.drop_index("ix_retrieval_traces_chat_message_id", table_name="retrieval_traces")
    op.drop_index("ix_retrieval_traces_session_id", table_name="retrieval_traces")
    op.drop_index("ix_retrieval_traces_user_id", table_name="retrieval_traces")
    op.drop_index("ix_retrieval_traces_session_created", table_name="retrieval_traces")
    op.drop_index("ix_retrieval_traces_user_created", table_name="retrieval_traces")
    op.drop_table("retrieval_traces")

