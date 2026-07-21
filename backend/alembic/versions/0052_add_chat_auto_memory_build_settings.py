"""Add configurable chat auto-memory build settings.

Revision ID: 0052_add_chat_auto_memory_build_settings
Revises: 0051_add_agent_tool_priors
Create Date: 2026-02-11 12:40:00.000000
"""

from alembic import op
import sqlalchemy as sa


revision = "0052_add_chat_auto_memory_build_settings"
down_revision = "0051_add_agent_tool_priors"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "user_preferences",
        sa.Column("auto_memory_build_enabled", sa.Boolean(), nullable=False, server_default=sa.text("true")),
    )
    op.add_column(
        "user_preferences",
        sa.Column("auto_memory_build_mode", sa.String(length=32), nullable=False, server_default=sa.text("'per_turn'")),
    )
    op.add_column(
        "user_preferences",
        sa.Column("auto_memory_build_min_messages", sa.Integer(), nullable=False, server_default=sa.text("3")),
    )
    op.add_column(
        "user_preferences",
        sa.Column("auto_memory_build_min_minutes", sa.Integer(), nullable=False, server_default=sa.text("10")),
    )


def downgrade() -> None:
    op.drop_column("user_preferences", "auto_memory_build_min_minutes")
    op.drop_column("user_preferences", "auto_memory_build_min_messages")
    op.drop_column("user_preferences", "auto_memory_build_mode")
    op.drop_column("user_preferences", "auto_memory_build_enabled")
