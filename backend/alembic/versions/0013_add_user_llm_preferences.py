"""Add user LLM preferences columns.

Revision ID: 0013_add_user_llm_preferences
Revises: 0012_add_template_jobs
Create Date: 2026-01-13 12:00:00.000000
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "0013_add_user_llm_preferences"
down_revision = "0012_add_template_jobs"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add LLM preference columns to user_preferences table
    op.add_column(
        "user_preferences",
        sa.Column("llm_provider", sa.String(length=50), nullable=True),
    )
    op.add_column(
        "user_preferences",
        sa.Column("llm_model", sa.String(length=100), nullable=True),
    )
    op.add_column(
        "user_preferences",
        sa.Column("llm_api_url", sa.String(length=500), nullable=True),
    )
    op.add_column(
        "user_preferences",
        sa.Column("llm_api_key", sa.String(length=500), nullable=True),
    )
    op.add_column(
        "user_preferences",
        sa.Column("llm_temperature", sa.Float(), nullable=True),
    )
    op.add_column(
        "user_preferences",
        sa.Column("llm_max_tokens", sa.Integer(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("user_preferences", "llm_max_tokens")
    op.drop_column("user_preferences", "llm_temperature")
    op.drop_column("user_preferences", "llm_api_key")
    op.drop_column("user_preferences", "llm_api_url")
    op.drop_column("user_preferences", "llm_model")
    op.drop_column("user_preferences", "llm_provider")
