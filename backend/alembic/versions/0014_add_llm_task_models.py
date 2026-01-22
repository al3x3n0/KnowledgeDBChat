"""Add per-task LLM model configuration.

Revision ID: 0014_add_llm_task_models
Revises: 0013_add_user_llm_preferences
Create Date: 2026-01-13 13:00:00.000000
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSON


# revision identifiers, used by Alembic.
revision = "0014_add_llm_task_models"
down_revision = "0013_add_user_llm_preferences"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add JSON column for per-task model overrides
    op.add_column(
        "user_preferences",
        sa.Column("llm_task_models", JSON, nullable=True),
    )


def downgrade() -> None:
    op.drop_column("user_preferences", "llm_task_models")
