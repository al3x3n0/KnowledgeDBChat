"""Add groundedness_score to chat_messages.

Revision ID: 0045_add_chat_groundedness_score
Revises: 0044_add_latex_compile_jobs
Create Date: 2026-02-05 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa


revision = "0045_add_chat_groundedness_score"
down_revision = "0044_add_latex_compile_jobs"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("chat_messages", sa.Column("groundedness_score", sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column("chat_messages", "groundedness_score")

