"""A person's saved LLM providers.

One API URL and one key per person meant every task went to the same
endpoint, and a per-task provider override could not reach a second one.
``llm_providers`` holds named connections (a local vLLM, a gateway), each
referred to as ``custom:<id>`` by ``llm_provider`` and
``llm_task_providers``; keys are stored encrypted. NULL means none saved.

Revision ID: 0111_user_llm_providers
Revises: 0110_campaign_plan_approval
"""

import sqlalchemy as sa

from alembic import op

revision = "0111_user_llm_providers"
down_revision = "0110_campaign_plan_approval"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "user_preferences", sa.Column("llm_providers", sa.JSON(), nullable=True)
    )


def downgrade() -> None:
    op.drop_column("user_preferences", "llm_providers")
