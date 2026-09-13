"""Record what a reasoning model thought, not only what it said.

DeepSeek and its kind return the chain of thought in a separate field and
charge it against max_tokens. The client read only the answer, so the tokens
were paid for and discarded: an agent's decision could be replayed but the
reasoning behind it could not, and a call that spent its whole budget thinking
looked simply empty.

Revision ID: 0091_add_llm_snapshot_reasoning
Revises: 0090_add_agent_retractions
"""

import sqlalchemy as sa
from alembic import op

revision = "0091_add_llm_snapshot_reasoning"
down_revision = "0090_add_agent_retractions"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "llm_call_snapshots", sa.Column("reasoning_text", sa.Text(), nullable=True)
    )
    op.add_column(
        "llm_call_snapshots", sa.Column("reasoning_tokens", sa.Integer(), nullable=True)
    )


def downgrade() -> None:
    op.drop_column("llm_call_snapshots", "reasoning_tokens")
    op.drop_column("llm_call_snapshots", "reasoning_text")
