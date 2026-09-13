"""Record whether the prompt cache actually hit.

The thinking prompt is deliberately split into a byte-stable prefix and a
volatile tail so the prefix can be cached, and Anthropic requests carry
cache_control breakpoints for the same reason. Nothing ever read back whether
any of it worked, so the whole arrangement rested on an assumption that could
not be checked. Providers report it on every call and we were discarding it.

Both columns are nullable, and that distinction is the point: NULL means the
provider said nothing about caching, 0 means it said the cache missed
completely. Averaging silence as zero would report a healthy cache as broken.

Revision ID: 0095_llm_cache_tokens
Revises: 0094_agent_pipelines
"""

import sqlalchemy as sa
from alembic import op

revision = "0095_llm_cache_tokens"
down_revision = "0094_agent_pipelines"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "llm_call_snapshots",
        sa.Column("cache_hit_tokens", sa.Integer(), nullable=True),
    )
    op.add_column(
        "llm_call_snapshots",
        sa.Column("cache_miss_tokens", sa.Integer(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("llm_call_snapshots", "cache_miss_tokens")
    op.drop_column("llm_call_snapshots", "cache_hit_tokens")
