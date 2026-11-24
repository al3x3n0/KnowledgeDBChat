"""Store prompts on custom MCP profiles."""

import sqlalchemy as sa

from alembic import op

revision = "0113_mcp_prompts"
down_revision = "0112_mcp_programs"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("api_keys", sa.Column("mcp_prompts", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("api_keys", "mcp_prompts")
