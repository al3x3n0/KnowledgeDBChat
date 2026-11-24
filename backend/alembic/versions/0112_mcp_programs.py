"""Bind MCP-only credentials to selected research programs."""

import sqlalchemy as sa

from alembic import op

revision = "0112_mcp_programs"
down_revision = "0111_user_llm_providers"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("api_keys", sa.Column("mcp_programs", sa.JSON(), nullable=True))


def downgrade() -> None:
    # Older code would accept these credentials at general API routes.
    op.execute("UPDATE api_keys SET is_active = false WHERE mcp_programs IS NOT NULL")
    op.drop_column("api_keys", "mcp_programs")
