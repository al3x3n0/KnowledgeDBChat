"""Add MCP configuration tables.

Revision ID: 0033_add_mcp_configuration
Revises: 0032_add_repo_report_jobs
Create Date: 2026-01-28
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0033_add_mcp_configuration"
down_revision = "0032_add_repo_report_jobs"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create mcp_tool_configs table - configures which tools are available per API key
    op.create_table(
        "mcp_tool_configs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "api_key_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("api_keys.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("tool_name", sa.String(50), nullable=False),
        sa.Column("is_enabled", sa.Boolean(), default=True),
        sa.Column("config", postgresql.JSON(), nullable=True),  # Tool-specific config
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            onupdate=sa.func.now(),
        ),
        sa.UniqueConstraint("api_key_id", "tool_name", name="uq_mcp_tool_config"),
    )

    # Create index on api_key_id
    op.create_index(
        "ix_mcp_tool_configs_api_key_id",
        "mcp_tool_configs",
        ["api_key_id"],
    )

    # Create mcp_source_access table - configures which sources an API key can access
    op.create_table(
        "mcp_source_access",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "api_key_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("api_keys.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "source_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("document_sources.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("can_read", sa.Boolean(), default=True),
        sa.Column("can_search", sa.Boolean(), default=True),
        sa.Column("can_chat", sa.Boolean(), default=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
        sa.UniqueConstraint("api_key_id", "source_id", name="uq_mcp_source_access"),
    )

    # Create indexes
    op.create_index(
        "ix_mcp_source_access_api_key_id",
        "mcp_source_access",
        ["api_key_id"],
    )

    # Add MCP-specific columns to api_keys table
    op.add_column(
        "api_keys",
        sa.Column("mcp_enabled", sa.Boolean(), default=True, server_default="true"),
    )
    op.add_column(
        "api_keys",
        sa.Column("allowed_tools", postgresql.JSON(), nullable=True),  # List of allowed tool names, null = all
    )
    op.add_column(
        "api_keys",
        sa.Column("source_access_mode", sa.String(20), default="all", server_default="all"),
        # "all" = access all sources, "restricted" = only mcp_source_access entries
    )


def downgrade() -> None:
    op.drop_column("api_keys", "source_access_mode")
    op.drop_column("api_keys", "allowed_tools")
    op.drop_column("api_keys", "mcp_enabled")
    op.drop_table("mcp_source_access")
    op.drop_table("mcp_tool_configs")
