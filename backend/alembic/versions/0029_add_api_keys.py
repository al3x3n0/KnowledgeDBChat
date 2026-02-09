"""Add API keys tables for external tool authentication.

Revision ID: 0029_add_api_keys
Revises: 0028_llm_usage
Create Date: 2026-01-27
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0029_add_api_keys"
down_revision = "0028_llm_usage"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create api_keys table
    op.create_table(
        "api_keys",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("key_prefix", sa.String(length=8), nullable=False, index=True),
        sa.Column("key_hash", sa.String(length=128), nullable=False),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("scopes", postgresql.JSON(), nullable=True),
        sa.Column("rate_limit_per_minute", sa.Integer(), default=60),
        sa.Column("rate_limit_per_day", sa.Integer(), default=10000),
        sa.Column("is_active", sa.Boolean(), default=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_used_ip", sa.String(length=45), nullable=True),
        sa.Column("usage_count", sa.Integer(), default=0),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
    )

    # Create index on user_id for faster lookups
    op.create_index(
        "ix_api_keys_user_id",
        "api_keys",
        ["user_id"],
    )

    # Create index on key_hash for validation lookups
    op.create_index(
        "ix_api_keys_key_hash",
        "api_keys",
        ["key_hash"],
        unique=True,
    )

    # Create api_key_usage_logs table
    op.create_table(
        "api_key_usage_logs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "api_key_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("api_keys.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("endpoint", sa.String(length=500), nullable=False),
        sa.Column("method", sa.String(length=10), nullable=False),
        sa.Column("status_code", sa.Integer(), nullable=True),
        sa.Column("ip_address", sa.String(length=45), nullable=True),
        sa.Column("user_agent", sa.String(length=500), nullable=True),
        sa.Column(
            "timestamp",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            index=True,
        ),
        sa.Column("response_time_ms", sa.Integer(), nullable=True),
    )

    # Create index for usage log queries
    op.create_index(
        "ix_api_key_usage_logs_api_key_id",
        "api_key_usage_logs",
        ["api_key_id"],
    )


def downgrade() -> None:
    op.drop_table("api_key_usage_logs")
    op.drop_table("api_keys")
