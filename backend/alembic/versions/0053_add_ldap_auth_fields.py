"""Add LDAP auth fields to users.

Revision ID: 0053_add_ldap_auth_fields
Revises: 0052_add_chat_auto_memory_build_settings
Create Date: 2026-02-12 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa


revision = "0053_add_ldap_auth_fields"
down_revision = "0052_add_chat_auto_memory_build_settings"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    existing_cols = {c["name"] for c in insp.get_columns("users")}

    if "auth_provider" not in existing_cols:
        op.add_column(
            "users",
            sa.Column("auth_provider", sa.String(length=20), nullable=False, server_default=sa.text("'local'")),
        )
    if "auth_subject" not in existing_cols:
        op.add_column("users", sa.Column("auth_subject", sa.String(length=512), nullable=True))
    if "auth_metadata" not in existing_cols:
        op.add_column("users", sa.Column("auth_metadata", sa.JSON(), nullable=True))

    # Indexes (idempotent)
    existing_idx = {i["name"] for i in insp.get_indexes("users")}
    if "ix_users_auth_provider" not in existing_idx:
        op.create_index("ix_users_auth_provider", "users", ["auth_provider"])
    if "ix_users_auth_subject" not in existing_idx:
        op.create_index("ix_users_auth_subject", "users", ["auth_subject"])


def downgrade() -> None:
    op.drop_index("ix_users_auth_subject", table_name="users")
    op.drop_index("ix_users_auth_provider", table_name="users")
    op.drop_column("users", "auth_metadata")
    op.drop_column("users", "auth_subject")
    op.drop_column("users", "auth_provider")
