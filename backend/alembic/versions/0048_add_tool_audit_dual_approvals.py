"""Add dual-approval fields to tool_execution_audits.

Revision ID: 0048_add_tool_audit_dual_approvals
Revises: 0047_add_tool_audit_policy_decision
Create Date: 2026-02-05 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0048_add_tool_audit_dual_approvals"
down_revision = "0047_add_tool_audit_policy_decision"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("tool_execution_audits", sa.Column("approval_mode", sa.String(length=32), nullable=True, server_default="owner_and_admin"))
    op.add_column("tool_execution_audits", sa.Column("owner_approved_by", postgresql.UUID(as_uuid=True), nullable=True))
    op.add_column("tool_execution_audits", sa.Column("owner_approved_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("tool_execution_audits", sa.Column("admin_approved_by", postgresql.UUID(as_uuid=True), nullable=True))
    op.add_column("tool_execution_audits", sa.Column("admin_approved_at", sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    op.drop_column("tool_execution_audits", "admin_approved_at")
    op.drop_column("tool_execution_audits", "admin_approved_by")
    op.drop_column("tool_execution_audits", "owner_approved_at")
    op.drop_column("tool_execution_audits", "owner_approved_by")
    op.drop_column("tool_execution_audits", "approval_mode")

