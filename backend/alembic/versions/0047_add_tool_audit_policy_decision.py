"""Add policy_decision to tool_execution_audits.

Revision ID: 0047_add_tool_audit_policy_decision
Revises: 0046_add_tool_policies
Create Date: 2026-02-05 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0047_add_tool_audit_policy_decision"
down_revision = "0046_add_tool_policies"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("tool_execution_audits", sa.Column("policy_decision", postgresql.JSONB(astext_type=sa.Text()), nullable=True))


def downgrade() -> None:
    op.drop_column("tool_execution_audits", "policy_decision")

