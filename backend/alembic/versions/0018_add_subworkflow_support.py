"""Add parent_execution_id for sub-workflow tracking.

Revision ID: 0018_add_subworkflow_support
Revises: 0017_add_presentation_jobs
Create Date: 2025-01-20 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = "0018_add_subworkflow_support"
down_revision = "0017_add_presentation_jobs"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add parent_execution_id column to workflow_executions for sub-workflow tracking
    op.add_column(
        "workflow_executions",
        sa.Column(
            "parent_execution_id",
            postgresql.UUID(as_uuid=True),
            nullable=True
        )
    )

    # Add foreign key constraint (self-referential)
    op.create_foreign_key(
        "fk_workflow_execution_parent",
        "workflow_executions",
        "workflow_executions",
        ["parent_execution_id"],
        ["id"],
        ondelete="SET NULL"
    )

    # Add index for faster parent/child lookups
    op.create_index(
        "ix_workflow_executions_parent_execution_id",
        "workflow_executions",
        ["parent_execution_id"]
    )


def downgrade() -> None:
    # Drop index first
    op.drop_index(
        "ix_workflow_executions_parent_execution_id",
        table_name="workflow_executions"
    )

    # Drop foreign key constraint
    op.drop_constraint(
        "fk_workflow_execution_parent",
        "workflow_executions",
        type_="foreignkey"
    )

    # Drop column
    op.drop_column("workflow_executions", "parent_execution_id")
