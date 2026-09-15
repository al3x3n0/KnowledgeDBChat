"""Say where a workflow came from, when a plugin put it there.

Plugins can now ship workflows. A workflow created that way is a *copy* -- rows
in `workflows`, `workflow_nodes` and `workflow_edges` owned by the installing
user, editable like any other, and deliberately not overwritten when the plugin
is updated, because somebody's edits are worth more than a version bump.

That copy needs to say where it came from. A flow that appeared without you
building it, with no way to tell which plugin put it there, is
indistinguishable from one you forgot writing.

A slug rather than a foreign key, on purpose: the workflow outlives
uninstalling the plugin and must not be removed by a cascade when somebody
deletes the bundle that seeded it.

Revision ID: 0100_workflow_plugin_origin
Revises: 0099_user_ui_preferences
"""

import sqlalchemy as sa

from alembic import op

revision = "0100_workflow_plugin_origin"
down_revision = "0099_user_ui_preferences"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "workflows",
        sa.Column("origin_plugin_slug", sa.String(length=32), nullable=True),
    )
    # Identity has to be something the owner cannot change. Matching a shipped
    # workflow by name means renaming it makes the next install create a second
    # copy.
    op.add_column(
        "workflows", sa.Column("origin_flow_id", sa.String(length=60), nullable=True)
    )
    op.create_index(
        "ix_workflows_origin_plugin_slug", "workflows", ["origin_plugin_slug"]
    )


def downgrade() -> None:
    op.drop_index("ix_workflows_origin_plugin_slug", table_name="workflows")
    op.drop_column("workflows", "origin_flow_id")
    op.drop_column("workflows", "origin_plugin_slug")
