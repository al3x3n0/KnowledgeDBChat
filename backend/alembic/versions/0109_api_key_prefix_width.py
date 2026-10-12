"""Widen api_keys.key_prefix to hold the prefix the service writes.

The column was VARCHAR(8). `APIKeyService.generate_key` stores the first 12
characters of the key ("taic_" and seven more), which Postgres refuses with
"value too long for type character varying(8)", so creating an API key
failed on every real deployment. SQLite does not enforce the width, which is
why no test saw it.

Revision ID: 0109_api_key_prefix_width
Revises: 0108_job_operator_flags
"""

import sqlalchemy as sa

from alembic import op

revision = "0109_api_key_prefix_width"
down_revision = "0108_job_operator_flags"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("api_keys") as batch:
        batch.alter_column(
            "key_prefix",
            existing_type=sa.String(length=8),
            type_=sa.String(length=16),
            existing_nullable=False,
        )


def downgrade() -> None:
    # Prefixes written since are 12 characters; cut them to fit.
    op.execute("UPDATE api_keys SET key_prefix = substr(key_prefix, 1, 8)")
    with op.batch_alter_table("api_keys") as batch:
        batch.alter_column(
            "key_prefix",
            existing_type=sa.String(length=16),
            type_=sa.String(length=8),
            existing_nullable=False,
        )
