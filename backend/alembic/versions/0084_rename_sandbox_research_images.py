"""Rename the sandbox research images off the retired knowledgedb namespace

The system-managed sandbox profiles were seeded in 0062 with
ghcr.io/knowledgedb/* images. That namespace is a retired project name and no
such image was ever published, so move the rows to the current one. 0062 is
left as it was: it is the historical record of what that migration did.

Only system-managed rows are touched, and only when they still hold the exact
old value, so an operator who repointed a profile at their own image keeps it.

Revision ID: 0084_rename_sandbox_research_images
Revises: 0083_add_indexes_declared_only_by_models
Create Date: 2026-08-10

"""

from alembic import op

revision = "0084_rename_sandbox_research_images"
down_revision = "0083_add_indexes_declared_only_by_models"
branch_labels = None
depends_on = None


RENAMES = (
    (
        "ghcr.io/knowledgedb/compiler-research:latest",
        "ghcr.io/al3x3n0/kdbc-compiler-research:latest",
    ),
    (
        "ghcr.io/knowledgedb/microarch-research:latest",
        "ghcr.io/al3x3n0/kdbc-microarch-research:latest",
    ),
)


def _swap(pairs) -> None:
    for old, new in pairs:
        op.execute(
            """
            UPDATE scientific_sandbox_profiles
               SET docker_image = '{new}'
             WHERE docker_image = '{old}'
               AND system_managed IS TRUE
            """.format(
                new=new, old=old
            )
        )


def upgrade() -> None:
    _swap(RENAMES)


def downgrade() -> None:
    _swap([(new, old) for old, new in RENAMES])
