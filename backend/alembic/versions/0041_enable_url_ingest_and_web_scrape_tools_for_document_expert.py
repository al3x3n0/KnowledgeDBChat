"""Enable web_scrape and ingest_url tools for Document Expert agent.

Revision ID: 0041_enable_url_ingest_and_web_scrape_tools_for_document_expert
Revises: 0040_ai_hub_feedback_profile_id
Create Date: 2026-01-30 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = "0041_enable_url_ingest_and_web_scrape_tools_for_document_expert"
down_revision = "0040_ai_hub_feedback_profile_id"
branch_labels = None
depends_on = None


TOOLS_TO_ADD = ["web_scrape", "ingest_url"]


def upgrade() -> None:
    conn = op.get_bind()

    agent_definitions = sa.table(
        "agent_definitions",
        sa.column("name", sa.String),
        sa.column("tool_whitelist", postgresql.JSON),
    )

    row = conn.execute(
        sa.select(agent_definitions.c.tool_whitelist).where(agent_definitions.c.name == "document_expert")
    ).fetchone()

    if not row:
        return

    tool_whitelist = row[0]
    if tool_whitelist is None:
        # None means all tools already allowed.
        return

    tools = list(tool_whitelist or [])
    changed = False
    for t in TOOLS_TO_ADD:
        if t not in tools:
            tools.append(t)
            changed = True

    if changed:
        conn.execute(
            agent_definitions.update()
            .where(agent_definitions.c.name == "document_expert")
            .values(tool_whitelist=tools)
        )


def downgrade() -> None:
    conn = op.get_bind()

    agent_definitions = sa.table(
        "agent_definitions",
        sa.column("name", sa.String),
        sa.column("tool_whitelist", postgresql.JSON),
    )

    row = conn.execute(
        sa.select(agent_definitions.c.tool_whitelist).where(agent_definitions.c.name == "document_expert")
    ).fetchone()

    if not row:
        return

    tool_whitelist = row[0]
    if tool_whitelist is None:
        return

    tools = [t for t in (tool_whitelist or []) if t not in set(TOOLS_TO_ADD)]
    conn.execute(
        agent_definitions.update()
        .where(agent_definitions.c.name == "document_expert")
        .values(tool_whitelist=tools)
    )

