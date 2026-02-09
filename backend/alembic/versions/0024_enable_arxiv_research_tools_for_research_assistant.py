"""Enable arXiv research tools for Research Assistant.

Revision ID: 0024_enable_arxiv_tools
Revises: 0023_add_presentation_started_at
Create Date: 2026-01-24
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "0024_enable_arxiv_tools"
down_revision = "0023_add_presentation_started_at"
branch_labels = None
depends_on = None


def upgrade() -> None:
    agent_definitions = sa.table(
        "agent_definitions",
        sa.column("name", sa.String),
        sa.column("capabilities", sa.JSON),
        sa.column("tool_whitelist", sa.JSON),
        sa.column("updated_at", sa.DateTime(timezone=True)),
    )

    bind = op.get_bind()
    row = bind.execute(
        sa.select(agent_definitions.c.capabilities, agent_definitions.c.tool_whitelist).where(
            agent_definitions.c.name == sa.literal("research_assistant")
        )
    ).first()
    if not row:
        return

    capabilities = list(row[0] or [])
    for cap in ["paper_search"]:
        if cap not in capabilities:
            capabilities.append(cap)

    tool_whitelist = row[1]
    if tool_whitelist is None:
        # null means "all tools"; leave it as-is.
        return

    tools = list(tool_whitelist or [])
    for t in ["search_arxiv", "ingest_arxiv_papers"]:
        if t not in tools:
            tools.append(t)

    bind.execute(
        agent_definitions.update()
        .where(agent_definitions.c.name == sa.literal("research_assistant"))
        .values(
            capabilities=capabilities,
            tool_whitelist=tools,
        )
    )


def downgrade() -> None:
    agent_definitions = sa.table(
        "agent_definitions",
        sa.column("name", sa.String),
        sa.column("capabilities", sa.JSON),
        sa.column("tool_whitelist", sa.JSON),
    )

    bind = op.get_bind()
    row = bind.execute(
        sa.select(agent_definitions.c.capabilities, agent_definitions.c.tool_whitelist).where(
            agent_definitions.c.name == sa.literal("research_assistant")
        )
    ).first()
    if not row:
        return

    capabilities = [c for c in (row[0] or []) if c != "paper_search"]
    tool_whitelist = row[1]
    if tool_whitelist is None:
        return
    tools = [t for t in (tool_whitelist or []) if t not in {"search_arxiv", "ingest_arxiv_papers"}]

    bind.execute(
        agent_definitions.update()
        .where(agent_definitions.c.name == sa.literal("research_assistant"))
        .values(
            capabilities=capabilities,
            tool_whitelist=tools,
        )
    )
