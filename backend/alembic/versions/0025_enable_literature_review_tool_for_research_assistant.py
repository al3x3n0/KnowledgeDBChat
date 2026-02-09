"""Enable literature_review_arxiv tool for Research Assistant.

Revision ID: 0025_enable_lit_review
Revises: 0024_enable_arxiv_tools
Create Date: 2026-01-24
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0025_enable_lit_review"
down_revision = "0024_enable_arxiv_tools"
branch_labels = None
depends_on = None


def upgrade() -> None:
    agent_definitions = sa.table(
        "agent_definitions",
        sa.column("name", sa.String),
        sa.column("tool_whitelist", sa.JSON),
    )

    bind = op.get_bind()
    row = bind.execute(
        sa.select(agent_definitions.c.tool_whitelist).where(
            agent_definitions.c.name == sa.literal("research_assistant")
        )
    ).first()
    if not row:
        return

    tool_whitelist = row[0]
    if tool_whitelist is None:
        return

    tools = list(tool_whitelist or [])
    if "literature_review_arxiv" not in tools:
        tools.append("literature_review_arxiv")

    bind.execute(
        agent_definitions.update()
        .where(agent_definitions.c.name == sa.literal("research_assistant"))
        .values(tool_whitelist=tools)
    )


def downgrade() -> None:
    agent_definitions = sa.table(
        "agent_definitions",
        sa.column("name", sa.String),
        sa.column("tool_whitelist", sa.JSON),
    )

    bind = op.get_bind()
    row = bind.execute(
        sa.select(agent_definitions.c.tool_whitelist).where(
            agent_definitions.c.name == sa.literal("research_assistant")
        )
    ).first()
    if not row:
        return

    tool_whitelist = row[0]
    if tool_whitelist is None:
        return

    tools = [t for t in (tool_whitelist or []) if t != "literature_review_arxiv"]
    bind.execute(
        agent_definitions.update()
        .where(agent_definitions.c.name == sa.literal("research_assistant"))
        .values(tool_whitelist=tools)
    )
