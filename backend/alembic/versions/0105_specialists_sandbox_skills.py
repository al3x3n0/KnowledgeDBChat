"""Let every built-in specialist use sandbox skills.

Migration 0104 gave the skill tools to the compiler optimization expert, the
one specialist a live request had been routed to and found without them. That
left a skill usable or not depending on which agent the router happened to
choose: the same sentence worked when it reached the generalist and silently
did nothing when it reached, say, the code expert.

A skill is a platform capability rather than one agent's speciality, so it is
offered wherever chat is. This adds the four tools to every *system* agent
that has a whitelist. Agents a user authored are deliberately not touched: a
whitelist someone wrote is a decision they made, and widening it for them is
not this migration's to do.

Revision ID: 0105_specialists_sandbox_skills
Revises: 0104_compiler_expert_sandbox_skills
"""

import json

import sqlalchemy as sa

from alembic import op

revision = "0105_specialists_sandbox_skills"
down_revision = "0104_compiler_expert_sandbox_skills"
branch_labels = None
depends_on = None

SKILL_TOOLS = [
    "list_sandbox_skills",
    "load_sandbox_skill",
    "run_sandbox_skill",
    "propose_sandbox_skill",
]

#: Already handled by 0104, and left to that migration's downgrade.
ALREADY_GRANTED = "compiler_optimization_expert"


def _system_whitelists(connection):
    rows = connection.execute(
        sa.text(
            "SELECT name, tool_whitelist FROM agent_definitions "
            "WHERE is_system = true AND name <> :skip"
        ),
        {"skip": ALREADY_GRANTED},
    ).fetchall()
    for name, current in rows:
        if isinstance(current, str):
            current = json.loads(current)
        # Null (SQL or JSON) means every tool, which already includes these.
        if isinstance(current, list):
            yield name, current


def _store(connection, name, tools) -> None:
    connection.execute(
        sa.text(
            "UPDATE agent_definitions SET tool_whitelist = CAST(:tools AS JSON) "
            "WHERE name = :name"
        ),
        {"tools": json.dumps(tools), "name": name},
    )


def upgrade() -> None:
    connection = op.get_bind()
    for name, current in list(_system_whitelists(connection)):
        added = [tool for tool in SKILL_TOOLS if tool not in current]
        if added:
            _store(connection, name, current + added)


def downgrade() -> None:
    connection = op.get_bind()
    for name, current in list(_system_whitelists(connection)):
        kept = [tool for tool in current if tool not in SKILL_TOOLS]
        if kept != current:
            _store(connection, name, kept)
