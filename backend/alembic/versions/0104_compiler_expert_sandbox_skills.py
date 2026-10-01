"""Let the compiler optimization expert use sandbox skills in chat.

A chat message is routed to a specialist by capability, and a specialist may
call only the tools on its whitelist. Asked to "use the ir_opcodes sandbox
skill" on a C function, the router chose this agent -- rightly, it is the one
for compiler questions -- and the planner then had no skill tool to offer it.
It planned nothing, and the answer described what the skill *would* have
printed.

The whitelist is doing its job; it was written before the tools existed. This
adds them to the one specialist whose subject they are. The other specialists
are left alone on purpose: a whitelist is a decision about what an agent is
for, and a document or report agent running sandbox code is not obviously
wanted.

Revision ID: 0104_compiler_expert_sandbox_skills
Revises: 0103_sandbox_skills
"""

import json

import sqlalchemy as sa

from alembic import op

revision = "0104_compiler_expert_sandbox_skills"
down_revision = "0103_sandbox_skills"
branch_labels = None
depends_on = None

AGENT = "compiler_optimization_expert"
SKILL_TOOLS = [
    "list_sandbox_skills",
    "load_sandbox_skill",
    "run_sandbox_skill",
    "propose_sandbox_skill",
]


def _whitelist(connection):
    row = connection.execute(
        sa.text("SELECT tool_whitelist FROM agent_definitions WHERE name = :name"),
        {"name": AGENT},
    ).first()
    if row is None:
        return None
    current = row[0]
    if isinstance(current, str):
        current = json.loads(current)
    # A null whitelist means every tool, which already includes these.
    return current if isinstance(current, list) else None


def _store(connection, tools) -> None:
    connection.execute(
        sa.text(
            "UPDATE agent_definitions SET tool_whitelist = CAST(:tools AS JSON) "
            "WHERE name = :name"
        ),
        {"tools": json.dumps(tools), "name": AGENT},
    )


def upgrade() -> None:
    connection = op.get_bind()
    current = _whitelist(connection)
    if current is None:
        return
    added = [tool for tool in SKILL_TOOLS if tool not in current]
    if added:
        _store(connection, current + added)


def downgrade() -> None:
    connection = op.get_bind()
    current = _whitelist(connection)
    if current is None:
        return
    _store(connection, [tool for tool in current if tool not in SKILL_TOOLS])
