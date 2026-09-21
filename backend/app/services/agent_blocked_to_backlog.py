"""File the platform gap that stopped a run, as coding work.

A run that blocks on its own bad input is the development loop working: write
code, read the diagnostic, fix line 12. A run that blocks because a *tool* is
missing something is a different thing entirely -- nothing the run can write
will help, and until someone edits the platform every later run meets the same
wall.

Those blockers arrive fully specified, because the tools say what they need:
``unknown mnemonic 'uaddw': add it to operand_arity, since guessing its operand
count emits assembly the assembler will reject``. That is a coding task with
its symptom, its evidence and its fix location already written down, and it sat
in a paused job waiting for a person to read the logs. Measured across one
session: five operator interventions, two of which were exactly this and
required editing a Rust table.

The discriminator is ``agent_failure_diagnosis.blames_the_submitted_code``,
which the escalation machinery already uses for the same distinction and whose
docstring states it plainly: a compiler pointing at a line in the submitted
source is the run's problem; ``clang: not found`` is the platform's.

What this does **not** do is fix anything. It files a draft item with
``auto_apply_enabled`` off, because a change to the platform that agents run on
deserves a person's eye before it lands -- and because an item that files
itself and then applies itself is two decisions taken by something that has
only earned one.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional

from sqlalchemy import select

logger = logging.getLogger(__name__)

#: A failure has to repeat before it is worth filing. Once may be a flake, a
#: busy host, a half-written argument the next iteration fixes; twice with the
#: same class of error is the run meeting a wall rather than tripping.
MIN_ATTEMPTS = 2

#: One item per (tool, error class) while an earlier one is still open. The
#: same missing mnemonic met by four runs is one piece of work, and four
#: identical items is a backlog nobody reads.
OPEN_STATUSES = ("draft", "ready", "in_progress", "proposed")


def _actions(state: Mapping[str, Any]) -> List[Dict[str, Any]]:
    taken = state.get("actions_taken") if isinstance(state, Mapping) else None
    return [a for a in (taken or []) if isinstance(a, Mapping)]


def blocker(state: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """The repeated, not-self-inflicted failure that stopped this run.

    Returns the tool, the error as the tool phrased it, its class and how many
    times it was met -- or ``None`` when the run's failures were its own code,
    happened once, or never happened at all.
    """
    from app.services import agent_failure_diagnosis as diagnosis

    counts: Dict[str, Dict[str, Any]] = {}
    for entry in _actions(state):
        action = entry.get("action") if isinstance(entry.get("action"), Mapping) else {}
        result = entry.get("result") if isinstance(entry.get("result"), Mapping) else {}
        tool = str(action.get("tool") or entry.get("tool") or "").strip()
        error = result.get("error")
        if not tool or not error:
            continue
        if result.get("success"):
            continue
        # The run's own code being wrong is the development loop, not a gap in
        # the platform. Filing those would bury the real ones.
        if diagnosis.blames_the_submitted_code(error):
            continue
        key = f"{tool}:{diagnosis.classify_error(error)}"
        row = counts.setdefault(
            key,
            {
                "tool": tool,
                "error": str(error)[:2000],
                "error_class": diagnosis.classify_error(error),
                "attempts": 0,
            },
        )
        row["attempts"] += 1
        # Keep the latest phrasing: a tool that says more the second time --
        # after a stale image is replaced, say -- is worth filing on.
        row["error"] = str(error)[:2000]

    worst = max(counts.values(), key=lambda r: r["attempts"], default=None)
    if not worst or worst["attempts"] < MIN_ATTEMPTS:
        return None
    return worst


async def file_blocker(job: Any, state: Mapping[str, Any], db: Any) -> Optional[Any]:
    """Open a coding backlog item for the blocker, unless one is already open.

    Returns the item, or ``None`` when there was nothing to file.
    """
    from app.models.coding_backlog import CodingBacklogItem

    found = blocker(state)
    if not found:
        return None

    tool, error_class = found["tool"], found["error_class"]
    marker = f"[auto] {tool}: {error_class}"

    existing = await db.execute(
        select(CodingBacklogItem).where(
            CodingBacklogItem.user_id == job.user_id,
            CodingBacklogItem.title == marker,
            CodingBacklogItem.status.in_(OPEN_STATUSES),
        )
    )
    if existing.scalars().first() is not None:
        return None

    item = CodingBacklogItem(
        user_id=job.user_id,
        title=marker,
        portfolio_goal=(
            f"A run blocked because {tool} could not do what it was asked, "
            f"{found['attempts']} times. The tool's own message says what it "
            "needs. Make the tool able to do it, or make it say why it cannot."
        ),
        status="draft",
        priority=60,
        scope="platform",
        failure_symptom=(
            f"{tool} failed {found['attempts']} times with the same class of "
            f"error, and the run stopped without its contract met."
        ),
        error_output=found["error"],
        # Never applied unattended: this edits the platform the agents
        # themselves run on.
        auto_apply_enabled=False,
    )
    db.add(item)
    logger.info(
        "Filed coding backlog item for %s blocked on %s (%s)",
        job.id,
        tool,
        error_class,
    )

    # Optionally start work on it. The filing above is unconditional because a
    # recorded blocker costs nothing; starting a run is what costs, so it is
    # what the flag governs. Nothing lands unattended either way: the item
    # carries auto_apply_enabled=False, which the coding runner resolves to
    # proposal_only.
    from app.core.config import settings

    if getattr(settings, "AGENT_BLOCKER_AUTO_CODING_ENABLED", False):
        try:
            from app.api.endpoints.coding_backlog import _create_orchestrator_job

            await db.flush()
            await _create_orchestrator_job(item, db=db)
            logger.info("Started backlog orchestration for %s", item.id)
        except Exception as exc:  # pragma: no cover - never fatal to the run
            logger.warning(f"Could not start orchestration for {item.id}: {exc}")

    return item


__all__ = ["MIN_ATTEMPTS", "blocker", "file_blocker"]
