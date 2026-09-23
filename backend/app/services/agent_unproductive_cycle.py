"""A run going in circles through calls that all succeed.

`agent_repeated_success` catches the SAME call repeated. It cannot catch a
cycle of *different* calls, and a run alternating two tools defeats it by
construction: measured on a live job, three `write_progress_report` calls
carried three different report texts, so three different signatures, and were
never flagged -- while the `recall_prior_findings` calls between them, which
were identical, were flagged and ignored.

What the two have in common is not repetition. It is that neither tool can
produce evidence, and the tool specs already say so: `produces=()` against
`simulate_c_workload`'s `('simulated_measurement',)`. That distinction is what
makes this safe to act on. Four simulations in a row produce no findings
either and are exactly what a careful run should be doing; four progress
reports are a run narrating instead of working.

Deliberately not keyed on findings alone. A tool that declares evidence and
returns none has measured something and come up empty, which is a result.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional, Sequence

logger = logging.getLogger(__name__)

#: Successful non-producing calls in a row before the run is told.
NOTE_AT = 4

#: ...and before it is told in the imperative. Two more calls is the cheapest
#: window that distinguishes a pause for orientation from a loop.
DIRECTIVE_AT = 6

#: Tools that satisfy a contract requirement without emitting a finding, so a
#: call to one is progress even though it produces no evidence. Derived from
#: the result keys a run can actually write.
CONTRACT_SATISFYING_TOOLS = frozenset(
    {"set_output_schema", "format_as_table", "format_as_report"}
)


def _succeeded(result: Any) -> bool:
    if not isinstance(result, Mapping):
        return False
    if result.get("error"):
        return False
    return bool(result.get("success", True))


def _identities(result: Any) -> set:
    """What claims this call put on the table, new or not."""
    if not isinstance(result, Mapping):
        return set()
    from app.services.agent_loop_policy import finding_identity

    findings = result.get("findings")
    if not isinstance(findings, list):
        return set()
    return {finding_identity(f) for f in findings if isinstance(f, dict)}


def can_produce_evidence(tool: str) -> bool:
    """Whether the tool's own spec says it can produce evidence.

    Unknown tools are treated as productive: a plugin or a tool this catalog
    does not carry should never be accused of padding a loop.
    """
    name = str(tool or "").strip()
    if not name:
        return True
    if name in CONTRACT_SATISFYING_TOOLS:
        return True
    try:
        from app.agent_core import tool_specs

        spec = tool_specs.STATIC_CATALOG.spec_for(name)
    except Exception:  # pragma: no cover - defensive
        return True
    if spec is None:
        return True
    return bool(getattr(spec, "produces", ()) or ())


def _tool_of(entry: Mapping[str, Any]) -> str:
    action = entry.get("action") if isinstance(entry.get("action"), Mapping) else {}
    return str(action.get("tool") or entry.get("tool") or "").strip()


def streak(state: Optional[Mapping[str, Any]]) -> List[str]:
    """The trailing run of successful calls that could not have advanced anything.

    Ends at the first call that failed, emitted a finding, or belongs to a tool
    that declares evidence -- any of which means the run was doing something.
    """
    if not isinstance(state, Mapping):
        return []
    taken = state.get("actions_taken")
    if not isinstance(taken, list):
        return []
    # Walk forward so "new" means new at the time the call was made. Recalling
    # ten findings the run already had is the case this exists for: the raw
    # count grows, the claims do not, and a detector keyed on "emitted any
    # finding" reads the repeat as work. Measured: 101 findings, 24 distinct.
    seen: set = set()
    productive_at: List[bool] = []
    tools: List[str] = []
    for entry in taken:
        if not isinstance(entry, Mapping):
            productive_at.append(True)
            tools.append("")
            continue
        result = entry.get("result")
        tool = _tool_of(entry)
        added = _identities(result) - seen
        seen |= _identities(result)
        productive_at.append(
            (not _succeeded(result))
            or bool(added)
            or (not tool)
            or can_produce_evidence(tool)
        )
        tools.append(tool)

    out: List[str] = []
    for tool, productive in zip(reversed(tools), reversed(productive_at)):
        if productive:
            break
        out.append(tool)
    out.reverse()
    return out


def analyze(
    state: Optional[Mapping[str, Any]],
    *,
    missing: Sequence[str] = (),
) -> Optional[Dict[str, Any]]:
    """A note for a run that has stopped advancing without anything failing.

    `missing` is what the contract still wants, so the note can say what to do
    instead of only what to stop doing -- a run told it is looping and not told
    the way out has been given the same information twice.
    """
    tools = streak(state)
    if len(tools) < NOTE_AT:
        return None

    unique = sorted(set(tools))
    note = (
        f"The last {len(tools)} calls all succeeded and none of them could "
        f"have advanced this run: {', '.join(unique)} "
        + ("declare" if len(unique) > 1 else "declares")
        + " no evidence, so no number of them will satisfy a contract or "
        "answer the goal."
    )
    wanted = [str(m).strip() for m in (missing or []) if str(m).strip()]
    if wanted:
        note += " Outstanding: " + ", ".join(wanted[:6]) + "."

    if len(tools) >= DIRECTIVE_AT:
        note += (
            " Stop reporting and reading. The next call must be one that "
            "produces what is outstanding, or the run is spending its "
            "remaining iterations the way it spent these."
        )
    else:
        note += (
            " Use what is already in this run's history and make the next "
            "call one that produces something."
        )
    return {"streak": len(tools), "tools": unique, "note": note}
