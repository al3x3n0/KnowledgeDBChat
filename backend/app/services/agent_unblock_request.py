"""What a blocked run actually needs, phrased as a question someone can answer.

A blocked run says "3 consecutive rounds produced no new findings (still 11);
stopping rather than spending the remaining iterations the same way" and names
the evidence it lacks. That is honest and it is not answerable: it describes
the shape of the stall, not the thing a person could supply to end it.

The run usually knows more than it says. Its history holds the tool that
failed, the words that tool used, and whether the tool ran at all -- which is
the difference between "this is broken, tell me if it is coming back" and "this
refused my call, tell me what it accepts". Those are different questions with
different answerers, and a queue that shows both as "resume?" wastes the
distinction.

Measured across one session: four runs blocked, and answering each took
reading the logs to work out what it wanted. Three of the four wanted something
stateable in a sentence -- an available tool, an accepted config shape, a
reachable upstream.

Deliberately derived rather than asked for. A run that could reliably phrase
its own need would not be stuck in the first place, and asking the model to
summarise its blocker is one more thing to get wrong on the way out.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

#: Kinds of need, ordered by how specifically they can be answered. The first
#: match wins, because a run whose tool never ran has nothing to say about what
#: that tool accepts.
TOOL_UNAVAILABLE = "tool_unavailable"
TOOL_REFUSES_INPUT = "tool_refuses_input"
EVIDENCE_UNREACHABLE = "evidence_unreachable"
UNKNOWN = "unknown"

MAX_QUOTE_CHARS = 300


def _failures(state: Mapping[str, Any]) -> List[Dict[str, Any]]:
    taken = state.get("actions_taken") if isinstance(state, Mapping) else None
    out: List[Dict[str, Any]] = []
    for entry in taken or []:
        if not isinstance(entry, Mapping):
            continue
        result = entry.get("result") if isinstance(entry.get("result"), Mapping) else {}
        error = result.get("error")
        if not error or result.get("success"):
            continue
        action = entry.get("action") if isinstance(entry.get("action"), Mapping) else {}
        tool = str(action.get("tool") or entry.get("tool") or "").strip()
        if tool:
            out.append({"tool": tool, "error": str(error)})
    return out


def _unreachable_evidence(missing: Sequence[str], job_type: str) -> Optional[str]:
    """A required finding type no tool this job type may call can produce."""
    from app.agent_core import tool_specs
    from app.services import agent_pipeline_vocabulary as vocabulary

    wanted = [
        str(m).split(":", 1)[1]
        for m in (missing or [])
        if str(m).startswith("finding_type:")
    ]
    if not wanted:
        return None
    by_name = {item.name: item for item in vocabulary.evidence_types()}
    runnable = set(tool_specs.STATIC_CATALOG.tools_for_job_type(job_type or "research"))
    for name in wanted:
        item = by_name.get(name)
        if item and item.producers and not any(p in runnable for p in item.producers):
            return name
    return None


def describe(
    state: Mapping[str, Any],
    *,
    missing: Sequence[str] = (),
    job_type: str = "research",
) -> Optional[Dict[str, Any]]:
    """The one thing worth asking about this stall, or None if nothing is.

    Returns a kind, the question, and who could answer it -- an operator, or a
    change to the platform. ``None`` means the run stalled without a blocker
    anyone can name, which is itself worth not dressing up as a question.
    """
    from app.services import agent_failure_diagnosis as diagnosis

    failures = _failures(state)

    for failure in reversed(failures):
        if diagnosis.could_not_run(failure["error"]):
            return {
                "kind": TOOL_UNAVAILABLE,
                "tool": failure["tool"],
                "question": (
                    f"Is {failure['tool']} available? It never ran, reporting: "
                    f"{failure['error'][:MAX_QUOTE_CHARS]}"
                ),
                "answerable_by": "platform_change",
            }

    for failure in reversed(failures):
        if diagnosis.blames_the_submitted_code(failure["error"]):
            continue
        # "What does this tool accept?" is only a question somebody can answer
        # when the tool actually judged the arguments. Every other failure --
        # an upstream that errored, a simulation that aborted, an async
        # ingestion that never landed -- wears the same shape in the ledger,
        # and asking an operator to describe the accepted input of a tool that
        # crashed sends a platform problem to a person as though it were a
        # typo. Measured on ten real stalls: five of seven questions were of
        # that kind. The predicate lives beside the other failure-shape
        # predicates rather than here, so the question "did the tool read the
        # arguments" has one answer wherever it is asked.
        if not diagnosis.describes_the_input(failure["error"]):
            continue
        return {
            "kind": TOOL_REFUSES_INPUT,
            "tool": failure["tool"],
            "question": (
                f"What does {failure['tool']} accept here? It refused with: "
                f"{failure['error'][:MAX_QUOTE_CHARS]}"
            ),
            "answerable_by": "operator",
        }

    stranded = _unreachable_evidence(missing, job_type)
    if stranded:
        return {
            "kind": EVIDENCE_UNREACHABLE,
            "evidence": stranded,
            "question": (
                f"Nothing this job type may call produces {stranded}. Should "
                "the contract ask for different evidence, or should the "
                "producing tool be allowed here?"
            ),
            "answerable_by": "platform_change",
        }

    return None


__all__ = [
    "EVIDENCE_UNREACHABLE",
    "TOOL_REFUSES_INPUT",
    "TOOL_UNAVAILABLE",
    "UNKNOWN",
    "describe",
]
