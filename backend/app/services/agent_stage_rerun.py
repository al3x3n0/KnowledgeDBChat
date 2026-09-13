"""A stage asking for an earlier stage to be redone.

The forward path assumes each stage was given what it needed. Sometimes it was
not, and the stage that finds out is the one downstream: an implement stage
discovers the specification left the tie-breaking unstated, a measure stage
discovers the implementation it was handed cannot be timed meaningfully. Today
that ends the run -- the contract goes unmet and the chain stops, correctly but
uselessly, because the thing that needs redoing is upstream and nothing can say
so.

This is that edge. What it is NOT is a retry: the stage does not run again on
the same inputs, an EARLIER stage runs again on a stated correction, and
everything after it is re-derived. That distinction is the whole reason a
backward edge is safe to have at all.

Three refusals keep it from becoming a loop:

* the target must be an edge the pipeline author declared. A run that can jump
  anywhere is not a pipeline.
* the run's shared budget must not be spent. Per-run and not per-edge: two
  stages that each send work back once are converging, and two stages sending
  it back and forth are the same two stages doing it for ever.
* the request must say what was wrong with the earlier output. "Try again" asks
  for the same work to be done the same way; the reason is the only thing that
  makes the second attempt different from the first.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

#: Where the run's spend is recorded. On the JOB's results rather than the
#: state, because state belongs to one stage and the budget belongs to the run.
REVISIT_LEDGER_KEY = "stage_revisits"

MIN_REASON_CHARS = 40


@dataclass(frozen=True)
class RerunRequest:
    """A validated request, or the reason it was refused."""

    stage: str
    reason: str
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error


def declared_targets(config: Optional[Dict[str, Any]]) -> List[str]:
    config = config if isinstance(config, dict) else {}
    targets = config.get("may_revisit")
    return (
        [str(t).strip() for t in targets if str(t).strip()]
        if isinstance(targets, list)
        else []
    )


def budget(config: Optional[Dict[str, Any]]) -> int:
    config = config if isinstance(config, dict) else {}
    try:
        return max(0, int(config.get("revisit_budget", 0)))
    except (TypeError, ValueError):
        return 0


def spent(results: Optional[Dict[str, Any]]) -> int:
    """How many backward edges this RUN has already taken.

    Read from the root job's results, which every stage inherits, so the count
    survives the stage that spent it ending.
    """
    results = results if isinstance(results, dict) else {}
    ledger = results.get(REVISIT_LEDGER_KEY)
    return len(ledger) if isinstance(ledger, list) else 0


def evaluate(
    *,
    stage: str,
    reason: str,
    config: Optional[Dict[str, Any]],
    results: Optional[Dict[str, Any]],
) -> RerunRequest:
    """Whether this stage may send the work back there, and why not."""
    target = str(stage or "").strip()
    said = str(reason or "").strip()
    allowed = declared_targets(config)

    if not target:
        return RerunRequest("", said, "stage is required: name the stage to redo.")

    if not allowed:
        return RerunRequest(
            target,
            said,
            "This stage declares no backward edges, so there is nothing to send "
            "the work back to. If an earlier stage is the problem, say so in "
            "the findings and let the run end: a stage that cannot meet its "
            "contract on what it was given is a real result.",
        )

    if target not in allowed:
        return RerunRequest(
            target,
            said,
            f"{target!r} is not a stage this one may send work back to. "
            f"Declared: {', '.join(allowed)}.",
        )

    remaining = budget(config) - spent(results)
    if remaining <= 0:
        return RerunRequest(
            target,
            said,
            "This run has spent its backward-edge budget. Going back again is "
            "the shape of a loop rather than a correction: finish on what you "
            "have and record what was missing, so the gap is visible.",
        )

    if len(said) < MIN_REASON_CHARS:
        return RerunRequest(
            target,
            said,
            "Say what was wrong with the earlier stage's output, concretely "
            "enough that redoing it can come out differently. Without that the "
            "stage repeats the work it already did, on the same inputs, and "
            "this run pays for it twice.",
        )

    return RerunRequest(target, said)


def record(
    results: Dict[str, Any],
    *,
    from_stage: str,
    to_stage: str,
    reason: str,
    iteration: int,
) -> Dict[str, Any]:
    """Write the hop into the run's ledger.

    Kept because a result reached after going back three times is not the same
    result as one reached first try, and nothing else would show it.
    """
    ledger = results.get(REVISIT_LEDGER_KEY)
    if not isinstance(ledger, list):
        ledger = []
    ledger.append(
        {
            "from": from_stage,
            "to": to_stage,
            "reason": reason[:2000],
            "iteration": int(iteration or 0),
        }
    )
    results[REVISIT_LEDGER_KEY] = ledger[-20:]
    return results
