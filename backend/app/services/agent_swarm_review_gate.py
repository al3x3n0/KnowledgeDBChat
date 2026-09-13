"""Whether a swarm's merged verdict should stop for a person.

A swarm costs several agents to run, and its whole product is one judgement:
these independent looks at the same question agreed, or they did not. Letting
that judgement complete silently means the expensive part happened and nobody
read the answer.

The opposite -- stopping for approval on every merge -- is worse in practice.
A gate a person meets on every clean run is a gate they learn to approve
without reading, and then it is costing attention while protecting nothing.

So the default holds on the verdicts that actually need a human and lets the
rest through:

  * **contested** -- two roles measured the same thing and materially
    disagreed. This is the output a swarm exists to produce.
  * **inconclusive** -- they measured the same thing through an instrument too
    imprecise to resolve it. Nearly as actionable as a disagreement, because
    the fix is usually cheap: rerun on a quiet machine.
  * **incomplete** -- roles were expected that never reached a terminal state,
    so the merge is over a subset and the missing one may be the dissenter.
  * **nothing cross-checked** -- the roles never spoke to the same subject, so
    the swarm produced N independent opinions and zero corroboration. It
    reported no conflict, but it also did not do the one thing it is for.

A clean corroboration is the case that does NOT need a person, and letting it
through is what keeps the gate worth reading when it does fire.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

#: Never hold. The merge completes and the verdict is read later, or not.
NEVER = "never"
#: Hold only on a verdict a person should see. The default.
ON_DISPUTE = "on_dispute"
#: Hold on every merge, however clean.
ALWAYS = "always"

POLICIES = (NEVER, ON_DISPUTE, ALWAYS)


@dataclass(frozen=True)
class GateDecision:
    """Whether to hold, under which policy, and why."""

    hold: bool
    policy: str
    reasons: List[str]

    def as_dict(self) -> Dict[str, Any]:
        return {"hold": self.hold, "policy": self.policy, "reasons": list(self.reasons)}


def _count(fan_in: Mapping[str, Any], key: str) -> int:
    try:
        return int(fan_in.get(key) or 0)
    except (TypeError, ValueError):
        return 0


def normalize_policy(value: Any, *, default: str = ON_DISPUTE) -> str:
    """A policy name, or the default for anything unrecognised.

    Deliberately forgiving: a typo in a job config should not turn a gate into
    an exception in the finalizer, which runs after the work is done and would
    lose it.
    """
    token = str(value or "").strip().lower().replace("-", "_")
    return token if token in POLICIES else default


def reasons_to_review(fan_in: Mapping[str, Any]) -> List[str]:
    """What about this merge a person should look at. Empty means nothing."""
    if not isinstance(fan_in, Mapping):
        return []

    reasons: List[str] = []

    contested = _count(fan_in, "contested_count")
    if contested:
        reasons.append(
            f"{contested} contested: roles measured the same thing and disagreed"
        )

    inconclusive = _count(fan_in, "inconclusive_count")
    if inconclusive:
        reasons.append(
            f"{inconclusive} cross-checked but unresolved: measured too noisily "
            "to tell agreement from coincidence"
        )

    # The merge already names this one; read it rather than recomputing, so the
    # gate and the summary can never disagree about whether a role is missing.
    for conflict in fan_in.get("conflicts") or []:
        if not isinstance(conflict, Mapping):
            continue
        if str(conflict.get("type") or "") == "incomplete_swarm":
            reasons.append(
                str(conflict.get("description") or "Not every role finished.")[:200]
            )
            break

    # Nothing was checked against anything. No conflict is reported because
    # there was no comparison to conflict -- which is exactly why it needs
    # saying out loud: the swarm ran and corroborated nothing.
    if fan_in.get("typed_agreement") is None and not _count(
        fan_in, "inconclusive_count"
    ):
        if _count(fan_in, "corroborated_count") == 0:
            reasons.append(
                "no two roles spoke to the same subject, so nothing was " "corroborated"
            )

    return reasons


def decide(
    fan_in: Optional[Mapping[str, Any]], policy: Any = ON_DISPUTE
) -> GateDecision:
    """Whether this merge stops for a person."""
    resolved = normalize_policy(policy)
    if not isinstance(fan_in, Mapping) or not fan_in:
        # Not a swarm merge at all: there is no verdict to review.
        return GateDecision(hold=False, policy=resolved, reasons=[])

    if resolved == NEVER:
        return GateDecision(hold=False, policy=resolved, reasons=[])

    reasons = reasons_to_review(fan_in)
    if resolved == ALWAYS:
        return GateDecision(
            hold=True,
            policy=resolved,
            reasons=reasons or ["policy is to review every swarm merge"],
        )

    return GateDecision(hold=bool(reasons), policy=resolved, reasons=reasons)
