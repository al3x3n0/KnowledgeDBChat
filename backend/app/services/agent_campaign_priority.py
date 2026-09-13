"""Decide which piece of a campaign's backlog is worth doing next.

A campaign worked its list in creation order, which is the same as having no
opinion: the tenth speculative offshoot of a line that has produced nothing
ranked exactly alongside a candidate that came out of a job that met its
contract. With a job budget in the tens and jobs that take tens of minutes,
order is most of what a campaign decides.

The scoring here is arithmetic over outcomes the campaign already recorded --
deliberately not a model call. A scheduler tick that consulted an LLM would
make the same campaign replay differently every time, and the reason all the
state lives in the database is so that it does not. What the arithmetic buys is
modest and worth stating plainly: it prefers work descending from jobs that
produced something, it discounts depth, and it stops lines that have twice
produced nothing. It is not a research director.

Three judgements, each with a failure mode it is guarding against:

  yield      work born from a job that produced the findings the campaign is
             looking for outranks work someone guessed at when writing the
             seed list -- but only where there is a settled outcome to read,
             since an unsettled parent is not evidence of anything.

  depth      each generation away from the seeds is discounted, because a
             campaign that spawns from its own spawn drifts from the goal it
             was given and nothing else pulls it back.

  cold       a line whose recent ancestors settled without meeting a contract
             and without target findings is abandoned rather than merely
             ranked last. Ranking last is not enough when the budget is finite:
             the item still runs eventually, and burns a job doing it.

Against cold-line abandonment there is a real objection -- a line may be cold
because the questions were hard rather than wrong. Two guards: seeds are never
dropped whatever their record, and dropping needs two settled ancestors, so a
single bad job never ends a line. A dropped item keeps its row and its reason,
so the decision can be read afterwards and argued with.

The anti-starvation cap is the counterweight to the yield preference. A
campaign whose jobs spawn their own successors can chase its own tail forever
while the seed list -- the work an actual person asked for -- never starts.
After a run of discovered items, a waiting seed goes next regardless of score.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

# Two settled ancestors that produced nothing. One is a bad job; two is a line.
COLD_RUN_LIMIT = 2
# How many self-spawned items may run back to back while seed work waits.
MAX_CONSECUTIVE_DISCOVERED = 3
# Per generation away from the seed list.
DEPTH_PENALTY = 0.25
# Each sibling from the same parent already launched discounts the next.
SIBLING_PENALTY = 0.12

SEED_BASE = 1.0
DISCOVERED_BASE = 1.0
YIELD_WEIGHT = 0.6


def _counts(outcome: Optional[Mapping[str, Any]]) -> Dict[str, int]:
    if not isinstance(outcome, Mapping):
        return {}
    counts = outcome.get("finding_counts")
    if not isinstance(counts, Mapping):
        return {}
    return {str(k): int(v or 0) for k, v in counts.items()}


def target_findings(
    outcome: Optional[Mapping[str, Any]], target_types: Sequence[str]
) -> int:
    """How many findings of the types this campaign is actually looking for."""
    counts = _counts(outcome)
    if not target_types:
        # No declared targets: any finding counts, which is the honest reading
        # of a campaign that never said what it was after.
        return sum(counts.values())
    return sum(counts.get(str(name), 0) for name in target_types)


def parent_yield(
    outcome: Optional[Mapping[str, Any]], target_types: Sequence[str]
) -> float:
    """What the job behind this item produced, as 0..1.

    Half for meeting its contract and half for producing the findings the
    campaign is after, because either alone is a weak signal: a contract can be
    met by a job that found nothing interesting, and findings can pile up in a
    job that never established anything.
    """
    if not isinstance(outcome, Mapping):
        return 0.0
    score = 0.0
    if outcome.get("contract_satisfied"):
        score += 0.5
    if target_findings(outcome, target_types) > 0:
        score += 0.5
    return score


def produced_nothing(
    outcome: Optional[Mapping[str, Any]], target_types: Sequence[str]
) -> bool:
    """A settled job that neither met its contract nor found what was wanted."""
    if not isinstance(outcome, Mapping):
        return False
    if outcome.get("contract_satisfied"):
        return False
    return target_findings(outcome, target_types) == 0


def is_cold(
    ancestry: Sequence[Optional[Mapping[str, Any]]], target_types: Sequence[str]
) -> bool:
    """Has this line produced nothing across enough consecutive ancestors?

    `ancestry` is nearest parent first. Unsettled ancestors (no outcome yet)
    stop the walk rather than counting either way -- a line is not cold because
    its parent has not finished.
    """
    run = 0
    for outcome in ancestry:
        if not isinstance(outcome, Mapping):
            return False
        if not produced_nothing(outcome, target_types):
            return False
        run += 1
        if run >= COLD_RUN_LIMIT:
            return True
    return False


def score(view: Mapping[str, Any], target_types: Sequence[str]) -> Tuple[float, str]:
    """Rank one pending item, and say in one line why it got that rank."""
    origin = str(view.get("origin") or "seed")
    generation = max(0, int(view.get("generation") or 0))
    siblings = max(0, int(view.get("siblings_launched") or 0))
    outcome = view.get("parent_outcome")

    reasons: List[str] = []
    if origin == "discovered":
        value = DISCOVERED_BASE
        yielded = parent_yield(outcome, target_types)
        value += YIELD_WEIGHT * yielded
        if not isinstance(outcome, Mapping):
            reasons.append("from a job that has not settled")
        elif yielded >= 1.0:
            reasons.append("from a job that met its contract and found targets")
        elif yielded > 0:
            reasons.append("from a job that produced something")
        else:
            reasons.append("from a job that produced nothing yet")
    else:
        value = SEED_BASE
        reasons.append("asked for in the seed list")

    if generation > 1:
        penalty = DEPTH_PENALTY * (generation - 1)
        value -= penalty
        reasons.append(f"generation {generation}")

    if siblings:
        value -= SIBLING_PENALTY * siblings
        reasons.append(f"{siblings} sibling(s) already run")

    return round(value, 4), "; ".join(reasons)


def starved_seed_index(
    pending: Sequence[Mapping[str, Any]], recent_origins: Sequence[str]
) -> Optional[int]:
    """Index of a seed that must go next because self-spawned work has had a run.

    Returns None when nothing is starved.
    """
    trailing = 0
    for origin in reversed(list(recent_origins)):
        if str(origin) != "discovered":
            break
        trailing += 1
    if trailing < MAX_CONSECUTIVE_DISCOVERED:
        return None
    for index, view in enumerate(pending):
        if str(view.get("origin") or "seed") == "seed":
            return index
    return None


def choose(
    pending: Sequence[Mapping[str, Any]],
    *,
    target_types: Sequence[str] = (),
    recent_origins: Sequence[str] = (),
) -> Optional[Dict[str, Any]]:
    """Pick the next item to run, with the reason attached.

    `pending` is in creation order, which stays the tie-break: two items the
    scoring cannot separate are done oldest first, so a campaign with nothing
    to go on behaves exactly as it did before.
    """
    if not pending:
        return None

    scored = []
    for index, view in enumerate(pending):
        value, reason = score(view, target_types)
        scored.append({"index": index, "score": value, "reason": reason})

    starved = starved_seed_index(pending, recent_origins)
    if starved is not None:
        chosen = dict(scored[starved])
        chosen["reason"] = (
            f"{MAX_CONSECUTIVE_DISCOVERED} self-spawned items ran in a row; "
            "taking waiting seed work next"
        )
        chosen["starved"] = True
        return chosen

    best = max(scored, key=lambda row: (row["score"], -row["index"]))
    return dict(best)


def describe(view: Mapping[str, Any], target_types: Sequence[str] = ()) -> str:
    value, reason = score(view, target_types)
    return f"{value:+.2f} ({reason})"
