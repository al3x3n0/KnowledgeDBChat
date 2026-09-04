"""When a looping stage should stop, beyond running out of iterations.

A pipeline stage may declare `loop: {max_iterations, until, dry_rounds}`. The
binding writes all three into the job config, and until now only
`max_iterations` was read: `until` and `dry_rounds` were documented options
that nothing honoured, which is worse than not offering them — an author who
writes `until: no_new_findings` gets a run that ignores it silently.

Two termination conditions, and they answer different questions:

    contract_satisfied   stop when the goal contract holds. This is the
                         executor's existing behaviour and needs nothing here:
                         a satisfied contract already ends the run.

    no_new_findings      stop when the last few rounds established nothing.
                         The condition a patch-and-test loop actually needs.
                         A run that patches, tests, reads the failures and
                         patches again is making progress; a run that produces
                         nothing new for two rounds running is stuck, and the
                         remaining iterations will be spent the same way.

`dry_rounds` defaults to two rather than one because one empty round is
ordinary. A round can come up empty while the agent reads context, and cutting
a run off for it would stop exactly the patient work that eventually lands.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

#: What a stage may ask for. An unrecognised value is ignored rather than
#: guessed at, and reported, because silently picking a policy the author did
#: not choose is how a run stops for a reason nobody can explain.
KNOWN_POLICIES = ("contract_satisfied", "no_new_findings")

DEFAULT_DRY_ROUNDS = 2


def record_round(state: Dict[str, Any], finding_count: int) -> None:
    """Note how many findings existed at the end of an iteration.

    Kept as a running list rather than a single "last count" so the policy can
    look back over several rounds, which is what `dry_rounds` means.
    """
    history = state.get("loop_finding_counts")
    if not isinstance(history, list):
        history = []
    history.append(int(finding_count))
    # Only the recent tail matters, and an unbounded list rides in the job
    # state that gets serialised every iteration.
    state["loop_finding_counts"] = history[-20:]


def should_stop(
    config: Optional[Dict[str, Any]], state: Optional[Dict[str, Any]]
) -> Tuple[bool, str]:
    """Whether a looping stage has stopped making progress.

    Returns (stop, reason). The reason is written into the job log, so it has
    to say what happened in terms someone reading the log can act on.
    """
    config = config if isinstance(config, dict) else {}
    state = state if isinstance(state, dict) else {}

    policy = str(config.get("loop_until") or "").strip().lower()
    if policy != "no_new_findings":
        # contract_satisfied, unset, or something unrecognised: the executor's
        # own contract handling decides, which is the right default.
        return False, ""

    dry_rounds = _as_int(config.get("loop_dry_rounds"), DEFAULT_DRY_ROUNDS)
    if dry_rounds < 1:
        dry_rounds = DEFAULT_DRY_ROUNDS

    history: List[int] = [
        int(n) for n in (state.get("loop_finding_counts") or []) if isinstance(n, int)
    ]
    # Need one round before the window to compare against: with dry_rounds=2
    # that is three observations, and the first two rounds of any run cannot
    # yet be dry by this definition.
    if len(history) < dry_rounds + 1:
        return False, ""

    window = history[-(dry_rounds + 1) :]
    if window[-1] > window[0]:
        return False, ""

    return True, (
        f"{dry_rounds} consecutive rounds produced no new findings "
        f"(still {window[-1]}); stopping rather than spending the remaining "
        "iterations the same way"
    )


def policy_warning(config: Optional[Dict[str, Any]]) -> str:
    """Say so when a stage asked for a policy that does not exist.

    Silence here means an author writes `until: whenever_ready` and gets the
    default with nothing to tell them why the run behaved as it did.
    """
    config = config if isinstance(config, dict) else {}
    policy = str(config.get("loop_until") or "").strip().lower()
    if not policy or policy in KNOWN_POLICIES:
        return ""
    return (
        f"Unknown loop policy {policy!r}; treating it as contract_satisfied. "
        f"Known policies: {', '.join(KNOWN_POLICIES)}"
    )


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
