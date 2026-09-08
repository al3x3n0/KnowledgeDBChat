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

#: Rounds a stage that declared its own `until` may produce nothing new before
#: it is treated as stuck. Higher than the default: its author chose a
#: different success condition, so being stopped this way is not what they
#: asked for, and a loop written on purpose is likelier to contain patient
#: work that records nothing for a while.
DECLARED_LOOP_DRY_ROUNDS = 5


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

    # `until` and dry rounds answer different questions, so one must not
    # silence the other. `until: contract_satisfied` says when the stage is
    # DONE; dry rounds say when it is STUCK, and a stage going in circles is
    # stuck whichever way its author wrote the success condition.
    #
    # Measured: an implement stage declaring `until: contract_satisfied` got a
    # real diagnostic at iteration 4 and then spent iterations 5 to 10 reading
    # documents and writing progress reports -- ten consecutive actions, no
    # code, no checks, nothing new recorded, and nothing in place to notice.
    # Before this, declaring any loop at all bought a stage out of stall
    # detection entirely.
    dry_rounds = _as_int(config.get("loop_dry_rounds"), DEFAULT_DRY_ROUNDS)
    if dry_rounds < 1:
        dry_rounds = DEFAULT_DRY_ROUNDS
    if policy != "no_new_findings":
        # A stage that declared a different success condition is given more
        # rope before being called stuck: it did not ask to be stopped this
        # way, and patient work is likelier where an author wrote a loop on
        # purpose.
        dry_rounds = max(
            dry_rounds,
            _as_int(config.get("loop_dry_rounds"), 0) or DECLARED_LOOP_DRY_ROUNDS,
        )

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
        + (
            f" (the stage asked to run until {policy}, which says when it is "
            "done, not whether it is getting anywhere)"
            if policy and policy != "no_new_findings"
            else ""
        )
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


#: State a run accumulates as evidence that it should give up. A correction
#: from an operator invalidates all of it: the run stopped because of what it
#: believed, and it has just been told that belief was wrong.
GIVE_UP_STATE_KEYS = (
    "loop_finding_counts",
    "loop_policy_stop_reason",
    "stopped_short_of_contract",
    "stopped_short_reason",
    "stalled_iterations",
)


def clear_give_up_state(state: Dict[str, Any]) -> None:
    """Let a corrected run start over on the question of whether it is stuck.

    Without this, resuming with a clue is a no-op that looks like the agent
    ignored it: the restored state still holds the dry-round history that made
    `should_stop` fire, so the run does its setup, stops before its first
    iteration, and blocks again on the same reason. Measured on a live run --
    `resumed > skill_profile_resolved > ... > loop_policy_stop`, zero work
    done in between.
    """
    for key in GIVE_UP_STATE_KEYS:
        state.pop(key, None)
