"""Judging a candidate against a baseline timed beside it.

Two separate `benchmark_c_snippet` calls are two measurements taken at
different moments on a host whose speed moves: on this machine repeated runs
of one program varied by up to 44%, so a 20% difference between two calls a
minute apart says nothing. A campaign concluded on exactly that -- one trial
of the candidate and no baseline at all.

`compare_c_snippets` times both programs in one container, alternating
(A B, B A, A B, ...), so drift lands on both arms. This module turns those
paired trials into a verdict, and the verdict has to be able to say "I cannot
tell":

* The host's own noise is measured in the same run, for free: consecutive
  BASELINE trials are an A/A comparison, and how far they disagree is how far
  two runs of the same program disagree here and now. A difference smaller
  than that is not a difference.
* A verdict of faster or slower also needs the rounds to agree: most of the
  paired trials must point the same way, so one lucky round cannot carry it.
* Too few rounds, or a workload too small to time on this host, is
  `cannot_tell`, with what to change.

The arithmetic is on log ratios, so 2x faster and 2x slower are symmetric.
"""

from __future__ import annotations

import math
from statistics import median
from typing import Any, Dict, List, Sequence

MIN_ROUNDS = 5

#: Share of rounds that must agree for faster/slower.
AGREEMENT = 0.8

#: The smallest noise band assumed even when the A/A trials happen to agree
#: closely. Two trials matching to the millisecond is luck, not precision.
MIN_NOISE = 0.02

#: Below this a workload is at the host's scheduling noise (measured here; see
#: agent_compiler_sandbox.NOISE_FLOOR_MS), and with a noise band this wide
#: nothing can be resolved.
NOISE_FLOOR_MS = 250
UNRESOLVABLE_NOISE = 0.25


def _quantile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def judge(candidate_ms: Sequence[int], baseline_ms: Sequence[int]) -> Dict[str, Any]:
    """The verdict for paired trials, round i of each arm taken together."""
    pairs = [
        (float(c), float(b))
        for c, b in zip(candidate_ms, baseline_ms)
        if c is not None and b is not None and c > 0 and b > 0
    ]
    out: Dict[str, Any] = {"rounds": len(pairs)}
    if len(pairs) < MIN_ROUNDS:
        out.update(
            verdict="cannot_tell",
            reason=(
                f"{len(pairs)} paired round(s); at least {MIN_ROUNDS} are needed "
                "before a difference can be told from the host's noise. Pass "
                f"rounds={max(MIN_ROUNDS, 8)} or more."
            ),
        )
        return out

    logs = [math.log(c / b) for c, b in pairs]
    baseline = [b for _, b in pairs]
    null = [
        abs(math.log(baseline[i + 1] / baseline[i])) for i in range(len(baseline) - 1)
    ]
    noise = max(MIN_NOISE, _quantile(null, 0.75))
    centre = median(logs)
    ratio = math.exp(centre)
    faster_rounds = sum(1 for x in logs if x < 0)
    slower_rounds = sum(1 for x in logs if x > 0)

    out.update(
        candidate_fastest_ms=int(min(c for c, _ in pairs)),
        baseline_fastest_ms=int(min(baseline)),
        # Candidate time over baseline time, per round and overall.
        ratio=round(ratio, 4),
        speedup=round(1.0 / ratio, 4),
        round_ratios=[round(math.exp(x), 4) for x in logs],
        noise_band=round(math.exp(noise) - 1.0, 4),
        faster_rounds=faster_rounds,
        slower_rounds=slower_rounds,
    )

    smallest = min(out["candidate_fastest_ms"], out["baseline_fastest_ms"])
    if smallest < NOISE_FLOOR_MS and noise >= math.log(1 + UNRESOLVABLE_NOISE):
        out.update(
            verdict="cannot_tell",
            reason=(
                f"At {smallest} ms a trial is too short to time on this host: "
                f"two runs of the baseline alone disagreed by up to "
                f"{out['noise_band'] * 100:.0f}%. Scale the work up until the "
                f"fastest trial clears ~{NOISE_FLOOR_MS} ms."
            ),
        )
        return out

    n = len(pairs)
    if centre < -noise and faster_rounds >= AGREEMENT * n:
        verdict = "faster"
        reason = (
            f"{out['speedup']:.2f}x faster than the baseline, beyond the host's "
            f"own noise of {out['noise_band'] * 100:.1f}%, in {faster_rounds} of "
            f"{n} rounds."
        )
    elif centre > noise and slower_rounds >= AGREEMENT * n:
        verdict = "slower"
        reason = (
            f"{out['ratio']:.2f}x the baseline's time, beyond the host's own "
            f"noise of {out['noise_band'] * 100:.1f}%, in {slower_rounds} of {n} "
            "rounds."
        )
    else:
        verdict = "indistinguishable"
        if abs(centre) <= noise:
            reason = (
                f"The difference ({(ratio - 1) * 100:+.1f}%) is within the host's "
                f"own noise ({out['noise_band'] * 100:.1f}%, two runs of the "
                "baseline against each other): this measurement cannot tell "
                "them apart."
            )
        else:
            reason = (
                f"The median difference is {(ratio - 1) * 100:+.1f}%, but only "
                f"{max(faster_rounds, slower_rounds)} of {n} rounds agree on "
                "its direction; the rounds disagree too much to call it."
            )
    out.update(verdict=verdict, reason=reason)
    return out


def interleaved_order(rounds: int) -> List[str]:
    """Which arm runs first in each round: A B, B A, A B, ...

    Alternating cancels a steady drift (a host warming up or cooling down),
    which would otherwise always favour whichever arm ran second.
    """
    return ["ab" if i % 2 == 0 else "ba" for i in range(max(0, int(rounds)))]


__all__ = ["MIN_ROUNDS", "interleaved_order", "judge"]
