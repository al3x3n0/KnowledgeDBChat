"""Verify the instrument, not just the result.

Goal contracts check what a run produced: enough findings, of the right types,
with uncertainty attached, inside physical bounds. Every one of those checks
assumes the tool that produced the numbers was working. Nothing checked that,
and on this project that assumption failed twice in a way no contract could
have caught:

* A wall-clock harness on a loaded host timed two **identical** dependent add
  chains at a ratio of 0.53, then 0.49 with the order reversed. It was
  measuring which block ran first. Every latency this project had measured was
  a ratio of exactly that kind.
* Later, on a quiet host, the same null control read a clean 1.0000 four times
  running while a chain of **known ratio 2.0** read 3.47. A disturbance that
  hits two identical chains cancels; one that hits two different chains does
  not. One control is not enough.

A control is a call to a tool whose correct answer is known in advance. If it
comes back wrong, nothing that tool produced in that window is evidence --
regardless of how well-formed, well-typed and in-bounds it looks.

Two rules the failures above earned:

**Bracket, do not precede.** A control before the measurement says the
instrument worked before. Hosts drift. A third measurement run in this project
was discarded because the control taken *immediately after* read 2.2012 against
a required 2.000 -- and its data showed exactly the damage that predicts,
`fsqrt` at 16.91 where the accepted runs read 10.01 and 10.43. A measurement is
trusted when a passing control sits on **both** sides of it.

**More than one control, testing different things.** A control the disturbance
cancels out of proves nothing. Each control here names what it catches that the
others do not.

This is deliberately not a tolerance on results. A cycle count whose error is
"small" is a cycle count whose error is unquantified. The band here is the
point past which the *instrument* is disqualified, which is a different kind of
judgement and a much wider one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

# --- the control programs -------------------------------------------------
#
# Both are the ones actually validated on the host: the null control moved from
# 0.09 on a loaded machine to 1.0000 on a quiet one, and the scale control
# caught a drifting host the null control called clean.
#
# Two details in these are load-bearing and were each arrived at by a wrong
# answer first:
#
#   * The statistic is the MEDIAN of per-round ratios. Pooling totals across
#     rounds reads 0.53, because one preempted block dominates a sum. ABBA
#     ordering, which cancels linear drift, made it worse and more variable
#     (0.61 to 1.16) -- so the disturbance is bursty, not a trend.
#   * The accumulators are pinned with `register long x asm("x9")` and passed
#     as operands. Naming them in the clobber list says "destroyed", which is
#     the opposite of carrying a value, and leaves them holding garbage.

_ROUNDS = 31
_ITERATIONS = 100_000
_UNROLL = 16

_ADD_CHAIN = "\n".join(['        "add %[x], %[x], #1\\n\\t"'] * _UNROLL)
_ADD_CHAIN_2X = "\n".join(['        "add %[y], %[y], #1\\n\\t"'] * (2 * _UNROLL))


def _control_program(target_body: str, target_reg: str, ops_divisor: int) -> str:
    return f"""
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static int cmp(const void *x, const void *y) {{
    double a = *(const double *)x, b = *(const double *)y;
    return (a > b) - (a < b);
}}

int main(void) {{
    register long a asm("x9") = 1;
    register long b asm("x10") = 1;
    struct timespec t0, t1;
    double ratios[{_ROUNDS}];
    for (int r = 0; r < {_ROUNDS}; r++) {{
        double ta, tb;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        for (long i = 0; i < {_ITERATIONS}; i++) {{
            asm volatile(
{_ADD_CHAIN}
                : [x] "+r"(a));
        }}
        clock_gettime(CLOCK_MONOTONIC, &t1);
        ta = (t1.tv_sec - t0.tv_sec) * 1e9 + (t1.tv_nsec - t0.tv_nsec);
        clock_gettime(CLOCK_MONOTONIC, &t0);
        for (long i = 0; i < {_ITERATIONS}; i++) {{
            asm volatile(
{target_body}
                : [{target_reg}] "+r"(b));
        }}
        clock_gettime(CLOCK_MONOTONIC, &t1);
        tb = (t1.tv_sec - t0.tv_sec) * 1e9 + (t1.tv_nsec - t0.tv_nsec);
        ratios[r] = (tb / (double){ops_divisor}) / (ta / (double){_UNROLL});
    }}
    qsort(ratios, {_ROUNDS}, sizeof(double), cmp);
    printf("control_ratio=%.6f\\n", ratios[{_ROUNDS} / 2]);
    return 0;
}}
"""


NULL_CONTROL_PROGRAM = _control_program(
    _ADD_CHAIN.replace("%[x]", "%[y]"), "y", _UNROLL
)
SCALE_CONTROL_PROGRAM = _control_program(_ADD_CHAIN_2X, "y", _UNROLL)


@dataclass(frozen=True)
class Control:
    """A call whose correct answer is known before it is made."""

    name: str
    tool: str
    params: Dict[str, Any]
    #: `key=value` line to read out of the tool's stdout.
    expect_key: str
    expect_value: float
    #: Fractional band. Wide on purpose: this disqualifies an instrument, it
    #: does not grade a result.
    tolerance: float
    #: What this catches that the other controls do not. Shown when it fails,
    #: because "a control failed" is not actionable and this is.
    catches: str


CONTROLS: Dict[str, List[Control]] = {
    "benchmark_c_snippet": [
        Control(
            name="null_control",
            tool="benchmark_c_snippet",
            params={
                "code": NULL_CONTROL_PROGRAM,
                "flags": "-O2",
                "repeat": 1,
                "label": "null control: two identical dependent add chains",
            },
            expect_key="control_ratio",
            expect_value=1.0,
            tolerance=0.05,
            catches=(
                "a host that is timing position rather than instructions. Two "
                "identical chains must cost the same; when they do not, every "
                "ratio this tool reports is a ratio of two positions."
            ),
        ),
        Control(
            name="scale_control",
            tool="benchmark_c_snippet",
            params={
                "code": SCALE_CONTROL_PROGRAM,
                "flags": "-O2",
                "repeat": 1,
                "label": "scale control: a chain of known ratio 2.0",
            },
            expect_key="control_ratio",
            expect_value=2.0,
            tolerance=0.05,
            catches=(
                "a host the null control calls clean. A disturbance hitting "
                "two identical chains cancels; this one gives the target twice "
                "the work, so it cannot cancel. Seen reading 3.47 while the "
                "null control read 1.0000 four times."
            ),
        ),
    ],
}


def controls_for(tool: str) -> List[Control]:
    return list(CONTROLS.get(str(tool or ""), []))


def is_controlled(tool: str) -> bool:
    """True when this tool's answers are only evidence if a control passed."""
    return bool(CONTROLS.get(str(tool or "")))


def controlled_tools() -> List[str]:
    return sorted(CONTROLS)


def read_measurement(output: Any, key: str) -> Optional[float]:
    """Pull `key=<number>` out of a tool's stdout."""
    text = output if isinstance(output, str) else str(output or "")
    match = re.search(rf"{re.escape(key)}\s*=\s*([-+0-9.eE]+)", text)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def judge(control: Control, result: Any) -> Dict[str, Any]:
    """Did the instrument give the answer that is known in advance?

    A result that cannot be read at all is a failure, not a skip. "The control
    did not run" and "the control passed" must never resolve the same way --
    that is the same rule the bundle verifier follows for a missing tool.
    """
    verdict: Dict[str, Any] = {
        "control": control.name,
        "tool": control.tool,
        "expected": control.expect_value,
        "tolerance": control.tolerance,
        "catches": control.catches,
    }

    if not isinstance(result, Mapping) or not result.get("success"):
        verdict.update(
            passed=False,
            actual=None,
            reason=(
                f"the control call itself failed: "
                f"{str((result or {}).get('error') if isinstance(result, Mapping) else result)[:200]}"
            ),
        )
        return verdict

    data = result.get("data") if isinstance(result.get("data"), Mapping) else {}
    stdout = data.get("stdout") or data.get("output") or ""
    actual = read_measurement(stdout, control.expect_key)
    if actual is None:
        verdict.update(
            passed=False,
            actual=None,
            reason=(
                f"the control ran but printed no {control.expect_key}=, so "
                "there is nothing to check it against"
            ),
        )
        return verdict

    band = abs(control.expect_value) * control.tolerance
    passed = abs(actual - control.expect_value) <= band
    verdict.update(
        passed=passed,
        actual=round(actual, 6),
        reason=(
            ""
            if passed
            else (
                f"{control.name} read {actual:.4f} against a required "
                f"{control.expect_value:.4f}. This catches {control.catches}"
            )
        ),
    )
    return verdict


# --- reading the run's own record -----------------------------------------


def _control_events(state: Mapping[str, Any], tool: str) -> List[Dict[str, Any]]:
    """Control verdicts recorded in this run, in order, for one tool."""
    events = state.get("instrument_controls")
    if not isinstance(events, Sequence):
        return []
    return [
        dict(event)
        for event in events
        if isinstance(event, Mapping) and str(event.get("tool") or "") == str(tool)
    ]


def _measurement_indices(state: Mapping[str, Any], tool: str) -> List[int]:
    """Positions in the action sequence where this tool was actually used."""
    actions = state.get("actions_taken")
    if not isinstance(actions, Sequence):
        return []
    found = []
    for index, entry in enumerate(actions):
        if not isinstance(entry, Mapping):
            continue
        action = entry.get("action") if isinstance(entry.get("action"), Mapping) else {}
        if str(action.get("tool") or "") == str(tool):
            found.append(index)
    return found


def bracket_status(state: Mapping[str, Any], tool: str) -> Dict[str, Any]:
    """Is every measurement by `tool` surrounded by passing controls?

    Before *and* after. A control that only precedes says the instrument was
    working before, and hosts drift mid-run -- which is how a measurement run
    on this project came back with fsqrt at 16.91 where two accepted runs read
    10.01 and 10.43, caught only by the control taken afterwards.
    """
    if not is_controlled(tool):
        return {"tool": tool, "controlled": False, "bracketed": True, "reason": ""}

    events = _control_events(state, tool)
    uses = _measurement_indices(state, tool)
    if not uses:
        return {"tool": tool, "controlled": True, "bracketed": True, "reason": ""}

    first_use, last_use = min(uses), max(uses)
    before = [
        e
        for e in events
        if e.get("passed") and int(e.get("at_action", -1)) <= first_use
    ]
    after = [
        e for e in events if e.get("passed") and int(e.get("at_action", -1)) >= last_use
    ]

    failures = [e for e in events if not e.get("passed")]
    names_before = {e.get("control") for e in before}
    names_after = {e.get("control") for e in after}
    required = {c.name for c in controls_for(tool)}

    missing_before = sorted(required - names_before)
    missing_after = sorted(required - names_after)

    reason = ""
    if failures:
        reason = failures[0].get("reason") or f"a control on {tool} failed"
    elif missing_before:
        reason = (
            f"{tool} was used without {', '.join(missing_before)} passing first, "
            "so nothing establishes the instrument was working"
        )
    elif missing_after:
        reason = (
            f"{tool} has no {', '.join(missing_after)} after its last "
            "measurement. A host can drift mid-run, and a control that only "
            "precedes cannot see it."
        )

    return {
        "tool": tool,
        "controlled": True,
        "bracketed": not reason,
        "reason": reason,
        "controls_run": len(events),
        "controls_failed": len(failures),
        "missing_before": missing_before,
        "missing_after": missing_after,
    }


def unverified_instruments(state: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Every controlled tool this run used without a passing bracket."""
    out = []
    for tool in controlled_tools():
        status = bracket_status(state, tool)
        if status.get("controlled") and not status.get("bracketed"):
            out.append(status)
    return out


def describe() -> List[str]:
    """What a contract is asking for when it requires verified instruments."""
    lines = []
    for tool, controls in sorted(CONTROLS.items()):
        names = ", ".join(c.name for c in controls)
        lines.append(
            f"{tool}: {names} must each pass before the first measurement and "
            "again after the last one"
        )
    return lines


# --- running them ---------------------------------------------------------


async def run_controls(
    call: Any,
    tool: str,
    state: Any,
    *,
    at_action: Optional[int] = None,
    when: str = "before",
) -> List[Dict[str, Any]]:
    """Run every control for `tool` and record the verdicts on the run state.

    `call` is an awaitable `call(tool_name, params) -> result`, so this module
    never has to know how tools are dispatched and can be tested without one.

    A control that raises is recorded as a failure. Swallowing it would leave
    the run with no verdict at all, which `bracket_status` then reads as an
    absent control -- the same outcome, reached less clearly.
    """
    if not is_controlled(tool):
        return []

    if at_action is None:
        actions = state.get("actions_taken") if isinstance(state, Mapping) else None
        at_action = len(actions) if isinstance(actions, Sequence) else 0

    verdicts: List[Dict[str, Any]] = []
    for control in controls_for(tool):
        try:
            result = await call(control.tool, dict(control.params))
            verdict = judge(control, result)
        except Exception as exc:  # pragma: no cover - defensive
            verdict = {
                "control": control.name,
                "tool": control.tool,
                "passed": False,
                "actual": None,
                "expected": control.expect_value,
                "catches": control.catches,
                "reason": f"the control could not be run: {str(exc)[:200]}",
            }
        verdict["at_action"] = int(at_action)
        verdict["when"] = when
        verdicts.append(verdict)

    if isinstance(state, dict):
        state.setdefault("instrument_controls", []).extend(verdicts)
    return verdicts


def needs_pre_control(state: Mapping[str, Any], tool: str) -> bool:
    """True when this tool is about to be used and has no control yet.

    Deliberately once per run rather than once per call: the controls are
    themselves measurements and running two of them around every single call
    would cost more than the work being measured.
    """
    if not is_controlled(tool):
        return False
    return not _control_events(state, tool)


def needs_post_control(state: Mapping[str, Any], tool: str) -> bool:
    """True when this tool has been used since its last control ran.

    This is the half that cannot be automated at call time, because nothing
    knows which measurement is the last one until the run tries to conclude.
    The evaluate phase asks this before checking the contract.
    """
    if not is_controlled(tool):
        return False
    uses = _measurement_indices(state, tool)
    if not uses:
        return False
    events = _control_events(state, tool)
    if not events:
        return True
    latest_control = max(int(e.get("at_action", -1)) for e in events)
    return latest_control < max(uses)
