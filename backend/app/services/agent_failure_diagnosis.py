"""Turn a repeated tool failure into a diagnosis, instead of another retry.

A tool that fails returns a message, and a message is a fine remedy the first
time. The failure mode this exists for is the second and third time: a run
called the compiler with `-march=native` four times and got the same refusal
four times, and the message it was reading could not have helped because the
flag is unsupported on this architecture at all.

Worse, an error can point somewhere the answer is not. A gem5 timeout advises
shrinking the workload, which is right for a slow simulation and useless for a
model that deadlocks: the workload that hung was 3,200 instructions, and no
amount of shrinking would ever have finished. Diagnosing it took a control run,
a bisect and a disassembly -- a protocol, and one worth naming rather than
rediscovering under iteration pressure.

So: the first failure is left alone, a repeat is called out, and a third says
what to do instead of trying again. What it recommends is what actually works
-- run the smallest possible input through the same tool, because that single
result splits the space in half:

    the control also fails   the tool or its environment is broken, and no
                             edit to the input will fix it
    the control succeeds     the input is at fault, so bisect it: remove one
                             element at a time until the failure flips
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, List, Mapping, Optional

# How many times the same failure may recur before each level of response.
CALL_OUT_AFTER = 2
ESCALATE_AFTER = 3

# A model rarely retries verbatim: it edits the code and calls again, so the
# params differ every time while the failure does not. Measured on a live run,
# benchmark_c_snippet failed seven times with five different compile errors and
# never once tripped the identical-call check. Same tool, same kind of failure,
# different arguments is its own signal and needs a higher bar, because
# genuinely making progress through a series of different errors looks the same
# from here for the first few attempts.
CLASS_ESCALATE_AFTER = 4

# Params that identify *what* was asked rather than how it was labelled.
# Labels and free-text notes change between otherwise identical calls and
# would hide a verbatim retry.
IGNORED_PARAMS = frozenset({"label", "notes", "subject", "title", "reason"})

_ERROR_CLASSES = (
    ("timeout", re.compile(r"timed out|timeout|deadline exceeded", re.I)),
    ("compilation", re.compile(r"compil|assembler|undefined reference|linker", re.I)),
    ("not_found", re.compile(r"not found|no such|does not exist|unknown \w+", re.I)),
    (
        "invalid_argument",
        re.compile(r"invalid|unsupported|not of the form|required", re.I),
    ),
    ("permission", re.compile(r"permission|denied|not allowlisted|disabled", re.I)),
    ("resource", re.compile(r"out of memory|oom|killed|no space|exceeds", re.I)),
)


def classify_error(text: Any) -> str:
    """Bucket an error message by what kind of problem it describes.

    Only used to tell one failure from another, so the buckets are coarse on
    purpose: an exact-text match would treat two timeouts differing by a
    duration as unrelated, which is how a repeat goes unnoticed.
    """
    message = str(text or "").strip()
    if not message:
        return "unknown"
    for name, pattern in _ERROR_CLASSES:
        if pattern.search(message):
            return name
    return "unknown"


def _canonical_params(params: Any) -> str:
    if not isinstance(params, Mapping):
        return ""
    salient = {
        key: value for key, value in params.items() if str(key) not in IGNORED_PARAMS
    }
    try:
        return json.dumps(salient, sort_keys=True, default=str)
    except Exception:  # pragma: no cover - defensive
        return str(sorted(salient))


def signature(tool: str, params: Any, error: Any) -> str:
    """Identify a failure by what was asked and how it broke."""
    digest = hashlib.sha256(_canonical_params(params).encode("utf-8")).hexdigest()[:16]
    return f"{class_signature(tool, error)}:{digest}"


def class_signature(tool: str, error: Any) -> str:
    """Identify a failure by tool and kind, ignoring what was asked."""
    return f"{str(tool).strip()}:{classify_error(error)}"


def _error_of(result: Any) -> str:
    if not isinstance(result, Mapping):
        return ""
    direct = result.get("error")
    if direct:
        return str(direct)
    data = result.get("data")
    if isinstance(data, Mapping) and data.get("error"):
        return str(data.get("error"))
    return ""


def _failed(result: Any) -> bool:
    if not isinstance(result, Mapping):
        return False
    if result.get("success") is False:
        return True
    return bool(result.get("error")) and not result.get("success")


def prior_failures(
    state: Mapping[str, Any], target: str, *, by_class: bool = False
) -> int:
    """Count earlier failures in this run matching a signature.

    `by_class` ignores the arguments, which is what catches a run editing its
    input between attempts and hitting the same wall each time.
    """
    actions = state.get("actions_taken")
    if not isinstance(actions, list):
        return 0
    seen = 0
    for entry in actions:
        if not isinstance(entry, dict):
            continue
        action = entry.get("action") if isinstance(entry.get("action"), dict) else {}
        result = entry.get("result") if isinstance(entry.get("result"), dict) else {}
        if not _failed(result):
            continue
        error = _error_of(result)
        found = (
            class_signature(action.get("tool"), error)
            if by_class
            else signature(action.get("tool"), action.get("params"), error)
        )
        if found == target:
            seen += 1
    return seen


def diagnostic_protocol(tool: str) -> List[str]:
    """The steps that separate a broken tool from a broken input."""
    return [
        f"Run {tool} on the smallest input that should work at all -- an empty "
        "or trivial one. This single result splits the problem in half.",
        "If that control also fails, the tool or its environment is at fault "
        "and no edit to your input will help. Report it and use another route "
        "to the same evidence.",
        "If the control succeeds, the input is at fault. Bisect it: remove or "
        "simplify one element at a time, keeping everything else identical, "
        "until the failure appears or disappears.",
        "Change one thing per attempt. Two changes at once cannot tell you "
        "which of them mattered.",
    ]


def analyze(
    action: Mapping[str, Any], result: Mapping[str, Any], state: Mapping[str, Any]
) -> Optional[Dict[str, Any]]:
    """Judge a fresh failure against what this run has already tried.

    Returns None for a first failure: the tool's own message is the remedy,
    and repeating it as guidance would only be noise.
    """
    if not _failed(result):
        return None

    tool = str((action or {}).get("tool") or "").strip()
    if not tool:
        return None
    error = _error_of(result)
    target = signature(tool, (action or {}).get("params"), error)
    # The failure being judged is already in the history by the time this runs
    # in some call paths and not in others, so count strictly earlier ones and
    # add this one, rather than trusting either arrangement.
    attempt = prior_failures(state, target) + 1

    by_class = class_signature(tool, error)
    class_attempt = prior_failures(state, by_class, by_class=True) + 1

    if attempt < CALL_OUT_AFTER:
        # The arguments changed, so this is not a verbatim retry -- but a run
        # that keeps rewriting its input and keeps hitting the same kind of
        # failure is not converging either, and nothing else would say so.
        if class_attempt >= CLASS_ESCALATE_AFTER:
            return {
                "signature": by_class,
                "attempt": class_attempt,
                "error_class": classify_error(error),
                "varied_arguments": True,
                "guidance": (
                    f"{tool} has failed {class_attempt} times in this run with "
                    f"{classify_error(error)} errors, each time with different "
                    "arguments. Editing the input and trying again is not "
                    "working. Establish what the tool does accept before "
                    "changing it further."
                ),
                "protocol": diagnostic_protocol(tool),
            }
        return None

    diagnosis: Dict[str, Any] = {
        "signature": target,
        "attempt": attempt,
        "error_class": classify_error(error),
    }

    if attempt < ESCALATE_AFTER:
        diagnosis["guidance"] = (
            f"This is attempt {attempt} at {tool} with the same arguments and "
            "the same failure. Repeating it will fail again. Change the call, "
            "or find out why it fails before calling it once more."
        )
        return diagnosis

    diagnosis["guidance"] = (
        f"{tool} has now failed {attempt} times with the same arguments and "
        "the same error. Stop retrying: the message is either not the real "
        "cause or not something a retry can fix. Diagnose it instead."
    )
    diagnosis["protocol"] = diagnostic_protocol(tool)
    if diagnosis["error_class"] == "timeout":
        # The trap this was written for: a timeout that advises shrinking the
        # input, on a workload already far too small to be slow.
        diagnosis["note"] = (
            "A timeout does not always mean the work was too large. If a much "
            "smaller input times out the same way, the tool is stuck rather "
            "than slow, and the size advice in the message is a dead end."
        )
    return diagnosis
