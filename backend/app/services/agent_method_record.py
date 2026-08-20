"""Record a way of doing something, with the evidence that it works.

Findings say what was learned about the subject. This is for what was learned
about *how to investigate* it, which is the part that transfers: the next job
on a different kernel still needs to know that measuring instruction latency
through C expressions does not work because the compiler reshapes them, and
that the way to check a timing harness is an operation whose answer is known
in advance.

That knowledge existed in this project only as prose a human wrote down. The
run that discovered it could store a memory, but nothing said what a method
record should contain, and nothing distinguished one validated by evidence
from a passing thought. Both failures matter: an unstructured note is not
reusable, and an unvalidated method reused with confidence is worse than none.

So a record must carry:

    procedure   the steps, in order, concrete enough to follow
    prevents    the wrong answer it exists to stop, which is what tells a
                future reader whether their situation is the same one
    evidence    finding types produced *in this run* that establish it

The evidence check is the same one `record_prediction` uses, for the same
reason: a run once predicted from a measurement it never obtained. A method
may be recorded without evidence, but only by saying so, and it is stored
marked as unvalidated rather than quietly indistinguishable.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Optional, Sequence

MEMORY_TYPE = "pattern"
MAX_STEPS = 12
MAX_FIELD_CHARS = 600

# Written into the record so a reader -- human or model -- can tell a method
# that was demonstrated from one that was merely asserted.
VALIDATED = "validated"
UNVALIDATED = "unvalidated"
NO_EVIDENCE = "none"


class MethodRecordError(ValueError):
    """The record cannot be written as asked."""


def _clean(value: Any, limit: int = MAX_FIELD_CHARS) -> str:
    return str(value or "").strip()[:limit]


def _steps(value: Any) -> List[str]:
    if isinstance(value, str):
        # Accept a numbered or newline-separated block, which is how a model
        # most often writes a procedure when not handed a list.
        raw = [re.sub(r"^\s*\d+[.)]\s*", "", line) for line in value.splitlines()]
    elif isinstance(value, (list, tuple)):
        raw = [str(item) for item in value]
    else:
        raw = []
    return [_clean(step, 300) for step in raw if _clean(step, 300)][:MAX_STEPS]


def build_record(
    *,
    name: str,
    procedure: Any,
    prevents: str,
    derived_from: Any,
    available_finding_types: Sequence[str],
    applies_to: Any = None,
    limits: str = "",
) -> Dict[str, Any]:
    """Validate a method record against what this run actually produced.

    Raises rather than returning an error dict, so a caller cannot store a
    half-built record by forgetting to check.
    """
    clean_name = _clean(name, 160)
    if not clean_name:
        raise MethodRecordError(
            "name is required: a method with no name cannot be recalled"
        )

    steps = _steps(procedure)
    if not steps:
        raise MethodRecordError(
            "procedure is required: a method without steps is an opinion, not "
            "something a later run can follow"
        )

    clean_prevents = _clean(prevents)
    if not clean_prevents:
        raise MethodRecordError(
            "prevents is required: state the wrong answer this method exists to "
            "stop. Without it a reader cannot tell whether their situation is "
            "the one it applies to"
        )

    if isinstance(derived_from, str):
        cited = [derived_from]
    elif isinstance(derived_from, (list, tuple)):
        cited = [str(item) for item in derived_from]
    else:
        cited = []
    cited = [_clean(item, 80) for item in cited if _clean(item, 80)]
    if not cited:
        raise MethodRecordError(
            "derived_from is required: name the finding types in this run that "
            f"establish the method, or pass ['{NO_EVIDENCE}'] to record it as "
            "unvalidated and say why in limits"
        )

    available = {str(x).strip() for x in available_finding_types if str(x).strip()}
    if cited == [NO_EVIDENCE]:
        status = UNVALIDATED
        evidence: List[str] = []
    else:
        missing = [item for item in cited if item not in available]
        if missing:
            raise MethodRecordError(
                f"This method says it derives from {', '.join(missing)}, but no "
                f"such finding exists in this run. Findings so far: "
                f"{', '.join(sorted(available)) or 'none'}. Demonstrate the "
                "method before recording it, or record it as unvalidated with "
                f"derived_from=['{NO_EVIDENCE}']"
            )
        status = VALIDATED
        evidence = cited

    targets = applies_to
    if isinstance(targets, str):
        targets = [targets]
    scope = [_clean(item, 80) for item in (targets or []) if _clean(item, 80)][:12]

    return {
        "name": clean_name,
        "procedure": steps,
        "prevents": clean_prevents,
        "applies_to": scope,
        "limits": _clean(limits),
        "evidence": evidence,
        "status": status,
    }


def render(record: Mapping[str, Any]) -> str:
    """Write the record as the text stored in memory.

    A fixed layout, because these are recalled as plain text into a later
    run's context: a reader has to be able to tell the procedure from the
    caveats without the structure having survived the round trip.
    """
    lines = [f"METHOD: {record.get('name')}"]
    status = record.get("status")
    if status == UNVALIDATED:
        lines.append(
            "STATUS: unvalidated -- recorded without evidence in the run that "
            "wrote it. Demonstrate it before relying on it."
        )
    else:
        lines.append(f"STATUS: validated by {', '.join(record.get('evidence') or [])}")
    if record.get("applies_to"):
        lines.append(f"APPLIES TO: {', '.join(record['applies_to'])}")
    lines.append("PROCEDURE:")
    lines.extend(
        f"  {i}. {step}" for i, step in enumerate(record.get("procedure") or [], 1)
    )
    lines.append(f"PREVENTS: {record.get('prevents')}")
    if record.get("limits"):
        lines.append(f"LIMITS: {record['limits']}")
    return "\n".join(lines)


def parse(content: str) -> Optional[Dict[str, Any]]:
    """Read a stored record back, for display and for reuse checks."""
    text = str(content or "")
    if not text.startswith("METHOD:"):
        return None
    record: Dict[str, Any] = {
        "name": "",
        "procedure": [],
        "prevents": "",
        "applies_to": [],
        "limits": "",
        "status": VALIDATED,
        "evidence": [],
    }
    section = ""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("METHOD:"):
            record["name"] = stripped[len("METHOD:") :].strip()
        elif stripped.startswith("STATUS:"):
            value = stripped[len("STATUS:") :].strip()
            record["status"] = (
                UNVALIDATED if value.startswith(UNVALIDATED) else VALIDATED
            )
            if record["status"] == VALIDATED and "validated by" in value:
                record["evidence"] = [
                    item.strip()
                    for item in value.split("validated by", 1)[1].split(",")
                    if item.strip()
                ]
        elif stripped.startswith("APPLIES TO:"):
            record["applies_to"] = [
                item.strip()
                for item in stripped[len("APPLIES TO:") :].split(",")
                if item.strip()
            ]
        elif stripped.startswith("PROCEDURE:"):
            section = "procedure"
        elif stripped.startswith("PREVENTS:"):
            section = ""
            record["prevents"] = stripped[len("PREVENTS:") :].strip()
        elif stripped.startswith("LIMITS:"):
            section = ""
            record["limits"] = stripped[len("LIMITS:") :].strip()
        elif section == "procedure" and stripped:
            record["procedure"].append(re.sub(r"^\d+\.\s*", "", stripped))
    return record


def tags_for(record: Mapping[str, Any]) -> List[str]:
    """Tags that make a method findable by the work it applies to."""
    tags = ["method", str(record.get("status") or VALIDATED)]
    tags.extend(str(item) for item in (record.get("applies_to") or [])[:6])
    return tags[:10]
