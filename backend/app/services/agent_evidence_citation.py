"""Match a cited piece of evidence to a finding the run actually produced.

`record_prediction` and `record_method` both refuse a citation naming evidence
that does not exist, which is the check that caught a run predicting from a
measurement it never obtained. The check has to stay strict about *whether* the
evidence exists and can afford to be generous about how it is spelled.

It was not, and a live run showed the cost. Asked what its prediction derived
from, a model wrote "simulated_measurement: fsqrt dependent chain O3CPU
simulation" -- the right finding type with a description appended -- and was
refused twice for it, then a third time for the same string. It had the
evidence and could not say so in the accepted form.

So: an exact type matches, a type followed by any description matches, and a
sentence mentioning known types matches all of them -- a citation naming two is
a claim on both, and keeping only the first would quietly narrow the evidence a
record rests on. A citation mentioning none of them still fails, because at
that point there is nothing to check it against.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Optional, Tuple


def resolve(cited: str, available: Iterable[str]) -> Optional[str]:
    """Return the finding type a citation names first, or None if it names none."""
    text = str(cited or "").strip()
    known = [str(t).strip() for t in available if str(t).strip()]
    if not text or not known:
        return None

    if text in known:
        return text

    lowered = text.lower()
    # "simulated_measurement: fsqrt chain" and "simulated_measurement (O3CPU)"
    # both name their type first and then say which one they mean.
    for candidate in sorted(known, key=len, reverse=True):
        if lowered.startswith(candidate.lower()):
            return candidate

    # A sentence that mentions exactly one known type is unambiguous. Tokenised
    # rather than substring-matched so that a type name embedded in a longer
    # word cannot count as a mention.
    mentioned = names_in(text, known)
    return mentioned[0] if mentioned else None


def names_in(cited: str, available: Iterable[str]) -> List[str]:
    """Every known finding type a citation mentions, in the order written."""
    text = str(cited or "")
    lowered = text.lower()
    known = [str(t).strip() for t in available if str(t).strip()]
    # Tokenised rather than substring-matched, so a type name embedded in a
    # longer word cannot count as a mention.
    tokens = set(re.findall(r"[a-z0-9_]+", lowered))
    found = [(lowered.find(c.lower()), c) for c in known if c.lower() in tokens]
    return [c for _, c in sorted(found)]


def resolve_all(
    cited: Iterable[str], available: Iterable[str]
) -> Tuple[List[str], List[str]]:
    """Split citations into the types they resolve to and the ones that fail."""
    known = [str(t).strip() for t in available if str(t).strip()]
    resolved: List[str] = []
    unresolved: List[str] = []
    for item in cited:
        matches = names_in(item, known)
        if not matches:
            # A prefix citation still counts even if the type is not a whole
            # token in the rest of the sentence.
            single = resolve(item, known)
            matches = [single] if single else []
        if not matches:
            unresolved.append(str(item))
            continue
        for match in matches:
            if match not in resolved:
                resolved.append(match)
    return resolved, unresolved


def explain_unresolved(unresolved: Iterable[str], available: Iterable[str]) -> str:
    """Say what was not found and, exactly, what may be written instead."""
    names = sorted({str(t).strip() for t in available if str(t).strip()})
    return (
        f"No finding of type {', '.join(str(u)[:80] for u in unresolved)} exists "
        "in this run. Cite one of these exact finding types: "
        + (", ".join(names) if names else "none recorded yet")
        + ". Obtain the measurement first, then cite what it returned."
    )
