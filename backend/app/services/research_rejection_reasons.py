"""What a rejection teaches, which is not the same for every rejection.

The monitor profile learns from triage: accepted items add their words to a
positive counter, rejected items to a negative one. That is right for one kind
of rejection and actively harmful for the others, because the words it learns
from are the item's *topic* words. Reject a weak paper about sparse attention
and the profile concludes you dislike "sparse attention" -- the topic you care
about most is the one you are punished for reading carelessly. The signal is
strongest exactly where it is most wrong, since the topics you see most are the
ones you reject most in absolute terms.

The fix is not a better weighting. It is that "this is not my subject" and
"this is my subject done badly" are different sentences, and only the first one
is about the words. So a rejection may now carry a reason, and the reason
decides whether the topic counters move at all.

``low_quality`` deliberately teaches **nothing**. The honest lesson there is
about a venue, a group or an author, none of which this profile models, and a
coarse stand-in -- downweighting every arXiv paper because one was thin -- would
be a worse error than silence. The UI says so rather than implying the click
changed something.

An absent reason keeps the old behaviour. Rows triaged before this existed were
recorded under those rules, and silently reinterpreting them would rewrite a
history nobody can check.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class RejectionReason:
    """One reason, and what the profile is entitled to conclude from it."""

    key: str
    label: str
    #: Whether the item's words are evidence about what you want to read.
    teaches_topic: bool
    #: Shown under the choice, so the effect of the click is never a surprise.
    effect: str


REASONS: Tuple[RejectionReason, ...] = (
    RejectionReason(
        key="off_topic",
        label="Not my subject",
        teaches_topic=True,
        effect="Shows fewer items using these words.",
    ),
    RejectionReason(
        key="low_quality",
        label="My subject, done badly",
        teaches_topic=False,
        effect="Changes nothing — the words were right, and the profile does "
        "not model venues or authors.",
    ),
    RejectionReason(
        key="already_known",
        label="I already know this",
        teaches_topic=False,
        effect="Changes nothing — this is a duplicate, not a preference.",
    ),
    RejectionReason(
        key="not_now",
        label="Not right now",
        teaches_topic=False,
        effect="Changes nothing — timing is not a topic.",
    ),
)

_BY_KEY: Dict[str, RejectionReason] = {reason.key: reason for reason in REASONS}

#: The vocabulary a request may use. Anything else is refused rather than
#: stored, or the column would fill with values no learner knows how to read.
VALID_KEYS: Tuple[str, ...] = tuple(_BY_KEY)


def normalize(raw: object) -> Optional[str]:
    """The stored form of a submitted reason, or ``None`` when unspecified."""
    key = str(raw or "").strip().lower()
    return key if key in _BY_KEY else None


def is_valid(raw: object) -> bool:
    """False only for a non-empty value outside the vocabulary."""
    return not str(raw or "").strip() or normalize(raw) is not None


def teaches_topic(raw: object) -> bool:
    """Whether this rejection is evidence about the words.

    Unspecified means yes: that is what every rejection meant before reasons
    existed, and rows recorded under those rules still say it.
    """
    reason = _BY_KEY.get(normalize(raw) or "")
    return True if reason is None else reason.teaches_topic


def describe() -> List[Dict[str, object]]:
    """The vocabulary, for a client that offers the choice."""
    return [
        {
            "key": reason.key,
            "label": reason.label,
            "teaches_topic": reason.teaches_topic,
            "effect": reason.effect,
        }
        for reason in REASONS
    ]


__all__ = [
    "REASONS",
    "VALID_KEYS",
    "RejectionReason",
    "describe",
    "is_valid",
    "normalize",
    "teaches_topic",
]
