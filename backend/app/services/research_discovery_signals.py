"""Why an inbox item was surfaced, in words a person can act on.

The monitor profile learns from what you keep and what you dismiss
(``research_monitor_profile_service``), and the research runner scores each
candidate against that profile before surfacing it. The scoring already knew
*which* words matched -- it iterates your tokens against the learned ones -- and
then threw them away, appending the constant string ``"token_bias"``. So the
inbox rendered::

    Discovery why: source_type:paper:3, token_bias, phrase_bias

which tells you an opinion exists without telling you what it is, and cannot be
argued with. This module keeps the terms, so the same item reads *matches
"sparse attention", a phrase from items you kept*.

**What the weight is not: a count of items.** The learner accumulates with
``Counter.update(tokens)``, so a word used three times in one accepted item
contributes three, and the stored number is a signed net over *occurrences*
across accepted minus rejected items. "You accepted this four times" would be a
false sentence built from a true number, so the phrasing here names the
direction and the evidence ("from items you kept") and leaves the number to a
tooltip. Source-type weights are not keeps at all -- they come from follow-up
outcomes -- and get their own wording for that reason.

The tokenizer below must stay identical to the one that fed the learner, or a
term would be explained that was never scored; ``test_research_discovery_signals``
asserts that against the real profile service rather than a copy.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

#: Mirrors ``ResearchMonitorProfileService.STOPWORDS``. Domain words ("paper",
#: "model", "dataset") are in it because they appear in everything a research
#: inbox holds and so separate nothing.
STOPWORDS = frozenset(
    """
    the and for with from that this into over under when where what which while
    your you are our their they them then than also only just more most less
    use using used make made help helps via can could should would may might
    will data dataset datasets model models train training eval evaluate
    evaluation assistant job jobs paper papers doc docs document documents
    research monitor
    """.split()
)

#: How many terms to keep. Four short clauses is a sentence; ten is a log line.
MAX_SIGNALS = 4

#: The slices the score is actually computed over. Explaining a term outside
#: them would name something that contributed nothing to the number shown.
_TOKEN_WINDOW = 10
_PHRASE_WINDOW = 6

#: Weights as applied to the score. Source type dominates deliberately: it is a
#: statement about outcomes rather than about wording.
_SOURCE_TYPE_MULTIPLIER = 6
_PHRASE_MULTIPLIER = 2

#: Most specific first. A phrase says more about why than a bare word, and a
#: word says more than "it is a paper", so that is the order a person reads.
_SPECIFICITY = {"phrase": 0, "token": 1, "source_type": 2}


def tokenize(text: str) -> List[str]:
    """Lowercase content words, exactly as the learner tokenizes them."""
    raw = re.findall(r"[a-zA-Z0-9_\\-]+", (text or "").lower())
    out: List[str] = []
    for word in raw:
        word = word.strip("_-")
        if len(word) < 3 or word in STOPWORDS:
            continue
        out.append(word)
    return out


def bigrams(tokens: Sequence[str]) -> List[str]:
    """Adjacent pairs -- the same phrases the profile stores."""
    return [f"{tokens[i]} {tokens[i + 1]}" for i in range(len(tokens) - 1)]


@dataclass(frozen=True)
class DiscoverySignal:
    """One learned term that moved this item's score, and by how much."""

    kind: str  # phrase | token | source_type
    term: str
    #: The learned weight from the profile: signed, net over occurrences in
    #: kept minus dismissed items. Not a count of items.
    weight: int

    @property
    def favourable(self) -> bool:
        return self.weight > 0

    def label(self) -> str:
        """The short form, for a chip in a list. The justification lives in
        :meth:`describe`, which a tooltip shows -- a list row has space for the
        term that matched, not for why the term is known."""
        if self.kind == "source_type":
            return f"{self.term} items"
        return f"“{self.term}”"

    def describe(self) -> str:
        """One clause, true to what the number behind it actually measures."""
        if self.kind == "source_type":
            # These come from follow-up outcomes, not from keeps.
            if self.favourable:
                return f"{self.term} items have led somewhere before"
            return f"{self.term} items have not led anywhere before"
        noun = "phrase" if self.kind == "phrase" else "word"
        kept = "kept" if self.favourable else "dismissed"
        return f"matches “{self.term}”, a {noun} from items you {kept}"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "term": self.term,
            "weight": self.weight,
            "favourable": self.favourable,
            "label": self.label(),
            "text": self.describe(),
        }


@dataclass(frozen=True)
class DiscoveryExplanation:
    """The score, and the terms that account for it."""

    score: int
    signals: Tuple[DiscoverySignal, ...]

    @property
    def reasons(self) -> List[str]:
        """Human clauses. Kept under the old ``discovery_reasons`` key so a
        reader that predates this module gets prose instead of a debug dump."""
        return [signal.describe() for signal in self.signals]

    def as_metadata(self) -> Dict[str, Any]:
        return {
            "discovery_score": self.score,
            "discovery_reasons": self.reasons,
            "discovery_signals": [signal.as_dict() for signal in self.signals],
            "score_explained": bool(self.signals),
        }


def _weights(bias: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = bias.get(key)
    return value if isinstance(value, Mapping) else {}


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def explain(
    *,
    bias: Optional[Mapping[str, Any]],
    title: Optional[str],
    summary: Optional[str] = None,
    item_type: Optional[str] = None,
) -> DiscoveryExplanation:
    """Score a candidate against a learned profile, keeping the reasons.

    The arithmetic is unchanged from the scorer this replaced -- the windows and
    multipliers above are that scorer's -- so an item scores what it always did.
    What is new is that the terms survive.
    """
    if not isinstance(bias, Mapping):
        return DiscoveryExplanation(score=0, signals=())

    token_scores = _weights(bias, "token_scores")
    phrase_scores = _weights(bias, "phrase_scores")
    source_type_scores = _weights(bias, "source_type_scores")

    tokens = tokenize(f"{title or ''} {summary or ''}".strip())
    phrases = bigrams(tokens)

    score = 0
    found: List[DiscoverySignal] = []
    seen: set = set()

    def remember(kind: str, term: str, weight: int) -> None:
        # Deduplicated because a repeated word contributes to the score twice
        # but is one reason; the weight shown stays the learned one.
        if weight and (kind, term) not in seen:
            seen.add((kind, term))
            found.append(DiscoverySignal(kind=kind, term=term, weight=weight))

    if item_type and item_type in source_type_scores:
        delta = _as_int(source_type_scores.get(item_type))
        score += delta * _SOURCE_TYPE_MULTIPLIER
        remember("source_type", item_type, delta)

    for token in tokens[:_TOKEN_WINDOW]:
        weight = _as_int(token_scores.get(token))
        score += weight
        remember("token", token, weight)

    for phrase in phrases[:_PHRASE_WINDOW]:
        weight = _as_int(phrase_scores.get(phrase))
        score += weight * _PHRASE_MULTIPLIER
        remember("phrase", phrase, weight)

    found.sort(key=lambda s: (_SPECIFICITY.get(s.kind, 9), -abs(s.weight), s.term))
    return DiscoveryExplanation(score=int(score), signals=tuple(found[:MAX_SIGNALS]))


__all__ = [
    "STOPWORDS",
    "DiscoveryExplanation",
    "DiscoverySignal",
    "bigrams",
    "explain",
    "tokenize",
]
