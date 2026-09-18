"""Evidence types a stage's goal is probably asking for.

Writing a contract is the hardest step in authoring a pipeline, and it is the
step that decides whether the pipeline can stop honestly. To write one you have
to know which of 51 evidence types exists, which tool produces it, and whether
that tool may run under your stage's job type. Get any of the three wrong and
the only feedback is a validation error on a stage you have already written.

Most of that is mechanical. "Identify research gaps" wants ``research_gap``
because ``identify_research_gaps`` produces it; "Create a presentation" wants
``research_presentation`` because ``generate_research_presentation`` does. The
words in a goal and the words in a producing tool are drawn from the same small
domain, so overlap between them is a usable signal.

It is a signal and not an answer, which is why this **suggests and never
applies**. Scoring picks candidates a person can recognise or reject in a
second; it cannot know that "analyse the papers" in this pipeline means
clustering rather than summarising. So each suggestion carries the tool that
produces it and the words that matched, because a suggestion you cannot check
is worse than none -- you would be trading one thing you did not understand for
another.

Deliberately not a model call. This runs while someone types, costs nothing,
and returns the same answer twice; the drafting service is where a model earns
its keep, on the harder problem of inventing whole stages.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from app.services import agent_pipeline_vocabulary as vocabulary

#: Words too common in a goal to mean anything about evidence. Kept small on
#: purpose: a stop list that grows starts deleting the domain words that carry
#: the signal ("analysis" and "report" are exactly what distinguishes stages).
_NOISE = frozenset(
    """
    a an the and or of for to in on with by from at into this that these those
    all any some each every it its as is are be been being do does did done
    then than so such very more most other others new newly
    stage step goal run job use using used make made get gets
    """.split()
)

#: Minimum overlap before a candidate is worth showing. Below this the match is
#: usually one incidental word, and a wrong suggestion presented confidently is
#: more expensive than an empty list.
_FLOOR = 0.15


def _words(text: str) -> List[str]:
    """Lowercase word stems, near enough for matching short domain phrases."""
    raw = re.findall(r"[a-zA-Z]+", str(text or "").lower())
    out: List[str] = []
    for word in raw:
        if word in _NOISE or len(word) < 3:
            continue
        # Crude singularisation: "papers" and "paper" are the same signal here.
        if len(word) > 4 and word.endswith("ies"):
            word = word[:-3] + "y"
        elif len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        out.append(word)
    return out


@dataclass(frozen=True)
class ContractSuggestion:
    """One evidence type a stage might require, and why it was suggested."""

    finding_type: str
    #: The tool whose name matched most strongly. What makes it checkable.
    produced_by: str
    #: Words shared by the goal and the evidence, in the goal's order.
    matched: Tuple[str, ...]
    #: 0..1, for ordering only. Not shown as a number: a precise-looking score
    #: on a guess invites more trust than the guess deserves.
    score: float
    typical_seconds: int
    perishable: bool

    def as_dict(self) -> Dict[str, Any]:
        return {
            "finding_type": self.finding_type,
            "produced_by": self.produced_by,
            "matched": list(self.matched),
            "typical_seconds": self.typical_seconds,
            "perishable": self.perishable,
        }


def _candidates_for_job_type(job_type: str) -> List[Any]:
    """Evidence this job type may actually produce.

    A suggestion the stage could not call is worse than no suggestion: the
    author accepts it, and the checker then refuses the stage for a reason
    that looks unrelated to what they just clicked.
    """
    from app.agent_core import tool_specs

    wanted = str(job_type or "").strip()
    runnable_here = set(tool_specs.STATIC_CATALOG.tools_for_job_type(wanted))
    out = []
    for evidence in vocabulary.evidence_types():
        allowed = tuple(evidence.job_types or ())
        if allowed and wanted and wanted not in allowed:
            continue
        # The evidence saying a job type may produce it is not the same as a
        # tool existing that this job type can call. ``literature_review``
        # declares no restriction and yet both its producers are reachable only
        # from chat or MCP, so suggesting it hands the author a contract that
        # can never be satisfied -- which is worse than suggesting nothing,
        # because it looks like an answer.
        if evidence.producers and not any(
            producer in runnable_here for producer in evidence.producers
        ):
            continue
        out.append(evidence)
    return out


def suggest(
    *,
    goal: str,
    job_type: str = "research",
    limit: int = 4,
    exclude: Optional[Sequence[str]] = None,
) -> List[ContractSuggestion]:
    """Evidence types this goal is probably asking for, best first.

    Empty when nothing scores above the floor, which is the right answer for a
    goal written in words the vocabulary does not share. An author who sees
    nothing looks at the vocabulary; an author who sees a bad guess may take it.
    """
    goal_words = _words(goal)
    if not goal_words:
        return []
    goal_set = set(goal_words)
    already = {str(x).strip() for x in (exclude or []) if str(x).strip()}

    scored: List[Tuple[float, ContractSuggestion]] = []
    for evidence in _candidates_for_job_type(job_type):
        if evidence.name in already:
            continue

        best_producer = ""
        best_overlap: set = set()
        best_ratio = 0.0
        # The evidence name and each producing tool are separate chances to
        # match: a goal may echo either "research gap" or "identify gaps".
        for source in (evidence.name, *evidence.producers):
            source_words = set(_words(source.replace("_", " ")))
            if not source_words:
                continue
            overlap = goal_set & source_words
            if not overlap:
                continue
            # Against the source, not the goal: a long goal should not dilute
            # a tool whose every word it contains.
            ratio = len(overlap) / len(source_words)
            if ratio > best_ratio:
                best_ratio, best_overlap, best_producer = ratio, overlap, source

        if best_ratio < _FLOOR:
            continue
        scored.append(
            (
                best_ratio,
                ContractSuggestion(
                    finding_type=evidence.name,
                    produced_by=(
                        best_producer
                        if best_producer in evidence.producers
                        else (evidence.producers[0] if evidence.producers else "")
                    ),
                    matched=tuple(w for w in goal_words if w in best_overlap),
                    score=round(best_ratio, 3),
                    typical_seconds=evidence.typical_seconds,
                    perishable=evidence.perishable,
                ),
            )
        )

    scored.sort(key=lambda pair: (-pair[0], pair[1].finding_type))
    return [suggestion for _, suggestion in scored[:limit]]


def suggest_for_spec(
    spec: Mapping[str, Any], limit: int = 4
) -> Dict[str, List[Dict[str, Any]]]:
    """Suggestions per stage id, for the stages that have no contract yet.

    Stages that already require something are left alone: the author has
    answered this question, and a suggestion beside their answer reads as a
    correction rather than an offer.
    """
    out: Dict[str, List[Dict[str, Any]]] = {}
    stages = spec.get("stages") if isinstance(spec, Mapping) else None
    for index, stage in enumerate(stages if isinstance(stages, list) else []):
        if not isinstance(stage, Mapping):
            continue
        stage_id = str(stage.get("id") or f"stage_{index + 1}").strip()
        contract = stage.get("contract")
        required = []
        if isinstance(contract, Mapping):
            counts = contract.get("required_finding_type_counts")
            names = contract.get("required_finding_types")
            if isinstance(counts, Mapping):
                required = list(counts)
            elif isinstance(names, (list, Mapping)):
                required = list(names)
        if required:
            continue
        found = suggest(
            goal=str(stage.get("goal") or ""),
            job_type=str(stage.get("job_type") or "research"),
            limit=limit,
        )
        if found:
            out[stage_id] = [s.as_dict() for s in found]
    return out


__all__ = ["ContractSuggestion", "suggest", "suggest_for_spec"]
