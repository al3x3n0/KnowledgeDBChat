"""Scoring and clustering helpers for the domain-research orchestrators.

These were nested inside ``run_domain_research_orchestrator``, a 2176-line
function, where they could not be called or tested by anything else. Two of
them -- ``safe_float`` and ``normalize_key`` -- were also duplicated verbatim
inside ``run_research_fleet_orchestrator``, which is what nesting a helper
costs: the second orchestrator could not reach the first one's copy, so it grew
its own.

Nothing here touches the database, the LLM or the job. They map values to
values, which is why they are the part of that function worth having on its
own: a scoring rule that can be read in isolation can be argued with, and one
buried in the middle of an orchestrator cannot.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple

#: Keywords that mark a paper or idea as belonging to a track, and the
#: instruction that goes to the model alongside them. Track-specific, because
#: "relevant" means something different for a compiler question than for a
#: microarchitecture one.
_COMPILER_KEYWORDS: Set[str] = {
    "llvm",
    "mlir",
    "pass",
    "passes",
    "ir",
    "vectorization",
    "vectorizer",
    "codegen",
    "scheduling",
    "fusion",
    "tiling",
    "register",
    "allocation",
    "pipeline",
    "kernel",
}

_MICROARCHITECTURE_KEYWORDS: Set[str] = {
    "cache",
    "ipc",
    "branch",
    "predictor",
    "latency",
    "bandwidth",
    "simd",
    "avx",
    "sve",
    "stall",
    "pipeline",
    "frontend",
    "backend",
    "throughput",
    "memory",
}

_GENERIC_KEYWORDS: Set[str] = {
    "benchmark",
    "performance",
    "compiler",
    "kernel",
    "cache",
    "vectorization",
    "latency",
    "throughput",
}

_COMPILER_PROMPT = (
    "Prioritize IR, passes, vectorization, codegen, kernels, compiler "
    "regressions, and optimization pipelines."
)
_MICROARCHITECTURE_PROMPT = (
    "Prioritize cache behavior, branch behavior, SIMD/ISA usage, stalls, "
    "bandwidth, and pipeline efficiency."
)
_GENERIC_PROMPT = (
    "Optimize for novel, evidence-backed, testable ideas across the available "
    "technical evidence."
)

#: How many clusters a signal list may produce, and how long a label may be.
MAX_SIGNAL_CLUSTERS = 8
MAX_CLUSTER_LABEL_CHARS = 180


def safe_float(value: Any, default: float = 0.0) -> float:
    """A float, or the default -- never an exception."""
    try:
        return float(value)
    except Exception:
        return default


def normalize_key(value: Any) -> str:
    """A stable identifier for arbitrary text: lowercase, non-alphanumerics collapsed."""
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")


def track_keyword_sets(track: str) -> Tuple[Set[str], str]:
    """The keywords that count as on-topic for a track, and the model's instruction.

    Returns copies: a caller that mutates the set it is given must not edit the
    rule for every later call in the process.
    """
    if track == "compiler":
        return set(_COMPILER_KEYWORDS), _COMPILER_PROMPT
    if track == "microarchitecture":
        return set(_MICROARCHITECTURE_KEYWORDS), _MICROARCHITECTURE_PROMPT
    return set(_GENERIC_KEYWORDS), _GENERIC_PROMPT


def track_fit_score(track: str, fields: List[str]) -> float:
    """How well some text fits a track, in [0, 1].

    Keyword counting, deliberately: it is cheap, it needs no model, and it is
    used to rank candidates before anything expensive looks at them. Empty text
    scores at the base rather than at zero, because "nothing to judge" is not
    the same claim as "judged and found irrelevant".
    """
    keywords, _prompt = track_keyword_sets(track)
    text = " ".join(
        str(value or "").strip().lower() for value in fields if str(value or "").strip()
    )
    if not text:
        return 0.5 if track == "generic" else 0.35
    hits = sum(1 for keyword in keywords if keyword in text)
    base = 0.45 if track == "generic" else 0.35
    return round(min(1.0, base + (0.08 * hits)), 4)


def signal_clusters_from_ideas(
    signals: List[str], ranked_ideas: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Group the run's signals and its best idea titles into labelled clusters.

    Deduplicated by normalized key, so the same theme arriving as a signal and
    again as an idea title becomes one cluster rather than two.
    """
    buckets: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for raw in list(signals or []) + [
        str(idea.get("title") or "") for idea in ranked_ideas[:6]
    ]:
        text = str(raw or "").strip()
        if not text:
            continue
        key = normalize_key(text)[:64] or f"cluster_{len(buckets) + 1}"
        if key in seen:
            continue
        seen.add(key)
        buckets.append(
            {
                "id": key,
                "label": text[:MAX_CLUSTER_LABEL_CHARS],
                "source_count": 1,
            }
        )
        if len(buckets) >= MAX_SIGNAL_CLUSTERS:
            break
    return buckets


#: The most supporting sources a candidate will carry.
MAX_SUPPORTING_SOURCES = 6


def match_evidence_sources(
    evidence_list: List[str],
    title: str,
    hypothesis: str,
    *,
    source_rows: List[Dict[str, Any]],
    minimum_supporting_sources: int,
) -> List[Dict[str, Any]]:
    """The sources that back a candidate, by title, path or token overlap.

    Note the top-up at the end: when matching finds fewer sources than the
    policy requires, the list is padded with sources that did NOT match, so a
    candidate can reach the minimum on sources unrelated to it. That is the
    behaviour as written and it is preserved here rather than corrected -- the
    tests pin it so the choice is visible, because a citation list padded to a
    quota reads downstream exactly like one that was earned.
    """
    haystacks = [str(title or "").strip(), str(hypothesis or "").strip()]
    haystacks.extend(
        [str(item or "").strip() for item in evidence_list if str(item or "").strip()]
    )
    refs: List[Dict[str, Any]] = []
    seen_refs: Set[str] = set()
    for source in source_rows:
        source_title = str(source.get("title") or "").strip()
        source_path = str(source.get("file_path") or "").strip()
        source_key = normalize_key(source_title)
        if not source_key:
            source_key = normalize_key(source_path)
        if not source_key:
            continue
        matched = False
        for text in haystacks:
            lowered = str(text or "").strip().lower()
            if not lowered:
                continue
            if source_title and (
                source_title.lower() in lowered or lowered in source_title.lower()
            ):
                matched = True
                break
            if source_path and (
                source_path.lower() in lowered or lowered in source_path.lower()
            ):
                matched = True
                break
            overlap = set(source_key.split("_")) & set(
                normalize_key(lowered).split("_")
            )
            if len([token for token in overlap if token]) >= 3:
                matched = True
                break
        if not matched:
            continue
        ref_key = f"{source.get('source_type')}:{source.get('id') or source_title}"
        if ref_key in seen_refs:
            continue
        seen_refs.add(ref_key)
        refs.append(
            {
                "source_type": source.get("source_type"),
                "id": source.get("id"),
                "title": source_title,
                "url": source.get("url"),
                "published": source.get("published"),
                "file_path": source.get("file_path"),
                "source_name": source.get("source_name"),
            }
        )
        if len(refs) >= MAX_SUPPORTING_SOURCES:
            break
    if len(refs) < minimum_supporting_sources:
        for source in source_rows:
            ref_key = (
                f"{source.get('source_type')}:{source.get('id') or source.get('title')}"
            )
            if ref_key in seen_refs:
                continue
            refs.append(
                {
                    "source_type": source.get("source_type"),
                    "id": source.get("id"),
                    "title": source.get("title"),
                    "url": source.get("url"),
                    "published": source.get("published"),
                }
            )
            seen_refs.add(ref_key)
            if len(refs) >= minimum_supporting_sources:
                break
    return refs[:MAX_SUPPORTING_SOURCES]


def build_candidate(
    item: Dict[str, Any],
    idx: int,
    *,
    domain: str,
    track_type: str,
    previous_idea_titles: Set[str],
    scoring_policy: Dict[str, Any],
    confidence_threshold: float,
    source_rows: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Score one proposed idea into a ranked candidate, or reject it as empty.

    Four subscores, combined by the policy's weights and then diluted by the
    model's own confidence and the track fit. ``passes_threshold`` is the gate
    that decides whether the candidate reaches a memo, and it is deliberately
    an AND across every subscore: a candidate cannot buy its way past a missing
    subscore with a high one elsewhere.
    """
    if not isinstance(item, dict):
        return None
    title = str(item.get("title") or "").strip()
    hypothesis = str(item.get("hypothesis") or "").strip()
    opportunity = str(item.get("opportunity") or "").strip()
    if not title and not hypothesis and not opportunity:
        return None
    evidence = item.get("supporting_evidence")
    if isinstance(evidence, list):
        evidence_list = [str(x).strip() for x in evidence if str(x).strip()][:6]
    else:
        evidence_list = [str(evidence).strip()] if str(evidence or "").strip() else []
    next_steps = [
        str(x).strip()
        for x in (
            item.get("next_steps") if isinstance(item.get("next_steps"), list) else []
        )
        if str(x).strip()
    ][:5]
    counterarguments = [
        str(x).strip()
        for x in (
            item.get("counterarguments")
            if isinstance(item.get("counterarguments"), list)
            else []
        )
        if str(x).strip()
    ][:4]
    normalized_title = title or hypothesis[:180] or f"{domain} hypothesis {idx + 1}"
    matched_sources = match_evidence_sources(
        evidence_list,
        normalized_title,
        hypothesis,
        source_rows=source_rows,
        minimum_supporting_sources=scoring_policy["minimum_supporting_sources"],
    )
    evidence_count = len(matched_sources)
    is_new = normalize_key(normalized_title) not in previous_idea_titles
    novelty_score = 0.9 if is_new else 0.35
    evidence_score = min(1.0, 0.35 + 0.2 * min(evidence_count, 3))
    testability_score = 0.45
    if next_steps:
        testability_score += 0.1 * min(len(next_steps), 3)
    if hypothesis:
        testability_score += 0.1
    testability_score = min(1.0, testability_score)
    llm_confidence = max(0.0, min(safe_float(item.get("confidence"), 0.55), 1.0))
    fit_score = track_fit_score(
        track_type,
        [
            normalized_title,
            hypothesis,
            opportunity,
            *evidence_list,
            *[str(source.get("title") or "") for source in matched_sources],
            *[str(source.get("file_path") or "") for source in matched_sources],
        ],
    )
    weighted = (
        novelty_score * scoring_policy["weights"]["novelty"]
        + evidence_score * scoring_policy["weights"]["evidence"]
        + testability_score * scoring_policy["weights"]["testability"]
    )
    overall_score = round(
        min(1.0, (weighted * 0.75) + (llm_confidence * 0.15) + (fit_score * 0.10)),
        4,
    )
    return {
        "id": f"idea_{idx + 1}",
        "title": normalized_title,
        "hypothesis": hypothesis or opportunity,
        "opportunity": opportunity,
        "supporting_evidence": evidence_list,
        "supporting_sources": matched_sources,
        "counterarguments": counterarguments,
        "confidence": llm_confidence,
        "novelty_score": round(novelty_score, 4),
        "evidence_score": round(evidence_score, 4),
        "testability_score": round(testability_score, 4),
        "track_fit_score": round(fit_score, 4),
        "overall_score": overall_score,
        "passes_threshold": (
            overall_score >= confidence_threshold
            and novelty_score >= scoring_policy["minimum_subscore"]
            and evidence_score >= scoring_policy["minimum_subscore"]
            and testability_score >= scoring_policy["minimum_subscore"]
            and evidence_count >= scoring_policy["minimum_supporting_sources"]
        ),
        "is_new": is_new,
        "next_steps": next_steps or ["Validate on a bounded benchmark slice"],
    }
