"""Letting a run withdraw a number it has shown to be wrong.

The retraction machinery existed and nothing could reach it. A run that
discovers a recalled finding is false -- which is what happened when an
`IrregularStreamBufferPrefetcher` result turned out to describe a mechanism
that issues no prefetches -- could add a contradicting finding, but the false
one stayed recallable at equal standing, with a job id and a measurement source
attached. The more useful a wrong claim sounds, the more likely the next run is
to build on it.

A retraction must cite what overturns the claim, checked the same way
`record_prediction` checks `derived_from`: against findings this run actually
produced. Without that it is an opinion with the power to delete evidence, and
the failure mode is a run that reasons its way out of an inconvenient number.
"""

from __future__ import annotations

import logging
from typing import Any, List, Mapping, Optional, Sequence

logger = logging.getLogger(__name__)

#: `<job_id>#<index>`, as `recall_prior_findings` now reports on every finding.
REF_SHAPE = "<job_id>#<index>"


def _own_finding_identities(state: Optional[Mapping[str, Any]]) -> List[str]:
    """What this run has established, as types and titles it may cite."""
    findings = (state or {}).get("findings") if isinstance(state, Mapping) else None
    out: List[str] = []
    for finding in findings or []:
        if not isinstance(finding, Mapping) or finding.get("recalled"):
            continue
        for field in ("type", "title", "subject"):
            value = str(finding.get(field) or "").strip()
            if value:
                out.append(value)
    return out


def check(
    ref: str,
    reason: str,
    contradicted_by: Sequence[str],
    state: Optional[Mapping[str, Any]],
) -> Optional[str]:
    """Why this retraction cannot be recorded, or None."""
    ref = str(ref or "").strip()
    if "#" not in ref:
        return (
            f"ref must address one finding as {REF_SHAPE}, which is the `ref` "
            "field on every finding recall_prior_findings returns. A job id "
            "alone names a job, not the claim inside it."
        )
    if len(str(reason or "").strip()) < 20:
        return (
            "reason must say what is wrong in enough detail for a later run to "
            "tell a number withdrawn for a harness defect from one withdrawn "
            "because the question changed."
        )

    cited = [str(c).strip() for c in (contradicted_by or []) if str(c).strip()]
    if not cited:
        return (
            "contradicted_by is required: name the findings THIS run produced "
            "that overturn the claim. A retraction that cites nothing is an "
            "opinion, and this one deletes evidence."
        )

    own = _own_finding_identities(state)
    if not own:
        return (
            "This run has produced no findings of its own, so nothing here can "
            "overturn anything. Measure the thing that contradicts the claim "
            "first, then retract it."
        )
    unknown = [c for c in cited if not any(c in o or o in c for o in own)]
    if unknown:
        return (
            "contradicted_by names "
            + ", ".join(repr(u) for u in unknown[:3])
            + ", which this run did not produce. It has: "
            + ", ".join(sorted(set(own))[:8])
            + "."
        )
    return None
