"""Whether several agents independently found the same thing.

This replaces matching on prose. The previous rule keyed agreement on the
lowercased text of a finding's title -- LLM-generated sentences, truncated to
280 characters, compared for exact equality. Two agents reaching the same
conclusion in different words never matched, so `support_count >= 2` could
essentially never fire: the merge reported "the roles did not agree" whether
or not they had, and every run produced a conflict. A consensus mechanism that
cannot recognise agreement is worse than none, because it reports a specific
falsehood rather than staying quiet.

What replaces it is the thing this codebase already has and the old rule
ignored: **typed evidence**. A finding is not a sentence, it is a
`benchmark_measurement` with a subject and a number, or an
`implementation_verified` with a pass count. Two agents agree when they
produce the same KIND of evidence about the same SUBJECT, and their values
agree. That is checkable without a language model and without guessing at
paraphrase.

Three rules decide a verdict, and each exists because collapsing it would
misreport something:

* **Corroboration requires distinct roles.** One agent producing a finding
  twice is one agent, not two. Agreement across roles is the whole point --
  a researcher and a verifier arriving separately at a number is evidence in
  a way that one agent repeating itself is not.
* **Numbers agree within a tolerance, never exactly.** Two benchmarks of the
  same code differ; demanding equality of floats reinvents the prose bug in
  arithmetic. Where a finding carries its own spread, that spread is the
  tolerance -- the run already measured how much it varies, and second-
  guessing it with a fixed percentage would discard a real measurement in
  favour of a constant.
* **Contested is a first-class outcome, reported with the values.** "The roles
  disagreed" is not useful. "researcher measured 12.5 ms, verifier measured
  48.0 ms" tells a person what to do next, and which of the two to distrust.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

#: Numeric fields worth comparing, in the order a reader cares about. Drawn
#: from what the measurement tools actually emit rather than invented: the
#: first one present on both sides is the one compared.
_COMPARABLE_FIELDS: Tuple[str, ...] = (
    "fastest_ms",
    "cycles",
    "speedup",
    "ratio",
    "throughput",
    "passed",
    "failed",
    "score",
)

#: Fields a finding may carry describing its own dispersion. Same list the
#: validity checker uses; a spread the run measured beats a constant we chose.
_SPREAD_FIELDS: Tuple[str, ...] = (
    "trial_spread",
    "relative_spread",
    "spread",
    "std_dev",
    "stddev",
    "uncertainty",
)

#: Used only where a finding reports no spread of its own. Deliberately loose:
#: it exists to avoid calling two roughly-equal numbers a disagreement, not to
#: certify them as equal.
DEFAULT_TOLERANCE = 0.10

#: Above this, "they agree" stops carrying information. Tolerance comes from
#: the claims' own reported spread, so a pair of measurements taken on a
#: machine too busy to measure anything arrives with an enormous one -- and
#: then any two numbers fall inside it. Measured, on the first swarm run whose
#: roles both benchmarked: trial spreads of 130% and 142% on a `saturated`
#: host turned 60ms vs 45ms into "corroborated (within 142%)", and the summary
#: reported agreement 1.0. Nothing was wrong with that arithmetic; the reader
#: was still misled, because at 142% two numbers 2.4x apart would also have
#: "agreed".
#:
#: So a comparison this loose resolves nothing and says so, the same way
#: `compare_to_claim` answers `incomparable` rather than forcing a verdict
#: between numbers that cannot be tested against each other.
#:
#: 50%: at that width a 1.5x difference passes as agreement, which for a
#: performance measurement is already useless. Deliberately not lower -- a run
#: reporting 40% variation is making a real statement about its own precision,
#: and overruling it with a constant is the mistake `_tolerance_for` exists to
#: avoid.
MAX_USEFUL_TOLERANCE = 0.50


def _as_number(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _dig(finding: Mapping[str, Any], field_name: str) -> Optional[float]:
    """A numeric field at the top level or one nesting down.

    Tools put their numbers in different places -- some at the top level, some
    under `data` or `details`. A comparison that only looked at the top level
    would silently treat two findings as incomparable whenever the value it
    needed was one level down.
    """
    direct = _as_number(finding.get(field_name))
    if direct is not None:
        return direct
    for container in ("data", "details", "metrics", "value"):
        nested = finding.get(container)
        if isinstance(nested, Mapping):
            found = _as_number(nested.get(field_name))
            if found is not None:
                return found
    return None


def subject_key(finding: Mapping[str, Any]) -> str:
    """What a finding is ABOUT, normalised.

    Prefers the explicit `subject` a measurement tool sets over the title,
    because a title is prose and a subject is a label. Falls back to the title
    only when there is nothing better, and normalises hard: two agents naming
    the same kernel should not be separated by capitalisation or punctuation.
    """
    raw = ""
    for candidate in ("subject", "label", "symbol", "target", "file", "path"):
        value = str(finding.get(candidate) or "").strip()
        if value:
            raw = value
            break
    if not raw:
        raw = str(finding.get("title") or "").strip()
    token = raw.lower()
    token = re.sub(r"[^a-z0-9]+", "_", token).strip("_")
    return token[:120]


@dataclass(frozen=True)
class RoleClaim:
    """One role's finding about one subject."""

    role: str
    value: Optional[float] = None
    field_name: str = ""
    spread: Optional[float] = None
    title: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "value": self.value,
            "field": self.field_name,
            "spread": self.spread,
            "title": self.title,
        }


@dataclass(frozen=True)
class ConsensusGroup:
    """What every role said about one kind of evidence about one subject."""

    finding_type: str
    subject: str
    claims: List[RoleClaim] = field(default_factory=list)
    #: corroborated | contested | inconclusive | uncorroborated
    verdict: str = "uncorroborated"
    #: Why, in a sentence a person can act on.
    detail: str = ""
    #: How far apart the numbers were, and how far apart they were allowed to
    #: be. Carried because a verdict without its resolution cannot be audited:
    #: "corroborated" means something different at 5% than at 45%.
    drift: Optional[float] = None
    tolerance: Optional[float] = None

    @property
    def roles(self) -> List[str]:
        return sorted({c.role for c in self.claims})

    def as_dict(self) -> Dict[str, Any]:
        return {
            "finding_type": self.finding_type,
            "subject": self.subject,
            "verdict": self.verdict,
            "detail": self.detail,
            "roles": self.roles,
            "claims": [c.as_dict() for c in self.claims],
            "drift": self.drift,
            "tolerance": self.tolerance,
        }


def _tolerance_for(claims: Sequence[RoleClaim]) -> float:
    """How far apart two numbers may be and still agree.

    The widest spread any claim reported, because a measurement that says it
    varies by 8% is telling you what agreement means for it. Only when nobody
    reported one does the default apply.
    """
    spreads = [c.spread for c in claims if c.spread is not None and c.spread > 0]
    if not spreads:
        return DEFAULT_TOLERANCE
    return max(max(spreads), DEFAULT_TOLERANCE)


def _resolution(claims: Sequence[RoleClaim]) -> Tuple[Optional[float], Optional[float]]:
    """How far apart the numbers were, and how far apart they could be.

    Returned alongside the verdict so a reader can see what "agree" meant here.
    """
    numbered = [c for c in claims if c.value is not None]
    if len(numbered) < 2:
        return None, None
    values = [c.value for c in numbered]
    low, high = min(values), max(values)
    base = max(abs(low), 1e-9)
    return abs(high - low) / base, _tolerance_for(numbered)


def _numeric_verdict(claims: Sequence[RoleClaim]) -> Tuple[str, str]:
    """Whether the numbers agree, and how to say so."""
    numbered = [c for c in claims if c.value is not None]
    if len(numbered) < 2:
        return "", ""
    values = [c.value for c in numbered]
    low, high = min(values), max(values)
    # Relative to the smaller magnitude: 1ms vs 2ms is a disagreement, and
    # 1000ms vs 1001ms is not, and an absolute threshold cannot say both.
    base = max(abs(low), 1e-9)
    drift = abs(high - low) / base
    tolerance = _tolerance_for(numbered)
    said = ", ".join(f"{c.role} {c.value:g}" for c in numbered)
    # A gap that survives even a generous window is a real disagreement, and
    # the more generous the window the more confident that verdict is. Checked
    # BEFORE the ceiling: a 400x difference between two noisy measurements is
    # not "unresolvable", it is the clearest result the swarm can produce, and
    # letting the ceiling swallow it would discard the signal along with the
    # noise.
    if drift > tolerance:
        return "contested", f"{said} — {drift:.0%} apart, beyond {tolerance:.0%}"
    if tolerance > MAX_USEFUL_TOLERANCE:
        # Inside the window -- but the window is wide enough to swallow the
        # question. Report the width rather than a verdict it cannot support.
        return (
            "inconclusive",
            f"{said} — {drift:.0%} apart, but these measurements report "
            f"{tolerance:.0%} of their own variation, too noisy to tell "
            "agreement from coincidence",
        )
    return "corroborated", f"{said} (within {tolerance:.0%})"


def group_claims(
    findings_by_role: Mapping[str, Sequence[Mapping[str, Any]]]
) -> List[ConsensusGroup]:
    """Group every role's findings by what kind of thing they are about.

    `findings_by_role` maps a role name to that role's findings. Roles rather
    than jobs, because two jobs running the same role are one opinion: the
    independence that makes agreement meaningful comes from the roles being
    different, not from there being several processes.
    """
    buckets: Dict[Tuple[str, str], List[RoleClaim]] = {}
    for role, findings in (findings_by_role or {}).items():
        role_token = str(role or "").strip().lower()
        if not role_token:
            continue
        for finding in findings or []:
            if not isinstance(finding, Mapping):
                continue
            finding_type = str(finding.get("type") or "").strip()
            if not finding_type:
                continue
            subject = subject_key(finding)
            value: Optional[float] = None
            used_field = ""
            for name in _COMPARABLE_FIELDS:
                found = _dig(finding, name)
                if found is not None:
                    value, used_field = found, name
                    break
            spread: Optional[float] = None
            for name in _SPREAD_FIELDS:
                found = _dig(finding, name)
                if found is not None and found > 0:
                    spread = found
                    break
            buckets.setdefault((finding_type, subject), []).append(
                RoleClaim(
                    role=role_token,
                    value=value,
                    field_name=used_field,
                    spread=spread,
                    title=str(finding.get("title") or "").strip()[:200],
                )
            )

    groups: List[ConsensusGroup] = []
    for (finding_type, subject), claims in buckets.items():
        distinct_roles = {c.role for c in claims}
        if len(distinct_roles) < 2:
            # One agent producing a finding twice is one agent. Saying so is
            # the difference between corroboration and an echo.
            groups.append(
                ConsensusGroup(
                    finding_type=finding_type,
                    subject=subject,
                    claims=list(claims),
                    verdict="uncorroborated",
                    detail=f"only {sorted(distinct_roles)[0]} produced this",
                )
            )
            continue

        verdict, detail = _numeric_verdict(claims)
        if not verdict:
            # No comparable number on either side. Two roles independently
            # producing the same KIND of evidence about the same subject is
            # still corroboration -- weaker than agreeing numbers, and it must
            # not be dressed up as more.
            verdict = "corroborated"
            detail = (
                f"{len(distinct_roles)} roles produced this; no comparable "
                "value to check"
            )
        drift, tolerance = _resolution(claims)
        groups.append(
            ConsensusGroup(
                finding_type=finding_type,
                subject=subject,
                claims=list(claims),
                verdict=verdict,
                detail=detail,
                drift=drift,
                tolerance=tolerance,
            )
        )

    # Contested first: a disagreement is the thing a person must look at, and
    # burying it under agreements is how a swarm's one useful output is missed.
    order = {
        "contested": 0,
        "inconclusive": 1,
        "corroborated": 2,
        "uncorroborated": 3,
    }
    groups.sort(key=lambda g: (order.get(g.verdict, 4), g.finding_type, g.subject))
    return groups


def summarize(groups: Iterable[ConsensusGroup]) -> Dict[str, Any]:
    """The swarm's result, as a reader needs it."""
    rows = list(groups)
    corroborated = [g for g in rows if g.verdict == "corroborated"]
    contested = [g for g in rows if g.verdict == "contested"]
    inconclusive = [g for g in rows if g.verdict == "inconclusive"]
    uncorroborated = [g for g in rows if g.verdict == "uncorroborated"]

    checkable = len(corroborated) + len(contested)
    return {
        "corroborated": [g.as_dict() for g in corroborated],
        "contested": [g.as_dict() for g in contested],
        "inconclusive": [g.as_dict() for g in inconclusive],
        "uncorroborated": [g.as_dict() for g in uncorroborated],
        "corroborated_count": len(corroborated),
        "contested_count": len(contested),
        "inconclusive_count": len(inconclusive),
        "uncorroborated_count": len(uncorroborated),
        # Of the things more than one role spoke to, how many they agreed on.
        # Deliberately not "agreement over everything": a finding only one role
        # produced is not evidence of agreement OR disagreement, and counting
        # it either way would make a swarm of silent roles look unanimous.
        #
        # An `inconclusive` group is out of the denominator for the same
        # reason. Two roles measured the same thing, but through an instrument
        # too imprecise to say whether the answers match -- scoring that as
        # agreement is what produced "agreement 1.0" for a pair of numbers 33%
        # apart, and scoring it as disagreement would be just as invented.
        "agreement": (len(corroborated) / checkable) if checkable else None,
    }
