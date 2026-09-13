"""A run's evidence, as something a person can judge.

Findings reach the UI today as `findings_count` -- a number. The chain this
whole system exists to produce (`algorithm_spec` -> `implementation_verified`
-> `benchmark_measurement` -> `reproduction_verdict`) is invisible, so you can
see THAT a stage produced four findings and not whether any of them is worth
believing. For a research pipeline that is the difference between a run you can
supervise and one you can only watch finish.

What this assembles, and why each part is here rather than left to the reader:

* **The findings themselves**, with their numbers. A `benchmark_measurement`
  carries `fastest_ms`, the language and flags it was built with, and the load
  on the machine while it ran. All of that decides whether the number means
  anything, and none of it was reachable.
* **Whether each carries a spread.** Asked through
  `agent_measurement_validity.has_uncertainty`, never reimplemented: two
  definitions of "reports error bars" would disagree the first time a tool
  named its dispersion something new, which has already happened once here.
* **The requirement each satisfies.** A contract asks for finding types; the
  useful view is per requirement, because the question is never "how many
  findings" but "is the thing I asked for actually there".
* **Perishability**, from the evidence map. Evidence a later change invalidates
  reads very differently from evidence that keeps.

Read-only, and deliberately so. It computes no verdicts of its own: the
contract result already in `results` is the authority on satisfaction, and this
shows what that verdict was reached from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from app.services import agent_evidence_map, agent_measurement_validity

#: Fields that identify or narrate a finding rather than measure anything.
#: Excluded from the value list so the numbers that decide whether to believe
#: it are not buried under provenance.
_BOILERPLATE = frozenset(
    {
        "id",
        "type",
        "title",
        "source",
        "source_id",
        "score",
        "note",
        "description",
        "summary",
        "content",
        "text",
        "url",
        "created_at",
        "timestamp",
    }
)

#: Shown first when present, in this order. Everything else follows
#: alphabetically. These are the fields a reader reaches for first: what was
#: measured, how much of it, and under what conditions.
_HEADLINE = (
    "subject",
    "verdict",
    "passed",
    "failed",
    "fastest_ms",
    "speedup",
    "ratio",
    "cycles",
    "language",
    "flags",
    "trial_spread",
    "load_per_cpu",
    "measurement_environment",
)

#: How many values to show for one finding. A finding with forty fields is a
#: tool result, and dumping it here would bury the four that matter.
_MAX_VALUES = 10


def _render(value: Any) -> Optional[str]:
    """One field as a short string, or None if it is not worth a row."""
    if value is None or isinstance(value, (dict,)):
        return None
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        # A list of trials is a sample count, not forty numbers. The trials
        # themselves are what `has_uncertainty` reads; the reader needs to know
        # there was more than one.
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value):
            return f"{len(value)} values"
        return f"{len(value)} items"
    if isinstance(value, float):
        return f"{value:.4g}"
    rendered = str(value).strip()
    if not rendered:
        return None
    return rendered[:200]


@dataclass(frozen=True)
class EvidenceValue:
    label: str
    value: str

    def as_dict(self) -> Dict[str, str]:
        return {"label": self.label, "value": self.value}


@dataclass(frozen=True)
class EvidenceItem:
    """One finding, as a reader needs to judge it."""

    index: int
    type: str
    title: str
    values: List[EvidenceValue] = field(default_factory=list)
    #: Whether it reports a spread. Asked of the validity service, not decided
    #: here.
    has_uncertainty: bool = False
    #: quiet / busy / saturated, for findings whose tool recorded it. A
    #: wall-clock number taken on a saturated host is not a measurement.
    measurement_environment: str = ""
    #: What the producing tool warned about this measurement, verbatim.
    warning: str = ""
    #: Evidence a later change invalidates, so it is never inherited from an
    #: upstream stage.
    perishable: bool = False
    #: A person rejected this result, and why. Advisory: nothing downstream is
    #: invalidated by it and the contract's verdict is unchanged. What it does
    #: is travel -- into a restart of the stage as an operator correction, and
    #: into the prompt of any run that would cite this evidence again.
    disputed: bool = False
    dispute_reason: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "type": self.type,
            "title": self.title,
            "values": [v.as_dict() for v in self.values],
            "has_uncertainty": self.has_uncertainty,
            "measurement_environment": self.measurement_environment,
            "warning": self.warning,
            "perishable": self.perishable,
            "disputed": self.disputed,
            "dispute_reason": self.dispute_reason,
        }


@dataclass(frozen=True)
class RequirementView:
    """One finding type the contract asked for, and what arrived."""

    finding_type: str
    #: Indices into the evidence list. Indices rather than copies so a reader
    #: sees one finding once, however many requirements it answers.
    satisfied_by: List[int] = field(default_factory=list)
    #: The contract asked for a spread on this type specifically.
    uncertainty_required: bool = False
    #: Requirements that arrived but without the spread they were required to
    #: carry. Named per requirement because that is where the author can act.
    missing_uncertainty: List[int] = field(default_factory=list)

    @property
    def satisfied(self) -> bool:
        return bool(self.satisfied_by)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "finding_type": self.finding_type,
            "satisfied": self.satisfied,
            "satisfied_by": list(self.satisfied_by),
            "uncertainty_required": self.uncertainty_required,
            "missing_uncertainty": list(self.missing_uncertainty),
        }


#: How many findings nobody asked for to render. Beyond this the page is
#: showing a sample, and says so.
_MAX_UNREQUESTED = 50

#: How many fields to lift out of one nested container. A tool result nested
#: whole would bury the finding it belongs to.
_MAX_NESTED = 6


def _values_of(finding: Mapping[str, Any]) -> List[EvidenceValue]:
    """The fields worth showing, headline ones first.

    Descends one level into nested objects, because that is where tools
    actually put their numbers: a `codegen_measurement` keeps `vector_ops` and
    `conditional_branches` under `codegen`, and a reader shown only the top
    level sees a subject and some flags -- the measurement itself invisible.

    The same trap `agent_measurement_validity._find_number` documents and
    guards against. Measured here on a real run: two of three measurement
    findings rendered no numbers at all.
    """
    candidates: Dict[str, str] = {}
    for key, raw in finding.items():
        name = str(key)
        if name in _BOILERPLATE or name.startswith("_"):
            continue
        if isinstance(raw, Mapping):
            # One level, labelled by its container so `codegen.vector_ops`
            # stays distinguishable from a top-level `vector_ops`.
            taken = 0
            for sub_key, sub_raw in raw.items():
                if taken >= _MAX_NESTED:
                    break
                sub_name = str(sub_key)
                if sub_name in _BOILERPLATE or sub_name.startswith("_"):
                    continue
                rendered = _render(sub_raw)
                if rendered is not None:
                    candidates[f"{name}.{sub_name}"] = rendered
                    taken += 1
            continue
        rendered = _render(raw)
        if rendered is not None:
            candidates[name] = rendered

    ordered: List[EvidenceValue] = []
    for name in _HEADLINE:
        if name in candidates:
            ordered.append(EvidenceValue(label=name, value=candidates.pop(name)))
    for name in sorted(candidates):
        ordered.append(EvidenceValue(label=name, value=candidates[name]))
    return ordered[:_MAX_VALUES]


def _contract_of(results: Mapping[str, Any]) -> Dict[str, Any]:
    """The contract as authored, out of the result the run recorded.

    Read from the run rather than from the job config: the config can be edited
    after the fact, and the question here is what this run was judged against.
    """
    block = results.get("goal_contract")
    if not isinstance(block, Mapping):
        return {}
    inner = block.get("contract")
    return dict(inner) if isinstance(inner, Mapping) else {}


def _required_types(contract: Mapping[str, Any]) -> List[str]:
    raw = contract.get("required_finding_types")
    if isinstance(raw, (list, tuple)):
        return [str(t).strip() for t in raw if str(t).strip()]
    return []


def _uncertainty_types(contract: Mapping[str, Any]) -> List[str]:
    validity = contract.get("validity")
    if not isinstance(validity, Mapping):
        return []
    raw = validity.get("require_uncertainty")
    if isinstance(raw, (list, tuple)):
        return [str(t).strip() for t in raw if str(t).strip()]
    return []


def build(job: Any, disputes: Optional[Mapping[int, str]] = None) -> Dict[str, Any]:
    """The evidence a run produced, and what it was asked for.

    `disputes` maps a finding index to the reason a person rejected it. Passed
    in rather than read here so this stays a pure function of a run and a
    verdict -- and because the disputes live in the retraction table, which is
    the one authority on what is no longer believed.
    """
    rejected: Mapping[int, str] = disputes or {}
    results = job.results if isinstance(getattr(job, "results", None), dict) else {}
    raw_findings = results.get("findings")
    findings = (
        [f for f in raw_findings if isinstance(f, dict)]
        if isinstance(raw_findings, list)
        else []
    )

    contract = _contract_of(results)
    required_names = set(_required_types(contract))

    # Which findings to render at all. Everything the contract asked for, plus
    # a bounded sample of everything else -- that ratio is the whole reason for
    # a cap: this database holds 4,247 `document` findings against 97
    # benchmarks, so an uncapped response is almost entirely evidence nobody
    # asked for, and the run that needs the cap is the one already slowest to
    # serve.
    #
    # Required evidence is never dropped. It is what the page is for, and a
    # truncated requirement would read as a requirement nothing answered --
    # the one message this view must never get wrong.
    rendered_indices: List[int] = []
    unrequested_total = 0
    for index, finding in enumerate(findings):
        finding_type = str(finding.get("type") or "").strip()
        if finding_type in required_names:
            rendered_indices.append(index)
            continue
        unrequested_total += 1
        if unrequested_total <= _MAX_UNREQUESTED or index in rejected:
            # A rejected finding is always shown, wherever it falls: someone
            # went to the trouble of rejecting it, and hiding it behind a cap
            # would make the rejection unreachable.
            rendered_indices.append(index)

    items: List[EvidenceItem] = []
    for index in rendered_indices:
        finding = findings[index]
        finding_type = str(finding.get("type") or "").strip()
        items.append(
            EvidenceItem(
                index=index,
                type=finding_type,
                title=str(finding.get("title") or finding.get("subject") or "").strip(),
                values=_values_of(finding),
                has_uncertainty=agent_measurement_validity.has_uncertainty(finding),
                measurement_environment=str(
                    finding.get("measurement_environment") or ""
                ).strip(),
                warning=str(finding.get("measurement_warning") or "").strip(),
                perishable=(
                    agent_evidence_map.is_perishable(finding_type)
                    if finding_type
                    else False
                ),
                disputed=index in rejected,
                dispute_reason=str(rejected.get(index) or ""),
            )
        )

    needs_spread = set(_uncertainty_types(contract))
    by_type: Dict[str, List[int]] = {}
    for item in items:
        if item.type:
            by_type.setdefault(item.type, []).append(item.index)

    by_index: Dict[int, EvidenceItem] = {item.index: item for item in items}
    requirements: List[RequirementView] = []
    for finding_type in _required_types(contract):
        matched = by_type.get(finding_type, [])
        wants_spread = finding_type in needs_spread
        requirements.append(
            RequirementView(
                finding_type=finding_type,
                satisfied_by=matched,
                uncertainty_required=wants_spread,
                missing_uncertainty=(
                    [i for i in matched if not by_index[i].has_uncertainty]
                    if wants_spread
                    else []
                ),
            )
        )

    # Findings of a type nothing asked for. Not a fault -- a run records what
    # it learns -- but they are the difference between the evidence that was
    # required and the evidence that exists, and a reader looking for one
    # should not have to wade through the other.
    unrequested = [item.index for item in items if item.type not in required_names]

    contract_block = (
        results.get("goal_contract")
        if isinstance(results.get("goal_contract"), Mapping)
        else {}
    )

    return {
        "job_id": str(getattr(job, "id", "") or ""),
        "evidence": [item.as_dict() for item in items],
        "requirements": [r.as_dict() for r in requirements],
        "unrequested": unrequested,
        # How many there are in total, so the page can say "showing 50 of
        # 4,247" rather than quietly presenting a sample as the whole.
        "unrequested_total": unrequested_total,
        "contract_enabled": bool(contract_block.get("enabled", False)),
        # A run resting on rejected evidence must not read as clean, even
        # though nothing about its contract changed. Advisory does not mean
        # invisible.
        "disputed_count": sum(1 for item in items if item.disputed),
        # The contract's own verdict, not one computed here. This view shows
        # what that verdict was reached from; it does not second-guess it.
        "contract_satisfied": bool(contract_block.get("satisfied", False)),
        "missing": [
            str(m)[:200]
            for m in (
                contract_block.get("missing")
                if isinstance(contract_block.get("missing"), list)
                else []
            )
        ][:20],
        # Predictions the run made and never settled with a measurement. The
        # question `predictions_measured` asks, shown whether or not the
        # contract asked for it: an unsettled prediction is a claim nobody
        # checked.
        "unsettled_predictions": agent_measurement_validity.unsettled_predictions(
            results
        )[:20],
    }
