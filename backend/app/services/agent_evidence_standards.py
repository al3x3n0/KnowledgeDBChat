"""What a plan a run proposed for itself must demand of its evidence.

A contract counts findings. Counting cannot tell a measurement from a number:
the first campaign that approved its own plan ran a benchmark stage whose
contract asked for one `benchmark_measurement`, got one trial of one program
with no baseline beside it, met its contract, and the campaign concluded --
correctly -- that its findings established nothing. The plan was within its
limits in size and cost, which is all self-approval checked, and its evidence
standard was whatever the drafting model happened to write.

So a self-proposed plan has the standard written into it before it is
checked, priced and shown for approval. Two rules, each one the validity
block already enforces (`agent_measurement_validity`), so nothing new decides
at run time:

  A BENCHMARK REPORTS ITS SPREAD. `require_uncertainty` on
  `benchmark_measurement`: one trial is not a measurement, and the tool
  already says so in a warning nobody was obliged to read.

  A BENCHMARK HAS A BASELINE BESIDE IT. `distinct_subjects` of
  `BASELINE_MIN_SUBJECTS` on `benchmark_measurement`: a time for the
  candidate alone says nothing about whether to adopt it. The second
  self-approved campaign timed fastmod and never plain `%`, and concluded,
  correctly, that it could not say whether fastmod was worth adopting. Timing
  the candidate twice is not a baseline, which is why this counts subjects
  rather than findings.

  AN IMPLEMENTATION IS CHECKED AGAINST AN ORACLE. A bound of
  `generated_cases >= ORACLE_MIN_CASES` on `implementation_verified`, met only
  by `check_implementation` with a reference and a generator. Hand-written
  cases are chosen by the author of the code, and a live stage's harness
  printed PASS whatever its input.

A rule the plan already states is left as its author wrote it, unless it is
weaker (a lower bound is raised). Person-authored pipelines are not touched:
their author chose their standard, and a person approves them.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Tuple

from app.services.agent_pipeline_spec import _required_types

#: Generated inputs an oracle check must have compared.
ORACLE_MIN_CASES = 50

#: Different subjects a benchmark stage must have timed: the candidate and
#: what it would replace.
BASELINE_MIN_SUBJECTS = 2

#: What the drafter is told, so the stages it writes can meet the standard
#: rather than discover it at run time.
DRAFTER_NOTE = (
    "Evidence standard for this plan (it will be enforced): a stage that "
    "requires benchmark_measurement must report a spread -- several trials, "
    "never one -- and must time the baseline (the plain implementation the "
    "candidate would replace) beside the candidate, each under its own label, "
    "so say so in its goal; a stage "
    "that requires implementation_verified must check the code with "
    "check_implementation against a reference implementation (the obvious, "
    f"slow, correct one) and a generator of at least {ORACLE_MIN_CASES} "
    "inputs, so say so in its goal."
)


def _validity(contract: Dict[str, Any]) -> Dict[str, Any]:
    validity = contract.get("validity")
    if not isinstance(validity, dict):
        validity = {}
        contract["validity"] = validity
    return validity


def apply(spec: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """The spec with the standard written into its contracts, and what changed."""
    raised = copy.deepcopy(spec) if isinstance(spec, dict) else {}
    notes: List[str] = []
    for stage in raised.get("stages") or []:
        if not isinstance(stage, dict):
            continue
        contract = stage.get("contract")
        if not isinstance(contract, dict):
            continue
        required = set(_required_types(contract))
        name = str(stage.get("id") or stage.get("name") or "stage")

        if "benchmark_measurement" in required:
            validity = _validity(contract)
            wanted = validity.get("require_uncertainty")
            wanted = list(wanted) if isinstance(wanted, (list, tuple)) else []
            if "benchmark_measurement" not in wanted:
                validity["require_uncertainty"] = wanted + ["benchmark_measurement"]
                notes.append(f"{name}: benchmark_measurement must report a spread")
            distinct = validity.get("distinct_subjects")
            if not isinstance(distinct, dict):
                distinct = {}
                validity["distinct_subjects"] = distinct
            try:
                current = int(distinct.get("benchmark_measurement") or 0)
            except (TypeError, ValueError):
                current = 0
            if current < BASELINE_MIN_SUBJECTS:
                distinct["benchmark_measurement"] = BASELINE_MIN_SUBJECTS
                notes.append(
                    f"{name}: benchmark_measurement must time a baseline "
                    "beside the candidate"
                )

        if "implementation_verified" in required:
            validity = _validity(contract)
            bounds = validity.get("bounds")
            if not isinstance(bounds, dict):
                bounds = {}
                validity["bounds"] = bounds
            rule = bounds.get("implementation_verified")
            current = rule.get("min") if isinstance(rule, dict) else None
            same_field = (
                isinstance(rule, dict) and rule.get("field") == "generated_cases"
            )
            try:
                strong_enough = same_field and float(current) >= ORACLE_MIN_CASES
            except (TypeError, ValueError):
                strong_enough = False
            if not strong_enough and (rule is None or same_field):
                bounds["implementation_verified"] = {
                    **(rule if isinstance(rule, dict) else {}),
                    "field": "generated_cases",
                    "min": ORACLE_MIN_CASES,
                }
                notes.append(
                    f"{name}: implementation_verified must compare against an "
                    f"oracle on at least {ORACLE_MIN_CASES} generated inputs"
                )
            elif not same_field:
                # A contract holds one bound per type; the plan's own stays.
                notes.append(
                    f"{name}: kept its own bound on implementation_verified."
                    f"{rule.get('field')}, so no oracle is required of it"
                )
    return raised, notes
