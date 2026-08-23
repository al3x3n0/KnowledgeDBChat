"""Deterministic soundness checks a goal contract can require of a run.

Goal contracts count outputs: two findings of this type, three artifacts of
that one. Counting cannot tell whether the outputs mean anything, and in
practice that is where the failures are. A throughput benchmark with fewer
independent chains than the machine keeps in flight returns exactly
latency/ways -- a property of the harness, not of the processor -- and four of
seven such results would have satisfied any contract expressible today. A
frequency measurement that implies a dependent integer add took 0.83 cycles is
impossible by construction and still counts as a measurement.

So these are the checks that separate "the job produced the required number of
things" from "the things are worth having". They are deliberately general:
each reads the run's own state, needs no database and no model call, and
returns the same verdict every time it is run on the same state.

The vocabulary a contract may use, under `contract["validity"]`:

    predictions_measured    every prediction recorded in this run was settled
                            with a measurement
    require_uncertainty     findings of these types must carry a spread, error
                            bar or sample count -- a bare number is not a
                            measurement
    bounds                  per finding type, a numeric field and the range it
                            must fall in, for values that are impossible
                            rather than merely surprising
    records_method          the run must record at least one method, so what
                            it learned about *how* to investigate outlives it
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

# Fields a finding may use to express how uncertain a measurement is. Any one
# of them satisfies `require_uncertainty`: the point is that the run reports
# its own dispersion somehow, not that it picks a particular spelling.
UNCERTAINTY_FIELDS = (
    "spread",
    # What benchmark_c_snippet actually names its dispersion. Omitting it made
    # a contract asking for error bars unsatisfiable by the one tool in this
    # codebase that reports them -- the two halves were built apart and did
    # not meet.
    "trial_spread",
    "relative_spread",
    "std_dev",
    "stddev",
    "error_bar",
    "uncertainty",
    "confidence_interval",
    "samples",
    "sample_count",
    "runs",
    "trials",
)


def _findings(state: Mapping[str, Any]) -> List[Dict[str, Any]]:
    raw = state.get("findings")
    return [f for f in raw if isinstance(f, dict)] if isinstance(raw, list) else []


def _actions(state: Mapping[str, Any]) -> List[Dict[str, Any]]:
    raw = state.get("actions_taken")
    return [a for a in raw if isinstance(a, dict)] if isinstance(raw, list) else []


def _as_number(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _find_number(finding: Mapping[str, Any], field: str) -> Optional[float]:
    """Look for a named numeric field on a finding or one level inside it.

    Tools put their numbers in different places -- some at the top level, some
    under `data` or `details` -- and a check that only looked at the top level
    would silently pass every finding whose value it could not see.
    """
    direct = _as_number(finding.get(field))
    if direct is not None:
        return direct
    for container in ("data", "details", "metrics", "value"):
        nested = finding.get(container)
        if isinstance(nested, Mapping):
            found = _as_number(nested.get(field))
            if found is not None:
                return found
    return None


SAMPLE_LIST_FIELDS = ("all_ms", "samples", "timings", "measurements")


def _has_uncertainty(finding: Mapping[str, Any]) -> bool:
    for container in (finding, finding.get("data"), finding.get("details")):
        if not isinstance(container, Mapping):
            continue
        for field in SAMPLE_LIST_FIELDS:
            value = container.get(field)
            # More than one trial is itself a statement about dispersion; a
            # single one says nothing and must not pass.
            if isinstance(value, (list, tuple)) and len(value) > 1:
                return True
    for field in UNCERTAINTY_FIELDS:
        if _find_number(finding, field) is not None:
            return True
        # A textual range ("3.2-3.9", "+/-12%") is a spread too; only an
        # entirely absent field means the run never reported one.
        for container in (finding, finding.get("data"), finding.get("details")):
            if (
                isinstance(container, Mapping)
                and str(container.get(field) or "").strip()
            ):
                return True
    return False


def unsettled_predictions(state: Mapping[str, Any]) -> List[str]:
    """Prediction ids recorded in this run that no measurement ever settled.

    Read from the run's own actions rather than the calibration table, because
    the contract is evaluated synchronously and because the question is about
    this run: a prediction settled by some other job is not this job's evidence.
    """
    recorded: List[str] = []
    settled: set = set()
    for entry in _actions(state):
        action = entry.get("action") if isinstance(entry.get("action"), dict) else {}
        result = entry.get("result") if isinstance(entry.get("result"), dict) else {}
        tool = str(action.get("tool") or "").strip()
        params = action.get("params") if isinstance(action.get("params"), dict) else {}

        if tool == "record_prediction" and bool(result.get("success")):
            data = result.get("data") if isinstance(result.get("data"), dict) else {}
            prediction_id = str(data.get("prediction_id") or "").strip()
            if prediction_id:
                recorded.append(prediction_id)
        elif tool == "record_measurement" and bool(result.get("success")):
            prediction_id = str(params.get("prediction_id") or "").strip()
            if prediction_id:
                settled.add(prediction_id)

    return [pid for pid in recorded if pid not in settled]


def evaluate(contract: Mapping[str, Any], state: Mapping[str, Any]) -> Dict[str, Any]:
    """Check every validity requirement a contract declares.

    Returns the unmet requirement labels, in the same shape the contract
    evaluator uses for its counting requirements, plus the detail an operator
    needs to see why a run was held back.
    """
    spec = contract.get("validity")
    if not isinstance(spec, Mapping) or not spec:
        return {"declared": False, "missing": [], "details": {}}

    missing: List[str] = []
    details: Dict[str, Any] = {}
    findings = _findings(state)

    if bool(spec.get("predictions_measured")):
        unsettled = unsettled_predictions(state)
        if unsettled:
            missing.append("validity:predictions_measured")
            details["unsettled_predictions"] = unsettled[:10]

    if bool(spec.get("instruments_verified")):
        # Checks the instrument rather than the result. Every other requirement
        # in this module assumes the tool that produced the numbers was
        # working, and on this project that assumption failed twice in ways no
        # count, bound or uncertainty check could see.
        from app.services import agent_tool_controls

        unverified = agent_tool_controls.unverified_instruments(state)
        if unverified:
            missing.append("validity:instruments_verified")
            details["unverified_instruments"] = [
                {"tool": u.get("tool"), "reason": u.get("reason")}
                for u in unverified[:5]
            ]

    if bool(spec.get("records_method")):
        recorded = [
            f for f in findings if str(f.get("type") or "").strip() == "method_recorded"
        ]
        if not recorded:
            missing.append("validity:records_method")

    required_uncertainty = spec.get("require_uncertainty")
    if isinstance(required_uncertainty, (list, tuple)):
        for type_name in [
            str(x).strip() for x in required_uncertainty if str(x).strip()
        ]:
            matching = [
                f for f in findings if str(f.get("type") or "").strip() == type_name
            ]
            bare = [f for f in matching if not _has_uncertainty(f)]
            # A type that never appeared is the counting requirements' problem,
            # not this check's: reporting it here would blame the wrong thing.
            if matching and bare:
                missing.append(f"validity:uncertainty:{type_name}")
                details.setdefault("without_uncertainty", {})[type_name] = len(bare)

    bounds = spec.get("bounds")
    if isinstance(bounds, Mapping):
        for type_name, rule in bounds.items():
            if not isinstance(rule, Mapping):
                continue
            field = str(rule.get("field") or "").strip()
            if not field:
                continue
            low, high = _as_number(rule.get("min")), _as_number(rule.get("max"))
            offenders: List[Any] = []
            for finding in findings:
                if str(finding.get("type") or "").strip() != str(type_name).strip():
                    continue
                value = _find_number(finding, field)
                if value is None:
                    continue
                if (low is not None and value < low) or (
                    high is not None and value > high
                ):
                    offenders.append(value)
            if offenders:
                missing.append(f"validity:bounds:{type_name}")
                details.setdefault("out_of_bounds", {})[str(type_name)] = {
                    "field": field,
                    "min": low,
                    "max": high,
                    "values": offenders[:5],
                }

    return {"declared": True, "missing": missing, "details": details}


def _explain_instruments(details: Mapping[str, Any]) -> List[str]:
    lines = []
    for entry in details.get("unverified_instruments") or []:
        tool = entry.get("tool")
        lines.append(
            f"{tool} produced numbers this run without a passing control on "
            f"both sides of them. {entry.get('reason')} "
            f"Run the control for {tool} before your first measurement and "
            "again after your last, and treat anything measured in between as "
            "unusable until both pass."
        )
    return lines


def explain(missing: Sequence[str], details: Mapping[str, Any]) -> List[str]:
    """Turn unmet validity labels into instructions that name the remedy.

    `validity:predictions_measured` tells a model nothing it can act on. The
    ids of the predictions it left open, and the tool that settles them, do.
    """
    lines: List[str] = []
    for label in missing:
        text = str(label)
        if text == "validity:instruments_verified":
            lines.extend(_explain_instruments(details))
        elif text == "validity:predictions_measured":
            open_ids = details.get("unsettled_predictions")
            listed = ", ".join(str(x) for x in open_ids[:3]) if open_ids else ""
            lines.append(
                "Predictions recorded in this run were never settled"
                + (f" ({listed})" if listed else "")
                + ". Run the referee and call record_measurement with each "
                "prediction_id; an unsettled prediction scores nothing."
            )
        elif text == "validity:records_method":
            lines.append(
                "This run has not recorded a method. Call record_method with "
                "what you learned about how to do this work -- the procedure, "
                "what it prevents, and the findings that establish it -- so the "
                "next job inherits it instead of rediscovering it."
            )
        elif text.startswith("validity:uncertainty:"):
            type_name = text.split(":", 2)[2]
            lines.append(
                f"Findings of type {type_name} report a bare number. Repeat the "
                "measurement and include the spread or sample count: a single "
                "sample cannot show whether a difference is real."
            )
        elif text.startswith("validity:bounds:"):
            type_name = text.split(":", 2)[2]
            rule = (details.get("out_of_bounds") or {}).get(type_name) or {}
            values = ", ".join(str(v) for v in (rule.get("values") or [])[:3])
            lines.append(
                f"Findings of type {type_name} carry impossible values for "
                f"{rule.get('field')}"
                + (f" ({values})" if values else "")
                + f"; the physical range is [{rule.get('min')}, {rule.get('max')}]. "
                "Something in the measurement is wrong -- diagnose it rather "
                "than recording the number."
            )
    return lines


def describe(spec: Any) -> Sequence[str]:
    """Human-readable lines for a validity block, for digests and prompts."""
    if not isinstance(spec, Mapping) or not spec:
        return []
    lines: List[str] = []
    if bool(spec.get("predictions_measured")):
        lines.append("every prediction must be settled with a measurement")
    if bool(spec.get("instruments_verified")):
        from app.services import agent_tool_controls

        lines.extend(agent_tool_controls.describe())
    if bool(spec.get("records_method")):
        lines.append(
            "the run must record at least one method (record_method): the "
            "procedure that worked, what it prevents, and its evidence"
        )
    required = spec.get("require_uncertainty")
    if isinstance(required, (list, tuple)) and required:
        lines.append(
            "these findings must report a spread or sample count: "
            + ", ".join(str(x) for x in required)
        )
    bounds = spec.get("bounds")
    if isinstance(bounds, Mapping):
        for type_name, rule in bounds.items():
            if not isinstance(rule, Mapping):
                continue
            lines.append(
                f"{type_name}.{rule.get('field')} must lie in "
                f"[{rule.get('min')}, {rule.get('max')}]"
            )
    return lines
