"""A run's evidence, as something a person can judge.

Findings reached the UI as `findings_count` -- a number. You could see THAT a
stage produced four findings and not whether any was worth believing, which for
a research pipeline is the difference between a run you can supervise and one
you can only watch finish.

So these tests are about the distinctions a count destroys: a measurement with
error bars against one without, a number taken on a quiet machine against one
taken on a saturated host, evidence the contract asked for against evidence it
merely happens to have.
"""

import pytest

from app.services import agent_evidence_view as view

pytestmark = pytest.mark.unit


class _Job:
    """The two attributes the view reads."""

    def __init__(self, results):
        self.id = "job-1"
        self.results = results


def _benchmark(**over):
    finding = {
        "type": "benchmark_measurement",
        "title": "kernel @ c -O3",
        "fastest_ms": 12.5,
        "all_ms": [12.5, 13.1, 12.9],
        "language": "c",
        "flags": "-O3",
        "load_per_cpu": 0.4,
        "measurement_environment": "quiet",
        "trial_spread": 0.05,
    }
    finding.update(over)
    return finding


def _results(findings, **contract):
    block = {"enabled": True, "satisfied": False, "missing": [], "contract": contract}
    return {"findings": findings, "goal_contract": block}


class TestWhatACountDestroys:
    def test_a_measurement_reports_whether_it_carries_a_spread(self):
        built = view.build(
            _Job(_results([_benchmark(), _benchmark(all_ms=[12.5], trial_spread=None)]))
        )

        assert built["evidence"][0]["has_uncertainty"] is True
        assert (
            built["evidence"][1]["has_uncertainty"] is False
        ), "one trial and no spread field says nothing about dispersion"

    def test_the_machine_a_number_was_taken_on_travels_with_it(self):
        """A wall-clock number taken on a saturated host is not a
        measurement, and only the finding knows which it was."""
        built = view.build(
            _Job(_results([_benchmark(measurement_environment="saturated")]))
        )

        assert built["evidence"][0]["measurement_environment"] == "saturated"

    def test_a_tools_warning_is_carried_verbatim(self):
        built = view.build(
            _Job(
                _results(
                    [_benchmark(measurement_warning="host was busy; treat as advisory")]
                )
            )
        )

        assert "advisory" in built["evidence"][0]["warning"]

    def test_perishable_evidence_says_so(self):
        """It is never inherited from an upstream stage, so a looping stage
        cannot keep a verdict it earned before its own edit."""
        built = view.build(_Job(_results([{"type": "command_result", "title": "ls"}])))

        assert built["evidence"][0]["perishable"] is True

    def test_the_numbers_that_decide_belief_are_shown_first(self):
        built = view.build(_Job(_results([_benchmark()])))
        labels = [v["label"] for v in built["evidence"][0]["values"]]

        assert labels[0] == "fastest_ms", "the measurement leads"
        assert "measurement_environment" in labels
        assert "title" not in labels, "provenance is not a measurement"

    def test_a_list_of_trials_is_shown_as_a_count_not_forty_numbers(self):
        built = view.build(_Job(_results([_benchmark(all_ms=[1.0] * 40)])))
        values = {v["label"]: v["value"] for v in built["evidence"][0]["values"]}

        assert values["all_ms"] == "40 values"


class TestTheContractIsTheQuestion:
    def test_each_requirement_names_the_findings_that_answer_it(self):
        built = view.build(
            _Job(
                _results(
                    [
                        {"type": "implementation_verified", "passed": 4},
                        _benchmark(),
                    ],
                    required_finding_types=[
                        "implementation_verified",
                        "benchmark_measurement",
                    ],
                )
            )
        )

        by_type = {r["finding_type"]: r for r in built["requirements"]}
        assert by_type["implementation_verified"]["satisfied_by"] == [0]
        assert by_type["benchmark_measurement"]["satisfied_by"] == [1]

    def test_a_requirement_nothing_answered_is_shown_unsatisfied(self):
        """The question is never "how many findings" but "is the thing I asked
        for actually there"."""
        built = view.build(
            _Job(
                _results(
                    [_benchmark()],
                    required_finding_types=[
                        "benchmark_measurement",
                        "reproduction_verdict",
                    ],
                )
            )
        )

        by_type = {r["finding_type"]: r for r in built["requirements"]}
        assert by_type["reproduction_verdict"]["satisfied"] is False
        assert by_type["reproduction_verdict"]["satisfied_by"] == []

    def test_a_requirement_answered_without_its_required_spread_is_named(self):
        """Arrived, and not in the form that was asked for. The contract knows;
        a reader could not see which finding was at fault."""
        built = view.build(
            _Job(
                _results(
                    [_benchmark(all_ms=[12.5], trial_spread=None)],
                    required_finding_types=["benchmark_measurement"],
                    validity={"require_uncertainty": ["benchmark_measurement"]},
                )
            )
        )

        requirement = built["requirements"][0]
        assert requirement["satisfied"] is True, "it did arrive"
        assert requirement["uncertainty_required"] is True
        assert requirement["missing_uncertainty"] == [0], "and it is the one at fault"

    def test_evidence_nobody_asked_for_is_separated_not_hidden(self):
        """A run records what it learns. Keeping it visible but apart is what
        lets the required evidence be found at all -- 4,247 `document`
        findings in this database against 97 benchmarks."""
        built = view.build(
            _Job(
                _results(
                    [_benchmark(), {"type": "document", "title": "a doc"}],
                    required_finding_types=["benchmark_measurement"],
                )
            )
        )

        assert built["unrequested"] == [1]
        assert len(built["evidence"]) == 2

    def test_the_contracts_own_verdict_is_carried_not_recomputed(self):
        """This view shows what the verdict was reached from. Deciding
        satisfaction a second way here would be a second authority, and the
        two would disagree."""
        results = _results([_benchmark()], required_finding_types=[])
        results["goal_contract"]["satisfied"] = True
        results["goal_contract"]["missing"] = ["something the run still wanted"]

        built = view.build(_Job(results))

        assert built["contract_satisfied"] is True
        assert built["missing"] == ["something the run still wanted"]


class TestItSurvivesRunsThatAreNotResearchRuns:
    def test_a_job_with_no_findings_at_all(self):
        built = view.build(_Job({}))

        assert built["evidence"] == []
        assert built["requirements"] == []
        assert built["contract_enabled"] is False

    def test_findings_that_are_not_dicts_are_skipped(self):
        built = view.build(_Job({"findings": ["a string", None, _benchmark()]}))

        assert len(built["evidence"]) == 1

    def test_a_run_with_no_contract_still_shows_its_evidence(self):
        """Most runs in this database have no contract. Their findings are
        still the only account of what happened."""
        built = view.build(_Job({"findings": [_benchmark()]}))

        assert len(built["evidence"]) == 1
        assert built["requirements"] == []
        assert built["unrequested"] == [0]

    def test_an_unsettled_prediction_is_surfaced_without_being_asked_for(self):
        """A claim nobody checked. `predictions_measured` asks this only when
        the contract requests it; a reader wants to know either way.

        Read from the run's ACTIONS, not its findings -- a prediction is a tool
        call that was or was not answered by a later one, and the finding a
        prediction produces says nothing about whether it was settled."""
        built = view.build(
            _Job(
                {
                    "actions_taken": [
                        {
                            "action": {"tool": "record_prediction", "params": {}},
                            "result": {
                                "success": True,
                                "data": {"prediction_id": "p1"},
                            },
                        }
                    ]
                }
            )
        )

        assert built["unsettled_predictions"] == ["p1"]

    def test_a_prediction_a_measurement_settled_is_not_reported(self):
        built = view.build(
            _Job(
                {
                    "actions_taken": [
                        {
                            "action": {"tool": "record_prediction", "params": {}},
                            "result": {
                                "success": True,
                                "data": {"prediction_id": "p1"},
                            },
                        },
                        {
                            "action": {
                                "tool": "record_measurement",
                                "params": {"prediction_id": "p1"},
                            },
                            "result": {"success": True},
                        },
                    ]
                }
            )
        )

        assert built["unsettled_predictions"] == []


class TestNumbersNestedOneLevelDown:
    """Tools put their numbers in different places, and a reader shown only
    the top level sees provenance with the measurement missing.

    Found on a real run rather than reasoned about: a `codegen_measurement`
    titled "6 vector ops, 4 conditional branches" rendered `subject` and
    `flags` and nothing else, because the counts live under `codegen`. The
    same trap `agent_measurement_validity._find_number` already guards.
    """

    def test_a_nested_measurement_is_lifted_out(self):
        built = view.build(
            _Job(
                {
                    "findings": [
                        {
                            "type": "codegen_measurement",
                            "title": "dotprod_O2 @ clang -O2: 6 vector ops",
                            "flags": "-O2",
                            "subject": "dotprod_O2",
                            "codegen": {
                                "calls": 0,
                                "vector_ops": 6,
                                "conditional_branches": 4,
                            },
                        }
                    ]
                }
            )
        )
        values = {v["label"]: v["value"] for v in built["evidence"][0]["values"]}

        assert values["codegen.vector_ops"] == "6"
        assert values["codegen.conditional_branches"] == "4"

    def test_the_container_name_is_kept_so_two_fields_stay_distinct(self):
        built = view.build(
            _Job(
                {
                    "findings": [
                        {
                            "type": "benchmark_measurement",
                            "cycles": 100,
                            "modelled": {"cycles": 90},
                        }
                    ]
                }
            )
        )
        values = {v["label"]: v["value"] for v in built["evidence"][0]["values"]}

        assert values["cycles"] == "100"
        assert values["modelled.cycles"] == "90"

    def test_a_large_nested_object_does_not_bury_the_finding(self):
        """A tool result nested whole is not a measurement."""
        built = view.build(
            _Job(
                {
                    "findings": [
                        {
                            "type": "benchmark_measurement",
                            "raw": {f"field_{i}": i for i in range(40)},
                        }
                    ]
                }
            )
        )

        assert len(built["evidence"][0]["values"]) <= 10


class TestRejectingAResult:
    """Advisory, and therefore the whole question is whether it is visible.

    A rejection that changes no verdict and marks nothing is a note in a
    drawer. These are the two places it has to show up: on the finding, and in
    the count that stops a run reading as clean.
    """

    def test_a_rejected_finding_carries_the_reason(self):
        built = view.build(
            _Job(_results([_benchmark()])), {0: "host was loaded; retake it"}
        )

        assert built["evidence"][0]["disputed"] is True
        assert built["evidence"][0]["dispute_reason"] == "host was loaded; retake it"

    def test_a_rejection_changes_no_verdict(self):
        """The contract's own answer is untouched. That is what advisory
        means, and it is the reason a rejection is safe to make."""
        results = _results([_benchmark()], required_finding_types=[])
        results["goal_contract"]["satisfied"] = True

        built = view.build(_Job(results), {0: "not sound"})

        assert built["contract_satisfied"] is True
        assert built["requirements"] == []

    def test_a_rejected_finding_still_satisfies_its_requirement(self):
        """It arrived. Whether it should be believed is a separate question
        from whether it exists, and conflating them would make a rejection
        silently take down the stage's contract."""
        built = view.build(
            _Job(
                _results(
                    [_benchmark()], required_finding_types=["benchmark_measurement"]
                )
            ),
            {0: "not sound"},
        )

        assert built["requirements"][0]["satisfied"] is True
        assert built["evidence"][0]["disputed"] is True

    def test_the_run_counts_what_was_rejected(self):
        built = view.build(
            _Job(_results([_benchmark(), _benchmark(), _benchmark()])),
            {0: "a", 2: "b"},
        )

        assert built["disputed_count"] == 2

    def test_no_rejections_is_the_ordinary_case(self):
        built = view.build(_Job(_results([_benchmark()])))

        assert built["disputed_count"] == 0
        assert built["evidence"][0]["disputed"] is False


class TestTheResponseIsBounded:
    """One job in this database holds thousands of `document` findings against
    a handful of measurements, so an uncapped response is almost entirely
    evidence nobody asked for -- and the run that most needs the cap is the one
    already slowest to serve.

    What the cap must never do is truncate the part the page exists for.
    """

    def _many(self, count, **over):
        base = [_benchmark()]
        base.extend({"type": "document", "title": f"doc {i}"} for i in range(count))
        return base

    def test_required_evidence_is_never_dropped(self):
        """A truncated requirement would read as a requirement nothing
        answered, which is the one message this view must not get wrong."""
        findings = [{"type": "document", "title": f"d{i}"} for i in range(300)]
        findings.append(_benchmark())
        built = view.build(
            _Job(_results(findings, required_finding_types=["benchmark_measurement"]))
        )

        assert built["requirements"][0]["satisfied"] is True
        assert built["requirements"][0]["satisfied_by"] == [300]

    def test_the_unasked_for_are_capped_and_the_total_reported(self):
        built = view.build(
            _Job(
                _results(
                    self._many(300), required_finding_types=["benchmark_measurement"]
                )
            )
        )

        assert len(built["unrequested"]) == 50
        assert (
            built["unrequested_total"] == 300
        ), "a sample presented as the whole is worse than a sample"

    def test_a_rejected_finding_is_shown_however_far_down_it_falls(self):
        """Someone went to the trouble of rejecting it; hiding it behind the
        cap would make the rejection unreachable."""
        built = view.build(
            _Job(
                _results(
                    self._many(300), required_finding_types=["benchmark_measurement"]
                )
            ),
            {250: "this document is not what it claims"},
        )

        assert any(e["index"] == 250 for e in built["evidence"])
        assert built["disputed_count"] == 1

    def test_a_small_run_is_untouched(self):
        built = view.build(_Job(_results(self._many(3), required_finding_types=[])))

        assert len(built["unrequested"]) == 4
        assert built["unrequested_total"] == 4
