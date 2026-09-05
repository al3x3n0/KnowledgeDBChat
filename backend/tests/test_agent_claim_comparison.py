"""Scoring a measurement against a paper's claim.

The verdict this module exists to make possible is `incomparable`. Reproduction
work goes wrong in both directions -- calling a mismatch a refutation when the
two numbers never tested each other, and calling a coincidence a confirmation
when they were never the same quantity -- and both failures look like a result.
"""

import pytest

from app.services import agent_claim_comparison as claims

pytestmark = pytest.mark.unit


def _compare(**kwargs):
    base = dict(
        claimed_value=3.0,
        measured_value=3.0,
        claimed_unit="x",
        measured_unit="x",
        measurement_source="wall clock, 5 trials",
    )
    base.update(kwargs)
    return claims.compare(**base)


class TestTheStraightforwardCases:
    def test_a_match_reproduces(self):
        result = _compare(measured_value=3.1)
        assert result.verdict == claims.VERDICT_REPRODUCED
        assert result.relative_error == pytest.approx(0.0333, abs=1e-3)

    def test_a_miss_does_not(self):
        result = _compare(measured_value=1.5)
        assert result.verdict == claims.VERDICT_NOT_REPRODUCED
        # The direction is the useful part: "we got half" is a different
        # finding from "we got double".
        assert "below" in result.summary
        assert result.ratio == pytest.approx(0.5)

    def test_the_tolerance_is_the_authors_to_set(self):
        # 2.2 against a claimed 3.0 is 26.7% off: outside the default band,
        # inside a 30% one.
        assert _compare(measured_value=2.2, tolerance=0.2).verdict == (
            claims.VERDICT_NOT_REPRODUCED
        )
        assert _compare(measured_value=2.2, tolerance=0.3).verdict == (
            claims.VERDICT_REPRODUCED
        )


class TestIncomparableIsNotFailure:
    """The verdict that keeps a run from claiming a finding it did not earn."""

    def test_different_units_are_refused_rather_than_scored(self):
        result = _compare(claimed_unit="ms", measured_unit="x")
        assert result.verdict == claims.VERDICT_INCOMPARABLE
        assert any("Units differ" in b for b in result.blockers)

    def test_an_absolute_time_from_another_machine_tests_nothing(self):
        # 40ms on a Xeon and 44ms on aarch64 are within any tolerance and mean
        # nothing whatsoever. Scoring them as agreement is the flattering
        # failure this exists to stop.
        result = _compare(
            claimed_value=40.0,
            measured_value=44.0,
            claimed_unit="ms",
            measured_unit="ms",
            claimed_conditions={"hardware": "Xeon 8280"},
            measured_conditions={"hardware": "Graviton3"},
        )
        assert result.verdict == claims.VERDICT_INCOMPARABLE
        assert any("different machines" in b for b in result.blockers)

    def test_a_ratio_survives_the_move_with_a_caveat(self):
        # A speedup mostly cancels the machine out, so this is comparable --
        # but the reader should be told the move happened.
        result = _compare(
            measured_value=2.9,
            claimed_conditions={"hardware": "Xeon 8280"},
            measured_conditions={"hardware": "Graviton3"},
        )
        assert result.verdict == claims.VERDICT_REPRODUCED
        assert any("approximately" in c for c in result.caveats)

    def test_a_different_input_size_tests_nothing(self):
        # An algorithmic advantage is a function of n. Measuring at a size that
        # fits in cache and reporting "not reproduced" refutes nothing.
        result = _compare(
            measured_value=1.2,
            claimed_conditions={"input_size": "1e7"},
            measured_conditions={"input_size": "1e4"},
        )
        assert result.verdict == claims.VERDICT_INCOMPARABLE
        assert any("function of size" in b for b in result.blockers)

    def test_a_claim_with_no_number_cannot_be_tested(self):
        # Papers say "substantially faster" constantly. The honest answer is
        # that there is nothing to score, not a guess scored generously.
        result = _compare(claimed_value=None)
        assert result.verdict == claims.VERDICT_INCOMPARABLE
        assert any("no numeric value" in b for b in result.blockers)

    def test_a_measurement_with_no_source_is_a_recollection(self):
        result = _compare(measurement_source="")
        assert result.verdict == claims.VERDICT_INCOMPARABLE
        assert any("recollection" in b for b in result.blockers)

    def test_a_zero_claim_does_not_divide(self):
        result = _compare(claimed_value=0.0)
        assert result.verdict == claims.VERDICT_INCOMPARABLE
        assert result.relative_error is None

    def test_every_blocker_is_named_not_just_the_first(self):
        # A run that fixes one condition and retries should not discover the
        # next blocker one round at a time.
        result = _compare(
            claimed_unit="ms",
            measured_unit="x",
            measurement_source="",
        )
        assert len(result.blockers) >= 2


class TestNotationRatherThanSubstance:
    def test_percent_and_multiplier_say_the_same_thing(self):
        # "40% faster" and "1.4x" agree. Reading 40 against 1.4 reports a
        # catastrophic failure that did not happen.
        result = claims.compare(
            claimed_value=40.0,
            claimed_unit="percent",
            measured_value=1.42,
            measured_unit="x",
            measurement_source="wall clock",
        )
        assert result.verdict == claims.VERDICT_REPRODUCED

    def test_units_are_read_past_their_spelling(self):
        for spelling in ("x", " X ", "×", "1.8x", "speedup", "times"):
            assert claims.unit_kind(spelling) == "relative", spelling
        for spelling in ("ms", "milliseconds", "cycles"):
            assert claims.unit_kind(spelling) == "absolute", spelling

    def test_a_compound_ratio_is_relative(self):
        assert claims.unit_kind("cycles_per_byte") == "relative"
        assert claims.unit_kind("bytes/cycle") == "relative"


class TestWhatTheVerdictCarries:
    def test_the_evidence_says_why_not_just_what(self):
        evidence = _compare(claimed_unit="ms", measured_unit="x").as_evidence()
        assert evidence["verdict"] == claims.VERDICT_INCOMPARABLE
        assert evidence["comparable"] is False
        assert evidence["blockers"]

    def test_an_uncaveated_comparison_still_reports_its_terms(self):
        evidence = _compare(measured_value=3.1).as_evidence()
        assert evidence["ratio"] == pytest.approx(1.0333, abs=1e-3)
        assert evidence["tolerance"] == claims.DEFAULT_TOLERANCE
        assert evidence["unit_kind"] == "relative"


class TestAnUntrustworthyMeasurementSettlesNothing:
    """The other way a comparison can be wrong: the number itself.

    The benchmark tool already reports how busy the host was and how far the
    trials spread, and nothing read it back when the number was scored. One
    run produced these two benchmarks minutes apart and scored them as
    `reproduced` (2.44x) and `not_reproduced` (0.72x):

        fastmod [1.248]                      baseline [3.049]
        fastmod [7.4, 2.3, 17.7, 6.8, 14.1]  baseline [25.5, 1.7, 2.0, 35.5, 4.7]

    The second is not a disagreement with the paper. It is a machine too busy
    to measure anything, and calling it a refutation claims a finding the run
    did not earn.
    """

    def test_the_readings_from_that_run_are_refused(self):
        concerns = claims.measurement_concerns(
            {
                "reported_metrics": {
                    "fastmod_ns_per_op": [7.428964, 2.325733, 17.688632, 6.846945],
                    "baseline_ns_per_op": [25.476773, 1.676083, 1.985749, 35.515865],
                }
            }
        )
        assert concerns, "the unstable readings were accepted"
        assert "fastmod_ns_per_op" in " ".join(concerns)

    def test_the_clean_reading_from_that_run_is_accepted(self):
        # The same run's good benchmark must still score, or the check would
        # cost more than it saves.
        assert (
            claims.measurement_concerns(
                {
                    "measurement_environment": "quiet",
                    "reported_metrics": {
                        "fastmod_ns_per_op": [1.248],
                        "baseline_ns_per_op": [3.049],
                    },
                }
            )
            == []
        )

    def test_a_busy_host_is_a_concern(self):
        concerns = claims.measurement_concerns(
            {"measurement_environment": "busy", "load_per_cpu": 4.2}
        )
        assert concerns and "busy" in concerns[0]
        assert "4.2" in concerns[0]

    def test_a_saturated_host_is_a_concern(self):
        assert claims.measurement_concerns({"measurement_environment": "saturated"})

    def test_a_quiet_host_is_not(self):
        assert claims.measurement_concerns({"measurement_environment": "quiet"}) == []

    def test_wide_trial_spread_is_a_concern(self):
        assert claims.measurement_concerns({"trial_spread": 1.4})

    def test_a_tight_spread_is_not(self):
        assert claims.measurement_concerns({"trial_spread": 0.05}) == []

    def test_a_single_trial_is_weaker_but_not_corrupted(self):
        # One reading is a weaker measurement than several and not a wrong
        # one; refusing it would reject sound numbers along with the noise.
        assert claims.measurement_concerns({"single_trial": True}) == []

    def test_nonsense_input_is_not_treated_as_evidence_of_anything(self):
        assert claims.measurement_concerns(None) == []
        assert claims.measurement_concerns({}) == []
        assert claims.measurement_concerns({"trial_spread": "wide"}) == []
        assert (
            claims.measurement_concerns({"reported_metrics": {"x": ["a", "b"]}}) == []
        )

    def test_a_metric_that_touches_zero_is_not_divided_by(self):
        assert (
            claims.measurement_concerns({"reported_metrics": {"x": [0.0, 5.0]}}) == []
        )
