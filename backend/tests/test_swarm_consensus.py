"""Whether several agents independently found the same thing.

The rule this replaces keyed agreement on the lowercased text of a finding's
title -- LLM prose, compared for exact equality. Two agents reaching the same
conclusion in different words never matched, so the merge reported "the roles
did not agree" whether or not they had. The first test here is that exact
case, because a consensus mechanism that cannot recognise agreement is worse
than none: it states a specific falsehood instead of staying quiet.
"""

import pytest

from app.services import agent_swarm_consensus as consensus

pytestmark = pytest.mark.unit


def _bench(subject, ms, **over):
    finding = {
        "type": "benchmark_measurement",
        "subject": subject,
        "title": f"{subject}: fastest {ms} ms",
        "fastest_ms": ms,
    }
    finding.update(over)
    return finding


class TestTheProseBug:
    def test_two_roles_agreeing_in_different_words_now_corroborate(self):
        """The exact failure of the old rule. These titles share not one
        character sequence, and they are the same claim."""
        groups = consensus.group_claims(
            {
                "researcher": [
                    {
                        "type": "bottleneck_attribution",
                        "subject": "dotprod",
                        "title": "The loop is memory bound on the accumulator",
                    }
                ],
                "verifier": [
                    {
                        "type": "bottleneck_attribution",
                        "subject": "dotprod",
                        "title": "Accumulator dependency stalls the pipeline",
                    }
                ],
            }
        )

        assert len(groups) == 1
        assert groups[0].verdict == "corroborated"
        assert groups[0].roles == ["researcher", "verifier"]

    def test_the_same_role_twice_is_not_corroboration(self):
        """One agent producing a finding twice is one agent. Counting it as
        two is how an echo becomes evidence."""
        groups = consensus.group_claims(
            {"researcher": [_bench("dotprod", 12.0), _bench("dotprod", 12.1)]}
        )

        assert groups[0].verdict == "uncorroborated"
        assert groups[0].roles == ["researcher"]


class TestNumbersAgreeWithinATolerance:
    def test_close_measurements_corroborate(self):
        groups = consensus.group_claims(
            {
                "researcher": [_bench("dotprod", 12.0)],
                "verifier": [_bench("dotprod", 12.4)],
            }
        )

        assert groups[0].verdict == "corroborated"
        assert "12" in groups[0].detail

    def test_far_apart_measurements_are_contested_with_the_values(self):
        """ "The roles disagreed" is not useful. Which value each produced is
        what tells a person what to do next."""
        groups = consensus.group_claims(
            {
                "researcher": [_bench("dotprod", 12.5)],
                "verifier": [_bench("dotprod", 48.0)],
            }
        )

        assert groups[0].verdict == "contested"
        assert "researcher 12.5" in groups[0].detail
        assert "verifier 48" in groups[0].detail

    def test_a_findings_own_spread_is_the_tolerance(self):
        """A measurement that reports it varies by 40% is telling you what
        agreement means for it. Overruling that with a fixed percentage
        discards a real measurement in favour of a constant."""
        groups = consensus.group_claims(
            {
                "researcher": [_bench("dotprod", 10.0, trial_spread=0.4)],
                "verifier": [_bench("dotprod", 13.0)],
            }
        )

        assert (
            groups[0].verdict == "corroborated"
        ), "30% apart, but the run itself measured 40% variation"

    def test_a_tight_spread_does_not_make_the_rule_stricter_than_default(self):
        """A run reporting 0.1% variance should not turn a 5% difference into
        a disagreement; the default floor exists to stop that."""
        groups = consensus.group_claims(
            {
                "researcher": [_bench("dotprod", 100.0, trial_spread=0.001)],
                "verifier": [_bench("dotprod", 105.0)],
            }
        )

        assert groups[0].verdict == "corroborated"

    def test_small_numbers_are_compared_relatively(self):
        """1ms against 2ms is a disagreement; 1000ms against 1001ms is not,
        and an absolute threshold cannot say both."""
        groups = consensus.group_claims(
            {
                "researcher": [_bench("tiny", 1.0)],
                "verifier": [_bench("tiny", 2.0)],
            }
        )

        assert groups[0].verdict == "contested"

    def test_a_number_nested_one_level_down_is_still_compared(self):
        groups = consensus.group_claims(
            {
                "researcher": [
                    {
                        "type": "codegen_measurement",
                        "subject": "dotprod",
                        "data": {"cycles": 100},
                    }
                ],
                "verifier": [
                    {
                        "type": "codegen_measurement",
                        "subject": "dotprod",
                        "data": {"cycles": 400},
                    }
                ],
            }
        )

        assert groups[0].verdict == "contested"


class TestWhatCountsAsTheSameThing:
    def test_different_subjects_never_merge(self):
        groups = consensus.group_claims(
            {
                "researcher": [_bench("dotprod", 12.0)],
                "verifier": [_bench("matmul", 12.0)],
            }
        )

        assert len(groups) == 2
        assert all(g.verdict == "uncorroborated" for g in groups)

    def test_different_evidence_types_never_merge(self):
        """A benchmark and an implementation check about the same subject are
        different claims, and merging them would count one as support for the
        other."""
        groups = consensus.group_claims(
            {
                "researcher": [_bench("dotprod", 12.0)],
                "verifier": [
                    {
                        "type": "implementation_verified",
                        "subject": "dotprod",
                        "passed": 4,
                    }
                ],
            }
        )

        assert len(groups) == 2

    def test_the_subject_is_normalised(self):
        groups = consensus.group_claims(
            {
                "researcher": [_bench("Dot Product", 12.0)],
                "verifier": [_bench("dot_product", 12.2)],
            }
        )

        assert len(groups) == 1
        assert groups[0].verdict == "corroborated"


class TestTheSummary:
    def test_agreement_is_over_what_was_checkable(self):
        """A finding only one role produced is evidence of neither agreement
        nor disagreement. Counting it either way would make a swarm of silent
        roles look unanimous."""
        groups = consensus.group_claims(
            {
                "researcher": [_bench("a", 10.0), _bench("solo", 5.0)],
                "verifier": [_bench("a", 10.2)],
            }
        )
        out = consensus.summarize(groups)

        assert out["corroborated_count"] == 1
        assert out["uncorroborated_count"] == 1
        assert out["agreement"] == 1.0, "1 of 1 checkable, not 1 of 2"

    def test_no_checkable_claims_yields_no_agreement_score(self):
        """None, not zero: nothing was checked, which is different from
        everything checked having failed."""
        out = consensus.summarize(
            consensus.group_claims({"researcher": [_bench("solo", 1.0)]})
        )

        assert out["agreement"] is None

    def test_contested_sorts_first(self):
        """A disagreement is the thing a person must look at; burying it under
        agreements is how a swarm's one useful output is missed."""
        groups = consensus.group_claims(
            {
                "researcher": [_bench("agree", 10.0), _bench("differ", 10.0)],
                "verifier": [_bench("agree", 10.1), _bench("differ", 90.0)],
            }
        )

        assert groups[0].verdict == "contested"
        assert groups[0].subject == "differ"


class TestItSurvivesRealisticJunk:
    def test_findings_without_a_type_are_skipped(self):
        assert consensus.group_claims({"researcher": [{"subject": "x"}]}) == []

    def test_non_dict_findings_are_skipped(self):
        groups = consensus.group_claims({"researcher": ["a string", None]})

        assert groups == []

    def test_an_empty_swarm(self):
        out = consensus.summarize(consensus.group_claims({}))

        assert out["agreement"] is None
        assert out["corroborated"] == []


class TestItReachesTheMergedResult:
    """The consensus module is only worth having if the fan-in uses it.

    These go through `build_swarm_fan_in_result`, because the defect was never
    in the matching alone -- it was that the merge asked prose whether the
    roles agreed and then reported the answer as fact.
    """

    def _payload(self, roles):
        return {
            "sibling_jobs": [
                {
                    "role": role,
                    "status": "completed",
                    "results": {"findings": findings},
                }
                for role, findings in roles.items()
            ],
            "expected_siblings": len(roles),
        }

    def test_agreeing_roles_are_no_longer_reported_as_low_alignment(self):
        """The old rule declared "low alignment" whenever two titles were not
        byte-identical, which was every swarm that ever ran."""
        from app.services.agent_swarm_fan_in import build_swarm_fan_in_result

        out = build_swarm_fan_in_result(
            self._payload(
                {
                    "researcher": [_bench("dotprod", 12.0)],
                    "verifier": [_bench("dotprod", 12.3)],
                }
            )
        )

        assert out["corroborated_count"] == 1
        assert not [c for c in out["conflicts"] if c["type"] == "low_alignment"]

    def test_a_real_disagreement_becomes_a_conflict_with_its_numbers(self):
        from app.services.agent_swarm_fan_in import build_swarm_fan_in_result

        out = build_swarm_fan_in_result(
            self._payload(
                {
                    "researcher": [_bench("dotprod", 12.0)],
                    "verifier": [_bench("dotprod", 96.0)],
                }
            )
        )

        contested = [
            c for c in out["conflicts"] if c["type"] == "contested_measurement"
        ]
        assert contested, "a swarm exists to surface exactly this"
        assert "12" in contested[0]["description"]
        assert "96" in contested[0]["description"]

    def test_silence_is_not_dissent_in_the_agreement_score(self):
        """Two of four roles agreeing on the one thing they both measured is
        full agreement on what was checkable. The old score divided by the
        sibling count and read 0.5, as though the quiet two had objected."""
        from app.services.agent_swarm_fan_in import build_swarm_fan_in_result

        out = build_swarm_fan_in_result(
            self._payload(
                {
                    "researcher": [_bench("dotprod", 10.0)],
                    "verifier": [_bench("dotprod", 10.1)],
                    "critic": [],
                    "synthesizer": [],
                }
            )
        )

        assert out["typed_agreement"] == 1.0


class TestAgreementThatResolvesNothing:
    """Tolerance comes from the claims' own reported spread, which is right --
    until the spread is enormous, and then every pair of numbers "agrees".

    Taken from the first swarm run whose roles both actually benchmarked:
    trial spreads of 130% and 142% on a host the tool itself labelled
    `saturated` turned 60ms vs 45ms into "corroborated (within 142%)", and the
    summary reported agreement 1.0. The arithmetic was right and the reader
    was still misled, because at that width two numbers 2.4x apart would have
    passed too.
    """

    def test_the_measured_case_is_no_longer_called_agreement(self):
        groups = consensus.group_claims(
            {
                "researcher": [_bench("dotprod", 60.0, trial_spread=1.3)],
                "verifier": [_bench("dotprod", 45.0, trial_spread=1.422)],
            }
        )

        assert groups[0].verdict == "inconclusive"

    def test_it_says_why_rather_than_just_refusing(self):
        groups = consensus.group_claims(
            {
                "researcher": [_bench("dotprod", 60.0, trial_spread=1.3)],
                "verifier": [_bench("dotprod", 45.0, trial_spread=1.422)],
            }
        )
        detail = groups[0].detail

        assert "142%" in detail, "the reader needs the width that defeated it"
        assert "60" in detail and "45" in detail, "and the numbers themselves"

    def test_it_does_not_count_as_agreement_in_the_score(self):
        """None, not 1.0 and not 0.0. Scoring it either way invents a result
        the measurement cannot support."""
        out = consensus.summarize(
            consensus.group_claims(
                {
                    "researcher": [_bench("dotprod", 60.0, trial_spread=1.3)],
                    "verifier": [_bench("dotprod", 45.0, trial_spread=1.422)],
                }
            )
        )

        assert out["agreement"] is None
        assert out["inconclusive_count"] == 1
        assert out["corroborated_count"] == 0
        assert out["contested_count"] == 0

    def test_a_run_that_reports_its_own_precision_is_still_believed(self):
        """The floor is not a licence to overrule a run's own spread. 40% is a
        real statement about precision and stays corroboration -- the ceiling
        exists for the case where the window has swallowed the question."""
        groups = consensus.group_claims(
            {
                "researcher": [_bench("dotprod", 10.0, trial_spread=0.4)],
                "verifier": [_bench("dotprod", 13.0)],
            }
        )

        assert groups[0].verdict == "corroborated"

    def test_a_wide_spread_does_not_hide_a_real_disagreement(self):
        """Ordering matters: numbers far enough apart to be contested even at
        this tolerance must not be laundered into "inconclusive"."""
        groups = consensus.group_claims(
            {
                "researcher": [_bench("dotprod", 10.0, trial_spread=1.4)],
                "verifier": [_bench("dotprod", 4000.0)],
            }
        )

        assert groups[0].verdict == "contested", (
            "140% noise spans about 2.4x and cannot explain 400x; the noise "
            "ceiling must not launder a real disagreement into 'unresolvable'"
        )

    def test_the_verdict_carries_the_numbers_it_was_made_from(self):
        """A verdict without its resolution cannot be audited: 'corroborated'
        means something different at 5% than at 45%."""
        groups = consensus.group_claims(
            {
                "researcher": [_bench("dotprod", 100.0)],
                "verifier": [_bench("dotprod", 104.0)],
            }
        )
        row = groups[0].as_dict()

        assert row["verdict"] == "corroborated"
        assert row["drift"] == pytest.approx(0.04, abs=0.005)
        assert row["tolerance"] == pytest.approx(consensus.DEFAULT_TOLERANCE)

    def test_an_unresolvable_group_sorts_above_the_agreements(self):
        """Same reason contested sorts first: the rows a person must act on
        must not be buried under the ones they need not."""
        groups = consensus.group_claims(
            {
                "researcher": [
                    _bench("clean", 100.0),
                    _bench("noisy", 60.0, trial_spread=1.3),
                ],
                "verifier": [_bench("clean", 101.0), _bench("noisy", 45.0)],
            }
        )

        assert [g.verdict for g in groups] == ["inconclusive", "corroborated"]
