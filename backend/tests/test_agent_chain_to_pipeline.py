"""A chain becomes a pipeline, or says why it cannot.

The refusals are the point of these tests. Four of the six chain trigger
conditions have no pipeline stage equivalent, and the failure mode to guard
against is not an exception -- it is a conversion that looks successful and
quietly changes what the chain did.
"""

import pytest

from app.services.agent_chain_to_pipeline import (
    NOT_CONVERTIBLE,
    ChainNotConvertible,
    convert,
    describe,
)
from app.services.agent_pipeline_spec import normalize, validate


def _steps(*triggers):
    return [
        {
            "step_name": f"Step {i + 1}",
            "job_type": "research",
            "goal_template": f"Do thing {i + 1}",
            "trigger_condition": t,
        }
        for i, t in enumerate(triggers)
    ]


class TestConverts:
    def test_a_linear_chain_becomes_a_path_of_stages(self):
        spec = convert(
            name="lit_review", chain_steps=_steps("on_complete", "on_complete")
        )
        assert [s["id"] for s in spec["stages"]] == ["step_1", "step_2"]
        # A chain is a path: each stage waits for the one before it.
        assert "depends_on" not in spec["stages"][0]
        assert spec["stages"][1]["depends_on"] == ["step_1"]

    def test_the_result_is_a_pipeline_the_real_parser_reads(self):
        spec = convert(
            name="lit_review", chain_steps=_steps("on_complete", "on_complete")
        )
        pipeline = normalize(spec)
        assert pipeline.name == "lit_review"
        assert len(pipeline.stages) == 2

    def test_a_converted_chain_does_not_yet_validate_and_that_is_the_point(self):
        """The conversion is faithful, which means it is not yet runnable.

        A chain never said what a step had to achieve, so every converted stage
        has an empty contract -- and `validate` refuses a contract that nothing
        can fail. That refusal *is* the migration's value: it names, per stage,
        the work the author now has to do. The pipeline model stores a spec
        that does not validate on purpose, so it can be read and repaired
        rather than refused at the door.
        """
        spec = convert(
            name="lit_review", chain_steps=_steps("on_complete", "on_complete")
        )
        problems = validate(normalize(spec))
        assert len(problems) == 2, problems
        assert all("contract" in p for p in problems)
        # Named per stage, so the author knows where to start.
        assert {p.split(":")[0] for p in problems} == {"step_1", "step_2"}

    def test_a_missing_trigger_is_on_complete(self):
        spec = convert(
            name="c",
            chain_steps=[{"step_name": "A"}, {"step_name": "B"}],
        )
        assert spec["stages"][1]["depends_on"] == ["a"]

    def test_on_approval_becomes_a_checkpoint_rather_than_being_lost(self):
        spec = convert(name="c", chain_steps=_steps("on_complete", "on_approval"))
        assert spec["stages"][1]["checkpoint"] is True

    def test_contracts_are_left_empty_rather_than_invented(self):
        # The chain never said what a step had to achieve. Writing a contract
        # here would assert something its author never did.
        spec = convert(name="c", chain_steps=_steps("on_complete"))
        assert spec["stages"][0]["contract"] == {}

    def test_step_config_survives_and_chain_defaults_merge_under_it(self):
        spec = convert(
            name="c",
            chain_steps=[
                {"step_name": "A", "config": {"sources": ["arxiv"], "depth": 2}},
            ],
            default_config={"depth": 1, "notify": True},
        )
        assert spec["stages"][0]["config"] == {
            "depth": 2,  # the step's own value wins
            "notify": True,  # the chain default carries
            "sources": ["arxiv"],
        }

    def test_two_steps_with_one_name_get_distinct_stage_ids(self):
        spec = convert(
            name="c",
            chain_steps=[{"step_name": "Review"}, {"step_name": "Review"}],
        )
        ids = [s["id"] for s in spec["stages"]]
        assert len(set(ids)) == 2, ids
        assert spec["stages"][1]["depends_on"] == [ids[0]]


class TestRefuses:
    @pytest.mark.parametrize("trigger", sorted(NOT_CONVERTIBLE))
    def test_every_unmappable_trigger_is_refused_not_approximated(self, trigger):
        with pytest.raises(ChainNotConvertible):
            convert(name="c", chain_steps=_steps("on_complete", trigger))

    def test_the_refusal_names_the_step_the_trigger_and_the_reason(self):
        with pytest.raises(ChainNotConvertible) as caught:
            convert(name="monitoring", chain_steps=_steps("on_complete", "on_findings"))
        error = caught.value
        assert error.chain_name == "monitoring"
        ((step, trigger, why),) = error.reasons
        assert step == "Step 2"
        assert trigger == "on_findings"
        assert "still running" in why

    def test_a_live_monitor_is_exactly_the_case_this_protects(self):
        # The real chain in this database: a monitor that raises an alert once
        # it has five findings, while it keeps monitoring. As a stage
        # dependency it would wait for the monitor to finish, which it never
        # does -- a conversion that succeeded here would silently stop the
        # alerts.
        monitoring = [
            {
                "step_name": "Topic Monitoring",
                "job_type": "monitor",
                "trigger_condition": "on_findings",
                "trigger_thresholds": {"findings_threshold": 5},
            },
            {"step_name": "Alert", "job_type": "research"},
        ]
        assert [t for _, t, _ in describe(monitoring)] == ["on_findings"]
        with pytest.raises(ChainNotConvertible):
            convert(name="continuous_monitoring_with_alerts", chain_steps=monitoring)

    def test_describe_reports_without_raising_so_callers_can_survey_first(self):
        problems = describe(_steps("on_complete", "on_fail", "on_progress"))
        assert [t for _, t, _ in problems] == ["on_fail", "on_progress"]

    def test_describe_is_empty_for_a_chain_that_converts(self):
        assert describe(_steps("on_complete", "on_approval")) == []
