"""Binding a pipeline to the chain that runs it, or refusing to.

The chain is a tree with one fan-in special case, so not every valid pipeline
is a runnable one. These tests are mostly about the refusals: the two available
approximations are both worse than saying no, because linearising a diamond
discards the parallelism the author asked for and dropping a cross edge starts
a stage before the evidence it needs.
"""

import pytest

from app.services import agent_pipeline_binding as pb
from app.services import agent_pipeline_spec as ps


def _stage(stage_id, types, **over):
    spec = {
        "id": stage_id,
        "goal": f"do {stage_id}",
        "contract": {"required_finding_types": list(types)},
    }
    spec.update(over)
    return spec


def _diamond():
    return ps.normalize(
        {
            "name": "diamond",
            "stages": [
                _stage("root", ["codegen_measurement"]),
                _stage("bench", ["benchmark_measurement"], depends_on=["root"]),
                _stage("sim", ["simulated_measurement"], depends_on=["root"]),
                _stage("compare", ["prediction_recorded"], depends_on=["bench", "sim"]),
            ],
        }
    )


def _child_named(job, stage_id):
    for child in (job.get("chain_config") or {}).get("child_jobs") or []:
        if child["config"]["pipeline_stage"] == stage_id:
            return child
    raise AssertionError(f"{stage_id} is not chained under {job['name']}")


class TestWhatItEmits:
    def test_a_linear_pipeline_nests_each_stage_under_the_one_before(self):
        pipeline = ps.normalize(
            {
                "name": "line",
                "stages": [
                    _stage("first", ["counter_trace"]),
                    _stage("second", ["predictability_ceiling"], depends_on=["first"]),
                ],
            }
        )
        bound = pb.bind(pipeline)
        assert len(bound.roots) == 1
        assert _child_named(bound.roots[0], "second")

    def test_the_contract_travels_into_the_job_config(self):
        """The stage's contract is the whole specification of the stage; if it
        does not reach the job, nothing enforces it."""
        bound = pb.bind(_diamond())
        contract = bound.roots[0]["config"]["goal_contract"]
        assert contract["required_finding_types"] == ["codegen_measurement"]

    def test_a_job_says_which_stage_of_which_pipeline_it_is(self):
        bound = pb.bind(_diamond())
        config = bound.roots[0]["config"]
        assert config["pipeline"] == "diamond"
        assert config["pipeline_stage"] == "root"

    def test_a_loop_becomes_the_iteration_budget_the_author_set(self):
        pipeline = ps.normalize(
            {
                "stages": [
                    _stage(
                        "a",
                        ["counter_trace"],
                        loop={"max_iterations": 7, "until": "no_new_findings"},
                    )
                ]
            }
        )
        job = pb.bind(pipeline).roots[0]
        assert job["max_iterations"] == 7
        assert job["config"]["loop_until"] == "no_new_findings"
        assert job["config"]["loop_dry_rounds"] >= 2

    def test_assumptions_are_carried_so_a_running_stage_knows_them(self):
        pipeline = ps.normalize(
            {
                "stages": [
                    _stage("a", ["counter_trace"]),
                    _stage(
                        "b",
                        ["predictability_ceiling"],
                        depends_on=["a"],
                        assumes=["counter_trace"],
                    ),
                ]
            }
        )
        child = _child_named(pb.bind(pipeline).roots[0], "b")
        assert child["config"]["pipeline_assumes"] == ["counter_trace"]

    def test_every_stage_reaches_the_binding(self):
        assert sorted(pb.bind(_diamond()).stage_ids()) == [
            "bench",
            "compare",
            "root",
            "sim",
        ]

    def test_only_the_keys_the_executor_reads_are_emitted(self):
        """A key create_chained_job does not read is a setting that looks
        applied and is not -- the failure this codebase keeps finding."""
        understood = {
            "name",
            "description",
            "job_type",
            "goal",
            "goal_criteria",
            "config",
            "chain_config",
            "max_iterations",
            "max_tool_calls",
            "max_llm_calls",
            "agent_definition_id",
        }
        bound = pb.bind(_diamond())

        def check(job):
            assert set(job) <= understood, f"unread keys: {set(job) - understood}"
            for child in (job.get("chain_config") or {}).get("child_jobs") or []:
                check(child)

        for root in bound.roots:
            check(root)


class TestFanIn:
    def test_a_diamond_is_expressible(self):
        assert pb.expressible(_diamond()) == []

    def test_both_branches_carry_the_converging_stage(self):
        """The gate runs on whichever sibling finishes last, so every sibling
        has to know what comes after; the group id is what stops two of them
        creating it."""
        bound = pb.bind(_diamond())
        root = bound.roots[0]
        for branch in ("bench", "sim"):
            child = _child_named(_child_named(root, branch), "compare")
            assert child["config"]["origin"] == pb.FAN_IN_ORIGIN
            assert child["config"]["swarm_fan_in_group_id"] == "diamond:compare"

    def test_each_branch_waits_for_its_siblings(self):
        bound = pb.bind(_diamond())
        for branch in ("bench", "sim"):
            data = _child_named(bound.roots[0], branch)["chain_config"]["chain_data"]
            assert data["swarm_fan_in_wait_for_all_siblings"] is True
            assert data["swarm_fan_in_expected_siblings"] == 2

    def test_a_stage_with_one_parent_is_not_marked_as_converging(self):
        bound = pb.bind(_diamond())
        assert "chain_data" not in bound.roots[0].get("chain_config", {})


class TestRefusals:
    def test_converging_stages_without_a_common_parent_are_refused(self):
        """Linearising this would throw away the parallelism, and dropping an
        edge would start c before its evidence exists."""
        cross = ps.normalize(
            {
                "stages": [
                    _stage("a", ["codegen_measurement"]),
                    _stage("b", ["benchmark_measurement"]),
                    _stage("c", ["prediction_recorded"], depends_on=["a", "b"]),
                ]
            }
        )
        problems = pb.expressible(cross)
        assert any("do not all branch from one earlier stage" in p for p in problems)
        with pytest.raises(ValueError):
            pb.bind(cross)

    def test_waiting_for_some_siblings_but_not_all_is_refused(self):
        """The gate waits for every sibling, so a stage that depends on two of
        three would wait for the third anyway -- silently, and later."""
        partial = ps.normalize(
            {
                "stages": [
                    _stage("root", ["codegen_measurement"]),
                    _stage("x", ["benchmark_measurement"], depends_on=["root"]),
                    _stage("y", ["simulated_measurement"], depends_on=["root"]),
                    _stage("z", ["cycle_model_measurement"], depends_on=["root"]),
                    _stage("join", ["prediction_recorded"], depends_on=["x", "y"]),
                ]
            }
        )
        problems = pb.expressible(partial)
        assert any("also branches to z" in p for p in problems)

    def test_an_invalid_pipeline_is_reported_before_its_shape_is_judged(self):
        """Told what is wrong with the research first, not what is wrong with
        the graph -- they want different fixes."""
        broken = ps.normalize({"stages": [_stage("a", ["unicorn_measurement"])]})
        problems = pb.expressible(broken)
        assert problems and "not valid yet" in problems[0]


class TestCheckpoints:
    def test_a_checkpoint_stage_does_not_chain_what_follows_it(self):
        pipeline = ps.normalize(
            {
                "stages": [
                    _stage("measure", ["counter_trace"], checkpoint=True),
                    _stage("after", ["predictability_ceiling"], depends_on=["measure"]),
                ]
            }
        )
        bound = pb.bind(pipeline)
        assert "chain_config" not in bound.roots[0]
        assert bound.stage_ids() == ["measure"]

    def test_the_held_edge_is_reported_rather_than_approximated(self):
        """None of on_complete, on_fail, on_any_end, on_progress or
        on_findings asks anybody. Picking the nearest one would call a stage
        gated when it is not."""
        pipeline = ps.normalize(
            {
                "stages": [
                    _stage("measure", ["counter_trace"], checkpoint=True),
                    _stage("after", ["predictability_ceiling"], depends_on=["measure"]),
                ]
            }
        )
        bound = pb.bind(pipeline)
        assert len(bound.deferred) == 1
        edge = bound.deferred[0]
        assert (edge.after, edge.launch) == ("measure", "after")
        assert "approval-gated" in edge.reason

    def test_the_description_says_what_is_being_held(self):
        pipeline = ps.normalize(
            {
                "name": "gated",
                "stages": [
                    _stage("measure", ["counter_trace"], checkpoint=True),
                    _stage("after", ["predictability_ceiling"], depends_on=["measure"]),
                ],
            }
        )
        text = "\n".join(pb.describe(pipeline))
        assert "held: after waits on measure" in text
