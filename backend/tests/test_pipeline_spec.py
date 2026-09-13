"""A pipeline is checked before it is run, or the check is worth nothing.

Every test here is about a refusal that costs nothing, standing in for a
failure that costs hours: evidence no tool produces, a stage built on a
measurement nothing takes, a loop with no bound, a budget in the wrong range.
"""

from app.services import agent_pipeline_spec as ps


def _stage(stage_id, types, **over):
    spec = {
        "id": stage_id,
        "goal": f"do {stage_id}",
        "contract": {"required_finding_types": list(types)},
    }
    spec.update(over)
    return spec


def _pipeline(*stages, name="p"):
    return ps.normalize({"name": name, "stages": list(stages)})


class TestReadingASpec:
    def test_a_pipeline_is_read_from_plain_config(self):
        pipeline = _pipeline(
            _stage("one", ["counter_trace"]),
            _stage("two", ["predictability_ceiling"], depends_on=["one"]),
        )
        assert pipeline.name == "p"
        assert [s.id for s in pipeline.stages] == ["one", "two"]
        assert pipeline.stages[1].depends_on == ("one",)

    def test_dependencies_may_be_written_as_a_string(self):
        pipeline = _pipeline(
            _stage("one", ["counter_trace"]),
            _stage("two", ["predictability_ceiling"], depends_on="one"),
        )
        assert pipeline.stages[1].depends_on == ("one",)

    def test_a_stage_without_a_loop_runs_once(self):
        pipeline = _pipeline(_stage("one", ["counter_trace"]))
        assert pipeline.stages[0].iterations() == 1

    def test_required_types_are_read_however_the_contract_writes_them(self):
        as_counts = ps.normalize(
            {
                "stages": [
                    {
                        "id": "a",
                        "contract": {
                            "required_finding_type_counts": {"counter_trace": 3}
                        },
                    }
                ]
            }
        )
        assert as_counts.stages[0].required_finding_types() == ["counter_trace"]


class TestRefusalsThatCostNothing:
    def test_evidence_no_tool_produces_is_refused(self):
        problems = ps.validate(_pipeline(_stage("a", ["unicorn_measurement"])))
        assert any("no tool produces unicorn_measurement" in p for p in problems)

    def test_a_stage_assuming_what_nothing_upstream_produces_is_refused(self):
        pipeline = _pipeline(
            _stage("first", ["codegen_measurement"]),
            _stage(
                "second",
                ["benchmark_measurement"],
                depends_on=["first"],
                assumes=["counter_trace"],
            ),
        )
        problems = ps.validate(pipeline)
        assert any("assumes 'counter_trace'" in p for p in problems)

    def test_an_assumption_a_parent_does_produce_is_accepted(self):
        pipeline = _pipeline(
            _stage("first", ["counter_trace"]),
            _stage(
                "second",
                ["predictability_ceiling"],
                depends_on=["first"],
                assumes=["counter_trace"],
            ),
        )
        assert ps.validate(pipeline) == []

    def test_an_assumption_inherited_from_a_grandparent_is_accepted(self):
        """Evidence carries down the chain, not just one step."""
        pipeline = _pipeline(
            _stage("first", ["counter_trace"]),
            _stage("second", ["predictability_ceiling"], depends_on=["first"]),
            _stage(
                "third",
                ["counter_tap_selection"],
                depends_on=["second"],
                assumes=["counter_trace"],
            ),
        )
        assert ps.validate(pipeline) == []

    def test_a_cycle_is_named_not_just_detected(self):
        pipeline = _pipeline(
            _stage("a", ["codegen_measurement"], depends_on=["b"]),
            _stage("b", ["benchmark_measurement"], depends_on=["a"]),
        )
        problems = ps.validate(pipeline)
        assert any("cycle" in p and "a" in p and "b" in p for p in problems)

    def test_a_stage_depending_on_itself_is_refused(self):
        problems = ps.validate(
            _pipeline(_stage("a", ["codegen_measurement"], depends_on=["a"]))
        )
        assert any("depends on itself" in p for p in problems)

    def test_an_unknown_dependency_is_refused(self):
        problems = ps.validate(
            _pipeline(_stage("a", ["codegen_measurement"], depends_on=["ghost"]))
        )
        assert any("unknown stage 'ghost'" in p for p in problems)

    def test_duplicate_stage_ids_are_refused(self):
        problems = ps.validate(
            _pipeline(
                _stage("a", ["codegen_measurement"]),
                _stage("a", ["benchmark_measurement"]),
            )
        )
        assert any("duplicate stage id" in p for p in problems)

    def test_a_contract_that_demands_nothing_is_refused(self):
        """A stage nothing can fail lengthens the pipeline without
        strengthening it."""
        problems = ps.validate(ps.normalize({"stages": [{"id": "a"}]}))
        assert any("nothing can fail it" in p for p in problems)

    def test_a_stage_that_only_waits_for_a_human_is_allowed_to_demand_nothing(self):
        pipeline = ps.normalize({"stages": [{"id": "gate", "checkpoint": True}]})
        assert ps.validate(pipeline) == []

    def test_an_empty_pipeline_is_refused(self):
        assert ps.validate(ps.normalize({"stages": []})) == ["pipeline has no stages"]


class TestLoopBounds:
    def test_a_loop_without_a_bound_is_refused(self):
        """The one failure nobody is present to interrupt."""
        pipeline = _pipeline(
            _stage("a", ["codegen_measurement"], loop={"until": "contract_satisfied"})
        )
        problems = ps.validate(pipeline)
        assert any("no iteration bound" in p for p in problems)

    def test_a_loop_over_the_cap_is_refused(self):
        pipeline = _pipeline(
            _stage(
                "a",
                ["codegen_measurement"],
                loop={"max_iterations": ps.MAX_LOOP_ITERATIONS + 1},
            )
        )
        assert any("over the" in p for p in ps.validate(pipeline))

    def test_an_unknown_stop_condition_is_refused(self):
        pipeline = _pipeline(
            _stage(
                "a",
                ["codegen_measurement"],
                loop={"max_iterations": 2, "until": "vibes"},
            )
        )
        assert any("known conditions are" in p for p in ps.validate(pipeline))

    def test_dry_rounds_default_above_one(self):
        """A round can come up empty and the next one not."""
        pipeline = _pipeline(
            _stage(
                "a",
                ["codegen_measurement"],
                loop={"max_iterations": 3, "until": "no_new_findings"},
            )
        )
        assert ps.validate(pipeline) == []
        assert pipeline.stages[0].loop.dry_rounds >= 2


class TestCompiling:
    def test_dependencies_come_before_dependants(self):
        pipeline = _pipeline(
            _stage("last", ["counter_tap_selection"], depends_on=["middle"]),
            _stage("first", ["counter_trace"]),
            _stage("middle", ["predictability_ceiling"], depends_on=["first"]),
        )
        assert ps.topological_order(pipeline) == ["first", "middle", "last"]

    def test_the_same_spec_always_compiles_the_same_way(self):
        """Two runs of one pipeline are only comparable if the plan is stable."""
        pipeline = _pipeline(
            _stage("a", ["counter_trace"]),
            _stage("b", ["codegen_measurement"]),
            _stage("c", ["benchmark_measurement"]),
        )
        assert ps.topological_order(pipeline) == ps.topological_order(pipeline)

    def test_a_stage_does_not_pay_for_evidence_it_inherits(self):
        """chain_for derives from nothing, which is right for one job and wrong
        for a stage: priced that way, cost grows with the depth of the graph
        and an affordable pipeline gets refused."""
        alone = ps.plan(_pipeline(_stage("solo", ["predictability_ceiling"])))
        assert "sample_hardware_counters" in alone.stages[0].tools

        chained = ps.plan(
            _pipeline(
                _stage("trace", ["counter_trace"]),
                _stage("ceiling", ["predictability_ceiling"], depends_on=["trace"]),
            )
        )
        second = next(s for s in chained.stages if s.stage_id == "ceiling")
        assert second.tools == ("measure_predictability",)
        assert second.seconds < alone.stages[0].seconds

    def test_a_loop_is_priced_at_its_worst_case(self):
        once = ps.plan(_pipeline(_stage("a", ["counter_trace"])))
        thrice = ps.plan(
            _pipeline(_stage("a", ["counter_trace"], loop={"max_iterations": 3}))
        )
        assert thrice.total_seconds == once.total_seconds * 3

    def test_the_critical_path_is_shorter_than_the_total_when_work_forks(self):
        """Spend and elapsed time are different questions: a pipeline can be
        affordable and still take a week."""
        compiled = ps.plan(
            _pipeline(
                _stage("root", ["codegen_measurement"]),
                _stage("left", ["benchmark_measurement"], depends_on=["root"]),
                _stage("right", ["simulated_measurement"], depends_on=["root"]),
            )
        )
        # Both branches wait on root, then run work of their own; the pipeline
        # spends both and elapses only the longer.
        assert compiled.critical_path_seconds < compiled.total_seconds

    def test_checkpoints_are_reported(self):
        compiled = ps.plan(
            _pipeline(
                _stage("a", ["counter_trace"]),
                _stage(
                    "b", ["predictability_ceiling"], depends_on=["a"], checkpoint=True
                ),
            )
        )
        assert compiled.checkpoints == ("b",)


class TestSayingWhatItDoesNotKnow:
    def test_a_tool_with_no_recorded_cost_is_named_not_counted_as_free(self):
        compiled = ps.plan(_pipeline(_stage("a", ["counter_tap_selection"])))
        assert "select_counter_taps" in compiled.unpriced

    def test_the_description_says_the_estimate_is_a_floor(self):
        text = "\n".join(
            ps.plan(_pipeline(_stage("a", ["counter_tap_selection"]))).describe()
        )
        assert "floor" in text

    def test_fitting_the_budget_on_an_incomplete_estimate_carries_a_caveat(self):
        """This is how a budget comes to sound generous and be short."""
        # model_parameters is produced by a tool with no recorded cost and no
        # prerequisites, so this prices at nothing and is not therefore free.
        verdict = ps.check_budget(_pipeline(_stage("a", ["model_parameters"])), 3600)
        assert verdict["affordable"] is True
        assert "floor rather than a prediction" in verdict["caveat"]


class TestAffording:
    def test_a_pipeline_that_cannot_afford_itself_is_refused_before_it_starts(self):
        verdict = ps.check_budget(_pipeline(_stage("a", ["counter_trace"])), 60)
        assert verdict["affordable"] is False
        assert "short by" in verdict["refusal"]

    def test_the_refusal_says_what_to_do_about_it(self):
        verdict = ps.check_budget(_pipeline(_stage("a", ["counter_trace"])), 60)
        for remedy in ("cut a stage", "lower a loop bound", "raise the budget"):
            assert remedy in verdict["refusal"]

    def test_a_pipeline_within_budget_is_allowed(self):
        verdict = ps.check_budget(_pipeline(_stage("a", ["counter_trace"])), 3600 * 5)
        assert verdict["affordable"] is True


class TestDescribing:
    def test_problems_are_reported_instead_of_a_plan(self):
        text = "\n".join(ps.describe(_pipeline(_stage("a", ["unicorn_measurement"]))))
        assert "cannot run" in text and "unicorn_measurement" in text

    def test_a_sound_pipeline_describes_its_plan(self):
        text = "\n".join(
            ps.describe(
                _pipeline(
                    _stage("trace", ["counter_trace"]),
                    _stage(
                        "ceiling",
                        ["predictability_ceiling"],
                        depends_on=["trace"],
                        assumes=["counter_trace"],
                    ),
                    name="study",
                )
            )
        )
        assert "study" in text
        assert "sample_hardware_counters" in text
        assert "measure_predictability" in text
