"""Measuring what a microarchitectural mechanism is worth.

Every test here is a wrong answer that was actually produced against the real
simulator before the guard existed. The numbers in them are the measured ones,
not illustrative: StridePrefetcher on gem5's default L1D issued 35 of 503,959
identified candidates and came back bit-identical to no prefetcher; the same
prefetcher on L2 issued 401,061 of 401,066 and was worth 1.78x; and adding a
prefetcher alongside a wider MSHR file read as a 2.59x prefetcher win that the
MSHRs delivered on their own.
"""

import pytest

from app.services import agent_gem5_mechanism as mech

STATS = """
---------- Begin Simulation Statistics ----------
simInsts                                      2000000
system.cpu.numCycles                          4824818
system.cpu.dcache.demandMisses::total          131928
hostSeconds                                     41.20
---------- End Simulation Statistics   ----------
"""


class TestTheConfoundRule:
    """The comparison must differ in the mechanism and in nothing else."""

    def test_a_geometry_change_alongside_the_mechanism_is_refused(self):
        """The 2.59x that was not the prefetcher."""
        confounds = mech.find_confounds(
            {"caches": {"l1d": {"mshrs": 4}}},
            {"caches": {"l1d": {"mshrs": 32, "prefetcher": "StridePrefetcher"}}},
        )

        assert confounds
        assert any("mshrs" in c for c in confounds)

    def test_the_mechanism_itself_is_not_a_confound(self):
        assert (
            mech.find_confounds(
                {"caches": {"l2": {"mshrs": 20}}},
                {"caches": {"l2": {"mshrs": 20, "prefetcher": "StridePrefetcher"}}},
            )
            == []
        )

    def test_a_geometry_change_carried_by_both_arms_is_allowed(self):
        """Widening the MSHRs is a legitimate study -- in both arms at once.

        Refusing it outright would make the tool unable to express the one
        configuration in which an L1 prefetcher can do anything.
        """
        assert (
            mech.find_confounds(
                {"caches": {"l1d": {"mshrs": 32}}},
                {"caches": {"l1d": {"mshrs": 32, "prefetcher": "StridePrefetcher"}}},
            )
            == []
        )

    def test_a_different_core_between_the_arms_is_a_confound(self):
        confounds = mech.find_confounds(
            {"cpu_type": "O3CPU"},
            {
                "cpu_type": "NeoverseV2",
                "caches": {"l2": {"prefetcher": "TaggedPrefetcher"}},
            },
        )

        assert any("cpu_type" in c for c in confounds)


class TestActivation:
    """A mechanism that never fires is not a negative result."""

    def test_a_mechanism_that_never_fired_is_an_error(self):
        verdict = mech.judge_activation(
            {"l1d.prefetcher": {"pfIdentified": 503959.0, "pfIssued": 0.0}}
        )

        assert verdict is not None
        assert "never fired" in verdict

    def test_the_starved_l1_prefetcher_is_caught_and_explained(self):
        """35 of 503,959 -- a run that is bit-identical to no prefetcher."""
        verdict = mech.judge_activation(
            {"l1d.prefetcher": {"pfIdentified": 503959.0, "pfIssued": 35.0}}
        )

        assert verdict is not None
        assert "mshrs" in verdict

    def test_a_mechanism_doing_its_work_passes(self):
        """401,061 of 401,066, the L2 prefetcher that was worth 1.78x."""
        assert (
            mech.judge_activation(
                {"l2.prefetcher": {"pfIdentified": 401066.0, "pfIssued": 401061.0}}
            )
            is None
        )

    def test_no_mechanism_to_check_is_not_a_failure(self):
        assert mech.judge_activation({}) is None


class TestWhereTheCountersLive:
    def test_an_l2_prefetcher_is_found_under_its_own_path(self):
        """The bug this replaced: a hardcoded `dcache` prefix reported the
        only prefetcher that worked as having done nothing."""
        activity = mech.mechanism_activity(
            {
                "system.l2cache.prefetcher.pfIdentified": 401066.0,
                "system.l2cache.prefetcher.pfIssued": 401061.0,
                "system.cpu.numCycles": 2706225.0,
            },
            {"caches": {"l2": {"prefetcher": "StridePrefetcher"}}},
        )

        assert activity == {
            "l2.prefetcher": {"pfIdentified": 401066.0, "pfIssued": 401061.0}
        }

    def test_a_level_with_no_mechanism_reports_none(self):
        assert (
            mech.mechanism_activity(
                {"system.cpu.dcache.prefetcher.pfIssued": 9.0},
                {"caches": {"l1d": {"prefetcher": "none"}}},
            )
            == {}
        )


class TestComparingTheArms:
    def test_identical_runs_are_reported_as_identical_not_as_zero_percent(self):
        """Three prefetcher parameters on this build are accepted, recorded as
        changed in config.ini, and change nothing. "0.0% faster" hides that."""
        stats = mech.parse_stats(STATS)

        result = mech.compare_arms(stats, dict(stats))

        assert result["identical_stats"] is True
        assert "deterministic" in result["note"]

    def test_a_real_speedup_is_a_ratio(self):
        result = mech.compare_arms(
            {"system.cpu.numCycles": 4824818.0},
            {"system.cpu.numCycles": 2706225.0},
        )

        assert result["identical_stats"] is False
        assert result["speedup"] == pytest.approx(1.7828, rel=1e-3)
        assert result["cycle_change_percent"] < 0

    def test_host_time_does_not_break_the_identity_check(self):
        """gem5 is deterministic; its host timings are not. Comparing those
        would fail every pair while proving nothing."""
        stats = mech.parse_stats(STATS)

        assert "hostSeconds" not in stats
        assert stats["system.cpu.numCycles"] == 4824818


class TestRefusalsBeforeAnythingRuns:
    async def test_a_confounded_pair_is_refused_without_simulating(self):
        result = await mech.simulate_mechanism(
            code="int main(void){return 0;}",
            baseline={"caches": {"l1d": {"mshrs": 4}}},
            variant={
                "caches": {"l1d": {"mshrs": 32, "prefetcher": "StridePrefetcher"}}
            },
        )

        assert result["success"] is False
        assert "2.59x" in result["error"]
        assert result["confounds"]

    async def test_a_variant_with_no_mechanism_says_what_one_looks_like(self):
        result = await mech.simulate_mechanism(
            code="int main(void){return 0;}", variant={}
        )

        assert result["success"] is False
        assert "StridePrefetcher" in result["error"]


class TestTheFindingItWrites:
    """A contract counts findings, so a tool that produces an evidence type
    and never writes one declares a capability the run cannot use."""

    def test_a_passing_comparison_is_written_as_evidence(self):
        finding = mech._finding(
            "L2 stride on strided scan",
            {
                "speedup": 1.8002,
                "baseline_cycles": 4824580.0,
                "variant_cycles": 2680036.0,
                "identical_stats": False,
            },
            {"l2.prefetcher": {"pfIssued": 409613.0, "pfIdentified": 409617.0}},
            {
                "caches": {
                    "l2": {
                        "prefetcher": "StridePrefetcher",
                        "replacement_policy": "LRURP",
                    }
                },
                "cpu": {"type": "O3CPU"},
            },
        )

        assert finding["type"] == "mechanism_comparison"
        assert finding["speedup"] == 1.8002
        assert "l2.prefetcher=StridePrefetcher" in finding["mechanisms"]
        assert "1.8002x" in finding["title"]

    def test_the_default_replacement_policy_is_not_reported_as_a_mechanism(self):
        """Every cache carries one whether or not it is being studied. Listing
        what the variant HAS rather than what it CHANGED would name LRURP as
        the mechanism under test in every prefetcher run."""
        finding = mech._finding(
            "",
            {"speedup": 1.8, "baseline_cycles": 2.0, "variant_cycles": 1.0},
            {},
            {
                "caches": {
                    "l2": {
                        "prefetcher": "StridePrefetcher",
                        "replacement_policy": "LRURP",
                    }
                }
            },
            {"caches": {"l2": {"prefetcher": "none", "replacement_policy": "LRURP"}}},
        )

        assert finding["mechanisms"] == ["l2.prefetcher=StridePrefetcher"]

    def test_a_replacement_policy_study_names_the_policy(self):
        finding = mech._finding(
            "",
            {"speedup": 1.07, "baseline_cycles": 2.0, "variant_cycles": 1.0},
            {},
            {"caches": {"l2": {"prefetcher": "none", "replacement_policy": "BRRIPRP"}}},
            {"caches": {"l2": {"prefetcher": "none", "replacement_policy": "LRURP"}}},
        )

        assert finding["mechanisms"] == ["l2.replacement_policy=BRRIPRP"]
        assert finding["subject"] == "l2.replacement_policy=BRRIPRP"


class TestHowTheFilterIsSpelled:
    """A live run asked for "prefetchers", was refused, asked again for
    "prefetcher", and got the same list. One action for a plural."""

    CATALOG = {"prefetcher": [], "replacement_policy": [], "conditional_predictor": []}

    def test_a_plural_means_the_category(self):
        assert mech.resolve_kind("prefetchers", self.CATALOG) == "prefetcher"

    def test_the_singular_still_works(self):
        assert mech.resolve_kind("prefetcher", self.CATALOG) == "prefetcher"

    def test_case_and_hyphens_do_not_matter(self):
        assert (
            mech.resolve_kind("Replacement-Policies", self.CATALOG)
            == "replacement_policy"
        )

    def test_nothing_asked_for_means_everything(self):
        assert mech.resolve_kind("", self.CATALOG) == ""

    def test_a_category_this_build_lacks_is_still_refused(self):
        """Quietly listing everything would hide a caller asking for a
        mechanism class that is not in this build at all."""
        assert mech.resolve_kind("value_predictor", self.CATALOG) is None

    def test_asking_for_all_is_the_same_as_asking_for_nothing(self):
        """The schema says "omit for all"; a live run wrote kind="all" and was
        refused. The two are the same request."""
        for spelling in ("all", "ALL", "any", "*"):
            assert mech.resolve_kind(spelling, self.CATALOG) == ""


class TestAnUnknownKindStillAnswers:
    """Three live runs lost an action to this parameter -- "prefetchers",
    "all", "cache" -- and every time the whole catalogue was what was wanted."""

    def test_an_unrecognised_kind_is_not_a_class_this_build_has(self):
        assert mech.resolve_kind("cache", {"prefetcher": [], "cpu_type": []}) is None


class TestCarryingAPlugin:
    """A mechanism the simulator does not ship is compiled in the same
    container that will dlopen it -- the only way the two agree about
    libstdc++ -- and built separately from the workload so a mistake in one is
    not reported as a mistake in the other."""

    def test_a_study_without_a_plugin_builds_nothing_extra(self):
        import inspect

        source = inspect.getsource(mech.run_configs)

        assert "if plugin_source.strip()" in source
        assert "exit 89" in source

    def test_the_plugin_and_the_workload_have_separate_exit_codes(self):
        """89 and 90. One error message telling a caller to check their C
        kernel when the fault is in their policy costs an iteration."""
        import inspect

        source = inspect.getsource(mech.run_configs)

        assert source.count("exit 89") == 1
        assert source.count("exit 90") == 1

    def test_simulate_mechanism_passes_the_plugin_through(self):
        import inspect

        signature = inspect.signature(mech.simulate_mechanism)

        assert "plugin_source" in signature.parameters
        assert signature.parameters["plugin_source"].default == ""
