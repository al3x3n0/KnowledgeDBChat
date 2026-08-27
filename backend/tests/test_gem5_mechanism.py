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


class TestTheNullControlSurvivesNaN:
    """gem5 prints `nan` for every averaged statistic whose denominator was
    zero -- avgBlocked::no_mshrs, ftq.occupancy::mean and a dozen more, in
    essentially every run. NaN is not equal to itself, so comparing the two
    dictionaries with == reported byte-identical runs as different. The check
    that exists to say "the variant changed nothing" never once fired on real
    output, and the tests that passed used statistics with no NaN in them.

    These parse from text rather than copying a dict. Two runs produce two
    files and therefore two distinct NaN objects, and dict equality
    short-circuits on identity -- a test that shared one NaN object would pass
    against the broken code too, which is how the gap survived in the first
    place.
    """

    TEXT = """
---------- Begin Simulation Statistics ----------
system.cpu.numCycles                           466578
system.l2cache.avgBlocked::no_mshrs               nan
system.cpu.ftq.occupancy::mean                    nan
---------- End Simulation Statistics   ----------
"""

    def parsed(self, cycles="466578"):
        return mech.parse_stats(self.TEXT.replace("466578", cycles))

    def test_two_identical_runs_with_nan_are_identical(self):
        assert mech.stats_identical(self.parsed(), self.parsed()) is True

    def test_plain_equality_would_have_missed_it(self):
        """The bug, pinned: this is what the code used to do, against two
        separately parsed files."""
        assert (self.parsed() == self.parsed()) is False

    def test_a_real_difference_is_still_a_difference(self):
        assert mech.stats_identical(self.parsed(), self.parsed("466579")) is False

    def test_a_nan_where_the_other_run_had_a_number_is_a_difference(self):
        numeric = self.parsed()
        numeric["system.cpu.ftq.occupancy::mean"] = 1.0

        assert mech.stats_identical(self.parsed(), numeric) is False

    def test_a_missing_statistic_is_a_difference(self):
        fewer = self.parsed()
        del fewer["system.cpu.numCycles"]

        assert mech.stats_identical(self.parsed(), fewer) is False

    def test_an_inert_variant_is_reported_as_identical(self):
        """End of the chain: what a caller sees when a mechanism was accepted,
        recorded as configured, and changed nothing."""
        result = mech.compare_arms(self.parsed(), self.parsed())

        assert result["identical_stats"] is True
        assert "deterministic" in result["note"]


class TestRefusingSourceThatIsNotAPlugin:
    """A live run wrote gem5-style C++ three times running -- `#include
    "mem/cache/prefetch/queued.hh"`, a class deriving from Queued, no extern
    "C" -- because that is what a gem5 prefetcher looks like everywhere except
    here. The description said the right thing and lost to a stronger prior.
    The compiler's answer to that source is a wall of C++ whose implied
    correction, find the gem5 headers, is the opposite of the truth."""

    GEM5_STYLE = """#include "mem/cache/prefetch/queued.hh"
#include "params/CustomNextLinePrefetcher.hh"
class CustomNextLine : public Queued { void calculatePrefetch(); };"""

    def test_gem5_internals_are_caught_before_the_compiler(self):
        message = mech.check_plugin_source(self.GEM5_STYLE)

        assert message is not None
        assert "no gem5 headers exist" in message.lower()

    def test_the_refusal_carries_a_whole_working_example(self):
        """Describing a contract nobody has seen does not teach it. The same
        gap left the AXIS tools unusable: the grammar was never shown."""
        message = mech.check_plugin_source(self.GEM5_STYLE)

        assert "gem5_pf_api_v1" in message
        assert "extern" in message and "GEM5_PF_ABI_VERSION" in message

    def test_it_names_the_header_for_what_was_being_written(self):
        """The first version of this message named the replacement-policy
        header for every plugin, including prefetchers -- so it corrected a
        wrong include with another wrong include."""
        assert "gem5_pf_plugin_abi.h" in mech.check_plugin_source(self.GEM5_STYLE)
        assert "gem5_rp_plugin_abi.h" in mech.check_plugin_source(
            '#include "mem/cache/replacement_policies/base.hh"\nclass X {};'
        )

    def test_a_missing_entry_symbol_is_caught(self):
        source = "#include <gem5_pf_plugin_abi.h>\nstatic int nothing;"

        assert "gem5_pf_api_v1" in mech.check_plugin_source(source)

    def test_a_mangled_entry_symbol_is_caught(self):
        source = (
            "#include <gem5_pf_plugin_abi.h>\n"
            "const Gem5PfApiV1 *gem5_pf_api_v1(void) { return 0; }"
        )
        message = mech.check_plugin_source(source)

        assert message is not None and "mangle" in message

    def test_source_naming_neither_abi_is_caught(self):
        assert "one of them" in mech.check_plugin_source("int main(void){return 0;}")

    def test_every_skeleton_passes_its_own_check(self):
        """A skeleton the checker would reject is advice that cannot be taken.
        Both are also verified to compile, load and run in the sandbox."""
        for kind, spec in mech.PLUGIN_KINDS.items():
            assert mech.check_plugin_source(spec["skeleton"]) is None, kind

    def test_every_skeleton_exports_what_the_shim_looks_for(self):
        for kind, spec in mech.PLUGIN_KINDS.items():
            assert 'extern "C"' in spec["skeleton"], kind
            assert spec["entry"] in spec["skeleton"], kind
            assert f"#include <{spec['header']}>" in spec["skeleton"], kind
