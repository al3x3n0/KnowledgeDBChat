"""Studies that take more than one simulation.

The numbers quoted are measured, not illustrative. Two kernels: a 4 MiB
stride-64B scan where idealising the L1 recovered 84.5% and the issue queue
3.4%, and a 256 KiB stride-16B read-modify-write where the same two recovered
28.6% and 11.7%. Neither ordering is guessable from a cycle count, which is the
reason these tools exist.
"""

import pytest

from app.services import agent_gem5_studies as st


class TestBuildingTheConfigurations:
    def test_an_idealisation_overlays_without_discarding_the_rest(self):
        merged = st._merge(
            {"cpu_type": "O3CPU", "caches": {"l2": {"size": "2MiB", "assoc": 8}}},
            st.IDEALISATIONS["l2_capacity"]["config"],
        )

        assert merged["cpu_type"] == "O3CPU"
        assert merged["caches"]["l2"]["size"] == "256MiB"

    def test_the_issue_queue_is_idealised_through_its_vector(self):
        """numIQEntries does not exist on O3CPU in gem5 25.1: the queue moved
        to instQueues[N]. Setting the old name runs clean and changes nothing
        -- on the kernel where the queue is demonstrably binding."""
        config = st.IDEALISATIONS["issue_queue"]["config"]

        assert "instQueues[*].numEntries" in config["cpu_params"]
        assert "numIQEntries" not in config["cpu_params"]

    def test_the_branch_predictor_is_not_claimed_to_be_idealised(self):
        """gem5 has no perfect predictor. Substituting the largest one bounds
        the cost from BELOW, the opposite direction from every other entry."""
        assert st.IDEALISATIONS["branch_prediction"]["bounds"] == "lower"
        assert st.IDEALISATIONS["issue_queue"].get("bounds", "upper") == "upper"


class TestSweepPaths:
    def test_a_path_reaches_into_a_prefetcher_named_as_a_string(self):
        """The docs show {"prefetcher": "StridePrefetcher"}, so that is the
        shape a sweep will usually be handed."""
        out = st._set_path(
            {"caches": {"l2": {"prefetcher": "StridePrefetcher"}}},
            "caches.l2.prefetcher.params.degree",
            8,
        )

        assert out["caches"]["l2"]["prefetcher"] == {
            "class": "StridePrefetcher",
            "params": {"degree": 8},
        }

    def test_the_original_configuration_is_not_mutated(self):
        original = {"caches": {"l2": {"prefetcher": "StridePrefetcher"}}}
        st._set_path(original, "caches.l2.prefetcher.params.degree", 8)

        assert original["caches"]["l2"]["prefetcher"] == "StridePrefetcher"

    def test_missing_levels_are_created(self):
        out = st._set_path({}, "caches.l1d.mshrs", 32)

        assert out == {"caches": {"l1d": {"mshrs": 32}}}


class TestReadingACurve:
    CURVE = [
        {"value": 1, "speedup": 1.05},
        {"value": 2, "speedup": 1.20},
        {"value": 4, "speedup": 1.38},
        {"value": 8, "speedup": 1.38},
        {"value": 16, "speedup": 1.38},
    ]

    def test_saturation_is_the_first_setting_that_gets_there(self):
        """Paying for a setting that stopped helping is the mistake this
        catches: 4 is as good as 16 here."""
        assert st._saturation_point(self.CURVE) == 4

    def test_a_curve_that_turns_around_is_not_monotonic(self):
        turning = self.CURVE[:3] + [{"value": 8, "speedup": 1.10}]

        assert st._is_monotonic(self.CURVE) is True
        assert st._is_monotonic(turning) is False

    def test_a_curve_that_never_helps_has_no_saturation_point(self):
        assert st._saturation_point([{"value": 1, "speedup": 0.0}]) is None


class TestSummarisingAcrossKernels:
    def test_the_geometric_mean_is_used_for_ratios(self):
        """Speedups are ratios; averaging them arithmetically overweights the
        wins and can report a gain for a mechanism that mostly loses."""
        assert st._geomean([2.0, 0.5]) == pytest.approx(1.0)

    def test_kernels_that_failed_do_not_drag_the_mean(self):
        assert st._geomean([1.5, None, 0.0]) == pytest.approx(1.5)


class TestRefusalsBeforeAnythingRuns:
    async def test_headroom_without_targets_lists_them(self):
        result = await st.measure_headroom(code="int main(void){return 0;}", targets=[])

        assert result["success"] is False
        assert "issue_queue" in result["error"]

    async def test_an_unknown_target_is_refused_with_the_catalogue(self):
        result = await st.measure_headroom(
            code="int main(void){return 0;}", targets=["magic_unit"]
        )

        assert result["success"] is False
        assert "magic_unit" in result["error"]

    async def test_a_sweep_of_one_point_is_not_a_sweep(self):
        result = await st.sweep_mechanism(
            code="int main(void){return 0;}",
            variant={"caches": {"l2": {"prefetcher": "StridePrefetcher"}}},
            vary="caches.l2.prefetcher.params.degree",
            values=[4],
        )

        assert result["success"] is False
        assert "simulate_mechanism" in result["error"]

    async def test_one_kernel_is_not_an_evaluation(self):
        result = await st.evaluate_across_kernels(
            kernels=[{"name": "a", "code": "int main(void){return 0;}"}],
            variant={"caches": {"l2": {"prefetcher": "StridePrefetcher"}}},
        )

        assert result["success"] is False
        assert "overturned" in result["error"]

    async def test_a_confounded_evaluation_is_refused(self):
        """Same rule as simulate_mechanism: geometry differing between the
        arms means the result cannot be attributed to the mechanism."""
        result = await st.evaluate_across_kernels(
            kernels=[
                {"name": "a", "code": "int main(void){return 0;}"},
                {"name": "b", "code": "int main(void){return 1;}"},
            ],
            baseline={"caches": {"l2": {"mshrs": 20}}},
            variant={"caches": {"l2": {"mshrs": 64, "prefetcher": "StridePrefetcher"}}},
        )

        assert result["success"] is False
        assert result["confounds"]


class TestNamingAStructure:
    """explain_bottleneck prints `dominant: "IQ"` beside
    `headroom_target: "issue_queue"`. A live run passed the first of those and
    lost an action. Two names for one structure in one result is the tool's
    fault."""

    def test_the_backpressure_label_names_the_idealisation(self):
        assert st.resolve_target("IQ") == "issue_queue"
        assert st.resolve_target("SQ") == "store_queue"

    def test_the_documented_name_still_works(self):
        assert st.resolve_target("issue_queue") == "issue_queue"

    def test_case_does_not_matter(self):
        assert st.resolve_target("L1D_Capacity") == "l1d_capacity"

    def test_a_name_that_means_nothing_is_left_to_be_refused(self):
        """l2_cache was a real live guess; it must still produce the
        catalogue rather than silently idealising something else."""
        assert st.resolve_target("l2_cache") == "l2_cache"
        assert "l2_cache" not in st.IDEALISATIONS
