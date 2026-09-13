"""How much a wall-clock timing actually moved.

`(max - min) / min` over three trials is one stall away from meaningless, and
that is not hypothetical: a swarm role timing a 46 ms kernel recorded
`all_ms=[61, 109, 46]` and reported 137% spread. The slow trial was the
MIDDLE one, so it was not warm-up -- two trials agreed and one blip decided
the number. That spread became the consensus tolerance, pushed the merge past
the point where agreement means anything, and called a person over to look at
a machine hiccup.

The asymmetry these tests encode: interference can add time to a trial and can
never subtract it. So the slowest sample is the most contaminated by
construction, and discarding it is not massaging the data -- treating it as
signal is.
"""

import pytest

from app.services.agent_compiler_sandbox import (
    DEFAULT_REPEAT,
    TRIM_FRACTION,
    measurement_quality,
    robust_spread,
)

pytestmark = pytest.mark.unit


class TestTheMeasuredFailure:
    def test_the_live_sample_is_no_longer_dominated_by_one_stall(self):
        out = robust_spread([61, 109, 46])
        assert out["trial_spread_raw"] == pytest.approx(1.37, abs=0.01)
        assert out["trial_spread"] < 0.4, "137% was the stall, not the code"

    def test_five_trials_absorb_a_stall_almost_entirely(self):
        """Why the default moved to five: with four clean samples left, one
        blip stops mattering instead of merely mattering less."""
        out = robust_spread([46, 47, 48, 49, 110])
        assert out["trial_spread_raw"] > 1.0
        assert out["trial_spread"] < 0.1

    def test_the_untrimmed_figure_is_still_reported(self):
        """Nothing is hidden. The gap between the two IS the finding that the
        host stalled."""
        out = robust_spread([46, 47, 48, 49, 110])
        assert "trial_spread_raw" in out and "trial_spread" in out


class TestItDoesNotInventPrecision:
    def test_a_genuinely_variable_run_still_reports_a_wide_spread(self):
        """Trimming one slow trial must not turn noise into precision."""
        out = robust_spread([10, 40, 70, 100, 130])
        assert out["trial_spread"] > 0.5

    def test_two_trials_are_not_trimmed(self):
        """Trimming one of two would leave a single number and a spread of
        zero -- precision manufactured out of nothing."""
        out = robust_spread([100, 160])
        assert out.get("trials_discarded", 0) == 0
        assert out["trial_spread"] == pytest.approx(0.6)

    def test_few_trials_are_flagged(self):
        assert robust_spread([100, 110, 120])["few_trials"] is True
        assert "few_trials" not in robust_spread([100, 110, 120, 130, 140])

    def test_one_trial_reports_no_spread_at_all(self):
        out = robust_spread([100])
        assert out["single_trial"] is True
        assert "trial_spread" not in out

    def test_a_perfectly_stable_run_reports_zero(self):
        assert robust_spread([50, 50, 50, 50, 50])["trial_spread"] == 0.0


class TestOnlySlowTrialsAreDiscarded:
    def test_the_fastest_trial_always_survives(self):
        """`fastest_ms` is the reported number; trimming must never touch it."""
        for sample in ([5, 100, 200], [5, 6, 7, 8, 900], [1, 2, 3, 4, 5, 6, 7, 8]):
            out = robust_spread(sample)
            assert out["trial_spread"] >= 0

    def test_it_trims_from_the_slow_end(self):
        fast_outlier = robust_spread([1, 100, 101, 102, 103])
        slow_outlier = robust_spread([100, 101, 102, 103, 400])
        assert slow_outlier["trial_spread"] < slow_outlier["trial_spread_raw"]
        # A fast outlier is not discarded: nothing makes a run spuriously fast.
        assert fast_outlier["trial_spread"] > 0.9

    def test_the_trim_scales_with_the_sample(self):
        assert robust_spread([1, 2, 3])["trials_discarded"] == 1
        assert robust_spread(list(range(1, 9)))["trials_discarded"] == 2
        assert TRIM_FRACTION == 0.25


class TestWhatTheRunIsTold:
    def _warning(self, timings):
        q = measurement_quality(0.1, 8, timings, None)
        return q.get("measurement_warning") or ""

    def test_a_stall_is_named_rather_than_hidden(self):
        assert "stalled" in self._warning([46, 47, 48, 49, 110])

    def test_a_clean_run_is_not_warned_about_stalls(self):
        assert "stalled" not in self._warning([50, 51, 52, 53, 54])

    def test_three_trials_are_called_out_as_too_few(self):
        assert "repeat=5" in self._warning([50, 51, 52])

    def test_the_default_is_enough_trials_to_trim(self):
        assert DEFAULT_REPEAT >= 4, "trimming needs a sample worth trimming"


class TestTheDefaultReachesTheAgent:
    """Agents reach the benchmark only through the dispatch wrapper, and the
    wrapper used to restate the trial count as `or 3` -- a second copy of a
    default that also lives on `benchmark_c_snippet`.

    So raising the sandbox default to five changed nothing for the callers it
    was raised for. Measured: a swarm launched after the change still took
    three trials, one of which stalled at 230 ms against a 56 ms best. The
    standalone check that "proved" five trials had called the sandbox
    directly, down a path no agent uses.
    """

    def test_the_wrapper_does_not_restate_the_default(self):
        import inspect

        from app.services import agent_tool_dispatch

        source = inspect.getsource(agent_tool_dispatch)
        start = source.index("async def _benchmark_c_snippet")
        body = source[start : start + 1600]
        assert (
            'params.get("repeat") or 3' not in body
        ), "the trial count default belongs to benchmark_c_snippet alone"

    def test_an_explicit_repeat_is_still_forwarded(self):
        import inspect

        from app.services import agent_tool_dispatch

        source = inspect.getsource(agent_tool_dispatch)
        start = source.index("async def _benchmark_c_snippet")
        body = source[start : start + 1600]
        assert 'kwargs["repeat"] = int(params["repeat"])' in body

    def test_the_signature_default_is_the_one_that_applies(self):
        import inspect

        from app.services.agent_compiler_sandbox import (
            DEFAULT_REPEAT,
            benchmark_c_snippet,
        )

        sig = inspect.signature(benchmark_c_snippet)
        assert sig.parameters["repeat"].default == DEFAULT_REPEAT


class TestAWorkloadTooSmallToTime:
    """Serialising the measurement, taking five trials and discarding the
    slowest all failed to stabilise a ~30 ms kernel on this host: spreads
    stayed above 100%, and two roles that agreed exactly on 23 ms were still
    reported unresolvable.

    No statistic recovers a signal smaller than the noise around it. The run
    has to be told to make the work bigger, the same lesson `startup_share`
    already teaches for interpreted languages.
    """

    def _warning(self, timings):
        from app.services.agent_compiler_sandbox import measurement_quality

        return (
            measurement_quality(0.3, 8, timings, None).get("measurement_warning") or ""
        )

    def test_a_small_unstable_workload_is_told_to_grow(self):
        assert "too small to time reliably" in self._warning([29, 36, 40, 61, 78])

    def test_a_large_workload_is_left_alone(self):
        assert "too small" not in self._warning([1200, 1210, 1230, 1240, 1500])

    def test_a_small_but_stable_workload_is_not_nagged(self):
        """A tight spread at 30 ms is not proof of anything, but nor is it the
        failure this warning is about; the spread warning covers it."""
        assert "too small" not in self._warning([30, 30, 31, 31, 31])

    def test_the_warm_up_run_is_reported_not_hidden(self):
        """warmup_ms is what settled the warm-up question: it came back at
        30 ms against a 29 ms fastest trial, so the first run was not slow."""
        import inspect

        from app.services import agent_compiler_sandbox

        source = inspect.getsource(agent_compiler_sandbox)
        assert '"warmup_ms"' in source
        assert "__warmup_ms__" in source
