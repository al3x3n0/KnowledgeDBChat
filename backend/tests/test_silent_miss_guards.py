"""Checks that could not tell must not report the reassuring answer.

Three defects in one day shared a shape: a check or a query looked in the
wrong place and returned "nothing found" where the honest answer was "I could
not tell". The null control compared NaN with == and never fired; a probe
discarded its own stderr and blamed the simulator for a full disk; a filter
searched three fields of a record and reported an absence of evidence that was
sitting in the next field along.

These are the two remaining instances of that shape, both found by sweeping
for it rather than by hitting them.
"""

from app.services.agent_compiler_sandbox import measurement_quality


class TestAnUnreadableLoadIsNotAQuietMachine:
    """The probe fell back to `echo 0` when /proc/loadavg could not be read,
    and 0 classifies as quiet -- so an unreadable probe blessed a timing taken
    on a saturated host as taken on an idle one. That is the single thing this
    sampling exists to prevent."""

    def test_an_unknown_load_claims_no_environment(self):
        quality = measurement_quality(None, None, [10, 11])

        assert "measurement_environment" not in quality
        assert "load_per_cpu" not in quality

    def test_a_genuinely_idle_machine_is_still_called_quiet(self):
        """The control: the honest answer must survive the fix."""
        quality = measurement_quality(0.4, 8, [10, 11])

        assert quality["measurement_environment"] == "quiet"

    def test_a_loaded_machine_is_still_called_saturated(self):
        assert (
            measurement_quality(16.0, 8, [10, 11])["measurement_environment"]
            == "saturated"
        )

    def test_the_probe_emits_a_non_numeric_sentinel(self):
        """A numeric fallback is the bug: it parses, and a parsed number is
        indistinguishable from a measured one."""
        import inspect

        from app.services import agent_compiler_sandbox

        source = inspect.getsource(agent_compiler_sandbox)

        assert "|| echo unknown" in source
        assert "/proc/loadavg 2>/dev/null'\n            ' || echo 0" not in source
