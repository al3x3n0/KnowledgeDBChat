"""A wall-clock timing must say what the machine was doing while it was taken.

On this host a competing workload ran ~285% CPU for an entire session, an
orphaned container raised load for an hour unnoticed, and identical gem5 runs
varied from 10s to over 150s. Nothing in the pipeline could tell that any of
those numbers were worthless, and the agent had no way to find out.

Only the wall-clock tool needs this. A simulated measurement reports modelled
cycles, which are the same on a busy machine as on a quiet one -- contention
changes how long the simulation takes, not what it reports.
"""

from __future__ import annotations

from app.services.agent_compiler_sandbox import measurement_quality


def test_a_quiet_machine_produces_no_warning():
    quality = measurement_quality(1.2, 8, [100, 101, 102])

    assert quality["measurement_environment"] == "quiet"
    assert "measurement_warning" not in quality


def test_a_busy_machine_is_reported_as_busy():
    quality = measurement_quality(6.0, 8, [100, 102])

    assert quality["measurement_environment"] == "busy"
    assert quality["load_per_cpu"] == 0.75
    assert "treat small differences as noise" in quality["measurement_warning"]


def test_a_saturated_machine_says_the_timing_reflects_competition():
    """The honest statement: this is not a property of the code."""
    quality = measurement_quality(13.0, 8, [100, 101])

    assert quality["measurement_environment"] == "saturated"
    assert "competition for the machine" in quality["measurement_warning"]


def test_unstable_trials_are_caught_even_on_a_quiet_machine():
    """Load and spread catch different things; a quiet host can still wobble."""
    quality = measurement_quality(0.4, 8, [100, 180])

    assert quality["measurement_environment"] == "quiet"
    assert quality["trial_spread"] == 0.8
    assert "80%" in quality["measurement_warning"]
    assert "not evidence" in quality["measurement_warning"]


def test_the_load_of_this_host_during_the_session_would_have_warned():
    """Load 8.67 across 8 CPUs, measured live while building this."""
    quality = measurement_quality(8.67, 8, [806, 704, 819, 670, 744])

    assert quality["measurement_environment"] == "busy"
    assert quality["trial_spread"] >= 0.2
    assert quality["measurement_warning"]


def test_a_single_trial_has_no_spread_to_report():
    """It reports no spread, and now says so: silence read as approval."""
    quality = measurement_quality(0.5, 8, [100])

    assert "trial_spread" not in quality
    assert quality["single_trial"] is True
    assert "one trial" in quality["measurement_warning"].lower()


def test_missing_samples_are_reported_as_nothing_rather_than_as_quiet():
    """An unknown environment must not read as a good one."""
    assert measurement_quality(None, None, None) == {}
    assert (
        measurement_quality(None, 8, [100, 101]).get("measurement_environment") is None
    )


def test_zero_timings_do_not_divide_by_zero():
    quality = measurement_quality(0.5, 8, [0, 0])

    assert "trial_spread" not in quality


def test_the_benchmark_tool_parses_the_samples_it_emits():
    """The markers the sandbox script echoes must survive parsing.

    Asserted against the same string handling the tool uses, because a marker
    renamed on one side and not the other would silently return to the old
    behaviour of reporting nothing about the environment.
    """
    stdout = "\n".join(
        [
            "__elapsed_ms__ 120",
            "__elapsed_ms__ 118",
            "ns_total=42",
            "__loadavg__ 8.67",
            "__cpus__ 8",
        ]
    )

    timings, load, cpus, program = [], None, None, []
    for line in stdout.splitlines():
        if line.startswith("__elapsed_ms__ "):
            timings.append(int(line.split()[1]))
        elif line.startswith("__loadavg__ "):
            load = float(line.split()[1])
        elif line.startswith("__cpus__ "):
            cpus = int(line.split()[1])
        else:
            program.append(line)

    assert timings == [120, 118]
    assert (load, cpus) == (8.67, 8)
    assert program == ["ns_total=42"], "markers must not leak into program output"
    assert measurement_quality(load, cpus, timings)["measurement_environment"] == "busy"


def test_the_sandbox_script_emits_both_markers():
    """A silent regression here turns the whole check off."""
    import inspect

    from app.services import agent_compiler_sandbox

    source = inspect.getsource(agent_compiler_sandbox.benchmark_c_snippet)

    assert "__loadavg__" in source
    assert "__cpus__" in source
    assert "/proc/loadavg" in source


def test_a_single_trial_says_why_it_is_not_enough():
    """A live run benchmarked once, was refused by a contract wanting error
    bars, and had nothing in the tool's output telling it what to change."""
    quality = measurement_quality(2.4, 8, [44])

    assert quality["single_trial"] is True
    assert "trial_spread" not in quality
    assert "repeat=5" in quality["measurement_warning"]
