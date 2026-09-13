import inspect

from app.services.agent_experiment_runner_service import (
    AgentExperimentRunnerService,
    _build_repeated_command_schedule,
)


def test_repeated_command_schedule_tracks_full_experiment_repetitions():
    schedule = _build_repeated_command_schedule(
        ["pytest -q", "python benchmark.py"],
        3,
    )

    assert schedule == [
        (1, "pytest -q"),
        (1, "python benchmark.py"),
        (2, "pytest -q"),
        (2, "python benchmark.py"),
        (3, "pytest -q"),
        (3, "python benchmark.py"),
    ]


def test_verification_reconciliation_runs_only_on_experiment_execution_path():
    experiment_runner = inspect.getsource(
        AgentExperimentRunnerService.run_experiment_runner
    )
    plan_generator = inspect.getsource(
        AgentExperimentRunnerService.run_experiment_plan_generate
    )

    assert "verification_reconciliation_service" in experiment_runner
    assert "verification_reconciliation_service" not in plan_generator
