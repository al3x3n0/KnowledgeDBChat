from types import SimpleNamespace

from app.services.autonomous_rnd_trajectory_service import (
    AutonomousRnDTrajectoryAdapter,
)


def test_projects_persisted_job_and_experiment_run_without_inventing_success():
    job = SimpleNamespace(
        id="job-1",
        status="completed",
        job_type="analysis",
        iteration=4,
        tool_calls_used=2,
        llm_calls_used=1,
        error=None,
        results={
            "evaluation_outcome": {
                "status": "failed",
                "claims": [{"id": "claim-1", "evidence_ids": ["finding-1"]}],
                "decision": {"classification": "no_regression"},
            },
            "findings": [
                {
                    "id": "finding-1",
                    "type": "benchmark_output",
                    "title": "Controlled benchmark",
                }
            ],
            "execution_strategy": {
                "step_events": [
                    {
                        "type": "step_completed",
                        "tool": "run_experiment",
                        "iteration": 2,
                    }
                ]
            },
            "experiment_run": {
                "ran": True,
                "runs": [
                    {"command": "compile", "ok": True},
                    {"command": "benchmark", "ok": False},
                ],
            },
        },
        output_artifacts=[{"type": "compiler_logs", "id": "artifact-1"}],
        execution_log=[],
    )
    run = SimpleNamespace(
        id="run-1",
        results={"measurement_summary": {"runtime_ms": 24.5}},
        config={
            "scientific_validation": {
                "measurement_summary": {
                    "repeat_count": 5,
                    "artifact_inventory": ["benchmark_output"],
                }
            }
        },
    )

    outcome = AutonomousRnDTrajectoryAdapter().build_outcome(job, experiment_runs=[run])

    assert outcome["status"] == "completed"
    assert outcome["schema_version"] == 3
    assert outcome["generated_by"] == "trajectory_adapter"
    assert outcome["evidence"][0]["id"] == "finding-1"
    assert {artifact["kind"] for artifact in outcome["artifacts"]} == {
        "compiler_logs",
        "benchmark_output",
    }
    assert outcome["actions"] == [
        {
            "tool": "run_experiment",
            "iteration": 2,
            "type": "step_completed",
        }
    ]
    assert outcome["experiment"]["repeat_count"] == 5
    assert outcome["experiment"]["all_commands_ok"] is False
    assert outcome["metrics"]["runtime_ms"] == 24.5
    assert outcome["trajectory"]["experiment_run_ids"] == ["run-1"]


def test_compact_action_ledger_drops_payloads_and_keeps_safe_provenance():
    actions = [
        {
            "action": {
                "tool": "run_custom_tool",
                "params": {
                    "tool_name": "remote-reviewer",
                    "inputs": {"api_key": "must-not-persist"},
                },
            },
            "result": {
                "success": True,
                "status": "completed",
                "tool_name": "remote-reviewer",
                "tool_type": "external_agent",
                "output": {
                    "output": {"answer": "raw response must not persist"},
                    "provenance": {
                        "external_agent_id": "agent-1",
                        "external_agent_name": "Remote Reviewer",
                        "endpoint_origin": "https://agents.example",
                        "capability": "review",
                        "request_id": "request-1",
                        "received_at": "2026-07-28T12:00:00+00:00",
                        "response_sha256": "abc123",
                        "response_bytes": 42,
                        "execution_time_ms": 15,
                        "unexpected_secret": "must-not-persist",
                    },
                },
            },
            "iteration": 2,
            "node": "act",
            "step_id": "step-2",
        }
    ]

    ledger = AutonomousRnDTrajectoryAdapter().compact_action_ledger(actions)

    assert ledger == [
        {
            "tool": "run_custom_tool",
            "success": True,
            "status": "completed",
            "iteration": 2,
            "node": "act",
            "step_id": "step-2",
            "delegated_tool_name": "remote-reviewer",
            "tool_type": "external_agent",
            "external_agent_provenance": {
                "external_agent_id": "agent-1",
                "external_agent_name": "Remote Reviewer",
                "endpoint_origin": "https://agents.example",
                "capability": "review",
                "request_id": "request-1",
                "received_at": "2026-07-28T12:00:00+00:00",
                "response_sha256": "abc123",
                "response_bytes": 42,
                "execution_time_ms": 15,
            },
        }
    ]
    assert "must-not-persist" not in repr(ledger)


def test_external_agent_provenance_becomes_unverified_evidence_not_a_claim():
    job = SimpleNamespace(
        id="job-3",
        status="completed",
        job_type="research",
        iteration=2,
        tool_calls_used=1,
        llm_calls_used=1,
        error=None,
        results={
            "actions": [
                {
                    "tool": "run_custom_tool",
                    "success": True,
                    "external_agent_provenance": {
                        "external_agent_id": "agent-1",
                        "external_agent_name": "Remote Reviewer",
                        "endpoint_origin": "https://agents.example",
                        "capability": "review",
                        "request_id": "request-1",
                        "response_sha256": "abc123",
                    },
                }
            ]
        },
        output_artifacts=[],
        execution_log=[],
    )

    outcome = AutonomousRnDTrajectoryAdapter().build_outcome(job)

    assert outcome["claims"] == []
    assert outcome["evidence"] == [
        {
            "id": "external-agent:request-1",
            "kind": "external_agent_response",
            "record_origin": "external_agent_gateway",
            "verification_status": "unverified",
            "verification_reason": "No explicit local verification link was recorded.",
            "source": "https://agents.example",
            "external_agent_id": "agent-1",
            "external_agent_name": "Remote Reviewer",
            "capability": "review",
            "request_id": "request-1",
            "response_sha256": "abc123",
        }
    ]
    assert outcome["verification_plan"]["task_count"] == 1


def test_compops_provenance_becomes_external_system_evidence():
    job = SimpleNamespace(
        id="job-compops",
        status="completed",
        job_type="research",
        iteration=1,
        tool_calls_used=1,
        llm_calls_used=1,
        error=None,
        results={
            "actions": [
                {
                    "tool": "run_custom_tool",
                    "success": True,
                    "external_agent_provenance": {
                        "external_agent_id": "compops-connection-1",
                        "external_agent_name": "CompOps",
                        "provider_type": "compops",
                        "endpoint_origin": "https://compops.example",
                        "capability": "compops.studies.report",
                        "request_id": "research-request-1",
                        "response_sha256": "def456",
                        "remote_references": {"study_id": "study-1"},
                    },
                }
            ]
        },
        output_artifacts=[],
        execution_log=[],
    )

    outcome = AutonomousRnDTrajectoryAdapter().build_outcome(job)

    external = next(
        item
        for item in outcome["evidence"]
        if item["kind"] == "external_system_response"
    )
    assert external["id"] == "external-system:research-request-1"
    assert external["external_system_type"] == "compops"
    assert external["remote_references"] == {"study_id": "study-1"}
    assert external["verification_status"] == "unverified"
    assert outcome["verification_plan"]["task_count"] == 1


def test_explicit_local_verification_link_promotes_external_evidence():
    job = SimpleNamespace(
        id="job-4",
        status="completed",
        job_type="research",
        iteration=3,
        tool_calls_used=2,
        llm_calls_used=1,
        error=None,
        results={
            "actions": [
                {
                    "tool": "run_custom_tool",
                    "success": True,
                    "external_agent_provenance": {
                        "external_agent_id": "agent-1",
                        "request_id": "request-1",
                        "response_sha256": "abc123",
                    },
                }
            ],
            "findings": [
                {
                    "id": "local-benchmark-1",
                    "type": "benchmark_output",
                }
            ],
            "structured_output": {
                "verification_links": [
                    {
                        "external_evidence_id": "external-agent:request-1",
                        "verdict": "supports",
                        "local_evidence_ids": ["local-benchmark-1"],
                        "artifact_kinds": ["compiler_logs"],
                    }
                ]
            },
            "experiment_run": {
                "ran": True,
                "runs": [{"ok": True}, {"ok": True}],
            },
        },
        output_artifacts=[{"id": "artifact-1", "type": "compiler_logs"}],
        execution_log=[],
    )

    outcome = AutonomousRnDTrajectoryAdapter().build_outcome(job)

    external = next(
        item
        for item in outcome["evidence"]
        if item["kind"] == "external_agent_response"
    )
    assert external["verification_status"] == "verified"
    assert outcome["evidence_verification"]["status_counts"]["verified"] == 1
    assert outcome["verification_plan"]["task_count"] == 0


def test_empty_trajectory_does_not_gain_claims_or_execution_success():
    job = SimpleNamespace(
        id="job-2",
        status="completed",
        job_type="research",
        iteration=1,
        tool_calls_used=0,
        llm_calls_used=1,
        error=None,
        results={"summary": "Everything worked."},
        output_artifacts=[],
        execution_log=[],
    )

    outcome = AutonomousRnDTrajectoryAdapter().build_outcome(job)

    assert outcome["claims"] == []
    assert outcome["evidence"] == []
    assert outcome["experiment"]["repeat_count"] == 0
    assert outcome["experiment"]["all_commands_ok"] is False
