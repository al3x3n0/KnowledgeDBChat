import asyncio
import hashlib
import uuid
from copy import deepcopy

from sqlalchemy import select

from app.models.agent_job import AgentJob, AgentJobStatus
from app.models.document import DocumentSource
from app.models.experiment import ExperimentPlan, ExperimentRun
from app.models.notification import Notification, NotificationType
from app.models.research_note import ResearchNote
from app.models.tool_audit import ToolExecutionAudit
from app.services.autonomous_rnd_verification_reconciliation_service import (
    autonomous_rnd_verification_reconciliation_service,
)


def _passing_regression_job(test_user, index):
    return AgentJob(
        name=f"Compiler evaluation trial {index}",
        goal="Reproduce compiler regression",
        job_type="analysis",
        user_id=test_user.id,
        status=AgentJobStatus.COMPLETED.value,
        progress=100,
        iteration=3,
        max_iterations=10,
        max_tool_calls=10,
        max_llm_calls=10,
        max_runtime_minutes=10,
        results={
            "evaluation_outcome": {
                "claims": [
                    {
                        "id": f"claim-{index}",
                        "evidence_ids": [f"evidence-{index}"],
                    }
                ],
                "evidence": [
                    {
                        "id": f"evidence-{index}",
                        "kind": "benchmark_output",
                    }
                ],
                "experiment": {
                    "repeat_count": 3,
                    "all_commands_ok": True,
                },
            },
            "execution_strategy": {
                "step_events": [
                    {
                        "type": "step_completed",
                        "tool": "run_experiment",
                        "iteration": 2,
                    }
                ]
            },
        },
        output_artifacts=[
            {"type": "compiler_logs"},
            {"type": "ir_or_codegen_artifacts"},
            {"type": "benchmark_output"},
        ],
    )


def test_lists_builtin_autonomous_rnd_eval_suites(client, auth_headers):
    response = client.get("/api/v1/autonomous-rnd-evals/suites", headers=auth_headers)

    assert response.status_code == 200
    payload = response.json()
    assert payload["suites"][0]["id"] == "compiler_research_v1"
    assert payload["suites"][0]["task_count"] == 3


def test_grades_persisted_agent_jobs_as_trials(
    client, auth_headers, db_session, test_user
):
    jobs = [_passing_regression_job(test_user, index) for index in range(3)]

    async def _seed():
        db_session.add_all(jobs)
        await db_session.commit()
        for job in jobs:
            await db_session.refresh(job)

    asyncio.get_event_loop().run_until_complete(_seed())

    response = client.post(
        "/api/v1/autonomous-rnd-evals/grade-jobs",
        headers=auth_headers,
        json={
            "suite_id": "compiler_research_v1",
            "trials": [
                {
                    "task_id": "compiler_regression_reproduce",
                    "job_ids": [str(job.id) for job in jobs],
                }
            ],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["evaluated_job_count"] == 3
    assert payload["report"]["tasks"][0]["passed_count"] == 3
    assert payload["report"]["tasks"][0]["pass_pow_k"] is True
    assert payload["report"]["pass_at_k"] == 0.333333
    assert payload["report"]["pass_pow_k"] == 0.333333


def test_rejects_unknown_eval_task(client, auth_headers):
    response = client.post(
        "/api/v1/autonomous-rnd-evals/grade-jobs",
        headers=auth_headers,
        json={
            "suite_id": "compiler_research_v1",
            "trials": [
                {
                    "task_id": "not-a-task",
                    "job_ids": ["00000000-0000-0000-0000-000000000001"],
                }
            ],
        },
    )

    assert response.status_code == 400
    assert "Unknown task ids" in response.json()["detail"]


def _verification_parent_job(test_user, request_id):
    return AgentJob(
        name=f"External evidence job {request_id}",
        goal="Review external evidence",
        job_type="analysis",
        user_id=test_user.id,
        status=AgentJobStatus.COMPLETED.value,
        progress=100,
        iteration=1,
        max_iterations=2,
        max_tool_calls=2,
        max_llm_calls=1,
        max_runtime_minutes=5,
        chain_depth=0,
        results={
            "actions": [
                {
                    "tool": "run_custom_tool",
                    "success": True,
                    "external_agent_provenance": {
                        "external_agent_id": "agent-1",
                        "capability": "general.review",
                        "request_id": request_id,
                        "response_sha256": "abc123",
                    },
                }
            ]
        },
        output_artifacts=[],
    )


def test_launches_verification_task_idempotently_after_explicit_approval(
    client, auth_headers, db_session, test_user
):
    request_id = "launch-request-1"
    parent = _verification_parent_job(test_user, request_id)
    note = ResearchNote(
        user_id=test_user.id,
        title="Verification note",
        content_markdown="# Verification",
    )
    source = DocumentSource(
        name=f"verification-source-{uuid.uuid4()}",
        source_type="file",
        config={},
        is_active=True,
    )

    async def _seed():
        db_session.add_all([parent, note, source])
        await db_session.commit()
        await db_session.refresh(parent)
        await db_session.refresh(note)
        await db_session.refresh(source)

    asyncio.get_event_loop().run_until_complete(_seed())
    evidence_id = f"external-agent:{request_id}"
    task_id = f"verify-{hashlib.sha256(evidence_id.encode()).hexdigest()[:12]}"
    request = {
        "approval_confirmed": True,
        "approval_note": "Approved bounded local verification",
        "research_note_id": str(note.id),
        "source_id": str(source.id),
        "sandbox_profile_id": "scientific-generic-sandbox",
        "commands": ["pytest -q"],
        "repeat_count": 2,
        "timeout_seconds": 30,
        "max_runtime_minutes": 2,
        "budget_limit": 1.0,
        "start_immediately": False,
    }
    path = (
        f"/api/v1/autonomous-rnd-evals/jobs/{parent.id}"
        f"/verification-tasks/{task_id}/launch"
    )
    outcome_path = f"/api/v1/autonomous-rnd-evals/jobs/{parent.id}/outcome"

    before_launch = client.get(outcome_path, headers=auth_headers)
    over_budget = {**request, "budget_limit": 26.0}
    rejected = client.post(path, headers=auth_headers, json=over_budget)
    first = client.post(path, headers=auth_headers, json=request)
    second = client.post(path, headers=auth_headers, json=request)
    after_launch = client.get(outcome_path, headers=auth_headers)

    assert before_launch.status_code == 200
    before_lifecycle = before_launch.json()["verification_lifecycle"]
    assert before_lifecycle["task_count"] == 1
    assert before_lifecycle["tasks"][0]["launch_status"] == "not_launched"
    assert [event["event_type"] for event in before_lifecycle["timeline"]] == [
        "proposal_created"
    ]
    assert rejected.status_code == 400
    assert "budget_limit exceeds" in rejected.json()["detail"]
    assert first.status_code == 200
    assert second.status_code == 200
    first_payload = first.json()
    second_payload = second.json()
    assert first_payload["created"] is True
    assert first_payload["queued"] is False
    assert first_payload["status"] == "planned"
    assert second_payload["created"] is False
    assert second_payload["experiment_plan_id"] == first_payload["experiment_plan_id"]
    assert second_payload["agent_job_id"] == first_payload["agent_job_id"]
    assert after_launch.status_code == 200
    launched_task = after_launch.json()["verification_lifecycle"]["tasks"][0]
    assert launched_task["launch_status"] == "planned"
    assert launched_task["approval_status"] == "approved"
    assert launched_task["budget"]["repeat_count"] == 2
    assert launched_task["agent_job_id"] == first_payload["agent_job_id"]
    launched_event_types = {
        event["event_type"]
        for event in after_launch.json()["verification_lifecycle"]["timeline"]
    }
    assert {
        "proposal_created",
        "approval_recorded",
        "verification_launched",
    }.issubset(launched_event_types)

    async def _load_created():
        return (
            await db_session.get(
                ExperimentPlan, uuid.UUID(first_payload["experiment_plan_id"])
            ),
            await db_session.get(
                ExperimentRun, uuid.UUID(first_payload["experiment_run_id"])
            ),
            await db_session.get(AgentJob, uuid.UUID(first_payload["agent_job_id"])),
            await db_session.get(
                ToolExecutionAudit, uuid.UUID(first_payload["audit_id"])
            ),
        )

    plan, run, job, audit = asyncio.get_event_loop().run_until_complete(_load_created())
    assert plan.generator == "autonomous_rnd_verification_planner"
    assert run.config["repeat_count"] == 2
    assert job.config["deterministic_runner"] == "experiment_runner"
    assert job.config["unsafe_code_exec_backend"] == "docker"
    assert job.config["unsafe_code_exec_docker_image"] == "python:3.11-slim"
    assert job.max_runtime_minutes == 2
    assert audit.approval_status == "approved"
    assert audit.tool_input["command_count"] == 1

    async def _reconcile_success():
        job.results = {
            "experiment_run": {
                "ran": True,
                "repeat_count": 2,
                "runs": [
                    {
                        "command": "pytest -q",
                        "repeat_index": 1,
                        "ok": True,
                        "stdout": "sensitive local output",
                    },
                    {
                        "command": "pytest -q",
                        "repeat_index": 2,
                        "ok": True,
                        "stdout": "sensitive local output",
                    },
                ],
            }
        }
        run.status = "succeeded"
        reconciled = await autonomous_rnd_verification_reconciliation_service.reconcile(
            verification_job=job,
            db=db_session,
        )
        reconciled_again = (
            await autonomous_rnd_verification_reconciliation_service.reconcile(
                verification_job=job,
                db=db_session,
            )
        )
        await db_session.commit()
        await db_session.refresh(parent)
        notifications = list(
            (
                await db_session.execute(
                    select(Notification).where(
                        Notification.notification_type
                        == NotificationType.AUTONOMOUS_RND_VERIFICATION_UPDATE,
                        Notification.related_entity_id == parent.id,
                    )
                )
            )
            .scalars()
            .all()
        )
        return reconciled, reconciled_again, notifications

    (
        reconciled,
        reconciled_again,
        notifications,
    ) = asyncio.get_event_loop().run_until_complete(_reconcile_success())
    after_reconciliation = client.get(outcome_path, headers=auth_headers)
    external_evidence = next(
        item
        for item in parent.results["evaluation_outcome"]["evidence"]
        if item["kind"] == "external_agent_response"
    )
    assert reconciled is True
    assert reconciled_again is True
    assert len(notifications) == 1
    assert notifications[0].data["verification_task_id"] == task_id
    assert notifications[0].data["verification_status"] == "verified"
    assert notifications[0].action_url.endswith(
        f"job={parent.id}&verification_task={task_id}"
    )
    assert "commands" not in notifications[0].data
    assert external_evidence["verification_status"] == "verified"
    assert parent.results["evaluation_outcome"]["verification_plan"]["task_count"] == 0
    assert "sensitive local output" not in repr(parent.results)
    reconciled_task = after_reconciliation.json()["verification_lifecycle"]["tasks"][0]
    assert reconciled_task["launch_status"] == "succeeded"
    assert reconciled_task["evidence_status"] == "verified"
    assert reconciled_task["reconciliation_status"] == "support_recorded"
    assert reconciled_task["reconciliation_recorded_at"]
    reconciled_timeline = after_reconciliation.json()["verification_lifecycle"][
        "timeline"
    ]
    assert [event["at"] for event in reconciled_timeline] == sorted(
        event["at"] for event in reconciled_timeline
    )
    assert {
        "execution_completed",
        "reconciliation_recorded",
    }.issubset(event["event_type"] for event in reconciled_timeline)

    signed_response = client.post(
        f"/api/v1/autonomous-rnd-evals/jobs/{parent.id}/verification-audit-snapshot",
        headers=auth_headers,
        json={"task_id": task_id, "status": "verified"},
    )
    assert signed_response.status_code == 200
    signed = signed_response.json()
    assert signed["snapshot"]["summary"]["task_count"] == 1
    assert signed["snapshot"]["filters"]["task_id"] == task_id
    registry_id = signed["snapshot"]["registry_id"]
    assert len(signed["integrity"]["sha256"]) == 64
    assert signed["integrity"]["signature_algorithm"] == "ed25519"
    assert signed["integrity"]["signature_encoding"] == "hex"
    assert len(signed["integrity"]["signature"]) == 128
    assert len(signed["integrity"]["public_key"]) == 64
    assert "sensitive local output" not in repr(signed)
    assert "commands" not in repr(signed)

    registry_response = client.get(
        f"/api/v1/autonomous-rnd-evals/verification-audit-snapshots/{registry_id}",
        headers=auth_headers,
    )
    registry_list_response = client.get(
        f"/api/v1/autonomous-rnd-evals/jobs/{parent.id}/verification-audit-snapshots",
        headers=auth_headers,
    )
    keys_response = client.get(
        "/api/v1/autonomous-rnd-evals/verification-audit-keys",
        headers=auth_headers,
    )
    assert registry_response.status_code == 200
    assert registry_response.json() == signed
    assert registry_list_response.status_code == 200
    registry_items = registry_list_response.json()["items"]
    assert registry_items[0]["registry_id"] == registry_id
    assert registry_items[0]["sha256"] == signed["integrity"]["sha256"]
    assert keys_response.status_code == 200
    signing_key = keys_response.json()["keys"][0]
    assert signing_key["status"] == "active"
    assert signing_key["public_key"] == signed["integrity"]["public_key"]
    assert len(signing_key["fingerprint_sha256"]) == 64

    verified_response = client.post(
        "/api/v1/autonomous-rnd-evals/verification-audit-snapshots/verify",
        headers=auth_headers,
        json=signed,
    )
    assert verified_response.status_code == 200
    assert verified_response.json()["valid"] is True

    tampered = deepcopy(signed)
    tampered["snapshot"]["job_status"] = "tampered"
    tampered_response = client.post(
        "/api/v1/autonomous-rnd-evals/verification-audit-snapshots/verify",
        headers=auth_headers,
        json=tampered,
    )
    assert tampered_response.status_code == 200
    assert tampered_response.json() == {
        "valid": False,
        "reason": "sha256_mismatch",
        "job_id": None,
        "registry_id": None,
        "sha256": None,
        "key_id": None,
    }


def test_verification_launch_requires_literal_approval_confirmation(
    client, auth_headers
):
    response = client.post(
        "/api/v1/autonomous-rnd-evals/jobs/"
        "00000000-0000-0000-0000-000000000001/"
        "verification-tasks/verify-example/launch",
        headers=auth_headers,
        json={
            "approval_confirmed": False,
            "approval_note": "Not approved",
            "research_note_id": "00000000-0000-0000-0000-000000000002",
            "source_id": "00000000-0000-0000-0000-000000000003",
            "sandbox_profile_id": "scientific-generic-sandbox",
            "commands": ["pytest -q"],
            "repeat_count": 2,
            "timeout_seconds": 30,
            "max_runtime_minutes": 2,
            "budget_limit": 1.0,
        },
    )

    assert response.status_code == 422


def _failing_regression_job(test_user, index):
    """Same shape as a passing trial, minus the evidence the graders require."""
    job = _passing_regression_job(test_user, index)
    job.results["evaluation_outcome"]["claims"] = []
    job.results["evaluation_outcome"]["evidence"] = []
    job.output_artifacts = []
    return job


def _grade_jobs(client, auth_headers, jobs, *, persist=False, label=None):
    payload = {
        "suite_id": "compiler_research_v1",
        "trials": [
            {
                "task_id": "compiler_regression_reproduce",
                "job_ids": [str(job.id) for job in jobs],
            }
        ],
        "persist": persist,
    }
    if label:
        payload["label"] = label
    return client.post(
        "/api/v1/autonomous-rnd-evals/grade-jobs",
        headers=auth_headers,
        json=payload,
    )


def test_persists_graded_run_and_compares_candidate_against_baseline(
    client, auth_headers, db_session, test_user
):
    passing = [_passing_regression_job(test_user, index) for index in range(3)]
    failing = [_failing_regression_job(test_user, 10 + index) for index in range(3)]

    async def _seed():
        db_session.add_all(passing + failing)
        await db_session.commit()
        for job in passing + failing:
            await db_session.refresh(job)

    asyncio.get_event_loop().run_until_complete(_seed())

    baseline_response = _grade_jobs(
        client, auth_headers, passing, persist=True, label="baseline run"
    )
    assert baseline_response.status_code == 200
    baseline_id = baseline_response.json()["run_id"]
    assert baseline_id is not None

    promote = client.post(
        f"/api/v1/autonomous-rnd-evals/runs/{baseline_id}/baseline",
        headers=auth_headers,
    )
    assert promote.status_code == 200
    assert promote.json()["is_baseline"] is True
    assert promote.json()["label"] == "baseline run"

    candidate_response = _grade_jobs(client, auth_headers, failing, persist=True)
    candidate_id = candidate_response.json()["run_id"]

    comparison = client.get(
        f"/api/v1/autonomous-rnd-evals/runs/{candidate_id}/comparison",
        headers=auth_headers,
    )
    assert comparison.status_code == 200
    payload = comparison.json()["comparison"]
    assert payload["baseline_run_id"] == baseline_id
    assert payload["has_regression"] is True
    assert payload["regressed_task_ids"] == ["compiler_regression_reproduce"]
    assert payload["metrics"]["pass_pow_k"]["delta"] < 0


def test_grade_jobs_does_not_persist_a_run_by_default(
    client, auth_headers, db_session, test_user
):
    jobs = [_passing_regression_job(test_user, 20 + index) for index in range(3)]

    async def _seed():
        db_session.add_all(jobs)
        await db_session.commit()
        for job in jobs:
            await db_session.refresh(job)

    asyncio.get_event_loop().run_until_complete(_seed())

    response = _grade_jobs(client, auth_headers, jobs)

    assert response.status_code == 200
    assert response.json()["run_id"] is None
    listing = client.get("/api/v1/autonomous-rnd-evals/runs", headers=auth_headers)
    assert listing.status_code == 200
    assert listing.json()["runs"] == []


def test_comparison_requires_a_baseline_for_the_suite(
    client, auth_headers, db_session, test_user
):
    jobs = [_passing_regression_job(test_user, 30 + index) for index in range(3)]

    async def _seed():
        db_session.add_all(jobs)
        await db_session.commit()
        for job in jobs:
            await db_session.refresh(job)

    asyncio.get_event_loop().run_until_complete(_seed())

    run_id = _grade_jobs(client, auth_headers, jobs, persist=True).json()["run_id"]

    response = client.get(
        f"/api/v1/autonomous-rnd-evals/runs/{run_id}/comparison",
        headers=auth_headers,
    )

    assert response.status_code == 404
    assert "No baseline run is set" in response.json()["detail"]


def test_eval_runs_are_scoped_to_their_owner(
    client, admin_headers, auth_headers, db_session, test_user
):
    jobs = [_passing_regression_job(test_user, 40 + index) for index in range(3)]

    async def _seed():
        db_session.add_all(jobs)
        await db_session.commit()
        for job in jobs:
            await db_session.refresh(job)

    asyncio.get_event_loop().run_until_complete(_seed())

    run_id = _grade_jobs(client, auth_headers, jobs, persist=True).json()["run_id"]

    response = client.get(
        f"/api/v1/autonomous-rnd-evals/runs/{run_id}", headers=admin_headers
    )

    assert response.status_code == 404
