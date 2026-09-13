import asyncio
from uuid import UUID

import pytest

from app.models.agent_job import AgentJob, AgentJobCheckpoint, AgentJobStatus
from app.models.domain_research_profile import DomainResearchProfile
from app.models.research_portfolio import ResearchPortfolio


def test_pause_action_persists_operator_intervention(
    client,
    db_session,
    test_user,
    auth_headers,
):
    job = AgentJob(
        name="Intervention Logging Job",
        goal="Verify operator intervention persistence",
        job_type="research",
        user_id=test_user.id,
        status=AgentJobStatus.RUNNING.value,
        progress=42,
        iteration=2,
        max_iterations=10,
        max_tool_calls=10,
        max_llm_calls=10,
        max_runtime_minutes=30,
        results={
            "execution_strategy": {
                "step_events": [],
            }
        },
    )

    async def _seed():
        db_session.add(job)
        await db_session.commit()
        await db_session.refresh(job)

    asyncio.get_event_loop().run_until_complete(_seed())

    response = client.post(
        f"/api/v1/agent-jobs/{job.id}/action",
        headers=auth_headers,
        json={
            "action": "pause",
            "checkpoint_note": "Paused for manual inspection",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == AgentJobStatus.PAUSED.value

    execution = (payload.get("results") or {}).get("execution_strategy") or {}
    interventions = execution.get("operator_interventions") or []
    assert len(interventions) == 1
    assert interventions[0]["action"] == "pause"
    assert interventions[0]["note"] == "Paused for manual inspection"
    assert interventions[0]["job_status_before"] == AgentJobStatus.RUNNING.value
    assert interventions[0]["job_status_after"] == AgentJobStatus.PAUSED.value
    assert interventions[0]["outcome_status"] == "applied"
    assert interventions[0]["outcome_reason"] == "Job remains paused after intervention"


def test_cancel_action_records_intervention_and_syncs_follow_up(
    client,
    db_session,
    test_user,
    auth_headers,
    monkeypatch,
):
    synced_jobs: list[str] = []

    async def _sync_follow_up(_db, job):
        synced_jobs.append(str(job.id))

    monkeypatch.setattr(
        "app.modules.autonomy.application.job_action_lifecycle."
        "sync_follow_up_outcome_for_job",
        _sync_follow_up,
    )
    job = AgentJob(
        name="Cancelable job",
        goal="Verify cancellation lifecycle",
        job_type="research",
        user_id=test_user.id,
        status=AgentJobStatus.PENDING.value,
        max_iterations=10,
        max_tool_calls=10,
        max_llm_calls=10,
        max_runtime_minutes=30,
        results={"execution_strategy": {}},
    )

    async def _seed():
        db_session.add(job)
        await db_session.commit()
        await db_session.refresh(job)

    asyncio.get_event_loop().run_until_complete(_seed())

    response = client.post(
        f"/api/v1/agent-jobs/{job.id}/action",
        headers=auth_headers,
        json={"action": "cancel", "checkpoint_note": "Stop requested"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == AgentJobStatus.CANCELLED.value
    assert payload["completed_at"] is not None
    assert synced_jobs == [str(job.id)]
    execution = (payload.get("results") or {}).get("execution_strategy") or {}
    intervention = execution["operator_interventions"][-1]
    assert intervention["action"] == "cancel"
    assert intervention["note"] == "Stop requested"


def test_generate_summary_action_queues_completed_job(
    client,
    db_session,
    test_user,
    auth_headers,
    monkeypatch,
):
    queued_jobs: list[str] = []
    monkeypatch.setattr(
        "app.tasks.agent_job_tasks.generate_job_summary.delay",
        lambda job_id: queued_jobs.append(job_id),
    )
    job = AgentJob(
        name="Completed job",
        goal="Generate a summary",
        job_type="research",
        user_id=test_user.id,
        status=AgentJobStatus.COMPLETED.value,
        max_iterations=10,
        max_tool_calls=10,
        max_llm_calls=10,
        max_runtime_minutes=30,
        results={},
    )

    async def _seed():
        db_session.add(job)
        await db_session.commit()
        await db_session.refresh(job)

    asyncio.get_event_loop().run_until_complete(_seed())

    response = client.post(
        f"/api/v1/agent-jobs/{job.id}/action",
        headers=auth_headers,
        json={"action": "generate_summary"},
    )

    assert response.status_code == 200
    assert queued_jobs == [str(job.id)]


def test_resume_action_queues_paused_job_and_records_intervention(
    client,
    db_session,
    test_user,
    auth_headers,
    monkeypatch,
):
    queued_jobs: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "app.tasks.agent_job_tasks.execute_agent_job_task.delay",
        lambda job_id, user_id: queued_jobs.append((job_id, user_id)),
    )
    job = AgentJob(
        name="Paused job",
        goal="Resume ordinary execution",
        job_type="research",
        user_id=test_user.id,
        status=AgentJobStatus.PAUSED.value,
        max_iterations=10,
        max_tool_calls=10,
        max_llm_calls=10,
        max_runtime_minutes=30,
        results={"execution_strategy": {}},
    )

    async def _seed():
        db_session.add(job)
        await db_session.commit()
        await db_session.refresh(job)

    asyncio.get_event_loop().run_until_complete(_seed())

    response = client.post(
        f"/api/v1/agent-jobs/{job.id}/action",
        headers=auth_headers,
        json={"action": "resume", "checkpoint_note": "Continue execution"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == AgentJobStatus.PENDING.value
    assert queued_jobs == [(str(job.id), str(test_user.id))]
    execution = (payload.get("results") or {}).get("execution_strategy") or {}
    intervention = execution["operator_interventions"][-1]
    assert intervention["action"] == "resume"
    assert intervention["note"] == "Continue execution"
    assert intervention["metadata"]["approval_checkpoint_pending"] is False


def test_resume_action_rejects_execution_reconciliation_checkpoint(
    client,
    db_session,
    test_user,
    auth_headers,
    monkeypatch,
):
    queued_jobs: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "app.tasks.agent_job_tasks.execute_agent_job_task.delay",
        lambda job_id, user_id: queued_jobs.append((job_id, user_id)),
    )
    pending = {
        "checkpoint_type": "execution_reconciliation",
        "checkpoint_id": "reconcile:resume-guard",
        "iteration": 2,
        "action": {"tool": "write_file", "params": {"path": "src/fix.py"}},
    }
    job = AgentJob(
        name="Interrupted tool action",
        goal="Require an explicit reconciliation decision",
        job_type="coding",
        user_id=test_user.id,
        status=AgentJobStatus.PAUSED.value,
        max_iterations=10,
        max_tool_calls=10,
        max_llm_calls=10,
        max_runtime_minutes=30,
        results={
            "execution_strategy": {
                "approval_checkpoints": {"pending": pending, "events": []}
            },
            "approval_checkpoint": pending,
        },
    )

    async def _seed():
        db_session.add(job)
        await db_session.commit()
        await db_session.refresh(job)

    asyncio.get_event_loop().run_until_complete(_seed())

    response = client.post(
        f"/api/v1/agent-jobs/{job.id}/action",
        headers=auth_headers,
        json={"action": "resume"},
    )

    assert response.status_code == 400
    assert "explicit approve, edit, skip, or reject" in response.json()["detail"]
    assert queued_jobs == []


@pytest.mark.parametrize(
    ("action", "expected_status", "expected_step_status", "should_queue"),
    [
        ("edit", AgentJobStatus.PENDING.value, "in_progress", True),
        ("skip", AgentJobStatus.PENDING.value, "skipped", True),
        ("reject", AgentJobStatus.PAUSED.value, "failed", False),
    ],
)
def test_checkpoint_decision_actions_update_state_and_queue_policy(
    action,
    expected_status,
    expected_step_status,
    should_queue,
    client,
    db_session,
    test_user,
    auth_headers,
    monkeypatch,
):
    queued_jobs: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "app.tasks.agent_job_tasks.execute_agent_job_task.delay",
        lambda job_id, user_id: queued_jobs.append((job_id, user_id)),
    )
    pending = {
        "checkpoint_type": "approval",
        "checkpoint_id": f"approval:{action}",
        "iteration": 2,
        "plan_step_id": "step_1",
        "plan_step_index": 0,
        "action": {
            "tool": "search_documents",
            "purpose": "Original purpose",
            "params": {"query": "compiler"},
        },
    }
    job = AgentJob(
        name=f"Checkpoint {action}",
        goal="Apply an explicit checkpoint decision",
        job_type="research",
        user_id=test_user.id,
        status=AgentJobStatus.PAUSED.value,
        max_iterations=10,
        max_tool_calls=10,
        max_llm_calls=10,
        max_runtime_minutes=30,
        results={
            "execution_strategy": {
                "approval_checkpoints": {"pending": pending, "events": []}
            },
            "approval_checkpoint": pending,
        },
    )
    checkpoint = AgentJobCheckpoint(
        job=job,
        iteration=2,
        phase="awaiting_approval",
        state={
            "approval_checkpoint_pending": pending,
            "execution_plan": [
                {
                    "step_id": "step_1",
                    "status": "in_progress",
                }
            ],
            "plan_step_index": 0,
            "step_events": [],
        },
        context={"reason": "approval"},
    )

    async def _seed():
        db_session.add(job)
        db_session.add(checkpoint)
        await db_session.commit()
        await db_session.refresh(job)

    asyncio.get_event_loop().run_until_complete(_seed())

    request_payload = {
        "action": action,
        "checkpoint_note": f"Operator selected {action}",
    }
    if action == "edit":
        request_payload["checkpoint_action_patch"] = {
            "purpose": "Refined purpose",
        }
    response = client.post(
        f"/api/v1/agent-jobs/{job.id}/action",
        headers=auth_headers,
        json=request_payload,
    )

    assert response.status_code == 200
    assert response.json()["status"] == expected_status
    expected_queue = [(str(job.id), str(test_user.id))] if should_queue else []
    assert queued_jobs == expected_queue

    async def _refresh():
        await db_session.refresh(checkpoint)

    asyncio.get_event_loop().run_until_complete(_refresh())
    assert checkpoint.state["approval_checkpoint_pending"] is None
    assert checkpoint.state["execution_plan"][0]["status"] == expected_step_status
    if action == "edit":
        assert (
            checkpoint.state["approval_override_action"]["purpose"] == "Refined purpose"
        )


def test_approve_interrupted_tool_retries_exact_journal_action(
    client,
    db_session,
    test_user,
    auth_headers,
    monkeypatch,
):
    queued_jobs: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "app.tasks.agent_job_tasks.execute_agent_job_task.delay",
        lambda job_id, user_id: queued_jobs.append((job_id, user_id)),
    )
    pending = {
        "checkpoint_type": "execution_reconciliation",
        "checkpoint_id": "reconcile:inv-1",
        "iteration": 3,
        "action": {
            "tool": "write_file",
            "purpose": "Apply reviewed patch",
            "params": {"path": "src/fix.py"},
        },
        "retryable_from_journal": True,
    }
    job = AgentJob(
        name="Interrupted coding action",
        goal="Resume safely",
        job_type="coding",
        user_id=test_user.id,
        status=AgentJobStatus.PAUSED.value,
        max_iterations=10,
        max_tool_calls=10,
        max_llm_calls=10,
        max_runtime_minutes=30,
        results={
            "execution_strategy": {
                "approval_checkpoints": {"pending": pending, "events": []}
            },
            "approval_checkpoint": pending,
        },
    )
    checkpoint = AgentJobCheckpoint(
        job=job,
        iteration=3,
        phase="awaiting_reconciliation",
        state={
            "approval_checkpoint_pending": pending,
            "execution_reconciliation_pending": pending,
            "execution_plan": [],
            "step_events": [],
        },
        context={"reason": "execution_reconciliation"},
    )

    async def _seed():
        db_session.add(job)
        db_session.add(checkpoint)
        await db_session.commit()
        await db_session.refresh(job)

    asyncio.get_event_loop().run_until_complete(_seed())

    response = client.post(
        f"/api/v1/agent-jobs/{job.id}/action",
        headers=auth_headers,
        json={"action": "approve", "checkpoint_note": "External effect absent; retry"},
    )

    assert response.status_code == 200
    assert queued_jobs == [(str(job.id), str(test_user.id))]

    async def _refresh():
        await db_session.refresh(checkpoint)

    asyncio.get_event_loop().run_until_complete(_refresh())
    assert checkpoint.state["approval_override_action"]["tool"] == "write_file"
    assert (
        checkpoint.state["approval_override_action"]["params"]["path"] == "src/fix.py"
    )
    assert checkpoint.state["execution_reconciliation_pending"] is None


def test_launch_tie_breaker_action_spawns_new_job(
    client,
    db_session,
    test_user,
    auth_headers,
    monkeypatch,
):
    queued_jobs: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "app.tasks.agent_job_tasks.execute_agent_job_task.delay",
        lambda job_id, user_id: queued_jobs.append((job_id, user_id)),
    )

    job = AgentJob(
        name="Bug Swarm Fan-in",
        goal="Resolve frontend save failure",
        job_type="synthesis",
        user_id=test_user.id,
        status=AgentJobStatus.PAUSED.value,
        config={
            "coding_swarm_enabled": True,
            "coding_swarm_profile": "bug_triage",
            "inherited_data": {
                "swarm": {
                    "sibling_jobs": [
                        {
                            "job_id": "slice-1",
                            "role": "Patcher",
                            "status": "completed",
                            "results": {"summary": "Likely fix is in the save handler"},
                        }
                    ]
                }
            },
        },
        results={
            "swarm_fan_in": {
                "review_state": "consensus_failed",
                "candidate_paths": [
                    {
                        "job_id": "slice-1",
                        "role": "Patcher",
                        "suspect_files": ["frontend/src/pages/DocumentsPage.tsx"],
                    }
                ],
            }
        },
    )

    async def _seed():
        db_session.add(job)
        await db_session.commit()
        await db_session.refresh(job)

    asyncio.get_event_loop().run_until_complete(_seed())

    response = client.post(
        f"/api/v1/agent-jobs/{job.id}/action",
        headers=auth_headers,
        json={"action": "launch_tie_breaker"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["parent_job_id"] == str(job.id)
    assert queued_jobs

    async def _refresh():
        await db_session.refresh(job)

    asyncio.get_event_loop().run_until_complete(_refresh())
    fan_in = (job.results or {}).get("swarm_fan_in") or {}
    assert fan_in["review_state"] == "tie_break_running"
    assert fan_in["tie_breaker_job_id"] == payload["id"]


def test_promote_swarm_candidate_action_spawns_repair_job(
    client,
    db_session,
    test_user,
    auth_headers,
    monkeypatch,
):
    queued_jobs: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "app.tasks.agent_job_tasks.execute_agent_job_task.delay",
        lambda job_id, user_id: queued_jobs.append((job_id, user_id)),
    )

    job = AgentJob(
        name="Bug Swarm Fan-in",
        goal="Resolve frontend save failure",
        job_type="synthesis",
        user_id=test_user.id,
        status=AgentJobStatus.PAUSED.value,
        config={
            "coding_swarm_enabled": True,
            "coding_swarm_profile": "bug_triage",
            "source_id": "repo-source-1",
            "scope": "frontend",
            "failure_symptom": "Save flow returns 500",
        },
        results={
            "swarm_fan_in": {
                "review_state": "needs_review",
                "candidate_paths": [
                    {
                        "job_id": "slice-1",
                        "role": "Patcher",
                        "suspect_files": ["frontend/src/pages/DocumentsPage.tsx"],
                        "recommended_commands": [
                            "CI=true npm --prefix frontend test -- --watchAll=false"
                        ],
                    }
                ],
                "recommended_commands": [
                    "CI=true npm --prefix frontend test -- --watchAll=false"
                ],
            }
        },
    )

    async def _seed():
        db_session.add(job)
        await db_session.commit()
        await db_session.refresh(job)

    asyncio.get_event_loop().run_until_complete(_seed())

    async def _fake_launch_repair(
        self, *, fan_in_job, db, merged, candidate_job_id="", manual_promotion=False
    ):
        child = AgentJob(
            name="Bug Triage Repair",
            goal="Repair",
            job_type="analysis",
            user_id=test_user.id,
            status=AgentJobStatus.PENDING.value,
            parent_job_id=fan_in_job.id,
            root_job_id=fan_in_job.root_job_id or fan_in_job.id,
        )
        db.add(child)
        await db.flush()
        return child

    monkeypatch.setattr(
        "app.services.autonomous_agent_executor.AutonomousAgentExecutor._launch_bug_triage_swarm_repair_job",
        _fake_launch_repair,
    )

    response = client.post(
        f"/api/v1/agent-jobs/{job.id}/action",
        headers=auth_headers,
        json={
            "action": "promote_swarm_candidate",
            "action_payload": {"candidate_job_id": "slice-1"},
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["parent_job_id"] == str(job.id)
    assert queued_jobs

    async def _refresh():
        await db_session.refresh(job)

    asyncio.get_event_loop().run_until_complete(_refresh())
    fan_in = (job.results or {}).get("swarm_fan_in") or {}
    assert fan_in["repair_chain_job_id"] == payload["id"]
    assert fan_in["review_state"] == "manual_promotion"


def test_promote_domain_research_job_creates_profile_and_portfolio(
    client,
    db_session,
    test_user,
    auth_headers,
    monkeypatch,
):
    queued_jobs: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "app.tasks.agent_job_tasks.execute_agent_job_task.delay",
        lambda job_id, user_id: queued_jobs.append((job_id, user_id)),
    )

    job = AgentJob(
        name="Compiler Optimization Scout",
        goal="Research compiler optimization opportunities",
        job_type="research",
        user_id=test_user.id,
        status=AgentJobStatus.COMPLETED.value,
        config={
            "launch_mode": "quick_start_domain_research",
            "domain": "Compiler optimization",
            "objective": "Find benchmark-backed opportunities",
            "source_scope": "kb_plus_arxiv_plus_repo",
            "track_type": "compiler",
            "monitor_queries": ["compiler optimization benchmark"],
            "benchmark_queries": ["llvm pass regression"],
            "automation_profile": "balanced",
            "automation_policy": {
                "confidence_threshold": 0.74,
                "auto_launch_follow_up": True,
                "auto_create_experiment_plans": True,
            },
            "quick_start": {"profile": "domain_research", "version": "v2"},
        },
    )

    async def _seed():
        db_session.add(job)
        await db_session.commit()
        await db_session.refresh(job)

    asyncio.get_event_loop().run_until_complete(_seed())

    response = client.post(
        f"/api/v1/agent-jobs/{job.id}/promote-domain-research",
        headers=auth_headers,
        json={
            "target_mode": "profile_with_portfolio",
            "profile": {
                "title": "Compiler Perf Monitor",
                "interval_minutes": 720,
            },
            "portfolio": {
                "title": "Compiler Research Fleet",
            },
            "start_profile_now": True,
            "run_portfolio_now": True,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["promotion_status"] == "promoted_to_profile_and_portfolio"
    assert payload["domain_research_profile_id"]
    assert payload["research_portfolio_id"]
    assert payload["profile"]["title"] == "Compiler Perf Monitor"
    assert payload["portfolio"]["title"] == "Compiler Research Fleet"
    assert (
        payload["source_job"]["promotion_status"] == "promoted_to_profile_and_portfolio"
    )
    assert (
        payload["source_job"]["promoted_domain_research_profile_id"]
        == payload["domain_research_profile_id"]
    )
    assert (
        payload["source_job"]["config"]["promotion"]["domain_research_profile_id"]
        == payload["domain_research_profile_id"]
    )
    assert len(queued_jobs) == 2

    async def _assert_db():
        await db_session.refresh(job)
        promotion = (job.config or {}).get("promotion") or {}
        assert (
            promotion["domain_research_profile_id"]
            == payload["domain_research_profile_id"]
        )
        assert promotion["research_portfolio_id"] == payload["research_portfolio_id"]

        profile = await db_session.get(
            DomainResearchProfile, UUID(payload["domain_research_profile_id"])
        )
        assert profile is not None
        assert profile.title == "Compiler Perf Monitor"
        assert profile.latest_run_job_id is not None

        portfolio = await db_session.get(
            ResearchPortfolio, UUID(payload["research_portfolio_id"])
        )
        assert portfolio is not None
        assert str(profile.id) in (portfolio.linked_profile_ids or [])
        assert portfolio.latest_run_job_id is not None

    asyncio.get_event_loop().run_until_complete(_assert_db())


def test_promote_domain_research_job_rejects_non_completed_jobs(
    client,
    db_session,
    test_user,
    auth_headers,
):
    job = AgentJob(
        name="Incomplete Domain Research",
        goal="Research retrieval quality",
        job_type="research",
        user_id=test_user.id,
        status=AgentJobStatus.RUNNING.value,
        config={
            "launch_mode": "quick_start_domain_research",
            "domain": "Retrieval quality",
            "objective": "Find ranking failures",
            "quick_start": {"profile": "domain_research", "version": "v2"},
        },
    )

    async def _seed():
        db_session.add(job)
        await db_session.commit()

    asyncio.get_event_loop().run_until_complete(_seed())

    response = client.post(
        f"/api/v1/agent-jobs/{job.id}/promote-domain-research",
        headers=auth_headers,
        json={"profile": {"title": "Retrieval Monitor"}},
    )

    assert response.status_code == 422


def test_promote_domain_research_job_rejects_duplicate_promotion(
    client,
    db_session,
    test_user,
    auth_headers,
):
    job = AgentJob(
        name="Already Promoted Research",
        goal="Research retrieval quality",
        job_type="research",
        user_id=test_user.id,
        status=AgentJobStatus.COMPLETED.value,
        config={
            "launch_mode": "quick_start_domain_research",
            "domain": "Retrieval quality",
            "objective": "Find ranking failures",
            "promotion": {
                "status": "promoted_to_profile",
                "domain_research_profile_id": "00000000-0000-0000-0000-000000000123",
            },
            "quick_start": {"profile": "domain_research", "version": "v2"},
        },
    )

    async def _seed():
        db_session.add(job)
        await db_session.commit()

    asyncio.get_event_loop().run_until_complete(_seed())

    response = client.post(
        f"/api/v1/agent-jobs/{job.id}/promote-domain-research",
        headers=auth_headers,
        json={"profile": {"title": "Retrieval Monitor"}},
    )

    assert response.status_code == 409


def test_promote_domain_research_job_attaches_to_existing_portfolio(
    client,
    db_session,
    test_user,
    auth_headers,
    monkeypatch,
):
    queued_jobs: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "app.tasks.agent_job_tasks.execute_agent_job_task.delay",
        lambda job_id, user_id: queued_jobs.append((job_id, user_id)),
    )

    portfolio = ResearchPortfolio(
        user_id=test_user.id,
        title="Existing Fleet",
        objective="Track opportunities",
        status="running",
        linked_profile_ids=["existing-profile-id"],
        automation_profile="balanced",
        automation_policy={},
    )
    job = AgentJob(
        name="Compiler Optimization Scout",
        goal="Research compiler optimization opportunities",
        job_type="research",
        user_id=test_user.id,
        status=AgentJobStatus.COMPLETED.value,
        config={
            "launch_mode": "quick_start_domain_research",
            "domain": "Compiler optimization",
            "objective": "Find benchmark-backed opportunities",
            "source_scope": "kb_plus_arxiv_plus_repo",
            "track_type": "compiler",
            "monitor_queries": ["compiler optimization benchmark"],
            "automation_profile": "balanced",
            "automation_policy": {
                "confidence_threshold": 0.74,
                "auto_launch_follow_up": True,
                "auto_create_experiment_plans": True,
            },
            "quick_start": {"profile": "domain_research", "version": "v2"},
        },
    )

    async def _seed():
        db_session.add(portfolio)
        db_session.add(job)
        await db_session.commit()
        await db_session.refresh(portfolio)
        await db_session.refresh(job)

    asyncio.get_event_loop().run_until_complete(_seed())

    response = client.post(
        f"/api/v1/agent-jobs/{job.id}/promote-domain-research",
        headers=auth_headers,
        json={
            "target_mode": "profile_with_portfolio",
            "portfolio_id": str(portfolio.id),
            "start_profile_now": False,
            "run_portfolio_now": False,
            "profile": {
                "title": "Compiler Perf Monitor",
            },
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["research_portfolio_id"] == str(portfolio.id)
    assert payload["portfolio"]["id"] == str(portfolio.id)
    assert queued_jobs == []

    async def _assert_db():
        await db_session.refresh(portfolio)
        assert payload["domain_research_profile_id"] in (
            portfolio.linked_profile_ids or []
        )

    asyncio.get_event_loop().run_until_complete(_assert_db())
