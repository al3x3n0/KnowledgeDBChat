from uuid import uuid4

import pytest

from app.models.agent_job import AgentJob, AgentJobStatus
from app.services.autonomous_agent_executor import AutonomousAgentExecutor


def _candidate_snapshot(job_id: str) -> dict:
    return {
        "type": "workspace_delta_snapshot",
        "snapshot_id": "candidate-handoff",
        "session_id": "coding-session-root",
        "job_id": job_id,
        "workspace_id": "workspace-patcher",
        "immutable": True,
        "persistence_complete": True,
        "base_digest": "base-digest",
        "files": [],
        "deleted_files": [],
    }


@pytest.mark.asyncio
async def test_tie_breaker_receives_read_execute_hydration_policy(
    db_session,
    test_user,
):
    executor = AutonomousAgentExecutor()
    fan_in_job = AgentJob(
        id=uuid4(),
        name="Coding fan-in",
        goal="Resolve parser failure",
        job_type="synthesis",
        user_id=test_user.id,
        status=AgentJobStatus.COMPLETED.value,
        config={
            "source_id": str(uuid4()),
            "coding_swarm_enabled": True,
            "coding_workspace_session_id": "coding-session-root",
        },
    )
    db_session.add(fan_in_job)
    await db_session.flush()
    snapshot = _candidate_snapshot("patcher-job")
    tie_breaker = await executor._launch_bug_triage_swarm_tie_breaker_job(
        fan_in_job=fan_in_job,
        db=db_session,
        merged={
            "candidate_paths": [
                {
                    "job_id": "patcher-job",
                    "role": "Primary Implementer",
                    "candidate_snapshot": snapshot,
                    "suspect_files": ["backend/app/parser.py"],
                    "recommended_commands": ["pytest -q"],
                }
            ],
            "conflicts": [],
        },
        swarm_payload={"sibling_jobs": []},
    )

    assert tie_breaker.config["coding_harness_role"] == "verifier"
    assert tie_breaker.config["coding_harness_may_mutate"] is False
    assert "hydrate_candidate_snapshot" in tie_breaker.config["allowed_tools"]
    assert "restore_workspace_checkpoint" in tie_breaker.config["blocked_tools"]
    assert tie_breaker.config["candidate_snapshots"] == [snapshot]
    assert tie_breaker.config["coding_workspace_session_id"] == "coding-session-root"
    available_tools = executor._get_tools_for_job_type(
        tie_breaker.job_type,
        tie_breaker.config,
    )
    assert "hydrate_candidate_snapshot" in available_tools
    assert "list_workspace_checkpoints" in available_tools
    assert "list_durable_workspace_checkpoints" in available_tools
    assert "restore_workspace_checkpoint" not in available_tools
    assert "restore_durable_workspace_checkpoint" not in available_tools


@pytest.mark.asyncio
async def test_repair_handoff_receives_mutation_owner_recovery_policy(
    db_session,
    test_user,
):
    executor = AutonomousAgentExecutor()
    fan_in_job = AgentJob(
        id=uuid4(),
        name="Coding fan-in",
        goal="Resolve parser failure",
        job_type="synthesis",
        user_id=test_user.id,
        status=AgentJobStatus.COMPLETED.value,
        config={
            "source_id": str(uuid4()),
            "coding_workspace_session_id": "coding-session-root",
        },
    )
    db_session.add(fan_in_job)
    await db_session.flush()
    snapshot = _candidate_snapshot("patcher-job")
    repair_job = await executor._launch_bug_triage_swarm_repair_job(
        fan_in_job=fan_in_job,
        db=db_session,
        merged={
            "winning_slice_id": "patcher-job",
            "winning_candidate_snapshot": snapshot,
            "candidate_paths": [
                {
                    "job_id": "patcher-job",
                    "role": "Primary Implementer",
                    "candidate_snapshot": snapshot,
                    "suspect_files": ["backend/app/parser.py"],
                    "recommended_commands": ["pytest -q"],
                }
            ],
            "recommended_commands": ["pytest -q"],
        },
    )

    assert repair_job.config["coding_harness_role"] == "patcher"
    assert repair_job.config["coding_harness_may_mutate"] is True
    assert "hydrate_candidate_snapshot" in repair_job.config["allowed_tools"]
    assert "restore_workspace_checkpoint" in repair_job.config["allowed_tools"]
    assert repair_job.config["candidate_snapshot"] == snapshot
    assert repair_job.config["swarm_handoff"]["candidate_snapshot"] == snapshot
    available_tools = executor._get_tools_for_job_type(
        repair_job.job_type,
        repair_job.config,
    )
    assert "hydrate_candidate_snapshot" in available_tools
    assert "create_workspace_checkpoint" in available_tools
    assert "restore_workspace_checkpoint" in available_tools
    assert "persist_durable_workspace_checkpoint" in available_tools
    assert "restore_durable_workspace_checkpoint" in available_tools
