from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.models.agent_job import AgentJob, AgentJobStatus
from app.services.agent_coding_workspace_session_service import (
    agent_coding_workspace_session_service,
)
from app.services.coding_workspace_manager import CodingWorkspace


def _job(*, may_mutate: bool = True) -> AgentJob:
    return AgentJob(
        id=uuid4(),
        name="Coding candidate",
        goal="Repair regression",
        job_type="analysis",
        user_id=uuid4(),
        status=AgentJobStatus.RUNNING.value,
        config={
            "coding_harness_enabled": True,
            "coding_harness_role": "patcher" if may_mutate else "verifier",
            "coding_harness_may_mutate": may_mutate,
        },
        results={"coding_harness": {"completion_eligible": True}},
    )


def test_bind_job_creates_stable_session_and_child_views():
    job = _job()

    first = agent_coding_workspace_session_service.bind_job(job)
    second = agent_coding_workspace_session_service.bind_job(job)
    child = agent_coding_workspace_session_service.child_session_config(
        job,
        role="patcher",
        role_index=3,
    )

    assert first == second == f"coding-session-{job.id}"
    assert job.config["coding_workspace_session"]["root_job_id"] == str(job.id)
    assert child["coding_workspace_session_id"] == first
    assert child["coding_workspace_session"]["workspace_view"] == (
        "isolated_role_candidate"
    )
    assert child["coding_workspace_session"]["role_index"] == 3


@pytest.mark.asyncio
async def test_persist_candidate_snapshot_attaches_immutable_exact_reference(tmp_path):
    job = _job()
    session_id = agent_coding_workspace_session_service.bind_job(job)
    workspace = CodingWorkspace(
        workspace_id="workspace-1",
        base_path=tmp_path,
        owner_job_id=str(job.id),
        session_id=session_id,
    )
    manifest = {
        "type": "workspace_snapshot",
        "job_id": str(job.id),
        "user_id": str(job.user_id),
        "workspace_id": workspace.workspace_id,
        "source_id": "source-1",
        "repo_url": None,
        "files": [
            {
                "path": "backend/app/parser.py",
                "object_path": f"workspaces/{job.id}/backend/app/parser.py",
                "size": 12,
                "sha256": "abc123",
                "status": "modified",
            }
        ],
        "deleted_files": ["backend/app/obsolete.py"],
        "base_digest": "base123",
        "base_files_count": 2,
        "persistence_complete": True,
        "changes_summary": {"modified": 1, "added": 0, "deleted": 1},
    }
    manager = SimpleNamespace(
        get=lambda workspace_id: workspace if workspace_id == "workspace-1" else None,
        persist_workspace=AsyncMock(
            return_value={"files_persisted": 1, "manifest": manifest}
        ),
    )
    executor = SimpleNamespace(workspace_manager=manager)
    state = {"coding_workspace_id": "workspace-1", "artifacts": []}

    snapshot = await agent_coding_workspace_session_service.persist_candidate_snapshot(
        executor,
        job,
        state,
    )

    assert snapshot["type"] == "workspace_delta_snapshot"
    assert snapshot["snapshot_id"].startswith("candidate-")
    assert snapshot["session_id"] == session_id
    assert snapshot["immutable"] is True
    assert snapshot["deleted_files"] == ["backend/app/obsolete.py"]
    reference = job.results["coding_harness"]["candidate_snapshot"]
    assert reference["snapshot_id"] == snapshot["snapshot_id"]
    assert reference["files"][0]["sha256"] == "abc123"
    assert reference["base_digest"] == "base123"
    assert job.output_artifacts == [snapshot]
    assert state["artifacts"] == [snapshot]

    duplicate = await agent_coding_workspace_session_service.persist_candidate_snapshot(
        executor,
        job,
        state,
    )
    assert duplicate["snapshot_id"] == snapshot["snapshot_id"]
    manager.persist_workspace.assert_awaited_once()


@pytest.mark.asyncio
async def test_snapshot_service_rejects_workspace_owned_by_another_job(tmp_path):
    job = _job()
    workspace = CodingWorkspace(
        workspace_id="workspace-foreign",
        base_path=tmp_path,
        owner_job_id=str(uuid4()),
    )
    manager = SimpleNamespace(
        get=lambda _workspace_id: workspace,
        persist_workspace=AsyncMock(),
    )

    snapshot = await agent_coding_workspace_session_service.persist_candidate_snapshot(
        SimpleNamespace(workspace_manager=manager),
        job,
        {"coding_workspace_id": workspace.workspace_id},
    )

    assert snapshot is None
    manager.persist_workspace.assert_not_awaited()


@pytest.mark.asyncio
async def test_partial_snapshot_is_recorded_but_never_promotable(tmp_path):
    job = _job()
    workspace = CodingWorkspace(
        workspace_id="workspace-partial",
        base_path=tmp_path,
        owner_job_id=str(job.id),
    )
    manifest = {
        "type": "workspace_snapshot",
        "job_id": str(job.id),
        "workspace_id": workspace.workspace_id,
        "files": [],
        "deleted_files": [],
        "failed_files": ["backend/app/parser.py"],
        "persistence_complete": False,
        "changes_summary": {"modified": 1, "added": 0, "deleted": 0},
    }
    manager = SimpleNamespace(
        get=lambda _workspace_id: workspace,
        persist_workspace=AsyncMock(
            return_value={"files_persisted": 0, "manifest": manifest}
        ),
    )

    snapshot = await agent_coding_workspace_session_service.persist_candidate_snapshot(
        SimpleNamespace(workspace_manager=manager),
        job,
        {"coding_workspace_id": workspace.workspace_id},
    )

    assert snapshot["persistence_complete"] is False
    assert job.results["coding_harness"]["workspace_snapshots"]
    assert "candidate_snapshot" not in job.results["coding_harness"]
