import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.services.agent_coding_durable_checkpoint_service import (
    agent_coding_durable_checkpoint_service,
)
from app.services.coding_workspace_manager import (
    CodingWorkspace,
    CodingWorkspaceManager,
    _sha256,
)


def _workspace(
    manager: CodingWorkspaceManager,
    *,
    owner_job_id: str,
) -> CodingWorkspace:
    base_path = Path(tempfile.mkdtemp(prefix="durable_checkpoint_test_")).resolve()
    files = {
        "src/a.py": b"original a\n",
        "src/remove.py": b"remove me\n",
    }
    original_hashes = {}
    for relative, content in files.items():
        target = base_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        original_hashes[relative] = _sha256(content)
    workspace = CodingWorkspace(
        workspace_id=str(uuid4()),
        base_path=base_path,
        source_id="source-1",
        owner_job_id=owner_job_id,
        session_id=f"coding-session-{owner_job_id}",
        original_hashes=original_hashes,
    )
    manager._workspaces[workspace.workspace_id] = workspace
    return workspace


@pytest.mark.asyncio
async def test_durable_checkpoint_survives_workspace_manager_restart(monkeypatch):
    job_id = str(uuid4())
    session_id = f"coding-session-{job_id}"
    job = SimpleNamespace(
        id=job_id,
        root_job_id=None,
        user_id=uuid4(),
        config={
            "coding_harness_enabled": True,
            "coding_harness_may_mutate": True,
            "coding_workspace_session_id": session_id,
        },
        output_artifacts=[],
        results={},
    )
    first_manager = CodingWorkspaceManager()
    first_workspace = _workspace(first_manager, owner_job_id=job_id)
    (first_workspace.base_path / "src/a.py").write_text("fixed a\n")
    (first_workspace.base_path / "src/remove.py").unlink()
    (first_workspace.base_path / "src/new.py").write_text("new file\n")
    first_state = {
        "coding_workspace_id": first_workspace.workspace_id,
        "artifacts": [],
    }
    stored_objects: dict[str, bytes] = {}

    from app.services.storage_service import storage_service

    async def _upload(object_path, content, _content_type):
        stored_objects[object_path] = content

    monkeypatch.setattr(storage_service, "initialize", AsyncMock())
    monkeypatch.setattr(
        storage_service, "upload_to_path", AsyncMock(side_effect=_upload)
    )

    manifest = await agent_coding_durable_checkpoint_service.persist(
        SimpleNamespace(workspace_manager=first_manager),
        job,
        first_state,
        label="Verified parser repair",
        reason="successful_verification",
    )

    assert manifest["type"] == "workspace_session_checkpoint"
    assert manifest["persistence_complete"] is True
    assert manifest["workspace_state_digest"]
    assert len(stored_objects) == 2
    assert job.output_artifacts == [manifest]
    assert first_state["coding_last_durable_checkpoint_id"] == (
        manifest["checkpoint_id"]
    )

    duplicate = await agent_coding_durable_checkpoint_service.persist(
        SimpleNamespace(workspace_manager=first_manager),
        job,
        first_state,
        reason="runtime_checkpoint",
    )
    assert duplicate["checkpoint_id"] == manifest["checkpoint_id"]
    assert len(stored_objects) == 2

    first_manager.cleanup_all()
    second_manager = CodingWorkspaceManager()
    second_workspace = _workspace(second_manager, owner_job_id=job_id)
    resumed_state = {
        "coding_workspace_id": second_workspace.workspace_id,
        "coding_last_durable_checkpoint_id": manifest["checkpoint_id"],
        "artifacts": [manifest],
    }

    async def _download(object_path):
        return stored_objects[object_path]

    monkeypatch.setattr(
        storage_service,
        "get_file_content",
        AsyncMock(side_effect=_download),
    )

    restored = await agent_coding_durable_checkpoint_service.restore(
        SimpleNamespace(workspace_manager=second_manager),
        job,
        resumed_state,
        checkpoint_id=manifest["checkpoint_id"],
    )

    assert restored["snapshot_id"] == manifest["checkpoint_id"]
    assert (second_workspace.base_path / "src/a.py").read_text() == "fixed a\n"
    assert (second_workspace.base_path / "src/new.py").read_text() == "new file\n"
    assert not (second_workspace.base_path / "src/remove.py").exists()
    assert resumed_state["coding_restored_durable_checkpoint_id"] == (
        manifest["checkpoint_id"]
    )
    second_manager.cleanup_all()


@pytest.mark.asyncio
async def test_durable_restore_rejects_checkpoint_from_another_session():
    job_id = str(uuid4())
    job = SimpleNamespace(
        id=job_id,
        root_job_id=None,
        user_id=uuid4(),
        config={
            "coding_harness_enabled": True,
            "coding_harness_may_mutate": True,
            "coding_workspace_session_id": f"coding-session-{job_id}",
        },
        output_artifacts=[
            {
                "type": "workspace_session_checkpoint",
                "checkpoint_id": "durable-foreign",
                "session_id": "coding-session-someone-else",
                "persistence_complete": True,
            }
        ],
        results={},
    )

    with pytest.raises(ValueError, match="not found for this session"):
        await agent_coding_durable_checkpoint_service.restore(
            SimpleNamespace(workspace_manager=CodingWorkspaceManager()),
            job,
            {"artifacts": []},
            checkpoint_id="durable-foreign",
        )


@pytest.mark.asyncio
async def test_durable_restore_requires_mutation_owner_permission():
    job = SimpleNamespace(
        id=str(uuid4()),
        root_job_id=None,
        config={
            "coding_harness_enabled": True,
            "coding_harness_may_mutate": False,
        },
        output_artifacts=[],
    )

    with pytest.raises(ValueError, match="mutation-owner permission"):
        await agent_coding_durable_checkpoint_service.restore(
            SimpleNamespace(workspace_manager=CodingWorkspaceManager()),
            job,
            {"artifacts": []},
            checkpoint_id="durable-forbidden",
        )
