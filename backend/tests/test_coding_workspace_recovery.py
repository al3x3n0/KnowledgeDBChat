import tempfile
from pathlib import Path
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.services.coding_workspace_manager import (
    CodingWorkspace,
    CodingWorkspaceManager,
    _sha256,
)


@pytest.fixture
def manager():
    workspace_manager = CodingWorkspaceManager()
    yield workspace_manager
    workspace_manager.cleanup_all()


def _make_workspace(
    manager: CodingWorkspaceManager,
    files: dict[str, str],
) -> CodingWorkspace:
    workspace_id = str(uuid4())
    base_path = Path(
        tempfile.mkdtemp(prefix=f"test_recovery_{workspace_id[:8]}_")
    ).resolve()
    original_hashes = {}
    for relative, content in files.items():
        target = base_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        data = content.encode("utf-8")
        target.write_bytes(data)
        original_hashes[relative] = _sha256(data)
    workspace = CodingWorkspace(
        workspace_id=workspace_id,
        base_path=base_path,
        original_hashes=original_hashes,
    )
    manager._workspaces[workspace_id] = workspace
    return workspace


def test_checkpoint_restore_preserves_current_state_as_recovery_point(manager):
    workspace = _make_workspace(
        manager,
        {
            "src/a.py": "original a\n",
            "src/b.py": "original b\n",
        },
    )
    checkpoint, error = manager.create_checkpoint(
        workspace,
        label="Known good baseline",
    )
    assert error is None

    (workspace.base_path / "src/a.py").write_text("broken a\n", encoding="utf-8")
    (workspace.base_path / "src/b.py").unlink()
    (workspace.base_path / "src/new.py").write_text("new\n", encoding="utf-8")

    restored, error = manager.restore_checkpoint(
        workspace,
        checkpoint["checkpoint_id"],
    )

    assert error is None
    assert (workspace.base_path / "src/a.py").read_text() == "original a\n"
    assert (workspace.base_path / "src/b.py").read_text() == "original b\n"
    assert not (workspace.base_path / "src/new.py").exists()
    assert restored["safety_checkpoint"]["kind"] == "pre_restore"
    assert len(manager.list_checkpoints(workspace)) == 2
    assert restored["status"]["changes_count"] == 0


def test_checkpoint_restore_keeps_git_metadata(manager):
    workspace = _make_workspace(manager, {"a.py": "one\n"})
    git_marker = workspace.base_path / ".git/HEAD"
    git_marker.parent.mkdir(parents=True)
    git_marker.write_text("ref: refs/heads/main\n")
    checkpoint, error = manager.create_checkpoint(workspace, label="baseline")
    assert error is None

    (workspace.base_path / "a.py").write_text("two\n")
    restored, error = manager.restore_checkpoint(
        workspace,
        checkpoint["checkpoint_id"],
        preserve_current=False,
    )

    assert error is None
    assert restored["safety_checkpoint"] is None
    assert git_marker.read_text() == "ref: refs/heads/main\n"


def test_failed_restore_rolls_back_to_automatic_safety_checkpoint(
    manager,
    monkeypatch,
):
    workspace = _make_workspace(manager, {"a.py": "baseline\n"})
    checkpoint, error = manager.create_checkpoint(workspace, label="baseline")
    assert error is None
    (workspace.base_path / "a.py").write_text("current work\n")

    original_restore_tree = manager._restore_tree
    attempts = 0

    def _fail_once(target_workspace, checkpoint_path):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            (target_workspace.base_path / "a.py").write_text("partial restore\n")
            raise OSError("simulated copy failure")
        original_restore_tree(target_workspace, checkpoint_path)

    monkeypatch.setattr(manager, "_restore_tree", _fail_once)

    result, error = manager.restore_checkpoint(
        workspace,
        checkpoint["checkpoint_id"],
    )

    assert result is None
    assert "simulated copy failure" in error
    assert (workspace.base_path / "a.py").read_text() == "current work\n"


@pytest.mark.asyncio
async def test_hydrate_candidate_snapshot_verifies_and_applies_exact_delta(
    manager,
    monkeypatch,
):
    workspace = _make_workspace(
        manager,
        {
            "src/a.py": "original a\n",
            "src/remove.py": "remove me\n",
        },
    )
    modified = b"fixed a\n"
    added = b"new file\n"
    job_id = "candidate-job"
    manifest = {
        "type": "workspace_delta_snapshot",
        "snapshot_id": "candidate-exact",
        "job_id": job_id,
        "immutable": True,
        "persistence_complete": True,
        "base_digest": manager._base_digest(workspace),
        "files": [
            {
                "path": "src/a.py",
                "object_path": f"workspaces/{job_id}/src/a.py",
                "sha256": _sha256(modified),
            },
            {
                "path": "src/new.py",
                "object_path": f"workspaces/{job_id}/src/new.py",
                "sha256": _sha256(added),
            },
        ],
        "deleted_files": ["src/remove.py"],
    }

    from app.services.storage_service import storage_service

    get_content = AsyncMock(side_effect=[modified, added])
    monkeypatch.setattr(storage_service, "get_file_content", get_content)

    result, error = await manager.hydrate_candidate_snapshot(workspace, manifest)

    assert error is None
    assert (workspace.base_path / "src/a.py").read_bytes() == modified
    assert (workspace.base_path / "src/new.py").read_bytes() == added
    assert not (workspace.base_path / "src/remove.py").exists()
    assert result["snapshot_id"] == "candidate-exact"
    assert result["safety_checkpoint"]["kind"] == "pre_hydrate"
    assert result["status"]["modified"] == ["src/a.py"]
    assert result["status"]["added"] == ["src/new.py"]
    assert result["status"]["deleted"] == ["src/remove.py"]


@pytest.mark.asyncio
async def test_hydration_rejects_baseline_drift_before_download(manager, monkeypatch):
    workspace = _make_workspace(manager, {"src/a.py": "original\n"})
    manifest = {
        "type": "workspace_delta_snapshot",
        "snapshot_id": "candidate-drifted",
        "job_id": "candidate-job",
        "immutable": True,
        "persistence_complete": True,
        "base_digest": "wrong-baseline",
        "files": [],
        "deleted_files": [],
    }

    from app.services.storage_service import storage_service

    get_content = AsyncMock()
    monkeypatch.setattr(storage_service, "get_file_content", get_content)

    result, error = await manager.hydrate_candidate_snapshot(workspace, manifest)

    assert result is None
    assert "baseline does not match" in error
    get_content.assert_not_awaited()
    assert manager.list_checkpoints(workspace) == []
    assert (workspace.base_path / "src/a.py").read_text() == "original\n"


@pytest.mark.asyncio
async def test_hydration_rejects_dirty_workspace_before_download(manager, monkeypatch):
    workspace = _make_workspace(manager, {"src/a.py": "original\n"})
    manifest = {
        "type": "workspace_delta_snapshot",
        "snapshot_id": "candidate-clean-only",
        "job_id": "candidate-job",
        "immutable": True,
        "persistence_complete": True,
        "base_digest": manager._base_digest(workspace),
        "files": [],
        "deleted_files": [],
    }
    (workspace.base_path / "src/a.py").write_text("local edits\n")

    from app.services.storage_service import storage_service

    get_content = AsyncMock()
    monkeypatch.setattr(storage_service, "get_file_content", get_content)

    result, error = await manager.hydrate_candidate_snapshot(workspace, manifest)

    assert result is None
    assert "clean baseline" in error
    get_content.assert_not_awaited()
    assert (workspace.base_path / "src/a.py").read_text() == "local edits\n"


@pytest.mark.asyncio
async def test_hydration_rejects_hash_mismatch_without_mutation(manager, monkeypatch):
    workspace = _make_workspace(manager, {"src/a.py": "original\n"})
    manifest = {
        "type": "workspace_delta_snapshot",
        "snapshot_id": "candidate-tampered",
        "job_id": "candidate-job",
        "immutable": True,
        "persistence_complete": True,
        "base_digest": manager._base_digest(workspace),
        "files": [
            {
                "path": "src/a.py",
                "object_path": "workspaces/candidate-job/src/a.py",
                "sha256": _sha256(b"expected\n"),
            }
        ],
        "deleted_files": [],
    }

    from app.services.storage_service import storage_service

    monkeypatch.setattr(
        storage_service,
        "get_file_content",
        AsyncMock(return_value=b"tampered\n"),
    )

    result, error = await manager.hydrate_candidate_snapshot(workspace, manifest)

    assert result is None
    assert "hash mismatch" in error
    assert manager.list_checkpoints(workspace) == []
    assert (workspace.base_path / "src/a.py").read_text() == "original\n"
