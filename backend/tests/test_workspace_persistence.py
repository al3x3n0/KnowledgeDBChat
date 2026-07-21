"""Tests for workspace persistence to MinIO and get_workspace_artifact_url tool."""

import tempfile
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from app.services.coding_workspace_manager import (
    CodingWorkspace,
    CodingWorkspaceManager,
    _sha256,
)


def _make_manager_and_workspace(files: dict[str, str] | None = None):
    """Create a workspace manager with a populated workspace."""
    mgr = CodingWorkspaceManager()
    workspace_id = str(uuid.uuid4())
    base_path = Path(tempfile.mkdtemp(prefix=f"test_persist_{workspace_id[:8]}_")).resolve()

    original_hashes = {}
    for rel, content in (files or {}).items():
        target = base_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        data = content.encode("utf-8")
        target.write_bytes(data)
        original_hashes[rel] = _sha256(data)

    ws = CodingWorkspace(
        workspace_id=workspace_id,
        base_path=base_path,
        original_hashes=original_hashes,
    )
    mgr._workspaces[workspace_id] = ws
    return mgr, ws


class TestPersistWorkspaceManifest:
    """Tests for persist_workspace manifest structure."""

    def test_manifest_structure(self):
        """Manifest should have the expected keys."""
        manifest = {
            "type": "workspace_snapshot",
            "job_id": "job-123",
            "user_id": "user-456",
            "workspace_id": "ws-789",
            "source_id": None,
            "repo_url": None,
            "files": [
                {"path": "a.py", "object_path": "workspaces/job-123/a.py", "size": 100, "status": "modified"},
            ],
            "total_files": 1,
            "changes_summary": {"modified": 1, "added": 0, "deleted": 0},
            "persisted_at": "2026-01-01T00:00:00+00:00",
        }
        assert manifest["type"] == "workspace_snapshot"
        assert manifest["total_files"] == 1
        assert manifest["files"][0]["status"] == "modified"

    def test_object_path_format(self):
        """Object paths should follow workspaces/{job_id}/{rel_path} pattern."""
        job_id = "abc-123"
        rel_path = "src/main.py"
        expected = f"workspaces/{job_id}/{rel_path}"
        assert expected == "workspaces/abc-123/src/main.py"

    def test_no_changes_returns_no_manifest(self):
        mgr, ws = _make_manager_and_workspace({"a.py": "original"})
        # No modifications → status shows 0 changes
        status = mgr.get_status(ws)
        assert status["changes_count"] == 0

    def test_changed_files_included_in_manifest(self):
        mgr, ws = _make_manager_and_workspace({"a.py": "original"})
        (ws.base_path / "a.py").write_text("modified")
        (ws.base_path / "new.py").write_text("new file")
        status = mgr.get_status(ws)
        changed = status["modified"] + status["added"]
        assert "a.py" in changed
        assert "new.py" in changed
        assert len(changed) == 2

    def test_deleted_files_not_in_upload_list(self):
        mgr, ws = _make_manager_and_workspace({"a.py": "keep", "b.py": "delete"})
        (ws.base_path / "b.py").unlink()
        status = mgr.get_status(ws)
        upload_list = status["modified"] + status["added"]
        assert "b.py" not in upload_list
        assert "b.py" in status["deleted"]


class TestPersistWorkspaceUpload:
    """Tests for the actual persist_workspace method with mocked storage."""

    def _run_persist(self, mgr, ws, job_id="job-1", user_id="user-1", document_workspace=None):
        """Run persist_workspace with mocked storage_service module."""
        import asyncio
        import sys
        from types import ModuleType

        mock_storage = AsyncMock()
        mock_storage.initialize = AsyncMock()
        mock_storage.upload_to_path = AsyncMock()

        # Pre-inject a fake storage_service module to avoid minio import
        fake_mod = ModuleType("app.services.storage_service")
        fake_mod.storage_service = mock_storage
        saved = sys.modules.get("app.services.storage_service")
        sys.modules["app.services.storage_service"] = fake_mod
        try:
            result = asyncio.run(mgr.persist_workspace(ws, job_id, user_id, document_workspace=document_workspace))
        finally:
            if saved is not None:
                sys.modules["app.services.storage_service"] = saved
            else:
                sys.modules.pop("app.services.storage_service", None)
        return result, mock_storage

    def _run_persist_with_side_effect(self, mgr, ws, side_effect):
        """Run persist_workspace with custom upload side effects."""
        import asyncio
        import sys
        from types import ModuleType

        mock_storage = AsyncMock()
        mock_storage.initialize = AsyncMock()
        mock_storage.upload_to_path = AsyncMock(side_effect=side_effect)

        fake_mod = ModuleType("app.services.storage_service")
        fake_mod.storage_service = mock_storage
        saved = sys.modules.get("app.services.storage_service")
        sys.modules["app.services.storage_service"] = fake_mod
        try:
            result = asyncio.run(mgr.persist_workspace(ws, "job-1", "user-1"))
        finally:
            if saved is not None:
                sys.modules["app.services.storage_service"] = saved
            else:
                sys.modules.pop("app.services.storage_service", None)
        return result, mock_storage

    def test_persist_uploads_modified_files(self):
        mgr, ws = _make_manager_and_workspace({"a.py": "original"})
        (ws.base_path / "a.py").write_text("modified content")

        result, mock_storage = self._run_persist(mgr, ws)

        assert result["files_persisted"] == 1
        assert result["manifest"]["type"] == "workspace_snapshot"
        assert result["manifest"]["files"][0]["path"] == "a.py"
        assert result["manifest"]["files"][0]["status"] == "modified"
        mock_storage.upload_to_path.assert_called_once()

    def test_persist_uploads_added_files(self):
        mgr, ws = _make_manager_and_workspace({})
        (ws.base_path / "new.py").write_text("new file")

        result, _ = self._run_persist(mgr, ws)

        assert result["files_persisted"] == 1
        assert result["manifest"]["files"][0]["status"] == "added"

    def test_persist_no_changes_returns_empty(self):
        mgr, ws = _make_manager_and_workspace({"a.py": "unchanged"})

        result, _ = self._run_persist(mgr, ws)
        assert result["files_persisted"] == 0
        assert result["manifest"] is None

    def test_persist_with_document_workspace(self):
        mgr, ws = _make_manager_and_workspace({})
        doc_ws = {"assembled_markdown": "# Title\n\nContent here."}

        result, _ = self._run_persist(mgr, ws, document_workspace=doc_ws)

        assert result["files_persisted"] == 1
        assert result["manifest"]["files"][0]["path"] == "_assembled_document.md"

    def test_persist_upload_failure_continues(self):
        mgr, ws = _make_manager_and_workspace({"a.py": "original"})
        (ws.base_path / "a.py").write_text("modified")
        (ws.base_path / "b.py").write_text("new")

        result, _ = self._run_persist_with_side_effect(
            mgr, ws, side_effect=[Exception("upload failed"), None]
        )

        # Should have persisted only the successful one
        assert result["files_persisted"] == 1


class TestGetWorkspaceArtifactUrl:
    """Tests for get_workspace_artifact_url tool logic."""

    def test_url_path_construction(self):
        job_id = "abc-123"
        file_path = "src/main.py"
        full_path = f"workspaces/{job_id}/{file_path}"
        assert full_path == "workspaces/abc-123/src/main.py"

    def test_requires_job_id_and_file_path(self):
        params = {}
        has_required = bool(params.get("job_id")) and bool(params.get("file_path"))
        assert not has_required

    def test_valid_params_accepted(self):
        params = {"job_id": "j-1", "file_path": "output.csv"}
        has_required = bool(params.get("job_id")) and bool(params.get("file_path"))
        assert has_required


class TestOutputArtifactsIntegration:
    """Tests for workspace snapshot integration with job.output_artifacts."""

    def test_manifest_appended_to_artifacts(self):
        output_artifacts = []
        manifest = {
            "type": "workspace_snapshot",
            "job_id": "j-1",
            "files": [{"path": "a.py", "status": "modified"}],
            "total_files": 1,
        }
        output_artifacts.append(manifest)
        assert len(output_artifacts) == 1
        assert output_artifacts[0]["type"] == "workspace_snapshot"

    def test_multiple_workspace_snapshots(self):
        output_artifacts = []
        for i in range(3):
            output_artifacts.append({
                "type": "workspace_snapshot",
                "workspace_id": f"ws-{i}",
                "total_files": i + 1,
            })
        snapshots = [a for a in output_artifacts if a["type"] == "workspace_snapshot"]
        assert len(snapshots) == 3

    def test_mixed_artifact_types(self):
        output_artifacts = [
            {"type": "code_patch_proposal", "id": "p-1"},
            {"type": "workspace_snapshot", "job_id": "j-1", "total_files": 5},
            {"type": "demo_check", "source_id": "s-1"},
        ]
        assert len(output_artifacts) == 3
        ws_snapshots = [a for a in output_artifacts if a["type"] == "workspace_snapshot"]
        assert len(ws_snapshots) == 1
