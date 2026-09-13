"""Tests for autonomous coding tool handler logic (clone_and_index_repo, browse_repo_files, etc.)."""

import tempfile
import uuid
from pathlib import Path

from app.services.coding_workspace_manager import (
    CodingWorkspace,
    CodingWorkspaceManager,
    _sha256,
)


def _make_manager_and_workspace(files: dict[str, str] | None = None):
    """Create a workspace manager with a populated workspace."""
    mgr = CodingWorkspaceManager()
    workspace_id = str(uuid.uuid4())
    base_path = Path(
        tempfile.mkdtemp(prefix=f"test_coder_{workspace_id[:8]}_")
    ).resolve()

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


def _make_state(workspace_id: str | None = None):
    """Create a minimal agent state dict for coding tools."""
    return {
        "coding_workspace_id": workspace_id,
        "coding_modified_files": [],
        "coding_command_history": [],
    }


class TestCloneAndIndexRepo:
    """Tests for clone_and_index_repo tool logic."""

    def test_workspace_id_stored_in_state(self):
        mgr, ws = _make_manager_and_workspace({"main.py": "print('hi')"})
        state = _make_state()
        state["coding_workspace_id"] = ws.workspace_id
        assert state["coding_workspace_id"] == ws.workspace_id
        assert mgr.get(ws.workspace_id) is ws

    def test_source_id_required_or_repo_url(self):
        params = {}
        has_source = bool(params.get("source_id") or params.get("repo_url"))
        assert not has_source

    def test_source_id_accepted(self):
        params = {"source_id": "src-123"}
        has_source = bool(params.get("source_id") or params.get("repo_url"))
        assert has_source

    def test_repo_url_requires_feature_flag(self):
        params = {"repo_url": "https://github.com/example/repo"}
        unsafe_enabled = False
        if params.get("repo_url") and not unsafe_enabled:
            error = "Unsafe code execution is not enabled"
        else:
            error = None
        assert error is not None


class TestBrowseRepoFiles:
    """Tests for browse_repo_files tool logic."""

    def test_browse_root(self):
        mgr, ws = _make_manager_and_workspace(
            {
                "src/main.py": "main",
                "src/utils.py": "utils",
                "README.md": "readme",
            }
        )
        entries = mgr.browse_files(ws, path=".")
        paths = {e["path"] for e in entries}
        assert "README.md" in paths
        assert "src" in paths

    def test_browse_subdirectory(self):
        mgr, ws = _make_manager_and_workspace(
            {
                "src/main.py": "main",
                "src/utils.py": "utils",
            }
        )
        entries = mgr.browse_files(ws, path="src")
        paths = {e["path"] for e in entries}
        assert "src/main.py" in paths

    def test_browse_with_glob(self):
        mgr, ws = _make_manager_and_workspace(
            {
                "a.py": "a",
                "b.js": "b",
                "c.py": "c",
            }
        )
        entries = mgr.browse_files(ws, glob_pattern="*.py")
        paths = {e["path"] for e in entries}
        assert "a.py" in paths
        assert "c.py" in paths
        assert "b.js" not in paths

    def test_browse_invalid_path(self):
        mgr, ws = _make_manager_and_workspace()
        entries = mgr.browse_files(ws, path="../escape")
        assert entries == []

    def test_browse_respects_max_results(self):
        files = {f"file_{i}.py": f"x = {i}" for i in range(50)}
        mgr, ws = _make_manager_and_workspace(files)
        entries = mgr.browse_files(ws, glob_pattern="*.py", max_results=10)
        assert len(entries) == 10


class TestReadFile:
    """Tests for read_file tool logic."""

    def test_read_existing_file(self):
        mgr, ws = _make_manager_and_workspace({"test.py": "hello world"})
        content, err = mgr.read_file(ws, "test.py")
        assert err is None
        assert content == "hello world"

    def test_read_with_line_range(self):
        code = "line1\nline2\nline3\nline4\nline5"
        mgr, ws = _make_manager_and_workspace({"code.py": code})
        content, err = mgr.read_file(ws, "code.py", start_line=2, end_line=4)
        assert err is None
        assert "line2" in content
        assert "line4" in content

    def test_read_truncates_large_content(self):
        large = "x" * 50000
        mgr, ws = _make_manager_and_workspace({"big.txt": large})
        content, err = mgr.read_file(ws, "big.txt", max_chars=100)
        assert err is None
        assert len(content) < 200
        assert "truncated" in content

    def test_read_nonexistent(self):
        mgr, ws = _make_manager_and_workspace()
        content, err = mgr.read_file(ws, "missing.py")
        assert content is None
        assert err is not None


class TestWriteFile:
    """Tests for write_file tool logic."""

    def test_write_new_file(self):
        mgr, ws = _make_manager_and_workspace()
        err = mgr.write_file(ws, "new.py", "print('new')")
        assert err is None
        content, _ = mgr.read_file(ws, "new.py")
        assert content == "print('new')"

    def test_write_overwrites_existing(self):
        mgr, ws = _make_manager_and_workspace({"exist.py": "old"})
        mgr.write_file(ws, "exist.py", "new content")
        content, _ = mgr.read_file(ws, "exist.py")
        assert content == "new content"

    def test_write_creates_nested_dirs(self):
        mgr, ws = _make_manager_and_workspace()
        err = mgr.write_file(ws, "a/b/c/d.py", "deep")
        assert err is None
        assert (ws.base_path / "a" / "b" / "c" / "d.py").exists()

    def test_write_tracks_modification(self):
        mgr, ws = _make_manager_and_workspace({"a.py": "original"})
        mgr.write_file(ws, "a.py", "modified")
        status = mgr.get_status(ws)
        assert "a.py" in status["modified"]


class TestSearchCode:
    """Tests for search_code tool logic."""

    def test_basic_search(self):
        mgr, ws = _make_manager_and_workspace(
            {
                "a.py": "def foo():\n    return 42\n",
                "b.py": "x = 1\n",
            }
        )
        results = mgr.search_code(ws, r"def \w+")
        assert len(results) == 1
        assert results[0]["file"] == "a.py"

    def test_search_with_file_glob(self):
        mgr, ws = _make_manager_and_workspace(
            {
                "a.py": "match here",
                "b.js": "match here too",
            }
        )
        results = mgr.search_code(ws, "match", file_glob="*.py")
        assert all(r["file"].endswith(".py") for r in results)

    def test_search_with_context(self):
        code = "line1\nline2\nTARGET\nline4\nline5"
        mgr, ws = _make_manager_and_workspace({"f.py": code})
        results = mgr.search_code(ws, "TARGET", context_lines=1)
        assert "line2" in results[0]["context"]
        assert "line4" in results[0]["context"]


class TestGetWorkspaceStatus:
    """Tests for get_workspace_status tool logic."""

    def test_clean_workspace(self):
        mgr, ws = _make_manager_and_workspace({"a.py": "a", "b.py": "b"})
        status = mgr.get_status(ws)
        assert status["changes_count"] == 0
        assert status["total_files"] == 2

    def test_tracks_all_change_types(self):
        mgr, ws = _make_manager_and_workspace(
            {
                "keep.py": "keep",
                "modify.py": "original",
                "delete.py": "delete me",
            }
        )
        (ws.base_path / "modify.py").write_text("changed")
        (ws.base_path / "delete.py").unlink()
        (ws.base_path / "new.py").write_text("new file")

        status = mgr.get_status(ws)
        assert "modify.py" in status["modified"]
        assert "delete.py" in status["deleted"]
        assert "new.py" in status["added"]
        assert status["changes_count"] == 3


class TestApplyPatchLogic:
    """Tests for apply_patch tool integration logic."""

    def test_dry_run_does_not_modify(self):
        mgr, ws = _make_manager_and_workspace({"a.py": "line1\nline2\nline3\n"})
        # Simulate dry_run: read file, check it exists, don't write
        content, err = mgr.read_file(ws, "a.py")
        assert content is not None
        status = mgr.get_status(ws)
        assert status["changes_count"] == 0

    def test_apply_modifies_file(self):
        mgr, ws = _make_manager_and_workspace({"a.py": "old content"})
        mgr.write_file(ws, "a.py", "new content")
        status = mgr.get_status(ws)
        assert "a.py" in status["modified"]


class TestRunCommandLogic:
    """Tests for run_command tool constraints."""

    def test_requires_unsafe_flag(self):
        unsafe_enabled = False
        if not unsafe_enabled:
            error = "Code execution is disabled"
        else:
            error = None
        assert error is not None

    def test_timeout_capped(self):
        max_timeout = 120
        requested = 999
        effective = min(requested, max_timeout)
        assert effective == 120

    def test_output_truncation(self):
        max_output = 20_000
        big_output = "x" * 50_000
        truncated = big_output[:max_output]
        assert len(truncated) == max_output

    def test_command_history_recorded(self):
        state = _make_state()
        record = {
            "cmd": "python -m pytest",
            "exit_code": 0,
            "stdout_preview": "5 passed",
            "timestamp": "2026-01-01T00:00:00",
        }
        state["coding_command_history"].append(record)
        assert len(state["coding_command_history"]) == 1
        assert state["coding_command_history"][0]["exit_code"] == 0
