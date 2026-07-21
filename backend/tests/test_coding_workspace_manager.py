"""Tests for CodingWorkspaceManager."""

import os
import tempfile
from pathlib import Path

import pytest

from app.services.coding_workspace_manager import (
    CodingWorkspace,
    CodingWorkspaceManager,
    _sha256,
    MAX_FILE_BYTES,
    MAX_WORKSPACE_BYTES,
)


@pytest.fixture
def manager():
    mgr = CodingWorkspaceManager()
    yield mgr
    mgr.cleanup_all()


def _make_workspace(manager: CodingWorkspaceManager, files: dict[str, str] | None = None) -> CodingWorkspace:
    """Create a workspace manually (without DB or git) for testing."""
    import uuid

    workspace_id = str(uuid.uuid4())
    base_path = Path(tempfile.mkdtemp(prefix=f"test_ws_{workspace_id[:8]}_")).resolve()

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
    manager._workspaces[workspace_id] = ws
    return ws


class TestPathSafety:
    """Tests for path traversal prevention."""

    def test_safe_resolve_normal_path(self, manager):
        ws = _make_workspace(manager, {"src/main.py": "print('hi')"})
        resolved = manager.safe_resolve(ws, "src/main.py")
        assert resolved is not None
        assert resolved == (ws.base_path / "src" / "main.py").resolve()

    def test_safe_resolve_rejects_dotdot(self, manager):
        ws = _make_workspace(manager)
        assert manager.safe_resolve(ws, "../etc/passwd") is None

    def test_safe_resolve_rejects_absolute_path(self, manager):
        ws = _make_workspace(manager)
        assert manager.safe_resolve(ws, "/etc/passwd") is None

    def test_safe_resolve_rejects_null_byte(self, manager):
        ws = _make_workspace(manager)
        assert manager.safe_resolve(ws, "file\x00.py") is None

    def test_safe_resolve_rejects_empty(self, manager):
        ws = _make_workspace(manager)
        assert manager.safe_resolve(ws, "") is None

    def test_safe_resolve_rejects_backslash_traversal(self, manager):
        ws = _make_workspace(manager)
        assert manager.safe_resolve(ws, "..\\etc\\passwd") is None

    def test_safe_resolve_nested_dotdot(self, manager):
        ws = _make_workspace(manager)
        assert manager.safe_resolve(ws, "a/b/../../..") is None


class TestFileOperations:
    """Tests for read_file, write_file, browse_files, search_code."""

    def test_read_file(self, manager):
        ws = _make_workspace(manager, {"hello.txt": "Hello World"})
        content, err = manager.read_file(ws, "hello.txt")
        assert err is None
        assert content == "Hello World"

    def test_read_file_with_line_range(self, manager):
        ws = _make_workspace(manager, {"lines.txt": "a\nb\nc\nd\ne"})
        content, err = manager.read_file(ws, "lines.txt", start_line=2, end_line=4)
        assert err is None
        assert content == "b\nc\nd\n"

    def test_read_file_not_found(self, manager):
        ws = _make_workspace(manager)
        content, err = manager.read_file(ws, "nonexistent.txt")
        assert content is None
        assert err is not None

    def test_read_file_rejects_traversal(self, manager):
        ws = _make_workspace(manager)
        content, err = manager.read_file(ws, "../secret.txt")
        assert content is None
        assert "Invalid path" in err

    def test_write_file(self, manager):
        ws = _make_workspace(manager)
        err = manager.write_file(ws, "new_file.py", "print('hello')")
        assert err is None
        content, _ = manager.read_file(ws, "new_file.py")
        assert content == "print('hello')"

    def test_write_file_creates_dirs(self, manager):
        ws = _make_workspace(manager)
        err = manager.write_file(ws, "deep/nested/dir/file.py", "x = 1")
        assert err is None
        assert (ws.base_path / "deep" / "nested" / "dir" / "file.py").exists()

    def test_write_file_rejects_traversal(self, manager):
        ws = _make_workspace(manager)
        err = manager.write_file(ws, "../escape.py", "bad")
        assert err is not None
        assert "traversal" in err.lower()

    def test_browse_files_lists_contents(self, manager):
        ws = _make_workspace(manager, {
            "src/a.py": "a",
            "src/b.py": "b",
            "README.md": "readme",
        })
        entries = manager.browse_files(ws)
        paths = {e["path"] for e in entries}
        assert "README.md" in paths
        assert "src" in paths

    def test_browse_files_with_glob(self, manager):
        ws = _make_workspace(manager, {
            "src/a.py": "a",
            "src/b.js": "b",
            "lib/c.py": "c",
        })
        entries = manager.browse_files(ws, glob_pattern="*.py")
        paths = {e["path"] for e in entries}
        assert "src/a.py" in paths
        assert "lib/c.py" in paths
        assert "src/b.js" not in paths

    def test_search_code_finds_pattern(self, manager):
        ws = _make_workspace(manager, {
            "src/main.py": "def hello():\n    return 'world'\n",
            "src/other.py": "x = 42\n",
        })
        results = manager.search_code(ws, r"def \w+")
        assert len(results) == 1
        assert results[0]["file"] == "src/main.py"
        assert results[0]["line"] == 1

    def test_search_code_invalid_regex(self, manager):
        ws = _make_workspace(manager)
        results = manager.search_code(ws, "[invalid")
        assert len(results) == 1
        assert "error" in results[0]

    def test_search_code_respects_max_results(self, manager):
        lines = "\n".join(f"match_{i} = True" for i in range(100))
        ws = _make_workspace(manager, {"big.py": lines})
        results = manager.search_code(ws, "match_", max_results=5)
        assert len(results) == 5


class TestWorkspaceStatus:
    """Tests for get_status change detection."""

    def test_no_changes(self, manager):
        ws = _make_workspace(manager, {"a.py": "original"})
        status = manager.get_status(ws)
        assert status["changes_count"] == 0
        assert status["modified"] == []
        assert status["added"] == []
        assert status["deleted"] == []

    def test_modified_file_detected(self, manager):
        ws = _make_workspace(manager, {"a.py": "original"})
        (ws.base_path / "a.py").write_text("modified")
        status = manager.get_status(ws)
        assert "a.py" in status["modified"]
        assert status["changes_count"] == 1

    def test_added_file_detected(self, manager):
        ws = _make_workspace(manager, {"a.py": "original"})
        (ws.base_path / "new.py").write_text("new file")
        status = manager.get_status(ws)
        assert "new.py" in status["added"]

    def test_deleted_file_detected(self, manager):
        ws = _make_workspace(manager, {"a.py": "original", "b.py": "also original"})
        (ws.base_path / "b.py").unlink()
        status = manager.get_status(ws)
        assert "b.py" in status["deleted"]

    def test_multiple_changes(self, manager):
        ws = _make_workspace(manager, {"a.py": "aa", "b.py": "bb", "c.py": "cc"})
        (ws.base_path / "a.py").write_text("modified")
        (ws.base_path / "b.py").unlink()
        (ws.base_path / "d.py").write_text("new")
        status = manager.get_status(ws)
        assert status["changes_count"] == 3
        assert "a.py" in status["modified"]
        assert "b.py" in status["deleted"]
        assert "d.py" in status["added"]


class TestWorkspaceLifecycle:
    """Tests for workspace creation, retrieval, and cleanup."""

    def test_get_returns_workspace(self, manager):
        ws = _make_workspace(manager, {"f.py": "x"})
        assert manager.get(ws.workspace_id) is ws

    def test_get_returns_none_for_unknown(self, manager):
        assert manager.get("nonexistent-id") is None

    def test_get_or_default_from_state(self, manager):
        ws = _make_workspace(manager)
        state = {"coding_workspace_id": ws.workspace_id}
        assert manager.get_or_default(None, state) is ws

    def test_cleanup_removes_directory(self, manager):
        ws = _make_workspace(manager, {"f.py": "x"})
        path = ws.base_path
        assert path.exists()
        manager.cleanup(ws.workspace_id)
        assert not path.exists()
        assert manager.get(ws.workspace_id) is None

    def test_cleanup_all(self, manager):
        ws1 = _make_workspace(manager, {"a.py": "a"})
        ws2 = _make_workspace(manager, {"b.py": "b"})
        manager.cleanup_all()
        assert not ws1.base_path.exists()
        assert not ws2.base_path.exists()
        assert manager.get(ws1.workspace_id) is None

    def test_cleanup_idempotent(self, manager):
        ws = _make_workspace(manager)
        manager.cleanup(ws.workspace_id)
        manager.cleanup(ws.workspace_id)  # should not raise


class TestSha256:
    """Tests for the _sha256 helper."""

    def test_deterministic(self):
        assert _sha256(b"hello") == _sha256(b"hello")

    def test_different_inputs(self):
        assert _sha256(b"hello") != _sha256(b"world")

    def test_returns_hex_string(self):
        h = _sha256(b"test")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)
