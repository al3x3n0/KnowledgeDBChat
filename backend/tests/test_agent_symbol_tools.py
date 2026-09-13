"""Tests for symbol-aware code retrieval tools (retrieve_repo_symbols, get_symbol_context, find_tests_for_symbol)."""

import tempfile
import uuid
from pathlib import Path

from app.services.coding_workspace_manager import (
    CodingWorkspace,
    CodingWorkspaceManager,
    _sha256,
)
from app.services.repo_symbol_index_service import RepoSymbolIndexService


def _make_workspace_with_code(files: dict[str, str]):
    """Create a workspace with code files for symbol testing."""
    mgr = CodingWorkspaceManager()
    workspace_id = str(uuid.uuid4())
    base_path = Path(tempfile.mkdtemp(prefix=f"test_sym_{workspace_id[:8]}_")).resolve()

    original_hashes = {}
    for rel, content in files.items():
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


SAMPLE_PYTHON = """\
class UserService:
    def __init__(self, db):
        self.db = db

    def get_user(self, user_id: int):
        return self.db.query(user_id)

    def create_user(self, name: str, email: str):
        return self.db.insert(name, email)

def helper_function(x, y):
    return x + y
"""

SAMPLE_TEST = """\
import pytest

def test_get_user():
    assert True

def test_create_user():
    assert True

def test_helper_function():
    assert True
"""


class TestRepoSymbolIndexService:
    """Verify RepoSymbolIndexService works with our workspace files."""

    def test_retrieve_finds_python_symbols(self):
        mgr, ws = _make_workspace_with_code({"src/service.py": SAMPLE_PYTHON})
        svc = RepoSymbolIndexService()
        result = svc.retrieve(
            repo_root=ws.base_path,
            query_keywords=["UserService"],
            include_paths=[],
            max_symbols=10,
            max_snippets=5,
        )
        symbols = result["symbol_matches"]
        assert len(symbols) > 0
        names = [s["symbol"] for s in symbols]
        assert "UserService" in names

    def test_retrieve_finds_functions(self):
        mgr, ws = _make_workspace_with_code({"src/service.py": SAMPLE_PYTHON})
        svc = RepoSymbolIndexService()
        result = svc.retrieve(
            repo_root=ws.base_path,
            query_keywords=["helper_function"],
            include_paths=[],
            max_symbols=10,
            max_snippets=5,
        )
        symbols = result["symbol_matches"]
        names = [s["symbol"] for s in symbols]
        assert "helper_function" in names

    def test_retrieve_with_include_paths(self):
        mgr, ws = _make_workspace_with_code(
            {
                "src/a.py": "def foo(): pass\n",
                "lib/b.py": "def bar(): pass\n",
            }
        )
        svc = RepoSymbolIndexService()
        result = svc.retrieve(
            repo_root=ws.base_path,
            query_keywords=["foo", "bar"],
            include_paths=["src/"],
            max_symbols=10,
            max_snippets=5,
        )
        paths = [s["path"] for s in result["symbol_matches"]]
        assert all(p.startswith("src/") for p in paths)

    def test_retrieve_empty_repo(self):
        mgr, ws = _make_workspace_with_code({})
        svc = RepoSymbolIndexService()
        result = svc.retrieve(
            repo_root=ws.base_path,
            query_keywords=["anything"],
            include_paths=[],
        )
        assert result["symbol_matches"] == []
        assert result["symbol_scan_files"] == 0

    def test_looks_like_test(self):
        svc = RepoSymbolIndexService()
        # /tests/ requires leading slash in the path
        assert svc._looks_like_test("src/tests/test_foo.py")
        assert svc._looks_like_test("src/__tests__/bar.test.ts")
        assert svc._looks_like_test("my_test.py")  # ends with _test.py
        assert svc._looks_like_test("component.test.ts")
        assert svc._looks_like_test("component.spec.tsx")
        assert not svc._looks_like_test("src/service.py")
        assert not svc._looks_like_test("utils/helper.js")


class TestRetrieveRepoSymbols:
    """Tests for retrieve_repo_symbols tool handler logic."""

    def test_query_keyword_splitting(self):
        query = "user-service create"
        keywords = [
            t.strip()
            for t in query.replace("-", " ").replace("_", " ").split()
            if t.strip()
        ]
        assert keywords == ["user", "service", "create"]

    def test_language_filter_ext_map(self):
        ext_map = {
            "python": {".py"},
            "typescript": {".ts", ".tsx"},
            "javascript": {".js", ".jsx"},
        }
        assert ".py" in ext_map["python"]
        assert ".tsx" in ext_map["typescript"]

    def test_language_filter_applied(self):
        mgr, ws = _make_workspace_with_code(
            {
                "a.py": "def foo(): pass\n",
                "b.ts": "function bar() {}\n",
            }
        )
        svc = RepoSymbolIndexService()
        result = svc.retrieve(
            repo_root=ws.base_path,
            query_keywords=["foo", "bar"],
            include_paths=[],
            max_symbols=20,
        )
        # Apply python filter post-hoc
        allowed_exts = {".py"}
        filtered = [
            s
            for s in result["symbol_matches"]
            if any(str(s.get("path", "")).endswith(ext) for ext in allowed_exts)
        ]
        assert all(s["path"].endswith(".py") for s in filtered)

    def test_max_results_respected(self):
        lines = "\n".join(f"def func_{i}(): pass" for i in range(30))
        mgr, ws = _make_workspace_with_code({"big.py": lines})
        svc = RepoSymbolIndexService()
        result = svc.retrieve(
            repo_root=ws.base_path,
            query_keywords=["func"],
            include_paths=[],
            max_symbols=5,
        )
        assert len(result["symbol_matches"]) <= 5

    def test_requires_workspace(self):
        """Handler should error if no workspace is active."""
        state = {"coding_workspace_id": None}
        mgr = CodingWorkspaceManager()
        ws = mgr.get_or_default(None, state)
        assert ws is None


class TestGetSymbolContext:
    """Tests for get_symbol_context tool handler logic."""

    def test_finds_exact_symbol(self):
        mgr, ws = _make_workspace_with_code({"src/service.py": SAMPLE_PYTHON})
        svc = RepoSymbolIndexService()
        result = svc.retrieve(
            repo_root=ws.base_path,
            query_keywords=["get_user"],
            include_paths=["src/service.py"],
            max_symbols=20,
        )
        matches = [s for s in result["symbol_matches"] if s["path"] == "src/service.py"]
        exact = [s for s in matches if s["symbol"] == "get_user"]
        assert len(exact) == 1
        assert exact[0]["symbol"] == "get_user"

    def test_reads_surrounding_code(self):
        mgr, ws = _make_workspace_with_code({"src/service.py": SAMPLE_PYTHON})
        svc = RepoSymbolIndexService()
        result = svc.retrieve(
            repo_root=ws.base_path,
            query_keywords=["get_user"],
            include_paths=["src/service.py"],
            max_symbols=20,
        )
        exact = [s for s in result["symbol_matches"] if s["symbol"] == "get_user"]
        target = exact[0]
        # Read surrounding context
        content, err = mgr.read_file(
            ws,
            "src/service.py",
            start_line=max(1, target.get("start_line", 1) - 5),
            end_line=(target.get("end_line") or target.get("start_line", 1)) + 5,
        )
        assert err is None
        assert "get_user" in content

    def test_related_symbols_found(self):
        mgr, ws = _make_workspace_with_code({"src/service.py": SAMPLE_PYTHON})
        svc = RepoSymbolIndexService()
        result = svc.retrieve(
            repo_root=ws.base_path,
            query_keywords=["get_user"],
            include_paths=["src/service.py"],
            max_symbols=20,
        )
        matches = [s for s in result["symbol_matches"] if s["path"] == "src/service.py"]
        # Should find other symbols in the same file
        all_names = {s["symbol"] for s in matches}
        # At minimum get_user and possibly create_user, UserService
        assert "get_user" in all_names

    def test_symbol_not_found_in_wrong_file(self):
        mgr, ws = _make_workspace_with_code(
            {
                "a.py": "def foo(): pass\n",
                "b.py": "def bar(): pass\n",
            }
        )
        svc = RepoSymbolIndexService()
        result = svc.retrieve(
            repo_root=ws.base_path,
            query_keywords=["foo"],
            include_paths=["b.py"],
            max_symbols=20,
        )
        matches = [s for s in result["symbol_matches"] if s["path"] == "b.py"]
        exact = [s for s in matches if s["symbol"] == "foo"]
        assert len(exact) == 0


class TestFindTestsForSymbol:
    """Tests for find_tests_for_symbol tool handler logic."""

    def test_finds_test_files(self):
        mgr, ws = _make_workspace_with_code(
            {
                "src/service.py": SAMPLE_PYTHON,
                "src/tests/test_service.py": SAMPLE_TEST,
            }
        )
        svc = RepoSymbolIndexService()
        result = svc.retrieve(
            repo_root=ws.base_path,
            query_keywords=["get_user", "test"],
            include_paths=[],
            max_symbols=30,
        )
        test_matches = list(result.get("related_tests", []))
        # Also check symbol_matches for test-looking paths
        for sym in result.get("symbol_matches", []):
            path_lower = str(sym.get("path", "")).lower()
            if svc._looks_like_test(path_lower):
                entry = {
                    "path": sym["path"],
                    "symbol": sym["symbol"],
                    "score": sym.get("score", 0),
                }
                if entry not in test_matches:
                    test_matches.append(entry)

        test_paths = {t["path"] for t in test_matches}
        assert "src/tests/test_service.py" in test_paths

    def test_finds_specific_test_functions(self):
        mgr, ws = _make_workspace_with_code(
            {
                "src/service.py": SAMPLE_PYTHON,
                "src/tests/test_service.py": SAMPLE_TEST,
            }
        )
        svc = RepoSymbolIndexService()
        result = svc.retrieve(
            repo_root=ws.base_path,
            query_keywords=["helper_function", "test"],
            include_paths=[],
            max_symbols=30,
        )
        all_symbols = [s["symbol"] for s in result.get("symbol_matches", [])]
        # Should find test_helper_function
        assert "test_helper_function" in all_symbols or "helper_function" in all_symbols

    def test_no_tests_returns_empty(self):
        mgr, ws = _make_workspace_with_code(
            {
                "src/service.py": SAMPLE_PYTHON,
            }
        )
        svc = RepoSymbolIndexService()
        result = svc.retrieve(
            repo_root=ws.base_path,
            query_keywords=["get_user", "test"],
            include_paths=[],
            max_symbols=30,
        )
        test_matches = result.get("related_tests", [])
        # No test files exist, so related_tests should be empty
        assert len(test_matches) == 0

    def test_requires_symbol_name(self):
        params = {}
        symbol_name = str(params.get("symbol_name", "")).strip()
        assert not symbol_name


class TestSymbolToolState:
    """Tests for symbol tool workspace state integration."""

    def test_workspace_lookup_from_state(self):
        mgr, ws = _make_workspace_with_code({"a.py": "def x(): pass\n"})
        state = {"coding_workspace_id": ws.workspace_id}
        found = mgr.get_or_default(None, state)
        assert found is ws

    def test_explicit_workspace_id_overrides_state(self):
        mgr, ws1 = _make_workspace_with_code({"a.py": "x = 1\n"})
        ws2_id = str(uuid.uuid4())
        ws2_path = Path(tempfile.mkdtemp(prefix="test_sym2_")).resolve()
        ws2 = CodingWorkspace(
            workspace_id=ws2_id, base_path=ws2_path, original_hashes={}
        )
        mgr._workspaces[ws2_id] = ws2

        state = {"coding_workspace_id": ws1.workspace_id}
        found = mgr.get_or_default(ws2_id, state)
        assert found is ws2

    def test_no_workspace_returns_none(self):
        mgr = CodingWorkspaceManager()
        state = {"coding_workspace_id": None}
        assert mgr.get_or_default(None, state) is None
