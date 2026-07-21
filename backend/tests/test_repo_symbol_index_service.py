from pathlib import Path

from app.services.repo_symbol_index_service import RepoSymbolIndexService


def test_retrieve_finds_python_symbols_and_snippets(tmp_path: Path):
    service_file = tmp_path / "backend" / "app" / "services"
    service_file.mkdir(parents=True, exist_ok=True)
    file_path = service_file / "timeout_service.py"
    file_path.write_text(
        "\n".join(
            [
                "class RetryController:",
                "    def __init__(self):",
                "        self.counter = 0",
                "",
                "def handle_timeout_retry(value: int) -> int:",
                "    return value + 1",
            ]
        ),
        encoding="utf-8",
    )

    index = RepoSymbolIndexService()
    result = index.retrieve(
        repo_root=tmp_path,
        query_keywords=["timeout", "retry"],
        include_paths=[],
        max_scan_files=50,
        max_symbols=10,
        max_snippets=5,
    )

    symbols = result["symbol_matches"]
    snippets = result["snippet_matches"]
    assert result["symbol_scan_files"] >= 1
    assert any(item["symbol"] == "handle_timeout_retry" for item in symbols)
    assert any("handle_timeout_retry" in item["code_excerpt"] for item in snippets)


def test_retrieve_detects_related_tests(tmp_path: Path):
    tests_dir = tmp_path / "backend" / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    test_file = tests_dir / "test_timeout_flow.py"
    test_file.write_text(
        "\n".join(
            [
                "def test_timeout_retry_flow():",
                "    assert True",
            ]
        ),
        encoding="utf-8",
    )

    index = RepoSymbolIndexService()
    result = index.retrieve(
        repo_root=tmp_path,
        query_keywords=["timeout", "retry"],
        include_paths=["backend/tests"],
        max_scan_files=20,
        max_symbols=10,
        max_snippets=5,
    )

    related = result["related_tests"]
    assert len(related) >= 1
    assert any(item["path"].endswith("test_timeout_flow.py") for item in related)


def test_retrieve_honors_include_paths(tmp_path: Path):
    included = tmp_path / "backend" / "app" / "services"
    excluded = tmp_path / "frontend" / "src"
    included.mkdir(parents=True, exist_ok=True)
    excluded.mkdir(parents=True, exist_ok=True)

    (included / "retry_worker.py").write_text(
        "def retry_timeout_worker():\n    return 1\n",
        encoding="utf-8",
    )
    (excluded / "retryWorker.ts").write_text(
        "export function retryTimeoutWorker() { return 1; }\n",
        encoding="utf-8",
    )

    index = RepoSymbolIndexService()
    result = index.retrieve(
        repo_root=tmp_path,
        query_keywords=["retry", "timeout"],
        include_paths=["backend/app/services"],
        max_scan_files=20,
        max_symbols=10,
        max_snippets=5,
    )

    symbol_paths = [item["path"] for item in result["symbol_matches"]]
    assert any(path.startswith("backend/app/services/") for path in symbol_paths)
    assert not any(path.startswith("frontend/src/") for path in symbol_paths)
