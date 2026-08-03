"""
Repository symbol-aware retrieval service (MVP).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple


class RepoSymbolIndexService:
    _allowed_exts = {".py", ".ts", ".tsx", ".js", ".jsx"}

    def retrieve(
        self,
        *,
        repo_root: Path,
        query_keywords: List[str],
        include_paths: List[str],
        max_scan_files: int = 1500,
        max_symbols: int = 20,
        max_snippets: int = 10,
    ) -> Dict[str, Any]:
        if not repo_root.exists():
            return {
                "symbol_matches": [],
                "snippet_matches": [],
                "related_tests": [],
                "symbol_scan_files": 0,
            }

        include_prefixes = [token for token in include_paths if token]
        symbol_rows: List[Dict[str, Any]] = []
        scanned = 0

        for file_path in repo_root.rglob("*"):
            if scanned >= max_scan_files:
                break
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in self._allowed_exts:
                continue
            rel_path = file_path.relative_to(repo_root).as_posix()
            if include_prefixes and not any(
                rel_path.startswith(prefix) for prefix in include_prefixes
            ):
                continue
            scanned += 1
            symbol_rows.extend(
                self._extract_symbols(file_path, rel_path, query_keywords)
            )

        symbol_rows.sort(
            key=lambda row: (-int(row.get("score", 0)), str(row.get("path", "")))
        )
        top_symbols = symbol_rows[:max_symbols]
        top_snippets = [
            self._symbol_to_snippet(repo_root, row)
            for row in top_symbols[:max_snippets]
        ]
        top_snippets = [row for row in top_snippets if row]

        related_tests: List[Dict[str, Any]] = []
        for row in top_symbols:
            path = str(row.get("path") or "").lower()
            if self._looks_like_test(path):
                related_tests.append(
                    {
                        "path": str(row.get("path") or ""),
                        "symbol": str(row.get("symbol") or ""),
                        "score": int(row.get("score", 0) or 0),
                    }
                )
            if len(related_tests) >= 8:
                break

        return {
            "symbol_matches": top_symbols,
            "snippet_matches": top_snippets,
            "related_tests": related_tests,
            "symbol_scan_files": scanned,
        }

    def _extract_symbols(
        self, file_path: Path, rel_path: str, query_keywords: List[str]
    ) -> List[Dict[str, Any]]:
        ext = file_path.suffix.lower()
        if ext == ".py":
            return self._extract_python_symbols(file_path, rel_path, query_keywords)
        return self._extract_regex_symbols(file_path, rel_path, query_keywords)

    def _extract_python_symbols(
        self, file_path: Path, rel_path: str, query_keywords: List[str]
    ) -> List[Dict[str, Any]]:
        try:
            source = file_path.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(source)
        except Exception:
            return []

        rows: List[Dict[str, Any]] = []
        for node in ast.walk(tree):
            kind = ""
            name = ""
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                kind = "function"
                name = str(node.name)
            elif isinstance(node, ast.ClassDef):
                kind = "class"
                name = str(node.name)
            else:
                continue
            if not name:
                continue
            start = int(getattr(node, "lineno", 1) or 1)
            end = int(getattr(node, "end_lineno", start) or start)
            score = self._score_symbol(rel_path, name, kind, query_keywords)
            if score <= 0:
                continue
            rows.append(
                {
                    "path": rel_path,
                    "symbol": name,
                    "kind": kind,
                    "start_line": start,
                    "end_line": end,
                    "score": score,
                    "why_relevant": self._why_relevant(rel_path, name, query_keywords),
                }
            )
        return rows

    def _extract_regex_symbols(
        self, file_path: Path, rel_path: str, query_keywords: List[str]
    ) -> List[Dict[str, Any]]:
        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return []
        rows: List[Dict[str, Any]] = []
        patterns: List[Tuple[str, re.Pattern[str]]] = [
            ("function", re.compile(r"\bfunction\s+([A-Za-z0-9_]+)\s*\(")),
            ("function", re.compile(r"\bconst\s+([A-Za-z0-9_]+)\s*=\s*\(.*?\)\s*=>")),
            ("class", re.compile(r"\bclass\s+([A-Za-z0-9_]+)\b")),
            ("test", re.compile(r"\b(?:test|it)\s*\(\s*[\"'`](.+?)[\"'`]")),
        ]
        lines = text.splitlines()
        for idx, line in enumerate(lines, start=1):
            for kind, pattern in patterns:
                match = pattern.search(line)
                if not match:
                    continue
                name = str(match.group(1) or "").strip()
                if not name:
                    continue
                score = self._score_symbol(rel_path, name, kind, query_keywords)
                if score <= 0:
                    continue
                rows.append(
                    {
                        "path": rel_path,
                        "symbol": name[:120],
                        "kind": kind,
                        "start_line": idx,
                        "end_line": min(idx + 4, len(lines)),
                        "score": score,
                        "why_relevant": self._why_relevant(
                            rel_path, name, query_keywords
                        ),
                    }
                )
        return rows

    def _score_symbol(
        self, path: str, symbol: str, kind: str, query_keywords: List[str]
    ) -> int:
        path_l = path.lower()
        symbol_l = symbol.lower()
        score = 0
        for token in query_keywords:
            token_l = token.lower()
            if token_l in symbol_l:
                score += 4
            if token_l in path_l:
                score += 3
            for part in token_l.replace("-", "_").split("_"):
                piece = part.strip()
                if piece and len(piece) > 2 and piece in symbol_l:
                    score += 2
        if kind == "test" or self._looks_like_test(path_l):
            score += 2
        if "/services/" in path_l or path_l.startswith("backend/app/services/"):
            score += 1
        return score

    def _why_relevant(self, path: str, symbol: str, query_keywords: List[str]) -> str:
        matches = []
        symbol_l = symbol.lower()
        path_l = path.lower()
        for token in query_keywords:
            token_l = token.lower()
            if token_l in symbol_l or token_l in path_l:
                matches.append(token_l)
            if len(matches) >= 3:
                break
        if not matches:
            return "Scored by structural relevance."
        return f"Matched query keywords: {', '.join(matches)}."

    def _symbol_to_snippet(
        self, repo_root: Path, row: Dict[str, Any]
    ) -> Dict[str, Any]:
        path = str(row.get("path") or "").strip()
        if not path:
            return {}
        full = repo_root / path
        try:
            lines = full.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            return {}
        start = max(1, int(row.get("start_line", 1) or 1))
        end = max(start, int(row.get("end_line", start) or start))
        start_clip = max(1, start - 2)
        end_clip = min(len(lines), end + 2)
        excerpt = "\n".join(lines[start_clip - 1 : end_clip])[:2000]
        return {
            "path": path,
            "symbol": str(row.get("symbol") or ""),
            "kind": str(row.get("kind") or ""),
            "start_line": start,
            "end_line": end,
            "why_relevant": str(row.get("why_relevant") or ""),
            "code_excerpt": excerpt,
        }

    def _looks_like_test(self, path_lower: str) -> bool:
        return (
            "/tests/" in path_lower
            or "__tests__" in path_lower
            or path_lower.endswith("_test.py")
            or path_lower.endswith(".test.ts")
            or path_lower.endswith(".test.tsx")
            or path_lower.endswith(".spec.ts")
            or path_lower.endswith(".spec.tsx")
        )
