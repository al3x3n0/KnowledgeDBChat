"""Repository symbol-aware retrieval: where is this defined, and what is near it.

Re-scans the repository on every call; it keeps no index of its own beyond a
per-file parse cache. Each language is read by a real parser -- `ast` for
Python, libclang for C/C++, tree-sitter for JS/TS -- in `repo_symbol_parsers`,
and a file whose parser is not installed is skipped and reported as
unreadable rather than searched by pattern.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from app.services import repo_symbol_parsers as parsers


class RepoSymbolIndexService:
    @staticmethod
    def reads(path: str) -> bool:
        """Whether this index can see symbols in a file of this kind at all."""
        return parsers.parser_available(path)

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
        needles = self._needles(query_keywords)
        symbol_rows: List[Dict[str, Any]] = []
        scanned = 0

        for file_path in repo_root.rglob("*"):
            if scanned >= max_scan_files:
                break
            if not file_path.is_file() or ".git" in file_path.parts:
                continue
            ext = file_path.suffix.lower()
            if ext not in parsers.ALL_EXTS or not parsers.parser_available(
                file_path.name
            ):
                continue
            rel_path = file_path.relative_to(repo_root).as_posix()
            if include_prefixes and not any(
                rel_path.startswith(prefix) for prefix in include_prefixes
            ):
                continue
            scanned += 1
            # A C/C++ parse costs ~0.5 s; skip the files where no symbol could
            # score -- no keyword in the path and none in the text. Python and
            # JS/TS parse in milliseconds and are always read, so a test file
            # still earns its structural bonus.
            if ext in parsers.CLANG_EXTS and needles:
                if not any(
                    n in rel_path.lower() for n in needles
                ) and not self._mentions(file_path, needles):
                    continue
            for name, kind, start, end in parsers.symbols_in(file_path, repo_root):
                score = self._score_symbol(rel_path, name, kind, query_keywords)
                if score <= 0:
                    continue
                symbol_rows.append(
                    {
                        "path": rel_path,
                        "symbol": name[:120],
                        "kind": kind,
                        "start_line": start,
                        "end_line": end,
                        "score": score,
                        "why_relevant": self._why_relevant(
                            rel_path, name, query_keywords
                        ),
                    }
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

    @staticmethod
    def _needles(query_keywords: List[str]) -> List[str]:
        """Every string whose presence could make a symbol score (see below)."""
        out: List[str] = []
        for token in query_keywords:
            token_l = str(token or "").lower().strip()
            if token_l:
                out.append(token_l)
            for part in token_l.replace("-", "_").split("_"):
                if len(part.strip()) > 2:
                    out.append(part.strip())
        return out

    @staticmethod
    def _mentions(file_path: Path, needles: List[str]) -> bool:
        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore").lower()
        except OSError:
            return False
        return any(n in text for n in needles)

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
