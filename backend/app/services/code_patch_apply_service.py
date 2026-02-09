"""
Unified diff apply service for CodePatchProposal.

Applies patch proposals to KnowledgeDB documents representing code files.
This is a best-effort applier intended for small, clean diffs produced by the Code Agent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class DiffHunk:
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: List[str]  # includes prefixes " ", "+", "-"


@dataclass
class FileDiff:
    path: str
    hunks: List[DiffHunk]


class UnifiedDiffApplyError(Exception):
    pass


class CodePatchApplyService:
    HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")

    def parse(self, diff_text: str) -> List[FileDiff]:
        lines = (diff_text or "").splitlines()
        i = 0
        files: List[FileDiff] = []

        current_path: Optional[str] = None
        current_hunks: List[DiffHunk] = []

        def _flush():
            nonlocal current_path, current_hunks
            if current_path and current_hunks:
                files.append(FileDiff(path=current_path, hunks=current_hunks))
            current_path = None
            current_hunks = []

        def _clean_path(p: str) -> str:
            p = (p or "").strip()
            if p.startswith("a/") or p.startswith("b/"):
                p = p[2:]
            return p

        while i < len(lines):
            line = lines[i]

            if line.startswith("--- "):
                # expect +++ next
                if i + 1 >= len(lines) or not lines[i + 1].startswith("+++ "):
                    raise UnifiedDiffApplyError("Invalid unified diff: missing +++ line")
                old_path = lines[i][4:].strip()
                new_path = lines[i + 1][4:].strip()
                if old_path == "/dev/null" or new_path == "/dev/null":
                    raise UnifiedDiffApplyError("Patch adds/deletes files; not supported in MVP")
                _flush()
                current_path = _clean_path(new_path)
                i += 2
                continue

            m = self.HUNK_RE.match(line)
            if m and current_path:
                old_start = int(m.group(1))
                old_count = int(m.group(2) or 1)
                new_start = int(m.group(3))
                new_count = int(m.group(4) or 1)
                i += 1
                hunk_lines: List[str] = []
                while i < len(lines):
                    hl = lines[i]
                    if hl.startswith(("--- ", "+++ ", "@@ ")):
                        break
                    if hl.startswith(("\\ No newline at end of file",)):
                        i += 1
                        continue
                    if hl[:1] in {" ", "+", "-"}:
                        hunk_lines.append(hl)
                        i += 1
                        continue
                    # ignore other metadata lines
                    i += 1
                current_hunks.append(
                    DiffHunk(
                        old_start=old_start,
                        old_count=old_count,
                        new_start=new_start,
                        new_count=new_count,
                        lines=hunk_lines,
                    )
                )
                continue

            i += 1

        _flush()
        return files

    def apply_to_text(self, original: str, file_diff: FileDiff) -> Tuple[str, dict]:
        """
        Apply a FileDiff to text content.

        Returns (new_text, debug_info).
        """
        original_lines = (original or "").splitlines()
        lines = list(original_lines)
        debug = {"path": file_diff.path, "hunks": [], "applied": 0}

        for hunk in file_diff.hunks:
            applied, info, lines = self._apply_hunk(lines, hunk)
            debug["hunks"].append(info)
            if applied:
                debug["applied"] += 1
            else:
                raise UnifiedDiffApplyError(f"Failed to apply hunk for {file_diff.path}: {info.get('error')}")

        new_text = "\n".join(lines)
        if original.endswith("\n"):
            new_text += "\n"
        return new_text, debug

    def _apply_hunk(self, lines: List[str], hunk: DiffHunk) -> Tuple[bool, dict, List[str]]:
        expected_idx = max(0, int(hunk.old_start) - 1)
        window = 8

        def _matches_at(start_idx: int) -> bool:
            idx = start_idx
            for hl in hunk.lines:
                prefix = hl[:1]
                content = hl[1:]
                if prefix in {" ", "-"}:
                    if idx >= len(lines) or lines[idx] != content:
                        return False
                    idx += 1
                elif prefix == "+":
                    continue
            return True

        start_idx = None
        if _matches_at(expected_idx):
            start_idx = expected_idx
        else:
            for delta in range(1, window + 1):
                for cand in (expected_idx - delta, expected_idx + delta):
                    if cand < 0:
                        continue
                    if _matches_at(cand):
                        start_idx = cand
                        break
                if start_idx is not None:
                    break

        if start_idx is None:
            return False, {"hunk": self._hunk_id(hunk), "error": "context mismatch"}, lines

        out_prefix = lines[:start_idx]
        idx = start_idx
        out_mid: List[str] = []
        for hl in hunk.lines:
            prefix = hl[:1]
            content = hl[1:]
            if prefix == " ":
                out_mid.append(content)
                idx += 1
            elif prefix == "-":
                idx += 1
            elif prefix == "+":
                out_mid.append(content)

        out_suffix = lines[idx:]
        new_lines = out_prefix + out_mid + out_suffix
        return True, {"hunk": self._hunk_id(hunk), "applied_at": start_idx + 1}, new_lines

    @staticmethod
    def _hunk_id(h: DiffHunk) -> str:
        return f"-{h.old_start},{h.old_count}+{h.new_start},{h.new_count}"


code_patch_apply_service = CodePatchApplyService()

