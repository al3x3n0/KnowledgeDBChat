"""
Minimal unified-diff application helpers.

We intentionally support only a single-file patch target (paper.tex) to keep this safe.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

_UNIFIED_HUNK_RE = re.compile(
    r"^@@\s+-(?P<old_start>\d+)(?:,(?P<old_count>\d+))?\s+\+(?P<new_start>\d+)(?:,(?P<new_count>\d+))?\s+@@"
)


def apply_unified_diff_to_text(*, original: str, diff_unified: str) -> Tuple[str, List[str]]:
    """
    Apply a unified diff to a single in-memory text buffer (paper.tex).

    Security: we only support patching a single file, and we require its filename to end with `paper.tex`
    if file headers are present.
    """
    original_lines = (original or "").replace("\r\n", "\n").split("\n")
    diff_lines = (diff_unified or "").replace("\r\n", "\n").split("\n")
    warnings: List[str] = []

    # Validate file headers if present.
    old_file: Optional[str] = None
    new_file: Optional[str] = None
    file_pairs = 0
    for line in diff_lines:
        if line.startswith("--- "):
            old_file = (line[4:].strip().split("\t", 1)[0] or "").strip()
            continue
        if line.startswith("+++ "):
            new_file = (line[4:].strip().split("\t", 1)[0] or "").strip()
            file_pairs += 1
            continue

    if file_pairs > 1:
        raise ValueError("Diff patches multiple files; only paper.tex is supported.")
    if file_pairs == 1:
        if not (str(old_file or "").endswith("paper.tex") and str(new_file or "").endswith("paper.tex")):
            raise ValueError("Diff must target paper.tex only.")

    def _find_subsequence(haystack: List[str], needle: List[str], *, expected_pos: Optional[int]) -> Optional[int]:
        if not needle:
            return expected_pos if expected_pos is not None else 0
        candidates: List[int] = []
        n = len(needle)
        for start in range(0, max(0, len(haystack) - n) + 1):
            if haystack[start : start + n] == needle:
                candidates.append(start)
        if not candidates:
            return None
        if expected_pos is None:
            if len(candidates) > 1:
                warnings.append(f"Hunk matched {len(candidates)} times; using first match at line {candidates[0] + 1}.")
            return candidates[0]
        best = min(candidates, key=lambda x: abs(x - expected_pos))
        if len(candidates) > 1:
            warnings.append(
                f"Hunk matched {len(candidates)} times; using closest match at line {best + 1} (expected ~{expected_pos + 1})."
            )
        return best

    i = 0
    while i < len(diff_lines):
        line = diff_lines[i]
        if line.startswith("@@ "):
            m = _UNIFIED_HUNK_RE.match(line)
            if not m:
                raise ValueError("Invalid unified diff hunk header.")
            old_start = int(m.group("old_start"))
            expected_pos = max(0, old_start - 1)
            i += 1

            hunk_lines: List[str] = []
            while i < len(diff_lines):
                nxt = diff_lines[i]
                if nxt.startswith("@@ "):
                    break
                if nxt.startswith("--- ") or nxt.startswith("+++ "):
                    break
                if nxt.startswith("\\ No newline at end of file"):
                    i += 1
                    continue
                hunk_lines.append(nxt)
                i += 1

            old_chunk: List[str] = []
            new_chunk: List[str] = []
            for hl in hunk_lines:
                if hl == "":
                    # This is not a valid unified-diff line; treat it as context.
                    old_chunk.append("")
                    new_chunk.append("")
                    continue
                prefix = hl[0]
                content = hl[1:] if len(hl) > 1 else ""
                if prefix == " ":
                    old_chunk.append(content)
                    new_chunk.append(content)
                elif prefix == "-":
                    old_chunk.append(content)
                elif prefix == "+":
                    new_chunk.append(content)
                else:
                    raise ValueError("Invalid unified diff line prefix.")

            if not old_chunk:
                warnings.append("Hunk contained no base lines; inserting at expected position.")
                insert_at = min(len(original_lines), expected_pos)
                original_lines[insert_at:insert_at] = new_chunk
                continue

            # Try applying at expected position first; otherwise search by content.
            pos: Optional[int] = None
            if original_lines[expected_pos : expected_pos + len(old_chunk)] == old_chunk:
                pos = expected_pos
            else:
                pos = _find_subsequence(original_lines, old_chunk, expected_pos=expected_pos)
            if pos is None:
                raise ValueError("Failed to apply diff: hunk context not found in current paper.tex.")

            original_lines[pos : pos + len(old_chunk)] = new_chunk
            continue

        i += 1

    patched = "\n".join(original_lines)
    if not patched.endswith("\n"):
        patched += "\n"
    return patched, warnings

