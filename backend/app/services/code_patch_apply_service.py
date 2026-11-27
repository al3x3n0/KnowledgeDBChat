"""
Unified diff apply service for CodePatchProposal.

Applies patch proposals to KnowledgeDB documents representing code files.
This is a best-effort applier intended for small, clean diffs produced by the Code Agent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from hashlib import sha256
from typing import List, Optional, Tuple
from uuid import UUID

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.job_dispatch import enqueue


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

    async def apply_to_library(
        self,
        db: AsyncSession,
        *,
        source_id: UUID,
        user_id: UUID,
        file_diffs: List[FileDiff],
        dry_run: bool,
        base_hashes: Optional[dict[str, str]] = None,
    ) -> Tuple[list[dict], list[dict]]:
        """Prepare all replacements before staging any writes in the caller's transaction."""
        from app.models.document import Document

        paths = {diff.path for diff in file_diffs}
        statement = (
            select(Document)
            .where(
                Document.source_id == source_id,
                or_(
                    Document.file_path.in_(paths),
                    Document.source_identifier.in_(paths),
                    Document.title.in_(paths),
                ),
            )
            .order_by(Document.id)
            .execution_options(populate_existing=True)
        )
        if not dry_run:
            statement = statement.with_for_update()
        documents = (await db.scalars(statement)).all()
        by_path: dict[str, list[Document]] = {}
        for document in documents:
            aliases = {document.file_path, document.source_identifier, document.title}
            for path in aliases.intersection(paths):
                by_path.setdefault(path, []).append(document)

        replacements = []
        files: list[dict] = []
        errors: list[dict] = []
        seen: set[UUID] = set()
        for diff in file_diffs:
            candidates = by_path.get(diff.path, [])
            if len(candidates) != 1:
                message = (
                    "Ambiguous document path" if candidates else "Document not found"
                )
                errors.append({"path": diff.path, "error": message})
                continue
            document = candidates[0]
            if document.id in seen:
                errors.append(
                    {"path": diff.path, "error": "Duplicate document in patch"}
                )
                continue
            seen.add(document.id)
            original = document.content or ""
            base_hash = (
                base_hashes.get(diff.path) if isinstance(base_hashes, dict) else None
            )
            if base_hash and sha256(original.encode("utf-8")).hexdigest() != base_hash:
                errors.append(
                    {
                        "path": diff.path,
                        "error": "Document changed since the workspace was created",
                    }
                )
                continue
            try:
                replacement, debug = self.apply_to_text(original, diff)
            except UnifiedDiffApplyError as exc:
                errors.append({"path": diff.path, "error": str(exc)})
                continue
            replacements.append((document, replacement))
            files.append(
                {"path": diff.path, "document_id": str(document.id), "debug": debug}
            )

        if errors:
            return [], errors
        if not dry_run:
            for document, replacement in replacements:
                document.content = replacement
                document.content_hash = sha256(replacement.encode("utf-8")).hexdigest()
                document.is_processed = False
                document.processing_error = None
                enqueue(
                    db,
                    "app.tasks.processing_tasks.reprocess_document",
                    str(document.id),
                    str(user_id),
                )
        return files, []

    def parse(self, diff_text: str) -> List[FileDiff]:
        lines = (diff_text or "").splitlines()
        i = 0
        files: List[FileDiff] = []

        current_path: Optional[str] = None
        current_hunks: List[DiffHunk] = []

        def _flush():
            nonlocal current_path, current_hunks
            if current_path:
                if not current_hunks:
                    raise UnifiedDiffApplyError(f"Missing hunks for {current_path}")
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

            if line.startswith(
                (
                    "Binary files ",
                    "GIT binary patch",
                    "rename from ",
                    "copy from ",
                    "old mode ",
                )
            ):
                raise UnifiedDiffApplyError(
                    "Library patches support text edits to existing files"
                )

            if line.startswith("--- "):
                # expect +++ next
                if i + 1 >= len(lines) or not lines[i + 1].startswith("+++ "):
                    raise UnifiedDiffApplyError(
                        "Invalid unified diff: missing +++ line"
                    )
                old_path = lines[i][4:].strip()
                new_path = lines[i + 1][4:].strip()
                if old_path == "/dev/null" or new_path == "/dev/null":
                    raise UnifiedDiffApplyError(
                        "Library patches support text edits to existing files"
                    )
                _flush()
                current_path = _clean_path(new_path)
                if not current_path or _clean_path(old_path) != current_path:
                    raise UnifiedDiffApplyError("Library patches cannot rename files")
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
                old_lines = new_lines = 0
                while i < len(lines) and (
                    old_lines < old_count or new_lines < new_count
                ):
                    hl = lines[i]
                    if hl.startswith(("\\ No newline at end of file",)):
                        i += 1
                        continue
                    prefix = hl[:1]
                    if prefix not in {" ", "-", "+"}:
                        break
                    old_lines += prefix != "+"
                    new_lines += prefix != "-"
                    hunk_lines.append(hl)
                    i += 1
                if (old_lines, new_lines) != (old_count, new_count):
                    raise UnifiedDiffApplyError(
                        f"Incomplete patch hunk for {current_path}"
                    )
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

        offset = 0
        for hunk in file_diff.hunks:
            applied, info, lines = self._apply_hunk(lines, hunk, offset)
            debug["hunks"].append(info)
            if applied:
                debug["applied"] += 1
                offset += hunk.new_count - hunk.old_count
            else:
                raise UnifiedDiffApplyError(
                    f"Failed to apply hunk for {file_diff.path}: {info.get('error')}"
                )

        new_text = "\n".join(lines)
        if original.endswith("\n"):
            new_text += "\n"
        return new_text, debug

    def _apply_hunk(
        self, lines: List[str], hunk: DiffHunk, offset: int = 0
    ) -> Tuple[bool, dict, List[str]]:
        expected_idx = max(0, hunk.old_start - (1 if hunk.old_count else 0) + offset)
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
            return (
                False,
                {"hunk": self._hunk_id(hunk), "error": "context mismatch"},
                lines,
            )

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
        return (
            True,
            {"hunk": self._hunk_id(hunk), "applied_at": start_idx + 1},
            new_lines,
        )

    @staticmethod
    def _hunk_id(h: DiffHunk) -> str:
        return f"-{h.old_start},{h.old_count}+{h.new_start},{h.new_count}"


code_patch_apply_service = CodePatchApplyService()
