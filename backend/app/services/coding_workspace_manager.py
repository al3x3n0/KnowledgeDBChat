"""
Coding Workspace Manager.

Manages temporary directory workspaces for autonomous coding agents.
Provides workspace lifecycle (create, get, cleanup), path safety
validation, and file change tracking via content hashes.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import shutil
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger


@dataclass
class CodingWorkspace:
    """A temporary coding workspace backed by a filesystem directory."""

    workspace_id: str
    base_path: Path
    source_id: Optional[str] = None
    repo_url: Optional[str] = None
    branch: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    original_hashes: Dict[str, str] = field(default_factory=dict)


# Maximum workspace size (total bytes of files written from KB source).
MAX_WORKSPACE_BYTES = 100 * 1024 * 1024  # 100 MB
# Maximum number of files to populate from a KB source.
MAX_SOURCE_FILES = 2000
# Maximum single file size when populating from KB.
MAX_FILE_BYTES = 5 * 1024 * 1024  # 5 MB


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class CodingWorkspaceManager:
    """Manages temporary coding workspaces for agent jobs."""

    def __init__(self) -> None:
        self._workspaces: Dict[str, CodingWorkspace] = {}

    # ------------------------------------------------------------------
    # Workspace creation
    # ------------------------------------------------------------------

    async def create_from_source(
        self,
        source_id: str,
        db: Any,  # AsyncSession
    ) -> CodingWorkspace:
        """Create a workspace by loading code documents from a KB source.

        Queries Document rows belonging to *source_id*, writes them into a
        temporary directory preserving their ``source_identifier`` /
        ``file_path`` / ``title`` as relative paths, and records SHA-256
        hashes of each original file for later change detection.
        """
        from sqlalchemy import select
        from app.models.document import Document

        workspace_id = str(uuid.uuid4())
        base_path = Path(tempfile.mkdtemp(prefix=f"agent_ws_{workspace_id[:8]}_"))

        # Query code documents from the source.
        result = await db.execute(
            select(Document)
            .where(Document.source_id == source_id)
            .order_by(Document.created_at.desc())
            .limit(MAX_SOURCE_FILES)
        )
        docs = result.scalars().all()

        original_hashes: Dict[str, str] = {}
        total_bytes = 0

        for doc in docs:
            # Determine relative path.
            rel = (
                getattr(doc, "source_identifier", None)
                or getattr(doc, "file_path", None)
                or getattr(doc, "title", None)
                or str(doc.id)
            )
            rel = str(rel).strip().lstrip("/")
            if not rel:
                continue

            content = getattr(doc, "content", None) or ""
            content_bytes = content.encode("utf-8", errors="replace")

            if len(content_bytes) > MAX_FILE_BYTES:
                content_bytes = content_bytes[:MAX_FILE_BYTES]
            total_bytes += len(content_bytes)
            if total_bytes > MAX_WORKSPACE_BYTES:
                logger.warning(
                    f"Workspace {workspace_id}: total size limit reached, stopping at {len(original_hashes)} files"
                )
                break

            # Safety: resolve and validate path.
            target = self._safe_resolve(base_path, rel)
            if target is None:
                continue

            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content_bytes)
            original_hashes[rel] = _sha256(content_bytes)

        ws = CodingWorkspace(
            workspace_id=workspace_id,
            base_path=base_path,
            source_id=source_id,
            original_hashes=original_hashes,
        )
        self._workspaces[workspace_id] = ws
        logger.info(
            f"Created workspace {workspace_id} from source {source_id} "
            f"with {len(original_hashes)} files in {base_path}"
        )
        return ws

    async def create_from_url(
        self,
        repo_url: str,
        branch: Optional[str] = None,
        timeout: int = 120,
    ) -> CodingWorkspace:
        """Create a workspace by cloning a git repository.

        Requires ``unsafe_code_execution_enabled`` to be checked by the
        caller before invoking this method.
        """
        workspace_id = str(uuid.uuid4())
        base_path = Path(tempfile.mkdtemp(prefix=f"agent_ws_{workspace_id[:8]}_"))

        cmd = ["git", "clone", "--depth", "1"]
        if branch:
            cmd += ["--branch", branch]
        cmd += [repo_url, str(base_path)]

        try:
            proc = await asyncio.wait_for(
                asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                ),
                timeout=timeout,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"git clone failed (exit {proc.returncode}): {stderr.decode('utf-8', errors='replace')[:500]}"
                )
        except asyncio.TimeoutError:
            raise RuntimeError(f"git clone timed out after {timeout}s")

        # Index original file hashes.
        original_hashes: Dict[str, str] = {}
        for root, _dirs, files in os.walk(base_path):
            # Skip .git directory.
            if ".git" in Path(root).parts:
                continue
            for fname in files:
                fpath = Path(root) / fname
                rel = str(fpath.relative_to(base_path))
                try:
                    original_hashes[rel] = _sha256(fpath.read_bytes())
                except OSError:
                    pass
                if len(original_hashes) >= MAX_SOURCE_FILES:
                    break

        ws = CodingWorkspace(
            workspace_id=workspace_id,
            base_path=base_path,
            repo_url=repo_url,
            branch=branch,
            original_hashes=original_hashes,
        )
        self._workspaces[workspace_id] = ws
        logger.info(
            f"Created workspace {workspace_id} from {repo_url} "
            f"with {len(original_hashes)} files in {base_path}"
        )
        return ws

    # ------------------------------------------------------------------
    # Workspace access
    # ------------------------------------------------------------------

    def get(self, workspace_id: str) -> Optional[CodingWorkspace]:
        return self._workspaces.get(workspace_id)

    def get_or_default(self, workspace_id: Optional[str], state: Dict[str, Any]) -> Optional[CodingWorkspace]:
        """Get workspace by explicit id or fall back to state['coding_workspace_id']."""
        wid = workspace_id or (state.get("coding_workspace_id") if state else None)
        if not wid:
            return None
        return self._workspaces.get(str(wid))

    # ------------------------------------------------------------------
    # Path safety
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_resolve(base: Path, rel_path: str) -> Optional[Path]:
        """Resolve *rel_path* under *base*, rejecting traversal attempts."""
        rel_path = rel_path.replace("\\", "/").strip()
        if not rel_path or rel_path.startswith("/") or "\0" in rel_path:
            return None
        # Reject explicit traversal components.
        parts = rel_path.split("/")
        if any(p == ".." for p in parts):
            return None
        resolved = (base / rel_path).resolve()
        # Ensure the resolved path is under base.
        try:
            resolved.relative_to(base.resolve())
        except ValueError:
            return None
        return resolved

    def safe_resolve(self, workspace: CodingWorkspace, rel_path: str) -> Optional[Path]:
        """Public wrapper for path resolution with traversal checks."""
        return self._safe_resolve(workspace.base_path, rel_path)

    # ------------------------------------------------------------------
    # File operations helpers
    # ------------------------------------------------------------------

    def browse_files(
        self,
        workspace: CodingWorkspace,
        path: str = ".",
        glob_pattern: Optional[str] = None,
        max_results: int = 200,
    ) -> List[Dict[str, Any]]:
        """List files in the workspace directory."""
        target = self._safe_resolve(workspace.base_path, path)
        if target is None or not target.is_dir():
            return []

        entries: List[Dict[str, Any]] = []
        if glob_pattern:
            for p in sorted(target.rglob(glob_pattern)):
                if ".git" in p.parts:
                    continue
                rel = str(p.relative_to(workspace.base_path))
                entries.append({
                    "path": rel,
                    "type": "dir" if p.is_dir() else "file",
                    "size": p.stat().st_size if p.is_file() else 0,
                })
                if len(entries) >= max_results:
                    break
        else:
            for p in sorted(target.iterdir()):
                if p.name == ".git":
                    continue
                rel = str(p.relative_to(workspace.base_path))
                entries.append({
                    "path": rel,
                    "type": "dir" if p.is_dir() else "file",
                    "size": p.stat().st_size if p.is_file() else 0,
                })
                if len(entries) >= max_results:
                    break

        return entries

    def read_file(
        self,
        workspace: CodingWorkspace,
        path: str,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
        max_chars: int = 20000,
    ) -> Tuple[Optional[str], Optional[str]]:
        """Read file contents. Returns (content, error)."""
        target = self.safe_resolve(workspace, path)
        if target is None:
            return None, "Invalid path"
        if not target.is_file():
            return None, f"Not a file: {path}"
        try:
            text = target.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            return None, str(e)

        if start_line or end_line:
            lines = text.splitlines(keepends=True)
            s = max(0, (start_line or 1) - 1)
            e = end_line or len(lines)
            text = "".join(lines[s:e])

        if len(text) > max_chars:
            text = text[:max_chars] + f"\n... (truncated at {max_chars} chars)"
        return text, None

    def write_file(
        self,
        workspace: CodingWorkspace,
        path: str,
        content: str,
        create_dirs: bool = True,
    ) -> Optional[str]:
        """Write file contents. Returns error string or None on success."""
        target = self.safe_resolve(workspace, path)
        if target is None:
            return "Invalid path (possible traversal)"
        if create_dirs:
            target.parent.mkdir(parents=True, exist_ok=True)
        try:
            target.write_text(content, encoding="utf-8")
        except OSError as e:
            return str(e)
        return None

    def search_code(
        self,
        workspace: CodingWorkspace,
        pattern: str,
        path: str = ".",
        file_glob: Optional[str] = None,
        max_results: int = 50,
        context_lines: int = 2,
    ) -> List[Dict[str, Any]]:
        """Search for a regex pattern in workspace files."""
        import re

        target = self._safe_resolve(workspace.base_path, path)
        if target is None:
            return []

        try:
            compiled = re.compile(pattern)
        except re.error:
            return [{"error": f"Invalid regex: {pattern}"}]

        results: List[Dict[str, Any]] = []
        files = target.rglob(file_glob) if file_glob else target.rglob("*")

        for fpath in sorted(files):
            if not fpath.is_file() or ".git" in fpath.parts:
                continue
            # Skip binary files.
            try:
                text = fpath.read_text(encoding="utf-8", errors="strict")
            except (OSError, UnicodeDecodeError):
                continue

            lines = text.splitlines()
            for i, line in enumerate(lines):
                if compiled.search(line):
                    start = max(0, i - context_lines)
                    end = min(len(lines), i + context_lines + 1)
                    results.append({
                        "file": str(fpath.relative_to(workspace.base_path)),
                        "line": i + 1,
                        "match": line.rstrip()[:300],
                        "context": "\n".join(lines[start:end]),
                    })
                    if len(results) >= max_results:
                        return results
        return results

    def get_status(self, workspace: CodingWorkspace) -> Dict[str, Any]:
        """Compare current workspace files against originals."""
        current_files: Dict[str, str] = {}
        for root, _dirs, files in os.walk(workspace.base_path):
            if ".git" in Path(root).parts:
                continue
            for fname in files:
                fpath = Path(root) / fname
                rel = str(fpath.relative_to(workspace.base_path))
                try:
                    current_files[rel] = _sha256(fpath.read_bytes())
                except OSError:
                    pass

        modified = []
        added = []
        deleted = []

        for path, orig_hash in workspace.original_hashes.items():
            cur_hash = current_files.get(path)
            if cur_hash is None:
                deleted.append(path)
            elif cur_hash != orig_hash:
                modified.append(path)

        for path in current_files:
            if path not in workspace.original_hashes:
                added.append(path)

        return {
            "workspace_id": workspace.workspace_id,
            "total_files": len(current_files),
            "original_files": len(workspace.original_hashes),
            "modified": sorted(modified),
            "added": sorted(added),
            "deleted": sorted(deleted),
            "changes_count": len(modified) + len(added) + len(deleted),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    async def persist_workspace(
        self,
        workspace: CodingWorkspace,
        job_id: str,
        user_id: str,
        document_workspace: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Upload changed files to MinIO and return a manifest.

        Called before cleanup so that workspace artifacts survive temp
        directory removal.
        """
        import mimetypes

        from app.services.storage_service import storage_service

        status = self.get_status(workspace)
        changed_paths = status["modified"] + status["added"]

        if not changed_paths and not (document_workspace and document_workspace.get("assembled_markdown")):
            return {"files_persisted": 0, "manifest": None}

        await storage_service.initialize()

        manifest_files: List[Dict[str, Any]] = []
        for rel_path in changed_paths:
            target = self._safe_resolve(workspace.base_path, rel_path)
            if target is None or not target.is_file():
                continue
            try:
                content_bytes = target.read_bytes()
            except OSError:
                continue

            object_path = f"workspaces/{job_id}/{rel_path}"
            mime = mimetypes.guess_type(rel_path)[0] or "application/octet-stream"
            try:
                await storage_service.upload_to_path(object_path, content_bytes, mime)
                manifest_files.append({
                    "path": rel_path,
                    "object_path": object_path,
                    "size": len(content_bytes),
                    "status": "modified" if rel_path in status["modified"] else "added",
                })
            except Exception as upload_err:
                logger.warning(f"Failed to persist {rel_path}: {upload_err}")

        # Persist assembled document markdown if present
        if document_workspace and isinstance(document_workspace, dict):
            assembled = document_workspace.get("assembled_markdown")
            if assembled and isinstance(assembled, str):
                md_bytes = assembled.encode("utf-8")
                md_path = f"workspaces/{job_id}/_assembled_document.md"
                try:
                    await storage_service.upload_to_path(md_path, md_bytes, "text/markdown")
                    manifest_files.append({
                        "path": "_assembled_document.md",
                        "object_path": md_path,
                        "size": len(md_bytes),
                        "status": "added",
                    })
                except Exception as upload_err:
                    logger.warning(f"Failed to persist assembled document: {upload_err}")

        manifest = {
            "type": "workspace_snapshot",
            "job_id": job_id,
            "user_id": user_id,
            "workspace_id": workspace.workspace_id,
            "source_id": workspace.source_id,
            "repo_url": workspace.repo_url,
            "files": manifest_files,
            "total_files": len(manifest_files),
            "changes_summary": {
                "modified": len(status["modified"]),
                "added": len(status["added"]),
                "deleted": len(status["deleted"]),
            },
            "persisted_at": datetime.now(timezone.utc).isoformat(),
        }

        logger.info(
            f"Persisted workspace {workspace.workspace_id} for job {job_id}: "
            f"{len(manifest_files)} files uploaded"
        )
        return {"files_persisted": len(manifest_files), "manifest": manifest}

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self, workspace_id: str) -> None:
        """Remove workspace temp directory and drop from registry."""
        ws = self._workspaces.pop(workspace_id, None)
        if ws and ws.base_path.exists():
            try:
                shutil.rmtree(ws.base_path, ignore_errors=True)
                logger.info(f"Cleaned up workspace {workspace_id}")
            except Exception as e:
                logger.warning(f"Failed to clean workspace {workspace_id}: {e}")

    def cleanup_all(self) -> None:
        """Clean up all active workspaces."""
        for wid in list(self._workspaces):
            self.cleanup(wid)
