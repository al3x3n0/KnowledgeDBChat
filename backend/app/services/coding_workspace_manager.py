"""
Coding Workspace Manager.

Manages temporary directory workspaces for autonomous coding agents.
Provides workspace lifecycle (create, get, cleanup), path safety
validation, and file change tracking via content hashes.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
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
    owner_job_id: Optional[str] = None
    session_id: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    original_hashes: Dict[str, str] = field(default_factory=dict)
    checkpoint_root: Optional[Path] = None
    checkpoints: Dict[str, Dict[str, Any]] = field(default_factory=dict)


# Maximum workspace size (total bytes of files written from KB source).
MAX_WORKSPACE_BYTES = 100 * 1024 * 1024  # 100 MB
# Maximum number of files to populate from a KB source.
MAX_SOURCE_FILES = 2000
# Maximum single file size when populating from KB.
MAX_FILE_BYTES = 5 * 1024 * 1024  # 5 MB
MAX_CHECKPOINTS = 8
MAX_CHECKPOINT_FILES = 5000
MAX_CHECKPOINT_BYTES = 100 * 1024 * 1024


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
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
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

    def get_or_default(
        self, workspace_id: Optional[str], state: Dict[str, Any]
    ) -> Optional[CodingWorkspace]:
        """Get workspace by explicit id or fall back to state['coding_workspace_id']."""
        wid = workspace_id or (state.get("coding_workspace_id") if state else None)
        if not wid:
            return None
        return self._workspaces.get(str(wid))

    # ------------------------------------------------------------------
    # Recovery checkpoints
    # ------------------------------------------------------------------

    @staticmethod
    def _base_digest(workspace: CodingWorkspace) -> str:
        return hashlib.sha256(
            json.dumps(
                workspace.original_hashes,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

    @staticmethod
    def workspace_state_digest(workspace: CodingWorkspace) -> str:
        """Hash the current non-git workspace tree for checkpoint deduplication."""
        current_hashes: Dict[str, str] = {}
        for root, _dirs, filenames in os.walk(workspace.base_path):
            if ".git" in Path(root).parts:
                continue
            for filename in filenames:
                file_path = Path(root) / filename
                relative = str(file_path.relative_to(workspace.base_path))
                try:
                    current_hashes[relative] = _sha256(file_path.read_bytes())
                except OSError:
                    continue
        return hashlib.sha256(
            json.dumps(
                current_hashes,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _checkpoint_public(metadata: Dict[str, Any]) -> Dict[str, Any]:
        return {key: value for key, value in metadata.items() if key not in {"path"}}

    def create_checkpoint(
        self,
        workspace: CodingWorkspace,
        *,
        label: str = "",
        kind: str = "manual",
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Create a bounded full-tree checkpoint outside the active workspace."""
        if len(workspace.checkpoints) >= MAX_CHECKPOINTS:
            return None, f"Checkpoint limit reached ({MAX_CHECKPOINTS})"

        checkpoint_id = f"checkpoint-{uuid.uuid4().hex[:16]}"
        if workspace.checkpoint_root is None:
            workspace.checkpoint_root = Path(
                tempfile.mkdtemp(prefix=f"agent_cp_{workspace.workspace_id[:8]}_")
            )
        target_root = workspace.checkpoint_root / checkpoint_id
        target_root.mkdir(parents=True, exist_ok=False)
        total_files = 0
        total_bytes = 0
        skipped_symlinks: List[str] = []

        try:
            for source in sorted(workspace.base_path.rglob("*")):
                relative = source.relative_to(workspace.base_path)
                if ".git" in relative.parts:
                    continue
                if source.is_symlink():
                    skipped_symlinks.append(str(relative))
                    continue
                if source.is_dir():
                    (target_root / relative).mkdir(parents=True, exist_ok=True)
                    continue
                if not source.is_file():
                    continue
                size = source.stat().st_size
                total_files += 1
                total_bytes += size
                if total_files > MAX_CHECKPOINT_FILES:
                    raise ValueError(
                        f"Checkpoint exceeds {MAX_CHECKPOINT_FILES} file limit"
                    )
                if total_bytes > MAX_CHECKPOINT_BYTES:
                    raise ValueError(
                        f"Checkpoint exceeds {MAX_CHECKPOINT_BYTES} byte limit"
                    )
                destination = target_root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
        except Exception as exc:
            shutil.rmtree(target_root, ignore_errors=True)
            return None, str(exc)

        metadata: Dict[str, Any] = {
            "checkpoint_id": checkpoint_id,
            "workspace_id": workspace.workspace_id,
            "label": str(label or "").strip()[:120],
            "kind": str(kind or "manual").strip()[:40],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "files_count": total_files,
            "total_bytes": total_bytes,
            "skipped_symlinks": skipped_symlinks[:50],
            "path": target_root,
        }
        workspace.checkpoints[checkpoint_id] = metadata
        return self._checkpoint_public(metadata), None

    def list_checkpoints(self, workspace: CodingWorkspace) -> List[Dict[str, Any]]:
        """Return checkpoint metadata without exposing host paths."""
        rows = [
            self._checkpoint_public(metadata)
            for metadata in workspace.checkpoints.values()
        ]
        return sorted(rows, key=lambda row: str(row.get("created_at") or ""))

    def restore_checkpoint(
        self,
        workspace: CodingWorkspace,
        checkpoint_id: str,
        *,
        preserve_current: bool = True,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Restore a full-tree checkpoint while retaining the repository .git dir."""
        metadata = workspace.checkpoints.get(str(checkpoint_id or "").strip())
        if not isinstance(metadata, dict):
            return None, "Checkpoint not found"
        checkpoint_path = metadata.get("path")
        if not isinstance(checkpoint_path, Path) or not checkpoint_path.is_dir():
            return None, "Checkpoint data is unavailable"

        safety_checkpoint = None
        if preserve_current:
            safety_checkpoint, error = self.create_checkpoint(
                workspace,
                label=f"Before restore {checkpoint_id}",
                kind="pre_restore",
            )
            if error:
                return None, f"Could not preserve current workspace: {error}"

        try:
            self._restore_tree(workspace, checkpoint_path)
        except Exception as exc:
            rollback_error = None
            if isinstance(safety_checkpoint, dict):
                safety_metadata = workspace.checkpoints.get(
                    str(safety_checkpoint.get("checkpoint_id") or "")
                )
                safety_path = (
                    safety_metadata.get("path")
                    if isinstance(safety_metadata, dict)
                    else None
                )
                if isinstance(safety_path, Path):
                    try:
                        self._restore_tree(workspace, safety_path)
                    except Exception as rollback_exc:
                        rollback_error = str(rollback_exc)
            message = f"Checkpoint restore failed: {exc}"
            if rollback_error:
                message += f"; safety rollback also failed: {rollback_error}"
            return None, message

        return {
            "workspace_id": workspace.workspace_id,
            "restored_checkpoint_id": checkpoint_id,
            "safety_checkpoint": safety_checkpoint,
            "status": self.get_status(workspace),
        }, None

    @staticmethod
    def _restore_tree(workspace: CodingWorkspace, checkpoint_path: Path) -> None:
        """Replace the non-git workspace tree from a checkpoint directory."""
        for child in workspace.base_path.iterdir():
            if child.name == ".git":
                continue
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()
        for source in sorted(checkpoint_path.rglob("*")):
            relative = source.relative_to(checkpoint_path)
            destination = workspace.base_path / relative
            if source.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
            elif source.is_file():
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)

    async def hydrate_candidate_snapshot(
        self,
        workspace: CodingWorkspace,
        manifest: Dict[str, Any],
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Apply a complete immutable candidate delta to its exact source baseline."""
        manifest_type = str(manifest.get("type") or "")
        if manifest_type not in {
            "workspace_delta_snapshot",
            "workspace_session_checkpoint",
        }:
            return None, "Candidate snapshot type is invalid"
        if manifest.get("immutable") is not True:
            return None, "Candidate snapshot is not immutable"
        if manifest.get("persistence_complete") is not True:
            return None, "Candidate snapshot persistence is incomplete"
        if str(manifest.get("base_digest") or "") != self._base_digest(workspace):
            return None, "Candidate snapshot baseline does not match this workspace"
        if int(self.get_status(workspace).get("changes_count") or 0) != 0:
            return None, "Candidate hydration requires a clean baseline workspace"

        job_id = str(manifest.get("job_id") or "").strip()
        files = manifest.get("files")
        deleted_files = manifest.get("deleted_files")
        if (
            not job_id
            or not isinstance(files, list)
            or not isinstance(deleted_files, list)
        ):
            return None, "Candidate snapshot manifest is incomplete"
        if len(files) > MAX_CHECKPOINT_FILES:
            return None, "Candidate snapshot exceeds file count limit"

        from app.services.storage_service import storage_service

        staging_root = Path(
            tempfile.mkdtemp(prefix=f"agent_hydrate_{workspace.workspace_id[:8]}_")
        )
        total_bytes = 0
        hydrated_paths: List[str] = []
        seen_paths: set[str] = set()
        try:
            for item in files:
                if not isinstance(item, dict):
                    raise ValueError(
                        "Candidate snapshot contains an invalid file entry"
                    )
                relative = str(item.get("path") or "").strip()
                object_path = str(item.get("object_path") or "").strip()
                if manifest_type == "workspace_session_checkpoint":
                    expected_object_path = (
                        "workspace_checkpoints/"
                        f"{str(manifest.get('session_id') or '').strip()}/"
                        f"{str(manifest.get('checkpoint_id') or '').strip()}/"
                        f"{relative}"
                    )
                else:
                    expected_object_path = f"workspaces/{job_id}/{relative}"
                destination = self._safe_resolve(staging_root, relative)
                if (
                    destination is None
                    or not relative
                    or object_path != expected_object_path
                    or relative in seen_paths
                ):
                    raise ValueError(f"Unsafe candidate file entry: {relative}")
                seen_paths.add(relative)
                content = await storage_service.get_file_content(object_path)
                expected_hash = str(item.get("sha256") or "").strip()
                if not expected_hash or _sha256(content) != expected_hash:
                    raise ValueError(f"Candidate file hash mismatch: {relative}")
                total_bytes += len(content)
                if total_bytes > MAX_WORKSPACE_BYTES:
                    raise ValueError("Candidate snapshot exceeds workspace size limit")
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(content)
                hydrated_paths.append(relative)

            safe_deleted: List[str] = []
            for raw_path in deleted_files:
                relative = str(raw_path or "").strip()
                if self.safe_resolve(workspace, relative) is None:
                    raise ValueError(f"Unsafe deleted file entry: {relative}")
                if relative in seen_paths:
                    raise ValueError(
                        f"Candidate path cannot be changed and deleted: {relative}"
                    )
                safe_deleted.append(relative)

            safety_checkpoint, error = self.create_checkpoint(
                workspace,
                label=f"Before hydrate {str(manifest.get('snapshot_id') or '')[:40]}",
                kind="pre_hydrate",
            )
            if error:
                raise ValueError(f"Could not checkpoint before hydration: {error}")

            try:
                for relative in hydrated_paths:
                    source = self._safe_resolve(staging_root, relative)
                    destination = self.safe_resolve(workspace, relative)
                    if source is None or destination is None:
                        raise ValueError(f"Unsafe candidate path: {relative}")
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, destination)
                for relative in safe_deleted:
                    target = self.safe_resolve(workspace, relative)
                    if target is not None and target.is_file():
                        target.unlink()
            except Exception:
                self.restore_checkpoint(
                    workspace,
                    str(safety_checkpoint.get("checkpoint_id") or ""),
                    preserve_current=False,
                )
                raise

            return {
                "workspace_id": workspace.workspace_id,
                "snapshot_id": str(manifest.get("snapshot_id") or ""),
                "hydrated_files": hydrated_paths,
                "deleted_files": safe_deleted,
                "safety_checkpoint": safety_checkpoint,
                "status": self.get_status(workspace),
            }, None
        except Exception as exc:
            return None, str(exc)
        finally:
            shutil.rmtree(staging_root, ignore_errors=True)

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
                entries.append(
                    {
                        "path": rel,
                        "type": "dir" if p.is_dir() else "file",
                        "size": p.stat().st_size if p.is_file() else 0,
                    }
                )
                if len(entries) >= max_results:
                    break
        else:
            for p in sorted(target.iterdir()):
                if p.name == ".git":
                    continue
                rel = str(p.relative_to(workspace.base_path))
                entries.append(
                    {
                        "path": rel,
                        "type": "dir" if p.is_dir() else "file",
                        "size": p.stat().st_size if p.is_file() else 0,
                    }
                )
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
                    results.append(
                        {
                            "file": str(fpath.relative_to(workspace.base_path)),
                            "line": i + 1,
                            "match": line.rstrip()[:300],
                            "context": "\n".join(lines[start:end]),
                        }
                    )
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

        if not changed_paths and not (
            document_workspace and document_workspace.get("assembled_markdown")
        ):
            return {"files_persisted": 0, "manifest": None}

        await storage_service.initialize()

        manifest_files: List[Dict[str, Any]] = []
        failed_files: List[str] = []
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
                manifest_files.append(
                    {
                        "path": rel_path,
                        "object_path": object_path,
                        "size": len(content_bytes),
                        "sha256": _sha256(content_bytes),
                        "status": "modified"
                        if rel_path in status["modified"]
                        else "added",
                    }
                )
            except Exception as upload_err:
                failed_files.append(rel_path)
                logger.warning(f"Failed to persist {rel_path}: {upload_err}")

        # Persist assembled document markdown if present
        if document_workspace and isinstance(document_workspace, dict):
            assembled = document_workspace.get("assembled_markdown")
            if assembled and isinstance(assembled, str):
                md_bytes = assembled.encode("utf-8")
                md_path = f"workspaces/{job_id}/_assembled_document.md"
                try:
                    await storage_service.upload_to_path(
                        md_path, md_bytes, "text/markdown"
                    )
                    manifest_files.append(
                        {
                            "path": "_assembled_document.md",
                            "object_path": md_path,
                            "size": len(md_bytes),
                            "status": "added",
                        }
                    )
                except Exception as upload_err:
                    logger.warning(
                        f"Failed to persist assembled document: {upload_err}"
                    )

        manifest = {
            "type": "workspace_snapshot",
            "job_id": job_id,
            "user_id": user_id,
            "workspace_id": workspace.workspace_id,
            "source_id": workspace.source_id,
            "repo_url": workspace.repo_url,
            "files": manifest_files,
            "deleted_files": list(status["deleted"]),
            "failed_files": failed_files,
            "total_files": len(manifest_files),
            "base_files_count": len(workspace.original_hashes),
            "base_digest": hashlib.sha256(
                json.dumps(
                    workspace.original_hashes,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            "persistence_complete": not failed_files
            and len(manifest_files) >= len(changed_paths),
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

    async def persist_durable_checkpoint(
        self,
        workspace: CodingWorkspace,
        *,
        job_id: str,
        user_id: str,
        session_id: str,
        label: str = "",
        reason: str = "periodic",
    ) -> Dict[str, Any]:
        """Persist an immutable workspace delta for cross-worker resumption."""
        import mimetypes

        from app.services.storage_service import storage_service

        normalized_session_id = str(session_id or "").strip()
        if not normalized_session_id or any(
            not (char.isalnum() or char in {"-", "_"}) for char in normalized_session_id
        ):
            raise ValueError("Invalid coding workspace session ID")

        status = self.get_status(workspace)
        changed_paths = list(status["modified"]) + list(status["added"])
        checkpoint_id = f"durable-{uuid.uuid4().hex[:20]}"
        object_prefix = f"workspace_checkpoints/{normalized_session_id}/{checkpoint_id}"
        manifest_files: List[Dict[str, Any]] = []
        failed_files: List[str] = []
        workspace_state_digest = self.workspace_state_digest(workspace)

        await storage_service.initialize()
        for relative in changed_paths:
            source = self.safe_resolve(workspace, relative)
            if source is None or not source.is_file():
                failed_files.append(relative)
                continue
            try:
                content = source.read_bytes()
            except OSError:
                failed_files.append(relative)
                continue
            object_path = f"{object_prefix}/{relative}"
            content_type = (
                mimetypes.guess_type(relative)[0] or "application/octet-stream"
            )
            try:
                await storage_service.upload_to_path(
                    object_path,
                    content,
                    content_type,
                )
                manifest_files.append(
                    {
                        "path": relative,
                        "object_path": object_path,
                        "size": len(content),
                        "sha256": _sha256(content),
                        "status": (
                            "modified" if relative in status["modified"] else "added"
                        ),
                    }
                )
            except Exception as exc:
                failed_files.append(relative)
                logger.warning(
                    f"Failed to persist durable checkpoint file {relative}: {exc}"
                )

        manifest = {
            "type": "workspace_session_checkpoint",
            "checkpoint_id": checkpoint_id,
            "snapshot_id": checkpoint_id,
            "checkpoint_kind": "resumable",
            "immutable": True,
            "persistence_complete": not failed_files
            and len(manifest_files) == len(changed_paths),
            "job_id": str(job_id),
            "user_id": str(user_id),
            "session_id": normalized_session_id,
            "workspace_id": workspace.workspace_id,
            "source_id": workspace.source_id,
            "repo_url": workspace.repo_url,
            "branch": workspace.branch,
            "base_digest": self._base_digest(workspace),
            "base_files_count": len(workspace.original_hashes),
            "workspace_state_digest": workspace_state_digest,
            "files": manifest_files,
            "deleted_files": list(status["deleted"]),
            "failed_files": failed_files,
            "changes_summary": {
                "modified": len(status["modified"]),
                "added": len(status["added"]),
                "deleted": len(status["deleted"]),
            },
            "label": str(label or "").strip()[:120],
            "reason": str(reason or "periodic").strip()[:80],
            "persisted_at": datetime.now(timezone.utc).isoformat(),
        }
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
        if ws and ws.checkpoint_root and ws.checkpoint_root.exists():
            shutil.rmtree(ws.checkpoint_root, ignore_errors=True)

    def cleanup_all(self) -> None:
        """Clean up all active workspaces."""
        for wid in list(self._workspaces):
            self.cleanup(wid)
