"""A window onto the environment a run worked in.

Addressed through the JOB, never as a standalone repository browser. That is
the difference between this and a general coding IDE: you are not opening a
repo, you are asking how a particular stage produced a particular result. A
`/workspaces/{id}` route would invite the other thing, and the other thing is
not what this product is for.

What it answers:

* **What did the run change?** From the hashes of the files it started with,
  which is the one thing reading the directory cannot tell you -- a directory
  shows what it ENDS with.
* **What is in there now?** The tree, and any file's contents.

What it deliberately does not answer: what changed *line by line*. Only the
hashes of the originals are kept, not their contents, so this can say a file
was modified and not how. Storing every original would multiply the disk cost
of a feature whose whole justification is that workspaces are already large
enough to need a retention window.

Read-only throughout. Writing into a workspace from here would put a human and
an agent on the same files with nothing arbitrating, and the run's own account
of what it did would stop being true.
"""

from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.endpoints.auth import get_current_active_user
from app.core.database import get_db
from app.models.coding_workspace import CodingWorkspaceRecord
from app.models.user import User

# One definition of who may see a run. Duplicating the ownership check here is
# exactly the drift that lets two endpoints disagree about who can read what.
from app.modules.autonomy.api.job_evidence import _get_visible_job
from app.schemas.agent_job import (
    AgentJobWorkspaceEntry,
    AgentJobWorkspaceFileResponse,
    AgentJobWorkspaceResponse,
)
from app.services.coding_workspace_manager import CodingWorkspaceManager

router = APIRouter()

#: One manager for reads. It holds no workspaces of its own -- everything comes
#: through the registry -- so it is only a bag of file helpers, and those are
#: the same ones the agent's tools use. A second implementation of "resolve a
#: path inside a workspace" is a second place for a traversal bug to live.
_reader = CodingWorkspaceManager()

#: Enough of a tree to orient in, not enough to serialise a node_modules.
_MAX_ENTRIES = 300


async def _record_for_job(
    *, job_id: UUID, db: AsyncSession
) -> Optional[CodingWorkspaceRecord]:
    """The workspace this run used, most recent first.

    A run can make more than one -- a stage that re-clones, a swarm member with
    its own copy -- and the last is the one that produced the results anyone is
    looking at.
    """
    rows = await db.execute(
        select(CodingWorkspaceRecord)
        .where(CodingWorkspaceRecord.owner_job_id == job_id)
        .order_by(CodingWorkspaceRecord.last_used_at.desc())
        .limit(1)
    )
    return rows.scalar_one_or_none()


async def _open(job_id: UUID, current_user: User, db: AsyncSession):
    """The job, its workspace record, and a live handle -- or the reason not.

    Three distinct "no"s, and they are not interchangeable: this run never made
    a workspace, it made one whose files have since been swept, or you may not
    see this run at all.
    """
    job = await _get_visible_job(job_id=job_id, current_user=current_user, db=db)
    record = await _record_for_job(job_id=job.id, db=db)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="This run did not use a coding workspace.",
        )
    workspace = await _reader.load_record(str(record.id), db)
    if workspace is None:
        raise HTTPException(
            status_code=status.HTTP_410_GONE,
            detail=(
                f"Workspace {record.id} is {record.status}: its files were "
                "removed after the retention window. The run's findings still "
                "reference it, but the environment itself is gone."
            ),
        )
    return job, record, workspace


@router.get("/{job_id}/workspace", response_model=AgentJobWorkspaceResponse)
async def get_job_workspace(
    job_id: UUID,
    path: str = Query(".", description="Directory within the workspace to list"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> AgentJobWorkspaceResponse:
    """The environment this run worked in, and what it changed."""
    _job, record, workspace = await _open(job_id, current_user, db)

    # The one question the directory cannot answer on its own.
    changes = _reader.get_status(workspace)
    entries = _reader.browse_files(workspace, path=path, max_results=_MAX_ENTRIES)

    changed = set(changes.get("modified") or []) | set(changes.get("added") or [])
    listed = [
        AgentJobWorkspaceEntry(
            path=str(entry.get("path") or ""),
            is_dir=bool(entry.get("is_dir")),
            size=int(entry.get("size") or 0),
            changed=str(entry.get("path") or "") in changed,
        )
        for entry in entries
    ]

    return AgentJobWorkspaceResponse(
        job_id=str(job_id),
        workspace_id=str(record.id),
        status=str(record.status or ""),
        source_id=record.source_id,
        repo_url=record.repo_url,
        branch=record.branch,
        path=path,
        entries=listed,
        truncated=len(entries) >= _MAX_ENTRIES,
        modified=[str(p) for p in (changes.get("modified") or [])][:200],
        added=[str(p) for p in (changes.get("added") or [])][:200],
        deleted=[str(p) for p in (changes.get("deleted") or [])][:200],
    )


@router.get("/{job_id}/workspace/file", response_model=AgentJobWorkspaceFileResponse)
async def get_job_workspace_file(
    job_id: UUID,
    path: str = Query(..., description="File within the workspace"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> AgentJobWorkspaceFileResponse:
    """One file, as it stands now.

    Resolution goes through the manager's own `safe_resolve`, which is what the
    agent's tools use: a path from a URL is exactly the input a traversal
    attempt arrives on, and this endpoint must not have its own opinion about
    what is inside the workspace.
    """
    _job, record, workspace = await _open(job_id, current_user, db)

    content, error = _reader.read_file(workspace, path)
    if error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=error)

    changes = _reader.get_status(workspace)
    changed = set(changes.get("modified") or []) | set(changes.get("added") or [])

    # The real line-level change, where the workspace has git to answer with.
    # None means git cannot say -- a workspace built from knowledge-base
    # documents has no repository and no stored originals -- which is a
    # different statement from "nothing changed", and the caller renders it as
    # such.
    diff = await _reader.unified_diff(workspace, path)

    return AgentJobWorkspaceFileResponse(
        job_id=str(job_id),
        workspace_id=str(record.id),
        path=path,
        content=content or "",
        changed=path in changed,
        diff=diff,
        diff_available=diff is not None,
    )
