"""HTTP boundary for a run's evidence.

The findings a run produced, what the contract asked for, and which of the two
met the other. Read-only: the contract result already recorded on the job is
the authority on whether it was satisfied, and this endpoint shows what that
verdict was reached from rather than recomputing it.

Separate from the job record endpoint because evidence is large and the job
list is not the place for it -- a list of forty runs should not carry forty
sets of findings to show a count.
"""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.endpoints.auth import get_current_active_user
from app.core.database import get_db
from app.models.agent_job import AgentJob
from app.models.agent_retraction import RetractionKind
from app.models.user import User
from app.schemas.agent_job import (
    AgentJobEvidenceDisputeRequest,
    AgentJobEvidenceDisputeResponse,
    AgentJobEvidenceResponse,
)
from app.services import agent_evidence_view, agent_retraction_service

router = APIRouter()


async def _get_visible_job(
    *, job_id: UUID, current_user: User, db: AsyncSession
) -> AgentJob:
    """The job, if it is this user's or they are an admin.

    404 rather than 403 for someone else's, matching every other read of a job:
    whether a run exists should not be discoverable by asking for it.
    """
    query = select(AgentJob).where(AgentJob.id == job_id)
    if not getattr(current_user, "is_admin", False):
        query = select(AgentJob).where(
            and_(AgentJob.id == job_id, AgentJob.user_id == current_user.id)
        )
    result = await db.execute(query)
    job = result.scalar_one_or_none()
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Agent job not found"
        )
    return job


@router.get("/{job_id}/evidence", response_model=AgentJobEvidenceResponse)
async def get_job_evidence(
    job_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> AgentJobEvidenceResponse:
    """What this run found, and whether it is what was asked for."""
    job = await _get_visible_job(job_id=job_id, current_user=current_user, db=db)
    disputes = await agent_retraction_service.disputed_findings(
        db, user_id=job.user_id, job_id=job.id
    )
    return AgentJobEvidenceResponse(**agent_evidence_view.build(job, disputes))


def _finding_count(job: AgentJob) -> int:
    results = job.results if isinstance(job.results, dict) else {}
    raw = results.get("findings")
    return len([f for f in raw if isinstance(f, dict)]) if isinstance(raw, list) else 0


@router.post(
    "/{job_id}/evidence/{index}/dispute",
    response_model=AgentJobEvidenceDisputeResponse,
)
async def dispute_job_evidence(
    job_id: UUID,
    index: int,
    payload: AgentJobEvidenceDisputeRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> AgentJobEvidenceDisputeResponse:
    """Reject one result, with a reason. Advisory, by design.

    Nothing downstream is invalidated and the contract's verdict is unchanged:
    a rejection that silently took down hours of dependent work would be a
    click nobody could risk. What it does instead is TRAVEL -- a restart of
    this stage begins with the rejection as an operator correction, and a run
    that would cite this evidence again is told it was withdrawn. A rejection
    nobody reads is a note in a drawer.
    """
    job = await _get_visible_job(job_id=job_id, current_user=current_user, db=db)

    total = _finding_count(job)
    if index < 0 or index >= total:
        # The index addresses a position in this run's findings. Rejecting one
        # that does not exist would record a dispute against nothing, and it
        # would never be shown.
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(f"This run has {total} findings; there is no finding {index}."),
        )

    reason = payload.reason.strip()
    try:
        await agent_retraction_service.retract(
            db,
            user_id=job.user_id,
            kind=RetractionKind.FINDING,
            ref=agent_retraction_service.finding_ref(job.id, index),
            reason=reason,
            source=f"operator:{current_user.username}",
            source_job_id=job.id,
        )
    except ValueError as error:
        # The service's own refusals -- chiefly the missing reason. Pydantic's
        # min_length accepts "   ", so whitespace reaches here and came back as
        # a 500: an ordinary typing mistake reported as a server fault.
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error))
    # The service flushes and leaves the transaction to its caller, so that an
    # agent can retract several things as one unit. An HTTP request is that
    # unit here, and without this the row is rolled back when the session
    # closes -- the dispute reports success and is gone on the next read.
    await db.commit()
    return AgentJobEvidenceDisputeResponse(
        job_id=str(job.id), index=index, reason=reason
    )


@router.delete("/{job_id}/evidence/{index}/dispute", status_code=status.HTTP_200_OK)
async def withdraw_job_evidence_dispute(
    job_id: UUID,
    index: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> dict:
    """Take a rejection back -- a measurement re-taken and found good after all.

    The reason this is possible at all is why disputes are computed on read
    rather than written into the run: a view that had already rewritten every
    dependent record would have no way back.
    """
    job = await _get_visible_job(job_id=job_id, current_user=current_user, db=db)
    ref = agent_retraction_service.finding_ref(job.id, index)
    rows = await agent_retraction_service.retractions(
        db, user_id=job.user_id, kind=RetractionKind.FINDING
    )
    removed = 0
    for row in rows:
        if str(row.subject_ref) == ref:
            if await agent_retraction_service.withdraw(db, row.id):
                removed += 1
    if removed:
        # Same reason as the dispute above: withdraw() flushes, the caller commits.
        await db.commit()
    return {"job_id": str(job.id), "index": index, "withdrawn": removed}
