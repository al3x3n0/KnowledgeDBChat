from hashlib import sha256
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import HTTPException

from app.api.endpoints.code_patches import apply_code_patch_to_kb, update_code_patch
from app.api.endpoints.patch_prs import (
    approve_patch_pr,
    create_patch_pr,
    get_patch_pr,
    merge_patch_pr,
    update_patch_pr,
)
from app.core.config import settings
from app.models.agent_job import AgentJob
from app.models.code_patch_proposal import CodePatchProposal
from app.models.document import Document, DocumentSource
from app.schemas.code_patch import CodePatchProposalUpdateRequest
from app.schemas.patch_pr import (
    PatchPRApproveRequest,
    PatchPRCreateRequest,
    PatchPRMergeRequest,
    PatchPRUpdateRequest,
)
from app.services import job_dispatch
from app.services.agent_coding_runner_service import AgentCodingRunnerService


def _diff(path: str, old: str, new: str) -> str:
    return f"--- a/{path}\n+++ b/{path}\n@@ -1 +1 @@\n-{old}\n+{new}\n"


@pytest.fixture
async def library_patch(db_session, test_user, monkeypatch):
    source = DocumentSource(
        name=f"patch-{uuid4()}",
        source_type="github",
        config={"requested_by_user_id": str(test_user.id)},
    )
    db_session.add(source)
    await db_session.flush()
    doc = Document(
        source_id=source.id,
        source_identifier="main.py",
        title="main.py",
        content="before\n",
        content_hash=sha256(b"before\n").hexdigest(),
        is_processed=True,
    )
    proposal = CodePatchProposal(
        user_id=test_user.id,
        source_id=source.id,
        title="A patch",
        diff_unified=_diff("main.py", "before", "after"),
    )
    db_session.add_all([doc, proposal])
    await db_session.commit()
    queued = []
    monkeypatch.setattr(
        job_dispatch,
        "resolve_task",
        lambda _: SimpleNamespace(
            delay=lambda *args, **kwargs: queued.append((args, kwargs)),
        ),
    )
    pr = await create_patch_pr(
        PatchPRCreateRequest(title="A patch", initial_proposal_id=str(proposal.id)),
        test_user,
        db_session,
    )
    return doc, proposal, pr, queued


@pytest.mark.asyncio
async def test_requires_persisted_approval_for_exact_diff(
    db_session, test_user, library_patch
):
    doc, proposal, pr, queued = library_patch
    request = PatchPRMergeRequest(dry_run=False, require_approved=False)
    with pytest.raises(HTTPException) as refused:
        await merge_patch_pr(pr.id, request, test_user, db_session)
    assert refused.value.status_code == 422
    with pytest.raises(HTTPException) as refused:
        await apply_code_patch_to_kb(proposal.id, test_user, db_session)
    assert refused.value.status_code == 409

    approved = await approve_patch_pr(
        pr.id,
        PatchPRApproveRequest(diff_sha256=pr.diff_sha256),
        test_user,
        db_session,
    )
    assert approved.approvals[0]["proposal_id"] == str(proposal.id)
    assert (await get_patch_pr(pr.id, test_user, db_session)).approval_valid

    proposal.diff_unified = _diff("main.py", "before", "new candidate")
    await db_session.commit()
    with pytest.raises(HTTPException) as refused:
        await approve_patch_pr(
            pr.id,
            PatchPRApproveRequest(diff_sha256=pr.diff_sha256),
            test_user,
            db_session,
        )
    assert refused.value.status_code == 409
    with pytest.raises(HTTPException) as refused:
        await merge_patch_pr(pr.id, request, test_user, db_session)
    assert refused.value.status_code == 422
    assert doc.content == "before\n"
    assert queued == []


@pytest.mark.asyncio
async def test_library_apply_is_atomic_and_idempotent(
    db_session, test_user, library_patch
):
    doc, proposal, pr, queued = library_patch
    valid_diff = proposal.diff_unified
    proposal.diff_unified += _diff("missing.py", "before", "after")
    await db_session.commit()
    await approve_patch_pr(pr.id, PatchPRApproveRequest(), test_user, db_session)

    request = PatchPRMergeRequest(dry_run=False)
    result = await merge_patch_pr(pr.id, request, test_user, db_session)
    await db_session.refresh(doc)
    assert not result.ok and not result.applied_files
    assert result.errors[0]["path"] == "missing.py"
    assert doc.content == "before\n" and doc.is_processed
    assert queued == []

    replacement = CodePatchProposal(
        user_id=test_user.id,
        source_id=proposal.source_id,
        title="Clean patch",
        diff_unified=valid_diff,
    )
    db_session.add(replacement)
    await db_session.commit()
    selected = await update_patch_pr(
        pr.id,
        PatchPRUpdateRequest(selected_proposal_id=str(replacement.id)),
        test_user,
        db_session,
    )
    assert selected.approvals == [] and not selected.approval_valid
    assert (
        await merge_patch_pr(pr.id, PatchPRMergeRequest(), test_user, db_session)
    ).ok
    assert doc.content == "before\n" and queued == []

    await approve_patch_pr(pr.id, PatchPRApproveRequest(), test_user, db_session)
    assert (await merge_patch_pr(pr.id, request, test_user, db_session)).ok
    assert (await merge_patch_pr(pr.id, request, test_user, db_session)).ok
    assert doc.content == "after\n" and not doc.is_processed
    assert len(queued) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("first_apply", ["review", "worker"])
async def test_a_proposal_can_change_library_content_only_once(
    db_session, test_user, library_patch, monkeypatch, first_apply
):
    doc, proposal, first_review, queued = library_patch
    proposal.diff_unified = "--- a/main.py\n+++ b/main.py\n@@ -0,0 +1 @@\n+inserted\n"
    await db_session.commit()
    with pytest.raises(HTTPException) as refused:
        await update_code_patch(
            proposal.id,
            CodePatchProposalUpdateRequest(status="applied"),
            test_user,
            db_session,
        )
    assert refused.value.status_code == 409

    second_review = await create_patch_pr(
        PatchPRCreateRequest(
            title="Another review", initial_proposal_id=str(proposal.id)
        ),
        test_user,
        db_session,
    )
    for review in (first_review, second_review):
        await approve_patch_pr(
            review.id, PatchPRApproveRequest(), test_user, db_session
        )

    monkeypatch.setattr(settings, "AGENT_KB_PATCH_APPLY_ENABLED", True)
    job = AgentJob(
        name="Apply patch",
        goal="Apply the test proposal",
        job_type="analysis",
        user_id=test_user.id,
        config={
            "apply_patch_to_kb": True,
            "dry_run": False,
            "proposal_id": str(proposal.id),
            "require_experiments_ok": False,
        },
    )
    db_session.add(job)
    await db_session.commit()
    runner = AgentCodingRunnerService()
    request = PatchPRMergeRequest(dry_run=False)
    if first_apply == "review":
        assert (
            await merge_patch_pr(first_review.id, request, test_user, db_session)
        ).ok
        assert (await apply_code_patch_to_kb(proposal.id, test_user, db_session))[
            "status"
        ] == "applied"
    else:
        result = await runner.run_code_patch_apply_to_kb(
            None, job=job, db=db_session, progress_callback=None
        )
        assert result["results"]["code_patch_kb_apply"]["did_apply"]

    with pytest.raises(HTTPException) as refused:
        await merge_patch_pr(second_review.id, request, test_user, db_session)
    assert refused.value.status_code == 409
    assert "already applied" in refused.value.detail
    with pytest.raises(HTTPException) as refused:
        await update_code_patch(
            proposal.id,
            CodePatchProposalUpdateRequest(status="proposed"),
            test_user,
            db_session,
        )
    assert refused.value.status_code == 409

    retry = await runner.run_code_patch_apply_to_kb(
        None, job=job, db=db_session, progress_callback=None
    )
    outcome = retry["results"]["code_patch_kb_apply"]
    assert outcome["ok"] and outcome["already_applied"] and not outcome["did_apply"]
    await db_session.refresh(doc)
    assert doc.content == "inserted\nbefore\n"
    assert len(queued) == 1
