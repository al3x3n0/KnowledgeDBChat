import asyncio
from datetime import datetime, timedelta
from uuid import UUID

from app.models.research_note import ResearchNote
from app.models.synthesis_job import SynthesisJob


def test_get_research_note_marks_completed_pending_reevaluation_as_review_ready(
    client, db_session, test_user, auth_headers
):
    completed_at = datetime.utcnow()
    note = ResearchNote(
        user_id=test_user.id,
        title="Review Ready Note",
        content_markdown="## Ranked hypotheses",
        structured_payload={
            "artifact_type": "hypothesis_reevaluation",
            "summary": "Reevaluated note",
            "hypotheses": [
                {
                    "id": "hyp-1",
                    "rank": 1,
                    "title": "Review-ready hypothesis",
                    "claim": "Draft completion should surface review CTA.",
                    "rationale": "Note fetch should reconcile pending status.",
                    "novelty_score": 0.7,
                    "evidence_score": 0.8,
                    "testability_score": 0.9,
                    "overall_score": 0.82,
                    "recommended_next_step": "Review the draft.",
                }
            ],
            "pending_reevaluation_job_id": "11111111-1111-1111-1111-111111111111",
            "pending_reevaluation_created_at": (
                completed_at - timedelta(minutes=5)
            ).isoformat(),
            "pending_reevaluation_status": "pending",
        },
    )
    job = SynthesisJob(
        id=UUID("11111111-1111-1111-1111-111111111111"),
        user_id=test_user.id,
        job_type="hypothesis_reevaluation",
        title="Queued draft",
        description="Queued from note",
        document_ids=[],
        paper_ids=[],
        research_note_id=note.id,
        output_format="markdown",
        output_style="technical",
        status="completed",
        progress=100,
        result_content="done",
        completed_at=completed_at,
    )

    async def _seed():
        db_session.add(note)
        await db_session.flush()
        job.research_note_id = note.id
        db_session.add(job)
        await db_session.commit()

    asyncio.get_event_loop().run_until_complete(_seed())

    response = client.get(f"/api/v1/research-notes/{note.id}", headers=auth_headers)

    assert response.status_code == 200
    payload = response.json()
    assert payload["structured_payload"]["pending_reevaluation_status"] == "completed"
    assert (
        payload["structured_payload"]["pending_reevaluation_completed_at"] is not None
    )


def test_get_research_note_marks_failed_pending_reevaluation_with_error(
    client, db_session, test_user, auth_headers
):
    note = ResearchNote(
        user_id=test_user.id,
        title="Failed Draft Note",
        content_markdown="## Ranked hypotheses",
        structured_payload={
            "artifact_type": "hypothesis_reevaluation",
            "summary": "Reevaluated note",
            "hypotheses": [
                {
                    "id": "hyp-1",
                    "rank": 1,
                    "title": "Failed hypothesis",
                    "claim": "Failure should surface on read.",
                    "rationale": "Note fetch should reconcile failures.",
                    "novelty_score": 0.7,
                    "evidence_score": 0.8,
                    "testability_score": 0.9,
                    "overall_score": 0.82,
                    "recommended_next_step": "Retry reevaluation.",
                }
            ],
            "pending_reevaluation_job_id": "22222222-2222-2222-2222-222222222222",
            "pending_reevaluation_created_at": datetime.utcnow().isoformat(),
            "pending_reevaluation_status": "pending",
        },
    )
    job = SynthesisJob(
        id=UUID("22222222-2222-2222-2222-222222222222"),
        user_id=test_user.id,
        job_type="hypothesis_reevaluation",
        title="Failed draft",
        description="Queued from note",
        document_ids=[],
        paper_ids=[],
        research_note_id=note.id,
        output_format="markdown",
        output_style="technical",
        status="failed",
        progress=100,
        result_content=None,
        error="Model timeout during reevaluation",
        completed_at=datetime.utcnow(),
    )

    async def _seed():
        db_session.add(note)
        await db_session.flush()
        job.research_note_id = note.id
        db_session.add(job)
        await db_session.commit()

    asyncio.get_event_loop().run_until_complete(_seed())

    response = client.get(f"/api/v1/research-notes/{note.id}", headers=auth_headers)

    assert response.status_code == 200
    payload = response.json()
    assert payload["structured_payload"]["pending_reevaluation_status"] == "failed"
    assert (
        "timeout" in payload["structured_payload"]["pending_reevaluation_error"].lower()
    )


def test_get_research_note_marks_completed_pending_reevaluation_as_stale_when_newer_evidence_exists(
    client, db_session, test_user, auth_headers
):
    completed_at = datetime.utcnow() - timedelta(minutes=10)
    last_appended_at = datetime.utcnow().isoformat()
    note = ResearchNote(
        user_id=test_user.id,
        title="Stale Draft Note",
        content_markdown="## Ranked hypotheses",
        structured_payload={
            "artifact_type": "hypothesis_reevaluation",
            "summary": "Reevaluated note",
            "hypotheses": [
                {
                    "id": "hyp-1",
                    "rank": 1,
                    "title": "Stale hypothesis",
                    "claim": "New evidence should stale a completed draft.",
                    "rationale": "The review CTA should not imply freshness.",
                    "novelty_score": 0.7,
                    "evidence_score": 0.8,
                    "testability_score": 0.9,
                    "overall_score": 0.82,
                    "recommended_next_step": "Start a fresh reevaluation.",
                }
            ],
            "last_appended_at": last_appended_at,
            "pending_reevaluation_job_id": "33333333-3333-3333-3333-333333333333",
            "pending_reevaluation_created_at": (
                completed_at - timedelta(minutes=5)
            ).isoformat(),
            "pending_reevaluation_status": "pending",
        },
    )
    job = SynthesisJob(
        id=UUID("33333333-3333-3333-3333-333333333333"),
        user_id=test_user.id,
        job_type="hypothesis_reevaluation",
        title="Completed stale draft",
        description="Queued from note",
        document_ids=[],
        paper_ids=[],
        research_note_id=note.id,
        output_format="markdown",
        output_style="technical",
        status="completed",
        progress=100,
        result_content="done",
        completed_at=completed_at,
    )

    async def _seed():
        db_session.add(note)
        await db_session.flush()
        job.research_note_id = note.id
        db_session.add(job)
        await db_session.commit()

    asyncio.get_event_loop().run_until_complete(_seed())

    response = client.get(f"/api/v1/research-notes/{note.id}", headers=auth_headers)

    assert response.status_code == 200
    payload = response.json()
    assert payload["structured_payload"]["pending_reevaluation_status"] == "stale"
