import asyncio
import hashlib
from datetime import datetime

from app.models.document import Document, DocumentSource
from app.models.research_note import ResearchNote
from app.models.research_paper import PaperClaim, ResearchPaper
from app.services.paper_extraction_service import paper_extraction_service


def _seed_arxiv_document(db_session, test_user):
    source = DocumentSource(
        name="arXiv Test Source",
        source_type="arxiv",
        config={"requested_by_user_id": str(test_user.id)},
    )
    content = "Compiler optimization paper content."
    document = Document(
        title="Compiler Optimization via Layouts",
        content=content,
        summary="A paper about compiler optimization via layouts.",
        content_hash=hashlib.sha256(content.encode()).hexdigest(),
        file_type="pdf",
        source_identifier="2401.12345",
        source=source,
        is_processed=True,
        url="https://arxiv.org/abs/2401.12345",
        extra_metadata={"authors": ["Ada Lovelace"], "categories": ["cs.PL"]},
    )

    async def _run():
        db_session.add(source)
        await db_session.flush()
        db_session.add(document)
        await db_session.commit()
        await db_session.refresh(document)
        await db_session.refresh(document, ["source"])
        return source, document

    return asyncio.get_event_loop().run_until_complete(_run())


def test_queue_document_extraction_job_preserves_previous_completed_state_on_failure(
    db_session,
    test_user,
):
    _, document = _seed_arxiv_document(db_session, test_user)

    async def _seed_paper():
        paper = ResearchPaper(
            user_id=test_user.id,
            document_id=document.id,
            source_id=document.source_id,
            arxiv_id=document.source_identifier,
            title=document.title,
            extraction_status="completed",
            extracted_at=datetime.utcnow(),
            extractor_version="paper_extraction_v1",
            summary="Existing extracted summary",
            mechanisms=["layout transform"],
            raw_extraction_payload={"summary": "Existing extracted summary", "claims": []},
        )
        db_session.add(paper)
        await db_session.commit()
        await db_session.refresh(paper)
        return paper

    original_paper = asyncio.get_event_loop().run_until_complete(_seed_paper())
    dispatched: list[str] = []

    async def _queue_and_fail():
        job = await paper_extraction_service.queue_document_extraction_job(
            db_session,
            document=document,
            user_id=test_user.id,
            force=True,
            dispatch_task=lambda job_id: dispatched.append(job_id),
        )
        await paper_extraction_service.mark_failed(db_session, job=job, error="llm failure")
        paper = await db_session.get(ResearchPaper, original_paper.id)
        return job, paper

    job, paper = asyncio.get_event_loop().run_until_complete(_queue_and_fail())

    assert dispatched == [str(job.id)]
    assert job.status == "failed"
    assert paper is not None
    assert paper.extraction_status == "completed"
    assert paper.summary == "Existing extracted summary"
    assert paper.raw_extraction_payload == {"summary": "Existing extracted summary", "claims": []}


def test_extract_research_papers_endpoint_queues_jobs_from_source(
    client,
    db_session,
    test_user,
    auth_headers,
    monkeypatch,
):
    from app.tasks.paper_extraction_tasks import extract_paper_job

    _, document = _seed_arxiv_document(db_session, test_user)
    queued: list[str] = []

    def _fake_delay(job_id: str):
        queued.append(job_id)

    monkeypatch.setattr(extract_paper_job, "delay", _fake_delay)

    response = client.post(
        "/api/v1/research/papers/extract",
        headers=auth_headers,
        json={"source_id": str(document.source_id), "force": False, "limit": 50},
    )

    assert response.status_code == 202
    payload = response.json()
    assert len(payload) == 1
    assert payload[0]["document_id"] == str(document.id)
    assert payload[0]["status"] == "pending"
    assert queued == [payload[0]["id"]]


def test_save_research_paper_as_note_persists_structured_payload(
    client,
    db_session,
    test_user,
    auth_headers,
):
    _, document = _seed_arxiv_document(db_session, test_user)

    async def _seed():
        paper = ResearchPaper(
            user_id=test_user.id,
            document_id=document.id,
            source_id=document.source_id,
            arxiv_id=document.source_identifier,
            title=document.title,
            extraction_status="completed",
            extracted_at=datetime.utcnow(),
            extractor_version="paper_extraction_v1",
            summary="Structured paper summary",
            mechanisms=["layout transform"],
            assumptions=["regular access patterns"],
            benchmarks=["PolyBench"],
            metrics=["runtime"],
            limitations=["not evaluated on irregular kernels"],
            raw_extraction_payload={"summary": "Structured paper summary", "claims": [{"statement": "Claim"}]},
        )
        db_session.add(paper)
        await db_session.flush()
        claim = PaperClaim(
            paper_id=paper.id,
            kind="performance",
            statement="If layout is optimized, runtime improves.",
            target_layer="midend",
            evidence_summary="Measured on PolyBench",
            rank=1,
        )
        db_session.add(claim)
        await db_session.commit()
        await db_session.refresh(paper)
        return paper

    paper = asyncio.get_event_loop().run_until_complete(_seed())

    response = client.post(
        f"/api/v1/research/papers/{paper.id}/save-as-note",
        headers=auth_headers,
        json={"tags": ["paper-extraction", "compiler"]},
    )

    assert response.status_code == 201
    payload = response.json()

    async def _load_note():
        return await db_session.get(ResearchNote, payload["id"])

    note = asyncio.get_event_loop().run_until_complete(_load_note())
    assert note is not None
    assert note.structured_payload["artifact_type"] == "paper_extraction"
    assert note.structured_payload["paper_id"] == str(paper.id)
    assert note.structured_payload["claims"][0]["statement"] == "If layout is optimized, runtime improves."
