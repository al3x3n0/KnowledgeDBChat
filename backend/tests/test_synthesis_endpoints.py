import asyncio
import hashlib
from datetime import datetime
from uuid import UUID

from app.models.research_note import ResearchNote
from app.models.document import Document, DocumentSource
from app.models.agent_job import AgentJob, AgentJobStatus
from app.models.domain_research_profile import DomainResearchProfile
from app.models.experiment import ExperimentPlan, ExperimentRun
from app.models.notification import Notification
from app.models.synthesis_job import SynthesisJob
from app.models.research_paper import ResearchPaper, PaperClaim


def test_synthesis_types_info_includes_decision_memo(client, auth_headers):
    response = client.get("/api/v1/synthesis/types/info", headers=auth_headers)

    assert response.status_code == 200
    payload = response.json()
    values = {item["value"] for item in payload["types"]}
    assert "decision_memo" in values


def test_create_decision_memo_synthesis_job(client, db_session, test_user, auth_headers, monkeypatch):
    from app.tasks.synthesis_tasks import execute_synthesis_task

    queued: list[tuple[str, str]] = []

    def _fake_delay(job_id: str, user_id: str):
        queued.append((job_id, user_id))

    monkeypatch.setattr(execute_synthesis_task, "delay", _fake_delay)

    source = DocumentSource(
        name="Test Decision Memo Source",
        source_type="file",
        config={},
    )
    document = Document(
        title="Source A",
        content="Claim one. Claim two.",
        summary="Summary for source A.",
        content_hash=hashlib.sha256(b"Claim one. Claim two.").hexdigest(),
        file_type="md",
        source_identifier="source-a",
        source=source,
        is_processed=True,
    )

    async def _seed():
        db_session.add(source)
        await db_session.flush()
        db_session.add(document)
        await db_session.commit()
        await db_session.refresh(document)

    asyncio.get_event_loop().run_until_complete(_seed())

    response = client.post(
        "/api/v1/synthesis",
        headers=auth_headers,
        json={
            "job_type": "decision_memo",
            "title": "Vendor Monitoring Memo",
            "document_ids": [str(document.id)],
            "topic": "Vendor posture",
            "options": {
                "audience": "procurement",
                "include_recommendations": True,
                "include_watchlist": True,
            },
            "output_format": "markdown",
            "output_style": "professional",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["job_type"] == "decision_memo"
    assert payload["topic"] == "Vendor posture"
    assert payload["options"]["audience"] == "procurement"
    assert len(queued) == 1
    assert queued[0][1] == str(test_user.id)

    async def _load_job():
        return await db_session.get(SynthesisJob, payload["id"])

    job = asyncio.get_event_loop().run_until_complete(_load_job())
    assert job is not None
    assert job.job_type == "decision_memo"


def test_create_decision_memo_with_search_query_only(client, test_user, auth_headers, monkeypatch):
    from app.tasks.synthesis_tasks import execute_synthesis_task

    queued: list[tuple[str, str]] = []

    def _fake_delay(job_id: str, user_id: str):
        queued.append((job_id, user_id))

    monkeypatch.setattr(execute_synthesis_task, "delay", _fake_delay)

    response = client.post(
        "/api/v1/synthesis",
        headers=auth_headers,
        json={
            "job_type": "decision_memo",
            "title": "Search-backed Memo",
            "document_ids": [],
            "search_query": "vendor pricing reliability updates",
            "topic": "Vendor monitoring",
            "output_format": "markdown",
            "output_style": "professional",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["job_type"] == "decision_memo"
    assert payload["document_ids"] == []
    assert payload["search_query"] == "vendor pricing reliability updates"
    assert len(queued) == 1


def test_save_completed_synthesis_job_as_research_note(client, db_session, test_user, auth_headers):
    source = DocumentSource(
        name="Completed Job Source",
        source_type="file",
        config={},
    )
    document = Document(
        title="Source B",
        content="Evidence for a decision memo.",
        summary="Decision memo evidence.",
        content_hash=hashlib.sha256(b"Evidence for a decision memo.").hexdigest(),
        file_type="md",
        source_identifier="source-b",
        source=source,
        is_processed=True,
    )
    job = SynthesisJob(
        user_id=test_user.id,
        job_type="decision_memo",
        title="Completed Memo",
        document_ids=[],
        topic="Market posture",
        options={"audience": "executive_leadership"},
        output_format="markdown",
        output_style="professional",
        status="completed",
        progress=100,
        result_content="# Completed Memo\n\n- Bottom line [Source: Source B]",
        result_metadata={"report_type": "decision_memo", "documents_analyzed": 1},
    )

    async def _seed():
        db_session.add(source)
        await db_session.flush()
        db_session.add(document)
        await db_session.flush()
        job.document_ids = [str(document.id)]
        db_session.add(job)
        await db_session.commit()
        await db_session.refresh(job)

    asyncio.get_event_loop().run_until_complete(_seed())

    response = client.post(
        f"/api/v1/synthesis/{job.id}/save-as-note",
        headers=auth_headers,
        json={},
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["title"] == "Completed Memo"
    assert payload["source_synthesis_job_id"] == str(job.id)
    assert payload["tags"] == ["decision-memo", "research-synthesis", "citations"]
    assert payload["attribution"]["saved_from_synthesis"]["job_type"] == "decision_memo"

    async def _load_note():
        return await db_session.get(ResearchNote, payload["id"])

    note = asyncio.get_event_loop().run_until_complete(_load_note())
    assert note is not None
    assert note.source_document_ids == [str(document.id)]


def test_create_gap_analysis_synthesis_job_from_papers(client, db_session, test_user, auth_headers, monkeypatch):
    from app.tasks.synthesis_tasks import execute_synthesis_task

    queued: list[tuple[str, str]] = []

    def _fake_delay(job_id: str, user_id: str):
        queued.append((job_id, user_id))

    monkeypatch.setattr(execute_synthesis_task, "delay", _fake_delay)

    source = DocumentSource(
        name="Paper Source",
        source_type="arxiv",
        config={},
    )
    document = Document(
        title="Paper-backed source",
        content="Paper content",
        summary="Paper summary",
        content_hash=hashlib.sha256(b"Paper content").hexdigest(),
        file_type="pdf",
        source_identifier="2401.99999",
        source=source,
        is_processed=True,
    )

    async def _seed():
        db_session.add(source)
        await db_session.flush()
        db_session.add(document)
        await db_session.flush()
        paper = ResearchPaper(
            user_id=test_user.id,
            document_id=document.id,
            source_id=source.id,
            arxiv_id="2401.99999",
            title="Paper-backed source",
            extraction_status="completed",
            extracted_at=datetime.utcnow(),
            summary="Structured summary",
        )
        db_session.add(paper)
        await db_session.commit()
        await db_session.refresh(paper)
        return paper

    paper = asyncio.get_event_loop().run_until_complete(_seed())

    response = client.post(
        "/api/v1/synthesis",
        headers=auth_headers,
        json={
            "job_type": "gap_analysis_hypotheses",
            "title": "Compiler hypotheses",
            "document_ids": [],
            "paper_ids": [str(paper.id)],
            "topic": "compiler optimization",
            "output_format": "markdown",
            "output_style": "technical",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["paper_ids"] == [str(paper.id)]
    assert queued


def test_save_completed_gap_analysis_as_structured_research_note(client, db_session, test_user, auth_headers):
    source = DocumentSource(
        name="Extracted Paper Source",
        source_type="arxiv",
        config={},
    )
    document = Document(
        title="Compiler layouts",
        content="Evidence",
        summary="Evidence summary",
        content_hash=hashlib.sha256(b"Evidence").hexdigest(),
        file_type="pdf",
        source_identifier="2402.11111",
        source=source,
        is_processed=True,
    )
    job = SynthesisJob(
        user_id=test_user.id,
        job_type="gap_analysis_hypotheses",
        title="Paper hypotheses",
        document_ids=[],
        paper_ids=[],
        topic="Compiler layouts",
        output_format="markdown",
        output_style="technical",
        status="completed",
        progress=100,
        result_content="# Hypotheses\n\n## Testable Hypotheses\n- Idea",
        result_metadata={
            "summary": "Cross-paper synthesis of compiler layout ideas.",
            "structured_hypotheses": [
                {
                    "id": "h1",
                    "rank": 1,
                    "title": "Layout plus scheduling",
                    "claim": "If layout transforms are paired with schedule-aware pressure control, then locality gains survive register pressure.",
                    "rationale": "Two papers optimize adjacent bottlenecks.",
                    "novelty_score": 0.84,
                    "evidence_score": 0.67,
                    "testability_score": 0.91,
                    "overall_score": 0.81,
                    "supporting_sources": [{"id": "paper-1", "title": "Compiler layouts"}],
                    "recommended_next_step": "Implement on three stencil kernels.",
                }
            ],
            "structured_gaps": ["No paper tested both mechanisms together."],
        },
    )

    async def _seed():
        db_session.add(source)
        await db_session.flush()
        db_session.add(document)
        await db_session.flush()
        paper = ResearchPaper(
            user_id=test_user.id,
            document_id=document.id,
            source_id=source.id,
            arxiv_id="2402.11111",
            title="Compiler layouts",
            extraction_status="completed",
            extracted_at=datetime.utcnow(),
            summary="Structured summary",
        )
        db_session.add(paper)
        await db_session.flush()
        claim = PaperClaim(
            paper_id=paper.id,
            kind="performance",
            statement="Layouts improve locality.",
            target_layer="midend",
            rank=1,
        )
        db_session.add(claim)
        await db_session.flush()
        job.paper_ids = [str(paper.id)]
        job.document_ids = []
        db_session.add(job)
        await db_session.commit()
        await db_session.refresh(job)
        return job, paper

    job, paper = asyncio.get_event_loop().run_until_complete(_seed())

    response = client.post(
        f"/api/v1/synthesis/{job.id}/save-as-note",
        headers=auth_headers,
        json={},
    )

    assert response.status_code == 201
    payload = response.json()

    async def _load_note():
        return await db_session.get(ResearchNote, payload["id"])

    note = asyncio.get_event_loop().run_until_complete(_load_note())
    assert note is not None
    assert note.structured_payload["research_mode"] == "paper_to_hypothesis"
    assert note.structured_payload["source_paper_ids"] == [str(paper.id)]
    assert note.structured_payload["hypotheses"][0]["title"] == "Layout plus scheduling"
    assert str(document.id) in (note.source_document_ids or [])


def test_create_hypothesis_reevaluation_job_from_research_note(client, db_session, test_user, auth_headers, monkeypatch):
    from app.tasks.synthesis_tasks import execute_synthesis_task

    queued: list[tuple[str, str]] = []

    def _fake_delay(job_id: str, user_id: str):
        queued.append((job_id, user_id))

    monkeypatch.setattr(execute_synthesis_task, "delay", _fake_delay)

    note = ResearchNote(
        user_id=test_user.id,
        title="Compiler hypotheses",
        content_markdown="## Ranked hypotheses",
        structured_payload={
            "artifact_type": "hypothesis_synthesis",
            "research_mode": "paper_to_hypothesis",
            "summary": "Structured memo",
            "hypotheses": [
                {
                    "id": "hyp-1",
                    "rank": 1,
                    "title": "Layout-aware scheduling",
                    "claim": "Joint transforms may preserve locality under pressure.",
                    "rationale": "Combined evidence from prior papers.",
                    "novelty_score": 0.7,
                    "evidence_score": 0.6,
                    "testability_score": 0.9,
                    "overall_score": 0.78,
                    "recommended_next_step": "Run kernel benchmarks.",
                    "experiment_evidence": [
                        {
                            "run_id": "run-1",
                            "status": "completed",
                            "summary": "Initial benchmark evidence",
                        }
                    ],
                }
            ],
        },
    )

    async def _seed():
        db_session.add(note)
        await db_session.commit()
        await db_session.refresh(note)
        return note

    note = asyncio.get_event_loop().run_until_complete(_seed())

    response = client.post(
        "/api/v1/synthesis",
        headers=auth_headers,
        json={
            "job_type": "hypothesis_reevaluation",
            "title": "Compiler hypothesis re-evaluation",
            "document_ids": [],
            "research_note_id": str(note.id),
            "output_format": "markdown",
            "output_style": "technical",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["job_type"] == "hypothesis_reevaluation"
    assert payload["research_note_id"] == str(note.id)
    assert queued


def test_save_completed_hypothesis_reevaluation_updates_target_note(client, db_session, test_user, auth_headers, monkeypatch):
    note = ResearchNote(
        user_id=test_user.id,
        title="Compiler hypotheses",
        content_markdown="## Ranked hypotheses",
        structured_payload={
            "artifact_type": "hypothesis_synthesis",
            "research_mode": "paper_to_hypothesis",
            "summary": "Structured memo",
            "source_paper_ids": ["paper-1"],
            "source_document_ids": ["doc-1"],
            "hypotheses": [
                {
                    "id": "hyp-1",
                    "rank": 1,
                    "title": "Layout-aware scheduling",
                    "claim": "Joint transforms may preserve locality under pressure.",
                    "rationale": "Combined evidence from prior papers.",
                    "novelty_score": 0.7,
                    "evidence_score": 0.6,
                    "testability_score": 0.9,
                    "overall_score": 0.78,
                    "recommended_next_step": "Run kernel benchmarks.",
                    "experiment_evidence": [
                        {
                            "run_id": "run-1",
                            "status": "completed",
                            "summary": "Initial benchmark evidence",
                            "autonomous_origin": {
                                "source_kind": "profile",
                                "source_id": "99999999-9999-9999-9999-999999999999",
                                "opportunity_id": "hyp-1",
                                "evidence_revision_at_launch": "seedrev1",
                            },
                        }
                    ],
                }
            ],
            "last_appended_run_id": "run-1",
            "last_appended_at": "2026-03-27T00:00:00Z",
            "pending_reevaluation_job_id": "11111111-1111-1111-1111-111111111111",
            "pending_reevaluation_created_at": "2026-03-27T01:00:00Z",
            "pending_reevaluation_reason": "new_experiment_evidence",
            "pending_reevaluation_source_run_ids": ["run-1"],
        },
    )
    job = SynthesisJob(
        user_id=test_user.id,
        job_type="hypothesis_reevaluation",
        title="Compiler hypothesis re-evaluation",
        description="Evidence-aware reprioritization",
        document_ids=[],
        paper_ids=[],
        research_note_id=note.id,
        output_format="markdown",
        output_style="technical",
        status="completed",
        progress=100,
        result_content="# Hypothesis Re-evaluation\n\nUpdated rankings.",
        result_metadata={
            "summary": "Evidence-aware reprioritization.",
            "reprioritization_summary": "Hypothesis one stays on top after positive run evidence.",
            "structured_hypotheses": [
                {
                    "id": "hyp-1",
                    "rank": 1,
                    "title": "Layout-aware scheduling",
                    "claim": "Joint transforms preserve locality under pressure.",
                    "rationale": "The new run supports the combined mechanism.",
                    "novelty_score": 0.71,
                    "evidence_score": 0.86,
                    "testability_score": 0.9,
                    "overall_score": 0.84,
                    "supporting_sources": [{"id": "paper-1", "title": "Compiler layouts"}],
                    "recommended_next_step": "Expand to multi-architecture validation.",
                }
            ],
            "priority_deltas": [
                {
                    "hypothesis_id": "hyp-1",
                    "previous_rank": 1,
                    "new_rank": 1,
                    "status": "unchanged",
                    "reason": "Positive evidence reinforced priority.",
                }
            ],
            "archived_hypothesis_ids": [],
        },
    )
    notification = Notification(
        user_id=test_user.id,
        notification_type="hypothesis_reevaluation_update",
        title="Reevaluation ready: Compiler hypotheses",
        message="Draft ready to review.",
        priority="normal",
        related_entity_type="research_note",
        related_entity_id=note.id,
        data={
            "note_id": str(note.id),
            "reevaluation_job_id": str(job.id),
            "reevaluation_status": "completed",
        },
        action_url=f"/research-notes?note={note.id}",
    )
    parent_job = AgentJob(
        user_id=test_user.id,
        name="Compiler Parent",
        goal="Track compiler opportunities",
        job_type="research",
        status=AgentJobStatus.COMPLETED.value,
        progress=100,
        config={},
    )
    profile = DomainResearchProfile(
        id=UUID("99999999-9999-9999-9999-999999999999"),
        user_id=test_user.id,
        title="Compiler Monitor",
        domain="Compiler",
        objective="Track compiler opportunities",
        status="active",
        latest_run_job_id=parent_job.id,
        latest_summary={
            "opportunities": [
                {
                    "opportunity_id": "hyp-1",
                    "title": "Layout-aware scheduling",
                    "hypothesis": "Joint transforms may preserve locality under pressure.",
                    "decision_state": "accepted",
                    "stage": "accepted",
                    "confidence": 0.6,
                    "readiness": 0.78,
                    "novelty": 0.7,
                    "source_note_ids": [],
                    "supporting_evidence": ["Initial benchmark evidence"],
                    "supporting_sources": [{"id": "paper-1", "title": "Compiler layouts"}],
                    "autonomous_origin": {
                        "source_kind": "profile",
                        "source_id": "99999999-9999-9999-9999-999999999999",
                        "opportunity_id": "hyp-1",
                        "evidence_revision_at_launch": "seedrev1",
                    },
                }
            ]
        },
    )
    queued_jobs: list[tuple[str, str]] = []

    async def _fake_create_follow_up(self, **kwargs):
        return type("ChildJob", (), {"id": "child-hyp-1"})()

    recorded_events: list[dict] = []

    async def _fake_record_event(*args, **kwargs):
        recorded_events.append(kwargs)
        return type("Evt", (), {"id": "evt-1"})()

    monkeypatch.setattr(
        "app.services.autonomous_agent_executor.AutonomousAgentExecutor._create_domain_research_follow_up_job",
        _fake_create_follow_up,
    )
    monkeypatch.setattr(
        "app.services.research_opportunity_reprioritization_service.execute_agent_job_task.delay",
        lambda job_id, user_id: queued_jobs.append((job_id, user_id)),
    )
    monkeypatch.setattr(
        "app.services.research_opportunity_reprioritization_service.record_autonomy_decision_event",
        _fake_record_event,
    )

    async def _seed():
        db_session.add(note)
        await db_session.flush()
        db_session.add(parent_job)
        await db_session.flush()
        profile.latest_run_job_id = parent_job.id
        db_session.add(job)
        db_session.add(profile)
        db_session.add(notification)
        await db_session.commit()
        await db_session.refresh(note)
        await db_session.refresh(job)
        return note, job, profile

    note, job, profile = asyncio.get_event_loop().run_until_complete(_seed())

    response = client.post(
        f"/api/v1/synthesis/{job.id}/save-as-note",
        headers=auth_headers,
        json={"target_note_id": str(note.id)},
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["id"] == str(note.id)
    assert payload["structured_payload"]["artifact_type"] == "hypothesis_reevaluation"
    assert payload["structured_payload"]["reprioritization_summary"] == "Hypothesis one stays on top after positive run evidence."
    assert payload["structured_payload"]["hypotheses"][0]["experiment_evidence"][0]["run_id"] == "run-1"
    assert payload["structured_payload"]["previous_hypotheses"][0]["overall_score"] == 0.78
    assert payload["structured_payload"]["previous_summary"] == "Structured memo"
    assert payload["structured_payload"]["previous_artifact_type"] == "hypothesis_synthesis"
    assert len(payload["structured_payload"]["reevaluation_history"]) == 1
    assert payload["structured_payload"]["reevaluation_history"][0]["job_id"] == str(job.id)
    assert payload["structured_payload"]["reevaluation_history"][0]["source_run_ids"] == ["run-1"]
    assert payload["structured_payload"]["reevaluation_history"][0]["outcome_status"] == "applied_to_source_note"
    assert payload["structured_payload"]["reevaluation_history"][0]["origin_source_kind"] == "profile"
    assert payload["structured_payload"]["reevaluation_history"][0]["origin_source_id"] == str(profile.id)
    assert payload["structured_payload"]["reevaluation_history"][0]["origin_opportunity_id"] == "hyp-1"
    assert payload["structured_payload"].get("pending_reevaluation_job_id") is None

    async def _load_note():
        return await db_session.get(ResearchNote, note.id)

    updated_note = asyncio.get_event_loop().run_until_complete(_load_note())
    assert updated_note is not None
    assert updated_note.source_synthesis_job_id == job.id
    assert updated_note.structured_payload["hypotheses"][0]["overall_score"] == 0.84
    assert updated_note.structured_payload["previous_hypotheses"][0]["id"] == "hyp-1"
    assert updated_note.structured_payload["reevaluation_history"][0]["outcome_status"] == "applied_to_source_note"
    assert updated_note.structured_payload["reevaluation_history"][0]["source_run_ids"] == ["run-1"]
    assert updated_note.structured_payload["reevaluation_history"][0]["origin_source_kind"] == "profile"
    assert updated_note.structured_payload["reevaluation_history"][0]["origin_source_id"] == str(profile.id)
    assert updated_note.structured_payload["reevaluation_history"][0]["origin_opportunity_id"] == "hyp-1"
    async def _load_job():
        return await db_session.get(SynthesisJob, job.id)
    updated_job = asyncio.get_event_loop().run_until_complete(_load_job())
    assert updated_job is not None
    assert updated_job.result_metadata["review_outcome_status"] == "applied_to_source_note"
    assert updated_job.result_metadata["review_target_note_id"] == str(note.id)
    async def _load_notification():
        return await db_session.get(Notification, notification.id)
    updated_notification = asyncio.get_event_loop().run_until_complete(_load_notification())
    assert updated_notification is not None
    assert updated_notification.is_dismissed is True
    assert updated_notification.is_read is True
    assert updated_notification.data["review_outcome_status"] == "applied_to_source_note"
    assert updated_notification.data["resolved_by_review"] is True
    updated_opportunity = profile.latest_summary["opportunities"][0]
    assert updated_opportunity["confidence"] == 0.86
    assert updated_opportunity["readiness"] == 0.84
    assert updated_opportunity["prior_confidence"] == 0.6
    assert updated_opportunity["reprioritization_reason"] == "Positive evidence reinforced priority."
    assert updated_opportunity["last_reevaluation_review_outcome"] == "applied_to_source_note"
    assert updated_opportunity["last_reevaluation_review_job_id"] == str(job.id)
    assert updated_opportunity["last_reevaluation_review_source_note_id"] == str(note.id)
    assert updated_opportunity["last_reevaluation_review_target_note_id"] == str(note.id)
    assert updated_opportunity["follow_up_review_status"] == "approved_launch"
    assert updated_opportunity["child_job_ids"] == ["child-hyp-1"]
    assert queued_jobs == [("child-hyp-1", str(test_user.id))]
    reevaluation_events = [event for event in recorded_events if event.get("event_type") == "reevaluation_applied_to_source_note"]
    assert len(reevaluation_events) == 1
    assert reevaluation_events[0]["deep_link"]["params"]["profileId"] == str(profile.id)
    assert reevaluation_events[0]["deep_link"]["params"]["opportunityId"] == "hyp-1"
    assert reevaluation_events[0]["metadata"]["source_note_id"] == str(note.id)
    assert reevaluation_events[0]["metadata"]["target_note_id"] == str(note.id)
    assert reevaluation_events[0]["metadata"]["reevaluation_job_id"] == str(job.id)


def test_review_completed_hypothesis_reevaluation_marks_dismissed(client, db_session, test_user, auth_headers, monkeypatch):
    note = ResearchNote(
        user_id=test_user.id,
        title="Compiler hypotheses",
        content_markdown="## Ranked hypotheses",
        structured_payload={
            "artifact_type": "hypothesis_synthesis",
            "hypotheses": [
                {
                    "id": "hyp-1",
                    "rank": 1,
                    "title": "Layout-aware scheduling",
                    "claim": "Joint transforms may preserve locality under pressure.",
                    "rationale": "Combined evidence from prior papers.",
                    "novelty_score": 0.7,
                    "evidence_score": 0.6,
                    "testability_score": 0.9,
                    "overall_score": 0.78,
                    "autonomous_origin": {
                        "source_kind": "profile",
                        "source_id": "99999999-9999-9999-9999-999999999999",
                        "opportunity_id": "hyp-1",
                    },
                }
            ],
        },
    )
    job = SynthesisJob(
        user_id=test_user.id,
        job_type="hypothesis_reevaluation",
        title="Compiler hypothesis re-evaluation",
        document_ids=[],
        paper_ids=[],
        research_note_id=note.id,
        output_format="markdown",
        output_style="technical",
        status="completed",
        progress=100,
        result_content="# Hypothesis Re-evaluation\n\nUpdated rankings.",
        result_metadata={
            "summary": "Evidence-aware reprioritization.",
            "structured_hypotheses": [
                {
                    "id": "hyp-1",
                    "rank": 1,
                    "title": "Layout-aware scheduling",
                    "experiment_evidence": [
                        {
                            "run_id": "run-1",
                            "autonomous_origin": {
                                "source_kind": "profile",
                                "source_id": "99999999-9999-9999-9999-999999999999",
                                "opportunity_id": "hyp-1",
                            },
                        }
                    ],
                }
            ],
        },
    )
    profile = DomainResearchProfile(
        id=UUID("99999999-9999-9999-9999-999999999999"),
        user_id=test_user.id,
        title="Compiler Monitor",
        domain="Compiler",
        objective="Track compiler opportunities",
        status="active",
        latest_summary={
            "opportunities": [
                {
                    "opportunity_id": "hyp-1",
                    "title": "Layout-aware scheduling",
                    "hypothesis": "Joint transforms may preserve locality under pressure.",
                    "decision_state": "accepted",
                    "stage": "accepted",
                    "confidence": 0.6,
                    "readiness": 0.78,
                    "novelty": 0.7,
                    "autonomous_origin": {
                        "source_kind": "profile",
                        "source_id": "99999999-9999-9999-9999-999999999999",
                        "opportunity_id": "hyp-1",
                    },
                }
            ]
        },
    )
    notification = Notification(
        user_id=test_user.id,
        notification_type="hypothesis_reevaluation_update",
        title="Reevaluation ready: Compiler hypotheses",
        message="Draft ready to review.",
        priority="normal",
        related_entity_type="research_note",
        related_entity_id=note.id,
        data={
            "note_id": str(note.id),
            "reevaluation_job_id": str(job.id),
            "reevaluation_status": "completed",
        },
        action_url=f"/research-notes?note={note.id}",
    )

    recorded_events: list[dict] = []

    async def _fake_record_event(*args, **kwargs):
        recorded_events.append(kwargs)
        return type("Evt", (), {"id": "evt-2"})()

    monkeypatch.setattr(
        "app.services.research_opportunity_reprioritization_service.record_autonomy_decision_event",
        _fake_record_event,
    )

    async def _seed():
        db_session.add(note)
        await db_session.flush()
        db_session.add(job)
        db_session.add(profile)
        db_session.add(notification)
        await db_session.commit()
        await db_session.refresh(job)
        await db_session.refresh(note)
        return job, note, profile

    job, note, profile = asyncio.get_event_loop().run_until_complete(_seed())

    response = client.post(
        f"/api/v1/synthesis/{job.id}/review",
        headers=auth_headers,
        json={"outcome_status": "dismissed", "outcome_note": "Superseded by fresher evidence."},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["review_outcome_status"] == "dismissed"
    assert payload["review_note"] == "Superseded by fresher evidence."
    assert payload["can_apply"] is False
    assert payload["can_dismiss"] is False

    async def _load_job():
        return await db_session.get(SynthesisJob, job.id)

    updated_job = asyncio.get_event_loop().run_until_complete(_load_job())
    assert updated_job is not None
    assert updated_job.result_metadata["review_outcome_status"] == "dismissed"
    assert updated_job.result_metadata["review_note"] == "Superseded by fresher evidence."
    async def _load_notification():
        return await db_session.get(Notification, notification.id)
    updated_notification = asyncio.get_event_loop().run_until_complete(_load_notification())
    assert updated_notification is not None
    assert updated_notification.is_dismissed is True
    assert updated_notification.is_read is True
    assert updated_notification.data["review_outcome_status"] == "dismissed"
    assert updated_notification.data["resolved_by_review"] is True
    updated_opportunity = profile.latest_summary["opportunities"][0]
    assert updated_opportunity["last_reevaluation_review_outcome"] == "dismissed"
    assert updated_opportunity["last_reevaluation_review_job_id"] == str(job.id)
    assert updated_opportunity["last_reevaluation_review_source_note_id"] == str(note.id)
    assert updated_opportunity["last_reevaluation_review_target_note_id"] is None
    assert updated_opportunity.get("follow_up_review_status") is None
    reevaluation_events = [event for event in recorded_events if event.get("event_type") == "reevaluation_dismissed"]
    assert len(reevaluation_events) == 1
    assert reevaluation_events[0]["metadata"]["source_note_id"] == str(note.id)
    assert reevaluation_events[0]["metadata"]["target_note_id"] is None
    assert reevaluation_events[0]["metadata"]["reevaluation_job_id"] == str(job.id)

    async def _load_note():
        return await db_session.get(ResearchNote, note.id)

    unchanged_note = asyncio.get_event_loop().run_until_complete(_load_note())
    assert unchanged_note is not None
    assert unchanged_note.source_synthesis_job_id is None


def test_create_compiler_regression_explanation_job(client, db_session, test_user, auth_headers, monkeypatch):
    from app.tasks.synthesis_tasks import execute_synthesis_task

    queued: list[tuple[str, str]] = []

    def _fake_delay(job_id: str, user_id: str):
        queued.append((job_id, user_id))

    monkeypatch.setattr(execute_synthesis_task, "delay", _fake_delay)

    note = ResearchNote(
        user_id=test_user.id,
        title="Compiler note",
        content_markdown="## Hypothesis\nInvestigate benchmark regressions.",
    )
    plan = ExperimentPlan(
        user_id=test_user.id,
        research_note_id=note.id,
        title="Compiler run plan",
        hypothesis_text="Compare compiler runs",
        plan={"objective": "compare compiler runs"},
    )
    run_a = ExperimentRun(
        user_id=test_user.id,
        experiment_plan_id=plan.id,
        name="Run A",
        status="completed",
        config={"scientific_validation": {"benchmark_family": "compiler_regression", "benchmark_suite_id": "compiler-llvm-regression-core", "benchmark_case_ids": ["case-instcombine-sroa"]}},
    )
    run_b = ExperimentRun(
        user_id=test_user.id,
        experiment_plan_id=plan.id,
        name="Run B",
        status="completed",
        config={"scientific_validation": {"benchmark_family": "compiler_regression", "benchmark_suite_id": "compiler-llvm-regression-core", "benchmark_case_ids": ["case-instcombine-sroa"]}},
    )

    async def _seed():
        db_session.add(note)
        await db_session.flush()
        plan.research_note_id = note.id
        db_session.add(plan)
        await db_session.flush()
        run_a.experiment_plan_id = plan.id
        run_b.experiment_plan_id = plan.id
        db_session.add(run_a)
        db_session.add(run_b)
        await db_session.commit()
        await db_session.refresh(run_a)
        await db_session.refresh(run_b)
        return run_a, run_b

    run_a, run_b = asyncio.get_event_loop().run_until_complete(_seed())

    response = client.post(
        "/api/v1/synthesis",
        headers=auth_headers,
        json={
            "job_type": "compiler_regression_explanation",
            "title": "Compiler regression explanation",
            "document_ids": [],
            "research_note_id": str(note.id),
            "experiment_run_ids": [str(run_a.id), str(run_b.id)],
            "primary_run_id": str(run_a.id),
            "comparison_run_id": str(run_b.id),
            "output_format": "markdown",
            "output_style": "technical",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["job_type"] == "compiler_regression_explanation"
    assert payload["options"]["primary_run_id"] == str(run_a.id)
    assert payload["options"]["comparison_run_id"] == str(run_b.id)
    assert queued


def test_save_compiler_regression_explanation_as_note(client, db_session, test_user, auth_headers):
    note = ResearchNote(
        user_id=test_user.id,
        title="Compiler note",
        content_markdown="## Hypothesis\nInvestigate benchmark regressions.",
    )
    job = SynthesisJob(
        user_id=test_user.id,
        job_type="compiler_regression_explanation",
        title="Compiler regression explanation",
        document_ids=[],
        output_format="markdown",
        output_style="technical",
        status="completed",
        progress=100,
        result_content="# Compiler Regression Explanation\n\nSummary",
        result_metadata={
            "summary": "Compile time regressed because vectorization remarks disappeared.",
            "regression_type": "compile_time",
            "source_run_ids": ["run-a", "run-b"],
            "primary_run_id": "run-a",
            "comparison_run_id": "run-b",
            "source_document_ids": ["doc-1"],
            "metric_deltas": [{"metric": "compile_time_ms", "primary": 1400, "comparison": 1200, "delta": 200}],
            "artifact_deltas": [{"kind": "remarks", "summary": "loop-vectorize remarks missing"}],
            "likely_causes": [{"title": "Vectorizer not firing", "confidence": "medium", "reason": "remarks disappeared"}],
            "supporting_signals": ["loop-vectorize remarks missing"],
            "confounders": ["Single-machine sample"],
            "recommended_next_steps": ["Diff pass remarks across both builds"],
            "benchmark_family": "compiler_regression",
            "benchmark_suite_id": "compiler-llvm-regression-core",
            "benchmark_case_ids": ["case-instcombine-sroa"],
            "benchmark_baseline_id": "baseline-llvm-main",
        },
    )

    async def _seed():
        db_session.add(note)
        await db_session.flush()
        job.research_note_id = note.id
        db_session.add(job)
        await db_session.commit()
        await db_session.refresh(job)

    asyncio.get_event_loop().run_until_complete(_seed())

    response = client.post(
        f"/api/v1/synthesis/{job.id}/save-as-note",
        headers=auth_headers,
        json={},
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["tags"] == ["compiler-regression-explanation", "performance-analysis"]
    assert payload["structured_payload"]["artifact_type"] == "compiler_regression_explanation"
    assert payload["structured_payload"]["primary_run_id"] == "run-a"
    assert payload["structured_payload"]["metric_deltas"][0]["metric"] == "compile_time_ms"


def test_create_compiler_patch_proposal_job_from_explanation_note(client, db_session, test_user, auth_headers, monkeypatch):
    from app.tasks.synthesis_tasks import execute_synthesis_task

    queued: list[tuple[str, str]] = []

    def _fake_delay(job_id: str, user_id: str):
        queued.append((job_id, user_id))

    monkeypatch.setattr(execute_synthesis_task, "delay", _fake_delay)

    note = ResearchNote(
        user_id=test_user.id,
        title="Compiler regression explanation",
        content_markdown="## Explanation",
        structured_payload={
            "artifact_type": "compiler_regression_explanation",
            "summary": "Compile time regressed because vectorization remarks disappeared.",
            "source_run_ids": ["run-a", "run-b"],
        },
    )

    async def _seed():
        db_session.add(note)
        await db_session.commit()
        await db_session.refresh(note)
        return note

    note = asyncio.get_event_loop().run_until_complete(_seed())

    response = client.post(
        "/api/v1/synthesis",
        headers=auth_headers,
        json={
            "job_type": "compiler_patch_proposal",
            "title": "Compiler patch proposal",
            "document_ids": [],
            "research_note_id": str(note.id),
            "output_format": "markdown",
            "output_style": "technical",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["job_type"] == "compiler_patch_proposal"
    assert payload["research_note_id"] == str(note.id)
    assert queued


def test_save_compiler_patch_proposal_as_note(client, db_session, test_user, auth_headers):
    note = ResearchNote(
        user_id=test_user.id,
        title="Compiler regression explanation",
        content_markdown="## Explanation",
        structured_payload={"artifact_type": "compiler_regression_explanation"},
    )
    job = SynthesisJob(
        user_id=test_user.id,
        job_type="compiler_patch_proposal",
        title="Compiler patch proposal",
        document_ids=[],
        output_format="markdown",
        output_style="technical",
        status="completed",
        progress=100,
        result_content="# Compiler Patch Proposal\n\nSummary",
        result_metadata={
            "proposal_summary": "Gate the vectorization heuristic with a narrow profitability check.",
            "target_area": "vectorization heuristic",
            "candidate_change": "Add a guard before enabling the transform when remarks indicate profitability is marginal.",
            "expected_effect": "Reduce compile-time regressions while preserving vectorization wins.",
            "mechanism": "Avoid costly transforms on marginal loops.",
            "supporting_evidence": ["loop-vectorize remarks disappeared", "compile_time_ms regressed"],
            "validation_plan": "Run the compiler regression suite and compare remarks plus compile time.",
            "risk_assessment": "May suppress beneficial vectorization in edge cases.",
            "rollback_or_guardrail": "Feature-flag the heuristic and keep a fast rollback path.",
            "source_run_ids": ["run-a", "run-b"],
            "source_explanation_note_id": str(note.id),
            "source_document_ids": ["doc-1"],
            "source_paper_ids": ["paper-1"],
            "benchmark_family": "compiler_regression",
            "benchmark_suite_id": "compiler-llvm-regression-core",
            "benchmark_case_ids": ["case-instcombine-sroa"],
            "benchmark_baseline_id": "baseline-llvm-main",
        },
    )

    async def _seed():
        db_session.add(note)
        await db_session.flush()
        job.research_note_id = note.id
        db_session.add(job)
        await db_session.commit()
        await db_session.refresh(job)

    asyncio.get_event_loop().run_until_complete(_seed())

    response = client.post(
        f"/api/v1/synthesis/{job.id}/save-as-note",
        headers=auth_headers,
        json={},
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["tags"] == ["compiler-patch-proposal", "compiler-proposal"]
    assert payload["structured_payload"]["artifact_type"] == "compiler_patch_proposal"
    assert payload["structured_payload"]["target_area"] == "vectorization heuristic"
    assert payload["structured_payload"]["candidate_change"].startswith("Add a guard")
    assert payload["structured_payload"]["source_explanation_note_id"] == str(note.id)


def test_create_compiler_patch_draft_job_from_proposal_note(client, db_session, test_user, auth_headers, monkeypatch):
    from app.tasks.synthesis_tasks import execute_synthesis_task

    queued: list[tuple[str, str]] = []

    def _fake_delay(job_id: str, user_id: str):
        queued.append((job_id, user_id))

    monkeypatch.setattr(execute_synthesis_task, "delay", _fake_delay)

    source = DocumentSource(
        name="Compiler Repo Source",
        source_type="github",
        config={"repos": ["llvm/llvm-project"]},
    )
    note = ResearchNote(
        user_id=test_user.id,
        title="Compiler patch proposal",
        content_markdown="## Proposal",
        structured_payload={
            "artifact_type": "compiler_patch_proposal",
            "proposal_summary": "Gate the vectorization heuristic with a narrow profitability check.",
        },
    )

    async def _seed():
        db_session.add(source)
        await db_session.flush()
        db_session.add(note)
        await db_session.commit()
        await db_session.refresh(source)
        await db_session.refresh(note)
        return source, note

    source, note = asyncio.get_event_loop().run_until_complete(_seed())

    response = client.post(
        "/api/v1/synthesis",
        headers=auth_headers,
        json={
            "job_type": "compiler_patch_draft",
            "title": "Compiler patch draft",
            "document_ids": [],
            "research_note_id": str(note.id),
            "source_id": str(source.id),
            "output_format": "markdown",
            "output_style": "technical",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["job_type"] == "compiler_patch_draft"
    assert payload["research_note_id"] == str(note.id)
    assert payload["options"]["source_id"] == str(source.id)
    assert queued


def test_save_compiler_patch_draft_as_note(client, db_session, test_user, auth_headers):
    note = ResearchNote(
        user_id=test_user.id,
        title="Compiler patch proposal",
        content_markdown="## Proposal",
        structured_payload={"artifact_type": "compiler_patch_proposal"},
    )
    job = SynthesisJob(
        user_id=test_user.id,
        job_type="compiler_patch_draft",
        title="Compiler patch draft",
        document_ids=[],
        output_format="markdown",
        output_style="technical",
        status="completed",
        progress=100,
        result_content="# Compiler Patch Draft\n\nSummary",
        result_metadata={
            "draft_summary": "Target vectorization profitability hooks in a narrow backend file set.",
            "source_proposal_note_id": str(note.id),
            "source_explanation_note_id": "note-exp-1",
            "source_id": "source-1",
            "source_name": "Compiler Repo Source",
            "target_files": ["llvm/lib/Transforms/Vectorize/LoopVectorize.cpp"],
            "target_symbols": ["LoopVectorizationPlanner"],
            "change_plan": ["Add a profitability guard before enabling the transform."],
            "proposed_code_regions": [{"file": "llvm/lib/Transforms/Vectorize/LoopVectorize.cpp", "symbol": "LoopVectorizationPlanner", "intent": "Gate marginal transforms"}],
            "validation_commands": ["ninja check-llvm", "llvm-lit test/Transforms/LoopVectorize"],
            "benchmark_validation_scope": ["compiler-llvm-regression-core"],
            "risk_checks": ["Verify no beneficial vectorization is lost on baseline kernels."],
            "rollback_steps": ["Revert the guard or disable it behind a flag."],
        },
    )

    async def _seed():
        db_session.add(note)
        await db_session.flush()
        job.research_note_id = note.id
        db_session.add(job)
        await db_session.commit()
        await db_session.refresh(job)

    asyncio.get_event_loop().run_until_complete(_seed())

    response = client.post(
        f"/api/v1/synthesis/{job.id}/save-as-note",
        headers=auth_headers,
        json={},
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["tags"] == ["compiler-patch-draft", "compiler-change-plan"]
    assert payload["structured_payload"]["artifact_type"] == "compiler_patch_draft"
    assert payload["structured_payload"]["source_proposal_note_id"] == str(note.id)
    assert payload["structured_payload"]["target_files"] == ["llvm/lib/Transforms/Vectorize/LoopVectorize.cpp"]
