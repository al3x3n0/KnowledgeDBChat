"""What a finished run leaves behind in the Library.

A run establishes findings and records methods, and until they are searchable
they exist only for whoever opens that run: invisible to a chat answer, to
search, and to the next run's recall. This is the record it leaves — and the
rules about what does not deserve one.
"""

import uuid

import pytest

from app.models.agent_job import AgentJob
from app.services.agent_runtime_finalizer import (
    _established_findings,
    _record_what_the_run_established,
)

pytestmark = pytest.mark.unit


def _job(results):
    return AgentJob(
        id=uuid.uuid4(),
        name="INT8 attention throughput",
        goal="Find what caps INT8 attention throughput",
        job_type="experiment",
        user_id=uuid.uuid4(),
        status="completed",
        results=results,
    )


class TestWhatCountsAsEstablished:
    def test_retrieval_hits_are_not_the_runs_own(self):
        job = _job(
            {
                "findings": [
                    {"type": "document", "title": "Something it read"},
                    {"type": "paper", "title": "A paper it read"},
                    {"type": "benchmark_measurement", "title": "It measured this"},
                ]
            }
        )
        established = _established_findings(job)
        assert [f["title"] for f in established] == ["It measured this"]

    def test_a_missing_or_malformed_findings_list_is_empty(self):
        assert _established_findings(_job(None)) == []
        assert _established_findings(_job({})) == []
        assert _established_findings(_job({"findings": "not a list"})) == []
        assert _established_findings(_job({"findings": [None, "text", 3]})) == []


class _FakeDocumentService:
    def __init__(self):
        self.reprocessed = []

    async def _get_or_create_agent_notes_source(self, db):
        class _Source:
            id = uuid.uuid4()

        return _Source()

    async def reprocess_document(self, doc_id, db, user_id=None):
        self.reprocessed.append(doc_id)


class _FakeExecutor:
    def __init__(self):
        self.document_service = _FakeDocumentService()


class _FakeDb:
    def __init__(self):
        self.added = []

    def add(self, obj):
        self.added.append(obj)

    async def commit(self):
        pass

    async def refresh(self, obj):
        if getattr(obj, "id", None) is None:
            obj.id = uuid.uuid4()


class TestTheRecordItLeaves:
    @pytest.mark.asyncio
    async def test_a_run_with_nothing_of_its_own_leaves_no_record(self):
        # A document saying only that a run happened is noise in search.
        job = _job({"findings": [{"type": "document", "title": "Only read this"}]})
        db, artifacts = _FakeDb(), []

        await _record_what_the_run_established(_FakeExecutor(), job, db, artifacts)

        assert db.added == []
        assert artifacts == []

    @pytest.mark.asyncio
    async def test_the_record_carries_the_findings_with_their_numbers(self):
        job = _job(
            {
                "findings": [
                    {
                        "type": "benchmark_measurement",
                        "title": "dotprod @ -O2",
                        "fastest_ms": 15,
                        "all_ms": [54, 18, 16, 15, 30],
                    }
                ],
                "conclusion": {"answer": "The arithmetic is not the ceiling."},
            }
        )
        db, artifacts = _FakeDb(), []

        await _record_what_the_run_established(_FakeExecutor(), job, db, artifacts)

        doc = db.added[0]
        assert "INT8 attention throughput" in doc.title
        assert "The arithmetic is not the ceiling." in doc.content
        # The numbers, not just the title — the same failure the run-conclusion
        # formatter exists to prevent.
        assert "fastest_ms=15" in doc.content
        assert "all_ms=15..54 (n=5)" in doc.content
        assert doc.extra_metadata["agent_job_id"] == str(job.id)
        assert doc.extra_metadata["finding_count"] == 1

    @pytest.mark.asyncio
    async def test_recorded_methods_are_written_out_in_full(self):
        job = _job(
            {
                "findings": [],
                "methods": [
                    {
                        "name": "control-through-same-tool",
                        "procedure": "Run a trivial control through the same tool.",
                        "prevents": "Blaming the input for a broken tool.",
                    }
                ],
            }
        )
        db, artifacts = _FakeDb(), []

        await _record_what_the_run_established(_FakeExecutor(), job, db, artifacts)

        content = db.added[0].content
        assert "control-through-same-tool" in content
        assert "Run a trivial control through the same tool." in content
        assert "Prevents: Blaming the input for a broken tool." in content

    @pytest.mark.asyncio
    async def test_the_record_is_indexed_and_announced_as_an_artifact(self):
        job = _job({"findings": [{"type": "insight", "title": "Upstream of L2"}]})
        executor, db, artifacts = _FakeExecutor(), _FakeDb(), []

        await _record_what_the_run_established(executor, job, db, artifacts)

        doc = db.added[0]
        # Unindexed it is invisible to search, which is the whole point.
        assert executor.document_service.reprocessed == [doc.id]
        assert artifacts == [
            {"type": "document", "id": str(doc.id), "title": doc.title}
        ]
        assert job.results["library_record"]["document_id"] == str(doc.id)

    @pytest.mark.asyncio
    async def test_finalizing_twice_does_not_leave_two_records(self):
        job = _job({"findings": [{"type": "insight", "title": "Upstream of L2"}]})
        executor, db, artifacts = _FakeExecutor(), _FakeDb(), []

        await _record_what_the_run_established(executor, job, db, artifacts)
        await _record_what_the_run_established(executor, job, db, artifacts)

        assert len(db.added) == 1
        assert len(artifacts) == 1

    @pytest.mark.asyncio
    async def test_a_failure_to_index_leaves_the_document_standing(self):
        # Unindexed it is still readable in the Library; losing it entirely
        # because the vector store was down would be the worse outcome.
        job = _job({"findings": [{"type": "insight", "title": "Upstream of L2"}]})
        executor, db, artifacts = _FakeExecutor(), _FakeDb(), []

        async def _boom(doc_id, db, user_id=None):
            raise RuntimeError("qdrant unreachable")

        executor.document_service.reprocess_document = _boom

        await _record_what_the_run_established(executor, job, db, artifacts)

        assert len(db.added) == 1
        assert artifacts[0]["id"] == str(db.added[0].id)
        assert job.results["library_record"]["document_id"] == str(db.added[0].id)
