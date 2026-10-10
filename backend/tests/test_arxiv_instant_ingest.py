"""Putting an arXiv paper into the knowledge base inside one request.

Two routes did this, each with its own copy of the code: `POST
/documents/ingest-arxiv-instant` and `POST /presentations/from-research`.
Neither copy could store a new paper. Each built a document without the
source it must belong to, called the async text splitter without awaiting it,
and built chunks without their content hash -- three separate failures, the
first of which the database refuses. One route answered 500; the other caught
everything and made the presentation with no sources and no word about it.

These run both routes with arXiv faked and everything else real: the rows,
the chunking, the document pipeline.
"""

from __future__ import annotations

import pytest
from sqlalchemy import func, select

from app.api.endpoints import documents as documents_endpoint
from app.api.endpoints import presentations as presentations_endpoint
from app.core.config import settings
from app.models.document import Document, DocumentChunk, DocumentSource
from app.models.presentation import PresentationJob
from app.schemas.document import InstantArxivIngestRequest
from app.services.vector_store import VectorStoreService

PAPER_TEXT = (
    "We study prefetching for irregular memory access patterns. "
    "Our stride-free predictor learns pointer chains from history. "
) * 40


class _Arxiv:
    """A connector that knows two papers and how often it was asked."""

    papers = {
        "2401.00001": "Irregular Prefetching",
        "2401.00002": "Pointer Chain Prediction",
    }
    fetched: list = []
    cleaned_up = 0
    fail_on: set = set()

    async def initialize(self, config):
        ids = config.get("paper_ids")
        self.wanted = list(ids) if ids else list(self.papers)
        return True

    async def list_documents(self):
        return [
            {
                "identifier": paper_id,
                "title": self.papers[paper_id],
                "url": f"https://arxiv.org/abs/{paper_id}",
                "author": "A. Author",
            }
            for paper_id in self.wanted
            if paper_id in self.papers
        ]

    async def get_document_content(self, identifier):
        if identifier in type(self).fail_on:
            raise RuntimeError("arXiv answered 503")
        type(self).fetched.append(identifier)
        return f"{self.papers[identifier]}\n\n{PAPER_TEXT}"

    async def get_document_metadata(self, identifier):
        return {"authors": ["A. Author", "B. Author"], "categories": ["cs.AR"]}

    async def cleanup(self):
        type(self).cleaned_up += 1


@pytest.fixture
def world(monkeypatch):
    """arXiv faked, the vector store recorded, the task queue recorded."""
    import app.services.connectors.arxiv_connector as arxiv_module
    from app.tasks.presentation_tasks import generate_presentation_task

    _Arxiv.fetched, _Arxiv.cleaned_up, _Arxiv.fail_on = [], 0, set()
    monkeypatch.setattr(arxiv_module, "ArxivConnector", _Arxiv)

    indexed, queued = [], []

    async def initialize(self, *args, **kwargs):
        return None

    async def add_document_chunks(self, document, chunks):
        indexed.append((document.id, len(chunks)))
        return [str(chunk.id) for chunk in chunks]

    monkeypatch.setattr(VectorStoreService, "initialize", initialize)
    monkeypatch.setattr(VectorStoreService, "add_document_chunks", add_document_chunks)
    monkeypatch.setattr(
        generate_presentation_task,
        "delay",
        lambda job_id, user_id: queued.append(job_id),
    )
    monkeypatch.setattr(settings, "KNOWLEDGE_GRAPH_ENABLED", False, raising=False)
    monkeypatch.setattr(settings, "AUTO_SUMMARIZE_ON_PROCESS", False, raising=False)
    monkeypatch.setattr(settings, "RAG_CHUNKING_STRATEGY", "fixed", raising=False)

    class World:
        pass

    world = World()
    world.indexed, world.queued = indexed, queued
    return world


async def _ingest(db, user, paper_id="2401.00001"):
    return await documents_endpoint.ingest_arxiv_instant(
        request=InstantArxivIngestRequest(
            arxiv_input=paper_id,
            auto_summarize=False,
            auto_enrich=False,
            auto_extract=False,
        ),
        current_user=user,
        db=db,
    )


async def _present(db, user, **overrides):
    params = dict(
        topic="irregular prefetching",
        slide_count=10,
        include_arxiv=True,
        arxiv_max_papers=5,
        style="technical",
        include_diagrams=True,
        current_user=user,
        db=db,
    )
    params.update(overrides)
    return await presentations_endpoint.create_research_presentation(**params)


async def _chunks_of(db, document_id):
    return (
        (
            await db.execute(
                select(DocumentChunk)
                .where(DocumentChunk.document_id == document_id)
                .order_by(DocumentChunk.chunk_index)
            )
        )
        .scalars()
        .all()
    )


class TestTheInstantIngestRoute:
    @pytest.mark.asyncio
    async def test_a_new_paper_becomes_a_searchable_document(
        self, db_session, test_user, world
    ):
        answer = await _ingest(db_session, test_user)

        document = await db_session.get(Document, answer.document_id)
        chunks = await _chunks_of(db_session, document.id)
        assert document.title == "Irregular Prefetching"
        assert document.source_identifier == "2401.00001"
        assert document.is_processed is True
        assert document.extra_metadata["authors"] == ["A. Author", "B. Author"]
        # It belongs to a source, as every document must.
        source = await db_session.get(DocumentSource, document.source_id)
        assert source is not None
        # Real chunks, each with the hash its column requires.
        assert len(chunks) > 1
        assert all(len(chunk.content_hash) == 64 for chunk in chunks)
        assert answer.chunks_created == len(chunks)
        assert answer.ready_for_chat is True
        assert world.indexed == [(document.id, len(chunks))]

    @pytest.mark.asyncio
    async def test_the_source_it_files_under_is_never_crawled(
        self, db_session, test_user, world
    ):
        answer = await _ingest(db_session, test_user)

        document = await db_session.get(Document, answer.document_id)
        source = await db_session.get(DocumentSource, document.source_id)
        # It has no queries to run; a sync sweep over it would be an error.
        assert source.is_active is False

    @pytest.mark.asyncio
    async def test_a_paper_already_indexed_is_not_fetched_again(
        self, db_session, test_user, world
    ):
        first = await _ingest(db_session, test_user)

        again = await _ingest(db_session, test_user)

        assert again.document_id == first.document_id
        assert again.background_tasks == ["already_indexed"]
        assert _Arxiv.fetched == ["2401.00001"]
        total = (await db_session.execute(select(func.count(Document.id)))).scalar()
        assert total == 1

    @pytest.mark.asyncio
    async def test_a_paper_arxiv_does_not_have(self, db_session, test_user, world):
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as refused:
            await _ingest(db_session, test_user, "2401.99999")

        assert refused.value.status_code == 404
        assert _Arxiv.cleaned_up == 1

    @pytest.mark.asyncio
    async def test_a_paper_that_could_not_be_indexed_is_not_called_ready(
        self, db_session, test_user, world, monkeypatch
    ):
        from fastapi import HTTPException

        async def broken(self, document, chunks):
            raise RuntimeError("vector store unreachable")

        monkeypatch.setattr(VectorStoreService, "add_document_chunks", broken)

        with pytest.raises(HTTPException) as refused:
            await _ingest(db_session, test_user)

        assert refused.value.status_code >= 500


class TestAPresentationFromResearch:
    @pytest.mark.asyncio
    async def test_the_papers_it_finds_become_its_sources(
        self, db_session, test_user, world
    ):
        answer = await _present(db_session, test_user)

        documents = (await db_session.execute(select(Document))).scalars().all()
        assert sorted(doc.source_identifier for doc in documents) == [
            "2401.00001",
            "2401.00002",
        ]
        assert all(doc.is_processed for doc in documents)
        assert sorted(answer.source_document_ids) == sorted(
            str(doc.id) for doc in documents
        )
        job = await db_session.get(PresentationJob, answer.id)
        assert sorted(job.source_document_ids) == sorted(answer.source_document_ids)
        assert world.queued == [str(job.id)]
        assert _Arxiv.cleaned_up == 1

    @pytest.mark.asyncio
    async def test_one_paper_failing_does_not_lose_the_others(
        self, db_session, test_user, world
    ):
        _Arxiv.fail_on = {"2401.00001"}

        answer = await _present(db_session, test_user)

        documents = (await db_session.execute(select(Document))).scalars().all()
        assert [doc.source_identifier for doc in documents] == ["2401.00002"]
        assert answer.source_document_ids == [str(documents[0].id)]
        assert _Arxiv.cleaned_up == 1

    @pytest.mark.asyncio
    async def test_a_paper_already_in_the_knowledge_base_is_reused(
        self, db_session, test_user, world
    ):
        first = await _ingest(db_session, test_user, "2401.00001")
        _Arxiv.fetched.clear()

        answer = await _present(db_session, test_user)

        assert str(first.document_id) in answer.source_document_ids
        assert _Arxiv.fetched == ["2401.00002"]

    @pytest.mark.asyncio
    async def test_without_arxiv_it_asks_nothing_of_it(
        self, db_session, test_user, world
    ):
        answer = await _present(db_session, test_user, include_arxiv=False)

        assert answer.source_document_ids == []
        assert _Arxiv.fetched == [] and _Arxiv.cleaned_up == 0
        assert world.queued == [str(answer.id)]
