"""Related documents and experiment outcomes, with rows present.

Two services on the corpus -> run -> review path that no test named.

Related documents (`paper_recommendation_service`):
* the query is built from the document itself, so its own chunks were the
  best matches and were then discarded: a document with more chunks than the
  window had no neighbours at all;
* a candidate with no document row was recommended with a null title --
  vectors outlive documents, and a chunk id was accepted as a document id.

Experiment outcomes (`experiment_outcome_service`):
* the profile or portfolio to update is named in the run's own config, and
  was updated whoever owned it.
"""

import hashlib
import inspect
from datetime import datetime
from uuid import uuid4

import pytest

from app.models.document import Document, DocumentChunk, DocumentSource
from app.models.experiment import ExperimentPlan, ExperimentRun
from app.models.research_portfolio import ResearchPortfolio
from app.services import paper_recommendation_service as recommendations
from app.services.experiment_outcome_service import (
    reconcile_experiment_run_outcome_to_originating_opportunity as reconcile,
)
from app.services.paper_recommendation_service import PaperRecommendationService
from app.services.vector_store import VectorStoreService

pytestmark = pytest.mark.unit

_SEARCH = inspect.signature(VectorStoreService.search)


# --------------------------------------------------------------------------
# related documents
# --------------------------------------------------------------------------


async def _documents(db, titles, chunks_of_first=0):
    source = DocumentSource(name=f"s-{uuid4().hex[:6]}", source_type="file", config={})
    db.add(source)
    await db.commit()
    docs = []
    for title in titles:
        doc = Document(
            title=title,
            content=f"{title} body",
            content_hash=hashlib.sha256(title.encode()).hexdigest(),
            source_id=source.id,
            source_identifier=f"test:{uuid4().hex}",
        )
        db.add(doc)
        docs.append(doc)
    await db.commit()
    for index in range(chunks_of_first):
        db.add(
            DocumentChunk(
                document_id=docs[0].id,
                content=f"chunk {index}",
                content_hash=hashlib.sha256(f"c{index}".encode()).hexdigest(),
                chunk_index=index,
            )
        )
    await db.commit()
    return docs


class _Store:
    """Ranked chunks; returns as many as it is asked for."""

    def __init__(self, ranked):
        self.ranked = ranked
        self.limits = []

    async def initialize(self, embedding_model=None, background=False):
        return None

    async def search(self, *args, **kwargs):
        bound = _SEARCH.bind(self, *args, **kwargs)
        bound.apply_defaults()
        self.limits.append(bound.arguments["limit"])
        return self.ranked[: bound.arguments["limit"]]


def _hit(document_id, score, chunk_id=None):
    return {
        "id": str(chunk_id or uuid4()),
        "score": score,
        "metadata": {"document_id": str(document_id), "chunk_id": str(uuid4())},
    }


def _service(store):
    service = PaperRecommendationService()
    service.vector_store = store
    return service


async def test_a_long_document_still_has_neighbours(db_session, monkeypatch):
    """Its own chunks outrank everything and used to fill the whole window."""
    monkeypatch.setattr(recommendations, "VECTOR_WINDOW", 3)
    long_doc, neighbour = await _documents(
        db_session, ["Long survey", "Neighbour"], chunks_of_first=10
    )
    store = _Store(
        [_hit(long_doc.id, 0.99) for _ in range(10)] + [_hit(neighbour.id, 0.5)]
    )

    related = await _service(store).related_documents(db_session, long_doc.id)

    assert store.limits == [13]
    assert [r["title"] for r in related] == ["Neighbour"]


async def test_a_document_that_no_longer_exists_is_not_recommended(db_session):
    doc, real = await _documents(db_session, ["Paper", "Still here"])
    store = _Store([_hit(uuid4(), 0.9), _hit(real.id, 0.4)])

    related = await _service(store).related_documents(db_session, doc.id)

    assert [r["title"] for r in related] == ["Still here"]
    assert all(r["title"] for r in related)


async def test_a_chunk_id_is_not_taken_for_a_document(db_session):
    doc, _other = await _documents(db_session, ["Paper", "Other"])
    # No document_id recorded on the chunk: nothing says whose it is.
    store = _Store([{"id": str(uuid4()), "score": 0.9, "metadata": {}}])

    assert await _service(store).related_documents(db_session, doc.id) == []


async def test_nothing_related_is_an_empty_list(db_session):
    (doc,) = await _documents(db_session, ["Alone"])

    assert await _service(_Store([])).related_documents(db_session, doc.id) == []


# --------------------------------------------------------------------------
# experiment outcomes
# --------------------------------------------------------------------------

OPPORTUNITY = "opp-1"


async def _portfolio(db, user):
    portfolio = ResearchPortfolio(
        user_id=user.id,
        title="Prefetching",
        objective="Find headroom",
        opportunities=[{"opportunity_id": OPPORTUNITY, "title": "Stride on L2"}],
    )
    db.add(portfolio)
    await db.commit()
    await db.refresh(portfolio)
    return portfolio


def _run_and_plan(user, portfolio, status="completed"):
    plan = ExperimentPlan(id=uuid4(), user_id=user.id, title="Plan", plan={})
    run = ExperimentRun(
        id=uuid4(),
        user_id=user.id,
        name="Run",
        status=status,
        summary="Stride gave 2.1x on the kernel.",
        completed_at=datetime(2026, 10, 12, 9, 0, 0),
        config={
            "execution_handoff": {
                "autonomous_origin": {
                    "source_kind": "portfolio",
                    "source_id": str(portfolio.id),
                    "opportunity_id": OPPORTUNITY,
                }
            }
        },
    )
    return run, plan


def _opportunity(portfolio):
    return next(
        row for row in portfolio.opportunities if row["opportunity_id"] == OPPORTUNITY
    )


async def test_a_finished_run_is_recorded_on_the_opportunity_it_came_from(
    db_session, test_user
):
    portfolio = await _portfolio(db_session, test_user)
    run, plan = _run_and_plan(test_user, portfolio)

    assert await reconcile(db_session, run=run, plan=plan) is True
    await db_session.commit()

    # Read back from the database: the list is a plain JSON column.
    portfolio_id = portfolio.id
    db_session.expire_all()
    stored = await db_session.get(ResearchPortfolio, portfolio_id)
    row = _opportunity(stored)
    assert row["latest_validation_status"] == "completed"
    assert row["follow_up_outcome_summary"] == "Stride gave 2.1x on the kernel."
    assert row["latest_validation_run_id"] == str(run.id)
    assert stored.latest_validation_run_ids == [str(run.id)]


async def test_a_run_cannot_record_on_someone_elses_opportunity(
    db_session, test_user, admin_user
):
    theirs = await _portfolio(db_session, admin_user)
    run, plan = _run_and_plan(test_user, theirs)

    assert await reconcile(db_session, run=run, plan=plan) is False
    await db_session.commit()

    portfolio_id = theirs.id
    db_session.expire_all()
    stored = await db_session.get(ResearchPortfolio, portfolio_id)
    assert "latest_validation_status" not in _opportunity(stored) or (
        _opportunity(stored)["latest_validation_status"] is None
    )
    assert not stored.latest_validation_run_ids


async def test_a_run_naming_no_origin_changes_nothing(db_session, test_user):
    portfolio = await _portfolio(db_session, test_user)
    run, plan = _run_and_plan(test_user, portfolio)
    run.config = {}

    assert await reconcile(db_session, run=run, plan=plan) is False
