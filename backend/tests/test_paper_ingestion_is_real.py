"""`papers_ingested` has to mean the paper is in the corpus.

`ingest_paper_by_id` ran an arXiv search, returned the metadata, stored
nothing, and emitted a `papers_ingested` finding anyway. Measured live: a
reproduction pipeline's first stage completed at 100% with its contract
satisfied while `research_papers` held zero rows and no document existed. The
next stage then searched the corpus for the paper, surfaced a DIFFERENT paper
by the same author left over from earlier work, and was one call from
specifying the wrong algorithm -- which every stage after it would have
implemented, measured and scored, each reporting success.
"""

import pytest

from app.agent_core.tool_specs.research import SPECS

pytestmark = pytest.mark.unit


def _spec(name):
    for spec in SPECS:
        if spec.name == name:
            return spec
    raise AssertionError(f"no such tool spec: {name}")


class TestTheToolPromisesOnlyWhatItDoes:
    def test_it_no_longer_offers_a_parameter_it_ignores(self):
        """`add_to_reading_list` was accepted and never read -- and being a
        STRING named like a boolean, the live run passed `true` and lost an
        iteration to the type error."""
        properties = _spec("ingest_paper_by_id").parameters["properties"]

        assert "add_to_reading_list" not in properties
        assert "arxiv_id" in properties

    def test_it_tells_the_model_to_read_by_id_not_by_search(self):
        """The substitution was silent because the finding named no document,
        so the only way downstream to find the paper was to search for it."""
        description = _spec("ingest_paper_by_id").description

        assert "document_ids" in description

    def test_both_producers_of_papers_ingested_claim_the_same_thing(self):
        """Two tools declare this evidence type. If one of them means
        something weaker, a contract requiring it is satisfied by whichever
        the model happens to pick."""
        producers = [s for s in SPECS if "papers_ingested" in (s.produces or ())]

        assert {s.name for s in producers} == {
            "ingest_arxiv_papers",
            "ingest_paper_by_id",
        }
        for spec in producers:
            assert "corpus" in (spec.consumes or ""), spec.name


@pytest.mark.asyncio
class TestItWaitsForTheDocumentToLand:
    async def test_no_document_after_the_wait_is_a_failure_not_a_success(
        self, monkeypatch
    ):
        """Ingestion runs in a Celery worker, so "queued" and "readable" are
        different facts. Only the second lets the next stage do its job."""
        from app.services import agent_tool_dispatch as dispatch

        async def _nothing_lands(source_id):
            return []

        monkeypatch.setattr(dispatch, "_wait_for_ingested_documents", _nothing_lands)

        result = await _call_ingest(monkeypatch, dispatch)

        assert result["success"] is False
        assert "not readable" in result["error"] or "no document" in result["error"]
        assert not result.get("findings"), "a failed ingestion claims no evidence"

    async def test_a_landed_document_is_named_in_the_finding(self, monkeypatch):
        from app.services import agent_tool_dispatch as dispatch

        async def _one_lands(source_id):
            return ["doc-1"]

        monkeypatch.setattr(dispatch, "_wait_for_ingested_documents", _one_lands)

        result = await _call_ingest(monkeypatch, dispatch)

        assert result["success"] is True
        finding = result["findings"][0]
        assert finding["type"] == "papers_ingested"
        assert finding["document_ids"] == ["doc-1"]
        assert finding["arxiv_id"] == "1805.10941"


async def _call_ingest(monkeypatch, dispatch):
    """Drive the real handler with only arXiv and the ingestion start stubbed."""

    class _Arxiv:
        async def search(self, **kwargs):
            return [{"title": "Fast Random Integer Generation in an Interval"}]

    class _Executor:
        arxiv_service = _Arxiv()

    class _AgentService:
        async def _tool_ingest_arxiv_papers(self, params, user_id, db):
            assert params["paper_ids"] == ["1805.10941"], "must ingest what was asked"
            return {"source_id": "src-1"}

    monkeypatch.setattr(
        "app.services.agent_service.AgentService", _AgentService, raising=False
    )

    provider = dispatch.build_autonomous_research_provider(_Executor())
    handler = provider._handlers["ingest_paper_by_id"]

    class _Ctx:
        user_id = "u-1"
        db = None
        state = {}

    return await handler({"arxiv_id": "1805.10941"}, _Ctx())


@pytest.mark.asyncio
class TestTheWaitDoesNotTouchTheCallersSession:
    """Two traps this project has already paid for, both reachable from a
    poll loop that borrows the caller's AsyncSession.

    A `rollback()` on a borrowed session expires every ORM object in it --
    `expire_on_commit=False` does not cover rollbacks -- so the executor's next
    attribute read becomes IO and raises MissingGreenlet somewhere unrelated.
    And holding that session in an open transaction for the length of the wait
    is what left the executor idle-in-transaction on the `agent_jobs` row while
    the lease heartbeat's UPDATE blocked behind it.
    """

    def test_the_waiter_takes_no_session_from_its_caller(self):
        import inspect

        from app.services.agent_tool_dispatch import _wait_for_ingested_documents

        parameters = inspect.signature(_wait_for_ingested_documents).parameters
        assert list(parameters) == ["source_id"], (
            "the waiter must open its own session; taking the caller's is how "
            "both traps become reachable"
        )

    def test_it_never_rolls_back_anything(self):
        """Checked on the body, not the whole function: the docstring names
        the trap on purpose, and a check that forbade the word would forbid
        explaining it."""
        import ast
        import inspect
        import textwrap

        from app.services.agent_tool_dispatch import _wait_for_ingested_documents

        tree = ast.parse(
            textwrap.dedent(inspect.getsource(_wait_for_ingested_documents))
        )
        calls = [
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        ]
        assert "rollback" not in calls
        assert "commit" not in calls

    def test_it_disposes_the_engine_it_creates(self):
        """`create_celery_session` builds a fresh engine, and unless
        CELERY_DB_USE_NULLPOOL is set that is a QueuePool holding real
        connections. A Celery task makes one per invocation; a tool can be
        called many times inside a single job."""
        import inspect

        from app.services.agent_tool_dispatch import _wait_for_ingested_documents

        source = inspect.getsource(_wait_for_ingested_documents)
        assert "engine.dispose()" in source
        assert "finally:" in source
