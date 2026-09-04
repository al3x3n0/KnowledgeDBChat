"""Choosing which memories a job starts with.

This asked a language model to sort a list of UUIDs, and it was the most
expensive thing in a run: 245 calls in one day for 646,989 reasoning tokens,
39s mean and 210s worst, all of it *before* the loop starts -- long enough
that a job could sit at iteration zero while its execution lease lapsed.

Routing it to the "fast" tier was the obvious fix and made it worse, measured
on the same ranking: deepseek-v4-flash spent 23,548 reasoning tokens and 291
seconds where pro had averaged 39. Both DeepSeek models reason before
answering, so there was no cheap tier to move it to. The mistake was using a
language model at all -- relevance between a goal and a memory is a similarity
question, and this codebase already answers those with embeddings.
"""

import pytest

from app.services.agent_job_memory_service import AgentJobMemoryService

pytestmark = pytest.mark.unit


class _Memory:
    def __init__(self, ident, content):
        self.id = ident
        self.content = content
        self.memory_type = "fact"


class _Job:
    def __init__(self, goal="benchmark a C kernel", job_type="research"):
        self.goal = goal
        self.job_type = job_type


class _Embedder:
    """Stands in for the ONNX encoder with a deterministic toy embedding.

    Each text becomes a vector of term counts over a fixed vocabulary, which is
    enough for cosine similarity to prefer the memory that shares words with
    the goal -- and, unlike a mock returning canned scores, it exercises the
    real normalisation and argsort.
    """

    VOCAB = ("benchmark", "kernel", "cake", "compiler", "recipe")

    def encode(self, texts, show_progress_bar=False):
        import numpy as np

        rows = []
        for text in texts:
            lowered = (text or "").lower()
            rows.append([float(lowered.count(word)) for word in self.VOCAB])
        return np.asarray(rows, dtype="float32")


def _service(embedder):
    service = AgentJobMemoryService()
    service._embedder = embedder
    return service


class TestItRanksBySimilarity:
    @pytest.mark.asyncio
    async def test_the_closest_memory_comes_first(self):
        memories = [
            _Memory("m1", "a cake recipe with butter"),
            _Memory("m2", "benchmark the kernel with a compiler"),
            _Memory("m3", "an unrelated note"),
        ]
        ranked = await _service(_Embedder())._rank_memories_by_relevance(
            _Job(), memories, "user", None, None
        )
        assert ranked[0].id == "m2"

    @pytest.mark.asyncio
    async def test_it_returns_every_memory_it_was_given(self):
        # Ranking reorders; it must never drop one, or the caller's limit
        # silently selects from a smaller set than it thinks.
        memories = [_Memory(f"m{i}", f"note {i}") for i in range(8)]
        ranked = await _service(_Embedder())._rank_memories_by_relevance(
            _Job(), memories, "user", None, None
        )
        assert sorted(m.id for m in ranked) == sorted(m.id for m in memories)

    @pytest.mark.asyncio
    async def test_the_same_input_gives_the_same_order(self):
        # The LLM ordering was not reproducible: two identical jobs could be
        # given different memories with nothing recording why.
        memories = [
            _Memory("m1", "benchmark kernel"),
            _Memory("m2", "cake recipe"),
            _Memory("m3", "compiler notes"),
        ]
        service = _service(_Embedder())
        first = await service._rank_memories_by_relevance(
            _Job(), memories, "user", None, None
        )
        second = await service._rank_memories_by_relevance(
            _Job(), memories, "user", None, None
        )
        assert [m.id for m in first] == [m.id for m in second]

    @pytest.mark.asyncio
    async def test_an_empty_memory_does_not_produce_nan(self):
        # A zero vector divided by its zero norm sorts unpredictably; an empty
        # memory should rank last, not randomly.
        memories = [
            _Memory("empty", ""),
            _Memory("relevant", "benchmark kernel benchmark kernel"),
        ]
        ranked = await _service(_Embedder())._rank_memories_by_relevance(
            _Job(), memories, "user", None, None
        )
        assert ranked[0].id == "relevant"
        assert len(ranked) == 2


class TestItDegradesRatherThanFails:
    @pytest.mark.asyncio
    async def test_no_encoder_falls_back_to_the_given_order(self):
        # Importance order, which is what the caller already used when the old
        # ranking raised.
        memories = [_Memory("m1", "a"), _Memory("m2", "b")]
        service = AgentJobMemoryService()

        async def _none():
            return None

        service._load_embedder = _none
        ranked = await service._rank_memories_by_relevance(
            _Job(), memories, "user", None, None
        )
        assert [m.id for m in ranked] == ["m1", "m2"]

    @pytest.mark.asyncio
    async def test_an_encoder_that_throws_does_not_end_memory_injection(self):
        class _Broken:
            def encode(self, texts, show_progress_bar=False):
                raise RuntimeError("model file is corrupt")

        memories = [_Memory("m1", "a"), _Memory("m2", "b")]
        ranked = await _service(_Broken())._rank_memories_by_relevance(
            _Job(), memories, "user", None, None
        )
        assert [m.id for m in ranked] == ["m1", "m2"]

    @pytest.mark.asyncio
    async def test_a_job_with_no_goal_is_not_ranked_against_nothing(self):
        memories = [_Memory("m1", "a"), _Memory("m2", "b")]
        ranked = await _service(_Embedder())._rank_memories_by_relevance(
            _Job(goal="", job_type=""), memories, "user", None, None
        )
        assert [m.id for m in ranked] == ["m1", "m2"]


class TestItCostsNoTokens:
    @pytest.mark.asyncio
    async def test_no_language_model_is_called(self):
        """The whole point. A service whose llm_service would explode proves
        the ranking never reaches for one."""

        class _Explodes:
            def __getattr__(self, name):
                raise AssertionError(
                    f"ranking called the language model ({name}); it costs "
                    "hundreds of seconds and tens of thousands of "
                    "reasoning tokens before the loop even starts"
                )

        service = _service(_Embedder())
        service.llm_service = _Explodes()
        memories = [_Memory("m1", "benchmark kernel"), _Memory("m2", "cake")]
        ranked = await service._rank_memories_by_relevance(
            _Job(), memories, "user", None, None
        )
        assert ranked[0].id == "m1"


class TestTheCandidateWindowIsStable:
    def test_the_query_breaks_ties_on_a_stable_key(self):
        """`LIMIT 50` over `ORDER BY importance_score` alone.

        Importance scores tie constantly, and Postgres is free to return a
        different fifty each time, so which memories a job was offered varied
        between identical runs. A deterministic ranking over a shuffled
        candidate set is still not reproducible.
        """
        from pathlib import Path

        source = Path("app/services/agent_job_memory_service.py")
        if not source.exists():  # pragma: no cover
            source = (
                Path(__file__).resolve().parents[1]
                / "app"
                / "services"
                / "agent_job_memory_service.py"
            )
        text = source.read_text()
        assert (
            "desc(ConversationMemory.importance_score), ConversationMemory.id" in text
        ), "the candidate window has no tiebreak and is not reproducible"
