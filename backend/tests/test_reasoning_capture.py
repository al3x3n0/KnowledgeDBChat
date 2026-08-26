"""Keep what a reasoning model thought, not only what it said.

DeepSeek and its kind return the chain of thought in a separate field and
charge it against max_tokens. The client read only the answer, so those tokens
were paid for and discarded: an agent's decision could be replayed while the
reasoning behind it could not, and a call that spent its entire budget
thinking looked simply empty.
"""

import pytest

from app.services import llm_service as mod


class _Recorder:
    """Captures what the snapshot recorder is handed."""

    def __init__(self):
        self.rows = []

    def add(self, row):
        self.rows.append(row)


def _response(reasoning=None, reasoning_tokens=None, content="the answer"):
    message = {"content": content}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    usage = {"prompt_tokens": 10, "completion_tokens": 900, "total_tokens": 910}
    if reasoning_tokens is not None:
        usage["completion_tokens_details"] = {"reasoning_tokens": reasoning_tokens}
    return {
        "id": "resp-1",
        "model": "deepseek-v4-pro",
        "usage": usage,
        "choices": [{"message": message, "finish_reason": "stop"}],
    }


class TestTheUsageRowStaysSmall:
    def test_size_is_recorded_rather_than_the_text(self):
        """A usage event is written for every call; a trace can run to
        thousands of tokens and would dwarf the rest of the row."""
        meta = mod._meta_from_completion(_response("thinking " * 500, 850))

        assert meta["reasoning_tokens"] == 850
        assert meta["reasoning_chars"] == len("thinking " * 500)
        assert "reasoning" not in meta, "the text must not ride along in usage"

    def test_a_model_that_does_not_reason_records_nothing(self):
        meta = mod._meta_from_completion(_response())
        assert meta["reasoning_chars"] == 0
        assert meta["reasoning_tokens"] is None


class TestTheTextReachesTheSnapshot:
    def test_it_is_carried_out_of_band(self):
        mod._LAST_REASONING.set(None)
        mod._meta_from_completion(_response("because the loop is dependent", 42))

        carried = mod._LAST_REASONING.get()
        assert carried == ("because the loop is dependent", 42)

    def test_nothing_is_carried_when_there_was_no_reasoning(self):
        mod._LAST_REASONING.set(("stale", 1))
        mod._meta_from_completion(_response())

        assert mod._LAST_REASONING.get() is None, "a later call must not inherit"

    @pytest.mark.asyncio
    async def test_concurrent_calls_do_not_read_each_other(self):
        """Instance state would cross-talk under LLM_MAX_CONCURRENCY; a
        ContextVar gives each task its own copy."""
        import asyncio

        async def one(text):
            mod._meta_from_completion(_response(text, 5))
            await asyncio.sleep(0)  # yield, so the tasks interleave
            return (mod._LAST_REASONING.get() or (None,))[0]

        got = await asyncio.gather(
            asyncio.create_task(one("first")), asyncio.create_task(one("second"))
        )
        assert got == ["first", "second"]


class TestBothWritersCoerceTheUserId:
    """The same defect, twice.

    str(job.user_id) into a UUID(as_uuid=True) column works on PostgreSQL and
    raises on SQLite. It was fixed for the usage event and left in place for the
    snapshot, so switching snapshots on reproduced it exactly -- at iteration 4
    of a live run, with 19,340 characters of captured reasoning that could not
    be written.
    """

    def test_no_uuid_column_is_written_from_a_raw_argument(self):
        import inspect

        source = inspect.getsource(mod)
        for writer in ("LLMUsageEvent(", "LLMCallSnapshot("):
            start = source.index(writer)
            first_field = source[start : start + 200]
            assert (
                "user_id=_usage_user_id(" in first_field
            ), f"{writer} must coerce user_id, not pass it through"

    def test_a_string_id_survives_the_round_trip(self):
        from uuid import uuid4

        u = uuid4()
        assert mod._usage_user_id(str(u)) == u
        assert mod._usage_user_id(u) == u
        assert mod._usage_user_id("not-a-uuid") is None


def test_the_recorder_defaults_to_the_call_that_just_happened():
    """Three snapshot sites, one of which remembered to pass the reasoning.

    The one that did not was the structured path -- the agent's decision phase,
    the only reasoning anybody would go looking for. Defaulting inside the
    recorder removes the chance to forget.
    """
    import inspect

    source = inspect.getsource(mod.LLMService._record_call_snapshot)
    assert (
        "_LAST_REASONING.get()" in source
    ), "the recorder must read the reasoning itself, not rely on callers"
