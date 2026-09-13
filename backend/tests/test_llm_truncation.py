"""Recovering a response the budget cut off.

The distinction being defended: truncation is not malformed output. A truncated
response is a *correct prefix* of the right answer, so closing it recovers work
the model already did, while treating it as malformed sends it back to rewrite
a plan it had got right -- on the same budget that just proved too small.

The other half is refusing to guess. A repair that invents a field the model
never wrote is worse than a failed parse, because the failure is visible and
the invention is not.
"""

import json

import pytest

from app.services import llm_truncation as trunc

pytestmark = pytest.mark.unit


class TestSpottingIt:
    def test_a_length_stop_is_truncation(self):
        assert trunc.is_truncated("length", "") is True
        assert trunc.is_truncated("max_tokens", "{") is True

    def test_a_normal_stop_is_not(self):
        assert trunc.is_truncated("stop", '{"a": 1}') is False
        assert trunc.is_truncated(None, "") is False
        assert trunc.is_truncated("tool_calls", "") is False


class TestAskingForMoreRoom:
    def test_it_doubles_rather_than_nudges(self):
        # The budget did not miss by a little: it was spent entirely on
        # reasoning before the answer began.
        assert trunc.next_budget(8000) == 16000

    def test_it_stops_at_the_ceiling(self):
        assert trunc.next_budget(20000) == trunc.MAX_RETRY_TOKENS

    def test_no_room_left_is_none_not_the_cap(self):
        # A caller already at the ceiling should report the truncation, not
        # repeat the identical call and describe it as a retry.
        assert trunc.next_budget(trunc.MAX_RETRY_TOKENS) is None
        assert trunc.next_budget(99999) is None

    def test_nonsense_budgets_do_not_become_retries(self):
        assert trunc.next_budget(0) is None
        assert trunc.next_budget(None) is None
        assert trunc.next_budget("many") is None


class TestClosingWhatWasStarted:
    def test_an_object_cut_after_a_complete_value(self):
        repaired = trunc.repair_truncated_json('{"tool": "run_tests", "args": {"x": 1}')
        assert json.loads(repaired) == {"tool": "run_tests", "args": {"x": 1}}

    def test_a_nested_structure_closes_in_the_right_order(self):
        repaired = trunc.repair_truncated_json('{"a": [1, 2, {"b": "c"}')
        assert json.loads(repaired) == {"a": [1, 2, {"b": "c"}]}

    def test_a_dangling_comma_is_dropped(self):
        repaired = trunc.repair_truncated_json('{"a": 1,')
        assert json.loads(repaired) == {"a": 1}

    def test_a_key_with_no_value_is_dropped_not_invented(self):
        # The model wrote the key and ran out before the value. Closing it with
        # null would put a field in the decision that nothing chose.
        repaired = trunc.repair_truncated_json('{"tool": "x", "arguments":')
        assert json.loads(repaired) == {"tool": "x"}

    def test_a_half_written_string_is_dropped(self):
        repaired = trunc.repair_truncated_json('{"tool": "run", "why": "because it')
        parsed = json.loads(repaired)
        assert parsed == {"tool": "run"}
        assert "why" not in parsed

    def test_leading_prose_before_the_json_is_skipped(self):
        # Models preface JSON with commentary constantly.
        repaired = trunc.repair_truncated_json('Here is my decision:\n{"tool": "a"')
        assert json.loads(repaired) == {"tool": "a"}


class TestRefusingToGuess:
    def test_balanced_json_is_not_a_truncation(self):
        # Returning a "repair" here would mask a different failure.
        assert trunc.repair_truncated_json('{"a": 1}') is None

    def test_text_with_no_json_at_all(self):
        assert trunc.repair_truncated_json("I could not decide.") is None
        assert trunc.repair_truncated_json("") is None
        assert trunc.repair_truncated_json(None) is None

    def test_too_little_arrived_to_close_honestly(self):
        assert trunc.repair_truncated_json('{"to') is None
        assert trunc.repair_truncated_json("{") is None

    def test_structurally_broken_is_not_unfinished(self):
        # A mismatched closer is the model writing something wrong, which is
        # what the LLM repair path is for.
        assert trunc.repair_truncated_json('{"a": [1, 2}') is None

    def test_the_repair_always_parses_or_is_none(self):
        """The invariant that makes this safe to try first.

        Every repair is either valid JSON or refused. A caller can attempt it
        unconditionally and fall back to the model when it declines.
        """
        cases = [
            '{"tool": "a", "args": {"b": [1, 2',
            '{"a": "b", "c": "d',
            '{"list": [{"x": 1}, {"y":',
            '{"n": 12',
            "[{",
            '[{"a": 1}, {"b": 2}',
            '{"a": {"b": {"c": "d"}}',
            "not json at all",
            '{"a": 1}',
            "",
        ]
        for case in cases:
            repaired = trunc.repair_truncated_json(case)
            if repaired is not None:
                json.loads(repaired)  # must not raise


class TestTheServiceRetriesOnce:
    """A response the budget cut off short gets more room, once.

    Measured before this existed: six calls in one run returned
    finish_reason='length' with completion_tokens exactly at the cap and no
    content, and the job could not leave iteration zero. The cause is known
    exactly at the point of failure, so reporting it was the wrong response.
    """

    @staticmethod
    def _reply(content, finish_reason, completion_tokens):
        return {
            "choices": [
                {"message": {"content": content}, "finish_reason": finish_reason}
            ],
            "usage": {"completion_tokens": completion_tokens},
            "model": "deepseek-v4-pro",
        }

    @pytest.mark.asyncio
    async def test_an_empty_truncated_reply_is_retried_with_more_room(
        self, monkeypatch
    ):
        from app.services.llm_service import LLMService

        service = LLMService()
        budgets = []

        class _Response:
            def __init__(self, payload):
                self._payload = payload

            def raise_for_status(self):
                return None

            def json(self):
                return self._payload

        async def fake_post(url, json=None, headers=None, timeout=None):
            budgets.append(json["max_tokens"])
            if len(budgets) == 1:
                return _Response(
                    TestTheServiceRetriesOnce._reply("", "length", json["max_tokens"])
                )
            return _Response(
                TestTheServiceRetriesOnce._reply('{"tool": "x"}', "stop", 12)
            )

        monkeypatch.setattr(service.client, "post", fake_post)
        monkeypatch.setattr(
            "app.services.llm_service.settings.DEEPSEEK_API_KEY", "test-key"
        )

        content, _meta = await service._make_deepseek_chat_request(
            model="deepseek-v4-pro",
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.2,
            max_tokens=8000,
        )
        assert content == '{"tool": "x"}'
        assert len(budgets) == 2
        assert budgets[1] > budgets[0], "the retry used the same budget"

    @pytest.mark.asyncio
    async def test_it_gives_up_after_one_retry(self, monkeypatch):
        # A second truncation means the prompt is the problem; doubling again
        # only spends more to fail the same way.
        from app.services.llm_service import LLMService, LLMServiceError

        service = LLMService()
        calls = []

        class _Response:
            def raise_for_status(self):
                return None

            def json(self):
                return TestTheServiceRetriesOnce._reply("", "length", 9999)

        async def fake_post(url, json=None, headers=None, timeout=None):
            calls.append(json["max_tokens"])
            return _Response()

        monkeypatch.setattr(service.client, "post", fake_post)
        monkeypatch.setattr(
            "app.services.llm_service.settings.DEEPSEEK_API_KEY", "test-key"
        )

        with pytest.raises(LLMServiceError) as caught:
            await service._make_deepseek_chat_request(
                model="deepseek-v4-pro",
                messages=[{"role": "user", "content": "hi"}],
                temperature=0.2,
                max_tokens=8000,
            )
        assert len(calls) == 2
        # And says so, rather than reporting a first failure that was not one.
        assert "already retried" in str(caught.value)

    @pytest.mark.asyncio
    async def test_an_empty_reply_that_was_not_truncated_still_reports(
        self, monkeypatch
    ):
        # finish_reason 'stop' with no content is a different failure, and
        # more room would not fix it.
        from app.services.llm_service import LLMService, LLMServiceError

        service = LLMService()
        calls = []

        class _Response:
            def raise_for_status(self):
                return None

            def json(self):
                return TestTheServiceRetriesOnce._reply("", "stop", 5)

        async def fake_post(url, json=None, headers=None, timeout=None):
            calls.append(1)
            return _Response()

        monkeypatch.setattr(service.client, "post", fake_post)
        monkeypatch.setattr(
            "app.services.llm_service.settings.DEEPSEEK_API_KEY", "test-key"
        )

        with pytest.raises(LLMServiceError):
            await service._make_deepseek_chat_request(
                model="deepseek-v4-pro",
                messages=[{"role": "user", "content": "hi"}],
                temperature=0.2,
                max_tokens=8000,
            )
        assert len(calls) == 1, "a non-truncated empty reply must not be retried"
