"""Reading back whether the prompt cache actually hit.

The thinking prompt is split into a byte-stable prefix and a volatile tail so
the prefix can be cached, and Anthropic requests carry cache_control
breakpoints for the same reason. Nothing read back whether any of it worked, so
the arrangement rested on an assumption nobody could check -- a cache
mechanism whose hit rate is never measured is indistinguishable from a comment.

The distinction these tests defend: NULL is not zero. Zero is a measured miss;
NULL is a provider that said nothing. Averaging silence as zero reports a
healthy cache as broken.
"""

import pytest

from app.services.llm_service import _cache_tokens, _meta_from_completion

pytestmark = pytest.mark.unit


class TestProvidersSpellItDifferently:
    def test_deepseek(self):
        usage = {
            "prompt_tokens": 91,
            "prompt_cache_hit_tokens": 64,
            "prompt_cache_miss_tokens": 27,
        }
        assert _cache_tokens(usage) == (64, 27)

    def test_anthropic(self):
        usage = {
            "cache_read_input_tokens": 1200,
            "cache_creation_input_tokens": 300,
        }
        assert _cache_tokens(usage) == (1200, 300)

    def test_openai_reports_hits_only_so_the_miss_is_inferred(self):
        usage = {"prompt_tokens": 100, "prompt_tokens_details": {"cached_tokens": 40}}
        assert _cache_tokens(usage) == (40, 60)

    def test_an_inferred_miss_needs_both_halves(self):
        # Without the total, a miss of "everything" would be an invented rate.
        usage = {"prompt_tokens_details": {"cached_tokens": 40}}
        assert _cache_tokens(usage) == (40, None)


class TestSilenceIsNotZero:
    def test_a_provider_that_says_nothing_reports_nothing(self):
        assert _cache_tokens({"prompt_tokens": 50}) == (None, None)
        assert _cache_tokens({}) == (None, None)
        assert _cache_tokens(None) == (None, None)

    def test_a_measured_total_miss_is_zero_not_none(self):
        # This is the shape of a cold prefix, and it must be distinguishable
        # from a provider that reports no cache data at all.
        usage = {"prompt_cache_hit_tokens": 0, "prompt_cache_miss_tokens": 91}
        assert _cache_tokens(usage) == (0, 91)

    def test_junk_values_do_not_become_numbers(self):
        usage = {"prompt_cache_hit_tokens": "lots", "prompt_cache_miss_tokens": None}
        assert _cache_tokens(usage) == (None, None)


class TestItReachesTheMeta:
    def test_a_completion_carries_its_cache_accounting(self):
        data = {
            "id": "abc",
            "model": "deepseek-v4-pro",
            "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 91,
                "prompt_cache_hit_tokens": 64,
                "prompt_cache_miss_tokens": 27,
            },
        }
        meta = _meta_from_completion(data)
        assert meta["cache_hit_tokens"] == 64
        assert meta["cache_miss_tokens"] == 27

    def test_a_completion_without_cache_data_says_none(self):
        data = {
            "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10},
        }
        meta = _meta_from_completion(data)
        assert meta["cache_hit_tokens"] is None
        assert meta["cache_miss_tokens"] is None
