import pytest


@pytest.mark.asyncio
async def test_generate_multi_queries_passes_user_settings_through():
    from app.services.query_processor import QueryProcessor
    from app.services.llm_service import UserLLMSettings

    qp = QueryProcessor()
    settings = UserLLMSettings(provider="deepseek", model="deepseek-chat")

    seen = {}

    class FakeLLM:
        async def generate_response(self, *, user_settings=None, **kwargs):
            seen["user_settings"] = user_settings
            # Return exactly one variation line.
            return "alt phrasing"

    variations = await qp.generate_multi_queries(
        "hello",
        llm_service=FakeLLM(),
        num_queries=2,
        user_settings=settings,
    )

    assert seen.get("user_settings") is settings
    assert "hello" in variations
    assert any(v == "alt phrasing" for v in variations)

