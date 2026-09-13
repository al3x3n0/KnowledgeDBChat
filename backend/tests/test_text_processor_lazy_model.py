from unittest.mock import AsyncMock, patch

import pytest

from app.services.text_processor import TextProcessor


def test_text_processor_does_not_load_semantic_model_during_construction():
    with patch("app.services.onnx_embeddings.load_text_embedder") as model_factory:
        processor = TextProcessor()

    model_factory.assert_not_called()
    assert processor.semantic_model is None
    assert processor._semantic_model_load_attempted is False


@pytest.mark.asyncio
async def test_semantic_model_is_loaded_only_once():
    processor = TextProcessor()
    model = object()

    with patch(
        "app.services.onnx_embeddings.load_text_embedder", return_value=model
    ) as model_factory:
        assert await processor._ensure_semantic_model() is True
        assert await processor._ensure_semantic_model() is True

    model_factory.assert_called_once()
    assert processor.semantic_model is model


@pytest.mark.asyncio
async def test_fixed_chunking_does_not_initialize_semantic_model():
    processor = TextProcessor()
    processor._ensure_semantic_model = AsyncMock(return_value=True)

    chunks = await processor.split_text(
        "A sufficiently long fixed-size paragraph that should remain available.",
        strategy="fixed",
    )

    processor._ensure_semantic_model.assert_not_awaited()
    assert chunks
