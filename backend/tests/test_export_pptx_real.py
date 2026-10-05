"""export_document builds a presentation that opens.

The other export tests replace the builder, because `pptx` used to be stubbed
for every test. With the real library this runs plan -> write -> assemble ->
export and opens the file it stored.
"""

import io
from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.services.agent_tool_dispatch import (
    AgentToolExecutionContext,
    build_autonomous_document_authoring_provider,
)
from app.services.storage_service import storage_service

pytestmark = pytest.mark.unit


async def test_an_exported_deck_opens_and_holds_every_line(monkeypatch):
    pptx = pytest.importorskip("pptx")
    stored = {}

    async def initialize():
        return None

    async def upload_to_path(path, content, content_type=None):
        stored[path] = content

    async def get_presigned_download_url(path, expiry=None):
        return f"https://storage.test/{path}"

    monkeypatch.setattr(storage_service, "initialize", initialize)
    monkeypatch.setattr(storage_service, "upload_to_path", upload_to_path)
    monkeypatch.setattr(
        storage_service, "get_presigned_download_url", get_presigned_download_url
    )

    provider = build_autonomous_document_authoring_provider(SimpleNamespace())
    state = {}
    ctx = AgentToolExecutionContext(
        mode="autonomous",
        db=None,
        service=None,
        user_id="u",
        job=SimpleNamespace(id=uuid4(), user_id="u", name="j", config={}),
        state=state,
    )

    async def call(tool, params):
        result = await provider._handlers[tool](params, ctx)
        assert result.get("success") is True, result
        return result

    await call(
        "plan_document",
        {
            "title": "Prefetcher Study",
            "sections": [{"id": "results", "title": "Results"}],
        },
    )
    lines = "\n".join(f"- finding number {i}" for i in range(15))
    await call("write_section", {"section_id": "results", "content": lines})
    await call("assemble_document", {"include_toc": False})
    result = await call("export_document", {"format": "pptx"})

    deck = pptx.Presentation(io.BytesIO(stored[result["data"]["object_path"]]))
    text = "\n".join(
        shape.text_frame.text
        for slide in deck.slides
        for shape in slide.shapes
        if shape.has_text_frame
    )
    for i in range(15):
        assert f"finding number {i}" in text
    assert "Results (cont.)" in text
