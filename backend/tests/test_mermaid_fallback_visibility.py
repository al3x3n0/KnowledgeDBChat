"""Rendering a diagram off-site should not be silent.

The compose ran Kroki without the Mermaid companion it delegates to, so local
rendering failed with a connection refused and every diagram was rendered by
the public kroki.io. It worked, so nothing looked wrong, and the diagram source
left the deployment on each call.
"""

import pytest

from app.services.mermaid_renderer import MermaidRenderer


@pytest.mark.asyncio
async def test_local_success_does_not_touch_the_fallback(monkeypatch):
    renderer = MermaidRenderer()
    seen = []

    async def _render(code, format="png", base_url=None):
        seen.append(base_url)
        return b"\x89PNG local"

    monkeypatch.setattr(renderer, "_render_via_kroki", _render)
    monkeypatch.setattr(renderer, "_validate_mermaid_code", lambda code: (True, None))

    await renderer.render_to_png("graph TD; A-->B;")

    assert seen == [renderer._kroki_url]
    assert renderer.last_render_used_fallback is False


@pytest.mark.asyncio
async def test_falling_back_is_recorded(monkeypatch, caplog):
    renderer = MermaidRenderer()

    async def _render(code, format="png", base_url=None):
        if base_url == renderer._kroki_url:
            raise RuntimeError("Connection refused: /127.0.0.1:8002")
        return b"\x89PNG remote"

    monkeypatch.setattr(renderer, "_render_via_kroki", _render)
    monkeypatch.setattr(renderer, "_validate_mermaid_code", lambda code: (True, None))
    monkeypatch.setattr(renderer, "_use_fallback", True)
    monkeypatch.setattr(renderer, "_fallback_url", "https://kroki.io")

    out = await renderer.render_to_png("graph TD; A-->B;")

    assert out == b"\x89PNG remote"
    assert renderer.last_render_used_fallback is True
