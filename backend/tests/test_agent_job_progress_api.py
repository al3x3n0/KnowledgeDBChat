"""Focused tests for the modular autonomous-job progress socket."""

import pytest

from app.modules.autonomy.api.job_progress import build_job_progress_api


class _WebSocket:
    def __init__(self):
        self.close_calls = []

    async def close(self, **kwargs):
        self.close_calls.append(kwargs)


@pytest.mark.asyncio
async def test_job_progress_rejects_invalid_token_before_opening_session():
    session_opened = False

    async def authenticate(_token):
        return None

    def session_factory():
        nonlocal session_opened
        session_opened = True
        raise AssertionError("session should not be opened")

    api = build_job_progress_api(
        authenticate_token=authenticate,
        session_factory=session_factory,
    )
    websocket = _WebSocket()

    await api.agent_job_progress_websocket(
        websocket=websocket,
        job_id="unused",
        token="invalid",
    )

    assert websocket.close_calls == [{"code": 4001, "reason": "Invalid token"}]
    assert session_opened is False
