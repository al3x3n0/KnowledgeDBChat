"""Tests for the follow-up queue application-to-HTTP adapter."""

from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import HTTPException

from app.modules.autonomy.api.follow_up_queue_actions import (
    build_follow_up_queue_action_api,
)
from app.modules.autonomy.application.follow_up_queue_inbox import (
    FollowUpQueueActionError,
)


@pytest.mark.asyncio
async def test_adapter_resolves_fresh_dependencies_and_returns_dispatch_result():
    dependencies = object()
    dependencies_factory = Mock(return_value=dependencies)
    expected = object()
    dispatcher = AsyncMock(return_value=expected)
    api = build_follow_up_queue_action_api(
        dependencies_factory=dependencies_factory,
        dispatcher=dispatcher,
    )

    result = await api.perform_follow_up_queue_action(
        item=object(),
        action="approve_launch",
        operator_note="Approved",
        db=object(),
        current_user=object(),
    )

    assert result is expected
    dependencies_factory.assert_called_once_with()
    assert dispatcher.await_args.kwargs["deps"] is dependencies
    assert dispatcher.await_args.kwargs["action"] == "approve_launch"
    assert dispatcher.await_args.kwargs["operator_note"] == "Approved"


@pytest.mark.asyncio
async def test_adapter_translates_follow_up_action_error_to_http_exception():
    dispatcher = AsyncMock(
        side_effect=FollowUpQueueActionError(
            status_code=422,
            detail="Recommendation is no longer safe",
        )
    )
    api = build_follow_up_queue_action_api(
        dependencies_factory=Mock(return_value=object()),
        dispatcher=dispatcher,
    )

    with pytest.raises(HTTPException) as exc_info:
        await api.perform_follow_up_queue_action(action="approve_launch")

    assert exc_info.value.status_code == 422
    assert exc_info.value.detail == "Recommendation is no longer safe"
