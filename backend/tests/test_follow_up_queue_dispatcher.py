"""Validation tests for follow-up queue target dispatch."""

import pytest

from app.modules.autonomy.application import (
    follow_up_queue_inbox,
    follow_up_queue_portfolios,
    follow_up_queue_profiles,
)
from app.modules.autonomy.application.follow_up_queue_dispatcher import (
    FollowUpQueueDispatcherDependencies,
    dispatch_follow_up_queue_action,
)
from app.modules.autonomy.application.follow_up_queue_inbox import (
    FollowUpQueueActionError,
)

DEPENDENCIES = FollowUpQueueDispatcherDependencies(
    load_learning_profile=lambda **_kwargs: None,
    build_follow_up_actions=lambda *_args, **_kwargs: [],
    launch_follow_up_action=lambda *_args, **_kwargs: None,
    build_portfolio_summary=lambda *_args, **_kwargs: {},
    build_profile_summary=lambda *_args, **_kwargs: {},
    classify_operator_review=lambda *_args, **_kwargs: None,
    sync_portfolio_queue_state=lambda **_kwargs: None,
    sync_profile_queue_state=lambda **_kwargs: None,
    resolve_portfolio_parent_job=lambda **_kwargs: None,
    resolve_profile_parent_job=lambda **_kwargs: None,
    execute_agent_job_task=object(),
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("action", "targets", "detail"),
    [
        ("retry", {}, "Unknown follow-up queue action"),
        ("approve_launch", {}, "Queue action target is required"),
        (
            "reject_launch",
            {"item": object(), "portfolio": object()},
            "Queue action target is ambiguous",
        ),
    ],
)
async def test_dispatcher_validates_action_and_exactly_one_target(
    action,
    targets,
    detail,
):
    with pytest.raises(FollowUpQueueActionError) as exc_info:
        await dispatch_follow_up_queue_action(
            action=action,
            operator_note=None,
            db=object(),
            current_user=object(),
            deps=DEPENDENCIES,
            **targets,
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == detail


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("module", "handler_name", "targets", "target_key"),
    [
        (
            follow_up_queue_inbox,
            "perform_inbox_follow_up_queue_action",
            {"item": object()},
            "item",
        ),
        (
            follow_up_queue_portfolios,
            "perform_portfolio_follow_up_queue_action",
            {"portfolio": object(), "portfolio_opportunity_id": "portfolio-opp"},
            "portfolio",
        ),
        (
            follow_up_queue_profiles,
            "perform_profile_follow_up_queue_action",
            {"profile": object(), "profile_opportunity_id": "profile-opp"},
            "profile",
        ),
    ],
)
async def test_dispatcher_routes_normalized_action_to_target_handler(
    monkeypatch,
    module,
    handler_name,
    targets,
    target_key,
):
    captured = {}
    expected = object()

    async def handler(**kwargs):
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(module, handler_name, handler)
    result = await dispatch_follow_up_queue_action(
        action=" APPROVE_LAUNCH ",
        operator_note="Approved",
        db=object(),
        current_user=object(),
        deps=DEPENDENCIES,
        **targets,
    )

    assert result is expected
    assert captured["action"] == "approve_launch"
    assert captured["operator_note"] == "Approved"
    assert captured[target_key] is targets[target_key]
