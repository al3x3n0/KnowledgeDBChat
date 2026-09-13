"""Tests for the research-inbox follow-up relaunch workflow."""

from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.modules.autonomy.application.follow_up_inbox_relaunch import (
    InboxFollowUpRelaunchDependencies,
    relaunch_inbox_follow_up,
)
from app.modules.autonomy.application.follow_up_queue_inbox import (
    FollowUpQueueActionError,
)


def _item(**overrides):
    values = {
        "id": uuid4(),
        "status": "accepted",
        "customer": "Acme",
        "follow_up_launch_status": "launched",
        "follow_up_outcome_status": "failed",
        "follow_up_recommendation_key": "deep_dive_chain",
        "follow_up_operator_decision": "approved_launch",
        "follow_up_decision": "approved_and_launched",
        "follow_up_job_id": uuid4(),
        "follow_up_chain_definition_id": None,
        "follow_up_launched_at": None,
        "follow_up_block_reason": "previous failure",
        "follow_up_budget_decision": "allow",
        "follow_up_budget_reason": "within budget",
        "follow_up_budget_throttle_state": "clear",
        "follow_up_customer_budget_decision": "allow",
        "follow_up_customer_budget_reason": "within customer budget",
        "follow_up_customer_budget_throttle_state": "clear",
        "follow_up_outcome_recorded_at": object(),
        "follow_up_outcome_summary": "failed verification",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _dependencies(*, actions=()):
    return InboxFollowUpRelaunchDependencies(
        load_learning_profile=AsyncMock(return_value={"token_scores": {}}),
        build_follow_up_actions=lambda *_args, **_kwargs: list(actions),
        launch_follow_up_action=AsyncMock(return_value=SimpleNamespace(id=uuid4())),
        project_relaunch_to_originating_opportunity=AsyncMock(),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("item", "status_code", "detail"),
    [
        (
            _item(status="new"),
            400,
            "Only accepted inbox items can relaunch a follow-up",
        ),
        (
            _item(follow_up_outcome_status="succeeded"),
            400,
            "Only failed or cancelled launched follow-ups can be relaunched",
        ),
    ],
)
async def test_relaunch_validates_item_state(item, status_code, detail):
    with pytest.raises(FollowUpQueueActionError) as exc_info:
        await relaunch_inbox_follow_up(
            item=item,
            operator_note=None,
            db=object(),
            current_user=SimpleNamespace(id=uuid4()),
            deps=_dependencies(),
        )

    assert exc_info.value.status_code == status_code
    assert exc_info.value.detail == detail


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("actions", "detail"),
    [
        ([], "Stored follow-up recommendation can no longer be resolved"),
        (
            [
                SimpleNamespace(
                    recommendation_key="deep_dive_chain",
                    autonomy_eligibility="manual_only",
                )
            ],
            "Stored follow-up recommendation is no longer safe to relaunch",
        ),
    ],
)
async def test_relaunch_requires_resolvable_safe_recommendation(actions, detail):
    with pytest.raises(FollowUpQueueActionError) as exc_info:
        await relaunch_inbox_follow_up(
            item=_item(),
            operator_note=None,
            db=object(),
            current_user=SimpleNamespace(id=uuid4()),
            deps=_dependencies(actions=actions),
        )

    assert exc_info.value.status_code == 422
    assert exc_info.value.detail == detail


@pytest.mark.asyncio
async def test_relaunch_launches_resets_outcome_and_projects_origin():
    item = _item(follow_up_outcome_status="cancelled")
    action = SimpleNamespace(
        recommendation_key="deep_dive_chain",
        autonomy_eligibility="auto_launchable",
        chain_create_payload=None,
    )
    dependencies = _dependencies(actions=[action])
    current_user = SimpleNamespace(id=uuid4())
    db = object()

    response = await relaunch_inbox_follow_up(
        item=item,
        operator_note=" Retry after repair. ",
        db=db,
        current_user=current_user,
        deps=dependencies,
    )

    launched = dependencies.launch_follow_up_action.await_args.args[0]
    assert launched is action
    assert item.follow_up_decision == "relaunched"
    assert item.follow_up_launch_status == "launched"
    assert item.follow_up_job_id == response.follow_up_job_id
    assert item.follow_up_block_reason == "Retry after repair."
    assert item.follow_up_outcome_status is None
    assert item.follow_up_outcome_recorded_at is None
    assert item.follow_up_outcome_summary is None
    assert item.follow_up_budget_decision is None
    dependencies.project_relaunch_to_originating_opportunity.assert_awaited_once_with(
        db=db,
        job=dependencies.launch_follow_up_action.return_value,
        launched_at=item.follow_up_launched_at,
    )
