"""Tests for queued follow-up target resolution from decision traces."""

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app.api.endpoints.agent_jobs import _trace_event_follow_up_target
from app.modules.autonomy.application.decision_trace_follow_up_targets import (
    DecisionTraceFollowUpTargetError,
    resolve_follow_up_target,
)


def _event(**overrides):
    values = {
        "source_kind": "domain_profile",
        "source_id": "profile-1",
        "event_type": "follow_up_queued",
        "decision_type": None,
        "event_metadata": {"profile_opportunity_id": "opp-profile-1"},
        "deep_link": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    ("event", "expected"),
    [
        (
            _event(),
            ("domain_profile", "profile-1", "opp-profile-1"),
        ),
        (
            _event(
                source_kind=" PORTFOLIO ",
                source_id=" fleet-1 ",
                event_type=None,
                decision_type="follow_up_queued_for_approval",
                event_metadata={"portfolio_opportunity_id": "opp-fleet-1"},
            ),
            ("portfolio", "fleet-1", "opp-fleet-1"),
        ),
        (
            _event(event_metadata={"opportunity_id": "opp-generic-1"}),
            ("domain_profile", "profile-1", "opp-generic-1"),
        ),
        (
            _event(
                event_metadata=None,
                deep_link={"params": {"opportunityId": "opp-link-1"}},
            ),
            ("domain_profile", "profile-1", "opp-link-1"),
        ),
    ],
)
def test_resolve_follow_up_target_supports_persisted_identifier_shapes(
    event,
    expected,
):
    assert resolve_follow_up_target(event) == expected


@pytest.mark.parametrize(
    ("event", "detail"),
    [
        (
            _event(source_kind="job"),
            "Decision trace event does not support follow-up approval actions",
        ),
        (
            _event(event_type="follow_up_failed"),
            "Decision trace event is not a pending follow-up approval",
        ),
        (
            _event(source_id=""),
            "Decision trace event is missing its follow-up owner identifier",
        ),
        (
            _event(event_metadata={}, deep_link={}),
            "Decision trace event is missing its follow-up opportunity identifier",
        ),
    ],
)
def test_resolve_follow_up_target_rejects_incomplete_or_ineligible_events(
    event,
    detail,
):
    with pytest.raises(DecisionTraceFollowUpTargetError) as exc_info:
        resolve_follow_up_target(event)

    assert exc_info.value.detail == detail


def test_endpoint_adapter_translates_target_error_to_http_422():
    with pytest.raises(HTTPException) as exc_info:
        _trace_event_follow_up_target(_event(source_kind="job"))

    assert exc_info.value.status_code == 422
    assert (
        exc_info.value.detail
        == "Decision trace event does not support follow-up approval actions"
    )
