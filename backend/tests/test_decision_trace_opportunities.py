"""Focused tests for bound opportunity decision-trace projection."""

from datetime import datetime
from unittest.mock import Mock

from app.modules.autonomy.application.decision_trace_opportunities import (
    OpportunityDecisionTraceDependencies,
    bind_opportunity_decision_trace,
)

NOW = datetime(2026, 8, 1, 12, 0, 0)


def test_bound_opportunity_trace_builder_injects_projection_dependencies():
    build_event = Mock(side_effect=lambda **kwargs: kwargs)
    build_context = Mock(return_value={"objective": "Verify compiler hotspot"})
    builder = bind_opportunity_decision_trace(
        deps=OpportunityDecisionTraceDependencies(
            parse_time=Mock(return_value=NOW),
            reason_label=Mock(return_value="Queued for approval"),
            build_event=build_event,
            build_operator_context=build_context,
        )
    )

    events = builder(
        source_kind="portfolio",
        source_id="fleet-1",
        source_label="Compiler Fleet",
        customer=None,
        opportunities=[
            {
                "opportunity_id": "opp-1",
                "title": "Vectorization regression",
                "updated_at": "2026-08-01T12:00:00Z",
                "follow_up_review_status": "pending_approval",
                "last_decision_reason_code": "follow_up_queued",
            }
        ],
        deep_link_params={"tab": "fleet", "fleetId": "fleet-1"},
        objective="Verify compiler hotspot",
    )

    assert len(events) == 1
    event = events[0]
    assert event["event_type"] == "follow_up_queued"
    assert event["event_time"] == NOW
    assert event["reason_label"] == "Queued for approval"
    assert event["deep_link"]["params"]["opportunityId"] == "opp-1"
    assert event["operator_context"] == {"objective": "Verify compiler hotspot"}
    build_event.assert_called_once()
    build_context.assert_called_once()


def test_bound_opportunity_trace_builder_skips_rows_without_event_time():
    build_event = Mock()
    builder = bind_opportunity_decision_trace(
        deps=OpportunityDecisionTraceDependencies(
            parse_time=Mock(return_value=None),
            reason_label=Mock(),
            build_event=build_event,
            build_operator_context=Mock(),
        )
    )

    events = builder(
        source_kind="domain_profile",
        source_id="profile-1",
        source_label="Compiler Frontier",
        customer=None,
        opportunities=[{"opportunity_id": "opp-1"}],
        deep_link_params={"tab": "domain"},
    )

    assert events == []
    build_event.assert_not_called()
