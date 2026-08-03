"""Resolve follow-up approval targets from persisted decision-trace events."""

from app.models.autonomy_decision_event import AutonomyDecisionEvent

FOLLOW_UP_SOURCE_KINDS = frozenset({"domain_profile", "portfolio"})
FOLLOW_UP_PENDING_EVENT_KINDS = frozenset(
    {"follow_up_queued", "follow_up_queued_for_approval"}
)


class DecisionTraceFollowUpTargetError(Exception):
    def __init__(self, detail: str):
        super().__init__(detail)
        self.detail = detail


def resolve_follow_up_target(
    event: AutonomyDecisionEvent,
) -> tuple[str, str, str]:
    """Return source kind, owner ID, and opportunity ID for a queued follow-up."""
    source_kind = str(event.source_kind or "").strip().lower()
    if source_kind not in FOLLOW_UP_SOURCE_KINDS:
        raise DecisionTraceFollowUpTargetError(
            "Decision trace event does not support follow-up approval actions"
        )

    event_kind = str(event.event_type or event.decision_type or "").strip().lower()
    if event_kind not in FOLLOW_UP_PENDING_EVENT_KINDS:
        raise DecisionTraceFollowUpTargetError(
            "Decision trace event is not a pending follow-up approval"
        )

    source_id = str(event.source_id or "").strip()
    if not source_id:
        raise DecisionTraceFollowUpTargetError(
            "Decision trace event is missing its follow-up owner identifier"
        )

    metadata = event.event_metadata if isinstance(event.event_metadata, dict) else {}
    deep_link = event.deep_link if isinstance(event.deep_link, dict) else {}
    deep_link_params = (
        deep_link.get("params") if isinstance(deep_link.get("params"), dict) else {}
    )
    opportunity_id = str(
        metadata.get("profile_opportunity_id")
        or metadata.get("portfolio_opportunity_id")
        or metadata.get("opportunity_id")
        or deep_link_params.get("opportunityId")
        or ""
    ).strip()
    if not opportunity_id:
        raise DecisionTraceFollowUpTargetError(
            "Decision trace event is missing its follow-up opportunity identifier"
        )

    return source_kind, source_id, opportunity_id
