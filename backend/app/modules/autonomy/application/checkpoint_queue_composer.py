"""Compose all autonomy checkpoint sources into one prioritized queue."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

from app.models.agent_job import AgentJob
from app.models.domain_research_profile import DomainResearchProfile
from app.models.research_inbox import ResearchInboxItem
from app.models.research_portfolio import ResearchPortfolio
from app.schemas.agent_job import AgentCheckpointQueueItemResponse

from . import (
    checkpoint_queue_inbox,
    checkpoint_queue_jobs,
    checkpoint_queue_monitors,
    checkpoint_queue_portfolios,
    checkpoint_queue_profiles,
)


@dataclass(frozen=True)
class CheckpointQueueCompositionDependencies:
    extract_approval_checkpoint: Callable[..., Any]
    extract_scheduler_state: Callable[..., Any]
    queue_customer_for_job: Callable[..., Any]
    present_job: Callable[..., Any]
    queue_priority_fields: Callable[..., Any]
    queue_evidence_summary_for_job: Callable[..., Any]
    queue_reason_label: Callable[..., Any]
    parse_optional_datetime: Callable[..., Any]
    extract_launch_mode: Callable[..., Any]
    build_policy_compat_fields: Callable[..., Any]
    safe_autonomy_recommendations: tuple[str, ...]
    build_follow_up_actions: Callable[..., Any]
    customer_profile_key: Callable[..., Any]
    build_portfolio_summary: Callable[..., Any]
    build_profile_summary: Callable[..., Any]
    classify_operator_review: Callable[..., Any]
    build_operator_context: Callable[..., Any]
    clean_text_list: Callable[..., Any]


def bind_checkpoint_queue_composer(
    *,
    deps: CheckpointQueueCompositionDependencies | None = None,
    dependencies_factory: Callable[[], CheckpointQueueCompositionDependencies]
    | None = None,
) -> Callable[..., list[AgentCheckpointQueueItemResponse]]:
    """Bind source dependencies once and return the endpoint-facing composer."""
    if deps is None and dependencies_factory is None:
        raise ValueError("Checkpoint queue composition dependencies are required")

    def build(*args: Any, **kwargs: Any) -> list[AgentCheckpointQueueItemResponse]:
        resolved_dependencies = dependencies_factory() if dependencies_factory else deps
        assert resolved_dependencies is not None
        return compose_checkpoint_queue(
            *args,
            **kwargs,
            deps=resolved_dependencies,
        )

    return build


def compose_checkpoint_queue(
    jobs: list[AgentJob],
    inbox_items: list[ResearchInboxItem],
    portfolios: list[ResearchPortfolio] | None = None,
    profiles: list[DomainResearchProfile] | None = None,
    *,
    deps: CheckpointQueueCompositionDependencies,
    learning_profiles: dict[str, dict[str, Any]] | None = None,
    monitor_health_rows: list[dict[str, Any]] | None = None,
    now: datetime | None = None,
) -> list[AgentCheckpointQueueItemResponse]:
    """Project all checkpoint sources and return one stable priority ordering."""
    reference = now or datetime.utcnow()
    items: list[AgentCheckpointQueueItemResponse] = []
    items.extend(
        checkpoint_queue_jobs.build_job_checkpoint_queue_items(
            jobs,
            now=reference,
            deps=checkpoint_queue_jobs.JobCheckpointQueueDependencies(
                extract_approval_checkpoint=deps.extract_approval_checkpoint,
                extract_scheduler_state=deps.extract_scheduler_state,
                queue_customer_for_job=deps.queue_customer_for_job,
                present_job=deps.present_job,
                queue_priority_fields=deps.queue_priority_fields,
                queue_evidence_summary_for_job=deps.queue_evidence_summary_for_job,
                queue_reason_label=deps.queue_reason_label,
                parse_optional_datetime=deps.parse_optional_datetime,
                extract_launch_mode=deps.extract_launch_mode,
            ),
        )
    )
    items.extend(
        checkpoint_queue_monitors.build_monitor_checkpoint_queue_items(
            jobs,
            monitor_health_rows or [],
            now=reference,
            deps=checkpoint_queue_monitors.MonitorCheckpointQueueDependencies(
                queue_customer_for_job=deps.queue_customer_for_job,
                present_job=deps.present_job,
                queue_priority_fields=deps.queue_priority_fields,
                build_policy_compat_fields=deps.build_policy_compat_fields,
                safe_autonomy_recommendations=deps.safe_autonomy_recommendations,
            ),
        )
    )
    items.extend(
        checkpoint_queue_inbox.build_inbox_checkpoint_queue_items(
            inbox_items,
            learning_profiles=learning_profiles or {},
            now=reference,
            deps=checkpoint_queue_inbox.InboxCheckpointQueueDependencies(
                build_follow_up_actions=deps.build_follow_up_actions,
                customer_profile_key=deps.customer_profile_key,
                queue_priority_fields=deps.queue_priority_fields,
                queue_reason_label=deps.queue_reason_label,
            ),
        )
    )
    items.extend(
        checkpoint_queue_portfolios.build_portfolio_checkpoint_queue_items(
            portfolios or [],
            now=reference,
            deps=checkpoint_queue_portfolios.PortfolioCheckpointQueueDependencies(
                build_summary_payload=deps.build_portfolio_summary,
                classify_operator_review=deps.classify_operator_review,
                parse_optional_datetime=deps.parse_optional_datetime,
                queue_priority_fields=deps.queue_priority_fields,
                queue_reason_label=deps.queue_reason_label,
                build_operator_context=deps.build_operator_context,
                clean_text_list=deps.clean_text_list,
            ),
        )
    )
    items.extend(
        checkpoint_queue_profiles.build_profile_checkpoint_queue_items(
            profiles or [],
            now=reference,
            deps=checkpoint_queue_profiles.ProfileCheckpointQueueDependencies(
                build_summary_payload=deps.build_profile_summary,
                classify_operator_review=deps.classify_operator_review,
                parse_optional_datetime=deps.parse_optional_datetime,
                queue_priority_fields=deps.queue_priority_fields,
                queue_reason_label=deps.queue_reason_label,
                build_operator_context=deps.build_operator_context,
            ),
        )
    )
    items.sort(
        key=lambda row: (
            -int(row.priority or 0),
            -float(row.priority_score or 0),
            -(row.created_at.timestamp() if row.created_at else 0.0),
        )
    )
    return items
