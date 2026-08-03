"""Project scientific validation runs into autonomy decision events."""

from dataclasses import dataclass
from typing import Any, Callable

from app.models.experiment import ExperimentRun
from app.schemas.agent_job import AgentDecisionTraceEventResponse


@dataclass(frozen=True)
class ValidationDecisionTraceDependencies:
    parse_time: Callable[..., Any]
    reason_label: Callable[..., Any]
    build_event: Callable[..., AgentDecisionTraceEventResponse]
    build_operator_context: Callable[..., dict[str, Any]]


def build_validation_decision_trace(
    runs: list[ExperimentRun],
    *,
    deps: ValidationDecisionTraceDependencies,
) -> list[AgentDecisionTraceEventResponse]:
    events: list[AgentDecisionTraceEventResponse] = []
    for run in runs:
        config = run.config if isinstance(run.config, dict) else {}
        validation = _dict(config.get("scientific_validation"))
        handoff = _dict(config.get("execution_handoff"))
        origin = _dict(handoff.get("autonomous_origin"))
        profile_snapshot = _dict(validation.get("profile_snapshot"))
        blocked_reason = str(
            validation.get("blocked_reason_code")
            or validation.get("blocked_reason")
            or ""
        ).strip()
        operator_actions = (
            validation.get("operator_actions")
            if isinstance(validation.get("operator_actions"), list)
            else []
        )
        hypothesis_id = (
            str(
                validation.get("hypothesis_id") or origin.get("opportunity_id") or ""
            ).strip()
            or None
        )
        profile_id = (
            str(validation.get("domain_research_profile_id") or "").strip() or None
        )
        portfolio_id = (
            str(validation.get("research_portfolio_id") or "").strip() or None
        )
        deep_link = _validation_link(
            run,
            profile_id=profile_id,
            portfolio_id=portfolio_id,
            hypothesis_id=hypothesis_id,
        )
        operator_context = deps.build_operator_context(
            objective=str(
                validation.get("decision_summary") or run.summary or ""
            ).strip()
            or None,
            domain=str(profile_snapshot.get("domain") or "").strip() or None,
            track_type=str(validation.get("track_type") or "").strip() or None,
            source_scope=str(validation.get("source_scope") or "").strip() or None,
            repo_source_ids=validation.get("repo_source_ids"),
            benchmark_queries=validation.get("benchmark_queries"),
            sandbox_profile_id=str(validation.get("sandbox_profile_id") or "").strip()
            or None,
            automation_profile=str(validation.get("automation_profile") or "").strip()
            or None,
            effective_policy=(
                validation.get("effective_policy")
                if isinstance(validation.get("effective_policy"), dict)
                else None
            ),
            confidence=validation.get("confidence"),
            readiness=validation.get("readiness"),
            linked_experiment_plan_ids=(
                [str(run.experiment_plan_id)] if run.experiment_plan_id else None
            ),
            linked_validation_run_ids=[str(run.id)] if run.id else None,
            child_job_ids=[str(run.agent_job_id)] if run.agent_job_id else None,
        )
        if blocked_reason:
            events.append(
                deps.build_event(
                    event_type="validation_blocked",
                    event_time=run.updated_at or run.created_at,
                    source_kind="validation_run",
                    source_id=str(run.id) if run.id else None,
                    source_label=run.name,
                    decision_type="validation_blocked",
                    reason_code=blocked_reason,
                    reason_label=deps.reason_label(blocked_reason),
                    status=run.status,
                    severity="high",
                    actor_mode="autonomous",
                    summary=f"{run.name}: validation blocked",
                    deep_link=deep_link,
                    metadata={
                        "experiment_plan_id": str(run.experiment_plan_id)
                        if run.experiment_plan_id
                        else None,
                        "opportunity_id": hypothesis_id,
                    },
                    suffix="blocked",
                    operator_context=operator_context,
                )
            )
        events.extend(
            _operator_action_events(
                run,
                operator_actions,
                hypothesis_id=hypothesis_id,
                deep_link=deep_link,
                operator_context=operator_context,
                deps=deps,
            )
        )
    return events


def _operator_action_events(
    run: ExperimentRun,
    actions: list[Any],
    *,
    hypothesis_id: str | None,
    deep_link: dict[str, Any],
    operator_context: dict[str, Any],
    deps: ValidationDecisionTraceDependencies,
) -> list[AgentDecisionTraceEventResponse]:
    events = []
    for index, action in enumerate(actions):
        if not isinstance(action, dict):
            continue
        event_time = deps.parse_time(
            action.get("at"),
            fallback=run.updated_at or run.created_at,
        )
        if event_time is None:
            continue
        action_name = str(action.get("action") or "operator_action").strip().lower()
        event_type = (
            "validation_requeued"
            if action_name in {"requeue", "retry", "restart"}
            else "validation_operator_action"
        )
        reason = str(action.get("outcome_status") or action_name or "").strip() or None
        events.append(
            deps.build_event(
                event_type=event_type,
                event_time=event_time,
                source_kind="validation_run",
                source_id=str(run.id) if run.id else None,
                source_label=run.name,
                decision_type=event_type,
                reason_code=reason,
                reason_label=deps.reason_label(reason),
                status=str(action.get("new_status") or run.status or "").strip()
                or None,
                severity="medium",
                actor_mode="operator",
                summary=f"{run.name}: {action_name.replace('_', ' ')}",
                operator_note=str(action.get("note") or "").strip() or None,
                before_state=(
                    {"status": action.get("previous_status")}
                    if action.get("previous_status")
                    else None
                ),
                after_state=(
                    {"status": action.get("new_status")}
                    if action.get("new_status")
                    else None
                ),
                deep_link=deep_link,
                metadata={
                    "linked_job_id": action.get("linked_job_id"),
                    "experiment_plan_id": str(run.experiment_plan_id)
                    if run.experiment_plan_id
                    else None,
                    "opportunity_id": hypothesis_id,
                },
                suffix=str(index),
                operator_context=operator_context,
            )
        )
    return events


def _validation_link(
    run: ExperimentRun,
    *,
    profile_id: str | None,
    portfolio_id: str | None,
    hypothesis_id: str | None,
) -> dict[str, Any]:
    job_params = {"job": str(run.agent_job_id)} if run.agent_job_id else {}
    if profile_id:
        return {
            "target_tab": "domain",
            "job_id": run.agent_job_id,
            "params": {
                "tab": "domain",
                "profileId": profile_id,
                **({"opportunityId": hypothesis_id} if hypothesis_id else {}),
                **job_params,
            },
            "label": "Open Domain",
        }
    if portfolio_id:
        return {
            "target_tab": "fleet",
            "job_id": run.agent_job_id,
            "params": {
                "tab": "fleet",
                "fleetId": portfolio_id,
                **({"opportunityId": hypothesis_id} if hypothesis_id else {}),
                **job_params,
            },
            "label": "Open Fleet",
        }
    return {
        "target_tab": "jobs",
        "job_id": run.agent_job_id,
        "params": job_params,
        "label": "Open Validation Job",
    }


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}
