"""Project coding-swarm execution results into compact API summaries."""

from typing import Any

from app.models.agent_job import AgentJob
from app.models.user import User
from app.services.agent_swarm_collaboration_service import extract_swarm_collaboration
from app.services.collaboration_service import build_collaboration_summary


def extract_swarm_summary(
    job: AgentJob,
    *,
    current_user_id: str | None = None,
    user_lookup: dict[str, User] | None = None,
) -> dict[str, Any] | None:
    """Build a compact swarm/fan-in summary for API consumers."""
    results = job.results if isinstance(job.results, dict) else {}
    execution_strategy = (
        results.get("execution_strategy")
        if isinstance(results.get("execution_strategy"), dict)
        else {}
    )
    swarm_execution = (
        execution_strategy.get("swarm")
        if isinstance(execution_strategy.get("swarm"), dict)
        else {}
    )
    fan_in = (
        results.get("swarm_fan_in")
        if isinstance(results.get("swarm_fan_in"), dict)
        else {}
    )
    config = job.config if isinstance(job.config, dict) else {}
    enabled = bool(
        config.get("swarm_child_jobs_enabled", False)
        or swarm_execution.get("enabled", False)
    )
    configured = bool(swarm_execution.get("configured", False) or swarm_execution)
    fan_in_enabled = bool(swarm_execution.get("fan_in_enabled", False))

    expected_siblings = int(fan_in.get("expected_siblings", 0) or 0)
    received_siblings = int(fan_in.get("received_siblings", 0) or 0)
    terminal_siblings = int(fan_in.get("terminal_siblings", 0) or 0)
    if expected_siblings <= 0:
        expected_siblings = int(swarm_execution.get("child_jobs_count", 0) or 0)
    if received_siblings <= 0 and expected_siblings > 0:
        received_siblings = expected_siblings
    if terminal_siblings <= 0 and received_siblings > 0:
        terminal_siblings = received_siblings

    roles: list[str] = []
    raw_roles = fan_in.get("roles")
    if isinstance(raw_roles, list) and raw_roles:
        roles = [str(role).strip() for role in raw_roles if str(role).strip()][:20]
    elif isinstance(swarm_execution.get("roles_assigned"), list):
        roles = [
            str(role).strip()
            for role in swarm_execution.get("roles_assigned", [])
            if str(role).strip()
        ][:20]

    confidence = (
        fan_in.get("confidence") if isinstance(fan_in.get("confidence"), dict) else {}
    )
    consensus_rows = (
        fan_in.get("consensus_findings")
        if isinstance(fan_in.get("consensus_findings"), list)
        else []
    )
    consensus_findings = [
        str(row.get("finding") or "").strip()[:280]
        for row in consensus_rows
        if isinstance(row, dict) and str(row.get("finding") or "").strip()
    ][:10]
    conflicts = (
        fan_in.get("conflicts") if isinstance(fan_in.get("conflicts"), list) else []
    )
    action_plan = (
        fan_in.get("action_plan") if isinstance(fan_in.get("action_plan"), list) else []
    )
    collaboration = extract_swarm_collaboration(job)

    if not any([enabled, configured, fan_in, swarm_execution]):
        return None

    return {
        "enabled": enabled,
        "configured": configured,
        "fan_in_enabled": fan_in_enabled,
        "fan_in_group_id": str(
            fan_in.get("fan_in_group_id")
            or swarm_execution.get("fan_in_group_id")
            or ""
        ).strip(),
        "roles": roles,
        "role_count": len(roles),
        "expected_siblings": expected_siblings,
        "received_siblings": received_siblings,
        "terminal_siblings": terminal_siblings,
        "consensus_count": len(consensus_rows),
        "consensus_findings": consensus_findings,
        "conflict_count": len(conflicts),
        "conflicts": conflicts[:10],
        "action_plan": action_plan[:10],
        "confidence": confidence,
        "winning_slice_id": str(fan_in.get("winning_slice_id") or "").strip() or None,
        "winning_role": str(fan_in.get("winning_role") or "").strip() or None,
        "promotion_reason": str(fan_in.get("promotion_reason") or "").strip() or None,
        "review_state": str(fan_in.get("review_state") or "").strip() or None,
        "review_reason": str(fan_in.get("review_reason") or "").strip() or None,
        "review_required": bool(fan_in.get("review_required", False)),
        "tie_breaker_attempted": bool(fan_in.get("tie_breaker_attempted", False)),
        "tie_breaker_job_id": str(fan_in.get("tie_breaker_job_id") or "").strip()
        or None,
        "tie_breaker_source_job_id": str(
            fan_in.get("tie_breaker_source_job_id") or ""
        ).strip()
        or None,
        "file_converged": bool(fan_in.get("file_converged", False)),
        "file_convergence_support": int(fan_in.get("file_convergence_support", 0) or 0),
        "top_file_cluster": fan_in.get("top_file_cluster")
        if isinstance(fan_in.get("top_file_cluster"), dict)
        else None,
        "command_converged": bool(fan_in.get("command_converged", False)),
        "command_convergence_support": int(
            fan_in.get("command_convergence_support", 0) or 0
        ),
        "top_command_cluster": fan_in.get("top_command_cluster")
        if isinstance(fan_in.get("top_command_cluster"), dict)
        else None,
        "repair_chain_job_id": str(fan_in.get("repair_chain_job_id") or "").strip()
        or None,
        "candidate_paths": fan_in.get("candidate_paths")[:10]
        if isinstance(fan_in.get("candidate_paths"), list)
        else [],
        "recommended_commands": fan_in.get("recommended_commands")[:10]
        if isinstance(fan_in.get("recommended_commands"), list)
        else [],
        "owner_user_id": str(collaboration.get("owner_user_id") or job.user_id),
        "shared_review": bool(collaboration.get("shared_review")),
        "shared_with_user_ids": collaboration.get("shared_with_user_ids") or [],
        "assigned_user_id": collaboration.get("assigned_user_id"),
        "assigned_at": collaboration.get("assigned_at"),
        "assigned_by_user_id": collaboration.get("assigned_by_user_id"),
        "review_note": collaboration.get("review_note"),
        "collaboration_summary": build_collaboration_summary(
            owner_user_id=str(collaboration.get("owner_user_id") or job.user_id),
            visibility=(
                "shared" if collaboration.get("shared_with_user_ids") else "private"
            ),
            shared_with_user_ids=list(collaboration.get("shared_with_user_ids") or []),
            assigned_user_id=str(collaboration.get("assigned_user_id") or "").strip()
            or None,
            assigned_by_user_id=str(
                collaboration.get("assigned_by_user_id") or ""
            ).strip()
            or None,
            assigned_at=str(collaboration.get("assigned_at") or "").strip() or None,
            note=str(collaboration.get("review_note") or "").strip() or None,
            current_user_id=current_user_id,
            user_lookup=user_lookup,
        ),
    }
