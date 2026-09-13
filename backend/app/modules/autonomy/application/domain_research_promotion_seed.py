"""Build reusable profile and portfolio seeds from domain-research jobs."""

from dataclasses import dataclass
from typing import Any, Callable

from app.models.agent_job import AgentJob


class DomainResearchPromotionSeedError(ValueError):
    """Raised when a job cannot seed domain-research promotion."""


@dataclass(frozen=True)
class DomainResearchPromotionSeedDependencies:
    extract_launch_mode: Callable[[dict], str]
    resolve_domain_automation_contract: Callable[..., tuple[str, dict[str, Any]]]
    normalize_portfolio_automation_profile: Callable[..., str]
    resolve_portfolio_automation_policy: Callable[..., dict[str, Any]]


def build_domain_research_promotion_seed(
    job: AgentJob,
    *,
    deps: DomainResearchPromotionSeedDependencies,
) -> dict[str, Any]:
    """Build bounded profile and portfolio creation payloads from a completed job."""
    config = job.config if isinstance(job.config, dict) else {}
    if deps.extract_launch_mode(config) != "quick_start_domain_research":
        raise DomainResearchPromotionSeedError(
            "Job is not a domain research quick start"
        )

    domain = str(config.get("domain") or "").strip()
    objective = str(config.get("objective") or "").strip()
    if not domain or not objective:
        raise DomainResearchPromotionSeedError(
            "Job is missing normalized domain research config"
        )

    automation_profile, automation_policy = deps.resolve_domain_automation_contract(
        automation_profile=config.get("automation_profile"),
        automation_policy=config.get("automation_policy"),
        current_snapshot={"validation_policy": config.get("validation_policy")}
        if isinstance(config.get("validation_policy"), dict)
        else None,
    )
    monitor_queries = _clean_list(config.get("monitor_queries"), limit=12)
    benchmark_queries = _clean_list(config.get("benchmark_queries"), limit=16)
    repo_source_ids = _clean_list(config.get("repo_source_ids"), limit=24)
    title = str(job.name or "").strip()[:200] or f"{domain[:120]} Monitor"
    sandbox_profile_id = str(config.get("sandbox_profile_id") or "").strip() or None

    return {
        "profile": {
            "title": title,
            "domain": domain,
            "objective": objective,
            "customer_context": str(config.get("customer_context") or "").strip()
            or None,
            "source_scope": str(
                config.get("source_scope") or "kb_plus_arxiv_plus_repo"
            ).strip(),
            "track_type": str(config.get("track_type") or "compiler").strip(),
            "research_mode": str(
                config.get("research_mode") or "literature_to_hypothesis"
            ).strip(),
            "monitor_queries": monitor_queries
            or [f"{domain} {objective}".strip()[:240]],
            "repo_source_ids": repo_source_ids or None,
            "benchmark_queries": benchmark_queries or None,
            "report_format": str(
                config.get("report_format") or "brief_and_report"
            ).strip(),
            "scoring_policy": (
                config.get("scoring_policy")
                if isinstance(config.get("scoring_policy"), dict)
                else None
            ),
            "selection_policy": (
                config.get("selection_policy")
                if isinstance(config.get("selection_policy"), dict)
                else None
            ),
            "automation_profile": automation_profile,
            "automation_policy": automation_policy,
            "sandbox_profile_id": sandbox_profile_id,
            "interval_minutes": int(config.get("interval_minutes") or 1440),
            "persist_artifacts": bool(config.get("persist_artifacts", True)),
            "auto_launch_follow_up": bool(
                automation_policy.get(
                    "auto_launch_follow_up",
                    config.get("auto_launch_follow_up", True),
                )
            ),
            "auto_create_experiment_plans": bool(
                automation_policy.get(
                    "auto_create_experiment_plans",
                    config.get("auto_create_experiment_plans", True),
                )
            ),
            "confidence_threshold": float(
                automation_policy.get(
                    "confidence_threshold",
                    config.get("confidence_threshold") or 0.7,
                )
            ),
            "max_documents": int(config.get("max_documents") or 10),
            "max_papers": int(config.get("max_papers") or 8),
            "start_immediately": False,
        },
        "portfolio": {
            "title": f"{domain[:160]} Fleet",
            "objective": objective,
            "sandbox_profile_id": sandbox_profile_id,
            "automation_profile": deps.normalize_portfolio_automation_profile(
                config.get("automation_profile"),
                default="balanced",
            ),
            "automation_policy": deps.resolve_portfolio_automation_policy(
                config.get("automation_profile"),
                config.get("automation_policy"),
            ),
            "start_immediately": False,
        },
    }


def _clean_list(value: Any, *, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()][:limit]
