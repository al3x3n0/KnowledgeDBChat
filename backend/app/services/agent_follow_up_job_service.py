"""Orchestration helpers extracted from AutonomousAgentExecutor."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from uuid import UUID

from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.agent_job import AgentJob, AgentJobStatus
from app.services.autonomy_service import build_domain_profile_compat_policy


class AgentFollowUpJobService:
    async def create_domain_research_follow_up_job(
        self,
        executor: Any,
        *,
        db: AsyncSession,
        job: AgentJob,
        domain: str,
        objective: str,
        customer_context: str,
        track_type: str,
        source_scope: str,
        top_idea: dict[str, Any],
        docs: list[dict[str, Any]],
        repo_documents: list[dict[str, Any]],
        papers: list[dict[str, Any]],
        repo_source_ids: list[str],
        benchmark_queries: list[str],
        automation_profile: Optional[str] = None,
        automation_policy: Optional[dict[str, Any]] = None,
        sandbox_profile_id: Optional[str] = None,
        profile_id: Optional[str] = None,
    ) -> Optional[AgentJob]:
        idea_title = str(top_idea.get("title") or "").strip() or domain
        idea_hypothesis = str(top_idea.get("hypothesis") or "").strip()
        normalized_automation_profile = str(automation_profile or "").strip().lower() or "balanced"
        normalized_automation_policy = dict(automation_policy) if isinstance(automation_policy, dict) else {}
        normalized_sandbox_profile_id = str(sandbox_profile_id or "").strip() or None
        normalized_profile_id = str(profile_id or "").strip() or None
        child = AgentJob(
            name=f"Domain Research Deep Dive - {idea_title[:80]}",
            description="Auto-launched deep-dive follow-up from domain research.",
            job_type="research",
            goal=(
                f"Deep-dive the strongest domain research idea for {domain}. "
                f"Objective: {objective}\nTop idea: {idea_title}\nHypothesis: {idea_hypothesis}"
            ).strip(),
            config={
                "launch_mode": "domain_research_follow_up",
                "follow_up_origin_job_id": str(job.id),
                "domain": domain,
                "objective": objective,
                "customer_context": customer_context,
                "track_type": track_type,
                "source_scope": source_scope,
                "profile_id": normalized_profile_id,
                "search_query": f"{domain} {idea_title}".strip()[:500],
                "monitor_queries": [idea_title][:4],
                "benchmark_queries": benchmark_queries[:8],
                "repo_source_ids": repo_source_ids[:24],
                "prefer_sources": ["documents", "arxiv", "repo"] if source_scope == "kb_plus_arxiv_plus_repo" else ["documents", "arxiv"],
                "max_documents": max(4, min(len(docs), 8)) or 6,
                "max_repo_documents": max(2, min(len(repo_documents), 8)) if repo_documents else 0,
                "max_papers": max(2, min(len(papers), 6)) or 4,
                "sandbox_profile_id": normalized_sandbox_profile_id,
                "automation_profile": normalized_automation_profile,
                "automation_policy": normalized_automation_policy,
                "validation_policy": build_domain_profile_compat_policy(normalized_automation_policy),
                "domain_research_follow_up": {
                    "idea": top_idea,
                    "parent_job_id": str(job.id),
                    "automation_profile": normalized_automation_profile,
                    "sandbox_profile_id": normalized_sandbox_profile_id,
                },
            },
            user_id=job.user_id,
            status=AgentJobStatus.PENDING.value,
            parent_job_id=job.id,
            root_job_id=job.root_job_id or job.id,
            chain_depth=int(job.chain_depth or 0) + 1,
            enable_memory=job.enable_memory,
            max_iterations=job.max_iterations,
            max_tool_calls=job.max_tool_calls,
            max_llm_calls=job.max_llm_calls,
            max_runtime_minutes=job.max_runtime_minutes,
        )
        db.add(child)
        await db.flush()
        return child
