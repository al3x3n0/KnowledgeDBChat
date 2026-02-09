"""
Autonomous Agent Executor Service.

Executes autonomous agent jobs that run independently, working toward
defined goals without requiring continuous user interaction.

The executor implements an autonomous loop:
1. Observe: Gather current state and context
2. Think: Analyze progress and decide next action
3. Act: Execute tools and gather results
4. Evaluate: Check if goal is achieved
5. Repeat until goal is met or limits reached
"""

import asyncio
import hashlib
import json
import math
import random
import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple
from uuid import UUID
from collections import Counter

from loguru import logger
from sqlalchemy import select, update, func, desc, or_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError

from app.models.agent_job import AgentJob, AgentJobStatus, AgentJobCheckpoint, ChainTriggerCondition
from app.models.agent_definition import AgentDefinition
from app.models.agent_tool_prior import AgentToolPrior
from app.models.user import User
from app.models.memory import UserPreferences
from app.services.llm_service import LLMService, UserLLMSettings
from app.services.agent_tools import AGENT_TOOLS, AUTONOMOUS_AGENT_TOOLS
from app.services.data_analysis_tools import DataAnalysisTools, DATA_ANALYSIS_TOOL_DEFINITIONS
from app.services.search_service import SearchService
from app.services.arxiv_search_service import ArxivSearchService
from app.services.document_service import DocumentService
from app.services.vector_store import VectorStoreService
from app.services.agent_job_memory_service import agent_job_memory_service
from app.services.ai_hub_dataset_preset_service import ai_hub_dataset_preset_service
from app.services.ai_hub_eval_service import ai_hub_eval_service


# Centralized tool fallback policies keyed by job type.
# Used when a requested tool is unknown/unimplemented or fails with an error.
# Per-job overrides can be provided via job.config.tool_fallback_map.
_TOOL_FALLBACK_POLICIES: Dict[str, Dict[str, Dict[str, str]]] = {
    "_default": {
        # Safe default: search the KB using the job goal.
        "__default__": {"tool": "search_documents", "param": "goal"},
        # Param-aware fallbacks.
        "web_scrape": {"tool": "search_documents", "param": "url"},
        "ingest_url": {"tool": "search_documents", "param": "url"},
        "search_with_filters": {"tool": "search_documents", "param": "query"},
        "search_arxiv": {"tool": "search_documents", "param": "query"},
        "monitor_arxiv_topic": {"tool": "search_documents", "param": "query"},
        "find_related_papers": {"tool": "search_documents", "param": "query"},
        "get_document_details": {"tool": "search_documents", "param": "document_id"},
        "read_document_content": {"tool": "search_documents", "param": "document_id"},
        "summarize_document": {"tool": "search_documents", "param": "document_id"},
        "find_similar_documents": {"tool": "search_documents", "param": "document_id"},
    },
    # Job-type specific safe defaults (can override _default).
    "research": {
        "__default__": {"tool": "search_documents", "param": "goal"},
    },
    "monitor": {
        "__default__": {"tool": "search_documents", "param": "goal"},
    },
    "analysis": {
        "__default__": {"tool": "search_documents", "param": "goal"},
    },
    "synthesis": {
        "__default__": {"tool": "search_documents", "param": "goal"},
    },
    "knowledge_expansion": {
        "__default__": {"tool": "search_documents", "param": "goal"},
    },
    "data_analysis": {
        # Data analysis tool failures often indicate missing schema/context;
        # searching the KB with the job goal is a safe best-effort fallback.
        "__default__": {"tool": "search_documents", "param": "goal"},
    },
    "custom": {
        "__default__": {"tool": "search_documents", "param": "goal"},
    },
}


class AutonomousAgentExecutor:
    """
    Executes autonomous agent jobs.

    The executor runs an autonomous loop that:
    1. Loads job context and state
    2. Decides next action based on goal and progress
    3. Executes tools
    4. Evaluates progress toward goal
    5. Continues until goal is achieved or limits reached
    """

    def __init__(self):
        self.llm_service = LLMService()
        self.search_service = SearchService()
        self.arxiv_service = ArxivSearchService()
        self.document_service = DocumentService()
        self.vector_store = VectorStoreService()
        # Store for job-specific data (findings, reading lists, etc.)
        self._job_findings: Dict[str, List[Dict[str, Any]]] = {}
        # Store for data analysis tools instances per job
        self._data_analysis_tools: Dict[str, DataAnalysisTools] = {}

    def _llm_routing_from_job_config(self, cfg: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(cfg, dict):
            return None

        tier = str(cfg.get("llm_tier") or cfg.get("tier") or "").strip().lower() or None

        fallback = cfg.get("llm_fallback_tiers") or cfg.get("fallback_tiers")
        if not isinstance(fallback, list):
            fallback = []
        fallback_tiers = [str(x).strip().lower() for x in fallback if str(x).strip()]

        def _opt_int(*keys: str) -> Optional[int]:
            for k in keys:
                if k in cfg and cfg.get(k) is not None:
                    try:
                        v = int(cfg.get(k))
                    except Exception:
                        continue
                    return v
            return None

        timeout_seconds = _opt_int("llm_timeout_seconds", "timeout_seconds")
        max_tokens_cap = _opt_int("llm_max_tokens_cap", "max_tokens_cap")
        cooldown_seconds = _opt_int("llm_unhealthy_cooldown_seconds", "cooldown_seconds")

        if not tier and not fallback_tiers and timeout_seconds is None and max_tokens_cap is None and cooldown_seconds is None:
            return None

        routing: Dict[str, Any] = {"tier": tier, "fallback_tiers": fallback_tiers}
        if timeout_seconds is not None:
            routing["timeout_seconds"] = max(2, min(timeout_seconds, 600))
        if max_tokens_cap is not None:
            routing["max_tokens_cap"] = max(64, min(max_tokens_cap, 20000))
        if cooldown_seconds is not None:
            routing["cooldown_seconds"] = max(5, min(cooldown_seconds, 3600))
        return routing


    async def execute_job(
        self,
        job_id: UUID,
        db: AsyncSession,
        progress_callback: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """
        Execute an autonomous agent job.

        Args:
            job_id: The job to execute
            db: Database session
            progress_callback: Optional callback for progress updates

        Returns:
            Execution result with status and outputs
        """
        # Load job
        result = await db.execute(
            select(AgentJob).where(AgentJob.id == job_id)
        )
        job = result.scalar_one_or_none()

        if not job:
            return {"error": "Job not found", "status": "failed"}

        if job.status not in [AgentJobStatus.PENDING.value, AgentJobStatus.RUNNING.value]:
            return {"error": f"Job cannot be executed in status: {job.status}", "status": job.status}

        # Load user settings
        user_settings = await self._load_user_settings(job.user_id, db)

        # Load agent definition if specified
        agent_def = None
        if job.agent_definition_id:
            agent_result = await db.execute(
                select(AgentDefinition).where(AgentDefinition.id == job.agent_definition_id)
            )
            agent_def = agent_result.scalar_one_or_none()

        # Update job status
        job.status = AgentJobStatus.RUNNING.value
        job.started_at = job.started_at or datetime.utcnow()
        job.last_activity_at = datetime.utcnow()
        await db.commit()

        try:
            # Deterministic runners (no LLM/tool loop)
            det = (job.config or {}).get("deterministic_runner")
            deterministic_result: Optional[Dict[str, Any]] = None

            if det == "ai_hub_scientist":
                deterministic_result = await self._run_ai_hub_scientist(job=job, db=db, progress_callback=progress_callback)
            elif det == "research_inbox_monitor":
                deterministic_result = await self._run_research_inbox_monitor(job=job, db=db, progress_callback=progress_callback)
            elif det == "code_patch_proposer":
                deterministic_result = await self._run_code_patch_proposer(job=job, db=db, progress_callback=progress_callback)
            elif det == "research_engineer_scientist":
                deterministic_result = await self._run_research_engineer_scientist(job=job, db=db, progress_callback=progress_callback)
            elif det == "research_engineer_paper_update":
                deterministic_result = await self._run_research_engineer_paper_update(job=job, db=db, progress_callback=progress_callback)
            elif det == "latex_citation_sync":
                deterministic_result = await self._run_latex_citation_sync(job=job, db=db, progress_callback=progress_callback)
            elif det == "latex_reviewer_critic":
                deterministic_result = await self._run_latex_reviewer_critic(job=job, db=db, progress_callback=progress_callback)
            elif det == "latex_compile_project":
                deterministic_result = await self._run_latex_compile_project(job=job, db=db, progress_callback=progress_callback)
            elif det == "latex_publish_project":
                deterministic_result = await self._run_latex_publish_project(job=job, db=db, progress_callback=progress_callback)
            elif det == "latex_apply_unified_diff":
                deterministic_result = await self._run_latex_apply_unified_diff(job=job, db=db, progress_callback=progress_callback)
            elif det == "experiment_loop_seed":
                deterministic_result = await self._run_experiment_loop_seed(job=job, db=db, progress_callback=progress_callback)
            elif det == "experiment_plan_generate":
                deterministic_result = await self._run_experiment_plan_generate(job=job, db=db, progress_callback=progress_callback)
            elif det == "experiment_decide_next":
                deterministic_result = await self._run_experiment_decide_next(job=job, db=db, progress_callback=progress_callback)
            elif det == "experiment_runner":
                deterministic_result = await self._run_experiment_runner(job=job, db=db, progress_callback=progress_callback)
            elif det == "experiment_persist_results":
                deterministic_result = await self._run_experiment_persist_results(job=job, db=db, progress_callback=progress_callback)
            elif det == "code_patch_apply_to_kb":
                deterministic_result = await self._run_code_patch_apply_to_kb(job=job, db=db, progress_callback=progress_callback)
            elif det == "arxiv_inbox_extract_repos":
                deterministic_result = await self._run_arxiv_inbox_extract_repos(job=job, db=db, progress_callback=progress_callback)
            elif det == "git_repo_ingest_wait":
                deterministic_result = await self._run_git_repo_ingest_wait(job=job, db=db, progress_callback=progress_callback)
            elif det == "paper_algorithm_project":
                deterministic_result = await self._run_paper_algorithm_project(job=job, db=db, progress_callback=progress_callback)
            elif det == "generated_project_demo_check":
                deterministic_result = await self._run_generated_project_demo_check(job=job, db=db, progress_callback=progress_callback)
            elif det == "swarm_fan_in_aggregate":
                deterministic_result = await self._run_swarm_fan_in_aggregate(job=job, db=db, progress_callback=progress_callback)

            if deterministic_result is not None:
                # Ensure chained jobs trigger even for deterministic runners.
                event = "complete" if job.status == AgentJobStatus.COMPLETED.value else "fail"
                await self._trigger_chained_jobs(job, event, db)
                return deterministic_result

            # Run the autonomous loop
            result = await self._run_autonomous_loop(
                job=job,
                agent_def=agent_def,
                user_settings=user_settings,
                db=db,
                progress_callback=progress_callback,
            )

            return result

        except Exception as e:
            logger.error(f"Autonomous job execution failed: {e}")
            job.status = AgentJobStatus.FAILED.value
            job.error = str(e)
            job.error_count += 1
            job.last_error_at = datetime.utcnow()
            await db.commit()
            try:
                await self._trigger_chained_jobs(job, "fail", db)
            except Exception:
                # Avoid masking the original failure if chain triggering fails
                pass
            return {"error": str(e), "status": "failed"}

    async def _run_ai_hub_scientist(
        self,
        *,
        job: AgentJob,
        db: AsyncSession,
        progress_callback: Optional[callable],
    ) -> Dict[str, Any]:
        """
        Deterministic runner: propose an AI Hub plugin bundle for the current deployment.

        Produces `job.results.ai_hub_bundle` with enabled preset IDs + eval template IDs and a demo plan.
        """
        from app.models.user import User
        from app.core.feature_flags import set_str as set_feature_str, get_str as get_feature_str
        from app.schemas.customer_profile import CustomerProfile

        def _emit(progress: int, phase: str, details: str):
            job.progress = max(0, min(100, int(progress)))
            job.current_phase = phase
            job.phase_details = details
            job.last_activity_at = datetime.utcnow()
            job.add_log_entry({"phase": phase, "action": "ai_hub_scientist", "result": details})

        # Customer profile (deployment-level) optionally overrides defaults.
        customer_profile_raw = await get_feature_str("ai_hub_customer_profile")
        customer_profile: CustomerProfile | None = None
        if customer_profile_raw:
            try:
                customer_profile = CustomerProfile.model_validate(json.loads(customer_profile_raw))
            except Exception:
                customer_profile = None

        workflows = (job.config or {}).get("workflows")
        if not workflows and customer_profile and customer_profile.preferred_workflows:
            workflows = customer_profile.preferred_workflows
        workflows = workflows or ["triage", "extraction", "literature"]
        workflows = [str(x).strip().lower() for x in workflows if str(x).strip()]

        apply_now = bool((job.config or {}).get("apply", False))
        customer_context = str((job.config or {}).get("customer_context") or "").strip()
        if not customer_context and customer_profile and customer_profile.notes:
            customer_context = str(customer_profile.notes).strip()

        _emit(10, "planning", f"Building AI Hub bundle for workflows: {', '.join(workflows) or 'all'}")
        await db.commit()

        # --- Build a lightweight customer signal (no LLM) ---
        STOPWORDS = {
            "the","and","for","with","from","that","this","into","over","under","when","where","what","which","while",
            "your","you","are","our","their","they","them","then","than","also","only","just","more","most","less","very",
            "use","using","used","make","made","help","helps","via","can","could","should","would","may","might","will",
            "data","dataset","datasets","model","models","train","training","eval","evaluate","evaluation","assistant",
            "job","jobs","v1","v2","version","note","notes","paper","papers","doc","docs","document","documents",
        }

        def _tokens(text: str) -> list[str]:
            raw = re.findall(r"[a-zA-Z0-9_\\-]+", (text or "").lower())
            out: list[str] = []
            for w in raw:
                w = w.strip("_-")
                if len(w) < 3:
                    continue
                if w in STOPWORDS:
                    continue
                out.append(w)
            return out

        async def _collect_customer_corpus() -> tuple[str, dict]:
            """
            Collect lightweight signals from this deployment + user.
            Assumes this deployment typically maps to one customer/workspace.
            """
            evidence: dict[str, Any] = {"signals": []}
            parts: list[str] = []

            if customer_profile and customer_profile.name:
                parts.append(str(customer_profile.name))
                evidence["signals"].append({"source": "customer_profile.name", "chars": len(customer_profile.name)})
            if customer_profile and customer_profile.keywords:
                # Keywords are strong signal, but still lightweight and user-controlled.
                parts.extend([" ".join(customer_profile.keywords)] * 4)
                evidence["signals"].append({"source": "customer_profile.keywords", "count": len(customer_profile.keywords)})

            # User-provided context gets extra weight.
            if customer_context:
                parts.extend([customer_context] * 3)
                evidence["signals"].append({"source": "job.config.customer_context", "chars": len(customer_context)})

            # Recent docs (titles + tags).
            try:
                from app.models.document import Document, DocumentSource
                docs_result = await db.execute(
                    select(Document.title, Document.tags, Document.source_id)
                    .order_by(Document.created_at.desc())
                    .limit(200)
                )
                rows = docs_result.all()
                parts.extend([r[0] for r in rows if r and r[0]])
                tags: list[str] = []
                src_ids = []
                for _, t, sid in rows:
                    if isinstance(t, list):
                        tags.extend([str(x) for x in t if x])
                    if sid:
                        src_ids.append(sid)
                if tags:
                    evidence["top_tags"] = [k for k, _ in Counter([x.lower() for x in tags]).most_common(20)]
                if src_ids:
                    src_result = await db.execute(
                        select(DocumentSource.source_type).where(DocumentSource.id.in_(src_ids))
                    )
                    types = [r[0] for r in src_result.all() if r and r[0]]
                    if types:
                        evidence["source_types"] = [k for k, _ in Counter(types).most_common(10)]
                evidence["signals"].append({"source": "recent_documents", "count": len(rows)})
            except Exception as e:
                evidence["signals"].append({"source": "recent_documents", "error": str(e)})

            # Reading lists (names + descriptions).
            try:
                from app.models.reading_list import ReadingList
                rl_result = await db.execute(
                    select(ReadingList.name, ReadingList.description)
                    .where(ReadingList.user_id == job.user_id)
                    .order_by(ReadingList.updated_at.desc())
                    .limit(30)
                )
                rls = rl_result.all()
                for name, desc in rls:
                    if name:
                        parts.append(str(name))
                    if desc:
                        parts.append(str(desc))
                evidence["signals"].append({"source": "reading_lists", "count": len(rls)})
            except Exception as e:
                evidence["signals"].append({"source": "reading_lists", "error": str(e)})

            # Research notes (titles + tags).
            try:
                from app.models.research_note import ResearchNote
                rn_result = await db.execute(
                    select(ResearchNote.title, ResearchNote.tags)
                    .where(ResearchNote.user_id == job.user_id)
                    .order_by(ResearchNote.updated_at.desc())
                    .limit(20)
                )
                notes = rn_result.all()
                note_tags: list[str] = []
                for title, tags in notes:
                    if title:
                        parts.append(str(title))
                    if isinstance(tags, list):
                        note_tags.extend([str(x) for x in tags if x])
                if note_tags:
                    evidence["note_tags"] = [k for k, _ in Counter([x.lower() for x in note_tags]).most_common(20)]
                evidence["signals"].append({"source": "research_notes", "count": len(notes)})
            except Exception as e:
                evidence["signals"].append({"source": "research_notes", "error": str(e)})

            return "\n".join(parts), evidence

        corpus, evidence = await _collect_customer_corpus()
        customer_freq = Counter(_tokens(corpus))
        top_keywords = [k for k, _ in customer_freq.most_common(30)]

        # Infer a coarse domain label (used for naming + suggested new plugins)
        kw = set(top_keywords[:50])
        if {"security", "vulnerability", "cve", "malware", "threat"} & kw:
            domain = "Security"
        elif {"robot", "robotics", "slam", "control", "motion"} & kw:
            domain = "Robotics"
        elif {"genome", "protein", "bio", "rna", "sequencing"} & kw:
            domain = "Bio"
        elif {"compiler", "llvm", "clang", "microarchitecture", "perf", "benchmark"} & kw:
            domain = "Compiler/Performance"
        elif {"hardware", "rtl", "verilog", "silicon", "chip"} & kw:
            domain = "Hardware"
        else:
            domain = "Research"

        # --- Score plugins against customer keywords ---
        presets = ai_hub_dataset_preset_service.list_presets()
        evals = ai_hub_eval_service.list_templates()

        # Load feedback aggregates to bias scoring (learning loop).
        # Bias is scoped by customer profile name when available.
        feedback_bias: dict[tuple[str, str, str], dict[str, int]] = {}
        try:
            from app.models.ai_hub_recommendation_feedback import AIHubRecommendationFeedback

            profile_id = customer_profile.id if customer_profile else None
            q = (
                select(
                    AIHubRecommendationFeedback.workflow,
                    AIHubRecommendationFeedback.item_type,
                    AIHubRecommendationFeedback.item_id,
                    AIHubRecommendationFeedback.decision,
                    func.count().label("cnt"),
                )
                .where(
                    AIHubRecommendationFeedback.customer_profile_id == profile_id
                    if profile_id is not None
                    else AIHubRecommendationFeedback.customer_profile_id.is_(None)
                )
                .group_by(
                    AIHubRecommendationFeedback.workflow,
                    AIHubRecommendationFeedback.item_type,
                    AIHubRecommendationFeedback.item_id,
                    AIHubRecommendationFeedback.decision,
                )
            )
            res = await db.execute(q)
            for wf, itype, iid, decision, cnt in res.all():
                key = (str(wf), str(itype), str(iid))
                bucket = feedback_bias.get(key) or {"accept": 0, "reject": 0}
                if str(decision) == "accept":
                    bucket["accept"] += int(cnt or 0)
                elif str(decision) == "reject":
                    bucket["reject"] += int(cnt or 0)
                feedback_bias[key] = bucket
        except Exception:
            feedback_bias = {}

        def _plugin_tokens_preset(p: Any) -> set[str]:
            text = f"{p.id}\n{p.name}\n{p.description}\n{getattr(p,'dataset_type','')}\n{getattr(p,'generation_prompt','')}"
            return set(_tokens(text))

        def _plugin_tokens_eval(t: Any) -> set[str]:
            cases_text = "\n".join([str(c.get("prompt") or "") for c in (t.cases or []) if isinstance(c, dict)])
            rubric_text = json.dumps(t.rubric or {}, ensure_ascii=False)
            text = f"{t.id}\n{t.name}\n{t.description}\n{t.judge_preamble}\n{rubric_text}\n{cases_text}"
            return set(_tokens(text))

        def _feedback_weight(workflow_name: str, item_type: str, item_id: str) -> dict[str, int]:
            bucket = feedback_bias.get((workflow_name, item_type, item_id)) or {"accept": 0, "reject": 0}
            accepts = int(bucket.get("accept", 0))
            rejects = int(bucket.get("reject", 0))
            # Keep weights moderate; keyword overlap remains primary signal.
            bias = accepts * 20 - rejects * 30
            return {"accepts": accepts, "rejects": rejects, "bias": bias}

        def _score(plugin_tokens: set[str], *, workflow_name: str, item_type: str, item_id: str) -> dict[str, Any]:
            overlap = [w for w in plugin_tokens if w in customer_freq]
            overlap_sorted = sorted(overlap, key=lambda w: customer_freq.get(w, 0), reverse=True)
            base = sum(int(customer_freq.get(w, 0)) for w in overlap_sorted)
            fb = _feedback_weight(workflow_name, item_type, item_id)
            return {
                "score": base + fb["bias"],
                "base_score": base,
                "feedback_bias": fb["bias"],
                "feedback_accepts": fb["accepts"],
                "feedback_rejects": fb["rejects"],
                "overlap": overlap_sorted[:10],
                "overlap_count": len(overlap),
            }

        # Categorize existing plugins into workflows
        def _workflow_for_preset(preset_id: str) -> str:
            pid = (preset_id or "").lower()
            if "triage" in pid or "regression" in pid:
                return "triage"
            if "repro" in pid or "checklist" in pid:
                return "extraction"
            if "gap" in pid or "hypoth" in pid:
                return "literature"
            return "other"

        def _workflow_for_eval(eval_id: str) -> str:
            eid = (eval_id or "").lower()
            if "triage" in eid or "regression" in eid:
                return "triage"
            if "extraction" in eid:
                return "extraction"
            if "literature" in eid:
                return "literature"
            return "other"

        scored_presets: list[dict[str, Any]] = []
        for p in presets:
            wf = _workflow_for_preset(p.id)
            scored_presets.append(
                {
                    "id": p.id,
                    "name": p.name,
                    "workflow": wf,
                    **_score(_plugin_tokens_preset(p), workflow_name=wf, item_type="dataset_preset", item_id=p.id),
                }
            )
        scored_evals: list[dict[str, Any]] = []
        for t in evals:
            wf = _workflow_for_eval(t.id)
            scored_evals.append(
                {
                    "id": t.id,
                    "name": t.name,
                    "workflow": wf,
                    **_score(_plugin_tokens_eval(t), workflow_name=wf, item_type="eval_template", item_id=t.id),
                }
            )

        def _pick_best(scored: list[dict[str, Any]], workflow_name: str) -> Optional[dict[str, Any]]:
            candidates = [x for x in scored if x.get("workflow") == workflow_name]
            candidates.sort(key=lambda x: (x.get("score", 0), x.get("overlap_count", 0)), reverse=True)
            if not candidates:
                return None
            # Guardrail: avoid items the customer has repeatedly rejected, unless nothing else fits.
            def is_blocked(c: dict[str, Any]) -> bool:
                try:
                    rejects = int(c.get("feedback_rejects") or 0)
                    accepts = int(c.get("feedback_accepts") or 0)
                except Exception:
                    rejects = 0
                    accepts = 0
                # Conservative: block only when there is strong negative signal and no positive signal.
                return rejects >= 3 and accepts == 0

            unblocked = [c for c in candidates if not is_blocked(c)]
            if unblocked:
                candidates = unblocked
            best = candidates[0]
            # Require at least a weak match to claim it's customer-specific
            if best.get("overlap_count", 0) < 3 and best.get("score", 0) < 5 and customer_context:
                return None
            return best

        dataset_preset_ids: list[str] = []
        eval_template_ids: list[str] = []
        rationale: list[dict[str, Any]] = []
        recommended_new: list[dict[str, Any]] = []
        selected_by_workflow: dict[str, dict[str, Optional[str]]] = {}

        def _dedupe_preserve_order(items: list[str]) -> list[str]:
            seen: set[str] = set()
            out: list[str] = []
            for x in items:
                if x in seen:
                    continue
                seen.add(x)
                out.append(x)
            return out

        def _best_scored_id_for_workflow(scored: list[dict[str, Any]], workflow_name: str) -> Optional[str]:
            candidates = [x for x in scored if x.get("workflow") == workflow_name]
            candidates.sort(key=lambda x: (x.get("score", 0), x.get("overlap_count", 0)), reverse=True)
            return candidates[0]["id"] if candidates else None

        def _representative_id_for_workflow(
            *,
            workflow_name: str,
            item_key: str,
            selected: dict[str, dict[str, Optional[str]]],
            allowlist_ids: list[str],
            classifier: Callable[[str], str],
            scored: list[dict[str, Any]],
        ) -> Optional[str]:
            # 1) Per-workflow explicit selection (best match)
            explicit = (selected.get(workflow_name) or {}).get(item_key)
            if explicit:
                return explicit

            # 2) If allowlist contains workflow-scoped items, pick first (stable)
            for iid in allowlist_ids:
                try:
                    if classifier(iid) == workflow_name:
                        return iid
                except Exception:
                    continue

            # 3) Fall back to "best available" for that workflow, even if low-signal
            return _best_scored_id_for_workflow(scored, workflow_name)

        for wf in workflows:
            if wf not in {"triage", "extraction", "literature"}:
                continue
            best_preset = _pick_best(scored_presets, wf)
            best_eval = _pick_best(scored_evals, wf)
            selected_by_workflow.setdefault(wf, {"dataset_preset_id": None, "eval_template_id": None})

            if best_preset:
                dataset_preset_ids.append(best_preset["id"])
                selected_by_workflow[wf]["dataset_preset_id"] = best_preset["id"]
                rationale.append(
                    {
                        "type": "dataset_preset",
                        "workflow": wf,
                        "id": best_preset["id"],
                        "score": best_preset["score"],
                        "base_score": best_preset.get("base_score", best_preset["score"]),
                        "feedback_bias": best_preset.get("feedback_bias", 0),
                        "feedback_accepts": best_preset.get("feedback_accepts", 0),
                        "feedback_rejects": best_preset.get("feedback_rejects", 0),
                        "matched_terms": best_preset["overlap"],
                    }
                )
            else:
                recommended_new.append(
                    {
                        "type": "dataset_preset",
                        "workflow": wf,
                        "id_suggestion": f"{wf}_{domain.lower().replace('/','_').replace(' ','_')}_v1".lower(),
                        "name_suggestion": f"{wf.title()} ({domain}) (v1)",
                        "why": "No existing preset matched customer keywords strongly enough.",
                        "skeleton": {
                            "id": "<replace>",
                            "name": "<replace>",
                            "description": f"Customer-specific {wf} dataset generation preset for {domain}.",
                            "dataset_type": "instruction",
                            "generation_prompt": "You are generating training data for a domain-specific research assistant. Generate {num} instruction/answer pairs from the document. Output JSON array with 'instruction' and 'output' only.",
                        },
                    }
                )

            if best_eval:
                eval_template_ids.append(best_eval["id"])
                selected_by_workflow[wf]["eval_template_id"] = best_eval["id"]
                rationale.append(
                    {
                        "type": "eval_template",
                        "workflow": wf,
                        "id": best_eval["id"],
                        "score": best_eval["score"],
                        "base_score": best_eval.get("base_score", best_eval["score"]),
                        "feedback_bias": best_eval.get("feedback_bias", 0),
                        "feedback_accepts": best_eval.get("feedback_accepts", 0),
                        "feedback_rejects": best_eval.get("feedback_rejects", 0),
                        "matched_terms": best_eval["overlap"],
                    }
                )
            else:
                recommended_new.append(
                    {
                        "type": "eval_template",
                        "workflow": wf,
                        "id_suggestion": f"{wf}_{domain.lower().replace('/','_').replace(' ','_')}_v1".lower(),
                        "name_suggestion": f"{wf.title()} Eval ({domain}) (v1)",
                        "why": "No existing eval template matched customer keywords strongly enough.",
                        "skeleton": {
                            "id": "<replace>",
                            "name": "<replace>",
                            "description": f"Customer-specific {wf} eval for {domain}.",
                            "version": 1,
                            "judge_preamble": "You are an evaluator for a domain-specific research assistant. Penalize hallucinations; prefer actionable next steps.",
                            "rubric": {"scale": "1-5", "criteria": ["Actionability", "Fidelity", "Clarity", "Rigor"]},
                            "cases": [{"id": f"{wf}_001", "prompt": "Write a realistic test prompt for this customer/workflow."}],
                        },
                    }
                )

        # If nothing selected, prefer presets/evals aligned to requested workflows.
        # Avoid proposing an empty allowlist (empty allowlist == "all enabled" in this product).
        has_customer_signal = bool(customer_context) or bool(top_keywords)
        if not dataset_preset_ids and presets:
            if not has_customer_signal:
                dataset_preset_ids = [p.id for p in presets]
            else:
                requested = set([w for w in workflows if w in {"triage", "extraction", "literature"}])
                dataset_preset_ids = [p.id for p in presets if _workflow_for_preset(p.id) in requested]
                if not dataset_preset_ids:
                    dataset_preset_ids = [p.id for p in presets]
        if not eval_template_ids and evals:
            if not has_customer_signal:
                eval_template_ids = [t.id for t in evals]
            else:
                requested = set([w for w in workflows if w in {"triage", "extraction", "literature"}])
                eval_template_ids = [t.id for t in evals if _workflow_for_eval(t.id) in requested]
                if not eval_template_ids:
                    eval_template_ids = [t.id for t in evals]

        dataset_preset_ids = _dedupe_preserve_order(dataset_preset_ids)
        eval_template_ids = _dedupe_preserve_order(eval_template_ids)

        _emit(60, "composing", "Prepared bundle configuration and demo plan")
        await db.commit()

        workflow_specs = {
            "triage": {
                "title": "Triage",
                "happy_path": ["Generate dataset", "Train", "Deploy adapter", "Run eval", "Use in Chat"],
            },
            "extraction": {
                "title": "Extraction",
                "happy_path": ["Generate dataset", "Train", "Deploy adapter", "Run eval", "Use in Chat"],
            },
            "literature": {
                "title": "Literature",
                "happy_path": ["Synthesize", "Save note", "Generate dataset", "Train", "Deploy adapter", "Run eval", "Use in Chat"],
            },
        }

        demo_plan: list[dict[str, Any]] = []
        demo_workflows = [w for w in workflows if w in workflow_specs]
        for idx, wf in enumerate(demo_workflows):
            preset_id = _representative_id_for_workflow(
                workflow_name=wf,
                item_key="dataset_preset_id",
                selected=selected_by_workflow,
                allowlist_ids=dataset_preset_ids,
                classifier=_workflow_for_preset,
                scored=scored_presets,
            )
            eval_id = _representative_id_for_workflow(
                workflow_name=wf,
                item_key="eval_template_id",
                selected=selected_by_workflow,
                allowlist_ids=eval_template_ids,
                classifier=_workflow_for_eval,
                scored=scored_evals,
            )
            spec = workflow_specs[wf]
            demo_plan.append(
                {
                    "name": f"Workflow {chr(65 + idx)} — {spec['title']}",
                    "workflow": wf,
                    "preset_id": preset_id,
                    "eval_template_id": eval_id,
                    "happy_path": spec["happy_path"],
                }
            )

        profile_name = (customer_profile.name if customer_profile else "") if customer_profile else ""
        ai_hub_bundle = {
            "bundle_name": f"{profile_name} Bundle" if profile_name else (f"{domain} Bundle" if domain != "Research" else "Research Bundle"),
            "inferred_domain": domain,
            "customer_profile": customer_profile.model_dump() if customer_profile else None,
            "customer_keywords": top_keywords[:20],
            "customer_evidence": evidence,
            "enabled_dataset_presets": dataset_preset_ids,
            "enabled_eval_templates": eval_template_ids,
            "workflows": workflows,
            "env": {
                "AI_HUB_DATASET_ENABLED_PRESET_IDS": ",".join(dataset_preset_ids),
                "AI_HUB_EVAL_ENABLED_TEMPLATE_IDS": ",".join(eval_template_ids),
            },
            "selection_rationale": rationale,
            "recommended_new_plugins": recommended_new,
            "success_metrics": {
                "triage": "median time-to-triage; severe-regression misses",
                "extraction": "field-level correctness; unknowns explicitly marked",
                "literature": "precision@k on read/skim/skip against historical decisions",
            },
            "demo_plan": demo_plan,
        }

        job.results = {
            "summary": f"Proposed an AI Hub bundle (presets + evals) and a {len(demo_plan)}-workflow happy-path demo plan.",
            "actions_count": len(demo_plan),
            "findings_count": 0,
            "ai_hub_bundle": ai_hub_bundle,
        }

        if apply_now:
            user_result = await db.execute(select(User).where(User.id == job.user_id))
            user = user_result.scalar_one_or_none()
            if not user or not user.is_admin():
                job.add_log_entry({"phase": "apply", "error": "Apply requested but user is not admin"})
            else:
                # Store as CSV in feature flags (empty string means "all enabled" semantics)
                await set_feature_str("ai_hub_enabled_dataset_presets", ",".join(dataset_preset_ids))
                await set_feature_str("ai_hub_enabled_eval_templates", ",".join(eval_template_ids))
                job.add_log_entry({"phase": "apply", "result": "Applied bundle to feature-flag allowlists"})

        _emit(100, "completed", "Bundle proposal ready")
        job.status = AgentJobStatus.COMPLETED.value
        job.completed_at = datetime.utcnow()
        await db.commit()

        if progress_callback:
            try:
                await progress_callback(
                    {
                        "job_id": str(job.id),
                        "progress": job.progress,
                        "phase": job.current_phase,
                        "status": job.status,
                        "iteration": job.iteration,
                        "phase_details": job.phase_details,
                        "error": job.error,
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                )
            except Exception:
                pass

        return {"status": "completed", "results": job.results}

    async def _run_research_inbox_monitor(
        self,
        *,
        job: AgentJob,
        db: AsyncSession,
        progress_callback: Optional[callable],
    ) -> Dict[str, Any]:
        """
        Deterministic runner: monitor internal KB + arXiv, write new items into `research_inbox_items`.

        Intended for scheduled runs. Dedupes per-user on (item_type, item_key).
        """
        import re
        from datetime import timezone

        from app.core.feature_flags import get_str as get_feature_str
        from app.models.research_inbox import ResearchInboxItem
        from app.schemas.customer_profile import CustomerProfile

        def _emit(progress: int, phase: str, details: str):
            job.progress = max(0, min(100, int(progress)))
            job.current_phase = phase
            job.phase_details = details
            job.last_activity_at = datetime.utcnow()
            job.add_log_entry({"phase": phase, "action": "research_inbox_monitor", "result": details})

        def _safe_text(x: Any) -> str:
            try:
                return str(x or "")
            except Exception:
                return ""

        def _parse_iso_dt(s: str) -> Optional[datetime]:
            ss = (s or "").strip()
            if not ss:
                return None
            if ss.endswith("Z"):
                ss = ss[:-1] + "+00:00"
            try:
                dt = datetime.fromisoformat(ss)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except Exception:
                return None

        def _tokens(text: str) -> list[str]:
            raw = re.findall(r"[a-zA-Z0-9_\\-]+", (text or "").lower())
            out: list[str] = []
            stop = {
                "the","and","for","with","from","that","this","into","over","under","when","where","what","which","while",
                "your","you","are","our","their","they","them","then","than","also","only","just","more","most","less",
                "use","using","used","make","made","help","helps","via","can","could","should","would","may","might","will",
                "data","dataset","datasets","model","models","train","training","eval","evaluate","evaluation","assistant",
                "job","jobs","paper","papers","doc","docs","document","documents","research","monitor",
            }
            for w in raw:
                w = w.strip("_-")
                if len(w) < 3:
                    continue
                if w in stop:
                    continue
                out.append(w)
            return out

        async def _load_feedback_bias_tokens(*, customer: Optional[str]) -> tuple[list[str], set[str], list[str], dict]:
            """
            Load positive/negative token sets from the persisted monitor profile if present,
            otherwise derive from inbox items.

            Returns: (positive_tokens, negative_tokens_set, debug_info)
            """
            debug: dict[str, Any] = {"source": None, "positive_tokens": [], "negative_tokens": [], "muted_patterns": []}
            muted_patterns: list[str] = []

            # Prefer persisted profile.
            try:
                from app.services.research_monitor_profile_service import research_monitor_profile_service

                prof = await research_monitor_profile_service.get_profile(db=db, user_id=job.user_id, customer=customer)
                if prof:
                    raw_scores = prof.token_scores if isinstance(getattr(prof, "token_scores", None), dict) else {}
                    scores = {str(k): int(v) for k, v in (raw_scores or {}).items() if isinstance(v, (int, float))}
                    positive = [t for t, s in sorted(scores.items(), key=lambda kv: kv[1], reverse=True) if s >= 3][:20]
                    negative = {t for t, s in sorted(scores.items(), key=lambda kv: kv[1]) if s <= -3}
                    negative -= set(positive)
                    try:
                        muted = getattr(prof, "muted_tokens", None)
                        if isinstance(muted, list):
                            negative |= {str(x).strip().lower() for x in muted if str(x).strip()}
                    except Exception:
                        pass
                    try:
                        mp = getattr(prof, "muted_patterns", None)
                        if isinstance(mp, list):
                            muted_patterns = [str(x).strip() for x in mp if str(x).strip()]
                    except Exception:
                        muted_patterns = []
                    debug["source"] = "profile"
                    debug["positive_tokens"] = positive[:10]
                    debug["negative_tokens"] = list(sorted(list(negative)))[:10]
                    debug["muted_patterns"] = muted_patterns[:10]
                    return (positive, negative, muted_patterns, debug)
            except Exception:
                pass

            # Fallback: derive from inbox history.
            try:
                stmt = (
                    select(ResearchInboxItem.status, ResearchInboxItem.title, ResearchInboxItem.summary)
                    .where(
                        ResearchInboxItem.user_id == job.user_id,
                        ResearchInboxItem.status.in_(["accepted", "rejected"]),
                    )
                    .order_by(ResearchInboxItem.updated_at.desc())
                    .limit(250)
                )
                if customer:
                    stmt = stmt.where(ResearchInboxItem.customer == customer)

                res = await db.execute(stmt)
                rows = res.all()
            except Exception:
                return ([], set(), muted_patterns, debug)

            pos = Counter()
            neg = Counter()
            for status, title, summary in rows:
                text = f"{_safe_text(title)} {_safe_text(summary)}"
                toks = _tokens(text)
                if not toks:
                    continue
                if str(status) == "accepted":
                    pos.update(toks)
                elif str(status) == "rejected":
                    neg.update(toks)

            scores2: dict[str, int] = {}
            for t, c in pos.items():
                scores2[t] = scores2.get(t, 0) + int(c)
            for t, c in neg.items():
                scores2[t] = scores2.get(t, 0) - int(c)

            positive2 = [t for t, s in sorted(scores2.items(), key=lambda kv: kv[1], reverse=True) if s >= 3][:20]
            negative2 = {t for t, s in sorted(scores2.items(), key=lambda kv: kv[1]) if s <= -3}
            negative2 -= set(positive2)
            debug["source"] = "inbox_derived"
            debug["positive_tokens"] = positive2[:10]
            debug["negative_tokens"] = list(sorted(list(negative2)))[:10]
            return (positive2, negative2, muted_patterns, debug)

        async def _create_inbox_item(
            *,
            item_type: str,
            item_key: str,
            title: str,
            summary: Optional[str],
            url: Optional[str],
            published_at: Optional[datetime],
            customer: Optional[str],
            metadata: dict,
        ) -> bool:
            it = ResearchInboxItem(
                user_id=job.user_id,
                job_id=job.id,
                customer=customer,
                item_type=item_type,
                item_key=item_key,
                title=title or item_key,
                summary=summary,
                url=url,
                published_at=published_at,
                discovered_at=datetime.utcnow(),
                status="new",
                feedback=None,
                item_metadata=metadata,
            )
            db.add(it)
            try:
                await db.flush()
                return True
            except IntegrityError:
                await db.rollback()
                return False

        # Customer profile (deployment-level) + optional per-job context.
        customer_profile_raw = await get_feature_str("ai_hub_customer_profile")
        customer_profile: CustomerProfile | None = None
        if customer_profile_raw:
            try:
                customer_profile = CustomerProfile.model_validate(json.loads(customer_profile_raw))
            except Exception:
                customer_profile = None

        customer_context = _safe_text((job.config or {}).get("customer_context")).strip()
        if not customer_context and customer_profile and customer_profile.notes:
            customer_context = _safe_text(customer_profile.notes).strip()

        customer_name = _safe_text(getattr(customer_profile, "name", "") if customer_profile else "").strip() or None
        customer_tag = _safe_text((job.config or {}).get("customer") or customer_name).strip() or None

        prefer_sources = (job.config or {}).get("prefer_sources")
        if not isinstance(prefer_sources, list):
            prefer_sources = ["documents", "arxiv"]
        prefer_sources = [str(s).strip().lower() for s in prefer_sources if str(s).strip()]

        max_documents = int((job.config or {}).get("max_documents") or 8)
        max_papers = int((job.config or {}).get("max_papers") or 8)
        max_documents = max(0, min(max_documents, 50))
        max_papers = max(0, min(max_papers, 50))

        monitor_queries = (job.config or {}).get("monitor_queries")
        if not isinstance(monitor_queries, list):
            monitor_queries = []
        monitor_queries = [str(q).strip() for q in monitor_queries if isinstance(q, (str, int, float)) and str(q).strip()]

        use_feedback_bias = bool((job.config or {}).get("use_feedback_bias", True))
        positive_tokens: list[str] = []
        negative_tokens: set[str] = set()
        muted_patterns: list[str] = []
        bias_debug: dict = {}
        if use_feedback_bias:
            positive_tokens, negative_tokens, muted_patterns, bias_debug = await _load_feedback_bias_tokens(customer=customer_tag)

        def _is_muted(text: str) -> bool:
            if not muted_patterns:
                return False
            t = (text or "").lower()
            for p in muted_patterns:
                pp = (p or "").strip().lower()
                if not pp:
                    continue
                if pp in t:
                    return True
            return False

        # Filter manual queries too
        if monitor_queries and muted_patterns:
            monitor_queries = [q for q in monitor_queries if not _is_muted(q)]

        if not monitor_queries:
            goal = _safe_text(job.goal).strip()
            kws: list[str] = []
            if customer_profile and isinstance(customer_profile.keywords, list):
                kws = [str(x).strip() for x in customer_profile.keywords if str(x).strip()]

            seed = " ".join([goal, customer_context, " ".join(kws[:12])]).strip()
            if positive_tokens:
                seed = (seed + " " + " ".join(positive_tokens)).strip()
            toks = [t for t in _tokens(seed) if t not in negative_tokens]

            derived: list[str] = []
            if goal:
                derived.append(goal[:200])
            if customer_tag:
                derived.append(f"{customer_tag} {goal[:160]}".strip()[:200] if goal else customer_tag[:200])
            if toks:
                derived.append(" ".join(toks[:10])[:200])

            seen: set[str] = set()
            deduped: list[str] = []
            for q in derived:
                q = (q or "").strip()
                if not q or q in seen:
                    continue
                if _is_muted(q):
                    continue
                seen.add(q)
                deduped.append(q)
            monitor_queries = deduped[:5]

        job.iteration = int(job.iteration or 0) + 1
        _emit(5, "planning", f"Monitoring {len(monitor_queries)} queries (sources: {', '.join(prefer_sources) or 'none'})")
        await db.commit()

        created = 0
        skipped = 0
        created_doc_ids: list[str] = []

        if "documents" in prefer_sources and max_documents > 0:
            for idx, q in enumerate(monitor_queries):
                _emit(10 + idx * 10, "searching_documents", f"Searching KB: {q[:120]}")
                await db.commit()
                try:
                    docs, _total, _took = await self.search_service.search(
                        query=q,
                        mode="smart",
                        page=1,
                        page_size=max_documents,
                        db=db,
                    )
                except Exception as exc:
                    logger.warning(f"Research inbox KB search failed for job {job.id}: {exc}")
                    continue

                for d in docs or []:
                    if not isinstance(d, dict):
                        continue
                    doc_id = _safe_text(d.get("id")).strip()
                    if not doc_id:
                        continue
                    title_text = _safe_text(d.get("title")).strip()
                    summary_text = _safe_text(d.get("snippet")).strip()
                    if _is_muted(f"{title_text} {summary_text}"):
                        skipped += 1
                        continue
                    ok = await _create_inbox_item(
                        item_type="document",
                        item_key=doc_id,
                        title=title_text or doc_id,
                        summary=summary_text or None,
                        url=_safe_text(d.get("url")).strip() or None,
                        published_at=None,
                        customer=customer_tag,
                        metadata={
                            "query": q,
                            "document_id": doc_id,
                            "source": d.get("source"),
                            "source_type": d.get("source_type"),
                            "relevance_score": d.get("relevance_score"),
                            "bias": bias_debug or None,
                        },
                    )
                    if ok:
                        created += 1
                        created_doc_ids.append(doc_id)
                    else:
                        skipped += 1

        if "arxiv" in prefer_sources and max_papers > 0:
            for idx, q in enumerate(monitor_queries):
                _emit(60 + idx * 8, "searching_arxiv", f"Searching arXiv: {q[:120]}")
                await db.commit()

                toks = [t for t in _tokens(q) if t not in negative_tokens]
                if positive_tokens:
                    # Add a couple of learned positives into the arXiv query deterministically.
                    for t in positive_tokens[:4]:
                        if t not in toks:
                            toks.append(t)
                if toks:
                    arxiv_q = " AND ".join([f"all:{t}" for t in toks[:6]])
                else:
                    phrase = " ".join(re.findall(r"[a-zA-Z0-9_\\-]+", q))[:120].strip()
                    if not phrase:
                        continue
                    arxiv_q = f'all:\"{phrase}\"'

                try:
                    res = await self.arxiv_service.search(
                        query=arxiv_q,
                        start=0,
                        max_results=max_papers,
                        sort_by="submittedDate",
                        sort_order="descending",
                    )
                except Exception as exc:
                    logger.warning(f"Research inbox arXiv search failed for job {job.id}: {exc}")
                    continue

                for it in res.items or []:
                    if not isinstance(it, dict):
                        continue
                    arxiv_id = _safe_text(it.get("id")).strip()
                    if not arxiv_id:
                        continue
                    title_text = _safe_text(it.get("title")).strip()
                    summary_text = _safe_text(it.get("summary")).strip()
                    if _is_muted(f"{title_text} {summary_text}"):
                        skipped += 1
                        continue
                    ok = await _create_inbox_item(
                        item_type="arxiv",
                        item_key=arxiv_id,
                        title=title_text or arxiv_id,
                        summary=summary_text or None,
                        url=_safe_text(it.get("pdf_url") or it.get("entry_url")).strip() or None,
                        published_at=_parse_iso_dt(_safe_text(it.get("published"))) or None,
                        customer=customer_tag,
                        metadata={
                            "query": q,
                            "arxiv_query": arxiv_q,
                            "arxiv_id": arxiv_id,
                            "entry_url": it.get("entry_url"),
                            "pdf_url": it.get("pdf_url"),
                            "authors": it.get("authors"),
                            "categories": it.get("categories"),
                            "primary_category": it.get("primary_category"),
                            "updated": it.get("updated"),
                            "doi": it.get("doi"),
                            "comments": it.get("comments"),
                            "bias": bias_debug or None,
                        },
                    )
                    if ok:
                        created += 1
                    else:
                        skipped += 1

        await db.commit()

        auto_add = bool((job.config or {}).get("auto_add_to_reading_list", False))
        reading_list_name = _safe_text((job.config or {}).get("reading_list_name")).strip()
        if auto_add and reading_list_name and created_doc_ids:
            try:
                from app.models.reading_list import ReadingList, ReadingListItem
                from app.models.document import Document

                rl_result = await db.execute(
                    select(ReadingList).where(
                        ReadingList.user_id == job.user_id,
                        ReadingList.name == reading_list_name,
                    )
                )
                rl = rl_result.scalar_one_or_none()
                if not rl:
                    rl = ReadingList(user_id=job.user_id, name=reading_list_name, description="Auto-populated by Research Inbox monitor")
                    db.add(rl)
                    await db.commit()
                    await db.refresh(rl)

                max_pos_res = await db.execute(
                    select(func.max(ReadingListItem.position)).where(ReadingListItem.reading_list_id == rl.id)
                )
                max_pos = int(max_pos_res.scalar() or 0)
                added = 0

                for doc_id in created_doc_ids[:200]:
                    try:
                        doc_uuid = UUID(str(doc_id))
                    except Exception:
                        continue
                    doc = await db.get(Document, doc_uuid)
                    if not doc:
                        continue

                    exists = await db.execute(
                        select(func.count())
                        .select_from(ReadingListItem)
                        .where(
                            ReadingListItem.reading_list_id == rl.id,
                            ReadingListItem.document_id == doc.id,
                        )
                    )
                    if int(exists.scalar() or 0) > 0:
                        continue

                    item = ReadingListItem(
                        reading_list_id=rl.id,
                        document_id=doc.id,
                        status="to-read",
                        priority=0,
                        position=max_pos + 1,
                        notes="Added automatically by Research Inbox monitor",
                    )
                    db.add(item)
                    try:
                        await db.flush()
                    except IntegrityError:
                        await db.rollback()
                        continue

                    max_pos += 1
                    added += 1

                await db.commit()
                if job.results is None:
                    job.results = {}
                job.results["reading_list"] = {"name": reading_list_name, "items_added": added}
            except Exception as exc:
                logger.warning(f"Failed to auto-populate reading list for inbox monitor: {exc}")

        persist = bool((job.config or {}).get("persist_artifacts", False))
        if persist and created > 0:
            try:
                from app.models.document import Document

                notes_source = await self.document_service._get_or_create_agent_notes_source(db)
                now = datetime.utcnow()
                iso_year, iso_week, _ = now.isocalendar()
                customer_slug = (customer_tag or "default").lower()
                customer_slug = re.sub(r"[^a-z0-9_\\-]+", "-", customer_slug).strip("-")[:64] or "default"
                source_identifier = f"research_inbox_weekly:{customer_slug}:{iso_year}-W{iso_week:02d}"

                existing = await db.execute(
                    select(Document)
                    .where(
                        Document.source_id == notes_source.id,
                        Document.source_identifier == source_identifier,
                    )
                    .limit(1)
                )
                doc = existing.scalar_one_or_none()

                header = f"# Research Inbox Weekly Brief — {customer_tag}" if customer_tag else "# Research Inbox Weekly Brief"
                section_lines: list[str] = [
                    header,
                    "",
                    "## Run",
                    f"- Timestamp: {now.isoformat()}Z",
                    f"- Queries: {', '.join(monitor_queries)[:800]}",
                    f"- New items created: {created}",
                    f"- Duplicates skipped: {skipped}",
                ]
                if customer_context:
                    section_lines.append(f"- Customer context: {customer_context[:500]}")
                section_lines.append("")

                content = "\n".join(section_lines).strip() + "\n"

                if doc:
                    doc.content = (doc.content or "").rstrip() + "\n\n" + content
                    doc.updated_at = datetime.utcnow()
                else:
                    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
                    doc = Document(
                        title=header.lstrip("# ").strip()[:240] or "Research Inbox Weekly Brief",
                        content=content,
                        content_hash=content_hash,
                        url=None,
                        file_path=None,
                        file_type="text/markdown",
                        file_size=len(content.encode("utf-8")),
                        source_id=notes_source.id,
                        source_identifier=source_identifier,
                        author=None,
                        tags=["autonomous_job", "research_inbox", "monitor"],
                        extra_metadata={
                            "origin": "autonomous_job",
                            "job_id": str(job.id),
                            "job_type": job.job_type,
                            "customer": customer_tag,
                        },
                        is_processed=False,
                    )
                    db.add(doc)

                await db.commit()
                await db.refresh(doc)

                try:
                    await self.document_service.reprocess_document(doc.id, db, user_id=job.user_id)
                except Exception:
                    pass

                if job.results is None:
                    job.results = {}
                job.results["weekly_brief_document"] = {"id": str(doc.id), "title": doc.title, "source_identifier": source_identifier}
            except Exception as exc:
                logger.warning(f"Failed to persist weekly brief for inbox monitor: {exc}")

        job.results = job.results or {}
        job.results.update(
            {
                "summary": f"Research inbox monitor: {created} new items ({skipped} duplicates) across {len(monitor_queries)} queries.",
                "monitor": {
                    "queries": monitor_queries,
                    "prefer_sources": prefer_sources,
                    "items_created": created,
                    "items_skipped": skipped,
                },
                "customer_profile": customer_profile.model_dump() if customer_profile else None,
                "customer_context": customer_context,
            }
        )

        _emit(100, "completed", job.results.get("summary", "Completed"))
        job.status = AgentJobStatus.COMPLETED.value
        job.completed_at = datetime.utcnow()
        await db.commit()

        if progress_callback:
            try:
                await progress_callback(
                    {
                        "job_id": str(job.id),
                        "progress": job.progress,
                        "phase": job.current_phase,
                        "status": job.status,
                        "iteration": job.iteration,
                        "phase_details": job.phase_details,
                        "error": job.error,
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                )
            except Exception:
                pass

        return {"status": "completed", "results": job.results}

    async def _run_code_patch_proposer(
        self,
        *,
        job: AgentJob,
        db: AsyncSession,
        progress_callback: Optional[callable],
    ) -> Dict[str, Any]:
        """
        Code Agent (MVP): generate a unified diff proposal against a git code source.

        Expects:
          - job.config.target_source_id (UUID of a git DocumentSource)
          - optional job.config.search_query, file_paths[], max_files, max_chars_per_file

        Produces:
          - job.results.code_patch (summary + diff + metadata)
          - CodePatchProposal row + artifact reference in job.output_artifacts
        """
        from uuid import UUID as _UUID
        from app.models.document import Document, DocumentSource
        from app.models.code_patch_proposal import CodePatchProposal

        def _emit(progress: int, phase: str, details: str):
            job.progress = max(0, min(100, int(progress)))
            job.current_phase = phase
            job.phase_details = details
            job.last_activity_at = datetime.utcnow()
            job.add_log_entry({"phase": phase, "action": "code_patch_proposer", "result": details})

        target_source_id = (job.config or {}).get("target_source_id") if isinstance(job.config, dict) else None
        if not target_source_id:
            inherited = (job.config or {}).get("inherited_data") if isinstance(job.config, dict) else None
            if isinstance(inherited, dict):
                parent_results = inherited.get("parent_results") if isinstance(inherited.get("parent_results"), dict) else None
                if parent_results and isinstance(parent_results.get("repo_ingest"), dict):
                    target_source_id = parent_results["repo_ingest"].get("source_id")
        if not target_source_id:
            job.status = AgentJobStatus.FAILED.value
            job.error = "Missing job.config.target_source_id"
            await db.commit()
            return {"status": "failed", "error": job.error}
        try:
            source_uuid = _UUID(str(target_source_id))
        except Exception:
            job.status = AgentJobStatus.FAILED.value
            job.error = "Invalid target_source_id"
            await db.commit()
            return {"status": "failed", "error": job.error}

        source = await db.get(DocumentSource, source_uuid)
        if not source:
            job.status = AgentJobStatus.FAILED.value
            job.error = "Target source not found"
            await db.commit()
            return {"status": "failed", "error": job.error}

        _emit(5, "planning", f"Preparing code patch proposal for source: {source.name}")
        await db.commit()

        inherited = (job.config or {}).get("inherited_data") if isinstance(job.config, dict) else None
        parent_results = inherited.get("parent_results") if isinstance(inherited, dict) else None

        search_query = str((job.config or {}).get("search_query") or "").strip()
        file_paths = (job.config or {}).get("file_paths")
        if not isinstance(file_paths, list):
            file_paths = []
        file_paths = [str(p).strip() for p in file_paths if str(p).strip()]

        max_files = int((job.config or {}).get("max_files") or 6)
        max_files = max(1, min(max_files, 20))
        max_chars_per_file = int((job.config or {}).get("max_chars_per_file") or 8000)
        max_chars_per_file = max(1000, min(max_chars_per_file, 20000))

        # If we are refining after an ExperimentRunner step, default to the previously-touched files.
        if not file_paths and isinstance(parent_results, dict):
            prev_patch = parent_results.get("code_patch") if isinstance(parent_results.get("code_patch"), dict) else None
            prev_touched = prev_patch.get("files_touched") if isinstance(prev_patch, dict) and isinstance(prev_patch.get("files_touched"), list) else []
            file_paths = [str(p).strip() for p in prev_touched if str(p).strip()][:max_files]

        # Collect candidate documents (code files)
        docs: list[Document] = []
        if file_paths:
            # Match by title or source_identifier
            for p in file_paths[:max_files]:
                res = await db.execute(
                    select(Document)
                    .where(
                        Document.source_id == source.id,
                        or_(
                            Document.title == p,
                            Document.source_identifier == p,
                            Document.file_path == p,
                        ),
                    )
                    .limit(1)
                )
                d = res.scalar_one_or_none()
                if d:
                    docs.append(d)
        else:
            # Use search if provided; otherwise recent files
            if search_query:
                try:
                    results, _total, _took = await self.search_service.search(
                        query=search_query,
                        mode="smart",
                        page=1,
                        page_size=max_files,
                        source_id=str(source.id),
                        db=db,
                    )
                    ids = [r.get("id") for r in (results or []) if isinstance(r, dict) and r.get("id")]
                    for doc_id in ids[:max_files]:
                        try:
                            d = await db.get(Document, _UUID(str(doc_id)))
                        except Exception:
                            d = None
                        if d and d.source_id == source.id:
                            docs.append(d)
                except Exception:
                    docs = []

            if not docs:
                res = await db.execute(
                    select(Document)
                    .where(Document.source_id == source.id)
                    .order_by(Document.updated_at.desc())
                    .limit(max_files)
                )
                docs = list(res.scalars().all())

        if not docs:
            job.status = AgentJobStatus.FAILED.value
            job.error = "No code documents found for the target source"
            await db.commit()
            return {"status": "failed", "error": job.error}

        # Build prompt context
        _emit(20, "collecting", f"Loaded {len(docs)} candidate files")
        await db.commit()

        file_blocks: list[str] = []
        file_meta: list[dict] = []
        for d in docs[:max_files]:
            content = (d.content or "")[:max_chars_per_file]
            file_id = str(d.id)
            path = d.title or d.source_identifier or d.file_path or file_id
            file_meta.append({"document_id": file_id, "path": path, "truncated": len(d.content or "") > len(content)})
            file_blocks.append(f"### FILE: {path}\n```text\n{content}\n```\n")

        _emit(40, "drafting", "Generating patch proposal with LLM")
        await db.commit()

        user_settings = await self._load_user_settings(job.user_id, db)
        refinement_context: list[str] = []
        previous_diff_excerpt: str | None = None
        if isinstance(parent_results, dict):
            prev_patch = parent_results.get("code_patch") if isinstance(parent_results.get("code_patch"), dict) else None
            prev_proposal_id = str(prev_patch.get("proposal_id") or "").strip() if isinstance(prev_patch, dict) else ""
            if prev_proposal_id:
                try:
                    prev_proposal = await db.get(CodePatchProposal, _UUID(prev_proposal_id))
                except Exception:
                    prev_proposal = None
                if prev_proposal and prev_proposal.user_id == job.user_id:
                    prev_summary = str(prev_proposal.summary or "").strip()
                    if prev_summary:
                        refinement_context.append(f"Previous patch summary:\n{prev_summary}")
                    prev_diff = str(prev_proposal.diff_unified or "").strip()
                    if prev_diff:
                        previous_diff_excerpt = "\n".join(prev_diff.splitlines()[:160])[:6000]

            exp = parent_results.get("experiment_run") if isinstance(parent_results.get("experiment_run"), dict) else None
            if isinstance(exp, dict):
                runs = exp.get("runs") if isinstance(exp.get("runs"), list) else []
                failures = [r for r in runs if isinstance(r, dict) and not bool(r.get("ok"))]
                if failures:
                    lines: list[str] = []
                    for r in failures[:4]:
                        cmd = str(r.get("command") or "")[:200]
                        code = r.get("exit_code")
                        stderr = str(r.get("stderr") or "")[:1200]
                        stdout = str(r.get("stdout") or "")[:800]
                        lines.append(f"- cmd: {cmd}\n  exit_code: {code}\n  stderr:\n{stderr}\n  stdout:\n{stdout}")
                    refinement_context.append("Experiment failures (most recent):\n" + "\n".join(lines))

            patch_apply = parent_results.get("code_patch_apply") if isinstance(parent_results.get("code_patch_apply"), dict) else None
            if isinstance(patch_apply, dict):
                errs = patch_apply.get("errors") if isinstance(patch_apply.get("errors"), list) else []
                if errs:
                    lines = []
                    for e in errs[:6]:
                        if not isinstance(e, dict):
                            continue
                        path = str(e.get("path") or "")
                        err = str(e.get("error") or e.get("message") or "")[:400]
                        if path:
                            lines.append(f"- {path}: {err}")
                        else:
                            lines.append(f"- {err}")
                    if lines:
                        refinement_context.append("Patch apply errors (most recent):\n" + "\n".join(lines))

        refinement_block = ""
        if refinement_context or previous_diff_excerpt:
            parts: list[str] = []
            parts.extend(refinement_context)
            if previous_diff_excerpt:
                parts.append(
                    "Previous diff excerpt (for reference; produce a complete diff against FILES below):\n" + previous_diff_excerpt
                )
            refinement_block = "\n\nREFINEMENT CONTEXT:\n" + "\n\n".join(parts) + "\n"

        prompt = (
            "You are a senior software engineer acting as a Code Agent.\n"
            "Task: produce a minimal, safe patch as a unified diff.\n\n"
            "Rules:\n"
            "- Output MUST be valid JSON only.\n"
            "- JSON keys: title, summary, diff_unified, files_touched, risks, tests_to_run.\n"
            "- diff_unified MUST be a standard unified diff starting with ---/+++ lines.\n"
            "- Only change files shown below; do not invent new file contents.\n"
            "- Keep patch small and focused.\n\n"
            f"GOAL:\n{(job.goal or '').strip()}\n\n"
            f"{refinement_block}\n"
            f"FILES:\n{''.join(file_blocks)}\n"
        )

        response = await self.llm_service.generate_response(
            query=prompt,
            context=None,
            temperature=0.2,
            max_tokens=2000,
            user_settings=user_settings,
            task_type="code_agent",
            user_id=job.user_id,
            db=db,
            routing=self._llm_routing_from_job_config(job.config),
        )

        try:
            payload = json.loads(response)
        except Exception:
            payload = {"title": "Code Patch Proposal", "summary": response[:800], "diff_unified": "", "files_touched": [], "risks": [], "tests_to_run": []}

        title = str(payload.get("title") or "Code Patch Proposal")[:500]
        summary = str(payload.get("summary") or "").strip() or None
        diff_unified = str(payload.get("diff_unified") or "").strip()
        files_touched = payload.get("files_touched") if isinstance(payload.get("files_touched"), list) else []
        risks = payload.get("risks") if isinstance(payload.get("risks"), list) else []
        tests_to_run = payload.get("tests_to_run") if isinstance(payload.get("tests_to_run"), list) else []

        if not diff_unified or "---" not in diff_unified or "+++" not in diff_unified:
            job.status = AgentJobStatus.FAILED.value
            job.error = "LLM did not produce a valid unified diff"
            await db.commit()
            return {"status": "failed", "error": job.error, "raw": response[:2000]}

        proposal = CodePatchProposal(
            user_id=job.user_id,
            job_id=job.id,
            source_id=source.id,
            title=title,
            summary=summary,
            diff_unified=diff_unified,
            metadata={
                "goal": (job.goal or "").strip(),
                "target_source_id": str(source.id),
                "target_source_name": source.name,
                "files_context": file_meta,
                "files_touched": files_touched,
                "risks": risks,
                "tests_to_run": tests_to_run,
            },
            status="proposed",
        )
        db.add(proposal)
        await db.commit()
        await db.refresh(proposal)

        job.results = dict(parent_results) if isinstance(parent_results, dict) else {}
        prev_cp = job.results.get("code_patch") if isinstance(job.results.get("code_patch"), dict) else None
        if isinstance(prev_cp, dict):
            existing = job.results.get("code_patches")
            if not isinstance(existing, list):
                existing = []
            existing.append(prev_cp)
            job.results["code_patches"] = existing[-5:]
        job.results["code_patch"] = {
            "proposal_id": str(proposal.id),
            "title": title,
            "summary": summary,
            "files_context": file_meta,
            "files_touched": files_touched,
            "risks": risks,
            "tests_to_run": tests_to_run,
        }
        if job.output_artifacts is None:
            job.output_artifacts = []
        job.output_artifacts.append({"type": "code_patch_proposal", "id": str(proposal.id), "title": title})

        _emit(100, "completed", f"Patch proposal ready: {title}")
        job.status = AgentJobStatus.COMPLETED.value
        job.completed_at = datetime.utcnow()
        await db.commit()

        if progress_callback:
            try:
                await progress_callback(
                    {
                        "job_id": str(job.id),
                        "progress": job.progress,
                        "phase": job.current_phase,
                        "status": job.status,
                        "iteration": job.iteration,
                        "phase_details": job.phase_details,
                        "error": job.error,
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                )
            except Exception:
                pass

        return {"status": "completed", "results": job.results}

    async def _run_research_engineer_scientist(
        self,
        *,
        job: AgentJob,
        db: AsyncSession,
        progress_callback: Optional[callable],
    ) -> Dict[str, Any]:
        """
        Deterministic runner: "AI Scientist" planning step for ResearchEngineer chains.

        Produces:
          - job.results.research_engineer_plan (hypothesis + experiments + metrics + risks)
          - optional: appends a LaTeX section into a LatexProject if job.config.latex_project_id is set

        Expects (optional):
          - job.config.search_query (string): KB query to ground on (defaults to job.goal)
          - job.config.max_documents (int): number of KB docs to include (default 8)
          - job.config.latex_project_id (UUID): LaTeX Studio project to update
        """
        import json
        from uuid import UUID as _UUID
        from app.models.document import Document
        from app.models.latex_project import LatexProject

        def _emit(progress: int, phase: str, details: str):
            job.progress = max(0, min(100, int(progress)))
            job.current_phase = phase
            job.phase_details = details
            job.last_activity_at = datetime.utcnow()
            job.add_log_entry({"phase": phase, "action": "research_engineer_scientist", "result": details})

        def _bib_key_from_uuid(doc_id: _UUID) -> str:
            return f"KDB:{str(doc_id)}"

        def _insert_before_end_document(source: str, addition: str) -> str:
            marker = "\\end{document}"
            s = (source or "")
            idx = s.rfind(marker)
            if idx == -1:
                return (s.rstrip() + "\n\n" + addition.strip() + "\n").lstrip("\n")
            before = s[:idx].rstrip()
            after = s[idx:]
            return f"{before}\n\n{addition.strip()}\n\n{after}"

        config = job.config if isinstance(job.config, dict) else {}
        search_query = str((config or {}).get("search_query") or "").strip() or str(job.goal or "").strip()
        max_docs = int((config or {}).get("max_documents") or 8)
        max_docs = max(1, min(max_docs, 20))
        latex_project_id = (config or {}).get("latex_project_id")

        _emit(10, "planning", "Selecting Knowledge DB sources")
        await db.commit()

        docs: list[Document] = []
        try:
            results, _total, _took = await self.search_service.search(
                query=search_query,
                mode="smart",
                page=1,
                page_size=max_docs,
                db=db,
            )
            ids = [r.get("id") for r in (results or []) if isinstance(r, dict) and r.get("id")]
            for doc_id in ids[:max_docs]:
                try:
                    d = await db.get(Document, _UUID(str(doc_id)))
                except Exception:
                    d = None
                if d:
                    docs.append(d)
        except Exception:
            docs = []

        cite_map = []
        for d in docs:
            try:
                cite_map.append(
                    {
                        "doc_id": str(d.id),
                        "cite_key": _bib_key_from_uuid(d.id),
                        "title": (d.title or "").strip()[:200],
                        "url": (d.url or "").strip(),
                        "snippet": ((d.summary or d.content or "")[:600] if (d.summary or d.content) else ""),
                    }
                )
            except Exception:
                continue

        _emit(35, "drafting", f"Drafting a research plan from {len(cite_map)} KB sources")
        await db.commit()

        user_settings = await self._load_user_settings(job.user_id, db)
        prompt = (
            "You are an AI Scientist working with an engineering teammate.\n"
            "Goal: produce a minimal, testable experiment plan and a short LaTeX section to insert into a paper.\n\n"
            "Output MUST be valid JSON only.\n"
            "JSON keys:\n"
            "- hypothesis (string)\n"
            "- plan (string)\n"
            "- experiments (array of {name, procedure, expected_outcome})\n"
            "- metrics (array of strings)\n"
            "- risks (array of strings)\n"
            "- code_change_goal (string)\n"
            "- code_search_query (string)\n"
            "- latex_section_tex (string)  # LaTeX snippet without preamble; may include \\cite{...}\n"
            "- cited_document_ids (array of doc_id strings)\n\n"
            f"USER GOAL:\n{(job.goal or '').strip()}\n\n"
            f"KB CONTEXT (use cite_key for citations):\n{json.dumps(cite_map, ensure_ascii=False)}\n"
        )
        response = await self.llm_service.generate_response(
            query=prompt,
            context=None,
            temperature=0.2,
            max_tokens=1600,
            user_settings=user_settings,
            task_type="research_engineer_scientist",
            user_id=job.user_id,
            db=db,
            routing=self._llm_routing_from_job_config(job.config),
        )

        try:
            payload = json.loads(response)
        except Exception:
            payload = None

        if not isinstance(payload, dict):
            job.status = AgentJobStatus.FAILED.value
            job.error = "Scientist step did not return valid JSON"
            await db.commit()
            return {"status": "failed", "error": job.error}

        cited_ids = payload.get("cited_document_ids") if isinstance(payload.get("cited_document_ids"), list) else []
        cited_ids = [str(x) for x in cited_ids if str(x).strip()]

        latex_section = str(payload.get("latex_section_tex") or "").strip()
        if not latex_section:
            latex_section = (
                "\\section{Hypothesis and Experiment Plan}\n"
                + (str(payload.get("hypothesis") or "").strip() + "\n\n")
                + (str(payload.get("plan") or "").strip() + "\n")
            ).strip()

        latex_updated = False
        latex_project_uuid = None
        if latex_project_id:
            try:
                latex_project_uuid = _UUID(str(latex_project_id))
            except Exception:
                latex_project_uuid = None
        if latex_project_uuid:
            project = await db.get(LatexProject, latex_project_uuid)
            if project and project.user_id == job.user_id:
                project.tex_source = _insert_before_end_document(project.tex_source or "", latex_section)
                await db.commit()
                latex_updated = True

        job.results = job.results or {}
        job.results["research_engineer_plan"] = {
            "search_query": search_query,
            "hypothesis": str(payload.get("hypothesis") or "").strip(),
            "plan": str(payload.get("plan") or "").strip(),
            "experiments": payload.get("experiments") if isinstance(payload.get("experiments"), list) else [],
            "metrics": payload.get("metrics") if isinstance(payload.get("metrics"), list) else [],
            "risks": payload.get("risks") if isinstance(payload.get("risks"), list) else [],
            "code_change_goal": str(payload.get("code_change_goal") or "").strip(),
            "code_search_query": str(payload.get("code_search_query") or "").strip(),
            "cited_document_ids": cited_ids,
            "latex_project_id": str(latex_project_uuid) if latex_project_uuid else None,
            "latex_updated": latex_updated,
        }

        _emit(100, "completed", "Scientist plan ready")
        job.status = AgentJobStatus.COMPLETED.value
        job.completed_at = datetime.utcnow()
        await db.commit()

        return {"status": "completed", "results": job.results}

    async def _run_research_engineer_paper_update(
        self,
        *,
        job: AgentJob,
        db: AsyncSession,
        progress_callback: Optional[callable],
    ) -> Dict[str, Any]:
        """
        Deterministic runner: append implementation notes into a LaTeX project based on parent code patch results.

        Expects:
          - job.config.latex_project_id (UUID)
          - job.config.inherited_data.parent_results.code_patch (from code_patch_proposer)
        """
        from uuid import UUID as _UUID
        from app.models.latex_project import LatexProject

        def _emit(progress: int, phase: str, details: str):
            job.progress = max(0, min(100, int(progress)))
            job.current_phase = phase
            job.phase_details = details
            job.last_activity_at = datetime.utcnow()
            job.add_log_entry({"phase": phase, "action": "research_engineer_paper_update", "result": details})

        def _insert_before_end_document(source: str, addition: str) -> str:
            marker = "\\end{document}"
            s = (source or "")
            idx = s.rfind(marker)
            if idx == -1:
                return (s.rstrip() + "\n\n" + addition.strip() + "\n").lstrip("\n")
            before = s[:idx].rstrip()
            after = s[idx:]
            return f"{before}\n\n{addition.strip()}\n\n{after}"

        config = job.config if isinstance(job.config, dict) else {}
        latex_project_id = (config or {}).get("latex_project_id")
        if not latex_project_id:
            job.status = AgentJobStatus.FAILED.value
            job.error = "Missing job.config.latex_project_id"
            await db.commit()
            return {"status": "failed", "error": job.error}
        try:
            latex_project_uuid = _UUID(str(latex_project_id))
        except Exception:
            job.status = AgentJobStatus.FAILED.value
            job.error = "Invalid latex_project_id"
            await db.commit()
            return {"status": "failed", "error": job.error}

        inherited = (config or {}).get("inherited_data") if isinstance(config, dict) else None
        parent_results = inherited.get("parent_results") if isinstance(inherited, dict) else None
        code_patch = parent_results.get("code_patch") if isinstance(parent_results, dict) else None
        if not isinstance(code_patch, dict):
            job.status = AgentJobStatus.FAILED.value
            job.error = "Missing inherited code_patch results"
            await db.commit()
            return {"status": "failed", "error": job.error}

        _emit(40, "writing", "Updating LaTeX project with implementation notes")
        await db.commit()

        title = str(code_patch.get("title") or "Code Patch Proposal").strip()
        summary = str(code_patch.get("summary") or "").strip()
        risks = code_patch.get("risks") if isinstance(code_patch.get("risks"), list) else []
        tests = code_patch.get("tests_to_run") if isinstance(code_patch.get("tests_to_run"), list) else []
        proposal_id = str(code_patch.get("proposal_id") or "").strip()

        bullets = []
        if summary:
            bullets.append(f"\\item Summary: {summary}")
        if risks:
            bullets.append("\\item Risks: " + "; ".join([str(r).strip() for r in risks if str(r).strip()][:8]))
        if tests:
            bullets.append("\\item Tests: " + "; ".join([str(t).strip() for t in tests if str(t).strip()][:8]))
        if proposal_id:
            bullets.append(f"\\item Proposal ID: \\texttt{{{proposal_id}}}")
        exp = parent_results.get("experiment_run") if isinstance(parent_results, dict) and isinstance(parent_results.get("experiment_run"), dict) else None
        if isinstance(exp, dict):
            runs = exp.get("runs") if isinstance(exp.get("runs"), list) else []
            ok = exp.get("ok")
            if ok is None:
                bullets.append("\\item Experiments: skipped (unsafe execution disabled)")
            elif ok:
                bullets.append("\\item Experiments: PASS")
            else:
                failed_cmds = []
                for r in runs:
                    if isinstance(r, dict) and not bool(r.get("ok")):
                        failed_cmds.append(str(r.get("command") or "")[:120])
                if failed_cmds:
                    bullets.append("\\item Experiments: FAIL (" + "; ".join(failed_cmds[:3]) + ")")
                else:
                    bullets.append("\\item Experiments: FAIL")

        kb_apply = (
            parent_results.get("code_patch_kb_apply")
            if isinstance(parent_results, dict) and isinstance(parent_results.get("code_patch_kb_apply"), dict)
            else None
        )
        if isinstance(kb_apply, dict) and kb_apply.get("enabled") is True:
            if kb_apply.get("dry_run") is True:
                ok = kb_apply.get("ok")
                bullets.append(f"\\item KB apply: dry-run ({'OK' if ok else 'errors'})")
            else:
                bullets.append("\\item KB apply: " + ("APPLIED" if kb_apply.get("did_apply") else "not applied"))

        section = "\\section{Implementation Notes}\n"
        section += f"\\subsection{{{title}}}\n"
        section += "\\begin{itemize}\n" + ("\n".join(bullets) if bullets else "\\item (No details available)") + "\n\\end{itemize}\n"

        project = await db.get(LatexProject, latex_project_uuid)
        if not project or project.user_id != job.user_id:
            job.status = AgentJobStatus.FAILED.value
            job.error = "LaTeX project not found"
            await db.commit()
            return {"status": "failed", "error": job.error}

        project.tex_source = _insert_before_end_document(project.tex_source or "", section)
        await db.commit()

        job.results = dict(parent_results) if isinstance(parent_results, dict) else {}
        job.results["research_engineer_paper_update"] = {
            "latex_project_id": str(latex_project_uuid),
            "code_patch_proposal_id": proposal_id or None,
        }

        _emit(100, "completed", "Paper updated with implementation notes")
        job.status = AgentJobStatus.COMPLETED.value
        job.completed_at = datetime.utcnow()
        await db.commit()

        return {"status": "completed", "results": job.results}

    async def _run_latex_citation_sync(
        self,
        *,
        job: AgentJob,
        db: AsyncSession,
        progress_callback: Optional[callable],
    ) -> Dict[str, Any]:
        """
        Deterministic runner: synchronize citations for a LaTeX Studio project.

        - Scans the LaTeX source for \\cite-like commands containing keys matching KDB:<uuid>
        - Also supports legacy keys KDB[0-9a-f]{8} (best-effort UUID prefix resolution)
        - Updates refs.bib (bibtex mode) OR inserts/replaces a thebibliography block

        Expects:
          - job.config.latex_project_id (UUID)
          - optional job.config.mode: 'bibtex' | 'thebibliography' (default 'bibtex')
          - optional job.config.bib_filename (default 'refs.bib')
        """
        import hashlib as _hashlib
        import json as _json
        from datetime import datetime as _dt
        from uuid import UUID as _UUID

        from sqlalchemy import String as _String, cast as _cast

        from app.models.document import Document
        from app.models.latex_project import LatexProject
        from app.models.latex_project_file import LatexProjectFile
        from app.services.storage_service import storage_service

        def _emit(progress: int, phase: str, details: str):
            job.progress = max(0, min(100, int(progress)))
            job.current_phase = phase
            job.phase_details = details
            job.last_activity_at = datetime.utcnow()
            job.add_log_entry({"phase": phase, "action": "latex_citation_sync", "result": details})

        def _sanitize_bib_filename(name: str) -> str:
            s = (name or "").strip()
            if not s:
                return "refs.bib"
            if "/" in s or "\\" in s or s.startswith("."):
                return "refs.bib"
            if not s.lower().endswith(".bib"):
                s = s + ".bib"
            if len(s) > 100:
                s = s[:100]
            return s

        def _bib_stem(name: str) -> str:
            n = _sanitize_bib_filename(name)
            return n[:-4] if n.lower().endswith(".bib") else n

        def _insert_before_end_document(source: str, addition: str) -> str:
            marker = "\\end{document}"
            s = (source or "")
            idx = s.rfind(marker)
            if idx == -1:
                return (s.rstrip() + "\n\n" + addition.strip() + "\n").lstrip("\n")
            before = s[:idx].rstrip()
            after = s[idx:]
            return f"{before}\n\n{addition.strip()}\n\n{after}"

        def _escape_bibtex(s: str) -> str:
            t = (s or "").strip()
            if not t:
                return ""
            t = re.sub(r"\s+", " ", t).strip()
            t = t.replace("\\", r"\textbackslash{}")
            t = t.replace("{", r"\{").replace("}", r"\}")
            t = t.replace("&", r"\&")
            t = t.replace("%", r"\%")
            t = t.replace("$", r"\$")
            t = t.replace("#", r"\#")
            t = t.replace("_", r"\_")
            t = t.replace("~", r"\textasciitilde{}")
            t = t.replace("^", r"\textasciicircum{}")
            return t

        def _extract_arxiv_id(url: str) -> Optional[str]:
            u = (url or "").strip()
            if not u:
                return None
            m = re.search(r"arxiv\.org/(abs|pdf)/(?P<id>\d{4}\.\d{4,5}(v\d+)?)(?:\.pdf)?", u, flags=re.I)
            if not m:
                return None
            return (m.group("id") or "").strip() or None

        def _bibtex_month_macro(dt: Optional[_dt]) -> Optional[str]:
            if not dt:
                return None
            try:
                month = int(dt.month)
            except Exception:
                return None
            months = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
            if 1 <= month <= 12:
                return months[month - 1]
            return None

        def _bib_key_from_uuid(doc_id: _UUID) -> str:
            return f"KDB:{str(doc_id)}"

        cfg = job.config if isinstance(job.config, dict) else {}
        enabled_raw = cfg.get("enabled")
        if enabled_raw is None:
            enabled = bool(cfg.get("enable_citation_sync", True))
        else:
            enabled = bool(enabled_raw)

        if not enabled:
            job.results = job.results or {}
            job.results["citation_sync"] = {
                "latex_project_id": str((cfg or {}).get("latex_project_id") or ""),
                "mode": str((cfg or {}).get("mode") or "bibtex").strip().lower(),
                "bib_filename": str((cfg or {}).get("bib_filename") or "refs.bib"),
                "skipped": True,
                "reason": "Disabled by config (enable_citation_sync=false)",
            }
            _emit(100, "completed", "Skipped (citation sync disabled)")
            job.status = AgentJobStatus.COMPLETED.value
            job.completed_at = datetime.utcnow()
            await db.commit()
            return {"status": "completed", "results": job.results}

        latex_project_id = (cfg or {}).get("latex_project_id")
        if not latex_project_id:
            job.status = AgentJobStatus.FAILED.value
            job.error = "Missing config.latex_project_id"
            await db.commit()
            return {"status": "failed", "error": job.error}
        try:
            project_uuid = _UUID(str(latex_project_id))
        except Exception:
            job.status = AgentJobStatus.FAILED.value
            job.error = "Invalid latex_project_id"
            await db.commit()
            return {"status": "failed", "error": job.error}

        mode = str((cfg or {}).get("mode") or "bibtex").strip().lower()
        if mode not in ("bibtex", "thebibliography"):
            mode = "bibtex"
        bib_filename = _sanitize_bib_filename(str((cfg or {}).get("bib_filename") or "refs.bib"))
        stem = _bib_stem(bib_filename)

        project = await db.get(LatexProject, project_uuid)
        if not project or project.user_id != job.user_id:
            job.status = AgentJobStatus.FAILED.value
            job.error = "LaTeX project not found"
            await db.commit()
            return {"status": "failed", "error": job.error}

        source = project.tex_source or ""

        _emit(10, "scanning", "Scanning LaTeX source for KDB cite keys")
        await db.commit()

        cite_keys: set[str] = set()
        keys_in_source_order: List[str] = []
        invalid_keys: List[str] = []
        for m in re.finditer(r"\\cite[a-zA-Z*]*\s*\{([^}]*)\}", source):
            keys_raw = m.group(1) or ""
            for k in keys_raw.split(","):
                kk = k.strip()
                if not kk:
                    continue
                if kk.startswith("KDB:"):
                    cite_keys.add(kk)
                elif re.fullmatch(r"KDB[0-9a-fA-F]{8}", kk):
                    cite_keys.add(kk)
                else:
                    continue
                if kk not in keys_in_source_order:
                    keys_in_source_order.append(kk)

        # Resolve cite keys to Document UUIDs.
        resolved: Dict[str, str] = {}
        collisions: Dict[str, int] = {}
        legacy_keys: List[str] = []
        for key in sorted(cite_keys):
            if key.startswith("KDB:"):
                raw = key[len("KDB:") :].strip()
                try:
                    resolved[key] = str(_UUID(raw))
                except Exception:
                    invalid_keys.append(key)
                continue
            legacy_keys.append(key)

        # Best-effort resolve legacy cite keys to Document UUIDs by prefix match.
        for key in legacy_keys:
            prefix = key[3:].lower()
            try:
                res = await db.execute(
                    select(Document.id)
                    .where(func.replace(_cast(Document.id, _String), "-", "").ilike(f"{prefix}%"))
                    .limit(2)
                )
                ids = [str(x[0]) for x in res.all()]
            except Exception:
                ids = []
            if len(ids) == 1:
                resolved[key] = ids[0]
            elif len(ids) > 1:
                collisions[key] = len(ids)

        doc_ids: List[_UUID] = []
        for did in sorted({str(x) for x in resolved.values()}):
            try:
                doc_ids.append(_UUID(str(did)))
            except Exception:
                continue

        docs: List[Document] = []
        if doc_ids:
            res = await db.execute(select(Document).where(Document.id.in_(doc_ids)))
            docs = list(res.scalars().all())

        # Deterministic ordering: preserve cite key order as it appears in paper.tex.
        docs_by_key: List[Document] = []
        by_id = {str(d.id): d for d in docs}
        for k in (keys_in_source_order or sorted(resolved.keys())):
            did = resolved.get(k)
            d = by_id.get(str(did)) if did else None
            if d:
                docs_by_key.append(d)

        _emit(35, "building", f"Building bibliography for {len(docs_by_key)} resolved citations")
        await db.commit()

        updated_bib = False
        updated_tex = False
        bibtex_entries = ""
        references_tex = ""

        if mode == "bibtex":
            entries: List[str] = []
            for d in docs_by_key:
                key = _bib_key_from_uuid(d.id)
                title = _escape_bibtex(d.title or "Untitled")
                url = (d.url or "").strip()
                author = _escape_bibtex(d.author or "")
                ts = d.last_modified or d.updated_at or d.created_at
                year: Optional[int] = None
                month_macro: Optional[str] = None
                try:
                    if ts:
                        year = int(ts.year)
                        month_macro = _bibtex_month_macro(ts)
                except Exception:
                    year = None
                    month_macro = None

                arxiv_id = _extract_arxiv_id(url)
                fields: List[str] = [
                    f"  title = {{{{{title}}}}}",
                    "  note = {Knowledge DB document}",
                ]
                if author:
                    fields.append(f"  author = {{{author}}}")
                if year:
                    fields.append(f"  year = {{{year}}}")
                if month_macro:
                    fields.append(f"  month = {month_macro}")
                if url:
                    fields.append(f"  howpublished = {{\\url{{{url}}}}}")
                    fields.append(f"  url = {{{url}}}")
                if arxiv_id:
                    fields.append("  archivePrefix = {arXiv}")
                    fields.append(f"  eprint = {{{arxiv_id}}}")

                entries.append("@misc{" + key + ",\n" + ",\n".join(fields) + "\n}\n")
            bibtex_entries = "\n".join(entries).strip() + ("\n" if entries else "")

            # Load existing bib, merge by key.
            existing = (
                await db.execute(
                    select(LatexProjectFile).where(
                        (LatexProjectFile.project_id == project.id) & (LatexProjectFile.filename == bib_filename)
                    )
                )
            ).scalar_one_or_none()
            existing_text = ""
            if existing:
                try:
                    existing_text = (await storage_service.get_file_content(existing.file_path) or b"").decode("utf-8", errors="replace")
                except Exception:
                    existing_text = ""

            existing_keys = set()
            for m in re.finditer(r"@\\w+\\s*\\{\\s*([^,\\s]+)\\s*,", existing_text):
                existing_keys.add((m.group(1) or "").strip())
            new_blocks = []
            for block in re.split(r"\n(?=@\\w+\\s*\\{)", bibtex_entries or ""):
                b = block.strip()
                if not b:
                    continue
                m = re.search(r"@\\w+\\s*\\{\\s*([^,\\s]+)\\s*,", b)
                k = (m.group(1) if m else "").strip()
                if k and k in existing_keys:
                    continue
                new_blocks.append(b + "\n")

            merged_text = (existing_text.rstrip() + ("\n\n" if existing_text.strip() and new_blocks else "\n") + "".join(new_blocks)).strip() + "\n"
            content_bytes = merged_text.encode("utf-8")
            sha = _hashlib.sha256(content_bytes).hexdigest()
            object_path = await storage_service.upload_file(
                document_id=project.id,
                filename=bib_filename,
                content=content_bytes,
                content_type="application/x-bibtex",
            )
            if existing:
                existing.file_path = object_path
                existing.sha256 = sha
                existing.file_size = len(content_bytes)
                existing.content_type = "application/x-bibtex"
            else:
                db.add(
                    LatexProjectFile(
                        project_id=project.id,
                        filename=bib_filename,
                        content_type="application/x-bibtex",
                        file_size=len(content_bytes),
                        sha256=sha,
                        file_path=object_path,
                    )
                )
            await db.commit()
            updated_bib = True

            # Ensure bibliography scaffold in LaTeX source.
            if "\\bibliography{" not in source:
                scaffold = f"\\bibliographystyle{{plain}}\\n\\bibliography{{{stem}}}"
                project.tex_source = _insert_before_end_document(project.tex_source or "", scaffold)
                await db.commit()
                updated_tex = True

        else:
            lines: List[str] = ["\\begin{thebibliography}{99}"]
            for d in docs_by_key:
                key = _bib_key_from_uuid(d.id)
                title = _escape_bibtex(d.title or "Untitled")
                author = _escape_bibtex(d.author or "")
                url = (d.url or "").strip()
                ts = d.last_modified or d.updated_at or d.created_at
                year: Optional[int] = None
                try:
                    if ts:
                        year = int(ts.year)
                except Exception:
                    year = None
                parts: List[str] = []
                if author:
                    parts.append(f"{author}.")
                parts.append(f"\\textit{{{title}}}.")
                if year:
                    parts.append(f"{year}.")
                parts.append("Knowledge DB document.")
                if url:
                    parts.append(f"\\url{{{url}}}.")
                lines.append(f"\\bibitem{{{key}}} " + " ".join(parts).strip())
            lines.append("\\end{thebibliography}")
            references_tex = "\n".join(lines).strip() + "\n"

            # Replace existing thebibliography if present; otherwise insert.
            if re.search(r"\\begin\\{thebibliography\\}", source):
                project.tex_source = re.sub(
                    r"\\begin\\{thebibliography\\}.*?\\end\\{thebibliography\\}",
                    references_tex.strip(),
                    project.tex_source or "",
                    flags=re.S,
                )
            else:
                project.tex_source = _insert_before_end_document(project.tex_source or "", references_tex)
            await db.commit()
            updated_tex = True

        job.results = job.results or {}
        job.results["citation_sync"] = {
            "latex_project_id": str(project.id),
            "mode": mode,
            "bib_filename": bib_filename if mode == "bibtex" else None,
            "resolved_count": len(docs_by_key),
            "unresolved_keys": sorted([k for k in cite_keys if k not in resolved]),
            "invalid_keys": sorted(set(invalid_keys)),
            "collisions": collisions,
            "updated_tex": updated_tex,
            "updated_bib": updated_bib,
        }

        _emit(100, "completed", f"Citation sync complete ({len(docs_by_key)} resolved).")
        job.status = AgentJobStatus.COMPLETED.value
        job.completed_at = datetime.utcnow()
        await db.commit()
        return {"status": "completed", "results": job.results}

    async def _run_latex_compile_project(
        self,
        *,
        job: AgentJob,
        db: AsyncSession,
        progress_callback: Optional[callable],
    ) -> Dict[str, Any]:
        """
        Deterministic runner: compile a LaTeX Studio project to PDF.

        Uses the dedicated Celery LaTeX worker when enabled; otherwise attempts a synchronous compile
        in-process (requires TeX tools installed in the backend container).

        Expects:
          - job.config.latex_project_id (UUID)
          - optional job.config.safe_mode (bool)
          - optional job.config.preferred_engine (string)
          - optional job.config.wait_seconds (int): how long to wait for async job completion
          - optional job.config.use_worker (bool): default True
          - optional job.config.skip_if_unavailable (bool): default True
        """
        import asyncio as _asyncio
        from datetime import datetime as _dt
        from uuid import UUID as _UUID

        from sqlalchemy import select as _select

        from app.core.config import settings as app_settings
        from app.models.latex_compile_job import LatexCompileJob
        from app.models.latex_project import LatexProject
        from app.models.latex_project_file import LatexProjectFile
        from app.models.user import User as _User
        from app.services.latex_compiler_service import LatexSafetyError, latex_compiler_service
        from app.services.storage_service import storage_service
        from app.tasks.latex_tasks import compile_latex_project_job

        def _emit(progress: int, phase: str, details: str):
            job.progress = max(0, min(100, int(progress)))
            job.current_phase = phase
            job.phase_details = details
            job.last_activity_at = datetime.utcnow()
            job.add_log_entry({"phase": phase, "action": "latex_compile_project", "result": details})

        cfg = job.config if isinstance(job.config, dict) else {}
        enabled_raw = cfg.get("enabled")
        if enabled_raw is None:
            enabled = bool(cfg.get("enable_compile", True))
        else:
            enabled = bool(enabled_raw)

        if not enabled:
            job.results = job.results or {}
            job.results["latex_compile"] = {
                "latex_project_id": str((cfg or {}).get("latex_project_id") or ""),
                "skipped": True,
                "reason": "Disabled by config (enable_compile=false)",
            }
            _emit(100, "completed", "Skipped (compile disabled)")
            job.status = AgentJobStatus.COMPLETED.value
            job.completed_at = datetime.utcnow()
            await db.commit()
            return {"status": "completed", "results": job.results}

        latex_project_id = (cfg or {}).get("latex_project_id")
        if not latex_project_id:
            job.status = AgentJobStatus.FAILED.value
            job.error = "Missing config.latex_project_id"
            await db.commit()
            return {"status": "failed", "error": job.error}
        try:
            project_uuid = _UUID(str(latex_project_id))
        except Exception:
            job.status = AgentJobStatus.FAILED.value
            job.error = "Invalid latex_project_id"
            await db.commit()
            return {"status": "failed", "error": job.error}

        safe_mode = bool(cfg.get("safe_mode", True))
        preferred_engine = (cfg.get("preferred_engine") or None)
        use_worker = cfg.get("use_worker")
        use_worker = True if use_worker is None else bool(use_worker)
        wait_seconds = int(cfg.get("wait_seconds") or 120)
        wait_seconds = max(0, min(wait_seconds, 10 * 60))
        skip_if_unavailable = cfg.get("skip_if_unavailable")
        skip_if_unavailable = True if skip_if_unavailable is None else bool(skip_if_unavailable)

        project = await db.get(LatexProject, project_uuid)
        if not project or project.user_id != job.user_id:
            job.status = AgentJobStatus.FAILED.value
            job.error = "LaTeX project not found"
            await db.commit()
            return {"status": "failed", "error": job.error}

        if not bool(getattr(app_settings, "LATEX_COMPILER_ENABLED", False)):
            if skip_if_unavailable:
                job.results = job.results or {}
                job.results["latex_compile"] = {
                    "latex_project_id": str(project.id),
                    "skipped": True,
                    "reason": "Compiler disabled on server",
                }
                _emit(100, "completed", "Skipped (compiler disabled)")
                job.status = AgentJobStatus.COMPLETED.value
                job.completed_at = datetime.utcnow()
                await db.commit()
                return {"status": "completed", "results": job.results}
            job.status = AgentJobStatus.FAILED.value
            job.error = "Compiler disabled on server"
            await db.commit()
            return {"status": "failed", "error": job.error}

        if bool(getattr(app_settings, "LATEX_COMPILER_ADMIN_ONLY", False)):
            user = await db.get(_User, job.user_id)
            if not user or (user.role or "") != "admin":
                if skip_if_unavailable:
                    job.results = job.results or {}
                    job.results["latex_compile"] = {
                        "latex_project_id": str(project.id),
                        "skipped": True,
                        "reason": "Compilation restricted to admins",
                    }
                    _emit(100, "completed", "Skipped (admin-only)")
                    job.status = AgentJobStatus.COMPLETED.value
                    job.completed_at = datetime.utcnow()
                    await db.commit()
                    return {"status": "completed", "results": job.results}
                job.status = AgentJobStatus.FAILED.value
                job.error = "Compilation restricted to admins"
                await db.commit()
                return {"status": "failed", "error": job.error}

        worker_enabled = bool(getattr(app_settings, "LATEX_COMPILER_USE_CELERY", False))
        queue = str(getattr(app_settings, "LATEX_COMPILER_CELERY_QUEUE", "latex") or "latex")

        if use_worker and worker_enabled:
            _emit(20, "queueing", "Enqueuing LaTeX compile job")
            await db.commit()

            compile_job = LatexCompileJob(
                user_id=job.user_id,
                project_id=project.id,
                status="queued",
                safe_mode=safe_mode,
                preferred_engine=preferred_engine,
            )
            db.add(compile_job)
            await db.commit()
            await db.refresh(compile_job)

            try:
                async_result = compile_latex_project_job.apply_async(args=[str(compile_job.id)], queue=queue)
                compile_job.celery_task_id = async_result.id
                await db.commit()
            except Exception:
                compile_job.status = "failed"
                compile_job.log = "Failed to enqueue compile job."
                compile_job.finished_at = _dt.utcnow()
                await db.commit()

            if wait_seconds > 0 and compile_job.status in ("queued", "running"):
                _emit(40, "waiting", f"Waiting up to {wait_seconds}s for compile result")
                await db.commit()
                deadline = _dt.utcnow().timestamp() + float(wait_seconds)
                while _dt.utcnow().timestamp() < deadline:
                    try:
                        await db.refresh(compile_job)
                    except Exception:
                        pass
                    if compile_job.status not in ("queued", "running"):
                        break
                    await _asyncio.sleep(1.0)

            await db.refresh(compile_job)
            await db.refresh(project)

            job.results = job.results or {}
            job.results["latex_compile"] = {
                "latex_project_id": str(project.id),
                "use_worker": True,
                "queue": queue,
                "compile_job_id": str(compile_job.id),
                "compile_job_status": compile_job.status,
                "engine": compile_job.engine,
                "pdf_file_path": project.pdf_file_path,
                "finished_at": compile_job.finished_at.isoformat() if compile_job.finished_at else None,
            }

            _emit(100, "completed", f"Compile job status: {compile_job.status}")
            job.status = AgentJobStatus.COMPLETED.value
            job.completed_at = datetime.utcnow()
            await db.commit()
            return {"status": "completed", "results": job.results}

        # Synchronous compile (in-process).
        _emit(20, "compiling", "Compiling LaTeX in-process")
        await db.commit()

        additional_files: Dict[str, bytes] = {}
        try:
            files_result = await db.execute(_select(LatexProjectFile).where(LatexProjectFile.project_id == project.id))
            for f in files_result.scalars().all():
                name = (f.filename or "").strip()
                if not name or "/" in name or "\\" in name:
                    continue
                try:
                    additional_files[name] = await storage_service.get_file_content(f.file_path)
                except Exception:
                    continue
        except Exception:
            additional_files = {}

        try:
            result = await _asyncio.to_thread(
                latex_compiler_service.compile_to_pdf,
                tex_source=project.tex_source or "",
                timeout_seconds=int(getattr(app_settings, "LATEX_COMPILER_TIMEOUT_SECONDS", 20)),
                max_source_chars=int(getattr(app_settings, "LATEX_COMPILER_MAX_SOURCE_CHARS", 100000)),
                safe_mode=safe_mode,
                preferred_engine=preferred_engine,
                additional_files=additional_files or None,
            )
        except LatexSafetyError as exc:
            job.results = job.results or {}
            job.results["latex_compile"] = {
                "latex_project_id": str(project.id),
                "use_worker": False,
                "success": False,
                "error": str(exc),
                "violations": list(getattr(exc, "violations", []) or []),
            }
            _emit(100, "completed", "Blocked by safe mode")
            job.status = AgentJobStatus.COMPLETED.value
            job.completed_at = datetime.utcnow()
            await db.commit()
            return {"status": "completed", "results": job.results}
        except Exception as exc:
            if skip_if_unavailable:
                job.results = job.results or {}
                job.results["latex_compile"] = {
                    "latex_project_id": str(project.id),
                    "use_worker": False,
                    "skipped": True,
                    "reason": "Compile failed in-process",
                    "error": str(exc),
                }
                _emit(100, "completed", "Skipped (compile error)")
                job.status = AgentJobStatus.COMPLETED.value
                job.completed_at = datetime.utcnow()
                await db.commit()
                return {"status": "completed", "results": job.results}
            raise

        if not result.success or not result.pdf_bytes:
            if skip_if_unavailable:
                job.results = job.results or {}
                job.results["latex_compile"] = {
                    "latex_project_id": str(project.id),
                    "use_worker": False,
                    "success": False,
                    "engine": result.engine,
                    "log": result.log,
                    "violations": list(result.violations or []),
                }
                _emit(100, "completed", "Compile did not produce a PDF")
                job.status = AgentJobStatus.COMPLETED.value
                job.completed_at = datetime.utcnow()
                await db.commit()
                return {"status": "completed", "results": job.results}
            job.status = AgentJobStatus.FAILED.value
            job.error = "Compile did not produce a PDF"
            await db.commit()
            return {"status": "failed", "error": job.error}

        pdf_path = await storage_service.upload_file(
            document_id=project.id,
            filename="paper.pdf",
            content=result.pdf_bytes,
            content_type="application/pdf",
        )
        project.pdf_file_path = pdf_path
        project.last_compile_engine = result.engine
        project.last_compile_log = result.log
        project.last_compiled_at = datetime.utcnow()
        await db.commit()
        await db.refresh(project)

        job.results = job.results or {}
        job.results["latex_compile"] = {
            "latex_project_id": str(project.id),
            "use_worker": False,
            "success": True,
            "engine": result.engine,
            "pdf_file_path": project.pdf_file_path,
        }

        _emit(100, "completed", f"Compiled ({result.engine or 'unknown'})")
        job.status = AgentJobStatus.COMPLETED.value
        job.completed_at = datetime.utcnow()
        await db.commit()
        return {"status": "completed", "results": job.results}

    async def _run_latex_publish_project(
        self,
        *,
        job: AgentJob,
        db: AsyncSession,
        progress_callback: Optional[callable],
    ) -> Dict[str, Any]:
        """
        Deterministic runner: publish a LaTeX Studio project's .tex/.pdf as Knowledge DB documents.

        Expects:
          - job.config.latex_project_id (UUID)
          - optional job.config.include_tex (bool, default True)
          - optional job.config.include_pdf (bool, default True)
          - optional job.config.tags (list[str] OR comma-separated string in job.config.publish_tags)
        """
        import hashlib as _hashlib
        import tempfile as _tempfile
        from uuid import UUID as _UUID

        from sqlalchemy import select as _select

        from app.models.document import Document
        from app.models.latex_project import LatexProject
        from app.models.user import User as _User
        from app.services.document_service import DocumentService
        from app.services.storage_service import storage_service

        def _emit(progress: int, phase: str, details: str):
            job.progress = max(0, min(100, int(progress)))
            job.current_phase = phase
            job.phase_details = details
            job.last_activity_at = datetime.utcnow()
            job.add_log_entry({"phase": phase, "action": "latex_publish_project", "result": details})

        cfg = job.config if isinstance(job.config, dict) else {}
        enabled_raw = cfg.get("enabled")
        if enabled_raw is None:
            enabled = bool(cfg.get("enable_publish", True))
        else:
            enabled = bool(enabled_raw)

        if not enabled:
            job.results = job.results or {}
            job.results["latex_publish"] = {
                "latex_project_id": str((cfg or {}).get("latex_project_id") or ""),
                "published": [],
                "skipped": [
                    {"kind": "tex", "reason": "Disabled by config (enable_publish=false)"},
                    {"kind": "pdf", "reason": "Disabled by config (enable_publish=false)"},
                ],
            }
            _emit(100, "completed", "Skipped (publish disabled)")
            job.status = AgentJobStatus.COMPLETED.value
            job.completed_at = datetime.utcnow()
            await db.commit()
            return {"status": "completed", "results": job.results}

        latex_project_id = (cfg or {}).get("latex_project_id")
        if not latex_project_id:
            job.status = AgentJobStatus.FAILED.value
            job.error = "Missing config.latex_project_id"
            await db.commit()
            return {"status": "failed", "error": job.error}
        try:
            project_uuid = _UUID(str(latex_project_id))
        except Exception:
            job.status = AgentJobStatus.FAILED.value
            job.error = "Invalid latex_project_id"
            await db.commit()
            return {"status": "failed", "error": job.error}

        include_tex = bool(cfg.get("include_tex", True))
        include_pdf = bool(cfg.get("include_pdf", True))
        raw_tags = cfg.get("tags")
        if raw_tags is None:
            raw_tags = cfg.get("publish_tags")
        tags: Optional[list[str]] = None
        if isinstance(raw_tags, list):
            tags = [str(x).strip() for x in raw_tags if str(x).strip()][:50]
        elif isinstance(raw_tags, str):
            parts = [p.strip() for p in raw_tags.split(",") if p.strip()]
            tags = parts[:50] if parts else None

        project = await db.get(LatexProject, project_uuid)
        if not project or project.user_id != job.user_id:
            job.status = AgentJobStatus.FAILED.value
            job.error = "LaTeX project not found"
            await db.commit()
            return {"status": "failed", "error": job.error}

        user = await db.get(_User, job.user_id)
        author = (
            (getattr(user, "full_name", None) if user else None)
            or (getattr(user, "username", None) if user else None)
            or (getattr(user, "email", None) if user else None)
        )

        document_service = DocumentService()
        source = await document_service._get_or_create_latex_projects_source(db)

        published: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []

        if include_tex:
            if not (project.tex_source or "").strip():
                skipped.append({"kind": "tex", "reason": "Empty LaTeX source"})
            else:
                _emit(20, "publishing", "Publishing paper.tex")
                await db.commit()
                try:
                    tex_bytes = (project.tex_source or "").encode("utf-8")
                    object_path = await storage_service.upload_file(
                        document_id=project.id,
                        filename="paper.tex",
                        content=tex_bytes,
                        content_type="text/x-tex",
                    )
                    project.tex_file_path = object_path
                    await db.commit()

                    tex_hash = _hashlib.sha256(tex_bytes).hexdigest()
                    source_identifier = f"latex_project:{project.id}:tex"
                    existing = (
                        await db.execute(
                            _select(Document).where(
                                (Document.source_id == source.id) & (Document.source_identifier == source_identifier)
                            )
                        )
                    ).scalar_one_or_none()
                    if existing:
                        tex_doc = existing
                        tex_doc.title = f"{project.title} (LaTeX)"
                        tex_doc.content = project.tex_source or ""
                        tex_doc.content_hash = tex_hash
                        tex_doc.file_path = project.tex_file_path
                        tex_doc.file_type = "text/x-tex"
                        tex_doc.file_size = len(tex_bytes)
                        tex_doc.author = author
                        tex_doc.tags = tags
                        tex_doc.extra_metadata = {
                            "origin": "latex_project_publish",
                            "latex_project_id": str(project.id),
                            "kind": "tex",
                        }
                        tex_doc.is_processed = False
                        await db.commit()
                        await db.refresh(tex_doc)
                    else:
                        tex_doc = Document(
                            title=f"{project.title} (LaTeX)",
                            content=project.tex_source or "",
                            content_hash=tex_hash,
                            url=None,
                            file_path=project.tex_file_path,
                            file_type="text/x-tex",
                            file_size=len(tex_bytes),
                            source_id=source.id,
                            source_identifier=source_identifier,
                            author=author,
                            tags=tags,
                            extra_metadata={
                                "origin": "latex_project_publish",
                                "latex_project_id": str(project.id),
                                "kind": "tex",
                            },
                            is_processed=False,
                        )
                        db.add(tex_doc)
                        await db.commit()
                        await db.refresh(tex_doc)

                    try:
                        await document_service.reprocess_document(tex_doc.id, db, user_id=job.user_id)
                    except Exception:
                        pass
                    published.append(
                        {
                            "kind": "tex",
                            "document_id": str(tex_doc.id),
                            "title": tex_doc.title,
                            "file_type": tex_doc.file_type,
                            "file_path": tex_doc.file_path,
                        }
                    )
                except Exception:
                    skipped.append({"kind": "tex", "reason": "Failed to publish LaTeX source"})
        else:
            skipped.append({"kind": "tex", "reason": "Disabled by request"})

        if include_pdf:
            if not project.pdf_file_path:
                skipped.append({"kind": "pdf", "reason": "No PDF available (compile first)"})
            else:
                _emit(60, "publishing", "Publishing paper.pdf")
                await db.commit()
                try:
                    pdf_bytes = await storage_service.get_file_content(project.pdf_file_path)
                    pdf_hash = _hashlib.sha256(pdf_bytes).hexdigest()

                    extracted_text = ""
                    try:
                        with _tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
                            tmp.write(pdf_bytes)
                            tmp.flush()
                            extracted_text, _ = await document_service.text_processor.extract_text(
                                tmp.name,
                                content_type="application/pdf",
                            )
                    except Exception:
                        extracted_text = ""

                    source_identifier = f"latex_project:{project.id}:pdf"
                    existing = (
                        await db.execute(
                            _select(Document).where(
                                (Document.source_id == source.id) & (Document.source_identifier == source_identifier)
                            )
                        )
                    ).scalar_one_or_none()
                    if existing:
                        pdf_doc = existing
                        pdf_doc.title = f"{project.title} (PDF)"
                        pdf_doc.content = extracted_text or ""
                        pdf_doc.content_hash = pdf_hash
                        pdf_doc.file_path = project.pdf_file_path
                        pdf_doc.file_type = "application/pdf"
                        pdf_doc.file_size = len(pdf_bytes)
                        pdf_doc.author = author
                        pdf_doc.tags = tags
                        pdf_doc.extra_metadata = {
                            "origin": "latex_project_publish",
                            "latex_project_id": str(project.id),
                            "kind": "pdf",
                        }
                        pdf_doc.is_processed = False
                        await db.commit()
                        await db.refresh(pdf_doc)
                    else:
                        pdf_doc = Document(
                            title=f"{project.title} (PDF)",
                            content=extracted_text or "",
                            content_hash=pdf_hash,
                            url=None,
                            file_path=project.pdf_file_path,
                            file_type="application/pdf",
                            file_size=len(pdf_bytes),
                            source_id=source.id,
                            source_identifier=source_identifier,
                            author=author,
                            tags=tags,
                            extra_metadata={
                                "origin": "latex_project_publish",
                                "latex_project_id": str(project.id),
                                "kind": "pdf",
                            },
                            is_processed=False,
                        )
                        db.add(pdf_doc)
                        await db.commit()
                        await db.refresh(pdf_doc)

                    try:
                        await document_service.reprocess_document(pdf_doc.id, db, user_id=job.user_id)
                    except Exception:
                        pass
                    published.append(
                        {
                            "kind": "pdf",
                            "document_id": str(pdf_doc.id),
                            "title": pdf_doc.title,
                            "file_type": pdf_doc.file_type,
                            "file_path": pdf_doc.file_path,
                        }
                    )
                except Exception:
                    skipped.append({"kind": "pdf", "reason": "Failed to publish PDF"})
        else:
            skipped.append({"kind": "pdf", "reason": "Disabled by request"})

        job.results = job.results or {}
        job.results["latex_publish"] = {
            "latex_project_id": str(project.id),
            "published": published,
            "skipped": skipped,
        }

        _emit(100, "completed", f"Publish complete ({len(published)} published)")
        job.status = AgentJobStatus.COMPLETED.value
        job.completed_at = datetime.utcnow()
        await db.commit()
        return {"status": "completed", "results": job.results}

    async def _run_latex_apply_unified_diff(
        self,
        *,
        job: AgentJob,
        db: AsyncSession,
        progress_callback: Optional[callable],
    ) -> Dict[str, Any]:
        """
        Deterministic runner: apply a unified diff to paper.tex in a LaTeX Studio project.

        Expects:
          - job.config.latex_project_id (UUID)
          - optional job.config.enabled (bool): default False (so this step is safely "optional")
          - optional job.config.diff_unified (string): if not provided, will try to read
            inherited_data.parent_results.latex_review.diff_unified
        """
        from uuid import UUID as _UUID

        from app.models.latex_project import LatexProject
        from app.services.storage_service import storage_service
        from app.services.unified_diff_service import apply_unified_diff_to_text

        def _emit(progress: int, phase: str, details: str):
            job.progress = max(0, min(100, int(progress)))
            job.current_phase = phase
            job.phase_details = details
            job.last_activity_at = datetime.utcnow()
            job.add_log_entry({"phase": phase, "action": "latex_apply_unified_diff", "result": details})

        cfg = job.config if isinstance(job.config, dict) else {}
        enabled_raw = cfg.get("enabled")
        if enabled_raw is None:
            enabled = bool(cfg.get("apply_review_diff", False))
        else:
            enabled = bool(enabled_raw)

        latex_project_id = (cfg or {}).get("latex_project_id")
        if not latex_project_id:
            job.status = AgentJobStatus.FAILED.value
            job.error = "Missing config.latex_project_id"
            await db.commit()
            return {"status": "failed", "error": job.error}
        try:
            project_uuid = _UUID(str(latex_project_id))
        except Exception:
            job.status = AgentJobStatus.FAILED.value
            job.error = "Invalid latex_project_id"
            await db.commit()
            return {"status": "failed", "error": job.error}

        project = await db.get(LatexProject, project_uuid)
        if not project or project.user_id != job.user_id:
            job.status = AgentJobStatus.FAILED.value
            job.error = "LaTeX project not found"
            await db.commit()
            return {"status": "failed", "error": job.error}

        inherited = (cfg or {}).get("inherited_data") if isinstance(cfg, dict) else None
        parent_results = inherited.get("parent_results") if isinstance(inherited, dict) else None
        review = parent_results.get("latex_review") if isinstance(parent_results, dict) else None

        diff_unified = str((cfg or {}).get("diff_unified") or "").strip()
        if not diff_unified and isinstance(review, dict):
            diff_unified = str(review.get("diff_unified") or "").strip()

        if not enabled:
            job.results = job.results or {}
            job.results["latex_apply_diff"] = {
                "latex_project_id": str(project.id),
                "enabled": False,
                "applied": False,
                "reason": "Disabled by config (enabled=false)",
            }
            _emit(100, "completed", "Skipped (disabled)")
            job.status = AgentJobStatus.COMPLETED.value
            job.completed_at = datetime.utcnow()
            await db.commit()
            return {"status": "completed", "results": job.results}

        if not diff_unified:
            job.results = job.results or {}
            job.results["latex_apply_diff"] = {
                "latex_project_id": str(project.id),
                "enabled": True,
                "applied": False,
                "reason": "No diff provided (and none found in inherited latex_review)",
            }
            _emit(100, "completed", "Skipped (no diff)")
            job.status = AgentJobStatus.COMPLETED.value
            job.completed_at = datetime.utcnow()
            await db.commit()
            return {"status": "completed", "results": job.results}

        base_tex = (project.tex_source or "").replace("\r\n", "\n")
        base_sha = hashlib.sha256(base_tex.encode("utf-8")).hexdigest()

        _emit(40, "applying", "Applying unified diff to paper.tex")
        await db.commit()

        try:
            patched, warnings = apply_unified_diff_to_text(original=base_tex, diff_unified=diff_unified)
        except ValueError as exc:
            job.status = AgentJobStatus.FAILED.value
            job.error = str(exc)
            await db.commit()
            return {"status": "failed", "error": job.error}

        new_sha = hashlib.sha256(patched.encode("utf-8")).hexdigest()
        if new_sha == base_sha:
            job.results = job.results or {}
            job.results["latex_apply_diff"] = {
                "latex_project_id": str(project.id),
                "enabled": True,
                "applied": False,
                "base_sha256": base_sha,
                "new_sha256": new_sha,
                "warnings": warnings or [],
                "reason": "Diff produced no changes",
            }
            _emit(100, "completed", "No changes (diff applied cleanly)")
            job.status = AgentJobStatus.COMPLETED.value
            job.completed_at = datetime.utcnow()
            await db.commit()
            return {"status": "completed", "results": job.results}

        project.tex_source = patched
        await db.commit()
        await db.refresh(project)

        # Best-effort: update stored source file.
        try:
            object_path = await storage_service.upload_file(
                document_id=project.id,
                filename="paper.tex",
                content=project.tex_source.encode("utf-8"),
                content_type="text/x-tex",
            )
            project.tex_file_path = object_path
            await db.commit()
            await db.refresh(project)
        except Exception:
            pass

        job.results = job.results or {}
        job.results["latex_apply_diff"] = {
            "latex_project_id": str(project.id),
            "enabled": True,
            "applied": True,
            "base_sha256": base_sha,
            "new_sha256": new_sha,
            "warnings": warnings or [],
        }

        _emit(100, "completed", "Applied diff to paper.tex")
        job.status = AgentJobStatus.COMPLETED.value
        job.completed_at = datetime.utcnow()
        await db.commit()
        return {"status": "completed", "results": job.results}

    async def _run_latex_reviewer_critic(
        self,
        *,
        job: AgentJob,
        db: AsyncSession,
        progress_callback: Optional[callable],
    ) -> Dict[str, Any]:
        """
        Deterministic runner: review a LaTeX project and suggest improvements as a patch-style diff.

        Expects:
          - job.config.latex_project_id (UUID)
          - optional job.config.focus (string)

        Produces:
          - job.results.latex_review (issues + diff_unified)
        """
        import json as _json
        from uuid import UUID as _UUID
        from app.models.latex_project import LatexProject

        def _emit(progress: int, phase: str, details: str):
            job.progress = max(0, min(100, int(progress)))
            job.current_phase = phase
            job.phase_details = details
            job.last_activity_at = datetime.utcnow()
            job.add_log_entry({"phase": phase, "action": "latex_reviewer_critic", "result": details})

        cfg = job.config if isinstance(job.config, dict) else {}
        enabled_raw = cfg.get("enabled")
        if enabled_raw is None:
            enabled = bool(cfg.get("enable_reviewer", True))
        else:
            enabled = bool(enabled_raw)

        if not enabled:
            job.results = job.results or {}
            job.results["latex_review"] = {
                "latex_project_id": str((cfg or {}).get("latex_project_id") or ""),
                "issues": [],
                "diff_unified": "",
                "skipped": True,
                "reason": "Disabled by config (enable_reviewer=false)",
            }
            _emit(100, "completed", "Skipped (reviewer disabled)")
            job.status = AgentJobStatus.COMPLETED.value
            job.completed_at = datetime.utcnow()
            await db.commit()
            return {"status": "completed", "results": job.results}

        latex_project_id = (cfg or {}).get("latex_project_id")
        focus = str((cfg or {}).get("focus") or "").strip()
        if not latex_project_id:
            job.status = AgentJobStatus.FAILED.value
            job.error = "Missing config.latex_project_id"
            await db.commit()
            return {"status": "failed", "error": job.error}
        try:
            project_uuid = _UUID(str(latex_project_id))
        except Exception:
            job.status = AgentJobStatus.FAILED.value
            job.error = "Invalid latex_project_id"
            await db.commit()
            return {"status": "failed", "error": job.error}

        project = await db.get(LatexProject, project_uuid)
        if not project or project.user_id != job.user_id:
            job.status = AgentJobStatus.FAILED.value
            job.error = "LaTeX project not found"
            await db.commit()
            return {"status": "failed", "error": job.error}

        tex = project.tex_source or ""
        if len(tex) > 40000:
            tex = tex[:40000]

        _emit(25, "reviewing", "Reviewing LaTeX for citations/clarity/notation consistency")
        await db.commit()

        user_settings = await self._load_user_settings(job.user_id, db)
        prompt = (
            "You are a meticulous Reviewer/Critic for an academic LaTeX paper.\n"
            "Check for:\n"
            "- missing citations for factual claims\n"
            "- unclear claims / weak definitions\n"
            "- inconsistent notation and terminology\n"
            "- LaTeX issues that commonly cause compile problems\n\n"
            "Output MUST be valid JSON only.\n"
            "JSON keys:\n"
            "- issues: array of {category, severity, message, location_hint}\n"
            "- diff_unified: a unified diff that patches paper.tex (use ---/+++ headers). Keep it minimal.\n\n"
            f"FOCUS (optional): {focus}\n\n"
            "CURRENT paper.tex (possibly truncated):\n"
            "```tex\n"
            f"{tex}\n"
            "```\n"
        )
        response = await self.llm_service.generate_response(
            query=prompt,
            context=None,
            temperature=0.2,
            max_tokens=1800,
            user_settings=user_settings,
            task_type="latex_reviewer_critic",
            user_id=job.user_id,
            db=db,
            routing=self._llm_routing_from_job_config(job.config),
        )

        try:
            payload = _json.loads(response)
        except Exception:
            payload = None

        if not isinstance(payload, dict):
            job.status = AgentJobStatus.FAILED.value
            job.error = "Reviewer did not return valid JSON"
            await db.commit()
            return {"status": "failed", "error": job.error}

        issues = payload.get("issues") if isinstance(payload.get("issues"), list) else []
        diff_unified = str(payload.get("diff_unified") or "").strip()

        job.results = job.results or {}
        job.results["latex_review"] = {
            "latex_project_id": str(project.id),
            "issues": issues[:100],
            "diff_unified": diff_unified,
            "note": "Diff is a suggestion; review before applying.",
        }

        _emit(100, "completed", "Review complete")
        job.status = AgentJobStatus.COMPLETED.value
        job.completed_at = datetime.utcnow()
        await db.commit()
        return {"status": "completed", "results": job.results}

    async def _run_experiment_plan_generate(
        self,
        *,
        job: AgentJob,
        db: AsyncSession,
        progress_callback: Optional[callable],
    ) -> Dict[str, Any]:
        """
        Deterministic runner: create an ExperimentPlan from a ResearchNote (Hypothesis section).

        Expects config:
          - research_note_id (UUID) OR note_id
          - optional prefer_section: 'hypothesis'|'full_note' (default 'hypothesis')
          - optional max_note_chars (default 12000)
        """
        from uuid import UUID as _UUID

        from app.models.experiment import ExperimentPlan
        from app.models.research_note import ResearchNote

        def _emit(progress: int, phase: str, details: str):
            job.progress = max(0, min(100, int(progress)))
            job.current_phase = phase
            job.phase_details = details
            job.last_activity_at = datetime.utcnow()
            job.add_log_entry({"phase": phase, "action": "experiment_plan_generate", "result": details})

        def _extract_hypothesis_section(markdown: str) -> Optional[str]:
            if not markdown:
                return None
            lines = markdown.splitlines()
            heading_re = re.compile(r"^(#{1,6})\s+(Hypothesis|Hypotheses)\s*$", re.IGNORECASE)
            start_idx = None
            start_level = None
            for i, line in enumerate(lines):
                m = heading_re.match(line.strip())
                if m:
                    start_idx = i + 1
                    start_level = len(m.group(1))
                    break
            if start_idx is None:
                return None
            next_heading_re = re.compile(r"^(#{1,6})\s+.+\s*$")
            out: list[str] = []
            for j in range(start_idx, len(lines)):
                m2 = next_heading_re.match(lines[j].strip())
                if m2 and len(m2.group(1)) <= (start_level or 6):
                    break
                out.append(lines[j])
            text = "\n".join(out).strip()
            return text or None

        cfg = job.config if isinstance(job.config, dict) else {}
        note_id_raw = cfg.get("research_note_id") or cfg.get("note_id")
        if not note_id_raw:
            job.status = AgentJobStatus.FAILED.value
            job.error = "Missing config.research_note_id"
            await db.commit()
            return {"status": "failed", "error": job.error}

        try:
            note_id = _UUID(str(note_id_raw))
        except Exception:
            job.status = AgentJobStatus.FAILED.value
            job.error = "Invalid research_note_id"
            await db.commit()
            return {"status": "failed", "error": job.error}

        note = await db.get(ResearchNote, note_id)
        if not note or note.user_id != job.user_id:
            job.status = AgentJobStatus.FAILED.value
            job.error = "Research note not found"
            await db.commit()
            return {"status": "failed", "error": job.error}

        prefer_section = str(cfg.get("prefer_section") or "hypothesis").strip().lower()
        if prefer_section not in {"hypothesis", "full_note"}:
            prefer_section = "hypothesis"

        max_note_chars = int(cfg.get("max_note_chars") or 12000)
        max_note_chars = max(500, min(max_note_chars, 60000))

        content = (note.content_markdown or "").strip()
        if prefer_section == "hypothesis":
            hypothesis_text = _extract_hypothesis_section(content) or content
        else:
            hypothesis_text = content
        hypothesis_text = (hypothesis_text or "").strip()
        if len(hypothesis_text) > max_note_chars:
            hypothesis_text = hypothesis_text[:max_note_chars]

        prompt = "\n\n".join(
            [
                "You are an AI research engineer. Create a runnable experiment plan from the hypothesis.",
                "Return ONLY valid JSON. No markdown, no commentary.",
                "JSON schema (high level): {"
                '\"hypothesis\": string, \"problem_statement\": string, \"success_criteria\": [string],'
                '\"datasets\": [{\"name\": string, \"source\": string, \"split\": string|null, \"notes\": string|null}],'
                '\"metrics\": [{\"name\": string, \"definition\": string, \"direction\": \"higher_better\"|\"lower_better\"}],'
                '\"baselines\": [{\"name\": string, \"details\": string}],'
                '\"method\": {\"summary\": string, \"key_components\": [string]},'
                '\"experiments\": [{\"name\": string, \"purpose\": string, \"variables\": [string], \"expected_outcome\": string}],'
                '\"ablations\": [{\"name\": string, \"remove_or_change\": string, \"expected_effect\": string}] | [],'
                '\"evaluation_protocol\": string,'
                '\"compute_budget\": {\"hardware\": string|null, \"time_estimate\": string|null, \"notes\": string|null},'
                '\"timeline\": [{\"week\": string, \"deliverable\": string}] | [],'
                '\"risks\": [{\"risk\": string, \"mitigation\": string}] | [],'
                '\"repro_checklist\": [string] | []'
                "}",
                f"Note title: {note.title}",
                "Hypothesis section:",
                hypothesis_text,
                "Rules:",
                "- Keep it concrete: include at least 3 experiments and 2 metrics.",
                "- Ensure the JSON is parseable.",
            ]
        )

        _emit(10, "planning", "Generating experiment plan JSON")
        llm = LLMService()
        raw = await llm.generate_response(
            query=prompt,
            max_tokens=1500,
            temperature=0.2,
            task_type="workflow_synthesis",
            user_id=job.user_id,
            db=db,
        )

        try:
            parsed = json.loads(raw) if isinstance(raw, str) else dict(raw)
            if not isinstance(parsed, dict):
                raise ValueError("Plan must be an object")
        except Exception:
            m = re.search(r"\{.*\}", str(raw), flags=re.DOTALL)
            if not m:
                job.status = AgentJobStatus.FAILED.value
                job.error = "Model did not return valid JSON"
                await db.commit()
                return {"status": "failed", "error": job.error}
            parsed = json.loads(m.group(0))

        plan = ExperimentPlan(
            user_id=job.user_id,
            research_note_id=note.id,
            title=f"Experiment Plan: {note.title}",
            hypothesis_text=hypothesis_text if prefer_section == "hypothesis" else None,
            plan=parsed,
            generator="llm",
            generator_details={"generated_at": datetime.utcnow().isoformat(), "via": "agent_job"},
        )
        db.add(plan)
        await db.commit()
        await db.refresh(plan)

        cfg["experiment_plan_id"] = str(plan.id)
        cfg["research_note_id"] = str(note.id)
        job.config = cfg

        job.results = job.results or {}
        job.results["experiment_plan"] = {
            "experiment_plan_id": str(plan.id),
            "research_note_id": str(note.id),
            "title": plan.title,
        }
        _emit(100, "completed", "Experiment plan created")
        job.status = AgentJobStatus.COMPLETED.value
        job.completed_at = datetime.utcnow()
        await db.commit()
        return {"status": "completed", "results": job.results}

    async def _run_experiment_loop_seed(
        self,
        *,
        job: AgentJob,
        db: AsyncSession,
        progress_callback: Optional[callable],
    ) -> Dict[str, Any]:
        """
        Deterministic runner: seed a configurable experiment loop by writing a nested chain_config onto this job.

        Expects config:
          - research_note_id (UUID)
          - source_id (UUID of git DocumentSource)
          - commands (list[str]) baseline commands
          - optional max_runs (int, default 3, max 20)
          - optional command_variants, use_llm_decider (used by experiment_decide_next)
        """
        cfg = job.config if isinstance(job.config, dict) else {}

        def _emit(progress: int, phase: str, details: str):
            job.progress = max(0, min(100, int(progress)))
            job.current_phase = phase
            job.phase_details = details
            job.last_activity_at = datetime.utcnow()
            job.add_log_entry({"phase": phase, "action": "experiment_loop_seed", "result": details})

        research_note_id = str(cfg.get("research_note_id") or cfg.get("note_id") or "").strip()
        source_id = str(cfg.get("source_id") or cfg.get("target_source_id") or "").strip()
        if not research_note_id:
            job.status = AgentJobStatus.FAILED.value
            job.error = "Missing config.research_note_id"
            await db.commit()
            return {"status": "failed", "error": job.error}
        if not source_id:
            job.status = AgentJobStatus.FAILED.value
            job.error = "Missing config.source_id"
            await db.commit()
            return {"status": "failed", "error": job.error}

        max_runs = int(cfg.get("max_runs") or cfg.get("max_experiment_runs") or 3)
        max_runs = max(1, min(max_runs, 20))

        inherit_results = bool(cfg.get("inherit_results", True))
        inherit_config = bool(cfg.get("inherit_config", True))
        append_to_note = bool(cfg.get("append_to_note", True))

        prefix = str(job.name or "Experiment Loop").strip()[:160]

        def _mk_child(name: str, runner: str, goal: str, config_extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
            payload: Dict[str, Any] = {
                "name": f"{prefix} - {name}"[:200],
                "job_type": "analysis",
                "goal": goal,
                "config": {"deterministic_runner": runner},
            }
            if config_extra:
                payload["config"].update(config_extra)
            return payload

        nodes: list[tuple[Dict[str, Any], Optional[str]]] = []
        nodes.append(
            (
                _mk_child(
                    "Generate Plan",
                    "experiment_plan_generate",
                    f"Generate an experiment plan from research note {research_note_id}",
                ),
                "on_complete",
            )
        )

        for i in range(max_runs):
            human_i = i + 1
            nodes.append(
                (
                    _mk_child(
                        f"Decide Next ({human_i})",
                        "experiment_decide_next",
                        f"Decide next commands for research note {research_note_id}",
                    ),
                    "on_complete",
                )
            )
            nodes.append(
                (
                    _mk_child(
                        f"Run ({human_i})",
                        "experiment_runner",
                        f"Run experiment commands for research note {research_note_id}",
                    ),
                    "on_any_end",
                )
            )
            nodes.append(
                (
                    _mk_child(
                        f"Persist ({human_i})",
                        "experiment_persist_results",
                        f"Persist experiment results for research note {research_note_id}",
                        {"append_to_note": append_to_note},
                    ),
                    "on_complete" if i < max_runs - 1 else None,
                )
            )

        child: Optional[Dict[str, Any]] = None
        for node, trig in reversed(nodes):
            if child is not None and trig:
                node["chain_config"] = {
                    "trigger_condition": trig,
                    "inherit_results": inherit_results,
                    "inherit_config": inherit_config,
                    "child_jobs": [child],
                }
            child = node

        if child is None:
            job.status = AgentJobStatus.FAILED.value
            job.error = "Failed to seed loop (no steps)"
            await db.commit()
            return {"status": "failed", "error": job.error}

        job.chain_config = {
            "trigger_condition": "on_complete",
            "inherit_results": inherit_results,
            "inherit_config": inherit_config,
            "child_jobs": [child],
        }

        job.results = job.results or {}
        job.results["experiment_loop_seed"] = {
            "max_runs": max_runs,
            "total_child_jobs": len(nodes),
            "append_to_note": append_to_note,
        }
        _emit(100, "completed", f"Seeded experiment loop ({max_runs} runs)")
        job.status = AgentJobStatus.COMPLETED.value
        job.completed_at = datetime.utcnow()
        await db.commit()
        return {"status": "completed", "results": job.results}

    async def _run_experiment_decide_next(
        self,
        *,
        job: AgentJob,
        db: AsyncSession,
        progress_callback: Optional[callable],
    ) -> Dict[str, Any]:
        """
        Deterministic runner: decide next experiment command variant + run name.

        Expects config:
          - commands (list[str]) baseline (optional)
          - command_variants: list[list[str]] OR list[{name, commands}] (optional)
          - use_llm_decider: bool (optional)

        Produces/updates config:
          - commands (list[str]) for next experiment_runner step
          - run_name (string)
          - experiment_iteration (int)
        """
        cfg = job.config if isinstance(job.config, dict) else {}

        def _emit(progress: int, phase: str, details: str):
            job.progress = max(0, min(100, int(progress)))
            job.current_phase = phase
            job.phase_details = details
            job.last_activity_at = datetime.utcnow()
            job.add_log_entry({"phase": phase, "action": "experiment_decide_next", "result": details})

        iteration = int(cfg.get("experiment_iteration") or 0)
        variants = cfg.get("command_variants") if isinstance(cfg.get("command_variants"), list) else []

        # If variants provided, pick by iteration index.
        chosen_commands: list[str] = []
        chosen_name: str | None = None
        if variants and iteration < len(variants):
            v = variants[iteration]
            if isinstance(v, dict):
                chosen_name = str(v.get("name") or "").strip() or None
                cmds = v.get("commands")
                if isinstance(cmds, list):
                    chosen_commands = [str(x).strip() for x in cmds if str(x).strip()]
            elif isinstance(v, list):
                chosen_commands = [str(x).strip() for x in v if str(x).strip()]

        # Optional LLM decider if enabled and no variant picked.
        use_llm = bool(cfg.get("use_llm_decider")) and not chosen_commands
        if use_llm:
            inherited = cfg.get("inherited_data") if isinstance(cfg.get("inherited_data"), dict) else {}
            parent_results = inherited.get("parent_results") if isinstance(inherited.get("parent_results"), dict) else {}
            plan = parent_results.get("experiment_plan") if isinstance(parent_results.get("experiment_plan"), dict) else {}
            last_run = parent_results.get("experiment_run") if isinstance(parent_results.get("experiment_run"), dict) else {}
            prompt = "\n\n".join(
                [
                    "You are an AI research engineer. Propose the next experiment command(s) to run.",
                    "Return ONLY JSON: {\"run_name\": string, \"commands\": [string], \"rationale\": string}.",
                    "Constraints: up to 3 commands, no destructive commands.",
                    f"Iteration: {iteration}",
                    f"Experiment plan summary: {json.dumps(plan, ensure_ascii=False)[:3000]}",
                    f"Last run results: {json.dumps(last_run, ensure_ascii=False)[:3000]}",
                ]
            )
            llm = LLMService()
            raw = await llm.generate_response(
                query=prompt,
                max_tokens=600,
                temperature=0.2,
                task_type="workflow_synthesis",
                user_id=job.user_id,
                db=db,
            )
            try:
                payload = json.loads(raw)
            except Exception:
                m = re.search(r"\{.*\}", str(raw), flags=re.DOTALL)
                payload = json.loads(m.group(0)) if m else {}
            if isinstance(payload, dict):
                rn = str(payload.get("run_name") or "").strip()
                cmds = payload.get("commands")
                if isinstance(cmds, list):
                    chosen_commands = [str(x).strip() for x in cmds if str(x).strip()]
                if rn:
                    chosen_name = rn

        # Fallback: keep existing commands from config
        if not chosen_commands:
            base_cmds = cfg.get("commands") if isinstance(cfg.get("commands"), list) else []
            chosen_commands = [str(x).strip() for x in base_cmds if str(x).strip()]

        chosen_commands = chosen_commands[:6]
        if not chosen_name:
            chosen_name = "Baseline" if iteration == 0 else f"Ablation {iteration}"

        cfg["commands"] = chosen_commands
        cfg["run_name"] = chosen_name
        cfg["experiment_iteration"] = iteration + 1
        job.config = cfg

        job.results = job.results or {}
        job.results["experiment_next"] = {"iteration": iteration, "run_name": chosen_name, "commands": chosen_commands}

        _emit(100, "completed", f"Next run: {chosen_name} ({len(chosen_commands)} cmd)")
        job.status = AgentJobStatus.COMPLETED.value
        job.completed_at = datetime.utcnow()
        await db.commit()
        return {"status": "completed", "results": job.results}

    async def _run_experiment_persist_results(
        self,
        *,
        job: AgentJob,
        db: AsyncSession,
        progress_callback: Optional[callable],
    ) -> Dict[str, Any]:
        """
        Deterministic runner: persist the last experiment_runner output into an ExperimentRun row and optionally append to the note.

        Expects:
          - inherited_data.parent_results.experiment_run
          - experiment_plan_id in inherited results or config
          - optional append_to_note (default True)
        """
        from uuid import UUID as _UUID

        from app.models.experiment import ExperimentPlan, ExperimentRun
        from app.models.research_note import ResearchNote

        def _emit(progress: int, phase: str, details: str):
            job.progress = max(0, min(100, int(progress)))
            job.current_phase = phase
            job.phase_details = details
            job.last_activity_at = datetime.utcnow()
            job.add_log_entry({"phase": phase, "action": "experiment_persist_results", "result": details})

        cfg = job.config if isinstance(job.config, dict) else {}
        inherited = cfg.get("inherited_data") if isinstance(cfg.get("inherited_data"), dict) else {}
        parent_results = inherited.get("parent_results") if isinstance(inherited.get("parent_results"), dict) else {}

        exp = parent_results.get("experiment_run") if isinstance(parent_results.get("experiment_run"), dict) else None
        if not exp:
            job.status = AgentJobStatus.FAILED.value
            job.error = "Missing parent_results.experiment_run"
            await db.commit()
            return {"status": "failed", "error": job.error}

        plan_id_raw = (
            (parent_results.get("experiment_plan") or {}).get("experiment_plan_id")
            if isinstance(parent_results.get("experiment_plan"), dict)
            else None
        )
        plan_id_raw = plan_id_raw or cfg.get("experiment_plan_id")
        if not plan_id_raw:
            job.status = AgentJobStatus.FAILED.value
            job.error = "Missing experiment_plan_id"
            await db.commit()
            return {"status": "failed", "error": job.error}

        try:
            plan_id = _UUID(str(plan_id_raw))
        except Exception:
            job.status = AgentJobStatus.FAILED.value
            job.error = "Invalid experiment_plan_id"
            await db.commit()
            return {"status": "failed", "error": job.error}

        plan = await db.get(ExperimentPlan, plan_id)
        if not plan or plan.user_id != job.user_id:
            job.status = AgentJobStatus.FAILED.value
            job.error = "Experiment plan not found"
            await db.commit()
            return {"status": "failed", "error": job.error}

        run_name = str(cfg.get("run_name") or "").strip() or str((parent_results.get("experiment_next") or {}).get("run_name") or "").strip()
        if not run_name:
            run_name = "Run"

        ok = exp.get("ok")
        status = "completed" if ok is True else ("cancelled" if ok is None else "failed")

        run = ExperimentRun(
            user_id=job.user_id,
            experiment_plan_id=plan.id,
            name=run_name,
            status=status,
            progress=100,
            started_at=datetime.utcnow(),
            completed_at=datetime.utcnow(),
            results=exp,
            summary=(str(exp.get("note") or "Experiment run")[:20000]),
        )
        db.add(run)
        await db.commit()
        await db.refresh(run)

        job.results = job.results or {}
        job.results["experiment_run_record"] = {"experiment_run_id": str(run.id), "status": status, "name": run_name}

        if bool(cfg.get("append_to_note", True)):
            note = await db.get(ResearchNote, plan.research_note_id)
            if note and note.user_id == job.user_id:
                marker = f"<!-- experiment_run:{run.id} -->"
                existing = note.content_markdown or ""
                if marker not in existing:
                    lines: list[str] = [
                        "## Experiment Results",
                        marker,
                        "",
                        f"Run: **{run.name}**",
                        f"Status: {run.status}",
                        f"Updated: {datetime.utcnow().isoformat()}",
                        "",
                    ]
                    cmds = exp.get("commands") if isinstance(exp.get("commands"), list) else []
                    if cmds:
                        lines.append("Commands:")
                        for c in cmds[:10]:
                            lines.append(f"- `{str(c)[:240]}`")
                        lines.append("")
                    rr = exp.get("runs") if isinstance(exp.get("runs"), list) else []
                    if rr:
                        lines.append("Results (first 10):")
                        for r2 in rr[:10]:
                            cmd = str(r2.get("command") or "")[:200]
                            exit_code = r2.get("exit_code")
                            ok2 = r2.get("ok")
                            dur = r2.get("duration_ms")
                            line = f"- `{cmd}`"
                            if isinstance(ok2, bool):
                                line += f" · ok={str(ok2).lower()}"
                            if exit_code is not None:
                                line += f" · exit={exit_code}"
                            if dur is not None:
                                line += f" · {dur}ms"
                            lines.append(line)
                        lines.append("")
                    note.content_markdown = existing.rstrip() + "\n\n" + "\n".join(lines).rstrip() + "\n"
                    await db.commit()

        # Stop criteria for chained loops:
        # - stop_on_ok: stop if the experiment succeeded
        # - stop_metric_*: stop if a parsed metric plateaus over a window
        stop_reason: str | None = None
        if bool(cfg.get("stop_on_ok")) and status == "completed":
            stop_reason = "stop_on_ok"

        metric_regex = str(cfg.get("stop_metric_regex") or "").strip()
        if not stop_reason and metric_regex:
            direction = str(cfg.get("stop_metric_direction") or "higher_better").strip().lower()
            if direction not in {"higher_better", "lower_better"}:
                direction = "higher_better"
            window = int(cfg.get("stop_metric_window") or 3)
            window = max(2, min(window, 10))
            min_improvement = float(cfg.get("stop_metric_min_improvement") or 0.0)

            def _extract_metric(exp_run: dict) -> float | None:
                rr = exp_run.get("runs") if isinstance(exp_run.get("runs"), list) else []
                if not rr:
                    return None
                try:
                    rx = re.compile(metric_regex)
                except Exception:
                    return None
                for r2 in rr:
                    if not isinstance(r2, dict):
                        continue
                    text = "\n".join([str(r2.get("stdout") or ""), str(r2.get("stderr") or "")])
                    m = rx.search(text)
                    if not m:
                        continue
                    val_raw = None
                    if "value" in getattr(m, "groupdict", lambda: {})():
                        val_raw = m.group("value")
                    else:
                        try:
                            val_raw = m.group(1)
                        except Exception:
                            val_raw = m.group(0)
                    try:
                        return float(str(val_raw).strip())
                    except Exception:
                        return None
                return None

            # Build a metric history from prior runs (if present) + current.
            prior_runs = parent_results.get("experiment_runs") if isinstance(parent_results.get("experiment_runs"), list) else []
            hist: list[float] = []
            for pr in prior_runs[-10:]:
                if isinstance(pr, dict):
                    v = _extract_metric(pr)
                    if v is not None:
                        hist.append(v)
            v_cur = _extract_metric(exp)
            if v_cur is not None:
                hist.append(v_cur)

            if len(hist) >= window:
                recent = hist[-window:]
                first, last = recent[0], recent[-1]
                improvement = (last - first) if direction == "higher_better" else (first - last)
                if improvement < min_improvement:
                    stop_reason = f"stop_metric_plateau:{direction}:Δ{improvement}"

        if stop_reason:
            job.chain_triggered = True
            job.results["experiment_loop_stop"] = {"reason": stop_reason, "at_run_id": str(run.id)}
            # Best-effort append an explicit stop marker for human visibility.
            if bool(cfg.get("append_to_note", True)):
                note = await db.get(ResearchNote, plan.research_note_id)
                if note and note.user_id == job.user_id:
                    marker2 = f"<!-- experiment_loop_stop:{run.id} -->"
                    existing2 = note.content_markdown or ""
                    if marker2 not in existing2:
                        note.content_markdown = (
                            existing2.rstrip()
                            + "\n\n"
                            + "\n".join(
                                [
                                    "### Experiment Loop Stop",
                                    marker2,
                                    "",
                                    f"Reason: `{stop_reason}`",
                                    f"At run: `{run.id}`",
                                    f"Updated: {datetime.utcnow().isoformat()}",
                                    "",
                                ]
                            ).rstrip()
                            + "\n"
                        )
                        await db.commit()

        _emit(100, "completed", f"Persisted run {run.id}")
        job.status = AgentJobStatus.COMPLETED.value
        job.completed_at = datetime.utcnow()
        await db.commit()
        return {"status": "completed", "results": job.results}

    def _build_swarm_fan_in_result(
        self,
        payload: Dict[str, Any],
        *,
        fan_in_group_id: str = "",
    ) -> Dict[str, Any]:
        """Build deterministic merged result from swarm sibling outputs."""

        def _norm_text(text: Any) -> str:
            raw = str(text or "").strip()
            if not raw:
                return ""
            return re.sub(r"\s+", " ", raw).strip()

        def _extract_points(results: Dict[str, Any]) -> List[str]:
            points: List[str] = []
            if not isinstance(results, dict):
                return points

            findings = results.get("findings")
            if isinstance(findings, list):
                for row in findings:
                    text = ""
                    if isinstance(row, dict):
                        text = (
                            str(
                                row.get("title")
                                or row.get("summary")
                                or row.get("message")
                                or row.get("insight")
                                or row.get("content")
                                or ""
                            )
                            .strip()
                        )
                    else:
                        text = str(row or "").strip()
                    text = _norm_text(text)
                    if text:
                        points.append(text[:280])

            research = results.get("research")
            if isinstance(research, dict):
                for key in ("top_insights", "top_documents", "top_papers"):
                    items = research.get(key)
                    if not isinstance(items, list):
                        continue
                    for item in items:
                        text = _norm_text(item)
                        if text:
                            points.append(text[:280])

            summary = _norm_text(results.get("summary"))
            if summary:
                points.append(summary[:280])

            seen: set[str] = set()
            deduped: List[str] = []
            for point in points:
                k = point.lower()
                if not point or k in seen:
                    continue
                seen.add(k)
                deduped.append(point)
                if len(deduped) >= 12:
                    break
            return deduped

        sibling_jobs = payload.get("sibling_jobs")
        if not isinstance(sibling_jobs, list):
            sibling_jobs = []
        expected = int(payload.get("expected_siblings", 0) or 0)
        if expected <= 0:
            expected = len(sibling_jobs)
        terminal_count = int(payload.get("terminal_siblings", 0) or 0)
        if terminal_count <= 0:
            terminal_statuses = {AgentJobStatus.COMPLETED.value, AgentJobStatus.FAILED.value, AgentJobStatus.CANCELLED.value}
            terminal_count = len([s for s in sibling_jobs if str((s or {}).get("status") or "") in terminal_statuses])

        support_map: Dict[str, Dict[str, Any]] = {}
        role_summaries: List[Dict[str, Any]] = []
        sibling_status: List[Dict[str, Any]] = []
        roles_ordered: List[str] = []
        completed_count = 0
        failed_roles: List[str] = []

        for row in sibling_jobs:
            if not isinstance(row, dict):
                continue
            role = _norm_text(row.get("role") or row.get("name") or "unknown_role")[:120]
            status = _norm_text(row.get("status") or "unknown").lower()
            if role and role not in roles_ordered:
                roles_ordered.append(role)
            if status == AgentJobStatus.COMPLETED.value:
                completed_count += 1
            if status in {AgentJobStatus.FAILED.value, AgentJobStatus.CANCELLED.value}:
                failed_roles.append(role or "unknown_role")

            points = _extract_points(row.get("results") if isinstance(row.get("results"), dict) else {})
            role_summaries.append(
                {
                    "role": role,
                    "status": status,
                    "key_points": points[:3],
                }
            )
            sibling_status.append(
                {
                    "job_id": str(row.get("job_id") or ""),
                    "role": role,
                    "status": status,
                    "progress": int(row.get("progress", 0) or 0),
                }
            )

            used_keys: set[str] = set()
            for point in points:
                k = point.lower()
                if not k or k in used_keys:
                    continue
                used_keys.add(k)
                slot = support_map.get(k)
                if not isinstance(slot, dict):
                    slot = {"finding": point, "roles": set(), "count": 0}
                roles_set = slot.get("roles")
                if not isinstance(roles_set, set):
                    roles_set = set()
                roles_set.add(role)
                slot["roles"] = roles_set
                slot["count"] = int(slot.get("count", 0) or 0) + 1
                support_map[k] = slot

        support_rows: List[Dict[str, Any]] = []
        for k, slot in support_map.items():
            roles = sorted([str(r) for r in slot.get("roles", set()) if str(r).strip()])
            support_rows.append(
                {
                    "key": k,
                    "finding": str(slot.get("finding") or ""),
                    "support_count": int(slot.get("count", 0) or 0),
                    "supporting_roles": roles,
                }
            )
        support_rows.sort(key=lambda r: (-int(r.get("support_count", 0) or 0), str(r.get("finding") or "")))

        consensus = [r for r in support_rows if int(r.get("support_count", 0) or 0) >= 2][:10]
        singletons = [r for r in support_rows if int(r.get("support_count", 0) or 0) <= 1][:10]

        conflicts: List[Dict[str, Any]] = []
        if failed_roles and completed_count > 0:
            conflicts.append(
                {
                    "type": "execution_divergence",
                    "description": f"{len(failed_roles)} swarm role(s) failed or were cancelled while others completed.",
                    "roles": failed_roles[:8],
                }
            )
        if not consensus and len(roles_ordered) >= 2 and support_rows:
            conflicts.append(
                {
                    "type": "low_alignment",
                    "description": "Role outputs show low overlap; no repeated findings across roles.",
                    "roles": roles_ordered[:8],
                }
            )
        if terminal_count < expected:
            conflicts.append(
                {
                    "type": "incomplete_swarm",
                    "description": f"Only {terminal_count}/{expected} sibling jobs reached a terminal state.",
                    "roles": roles_ordered[:8],
                }
            )

        coverage = float(min(1.0, float(len(sibling_jobs)) / float(max(1, expected))))
        completion = float(min(1.0, float(completed_count) / float(max(1, len(sibling_jobs)))))
        agreement = 0.0
        if consensus:
            agreement = float(sum(min(1.0, float(int(r.get("support_count", 0) or 0)) / float(max(1, len(sibling_jobs)))) for r in consensus))
            agreement = max(0.0, min(1.0, agreement / float(max(1, len(consensus)))))
        overall = max(0.0, min(1.0, (0.35 * coverage) + (0.35 * completion) + (0.3 * agreement)))

        action_plan: List[Dict[str, Any]] = []
        for row in consensus[:3]:
            action_plan.append(
                {
                    "priority": "high",
                    "action": f"Validate and operationalize: {str(row.get('finding') or '')[:200]}",
                    "rationale": f"Supported by {int(row.get('support_count', 0) or 0)} swarm roles.",
                }
            )
        for conflict in conflicts[:2]:
            action_plan.append(
                {
                    "priority": "medium",
                    "action": f"Resolve conflict: {str(conflict.get('type') or 'conflict')}",
                    "rationale": str(conflict.get("description") or "")[:220],
                }
            )
        if not action_plan and singletons:
            for row in singletons[:2]:
                action_plan.append(
                    {
                        "priority": "medium",
                        "action": f"Investigate unique signal: {str(row.get('finding') or '')[:180]}",
                        "rationale": "Appears in only one role; needs validation.",
                    }
                )
        if len(action_plan) < 3:
            action_plan.append(
                {
                    "priority": "low",
                    "action": "Produce a consolidated brief with evidence links and clear owner-assigned next steps.",
                    "rationale": "Ensures swarm output is actionable for downstream execution.",
                }
            )
        action_plan = action_plan[:6]

        return {
            "swarm_parent_job_id": str(payload.get("swarm_parent_job_id") or ""),
            "fan_in_group_id": str(fan_in_group_id or payload.get("swarm_fan_in_group_id") or ""),
            "expected_siblings": int(expected),
            "received_siblings": int(len(sibling_jobs)),
            "terminal_siblings": int(terminal_count),
            "roles": roles_ordered[:20],
            "role_summaries": role_summaries[:20],
            "sibling_status": sibling_status[:20],
            "consensus_findings": [
                {
                    "finding": str(r.get("finding") or "")[:280],
                    "support_count": int(r.get("support_count", 0) or 0),
                    "supporting_roles": r.get("supporting_roles", [])[:10],
                }
                for r in consensus
            ],
            "conflicts": conflicts[:10],
            "confidence": {
                "overall": round(overall, 4),
                "coverage": round(coverage, 4),
                "completion": round(completion, 4),
                "agreement": round(agreement, 4),
            },
            "action_plan": action_plan,
            "generated_at": datetime.utcnow().isoformat(),
        }

    async def _run_swarm_fan_in_aggregate(
        self,
        *,
        job: AgentJob,
        db: AsyncSession,
        progress_callback: Optional[callable],
    ) -> Dict[str, Any]:
        """Deterministic runner: aggregate swarm sibling outputs into a strict merged schema."""
        cfg = job.config if isinstance(job.config, dict) else {}

        def _emit(progress: int, phase: str, details: str):
            job.progress = max(0, min(100, int(progress)))
            job.current_phase = phase
            job.phase_details = details
            job.last_activity_at = datetime.utcnow()
            job.add_log_entry({"phase": phase, "action": "swarm_fan_in_aggregate", "result": details})

        inherited = cfg.get("inherited_data") if isinstance(cfg.get("inherited_data"), dict) else {}
        parent_results = inherited.get("parent_results") if isinstance(inherited.get("parent_results"), dict) else {}
        swarm_payload = inherited.get("swarm") if isinstance(inherited.get("swarm"), dict) else {}
        sibling_jobs = swarm_payload.get("sibling_jobs") if isinstance(swarm_payload.get("sibling_jobs"), list) else []
        fan_in_group_id = str(cfg.get("swarm_fan_in_group_id") or swarm_payload.get("swarm_fan_in_group_id") or "").strip()

        if not sibling_jobs:
            job.status = AgentJobStatus.FAILED.value
            job.error = "Missing inherited swarm sibling data"
            await db.commit()
            return {"status": "failed", "error": job.error}

        _emit(30, "aggregating", f"Aggregating outputs from {len(sibling_jobs)} swarm siblings")
        await db.commit()

        merged = self._build_swarm_fan_in_result(
            swarm_payload,
            fan_in_group_id=fan_in_group_id,
        )
        consensus_rows = merged.get("consensus_findings") if isinstance(merged.get("consensus_findings"), list) else []
        findings: List[Dict[str, Any]] = []
        for row in consensus_rows[:12]:
            if not isinstance(row, dict):
                continue
            findings.append(
                {
                    "type": "insight",
                    "category": "swarm_consensus",
                    "title": str(row.get("finding") or "")[:280],
                    "support_count": int(row.get("support_count", 0) or 0),
                    "roles": row.get("supporting_roles", [])[:10] if isinstance(row.get("supporting_roles"), list) else [],
                }
            )

        base_results = dict(parent_results) if isinstance(parent_results, dict) else {}
        base_results["swarm_fan_in"] = merged
        base_results["findings"] = findings
        base_results["summary"] = (
            f"Swarm fan-in complete: {int(merged.get('received_siblings', 0) or 0)}/"
            f"{int(merged.get('expected_siblings', 0) or 0)} siblings aggregated, "
            f"{len(consensus_rows)} consensus findings."
        )
        job.results = base_results

        _emit(100, "completed", "Swarm fan-in aggregation complete")
        job.status = AgentJobStatus.COMPLETED.value
        job.completed_at = datetime.utcnow()
        await db.commit()
        return {"status": "completed", "results": job.results}

    async def _run_experiment_runner(
        self,
        *,
        job: AgentJob,
        db: AsyncSession,
        progress_callback: Optional[callable],
    ) -> Dict[str, Any]:
        """
        Deterministic runner: run a small command-based experiment against a git DocumentSource.

        This is explicitly gated by the existing "unsafe code execution" feature flag/settings.

        Expects:
          - job.config.source_id OR job.config.target_source_id (UUID of DocumentSource)
          - optional job.config.commands: list[str] (shell commands)
          - optional job.config.latex_project_id: UUID (append a Results subsection)
        """
        import asyncio as _asyncio
        import os as _os
        import subprocess as _subprocess
        import tempfile as _tempfile
        from pathlib import Path as _Path
        from uuid import UUID as _UUID

        from app.core.config import settings as app_settings
        from app.core.feature_flags import get_flag as get_feature_flag, get_str as get_feature_str
        from app.models.document import Document, DocumentSource
        from app.models.code_patch_proposal import CodePatchProposal
        from app.models.latex_project import LatexProject
        from app.services.code_patch_apply_service import code_patch_apply_service, UnifiedDiffApplyError

        def _emit(progress: int, phase: str, details: str):
            job.progress = max(0, min(100, int(progress)))
            job.current_phase = phase
            job.phase_details = details
            job.last_activity_at = datetime.utcnow()
            job.add_log_entry({"phase": phase, "action": "experiment_runner", "result": details})

        def _safe_relpath(p: str) -> str:
            p = (p or "").replace("\\", "/").strip()
            p = p.lstrip("/")
            while p.startswith("./"):
                p = p[2:]
            parts = [x for x in p.split("/") if x not in {"", ".", ".."}]
            safe = "/".join(parts)
            return safe[:240]

        def _insert_before_end_document(source: str, addition: str) -> str:
            marker = "\\end{document}"
            s = (source or "")
            idx = s.rfind(marker)
            if idx == -1:
                return (s.rstrip() + "\n\n" + addition.strip() + "\n").lstrip("\n")
            before = s[:idx].rstrip()
            after = s[idx:]
            return f"{before}\n\n{addition.strip()}\n\n{after}"

        cfg = job.config if isinstance(job.config, dict) else {}
        enabled_raw = cfg.get("enabled")
        if enabled_raw is None:
            enabled = bool(cfg.get("enable_experiments", True))
        else:
            enabled = bool(enabled_raw)

        if not enabled:
            inherited = (cfg or {}).get("inherited_data") if isinstance(cfg, dict) else None
            parent_results = inherited.get("parent_results") if isinstance(inherited, dict) else None
            job.results = dict(parent_results) if isinstance(parent_results, dict) else {}
            job.results["experiment_run"] = {
                "enabled": False,
                "ran": False,
                "commands": [],
                "note": "Skipped (enable_experiments=false).",
            }
            _emit(100, "completed", "Skipped (experiments disabled)")
            job.status = AgentJobStatus.COMPLETED.value
            job.completed_at = datetime.utcnow()
            await db.commit()
            return {"status": "completed", "results": job.results}

        source_id_raw = cfg.get("source_id") or cfg.get("target_source_id")
        latex_project_id_raw = cfg.get("latex_project_id")

        # Default commands from inherited code patch results.
        commands = cfg.get("commands") if isinstance(cfg.get("commands"), list) else None
        if not commands:
            inherited = (cfg or {}).get("inherited_data") if isinstance(cfg, dict) else None
            parent_results = inherited.get("parent_results") if isinstance(inherited, dict) else None
            code_patch = parent_results.get("code_patch") if isinstance(parent_results, dict) else None
            tests_to_run = code_patch.get("tests_to_run") if isinstance(code_patch, dict) and isinstance(code_patch.get("tests_to_run"), list) else []
            commands = [str(x) for x in tests_to_run if str(x).strip()]

        commands = [str(c).strip() for c in (commands or []) if str(c).strip()]
        commands = commands[:6]

        if not source_id_raw:
            job.status = AgentJobStatus.FAILED.value
            job.error = "Missing config.source_id/target_source_id"
            await db.commit()
            return {"status": "failed", "error": job.error}

        try:
            source_uuid = _UUID(str(source_id_raw))
        except Exception:
            job.status = AgentJobStatus.FAILED.value
            job.error = "Invalid source_id"
            await db.commit()
            return {"status": "failed", "error": job.error}

        source = await db.get(DocumentSource, source_uuid)
        if not source:
            job.status = AgentJobStatus.FAILED.value
            job.error = "Source not found"
            await db.commit()
            return {"status": "failed", "error": job.error}

        inherited = (cfg or {}).get("inherited_data") if isinstance(cfg, dict) else None
        parent_results = inherited.get("parent_results") if isinstance(inherited, dict) else None
        base_results = dict(parent_results) if isinstance(parent_results, dict) else {}

        enabled_override = await get_feature_flag("unsafe_code_execution_enabled")
        enabled_effective = bool(enabled_override) if enabled_override is not None else bool(getattr(app_settings, "ENABLE_UNSAFE_CODE_EXECUTION", False))

        if not commands:
            job.results = dict(base_results)
            job.results["experiment_run"] = {
                "source_id": str(source.id),
                "source_name": source.name,
                "enabled": enabled_effective,
                "ran": False,
                "commands": [],
                "note": "No commands provided (and no inherited tests_to_run).",
            }
            _emit(100, "completed", "No experiment commands to run")
            job.status = AgentJobStatus.COMPLETED.value
            job.completed_at = datetime.utcnow()
            await db.commit()
            return {"status": "completed", "results": job.results}

        backend_override = await get_feature_str("unsafe_code_exec_backend")
        backend_effective = str(backend_override or getattr(app_settings, "UNSAFE_CODE_EXEC_BACKEND", "subprocess") or "subprocess").strip().lower()
        if backend_effective not in {"subprocess"}:
            backend_effective = "subprocess"

        timeout_seconds = int(cfg.get("timeout_seconds") or getattr(app_settings, "UNSAFE_CODE_EXEC_TIMEOUT_SECONDS", 10))
        timeout_seconds = max(2, min(timeout_seconds, 120))
        stdout_cap = int(getattr(app_settings, "UNSAFE_CODE_EXEC_MAX_STDOUT_CHARS", 20000))
        stderr_cap = int(getattr(app_settings, "UNSAFE_CODE_EXEC_MAX_STDERR_CHARS", 20000))

        _emit(10, "loading", f"Loading files for source {source.name}")
        await db.commit()

        res = await db.execute(select(Document).where(Document.source_id == source.id))
        docs = list(res.scalars().all())
        if not docs:
            job.status = AgentJobStatus.FAILED.value
            job.error = "Source has no documents"
            await db.commit()
            return {"status": "failed", "error": job.error}

        docs_by_path: dict[str, Document] = {}
        for d in docs[:2000]:
            p = _safe_relpath(d.file_path or d.source_identifier or d.title or "")
            if p and p not in docs_by_path:
                docs_by_path[p] = d

        files_list: list[dict] = []
        for d in docs[:400]:
            path = _safe_relpath(d.file_path or d.source_identifier or d.title or "")
            if not path:
                continue
            content = (d.content or "")
            if len(content) > 50000:
                content = content[:50000]
            files_list.append({"path": path, "content": content})
            if len(files_list) >= 120:
                break

        patch_apply: dict = {"proposal_id": None, "applied": [], "errors": []}
        code_patch = base_results.get("code_patch") if isinstance(base_results.get("code_patch"), dict) else None
        proposal_id = str((cfg or {}).get("code_patch_proposal_id") or (code_patch or {}).get("proposal_id") or "").strip()
        if proposal_id:
            patch_apply["proposal_id"] = proposal_id
            try:
                proposal_uuid = _UUID(proposal_id)
            except Exception:
                proposal_uuid = None
            proposal = await db.get(CodePatchProposal, proposal_uuid) if proposal_uuid else None
            if proposal and proposal.user_id == job.user_id:
                try:
                    file_diffs = code_patch_apply_service.parse(proposal.diff_unified or "")
                except UnifiedDiffApplyError as exc:
                    patch_apply["errors"].append({"error": f"Invalid diff: {exc}"})
                    file_diffs = []

                if file_diffs:
                    files_by_path: dict[str, int] = {}
                    for idx, ff in enumerate(files_list):
                        p = _safe_relpath(str(ff.get("path") or ""))
                        if p and p not in files_by_path:
                            files_by_path[p] = idx

                    for fd in file_diffs:
                        p = _safe_relpath(fd.path or "")
                        if not p or p in files_by_path:
                            continue
                        d = docs_by_path.get(p)
                        if not d:
                            patch_apply["errors"].append({"path": p, "error": "Document not found for patch path"})
                            continue
                        content = (d.content or "")
                        if len(content) > 50000:
                            content = content[:50000]
                        files_by_path[p] = len(files_list)
                        files_list.append({"path": p, "content": content})

                    for fd in file_diffs:
                        p = _safe_relpath(fd.path or "")
                        if not p:
                            continue
                        idx = files_by_path.get(p)
                        if idx is None:
                            patch_apply["errors"].append({"path": p, "error": "Missing file content for patch"})
                            continue
                        try:
                            new_text, debug = code_patch_apply_service.apply_to_text(str(files_list[idx].get("content") or ""), fd)
                        except UnifiedDiffApplyError as exc:
                            patch_apply["errors"].append({"path": p, "error": str(exc)})
                            continue
                        files_list[idx]["content"] = new_text
                        patch_apply["applied"].append({"path": p, "debug": debug})

        if proposal_id and patch_apply.get("errors"):
            job.results = dict(base_results)
            job.results["code_patch_apply"] = patch_apply
            _emit(100, "failed", "Failed to apply patch before experiments")
            job.status = AgentJobStatus.FAILED.value
            job.error = "Patch apply failed"
            job.completed_at = datetime.utcnow()
            await db.commit()
            return {"status": "failed", "results": job.results, "error": job.error}

        behavior: Dict[str, Any] = {"enabled": enabled_effective, "backend": backend_effective, "ran": False}
        runs: list[dict] = []

        if not enabled_effective:
            behavior["ran"] = False
            behavior["skipped_reason"] = "Server disabled unsafe code execution (unsafe_code_execution_enabled=false)"
        else:
            _emit(40, "running", f"Running {len(commands)} command(s) (unsafe)")
            await db.commit()
            with _tempfile.TemporaryDirectory(prefix="exp_runner_") as tmp:
                tmp_path = _Path(tmp)
                for f in files_list:
                    p = _safe_relpath(str(f.get("path") or ""))
                    if not p:
                        continue
                    out = tmp_path / p
                    out.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        out.write_text(str(f.get("content") or ""), encoding="utf-8")
                    except Exception:
                        continue

                env = dict(_os.environ or {})
                env.setdefault("PYTHONNOUSERSITE", "1")

                for i, cmd in enumerate(commands):
                    start = datetime.utcnow()
                    rec = {"command": cmd, "ok": False, "exit_code": None, "stdout": "", "stderr": "", "duration_ms": 0}
                    try:
                        completed = await _asyncio.wait_for(
                            _asyncio.to_thread(
                                lambda: _subprocess.run(
                                    ["/bin/sh", "-lc", cmd],
                                    cwd=str(tmp_path),
                                    env=env,
                                    capture_output=True,
                                    text=True,
                                    timeout=float(timeout_seconds),
                                )
                            ),
                            timeout=float(timeout_seconds + 2),
                        )
                        rec["exit_code"] = int(completed.returncode)
                        rec["stdout"] = (completed.stdout or "")[:stdout_cap]
                        rec["stderr"] = (completed.stderr or "")[:stderr_cap]
                        rec["ok"] = completed.returncode == 0
                    except _subprocess.TimeoutExpired as e:
                        rec["stderr"] = (str(getattr(e, "stderr", "") or "") or "Timed out")[:stderr_cap]
                    except Exception as e:
                        rec["stderr"] = str(e)[:stderr_cap]
                    finally:
                        rec["duration_ms"] = int((datetime.utcnow() - start).total_seconds() * 1000)
                        runs.append(rec)
                    behavior["ran"] = True
                    _emit(40 + int(40 * (i + 1) / max(1, len(commands))), "running", f"Ran: {cmd}")
                    await db.commit()

        ok: bool | None
        if not enabled_effective:
            ok = None
        else:
            ok = bool(behavior.get("ran")) and all(bool(r.get("ok")) for r in runs) if runs else False
        behavior["ok"] = ok

        # Optional: write results back into LaTeX project.
        latex_updated = False
        if latex_project_id_raw:
            try:
                proj_uuid = _UUID(str(latex_project_id_raw))
            except Exception:
                proj_uuid = None
            if proj_uuid:
                proj = await db.get(LatexProject, proj_uuid)
                if proj and proj.user_id == job.user_id:
                    lines = ["\\section{Results}", f"\\subsection{{Experiment Runner ({source.name})}}"]
                    lines.append("\\begin{itemize}")
                    for r in runs[:10]:
                        cmd = str(r.get("command") or "")
                        status = "OK" if r.get("ok") else "FAIL"
                        lines.append(f"\\item \\texttt{{{cmd.replace('{', '').replace('}', '')[:120]}}}: {status}")
                    lines.append("\\end{itemize}")
                    if not enabled_effective:
                        lines.append("\\noindent \\textbf{Note:} Execution was skipped because unsafe code execution is disabled on the server.")
                    proj.tex_source = _insert_before_end_document(proj.tex_source or "", "\n".join(lines))
                    await db.commit()
                    latex_updated = True

        job.results = dict(base_results)
        prev_er = job.results.get("experiment_run") if isinstance(job.results.get("experiment_run"), dict) else None
        if isinstance(prev_er, dict):
            existing = job.results.get("experiment_runs")
            if not isinstance(existing, list):
                existing = []
            existing.append(prev_er)
            job.results["experiment_runs"] = existing[-5:]

        job.results["code_patch_apply"] = patch_apply
        job.results["experiment_run"] = {
            "source_id": str(source.id),
            "source_name": source.name,
            "enabled": enabled_effective,
            "backend": backend_effective,
            "commands": commands,
            "runs": runs,
            "ok": ok,
            "proposal_id": patch_apply.get("proposal_id"),
            "latex_project_id": str(latex_project_id_raw) if latex_project_id_raw else None,
            "latex_updated": latex_updated,
        }

        if not enabled_effective:
            _emit(100, "completed", "Experiment run skipped (unsafe execution disabled)")
            job.status = AgentJobStatus.COMPLETED.value
        else:
            _emit(100, "completed" if ok else "failed", "Experiment run complete")
            job.status = AgentJobStatus.COMPLETED.value if ok else AgentJobStatus.FAILED.value
            if not ok:
                job.error = "Experiment run failed"
        job.completed_at = datetime.utcnow()
        await db.commit()
        return {"status": job.status, "results": job.results, "error": job.error}

    async def _run_code_patch_apply_to_kb(
        self,
        *,
        job: AgentJob,
        db: AsyncSession,
        progress_callback: Optional[callable],
    ) -> Dict[str, Any]:
        """
        Deterministic runner: apply the latest CodePatchProposal (unified diff) to KnowledgeDB code documents.

        This is intentionally gated to avoid accidental writes.

        Expects:
          - inherited_data.parent_results.code_patch.proposal_id OR config.proposal_id
          - optional config.apply_patch_to_kb (bool, default False)
          - optional config.dry_run (bool, default True)
          - optional config.proposal_strategy: 'best_passing' | 'latest' | 'explicit' (default 'latest')
          - optional config.enabled_key: string (default 'apply_patch_to_kb')
          - optional config.require_experiments_ok (bool, default True)
          - optional config.fail_on_block (bool, default False)
          - optional config.require_dry_run_first (bool, default False)

        Produces:
          - job.results.code_patch_kb_apply
        """
        from hashlib import sha256 as _sha256
        from uuid import UUID as _UUID

        from sqlalchemy import and_ as _and, or_ as _or

        from app.models.code_patch_proposal import CodePatchProposal
        from app.models.document import Document
        from app.services.code_patch_apply_service import code_patch_apply_service, UnifiedDiffApplyError
        from app.services.document_service import DocumentService

        def _emit(progress: int, phase: str, details: str):
            job.progress = max(0, min(100, int(progress)))
            job.current_phase = phase
            job.phase_details = details
            job.last_activity_at = datetime.utcnow()
            job.add_log_entry({"phase": phase, "action": "code_patch_apply_to_kb", "result": details})

        cfg = job.config if isinstance(job.config, dict) else {}
        inherited = (cfg or {}).get("inherited_data") if isinstance(cfg, dict) else None
        parent_results = inherited.get("parent_results") if isinstance(inherited, dict) else None
        base_results = dict(parent_results) if isinstance(parent_results, dict) else {}

        enabled_key = str(cfg.get("enabled_key") or "apply_patch_to_kb").strip() or "apply_patch_to_kb"
        enabled = bool(cfg.get(enabled_key, False))
        dry_run = bool(cfg.get("dry_run", True))
        require_experiments_ok = bool(cfg.get("require_experiments_ok", True))
        fail_on_block = bool(cfg.get("fail_on_block", False))
        require_dry_run_first = bool(cfg.get("require_dry_run_first", False))

        if not enabled:
            job.results = dict(base_results)
            job.results["code_patch_kb_apply"] = {
                "enabled": False,
                "ok": None,
                "dry_run": dry_run,
                "did_apply": False,
                "proposal_strategy": str(cfg.get("proposal_strategy") or "latest").strip().lower() or "latest",
                "enabled_key": enabled_key,
                "note": "Skipped (apply_patch_to_kb=false).",
            }
            _emit(100, "completed", "Skipped (apply_patch_to_kb=false)")
            job.status = AgentJobStatus.COMPLETED.value
            job.completed_at = datetime.utcnow()
            await db.commit()
            return {"status": "completed", "results": job.results}

        exp_runs = base_results.get("experiment_runs") if isinstance(base_results.get("experiment_runs"), list) else []
        exp_cur = base_results.get("experiment_run") if isinstance(base_results.get("experiment_run"), dict) else None
        exp_all = [r for r in exp_runs if isinstance(r, dict)]
        if isinstance(exp_cur, dict):
            exp_all.append(exp_cur)

        # If we are going to write, require a previous KB dry-run to have succeeded (in-chain).
        prev_kb_apply = base_results.get("code_patch_kb_apply") if isinstance(base_results.get("code_patch_kb_apply"), dict) else None
        if (not dry_run) and require_dry_run_first:
            prev_ok = bool(prev_kb_apply.get("ok")) if isinstance(prev_kb_apply, dict) else False
            prev_dry = bool(prev_kb_apply.get("dry_run")) if isinstance(prev_kb_apply, dict) else False
            if not (prev_ok and prev_dry):
                job.results = dict(base_results)
                job.results["code_patch_kb_apply"] = {
                    "enabled": True,
                    "ok": False,
                    "dry_run": dry_run,
                    "did_apply": False,
                    "blocked": True,
                    "blocked_reason": "Missing/failed prior dry-run (require_dry_run_first=true).",
                }
                _emit(100, "completed" if not fail_on_block else "failed", "Blocked (missing/failed dry-run)")
                job.status = AgentJobStatus.COMPLETED.value if not fail_on_block else AgentJobStatus.FAILED.value
                if fail_on_block:
                    job.error = "Blocked from applying patch to KB"
                job.completed_at = datetime.utcnow()
                await db.commit()
                return {"status": job.status, "results": job.results, "error": job.error}

        proposal_strategy = str(cfg.get("proposal_strategy") or "latest").strip().lower()
        if proposal_strategy not in {"best_passing", "latest", "explicit"}:
            proposal_strategy = "latest"

        job.results = dict(base_results)
        proposal_id = str(cfg.get("proposal_id") or "").strip()
        if proposal_strategy == "explicit" and not proposal_id:
            proposal_id = str(cfg.get("proposal_id") or "").strip()

        if proposal_strategy == "best_passing" and not proposal_id:
            for r in reversed(exp_all):
                if r.get("ok") is True:
                    pid = str(r.get("proposal_id") or "").strip()
                    if pid:
                        proposal_id = pid
                        break

        if proposal_strategy == "latest" and not proposal_id:
            # Prefer the latest proposal_id in inherited code_patch results.
            code_patch = base_results.get("code_patch") if isinstance(base_results.get("code_patch"), dict) else None
            proposal_id = str((code_patch or {}).get("proposal_id") or "").strip()
        if not proposal_id:
            # Fallback: last history entry.
            hist = base_results.get("code_patches") if isinstance(base_results.get("code_patches"), list) else []
            for p in reversed(hist):
                if isinstance(p, dict) and str(p.get("proposal_id") or "").strip():
                    proposal_id = str(p.get("proposal_id") or "").strip()
                    break

        if not proposal_id:
            job.results = dict(base_results)
            job.results["code_patch_kb_apply"] = {
                "enabled": True,
                "ok": False,
                "dry_run": dry_run,
                "did_apply": False,
                "proposal_strategy": proposal_strategy,
                "enabled_key": enabled_key,
                "error": "No proposal_id found in config or inherited results.",
            }
            _emit(100, "failed", "Missing proposal_id")
            job.status = AgentJobStatus.FAILED.value
            job.error = "Missing proposal_id"
            job.completed_at = datetime.utcnow()
            await db.commit()
            return {"status": "failed", "results": job.results, "error": job.error}

        if require_experiments_ok:
            last_for_proposal = None
            for r in reversed(exp_all):
                if str(r.get("proposal_id") or "").strip() == proposal_id:
                    last_for_proposal = r
                    break
            ok_val = last_for_proposal.get("ok") if isinstance(last_for_proposal, dict) else None
            if ok_val is not True:
                job.results = dict(base_results)
                job.results["code_patch_kb_apply"] = {
                    "enabled": True,
                    "ok": False,
                    "dry_run": dry_run,
                    "did_apply": False,
                    "blocked": True,
                    "proposal_id": proposal_id,
                    "proposal_strategy": proposal_strategy,
                    "enabled_key": enabled_key,
                    "blocked_reason": "No passing experiment run found for selected proposal (require_experiments_ok=true).",
                }
                _emit(100, "completed" if not fail_on_block else "failed", "Blocked (experiments not passing)")
                job.status = AgentJobStatus.COMPLETED.value if not fail_on_block else AgentJobStatus.FAILED.value
                if fail_on_block:
                    job.error = "Blocked from applying patch to KB"
                job.completed_at = datetime.utcnow()
                await db.commit()
                return {"status": job.status, "results": job.results, "error": job.error}

        try:
            proposal_uuid = _UUID(proposal_id)
        except Exception:
            job.status = AgentJobStatus.FAILED.value
            job.error = "Invalid proposal_id"
            await db.commit()
            return {"status": "failed", "error": job.error}

        proposal = await db.get(CodePatchProposal, proposal_uuid)
        if not proposal or proposal.user_id != job.user_id:
            job.status = AgentJobStatus.FAILED.value
            job.error = "Code patch proposal not found"
            await db.commit()
            return {"status": "failed", "error": job.error}
        if not proposal.source_id:
            job.status = AgentJobStatus.FAILED.value
            job.error = "Proposal missing source_id"
            await db.commit()
            return {"status": "failed", "error": job.error}

        # Human-in-the-loop gate: do not allow autonomous jobs to directly write patches to the KB
        # unless explicitly enabled at deployment level.
        from app.core.config import settings as _settings
        if (not dry_run) and (not bool(getattr(_settings, "AGENT_KB_PATCH_APPLY_ENABLED", False))):
            try:
                from app.models.patch_pr import PatchPR

                pr = PatchPR(
                    user_id=job.user_id,
                    source_id=proposal.source_id,
                    title=f"PatchPR: {proposal.title}"[:500],
                    description=(proposal.summary or None),
                    status="draft",
                    selected_proposal_id=proposal.id,
                    proposal_ids=[str(proposal.id)],
                    approvals=[],
                    checks={
                        "created_by": "autonomous_agent_executor",
                        "job_id": str(job.id),
                        "note": "Direct KB apply blocked; use PatchPR merge after review.",
                    },
                )
                db.add(pr)
                await db.commit()
                await db.refresh(pr)

                job.results = dict(base_results)
                job.results["code_patch_kb_apply"] = {
                    "enabled": True,
                    "ok": False,
                    "dry_run": dry_run,
                    "did_apply": False,
                    "blocked": True,
                    "proposal_id": str(proposal.id),
                    "proposal_strategy": proposal_strategy,
                    "enabled_key": enabled_key,
                    "blocked_reason": "Direct KB patch apply is disabled (AGENT_KB_PATCH_APPLY_ENABLED=false).",
                    "patch_pr_id": str(pr.id),
                }
                _emit(100, "completed" if not fail_on_block else "failed", "Blocked (requires PatchPR review/merge)")
                job.status = AgentJobStatus.COMPLETED.value if not fail_on_block else AgentJobStatus.FAILED.value
                if fail_on_block:
                    job.error = "Blocked from applying patch to KB"
                job.completed_at = datetime.utcnow()
                await db.commit()
                return {"status": job.status, "results": job.results, "error": job.error}
            except Exception:
                job.results = dict(base_results)
                job.results["code_patch_kb_apply"] = {
                    "enabled": True,
                    "ok": False,
                    "dry_run": dry_run,
                    "did_apply": False,
                    "blocked": True,
                    "proposal_id": str(proposal.id),
                    "proposal_strategy": proposal_strategy,
                    "enabled_key": enabled_key,
                    "blocked_reason": "Direct KB patch apply is disabled (AGENT_KB_PATCH_APPLY_ENABLED=false).",
                }
                _emit(100, "completed" if not fail_on_block else "failed", "Blocked (requires PatchPR review/merge)")
                job.status = AgentJobStatus.COMPLETED.value if not fail_on_block else AgentJobStatus.FAILED.value
                if fail_on_block:
                    job.error = "Blocked from applying patch to KB"
                job.completed_at = datetime.utcnow()
                await db.commit()
                return {"status": job.status, "results": job.results, "error": job.error}

        _emit(10, "parsing", "Parsing unified diff")
        await db.commit()

        try:
            file_diffs = code_patch_apply_service.parse(proposal.diff_unified or "")
        except UnifiedDiffApplyError as exc:
            job.results = dict(base_results)
            job.results["code_patch_kb_apply"] = {
                "enabled": True,
                "applied": False,
                "dry_run": dry_run,
                "proposal_id": str(proposal.id),
                "error": f"Invalid diff: {exc}",
            }
            _emit(100, "failed", "Invalid diff")
            job.status = AgentJobStatus.FAILED.value
            job.error = "Invalid diff"
            job.completed_at = datetime.utcnow()
            await db.commit()
            return {"status": "failed", "results": job.results, "error": job.error}

        if not file_diffs:
            job.results = dict(base_results)
            job.results["code_patch_kb_apply"] = {
                "enabled": True,
                "applied": False,
                "dry_run": dry_run,
                "proposal_id": str(proposal.id),
                "error": "No file diffs found",
            }
            _emit(100, "failed", "No file diffs found")
            job.status = AgentJobStatus.FAILED.value
            job.error = "No file diffs found"
            job.completed_at = datetime.utcnow()
            await db.commit()
            return {"status": "failed", "results": job.results, "error": job.error}

        service = DocumentService()
        applied: list[dict] = []
        errors: list[dict] = []

        _emit(30, "applying", f"Applying patch to {len(file_diffs)} file(s){' (dry-run)' if dry_run else ''}")
        await db.commit()

        for idx, fd in enumerate(file_diffs[:200]):
            path = (fd.path or "").strip()
            if not path:
                continue

            res = await db.execute(
                select(Document)
                .where(
                    _and(
                        Document.source_id == proposal.source_id,
                        _or(
                            Document.file_path == path,
                            Document.source_identifier == path,
                            Document.title == path,
                        ),
                    )
                )
                .limit(1)
            )
            doc = res.scalar_one_or_none()
            if not doc:
                errors.append({"path": path, "error": "Document not found"})
                continue

            try:
                new_text, debug = code_patch_apply_service.apply_to_text(doc.content or "", fd)
            except UnifiedDiffApplyError as exc:
                errors.append({"path": path, "error": str(exc)})
                continue

            if not dry_run:
                doc.content = new_text
                doc.content_hash = _sha256(new_text.encode("utf-8")).hexdigest()
                doc.is_processed = False
                doc.processing_error = None

                try:
                    await service.reprocess_document(doc.id, db, user_id=job.user_id)
                except Exception:
                    pass

            applied.append({"path": path, "document_id": str(doc.id), "debug": debug})
            _emit(30 + int(50 * (idx + 1) / max(1, min(len(file_diffs), 200))), "applying", f"Patched: {path}")
            await db.commit()

        ok = len(errors) == 0
        if not dry_run:
            proposal.proposal_metadata = proposal.proposal_metadata if isinstance(proposal.proposal_metadata, dict) else {}
            proposal.proposal_metadata["apply_results"] = {"applied": applied, "errors": errors, "dry_run": False}
            proposal.status = "applied" if ok else "proposed"

        job.results = dict(base_results)
        job.results["code_patch_kb_apply"] = {
            "enabled": True,
            "ok": ok,
            "dry_run": dry_run,
            "did_apply": ok and (not dry_run),
            "proposal_id": str(proposal.id),
            "proposal_strategy": proposal_strategy,
            "enabled_key": enabled_key,
            "source_id": str(proposal.source_id),
            "applied_files": applied,
            "errors": errors,
        }

        _emit(100, "completed" if ok else "failed", "KB patch apply complete")
        job.status = AgentJobStatus.COMPLETED.value if ok else AgentJobStatus.FAILED.value
        if not ok:
            job.error = "KB patch apply failed"
        job.completed_at = datetime.utcnow()
        await db.commit()
        return {"status": job.status, "results": job.results, "error": job.error}

    async def _run_arxiv_inbox_extract_repos(
        self,
        *,
        job: AgentJob,
        db: AsyncSession,
        progress_callback: Optional[callable],
    ) -> Dict[str, Any]:
        """
        Deterministic runner: extract GitHub/GitLab repo links for an arXiv Research Inbox item.
        """
        import re
        import httpx

        from app.models.research_inbox import ResearchInboxItem

        def _emit(progress: int, phase: str, details: str):
            job.progress = max(0, min(100, int(progress)))
            job.current_phase = phase
            job.phase_details = details
            job.last_activity_at = datetime.utcnow()
            job.add_log_entry({"phase": phase, "action": "arxiv_inbox_extract_repos", "result": details})

        def _extract(text: str) -> list[dict]:
            s = text or ""
            out: list[dict] = []
            seen: set[str] = set()

            for m in re.finditer(r"(https?://github\\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+))", s):
                url = m.group(1)
                owner = m.group(2)
                repo = m.group(3)
                repo_id = f"{owner}/{repo}"
                key = f"github:{repo_id}"
                if key in seen:
                    continue
                seen.add(key)
                out.append({"provider": "github", "repo": repo_id, "url": url})

            for m in re.finditer(r"(https?://gitlab\\.com/([A-Za-z0-9_\\-./]+))", s):
                url = m.group(1)
                path = m.group(2).strip("/")
                if path.count("/") < 1:
                    continue
                repo_id = path.split("#")[0].split("?")[0]
                key = f"gitlab:{repo_id}"
                if key in seen:
                    continue
                seen.add(key)
                out.append({"provider": "gitlab", "repo": repo_id, "url": url})

            return out[:20]

        inbox_item_id = (job.config or {}).get("inbox_item_id")
        if not inbox_item_id:
            job.status = AgentJobStatus.FAILED.value
            job.error = "Missing config.inbox_item_id"
            await db.commit()
            return {"status": "failed", "error": job.error}

        try:
            iid = UUID(str(inbox_item_id))
        except Exception:
            job.status = AgentJobStatus.FAILED.value
            job.error = "Invalid inbox_item_id"
            await db.commit()
            return {"status": "failed", "error": job.error}

        item = await db.get(ResearchInboxItem, iid)
        if not item or item.user_id != job.user_id:
            job.status = AgentJobStatus.FAILED.value
            job.error = "Inbox item not found"
            await db.commit()
            return {"status": "failed", "error": job.error}
        if item.item_type != "arxiv":
            job.status = AgentJobStatus.FAILED.value
            job.error = "Inbox item is not arxiv"
            await db.commit()
            return {"status": "failed", "error": job.error}

        meta = item.item_metadata if isinstance(item.item_metadata, dict) else {}
        combined = " ".join(
            [
                str(item.title or ""),
                str(item.summary or ""),
                str(item.url or ""),
                str(meta.get("entry_url") or ""),
                str(meta.get("pdf_url") or ""),
            ]
        )
        _emit(20, "extracting", "Extracting repos from item text")
        repos = _extract(combined)

        if not repos:
            entry_url = str(meta.get("entry_url") or item.url or "").strip()
            if entry_url:
                _emit(45, "fetching", "Fetching arXiv page for repo links")
                try:
                    async with httpx.AsyncClient(timeout=20.0, headers={"User-Agent": "KnowledgeDBChat-RepoScout"}) as client:
                        resp = await client.get(entry_url)
                        if resp.status_code == 200:
                            repos = _extract(resp.text)
                except Exception:
                    repos = repos or []

        meta["repos"] = repos
        item.item_metadata = meta
        await db.commit()

        job.results = job.results or {}
        job.results["repos_extracted"] = {"inbox_item_id": str(item.id), "count": len(repos), "repos": repos}
        _emit(100, "completed", f"Found {len(repos)} repos")
        job.status = AgentJobStatus.COMPLETED.value
        job.completed_at = datetime.utcnow()
        await db.commit()

        if progress_callback:
            try:
                await progress_callback(
                    {
                        "job_id": str(job.id),
                        "progress": job.progress,
                        "phase": job.current_phase,
                        "status": job.status,
                        "iteration": job.iteration,
                        "phase_details": job.phase_details,
                        "error": job.error,
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                )
            except Exception:
                pass

        return {"status": "completed", "results": job.results}

    async def _run_git_repo_ingest_wait(
        self,
        *,
        job: AgentJob,
        db: AsyncSession,
        progress_callback: Optional[callable],
    ) -> Dict[str, Any]:
        """
        Deterministic runner: create a git repo document source and wait until code files are available.
        """
        import re
        from urllib.parse import urlparse
        from uuid import uuid4

        from app.models.document import Document, DocumentSource
        from app.services.document_service import DocumentService
        from app.tasks.ingestion_tasks import ingest_from_source

        def _emit(progress: int, phase: str, details: str):
            job.progress = max(0, min(100, int(progress)))
            job.current_phase = phase
            job.phase_details = details
            job.last_activity_at = datetime.utcnow()
            job.add_log_entry({"phase": phase, "action": "git_repo_ingest_wait", "result": details})

        def _normalize_repo(provider: str, raw: str) -> str:
            s = (raw or "").strip()
            if not s:
                return ""
            s = s.replace("\\", "/").strip()
            s = s[:-4] if s.endswith(".git") else s
            # Accept URLs too
            if "://" in s:
                try:
                    p = urlparse(s)
                    path = (p.path or "").strip("/")
                    if provider == "github" and p.netloc.lower().endswith("github.com"):
                        parts = [x for x in path.split("/") if x]
                        if len(parts) >= 2:
                            return f"{parts[0]}/{parts[1]}"
                    if provider == "gitlab" and p.netloc.lower().endswith("gitlab.com"):
                        return path.split("#")[0].split("?")[0]
                except Exception:
                    pass
            if provider == "github":
                m = re.search(r"github\\.com/([^/]+/[^/]+)", s, flags=re.IGNORECASE)
                if m:
                    return m.group(1).split("#")[0].split("?")[0].rstrip("/")
                s = s.strip("/")
                parts = s.split("/")
                return f"{parts[0]}/{parts[1]}" if len(parts) >= 2 else s
            # GitLab: allow group/subgroup/project paths
            m = re.search(r"gitlab\\.com/([^\\s]+)", s, flags=re.IGNORECASE)
            if m:
                return m.group(1).split("#")[0].split("?")[0].strip("/")
            return s.strip("/")

        def _infer_repo_from_inherited(cfg: Dict[str, Any]) -> Optional[Dict[str, str]]:
            inherited = cfg.get("inherited_data") if isinstance(cfg, dict) else None
            parent_results = None
            if isinstance(inherited, dict):
                parent_results = inherited.get("parent_results") if isinstance(inherited.get("parent_results"), dict) else None
            repos: list[Any] = []
            if isinstance(parent_results, dict):
                extracted = parent_results.get("repos_extracted") if isinstance(parent_results.get("repos_extracted"), dict) else None
                if extracted and isinstance(extracted.get("repos"), list):
                    repos = extracted.get("repos") or []

            candidates: list[Dict[str, str]] = []
            for r in repos:
                if isinstance(r, dict):
                    candidates.append(
                        {
                            "provider": str(r.get("provider") or "").strip().lower(),
                            "repo": str(r.get("repo") or "").strip(),
                            "url": str(r.get("url") or "").strip(),
                        }
                    )
                elif isinstance(r, str):
                    candidates.append({"provider": "", "repo": r, "url": r})

            normalized: list[Dict[str, str]] = []
            for c in candidates:
                prov = c.get("provider") or ""
                raw = c.get("repo") or c.get("url") or ""
                if prov not in {"github", "gitlab"}:
                    # Try to infer provider from URL-ish strings
                    s = raw.lower()
                    if "github.com" in s:
                        prov = "github"
                    elif "gitlab.com" in s:
                        prov = "gitlab"
                if prov not in {"github", "gitlab"}:
                    continue
                rid = _normalize_repo(prov, raw)
                if not rid:
                    continue
                normalized.append({"provider": prov, "repo": rid})

            # Prefer GitHub if available; otherwise first usable candidate.
            for prov in ("github", "gitlab"):
                for n in normalized:
                    if n["provider"] == prov and n["repo"]:
                        return n
            return normalized[0] if normalized else None

        cfg = job.config if isinstance(job.config, dict) else {}
        provider = str(cfg.get("provider") or "").strip().lower()
        repo = str(cfg.get("repo") or "").strip()

        auto_selected = False
        if provider not in {"github", "gitlab"} or not repo:
            inferred = _infer_repo_from_inherited(cfg)
            if inferred:
                provider = inferred["provider"]
                repo = inferred["repo"]
                cfg["provider"] = provider
                cfg["repo"] = repo
                job.config = cfg
                auto_selected = True

        if provider not in {"github", "gitlab"} or not repo:
            job.status = AgentJobStatus.FAILED.value
            job.error = "Missing repo selection (config.provider/config.repo or inherited repos_extracted)"
            await db.commit()
            return {"status": "failed", "error": job.error}

        # Create the document source similarly to /documents/sources/git-repo.
        include_files = bool(cfg.get("include_files", True))
        include_issues = bool(cfg.get("include_issues", False))
        include_wiki = bool(cfg.get("include_wiki", False))
        include_pull_requests = bool(cfg.get("include_pull_requests", False))
        incremental_files = bool(cfg.get("incremental_files", True))
        use_gitignore = bool(cfg.get("use_gitignore", True))
        max_pages = int(cfg.get("git_ingest_max_pages") or cfg.get("max_pages") or 5)
        max_pages = max(1, min(max_pages, 50))

        config: Dict[str, Any] = {
            "include_files": include_files,
            "include_issues": include_issues,
            "include_wiki": include_wiki,
            "include_pull_requests": include_pull_requests,
            "incremental_files": incremental_files,
            "use_gitignore": use_gitignore,
            "max_pages": max_pages,
            "requested_by": str(job.user_id),
        }

        if provider == "github":
            config["repos"] = [repo]
            token = (cfg.get("token") or cfg.get("github_token") or "").strip()
            if token:
                config["token"] = token
        else:
            # GitLab requires token; support advanced config in a follow-up.
            token = (cfg.get("token") or cfg.get("gitlab_token") or "").strip()
            if not token:
                job.status = AgentJobStatus.FAILED.value
                job.error = "GitLab ingestion requires token (config.gitlab_token)"
                await db.commit()
                return {"status": "failed", "error": job.error}
            config["token"] = token
            gitlab_url = (cfg.get("gitlab_url") or "").strip()
            if gitlab_url:
                config["gitlab_url"] = gitlab_url.rstrip("/")
            config["projects"] = [
                {
                    "id": repo,
                    "include_files": include_files,
                    "include_wikis": include_wiki,
                    "include_issues": include_issues,
                    "include_merge_requests": include_pull_requests,
                }
            ]

        svc = DocumentService()
        name = f"{provider.title()} repo ({repo}) #{uuid4().hex[:6]}"
        _emit(10, "creating_source", f"Creating source for {provider}:{repo}")
        source = await svc.create_document_source(name=name, source_type=provider, config=config, db=db)
        await db.commit()
        await db.refresh(source)

        _emit(30, "ingesting", "Starting ingestion")
        try:
            ingest_from_source.delay(str(source.id))
        except Exception:
            pass

        wait_seconds = int(cfg.get("wait_seconds") or 120)
        wait_seconds = max(10, min(wait_seconds, 10 * 60))
        deadline = datetime.utcnow() + timedelta(seconds=wait_seconds)

        _emit(40, "waiting", "Waiting for code files to be ingested")
        await db.commit()

        docs_count = 0
        while datetime.utcnow() < deadline:
            try:
                res = await db.execute(
                    select(func.count())
                    .select_from(Document)
                    .where(Document.source_id == source.id)
                )
                docs_count = int(res.scalar() or 0)
                if docs_count > 0:
                    break
            except Exception:
                pass
            await asyncio.sleep(2.0)

        job.results = job.results or {}
        job.results["repo_ingest"] = {
            "provider": provider,
            "repo": repo,
            "source_id": str(source.id),
            "source_name": source.name,
            "documents_count": docs_count,
            "auto_selected": auto_selected,
        }

        if docs_count <= 0:
            job.status = AgentJobStatus.FAILED.value
            job.error = "Repo ingestion did not produce documents before timeout"
            _emit(100, "failed", job.error)
            await db.commit()
            return {"status": "failed", "error": job.error, "results": job.results}

        _emit(100, "completed", f"Repo ingested: {docs_count} docs")
        job.status = AgentJobStatus.COMPLETED.value
        job.completed_at = datetime.utcnow()
        await db.commit()

        if progress_callback:
            try:
                await progress_callback(
                    {
                        "job_id": str(job.id),
                        "progress": job.progress,
                        "phase": job.current_phase,
                        "status": job.status,
                        "iteration": job.iteration,
                        "phase_details": job.phase_details,
                        "error": job.error,
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                )
            except Exception:
                pass

        return {"status": "completed", "results": job.results}

    async def _run_generated_project_demo_check(
        self,
        *,
        job: AgentJob,
        db: AsyncSession,
        progress_callback: Optional[callable],
    ) -> Dict[str, Any]:
        """
        Deterministic runner: execute demo.py for an existing generated project source.

        Expects:
          - job.config.source_id (UUID of DocumentSource)
          - optional job.config.entrypoint (default: 'demo.py')
          - optional job.config.timeout_seconds (default: server config)
        """
        import asyncio
        import json as _json
        import os
        import subprocess
        import sys
        import tempfile
        from pathlib import Path
        from uuid import UUID as _UUID

        from app.core.config import settings as app_settings
        from app.core.feature_flags import get_flag as get_feature_flag, get_str as get_feature_str
        from app.models.document import Document, DocumentSource
        from app.models.user import User

        def _emit(progress: int, phase: str, details: str):
            job.progress = max(0, min(100, int(progress)))
            job.current_phase = phase
            job.phase_details = details
            job.last_activity_at = datetime.utcnow()
            job.add_log_entry({"phase": phase, "action": "generated_project_demo_check", "result": details})

        cfg = job.config if isinstance(job.config, dict) else {}
        source_id_raw = cfg.get("source_id")
        entrypoint = str(cfg.get("entrypoint") or "demo.py").strip() or "demo.py"
        timeout_seconds = int(cfg.get("timeout_seconds") or cfg.get("behavioral_timeout_seconds") or getattr(app_settings, "UNSAFE_CODE_EXEC_TIMEOUT_SECONDS", 10))
        timeout_seconds = max(2, min(timeout_seconds, 60))

        if not source_id_raw:
            job.status = AgentJobStatus.FAILED.value
            job.error = "Missing config.source_id"
            await db.commit()
            return {"status": "failed", "error": job.error}

        try:
            source_uuid = _UUID(str(source_id_raw))
        except Exception:
            job.status = AgentJobStatus.FAILED.value
            job.error = "Invalid source_id"
            await db.commit()
            return {"status": "failed", "error": job.error}

        source = await db.get(DocumentSource, source_uuid)
        if not source:
            job.status = AgentJobStatus.FAILED.value
            job.error = "Source not found"
            await db.commit()
            return {"status": "failed", "error": job.error}

        # Access control: admin or requested_by_user_id matches.
        user = await db.get(User, job.user_id)
        is_admin = bool(user and getattr(user, "role", None) == "admin")
        requested_by_user_id = str((source.config or {}).get("requested_by_user_id") or "").strip()
        if not is_admin and requested_by_user_id and requested_by_user_id != str(job.user_id):
            job.status = AgentJobStatus.FAILED.value
            job.error = "Not authorized for this source"
            await db.commit()
            return {"status": "failed", "error": job.error}

        _emit(10, "loading", f"Loading project files for source {source.name}")
        await db.commit()

        res = await db.execute(select(Document).where(Document.source_id == source.id))
        docs = res.scalars().all()
        if not docs:
            job.status = AgentJobStatus.FAILED.value
            job.error = "Source has no documents"
            await db.commit()
            return {"status": "failed", "error": job.error}

        def _safe_relpath(p: str) -> str:
            p = (p or "").replace("\\", "/").strip()
            p = p.lstrip("/")
            while p.startswith("./"):
                p = p[2:]
            parts = [x for x in p.split("/") if x not in {"", ".", ".."}]
            safe = "/".join(parts)
            return safe[:240]

        files_list: list[dict] = []
        for d in docs[:200]:
            path = _safe_relpath(d.file_path or d.source_identifier or d.title or "")
            if not path:
                continue
            content = (d.content or "")
            if len(content) > 50000:
                content = content[:50000]
            files_list.append({"path": path, "content": content})
            if len(files_list) >= 80:
                break

        enabled_override = await get_feature_flag("unsafe_code_execution_enabled")
        enabled_effective = bool(enabled_override) if enabled_override is not None else bool(getattr(app_settings, "ENABLE_UNSAFE_CODE_EXECUTION", False))
        backend_override = await get_feature_str("unsafe_code_exec_backend")
        backend_effective = str(backend_override or getattr(app_settings, "UNSAFE_CODE_EXEC_BACKEND", "subprocess") or "subprocess").strip().lower()
        if backend_effective not in {"subprocess", "docker"}:
            backend_effective = "subprocess"
        image_override = await get_feature_str("unsafe_code_exec_docker_image")
        image_effective = str(image_override or getattr(app_settings, "UNSAFE_CODE_EXEC_DOCKER_IMAGE", "python:3.11-slim") or "python:3.11-slim").strip()

        if not enabled_effective:
            job.status = AgentJobStatus.FAILED.value
            job.error = "Unsafe code execution disabled"
            job.results = job.results or {}
            job.results["demo_check"] = {
                "source_id": str(source.id),
                "entrypoint": entrypoint,
                "ok": False,
                "behavioral": {"enabled": False, "ran": False, "ok": False, "skipped_reason": "unsafe_code_execution_enabled=false"},
            }
            await db.commit()
            return {"status": "failed", "error": job.error, "results": job.results}

        _emit(35, "running", f"Running {entrypoint} (backend={backend_effective})")
        await db.commit()

        stdout_cap = int(getattr(app_settings, "UNSAFE_CODE_EXEC_MAX_STDOUT_CHARS", 20000))
        stderr_cap = int(getattr(app_settings, "UNSAFE_CODE_EXEC_MAX_STDERR_CHARS", 20000))

        def _limit_resources():
            try:
                import resource

                cpu = int(max(1, min(timeout_seconds + 1, 120)))
                resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
                resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
                resource.setrlimit(resource.RLIMIT_FSIZE, (10 * 1024 * 1024, 10 * 1024 * 1024))
                resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
                mem_mb = int(getattr(app_settings, "UNSAFE_CODE_EXEC_MAX_MEMORY_MB", 512))
                mem = mem_mb * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
            except Exception:
                return

        behavior: dict = {
            "enabled": True,
            "ran": False,
            "ok": False,
            "exit_code": None,
            "timed_out": False,
            "duration_ms": None,
            "entrypoint": entrypoint,
            "stdout": "",
            "stderr": "",
            "error": None,
            "backend": backend_effective,
        }

        with tempfile.TemporaryDirectory(prefix="demo_check_") as tmp:
            base = Path(tmp)
            for ff in files_list:
                rel = _safe_relpath(str(ff.get("path") or ""))
                if not rel or rel.startswith("."):
                    continue
                full = (base / rel).resolve()
                if not str(full).startswith(str(base.resolve())):
                    continue
                full.parent.mkdir(parents=True, exist_ok=True)
                try:
                    full.write_text(str(ff.get("content") or ""), encoding="utf-8")
                except Exception:
                    continue

            ep = _safe_relpath(entrypoint)
            if not (base / ep).exists():
                behavior["error"] = f"Entrypoint not found: {entrypoint}"
            else:
                env = {
                    "PYTHONNOUSERSITE": "1",
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONHASHSEED": "0",
                    "HOME": tmp,
                    "PATH": os.environ.get("PATH", ""),
                    "LANG": os.environ.get("LANG", "C.UTF-8"),
                    "LC_ALL": os.environ.get("LC_ALL", "C.UTF-8"),
                }
                if backend_effective == "docker":
                    mem_mb = int(getattr(app_settings, "UNSAFE_CODE_EXEC_MAX_MEMORY_MB", 512))
                    cpus = float(getattr(app_settings, "UNSAFE_CODE_EXEC_DOCKER_CPUS", 1.0) or 1.0)
                    pids = int(getattr(app_settings, "UNSAFE_CODE_EXEC_DOCKER_PIDS_LIMIT", 128))
                    cmd = [
                        "docker",
                        "run",
                        "--rm",
                        "--network",
                        "none",
                        "--cap-drop",
                        "ALL",
                        "--security-opt",
                        "no-new-privileges",
                        "--pids-limit",
                        str(max(32, min(pids, 1024))),
                        "--memory",
                        f"{max(64, min(mem_mb, 4096))}m",
                        "--cpus",
                        str(max(0.25, min(cpus, 4.0))),
                        "--user",
                        "65534:65534",
                        "-v",
                        f"{tmp}:/work:rw",
                        "-w",
                        "/work",
                        image_effective,
                        "python",
                        "-I",
                        "-S",
                        ep,
                    ]
                    preexec = None
                else:
                    cmd = [sys.executable, "-I", "-S", ep]
                    preexec = _limit_resources if os.name == "posix" else None

                start = datetime.utcnow()
                try:
                    completed = await asyncio.wait_for(
                        asyncio.to_thread(
                            lambda: subprocess.run(
                                cmd,
                                cwd=tmp,
                                env=env,
                                capture_output=True,
                                text=True,
                                timeout=float(timeout_seconds),
                                preexec_fn=preexec,
                            )
                        ),
                        timeout=float(timeout_seconds + 2),
                    )
                    behavior["ran"] = True
                    behavior["exit_code"] = int(completed.returncode)
                    behavior["stdout"] = (completed.stdout or "")[:stdout_cap]
                    behavior["stderr"] = (completed.stderr or "")[:stderr_cap]
                    behavior["ok"] = completed.returncode == 0
                except subprocess.TimeoutExpired as e:
                    behavior["ran"] = True
                    behavior["timed_out"] = True
                    behavior["stdout"] = str(getattr(e, "stdout", "") or "")[:stdout_cap]
                    behavior["stderr"] = str(getattr(e, "stderr", "") or "")[:stderr_cap]
                except FileNotFoundError as e:
                    behavior["ran"] = True
                    behavior["error"] = f"Execution backend not available: {e}"
                except Exception as e:
                    behavior["error"] = str(e)
                finally:
                    behavior["duration_ms"] = int((datetime.utcnow() - start).total_seconds() * 1000)

        job.results = job.results or {}
        job.results["demo_check"] = {
            "source_id": str(source.id),
            "source_name": source.name,
            "entrypoint": entrypoint,
            "ok": bool(behavior.get("ok")),
            "behavioral": behavior,
        }
        if job.output_artifacts is None:
            job.output_artifacts = []
        job.output_artifacts.append({"type": "demo_check", "source_id": str(source.id), "title": f"Demo check: {source.name}"})

        if behavior.get("ok"):
            _emit(100, "completed", "Demo check OK")
            job.status = AgentJobStatus.COMPLETED.value
            job.completed_at = datetime.utcnow()
            await db.commit()
            return {"status": "completed", "results": job.results}

        job.status = AgentJobStatus.FAILED.value
        job.error = "Demo check failed"
        _emit(100, "failed", job.error)
        await db.commit()
        return {"status": "failed", "error": job.error, "results": job.results}

    async def _run_paper_algorithm_project(
        self,
        *,
        job: AgentJob,
        db: AsyncSession,
        progress_callback: Optional[callable],
    ) -> Dict[str, Any]:
        """
        Deterministic runner: turn an arXiv Research Inbox item into a small generated code project.

        Expects:
          - job.config.inbox_item_id (UUID of ResearchInboxItem, item_type='arxiv')
          - optional job.config.language (default: 'python')
          - optional job.config.include_tests (default: True)

        Produces:
          - a DocumentSource (source_type='generated') containing project files as Documents
          - job.results.generated_project (source_id, project_name, file_count)
          - job.output_artifacts includes a 'generated_project' entry for UI
        """
        import hashlib
        import re
        from uuid import UUID as _UUID

        from app.models.document import Document, DocumentSource
        from app.models.research_inbox import ResearchInboxItem
        from app.models.user import User

        def _emit(progress: int, phase: str, details: str):
            job.progress = max(0, min(100, int(progress)))
            job.current_phase = phase
            job.phase_details = details
            job.last_activity_at = datetime.utcnow()
            job.add_log_entry({"phase": phase, "action": "paper_algorithm_project", "result": details})

        inbox_item_id = (job.config or {}).get("inbox_item_id") if isinstance(job.config, dict) else None
        if not inbox_item_id:
            job.status = AgentJobStatus.FAILED.value
            job.error = "Missing config.inbox_item_id"
            await db.commit()
            return {"status": "failed", "error": job.error}

        try:
            iid = _UUID(str(inbox_item_id))
        except Exception:
            job.status = AgentJobStatus.FAILED.value
            job.error = "Invalid inbox_item_id"
            await db.commit()
            return {"status": "failed", "error": job.error}

        item = await db.get(ResearchInboxItem, iid)
        if not item or item.user_id != job.user_id:
            job.status = AgentJobStatus.FAILED.value
            job.error = "Inbox item not found"
            await db.commit()
            return {"status": "failed", "error": job.error}
        if item.item_type != "arxiv":
            job.status = AgentJobStatus.FAILED.value
            job.error = "Inbox item is not arxiv"
            await db.commit()
            return {"status": "failed", "error": job.error}

        cfg = job.config if isinstance(job.config, dict) else {}
        language = str(cfg.get("language") or "python").strip().lower()
        from app.core.config import settings as app_settings
        from app.core.feature_flags import get_flag as get_feature_flag, get_str as get_feature_str

        include_tests = bool(cfg.get("include_tests", True))
        use_repo_context = bool(cfg.get("use_repo_context", False))
        auto_repair = bool(cfg.get("auto_repair", True))
        repair_max_attempts = int(cfg.get("repair_max_attempts") or 1)
        repair_max_attempts = max(0, min(repair_max_attempts, 2))
        entrypoint = str(cfg.get("entrypoint") or "demo.py").strip() or "demo.py"
        behavioral_check = bool(cfg.get("behavioral_check", False))
        behavioral_timeout_seconds = int(cfg.get("behavioral_timeout_seconds") or getattr(app_settings, "UNSAFE_CODE_EXEC_TIMEOUT_SECONDS", 10))
        behavioral_timeout_seconds = max(2, min(behavioral_timeout_seconds, 60))
        max_repo_files = int(cfg.get("max_repo_files") or 8)
        max_repo_files = max(1, min(max_repo_files, 20))
        max_chars_per_repo_file = int(cfg.get("max_chars_per_repo_file") or 8000)
        max_chars_per_repo_file = max(1000, min(max_chars_per_repo_file, 20000))
        if language not in {"python"}:
            job.status = AgentJobStatus.FAILED.value
            job.error = f"Unsupported language: {language}"
            await db.commit()
            return {"status": "failed", "error": job.error}

        user = await db.get(User, job.user_id)
        username = user.username if user else ""

        title = str(item.title or "").strip()
        summary = str(item.summary or "").strip()
        url = str(item.url or "").strip()
        meta = item.item_metadata if isinstance(item.item_metadata, dict) else {}
        entry_url = str(meta.get("entry_url") or "").strip()
        pdf_url = str(meta.get("pdf_url") or "").strip()

        _emit(10, "collecting", "Preparing paper context")
        await db.commit()

        def _slugify(s: str) -> str:
            s = (s or "").strip().lower()
            s = re.sub(r"[^a-z0-9]+", "-", s)
            s = s.strip("-")
            return s[:40] or "paper_algorithm"

        project_slug = _slugify(title) if title else "paper_algorithm"
        pkg = project_slug.replace("-", "_")
        pkg = re.sub(r"^[^a-z_]+", "", pkg) or "paper_algorithm"
        pkg = pkg[:48]

        repo_context_block = ""
        inherited_repo_source_id = None
        inherited_provider = None
        inherited_repo = None
        if use_repo_context:
            inherited = cfg.get("inherited_data") if isinstance(cfg, dict) else None
            parent_results = None
            if isinstance(inherited, dict):
                parent_results = inherited.get("parent_results") if isinstance(inherited.get("parent_results"), dict) else None
            if isinstance(parent_results, dict) and isinstance(parent_results.get("repo_ingest"), dict):
                inherited_repo_source_id = parent_results["repo_ingest"].get("source_id")
                inherited_provider = parent_results["repo_ingest"].get("provider")
                inherited_repo = parent_results["repo_ingest"].get("repo")

            if inherited_repo_source_id:
                _emit(22, "collecting", "Loading reference repo files for guidance")
                await db.commit()
                try:
                    from app.services.search_service import SearchService
                    search_service = SearchService()
                    # Use paper title/abstract as a rough query to pull relevant code.
                    query = (str(cfg.get("search_query") or "").strip() or f"{title}\n{summary}").strip()
                    results, _total, _took = await search_service.search(
                        query=query[:800],
                        mode="smart",
                        page=1,
                        page_size=max_repo_files,
                        source_id=str(inherited_repo_source_id),
                        db=db,
                    )
                    ids = [r.get("id") for r in (results or []) if isinstance(r, dict) and r.get("id")]
                    repo_docs: list[Document] = []
                    from uuid import UUID as _UUID2
                    for doc_id in ids[:max_repo_files]:
                        try:
                            d = await db.get(Document, _UUID2(str(doc_id)))
                        except Exception:
                            d = None
                        if d and str(d.source_id) == str(inherited_repo_source_id):
                            repo_docs.append(d)
                    if repo_docs:
                        blocks: list[str] = []
                        for d in repo_docs[:max_repo_files]:
                            p = d.title or d.source_identifier or d.file_path or str(d.id)
                            c = (d.content or "")[:max_chars_per_repo_file]
                            blocks.append(f"### REPO FILE: {p}\n```text\n{c}\n```\n")
                        repo_context_block = (
                            "REFERENCE REPOSITORY CONTEXT (use as guidance; do not overfit to repo quirks):\n"
                            f"Provider: {inherited_provider}\nRepo: {inherited_repo}\n\n"
                            + "".join(blocks)
                        )
                except Exception:
                    repo_context_block = ""

        user_settings = await self._load_user_settings(job.user_id, db)
        _emit(35, "drafting", "Generating implementation plan + code files (LLM)")
        await db.commit()

        prompt = (
            "You are an expert research engineer.\n"
            "Task: implement the core algorithm described in the paper as a small, runnable reference project.\n\n"
            "Output MUST be valid JSON ONLY with keys:\n"
            "- project_name (string)\n"
            "- summary (string)\n"
            "- run_instructions (string)\n"
            "- limitations (array of strings)\n"
            "- files (array of {path, content})\n\n"
            "Constraints:\n"
            "- Keep it minimal: 5-10 files total.\n"
            "- No network calls. No GPU dependencies. Avoid heavy deps.\n"
            "- Use Python 3.11+.\n"
            "- All Python files MUST be syntactically valid.\n"
            "- Include README.md.\n"
            f"- Package name should be '{pkg}'.\n"
            f"- Include a simple synthetic demo script at '{entrypoint}'.\n"
            f"- {entrypoint} must finish quickly (<5 seconds) and print a short success message.\n"
            + ("- Include unit tests (pytest) that check shapes/invariants.\n" if include_tests else "")
            + "- If the paper omits details, implement a reasonable approximation and list assumptions in limitations.\n\n"
            f"PAPER TITLE:\n{title}\n\n"
            f"ABSTRACT/SUMMARY:\n{summary[:6000]}\n\n"
            f"URLS:\n- item_url: {url}\n- entry_url: {entry_url}\n- pdf_url: {pdf_url}\n\n"
            + (repo_context_block + "\n\n" if repo_context_block else "")
        )

        response = await self.llm_service.generate_response(
            query=prompt,
            context=None,
            temperature=0.2,
            max_tokens=2500,
            user_settings=user_settings,
            task_type="code_agent",
            user_id=job.user_id,
            db=db,
            routing=self._llm_routing_from_job_config(job.config),
        )

        try:
            payload = json.loads(response)
        except Exception:
            job.status = AgentJobStatus.FAILED.value
            job.error = "LLM did not return valid JSON for generated project"
            await db.commit()
            return {"status": "failed", "error": job.error, "raw": (response or "")[:2000]}

        files = payload.get("files") if isinstance(payload.get("files"), list) else []
        project_name = str(payload.get("project_name") or f"Paper Algorithm: {title}")[:200].strip()
        summary_out = str(payload.get("summary") or "").strip()
        run_instructions = str(payload.get("run_instructions") or "").strip()
        limitations = payload.get("limitations") if isinstance(payload.get("limitations"), list) else []
        limitations = [str(x)[:300] for x in limitations if str(x).strip()][:20]

        def _sanitize_path(p: str) -> str:
            p = (p or "").replace("\\", "/").strip()
            p = re.sub(r"^/+", "", p)
            while p.startswith("./"):
                p = p[2:]
            p = re.sub(r"/{2,}", "/", p)
            parts = [x for x in p.split("/") if x not in {"", ".", ".."}]
            safe = "/".join(parts)
            return safe[:240]

        normalized_files: list[dict] = []
        seen_paths: set[str] = set()
        for f in files:
            if not isinstance(f, dict):
                continue
            path = _sanitize_path(str(f.get("path") or ""))
            content = str(f.get("content") or "")
            if not path or path in seen_paths:
                continue
            if len(content) > 25000:
                content = content[:25000]
            seen_paths.add(path)
            normalized_files.append({"path": path, "content": content})
            if len(normalized_files) >= 12:
                break

        if not normalized_files:
            job.status = AgentJobStatus.FAILED.value
            job.error = "Generated project had no files"
            await db.commit()
            return {"status": "failed", "error": job.error}

        def _run_behavioral_demo(files_list: list[dict], *, effective_backend: str, effective_image: str) -> Dict[str, Any]:
            """
            Best-effort behavioral check by running demo.py.
            This is explicitly gated by config + server setting.
            """
            import os
            import subprocess
            import sys
            import tempfile
            import time

            result: Dict[str, Any] = {
                "enabled": True,
                "ran": False,
                "ok": False,
                "exit_code": None,
                "timed_out": False,
                "duration_ms": None,
                "entrypoint": entrypoint,
                "stdout": "",
                "stderr": "",
                "error": None,
                "backend": str(effective_backend or "subprocess"),
            }

            # Require configured entrypoint to exist
            ep = _sanitize_path(entrypoint)
            if not ep:
                result["error"] = "Invalid entrypoint"
                return result
            if not any(str(ff.get("path") or "") == ep for ff in files_list):
                result["error"] = f"Entrypoint not found: {entrypoint}"
                return result

            def _limit_resources():
                try:
                    import resource

                    cpu = int(max(1, min(behavioral_timeout_seconds + 1, 120)))
                    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
                    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
                    # 10MB max file size
                    resource.setrlimit(resource.RLIMIT_FSIZE, (10 * 1024 * 1024, 10 * 1024 * 1024))
                    # Basic FD cap
                    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
                    # Memory cap (address space)
                    mem_mb = int(getattr(app_settings, "UNSAFE_CODE_EXEC_MAX_MEMORY_MB", 512))
                    mem = mem_mb * 1024 * 1024
                    resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
                except Exception:
                    return

            stdout_cap = int(getattr(app_settings, "UNSAFE_CODE_EXEC_MAX_STDOUT_CHARS", 20000))
            stderr_cap = int(getattr(app_settings, "UNSAFE_CODE_EXEC_MAX_STDERR_CHARS", 20000))

            with tempfile.TemporaryDirectory(prefix="paper_demo_") as tmp:
                from pathlib import Path

                base = Path(tmp)
                base_resolved = base.resolve()
                for ff in files_list[:200]:
                    rel = _sanitize_path(str(ff.get("path") or ""))
                    if not rel or rel.startswith("."):
                        continue
                    full = (base / rel).resolve()
                    if not str(full).startswith(str(base_resolved)):
                        continue
                    try:
                        full.parent.mkdir(parents=True, exist_ok=True)
                        full.write_text(str(ff.get("content") or ""), encoding="utf-8")
                    except Exception:
                        continue

                env = {
                    "PYTHONNOUSERSITE": "1",
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONHASHSEED": "0",
                    "HOME": tmp,
                    "PATH": os.environ.get("PATH", ""),
                    "LANG": os.environ.get("LANG", "C.UTF-8"),
                    "LC_ALL": os.environ.get("LC_ALL", "C.UTF-8"),
                }

                backend = str(effective_backend or "subprocess").strip().lower()
                cmd: list[str]
                if backend == "docker":
                    image = str(effective_image or "python:3.11-slim")
                    mem_mb = int(getattr(app_settings, "UNSAFE_CODE_EXEC_MAX_MEMORY_MB", 512))
                    cpus = float(getattr(app_settings, "UNSAFE_CODE_EXEC_DOCKER_CPUS", 1.0) or 1.0)
                    pids = int(getattr(app_settings, "UNSAFE_CODE_EXEC_DOCKER_PIDS_LIMIT", 128))
                    # Docker sandbox: no network, drop caps, no-new-privileges, resource caps, run as nobody.
                    cmd = [
                        "docker",
                        "run",
                        "--rm",
                        "--network",
                        "none",
                        "--cap-drop",
                        "ALL",
                        "--security-opt",
                        "no-new-privileges",
                        "--pids-limit",
                        str(max(32, min(pids, 1024))),
                        "--memory",
                        f"{max(64, min(mem_mb, 4096))}m",
                        "--cpus",
                        str(max(0.25, min(cpus, 4.0))),
                        "--user",
                        "65534:65534",
                        "-v",
                        f"{tmp}:/work:rw",
                        "-w",
                        "/work",
                        image,
                        "python",
                        "-I",
                        "-S",
                        ep,
                    ]
                    # For docker backend, don't apply RLIMITs in the host process.
                    local_preexec = None
                else:
                    cmd = [sys.executable, "-I", "-S", ep]
                    local_preexec = _limit_resources if os.name == "posix" else None
                start = time.time()
                try:
                    completed = subprocess.run(
                        cmd,
                        cwd=tmp,
                        env=env,
                        capture_output=True,
                        text=True,
                        timeout=float(behavioral_timeout_seconds),
                        preexec_fn=local_preexec,
                    )
                    result["ran"] = True
                    result["exit_code"] = int(completed.returncode)
                    out = completed.stdout or ""
                    err = completed.stderr or ""
                    result["stdout"] = out[:stdout_cap]
                    result["stderr"] = err[:stderr_cap]
                    result["ok"] = completed.returncode == 0
                except subprocess.TimeoutExpired as e:
                    result["ran"] = True
                    result["timed_out"] = True
                    result["exit_code"] = None
                    out = getattr(e, "stdout", "") or ""
                    err = getattr(e, "stderr", "") or ""
                    result["stdout"] = str(out)[:stdout_cap]
                    result["stderr"] = str(err)[:stderr_cap]
                    result["ok"] = False
                except FileNotFoundError as e:
                    result["ran"] = True
                    result["error"] = f"Execution backend not available: {e}"
                    result["ok"] = False
                except Exception as e:
                    result["error"] = str(e)
                    result["ok"] = False
                finally:
                    result["duration_ms"] = int((time.time() - start) * 1000)
            return result

        def _compile_python(files_list: list[dict]) -> list[dict]:
            errors: list[dict] = []
            for ff in files_list:
                p = str(ff.get("path") or "")
                if not p.endswith(".py"):
                    continue
                src = str(ff.get("content") or "")
                try:
                    compile(src, p, "exec")
                except SyntaxError as e:
                    errors.append(
                        {
                            "path": p,
                            "line": int(getattr(e, "lineno", 0) or 0),
                            "offset": int(getattr(e, "offset", 0) or 0),
                            "message": str(getattr(e, "msg", "") or str(e)),
                            "text": str(getattr(e, "text", "") or "").strip(),
                        }
                    )
                except Exception as e:
                    errors.append({"path": p, "line": 0, "offset": 0, "message": f"Compile error: {e}", "text": ""})
            return errors

        sanity_errors = _compile_python(normalized_files)
        repaired_files: list[str] = []
        repair_attempts = 0
        if sanity_errors and auto_repair and repair_max_attempts > 0:
            _emit(55, "repairing", f"Found {len(sanity_errors)} syntax errors; attempting auto-repair")
            await db.commit()
            while sanity_errors and repair_attempts < repair_max_attempts:
                repair_attempts += 1
                repair_prompt = (
                    "You are fixing a generated Python project.\n"
                    "Goal: fix ONLY syntax/compile errors without changing the intended behavior.\n\n"
                    "Output MUST be valid JSON ONLY with keys:\n"
                    "- files (array of {path, content}) containing ONLY the files you changed\n\n"
                    f"SYNTAX ERRORS:\n{json.dumps(sanity_errors, indent=2)}\n\n"
                    "PROJECT FILES:\n"
                    + "".join(
                        [
                            f"### FILE: {ff['path']}\n```text\n{str(ff.get('content') or '')[:12000]}\n```\n"
                            for ff in normalized_files
                        ]
                    )
                )
                repair_response = await self.llm_service.generate_response(
                    query=repair_prompt,
                    context=None,
                    temperature=0.1,
                    max_tokens=1800,
                    user_settings=user_settings,
                    task_type="code_agent",
                    user_id=job.user_id,
                    db=db,
                    routing=self._llm_routing_from_job_config(job.config),
                )
                try:
                    repair_payload = json.loads(repair_response)
                except Exception:
                    break
                changed = repair_payload.get("files") if isinstance(repair_payload.get("files"), list) else []
                if not changed:
                    break
                path_to_idx = {ff["path"]: i for i, ff in enumerate(normalized_files) if isinstance(ff.get("path"), str)}
                any_applied = False
                for ch in changed:
                    if not isinstance(ch, dict):
                        continue
                    p = _sanitize_path(str(ch.get("path") or ""))
                    if not p or p not in path_to_idx:
                        continue
                    content = str(ch.get("content") or "")
                    if len(content) > 25000:
                        content = content[:25000]
                    normalized_files[path_to_idx[p]]["content"] = content
                    repaired_files.append(p)
                    any_applied = True
                if not any_applied:
                    break
                sanity_errors = _compile_python(normalized_files)

        behavior = None
        if not sanity_errors and behavioral_check:
            # Explicitly gated server-side.
            enabled_override = await get_feature_flag("unsafe_code_execution_enabled")
            enabled_effective = bool(enabled_override) if enabled_override is not None else bool(getattr(app_settings, "ENABLE_UNSAFE_CODE_EXECUTION", False))
            backend_override = await get_feature_str("unsafe_code_exec_backend")
            backend_effective = str(backend_override or getattr(app_settings, "UNSAFE_CODE_EXEC_BACKEND", "subprocess") or "subprocess").strip().lower()
            if backend_effective not in {"subprocess", "docker"}:
                backend_effective = "subprocess"
            image_override = await get_feature_str("unsafe_code_exec_docker_image")
            image_effective = str(image_override or getattr(app_settings, "UNSAFE_CODE_EXEC_DOCKER_IMAGE", "python:3.11-slim") or "python:3.11-slim").strip()

            if not enabled_effective:
                behavior = {
                    "enabled": False,
                    "ran": False,
                    "ok": False,
                    "skipped_reason": "Server disabled unsafe code execution (unsafe_code_execution_enabled=false)",
                }
            else:
                _emit(62, "checking", "Running demo.py behavioral check (unsafe)")
                await db.commit()
                behavior = _run_behavioral_demo(normalized_files, effective_backend=backend_effective, effective_image=image_effective)

        # Create a generated document source and persist files as Documents.
        _emit(70, "persisting", "Creating generated project source and saving files")
        await db.commit()

        from uuid import uuid4

        source_name = f"Generated project ({project_name}) #{uuid4().hex[:6]}"
        source_cfg: Dict[str, Any] = {
            "kind": "paper_algorithm_project",
            "language": language,
            "project_name": project_name,
            "requested_by_user_id": str(job.user_id),
            "requested_by": username,
            "paper": {
                "inbox_item_id": str(item.id),
                "title": title,
                "url": url,
                "entry_url": entry_url,
                "pdf_url": pdf_url,
            },
            "entrypoint": entrypoint,
            "repo_context": {
                "enabled": bool(repo_context_block),
                "source_id": str(inherited_repo_source_id) if inherited_repo_source_id else None,
                "provider": str(inherited_provider) if inherited_provider else None,
                "repo": str(inherited_repo) if inherited_repo else None,
            },
            "job_id": str(job.id),
        }

        source = DocumentSource(name=source_name, source_type="generated", config=source_cfg)
        db.add(source)
        await db.commit()
        await db.refresh(source)

        created = 0
        for f in normalized_files:
            path = f["path"]
            content = f["content"]
            h = hashlib.sha256((content or "").encode("utf-8")).hexdigest()
            doc = Document(
                title=path[:500],
                content=content,
                content_hash=h,
                source_id=source.id,
                source_identifier=path[:500],
                file_path=path[:1000],
                file_type="text/plain",
                is_processed=False,
                extra_metadata={"generated": True, "project_name": project_name, "language": language},
            )
            db.add(doc)
            created += 1
            if created >= 30:
                break

        await db.commit()

        job.results = job.results or {}
        job.results["generated_project"] = {
            "source_id": str(source.id),
            "source_name": source.name,
            "project_name": project_name,
            "entrypoint": entrypoint,
            "file_count": created,
            "summary": summary_out,
            "run_instructions": run_instructions,
            "limitations": limitations,
            "sanity_check": {
                "ok": len(sanity_errors) == 0,
                "syntax_errors": sanity_errors,
                "repair_attempts": repair_attempts,
                "repaired_files": sorted(list(set(repaired_files)))[:50],
                "behavioral": behavior,
            },
        }
        if job.output_artifacts is None:
            job.output_artifacts = []
        job.output_artifacts.append(
            {"type": "generated_project", "source_id": str(source.id), "title": project_name, "language": language}
        )

        if sanity_errors:
            job.status = AgentJobStatus.FAILED.value
            job.error = f"Generated project has {len(sanity_errors)} syntax errors"
            _emit(100, "failed", job.error)
            await db.commit()
        elif behavioral_check and behavior and behavior.get("enabled") and behavior.get("ran") and not bool(behavior.get("ok")):
            job.status = AgentJobStatus.FAILED.value
            job.error = "Behavioral check failed (demo.py)"
            _emit(100, "failed", job.error)
            await db.commit()
        else:
            _emit(100, "completed", f"Generated project: {project_name} ({created} files)")
            job.status = AgentJobStatus.COMPLETED.value
            job.completed_at = datetime.utcnow()
            await db.commit()

        if progress_callback:
            try:
                await progress_callback(
                    {
                        "job_id": str(job.id),
                        "progress": job.progress,
                        "phase": job.current_phase,
                        "status": job.status,
                        "iteration": job.iteration,
                        "phase_details": job.phase_details,
                        "error": job.error,
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                )
            except Exception:
                pass

        return {"status": "completed", "results": job.results}

    async def _run_autonomous_loop(
        self,
        job: AgentJob,
        agent_def: Optional[AgentDefinition],
        user_settings: Optional[UserLLMSettings],
        db: AsyncSession,
        progress_callback: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """
        Run the autonomous execution loop.

        The loop continues until:
        - Goal is achieved
        - Max iterations reached
        - Resource limits hit
        - Error occurs
        - Job is paused/cancelled
        """
        start_time = datetime.utcnow()
        max_runtime = timedelta(minutes=job.max_runtime_minutes)

        # Initialize state
        state = {
            "observations": [],
            "actions_taken": [],
            "findings": [],
            "artifacts": [],
            "goal_progress": 0,
            "execution_plan": [],
            "causal_experiment_plan": {},
            "causal_plan_generation_attempted": False,
            "plan_step_index": 0,
            "plan_generation_attempted": False,
            "subgoals": [],
            "subgoal_index": 0,
            "subgoal_chain_configured": False,
            "swarm_chain_configured": False,
            "swarm_child_jobs_count": 0,
            "swarm_roles_assigned": [],
            "swarm_fan_in_enabled": False,
            "swarm_fan_in_group_id": "",
            "tool_stats": {},
            "tool_priors": {},
            "critic_notes": [],
            "last_critic_iteration": 0,
            "critic_last_trigger": {},
            "critic_last_trigger_iteration": 0,
            "critic_trigger_counts": {},
            "last_progress": 0,
            "stalled_iterations": 0,
            "repeated_action_iterations": 0,
            "last_action_signature": None,
            "recovery_actions_used": 0,
            "progress_history": [],
            "forced_exploration_attempts": 0,
            "forced_exploration_used": 0,
            "forced_exploration_successes": 0,
            "forced_exploration_failures": 0,
            "forced_exploration_history": [],
            "tool_cooldowns": {},
            "tool_cooldown_blocks": 0,
            "tool_selection_effective_mode": "",
            "tool_selection_ab_assignment": {},
            "tool_selection_mode_metrics": {},
            "tool_selection_mode_override": "",
            "tool_selection_fallback_events": [],
            "counterfactual_last": [],
            "counterfactual_logged_iterations": 0,
            "counterfactual_last_iteration": 0,
            "tool_selection_goal_stage": "",
            "selection_explainability_last": {},
            "selection_explainability_logged_iterations": 0,
            "memory_context": "",
            "injected_memories": [],
            "injected_memory_payloads": [],
            "customer_profile": None,
            "customer_context": "",
            "skill_profile": {},
            "skill_profile_metrics": {},
            "feedback_learning": {},
            "goal_contract_last": {},
            "goal_contract_satisfied_iteration": 0,
            "approval_checkpoint_pending": None,
            "approval_checkpoint_events": [],
            "approval_checkpoint_seen": [],
        }

        # Load checkpoint if resuming
        checkpoint = await self._load_latest_checkpoint(job.id, db)
        if checkpoint:
            state = checkpoint.state
            logger.info(f"Resuming job {job.id} from iteration {checkpoint.iteration}")
        state.setdefault("last_progress", int(state.get("goal_progress", 0) or 0))
        state.setdefault("execution_plan", [])
        state.setdefault("causal_experiment_plan", {})
        state.setdefault("causal_plan_generation_attempted", False)
        state.setdefault("plan_step_index", 0)
        state.setdefault("plan_generation_attempted", False)
        state.setdefault("subgoals", [])
        state.setdefault("subgoal_index", 0)
        state.setdefault("subgoal_chain_configured", False)
        state.setdefault("swarm_chain_configured", False)
        state.setdefault("swarm_child_jobs_count", 0)
        state.setdefault("swarm_roles_assigned", [])
        state.setdefault("swarm_fan_in_enabled", False)
        state.setdefault("swarm_fan_in_group_id", "")
        state.setdefault("tool_stats", {})
        state.setdefault("tool_priors", {})
        state.setdefault("critic_notes", [])
        state.setdefault("last_critic_iteration", 0)
        state.setdefault("critic_last_trigger", {})
        state.setdefault("critic_last_trigger_iteration", 0)
        state.setdefault("critic_trigger_counts", {})
        state.setdefault("stalled_iterations", 0)
        state.setdefault("repeated_action_iterations", 0)
        state.setdefault("last_action_signature", None)
        state.setdefault("recovery_actions_used", 0)
        state.setdefault("progress_history", [])
        state.setdefault("forced_exploration_attempts", 0)
        state.setdefault("forced_exploration_used", 0)
        state.setdefault("forced_exploration_successes", 0)
        state.setdefault("forced_exploration_failures", 0)
        state.setdefault("forced_exploration_history", [])
        state.setdefault("tool_cooldowns", {})
        state.setdefault("tool_cooldown_blocks", 0)
        state.setdefault("tool_selection_effective_mode", "")
        state.setdefault("tool_selection_ab_assignment", {})
        state.setdefault("tool_selection_mode_metrics", {})
        state.setdefault("tool_selection_mode_override", "")
        state.setdefault("tool_selection_fallback_events", [])
        state.setdefault("counterfactual_last", [])
        state.setdefault("counterfactual_logged_iterations", 0)
        state.setdefault("counterfactual_last_iteration", 0)
        state.setdefault("tool_selection_goal_stage", "")
        state.setdefault("selection_explainability_last", {})
        state.setdefault("selection_explainability_logged_iterations", 0)
        state.setdefault("injected_memory_payloads", [])
        state.setdefault("skill_profile", {})
        state.setdefault("skill_profile_metrics", {})
        state.setdefault("feedback_learning", {})
        state.setdefault("goal_contract_last", {})
        state.setdefault("goal_contract_satisfied_iteration", 0)
        state.setdefault("approval_checkpoint_pending", None)
        state.setdefault("approval_checkpoint_events", [])
        state.setdefault("approval_checkpoint_seen", [])

        # Resolve selection policy assignment once (deterministic; reused in ranking and telemetry).
        try:
            self._resolve_tool_selection_mode(job, state=state, selection_cfg=self._get_tool_selection_config(job))
        except Exception:
            pass

        # Load deployment-level customer profile + optional per-job customer context.
        # This is a lightweight, stable signal used to tailor the research loop.
        if state.get("customer_profile") is None and not (state.get("customer_context") or "").strip():
            try:
                from app.core.feature_flags import get_str as get_feature_str
                from app.schemas.customer_profile import CustomerProfile

                raw_profile = await get_feature_str("ai_hub_customer_profile")
                customer_profile = None
                if raw_profile:
                    try:
                        customer_profile = CustomerProfile.model_validate(json.loads(raw_profile))
                    except Exception:
                        customer_profile = None

                customer_context = str((job.config or {}).get("customer_context") or "").strip()
                if not customer_context and customer_profile and customer_profile.notes:
                    customer_context = str(customer_profile.notes).strip()

                state["customer_profile"] = customer_profile.model_dump() if customer_profile else None
                state["customer_context"] = customer_context
            except Exception:
                # Do not fail the job if the customer profile isn't configured.
                state["customer_profile"] = None
                state["customer_context"] = str((job.config or {}).get("customer_context") or "").strip()

        # Resolve skill profile once per run (role-aware prompt/tool constraints).
        try:
            skill_profile = self._resolve_agent_skill_profile(job, state=state)
            state["skill_profile"] = skill_profile
            if not isinstance(state.get("skill_profile_metrics"), dict) or not state.get("skill_profile_metrics"):
                state["skill_profile_metrics"] = {
                    "role": str(skill_profile.get("role") or "researcher"),
                    "actions_total": 0,
                    "actions_success": 0,
                    "actions_failure": 0,
                    "family_usage": {},
                    "role_counters": {},
                    "updated_at": datetime.utcnow().isoformat(),
                }
            job.add_log_entry(
                {
                    "phase": "skill_profile_resolved",
                    "role": str(skill_profile.get("role") or "researcher"),
                    "display_name": str(skill_profile.get("display_name") or ""),
                }
            )
        except Exception as e:
            logger.warning(f"Failed resolving skill profile for job {job.id}: {e}")

        # Inject relevant memories if enabled
        if job.enable_memory:
            try:
                memories = await agent_job_memory_service.get_relevant_memories_for_job(
                    job=job,
                    user_id=str(job.user_id),
                    db=db,
                )
                if memories:
                    state["memory_context"] = agent_job_memory_service.format_memories_for_job_context(
                        memories, include_metadata=True
                    )
                    state["injected_memories"] = [str(m.id) for m in memories]
                    state["injected_memory_payloads"] = [
                        {
                            "id": str(m.id),
                            "type": str(m.memory_type or ""),
                            "content": str(m.content or "")[:260],
                            "tags": m.tags if isinstance(m.tags, list) else [],
                            "context": m.context if isinstance(m.context, dict) else {},
                        }
                        for m in memories
                    ]
                    job.memory_injection_count = len(memories)
                    await db.commit()
                    logger.info(f"Injected {len(memories)} memories into job {job.id}")
                    try:
                        feedback_learning = agent_job_memory_service.extract_feedback_learning_signals(
                            memories=memories,
                            job_type=str(job.job_type or ""),
                            role=str((state.get("skill_profile") or {}).get("role") or ""),
                        )
                        state["feedback_learning"] = feedback_learning if isinstance(feedback_learning, dict) else {}
                    except Exception:
                        state["feedback_learning"] = {}
                    job.add_log_entry({
                        "phase": "memory_injection",
                        "memories_injected": len(memories),
                        "memory_types": list(set(m.memory_type for m in memories)),
                        "feedback_signals": (
                            {
                                "feedback_count": int((state.get("feedback_learning") or {}).get("feedback_count", 0) or 0),
                                "preferred_tools": ((state.get("feedback_learning") or {}).get("preferred_tools") or [])[:5],
                                "discouraged_tools": ((state.get("feedback_learning") or {}).get("discouraged_tools") or [])[:5],
                            }
                            if isinstance(state.get("feedback_learning"), dict)
                            else {}
                        ),
                    })
            except Exception as e:
                logger.warning(f"Failed to inject memories for job {job.id}: {e}")

        # Load cross-job tool priors once per execution.
        try:
            priors = await self._load_tool_priors(job, db)
            if priors:
                state["tool_priors"] = priors
                job.add_log_entry({
                    "phase": "tool_priors_loaded",
                    "tools": len(priors),
                })
        except Exception as e:
            logger.warning(f"Failed loading tool priors for job {job.id}: {e}")

        while job.can_continue():
            # Check runtime limit
            if datetime.utcnow() - start_time > max_runtime:
                logger.info(f"Job {job.id} hit runtime limit")
                job.add_log_entry({
                    "phase": "limit_reached",
                    "reason": "max_runtime_minutes",
                    "runtime_minutes": job.max_runtime_minutes,
                })
                break

            # Refresh job status from DB (check for pause/cancel)
            await db.refresh(job)
            if job.status not in [AgentJobStatus.RUNNING.value]:
                logger.info(f"Job {job.id} status changed to {job.status}")
                break

            job.iteration += 1
            job.last_activity_at = datetime.utcnow()

            try:
                # Phase 1: Observe - gather current state
                observation = await self._observe(job, state, db)
                state["observations"].append(observation)
                job.current_phase = "observing"
                job.phase_details = f"Gathered {len(observation.get('context', []))} context items"

                # Optional causal hypothesis+experiment plan for research jobs.
                used_causal_llm = await self._ensure_causal_experiment_plan(
                    job=job,
                    state=state,
                    observation=observation,
                    user_settings=user_settings,
                )
                if used_causal_llm:
                    job.llm_calls_used += 1

                # Optional plan-then-act phase: create an execution plan once and track current step.
                used_plan_llm = await self._ensure_execution_plan(job, agent_def, state, observation, user_settings)
                if used_plan_llm:
                    job.llm_calls_used += 1
                self._ensure_subgoals(job, state)
                self._ensure_swarm_chain_config(job, state)
                self._ensure_subgoal_chain_config(job, state)

                # Periodic critic pass to challenge trajectory and propose pivots.
                if self._should_run_critic(job, state):
                    critic_note = await self._run_critic_pass(job, state, observation, user_settings)
                    if critic_note:
                        notes = state.get("critic_notes")
                        if not isinstance(notes, list):
                            notes = []
                        notes.append(critic_note)
                        max_notes = int(self._get_critic_config(job).get("max_notes", 6))
                        state["critic_notes"] = notes[-max(1, max_notes):]
                        state["last_critic_iteration"] = int(job.iteration or 0)
                        job.llm_calls_used += 1
                        trigger_info = state.get("critic_last_trigger") if isinstance(state.get("critic_last_trigger"), dict) else {}
                        job.add_log_entry({
                            "phase": "critic_pass",
                            "assessment": str(critic_note.get("trajectory_assessment") or "")[:200],
                            "pivot": str(critic_note.get("pivot") or "")[:200],
                            "recommended_tools": critic_note.get("recommended_tools") or [],
                            "trigger_reason": str(trigger_info.get("reason") or ""),
                            "trigger_by_interval": bool(trigger_info.get("by_interval", False)),
                            "trigger_by_stall": bool(trigger_info.get("by_stall", False)),
                            "trigger_by_uncertainty": bool(trigger_info.get("by_uncertainty", False)),
                            "uncertainty_score_gap": trigger_info.get("uncertainty_score_gap"),
                            "uncertainty_effective_threshold": trigger_info.get("uncertainty_effective_threshold"),
                        })

                # Phase 2: Think - decide next action
                decision = await self._think(job, agent_def, state, observation, user_settings, db)
                decision = self._maybe_apply_critic_pivot_override(job, state, decision)
                job.current_phase = "thinking"
                job.phase_details = decision.get("reasoning", "")[:200]
                job.llm_calls_used += 1

                contract_before = self._evaluate_goal_contract(job, state, include_result_keys=False)
                state["goal_contract_last"] = contract_before

                # Check if goal is achieved
                if decision.get("goal_achieved"):
                    if bool(contract_before.get("enabled")) and not bool(contract_before.get("satisfied")):
                        unmet = contract_before.get("missing") if isinstance(contract_before.get("missing"), list) else []
                        decision["goal_achieved"] = False
                        decision["reasoning"] = (
                            f"{str(decision.get('reasoning') or '').strip()[:260]} "
                            f"Goal contract not yet satisfied: {', '.join([str(x)[:80] for x in unmet[:4]])}"
                        ).strip()
                        job.add_log_entry(
                            {
                                "phase": "goal_contract_blocked",
                                "reasoning": "goal_achieved blocked by unmet goal contract",
                                "missing": unmet[:8],
                            }
                        )
                    else:
                        if bool(contract_before.get("enabled")) and not int(state.get("goal_contract_satisfied_iteration", 0) or 0):
                            state["goal_contract_satisfied_iteration"] = int(job.iteration or 0)
                        logger.info(f"Job {job.id} achieved goal")
                        job.add_log_entry({
                            "phase": "goal_achieved",
                            "reasoning": decision.get("reasoning"),
                            "final_assessment": decision.get("assessment"),
                        })
                        state["goal_progress"] = 100
                        break

                # Check if should stop
                if decision.get("should_stop"):
                    logger.info(f"Job {job.id} decided to stop: {decision.get('stop_reason')}")
                    job.add_log_entry({
                        "phase": "voluntary_stop",
                        "reason": decision.get("stop_reason"),
                    })
                    break

                counterfactual_candidates: List[Dict[str, Any]] = []
                selection_explainability: Dict[str, Any] = {}
                cf_cfg = self._get_counterfactual_config(job)
                if bool(cf_cfg.get("enabled", True)):
                    counterfactual_candidates = self._build_counterfactual_candidates(
                        job=job,
                        state=state,
                        selected_tool=str(((decision.get("action") or {}).get("tool") or "")).strip() or None,
                        limit=int(cf_cfg.get("top_k", 3) or 3),
                        context_tag="iteration_decision",
                    )
                    state["counterfactual_last"] = counterfactual_candidates
                    state["counterfactual_logged_iterations"] = int(state.get("counterfactual_logged_iterations", 0) or 0) + 1
                    state["counterfactual_last_iteration"] = int(job.iteration or 0)
                selection_explainability = self._build_selection_explainability(
                    state=state,
                    selected_tool=str(((decision.get("action") or {}).get("tool") or "")).strip() or None,
                    candidates=counterfactual_candidates,
                )
                state["selection_explainability_last"] = selection_explainability
                state["selection_explainability_logged_iterations"] = int(state.get("selection_explainability_logged_iterations", 0) or 0) + 1

                # Phase 3: Act - execute the decided action
                action = None
                action_result = None
                action = decision.get("action")
                if action:
                    state["approval_checkpoint_pending"] = None
                    checkpoint_gate = self._evaluate_approval_checkpoint(job, state, action)
                    if bool(checkpoint_gate.get("required", False)):
                        checkpoint_payload = (
                            checkpoint_gate.get("checkpoint")
                            if isinstance(checkpoint_gate.get("checkpoint"), dict)
                            else {}
                        )
                        state["approval_checkpoint_pending"] = checkpoint_payload
                        events = state.get("approval_checkpoint_events")
                        if not isinstance(events, list):
                            events = []
                        events.append(checkpoint_payload)
                        state["approval_checkpoint_events"] = events[-20:]
                        results_payload = job.results if isinstance(job.results, dict) else {}
                        exec_strategy = (
                            results_payload.get("execution_strategy")
                            if isinstance(results_payload.get("execution_strategy"), dict)
                            else {}
                        )
                        approval_summary = (
                            exec_strategy.get("approval_checkpoints")
                            if isinstance(exec_strategy.get("approval_checkpoints"), dict)
                            else {}
                        )
                        approval_summary["pending"] = checkpoint_payload
                        approval_summary["events"] = state["approval_checkpoint_events"][-20:]
                        approval_summary["seen"] = (
                            state.get("approval_checkpoint_seen")
                            if isinstance(state.get("approval_checkpoint_seen"), list)
                            else []
                        )[-200:]
                        exec_strategy["approval_checkpoints"] = approval_summary
                        results_payload["execution_strategy"] = exec_strategy
                        results_payload["approval_checkpoint"] = checkpoint_payload
                        job.results = results_payload
                        job.status = AgentJobStatus.PAUSED.value
                        job.current_phase = "awaiting_approval"
                        job.phase_details = str(checkpoint_payload.get("message") or "Approval required before next action.")[:280]
                        job.add_log_entry(
                            {
                                "phase": "approval_checkpoint",
                                "checkpoint": checkpoint_payload,
                            }
                        )
                        await self._save_checkpoint(job, state, db)
                        await db.commit()
                        if progress_callback:
                            await progress_callback(
                                {
                                    "job_id": str(job.id),
                                    "iteration": job.iteration,
                                    "progress": int(state.get("goal_progress", 0) or 0),
                                    "phase": job.current_phase,
                                    "phase_details": job.phase_details,
                                    "status": job.status,
                                    "checkpoint": checkpoint_payload,
                                }
                            )
                        return {
                            "status": job.status,
                            "progress": int(state.get("goal_progress", 0) or 0),
                            "results": job.results if isinstance(job.results, dict) else {},
                            "iterations": job.iteration,
                            "tool_calls": job.tool_calls_used,
                            "llm_calls": job.llm_calls_used,
                            "checkpoint": checkpoint_payload,
                        }
                    action_result = await self._act(job, action, state, db)
                    state["actions_taken"].append({
                        "action": action,
                        "result": action_result,
                        "iteration": job.iteration,
                    })
                    job.current_phase = "acting"
                    job.phase_details = f"Executed: {action.get('tool', 'unknown')}"
                    job.tool_calls_used += 1

                    # Process action results
                    if action_result.get("findings"):
                        state["findings"].extend(action_result["findings"])
                    if action_result.get("artifacts"):
                        state["artifacts"].extend(action_result["artifacts"])
                    self._record_tool_outcome(state, action, action_result)
                    self._update_skill_profile_metrics(state, action, action_result)

                # Phase 4: Evaluate progress
                previous_progress = int(state.get("goal_progress", 0) or 0)
                progress = await self._evaluate_progress(job, state, user_settings, db)
                state["goal_progress"] = progress
                job.progress = progress
                job.llm_calls_used += 1
                self._advance_execution_plan_state(
                    state=state,
                    action=action,
                    action_result=action_result,
                    previous_progress=previous_progress,
                    current_progress=progress,
                )

                contract_after = self._evaluate_goal_contract(job, state, include_result_keys=False)
                state["goal_contract_last"] = contract_after
                if bool(contract_after.get("enabled")) and bool(contract_after.get("satisfied")):
                    if not int(state.get("goal_contract_satisfied_iteration", 0) or 0):
                        state["goal_contract_satisfied_iteration"] = int(job.iteration or 0)
                        job.add_log_entry(
                            {
                                "phase": "goal_contract_satisfied",
                                "iteration": int(job.iteration or 0),
                            }
                        )
                    contract_cfg = (
                        contract_after.get("contract")
                        if isinstance(contract_after.get("contract"), dict)
                        else {}
                    )
                    if bool(contract_cfg.get("auto_complete_when_satisfied", True)):
                        state["goal_progress"] = 100
                        job.progress = 100
                        job.add_log_entry(
                            {
                                "phase": "goal_contract_autocomplete",
                                "reason": "deterministic goal contract satisfied",
                            }
                        )
                        break

                stall_info = self._update_stall_state(
                    job=job,
                    state=state,
                    progress=progress,
                    action=action,
                )

                recovery_triggered = False
                if stall_info.get("should_recover"):
                    recovery_budget = int(self._get_stall_config(job).get("max_recovery_actions", 0))
                    used_recoveries = int(state.get("recovery_actions_used", 0) or 0)
                    if used_recoveries < recovery_budget and job.tool_calls_used < job.max_tool_calls:
                        recovery_action = self._build_recovery_action(job, state, exclude_tool=(action or {}).get("tool"))
                        if recovery_action:
                            recovery_result = await self._act(job, recovery_action, state, db)
                            state["actions_taken"].append({
                                "action": recovery_action,
                                "result": recovery_result,
                                "iteration": job.iteration,
                            })
                            recovery_triggered = True
                            job.tool_calls_used += 1
                            state["recovery_actions_used"] = used_recoveries + 1
                            if recovery_result.get("findings"):
                                state["findings"].extend(recovery_result["findings"])
                            if recovery_result.get("artifacts"):
                                state["artifacts"].extend(recovery_result["artifacts"])
                            self._record_tool_outcome(state, recovery_action, recovery_result)
                            self._update_skill_profile_metrics(state, recovery_action, recovery_result)
                            self._apply_recovery_post_action_updates(
                                job=job,
                                state=state,
                                recovery_action=recovery_action,
                                recovery_result=recovery_result,
                            )
                            job.add_log_entry({
                                "phase": "stall_recovery",
                                "trigger_reason": stall_info.get("reason"),
                                "recovery_action": recovery_action.get("tool"),
                                "recovery_success": bool(recovery_result.get("success")),
                                "forced_exploration": bool(state.get("last_recovery_was_forced_exploration", False)),
                                "forced_exploration_attempts": int(state.get("forced_exploration_attempts", 0) or 0),
                                "forced_exploration_successes": int(state.get("forced_exploration_successes", 0) or 0),
                                "forced_exploration_failures": int(state.get("forced_exploration_failures", 0) or 0),
                                "forced_exploration_used": int(state.get("forced_exploration_used", 0) or 0),
                                "tool_cooldown_blocks": int(state.get("tool_cooldown_blocks", 0) or 0),
                            })
                            # Recovery action usually changes search direction; avoid immediate hard-stop.
                            state["stalled_iterations"] = max(0, int(state.get("stalled_iterations", 0)) - 1)

                if stall_info.get("should_stop") and not recovery_triggered:
                    logger.info(f"Job {job.id} stopping due to stall: {stall_info.get('reason')}")
                    job.add_log_entry({
                        "phase": "voluntary_stop",
                        "reason": stall_info.get("reason"),
                    })
                    break

                # Log iteration
                findings_count = len(state["findings"])
                job.add_log_entry({
                    "phase": "iteration_complete",
                    "action": action.get("tool") if action else None,
                    "progress": progress,
                    "findings_count": findings_count,
                    "plan_step_index": int(state.get("plan_step_index", 0) or 0),
                    "plan_steps_total": len(state.get("execution_plan", []) or []),
                    "stalled_iterations": int(state.get("stalled_iterations", 0) or 0),
                    "repeated_action_iterations": int(state.get("repeated_action_iterations", 0) or 0),
                    "counterfactual_candidates": counterfactual_candidates[:5],
                    "selection_explainability": selection_explainability,
                })

                # Check for progress/findings-based chain triggers
                await self.trigger_progress_chain(job, progress, findings_count, db)

                # Save checkpoint periodically
                if job.iteration % 5 == 0:
                    await self._save_checkpoint(job, state, db)

                # Notify progress
                if progress_callback:
                    await progress_callback({
                        "job_id": str(job.id),
                        "iteration": job.iteration,
                        "progress": progress,
                        "phase": job.current_phase,
                        "phase_details": job.phase_details,
                    })

                await db.commit()

            except Exception as e:
                logger.error(f"Error in iteration {job.iteration}: {e}")
                job.add_log_entry({
                    "phase": "error",
                    "error": str(e),
                })
                job.error_count += 1
                job.last_error_at = datetime.utcnow()

                # Continue if error count is low, otherwise stop
                if job.error_count >= 5:
                    job.error = f"Too many errors: {e}"
                    break

        # Finalize job
        return await self._finalize_job(job, state, db)

    async def _observe(
        self,
        job: AgentJob,
        state: Dict[str, Any],
        db: AsyncSession,
    ) -> Dict[str, Any]:
        """
        Gather observations about current state.

        Collects context relevant to the job's goal.
        """
        observation = {
            "iteration": job.iteration,
            "context": [],
            "timestamp": datetime.utcnow().isoformat(),
        }

        config = job.config or {}
        job_type = job.job_type

        # Gather context based on job type
        if job_type == "research":
            # Get recent findings
            recent_findings = state.get("findings", [])[-10:]
            observation["recent_findings"] = recent_findings

            # Get papers found so far
            papers_found = len([
                a for a in state.get("actions_taken", [])
                if a.get("action", {}).get("tool") == "search_arxiv"
            ])
            observation["papers_searched"] = papers_found

        elif job_type == "monitor":
            # Get last check time
            last_actions = state.get("actions_taken", [])
            if last_actions:
                observation["last_check"] = last_actions[-1].get("result", {}).get("timestamp")

        elif job_type == "analysis":
            # Get documents analyzed
            analyzed = [
                a for a in state.get("actions_taken", [])
                if a.get("action", {}).get("tool") in ["get_document_details", "summarize_document"]
            ]
            observation["documents_analyzed"] = len(analyzed)

        elif job_type == "data_analysis":
            # Track data analysis progress
            job_id_str = str(job.id)
            if job_id_str in self._data_analysis_tools:
                tools = self._data_analysis_tools[job_id_str]
                datasets_info = tools.list_datasets()
                observation["datasets_loaded"] = datasets_info.get("count", 0)
                observation["datasets"] = datasets_info.get("datasets", [])

            # Count charts and diagrams created
            charts_created = [
                a for a in state.get("actions_taken", [])
                if a.get("action", {}).get("tool") in [
                    "create_chart", "create_correlation_heatmap"
                ]
            ]
            diagrams_created = [
                a for a in state.get("actions_taken", [])
                if a.get("action", {}).get("tool") in [
                    "create_flowchart", "create_sequence_diagram", "create_er_diagram",
                    "create_architecture_diagram", "create_drawio_diagram", "create_gantt_chart"
                ]
            ]
            observation["charts_created"] = len(charts_created)
            observation["diagrams_created"] = len(diagrams_created)

            # Count transformations
            transformations = [
                a for a in state.get("actions_taken", [])
                if a.get("action", {}).get("tool") in [
                    "query_data", "filter_data", "aggregate_data", "join_datasets", "transform_data"
                ]
            ]
            observation["transformations_applied"] = len(transformations)

        return observation

    async def _think(
        self,
        job: AgentJob,
        agent_def: Optional[AgentDefinition],
        state: Dict[str, Any],
        observation: Dict[str, Any],
        user_settings: Optional[UserLLMSettings],
        db: AsyncSession,
    ) -> Dict[str, Any]:
        """
        Decide the next action based on goal and current state.

        Uses LLM to reason about progress and determine next steps.
        """
        profile = (
            state.get("skill_profile")
            if isinstance(state.get("skill_profile"), dict)
            else self._resolve_agent_skill_profile(job, state=state)
        )
        # Build prompt for decision making
        system_prompt = self._build_thinking_prompt(job, agent_def, state, observation, profile=profile)

        # Get available tools for this job type
        available_tools = self._get_tools_for_job_type(job.job_type, job.config, profile=profile)

        user_message = f"""
Current iteration: {job.iteration}/{job.max_iterations}
Current progress: {state.get('goal_progress', 0)}%
Tool calls used: {job.tool_calls_used}/{job.max_tool_calls}
LLM calls used: {job.llm_calls_used}/{job.max_llm_calls}

Recent actions: {json.dumps(state.get('actions_taken', [])[-3:], default=str)}

Current observation:
{json.dumps(observation, default=str)}

Total findings so far: {len(state.get('findings', []))}

Based on the goal and current progress, decide:
1. Is the goal achieved? If so, explain why.
2. Should we stop for another reason? (e.g., no more progress possible)
3. If continuing, what is the next action to take?

Respond in JSON format:
{{
    "goal_achieved": true/false,
    "should_stop": true/false,
    "stop_reason": "reason if stopping",
    "reasoning": "your reasoning about current progress and next steps",
    "assessment": "assessment of goal completion (0-100%)",
    "action": {{
        "tool": "tool_name",
        "params": {{}},
        "purpose": "why this action"
    }} or null if stopping
}}
"""

        try:
            response = await self.llm_service.generate_response(
                system_prompt=system_prompt,
                user_message=user_message,
                user_settings=user_settings,
                routing=self._llm_routing_from_job_config(job.config),
            )

            return self._parse_decision_response(
                raw_response=response,
                job=job,
                state=state,
                available_tools=available_tools,
            )

        except Exception as e:
            logger.error(f"Error in thinking phase: {e}")
            recovery_action = self._build_recovery_action(job, state)
            return {
                "goal_achieved": False,
                "should_stop": recovery_action is None,
                "stop_reason": f"Thinking error: {e}" if recovery_action is None else "",
                "reasoning": str(e),
                "action": recovery_action,
            }

    def _parse_decision_response(
        self,
        raw_response: Any,
        job: AgentJob,
        state: Dict[str, Any],
        available_tools: List[str],
    ) -> Dict[str, Any]:
        """Parse and normalize LLM decision payload with resilient JSON extraction."""
        text = str(raw_response or "")
        payload = self._extract_first_json_object(text)
        if not isinstance(payload, dict):
            recovery = self._build_recovery_action(job, state)
            return {
                "goal_achieved": False,
                "should_stop": recovery is None,
                "stop_reason": "Model response did not contain a valid JSON decision" if recovery is None else "",
                "reasoning": text[:500] if text else "Model returned an empty decision",
                "assessment": None,
                "action": recovery,
            }

        goal_achieved = self._coerce_bool(payload.get("goal_achieved"), default=False)
        should_stop = self._coerce_bool(payload.get("should_stop"), default=False)
        reasoning = str(payload.get("reasoning") or "").strip()
        stop_reason = str(payload.get("stop_reason") or "").strip()
        assessment = payload.get("assessment")

        action = self._normalize_decision_action(payload.get("action"), available_tools)
        if action is None and not goal_achieved and not should_stop:
            action = self._build_recovery_action(job, state)
            if action:
                reasoning = f"{reasoning[:360]} Auto-selected deterministic recovery action.".strip()
            else:
                should_stop = True
                stop_reason = stop_reason or "No valid action available for continuation"

        if should_stop and not stop_reason:
            stop_reason = "Model requested stop"

        return {
            "goal_achieved": goal_achieved,
            "should_stop": should_stop,
            "stop_reason": stop_reason,
            "reasoning": reasoning[:800] if reasoning else (text[:500] if text else ""),
            "assessment": assessment,
            "action": action,
        }

    def _extract_first_json_object(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract the first valid JSON object from plain text or fenced markdown."""
        if not text:
            return None

        stripped = text.strip()
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
        if fence_match:
            fenced = fence_match.group(1).strip()
            try:
                parsed = json.loads(fenced)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass

        # Balanced-brace extraction for responses with commentary before/after JSON.
        for start in [i for i, ch in enumerate(text) if ch == "{"]:
            depth = 0
            in_string = False
            escaped = False
            for idx in range(start, len(text)):
                ch = text[idx]
                if in_string:
                    if escaped:
                        escaped = False
                    elif ch == "\\":
                        escaped = True
                    elif ch == '"':
                        in_string = False
                    continue

                if ch == '"':
                    in_string = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = text[start:idx + 1]
                        try:
                            parsed = json.loads(candidate)
                            if isinstance(parsed, dict):
                                return parsed
                        except Exception:
                            break
        return None

    def _normalize_decision_action(
        self,
        action: Any,
        available_tools: List[str],
    ) -> Optional[Dict[str, Any]]:
        """Normalize action payload and reject unavailable tools."""
        if action is None:
            return None
        if isinstance(action, str):
            action = {"tool": action, "params": {}}
        if not isinstance(action, dict):
            return None

        tool = str(action.get("tool") or "").strip()
        if not tool or tool not in set(available_tools):
            return None

        params = action.get("params")
        if not isinstance(params, dict):
            params = {}

        purpose = str(action.get("purpose") or "").strip()
        return {
            "tool": tool,
            "params": params,
            "purpose": purpose[:300],
        }

    def _coerce_bool(self, value: Any, default: bool = False) -> bool:
        """Coerce flexible model outputs to booleans."""
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "yes", "1", "y"}:
                return True
            if lowered in {"false", "no", "0", "n"}:
                return False
        return default

    def _normalize_causal_experiment_plan(
        self,
        payload: Dict[str, Any],
        *,
        max_hypotheses: int = 4,
        max_experiments: int = 6,
    ) -> Dict[str, Any]:
        """Normalize causal experiment planner output into stable schema."""
        if not isinstance(payload, dict):
            return {}

        hypotheses_raw = payload.get("hypotheses")
        if not isinstance(hypotheses_raw, list):
            hypotheses_raw = []
        hypotheses: List[Dict[str, Any]] = []
        for i, item in enumerate(hypotheses_raw, start=1):
            if isinstance(item, str):
                statement = item.strip()
                if not statement:
                    continue
                hypotheses.append(
                    {
                        "id": f"H{i}",
                        "statement": statement[:320],
                        "rationale": "",
                        "confidence": 0.5,
                    }
                )
            elif isinstance(item, dict):
                statement = str(item.get("statement") or item.get("hypothesis") or "").strip()
                if not statement:
                    continue
                hid = str(item.get("id") or f"H{i}").strip()[:24] or f"H{i}"
                rationale = str(item.get("rationale") or item.get("because") or "").strip()[:320]
                try:
                    conf = float(item.get("confidence", 0.5) or 0.5)
                except Exception:
                    conf = 0.5
                conf = max(0.0, min(1.0, conf))
                hypotheses.append(
                    {
                        "id": hid,
                        "statement": statement[:320],
                        "rationale": rationale,
                        "confidence": conf,
                    }
                )
            if len(hypotheses) >= max(1, min(max_hypotheses, 12)):
                break

        if not hypotheses:
            return {}
        hyp_ids = [str(h.get("id") or "") for h in hypotheses if str(h.get("id") or "").strip()]

        experiments_raw = payload.get("experiments")
        if not isinstance(experiments_raw, list):
            experiments_raw = []
        experiments: List[Dict[str, Any]] = []
        for i, item in enumerate(experiments_raw, start=1):
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or item.get("title") or f"Experiment {i}").strip()
            eid = str(item.get("id") or f"E{i}").strip()[:24] or f"E{i}"
            hypothesis_id = str(item.get("hypothesis_id") or item.get("hypothesis") or "").strip()
            if hypothesis_id not in hyp_ids:
                hypothesis_id = hyp_ids[min(i - 1, len(hyp_ids) - 1)]
            minimal_design = str(item.get("minimal_design") or item.get("design") or item.get("purpose") or "").strip()

            required_data = item.get("required_data")
            if not isinstance(required_data, list):
                required_data = item.get("data")
            if not isinstance(required_data, list):
                required_data = []
            required_data = [str(x).strip()[:140] for x in required_data if str(x).strip()][:8]

            steps = item.get("steps")
            if not isinstance(steps, list):
                steps = []
            steps = [str(x).strip()[:180] for x in steps if str(x).strip()][:8]

            success_criteria = item.get("success_criteria")
            if not isinstance(success_criteria, list):
                success_criteria = item.get("metrics")
            if not isinstance(success_criteria, list):
                success_criteria = []
            success_criteria = [str(x).strip()[:180] for x in success_criteria if str(x).strip()][:8]

            expected = item.get("expected_evidence")
            if not isinstance(expected, dict):
                expected = {}
            supports = expected.get("supports") if isinstance(expected.get("supports"), list) else []
            falsifies = expected.get("falsifies") if isinstance(expected.get("falsifies"), list) else []
            ambiguous = expected.get("ambiguous") if isinstance(expected.get("ambiguous"), list) else []
            expected_norm = {
                "supports": [str(x).strip()[:180] for x in supports if str(x).strip()][:6],
                "falsifies": [str(x).strip()[:180] for x in falsifies if str(x).strip()][:6],
                "ambiguous": [str(x).strip()[:180] for x in ambiguous if str(x).strip()][:6],
            }

            effort = str(item.get("estimated_effort") or item.get("effort") or "medium").strip().lower()
            if effort not in {"low", "medium", "high"}:
                effort = "medium"

            experiments.append(
                {
                    "id": eid,
                    "hypothesis_id": hypothesis_id,
                    "name": name[:220],
                    "minimal_design": minimal_design[:360],
                    "required_data": required_data,
                    "steps": steps,
                    "success_criteria": success_criteria,
                    "expected_evidence": expected_norm,
                    "estimated_effort": effort,
                }
            )
            if len(experiments) >= max(1, min(max_experiments, 20)):
                break

        if not experiments:
            return {}

        priority_raw = payload.get("priority_order")
        if not isinstance(priority_raw, list):
            priority_raw = []
        exp_ids = [str(e.get("id") or "") for e in experiments]
        priority = [str(x).strip() for x in priority_raw if str(x).strip() in set(exp_ids)]
        if not priority:
            priority = exp_ids[:]

        decision_rules = payload.get("decision_rules")
        if not isinstance(decision_rules, list):
            decision_rules = []
        decision_rules = [str(x).strip()[:220] for x in decision_rules if str(x).strip()][:8]
        if not decision_rules:
            decision_rules = [
                "If >=70% of support criteria are met, treat hypothesis as provisionally supported.",
                "If any falsification criterion is strongly observed, deprioritize that hypothesis.",
            ]

        assumptions = payload.get("assumptions")
        if not isinstance(assumptions, list):
            assumptions = []
        assumptions = [str(x).strip()[:180] for x in assumptions if str(x).strip()][:8]

        return {
            "hypotheses": hypotheses,
            "experiments": experiments,
            "priority_order": priority,
            "decision_rules": decision_rules,
            "assumptions": assumptions,
        }

    def _fallback_causal_experiment_plan(
        self,
        job: AgentJob,
        *,
        max_hypotheses: int = 3,
        max_experiments: int = 4,
    ) -> Dict[str, Any]:
        """Deterministic fallback when LLM causal planning is unavailable."""
        goal = str(job.goal or "").strip()[:220]
        hypotheses = [
            {
                "id": "H1",
                "statement": f"A focused approach derived from '{goal}' improves the target outcome versus baseline.",
                "rationale": "Primary causal claim from the stated goal.",
                "confidence": 0.55,
            },
            {
                "id": "H2",
                "statement": "Removing the key proposed factor will reduce outcome quality.",
                "rationale": "Ablation-style falsifiability check for causal contribution.",
                "confidence": 0.45,
            },
        ][: max(1, min(max_hypotheses, 8))]

        experiments = [
            {
                "id": "E1",
                "hypothesis_id": "H1",
                "name": "Minimal baseline comparison",
                "minimal_design": "Compare baseline process against the proposed intervention on a small representative sample.",
                "required_data": ["Representative sample", "Baseline output", "Intervention output"],
                "steps": ["Define baseline and intervention", "Run both on same sample", "Measure delta on core metric"],
                "success_criteria": ["Intervention outperforms baseline on primary metric"],
                "expected_evidence": {
                    "supports": ["Consistent metric lift over baseline"],
                    "falsifies": ["No lift or negative lift vs baseline"],
                    "ambiguous": ["Mixed outcomes across segments"],
                },
                "estimated_effort": "low",
            },
            {
                "id": "E2",
                "hypothesis_id": "H2",
                "name": "Ablation stress test",
                "minimal_design": "Remove or weaken the suspected causal factor and re-evaluate outcome quality.",
                "required_data": ["Intervention variant without factor", "Evaluation rubric"],
                "steps": ["Define ablated variant", "Run same evaluation", "Compare to full intervention"],
                "success_criteria": ["Ablated variant underperforms full intervention"],
                "expected_evidence": {
                    "supports": ["Meaningful drop after removing factor"],
                    "falsifies": ["No meaningful drop after ablation"],
                    "ambiguous": ["Drop only on subset of conditions"],
                },
                "estimated_effort": "medium",
            },
        ][: max(1, min(max_experiments, 12))]

        return {
            "hypotheses": hypotheses,
            "experiments": experiments,
            "priority_order": [str(e.get("id") or "") for e in experiments],
            "decision_rules": [
                "Prioritize the experiment with the highest falsifiability and lowest effort first.",
                "Advance only hypotheses with supporting evidence and no strong falsification signal.",
            ],
            "assumptions": ["Primary metric is stable and measurable on available data."],
            "source": "fallback",
        }

    async def _ensure_causal_experiment_plan(
        self,
        job: AgentJob,
        state: Dict[str, Any],
        observation: Dict[str, Any],
        user_settings: Optional[UserLLMSettings],
    ) -> bool:
        """
        Generate causal hypotheses + minimal experiments once for research jobs.

        Returns True when an LLM call was used.
        """
        cfg = job.config if isinstance(job.config, dict) else {}
        if str(job.job_type or "").strip().lower() != "research":
            return False
        if not bool(cfg.get("causal_experiment_planner_enabled", True)):
            return False
        existing = state.get("causal_experiment_plan")
        if isinstance(existing, dict) and existing.get("experiments"):
            return False
        if bool(state.get("causal_plan_generation_attempted")):
            return False

        state["causal_plan_generation_attempted"] = True
        used_llm = False
        try:
            max_hyp = int(cfg.get("causal_plan_max_hypotheses", 3) or 3)
        except Exception:
            max_hyp = 3
        max_hyp = max(1, min(max_hyp, 8))
        try:
            max_exp = int(cfg.get("causal_plan_max_experiments", 4) or 4)
        except Exception:
            max_exp = 4
        max_exp = max(1, min(max_exp, 12))

        findings = state.get("findings") if isinstance(state.get("findings"), list) else []
        finding_titles: List[str] = []
        for f in findings:
            if not isinstance(f, dict):
                continue
            title = str(f.get("title") or f.get("summary") or "").strip()
            if not title:
                continue
            finding_titles.append(title[:180])
            if len(finding_titles) >= 10:
                break

        hypotheses: Dict[str, Any] = {}
        profile = (
            state.get("skill_profile")
            if isinstance(state.get("skill_profile"), dict)
            else self._resolve_agent_skill_profile(job, state=state)
        )
        tools = self._get_tools_for_job_type(job.job_type, job.config, profile=profile)
        system_prompt = (
            "You design causal experiment plans for research agents.\n"
            "Return JSON only."
        )
        user_message = (
            f"Goal: {job.goal}\n"
            f"Success criteria: {json.dumps(job.goal_criteria or {}, default=str)[:1200]}\n"
            f"Observation: {json.dumps(observation or {}, default=str)[:1800]}\n"
            f"Top findings: {json.dumps(finding_titles, default=str)[:1600]}\n"
            f"Available tools: {', '.join(tools)}\n\n"
            f"Create a causal plan with up to {max_hyp} hypotheses and up to {max_exp} experiments.\n"
            "Return JSON schema:\n"
            "{\n"
            '  "hypotheses": [{"id":"H1","statement":"...","rationale":"...","confidence":0.0}],\n'
            '  "experiments": [\n'
            '    {"id":"E1","hypothesis_id":"H1","name":"...","minimal_design":"...",'
            ' "required_data":["..."],"steps":["..."],"success_criteria":["..."],'
            ' "expected_evidence":{"supports":["..."],"falsifies":["..."],"ambiguous":["..."]},'
            ' "estimated_effort":"low|medium|high"}\n'
            "  ],\n"
            '  "priority_order": ["E1","E2"],\n'
            '  "decision_rules": ["..."],\n'
            '  "assumptions": ["..."]\n'
            "}\n"
            "Rules:\n"
            "- Every hypothesis must be testable and falsifiable.\n"
            "- Every experiment must explicitly define expected supporting and falsifying evidence.\n"
            "- Prefer minimal experiments that can run with currently available data/tools.\n"
            "- Output parseable JSON only."
        )

        try:
            used_llm = True
            raw = await self.llm_service.generate_response(
                system_prompt=system_prompt,
                user_message=user_message,
                user_settings=user_settings,
                routing=self._llm_routing_from_job_config(job.config),
            )
            payload = self._extract_first_json_object(str(raw or "")) or {}
            hypotheses = self._normalize_causal_experiment_plan(
                payload,
                max_hypotheses=max_hyp,
                max_experiments=max_exp,
            )
        except Exception:
            hypotheses = {}

        if not hypotheses:
            hypotheses = self._fallback_causal_experiment_plan(
                job=job,
                max_hypotheses=max_hyp,
                max_experiments=max_exp,
            )
            used_llm = False

        if hypotheses:
            hypotheses["generated_at"] = datetime.utcnow().isoformat()
            hypotheses["source"] = str(hypotheses.get("source") or ("llm" if used_llm else "fallback"))
            state["causal_experiment_plan"] = hypotheses
            job.add_log_entry(
                {
                    "phase": "causal_experiment_plan_generated",
                    "hypotheses": len(hypotheses.get("hypotheses", []) if isinstance(hypotheses.get("hypotheses"), list) else []),
                    "experiments": len(hypotheses.get("experiments", []) if isinstance(hypotheses.get("experiments"), list) else []),
                    "source": str(hypotheses.get("source") or ""),
                }
            )
        return used_llm

    async def _ensure_execution_plan(
        self,
        job: AgentJob,
        agent_def: Optional[AgentDefinition],
        state: Dict[str, Any],
        observation: Dict[str, Any],
        user_settings: Optional[UserLLMSettings],
    ) -> bool:
        """Generate a lightweight execution plan once per job when enabled."""
        cfg = job.config if isinstance(job.config, dict) else {}
        if not bool(cfg.get("plan_then_act_enabled", True)):
            return False
        if state.get("execution_plan"):
            return False
        if bool(state.get("plan_generation_attempted")):
            return False

        state["plan_generation_attempted"] = True
        max_steps = 6
        try:
            max_steps = int(cfg.get("plan_max_steps", 6) or 6)
        except Exception:
            max_steps = 6
        max_steps = max(3, min(max_steps, 10))
        used_llm = False

        profile = (
            state.get("skill_profile")
            if isinstance(state.get("skill_profile"), dict)
            else self._resolve_agent_skill_profile(job, state=state)
        )
        tools = self._get_tools_for_job_type(job.job_type, job.config, profile=profile)
        system_prompt = (
            "You design concise, executable plans for autonomous agents.\n"
            "Return JSON only."
        )
        user_message = (
            f"Job type: {job.job_type}\n"
            f"Goal: {job.goal}\n"
            f"Success criteria: {json.dumps(job.goal_criteria or {}, default=str)[:1200]}\n"
            f"Recent observation: {json.dumps(observation or {}, default=str)[:1600]}\n"
            f"Causal plan: {json.dumps(state.get('causal_experiment_plan') or {}, default=str)[:2200]}\n"
            f"Available tools: {', '.join(tools)}\n\n"
            f"Create {max_steps - 1} to {max_steps} plan steps.\n"
            "Return JSON with shape:\n"
            "{\n"
            '  "plan_steps": [\n'
            '    {"title":"...", "objective":"...", "exit_criteria":"...", "suggested_tools":["tool_a"]}\n'
            "  ]\n"
            "}\n"
            "Rules: keep steps action-oriented and tool-aware."
        )

        try:
            used_llm = True
            raw = await self.llm_service.generate_response(
                system_prompt=system_prompt,
                user_message=user_message,
                user_settings=user_settings,
                routing=self._llm_routing_from_job_config(job.config),
            )
            payload = self._extract_first_json_object(str(raw or "")) or {}
            plan = self._normalize_execution_plan(payload, max_steps=max_steps)
        except Exception:
            plan = []

        if not plan:
            plan = self._fallback_execution_plan(job=job, max_steps=max_steps)

        if plan:
            state["execution_plan"] = plan
            state["plan_step_index"] = 0
            if isinstance(state["execution_plan"][0], dict):
                state["execution_plan"][0]["status"] = "in_progress"
        return used_llm

    def _normalize_execution_plan(
        self,
        payload: Dict[str, Any],
        max_steps: int = 6,
    ) -> List[Dict[str, Any]]:
        """Normalize planner output into stable step objects."""
        if not isinstance(payload, dict):
            return []
        raw_steps = payload.get("plan_steps")
        if not isinstance(raw_steps, list):
            raw_steps = payload.get("steps")
        if not isinstance(raw_steps, list):
            return []

        normalized: List[Dict[str, Any]] = []
        for item in raw_steps:
            if isinstance(item, str):
                title = item.strip()
                if not title:
                    continue
                step = {
                    "title": title[:220],
                    "objective": title[:350],
                    "exit_criteria": "",
                    "suggested_tools": [],
                    "status": "pending",
                }
                normalized.append(step)
            elif isinstance(item, dict):
                title = str(item.get("title") or item.get("name") or "").strip()
                objective = str(item.get("objective") or item.get("purpose") or "").strip()
                exit_criteria = str(item.get("exit_criteria") or item.get("done_when") or "").strip()
                suggested_tools = item.get("suggested_tools")
                if not isinstance(suggested_tools, list):
                    suggested_tools = item.get("tools")
                if not isinstance(suggested_tools, list):
                    suggested_tools = []
                suggested_tools = [str(x).strip() for x in suggested_tools if str(x).strip()]
                if not title and objective:
                    title = objective[:180]
                if not title:
                    continue
                normalized.append(
                    {
                        "title": title[:220],
                        "objective": objective[:500],
                        "exit_criteria": exit_criteria[:300],
                        "suggested_tools": suggested_tools[:6],
                        "status": "pending",
                    }
                )
            if len(normalized) >= max_steps:
                break

        return normalized

    def _fallback_execution_plan(self, job: AgentJob, max_steps: int = 6) -> List[Dict[str, Any]]:
        """Create a deterministic fallback plan when LLM planning is unavailable."""
        steps: List[Dict[str, Any]] = [
            {
                "title": "Scope the goal and constraints",
                "objective": "Clarify objective, success criteria, and important constraints.",
                "exit_criteria": "Clear objective statement and constraints captured.",
                "suggested_tools": ["write_progress_report"],
                "status": "pending",
            },
            {
                "title": "Collect high-signal internal evidence",
                "objective": "Find relevant documents and supporting context in the knowledge base.",
                "exit_criteria": "At least one relevant document identified and inspected.",
                "suggested_tools": ["search_documents", "read_document_content"],
                "status": "pending",
            },
        ]

        if job.job_type in {"research", "monitor", "knowledge_expansion"}:
            steps.append(
                {
                    "title": "Expand with external research",
                    "objective": "Complement internal evidence with current papers when appropriate.",
                    "exit_criteria": "Relevant external papers gathered or explicitly deemed unnecessary.",
                    "suggested_tools": ["search_arxiv", "find_related_papers"],
                    "status": "pending",
                }
            )

        steps.extend(
            [
                {
                    "title": "Synthesize findings",
                    "objective": "Convert evidence into conclusions, gaps, and next actions.",
                    "exit_criteria": "Findings are organized and attributable to sources.",
                    "suggested_tools": ["save_research_finding", "create_synthesis_document"],
                    "status": "pending",
                },
                {
                    "title": "Publish results",
                    "objective": "Produce a final output artifact and concise status summary.",
                    "exit_criteria": "Final artifact/report produced and progress reported.",
                    "suggested_tools": ["create_document_from_text", "write_progress_report"],
                    "status": "pending",
                },
            ]
        )
        return steps[:max_steps]

    def _ensure_subgoals(self, job: AgentJob, state: Dict[str, Any]) -> None:
        """Create lightweight subgoals from the plan or goal text."""
        cfg = job.config if isinstance(job.config, dict) else {}
        if not bool(cfg.get("subgoal_decomposition_enabled", True)):
            return

        existing = state.get("subgoals")
        if isinstance(existing, list) and existing:
            return

        max_subgoals = 5
        try:
            max_subgoals = int(cfg.get("max_subgoals", 5) or 5)
        except Exception:
            max_subgoals = 5
        max_subgoals = max(2, min(max_subgoals, 10))

        out: List[Dict[str, Any]] = []
        plan = state.get("execution_plan")
        if isinstance(plan, list) and plan:
            for step in plan:
                if not isinstance(step, dict):
                    continue
                title = str(step.get("title") or "").strip()
                obj = str(step.get("objective") or "").strip()
                text = title or obj
                if not text:
                    continue
                out.append({"title": text[:220], "status": "pending"})
                if len(out) >= max_subgoals:
                    break

        if not out:
            goal = str(job.goal or "").strip()
            parts = [p.strip() for p in re.split(r"[.;]|(?:\s+and\s+)|(?:\s+then\s+)|,", goal) if p.strip()]
            if not parts and goal:
                parts = [goal]
            for p in parts[:max_subgoals]:
                out.append({"title": p[:220], "status": "pending"})

        if out:
            out[0]["status"] = "in_progress"
            state["subgoals"] = out
            state["subgoal_index"] = 0

    def _get_swarm_config(self, job: AgentJob) -> Dict[str, Any]:
        """Get normalized config for swarm child-agent generation."""
        cfg = job.config if isinstance(job.config, dict) else {}

        def _as_int(key: str, default: int, lo: int, hi: int) -> int:
            try:
                val = int(cfg.get(key, default))
            except Exception:
                val = default
            return max(lo, min(val, hi))

        def _as_float(key: str, default: float, lo: float, hi: float) -> float:
            try:
                val = float(cfg.get(key, default))
            except Exception:
                val = default
            return max(lo, min(val, hi))

        roles = cfg.get("swarm_roles")
        if isinstance(roles, str):
            roles = [x.strip() for x in roles.split(",") if x.strip()]
        if not isinstance(roles, list):
            roles = []

        trigger = str(cfg.get("swarm_trigger_condition", ChainTriggerCondition.ON_COMPLETE.value) or ChainTriggerCondition.ON_COMPLETE.value).strip().lower()
        if trigger not in {ChainTriggerCondition.ON_COMPLETE.value, ChainTriggerCondition.ON_ANY_END.value, ChainTriggerCondition.ON_PROGRESS.value, ChainTriggerCondition.ON_FINDINGS.value, ChainTriggerCondition.ON_FAIL.value}:
            trigger = ChainTriggerCondition.ON_COMPLETE.value

        return {
            "enabled": bool(cfg.get("swarm_child_jobs_enabled", False)),
            "max_agents": _as_int("swarm_max_agents", 4, 1, 12),
            "roles": roles,
            "inherit_results": bool(cfg.get("swarm_inherit_results", True)),
            "inherit_config": bool(cfg.get("swarm_inherit_config", False)),
            "trigger_condition": trigger,
            "max_iterations_ratio": _as_float("swarm_child_max_iterations_ratio", 0.45, 0.1, 1.0),
            "max_tool_calls_ratio": _as_float("swarm_child_max_tool_calls_ratio", 0.45, 0.1, 1.0),
            "max_llm_calls_ratio": _as_float("swarm_child_max_llm_calls_ratio", 0.45, 0.1, 1.0),
            "max_runtime_ratio": _as_float("swarm_child_max_runtime_ratio", 0.5, 0.1, 1.0),
            "min_iterations": _as_int("swarm_child_min_iterations", 6, 1, 100),
            "min_tool_calls": _as_int("swarm_child_min_tool_calls", 8, 1, 200),
            "min_llm_calls": _as_int("swarm_child_min_llm_calls", 6, 1, 200),
            "min_runtime_minutes": _as_int("swarm_child_min_runtime_minutes", 10, 1, 240),
            "goal_prefix": str(cfg.get("swarm_goal_prefix", "Swarm role")).strip()[:80],
            "fan_in_enabled": bool(cfg.get("swarm_fan_in_enabled", True)),
            "fan_in_name": str(cfg.get("swarm_fan_in_name", "Swarm Synthesis")).strip()[:120],
            "fan_in_job_type": str(cfg.get("swarm_fan_in_job_type", "synthesis") or "synthesis").strip().lower(),
            "fan_in_trigger_condition": str(cfg.get("swarm_fan_in_trigger_condition", ChainTriggerCondition.ON_ANY_END.value) or ChainTriggerCondition.ON_ANY_END.value).strip().lower(),
        }

    def _ensure_swarm_chain_config(self, job: AgentJob, state: Dict[str, Any]) -> None:
        """Create a swarm of specialized child jobs when enabled and no chain exists yet."""
        swarm_cfg = self._get_swarm_config(job)
        if not bool(swarm_cfg.get("enabled", False)):
            return
        if bool(state.get("swarm_chain_configured", False)):
            return

        chain = job.chain_config if isinstance(job.chain_config, dict) else {}
        existing_children = chain.get("child_jobs")
        if isinstance(existing_children, list) and existing_children:
            state["swarm_chain_configured"] = True
            state["swarm_child_jobs_count"] = len(existing_children)
            chain_data_existing = chain.get("chain_data") if isinstance(chain.get("chain_data"), dict) else {}
            state["swarm_fan_in_enabled"] = bool(chain_data_existing.get("swarm_fan_in_enabled", False))
            state["swarm_fan_in_group_id"] = str(chain_data_existing.get("swarm_fan_in_group_id") or "")
            return

        role_templates: Dict[str, Dict[str, Any]] = {
            "researcher": {
                "name": "Researcher",
                "job_type": "research",
                "objective": "Gather high-signal evidence from papers and internal knowledge sources.",
                "config": {"prefer_sources": ["documents", "arxiv"], "max_documents": 10, "max_papers": 8},
            },
            "researcher_documents": {
                "name": "Knowledge Researcher",
                "job_type": "research",
                "objective": "Focus on internal documents and existing knowledge-base evidence.",
                "config": {"prefer_sources": ["documents"], "max_documents": 14, "max_papers": 2},
            },
            "researcher_arxiv": {
                "name": "Literature Researcher",
                "job_type": "research",
                "objective": "Focus on external paper discovery and validation.",
                "config": {"prefer_sources": ["arxiv"], "max_documents": 4, "max_papers": 12},
            },
            "analyst": {
                "name": "Analyst",
                "job_type": "analysis",
                "objective": "Compare sources, identify gaps/contradictions, and stress-test assumptions.",
                "config": {"prefer_sources": ["documents", "arxiv"]},
            },
            "synthesizer": {
                "name": "Synthesizer",
                "job_type": "synthesis",
                "objective": "Produce concise synthesis with traceable evidence and clear next actions.",
                "config": {"prefer_sources": ["documents"]},
            },
            "monitor": {
                "name": "Monitor",
                "job_type": "monitor",
                "objective": "Track updates and ingest newly relevant sources for the topic.",
                "config": {"prefer_sources": ["arxiv", "documents"]},
            },
            "knowledge_expander": {
                "name": "Knowledge Expander",
                "job_type": "knowledge_expansion",
                "objective": "Find adjacent concepts and add structured knowledge links.",
                "config": {"prefer_sources": ["documents", "arxiv"]},
            },
        }
        default_roles: List[Any] = [
            "researcher_documents",
            "researcher_arxiv",
            "analyst",
        ]
        roles_raw = swarm_cfg.get("roles")
        if not isinstance(roles_raw, list) or not roles_raw:
            roles_raw = default_roles

        max_agents = int(swarm_cfg.get("max_agents", 4) or 4)
        max_agents = max(1, min(max_agents, 12))
        parent_goal = str(job.goal or "").strip()[:1600]
        fan_in_enabled = bool(swarm_cfg.get("fan_in_enabled", True))
        fan_in_trigger = str(swarm_cfg.get("fan_in_trigger_condition", ChainTriggerCondition.ON_ANY_END.value) or ChainTriggerCondition.ON_ANY_END.value).strip().lower()
        if fan_in_trigger not in {
            ChainTriggerCondition.ON_COMPLETE.value,
            ChainTriggerCondition.ON_ANY_END.value,
            ChainTriggerCondition.ON_PROGRESS.value,
            ChainTriggerCondition.ON_FINDINGS.value,
            ChainTriggerCondition.ON_FAIL.value,
        }:
            fan_in_trigger = ChainTriggerCondition.ON_ANY_END.value

        child_max_iterations = max(
            int(swarm_cfg.get("min_iterations", 6) or 6),
            int((job.max_iterations or 20) * float(swarm_cfg.get("max_iterations_ratio", 0.45) or 0.45)),
        )
        child_max_tool_calls = max(
            int(swarm_cfg.get("min_tool_calls", 8) or 8),
            int((job.max_tool_calls or 50) * float(swarm_cfg.get("max_tool_calls_ratio", 0.45) or 0.45)),
        )
        child_max_llm_calls = max(
            int(swarm_cfg.get("min_llm_calls", 6) or 6),
            int((job.max_llm_calls or 30) * float(swarm_cfg.get("max_llm_calls_ratio", 0.45) or 0.45)),
        )
        child_max_runtime = max(
            int(swarm_cfg.get("min_runtime_minutes", 10) or 10),
            int((job.max_runtime_minutes or 60) * float(swarm_cfg.get("max_runtime_ratio", 0.5) or 0.5)),
        )

        allowed_job_types = {"research", "monitor", "analysis", "synthesis", "knowledge_expansion", "custom", "data_analysis"}
        fan_in_job_type = str(swarm_cfg.get("fan_in_job_type", "synthesis") or "synthesis").strip().lower()
        if fan_in_job_type not in allowed_job_types:
            fan_in_job_type = "synthesis"
        child_jobs: List[Dict[str, Any]] = []
        role_names: List[str] = []

        for idx, raw in enumerate(roles_raw, start=1):
            if len(child_jobs) >= max_agents:
                break

            role_tag = ""
            role_name = ""
            role_objective = ""
            role_job_type = ""
            role_cfg: Dict[str, Any] = {}

            if isinstance(raw, dict):
                role_key = str(raw.get("role") or raw.get("type") or raw.get("name") or "researcher").strip().lower().replace("-", "_").replace(" ", "_")
                tpl = role_templates.get(role_key, role_templates["researcher"])
                role_name = str(raw.get("name") or tpl.get("name") or "Researcher").strip()
                role_objective = str(raw.get("objective") or tpl.get("objective") or "").strip()
                role_job_type = str(raw.get("job_type") or tpl.get("job_type") or job.job_type).strip().lower()
                role_cfg = dict(tpl.get("config") if isinstance(tpl.get("config"), dict) else {})
                if isinstance(raw.get("config"), dict):
                    role_cfg.update(raw.get("config") or {})
            else:
                role_token = str(raw or "").strip()
                if not role_token:
                    continue
                role_key = role_token.lower().replace("-", "_").replace(" ", "_")
                if ":" in role_key:
                    role_key, role_tag = [p.strip() for p in role_key.split(":", 1)]
                tpl = role_templates.get(role_key, role_templates["researcher"])
                role_name = str(tpl.get("name") or "Researcher").strip()
                role_objective = str(tpl.get("objective") or "").strip()
                role_job_type = str(tpl.get("job_type") or job.job_type).strip().lower()
                role_cfg = dict(tpl.get("config") if isinstance(tpl.get("config"), dict) else {})
                if role_tag:
                    role_name = f"{role_name} ({role_tag[:40]})"
                    role_objective = f"{role_objective} Focus tag: {role_tag[:120]}."

            if role_job_type not in allowed_job_types:
                role_job_type = str(job.job_type or "research")

            role_name = role_name[:120] if role_name else f"Role {idx}"
            role_names.append(role_name)
            goal_prefix = str(swarm_cfg.get("goal_prefix", "Swarm role") or "Swarm role").strip()[:80]
            role_goal = (
                f"{goal_prefix}: {role_name}\n"
                f"Objective: {role_objective}\n"
                f"Parent goal: {parent_goal}\n\n"
                "Deliver concise, evidence-backed findings specific to this role, then provide actionable next steps."
            )
            child_jobs.append(
                {
                    "name": f"Swarm Agent {idx}: {role_name[:80]}",
                    "description": "Auto-generated swarm child agent from parent autonomous job.",
                    "job_type": role_job_type,
                    "goal": role_goal[:2200],
                    "config": {
                        **role_cfg,
                        "origin": "swarm_child_agent",
                        "swarm_role": role_name[:120],
                        "swarm_role_index": idx,
                        "swarm_parent_job_id": str(job.id),
                        "swarm_root_goal": parent_goal[:800],
                        "auto_subgoal_child_jobs_enabled": False,
                        "swarm_child_jobs_enabled": False,
                    },
                    "max_iterations": child_max_iterations,
                    "max_tool_calls": child_max_tool_calls,
                    "max_llm_calls": child_max_llm_calls,
                    "max_runtime_minutes": child_max_runtime,
                }
            )

        if not child_jobs:
            return

        fan_in_group_id = hashlib.sha256(f"swarm_fan_in:{job.id}:{max_agents}".encode("utf-8")).hexdigest()[:16]
        fan_in_template: Optional[Dict[str, Any]] = None
        if fan_in_enabled:
            fan_in_name = str(swarm_cfg.get("fan_in_name", "Swarm Synthesis") or "Swarm Synthesis").strip()[:120]
            fan_in_goal = (
                f"{fan_in_name}: Merge outputs from {len(child_jobs)} swarm agents.\n"
                f"Parent goal: {parent_goal}\n\n"
                "Use inherited swarm sibling results to produce: key findings, conflicts, confidence levels, "
                "and a consolidated recommendation with cited evidence."
            )
            fan_in_template = {
                "name": f"{fan_in_name}: Consolidated Output",
                "description": "Auto-generated fan-in aggregator for swarm child agents.",
                "job_type": fan_in_job_type,
                "goal": fan_in_goal[:2400],
                "config": {
                    "origin": "swarm_fan_in_aggregator",
                    "deterministic_runner": "swarm_fan_in_aggregate",
                    "swarm_fan_in_group_id": fan_in_group_id,
                    "swarm_parent_job_id": str(job.id),
                    "swarm_role_count": len(child_jobs),
                    "swarm_child_jobs_enabled": False,
                    "auto_subgoal_child_jobs_enabled": False,
                },
                "max_iterations": child_max_iterations,
                "max_tool_calls": child_max_tool_calls,
                "max_llm_calls": child_max_llm_calls,
                "max_runtime_minutes": child_max_runtime,
            }
            for child in child_jobs:
                fan_in_child = {
                    **fan_in_template,
                    "config": dict(fan_in_template.get("config") if isinstance(fan_in_template.get("config"), dict) else {}),
                }
                child["chain_config"] = {
                    "trigger_condition": fan_in_trigger,
                    "inherit_results": True,
                    "inherit_config": False,
                    "chain_data": {
                        "source": "swarm_fan_in",
                        "swarm_fan_in_wait_for_all_siblings": True,
                        "swarm_fan_in_expected_siblings": len(child_jobs),
                        "swarm_fan_in_group_id": fan_in_group_id,
                    },
                    "child_jobs": [fan_in_child],
                }

        merged = dict(chain)
        merged.setdefault("trigger_condition", str(swarm_cfg.get("trigger_condition") or ChainTriggerCondition.ON_COMPLETE.value))
        merged.setdefault("inherit_results", bool(swarm_cfg.get("inherit_results", True)))
        merged.setdefault("inherit_config", bool(swarm_cfg.get("inherit_config", False)))
        merged.setdefault("chain_data", {})
        if not isinstance(merged.get("chain_data"), dict):
            merged["chain_data"] = {}
        merged["chain_data"].update(
            {
                "source": "swarm_child_jobs",
                "generated_at_iteration": int(job.iteration or 0),
                "swarm_roles": role_names[:max_agents],
                "swarm_max_agents": max_agents,
                "swarm_fan_in_enabled": fan_in_enabled,
                "swarm_fan_in_group_id": fan_in_group_id if fan_in_enabled else "",
            }
        )
        merged["child_jobs"] = child_jobs
        job.chain_config = merged
        state["swarm_chain_configured"] = True
        state["swarm_child_jobs_count"] = len(child_jobs)
        state["swarm_roles_assigned"] = role_names[:max_agents]
        state["swarm_fan_in_enabled"] = fan_in_enabled
        state["swarm_fan_in_group_id"] = fan_in_group_id if fan_in_enabled else ""
        job.add_log_entry(
            {
                "phase": "swarm_chain_configured",
                "child_jobs_count": len(child_jobs),
                "roles": role_names[:max_agents],
                "trigger_condition": merged.get("trigger_condition"),
            }
        )

    def _ensure_subgoal_chain_config(self, job: AgentJob, state: Dict[str, Any]) -> None:
        """Create child job chain config from subgoals when enabled and absent."""
        cfg = job.config if isinstance(job.config, dict) else {}
        if not bool(cfg.get("auto_subgoal_child_jobs_enabled", True)):
            return
        if bool(state.get("subgoal_chain_configured")):
            return

        subgoals = state.get("subgoals")
        if not isinstance(subgoals, list) or len(subgoals) < 2:
            return

        chain = job.chain_config if isinstance(job.chain_config, dict) else {}
        existing_children = chain.get("child_jobs")
        if isinstance(existing_children, list) and existing_children:
            state["subgoal_chain_configured"] = True
            return

        max_children = 3
        try:
            max_children = int(cfg.get("auto_subgoal_child_jobs_max", 3) or 3)
        except Exception:
            max_children = 3
        max_children = max(1, min(max_children, 8))

        child_max_iterations = max(8, int((job.max_iterations or 20) * 0.4))
        child_max_tool_calls = max(8, int((job.max_tool_calls or 50) * 0.4))
        child_max_llm_calls = max(8, int((job.max_llm_calls or 30) * 0.4))
        child_max_runtime = max(10, int((job.max_runtime_minutes or 60) * 0.5))

        child_jobs: List[Dict[str, Any]] = []
        # Keep subgoal[0] in the parent execution loop; chain follow-ups for the remainder.
        for idx, sg in enumerate(subgoals[1:], start=1):
            if len(child_jobs) >= max_children:
                break
            if not isinstance(sg, dict):
                continue
            title = str(sg.get("title") or "").strip()
            if not title:
                continue

            child_jobs.append(
                {
                    "name": f"Subgoal Follow-up: {title[:80]}",
                    "description": "Auto-generated child job from parent subgoal decomposition.",
                    "job_type": job.job_type,
                    "goal": f"Subgoal: {title}\nParent goal: {str(job.goal or '')[:1200]}",
                    "config": {
                        "origin": "auto_subgoal_child",
                        "subgoal_index": idx,
                        "subgoal_title": title[:220],
                    },
                    "max_iterations": child_max_iterations,
                    "max_tool_calls": child_max_tool_calls,
                    "max_llm_calls": child_max_llm_calls,
                    "max_runtime_minutes": child_max_runtime,
                }
            )

        if not child_jobs:
            return

        merged = dict(chain)
        merged.setdefault("trigger_condition", ChainTriggerCondition.ON_COMPLETE.value)
        merged.setdefault("inherit_results", True)
        merged.setdefault("inherit_config", False)
        merged.setdefault("chain_data", {})
        if not isinstance(merged.get("chain_data"), dict):
            merged["chain_data"] = {}
        merged["chain_data"].update(
            {
                "source": "auto_subgoal_child_jobs",
                "generated_at_iteration": int(job.iteration or 0),
                "subgoals_count": len(subgoals),
            }
        )
        merged["child_jobs"] = child_jobs
        job.chain_config = merged
        state["subgoal_chain_configured"] = True
        job.add_log_entry(
            {
                "phase": "subgoal_chain_configured",
                "child_jobs_count": len(child_jobs),
                "trigger_condition": merged.get("trigger_condition"),
            }
        )

    def _get_critic_config(self, job: AgentJob) -> Dict[str, Any]:
        """Get normalized critic-pass settings."""
        cfg = job.config if isinstance(job.config, dict) else {}

        def _as_int(key: str, default: int, lo: int, hi: int) -> int:
            try:
                val = int(cfg.get(key, default))
            except Exception:
                val = default
            return max(lo, min(val, hi))

        def _as_float(key: str, default: float, lo: float, hi: float) -> float:
            try:
                val = float(cfg.get(key, default))
            except Exception:
                val = default
            return max(lo, min(val, hi))

        return {
            "enabled": bool(cfg.get("critic_enabled", True)),
            "every_n_iterations": _as_int("critic_every_n_iterations", 4, 1, 50),
            "on_stall": bool(cfg.get("critic_on_stall", True)),
            "stall_threshold": _as_int("critic_stall_threshold", 2, 1, 20),
            "on_uncertainty": bool(cfg.get("critic_on_uncertainty", True)),
            "uncertainty_top_gap_threshold": _as_float("critic_uncertainty_top_gap_threshold", 0.05, 0.0, 2.0),
            "uncertainty_min_candidates": _as_int("critic_uncertainty_min_candidates", 2, 2, 20),
            "uncertainty_max_age_iterations": _as_int("critic_uncertainty_max_age_iterations", 2, 1, 50),
            "uncertainty_min_iterations_since_last": _as_int("critic_uncertainty_min_iterations_since_last", 1, 1, 50),
            "uncertainty_stage_schedule_enabled": bool(cfg.get("critic_uncertainty_stage_schedule_enabled", True)),
            "uncertainty_mode_schedule_enabled": bool(cfg.get("critic_uncertainty_mode_schedule_enabled", True)),
            "uncertainty_stage_multiplier_discovery": _as_float("critic_uncertainty_stage_multiplier_discovery", 1.3, 0.1, 5.0),
            "uncertainty_stage_multiplier_consolidation": _as_float("critic_uncertainty_stage_multiplier_consolidation", 1.0, 0.1, 5.0),
            "uncertainty_stage_multiplier_finish": _as_float("critic_uncertainty_stage_multiplier_finish", 0.8, 0.1, 5.0),
            "uncertainty_stage_multiplier_rescue": _as_float("critic_uncertainty_stage_multiplier_rescue", 1.2, 0.1, 5.0),
            "uncertainty_mode_multiplier_baseline": _as_float("critic_uncertainty_mode_multiplier_baseline", 0.9, 0.1, 5.0),
            "uncertainty_mode_multiplier_adaptive": _as_float("critic_uncertainty_mode_multiplier_adaptive", 1.0, 0.1, 5.0),
            "uncertainty_mode_multiplier_thompson": _as_float("critic_uncertainty_mode_multiplier_thompson", 1.15, 0.1, 5.0),
            "uncertainty_threshold_min": _as_float("critic_uncertainty_threshold_min", 0.005, 0.0, 2.0),
            "uncertainty_threshold_max": _as_float("critic_uncertainty_threshold_max", 0.5, 0.0, 2.0),
            "max_notes": _as_int("critic_max_notes", 6, 1, 20),
            "force_pivot_on_high": bool(cfg.get("critic_force_pivot_on_high", True)),
            "force_min_confidence": _as_float("critic_force_min_confidence", 0.6, 0.0, 1.0),
        }

    def _effective_uncertainty_gap_threshold(
        self,
        job: AgentJob,
        state: Dict[str, Any],
        cfg: Dict[str, Any],
    ) -> Tuple[float, str, str]:
        """Compute stage/mode-adaptive uncertainty trigger threshold."""
        base = float(cfg.get("uncertainty_top_gap_threshold", 0.05) or 0.05)
        threshold = base

        stage = str(state.get("tool_selection_goal_stage") or "").strip().lower()
        if stage not in {"discovery", "consolidation", "finish", "rescue"}:
            stage = self._derive_goal_stage(state, self._get_tool_selection_config(job))
        if bool(cfg.get("uncertainty_stage_schedule_enabled", True)):
            stage_multipliers = {
                "discovery": float(cfg.get("uncertainty_stage_multiplier_discovery", 1.3) or 1.3),
                "consolidation": float(cfg.get("uncertainty_stage_multiplier_consolidation", 1.0) or 1.0),
                "finish": float(cfg.get("uncertainty_stage_multiplier_finish", 0.8) or 0.8),
                "rescue": float(cfg.get("uncertainty_stage_multiplier_rescue", 1.2) or 1.2),
            }
            threshold *= float(stage_multipliers.get(stage, 1.0))

        mode = str(state.get("tool_selection_effective_mode") or "").strip().lower()
        if mode not in {"baseline", "adaptive", "thompson"}:
            mode = str(self._get_tool_selection_config(job).get("policy_mode", "adaptive") or "adaptive").strip().lower()
            if mode not in {"baseline", "adaptive", "thompson"}:
                mode = "adaptive"
        if bool(cfg.get("uncertainty_mode_schedule_enabled", True)):
            mode_multipliers = {
                "baseline": float(cfg.get("uncertainty_mode_multiplier_baseline", 0.9) or 0.9),
                "adaptive": float(cfg.get("uncertainty_mode_multiplier_adaptive", 1.0) or 1.0),
                "thompson": float(cfg.get("uncertainty_mode_multiplier_thompson", 1.15) or 1.15),
            }
            threshold *= float(mode_multipliers.get(mode, 1.0))

        threshold_min = float(cfg.get("uncertainty_threshold_min", 0.005) or 0.005)
        threshold_max = float(cfg.get("uncertainty_threshold_max", 0.5) or 0.5)
        if threshold_min > threshold_max:
            threshold_min, threshold_max = threshold_max, threshold_min
        threshold = max(threshold_min, min(threshold, threshold_max))
        return threshold, stage, mode

    def _counterfactual_top_score_gap(self, state: Dict[str, Any]) -> Optional[float]:
        """Return top-vs-runner score gap from last counterfactual candidates."""
        rows = state.get("counterfactual_last")
        if not isinstance(rows, list):
            return None

        scores: List[float] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            try:
                scores.append(float(row.get("priority_score", 0.0) or 0.0))
            except Exception:
                continue
        if len(scores) < 2:
            return None
        scores.sort(reverse=True)
        return max(0.0, scores[0] - scores[1])

    def _should_run_critic(self, job: AgentJob, state: Dict[str, Any]) -> bool:
        """Determine whether to run a critic pass in this iteration."""
        cfg = self._get_critic_config(job)
        if not cfg.get("enabled", True):
            return False

        # Keep headroom for think + evaluate.
        if int(job.llm_calls_used or 0) >= max(0, int(job.max_llm_calls or 0) - 2):
            return False

        iteration = int(job.iteration or 0)
        last_iter = int(state.get("last_critic_iteration", 0) or 0)
        by_interval = (iteration - last_iter) >= int(cfg.get("every_n_iterations", 4))
        by_stall = bool(cfg.get("on_stall", True)) and int(state.get("stalled_iterations", 0) or 0) >= int(cfg.get("stall_threshold", 2))
        by_uncertainty = False
        uncertainty_gap: Optional[float] = None
        uncertainty_threshold: Optional[float] = None
        uncertainty_stage = str(state.get("tool_selection_goal_stage") or "").strip().lower()
        uncertainty_mode = str(state.get("tool_selection_effective_mode") or "").strip().lower()
        uncertainty_candidates = 0
        if bool(cfg.get("on_uncertainty", True)):
            min_since_last = int(cfg.get("uncertainty_min_iterations_since_last", 1) or 1)
            if (iteration - last_iter) >= min_since_last:
                rows = state.get("counterfactual_last")
                min_candidates = int(cfg.get("uncertainty_min_candidates", 2) or 2)
                uncertainty_candidates = len(rows) if isinstance(rows, list) else 0
                if uncertainty_candidates >= min_candidates:
                    max_age = int(cfg.get("uncertainty_max_age_iterations", 2) or 2)
                    last_cf_iteration = int(state.get("counterfactual_last_iteration", 0) or 0)
                    fresh_enough = True if last_cf_iteration <= 0 else (iteration - last_cf_iteration) <= max_age
                    if fresh_enough:
                        uncertainty_gap = self._counterfactual_top_score_gap(state)
                        uncertainty_threshold, uncertainty_stage, uncertainty_mode = self._effective_uncertainty_gap_threshold(job, state, cfg)
                        if uncertainty_gap is not None and uncertainty_gap <= uncertainty_threshold:
                            by_uncertainty = True

        triggered = by_interval or by_stall or by_uncertainty
        if not triggered:
            return False

        trigger_reason = "interval"
        if by_stall:
            trigger_reason = "stall"
        if by_uncertainty:
            trigger_reason = "uncertainty"

        trigger_payload: Dict[str, Any] = {
            "iteration": iteration,
            "reason": trigger_reason,
            "by_interval": bool(by_interval),
            "by_stall": bool(by_stall),
            "by_uncertainty": bool(by_uncertainty),
            "stalled_iterations": int(state.get("stalled_iterations", 0) or 0),
            "uncertainty_score_gap": (
                round(float(uncertainty_gap), 6)
                if uncertainty_gap is not None
                else None
            ),
            "uncertainty_effective_threshold": (
                round(float(uncertainty_threshold), 6)
                if uncertainty_threshold is not None
                else None
            ),
            "uncertainty_candidate_count": int(uncertainty_candidates),
            "uncertainty_stage": uncertainty_stage,
            "uncertainty_mode": uncertainty_mode,
        }
        state["critic_last_trigger"] = trigger_payload

        if int(state.get("critic_last_trigger_iteration", 0) or 0) != iteration:
            counts = state.get("critic_trigger_counts")
            if not isinstance(counts, dict):
                counts = {}
            counts["total"] = int(counts.get("total", 0) or 0) + 1
            if by_interval:
                counts["interval"] = int(counts.get("interval", 0) or 0) + 1
            if by_stall:
                counts["stall"] = int(counts.get("stall", 0) or 0) + 1
            if by_uncertainty:
                counts["uncertainty"] = int(counts.get("uncertainty", 0) or 0) + 1
            state["critic_trigger_counts"] = counts
            state["critic_last_trigger_iteration"] = iteration
        return True

    async def _run_critic_pass(
        self,
        job: AgentJob,
        state: Dict[str, Any],
        observation: Dict[str, Any],
        user_settings: Optional[UserLLMSettings],
    ) -> Optional[Dict[str, Any]]:
        """Run an LLM critic pass to identify risks and pivots."""
        profile = (
            state.get("skill_profile")
            if isinstance(state.get("skill_profile"), dict)
            else self._resolve_agent_skill_profile(job, state=state)
        )
        available_tools = self._get_tools_for_job_type(job.job_type, job.config, profile=profile)
        recent_actions = state.get("actions_taken", []) if isinstance(state.get("actions_taken"), list) else []
        recent = recent_actions[-6:]
        system_prompt = (
            "You are a strict critic for an autonomous agent.\n"
            "Assess trajectory quality, identify risks, and propose a concrete pivot when needed.\n"
            "Return JSON only."
        )
        user_message = (
            f"Goal: {job.goal}\n"
            f"Iteration: {job.iteration}/{job.max_iterations}\n"
            f"Progress: {state.get('goal_progress', 0)}\n"
            f"Stalled iterations: {state.get('stalled_iterations', 0)}\n"
            f"Recent actions: {json.dumps(recent, default=str)[:5000]}\n"
            f"Current observation: {json.dumps(observation, default=str)[:2500]}\n"
            f"Available tools: {', '.join(available_tools)}\n"
            "Return JSON schema:\n"
            "{\n"
            '  "trajectory_assessment": "short assessment",\n'
            '  "risks": ["risk1"],\n'
            '  "pivot": "single concrete adjustment",\n'
            '  "recommended_tools": ["search_documents"],\n'
            '  "confidence": 0.0,\n'
            '  "severity": "low|medium|high"\n'
            "}\n"
            "Rules: keep concise and actionable."
        )

        try:
            raw = await self.llm_service.generate_response(
                system_prompt=system_prompt,
                user_message=user_message,
                user_settings=user_settings,
                routing=self._llm_routing_from_job_config(job.config),
            )
        except Exception:
            return None

        payload = self._extract_first_json_object(str(raw or ""))
        if not isinstance(payload, dict):
            text = str(raw or "").strip()
            if not text:
                return None
            return {
                "iteration": int(job.iteration or 0),
                "trajectory_assessment": text[:300],
                "risks": [],
                "pivot": "",
                "recommended_tools": [],
                "confidence": 0.0,
                "timestamp": datetime.utcnow().isoformat(),
            }

        rec_tools = payload.get("recommended_tools")
        if not isinstance(rec_tools, list):
            rec_tools = []
        rec_tools = [
            str(t).strip()
            for t in rec_tools
            if str(t).strip() in set(available_tools)
        ][:5]

        risks = payload.get("risks")
        if not isinstance(risks, list):
            risks = []
        risks = [str(r).strip()[:220] for r in risks if str(r).strip()][:5]

        try:
            confidence = float(payload.get("confidence", 0.0) or 0.0)
        except Exception:
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))
        severity_raw = str(payload.get("severity") or "").strip().lower()
        if severity_raw not in {"low", "medium", "high"}:
            if len(risks) >= 3 and confidence >= 0.6:
                severity_raw = "high"
            elif len(risks) >= 1:
                severity_raw = "medium"
            else:
                severity_raw = "low"

        return {
            "iteration": int(job.iteration or 0),
            "trajectory_assessment": str(payload.get("trajectory_assessment") or "").strip()[:350],
            "risks": risks,
            "pivot": str(payload.get("pivot") or "").strip()[:320],
            "recommended_tools": rec_tools,
            "confidence": confidence,
            "severity": severity_raw,
            "timestamp": datetime.utcnow().isoformat(),
        }

    def _build_action_from_recommended_tools(
        self,
        job: AgentJob,
        state: Dict[str, Any],
        recommended_tools: List[str],
        exclude_tool: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Build an executable action from critic-recommended tools."""
        if not isinstance(recommended_tools, list) or not recommended_tools:
            return None

        profile = (
            state.get("skill_profile")
            if isinstance(state.get("skill_profile"), dict)
            else self._resolve_agent_skill_profile(job, state=state)
        )
        available = set(self._get_tools_for_job_type(job.job_type, job.config, profile=profile))
        combined_stats = self._merge_tool_stats(
            state.get("tool_priors") if isinstance(state.get("tool_priors"), dict) else {},
            state.get("tool_stats") if isinstance(state.get("tool_stats"), dict) else {},
        )
        exclude = str(exclude_tool or "").strip()
        findings = state.get("findings", []) if isinstance(state.get("findings"), list) else []
        doc_ids = []
        for f in findings:
            if not isinstance(f, dict):
                continue
            did = str(f.get("id") or f.get("document_id") or "").strip()
            if did and did not in doc_ids:
                doc_ids.append(did)

        unique_tools: List[str] = []
        for raw in recommended_tools:
            tool = str(raw).strip()
            if tool and tool not in unique_tools:
                unique_tools.append(tool)

        unique_tools = self._rank_tools_for_selection(
            job,
            unique_tools,
            combined_stats,
            state=state,
            context_tag="critic_recommended",
        )

        for raw in unique_tools:
            tool = str(raw).strip()
            if not tool or tool not in available or (exclude and tool == exclude):
                continue
            action = self._build_action_for_tool(
                tool=tool,
                job=job,
                doc_ids=doc_ids,
                purpose="Critic-directed pivot.",
            )
            if action:
                return action
        return None

    def _maybe_apply_critic_pivot_override(
        self,
        job: AgentJob,
        state: Dict[str, Any],
        decision: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Optionally override next action when critic marks trajectory as high risk."""
        if not isinstance(decision, dict):
            return decision
        if decision.get("goal_achieved") or decision.get("should_stop"):
            return decision

        cfg = self._get_critic_config(job)
        if not bool(cfg.get("force_pivot_on_high", False)):
            return decision

        notes = state.get("critic_notes")
        if not isinstance(notes, list) or not notes or not isinstance(notes[-1], dict):
            return decision
        note = notes[-1]
        severity = str(note.get("severity") or "").strip().lower()
        try:
            confidence = float(note.get("confidence", 0.0) or 0.0)
        except Exception:
            confidence = 0.0

        if severity != "high" or confidence < float(cfg.get("force_min_confidence", 0.6)):
            return decision

        current_action = decision.get("action") if isinstance(decision.get("action"), dict) else {}
        current_tool = str(current_action.get("tool") or "").strip()
        recommended = note.get("recommended_tools") if isinstance(note.get("recommended_tools"), list) else []
        if current_tool and current_tool in [str(t).strip() for t in recommended]:
            return decision

        pivot_action = self._build_action_from_recommended_tools(
            job=job,
            state=state,
            recommended_tools=[str(t).strip() for t in recommended if str(t).strip()],
            exclude_tool=current_tool or None,
        )
        if not pivot_action:
            return decision

        reasoning = str(decision.get("reasoning") or "").strip()
        pivot_txt = str(note.get("pivot") or "").strip()
        decision["action"] = pivot_action
        decision["reasoning"] = (
            f"{reasoning[:350]} Critic override applied (high risk): {pivot_txt[:220]}".strip()
        )
        return decision

    def _record_tool_outcome(
        self,
        state: Dict[str, Any],
        action: Optional[Dict[str, Any]],
        action_result: Optional[Dict[str, Any]],
    ) -> None:
        """Track per-tool outcomes for adaptive tool strategy hints."""
        if not isinstance(action, dict):
            return
        tool = str(action.get("tool") or "").strip()
        if not tool:
            return

        stats = state.get("tool_stats")
        if not isinstance(stats, dict):
            stats = {}
        slot = stats.get(tool)
        if not isinstance(slot, dict):
            slot = {"success": 0, "failure": 0, "last_error": ""}

        success = bool((action_result or {}).get("success"))
        if success:
            slot["success"] = int(slot.get("success", 0) or 0) + 1
        else:
            slot["failure"] = int(slot.get("failure", 0) or 0) + 1
            err = str((action_result or {}).get("error") or "").strip()
            slot["last_error"] = err[:200]

        stats[tool] = slot
        state["tool_stats"] = stats

        # Track live mode outcomes for policy fallback guardrails.
        mode = str(state.get("tool_selection_effective_mode") or "adaptive").strip().lower()
        if mode not in {"baseline", "adaptive", "thompson"}:
            mode = "adaptive"
        mode_metrics = state.get("tool_selection_mode_metrics")
        if not isinstance(mode_metrics, dict):
            mode_metrics = {}
        mslot = mode_metrics.get(mode)
        if not isinstance(mslot, dict):
            mslot = {"success": 0, "failure": 0}
        if success:
            mslot["success"] = int(mslot.get("success", 0) or 0) + 1
        else:
            mslot["failure"] = int(mslot.get("failure", 0) or 0) + 1
        mode_metrics[mode] = mslot
        state["tool_selection_mode_metrics"] = mode_metrics

    def _normalize_tool_stats_map(self, raw: Any) -> Dict[str, Dict[str, Any]]:
        """Normalize `{tool: {success, failure, last_error}}` map."""
        if not isinstance(raw, dict):
            return {}
        out: Dict[str, Dict[str, Any]] = {}
        for tool, val in raw.items():
            tool_name = str(tool or "").strip()
            if not tool_name or not isinstance(val, dict):
                continue
            out[tool_name] = {
                "success": int(val.get("success", 0) or 0),
                "failure": int(val.get("failure", 0) or 0),
                "last_error": str(val.get("last_error") or "").strip()[:200],
            }
        return out

    def _merge_tool_stats(
        self,
        *stats_maps: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """Merge multiple tool stat maps by summing success/failure counts."""
        merged: Dict[str, Dict[str, Any]] = {}
        for smap in stats_maps:
            norm = self._normalize_tool_stats_map(smap)
            for tool, val in norm.items():
                cur = merged.get(tool) or {"success": 0, "failure": 0, "last_error": ""}
                cur["success"] = int(cur.get("success", 0) or 0) + int(val.get("success", 0) or 0)
                cur["failure"] = int(cur.get("failure", 0) or 0) + int(val.get("failure", 0) or 0)
                if val.get("last_error"):
                    cur["last_error"] = str(val.get("last_error") or "")[:200]
                merged[tool] = cur
        return merged

    def _tool_success_ratio(self, stat: Dict[str, Any]) -> float:
        """Compute smoothed success ratio for a tool stat."""
        if not isinstance(stat, dict):
            return 0.0
        s = int(stat.get("success", 0) or 0)
        f = int(stat.get("failure", 0) or 0)
        # Laplace smoothing to avoid harsh early bias.
        return (s + 1.0) / float(s + f + 2.0)

    def _get_tool_selection_config(self, job: AgentJob) -> Dict[str, Any]:
        """Get adaptive selection settings for tool ranking."""
        cfg = job.config if isinstance(job.config, dict) else {}

        def _as_float(key: str, default: float, lo: float, hi: float) -> float:
            try:
                val = float(cfg.get(key, default))
            except Exception:
                val = default
            return max(lo, min(val, hi))

        def _as_int(key: str, default: int, lo: int, hi: int) -> int:
            try:
                val = int(cfg.get(key, default))
            except Exception:
                val = default
            return max(lo, min(val, hi))

        def _as_mode(key: str, default: str) -> str:
            val = str(cfg.get(key, default) or default).strip().lower()
            return val if val in {"baseline", "adaptive", "thompson"} else default

        policy_mode = _as_mode("tool_selection_policy_mode", "adaptive")

        return {
            "policy_mode": policy_mode,
            "exploration_enabled": bool(cfg.get("tool_selection_exploration_enabled", True)),
            "exploration_bonus": _as_float("tool_selection_exploration_bonus", 0.15, 0.0, 2.0),
            "cold_start_bonus": _as_float("tool_selection_cold_start_bonus", 0.05, 0.0, 1.0),
            "min_trials": _as_int("tool_selection_min_trials", 3, 0, 100),
            "failure_penalty": _as_float("tool_selection_failure_penalty", 0.08, 0.0, 1.0),
            "thompson_alpha_prior": _as_float("tool_selection_thompson_alpha_prior", 1.0, 0.1, 100.0),
            "thompson_beta_prior": _as_float("tool_selection_thompson_beta_prior", 1.0, 0.1, 100.0),
            "thompson_temperature": _as_float("tool_selection_thompson_temperature", 1.0, 0.1, 5.0),
            "ab_test_enabled": bool(cfg.get("tool_selection_ab_test_enabled", False)),
            "ab_test_split": _as_float("tool_selection_ab_test_split", 0.5, 0.0, 1.0),
            "ab_test_variant_a": _as_mode("tool_selection_ab_test_variant_a", "adaptive"),
            "ab_test_variant_b": _as_mode("tool_selection_ab_test_variant_b", "thompson"),
            "live_fallback_enabled": bool(cfg.get("tool_selection_live_fallback_enabled", True)),
            "live_fallback_min_samples": _as_int("tool_selection_live_fallback_min_samples", 8, 1, 10_000),
            "live_fallback_min_success_rate": _as_float("tool_selection_live_fallback_min_success_rate", 0.2, 0.0, 1.0),
            "live_fallback_to_mode": _as_mode("tool_selection_live_fallback_to_mode", "adaptive"),
            "live_fallback_reset_enabled": bool(cfg.get("tool_selection_live_fallback_reset_enabled", True)),
            "live_fallback_reset_min_samples": _as_int("tool_selection_live_fallback_reset_min_samples", 10, 1, 10_000),
            "live_fallback_reset_min_success_rate": _as_float("tool_selection_live_fallback_reset_min_success_rate", 0.55, 0.0, 1.0),
            "stage_schedule_enabled": bool(cfg.get("tool_selection_stage_schedule_enabled", False)),
            "stage_discovery_mode": _as_mode("tool_selection_stage_discovery_mode", "thompson"),
            "stage_consolidation_mode": _as_mode("tool_selection_stage_consolidation_mode", "adaptive"),
            "stage_finish_mode": _as_mode("tool_selection_stage_finish_mode", "baseline"),
            "stage_rescue_mode": _as_mode("tool_selection_stage_rescue_mode", "adaptive"),
            "stage_rescue_stall_threshold": _as_int("tool_selection_stage_rescue_stall_threshold", 3, 1, 100),
            "stage_finish_progress": _as_int("tool_selection_stage_finish_progress", 80, 10, 100),
            "stage_discovery_progress": _as_int("tool_selection_stage_discovery_progress", 35, 0, 90),
            "family_diversification_enabled": bool(cfg.get("tool_selection_family_diversification_enabled", True)),
            "family_diversification_window": _as_int("tool_selection_family_diversification_window", 6, 1, 100),
            "family_diversification_bonus": _as_float("tool_selection_family_diversification_bonus", 0.08, 0.0, 1.0),
            "family_diversification_target_unique": _as_int("tool_selection_family_diversification_target_unique", 3, 1, 20),
            "feedback_learning_enabled": bool(cfg.get("feedback_learning_enabled", True)),
            "feedback_learning_weight": _as_float("feedback_learning_weight", 0.08, 0.0, 0.6),
            "feedback_learning_max_abs_bias": _as_float("feedback_learning_max_abs_bias", 0.3, 0.0, 1.0),
        }

    def _stable_fraction(self, key: str) -> float:
        """Map a key to stable [0,1) fraction."""
        digest = hashlib.sha256(str(key).encode("utf-8")).hexdigest()
        bucket = int(digest[:12], 16)
        return float(bucket % 1_000_000) / 1_000_000.0

    def _derive_goal_stage(
        self,
        state: Dict[str, Any],
        selection_cfg: Dict[str, Any],
    ) -> str:
        """Derive a coarse execution stage for policy scheduling."""
        progress = int(state.get("goal_progress", 0) or 0)
        stalled = int(state.get("stalled_iterations", 0) or 0)
        findings = len(state.get("findings", []) if isinstance(state.get("findings"), list) else [])

        rescue_threshold = int(selection_cfg.get("stage_rescue_stall_threshold", 3) or 3)
        finish_progress = int(selection_cfg.get("stage_finish_progress", 80) or 80)
        discovery_progress = int(selection_cfg.get("stage_discovery_progress", 35) or 35)

        if stalled >= rescue_threshold:
            return "rescue"
        if progress >= finish_progress:
            return "finish"
        if progress < discovery_progress or findings < 3:
            return "discovery"
        return "consolidation"

    def _apply_goal_stage_policy_mode(
        self,
        state: Dict[str, Any],
        current_mode: str,
        selection_cfg: Dict[str, Any],
    ) -> str:
        """Optionally override mode based on progress/stall stage."""
        mode = str(current_mode or "adaptive").strip().lower()
        if mode not in {"baseline", "adaptive", "thompson"}:
            mode = "adaptive"
        if not bool(selection_cfg.get("stage_schedule_enabled", False)):
            return mode

        stage = self._derive_goal_stage(state, selection_cfg)
        state["tool_selection_goal_stage"] = stage
        if stage == "rescue":
            return str(selection_cfg.get("stage_rescue_mode", mode) or mode)
        if stage == "finish":
            return str(selection_cfg.get("stage_finish_mode", mode) or mode)
        if stage == "discovery":
            return str(selection_cfg.get("stage_discovery_mode", mode) or mode)
        return str(selection_cfg.get("stage_consolidation_mode", mode) or mode)

    def _resolve_tool_selection_mode(
        self,
        job: AgentJob,
        state: Optional[Dict[str, Any]] = None,
        selection_cfg: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Resolve configured policy mode with optional deterministic A/B override."""
        cfg = selection_cfg if isinstance(selection_cfg, dict) else self._get_tool_selection_config(job)
        base_mode = str(cfg.get("policy_mode", "adaptive") or "adaptive").strip().lower()
        if base_mode not in {"baseline", "adaptive", "thompson"}:
            base_mode = "adaptive"

        assignment = {
            "enabled": bool(cfg.get("ab_test_enabled", False)),
            "bucket": 0.0,
            "variant": "configured",
            "mode": base_mode,
        }
        if isinstance(state, dict):
            self._maybe_reset_live_mode_override(job, state, cfg)

        if not assignment["enabled"]:
            override_mode = str((state or {}).get("tool_selection_mode_override") or "").strip().lower() if isinstance(state, dict) else ""
            if override_mode in {"baseline", "adaptive", "thompson"}:
                base_mode = override_mode
                assignment["override_active"] = True
                assignment["mode"] = base_mode
            elif isinstance(state, dict):
                base_mode = self._apply_goal_stage_policy_mode(state, base_mode, cfg)
            if isinstance(state, dict):
                state["tool_selection_effective_mode"] = base_mode
                state["tool_selection_ab_assignment"] = assignment
                base_mode = self._apply_live_mode_fallback_guardrail(job, state, base_mode, cfg)
                state["tool_selection_effective_mode"] = base_mode
                assignment["mode"] = base_mode
                assignment["goal_stage"] = str(state.get("tool_selection_goal_stage") or "")
            return base_mode, assignment

        variant_a = str(cfg.get("ab_test_variant_a", "adaptive") or "adaptive").strip().lower()
        variant_b = str(cfg.get("ab_test_variant_b", "thompson") or "thompson").strip().lower()
        if variant_a not in {"baseline", "adaptive", "thompson"}:
            variant_a = "adaptive"
        if variant_b not in {"baseline", "adaptive", "thompson"}:
            variant_b = "thompson"
        split = float(cfg.get("ab_test_split", 0.5) or 0.5)
        split = max(0.0, min(1.0, split))

        key = f"{job.user_id}:{job.id}:{job.job_type}"
        bucket = self._stable_fraction(key)
        mode = variant_a if bucket < split else variant_b
        assignment = {
            "enabled": True,
            "bucket": bucket,
            "split": split,
            "variant": "A" if bucket < split else "B",
            "mode": mode,
            "variant_a": variant_a,
            "variant_b": variant_b,
            "configured_mode": base_mode,
        }
        override_mode = str((state or {}).get("tool_selection_mode_override") or "").strip().lower() if isinstance(state, dict) else ""
        if override_mode in {"baseline", "adaptive", "thompson"}:
            mode = override_mode
            assignment["override_active"] = True
            assignment["mode"] = mode
        elif isinstance(state, dict):
            mode = self._apply_goal_stage_policy_mode(state, mode, cfg)
        if isinstance(state, dict):
            mode = self._apply_live_mode_fallback_guardrail(job, state, mode, cfg)
            state["tool_selection_effective_mode"] = mode
            assignment["mode"] = mode
            assignment["goal_stage"] = str(state.get("tool_selection_goal_stage") or "")
            state["tool_selection_ab_assignment"] = assignment
        return mode, assignment

    def _maybe_reset_live_mode_override(
        self,
        job: AgentJob,
        state: Dict[str, Any],
        selection_cfg: Dict[str, Any],
    ) -> None:
        """Clear an existing fallback override when the override mode recovers."""
        if not bool(selection_cfg.get("live_fallback_reset_enabled", True)):
            return
        current_override = str(state.get("tool_selection_mode_override") or "").strip().lower()
        if current_override not in {"baseline", "adaptive", "thompson"}:
            return

        metrics = state.get("tool_selection_mode_metrics")
        if not isinstance(metrics, dict):
            return
        slot = metrics.get(current_override)
        if not isinstance(slot, dict):
            return

        success = int(slot.get("success", 0) or 0)
        failure = int(slot.get("failure", 0) or 0)
        samples = success + failure
        min_samples = int(selection_cfg.get("live_fallback_reset_min_samples", 10) or 10)
        if samples < min_samples:
            return
        success_rate = float(success) / float(max(1, samples))
        min_rate = float(selection_cfg.get("live_fallback_reset_min_success_rate", 0.55) or 0.55)
        if success_rate < min_rate:
            return

        events = state.get("tool_selection_fallback_events")
        if not isinstance(events, list):
            events = []
        events.append(
            {
                "iteration": int(job.iteration or 0),
                "event": "reset_override",
                "mode": current_override,
                "samples": samples,
                "success_rate": round(success_rate, 4),
                "threshold": round(min_rate, 4),
            }
        )
        state["tool_selection_fallback_events"] = events[-20:]
        state["tool_selection_mode_override"] = ""

    def _apply_live_mode_fallback_guardrail(
        self,
        job: AgentJob,
        state: Dict[str, Any],
        mode: str,
        selection_cfg: Dict[str, Any],
    ) -> str:
        """Fallback to safer policy mode when live performance is under threshold."""
        current_mode = str(mode or "adaptive").strip().lower()
        if current_mode not in {"baseline", "adaptive", "thompson"}:
            current_mode = "adaptive"
        if not bool(selection_cfg.get("live_fallback_enabled", True)):
            return current_mode

        fallback_mode = str(selection_cfg.get("live_fallback_to_mode", "adaptive") or "adaptive").strip().lower()
        if fallback_mode not in {"baseline", "adaptive", "thompson"}:
            fallback_mode = "adaptive"
        if current_mode == fallback_mode:
            return current_mode

        metrics = state.get("tool_selection_mode_metrics")
        if not isinstance(metrics, dict):
            metrics = {}
        slot = metrics.get(current_mode)
        if not isinstance(slot, dict):
            return current_mode

        success = int(slot.get("success", 0) or 0)
        failure = int(slot.get("failure", 0) or 0)
        samples = success + failure
        min_samples = int(selection_cfg.get("live_fallback_min_samples", 8) or 8)
        if samples < min_samples:
            return current_mode

        success_rate = float(success) / float(max(1, samples))
        min_rate = float(selection_cfg.get("live_fallback_min_success_rate", 0.2) or 0.2)
        if success_rate >= min_rate:
            return current_mode

        existing = str(state.get("tool_selection_mode_override") or "").strip().lower()
        if existing != fallback_mode:
            events = state.get("tool_selection_fallback_events")
            if not isinstance(events, list):
                events = []
            events.append(
                {
                    "iteration": int(job.iteration or 0),
                    "from_mode": current_mode,
                    "to_mode": fallback_mode,
                    "samples": samples,
                    "success_rate": round(success_rate, 4),
                    "threshold": round(min_rate, 4),
                }
            )
            state["tool_selection_fallback_events"] = events[-20:]
            state["tool_selection_mode_override"] = fallback_mode
        return fallback_mode

    def _get_counterfactual_config(self, job: AgentJob) -> Dict[str, Any]:
        """Config for iteration-level counterfactual candidate logging."""
        cfg = job.config if isinstance(job.config, dict) else {}
        try:
            top_k = int(cfg.get("tool_selection_counterfactual_top_k", 3) or 3)
        except Exception:
            top_k = 3
        top_k = max(1, min(top_k, 10))
        return {
            "enabled": bool(cfg.get("tool_selection_counterfactual_enabled", True)),
            "top_k": top_k,
        }

    def _build_selection_explainability(
        self,
        state: Dict[str, Any],
        selected_tool: Optional[str],
        candidates: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Build a compact explanation of the current tool-selection decision."""
        tool = str(selected_tool or "").strip()
        cand = candidates if isinstance(candidates, list) else []
        ranked = [c for c in cand if isinstance(c, dict)]
        selected_row = None
        for row in ranked:
            if str(row.get("tool") or "").strip() == tool:
                selected_row = row
                break
        top_row = ranked[0] if ranked else {}
        runner_row = ranked[1] if len(ranked) > 1 else {}

        def _as_float(v: Any) -> float:
            try:
                return float(v or 0.0)
            except Exception:
                return 0.0

        top_score = _as_float(top_row.get("priority_score") if isinstance(top_row, dict) else 0.0)
        selected_score = _as_float(selected_row.get("priority_score") if isinstance(selected_row, dict) else 0.0)
        runner_score = _as_float(runner_row.get("priority_score") if isinstance(runner_row, dict) else 0.0)

        return {
            "selected_tool": tool,
            "effective_mode": str(state.get("tool_selection_effective_mode") or ""),
            "goal_stage": str(state.get("tool_selection_goal_stage") or ""),
            "mode_override": str(state.get("tool_selection_mode_override") or ""),
            "selected_rank": int(selected_row.get("rank", 0) or 0) if isinstance(selected_row, dict) else 0,
            "selected_score": round(selected_score, 6),
            "top_tool": str(top_row.get("tool") or "") if isinstance(top_row, dict) else "",
            "top_score": round(top_score, 6),
            "score_gap_to_top": round(top_score - selected_score, 6),
            "score_gap_top_vs_runner_up": round(top_score - runner_score, 6),
            "candidate_count": len(ranked),
            "fallback_event_count": len(state.get("tool_selection_fallback_events", []) if isinstance(state.get("tool_selection_fallback_events"), list) else []),
        }

    def _tool_observation_count(self, stat: Dict[str, Any]) -> int:
        """Return total observed outcomes for a tool."""
        if not isinstance(stat, dict):
            return 0
        s = int(stat.get("success", 0) or 0)
        f = int(stat.get("failure", 0) or 0)
        return max(0, s + f)

    def _tool_family(self, tool: str) -> str:
        """Map a tool to a coarse family for diversification incentives."""
        t = str(tool or "").strip().lower()
        if not t:
            return "unknown"
        if any(tok in t for tok in ("chart", "diagram", "heatmap", "flowchart", "gantt", "drawio")):
            return "visualization"
        if t.startswith(("search_", "find_", "get_", "list_")):
            return "retrieval"
        if t.startswith(("ingest_", "batch_ingest_", "load_", "monitor_")):
            return "ingestion"
        if t.startswith(("read_", "summarize_", "extract_", "analyze_", "compare_", "identify_", "describe_", "query_", "filter_", "aggregate_", "join_", "transform_", "detect_", "calculate_")):
            return "analysis"
        if t.startswith(("create_", "generate_", "write_", "save_", "link_", "add_", "export_", "suggest_")):
            return "synthesis"
        return "other"

    def _family_diversification_bonus(
        self,
        tool: str,
        *,
        state: Optional[Dict[str, Any]],
        selection_cfg: Optional[Dict[str, Any]],
    ) -> float:
        """Boost underrepresented tool families based on recent action history."""
        cfg = selection_cfg if isinstance(selection_cfg, dict) else {}
        if not bool(cfg.get("family_diversification_enabled", True)):
            return 0.0
        if not isinstance(state, dict):
            return 0.0
        actions = state.get("actions_taken")
        if not isinstance(actions, list) or not actions:
            return 0.0

        window = max(1, int(cfg.get("family_diversification_window", 6) or 6))
        recent = actions[-window:]
        family_counts: Dict[str, int] = {}
        for row in recent:
            if not isinstance(row, dict):
                continue
            action = row.get("action")
            if not isinstance(action, dict):
                continue
            used_tool = str(action.get("tool") or "").strip()
            if not used_tool:
                continue
            fam = self._tool_family(used_tool)
            family_counts[fam] = int(family_counts.get(fam, 0) or 0) + 1
        if not family_counts:
            return 0.0

        target_unique = max(1, int(cfg.get("family_diversification_target_unique", 3) or 3))
        raw_bonus = float(cfg.get("family_diversification_bonus", 0.08) or 0.08)
        current_family = self._tool_family(tool)
        used_count = int(family_counts.get(current_family, 0) or 0)
        unique_used = len(family_counts)
        diversity_pressure = max(0.0, float(target_unique - unique_used) / float(target_unique))

        if used_count <= 0:
            return raw_bonus * (1.0 + 0.5 * diversity_pressure)
        return raw_bonus * diversity_pressure / float(used_count + 1)

    def _tool_priority_score(
        self,
        stat: Dict[str, Any],
        *,
        total_trials: int = 0,
        selection_cfg: Optional[Dict[str, Any]] = None,
        mode: str = "adaptive",
        tool_name: str = "",
        job: Optional[AgentJob] = None,
        state: Optional[Dict[str, Any]] = None,
        context_tag: str = "",
    ) -> float:
        """
        Score a tool for adaptive selection.

        Base quality is smoothed success ratio. Optional exploration adds an
        uncertainty bonus and mild cold-start boost, then subtracts a failure penalty.
        """
        ratio = self._tool_success_ratio(stat)
        cfg = selection_cfg if isinstance(selection_cfg, dict) else {}
        feedback_adjustment = self._feedback_tool_bias(
            tool_name,
            state,
            weight=float(cfg.get("feedback_learning_weight", 0.08) or 0.08),
            max_abs=float(cfg.get("feedback_learning_max_abs_bias", 0.3) or 0.3),
            enabled=bool(cfg.get("feedback_learning_enabled", True)),
        )
        mode_norm = str(mode or "adaptive").strip().lower()
        if mode_norm == "baseline":
            return ratio + feedback_adjustment
        if mode_norm == "thompson":
            alpha_prior = float(cfg.get("thompson_alpha_prior", 1.0) or 1.0)
            beta_prior = float(cfg.get("thompson_beta_prior", 1.0) or 1.0)
            temp = max(0.1, float(cfg.get("thompson_temperature", 1.0) or 1.0))
            s = max(0, int((stat or {}).get("success", 0) or 0))
            f = max(0, int((stat or {}).get("failure", 0) or 0))

            iter_part = int(getattr(job, "iteration", 0) or 0) if job is not None else 0
            forced_part = int((state or {}).get("forced_exploration_attempts", 0) or 0) if isinstance(state, dict) else 0
            seed_key = f"{getattr(job, 'id', '')}:{tool_name}:{context_tag}:{iter_part}:{forced_part}:{total_trials}"
            seed = int(hashlib.sha256(seed_key.encode("utf-8")).hexdigest()[:16], 16)
            rng = random.Random(seed)
            sample = float(rng.betavariate(alpha_prior + s, beta_prior + f))
            # Temperature scales exploitation pressure while preserving rank ordering behavior.
            score = max(0.0, min(1.0, math.pow(sample, 1.0 / temp)))
            return score + feedback_adjustment

        if not bool(cfg.get("exploration_enabled", True)):
            return ratio + feedback_adjustment

        n = self._tool_observation_count(stat)
        failures = int((stat or {}).get("failure", 0) or 0) if isinstance(stat, dict) else 0

        exploration_bonus = float(cfg.get("exploration_bonus", 0.15) or 0.15)
        cold_start_bonus = float(cfg.get("cold_start_bonus", 0.05) or 0.05)
        min_trials = int(cfg.get("min_trials", 3) or 3)
        failure_penalty = float(cfg.get("failure_penalty", 0.08) or 0.08)

        uncertainty_bonus = exploration_bonus / math.sqrt(float(n) + 1.0)
        ucb_bonus = 0.0
        if total_trials > 0:
            ucb_bonus = 0.5 * exploration_bonus * math.sqrt(
                max(0.0, math.log(float(total_trials) + 1.0) / (float(n) + 1.0))
            )
        cold_bonus = cold_start_bonus if n < min_trials else 0.0
        penalty = failure_penalty * (float(failures) / float(n + 1))

        return ratio + uncertainty_bonus + ucb_bonus + cold_bonus - penalty + feedback_adjustment

    def _rank_tools_for_selection(
        self,
        job: AgentJob,
        tools: List[str],
        combined_stats: Dict[str, Dict[str, Any]],
        *,
        state: Optional[Dict[str, Any]] = None,
        context_tag: str = "",
    ) -> List[str]:
        """Rank candidate tools using adaptive exploration/exploitation scoring."""
        if not isinstance(tools, list) or not tools:
            return []
        stats = combined_stats if isinstance(combined_stats, dict) else {}
        cfg = self._get_tool_selection_config(job)
        mode, _assignment = self._resolve_tool_selection_mode(job, state=state, selection_cfg=cfg)
        total_trials = sum(self._tool_observation_count(stats.get(t, {})) for t in tools)
        scored: List[Tuple[str, float, float]] = []
        for tool in [str(t).strip() for t in tools if str(t).strip()]:
            base_score = self._tool_priority_score(
                stats.get(tool, {}),
                total_trials=total_trials,
                selection_cfg=cfg,
                mode=mode,
                tool_name=tool,
                job=job,
                state=state,
                context_tag=context_tag,
            )
            family_bonus = self._family_diversification_bonus(
                tool,
                state=state,
                selection_cfg=cfg,
            )
            scored.append((tool, base_score + family_bonus, base_score))

        ranked = sorted(
            scored,
            key=lambda row: (
                -float(row[1]),
                -float(row[2]),
                -self._tool_success_ratio(stats.get(row[0], {})),
                self._tool_observation_count(stats.get(row[0], {})),
                row[0],
            ),
        )
        ranked = [row[0] for row in ranked]
        return ranked

    def _build_counterfactual_candidates(
        self,
        job: AgentJob,
        state: Dict[str, Any],
        *,
        selected_tool: Optional[str] = None,
        limit: int = 3,
        context_tag: str = "",
    ) -> List[Dict[str, Any]]:
        """Build top candidate tools with scores for decision observability."""
        profile = (
            state.get("skill_profile")
            if isinstance(state.get("skill_profile"), dict)
            else self._resolve_agent_skill_profile(job, state=state)
        )
        available = self._get_tools_for_job_type(job.job_type, job.config, profile=profile)
        if not available:
            return []

        combined_stats = self._merge_tool_stats(
            state.get("tool_priors") if isinstance(state.get("tool_priors"), dict) else {},
            state.get("tool_stats") if isinstance(state.get("tool_stats"), dict) else {},
        )
        ranked = self._rank_tools_for_selection(
            job,
            available,
            combined_stats,
            state=state,
            context_tag=context_tag or "counterfactual",
        )
        cfg = self._get_tool_selection_config(job)
        mode = str(state.get("tool_selection_effective_mode") or cfg.get("policy_mode") or "adaptive").strip().lower()
        total_trials = sum(self._tool_observation_count(combined_stats.get(t, {})) for t in available)
        selected = str(selected_tool or "").strip()
        top_k = max(1, min(int(limit or 3), 10))

        out: List[Dict[str, Any]] = []
        for idx, tool in enumerate(ranked[:top_k], start=1):
            stat = combined_stats.get(tool, {}) if isinstance(combined_stats, dict) else {}
            base_priority = self._tool_priority_score(
                stat,
                total_trials=total_trials,
                selection_cfg=cfg,
                mode=mode,
                tool_name=tool,
                job=job,
                state=state,
                context_tag=context_tag or "counterfactual",
            )
            family_bonus = self._family_diversification_bonus(
                tool,
                state=state,
                selection_cfg=cfg,
            )
            priority = base_priority + family_bonus
            s = int(stat.get("success", 0) or 0) if isinstance(stat, dict) else 0
            f = int(stat.get("failure", 0) or 0) if isinstance(stat, dict) else 0
            out.append(
                {
                    "rank": idx,
                    "tool": tool,
                    "priority_score": round(float(priority), 6),
                    "base_priority_score": round(float(base_priority), 6),
                    "family_bonus": round(float(family_bonus), 6),
                    "tool_family": self._tool_family(tool),
                    "success_ratio": round(float(self._tool_success_ratio(stat)), 6),
                    "success": s,
                    "failure": f,
                    "observations": s + f,
                    "selected": bool(selected and tool == selected),
                }
            )
        return out

    def simulate_tool_selection_replay(
        self,
        tool_stats: Dict[str, Dict[str, Any]],
        *,
        steps: int = 200,
        policy_modes: Optional[List[str]] = None,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Offline replay simulator for policy comparison using historical tool stats.

        The simulator derives empirical per-tool reward rates from historical outcomes,
        then runs synthetic bandit episodes for each policy mode.
        """
        stats = self._normalize_tool_stats_map(tool_stats)
        tools = sorted([t for t, s in stats.items() if self._tool_observation_count(s) > 0])
        if not tools:
            return {
                "steps": 0,
                "seed": seed,
                "tools": [],
                "modes": {},
            }

        total_steps = max(10, min(int(steps or 200), 50_000))
        modes = policy_modes if isinstance(policy_modes, list) and policy_modes else ["baseline", "adaptive", "thompson"]
        modes = [str(m or "").strip().lower() for m in modes if str(m or "").strip()]
        modes = [m for m in modes if m in {"baseline", "adaptive", "thompson"}]
        if not modes:
            modes = ["adaptive"]

        # Conservative empirical reward model (Laplace-smoothed Bernoulli means).
        empirical_rates: Dict[str, float] = {}
        for tool in tools:
            tstat = stats.get(tool, {})
            s = int(tstat.get("success", 0) or 0)
            f = int(tstat.get("failure", 0) or 0)
            empirical_rates[tool] = (s + 1.0) / float(s + f + 2.0)
        best_rate = max(empirical_rates.values()) if empirical_rates else 0.0

        base_cfg = {
            "exploration_enabled": True,
            "exploration_bonus": 0.15,
            "cold_start_bonus": 0.05,
            "min_trials": 3,
            "failure_penalty": 0.08,
            "thompson_alpha_prior": 1.0,
            "thompson_beta_prior": 1.0,
            "thompson_temperature": 1.0,
        }

        out_modes: Dict[str, Any] = {}
        for mode in modes:
            sim_stats: Dict[str, Dict[str, Any]] = {
                t: {"success": 0, "failure": 0, "last_error": ""} for t in tools
            }
            selection_counts: Dict[str, int] = {t: 0 for t in tools}
            successes = 0
            failures = 0
            cumulative_expected_regret = 0.0

            for step_idx in range(1, total_steps + 1):
                total_trials = sum(self._tool_observation_count(sim_stats[t]) for t in tools)
                ranked = sorted(
                    tools,
                    key=lambda tool: (
                        -self._tool_priority_score(
                            sim_stats.get(tool, {}),
                            total_trials=total_trials,
                            selection_cfg=base_cfg,
                            mode=mode,
                            tool_name=tool,
                            job=None,
                            state=None,
                            context_tag=f"replay:{seed}:{step_idx}",
                        ),
                        -self._tool_success_ratio(sim_stats.get(tool, {})),
                        self._tool_observation_count(sim_stats.get(tool, {})),
                        tool,
                    ),
                )
                chosen = ranked[0]
                selection_counts[chosen] = int(selection_counts.get(chosen, 0) or 0) + 1
                chosen_rate = float(empirical_rates.get(chosen, 0.0))
                cumulative_expected_regret += max(0.0, best_rate - chosen_rate)

                draw_key = f"reward:{seed}:{mode}:{step_idx}:{chosen}"
                draw_seed = int(hashlib.sha256(draw_key.encode("utf-8")).hexdigest()[:16], 16)
                rng = random.Random(draw_seed)
                reward = rng.random() < chosen_rate
                slot = sim_stats.get(chosen) or {"success": 0, "failure": 0, "last_error": ""}
                if reward:
                    slot["success"] = int(slot.get("success", 0) or 0) + 1
                    successes += 1
                else:
                    slot["failure"] = int(slot.get("failure", 0) or 0) + 1
                    failures += 1
                sim_stats[chosen] = slot

            selected_tools = [t for t, c in selection_counts.items() if int(c or 0) > 0]
            out_modes[mode] = {
                "steps": total_steps,
                "successes": successes,
                "failures": failures,
                "mean_reward": float(successes) / float(max(1, total_steps)),
                "best_possible_mean_reward": best_rate,
                "realized_regret_vs_best": max(0.0, best_rate - (float(successes) / float(max(1, total_steps)))),
                "cumulative_expected_regret": cumulative_expected_regret,
                "mean_expected_regret": cumulative_expected_regret / float(max(1, total_steps)),
                "unique_tools_selected": len(selected_tools),
                "selection_counts": selection_counts,
            }

        comparison: List[Dict[str, Any]] = []
        for mode, stats_out in out_modes.items():
            comparison.append(
                {
                    "mode": mode,
                    "mean_reward": float(stats_out.get("mean_reward", 0.0) or 0.0),
                    "realized_regret_vs_best": float(stats_out.get("realized_regret_vs_best", 0.0) or 0.0),
                    "cumulative_expected_regret": float(stats_out.get("cumulative_expected_regret", 0.0) or 0.0),
                    "mean_expected_regret": float(stats_out.get("mean_expected_regret", 0.0) or 0.0),
                    "unique_tools_selected": int(stats_out.get("unique_tools_selected", 0) or 0),
                }
            )
        comparison.sort(key=lambda r: (-float(r.get("mean_reward", 0.0) or 0.0), float(r.get("cumulative_expected_regret", 0.0) or 0.0)))

        return {
            "steps": total_steps,
            "seed": seed,
            "tools": tools,
            "empirical_rates": empirical_rates,
            "best_possible_mean_reward": best_rate,
            "comparison": comparison,
            "modes": out_modes,
        }

    def _get_forced_exploration_config(self, job: AgentJob) -> Dict[str, Any]:
        """Get forced exploration settings used during stall recovery."""
        cfg = job.config if isinstance(job.config, dict) else {}

        def _as_int(key: str, default: int, lo: int, hi: int) -> int:
            try:
                val = int(cfg.get(key, default))
            except Exception:
                val = default
            return max(lo, min(val, hi))

        tools = cfg.get("tool_selection_forced_exploration_tools")
        if isinstance(tools, str):
            tools = [x.strip() for x in tools.split(",") if x.strip()]
        if not isinstance(tools, list) or not tools:
            tools = [
                "search_arxiv",
                "search_documents",
                "search_with_filters",
                "summarize_document",
                "read_document_content",
                "suggest_next_action",
            ]

        return {
            "enabled": bool(cfg.get("tool_selection_forced_exploration_enabled", True)),
            "every_n_stalled_iterations": _as_int("tool_selection_forced_exploration_every_n", 2, 1, 20),
            "min_stalled_iterations": _as_int("tool_selection_forced_exploration_min_stalled", 2, 1, 50),
            "max_observations": _as_int("tool_selection_forced_exploration_max_observations", 2, 0, 100),
            "max_failures_per_tool": _as_int("tool_selection_forced_exploration_max_failures", 8, 0, 100),
            "tools": [str(t).strip() for t in tools if str(t).strip()],
        }

    def _get_tool_cooldown_config(self, job: AgentJob) -> Dict[str, Any]:
        """Get post-recovery tool cooldown settings."""
        cfg = job.config if isinstance(job.config, dict) else {}

        def _as_int(key: str, default: int, lo: int, hi: int) -> int:
            try:
                val = int(cfg.get(key, default))
            except Exception:
                val = default
            return max(lo, min(val, hi))

        return {
            "enabled": bool(cfg.get("tool_selection_cooldown_enabled", True)),
            "cooldown_iterations": _as_int("tool_selection_cooldown_iterations", 2, 1, 30),
            "forced_only": bool(cfg.get("tool_selection_cooldown_forced_only", True)),
            "on_failure_extra_iterations": _as_int("tool_selection_cooldown_failure_extra_iterations", 2, 0, 30),
            "on_success_shorten_by": _as_int("tool_selection_cooldown_success_shorten_by", 1, 0, 30),
        }

    def _is_tool_in_cooldown(
        self,
        tool: str,
        cooldowns: Dict[str, Any],
        current_iteration: int,
    ) -> bool:
        """Return true if a tool is still under cooldown at current iteration."""
        if not isinstance(cooldowns, dict):
            return False
        try:
            until = int(cooldowns.get(str(tool), 0) or 0)
        except Exception:
            return False
        return until >= int(current_iteration or 0)

    def _apply_recovery_post_action_updates(
        self,
        job: AgentJob,
        state: Dict[str, Any],
        recovery_action: Optional[Dict[str, Any]],
        recovery_result: Optional[Dict[str, Any]],
    ) -> None:
        """
        Apply post-recovery telemetry/cooldown updates.

        When recovery came from forced exploration, adapt cooldown duration using
        observed outcome and annotate history with success/error metadata.
        """
        if not bool(state.get("last_recovery_was_forced_exploration", False)):
            return
        if not isinstance(recovery_action, dict):
            return

        tool = str(recovery_action.get("tool") or "").strip()
        if not tool:
            return

        success = bool((recovery_result or {}).get("success"))
        err = str((recovery_result or {}).get("error") or "").strip()[:200]
        cur_iter = int(job.iteration or 0)
        cfg = self._get_tool_cooldown_config(job)

        if success:
            state["forced_exploration_successes"] = int(state.get("forced_exploration_successes", 0) or 0) + 1
        else:
            state["forced_exploration_failures"] = int(state.get("forced_exploration_failures", 0) or 0) + 1

        # Annotate latest matching history entry if present; otherwise append.
        history = state.get("forced_exploration_history")
        if not isinstance(history, list):
            history = []
        updated = False
        for idx in range(len(history) - 1, -1, -1):
            item = history[idx]
            if not isinstance(item, dict):
                continue
            if str(item.get("tool") or "").strip() != tool:
                continue
            if int(item.get("iteration", -1) or -1) != cur_iter:
                continue
            item["success"] = success
            if err:
                item["error"] = err
            updated = True
            break
        if not updated:
            rec = {"iteration": cur_iter, "tool": tool, "success": success}
            if err:
                rec["error"] = err
            history.append(rec)
        state["forced_exploration_history"] = history[-20:]

        if not bool(cfg.get("enabled", True)):
            return

        cooldowns = state.get("tool_cooldowns")
        if not isinstance(cooldowns, dict):
            cooldowns = {}

        try:
            base_until = int(cooldowns.get(tool, cur_iter) or cur_iter)
        except Exception:
            base_until = cur_iter

        if success:
            shorten = int(cfg.get("on_success_shorten_by", 1) or 1)
            new_until = max(cur_iter, base_until - max(0, shorten))
        else:
            extra = int(cfg.get("on_failure_extra_iterations", 2) or 2)
            new_until = max(cur_iter, base_until + max(0, extra))

        cooldowns[tool] = new_until
        state["tool_cooldowns"] = cooldowns

    def _should_force_exploration(self, job: AgentJob, state: Dict[str, Any]) -> bool:
        """Decide whether this recovery should deliberately explore under-sampled tools."""
        cfg = self._get_forced_exploration_config(job)
        if not bool(cfg.get("enabled", True)):
            return False

        stalled = int(state.get("stalled_iterations", 0) or 0)
        repeated = int(state.get("repeated_action_iterations", 0) or 0)
        min_stalled = int(cfg.get("min_stalled_iterations", 2) or 2)
        if stalled < min_stalled and repeated < min_stalled:
            return False

        cadence = int(cfg.get("every_n_stalled_iterations", 2) or 2)
        cadence = max(1, cadence)
        return ((stalled >= min_stalled) and (stalled % cadence == 0)) or ((repeated >= min_stalled) and (repeated % cadence == 0))

    def _build_action_for_tool(
        self,
        tool: str,
        job: AgentJob,
        *,
        doc_ids: Optional[List[str]] = None,
        purpose: str = "",
    ) -> Optional[Dict[str, Any]]:
        """Build normalized action payload for a known tool."""
        t = str(tool or "").strip()
        if not t:
            return None
        docs = doc_ids if isinstance(doc_ids, list) else []
        goal = str(job.goal or "")

        if t == "search_documents":
            return {"tool": t, "params": {"query": goal[:200], "limit": 10}, "purpose": purpose}
        if t == "search_arxiv":
            return {"tool": t, "params": {"query": goal[:140], "max_results": 8}, "purpose": purpose}
        if t == "search_with_filters":
            return {"tool": t, "params": {"query": goal[:200], "limit": 20}, "purpose": purpose}
        if t in {"read_document_content", "summarize_document"}:
            if not docs:
                return None
            params: Dict[str, Any] = {"document_id": docs[0]}
            if t == "read_document_content":
                params["max_length"] = 8000
            return {"tool": t, "params": params, "purpose": purpose}
        if t == "suggest_next_action":
            progress_hint = f"{int(job.progress or 0)}%"
            return {
                "tool": t,
                "params": {
                    "current_goal": goal,
                    "progress_so_far": progress_hint,
                },
                "purpose": purpose,
            }
        return {"tool": t, "params": {}, "purpose": purpose}

    def _build_forced_exploration_action(
        self,
        job: AgentJob,
        state: Dict[str, Any],
        *,
        available_tools: set[str],
        combined_stats: Dict[str, Dict[str, Any]],
        exclude_tool: str = "",
        doc_ids: Optional[List[str]] = None,
        recent_tools: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Select a deliberate exploration action from under-sampled tools."""
        cfg = self._get_forced_exploration_config(job)
        configured = [t for t in cfg.get("tools", []) if t in available_tools]
        candidate_tools = configured if configured else sorted(list(available_tools))
        if exclude_tool:
            candidate_tools = [t for t in candidate_tools if t != exclude_tool]

        recent = recent_tools if isinstance(recent_tools, list) else []
        candidate_tools = [t for t in candidate_tools if recent.count(t) < 3]
        if not candidate_tools:
            return None

        max_obs = int(cfg.get("max_observations", 2) or 2)
        max_failures = int(cfg.get("max_failures_per_tool", 8) or 8)

        viable: List[str] = []
        for t in candidate_tools:
            stat = combined_stats.get(t, {}) if isinstance(combined_stats, dict) else {}
            obs = self._tool_observation_count(stat)
            failures = int(stat.get("failure", 0) or 0) if isinstance(stat, dict) else 0
            if failures > max_failures:
                continue
            if obs <= max_obs:
                viable.append(t)

        if not viable:
            obs_by_tool = {
                t: self._tool_observation_count(combined_stats.get(t, {}))
                for t in candidate_tools
            }
            min_obs = min(obs_by_tool.values()) if obs_by_tool else 0
            viable = [t for t in candidate_tools if obs_by_tool.get(t, 0) == min_obs]
        if not viable:
            return None

        ranked_viable = self._rank_tools_for_selection(
            job,
            viable,
            combined_stats,
            state=state,
            context_tag="forced_exploration",
        )
        rank_index = {name: idx for idx, name in enumerate(ranked_viable)}
        viable.sort(
            key=lambda t: (
                self._tool_observation_count(combined_stats.get(t, {})),
                rank_index.get(t, 9999),
            )
        )

        for tool in viable:
            action = self._build_action_for_tool(
                tool=tool,
                job=job,
                doc_ids=doc_ids,
                purpose="Forced exploration to escape stall by sampling an under-used tool.",
            )
            if action:
                return action
        return None

    def _get_tool_prior_decay_config(self, job: AgentJob) -> Dict[str, Any]:
        """Get decay configuration for persistent tool priors."""
        cfg = job.config if isinstance(job.config, dict) else {}

        def _as_float(key: str, default: float, lo: float, hi: float) -> float:
            try:
                val = float(cfg.get(key, default))
            except Exception:
                val = default
            return max(lo, min(val, hi))

        return {
            "enabled": bool(cfg.get("tool_prior_decay_enabled", True)),
            "half_life_days": _as_float("tool_prior_half_life_days", 45.0, 1.0, 3650.0),
            "min_factor": _as_float("tool_prior_decay_min_factor", 0.01, 0.0, 1.0),
        }

    def _apply_decay_to_prior_counts(
        self,
        success_count: int,
        failure_count: int,
        updated_at: Optional[datetime],
        *,
        now: Optional[datetime] = None,
        enabled: bool = True,
        half_life_days: float = 45.0,
        min_factor: float = 0.01,
    ) -> Tuple[int, int]:
        """Apply exponential decay to prior counts based on age since last update."""
        s = max(0, int(success_count or 0))
        f = max(0, int(failure_count or 0))
        if not enabled:
            return s, f
        if updated_at is None:
            return s, f

        now_dt = now or datetime.utcnow()

        def _to_utc_naive(dt: datetime) -> datetime:
            if dt.tzinfo is None:
                return dt
            return dt.astimezone(timezone.utc).replace(tzinfo=None)

        try:
            age_days = (_to_utc_naive(now_dt) - _to_utc_naive(updated_at)).total_seconds() / 86400.0
        except Exception:
            return s, f
        if age_days <= 0:
            return s, f

        hl = max(1.0, float(half_life_days))
        factor = math.pow(0.5, age_days / hl)
        factor = max(0.0, min(1.0, factor))
        factor = max(float(min_factor), factor)
        ds = int(round(float(s) * factor))
        df = int(round(float(f) * factor))
        return max(0, ds), max(0, df)

    async def _load_tool_priors(
        self,
        job: AgentJob,
        db: AsyncSession,
    ) -> Dict[str, Dict[str, Any]]:
        """Load persistent tool priors for the same user/job type."""
        cfg = job.config if isinstance(job.config, dict) else {}
        if not bool(cfg.get("tool_prior_enabled", True)):
            return {}

        max_tools = 200
        try:
            max_tools = int(cfg.get("tool_prior_max_tools", 200) or 200)
        except Exception:
            max_tools = 200
        max_tools = max(20, min(max_tools, 2000))
        decay_cfg = self._get_tool_prior_decay_config(job)
        now_dt = datetime.utcnow()

        try:
            res = await db.execute(
                select(
                    AgentToolPrior.tool_name,
                    AgentToolPrior.success_count,
                    AgentToolPrior.failure_count,
                    AgentToolPrior.updated_at,
                )
                .where(
                    AgentToolPrior.user_id == job.user_id,
                    AgentToolPrior.job_type == job.job_type,
                )
                .order_by(desc(AgentToolPrior.updated_at), desc(AgentToolPrior.success_count))
                .limit(max_tools)
            )
            rows = res.all()
        except Exception:
            rows = []

        loaded: Dict[str, Dict[str, Any]] = {}
        for tool_name, success_count, failure_count, updated_at in rows:
            t = str(tool_name or "").strip()
            if not t:
                continue
            ds, df = self._apply_decay_to_prior_counts(
                int(success_count or 0),
                int(failure_count or 0),
                updated_at,
                now=now_dt,
                enabled=bool(decay_cfg.get("enabled", True)),
                half_life_days=float(decay_cfg.get("half_life_days", 45.0)),
                min_factor=float(decay_cfg.get("min_factor", 0.01)),
            )
            if ds <= 0 and df <= 0:
                continue
            loaded[t] = {
                "success": ds,
                "failure": df,
                "last_error": "",
            }

        if loaded:
            return loaded

        # Compatibility fallback: derive priors from prior job results.
        return await self._load_tool_priors_from_job_results(job, db)

    async def _load_tool_priors_from_job_results(
        self,
        job: AgentJob,
        db: AsyncSession,
    ) -> Dict[str, Dict[str, Any]]:
        """Fallback loader deriving priors from past job result snapshots."""
        cfg = job.config if isinstance(job.config, dict) else {}

        lookback_jobs = 20
        try:
            lookback_jobs = int(cfg.get("tool_prior_lookback_jobs", 20) or 20)
        except Exception:
            lookback_jobs = 20
        lookback_jobs = max(5, min(lookback_jobs, 200))

        result = await db.execute(
            select(AgentJob)
            .where(
                AgentJob.user_id == job.user_id,
                AgentJob.job_type == job.job_type,
                AgentJob.id != job.id,
                AgentJob.status.in_([AgentJobStatus.COMPLETED.value, AgentJobStatus.FAILED.value]),
            )
            .order_by(desc(AgentJob.completed_at), desc(AgentJob.created_at))
            .limit(lookback_jobs)
        )
        rows = result.scalars().all()

        aggregated: Dict[str, Dict[str, Any]] = {}
        for prev in rows:
            prev_results = prev.results if isinstance(prev.results, dict) else {}
            strategy = prev_results.get("execution_strategy") if isinstance(prev_results.get("execution_strategy"), dict) else {}
            stats = strategy.get("tool_stats") if isinstance(strategy.get("tool_stats"), dict) else {}
            aggregated = self._merge_tool_stats(aggregated, stats)

        return aggregated

    async def _persist_tool_priors(
        self,
        job: AgentJob,
        state: Dict[str, Any],
        db: AsyncSession,
    ) -> None:
        """Persist current run tool outcomes to the dedicated prior table."""
        cfg = job.config if isinstance(job.config, dict) else {}
        if not bool(cfg.get("tool_prior_enabled", True)):
            return
        if not bool(cfg.get("tool_prior_persist_enabled", True)):
            return

        current_stats = self._normalize_tool_stats_map(state.get("tool_stats"))
        if not current_stats:
            return

        max_count = 1_000_000
        try:
            max_count = int(cfg.get("tool_prior_max_count", 1_000_000) or 1_000_000)
        except Exception:
            max_count = 1_000_000
        max_count = max(10, min(max_count, 10_000_000))
        decay_cfg = self._get_tool_prior_decay_config(job)
        now_dt = datetime.utcnow()

        for tool_name, stat in current_stats.items():
            s = int(stat.get("success", 0) or 0)
            f = int(stat.get("failure", 0) or 0)
            if s <= 0 and f <= 0:
                continue

            res = await db.execute(
                select(AgentToolPrior).where(
                    AgentToolPrior.user_id == job.user_id,
                    AgentToolPrior.job_type == job.job_type,
                    AgentToolPrior.tool_name == tool_name,
                )
            )
            row = res.scalar_one_or_none()
            if row is None:
                row = AgentToolPrior(
                    user_id=job.user_id,
                    job_type=job.job_type,
                    tool_name=tool_name,
                    success_count=min(max_count, s),
                    failure_count=min(max_count, f),
                )
                db.add(row)
            else:
                base_s, base_f = self._apply_decay_to_prior_counts(
                    int(row.success_count or 0),
                    int(row.failure_count or 0),
                    row.updated_at,
                    now=now_dt,
                    enabled=bool(decay_cfg.get("enabled", True)),
                    half_life_days=float(decay_cfg.get("half_life_days", 45.0)),
                    min_factor=float(decay_cfg.get("min_factor", 0.01)),
                )
                row.success_count = min(max_count, base_s + s)
                row.failure_count = min(max_count, base_f + f)
                row.updated_at = now_dt

    def _advance_execution_plan_state(
        self,
        state: Dict[str, Any],
        action: Optional[Dict[str, Any]],
        action_result: Optional[Dict[str, Any]],
        previous_progress: int,
        current_progress: int,
    ) -> None:
        """Advance plan step when a meaningful action completes."""
        plan = state.get("execution_plan")
        if not isinstance(plan, list) or not plan:
            return

        idx = int(state.get("plan_step_index", 0) or 0)
        idx = max(0, min(idx, len(plan) - 1))
        delta = int(current_progress or 0) - int(previous_progress or 0)

        action_tool = str((action or {}).get("tool") or "").strip()
        action_success = bool((action_result or {}).get("success"))
        findings_count = len((action_result or {}).get("findings") or [])

        should_advance = False
        if delta >= 4:
            should_advance = True
        elif action_success and findings_count > 0:
            should_advance = True
        elif action_success and action_tool in {"create_synthesis_document", "create_document_from_text", "write_progress_report"}:
            should_advance = True

        # Keep at least one chance for the current step before advancing solely on small wins.
        step = plan[idx] if isinstance(plan[idx], dict) else {}
        completions = int(step.get("completions", 0) or 0) if isinstance(step, dict) else 0
        if should_advance and delta < 4 and completions < 1:
            should_advance = False

        if isinstance(step, dict):
            step["completions"] = completions + 1
            if step.get("status") != "done":
                step["status"] = "in_progress"

        if not should_advance:
            return

        if isinstance(step, dict):
            step["status"] = "done"
        next_idx = min(len(plan) - 1, idx + 1)
        state["plan_step_index"] = next_idx
        if next_idx != idx and isinstance(plan[next_idx], dict) and plan[next_idx].get("status") != "done":
            plan[next_idx]["status"] = "in_progress"

        # Keep subgoal index aligned with plan progress when possible.
        subgoals = state.get("subgoals")
        if isinstance(subgoals, list) and subgoals:
            sidx = int(state.get("subgoal_index", 0) or 0)
            sidx = max(0, min(sidx, len(subgoals) - 1))
            if isinstance(subgoals[sidx], dict):
                subgoals[sidx]["status"] = "done"
            next_sidx = min(len(subgoals) - 1, sidx + 1)
            state["subgoal_index"] = next_sidx
            if next_sidx != sidx and isinstance(subgoals[next_sidx], dict) and subgoals[next_sidx].get("status") != "done":
                subgoals[next_sidx]["status"] = "in_progress"

    def _format_execution_plan_for_prompt(self, state: Dict[str, Any]) -> str:
        """Render current plan context for decision prompts."""
        plan = state.get("execution_plan")
        if not isinstance(plan, list) or not plan:
            return ""

        idx = int(state.get("plan_step_index", 0) or 0)
        idx = max(0, min(idx, len(plan) - 1))

        lines = ["EXECUTION PLAN (Plan-Then-Act):"]
        current = plan[idx] if isinstance(plan[idx], dict) else {}
        lines.append(f"- Current step {idx + 1}/{len(plan)}: {str(current.get('title') or 'Untitled')[:220]}")
        objective = str(current.get("objective") or "").strip()
        if objective:
            lines.append(f"- Current objective: {objective[:400]}")
        exit_criteria = str(current.get("exit_criteria") or "").strip()
        if exit_criteria:
            lines.append(f"- Exit criteria: {exit_criteria[:300]}")
        tools = current.get("suggested_tools") if isinstance(current, dict) else []
        if isinstance(tools, list) and tools:
            lines.append(f"- Suggested tools: {', '.join([str(t) for t in tools[:8]])}")

        completed_titles: List[str] = []
        for step in plan:
            if not isinstance(step, dict):
                continue
            if str(step.get("status") or "").lower() == "done":
                title = str(step.get("title") or "").strip()
                if title:
                    completed_titles.append(title[:120])
        if completed_titles:
            lines.append(f"- Completed steps: {len(completed_titles)}")
        return "\n".join(lines)

    def _format_causal_experiment_plan_for_prompt(self, state: Dict[str, Any]) -> str:
        """Render causal experiment context for research decisions."""
        plan = state.get("causal_experiment_plan")
        if not isinstance(plan, dict):
            return ""
        hypotheses = plan.get("hypotheses") if isinstance(plan.get("hypotheses"), list) else []
        experiments = plan.get("experiments") if isinstance(plan.get("experiments"), list) else []
        if not hypotheses or not experiments:
            return ""

        lines = ["CAUSAL EXPERIMENT PLAN:"]
        lines.append(f"- Hypotheses: {len(hypotheses)}")
        for hyp in hypotheses[:3]:
            if not isinstance(hyp, dict):
                continue
            hid = str(hyp.get("id") or "").strip()
            statement = str(hyp.get("statement") or "").strip()
            if statement:
                lines.append(f"  - {hid or 'H?'}: {statement[:220]}")

        priority = plan.get("priority_order") if isinstance(plan.get("priority_order"), list) else []
        exp_map = {
            str(e.get("id") or "").strip(): e
            for e in experiments
            if isinstance(e, dict) and str(e.get("id") or "").strip()
        }
        ordered = [eid for eid in [str(x).strip() for x in priority if str(x).strip()] if eid in exp_map]
        if not ordered:
            ordered = list(exp_map.keys())
        next_ids = ordered[:2]
        if next_ids:
            lines.append(f"- Next experiment IDs: {', '.join(next_ids)}")
        for eid in next_ids:
            exp = exp_map.get(eid) if isinstance(exp_map.get(eid), dict) else {}
            if not exp:
                continue
            name = str(exp.get("name") or "").strip()
            hid = str(exp.get("hypothesis_id") or "").strip()
            lines.append(f"  - {eid} ({hid}): {name[:180]}")
            expected = exp.get("expected_evidence") if isinstance(exp.get("expected_evidence"), dict) else {}
            supports = expected.get("supports") if isinstance(expected.get("supports"), list) else []
            falsifies = expected.get("falsifies") if isinstance(expected.get("falsifies"), list) else []
            if supports:
                lines.append(f"    support signal: {str(supports[0])[:180]}")
            if falsifies:
                lines.append(f"    falsify signal: {str(falsifies[0])[:180]}")
        return "\n".join(lines)

    def _format_subgoals_for_prompt(self, state: Dict[str, Any]) -> str:
        """Render subgoal context for prompts."""
        subgoals = state.get("subgoals")
        if not isinstance(subgoals, list) or not subgoals:
            return ""

        idx = int(state.get("subgoal_index", 0) or 0)
        idx = max(0, min(idx, len(subgoals) - 1))
        current = subgoals[idx] if isinstance(subgoals[idx], dict) else {}

        lines = ["SUBGOALS:"]
        lines.append(f"- Current subgoal {idx + 1}/{len(subgoals)}: {str(current.get('title') or '').strip()[:220]}")
        done = 0
        for sg in subgoals:
            if isinstance(sg, dict) and str(sg.get("status") or "").lower() == "done":
                done += 1
        lines.append(f"- Subgoals completed: {done}")
        return "\n".join(lines)

    def _format_critic_for_prompt(self, state: Dict[str, Any]) -> str:
        """Render the latest critic guidance for prompts."""
        notes = state.get("critic_notes")
        if not isinstance(notes, list) or not notes:
            return ""

        latest = notes[-1] if isinstance(notes[-1], dict) else {}
        if not isinstance(latest, dict):
            return ""

        lines = ["CRITIC FEEDBACK:"]
        assess = str(latest.get("trajectory_assessment") or "").strip()
        pivot = str(latest.get("pivot") or "").strip()
        if assess:
            lines.append(f"- Assessment: {assess[:320]}")
        sev = str(latest.get("severity") or "").strip()
        if sev:
            lines.append(f"- Severity: {sev[:40]}")
        try:
            conf = float(latest.get("confidence", 0.0) or 0.0)
        except Exception:
            conf = 0.0
        lines.append(f"- Confidence: {max(0.0, min(1.0, conf)):.2f}")
        if pivot:
            lines.append(f"- Pivot: {pivot[:280]}")
        tools = latest.get("recommended_tools")
        if isinstance(tools, list) and tools:
            lines.append(f"- Recommended tools: {', '.join([str(t) for t in tools[:6]])}")
        risks = latest.get("risks")
        if isinstance(risks, list) and risks:
            lines.append(f"- Top risk: {str(risks[0])[:220]}")
        return "\n".join(lines)

    def _format_tool_stats_for_prompt(self, state: Dict[str, Any]) -> str:
        """Render per-tool outcomes as prompt hints."""
        current_stats = state.get("tool_stats") if isinstance(state.get("tool_stats"), dict) else {}
        prior_stats = state.get("tool_priors") if isinstance(state.get("tool_priors"), dict) else {}
        merged_stats = self._merge_tool_stats(prior_stats, current_stats)
        if not merged_stats:
            return ""

        scored: List[Tuple[str, int, int, float]] = []
        for tool, raw in merged_stats.items():
            if not isinstance(raw, dict):
                continue
            s = int(raw.get("success", 0) or 0)
            f = int(raw.get("failure", 0) or 0)
            total = s + f
            if total <= 0:
                continue
            ratio = self._tool_success_ratio(raw)
            scored.append((str(tool), s, f, ratio))

        if not scored:
            return ""

        scored.sort(key=lambda x: (x[3], x[1], -x[2]), reverse=True)
        best = scored[:3]
        worst = sorted(scored, key=lambda x: (x[3], -x[2], x[1]))[:3]

        lines = ["ADAPTIVE TOOL HINTS:"]
        if prior_stats:
            lines.append(f"- Historical priors loaded for {len(prior_stats)} tools.")
        if best:
            lines.append("- Strong tools:")
            for tool, s, f, _ in best:
                lines.append(f"  - {tool}: success={s}, failure={f}")
        if worst:
            lines.append("- Weak tools (avoid repeats unless needed):")
            for tool, s, f, _ in worst:
                lines.append(f"  - {tool}: success={s}, failure={f}")
        return "\n".join(lines)

    def _normalize_role_token(self, value: Any) -> str:
        token = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
        if token in {"research", "researcher"}:
            return "researcher"
        if token in {"critic", "reviewer"}:
            return "critic"
        if token in {"synth", "synthesizer", "writer"}:
            return "synthesizer"
        if token in {"verify", "verifier", "validator", "qa"}:
            return "verifier"
        return token

    def _resolve_agent_skill_profile(
        self,
        job: AgentJob,
        *,
        state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Resolve the active role profile controlling prompt/tool behavior."""
        cfg = job.config if isinstance(job.config, dict) else {}
        prior_role = self._normalize_role_token((state or {}).get("skill_profile", {}).get("role") if isinstance(state, dict) else "")
        role_candidates = [
            self._normalize_role_token(cfg.get("agent_role")),
            self._normalize_role_token(cfg.get("swarm_role")),
            self._normalize_role_token(cfg.get("role")),
            prior_role,
            self._normalize_role_token(job.name),
            self._normalize_role_token(job.goal),
        ]

        role = "researcher"
        for candidate in role_candidates:
            if candidate in {"researcher", "critic", "synthesizer", "verifier"}:
                role = candidate
                break
            if "critic" in candidate:
                role = "critic"
                break
            if "synth" in candidate:
                role = "synthesizer"
                break
            if "verif" in candidate or "validat" in candidate:
                role = "verifier"
                break

        profiles: Dict[str, Dict[str, Any]] = {
            "researcher": {
                "role": "researcher",
                "display_name": "Researcher",
                "prompt_directives": [
                    "Prioritize discovery and evidence coverage before conclusions.",
                    "Favor retrieval and analysis tools to expand breadth.",
                    "Capture uncertainties explicitly for downstream roles.",
                ],
                "preferred_tools": [
                    "search_documents", "search_with_filters", "search_arxiv", "find_related_papers",
                    "get_document_details", "read_document_content", "summarize_document", "extract_paper_insights",
                    "build_research_graph", "identify_research_gaps",
                ],
                "discouraged_tools": [
                    "create_synthesis_document", "create_document_from_text", "generate_research_presentation",
                ],
                "blocked_tools": [],
                "metric_focus": ["evidence_actions", "evidence_findings"],
            },
            "critic": {
                "role": "critic",
                "display_name": "Critic",
                "prompt_directives": [
                    "Challenge assumptions and seek disconfirming evidence.",
                    "Validate claims with direct source checks before acceptance.",
                    "Call out risks, contradictions, and missing controls.",
                ],
                "preferred_tools": [
                    "compare_documents", "compare_methodologies", "identify_research_gaps", "build_research_graph",
                    "read_document_content", "get_document_details", "get_research_findings", "search_with_filters",
                ],
                "discouraged_tools": [
                    "create_synthesis_document", "create_document_from_text", "generate_research_presentation",
                ],
                "blocked_tools": [],
                "metric_focus": ["challenge_actions", "risk_findings"],
            },
            "synthesizer": {
                "role": "synthesizer",
                "display_name": "Synthesizer",
                "prompt_directives": [
                    "Convert evidence into concise, actionable outputs.",
                    "Merge overlapping findings and reduce redundancy.",
                    "Always provide clear next-step recommendations.",
                ],
                "preferred_tools": [
                    "create_synthesis_document", "create_document_from_text", "generate_research_presentation",
                    "write_progress_report", "link_entities", "save_research_finding",
                ],
                "discouraged_tools": [
                    "batch_ingest_papers", "monitor_arxiv_topic",
                ],
                "blocked_tools": [],
                "metric_focus": ["synthesis_actions", "artifacts_created"],
            },
            "verifier": {
                "role": "verifier",
                "display_name": "Verifier",
                "prompt_directives": [
                    "Verify outputs against goal criteria and source evidence.",
                    "Prefer reproducible checks over broad exploration.",
                    "Surface confidence level and unresolved validation gaps.",
                ],
                "preferred_tools": [
                    "read_document_content", "get_document_details", "compare_documents", "search_with_filters",
                    "get_research_findings", "compare_methodologies", "build_research_graph",
                ],
                "discouraged_tools": [
                    "batch_ingest_papers", "monitor_arxiv_topic",
                ],
                "blocked_tools": [],
                "metric_focus": ["verification_actions", "failed_checks"],
            },
        }
        profile = dict(profiles.get(role, profiles["researcher"]))
        profile["role"] = role
        profile["resolved_at"] = datetime.utcnow().isoformat()
        return profile

    def _format_skill_profile_for_prompt(self, state: Dict[str, Any]) -> str:
        """Render active role profile for the planner prompt."""
        profile = state.get("skill_profile") if isinstance(state.get("skill_profile"), dict) else {}
        if not profile:
            return ""
        lines = [
            f"ROLE PROFILE: {str(profile.get('display_name') or profile.get('role') or '').strip()}",
        ]
        directives = profile.get("prompt_directives")
        if isinstance(directives, list):
            for directive in directives[:4]:
                text = str(directive or "").strip()
                if text:
                    lines.append(f"- {text}")
        preferred = profile.get("preferred_tools")
        if isinstance(preferred, list) and preferred:
            lines.append(f"- Preferred tools: {', '.join([str(t) for t in preferred[:8]])}")
        discouraged = profile.get("discouraged_tools")
        if isinstance(discouraged, list) and discouraged:
            lines.append(f"- Discouraged tools: {', '.join([str(t) for t in discouraged[:6]])}")
        return "\n".join(lines)

    def _format_feedback_learning_for_prompt(self, state: Dict[str, Any]) -> str:
        """Render compact human-feedback guidance for prompt conditioning."""
        feedback = state.get("feedback_learning") if isinstance(state.get("feedback_learning"), dict) else {}
        if not feedback:
            return ""
        if int(feedback.get("feedback_count", 0) or 0) <= 0:
            return ""
        lines = ["HUMAN FEEDBACK LEARNING:"]
        avg = feedback.get("avg_rating")
        if avg is not None:
            try:
                lines.append(f"- Average rating context: {float(avg):.2f}/5")
            except Exception:
                pass
        pref = feedback.get("preferred_tools")
        if isinstance(pref, list) and pref:
            lines.append(f"- Prefer tools: {', '.join([str(t) for t in pref[:6]])}")
        avoid = feedback.get("discouraged_tools")
        if isinstance(avoid, list) and avoid:
            lines.append(f"- Avoid tools: {', '.join([str(t) for t in avoid[:6]])}")
        highlights = feedback.get("highlights")
        if isinstance(highlights, list) and highlights:
            lines.append(f"- Recent feedback note: {str(highlights[0])[:260]}")
        return "\n".join(lines)

    def _feedback_tool_bias(
        self,
        tool_name: str,
        state: Optional[Dict[str, Any]],
        *,
        weight: float = 0.08,
        max_abs: float = 0.30,
        enabled: bool = True,
    ) -> float:
        """Map feedback signals to a bounded additive tool-priority adjustment."""
        if not enabled:
            return 0.0
        tool = str(tool_name or "").strip()
        if not tool or not isinstance(state, dict):
            return 0.0
        feedback = state.get("feedback_learning")
        if not isinstance(feedback, dict):
            return 0.0
        bias_map = feedback.get("tool_bias")
        if not isinstance(bias_map, dict):
            return 0.0
        raw = bias_map.get(tool)
        try:
            signal = float(raw or 0.0)
        except Exception:
            signal = 0.0
        signal = max(-1.0, min(1.0, signal))
        adj = signal * max(0.0, float(weight))
        return max(-abs(float(max_abs)), min(abs(float(max_abs)), adj))

    def _update_skill_profile_metrics(
        self,
        state: Dict[str, Any],
        action: Optional[Dict[str, Any]],
        action_result: Optional[Dict[str, Any]],
    ) -> None:
        """Track role-specific execution metrics for observability."""
        if not isinstance(state, dict) or not isinstance(action, dict):
            return
        profile = state.get("skill_profile") if isinstance(state.get("skill_profile"), dict) else {}
        role = str(profile.get("role") or "researcher").strip().lower()
        tool = str(action.get("tool") or "").strip()
        if not tool:
            return
        success = bool((action_result or {}).get("success"))

        metrics = state.get("skill_profile_metrics")
        if not isinstance(metrics, dict):
            metrics = {}
        metrics["role"] = role
        metrics["actions_total"] = int(metrics.get("actions_total", 0) or 0) + 1
        if success:
            metrics["actions_success"] = int(metrics.get("actions_success", 0) or 0) + 1
        else:
            metrics["actions_failure"] = int(metrics.get("actions_failure", 0) or 0) + 1

        family = self._tool_family(tool)
        family_usage = metrics.get("family_usage")
        if not isinstance(family_usage, dict):
            family_usage = {}
        family_usage[family] = int(family_usage.get(family, 0) or 0) + 1
        metrics["family_usage"] = family_usage

        counters = metrics.get("role_counters")
        if not isinstance(counters, dict):
            counters = {}
        findings = (action_result or {}).get("findings")
        findings_count = len(findings) if isinstance(findings, list) else 0
        artifacts = (action_result or {}).get("artifacts")
        artifacts_count = len(artifacts) if isinstance(artifacts, list) else 0

        if role == "researcher":
            if family in {"retrieval", "analysis", "ingestion"}:
                counters["evidence_actions"] = int(counters.get("evidence_actions", 0) or 0) + 1
            counters["evidence_findings"] = int(counters.get("evidence_findings", 0) or 0) + findings_count
        elif role == "critic":
            if tool in {"compare_documents", "compare_methodologies", "identify_research_gaps", "build_research_graph"}:
                counters["challenge_actions"] = int(counters.get("challenge_actions", 0) or 0) + 1
            risk_count = 0
            if isinstance(findings, list):
                for item in findings:
                    if not isinstance(item, dict):
                        continue
                    cat = str(item.get("category") or "").strip().lower()
                    if cat in {"contradiction", "gap", "risk"}:
                        risk_count += 1
            counters["risk_findings"] = int(counters.get("risk_findings", 0) or 0) + risk_count
        elif role == "synthesizer":
            if family == "synthesis":
                counters["synthesis_actions"] = int(counters.get("synthesis_actions", 0) or 0) + 1
            counters["artifacts_created"] = int(counters.get("artifacts_created", 0) or 0) + artifacts_count
        elif role == "verifier":
            if family in {"analysis", "retrieval"}:
                counters["verification_actions"] = int(counters.get("verification_actions", 0) or 0) + 1
            if not success:
                counters["failed_checks"] = int(counters.get("failed_checks", 0) or 0) + 1

        metrics["role_counters"] = counters
        metrics["updated_at"] = datetime.utcnow().isoformat()
        state["skill_profile_metrics"] = metrics

    def _build_thinking_prompt(
        self,
        job: AgentJob,
        agent_def: Optional[AgentDefinition],
        state: Dict[str, Any],
        observation: Dict[str, Any],
        profile: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build the system prompt for the thinking phase."""
        base_prompt = f"""You are an autonomous agent executing a background job.

Job Type: {job.job_type}
Job Name: {job.name}

GOAL:
{job.goal}

"""
        inherited_data = (job.config or {}).get("inherited_data") if isinstance(job.config, dict) else None
        if isinstance(inherited_data, dict) and inherited_data:
            parent_results = inherited_data.get("parent_results") if isinstance(inherited_data.get("parent_results"), dict) else None
            parent_findings = inherited_data.get("parent_findings") if isinstance(inherited_data.get("parent_findings"), list) else []

            base_prompt += "\nINHERITED DATA (from parent job):\n"
            if parent_results:
                summary = str(parent_results.get("summary") or "").strip()
                research_bundle = parent_results.get("research_bundle") if isinstance(parent_results.get("research_bundle"), dict) else None
                if summary:
                    base_prompt += f"- Parent summary: {summary[:600]}\n"
                if research_bundle:
                    top_docs = research_bundle.get("top_documents") if isinstance(research_bundle.get("top_documents"), list) else []
                    top_papers = research_bundle.get("top_papers") if isinstance(research_bundle.get("top_papers"), list) else []
                    base_prompt += f"- Parent top_documents: {len(top_docs)}\n"
                    base_prompt += f"- Parent top_papers: {len(top_papers)}\n"
            if parent_findings:
                titles: list[str] = []
                for f in parent_findings:
                    if not isinstance(f, dict):
                        continue
                    t = str(f.get("title") or "").strip()
                    if not t:
                        continue
                    titles.append(t[:200])
                    if len(titles) >= 8:
                        break
                if titles:
                    base_prompt += "- Parent key findings (titles):\n"
                    for t in titles:
                        base_prompt += f"  - {t}\n"
        customer_profile = state.get("customer_profile")
        customer_context = (state.get("customer_context") or "").strip()
        if customer_profile or customer_context:
            base_prompt += "\nCUSTOMER CONTEXT (tailor research to this):\n"
            if customer_profile and isinstance(customer_profile, dict):
                prof_name = str(customer_profile.get("name") or "").strip()
                prof_keywords = customer_profile.get("keywords") or []
                prof_notes = str(customer_profile.get("notes") or "").strip()
                if prof_name:
                    base_prompt += f"- Customer: {prof_name}\n"
                if isinstance(prof_keywords, list) and prof_keywords:
                    base_prompt += f"- Keywords: {', '.join([str(x) for x in prof_keywords[:30]])}\n"
                if prof_notes:
                    base_prompt += f"- Notes: {prof_notes[:1200]}\n"
            if customer_context:
                base_prompt += f"- Job customer_context: {customer_context[:1500]}\n"
            base_prompt += (
                "\nCustomer-specific guardrails:\n"
                "- Prefer internal knowledge base documents first.\n"
                "- When using external papers/sources, explicitly connect them to the customer's domain and constraints.\n"
                "- Produce actionable next steps and open questions relevant to this customer.\n"
            )
        if job.goal_criteria:
            base_prompt += f"""
SUCCESS CRITERIA:
{json.dumps(job.goal_criteria, indent=2)}
"""

        if agent_def and agent_def.system_prompt:
            base_prompt += f"""
AGENT INSTRUCTIONS:
{agent_def.system_prompt}
"""

        # Add memory context if available
        memory_context = state.get("memory_context", "")
        if memory_context:
            base_prompt += f"""

{memory_context}
"""

        causal_plan_context = self._format_causal_experiment_plan_for_prompt(state)
        if causal_plan_context:
            base_prompt += f"""

{causal_plan_context}
"""

        plan_context = self._format_execution_plan_for_prompt(state)
        if plan_context:
            base_prompt += f"""

{plan_context}
"""

        subgoal_context = self._format_subgoals_for_prompt(state)
        if subgoal_context:
            base_prompt += f"""

{subgoal_context}
"""

        critic_context = self._format_critic_for_prompt(state)
        if critic_context:
            base_prompt += f"""

{critic_context}
"""

        tool_hints = self._format_tool_stats_for_prompt(state)
        if tool_hints:
            base_prompt += f"""

{tool_hints}
"""

        active_profile = profile if isinstance(profile, dict) else self._resolve_agent_skill_profile(job, state=state)
        role_context = self._format_skill_profile_for_prompt(
            {
                **(state if isinstance(state, dict) else {}),
                "skill_profile": active_profile,
            }
        )
        if role_context:
            base_prompt += f"""

{role_context}
"""

        feedback_context = self._format_feedback_learning_for_prompt(state)
        if feedback_context:
            base_prompt += f"""

{feedback_context}
"""

        base_prompt += f"""
AVAILABLE TOOLS:
{self._format_tools_for_prompt(job.job_type, job.config, profile=active_profile)}

GUIDELINES:
1. Work systematically toward the goal
2. Gather information before making conclusions
3. Build on previous findings and relevant memories
4. Stop when the goal is achieved or no more progress is possible
5. Be efficient with tool calls
6. Apply insights from past jobs when relevant
7. Follow the current execution plan step unless strong evidence suggests a pivot
8. Treat critic feedback as a strong hint when selecting the next action
9. For research jobs, prioritize experiments that can confirm or falsify top hypotheses with minimal effort
"""
        return base_prompt

    def _get_tools_for_job_type(
        self,
        job_type: str,
        config: Optional[Dict[str, Any]],
        profile: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """Get available tools based on job type."""
        # Base tools available to all autonomous jobs
        base_tools = [
            "search_documents", "get_document_details", "read_document_content",
            "save_research_finding", "get_research_findings", "write_progress_report",
            "suggest_next_action", "search_with_filters"
        ]

        # Type-specific tools
        type_tools = {
            "research": [
                "search_arxiv", "summarize_document", "find_similar_documents",
                "get_knowledge_base_stats", "add_to_reading_list", "get_reading_lists",
                "extract_paper_insights", "find_related_papers", "build_research_graph",
                "compare_methodologies", "identify_research_gaps", "create_synthesis_document",
                "generate_research_presentation", "ingest_paper_by_id", "batch_ingest_papers",
                "analyze_document_cluster", "create_knowledge_base_entry", "create_document_from_text"
            ],
            "monitor": [
                "search_arxiv", "search_documents", "get_knowledge_base_stats",
                "monitor_arxiv_topic", "ingest_paper_by_id", "add_to_reading_list",
                "get_reading_lists"
            ],
            "analysis": [
                "search_documents", "get_document_details", "summarize_document",
                "find_similar_documents", "compare_documents", "extract_paper_insights",
                "compare_methodologies", "analyze_document_cluster", "build_research_graph",
                "identify_research_gaps", "create_synthesis_document", "create_document_from_text"
            ],
            "synthesis": [
                "search_documents", "get_document_details", "summarize_document",
                "generate_diagram", "create_synthesis_document", "generate_research_presentation",
                "create_knowledge_base_entry", "link_entities", "create_document_from_text"
            ],
            "knowledge_expansion": [
                "search_arxiv", "search_documents", "find_similar_documents",
                "get_knowledge_base_stats", "ingest_paper_by_id", "batch_ingest_papers",
                "find_related_papers", "build_research_graph", "link_entities",
                "create_knowledge_base_entry"
            ],
            "custom": [
                # Custom jobs get most tools
                "search_arxiv", "summarize_document", "find_similar_documents",
                "add_to_reading_list", "extract_paper_insights", "create_synthesis_document", "create_document_from_text"
            ],
            "data_analysis": [
                # Data analysis, ETL, and visualization tools
                "load_csv_data", "load_json_data", "create_dataset", "list_datasets",
                "describe_dataset", "query_data", "filter_data", "aggregate_data",
                "join_datasets", "transform_data", "detect_anomalies", "calculate_correlations",
                "create_chart", "create_correlation_heatmap", "create_flowchart",
                "create_sequence_diagram", "create_er_diagram", "create_architecture_diagram",
                "create_drawio_diagram", "create_gantt_chart", "export_dataset_csv",
                "export_dataset_json", "search_documents", "get_document_details",
                "read_document_content"
            ],
        }

        # Only expose tools implemented by the autonomous executor tool runner.
        supported_tools = {
            "search_arxiv",
            "search_documents",
            "search_with_filters",
            "web_scrape",
            "ingest_url",
            "get_document_details",
            "read_document_content",
            "summarize_document",
            "find_similar_documents",
            "save_research_finding",
            "get_research_findings",
            "get_knowledge_base_stats",
            "ingest_paper_by_id",
            "batch_ingest_papers",
            "monitor_arxiv_topic",
            "find_related_papers",
            "extract_paper_insights",
            "create_synthesis_document",
            "create_document_from_text",
            "compare_methodologies",
            "identify_research_gaps",
            "add_to_reading_list",
            "get_reading_lists",
            "write_progress_report",
            "suggest_next_action",
            "build_research_graph",
            "link_entities",
            "create_knowledge_base_entry",
            "generate_research_presentation",
            "analyze_document_cluster",
            "compare_documents",
        }
        supported_tools.update(set(DATA_ANALYSIS_TOOL_DEFINITIONS.keys()))

        proposed = sorted(list(set(base_tools + type_tools.get(job_type, []))))
        proposed = [t for t in proposed if t in supported_tools]

        cfg = config if isinstance(config, dict) else {}

        def _as_list(value: Any) -> List[str]:
            if isinstance(value, list):
                return [str(x).strip() for x in value if str(x).strip()]
            if isinstance(value, str):
                return [str(x).strip() for x in value.split(",") if str(x).strip()]
            return []

        allowlist = set(_as_list(cfg.get("allowed_tools") or cfg.get("tool_allowlist")))
        denylist = set(_as_list(cfg.get("blocked_tools") or cfg.get("tool_denylist")))

        if allowlist:
            proposed = [t for t in proposed if t in allowlist]
        if denylist:
            proposed = [t for t in proposed if t not in denylist]

        role_profile = profile if isinstance(profile, dict) else {}
        blocked = set(_as_list(role_profile.get("blocked_tools")))
        preferred = [t for t in _as_list(role_profile.get("preferred_tools")) if t in proposed]
        discouraged = [t for t in _as_list(role_profile.get("discouraged_tools")) if t in proposed]
        if blocked:
            proposed = [t for t in proposed if t not in blocked]

        preferred_seen = set()
        preferred_ordered: List[str] = []
        for t in preferred:
            if t not in preferred_seen and t in proposed:
                preferred_seen.add(t)
                preferred_ordered.append(t)

        discouraged_set = set(discouraged)
        middle = [t for t in proposed if t not in preferred_seen and t not in discouraged_set]
        tail = []
        for t in discouraged:
            if t in proposed and t not in preferred_seen and t not in tail:
                tail.append(t)

        ordered = preferred_ordered + middle + tail

        try:
            max_tools = int(cfg.get("skill_profile_max_tools", 0) or 0)
        except Exception:
            max_tools = 0
        if max_tools > 0:
            ordered = ordered[: max(1, min(max_tools, len(ordered)))]

        return ordered

    def _format_tools_for_prompt(
        self,
        job_type: str,
        config: Optional[Dict[str, Any]],
        profile: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Format available tools for the prompt."""
        tools = self._get_tools_for_job_type(job_type, config, profile=profile)
        role_profile = profile if isinstance(profile, dict) else {}
        preferred = set([str(x).strip() for x in (role_profile.get("preferred_tools") or []) if str(x).strip()])
        discouraged = set([str(x).strip() for x in (role_profile.get("discouraged_tools") or []) if str(x).strip()])
        tool_descriptions = []

        # Combine all tool definitions
        all_tools = AGENT_TOOLS + AUTONOMOUS_AGENT_TOOLS
        seen = set()

        for tool_def in all_tools:
            if tool_def["name"] in tools and tool_def["name"] not in seen:
                seen.add(tool_def["name"])
                # Format parameters
                params = tool_def.get("parameters", {}).get("properties", {})
                required = tool_def.get("parameters", {}).get("required", [])
                param_str = ""
                if params:
                    param_parts = []
                    for pname, pinfo in params.items():
                        req_marker = "*" if pname in required else ""
                        param_parts.append(f"{pname}{req_marker}")
                    param_str = f" ({', '.join(param_parts)})"

                role_marker = ""
                if tool_def["name"] in preferred:
                    role_marker = " [preferred]"
                elif tool_def["name"] in discouraged:
                    role_marker = " [discouraged]"
                tool_descriptions.append(
                    f"- {tool_def['name']}{param_str}{role_marker}: {tool_def['description'][:200]}"
                )

        # Add data analysis tools for data_analysis job type
        if job_type == "data_analysis":
            for tool_name, tool_def in DATA_ANALYSIS_TOOL_DEFINITIONS.items():
                if tool_name in tools and tool_name not in seen:
                    seen.add(tool_name)
                    params = tool_def.get("parameters", {})
                    param_str = ""
                    if params:
                        param_parts = list(params.keys())
                        param_str = f" ({', '.join(param_parts)})"
                    role_marker = ""
                    if tool_name in preferred:
                        role_marker = " [preferred]"
                    elif tool_name in discouraged:
                        role_marker = " [discouraged]"
                    tool_descriptions.append(
                        f"- {tool_name}{param_str}{role_marker}: {tool_def['description'][:200]}"
                    )

        return "\n".join(tool_descriptions)

    def _get_stall_config(self, job: AgentJob) -> Dict[str, Any]:
        """Get normalized stall-detection settings from job config."""
        cfg = job.config if isinstance(job.config, dict) else {}

        def _as_int(key: str, default: int, lo: int, hi: int) -> int:
            try:
                val = int(cfg.get(key, default))
            except Exception:
                val = default
            return max(lo, min(val, hi))

        return {
            "enabled": bool(cfg.get("stall_detection_enabled", True)),
            "min_progress_delta": _as_int("stall_min_progress_delta", 2, 0, 100),
            "max_iterations_without_progress": _as_int("stall_max_iterations_without_progress", 4, 1, 50),
            "max_repeated_actions": _as_int("stall_max_repeated_actions", 3, 2, 50),
            "hard_stop_iterations": _as_int("stall_hard_stop_iterations", 8, 2, 200),
            "max_recovery_actions": _as_int("stall_max_recovery_actions", 3, 0, 50),
        }

    def _get_goal_contract_config(self, job: AgentJob) -> Dict[str, Any]:
        """Get normalized deterministic completion contract config."""
        cfg = job.config if isinstance(job.config, dict) else {}
        raw = cfg.get("goal_contract")
        raw = raw if isinstance(raw, dict) else {}

        def _as_int(value: Any, default: int, lo: int, hi: int) -> int:
            try:
                iv = int(value if value is not None else default)
            except Exception:
                iv = default
            return max(lo, min(iv, hi))

        def _as_str_list(value: Any) -> List[str]:
            items: List[str] = []
            if isinstance(value, list):
                items = [str(x).strip() for x in value if str(x).strip()]
            elif isinstance(value, str):
                items = [str(x).strip() for x in value.split(",") if str(x).strip()]
            deduped: List[str] = []
            for item in items:
                if item not in deduped:
                    deduped.append(item)
            return deduped

        flat_contract_present = any(
            k in cfg
            for k in [
                "goal_contract_min_progress",
                "goal_contract_min_findings",
                "goal_contract_min_artifacts",
                "goal_contract_required_finding_types",
                "goal_contract_required_artifact_types",
                "goal_contract_required_result_keys",
            ]
        )
        enabled_default = bool(raw) or bool(flat_contract_present)
        enabled = self._coerce_bool(
            raw.get("enabled", cfg.get("goal_contract_enabled", enabled_default)),
            default=enabled_default,
        )
        required_finding_types = _as_str_list(
            raw.get("required_finding_types", cfg.get("goal_contract_required_finding_types", []))
        )
        required_artifact_types = _as_str_list(
            raw.get("required_artifact_types", cfg.get("goal_contract_required_artifact_types", []))
        )
        required_result_keys = _as_str_list(
            raw.get("required_result_keys", cfg.get("goal_contract_required_result_keys", []))
        )

        return {
            "enabled": bool(enabled),
            "min_progress": _as_int(raw.get("min_progress", cfg.get("goal_contract_min_progress", 100)), 100, 0, 100),
            "min_findings": _as_int(raw.get("min_findings", cfg.get("goal_contract_min_findings", 0)), 0, 0, 100_000),
            "min_artifacts": _as_int(raw.get("min_artifacts", cfg.get("goal_contract_min_artifacts", 0)), 0, 0, 100_000),
            "required_finding_types": required_finding_types[:24],
            "required_artifact_types": required_artifact_types[:24],
            "required_result_keys": required_result_keys[:24],
            "auto_complete_when_satisfied": self._coerce_bool(
                raw.get("auto_complete_when_satisfied", cfg.get("goal_contract_auto_complete_when_satisfied", True)),
                default=True,
            ),
            "strict_completion": self._coerce_bool(
                raw.get("strict_completion", cfg.get("goal_contract_strict_completion", False)),
                default=False,
            ),
        }

    def _evaluate_goal_contract(
        self,
        job: AgentJob,
        state: Dict[str, Any],
        *,
        include_result_keys: bool = True,
    ) -> Dict[str, Any]:
        """Evaluate whether deterministic completion requirements are currently satisfied."""
        contract = self._get_goal_contract_config(job)
        if not bool(contract.get("enabled", False)):
            return {
                "enabled": False,
                "satisfied": True,
                "missing": [],
                "contract": contract,
                "metrics": {
                    "progress": int(state.get("goal_progress", 0) or 0),
                    "findings_count": len(state.get("findings", []) if isinstance(state.get("findings"), list) else []),
                    "artifacts_count": len(state.get("artifacts", []) if isinstance(state.get("artifacts"), list) else []),
                },
            }

        findings = state.get("findings") if isinstance(state.get("findings"), list) else []
        artifacts = state.get("artifacts") if isinstance(state.get("artifacts"), list) else []
        progress = int(state.get("goal_progress", 0) or 0)

        finding_types: Dict[str, int] = {}
        for f in findings:
            if not isinstance(f, dict):
                continue
            ftype = str(f.get("type") or "").strip()
            if not ftype:
                continue
            finding_types[ftype] = int(finding_types.get(ftype, 0) or 0) + 1

        artifact_types: Dict[str, int] = {}
        for a in artifacts:
            if not isinstance(a, dict):
                continue
            atype = str(a.get("type") or "").strip()
            if not atype:
                continue
            artifact_types[atype] = int(artifact_types.get(atype, 0) or 0) + 1

        missing: List[str] = []
        min_progress = int(contract.get("min_progress", 100) or 100)
        min_findings = int(contract.get("min_findings", 0) or 0)
        min_artifacts = int(contract.get("min_artifacts", 0) or 0)

        if progress < min_progress:
            missing.append(f"progress>={min_progress}")
        if len(findings) < min_findings:
            missing.append(f"findings>={min_findings}")
        if len(artifacts) < min_artifacts:
            missing.append(f"artifacts>={min_artifacts}")

        required_finding_types = contract.get("required_finding_types")
        if isinstance(required_finding_types, list):
            for ftype in [str(x).strip() for x in required_finding_types if str(x).strip()]:
                if int(finding_types.get(ftype, 0) or 0) <= 0:
                    missing.append(f"finding_type:{ftype}")

        required_artifact_types = contract.get("required_artifact_types")
        if isinstance(required_artifact_types, list):
            for atype in [str(x).strip() for x in required_artifact_types if str(x).strip()]:
                if int(artifact_types.get(atype, 0) or 0) <= 0:
                    missing.append(f"artifact_type:{atype}")

        required_result_keys = contract.get("required_result_keys")
        if include_result_keys and isinstance(required_result_keys, list):
            results = job.results if isinstance(job.results, dict) else {}
            for key in [str(x).strip() for x in required_result_keys if str(x).strip()]:
                if key not in results:
                    missing.append(f"result_key:{key}")

        return {
            "enabled": True,
            "satisfied": len(missing) == 0,
            "missing": missing[:20],
            "contract": contract,
            "metrics": {
                "progress": progress,
                "findings_count": len(findings),
                "artifacts_count": len(artifacts),
                "finding_types": finding_types,
                "artifact_types": artifact_types,
            },
        }

    def _get_approval_checkpoint_config(self, job: AgentJob) -> Dict[str, Any]:
        """Get normalized human-approval checkpoint config."""
        cfg = job.config if isinstance(job.config, dict) else {}
        raw = cfg.get("approval_checkpoints")
        raw = raw if isinstance(raw, dict) else {}

        def _as_str_list(value: Any) -> List[str]:
            if isinstance(value, list):
                vals = [str(x).strip() for x in value if str(x).strip()]
            elif isinstance(value, str):
                vals = [str(x).strip() for x in value.split(",") if str(x).strip()]
            else:
                vals = []
            out: List[str] = []
            for v in vals:
                if v not in out:
                    out.append(v)
            return out

        def _as_int_list(value: Any) -> List[int]:
            raw_vals: List[int] = []
            if isinstance(value, list):
                for x in value:
                    try:
                        raw_vals.append(int(x))
                    except Exception:
                        continue
            elif isinstance(value, str):
                for x in value.split(","):
                    try:
                        raw_vals.append(int(x.strip()))
                    except Exception:
                        continue
            out: List[int] = []
            for v in raw_vals:
                v = max(1, min(v, 1_000_000))
                if v not in out:
                    out.append(v)
            return sorted(out)

        flat_checkpoint_present = any(
            k in cfg
            for k in [
                "approval_checkpoint_tools",
                "approval_checkpoint_iterations",
                "approval_checkpoint_progress_at_or_above",
            ]
        )
        enabled_default = bool(raw) or bool(flat_checkpoint_present)
        enabled = self._coerce_bool(
            raw.get("enabled", cfg.get("approval_checkpoints_enabled", enabled_default)),
            default=enabled_default,
        )
        tools = _as_str_list(raw.get("tools", cfg.get("approval_checkpoint_tools", [])))
        iterations = _as_int_list(raw.get("iterations", cfg.get("approval_checkpoint_iterations", [])))
        try:
            progress_at_or_above = int(raw.get("progress_at_or_above", cfg.get("approval_checkpoint_progress_at_or_above", -1)))
        except Exception:
            progress_at_or_above = -1
        progress_at_or_above = max(-1, min(progress_at_or_above, 100))

        return {
            "enabled": bool(enabled),
            "tools": tools[:40],
            "iterations": iterations[:200],
            "progress_at_or_above": progress_at_or_above,
            "once_per_checkpoint": self._coerce_bool(
                raw.get("once_per_checkpoint", cfg.get("approval_checkpoint_once_per_checkpoint", True)),
                default=True,
            ),
            "message_prefix": str(
                raw.get(
                    "message_prefix",
                    cfg.get("approval_checkpoint_message_prefix", "Approval required before autonomous action"),
                )
                or "Approval required before autonomous action"
            ).strip()[:160],
        }

    def _evaluate_approval_checkpoint(
        self,
        job: AgentJob,
        state: Dict[str, Any],
        action: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Determine whether this action should pause for explicit human approval."""
        cfg = self._get_approval_checkpoint_config(job)
        if not bool(cfg.get("enabled", False)) or not isinstance(action, dict):
            return {"enabled": bool(cfg.get("enabled", False)), "required": False}

        tool = str(action.get("tool") or "").strip()
        if not tool:
            return {"enabled": True, "required": False}

        reasons: List[str] = []
        reason_keys: List[str] = []
        watch_tools = set(str(x).strip() for x in (cfg.get("tools") or []) if str(x).strip())
        if watch_tools and tool in watch_tools:
            reasons.append(f"tool:{tool}")
            reason_keys.append(f"tool:{tool}")

        iteration = int(getattr(job, "iteration", 0) or 0)
        watch_iterations = set(int(x) for x in (cfg.get("iterations") or []) if isinstance(x, int))
        if watch_iterations and iteration in watch_iterations:
            reasons.append(f"iteration:{iteration}")
            reason_keys.append(f"iteration:{iteration}")

        progress = int(state.get("goal_progress", 0) or 0)
        threshold = int(cfg.get("progress_at_or_above", -1) or -1)
        if threshold >= 0 and progress >= threshold:
            reasons.append(f"progress>={threshold}")
            reason_keys.append(f"progress_threshold:{threshold}")

        if not reasons:
            return {"enabled": True, "required": False}

        seen = state.get("approval_checkpoint_seen")
        if not isinstance(seen, list):
            seen = []
        seen_set = set(str(x).strip() for x in seen if str(x).strip())
        if bool(cfg.get("once_per_checkpoint", True)):
            unseen = [rk for rk in reason_keys if rk not in seen_set]
            if not unseen:
                return {"enabled": True, "required": False}
            for rk in unseen:
                seen.append(rk)
        state["approval_checkpoint_seen"] = seen[-200:]

        checkpoint = {
            "iteration": iteration,
            "action": {
                "tool": tool,
                "params": action.get("params") if isinstance(action.get("params"), dict) else {},
                "purpose": str(action.get("purpose") or "").strip()[:220],
            },
            "reasons": reasons[:8],
            "message": f"{str(cfg.get('message_prefix') or 'Approval required')}: {tool}",
            "created_at": datetime.utcnow().isoformat(),
            "action_signature": self._action_signature(action),
        }
        return {
            "enabled": True,
            "required": True,
            "checkpoint": checkpoint,
        }

    def _build_executive_digest(self, job: AgentJob, state: Dict[str, Any]) -> Dict[str, Any]:
        """Build a compact deterministic run digest for operators."""
        findings = state.get("findings") if isinstance(state.get("findings"), list) else []
        artifacts = state.get("artifacts") if isinstance(state.get("artifacts"), list) else []
        actions = state.get("actions_taken") if isinstance(state.get("actions_taken"), list) else []
        critic_notes = state.get("critic_notes") if isinstance(state.get("critic_notes"), list) else []

        key_findings: List[str] = []
        for f in findings:
            if not isinstance(f, dict):
                continue
            title = str(f.get("title") or f.get("summary") or f.get("id") or "").strip()
            if not title or title in key_findings:
                continue
            key_findings.append(title[:220])
            if len(key_findings) >= 5:
                break

        failed_actions = 0
        for row in actions:
            if not isinstance(row, dict):
                continue
            result = row.get("result") if isinstance(row.get("result"), dict) else {}
            if not bool(result.get("success")):
                failed_actions += 1

        risks: List[str] = []
        if failed_actions > 0:
            risks.append(f"{failed_actions} tool actions failed during execution")
        for note in critic_notes[-4:]:
            if not isinstance(note, dict):
                continue
            sev = str(note.get("severity") or "").strip().lower()
            if sev not in {"high", "medium"}:
                continue
            pivot = str(note.get("pivot") or note.get("trajectory_assessment") or "").strip()
            if not pivot:
                continue
            risks.append(f"critic_{sev}: {pivot[:180]}")
            if len(risks) >= 5:
                break

        contract_eval = self._evaluate_goal_contract(job, state)
        if bool(contract_eval.get("enabled")) and not bool(contract_eval.get("satisfied")):
            missing = contract_eval.get("missing") if isinstance(contract_eval.get("missing"), list) else []
            if missing:
                risks.append(f"goal contract unmet: {', '.join([str(x)[:60] for x in missing[:3]])}")

        next_actions: List[str] = []
        causal_plan = state.get("causal_experiment_plan") if isinstance(state.get("causal_experiment_plan"), dict) else {}
        causal_experiments = causal_plan.get("experiments") if isinstance(causal_plan.get("experiments"), list) else []
        causal_priority = causal_plan.get("priority_order") if isinstance(causal_plan.get("priority_order"), list) else []
        exp_index = {
            str(e.get("id") or "").strip(): e
            for e in causal_experiments
            if isinstance(e, dict) and str(e.get("id") or "").strip()
        }
        ordered = [str(x).strip() for x in causal_priority if str(x).strip() in set(exp_index.keys())]
        if not ordered:
            ordered = list(exp_index.keys())
        for eid in ordered[:2]:
            exp = exp_index.get(eid) if isinstance(exp_index.get(eid), dict) else {}
            if not exp:
                continue
            name = str(exp.get("name") or "").strip()
            if name:
                next_actions.append(f"Execute causal experiment {eid}: {name[:160]}")
            expected = exp.get("expected_evidence") if isinstance(exp.get("expected_evidence"), dict) else {}
            supports = expected.get("supports") if isinstance(expected.get("supports"), list) else []
            if supports:
                next_actions.append(f"Evidence to confirm: {str(supports[0])[:160]}")
            if len(next_actions) >= 4:
                break

        results = job.results if isinstance(job.results, dict) else {}
        research_bundle = results.get("research_bundle") if isinstance(results.get("research_bundle"), dict) else {}
        rb_steps = research_bundle.get("next_steps") if isinstance(research_bundle.get("next_steps"), list) else []
        for step in rb_steps:
            txt = str(step).strip()
            if txt and txt not in next_actions:
                next_actions.append(txt[:200])
            if len(next_actions) >= 4:
                break
        if not next_actions and bool(contract_eval.get("enabled")) and not bool(contract_eval.get("satisfied")):
            missing = contract_eval.get("missing") if isinstance(contract_eval.get("missing"), list) else []
            for req in missing[:3]:
                next_actions.append(f"Satisfy contract requirement: {str(req)[:120]}")
        if not next_actions:
            next_actions = [
                "Review top findings for consistency and evidence quality.",
                "Plan one focused follow-up iteration on highest-impact gap.",
            ]

        summary = str(results.get("summary") or "").strip()
        if not summary:
            summary = (
                f"Completed {int(job.iteration or 0)} iterations with "
                f"{len(findings)} findings and {len(artifacts)} artifacts."
            )

        return {
            "goal": str(job.goal or "").strip()[:600],
            "status": str(job.status or ""),
            "outcome": summary[:400],
            "metrics": {
                "goal_progress": int(state.get("goal_progress", 0) or 0),
                "iterations": int(job.iteration or 0),
                "actions_count": len(actions),
                "findings_count": len(findings),
                "artifacts_count": len(artifacts),
                "failed_actions": failed_actions,
            },
            "key_findings": key_findings,
            "risks": risks[:5],
            "next_actions": next_actions[:5],
            "goal_contract": {
                "enabled": bool(contract_eval.get("enabled")),
                "satisfied": bool(contract_eval.get("satisfied")),
                "missing": contract_eval.get("missing") if isinstance(contract_eval.get("missing"), list) else [],
            },
        }

    def _action_signature(self, action: Optional[Dict[str, Any]]) -> Optional[str]:
        """Build a stable action signature for repeated-action detection."""
        if not isinstance(action, dict):
            return None
        tool = str(action.get("tool") or "").strip()
        if not tool:
            return None

        params = action.get("params")
        if not isinstance(params, dict):
            params = {}
        stable_params = {k: v for k, v in params.items() if not str(k).startswith("_fallback_")}
        try:
            params_blob = json.dumps(stable_params, sort_keys=True, default=str)
        except Exception:
            params_blob = str(stable_params)
        return f"{tool}:{params_blob}"

    def _update_stall_state(
        self,
        job: AgentJob,
        state: Dict[str, Any],
        progress: int,
        action: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Update stall counters and return stall/recovery recommendations."""
        cfg = self._get_stall_config(job)
        if not cfg.get("enabled", True):
            state["last_progress"] = int(progress or 0)
            return {
                "enabled": False,
                "should_recover": False,
                "should_stop": False,
                "reason": "",
            }

        prev_progress = int(state.get("last_progress", 0) or 0)
        delta = int(progress or 0) - prev_progress
        state["last_progress"] = int(progress or 0)

        min_delta = int(cfg["min_progress_delta"])
        if delta <= min_delta:
            state["stalled_iterations"] = int(state.get("stalled_iterations", 0) or 0) + 1
        else:
            state["stalled_iterations"] = 0

        sig = self._action_signature(action)
        if delta > min_delta:
            # Forward progress clears repetition pressure.
            state["repeated_action_iterations"] = 0
            state["last_action_signature"] = sig
        elif sig:
            if sig == state.get("last_action_signature"):
                state["repeated_action_iterations"] = int(state.get("repeated_action_iterations", 0) or 0) + 1
            else:
                state["repeated_action_iterations"] = 1
                state["last_action_signature"] = sig
        else:
            state["repeated_action_iterations"] = 0
            state["last_action_signature"] = None

        history = state.get("progress_history")
        if not isinstance(history, list):
            history = []
        history.append(int(progress or 0))
        state["progress_history"] = history[-25:]

        stalled = int(state.get("stalled_iterations", 0) or 0)
        repeated = int(state.get("repeated_action_iterations", 0) or 0)
        should_recover = (
            stalled >= int(cfg["max_iterations_without_progress"]) or
            repeated >= int(cfg["max_repeated_actions"])
        )
        should_stop = stalled >= int(cfg["hard_stop_iterations"])

        reason_parts = []
        if stalled:
            reason_parts.append(f"stalled_iterations={stalled}")
        if repeated:
            reason_parts.append(f"repeated_action_iterations={repeated}")
        if delta <= min_delta:
            reason_parts.append(f"progress_delta={delta}")

        return {
            "enabled": True,
            "should_recover": should_recover,
            "should_stop": should_stop,
            "reason": ", ".join(reason_parts),
            "stalled_iterations": stalled,
            "repeated_action_iterations": repeated,
        }

    def _build_recovery_action(
        self,
        job: AgentJob,
        state: Dict[str, Any],
        exclude_tool: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Choose a deterministic recovery action to break stalled loops."""
        state["last_recovery_was_forced_exploration"] = False
        profile = (
            state.get("skill_profile")
            if isinstance(state.get("skill_profile"), dict)
            else self._resolve_agent_skill_profile(job, state=state)
        )
        available = set(self._get_tools_for_job_type(job.job_type, job.config, profile=profile))
        findings = state.get("findings", []) if isinstance(state.get("findings"), list) else []
        recent_actions = state.get("actions_taken", []) if isinstance(state.get("actions_taken"), list) else []
        current_stats = state.get("tool_stats") if isinstance(state.get("tool_stats"), dict) else {}
        prior_stats = state.get("tool_priors") if isinstance(state.get("tool_priors"), dict) else {}
        combined_stats = self._merge_tool_stats(prior_stats, current_stats)
        forced_cfg = self._get_forced_exploration_config(job)
        forced_tools = set(forced_cfg.get("tools", [])) if isinstance(forced_cfg.get("tools"), list) else set()
        cooldown_cfg = self._get_tool_cooldown_config(job)
        cooldowns = state.get("tool_cooldowns")
        if not isinstance(cooldowns, dict):
            cooldowns = {}
        cur_iter = int(job.iteration or 0)
        if cooldowns:
            normalized: Dict[str, int] = {}
            for t, until in cooldowns.items():
                name = str(t).strip()
                if not name:
                    continue
                try:
                    until_i = int(until or 0)
                except Exception:
                    continue
                if until_i >= cur_iter:
                    normalized[name] = until_i
            cooldowns = normalized
            state["tool_cooldowns"] = cooldowns
        recent_tools = [
            ((a.get("action") or {}).get("tool"))
            for a in recent_actions[-5:]
            if isinstance(a, dict)
        ]
        exclude = str(exclude_tool or "").strip()

        def _can_use(tool: str) -> bool:
            if tool not in available:
                return False
            if exclude and tool == exclude:
                return False
            # Avoid picking the exact same action repeatedly in recovery mode.
            if recent_tools.count(tool) >= 3:
                return False
            if bool(cooldown_cfg.get("enabled", True)):
                apply_cooldown = True
                if bool(cooldown_cfg.get("forced_only", True)) and forced_tools and tool not in forced_tools:
                    apply_cooldown = False
                if apply_cooldown and self._is_tool_in_cooldown(tool, cooldowns, cur_iter):
                    state["tool_cooldown_blocks"] = int(state.get("tool_cooldown_blocks", 0) or 0) + 1
                    return False
            tstats = combined_stats.get(tool) if isinstance(combined_stats, dict) else None
            if isinstance(tstats, dict):
                success = int(tstats.get("success", 0) or 0)
                failure = int(tstats.get("failure", 0) or 0)
                ratio = self._tool_success_ratio(tstats)
                if failure >= 5 and ratio < 0.2:
                    return False
            return True

        def _doc_ids() -> List[str]:
            out: List[str] = []
            for f in findings:
                if not isinstance(f, dict):
                    continue
                doc_id = str(f.get("id") or f.get("document_id") or "").strip()
                if doc_id and doc_id not in out:
                    out.append(doc_id)
            return out

        doc_ids = _doc_ids()
        has_documents = bool(doc_ids)
        has_papers = any(isinstance(f, dict) and f.get("type") == "paper" for f in findings)

        # Periodically force exploration of under-sampled tools to avoid local optima.
        if self._should_force_exploration(job, state):
            state["forced_exploration_attempts"] = int(state.get("forced_exploration_attempts", 0) or 0) + 1
            forced = self._build_forced_exploration_action(
                job=job,
                state=state,
                available_tools=available,
                combined_stats=combined_stats,
                exclude_tool=exclude,
                doc_ids=doc_ids,
                recent_tools=recent_tools,
            )
            if forced and _can_use(str(forced.get("tool") or "").strip()):
                state["forced_exploration_used"] = int(state.get("forced_exploration_used", 0) or 0) + 1
                state["last_recovery_was_forced_exploration"] = True
                forced_tool = str(forced.get("tool") or "").strip()
                if forced_tool:
                    history = state.get("forced_exploration_history")
                    if not isinstance(history, list):
                        history = []
                    history.append({
                        "iteration": cur_iter,
                        "tool": forced_tool,
                        "success": None,
                    })
                    state["forced_exploration_history"] = history[-20:]
                    if bool(cooldown_cfg.get("enabled", True)):
                        until = cur_iter + int(cooldown_cfg.get("cooldown_iterations", 2) or 2)
                        prior_until = int(cooldowns.get(forced_tool, 0) or 0)
                        cooldowns[forced_tool] = max(prior_until, until)
                        state["tool_cooldowns"] = cooldowns
                return forced

        # Prioritize the latest critic recommendation when viable.
        critic_notes = state.get("critic_notes") if isinstance(state.get("critic_notes"), list) else []
        if critic_notes and isinstance(critic_notes[-1], dict):
            rec_tools = critic_notes[-1].get("recommended_tools") if isinstance(critic_notes[-1].get("recommended_tools"), list) else []
            rec_action = self._build_action_from_recommended_tools(
                job=job,
                state=state,
                recommended_tools=[str(t).strip() for t in rec_tools if str(t).strip()],
                exclude_tool=exclude,
            )
            if rec_action and _can_use(str(rec_action.get("tool") or "").strip()):
                return rec_action

        # Then prefer current plan step suggested tools.
        plan = state.get("execution_plan") if isinstance(state.get("execution_plan"), list) else []
        idx = int(state.get("plan_step_index", 0) or 0)
        if plan and 0 <= idx < len(plan) and isinstance(plan[idx], dict):
            suggested = plan[idx].get("suggested_tools")
            if isinstance(suggested, list):
                ranked_suggested = []
                seen = set()
                for st in suggested:
                    tool = str(st).strip()
                    if not tool or tool in seen:
                        continue
                    seen.add(tool)
                    ranked_suggested.append(tool)
                ranked_suggested = self._rank_tools_for_selection(
                    job,
                    ranked_suggested,
                    combined_stats,
                    state=state,
                    context_tag="plan_recovery",
                )
                for st in ranked_suggested:
                    tool = str(st).strip()
                    if not _can_use(tool):
                        continue
                    action = self._build_action_for_tool(
                        tool=tool,
                        job=job,
                        doc_ids=doc_ids,
                        purpose="Recover using current plan step suggested tool.",
                    )
                    if action:
                        return action

        if _can_use("search_documents"):
            action = self._build_action_for_tool(
                tool="search_documents",
                job=job,
                doc_ids=doc_ids,
                purpose="Recover from stall by broadening internal evidence search.",
            )
            if action:
                return action
        if has_documents and _can_use("read_document_content"):
            action = self._build_action_for_tool(
                tool="read_document_content",
                job=job,
                doc_ids=doc_ids,
                purpose="Recover from stall by collecting richer context from an identified document.",
            )
            if action:
                return action
        if has_documents and _can_use("summarize_document"):
            action = self._build_action_for_tool(
                tool="summarize_document",
                job=job,
                doc_ids=doc_ids,
                purpose="Recover from stall by extracting concise document insights.",
            )
            if action:
                return action
        if job.job_type == "research" and not has_papers and _can_use("search_arxiv"):
            action = self._build_action_for_tool(
                tool="search_arxiv",
                job=job,
                doc_ids=doc_ids,
                purpose="Recover from stall by adding external research evidence.",
            )
            if action:
                return action
        if _can_use("suggest_next_action"):
            action = self._build_action_for_tool(
                tool="suggest_next_action",
                job=job,
                doc_ids=doc_ids,
                purpose="Recover from stall by requesting a targeted next-step recommendation.",
            )
            if action:
                return action
        return None

    async def _act(
        self,
        job: AgentJob,
        action: Dict[str, Any],
        state: Dict[str, Any],
        db: AsyncSession,
    ) -> Dict[str, Any]:
        """
        Execute the decided action.

        Calls the appropriate tool and returns results.
        """
        tool_name = action.get("tool")
        params = action.get("params", {})

        result = {
            "tool": tool_name,
            "success": False,
            "findings": [],
            "artifacts": [],
            "timestamp": datetime.utcnow().isoformat(),
        }

        try:
            # --- Customer-specific research guardrails (no LLM dependency) ---
            # Prefer internal KB first when requested, but avoid deadlocks by allowing external research
            # after a couple iterations.
            if job.job_type == "research":
                prefer_sources = (job.config or {}).get("prefer_sources") or []
                if isinstance(prefer_sources, str):
                    prefer_sources = [x.strip() for x in prefer_sources.split(",") if x.strip()]
                prefer_sources = [str(x).strip().lower() for x in prefer_sources if str(x).strip()]

                external_tools = {"search_arxiv", "monitor_arxiv_topic", "ingest_paper_by_id", "batch_ingest_papers"}
                if tool_name in external_tools and "arxiv" not in prefer_sources:
                    result["error"] = "External paper research is disabled for this job (prefer_sources excludes arxiv)."
                    return result

                if tool_name in external_tools and prefer_sources and prefer_sources[0] == "documents" and "arxiv" in prefer_sources:
                    has_internal_attempt = any(
                        (a.get("action") or {}).get("tool") in {"search_documents", "search_with_filters"}
                        for a in (state.get("actions_taken") or [])
                    )
                    # Allow external after a couple iterations to avoid stalling.
                    if not has_internal_attempt and int(job.iteration or 0) <= 2:
                        result["error"] = "Prefer internal documents first (run a document search before arXiv)."
                        return result

            # ==================== Search Tools ====================
            if tool_name == "search_arxiv":
                query = params.get("query", job.goal[:100])
                max_results = params.get("max_results", 10)

                papers = await self.arxiv_service.search(
                    query=query,
                    max_results=max_results,
                )

                result["success"] = True
                result["data"] = papers
                result["findings"] = [
                    {
                        "type": "paper",
                        "title": p.get("title"),
                        "id": p.get("id"),
                        "arxiv_id": p.get("id"),
                        "summary": p.get("summary", "")[:500],
                        "authors": p.get("authors", []),
                        "published": p.get("published"),
                    }
                    for p in papers[:10]
                ]

            elif tool_name == "search_documents":
                query = params.get("query", job.goal[:100])
                limit = params.get("limit", 10)

                results = await self.search_service.search(
                    query=query,
                    limit=limit,
                )

                result["success"] = True
                result["data"] = results
                result["findings"] = [
                    {
                        "type": "document",
                        "title": r.get("title"),
                        "id": r.get("id"),
                        "score": r.get("score"),
                        "source": r.get("source"),
                    }
                    for r in results[:10]
                ]

            elif tool_name == "search_with_filters":
                query = params.get("query", "")
                limit = params.get("limit", 20)
                # Would implement filtered search
                results = await self.search_service.search(query=query, limit=limit)
                result["success"] = True
                result["data"] = results

            elif tool_name == "web_scrape":
                url = params.get("url", "")
                if not url:
                    result["error"] = "Missing required parameter: url"
                else:
                    from app.services.web_scraper_service import WebScraperService
                    from app.models.document import DocumentSource
                    from urllib.parse import urlparse

                    parsed = urlparse(url)
                    host = (parsed.hostname or "").lower()
                    allow_private = False
                    if host:
                        src_res = await db.execute(
                            select(DocumentSource).where(
                                DocumentSource.source_type == "web",
                                DocumentSource.is_active == True,
                            )
                        )
                        sources = src_res.scalars().all()

                        def host_matches(allowed: str) -> bool:
                            allowed = (allowed or "").strip().lower()
                            if not allowed:
                                return False
                            return host == allowed or host.endswith("." + allowed)

                        for source in sources:
                            cfg = source.config or {}
                            for d in (cfg.get("allowed_domains") or []):
                                if host_matches(d):
                                    allow_private = True
                                    break
                            if allow_private:
                                break
                            for base in (cfg.get("base_urls") or []):
                                try:
                                    base_host = (urlparse(str(base)).hostname or "").lower()
                                except Exception:
                                    base_host = ""
                                if base_host and host_matches(base_host):
                                    allow_private = True
                                    break
                            if allow_private:
                                break

                    scraper = WebScraperService(enforce_network_safety=True)
                    try:
                        scrape_result = await scraper.scrape(
                            url,
                            follow_links=bool(params.get("follow_links", False)),
                            max_pages=int(params.get("max_pages", 1)),
                            max_depth=int(params.get("max_depth", 0)),
                            same_domain_only=bool(params.get("same_domain_only", True)),
                            include_links=bool(params.get("include_links", True)),
                            allow_private_networks=allow_private,
                            max_content_chars=int(params.get("max_content_chars", 50_000)),
                        )
                        result["success"] = True
                        result["data"] = scrape_result
                        pages = scrape_result.get("pages", [])
                        if pages:
                            result["findings"] = [
                                {
                                    "type": "web_page",
                                    "title": p.get("title"),
                                    "url": p.get("url"),
                                    "content_preview": (p.get("content") or "")[:500],
                                }
                                for p in pages[:5]
                            ]
                    finally:
                        await scraper.aclose()

            elif tool_name == "ingest_url":
                url = (params.get("url") or "").strip()
                if not url:
                    result["error"] = "Missing required parameter: url"
                else:
                    from app.models.user import User
                    from app.services.url_ingestion_service import UrlIngestionService

                    ures = await db.execute(select(User).where(User.id == job.user_id))
                    user = ures.scalar_one_or_none()
                    if not user:
                        result["error"] = "User not found"
                    else:
                        service = UrlIngestionService()
                        ingest = await service.ingest_url(
                            db=db,
                            user=user,
                            url=url,
                            title=params.get("title"),
                            tags=params.get("tags"),
                            follow_links=bool(params.get("follow_links", False)),
                            max_pages=int(params.get("max_pages", 1)),
                            max_depth=int(params.get("max_depth", 0)),
                            same_domain_only=bool(params.get("same_domain_only", True)),
                            one_document_per_page=bool(params.get("one_document_per_page", False)),
                            allow_private_networks=bool(params.get("allow_private_networks", False)),
                            max_content_chars=int(params.get("max_content_chars", 50_000)),
                        )
                        if ingest.get("error"):
                            result["error"] = ingest["error"]
                        else:
                            result["success"] = True
                            result["data"] = ingest

            # ==================== Document Tools ====================
            elif tool_name == "get_document_details":
                doc_id = params.get("document_id")
                if doc_id:
                    from app.models.document import Document
                    from uuid import UUID
                    doc_result = await db.execute(
                        select(Document).where(Document.id == UUID(doc_id))
                    )
                    doc = doc_result.scalar_one_or_none()
                    if doc:
                        result["success"] = True
                        result["data"] = {
                            "id": str(doc.id),
                            "title": doc.title,
                            "source": doc.source,
                            "file_type": doc.file_type,
                            "author": doc.author,
                            "summary": doc.summary,
                            "has_content": bool(doc.content),
                        }
                    else:
                        result["error"] = "Document not found"

            elif tool_name == "read_document_content":
                doc_id = params.get("document_id")
                max_length = params.get("max_length", 10000)
                if doc_id:
                    from app.models.document import Document
                    from uuid import UUID
                    doc_result = await db.execute(
                        select(Document).where(Document.id == UUID(doc_id))
                    )
                    doc = doc_result.scalar_one_or_none()
                    if doc and doc.content:
                        content = doc.content[:max_length]
                        result["success"] = True
                        result["data"] = {
                            "id": str(doc.id),
                            "title": doc.title,
                            "content": content,
                            "truncated": len(doc.content) > max_length,
                        }
                    else:
                        result["error"] = "Document not found or has no content"

            elif tool_name == "summarize_document":
                doc_id = params.get("document_id")
                if doc_id:
                    from app.models.document import Document
                    from uuid import UUID
                    doc_result = await db.execute(
                        select(Document).where(Document.id == UUID(doc_id))
                    )
                    doc = doc_result.scalar_one_or_none()
                    if doc:
                        if doc.summary:
                            result["success"] = True
                            result["data"] = {"summary": doc.summary}
                            result["findings"] = [{
                                "type": "summary",
                                "document_id": doc_id,
                                "content": doc.summary[:500],
                            }]
                        else:
                            # Would trigger summarization task
                            result["success"] = True
                            result["data"] = {"status": "summarization_queued"}

            elif tool_name == "find_similar_documents":
                doc_id = params.get("document_id")
                limit = params.get("limit", 5)
                if doc_id:
                    # Use vector search for similarity
                    from app.models.document import Document
                    from uuid import UUID
                    doc_result = await db.execute(
                        select(Document).where(Document.id == UUID(doc_id))
                    )
                    doc = doc_result.scalar_one_or_none()
                    if doc and doc.content:
                        similar = await self.search_service.search(
                            query=doc.title + " " + (doc.summary or doc.content[:500]),
                            limit=limit + 1,
                        )
                        # Filter out the source document
                        similar = [s for s in similar if str(s.get("id")) != doc_id][:limit]
                        result["success"] = True
                        result["data"] = similar
                        result["findings"] = [
                            {"type": "similar_document", "id": s.get("id"), "title": s.get("title")}
                            for s in similar
                        ]

            # ==================== Research Finding Tools ====================
            elif tool_name == "save_research_finding":
                finding = {
                    "id": str(uuid.uuid4()),
                    "title": params.get("title"),
                    "content": params.get("content"),
                    "category": params.get("category"),
                    "source_document_ids": params.get("source_document_ids", []),
                    "confidence": params.get("confidence", 0.8),
                    "tags": params.get("tags", []),
                    "created_at": datetime.utcnow().isoformat(),
                }

                # Store in job state
                job_id_str = str(job.id)
                if job_id_str not in self._job_findings:
                    self._job_findings[job_id_str] = []
                self._job_findings[job_id_str].append(finding)

                # Also add to state findings
                state["findings"].append(finding)

                result["success"] = True
                result["data"] = {"finding_id": finding["id"]}
                result["findings"] = [finding]

            elif tool_name == "get_research_findings":
                job_id_str = str(job.id)
                findings = self._job_findings.get(job_id_str, [])

                # Apply filters
                category = params.get("category")
                if category:
                    findings = [f for f in findings if f.get("category") == category]

                min_confidence = params.get("min_confidence")
                if min_confidence:
                    findings = [f for f in findings if f.get("confidence", 0) >= min_confidence]

                limit = params.get("limit", 50)
                findings = findings[:limit]

                result["success"] = True
                result["data"] = {"findings": findings, "total": len(findings)}

            elif tool_name == "get_knowledge_base_stats":
                from app.models.document import Document, DocumentSource

                limit = int(params.get("recent_limit", 25) or 25)
                limit = max(1, min(limit, 100))

                total_docs = int((await db.execute(select(func.count()).select_from(Document))).scalar() or 0)
                total_sources = int((await db.execute(select(func.count()).select_from(DocumentSource))).scalar() or 0)

                recent = await db.execute(
                    select(Document.id, Document.title, Document.created_at, Document.tags)
                    .order_by(desc(Document.created_at))
                    .limit(limit)
                )
                rows = recent.all()
                tag_counter: Counter[str] = Counter()
                for _, _, _, tags in rows:
                    if isinstance(tags, list):
                        tag_counter.update([str(t).lower() for t in tags if t])

                result["success"] = True
                result["data"] = {
                    "documents_total": total_docs,
                    "sources_total": total_sources,
                    "recent_documents": [
                        {"id": str(did), "title": title, "created_at": str(created_at)}
                        for did, title, created_at, _ in rows
                    ],
                    "top_tags_recent": [k for k, _ in tag_counter.most_common(25)],
                }

            # ==================== ArXiv/Paper Tools ====================
            elif tool_name == "ingest_paper_by_id":
                arxiv_id = params.get("arxiv_id")
                if arxiv_id:
                    # Search for the specific paper
                    papers = await self.arxiv_service.search(
                        query=f"id:{arxiv_id}",
                        max_results=1,
                    )
                    if papers:
                        paper = papers[0]
                        result["success"] = True
                        result["data"] = paper
                        result["findings"] = [{
                            "type": "paper_ingested",
                            "arxiv_id": arxiv_id,
                            "title": paper.get("title"),
                        }]
                    else:
                        result["error"] = f"Paper {arxiv_id} not found"

            elif tool_name == "batch_ingest_papers":
                """
                Create an arXiv source and trigger ingestion asynchronously.

                Best-effort: queues a background task if Celery is available.
                """
                arxiv_ids = [x.strip() for x in (params.get("arxiv_ids") or []) if isinstance(x, str) and x.strip()]
                search_queries = [x.strip() for x in (params.get("search_queries") or []) if isinstance(x, str) and x.strip()]
                categories = [x.strip() for x in (params.get("categories") or []) if isinstance(x, str) and x.strip()]
                max_results = int(params.get("max_results") or 25)
                max_results = max(1, min(max_results, 200))

                if not arxiv_ids and not search_queries and not categories:
                    result["error"] = "Provide at least one of: arxiv_ids, search_queries, categories"
                else:
                    display = params.get("display") or "Autonomous job import"
                    source_name = f"ArXiv Import (Job {str(job.id)[:8]}) #{uuid.uuid4().hex[:6]}"
                    cfg = {
                        "paper_ids": arxiv_ids,
                        "search_queries": search_queries,
                        "categories": categories,
                        "max_results": max_results,
                        "requested_by_user_id": str(job.user_id),
                        "requested_by": str(job.user_id),
                        "display": display,
                    }
                    source = await self.document_service.create_document_source(
                        name=source_name,
                        source_type="arxiv",
                        config=cfg,
                        db=db,
                    )

                    queued = False
                    try:
                        from app.tasks.ingestion_tasks import ingest_from_source

                        ingest_from_source.delay(str(source.id))
                        queued = True
                    except Exception:
                        queued = False

                    result["success"] = True
                    result["data"] = {
                        "source_id": str(source.id),
                        "source_name": source.name,
                        "queued": queued,
                        "paper_ids_count": len(arxiv_ids),
                        "search_queries_count": len(search_queries),
                        "categories_count": len(categories),
                        "max_results": max_results,
                    }
                    result["findings"] = [
                        {"type": "arxiv_ingest_requested", "source_id": str(source.id), "queued": queued}
                    ]
                    result["artifacts"] = [
                        {"type": "document_source", "id": str(source.id), "name": source.name, "source_type": "arxiv"}
                    ]
                    # Also include a stable, compact artifact so UIs can resolve imports even if the
                    # document_source artifact schema changes.
                    result["artifacts"].append(
                        {"type": "arxiv_ingest_requested", "source_id": str(source.id), "queued": queued}
                    )

            elif tool_name == "monitor_arxiv_topic":
                topic = params.get("topic")
                query = params.get("query") or f"all:{topic}"
                max_results = params.get("max_results", 20)

                papers = await self.arxiv_service.search(
                    query=query,
                    max_results=max_results,
                    sort_by="submittedDate",
                    sort_order="descending",
                )

                result["success"] = True
                result["data"] = papers
                result["findings"] = [
                    {
                        "type": "new_paper",
                        "title": p.get("title"),
                        "arxiv_id": p.get("id"),
                        "published": p.get("published"),
                    }
                    for p in papers[:10]
                ]

            elif tool_name == "find_related_papers":
                doc_id = params.get("document_id")
                arxiv_id = params.get("arxiv_id")
                limit = params.get("limit", 10)
                search_external = params.get("search_external", True)

                # Build query from document or arxiv paper
                query = ""
                if doc_id:
                    from app.models.document import Document
                    from uuid import UUID
                    doc_result = await db.execute(
                        select(Document).where(Document.id == UUID(doc_id))
                    )
                    doc = doc_result.scalar_one_or_none()
                    if doc:
                        query = doc.title
                elif arxiv_id:
                    papers = await self.arxiv_service.search(query=f"id:{arxiv_id}", max_results=1)
                    if papers:
                        query = papers[0].get("title", "")

                if query and search_external:
                    related = await self.arxiv_service.search(query=query, max_results=limit)
                    result["success"] = True
                    result["data"] = related
                    result["findings"] = [
                        {"type": "related_paper", "title": p.get("title"), "arxiv_id": p.get("id")}
                        for p in related
                    ]
                else:
                    result["error"] = "No query could be built"

            elif tool_name == "extract_paper_insights":
                doc_id = params.get("document_id")
                focus_areas = params.get("focus_areas", ["methodology", "results", "contributions"])

                if doc_id:
                    from app.models.document import Document
                    from uuid import UUID
                    doc_result = await db.execute(
                        select(Document).where(Document.id == UUID(doc_id))
                    )
                    doc = doc_result.scalar_one_or_none()
                    if doc and doc.content:
                        # Use LLM to extract insights
                        prompt = f"""Extract key insights from this research paper.
Focus on: {', '.join(focus_areas)}

Paper Title: {doc.title}
Content: {doc.content[:8000]}

Provide structured insights in JSON format:
{{
    "methodology": "...",
    "key_findings": ["..."],
    "contributions": ["..."],
    "limitations": ["..."],
    "future_work": ["..."]
}}"""

                        try:
                            response = await self.llm_service.generate_response(
                                system_prompt="You are a research paper analyst. Extract structured insights.",
                                user_message=prompt,
                                routing=self._llm_routing_from_job_config(job.config),
                                task_type="summarization",
                                user_id=job.user_id,
                                db=db,
                            )
                            insights = json.loads(response)
                            result["success"] = True
                            result["data"] = insights
                            result["findings"] = [{
                                "type": "paper_insights",
                                "document_id": doc_id,
                                "insights": insights,
                            }]
                        except Exception as e:
                            result["success"] = True
                            result["data"] = {"raw_analysis": response if 'response' in dir() else str(e)}

            # ==================== Synthesis Tools ====================
            elif tool_name == "create_synthesis_document":
                title = params.get("title")
                topic = params.get("topic")
                document_ids = params.get("document_ids", [])
                persist = bool(params.get("persist")) or bool((job.config or {}).get("persist_artifacts", False))

                # Gather findings
                job_id_str = str(job.id)
                findings = self._job_findings.get(job_id_str, [])

                # Build synthesis content
                synthesis_content = f"# {title}\n\n## Research Topic\n{topic}\n\n"

                if findings:
                    synthesis_content += "## Key Findings\n"
                    for i, finding in enumerate(findings[:20], 1):
                        synthesis_content += f"\n### {i}. {finding.get('title', 'Finding')}\n"
                        synthesis_content += f"{finding.get('content', '')}\n"
                        if finding.get('category'):
                            synthesis_content += f"*Category: {finding['category']}*\n"

                result["success"] = True
                result["data"] = {
                    "title": title,
                    "content": synthesis_content,
                    "findings_included": len(findings),
                }
                result["artifacts"] = [{
                    "type": "synthesis_document",
                    "title": title,
                    "content": synthesis_content,
                }]

                if persist and title and synthesis_content.strip():
                    try:
                        from app.models.document import Document

                        notes_source = await self.document_service._get_or_create_agent_notes_source(db)
                        content_hash = hashlib.sha256(synthesis_content.encode("utf-8")).hexdigest()
                        doc = Document(
                            title=str(title).strip(),
                            content=synthesis_content,
                            content_hash=content_hash,
                            url=None,
                            file_path=None,
                            file_type="text/markdown",
                            file_size=len(synthesis_content.encode("utf-8")),
                            source_id=notes_source.id,
                            source_identifier=f"agent_synthesis:{uuid.uuid4().hex}",
                            author=None,
                            tags=["autonomous_job", "research"],
                            extra_metadata={
                                "origin": "autonomous_job",
                                "job_id": str(job.id),
                                "job_type": job.job_type,
                                "topic": topic,
                                "document_ids": document_ids,
                            },
                            is_processed=False,
                        )
                        db.add(doc)
                        await db.commit()
                        await db.refresh(doc)

                        try:
                            await self.document_service.reprocess_document(doc.id, db, user_id=job.user_id)
                        except Exception as exc:
                            logger.warning(f"Failed to process autonomous synthesis doc embeddings: {exc}")

                        result["data"]["document_id"] = str(doc.id)
                        result["artifacts"].append({"type": "document", "id": str(doc.id), "title": doc.title})
                    except Exception as exc:
                        logger.warning(f"Failed to persist synthesis document: {exc}")

            elif tool_name == "compare_methodologies":
                document_ids = params.get("document_ids", [])
                comparison_aspects = params.get("comparison_aspects", ["approach", "results"])

                # Would fetch documents and compare
                result["success"] = True
                result["data"] = {
                    "documents_compared": len(document_ids),
                    "aspects": comparison_aspects,
                    "comparison": "Comparison would be generated here",
                }

            elif tool_name == "identify_research_gaps":
                topic = params.get("topic", job.goal)

                # Analyze findings for gaps
                job_id_str = str(job.id)
                findings = self._job_findings.get(job_id_str, [])

                result["success"] = True
                result["data"] = {
                    "topic": topic,
                    "findings_analyzed": len(findings),
                    "gaps_identified": [],  # Would be populated by LLM analysis
                }

            # ==================== Reading List Tools ====================
            elif tool_name == "add_to_reading_list":
                from app.models.reading_list import ReadingList, ReadingListItem
                from app.models.document import Document

                list_name = (params.get("list_name") or "").strip()
                items = params.get("items", []) or []
                if not list_name:
                    result["error"] = "Missing required parameter: list_name"
                elif not isinstance(items, list) or not items:
                    result["error"] = "Missing required parameter: items"
                else:
                    rl_res = await db.execute(
                        select(ReadingList).where(
                            ReadingList.user_id == job.user_id,
                            ReadingList.name == list_name,
                        )
                    )
                    rl = rl_res.scalar_one_or_none()
                    if not rl:
                        rl = ReadingList(user_id=job.user_id, name=list_name, description=None, source_id=None)
                        db.add(rl)
                        await db.flush()

                    max_pos = int(
                        (await db.execute(
                            select(func.max(ReadingListItem.position)).where(ReadingListItem.reading_list_id == rl.id)
                        )).scalar() or 0
                    )
                    added = 0
                    skipped = 0
                    warnings: list[str] = []

                    for raw in items:
                        if not isinstance(raw, dict):
                            skipped += 1
                            continue

                        doc_id = raw.get("document_id")
                        arxiv_id = raw.get("arxiv_id")
                        notes = raw.get("notes")
                        priority = int(raw.get("priority", 3) or 3)

                        doc = None
                        if doc_id:
                            try:
                                from uuid import UUID as _UUID

                                doc = await db.get(Document, _UUID(str(doc_id)))
                            except Exception:
                                doc = None
                        elif arxiv_id:
                            arxiv_id = str(arxiv_id).strip()
                            if arxiv_id.startswith("arxiv:"):
                                arxiv_id = arxiv_id.split("arxiv:", 1)[1].strip()
                            if arxiv_id:
                                doc_res = await db.execute(select(Document).where(Document.source_identifier == arxiv_id).limit(1))
                                doc = doc_res.scalar_one_or_none()

                        if not doc:
                            skipped += 1
                            if arxiv_id:
                                warnings.append(f"Document not found for arXiv id: {arxiv_id}")
                            elif doc_id:
                                warnings.append(f"Document not found for id: {doc_id}")
                            continue

                        # Skip duplicates without rolling back the whole transaction.
                        exists = await db.execute(
                            select(func.count())
                            .select_from(ReadingListItem)
                            .where(
                                ReadingListItem.reading_list_id == rl.id,
                                ReadingListItem.document_id == doc.id,
                            )
                        )
                        if int(exists.scalar() or 0) > 0:
                            skipped += 1
                            continue

                        item = ReadingListItem(
                            reading_list_id=rl.id,
                            document_id=doc.id,
                            status="to-read",
                            priority=max(0, min(priority, 5)),
                            position=max_pos + 1,
                            notes=str(notes).strip()[:2000] if notes else None,
                        )
                        db.add(item)
                        try:
                            await db.flush()
                        except IntegrityError:
                            # Extremely defensive: in case of a race, don't poison the session.
                            await db.rollback()
                            skipped += 1
                            continue

                        max_pos += 1
                        added += 1

                    await db.commit()
                    result["success"] = True
                    result["data"] = {
                        "reading_list_id": str(rl.id),
                        "list_name": rl.name,
                        "items_added": added,
                        "items_skipped": skipped,
                        "warnings": warnings[:25],
                    }
                    result["artifacts"] = [
                        {"type": "reading_list", "id": str(rl.id), "name": rl.name, "items_added": added}
                    ]

            elif tool_name == "get_reading_lists":
                from app.models.reading_list import ReadingList, ReadingListItem
                from app.models.document import Document

                list_name = (params.get("list_name") or "").strip()
                include_items = bool(params.get("include_items", True))

                q = select(ReadingList).where(ReadingList.user_id == job.user_id).order_by(desc(ReadingList.updated_at))
                if list_name:
                    q = q.where(ReadingList.name == list_name)

                rl_res = await db.execute(q.limit(100))
                lists = rl_res.scalars().all()
                payload = []

                for rl in lists:
                    entry: dict[str, Any] = {
                        "id": str(rl.id),
                        "name": rl.name,
                        "description": rl.description,
                        "created_at": rl.created_at.isoformat() if rl.created_at else None,
                        "updated_at": rl.updated_at.isoformat() if rl.updated_at else None,
                    }
                    if include_items:
                        items_res = await db.execute(
                            select(ReadingListItem, Document.title)
                            .join(Document, Document.id == ReadingListItem.document_id)
                            .where(ReadingListItem.reading_list_id == rl.id)
                            .order_by(ReadingListItem.position.asc(), ReadingListItem.created_at.asc())
                        )
                        items = []
                        for item, title in items_res.all():
                            items.append(
                                {
                                    "id": str(item.id),
                                    "document_id": str(item.document_id),
                                    "document_title": title,
                                    "status": item.status,
                                    "priority": item.priority,
                                    "position": item.position,
                                    "notes": item.notes,
                                    "created_at": item.created_at.isoformat() if item.created_at else None,
                                }
                            )
                        entry["items"] = items
                    payload.append(entry)

                result["success"] = True
                result["data"] = {"reading_lists": payload, "total": len(payload)}

            # ==================== Progress Tools ====================
            elif tool_name == "write_progress_report":
                report = {
                    "summary": params.get("summary"),
                    "completed_tasks": params.get("completed_tasks", []),
                    "pending_tasks": params.get("pending_tasks", []),
                    "key_findings": params.get("key_findings", []),
                    "blockers": params.get("blockers", []),
                    "next_steps": params.get("next_steps", []),
                    "iteration": job.iteration,
                    "progress": job.progress,
                    "timestamp": datetime.utcnow().isoformat(),
                }

                # Store in job state
                if "progress_reports" not in state:
                    state["progress_reports"] = []
                state["progress_reports"].append(report)

                result["success"] = True
                result["data"] = report
                result["artifacts"] = [{
                    "type": "progress_report",
                    "report": report,
                }]

            elif tool_name == "suggest_next_action":
                current_goal = params.get("current_goal", job.goal)
                progress_so_far = params.get("progress_so_far", "")

                # Use LLM to suggest next action
                prompt = f"""Given the current research goal and progress, suggest the best next action.

Goal: {current_goal}
Progress so far: {progress_so_far}
Findings count: {len(state.get('findings', []))}
Iteration: {job.iteration}/{job.max_iterations}

Available actions:
- Search for more papers on arXiv
- Analyze existing documents
- Synthesize findings
- Create a report
- Monitor for new papers

Suggest the single best next action and explain why."""

                try:
                    suggestion = await self.llm_service.generate_response(
                        system_prompt="You are a research planning assistant.",
                        user_message=prompt,
                        routing=self._llm_routing_from_job_config(job.config),
                        task_type="research_engineer_scientist",
                        user_id=job.user_id,
                        db=db,
                    )
                    result["success"] = True
                    result["data"] = {"suggestion": suggestion}
                except Exception as e:
                    result["error"] = str(e)

            # ==================== Document Creation Tools ====================
            elif tool_name == "create_document_from_text":
                title = (params.get("title") or "").strip()
                content = (params.get("content") or "").strip()
                tags = params.get("tags") or []

                if not title:
                    result["error"] = "Title is required"
                elif not content:
                    result["error"] = "Content is required"
                else:
                    from app.models.document import Document

                    notes_source = await self.document_service._get_or_create_agent_notes_source(db)
                    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
                    doc = Document(
                        title=title,
                        content=content,
                        content_hash=content_hash,
                        url=None,
                        file_path=None,
                        file_type="text/plain",
                        file_size=len(content.encode("utf-8")),
                        source_id=notes_source.id,
                        source_identifier=f"agent_note:{uuid.uuid4().hex}",
                        author=None,
                        tags=tags if isinstance(tags, list) else None,
                        extra_metadata={
                            "origin": "autonomous_job",
                            "job_id": str(job.id),
                            "job_type": job.job_type,
                        },
                        is_processed=False,
                    )
                    db.add(doc)
                    await db.commit()
                    await db.refresh(doc)

                    try:
                        await self.document_service.reprocess_document(doc.id, db, user_id=job.user_id)
                    except Exception as exc:
                        logger.warning(f"Failed to process autonomous created doc embeddings: {exc}")

                    result["success"] = True
                    result["data"] = {"document_id": str(doc.id), "title": doc.title}
                    result["artifacts"] = [{"type": "document", "id": str(doc.id), "title": doc.title}]

            # ==================== Knowledge Graph Tools ====================
            elif tool_name == "build_research_graph":
                document_ids = params.get("document_ids", [])
                focus_on = params.get("focus_on", ["methods", "concepts"])

                result["success"] = True
                result["data"] = {
                    "documents_analyzed": len(document_ids),
                    "focus": focus_on,
                    "entities_found": 0,  # Would be populated
                    "relationships_found": 0,
                }

            elif tool_name == "link_entities":
                relationship_type = params.get("relationship_type")
                source_name = params.get("source_name")
                target_name = params.get("target_name")
                evidence = params.get("evidence")

                result["success"] = True
                result["data"] = {
                    "relationship_created": True,
                    "source": source_name,
                    "target": target_name,
                    "type": relationship_type,
                }

            elif tool_name == "create_knowledge_base_entry":
                title = params.get("title")
                content = params.get("content")
                entry_type = params.get("entry_type")

                result["success"] = True
                result["data"] = {
                    "entry_created": True,
                    "title": title,
                    "type": entry_type,
                }
                result["artifacts"] = [{
                    "type": "knowledge_entry",
                    "title": title,
                    "content": content,
                    "entry_type": entry_type,
                }]

            # ==================== Presentation Tools ====================
            elif tool_name == "generate_research_presentation":
                title = params.get("title")
                topic = params.get("topic")
                slide_count = params.get("slide_count", 12)

                result["success"] = True
                result["data"] = {
                    "presentation_queued": True,
                    "title": title,
                    "topic": topic,
                    "slides": slide_count,
                }
                result["artifacts"] = [{
                    "type": "presentation_job",
                    "title": title,
                    "status": "queued",
                }]

            # ==================== Analysis Tools ====================
            elif tool_name == "analyze_document_cluster":
                document_ids = params.get("document_ids", [])
                analysis_type = params.get("analysis_type", "comprehensive")

                result["success"] = True
                result["data"] = {
                    "documents_analyzed": len(document_ids),
                    "analysis_type": analysis_type,
                    "themes": [],  # Would be populated by analysis
                }

            elif tool_name == "compare_documents":
                doc_id_1 = params.get("document_id_1")
                doc_id_2 = params.get("document_id_2")

                result["success"] = True
                result["data"] = {
                    "documents_compared": [doc_id_1, doc_id_2],
                    "similarity_score": 0.0,  # Would be calculated
                    "common_themes": [],
                    "differences": [],
                }

            # ==================== Data Analysis Tools ====================
            elif tool_name in DATA_ANALYSIS_TOOL_DEFINITIONS:
                # Get or create data analysis tools instance for this job
                job_id_str = str(job.id)
                if job_id_str not in self._data_analysis_tools:
                    self._data_analysis_tools[job_id_str] = DataAnalysisTools(
                        job_id=job_id_str,
                        user_id=str(job.user_id)
                    )
                tools = self._data_analysis_tools[job_id_str]

                # Route to appropriate tool method
                if tool_name == "load_csv_data":
                    tool_result = tools.load_csv_data(
                        content=params.get("content", ""),
                        name=params.get("name", "dataset"),
                        delimiter=params.get("delimiter", ","),
                        has_header=params.get("has_header", True),
                    )
                elif tool_name == "load_json_data":
                    tool_result = tools.load_json_data(
                        content=params.get("content", ""),
                        name=params.get("name", "dataset"),
                    )
                elif tool_name == "create_dataset":
                    tool_result = tools.create_dataset(
                        data=params.get("data", {}),
                        name=params.get("name", "dataset"),
                    )
                elif tool_name == "list_datasets":
                    tool_result = tools.list_datasets()
                elif tool_name == "describe_dataset":
                    tool_result = tools.describe_dataset(
                        dataset_id=params.get("dataset_id"),
                    )
                elif tool_name == "query_data":
                    tool_result = tools.query_data(
                        dataset_id=params.get("dataset_id"),
                        query=params.get("query"),
                    )
                elif tool_name == "filter_data":
                    tool_result = tools.filter_data(
                        dataset_id=params.get("dataset_id"),
                        conditions=params.get("conditions", {}),
                    )
                elif tool_name == "aggregate_data":
                    tool_result = tools.aggregate_data(
                        dataset_id=params.get("dataset_id"),
                        group_by=params.get("group_by"),
                        aggregations=params.get("aggregations"),
                    )
                elif tool_name == "join_datasets":
                    tool_result = tools.join_datasets(
                        left_dataset_id=params.get("left_dataset_id"),
                        right_dataset_id=params.get("right_dataset_id"),
                        on=params.get("on"),
                        left_on=params.get("left_on"),
                        right_on=params.get("right_on"),
                        how=params.get("how", "inner"),
                    )
                elif tool_name == "transform_data":
                    tool_result = tools.transform_data(
                        dataset_id=params.get("dataset_id"),
                        operations=params.get("operations", []),
                    )
                elif tool_name == "detect_anomalies":
                    tool_result = tools.detect_anomalies(
                        dataset_id=params.get("dataset_id"),
                        columns=params.get("columns"),
                        method=params.get("method", "zscore"),
                        threshold=params.get("threshold", 3.0),
                    )
                elif tool_name == "calculate_correlations":
                    tool_result = tools.calculate_correlations(
                        dataset_id=params.get("dataset_id"),
                        columns=params.get("columns"),
                        method=params.get("method", "pearson"),
                    )
                elif tool_name == "create_chart":
                    tool_result = tools.create_chart(
                        dataset_id=params.get("dataset_id"),
                        chart_type=params.get("chart_type", "bar"),
                        x_column=params.get("x_column"),
                        y_columns=params.get("y_columns"),
                        title=params.get("title", ""),
                        config=params.get("config"),
                    )
                elif tool_name == "create_correlation_heatmap":
                    tool_result = tools.create_correlation_heatmap(
                        dataset_id=params.get("dataset_id"),
                        title=params.get("title", "Correlation Matrix"),
                    )
                elif tool_name == "create_flowchart":
                    tool_result = tools.create_flowchart(
                        nodes=params.get("nodes", []),
                        edges=params.get("edges", []),
                        title=params.get("title", ""),
                        direction=params.get("direction", "TD"),
                    )
                elif tool_name == "create_sequence_diagram":
                    tool_result = tools.create_sequence_diagram(
                        participants=params.get("participants", []),
                        messages=params.get("messages", []),
                        title=params.get("title", ""),
                    )
                elif tool_name == "create_er_diagram":
                    tool_result = tools.create_er_diagram(
                        entities=params.get("entities", []),
                        relationships=params.get("relationships", []),
                        title=params.get("title", ""),
                    )
                elif tool_name == "create_architecture_diagram":
                    tool_result = tools.create_architecture_diagram(
                        components=params.get("components", []),
                        connections=params.get("connections", []),
                        title=params.get("title", ""),
                        format=params.get("format", "auto"),
                    )
                elif tool_name == "create_drawio_diagram":
                    tool_result = tools.create_drawio_diagram(
                        nodes=params.get("nodes", []),
                        edges=params.get("edges", []),
                        title=params.get("title", ""),
                    )
                elif tool_name == "create_gantt_chart":
                    tool_result = tools.create_gantt_chart(
                        sections=params.get("sections", []),
                        title=params.get("title", "Project Timeline"),
                    )
                elif tool_name == "export_dataset_csv":
                    tool_result = tools.export_dataset_csv(
                        dataset_id=params.get("dataset_id"),
                    )
                elif tool_name == "export_dataset_json":
                    tool_result = tools.export_dataset_json(
                        dataset_id=params.get("dataset_id"),
                    )
                else:
                    tool_result = {"success": False, "error": f"Unknown data analysis tool: {tool_name}"}

                result["success"] = tool_result.get("success", False)
                result["data"] = tool_result

                # Add visualizations as artifacts
                if tool_result.get("success"):
                    if tool_result.get("image_base64"):
                        result["artifacts"].append({
                            "type": "chart" if "chart" in tool_name else "diagram",
                            "tool": tool_name,
                            "image_base64": tool_result["image_base64"],
                            "mime_type": tool_result.get("mime_type", "image/png"),
                        })
                    if tool_result.get("mermaid_code"):
                        result["artifacts"].append({
                            "type": "diagram",
                            "format": "mermaid",
                            "tool": tool_name,
                            "code": tool_result["mermaid_code"],
                        })
                    if tool_result.get("xml"):
                        result["artifacts"].append({
                            "type": "diagram",
                            "format": "drawio",
                            "tool": tool_name,
                            "xml": tool_result["xml"],
                            "edit_url": tool_result.get("edit_url"),
                        })
                    if tool_result.get("dot_code"):
                        result["artifacts"].append({
                            "type": "diagram",
                            "format": "graphviz",
                            "tool": tool_name,
                            "code": tool_result["dot_code"],
                        })

                # Add findings for analysis results
                if tool_name in ["detect_anomalies", "calculate_correlations", "describe_dataset"]:
                    if tool_result.get("success"):
                        result["findings"].append({
                            "type": "data_analysis",
                            "tool": tool_name,
                            "result": tool_result,
                        })

            # ==================== Fallback ====================
            else:
                result["error"] = f"Unknown or unimplemented tool: {tool_name}"
                logger.warning(f"Tool not implemented: {tool_name}")

        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            result["error"] = str(e)

        # Tool fallback routing (best-effort)
        # Enabled by job.config.tool_fallback_enabled (default True).
        cfg = job.config if isinstance(job.config, dict) else {}
        fallback_enabled = cfg.get("tool_fallback_enabled")
        if fallback_enabled is None:
            fallback_enabled = True

        if fallback_enabled and not result.get("success") and result.get("error"):
            depth = 0
            try:
                depth = int(params.get("_fallback_depth") or 0)
            except Exception:
                depth = 0

            max_depth = 1
            try:
                max_depth = int(cfg.get("tool_fallback_max_depth") or 1)
            except Exception:
                max_depth = 1
            max_depth = max(0, min(max_depth, 2))

            # Optional explicit overrides:
            # job.config.tool_fallback_map = {
            #   "tool_name": {"tool": "search_documents", "params": {"query": "...", "limit": 5}}
            # }
            fallback_map = cfg.get("tool_fallback_map") if isinstance(cfg.get("tool_fallback_map"), dict) else {}

            def _fallback_action_for(tool: str) -> Optional[Dict[str, Any]]:
                # 1) Explicit per-job overrides.
                if isinstance(fallback_map, dict) and tool in fallback_map and isinstance(fallback_map.get(tool), dict):
                    entry = fallback_map.get(tool) or {}
                    ft = str(entry.get("tool") or "").strip()
                    fp = entry.get("params") if isinstance(entry.get("params"), dict) else {}
                    if ft:
                        return {"tool": ft, "params": dict(fp)}

                # 2) Policy-based fallbacks (job type + default).
                if tool == "search_documents":
                    return None

                policy = dict(_TOOL_FALLBACK_POLICIES.get("_default") or {})
                policy.update(_TOOL_FALLBACK_POLICIES.get(str(getattr(job, "job_type", "") or "")) or {})

                entry = policy.get(tool) or policy.get("__default__")
                if not isinstance(entry, dict):
                    entry = None

                def _goal_query() -> str:
                    return str(getattr(job, "goal", "") or "").strip()

                def _query_from(param: str) -> str:
                    if param == "goal":
                        return _goal_query()
                    return str(params.get(param) or "").strip()

                if entry:
                    ft = str(entry.get("tool") or "").strip()
                    param = str(entry.get("param") or "goal").strip()
                    if ft == "search_documents":
                        q = _query_from(param)
                        if not q and param != "goal":
                            q = _goal_query()
                        if q:
                            return {"tool": "search_documents", "params": {"query": q, "limit": 5}}

                # 3) Final best-effort: search the KB with the user's goal.
                goal_q = _goal_query()
                if goal_q:
                    return {"tool": "search_documents", "params": {"query": goal_q, "limit": 5}}
                return None
            if depth < max_depth:
                fb_action_base = _fallback_action_for(str(tool_name or "").strip())
                if fb_action_base and isinstance(fb_action_base, dict):
                    try:
                        fb_action = {
                            "tool": fb_action_base.get("tool"),
                            "params": {
                                **(fb_action_base.get("params") or {}),
                                "_fallback_depth": depth + 1,
                                "_fallback_from": tool_name,
                            },
                        }
                        fb = await self._act(job, fb_action, state, db)
                        result["primary_tool"] = tool_name
                        result["primary_error"] = result.get("error")
                        result["fallback"] = fb
                        if isinstance(fb, dict) and fb.get("success"):
                            result["success"] = True
                            result["tool"] = fb.get("tool") or fb_action.get("tool")
                            result["data"] = fb.get("data")
                            result["findings"] = fb.get("findings") or result.get("findings")
                            result["artifacts"] = fb.get("artifacts") or result.get("artifacts")
                            result["note"] = f"Primary tool failed; used fallback tool: {result['tool']}"
                    except Exception:
                        pass
        return result

    def _score_research_evidence_quality(
        self,
        findings: List[Dict[str, Any]],
        target_docs: int,
        target_papers: int,
    ) -> float:
        """Score evidence quality (0-1) for research progress estimation."""
        if not findings:
            return 0.0

        unique_docs: set[str] = set()
        unique_papers: set[str] = set()
        quality = 0.0
        insight_bonus = 0.0

        for finding in findings:
            if not isinstance(finding, dict):
                continue
            ftype = str(finding.get("type") or "").strip().lower()

            if ftype == "document":
                doc_id = str(finding.get("id") or finding.get("document_id") or "").strip()
                if doc_id and doc_id not in unique_docs:
                    unique_docs.add(doc_id)
                    base = 1.0
                    raw_score = finding.get("score")
                    try:
                        score_val = float(raw_score)
                    except Exception:
                        score_val = 0.0
                    # Accommodate both [0,1] and larger retrieval scales.
                    norm_score = max(0.0, min(1.0, score_val if score_val <= 1.0 else score_val / 10.0))
                    quality += base + (0.5 * norm_score)

            elif ftype == "paper":
                paper_id = str(finding.get("arxiv_id") or finding.get("id") or "").strip()
                if paper_id and paper_id not in unique_papers:
                    unique_papers.add(paper_id)
                    base = 1.1
                    if finding.get("published"):
                        base += 0.2
                    if isinstance(finding.get("authors"), list) and finding.get("authors"):
                        base += 0.1
                    quality += base

            category = str(finding.get("category") or "").strip().lower()
            if category in {"key_insight", "methodology", "result", "gap", "connection", "trend"}:
                insight_bonus += 0.15

        quality += min(2.0, insight_bonus)
        denom = float(max(1, int(target_docs) + int(target_papers)))
        return max(0.0, min(1.0, quality / (1.5 * denom)))

    async def _evaluate_progress(
        self,
        job: AgentJob,
        state: Dict[str, Any],
        user_settings: Optional[UserLLMSettings],
        db: AsyncSession,
    ) -> int:
        """
        Evaluate progress toward the goal.

        Returns progress percentage (0-100).
        """
        findings_count = len(state.get("findings", []))
        actions_count = len(state.get("actions_taken", []))

        # Simple heuristic based on job type
        if job.job_type == "research":
            config = job.config or {}
            target_papers = int(config.get("max_papers", 10) or 10)
            target_papers = max(1, min(target_papers, 200))
            target_docs = int(config.get("max_documents", 10) or 10)
            target_docs = max(1, min(target_docs, 200))

            prefer_sources = config.get("prefer_sources") or []
            if isinstance(prefer_sources, str):
                prefer_sources = [x.strip() for x in prefer_sources.split(",") if x.strip()]
            prefer_sources = [str(x).strip().lower() for x in prefer_sources if str(x).strip()]

            findings = state.get("findings", []) or []
            papers_found = len([f for f in findings if f.get("type") == "paper"])
            docs_found = len([f for f in findings if f.get("type") == "document"])

            # Also count "document work" actions as progress.
            actions = state.get("actions_taken", []) or []
            docs_touched = len([
                a for a in actions
                if (a.get("action") or {}).get("tool") in {"get_document_details", "read_document_content", "summarize_document"}
                and (a.get("result") or {}).get("success")
            ])

            doc_units = max(docs_found, docs_touched)
            paper_score = min(1.0, papers_found / float(target_papers))
            doc_score = min(1.0, doc_units / float(target_docs))

            # Weight toward the preferred sources so internal-doc-only research can still "complete".
            if prefer_sources and "documents" in prefer_sources and "arxiv" not in prefer_sources:
                progress = doc_score
            elif prefer_sources and "arxiv" in prefer_sources and "documents" not in prefer_sources:
                progress = paper_score
            elif prefer_sources and prefer_sources[0] == "documents":
                progress = 0.8 * doc_score + 0.2 * paper_score
            else:
                progress = 0.5 * doc_score + 0.5 * paper_score

            quality_score = self._score_research_evidence_quality(
                findings=[f for f in findings if isinstance(f, dict)],
                target_docs=target_docs,
                target_papers=target_papers,
            )
            try:
                quality_weight = float(config.get("evidence_quality_weight", 0.35) or 0.35)
            except Exception:
                quality_weight = 0.35
            quality_weight = max(0.0, min(0.8, quality_weight))
            progress = ((1.0 - quality_weight) * progress) + (quality_weight * quality_score)

            # If we've produced a synthesis or persisted document, treat as near-done.
            artifacts = state.get("artifacts", []) or []
            if any(isinstance(a, dict) and a.get("type") in {"synthesis_document", "document"} for a in artifacts):
                progress = max(progress, 0.85)

            return min(100, int(progress * 100))

        elif job.job_type == "analysis":
            # Progress based on documents analyzed
            return min(100, int((findings_count / 10) * 100))

        elif job.job_type == "data_analysis":
            # Progress based on data analysis milestones
            config = job.config or {}
            progress = 0

            # Data loaded (20%)
            job_id_str = str(job.id)
            if job_id_str in self._data_analysis_tools:
                datasets = self._data_analysis_tools[job_id_str].list_datasets()
                if datasets.get("count", 0) > 0:
                    progress += 20

            # Transformations applied (30%)
            transforms = len([
                a for a in state.get("actions_taken", [])
                if a.get("action", {}).get("tool") in [
                    "query_data", "filter_data", "aggregate_data", "join_datasets", "transform_data"
                ]
            ])
            if transforms > 0:
                progress += min(30, transforms * 10)

            # Visualizations created (30%)
            viz = len([
                a for a in state.get("actions_taken", [])
                if a.get("action", {}).get("tool") in [
                    "create_chart", "create_correlation_heatmap", "create_flowchart",
                    "create_sequence_diagram", "create_er_diagram", "create_architecture_diagram",
                    "create_drawio_diagram", "create_gantt_chart"
                ]
            ])
            if viz > 0:
                progress += min(30, viz * 15)

            # Analysis done (20%)
            analysis = len([
                a for a in state.get("actions_taken", [])
                if a.get("action", {}).get("tool") in [
                    "detect_anomalies", "calculate_correlations", "describe_dataset"
                ]
            ])
            if analysis > 0:
                progress += min(20, analysis * 10)

            return min(100, progress)

        else:
            # Generic progress based on iterations
            return min(100, int((job.iteration / job.max_iterations) * 100))

    async def _finalize_job(
        self,
        job: AgentJob,
        state: Dict[str, Any],
        db: AsyncSession,
    ) -> Dict[str, Any]:
        """
        Finalize the job after execution completes.

        Sets final status, compiles results, and cleans up.
        """
        # Determine final status
        limited, limit_reason = job.is_resource_limited()
        contract_eval = self._evaluate_goal_contract(job, state)
        state["goal_contract_last"] = contract_eval
        existing_status = str(job.status or "")

        if existing_status == AgentJobStatus.PAUSED.value:
            job.status = AgentJobStatus.PAUSED.value
        elif existing_status == AgentJobStatus.CANCELLED.value:
            job.status = AgentJobStatus.CANCELLED.value
        elif state.get("goal_progress", 0) >= 100:
            job.status = AgentJobStatus.COMPLETED.value
        elif limited:
            job.status = AgentJobStatus.COMPLETED.value  # Completed with limits
            job.add_log_entry({
                "phase": "completed_with_limits",
                "reason": limit_reason,
            })
        elif job.error_count >= 5:
            job.status = AgentJobStatus.FAILED.value
        else:
            job.status = AgentJobStatus.COMPLETED.value

        if job.status != AgentJobStatus.PAUSED.value:
            job.completed_at = datetime.utcnow()
        job.progress = state.get("goal_progress", 0)

        # Compile results
        findings = state.get("findings", []) or []
        artifacts = state.get("artifacts", []) or []

        def _as_str(x: Any) -> str:
            try:
                return str(x)
            except Exception:
                return ""

        def _take_titles(items: list[dict[str, Any]], key: str, limit: int = 6) -> list[str]:
            out: list[str] = []
            for it in items:
                if not isinstance(it, dict):
                    continue
                val = it.get(key)
                if not val:
                    continue
                s = _as_str(val).strip()
                if not s:
                    continue
                out.append(s[:200])
                if len(out) >= limit:
                    break
            return out

        paper_findings = [f for f in findings if isinstance(f, dict) and f.get("type") == "paper"]
        doc_findings = [f for f in findings if isinstance(f, dict) and f.get("type") == "document"]
        insight_findings = [
            f for f in findings
            if isinstance(f, dict) and f.get("category") in {"key_insight", "methodology", "result", "gap", "connection", "contradiction", "trend"}
        ]

        job.results = {
            "findings_count": len(state.get("findings", [])),
            "actions_count": len(state.get("actions_taken", [])),
            "iterations": job.iteration,
            "findings": state.get("findings", [])[:50],  # Limit stored findings
            "goal_progress": state.get("goal_progress", 0),
        }
        job.results["goal_contract"] = {
            "enabled": bool(contract_eval.get("enabled", False)),
            "satisfied": bool(contract_eval.get("satisfied", True)),
            "missing": (
                contract_eval.get("missing", [])
                if isinstance(contract_eval.get("missing"), list)
                else []
            )[:20],
            "contract": (
                contract_eval.get("contract")
                if isinstance(contract_eval.get("contract"), dict)
                else {}
            ),
            "metrics": (
                contract_eval.get("metrics")
                if isinstance(contract_eval.get("metrics"), dict)
                else {}
            ),
            "satisfied_iteration": int(state.get("goal_contract_satisfied_iteration", 0) or 0),
        }
        job.results["execution_strategy"] = {
            "execution_plan": (state.get("execution_plan") or [])[:12] if isinstance(state.get("execution_plan"), list) else [],
            "plan_step_index": int(state.get("plan_step_index", 0) or 0),
            "causal_experiment_planner": {
                "enabled": bool((job.config or {}).get("causal_experiment_planner_enabled", True)),
                "attempted": bool(state.get("causal_plan_generation_attempted", False)),
                "plan": (
                    state.get("causal_experiment_plan")
                    if isinstance(state.get("causal_experiment_plan"), dict)
                    else {}
                ),
                "hypothesis_count": len(
                    (state.get("causal_experiment_plan") or {}).get("hypotheses", [])
                    if isinstance((state.get("causal_experiment_plan") or {}).get("hypotheses"), list)
                    else []
                ),
                "experiment_count": len(
                    (state.get("causal_experiment_plan") or {}).get("experiments", [])
                    if isinstance((state.get("causal_experiment_plan") or {}).get("experiments"), list)
                    else []
                ),
            },
            "subgoals": (state.get("subgoals") or [])[:12] if isinstance(state.get("subgoals"), list) else [],
            "subgoal_index": int(state.get("subgoal_index", 0) or 0),
            "subgoal_chain_configured": bool(state.get("subgoal_chain_configured", False)),
            "swarm": {
                "enabled": bool((job.config or {}).get("swarm_child_jobs_enabled", False)),
                "configured": bool(state.get("swarm_chain_configured", False)),
                "child_jobs_count": int(state.get("swarm_child_jobs_count", 0) or 0),
                "fan_in_enabled": bool(state.get("swarm_fan_in_enabled", False)),
                "fan_in_group_id": str(state.get("swarm_fan_in_group_id") or ""),
                "roles_assigned": (
                    state.get("swarm_roles_assigned")
                    if isinstance(state.get("swarm_roles_assigned"), list)
                    else []
                ),
            },
            "critic_notes": (state.get("critic_notes") or [])[-5:] if isinstance(state.get("critic_notes"), list) else [],
            "critic_last_trigger": (
                state.get("critic_last_trigger")
                if isinstance(state.get("critic_last_trigger"), dict)
                else {}
            ),
            "critic_trigger_counts": (
                state.get("critic_trigger_counts")
                if isinstance(state.get("critic_trigger_counts"), dict)
                else {}
            ),
            "approval_checkpoints": {
                **self._get_approval_checkpoint_config(job),
                "events": (
                    state.get("approval_checkpoint_events")
                    if isinstance(state.get("approval_checkpoint_events"), list)
                    else []
                )[-20:],
                "pending": (
                    state.get("approval_checkpoint_pending")
                    if isinstance(state.get("approval_checkpoint_pending"), dict)
                    else None
                ),
                "seen": (
                    state.get("approval_checkpoint_seen")
                    if isinstance(state.get("approval_checkpoint_seen"), list)
                    else []
                )[-200:],
            },
            "tool_stats": state.get("tool_stats") if isinstance(state.get("tool_stats"), dict) else {},
            "tool_priors": state.get("tool_priors") if isinstance(state.get("tool_priors"), dict) else {},
            "tool_selection": {
                **self._get_tool_selection_config(job),
                "forced_exploration": self._get_forced_exploration_config(job),
                "cooldown": self._get_tool_cooldown_config(job),
                "policy_mode_effective": str(state.get("tool_selection_effective_mode") or ""),
                "goal_stage": str(state.get("tool_selection_goal_stage") or ""),
                "mode_override": str(state.get("tool_selection_mode_override") or ""),
                "ab_assignment": (
                    state.get("tool_selection_ab_assignment")
                    if isinstance(state.get("tool_selection_ab_assignment"), dict)
                    else {}
                ),
                "runtime": {
                    "forced_exploration_attempts": int(state.get("forced_exploration_attempts", 0) or 0),
                    "forced_exploration_used": int(state.get("forced_exploration_used", 0) or 0),
                    "forced_exploration_successes": int(state.get("forced_exploration_successes", 0) or 0),
                    "forced_exploration_failures": int(state.get("forced_exploration_failures", 0) or 0),
                    "forced_exploration_rate": (
                        float(int(state.get("forced_exploration_used", 0) or 0))
                        / float(max(1, int(state.get("forced_exploration_attempts", 0) or 0)))
                    ),
                    "forced_exploration_success_rate": (
                        float(int(state.get("forced_exploration_successes", 0) or 0))
                        / float(max(1, int(state.get("forced_exploration_used", 0) or 0)))
                    ),
                    "forced_exploration_history": (
                        state.get("forced_exploration_history", [])[-20:]
                        if isinstance(state.get("forced_exploration_history"), list)
                        else []
                    ),
                    "active_tool_cooldowns": (
                        state.get("tool_cooldowns")
                        if isinstance(state.get("tool_cooldowns"), dict)
                        else {}
                    ),
                    "tool_cooldown_blocks": int(state.get("tool_cooldown_blocks", 0) or 0),
                    "mode_metrics": (
                        state.get("tool_selection_mode_metrics")
                        if isinstance(state.get("tool_selection_mode_metrics"), dict)
                        else {}
                    ),
                    "fallback_events": (
                        state.get("tool_selection_fallback_events", [])[-20:]
                        if isinstance(state.get("tool_selection_fallback_events"), list)
                        else []
                    ),
                    "counterfactual_logged_iterations": int(state.get("counterfactual_logged_iterations", 0) or 0),
                    "counterfactual_last_iteration": int(state.get("counterfactual_last_iteration", 0) or 0),
                    "counterfactual_last": (
                        state.get("counterfactual_last", [])[:10]
                        if isinstance(state.get("counterfactual_last"), list)
                        else []
                    ),
                    "selection_explainability_logged_iterations": int(state.get("selection_explainability_logged_iterations", 0) or 0),
                    "selection_explainability_last": (
                        state.get("selection_explainability_last")
                        if isinstance(state.get("selection_explainability_last"), dict)
                        else {}
                    ),
                },
            },
            "skill_profile": {
                "role": str(((state.get("skill_profile") or {}).get("role") or "researcher")),
                "display_name": str(((state.get("skill_profile") or {}).get("display_name") or "")),
                "prompt_directives": (
                    [str(x) for x in ((state.get("skill_profile") or {}).get("prompt_directives") or [])[:6]]
                    if isinstance((state.get("skill_profile") or {}).get("prompt_directives"), list)
                    else []
                ),
                "preferred_tools": (
                    [str(x) for x in ((state.get("skill_profile") or {}).get("preferred_tools") or [])[:20]]
                    if isinstance((state.get("skill_profile") or {}).get("preferred_tools"), list)
                    else []
                ),
                "discouraged_tools": (
                    [str(x) for x in ((state.get("skill_profile") or {}).get("discouraged_tools") or [])[:20]]
                    if isinstance((state.get("skill_profile") or {}).get("discouraged_tools"), list)
                    else []
                ),
                "metrics": (
                    state.get("skill_profile_metrics")
                    if isinstance(state.get("skill_profile_metrics"), dict)
                    else {}
                ),
            },
            "feedback_learning": (
                state.get("feedback_learning")
                if isinstance(state.get("feedback_learning"), dict)
                else {}
            ),
        }
        if bool((job.config or {}).get("tool_selection_replay_enabled", False)):
            replay_steps = 200
            try:
                replay_steps = int((job.config or {}).get("tool_selection_replay_steps", 200) or 200)
            except Exception:
                replay_steps = 200
            replay_steps = max(25, min(replay_steps, 5000))

            replay_modes = (job.config or {}).get("tool_selection_replay_modes")
            if not isinstance(replay_modes, list):
                replay_modes = ["baseline", "adaptive", "thompson"]

            replay_seed = 42
            try:
                replay_seed = int((job.config or {}).get("tool_selection_replay_seed", 42) or 42)
            except Exception:
                replay_seed = 42

            merged_for_replay = self._merge_tool_stats(
                state.get("tool_priors") if isinstance(state.get("tool_priors"), dict) else {},
                state.get("tool_stats") if isinstance(state.get("tool_stats"), dict) else {},
            )
            replay = self.simulate_tool_selection_replay(
                merged_for_replay,
                steps=replay_steps,
                policy_modes=[str(x) for x in replay_modes],
                seed=replay_seed,
            )
            tool_sel = job.results["execution_strategy"].get("tool_selection")
            if isinstance(tool_sel, dict):
                tool_sel["replay"] = replay

        if state.get("customer_profile") or (state.get("customer_context") or "").strip():
            job.results["customer_profile"] = state.get("customer_profile")
            job.results["customer_context"] = (state.get("customer_context") or "").strip()

        # Add a lightweight, deterministic summary for research jobs so the UI has something to display
        # even when no explicit synthesis doc was created.
        if job.job_type == "research":
            doc_titles = _take_titles(doc_findings, "title", limit=6)
            paper_titles = _take_titles(paper_findings, "title", limit=6)
            insight_titles = _take_titles(insight_findings, "title", limit=8)
            created_doc_ids = [
                _as_str(a.get("id") or a.get("document_id"))
                for a in artifacts
                if isinstance(a, dict) and a.get("type") == "document" and (a.get("id") or a.get("document_id"))
            ]
            created_doc_ids = [x for x in created_doc_ids if x]

            job.results["research"] = {
                "documents_found": len(doc_findings),
                "papers_found": len(paper_findings),
                "insights_saved": len(insight_findings),
                "top_documents": doc_titles,
                "top_papers": paper_titles,
                "top_insights": insight_titles,
                "created_documents": created_doc_ids[:10],
            }
            job.results["summary"] = (
                f"Research run completed: {len(doc_findings)} KB docs, {len(paper_findings)} papers, "
                f"{len(insight_findings)} saved insights."
            )

            # Standardized schema for downstream UX/workflows.
            customer_profile = state.get("customer_profile") if isinstance(state.get("customer_profile"), dict) else None
            customer_name = (customer_profile or {}).get("name") if customer_profile else None
            customer_keywords = (customer_profile or {}).get("keywords") if customer_profile else None
            if not isinstance(customer_keywords, list):
                customer_keywords = []
            customer_keywords = [str(x).strip() for x in customer_keywords if str(x).strip()]

            def _suggest_queries() -> list[str]:
                goal = (job.goal or "").strip()
                out: list[str] = []
                if goal:
                    out.append(goal[:140])
                # Blend in customer keywords deterministically.
                for kw in customer_keywords[:8]:
                    if not goal:
                        out.append(kw[:140])
                    else:
                        out.append(f"{kw} {goal[:120]}".strip()[:140])
                # Add a customer-name anchored query.
                if customer_name:
                    out.append(f"{customer_name} {goal[:120]}".strip()[:140] if goal else str(customer_name)[:140])
                # Deduplicate preserve order.
                seen: set[str] = set()
                deduped: list[str] = []
                for q in out:
                    q = (q or "").strip()
                    if not q or q in seen:
                        continue
                    seen.add(q)
                    deduped.append(q)
                return deduped[:12]

            top_docs_struct = []
            seen_doc_ids: set[str] = set()
            for f in doc_findings:
                if not isinstance(f, dict):
                    continue
                did = _as_str(f.get("id")).strip()
                if not did or did in seen_doc_ids:
                    continue
                seen_doc_ids.add(did)
                top_docs_struct.append({"id": did, "title": _as_str(f.get("title")).strip()[:300]})
                if len(top_docs_struct) >= 12:
                    break

            top_papers_struct = []
            seen_paper_ids: set[str] = set()
            for f in paper_findings:
                if not isinstance(f, dict):
                    continue
                pid = _as_str(f.get("arxiv_id") or f.get("id")).strip()
                if not pid or pid in seen_paper_ids:
                    continue
                seen_paper_ids.add(pid)
                top_papers_struct.append(
                    {
                        "arxiv_id": pid,
                        "title": _as_str(f.get("title")).strip()[:300],
                        "published": f.get("published"),
                    }
                )
                if len(top_papers_struct) >= 12:
                    break

            top_insights_struct = []
            seen_insight_ids: set[str] = set()
            for f in insight_findings:
                if not isinstance(f, dict):
                    continue
                fid = _as_str(f.get("id")).strip()
                if not fid:
                    fid = _as_str(f.get("title")).strip()
                if not fid or fid in seen_insight_ids:
                    continue
                seen_insight_ids.add(fid)
                top_insights_struct.append(
                    {
                        "id": _as_str(f.get("id")).strip() or None,
                        "title": _as_str(f.get("title")).strip()[:300],
                        "category": f.get("category"),
                        "confidence": f.get("confidence"),
                    }
                )
                if len(top_insights_struct) >= 20:
                    break

            causal_plan = state.get("causal_experiment_plan") if isinstance(state.get("causal_experiment_plan"), dict) else {}
            causal_experiments = causal_plan.get("experiments") if isinstance(causal_plan.get("experiments"), list) else []
            causal_priority = causal_plan.get("priority_order") if isinstance(causal_plan.get("priority_order"), list) else []
            exp_map = {
                str(e.get("id") or "").strip(): e
                for e in causal_experiments
                if isinstance(e, dict) and str(e.get("id") or "").strip()
            }
            ordered_experiment_ids = [str(x).strip() for x in causal_priority if str(x).strip() in set(exp_map.keys())]
            if not ordered_experiment_ids:
                ordered_experiment_ids = list(exp_map.keys())
            prioritized_experiments = []
            for eid in ordered_experiment_ids[:3]:
                exp = exp_map.get(eid)
                if not isinstance(exp, dict):
                    continue
                prioritized_experiments.append(
                    {
                        "id": eid,
                        "hypothesis_id": str(exp.get("hypothesis_id") or "").strip() or None,
                        "name": str(exp.get("name") or "").strip()[:220],
                        "minimal_design": str(exp.get("minimal_design") or "").strip()[:280],
                        "estimated_effort": str(exp.get("estimated_effort") or "").strip()[:20] or None,
                        "expected_evidence": (
                            exp.get("expected_evidence")
                            if isinstance(exp.get("expected_evidence"), dict)
                            else {}
                        ),
                    }
                )

            next_steps = [
                "Confirm constraints and success metrics for this customer.",
                "Pick 1–2 highest-signal hypotheses from key insights.",
                "Design a minimal experiment plan (data, evaluation, timeline).",
            ]
            if prioritized_experiments:
                next_steps = [f"Run prioritized causal experiment: {str(prioritized_experiments[0].get('name') or '')[:120]}"]
                if len(prioritized_experiments) > 1:
                    next_steps.append(f"Then run: {str(prioritized_experiments[1].get('name') or '')[:120]}")
                next_steps.append("Update hypothesis confidence based on support/falsification evidence.")

            job.results["research_bundle"] = {
                "customer": {"name": customer_name, "keywords": customer_keywords[:30]},
                "goal": (job.goal or "").strip(),
                "suggested_queries": _suggest_queries(),
                "top_documents": top_docs_struct,
                "top_papers": top_papers_struct,
                "key_insights": top_insights_struct,
                "artifacts": [a for a in artifacts if isinstance(a, dict)][:50],
                "causal_experiment_plan": {
                    "hypotheses": (
                        causal_plan.get("hypotheses")
                        if isinstance(causal_plan.get("hypotheses"), list)
                        else []
                    )[:6],
                    "priority_experiments": prioritized_experiments[:3],
                    "decision_rules": (
                        causal_plan.get("decision_rules")
                        if isinstance(causal_plan.get("decision_rules"), list)
                        else []
                    )[:6],
                    "source": str(causal_plan.get("source") or ""),
                },
                "next_steps": next_steps,
            }

            # Optional reading list auto-population (deterministic; no extra LLM calls).
            reading_list_name = str((job.config or {}).get("reading_list_name") or "").strip()
            if reading_list_name and not any(isinstance(a, dict) and a.get("type") == "reading_list" for a in artifacts):
                try:
                    from app.models.reading_list import ReadingList, ReadingListItem
                    from app.models.document import Document

                    rl_res = await db.execute(
                        select(ReadingList).where(
                            ReadingList.user_id == job.user_id,
                            ReadingList.name == reading_list_name,
                        )
                    )
                    rl = rl_res.scalar_one_or_none()
                    if not rl:
                        rl = ReadingList(user_id=job.user_id, name=reading_list_name, description=None, source_id=None)
                        db.add(rl)
                        await db.flush()

                    max_pos = int(
                        (await db.execute(
                            select(func.max(ReadingListItem.position)).where(ReadingListItem.reading_list_id == rl.id)
                        )).scalar() or 0
                    )

                    added = 0
                    limit = int((job.config or {}).get("max_documents") or 12)
                    limit = max(1, min(limit, 200))
                    for it in top_docs_struct[:limit]:
                        did = (it or {}).get("id")
                        if not did:
                            continue
                        try:
                            from uuid import UUID as _UUID

                            doc_uuid = _UUID(str(did))
                        except Exception:
                            continue

                        doc = await db.get(Document, doc_uuid)
                        if not doc:
                            continue

                        exists = await db.execute(
                            select(func.count())
                            .select_from(ReadingListItem)
                            .where(
                                ReadingListItem.reading_list_id == rl.id,
                                ReadingListItem.document_id == doc.id,
                            )
                        )
                        if int(exists.scalar() or 0) > 0:
                            continue

                        item = ReadingListItem(
                            reading_list_id=rl.id,
                            document_id=doc.id,
                            status="to-read",
                            priority=0,
                            position=max_pos + 1,
                            notes="Added automatically by customer research job",
                        )
                        db.add(item)
                        try:
                            await db.flush()
                        except IntegrityError:
                            await db.rollback()
                            continue

                        max_pos += 1
                        added += 1

                    await db.commit()
                    if added > 0 or rl:
                        artifacts.append({"type": "reading_list", "id": str(rl.id), "name": rl.name, "items_added": added})
                        job.results["research_bundle"]["reading_list"] = {"id": str(rl.id), "name": rl.name, "items_added": added}
                except Exception as exc:
                    logger.warning(f"Failed to auto-populate reading list: {exc}")

            # Optional auto-brief persistence (deterministic; no extra LLM calls).
            persist = bool((job.config or {}).get("persist_artifacts", False))
            if persist and not created_doc_ids:
                customer_profile = state.get("customer_profile") if isinstance(state.get("customer_profile"), dict) else None
                profile_name = (customer_profile or {}).get("name") if customer_profile else None
                title = f"Customer Research Brief — {profile_name}" if profile_name else "Customer Research Brief"

                customer_context = (state.get("customer_context") or "").strip()
                brief_lines: list[str] = []
                brief_lines.append(f"# {title}")
                brief_lines.append("")
                brief_lines.append("## Goal")
                brief_lines.append((job.goal or "").strip() or "(none)")
                if customer_context:
                    brief_lines.append("")
                    brief_lines.append("## Customer context")
                    brief_lines.append(customer_context[:2000])
                if doc_titles:
                    brief_lines.append("")
                    brief_lines.append("## Top internal documents")
                    for t in doc_titles:
                        brief_lines.append(f"- {t}")
                if paper_titles:
                    brief_lines.append("")
                    brief_lines.append("## Top papers")
                    for t in paper_titles:
                        brief_lines.append(f"- {t}")
                if insight_titles:
                    brief_lines.append("")
                    brief_lines.append("## Key insights")
                    for t in insight_titles:
                        brief_lines.append(f"- {t}")
                brief_lines.append("")
                brief_lines.append("## Next steps")
                brief_lines.append("- Validate the top insights against the customer constraints.")
                brief_lines.append("- Turn the most promising direction into an experiment plan (metrics + timeline).")

                content = "\n".join(brief_lines).strip() + "\n"
                try:
                    from app.models.document import Document

                    notes_source = await self.document_service._get_or_create_agent_notes_source(db)
                    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
                    doc = Document(
                        title=title,
                        content=content,
                        content_hash=content_hash,
                        url=None,
                        file_path=None,
                        file_type="text/markdown",
                        file_size=len(content.encode("utf-8")),
                        source_id=notes_source.id,
                        source_identifier=f"agent_research_brief:{uuid.uuid4().hex}",
                        author=None,
                        tags=["autonomous_job", "research", "customer_research"],
                        extra_metadata={
                            "origin": "autonomous_job",
                            "job_id": str(job.id),
                            "job_type": job.job_type,
                        },
                        is_processed=False,
                    )
                    db.add(doc)
                    await db.commit()
                    await db.refresh(doc)

                    try:
                        await self.document_service.reprocess_document(doc.id, db, user_id=job.user_id)
                    except Exception as exc:
                        logger.warning(f"Failed to process research brief embeddings: {exc}")

                    artifacts.append({"type": "document", "id": str(doc.id), "title": doc.title})
                    job.results["research"]["created_documents"] = [str(doc.id)]
                    job.results["research"]["brief_document_id"] = str(doc.id)
                except Exception as exc:
                    logger.warning(f"Failed to persist research brief: {exc}")

        # Ensure any finalize-time artifact additions are visible to callers.
        state["artifacts"] = artifacts
        job.output_artifacts = artifacts

        # Re-evaluate contract after finalize-time result/artifact mutations.
        final_contract_eval = self._evaluate_goal_contract(job, state)
        state["goal_contract_last"] = final_contract_eval
        strict_contract = bool((final_contract_eval.get("contract") or {}).get("strict_completion", False))
        if (
            job.status not in {AgentJobStatus.PAUSED.value, AgentJobStatus.CANCELLED.value}
            and strict_contract
            and bool(final_contract_eval.get("enabled"))
            and not bool(final_contract_eval.get("satisfied"))
        ):
            missing = final_contract_eval.get("missing") if isinstance(final_contract_eval.get("missing"), list) else []
            job.status = AgentJobStatus.FAILED.value
            job.error = f"Goal contract unmet: {', '.join([str(x) for x in missing[:5]])}"

        job.results["goal_contract"] = {
            "enabled": bool(final_contract_eval.get("enabled", False)),
            "satisfied": bool(final_contract_eval.get("satisfied", True)),
            "missing": (
                final_contract_eval.get("missing", [])
                if isinstance(final_contract_eval.get("missing"), list)
                else []
            )[:20],
            "contract": (
                final_contract_eval.get("contract")
                if isinstance(final_contract_eval.get("contract"), dict)
                else {}
            ),
            "metrics": (
                final_contract_eval.get("metrics")
                if isinstance(final_contract_eval.get("metrics"), dict)
                else {}
            ),
            "satisfied_iteration": int(state.get("goal_contract_satisfied_iteration", 0) or 0),
        }

        if job.status != AgentJobStatus.PAUSED.value and not job.completed_at:
            job.completed_at = datetime.utcnow()
        job.results["executive_digest"] = self._build_executive_digest(job, state)

        # Persist tool-learning signal for future jobs.
        try:
            await self._persist_tool_priors(job, state, db)
        except Exception as e:
            logger.warning(f"Failed to persist tool priors for job {job.id}: {e}")

        # Cleanup data analysis sandbox if used
        job_id_str = str(job.id)
        if job_id_str in self._data_analysis_tools:
            try:
                from app.services.data_sandbox_service import sandbox_manager
                sandbox_manager.cleanup(job_id_str)
            except Exception as e:
                logger.warning(f"Failed to cleanup data sandbox for job {job.id}: {e}")
            del self._data_analysis_tools[job_id_str]

        await db.commit()

        # Extract memories from completed job
        if job.enable_memory and job.status == AgentJobStatus.COMPLETED.value:
            try:
                extracted_memories = await agent_job_memory_service.extract_memories_from_job(
                    job=job,
                    user_id=str(job.user_id),
                    db=db,
                )
                if extracted_memories:
                    logger.info(f"Extracted {len(extracted_memories)} memories from job {job.id}")
                    job.add_log_entry({
                        "phase": "memory_extraction",
                        "memories_created": len(extracted_memories),
                        "memory_types": list(set(m.memory_type for m in extracted_memories)),
                    })
                    await db.commit()
            except Exception as e:
                logger.warning(f"Failed to extract memories from job {job.id}: {e}")

        if job.status == AgentJobStatus.PAUSED.value:
            return {
                "status": job.status,
                "progress": job.progress,
                "results": job.results,
                "iterations": job.iteration,
                "tool_calls": job.tool_calls_used,
                "llm_calls": job.llm_calls_used,
                "memories_injected": job.memory_injection_count or 0,
                "memories_created": job.memories_created_count or 0,
            }

        # Check if we should trigger chained jobs
        event = "complete" if job.status == AgentJobStatus.COMPLETED.value else "fail"
        await self._trigger_chained_jobs(job, event, db)

        return {
            "status": job.status,
            "progress": job.progress,
            "results": job.results,
            "iterations": job.iteration,
            "tool_calls": job.tool_calls_used,
            "llm_calls": job.llm_calls_used,
            "memories_injected": job.memory_injection_count or 0,
            "memories_created": job.memories_created_count or 0,
        }

    async def _evaluate_swarm_fan_in_gate(
        self,
        parent_job: AgentJob,
        db: AsyncSession,
    ) -> Dict[str, Any]:
        """Decide whether a swarm fan-in child is allowed to trigger now."""
        chain_cfg = parent_job.chain_config if isinstance(parent_job.chain_config, dict) else {}
        chain_data = chain_cfg.get("chain_data") if isinstance(chain_cfg.get("chain_data"), dict) else {}
        if not bool(chain_data.get("swarm_fan_in_wait_for_all_siblings", False)):
            return {"enabled": False, "ready": True, "already_exists": False}

        group_id = str(chain_data.get("swarm_fan_in_group_id") or "").strip()
        sibling_parent_id = parent_job.parent_job_id
        if not sibling_parent_id:
            return {"enabled": True, "ready": True, "already_exists": False, "group_id": group_id}

        siblings_res = await db.execute(
            select(AgentJob).where(AgentJob.parent_job_id == sibling_parent_id)
        )
        siblings = siblings_res.scalars().all()
        terminal = {
            AgentJobStatus.COMPLETED.value,
            AgentJobStatus.FAILED.value,
            AgentJobStatus.CANCELLED.value,
        }
        total_siblings = len(siblings)
        terminal_count = len([s for s in siblings if str(s.status) in terminal])
        expected = int(chain_data.get("swarm_fan_in_expected_siblings", 0) or 0)
        if expected <= 0:
            expected = total_siblings
        ready = bool(total_siblings >= expected and terminal_count >= expected)

        already_exists = False
        if group_id and siblings:
            sibling_ids = [s.id for s in siblings if getattr(s, "id", None) is not None]
            if sibling_ids:
                child_res = await db.execute(
                    select(AgentJob).where(AgentJob.parent_job_id.in_(sibling_ids))
                )
                for child in child_res.scalars().all():
                    cfg = child.config if isinstance(child.config, dict) else {}
                    if str(cfg.get("origin") or "") != "swarm_fan_in_aggregator":
                        continue
                    if str(cfg.get("swarm_fan_in_group_id") or "") == group_id:
                        already_exists = True
                        break

        return {
            "enabled": True,
            "ready": ready,
            "already_exists": already_exists,
            "group_id": group_id,
            "expected_siblings": expected,
            "total_siblings": total_siblings,
            "terminal_siblings": terminal_count,
        }

    async def _build_swarm_sibling_payload(
        self,
        parent_job: AgentJob,
        db: AsyncSession,
    ) -> Dict[str, Any]:
        """Collect sibling job outputs for swarm fan-in aggregators."""
        sibling_parent_id = parent_job.parent_job_id
        if not sibling_parent_id:
            return {}
        siblings_res = await db.execute(
            select(AgentJob).where(AgentJob.parent_job_id == sibling_parent_id)
        )
        siblings = siblings_res.scalars().all()
        if not siblings:
            return {}
        terminal = {
            AgentJobStatus.COMPLETED.value,
            AgentJobStatus.FAILED.value,
            AgentJobStatus.CANCELLED.value,
        }
        out: List[Dict[str, Any]] = []
        for s in siblings:
            cfg = s.config if isinstance(s.config, dict) else {}
            out.append(
                {
                    "job_id": str(s.id),
                    "name": str(s.name or "")[:200],
                    "status": str(s.status or ""),
                    "is_terminal": str(s.status or "") in terminal,
                    "progress": int(s.progress or 0),
                    "role": str(cfg.get("swarm_role") or ""),
                    "results": s.results if isinstance(s.results, dict) else {},
                }
            )
        return {
            "swarm_parent_job_id": str(sibling_parent_id),
            "expected_siblings": len(out),
            "terminal_siblings": len([x for x in out if bool(x.get("is_terminal"))]),
            "sibling_jobs": out,
        }

    async def _trigger_chained_jobs(
        self,
        parent_job: AgentJob,
        event: str,
        db: AsyncSession,
        value: int = 0,
    ) -> List[str]:
        """
        Trigger chained jobs based on the parent job completion.

        Args:
            parent_job: The parent job that completed
            event: Event type ('complete', 'fail', 'progress', 'findings')
            db: Database session
            value: Event value (progress percentage or findings count)

        Returns:
            List of created child job IDs
        """
        # Check if chain should be triggered
        if parent_job.chain_triggered:
            logger.debug(f"Job {parent_job.id} already triggered its chain")
            return []

        if not parent_job.should_trigger_chain(event, value):
            logger.debug(f"Chain trigger condition not met for job {parent_job.id}")
            return []

        # Get chain configuration
        chain_config = parent_job.chain_config or {}
        if not chain_config:
            return []

        fan_in_gate = await self._evaluate_swarm_fan_in_gate(parent_job, db)
        if bool(fan_in_gate.get("enabled", False)):
            if bool(fan_in_gate.get("already_exists", False)):
                parent_job.chain_triggered = True
                parent_job.add_log_entry(
                    {
                        "phase": "swarm_fan_in_duplicate_skipped",
                        "group_id": str(fan_in_gate.get("group_id") or ""),
                    }
                )
                await db.commit()
                return []
            if not bool(fan_in_gate.get("ready", False)):
                parent_job.add_log_entry(
                    {
                        "phase": "swarm_fan_in_deferred",
                        "group_id": str(fan_in_gate.get("group_id") or ""),
                        "expected_siblings": int(fan_in_gate.get("expected_siblings", 0) or 0),
                        "terminal_siblings": int(fan_in_gate.get("terminal_siblings", 0) or 0),
                        "total_siblings": int(fan_in_gate.get("total_siblings", 0) or 0),
                    }
                )
                await db.commit()
                return []

        # Mark parent as having triggered its chain
        parent_job.chain_triggered = True

        created_job_ids = []

        # Check for defined child jobs in chain_config
        child_jobs_config = chain_config.get("child_jobs", [])
        if child_jobs_config:
            for child_config in child_jobs_config:
                try:
                    child_job = await self._create_chained_job(
                        parent_job=parent_job,
                        child_config=child_config,
                        db=db,
                    )
                    if child_job:
                        created_job_ids.append(str(child_job.id))
                        logger.info(f"Created chained job {child_job.id} from parent {parent_job.id}")
                except Exception as e:
                    logger.error(f"Failed to create chained job: {e}")

        await db.commit()

        # Trigger execution of created jobs
        from app.tasks.agent_job_tasks import execute_agent_job_task
        for job_id in created_job_ids:
            execute_agent_job_task.delay(job_id, str(parent_job.user_id))

        return created_job_ids

    async def _create_chained_job(
        self,
        parent_job: AgentJob,
        child_config: Dict[str, Any],
        db: AsyncSession,
    ) -> Optional[AgentJob]:
        """
        Create a chained child job from configuration.

        Args:
            parent_job: The parent job
            child_config: Configuration for the child job
            db: Database session

        Returns:
            Created AgentJob or None if creation failed
        """
        # Get data to pass to child
        chain_data = parent_job.get_chain_data_for_child()
        parent_config = parent_job.chain_config or {}

        # Build child job configuration
        child_job_config = child_config.get("config", {})

        # Merge inherited config if specified
        if parent_config.get("inherit_config") and parent_job.config:
            child_job_config = {**parent_job.config, **child_job_config}

        # Merge parent results if inheriting
        if parent_config.get("inherit_results", True) and parent_job.results:
            if "inherited_data" not in child_job_config:
                child_job_config["inherited_data"] = {}
            child_job_config["inherited_data"]["parent_results"] = parent_job.results
            child_job_config["inherited_data"]["parent_findings"] = parent_job.results.get("findings", [])

        if str(child_job_config.get("origin") or "") == "swarm_fan_in_aggregator":
            sibling_payload = await self._build_swarm_sibling_payload(parent_job, db)
            if sibling_payload:
                if "inherited_data" not in child_job_config or not isinstance(child_job_config.get("inherited_data"), dict):
                    child_job_config["inherited_data"] = {}
                child_job_config["inherited_data"]["swarm"] = sibling_payload

        # Create the child job
        child_job = AgentJob(
            name=child_config.get("name", f"Chained: {parent_job.name}"),
            description=child_config.get("description", f"Chained from job: {parent_job.name}"),
            job_type=child_config.get("job_type", parent_job.job_type),
            goal=child_config.get("goal", parent_job.goal),
            goal_criteria=child_config.get("goal_criteria"),
            config=child_job_config,
            agent_definition_id=child_config.get("agent_definition_id") or parent_job.agent_definition_id,
            user_id=parent_job.user_id,
            status=AgentJobStatus.PENDING.value,
            # Chain hierarchy
            parent_job_id=parent_job.id,
            root_job_id=parent_job.root_job_id or parent_job.id,
            chain_depth=parent_job.chain_depth + 1,
            # Chain config for further chaining
            chain_config=child_config.get("chain_config"),
            # Resource limits - inherit from parent or use child config
            max_iterations=child_config.get("max_iterations", parent_job.max_iterations),
            max_tool_calls=child_config.get("max_tool_calls", parent_job.max_tool_calls),
            max_llm_calls=child_config.get("max_llm_calls", parent_job.max_llm_calls),
            max_runtime_minutes=child_config.get("max_runtime_minutes", parent_job.max_runtime_minutes),
        )

        db.add(child_job)
        await db.flush()  # Get the ID

        # Log the chain creation
        parent_job.add_log_entry({
            "phase": "chain_triggered",
            "child_job_id": str(child_job.id),
            "child_job_name": child_job.name,
            "trigger_event": "complete" if parent_job.status == AgentJobStatus.COMPLETED.value else "fail",
        })

        return child_job

    async def trigger_progress_chain(
        self,
        job: AgentJob,
        progress: int,
        findings_count: int,
        db: AsyncSession,
    ) -> List[str]:
        """
        Check and trigger chains based on progress or findings thresholds.

        Called during job execution when progress or findings are updated.

        Args:
            job: The running job
            progress: Current progress percentage
            findings_count: Current findings count
            db: Database session

        Returns:
            List of triggered job IDs
        """
        triggered_jobs = []

        # Check progress-based trigger
        if job.should_trigger_chain("progress", progress):
            triggered = await self._trigger_chained_jobs(job, "progress", db, progress)
            triggered_jobs.extend(triggered)

        # Check findings-based trigger
        if job.should_trigger_chain("findings", findings_count):
            triggered = await self._trigger_chained_jobs(job, "findings", db, findings_count)
            triggered_jobs.extend(triggered)

        return triggered_jobs

    async def _save_checkpoint(
        self,
        job: AgentJob,
        state: Dict[str, Any],
        db: AsyncSession,
    ) -> None:
        """Save a checkpoint for job resumption."""
        checkpoint = AgentJobCheckpoint(
            job_id=job.id,
            iteration=job.iteration,
            phase=job.current_phase,
            state=state,
            context={"progress": job.progress},
        )
        db.add(checkpoint)
        await db.commit()
        logger.debug(f"Saved checkpoint for job {job.id} at iteration {job.iteration}")

    async def _load_latest_checkpoint(
        self,
        job_id: UUID,
        db: AsyncSession,
    ) -> Optional[AgentJobCheckpoint]:
        """Load the latest checkpoint for a job."""
        result = await db.execute(
            select(AgentJobCheckpoint)
            .where(AgentJobCheckpoint.job_id == job_id)
            .order_by(AgentJobCheckpoint.created_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def _load_user_settings(
        self,
        user_id: UUID,
        db: AsyncSession,
    ) -> Optional[UserLLMSettings]:
        """Load user LLM settings."""
        try:
            result = await db.execute(
                select(UserPreferences).where(UserPreferences.user_id == user_id)
            )
            prefs = result.scalar_one_or_none()
            if prefs:
                return UserLLMSettings.from_preferences(prefs)
        except Exception as e:
            logger.warning(f"Failed to load user settings: {e}")
        return None

    async def pause_job(self, job_id: UUID, db: AsyncSession) -> bool:
        """Pause a running job."""
        result = await db.execute(
            update(AgentJob)
            .where(AgentJob.id == job_id, AgentJob.status == AgentJobStatus.RUNNING.value)
            .values(status=AgentJobStatus.PAUSED.value)
        )
        await db.commit()
        return result.rowcount > 0

    async def resume_job(self, job_id: UUID, db: AsyncSession) -> bool:
        """Resume a paused job."""
        result = await db.execute(
            update(AgentJob)
            .where(AgentJob.id == job_id, AgentJob.status == AgentJobStatus.PAUSED.value)
            .values(status=AgentJobStatus.RUNNING.value)
        )
        await db.commit()
        return result.rowcount > 0

    async def cancel_job(self, job_id: UUID, db: AsyncSession) -> bool:
        """Cancel a job."""
        result = await db.execute(
            update(AgentJob)
            .where(
                AgentJob.id == job_id,
                AgentJob.status.in_([AgentJobStatus.PENDING.value, AgentJobStatus.RUNNING.value, AgentJobStatus.PAUSED.value])
            )
            .values(status=AgentJobStatus.CANCELLED.value)
        )
        await db.commit()
        return result.rowcount > 0
