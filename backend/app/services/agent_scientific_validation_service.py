"""Orchestration helpers extracted from AutonomousAgentExecutor."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.agent_job import AgentJob, AgentJobStatus
from app.services.project_profile_service import build_project_profile


class AgentScientificValidationService:
    def scientific_validation_context_key(
        self,
        executor: Any,
        *,
        profile_id: Optional[str] = None,
        portfolio_id: Optional[str] = None,
        hypothesis_id: Optional[str] = None,
    ) -> str:
        bits = [
            str(profile_id or "").strip(),
            str(portfolio_id or "").strip(),
            str(hypothesis_id or "").strip(),
        ]
        return "|".join(bits)

    async def update_scientific_validation_summary_links(
        self,
        executor: Any,
        *,
        db: AsyncSession,
        profile_id: Optional[str] = None,
        portfolio_id: Optional[str] = None,
        run_id: Optional[str] = None,
        run_record: Optional[dict[str, Any]] = None,
    ) -> None:
        from app.models.domain_research_profile import DomainResearchProfile
        from app.models.research_portfolio import ResearchPortfolio

        run_id_text = str(run_id or "").strip()
        if not run_id_text:
            return

        if profile_id:
            try:
                profile = await db.get(DomainResearchProfile, UUID(str(profile_id)))
            except Exception:
                profile = None
            if profile is not None:
                rows = [
                    str(v).strip()
                    for v in (profile.latest_validation_run_ids or [])
                    if str(v).strip()
                ]
                if run_id_text not in rows:
                    rows.append(run_id_text)
                profile.latest_validation_run_ids = rows[-20:]
                summary = (
                    profile.latest_summary
                    if isinstance(profile.latest_summary, dict)
                    else {}
                )
                validation_runs = (
                    summary.get("validation_runs")
                    if isinstance(summary.get("validation_runs"), list)
                    else []
                )
                if isinstance(run_record, dict):
                    validation_runs = [
                        row
                        for row in validation_runs
                        if isinstance(row, dict)
                        and str(row.get("run_id") or "").strip() != run_id_text
                    ]
                    validation_runs.append(run_record)
                summary["validation_runs"] = validation_runs[-20:]
                summary["latest_validation_run_ids"] = profile.latest_validation_run_ids
                profile.latest_summary = summary

        if portfolio_id:
            try:
                portfolio = await db.get(ResearchPortfolio, UUID(str(portfolio_id)))
            except Exception:
                portfolio = None
            if portfolio is not None:
                rows = [
                    str(v).strip()
                    for v in (portfolio.latest_validation_run_ids or [])
                    if str(v).strip()
                ]
                if run_id_text not in rows:
                    rows.append(run_id_text)
                portfolio.latest_validation_run_ids = rows[-30:]
                summary = (
                    portfolio.latest_summary
                    if isinstance(portfolio.latest_summary, dict)
                    else {}
                )
                validation_runs = (
                    summary.get("validation_runs")
                    if isinstance(summary.get("validation_runs"), list)
                    else []
                )
                if isinstance(run_record, dict):
                    validation_runs = [
                        row
                        for row in validation_runs
                        if isinstance(row, dict)
                        and str(row.get("run_id") or "").strip() != run_id_text
                    ]
                    validation_runs.append(run_record)
                summary["validation_runs"] = validation_runs[-30:]
                summary[
                    "latest_validation_run_ids"
                ] = portfolio.latest_validation_run_ids
                portfolio.latest_summary = summary

    async def create_scientific_validation_run(
        self,
        executor: Any,
        *,
        db: AsyncSession,
        parent_job: AgentJob,
        experiment_plan: Any,
        track_type: str,
        objective: str,
        hypothesis_title: str,
        hypothesis_text: str,
        validation_policy: dict[str, Any],
        sandbox_profile_id: Optional[str],
        repo_source_ids: list[str],
        benchmark_queries: list[str],
        supporting_evidence: list[str],
        supporting_sources: list[dict[str, Any]],
        profile_id: Optional[str] = None,
        portfolio_id: Optional[str] = None,
        hypothesis_id: Optional[str] = None,
        originating_job_id: Optional[str] = None,
    ) -> dict[str, Any]:
        from app.models.experiment import ExperimentRun
        from app.services.scientific_validation_service import (
            build_scientific_validation_recipe,
            evaluate_scientific_validation_capabilities,
            get_scientific_sandbox_profile,
            get_scientific_validation_runtime_limits,
            normalize_validation_policy,
        )

        normalized_policy = normalize_validation_policy(validation_policy)
        run_status = "blocked"
        source_id = str((repo_source_ids or [None])[0] or "").strip()
        sandbox_profile = await get_scientific_sandbox_profile(
            db, sandbox_profile_id, track_type=track_type
        )
        runtime_limits = get_scientific_validation_runtime_limits()
        context_key = executor._scientific_validation_context_key(
            profile_id=profile_id,
            portfolio_id=portfolio_id,
            hypothesis_id=hypothesis_id,
        )
        decision: dict[str, Any] = {
            "run_id": None,
            "status": "blocked",
            "reason_code": "",
            "source_id": source_id or None,
            "recipe_family": None,
            "sandbox_profile_id": str(
                (sandbox_profile or {}).get("id") or sandbox_profile_id or ""
            ).strip()
            or None,
            "job_id": None,
        }

        if not source_id:
            decision["reason_code"] = "missing_repo_source"
        elif not sandbox_profile:
            decision["reason_code"] = "missing_sandbox_profile"
        else:
            project_profile = await build_project_profile(
                parent_job, db, source_id=source_id, max_files=300
            )
            verification_commands = executor._select_verification_commands_from_profile(
                project_profile, max_commands=4
            )
            retry_profile = executor._get_bootstrap_and_fallback_commands_from_profile(
                project_profile,
                primary_commands=verification_commands,
                max_install=3,
                max_fallback=3,
            )
            recipe = build_scientific_validation_recipe(
                track_type=track_type,
                objective=objective,
                hypothesis_title=hypothesis_title,
                hypothesis_text=hypothesis_text,
                benchmark_queries=benchmark_queries,
                verification_commands=verification_commands,
                bootstrap_commands=retry_profile.get("install")
                if isinstance(retry_profile.get("install"), list)
                else [],
                fallback_commands=retry_profile.get("fallback")
                if isinstance(retry_profile.get("fallback"), list)
                else [],
                supporting_evidence=supporting_evidence,
                supporting_sources=supporting_sources,
            )
            decision["recipe_family"] = (
                str(recipe.get("recipe_family") or "").strip() or None
            )
            capability_check = evaluate_scientific_validation_capabilities(
                source_id=source_id,
                sandbox_profile=sandbox_profile,
                recipe=recipe,
            )
            decision["capability_check"] = capability_check
            budget_limit = min(
                float(normalized_policy.get("max_validation_budget_per_run") or 25.0),
                float(
                    (sandbox_profile or {}).get("budget_limit_default")
                    or runtime_limits["max_budget_per_run"]
                ),
                float(runtime_limits["max_budget_per_run"]),
            )
            timeout_seconds = min(
                int(normalized_policy.get("max_validation_runtime_minutes") or 20) * 60,
                int(
                    (sandbox_profile or {}).get("timeout_seconds")
                    or runtime_limits["max_timeout_seconds"]
                ),
                int(runtime_limits["max_timeout_seconds"]),
            )

            recent_stmt = (
                select(ExperimentRun)
                .where(ExperimentRun.user_id == parent_job.user_id)
                .order_by(ExperimentRun.created_at.desc())
                .limit(80)
            )
            recent_runs = list((await db.execute(recent_stmt)).scalars().all())
            active_count = 0
            consecutive_failures = 0
            latest_failure_at: Optional[datetime] = None
            for recent in recent_runs:
                cfg = recent.config if isinstance(recent.config, dict) else {}
                meta = (
                    cfg.get("scientific_validation")
                    if isinstance(cfg.get("scientific_validation"), dict)
                    else {}
                )
                if str(meta.get("context_key") or "").strip() != context_key:
                    continue
                if recent.status in {"queued", "provisioning", "running"}:
                    active_count += 1
                if recent.status not in {"failed", "blocked"}:
                    break
                consecutive_failures += 1
                if latest_failure_at is None:
                    latest_failure_at = (
                        recent.completed_at or recent.updated_at or recent.created_at
                    )

            backoff = (
                normalized_policy.get("validation_backoff_policy")
                if isinstance(normalized_policy.get("validation_backoff_policy"), dict)
                else {}
            )
            cooldown_minutes = max(
                5, min(int(backoff.get("cooldown_minutes") or 180), 10080)
            )
            max_consecutive_failures = max(
                1, min(int(backoff.get("max_consecutive_failures") or 2), 10)
            )

            recipe_benchmark_family = str(recipe.get("benchmark_family") or "").strip()
            profile_benchmark_families = set(
                str(item).strip()
                for item in (
                    (sandbox_profile or {}).get("allowed_benchmark_families") or []
                )
                if str(item).strip()
            )
            profile_perf_collectors = set(
                str(item).strip()
                for item in (
                    (sandbox_profile or {}).get("allowed_perf_collectors") or []
                )
                if str(item).strip()
            )
            recipe_perf_collectors = set(
                str(item).strip()
                for item in (recipe.get("allowed_perf_collectors") or [])
                if str(item).strip()
            )
            if str(
                (sandbox_profile or {}).get("backend") or ""
            ).strip().lower() not in {"docker", "subprocess"}:
                decision["reason_code"] = "unsupported_backend"
            elif str(
                (sandbox_profile or {}).get("backend") or ""
            ).strip().lower() == "docker" and str(
                (sandbox_profile or {}).get("docker_image") or ""
            ).strip() not in set(
                runtime_limits["allowed_docker_images"]
            ):
                decision["reason_code"] = "disallowed_image"
            elif (
                recipe_benchmark_family
                and recipe_benchmark_family not in profile_benchmark_families
            ):
                decision["reason_code"] = "unsupported_benchmark_family"
            elif recipe_perf_collectors and not recipe_perf_collectors.issubset(
                profile_perf_collectors
            ):
                decision["reason_code"] = "recipe_profile_mismatch"
            elif not bool(capability_check.get("ok")):
                decision["reason_code"] = "missing_capability"
            elif not recipe.get("commands"):
                decision["reason_code"] = "recipe_compile_failed"
            elif active_count >= int(
                normalized_policy.get("max_concurrent_validation_runs") or 1
            ):
                decision["reason_code"] = "concurrency_limit"
            elif (
                consecutive_failures >= max_consecutive_failures
                and latest_failure_at is not None
                and (datetime.utcnow() - latest_failure_at).total_seconds()
                < cooldown_minutes * 60
            ):
                decision["reason_code"] = "backoff_cooldown"
            else:
                run_status = "queued"
                decision["status"] = "queued"

            run_config = {
                "source_id": source_id,
                "commands": recipe.get("commands")
                if isinstance(recipe.get("commands"), list)
                else [],
                "bootstrap_commands": recipe.get("bootstrap_commands")
                if isinstance(recipe.get("bootstrap_commands"), list)
                else [],
                "fallback_commands": recipe.get("fallback_commands")
                if isinstance(recipe.get("fallback_commands"), list)
                else [],
                "timeout_seconds": timeout_seconds,
                "unsafe_code_exec_backend": str(
                    (sandbox_profile or {}).get("backend") or "docker"
                ),
                "unsafe_code_exec_docker_image": str(
                    (sandbox_profile or {}).get("docker_image") or ""
                ),
                "unsafe_code_exec_max_memory_mb": int(
                    (
                        ((sandbox_profile or {}).get("resource_caps") or {}).get(
                            "memory_mb"
                        )
                        or 2048
                    )
                ),
                "unsafe_code_exec_docker_cpus": float(
                    (
                        ((sandbox_profile or {}).get("resource_caps") or {}).get("cpus")
                        or 1.0
                    )
                ),
                "unsafe_code_exec_docker_pids_limit": int(
                    (
                        ((sandbox_profile or {}).get("resource_caps") or {}).get(
                            "pids_limit"
                        )
                        or 128
                    )
                ),
                "execution_handoff": {
                    "execution_handoff_version": 1,
                    "selected_hypothesis_ids": [str(hypothesis_id or "").strip()]
                    if str(hypothesis_id or "").strip()
                    else [],
                    "supporting_sources": supporting_sources[:8],
                    "autonomous_origin": {
                        "source_kind": "profile"
                        if str(profile_id or "").strip()
                        else ("portfolio" if str(portfolio_id or "").strip() else None),
                        "source_id": str(profile_id or portfolio_id or "").strip()
                        or None,
                        "opportunity_id": str(hypothesis_id or "").strip() or None,
                        "evidence_revision_at_launch": str(
                            (
                                (experiment_plan.generator_details or {})
                                if isinstance(experiment_plan.generator_details, dict)
                                else {}
                            ).get("evidence_revision_at_launch")
                            or (
                                (
                                    (
                                        (experiment_plan.plan or {})
                                        if isinstance(experiment_plan.plan, dict)
                                        else {}
                                    ).get("provenance")
                                    or {}
                                ).get("autonomous_origin")
                                or {}
                            ).get("evidence_revision_at_launch")
                            or ""
                        ).strip()
                        or None,
                    },
                },
                "scientific_validation": {
                    "validation_kind": "scientific_validation",
                    "context_key": context_key,
                    "sandbox_profile_id": str((sandbox_profile or {}).get("id") or ""),
                    "profile_snapshot": deepcopy(sandbox_profile),
                    "recipe_family": str(recipe.get("recipe_family") or ""),
                    "recipe_id": str(recipe.get("recipe_id") or ""),
                    "recipe_version": int(recipe.get("recipe_version") or 1),
                    "benchmark_family": str(recipe.get("benchmark_family") or ""),
                    "benchmark_queries": benchmark_queries[:8],
                    "allowed_perf_collectors": recipe.get("allowed_perf_collectors")
                    if isinstance(recipe.get("allowed_perf_collectors"), list)
                    else [],
                    "required_capabilities": recipe.get("required_capabilities")
                    if isinstance(recipe.get("required_capabilities"), list)
                    else [],
                    "capability_check": capability_check,
                    "artifact_collection_rules": recipe.get("artifact_collection_rules")
                    if isinstance(recipe.get("artifact_collection_rules"), list)
                    else [],
                    "success_criteria": recipe.get("success_criteria")
                    if isinstance(recipe.get("success_criteria"), list)
                    else [],
                    "decision_summary": str(recipe.get("decision_summary") or "")[
                        :2000
                    ],
                    "baseline_comparison": recipe.get("baseline_comparison")
                    if isinstance(recipe.get("baseline_comparison"), dict)
                    else {},
                    "domain_research_profile_id": str(profile_id or "").strip() or None,
                    "research_portfolio_id": str(portfolio_id or "").strip() or None,
                    "hypothesis_id": str(hypothesis_id or "").strip() or None,
                    "originating_job_id": str(originating_job_id or parent_job.id),
                    "track_type": str(track_type or "generic"),
                    "budget_limit": budget_limit,
                    "runtime_limit_minutes": max(1, timeout_seconds // 60),
                    "supporting_evidence": supporting_evidence[:8],
                    "supporting_sources": supporting_sources[:8],
                    "recipe_snapshot": deepcopy(recipe),
                    "blocked_reason_code": decision["reason_code"] or None,
                },
            }

            run = ExperimentRun(
                user_id=parent_job.user_id,
                experiment_plan_id=experiment_plan.id,
                name=f"Validation Run: {hypothesis_title[:180]}",
                status=run_status,
                progress=0 if run_status == "queued" else 100,
                config=run_config,
                summary=(
                    str(
                        recipe.get("decision_summary")
                        or f"Validation run for {hypothesis_title}"
                    ).strip()[:20000]
                ),
                completed_at=datetime.utcnow() if run_status == "blocked" else None,
            )
            db.add(run)
            await db.flush()
            decision["run_id"] = str(run.id)

            run_record = {
                "run_id": str(run.id),
                "status": run.status,
                "recipe_family": decision["recipe_family"],
                "sandbox_profile_id": decision["sandbox_profile_id"],
                "reason_code": decision["reason_code"] or None,
                "hypothesis_id": str(hypothesis_id or "").strip() or None,
                "profile_id": str(profile_id or "").strip() or None,
                "portfolio_id": str(portfolio_id or "").strip() or None,
            }
            await executor._update_scientific_validation_summary_links(
                db=db,
                profile_id=profile_id,
                portfolio_id=portfolio_id,
                run_id=str(run.id),
                run_record=run_record,
            )

            if run_status == "queued":
                child_job = AgentJob(
                    name=f"Scientific Validation - {hypothesis_title[:80]}",
                    description="Recipe-backed scientific validation run launched from autonomous research.",
                    job_type="analysis",
                    goal=f"Execute a bounded scientific validation run for '{hypothesis_title}'.",
                    config={
                        **run_config,
                        "deterministic_runner": "experiment_runner",
                        "experiment_plan_id": str(experiment_plan.id),
                        "experiment_run_id": str(run.id),
                    },
                    user_id=parent_job.user_id,
                    status=AgentJobStatus.PENDING.value,
                    parent_job_id=parent_job.id,
                    root_job_id=parent_job.root_job_id or parent_job.id,
                    chain_depth=int(parent_job.chain_depth or 0) + 1,
                    max_iterations=1,
                    max_tool_calls=0,
                    max_llm_calls=0,
                    max_runtime_minutes=max(
                        5,
                        min(
                            int(
                                normalized_policy.get("max_validation_runtime_minutes")
                                or 20
                            ),
                            240,
                        ),
                    ),
                )
                db.add(child_job)
                await db.flush()
                run.agent_job_id = child_job.id
                decision["job_id"] = str(child_job.id)
                run_record["job_id"] = str(child_job.id)
                await executor._update_scientific_validation_summary_links(
                    db=db,
                    profile_id=profile_id,
                    portfolio_id=portfolio_id,
                    run_id=str(run.id),
                    run_record=run_record,
                )
            else:
                run.results = {
                    "scientific_validation": {
                        "status": "blocked",
                        "reason_code": decision["reason_code"],
                        "recipe_family": decision["recipe_family"],
                        "sandbox_profile_id": decision["sandbox_profile_id"],
                        "recipe_id": str(recipe.get("recipe_id") or ""),
                        "recipe_version": int(recipe.get("recipe_version") or 1),
                        "capability_check": capability_check,
                    }
                }

            return decision

        blocked_run = ExperimentRun(
            user_id=parent_job.user_id,
            experiment_plan_id=experiment_plan.id,
            name=f"Validation Run: {hypothesis_title[:180]}",
            status="blocked",
            progress=100,
            config={
                "scientific_validation": {
                    "validation_kind": "scientific_validation",
                    "context_key": context_key,
                    "sandbox_profile_id": decision["sandbox_profile_id"],
                    "domain_research_profile_id": str(profile_id or "").strip() or None,
                    "research_portfolio_id": str(portfolio_id or "").strip() or None,
                    "hypothesis_id": str(hypothesis_id or "").strip() or None,
                    "originating_job_id": str(originating_job_id or parent_job.id),
                    "track_type": str(track_type or "generic"),
                    "blocked_reason_code": decision["reason_code"],
                }
            },
            results={
                "scientific_validation": {
                    "status": "blocked",
                    "reason_code": decision["reason_code"],
                    "sandbox_profile_id": decision["sandbox_profile_id"],
                    "source_id": source_id or None,
                }
            },
            summary=f"Scientific validation blocked: {decision['reason_code'] or 'unknown_reason'}",
            completed_at=datetime.utcnow(),
        )
        db.add(blocked_run)
        await db.flush()
        decision["run_id"] = str(blocked_run.id)
        await executor._update_scientific_validation_summary_links(
            db=db,
            profile_id=profile_id,
            portfolio_id=portfolio_id,
            run_id=str(blocked_run.id),
            run_record={
                "run_id": str(blocked_run.id),
                "status": "blocked",
                "reason_code": decision["reason_code"] or "unknown_reason",
                "sandbox_profile_id": decision["sandbox_profile_id"],
                "hypothesis_id": str(hypothesis_id or "").strip() or None,
                "profile_id": str(profile_id or "").strip() or None,
                "portfolio_id": str(portfolio_id or "").strip() or None,
            },
        )
        return decision
