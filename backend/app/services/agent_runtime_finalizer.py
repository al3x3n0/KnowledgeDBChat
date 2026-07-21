"""Finalization helper for autonomous runtime jobs."""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime
from typing import Any, Dict

from loguru import logger
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.agent_job import AgentJob, AgentJobStatus
from app.services.agent_job_memory_service import agent_job_memory_service


async def finalize_job(executor: Any, job: AgentJob, state: Dict[str, Any], db: AsyncSession) -> Dict[str, Any]:
    """Finalize a runtime job and build the terminal result payload."""
    # Determine final status
    limited, limit_reason = job.is_resource_limited()
    contract_eval = executor._evaluate_goal_contract(job, state)
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
    job.results["source_scope_id"] = executor._resolve_default_source_scope(job)

    # Structured output from set_output_schema tool
    output_schema = state.get("output_schema")
    if isinstance(output_schema, dict) and output_schema:
        job.results["structured_output"] = output_schema

    # Formatted outputs from format_as_table / format_as_report tools
    formatted_outputs = state.get("formatted_outputs", [])
    if isinstance(formatted_outputs, list) and formatted_outputs:
        job.results["formatted_outputs"] = formatted_outputs[-20:]

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
    execution_graph_nodes = (
        state.get("execution_graph_nodes")
        if isinstance(state.get("execution_graph_nodes"), list)
        else []
    )
    execution_graph_edges = (
        state.get("execution_graph_edges")
        if isinstance(state.get("execution_graph_edges"), list)
        else []
    )
    execution_graph_dag_stats = executor._build_execution_graph_stats(
        execution_graph_nodes,
        execution_graph_edges,
    )
    execution_graph_health = executor._build_execution_graph_health(execution_graph_dag_stats)
    execution_graph_recommendations = executor._build_execution_graph_recommendations(execution_graph_health)

    job.results["execution_strategy"] = {
        "execution_mode": str(state.get("execution_mode") or "adaptive"),
        "execution_plan": (state.get("execution_plan") or [])[:12] if isinstance(state.get("execution_plan"), list) else [],
        "plan_step_index": int(state.get("plan_step_index", 0) or 0),
        "plan_completed": bool(state.get("plan_completed", False)),
        "step_events": (state.get("step_events") or [])[-300:] if isinstance(state.get("step_events"), list) else [],
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
            **executor._get_approval_checkpoint_config(job),
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
        "execution_graph": {
            **executor._get_execution_graph_config(job),
            "verification_attempts": int(state.get("verification_attempts", 0) or 0),
            "verification_successes": int(state.get("verification_successes", 0) or 0),
            "summarization_attempts": int(state.get("summarization_attempts", 0) or 0),
            "summarization_successes": int(state.get("summarization_successes", 0) or 0),
            "nodes": execution_graph_nodes[-200:],
            "edges": execution_graph_edges[-400:],
            "dag_stats": execution_graph_dag_stats,
            "graph_health": execution_graph_health,
            "recommended_actions": execution_graph_recommendations,
            "verification_actions": (
                state.get("verification_actions")
                if isinstance(state.get("verification_actions"), list)
                else []
            )[-50:],
            "summarization_actions": (
                state.get("summarization_actions")
                if isinstance(state.get("summarization_actions"), list)
                else []
            )[-50:],
        },
        "tool_stats": state.get("tool_stats") if isinstance(state.get("tool_stats"), dict) else {},
        "tool_priors": state.get("tool_priors") if isinstance(state.get("tool_priors"), dict) else {},
        "scope_guard": {
            **executor._get_scope_guard_config(job),
            "blocks": int(state.get("scope_guard_blocks", 0) or 0),
            "events": (
                state.get("scope_guard_events")
                if isinstance(state.get("scope_guard_events"), list)
                else []
            )[-20:],
        },
        "scope_observability": {
            "resolved_scope_id": executor._resolve_default_source_scope(job),
            "scope_source": executor._resolve_scope_source(job),
            "events": (
                state.get("scope_events")
                if isinstance(state.get("scope_events"), list)
                else []
            )[-100:],
            "event_counts": {
                "resolved_scope": len(
                    [
                        e for e in (state.get("scope_events") or [])
                        if isinstance(e, dict) and str(e.get("type") or "") == "resolved_scope"
                    ]
                ),
                "tool_scope": len(
                    [
                        e for e in (state.get("scope_events") or [])
                        if isinstance(e, dict) and str(e.get("type") or "") == "tool_scope"
                    ]
                ),
                "tool_result_scope": len(
                    [
                        e for e in (state.get("scope_events") or [])
                        if isinstance(e, dict) and str(e.get("type") or "") == "tool_result_scope"
                    ]
                ),
            },
        },
        "tool_selection": {
            **executor._get_tool_selection_config(job),
            "forced_exploration": executor._get_forced_exploration_config(job),
            "cooldown": executor._get_tool_cooldown_config(job),
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
        "memory_persistence": {
            "enabled": bool(job.enable_memory),
            "runtime": (
                state.get("memory_runtime")
                if isinstance(state.get("memory_runtime"), dict)
                else {}
            ),
            "policy": (
                state.get("memory_extraction_policy")
                if isinstance(state.get("memory_extraction_policy"), dict)
                else executor._resolve_memory_extraction_policy(job)
            ),
            "injected_count": len(state.get("injected_memories", []) if isinstance(state.get("injected_memories"), list) else []),
            "injected_memory_ids": (
                [str(x) for x in (state.get("injected_memories") or [])[:20]]
                if isinstance(state.get("injected_memories"), list)
                else []
            ),
            "extraction": (
                state.get("memory_extraction")
                if isinstance(state.get("memory_extraction"), dict)
                else {}
            ),
        },
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

        merged_for_replay = executor._merge_tool_stats(
            state.get("tool_priors") if isinstance(state.get("tool_priors"), dict) else {},
            state.get("tool_stats") if isinstance(state.get("tool_stats"), dict) else {},
        )
        replay = executor.simulate_tool_selection_replay(
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
    if isinstance(state.get("project_profile"), dict) and state.get("project_profile"):
        job.results["project_profile"] = state.get("project_profile")

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

                notes_source = await executor.document_service._get_or_create_agent_notes_source(db)
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
                    await executor.document_service.reprocess_document(doc.id, db, user_id=job.user_id)
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
    final_contract_eval = executor._evaluate_goal_contract(job, state)
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
    job.results["executive_digest"] = executor._build_executive_digest(job, state)

    # Persist tool-learning signal for future jobs.
    try:
        await executor._persist_tool_priors(job, state, db)
    except Exception as e:
        logger.warning(f"Failed to persist tool priors for job {job.id}: {e}")

    # Cleanup data analysis sandbox if used
    job_id_str = str(job.id)
    if job_id_str in executor._data_analysis_tools:
        try:
            from app.services.data_sandbox_service import sandbox_manager
            sandbox_manager.cleanup(job_id_str)
        except Exception as e:
            logger.warning(f"Failed to cleanup data sandbox for job {job.id}: {e}")
        del executor._data_analysis_tools[job_id_str]

    await db.commit()

    # Extract memories from terminal job states based on memory policy.
    memory_policy = (
        state.get("memory_extraction_policy")
        if isinstance(state.get("memory_extraction_policy"), dict)
        else executor._resolve_memory_extraction_policy(job)
    )
    job_status_token = str(job.status or "").strip().lower()
    extract_statuses = (
        [str(x).strip().lower() for x in (memory_policy.get("extract_on_statuses") or []) if str(x).strip()]
        if isinstance(memory_policy, dict)
        else []
    )
    should_extract_memories = (
        bool(job.enable_memory)
        and job.status != AgentJobStatus.PAUSED.value
        and job_status_token in set(extract_statuses)
    )
    if should_extract_memories:
        try:
            allowlist: Optional[List[str]] = None
            if job_status_token == AgentJobStatus.FAILED.value:
                failed_types = memory_policy.get("failed_extraction_types") if isinstance(memory_policy, dict) else None
                if isinstance(failed_types, list):
                    allowlist = [str(x).strip().lower() for x in failed_types if str(x).strip()]
            elif job_status_token == AgentJobStatus.COMPLETED.value:
                completed_types = memory_policy.get("completed_extraction_types") if isinstance(memory_policy, dict) else None
                if isinstance(completed_types, list) and completed_types:
                    allowlist = [str(x).strip().lower() for x in completed_types if str(x).strip()]

            plan_rows = state.get("execution_plan") if isinstance(state.get("execution_plan"), list) else []
            extraction_context = {
                "execution_mode": str(state.get("execution_mode") or "adaptive"),
                "plan_completed": bool(state.get("plan_completed", False)),
                "plan_step_index": int(state.get("plan_step_index", 0) or 0),
                "plan_steps_total": len(plan_rows),
            }
            extraction_stats: Dict[str, Any] = {}
            extracted_memories = await agent_job_memory_service.extract_memories_from_job(
                job=job,
                user_id=str(job.user_id),
                db=db,
                memory_types_allowlist=allowlist,
                context_overrides=extraction_context,
                extraction_reason=f"auto_{job_status_token}",
                stats_out=extraction_stats,
            )
            extraction_summary = {
                "status": "completed",
                "reason": f"auto_{job_status_token}",
                "created_count": len(extracted_memories),
                "allowlist": allowlist[:12] if isinstance(allowlist, list) else [],
                "extracted_types": list(set(str(m.memory_type) for m in extracted_memories))[:12],
                "parsed_count": int(extraction_stats.get("parsed_count", 0) or 0),
                "candidate_count": int(extraction_stats.get("candidate_count", 0) or 0),
                "skipped_duplicates": int(extraction_stats.get("skipped_duplicates", 0) or 0),
                "dedup_existing_signature_count": int(
                    extraction_stats.get("dedup_existing_signature_count", 0) or 0
                ),
                "is_relaunch_chain": bool(extraction_stats.get("is_relaunch_chain", False)),
                "relaunch_root_job_id": (
                    str(extraction_stats.get("relaunch_root_job_id") or "").strip() or None
                ),
                "at": datetime.utcnow().isoformat(),
            }
            state["memory_extraction"] = extraction_summary
            if extracted_memories:
                logger.info(f"Extracted {len(extracted_memories)} memories from job {job.id}")
                job.add_log_entry({
                    "phase": "memory_extraction",
                    "memories_created": len(extracted_memories),
                    "memory_types": list(set(m.memory_type for m in extracted_memories)),
                    "reason": extraction_summary.get("reason"),
                    "skipped_duplicates": int(extraction_summary.get("skipped_duplicates", 0) or 0),
                })
            results_payload = job.results if isinstance(job.results, dict) else {}
            exec_strategy = (
                results_payload.get("execution_strategy")
                if isinstance(results_payload.get("execution_strategy"), dict)
                else {}
            )
            mem_persistence = (
                exec_strategy.get("memory_persistence")
                if isinstance(exec_strategy.get("memory_persistence"), dict)
                else {}
            )
            mem_persistence["policy"] = memory_policy if isinstance(memory_policy, dict) else {}
            mem_persistence["extraction"] = extraction_summary
            mem_persistence["injected_count"] = len(
                state.get("injected_memories", [])
                if isinstance(state.get("injected_memories"), list)
                else []
            )
            exec_strategy["memory_persistence"] = mem_persistence
            results_payload["execution_strategy"] = exec_strategy
            job.results = results_payload
            await db.commit()
        except Exception as e:
            logger.warning(f"Failed to extract memories from job {job.id}: {e}")
            extraction_error = {
                "status": "failed",
                "reason": f"auto_{job_status_token}",
                "error": str(e)[:500],
                "at": datetime.utcnow().isoformat(),
            }
            state["memory_extraction"] = extraction_error
            try:
                results_payload = job.results if isinstance(job.results, dict) else {}
                exec_strategy = (
                    results_payload.get("execution_strategy")
                    if isinstance(results_payload.get("execution_strategy"), dict)
                    else {}
                )
                mem_persistence = (
                    exec_strategy.get("memory_persistence")
                    if isinstance(exec_strategy.get("memory_persistence"), dict)
                    else {}
                )
                mem_persistence["policy"] = memory_policy if isinstance(memory_policy, dict) else {}
                mem_persistence["extraction"] = extraction_error
                exec_strategy["memory_persistence"] = mem_persistence
                results_payload["execution_strategy"] = exec_strategy
                job.results = results_payload
                await db.commit()
            except Exception:
                pass

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
    await executor._trigger_chained_jobs(job, event, db)

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
