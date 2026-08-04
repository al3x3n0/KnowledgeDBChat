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

import hashlib
import json
import random
import re
from copy import deepcopy
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from loguru import logger
from sqlalchemy import desc, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.attributes import flag_modified

from app.agent_core.runtime import AgentRuntimeRunner
from app.models.agent_definition import AgentDefinition
from app.models.agent_job import (
    AgentJob,
    AgentJobCheckpoint,
    AgentJobStatus,
    ChainTriggerCondition,
)
from app.models.agent_tool_prior import AgentToolPrior
from app.models.memory import UserPreferences
from app.services import (
    agent_execution_graph,
    agent_plan_normalization,
    agent_prompt_sections,
    agent_tool_scoring,
)
from app.services.agent_action_service import AgentActionService
from app.services.agent_chain_orchestration_service import (
    AgentChainOrchestrationService,
)
from app.services.agent_checkpoint_service import AgentCheckpointService
from app.services.agent_coding_runner_service import AgentCodingRunnerService
from app.services.agent_decision_parser import AgentDecisionParser
from app.services.agent_deterministic_runner_registry import (
    build_deterministic_runner_registry,
)
from app.services.agent_execution_journal_service import agent_execution_journal_service
from app.services.agent_execution_planner import (
    AgentExecutionPlanner,
    ExecutionPlan,
    PlanStep,
)
from app.services.agent_experiment_runner_service import AgentExperimentRunnerService
from app.services.agent_follow_up_job_service import AgentFollowUpJobService
from app.services.agent_goal_contract_service import AgentGoalContractService
from app.services.agent_ingestion_demo_runner_service import (
    AgentIngestionDemoRunnerService,
)
from app.services.agent_job_memory_service import agent_job_memory_service
from app.services.agent_latex_runner_service import AgentLatexRunnerService
from app.services.agent_observation_service import AgentObservationService
from app.services.agent_progress_evaluation_service import (
    AgentProgressEvaluationService,
)
from app.services.agent_research_runner_service import AgentResearchRunnerService
from app.services.agent_runtime_finalizer import finalize_job
from app.services.agent_runtime_policy_service import AgentRuntimePolicyService
from app.services.agent_runtime_state_service import initialize_runtime_state
from app.services.agent_scientific_validation_service import (
    AgentScientificValidationService,
)
from app.services.agent_skill_profile_service import AgentSkillProfileService
from app.services.agent_thinking_service import AgentThinkingService
from app.services.agent_tool_dispatch import (
    AgentToolRegistry,
    build_autonomous_collaboration_provider,
    build_autonomous_data_analysis_provider,
    build_autonomous_document_authoring_provider,
    build_autonomous_document_provider,
    build_autonomous_kg_provider,
    build_autonomous_media_provider,
    build_autonomous_memory_provider,
    build_autonomous_notification_visualization_provider,
    build_autonomous_observability_provider,
    build_autonomous_output_state_provider,
    build_autonomous_project_bootstrap_provider,
    build_autonomous_reasoning_provider,
    build_autonomous_research_provider,
    build_autonomous_scheduling_provider,
    build_autonomous_snapshot_provider,
    build_autonomous_symbol_retrieval_provider,
    build_autonomous_web_research_provider,
    build_autonomous_workflow_provider,
    build_autonomous_workspace_mutation_provider,
    build_autonomous_workspace_read_provider,
)
from app.services.agent_tools import AGENT_TOOLS, AUTONOMOUS_AGENT_TOOLS
from app.services.arxiv_search_service import ArxivSearchService
from app.services.data_analysis_tools import (
    DATA_ANALYSIS_TOOL_DEFINITIONS,
    DataAnalysisTools,
)
from app.services.llm_service import LLMService, UserLLMSettings
from app.services.project_profile_service import (
    build_project_profile,
    format_project_profile_for_prompt,
)
from app.services.search_service import SearchService
from app.services.vector_store import VectorStoreService

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


class _AutonomousRuntimeAdapter:
    """App-layer adapter that exposes executor phases to the core runtime runner."""

    def __init__(
        self,
        *,
        executor: "AutonomousAgentExecutor",
        job: AgentJob,
        agent_def: Optional[AgentDefinition],
        user_settings: Optional[UserLLMSettings],
        state: Dict[str, Any],
        db: AsyncSession,
        start_time: datetime,
        max_runtime: timedelta,
        progress_callback: Optional[callable],
    ) -> None:
        self.executor = executor
        self.job = job
        self.agent_def = agent_def
        self.user_settings = user_settings
        self.state = state
        self.db = db
        self.start_time = start_time
        self.max_runtime = max_runtime
        self.progress_callback = progress_callback
        self.counterfactual_candidates: List[Dict[str, Any]] = []
        self.selection_explainability: Dict[str, Any] = {}

    async def can_continue(self) -> bool:
        if not self.job.can_continue():
            return False
        if datetime.utcnow() - self.start_time > self.max_runtime:
            logger.info(f"Job {self.job.id} hit runtime limit")
            self.job.add_log_entry(
                {
                    "phase": "limit_reached",
                    "reason": "max_runtime_minutes",
                    "runtime_minutes": self.job.max_runtime_minutes,
                }
            )
            return False
        await self.db.refresh(self.job)
        if self.job.status not in [AgentJobStatus.RUNNING.value]:
            logger.info(f"Job {self.job.id} status changed to {self.job.status}")
            return False
        return True

    async def on_iteration_start(self) -> None:
        self.job.iteration += 1
        self.job.last_activity_at = datetime.utcnow()

    async def observe_phase(self) -> Dict[str, Any]:
        observation = await self.executor.observation_service.observe(
            self.executor,
            self.job,
            self.state,
            self.db,
        )
        self.state["observations"].append(observation)
        self.job.current_phase = "observing"
        self.job.phase_details = (
            f"Gathered {len(observation.get('context', []))} context items"
        )
        resolved_scope = self.executor._resolve_default_source_scope(self.job)
        self.executor._append_scope_event(
            self.state,
            {
                "type": "resolved_scope",
                "timestamp": datetime.utcnow().isoformat(),
                "iteration": int(self.job.iteration or 0),
                "source_id": resolved_scope,
                "scope_source": self.executor._resolve_scope_source(self.job),
            },
        )

        used_causal_llm = await self.executor._ensure_causal_experiment_plan(
            job=self.job,
            state=self.state,
            observation=observation,
            user_settings=self.user_settings,
        )
        if used_causal_llm:
            self.job.llm_calls_used += 1

        if self.executor._resolve_execution_mode(
            self.job, state=self.state
        ) == "plan_and_execute" and not self.state.get("execution_plan"):
            self.job.current_phase = "planning"
            self.job.phase_details = "Generating execution plan"
        used_plan_llm = await self.executor._ensure_execution_plan(
            self.job,
            self.agent_def,
            self.state,
            observation,
            self.user_settings,
        )
        if used_plan_llm:
            self.job.llm_calls_used += 1
        self.executor._ensure_subgoals(self.job, self.state)
        self.executor._ensure_swarm_chain_config(self.job, self.state)
        self.executor._ensure_subgoal_chain_config(self.job, self.state)

        if self.executor._should_run_critic(self.job, self.state):
            critic_note = await self.executor._run_critic_pass(
                self.job, self.state, observation, self.user_settings
            )
            if critic_note:
                notes = self.state.get("critic_notes")
                if not isinstance(notes, list):
                    notes = []
                notes.append(critic_note)
                max_notes = int(
                    self.executor._get_critic_config(self.job).get("max_notes", 6)
                )
                self.state["critic_notes"] = notes[-max(1, max_notes) :]
                self.state["last_critic_iteration"] = int(self.job.iteration or 0)
                self.job.llm_calls_used += 1
                trigger_info = (
                    self.state.get("critic_last_trigger")
                    if isinstance(self.state.get("critic_last_trigger"), dict)
                    else {}
                )
                self.job.add_log_entry(
                    {
                        "phase": "critic_pass",
                        "assessment": str(
                            critic_note.get("trajectory_assessment") or ""
                        )[:200],
                        "pivot": str(critic_note.get("pivot") or "")[:200],
                        "recommended_tools": critic_note.get("recommended_tools") or [],
                        "trigger_reason": str(trigger_info.get("reason") or ""),
                        "trigger_by_interval": bool(
                            trigger_info.get("by_interval", False)
                        ),
                        "trigger_by_stall": bool(trigger_info.get("by_stall", False)),
                        "trigger_by_uncertainty": bool(
                            trigger_info.get("by_uncertainty", False)
                        ),
                        "uncertainty_score_gap": trigger_info.get(
                            "uncertainty_score_gap"
                        ),
                        "uncertainty_effective_threshold": trigger_info.get(
                            "uncertainty_effective_threshold"
                        ),
                    }
                )

        if self.state.get("execution_plan") and not self.state.get("plan_completed"):
            replan_trigger = self.executor.planner.evaluate_replan_triggers(
                self.job,
                self.state,
                config=self.job.config if isinstance(self.job.config, dict) else None,
            )
            if replan_trigger:
                available_tools = self.executor._get_tools_for_job_type(
                    self.job.job_type,
                    self.job.config,
                    profile=self.state.get("skill_profile")
                    if isinstance(self.state.get("skill_profile"), dict)
                    else None,
                )
                revised = await self.executor.planner.replan(
                    job=self.job,
                    state=self.state,
                    observation=observation,
                    user_settings=self.user_settings,
                    available_tools=available_tools,
                    routing=self.executor._llm_routing_from_job_config(self.job.config),
                    trigger_reason=replan_trigger,
                )
                self.executor._apply_revised_plan(self.state, revised)
                self.job.llm_calls_used += 1
                self.job.add_log_entry(
                    {
                        "phase": "replan",
                        "trigger": replan_trigger,
                        "version": revised.version,
                        "steps_total": len(revised.steps),
                    }
                )

        return observation

    async def think_phase(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        decision = await self.executor.thinking_service.think(
            self.executor,
            self.job,
            self.agent_def,
            self.state,
            observation,
            self.user_settings,
            self.db,
        )
        decision = self.executor._maybe_apply_critic_pivot_override(
            self.job, self.state, decision
        )
        self.job.current_phase = "thinking"
        self.job.phase_details = decision.get("reasoning", "")[:200]
        self.job.llm_calls_used += 1

        contract_before = self.executor._evaluate_goal_contract(
            self.job, self.state, include_result_keys=False
        )
        self.state["goal_contract_last"] = contract_before

        if decision.get("goal_achieved"):
            if (
                self.executor._resolve_execution_mode(self.job, state=self.state)
                == "plan_and_execute"
                and self.state.get("execution_plan")
                and not self.executor._is_execution_plan_complete(self.state)
            ):
                decision["goal_achieved"] = False
                decision["reasoning"] = (
                    f"{str(decision.get('reasoning') or '').strip()[:260]} "
                    "Plan-and-execute mode requires completing the active execution plan before final stop."
                ).strip()
                self.job.add_log_entry(
                    {"phase": "goal_achieved_blocked", "reason": "plan_not_completed"}
                )
            elif bool(contract_before.get("enabled")) and not bool(
                contract_before.get("satisfied")
            ):
                unmet = (
                    contract_before.get("missing")
                    if isinstance(contract_before.get("missing"), list)
                    else []
                )
                decision["goal_achieved"] = False
                decision["reasoning"] = (
                    f"{str(decision.get('reasoning') or '').strip()[:260]} "
                    f"Goal contract not yet satisfied: {', '.join([str(x)[:80] for x in unmet[:4]])}"
                ).strip()
                self.job.add_log_entry(
                    {
                        "phase": "goal_contract_blocked",
                        "reasoning": "goal_achieved blocked by unmet goal contract",
                        "missing": unmet[:8],
                    }
                )
            else:
                if bool(contract_before.get("enabled")) and not int(
                    self.state.get("goal_contract_satisfied_iteration", 0) or 0
                ):
                    self.state["goal_contract_satisfied_iteration"] = int(
                        self.job.iteration or 0
                    )
                logger.info(f"Job {self.job.id} achieved goal")
                self.job.add_log_entry(
                    {
                        "phase": "goal_achieved",
                        "reasoning": decision.get("reasoning"),
                        "final_assessment": decision.get("assessment"),
                    }
                )
                self.state["goal_progress"] = 100

        if decision.get("should_stop"):
            logger.info(
                f"Job {self.job.id} decided to stop: {decision.get('stop_reason')}"
            )
            self.job.add_log_entry(
                {"phase": "voluntary_stop", "reason": decision.get("stop_reason")}
            )

        return decision

    async def act_phase(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        self.counterfactual_candidates = []
        self.selection_explainability = {}
        cf_cfg = self.executor._get_counterfactual_config(self.job)
        if bool(cf_cfg.get("enabled", True)):
            self.counterfactual_candidates = (
                self.executor._build_counterfactual_candidates(
                    job=self.job,
                    state=self.state,
                    selected_tool=str(
                        ((decision.get("action") or {}).get("tool") or "")
                    ).strip()
                    or None,
                    limit=int(cf_cfg.get("top_k", 3) or 3),
                    context_tag="iteration_decision",
                )
            )
            self.state["counterfactual_last"] = self.counterfactual_candidates
            self.state["counterfactual_logged_iterations"] = (
                int(self.state.get("counterfactual_logged_iterations", 0) or 0) + 1
            )
            self.state["counterfactual_last_iteration"] = int(self.job.iteration or 0)
        self.selection_explainability = self.executor._build_selection_explainability(
            state=self.state,
            selected_tool=str(
                ((decision.get("action") or {}).get("tool") or "")
            ).strip()
            or None,
            candidates=self.counterfactual_candidates,
        )
        self.state["selection_explainability_last"] = self.selection_explainability
        self.state["selection_explainability_logged_iterations"] = (
            int(self.state.get("selection_explainability_logged_iterations", 0) or 0)
            + 1
        )

        action = decision.get("action")
        action_result = None
        verification_action = None
        verification_result = None
        summarize_action = None
        summarize_result = None
        checkpoint_override_applied = False
        approved_override = self.state.get("approval_override_action")
        if isinstance(approved_override, dict) and approved_override:
            action = approved_override
            decision["action"] = approved_override
            self.state["approval_override_action"] = None
            checkpoint_override_applied = True
            self.executor._append_step_event(
                self.state,
                {
                    "type": "checkpoint_override_applied",
                    "iteration": int(self.job.iteration or 0),
                    "tool": str((approved_override.get("tool") or "")).strip() or None,
                },
            )
            self.job.add_log_entry(
                {
                    "phase": "approval_override_applied",
                    "tool": str((approved_override.get("tool") or "")).strip(),
                }
            )
        if action:
            if not checkpoint_override_applied:
                action = self.executor._enforce_plan_step_action(
                    self.job, self.state, action
                )
            decision["action"] = action
            plan_rows = (
                self.state.get("execution_plan")
                if isinstance(self.state.get("execution_plan"), list)
                else []
            )
            plan_idx = int(self.state.get("plan_step_index", 0) or 0)
            plan_idx = max(0, min(plan_idx, len(plan_rows) - 1)) if plan_rows else 0
            active_step = (
                plan_rows[plan_idx]
                if plan_rows and isinstance(plan_rows[plan_idx], dict)
                else {}
            )
            active_step_id = str(
                active_step.get("step_id") or f"step_{plan_idx + 1}"
            ).strip()

            effective_action = self.executor._apply_default_scope_to_action(
                dict(action), self.job
            )
            req_params = (
                action.get("params") if isinstance(action.get("params"), dict) else {}
            )
            eff_params = (
                effective_action.get("params")
                if isinstance(effective_action.get("params"), dict)
                else {}
            )
            self.executor._append_scope_event(
                self.state,
                {
                    "type": "tool_scope",
                    "timestamp": datetime.utcnow().isoformat(),
                    "iteration": int(self.job.iteration or 0),
                    "tool": str(action.get("tool") or ""),
                    "requested_source_id": str(
                        req_params.get("source_id") or ""
                    ).strip()
                    or None,
                    "effective_source_id": str(
                        eff_params.get("source_id") or ""
                    ).strip()
                    or None,
                    "scope_source": self.executor._resolve_scope_source(self.job),
                },
            )
            self.state["approval_checkpoint_pending"] = None
            checkpoint_gate = self.executor._evaluate_approval_checkpoint(
                self.job, self.state, action
            )
            if bool(checkpoint_gate.get("required", False)):
                checkpoint_payload = (
                    checkpoint_gate.get("checkpoint")
                    if isinstance(checkpoint_gate.get("checkpoint"), dict)
                    else {}
                )
                checkpoint_payload["plan_step_id"] = active_step_id
                checkpoint_payload["plan_step_index"] = int(plan_idx)
                if isinstance(active_step, dict):
                    active_step["status"] = "waiting_approval"
                    active_step["waiting_since"] = datetime.utcnow().isoformat()
                    active_step["pending_action"] = {
                        "tool": str((action.get("tool") or "")).strip()
                    }
                self.executor._append_step_event(
                    self.state,
                    {
                        "type": "checkpoint_waiting",
                        "iteration": int(self.job.iteration or 0),
                        "plan_step_id": active_step_id,
                        "plan_step_index": int(plan_idx),
                        "tool": str((action.get("tool") or "")).strip() or None,
                        "reason": str(
                            (checkpoint_payload.get("message") or "")
                        ).strip()[:260],
                    },
                )
                self.state["approval_checkpoint_pending"] = checkpoint_payload
                events = self.state.get("approval_checkpoint_events")
                if not isinstance(events, list):
                    events = []
                events.append(checkpoint_payload)
                self.state["approval_checkpoint_events"] = events[-20:]
                results_payload = (
                    self.job.results if isinstance(self.job.results, dict) else {}
                )
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
                approval_summary["events"] = self.state["approval_checkpoint_events"][
                    -20:
                ]
                approval_summary["seen"] = (
                    self.state.get("approval_checkpoint_seen")
                    if isinstance(self.state.get("approval_checkpoint_seen"), list)
                    else []
                )[-200:]
                exec_strategy["approval_checkpoints"] = approval_summary
                results_payload["approval_checkpoint"] = checkpoint_payload
                self.job.results = results_payload
                self.executor._persist_runtime_execution_strategy(self.job, self.state)
                self.job.status = AgentJobStatus.PAUSED.value
                self.job.current_phase = "awaiting_approval"
                self.job.phase_details = str(
                    checkpoint_payload.get("message")
                    or "Approval required before next action."
                )[:280]
                self.job.add_log_entry(
                    {"phase": "approval_checkpoint", "checkpoint": checkpoint_payload}
                )
                await self.executor._save_checkpoint(self.job, self.state, self.db)
                await self.db.commit()
                if self.progress_callback:
                    self.executor._persist_runtime_execution_strategy(
                        self.job, self.state
                    )
                    exec_runtime = (
                        ((self.job.results or {}).get("execution_strategy") or {}).get(
                            "execution_graph_runtime"
                        )
                        if isinstance(
                            (self.job.results or {}).get("execution_strategy"), dict
                        )
                        else {}
                    )
                    scope_runtime = (
                        ((self.job.results or {}).get("execution_strategy") or {}).get(
                            "scope_observability_runtime"
                        )
                        if isinstance(
                            (self.job.results or {}).get("execution_strategy"), dict
                        )
                        else {}
                    )
                    await self.progress_callback(
                        {
                            "job_id": str(self.job.id),
                            "iteration": self.job.iteration,
                            "progress": int(self.state.get("goal_progress", 0) or 0),
                            "phase": self.job.current_phase,
                            "phase_details": self.job.phase_details,
                            "execution_graph_runtime": exec_runtime,
                            "scope_observability_runtime": scope_runtime,
                            "approval_checkpoint": checkpoint_payload,
                        }
                    )
                return {
                    "terminal_result": {
                        "status": self.job.status,
                        "progress": int(self.state.get("goal_progress", 0) or 0),
                        "results": self.job.results
                        if isinstance(self.job.results, dict)
                        else {},
                        "iterations": self.job.iteration,
                        "tool_calls": self.job.tool_calls_used,
                        "llm_calls": self.job.llm_calls_used,
                        "checkpoint": checkpoint_payload,
                    }
                }

            action_result = await self.executor.action_service.act(
                self.executor,
                self.job,
                action,
                self.state,
                self.db,
            )
            self.state["actions_taken"].append(
                {
                    "action": action,
                    "result": action_result,
                    "iteration": self.job.iteration,
                    "node": "act",
                    "step_id": active_step_id,
                }
            )
            self.executor._append_execution_graph_node(
                self.state,
                {
                    "id": active_step_id,
                    "type": "act",
                    "iteration": int(self.job.iteration or 0),
                    "tool": str(action.get("tool") or ""),
                    "success": bool(action_result.get("success", False)),
                },
            )
            self.executor._append_scope_event(
                self.state,
                {
                    "type": "tool_result_scope",
                    "timestamp": datetime.utcnow().isoformat(),
                    "iteration": int(self.job.iteration or 0),
                    "tool": str(action.get("tool") or ""),
                    "success": bool(action_result.get("success", False)),
                    "blocked_by_scope_guard": bool(action_result.get("scope_guard")),
                    "error": str(action_result.get("error") or "")[:260]
                    if action_result.get("error")
                    else "",
                },
            )
            self.job.current_phase = "acting"
            self.job.phase_details = f"Executed: {action.get('tool', 'unknown')}"
            self.job.tool_calls_used += 1
            if action_result.get("findings"):
                self.state["findings"].extend(action_result["findings"])
            if action_result.get("artifacts"):
                self.state["artifacts"].extend(action_result["artifacts"])
            self.executor._record_tool_outcome(self.state, action, action_result)
            self.executor._update_skill_profile_metrics(
                self.state, action, action_result
            )

            if bool(action_result.get("deferred_external")):
                await self.executor.checkpoint_service.save_checkpoint(
                    job=self.job,
                    state=self.state,
                    db=self.db,
                    reason="waiting_external",
                )
                return {
                    "terminal_result": {
                        "status": self.job.status,
                        "progress": int(self.state.get("goal_progress", 0) or 0),
                        "results": self.job.results
                        if isinstance(self.job.results, dict)
                        else {},
                        "iterations": self.job.iteration,
                        "tool_calls": self.job.tool_calls_used,
                        "llm_calls": self.job.llm_calls_used,
                        "external_call": action_result.get("data"),
                    }
                }

            graph_cfg = self.executor._get_execution_graph_config(self.job)
            verify_on_tools = set(
                [
                    str(x).strip()
                    for x in (graph_cfg.get("verify_on_tools") or [])
                    if str(x).strip()
                ]
            )
            primary_tool = str(action.get("tool") or "").strip()
            should_verify = (
                bool(graph_cfg.get("enabled", True))
                and bool(graph_cfg.get("verify_enabled", True))
                and bool(action_result.get("success", False))
                and primary_tool in verify_on_tools
            )
            if should_verify and self.job.tool_calls_used < self.job.max_tool_calls:
                verification_action = self.executor._build_verification_action(
                    self.job, action, action_result
                )
                if verification_action:
                    verify_node_id = f"{active_step_id}.verify"
                    verification_result = await self.executor.action_service.act(
                        self.executor,
                        self.job,
                        verification_action,
                        self.state,
                        self.db,
                    )
                    self.state["actions_taken"].append(
                        {
                            "action": verification_action,
                            "result": verification_result,
                            "iteration": self.job.iteration,
                            "node": "verify",
                            "parent_tool": primary_tool,
                            "step_id": verify_node_id,
                            "depends_on": [active_step_id],
                        }
                    )
                    self.executor._append_execution_graph_node(
                        self.state,
                        {
                            "id": verify_node_id,
                            "type": "verify",
                            "iteration": int(self.job.iteration or 0),
                            "tool": str(verification_action.get("tool") or ""),
                            "success": bool(verification_result.get("success", False)),
                        },
                    )
                    self.executor._append_execution_graph_edge(
                        self.state,
                        {
                            "from": active_step_id,
                            "to": verify_node_id,
                            "type": "verify_after",
                            "iteration": int(self.job.iteration or 0),
                        },
                    )
                    self.job.tool_calls_used += 1
                    self.state["verification_attempts"] = (
                        int(self.state.get("verification_attempts", 0) or 0) + 1
                    )
                    if bool(verification_result.get("success", False)):
                        self.state["verification_successes"] = (
                            int(self.state.get("verification_successes", 0) or 0) + 1
                        )
                    if verification_result.get("findings"):
                        self.state["findings"].extend(verification_result["findings"])
                    if verification_result.get("artifacts"):
                        self.state["artifacts"].extend(verification_result["artifacts"])
                    self.executor._record_tool_outcome(
                        self.state, verification_action, verification_result
                    )
                    self.executor._update_skill_profile_metrics(
                        self.state, verification_action, verification_result
                    )
                    ver_rows = self.state.get("verification_actions")
                    if not isinstance(ver_rows, list):
                        ver_rows = []
                    ver_rows.append(
                        {
                            "iteration": int(self.job.iteration or 0),
                            "parent_tool": primary_tool,
                            "tool": str(verification_action.get("tool") or ""),
                            "success": bool(verification_result.get("success", False)),
                        }
                    )
                    self.state["verification_actions"] = ver_rows[-200:]
                    self.job.add_log_entry(
                        {
                            "phase": "verify_node",
                            "parent_tool": primary_tool,
                            "tool": str(verification_action.get("tool") or ""),
                            "success": bool(verification_result.get("success", False)),
                        }
                    )

            summarize_every = int(graph_cfg.get("summarize_every_n_iterations", 1) or 1)
            should_summarize = (
                bool(graph_cfg.get("enabled", True))
                and bool(graph_cfg.get("summarize_enabled", True))
                and bool(action_result.get("success", False))
                and (int(self.job.iteration or 0) % max(1, summarize_every) == 0)
            )
            if should_summarize and self.job.tool_calls_used < self.job.max_tool_calls:
                summarize_action = self.executor._build_summarize_action(
                    self.job,
                    self.state,
                    action,
                    action_result,
                    verification_action,
                    verification_result,
                )
                if summarize_action:
                    summarize_node_id = f"{active_step_id}.summarize"
                    summarize_dep = (
                        f"{active_step_id}.verify"
                        if isinstance(verification_action, dict)
                        else active_step_id
                    )
                    summarize_result = await self.executor.action_service.act(
                        self.executor,
                        self.job,
                        summarize_action,
                        self.state,
                        self.db,
                    )
                    self.state["actions_taken"].append(
                        {
                            "action": summarize_action,
                            "result": summarize_result,
                            "iteration": self.job.iteration,
                            "node": "summarize",
                            "parent_tool": primary_tool,
                            "step_id": summarize_node_id,
                            "depends_on": [summarize_dep],
                        }
                    )
                    self.executor._append_execution_graph_node(
                        self.state,
                        {
                            "id": summarize_node_id,
                            "type": "summarize",
                            "iteration": int(self.job.iteration or 0),
                            "tool": str(summarize_action.get("tool") or ""),
                            "success": bool(summarize_result.get("success", False)),
                        },
                    )
                    self.executor._append_execution_graph_edge(
                        self.state,
                        {
                            "from": summarize_dep,
                            "to": summarize_node_id,
                            "type": "summarize_after",
                            "iteration": int(self.job.iteration or 0),
                        },
                    )
                    self.job.tool_calls_used += 1
                    self.state["summarization_attempts"] = (
                        int(self.state.get("summarization_attempts", 0) or 0) + 1
                    )
                    if bool(summarize_result.get("success", False)):
                        self.state["summarization_successes"] = (
                            int(self.state.get("summarization_successes", 0) or 0) + 1
                        )
                    if summarize_result.get("findings"):
                        self.state["findings"].extend(summarize_result["findings"])
                    if summarize_result.get("artifacts"):
                        self.state["artifacts"].extend(summarize_result["artifacts"])
                    self.executor._record_tool_outcome(
                        self.state, summarize_action, summarize_result
                    )
                    self.executor._update_skill_profile_metrics(
                        self.state, summarize_action, summarize_result
                    )
                    sum_rows = self.state.get("summarization_actions")
                    if not isinstance(sum_rows, list):
                        sum_rows = []
                    sum_rows.append(
                        {
                            "iteration": int(self.job.iteration or 0),
                            "parent_tool": primary_tool,
                            "tool": str(summarize_action.get("tool") or ""),
                            "success": bool(summarize_result.get("success", False)),
                        }
                    )
                    self.state["summarization_actions"] = sum_rows[-200:]
                    self.job.add_log_entry(
                        {
                            "phase": "summarize_node",
                            "parent_tool": primary_tool,
                            "tool": str(summarize_action.get("tool") or ""),
                            "success": bool(summarize_result.get("success", False)),
                        }
                    )

        return {
            "action": action,
            "action_result": action_result,
            "verification_action": verification_action,
            "verification_result": verification_result,
            "summarize_action": summarize_action,
            "summarize_result": summarize_result,
        }

    async def evaluate_phase(
        self, decision: Dict[str, Any], action_bundle: Dict[str, Any]
    ) -> Dict[str, Any]:
        action = action_bundle.get("action")
        action_result = action_bundle.get("action_result")
        previous_progress = int(self.state.get("goal_progress", 0) or 0)
        progress = await self.executor.progress_evaluation_service.evaluate_progress(
            self.executor,
            self.job,
            self.state,
            self.user_settings,
            self.db,
        )
        self.state["goal_progress"] = progress
        self.job.progress = progress
        self.job.llm_calls_used += 1
        self.executor._advance_execution_plan_state(
            state=self.state,
            action=action,
            action_result=action_result,
            previous_progress=previous_progress,
            current_progress=progress,
            iteration=int(self.job.iteration or 0),
        )

        contract_after = self.executor._evaluate_goal_contract(
            self.job, self.state, include_result_keys=False
        )
        self.state["goal_contract_last"] = contract_after
        if bool(contract_after.get("enabled")) and bool(
            contract_after.get("satisfied")
        ):
            if not int(self.state.get("goal_contract_satisfied_iteration", 0) or 0):
                self.state["goal_contract_satisfied_iteration"] = int(
                    self.job.iteration or 0
                )
                self.job.add_log_entry(
                    {
                        "phase": "goal_contract_satisfied",
                        "iteration": int(self.job.iteration or 0),
                    }
                )
            contract_cfg = (
                contract_after.get("contract")
                if isinstance(contract_after.get("contract"), dict)
                else {}
            )
            if bool(contract_cfg.get("auto_complete_when_satisfied", True)):
                self.state["goal_progress"] = 100
                self.job.progress = 100
                self.job.add_log_entry(
                    {
                        "phase": "goal_contract_autocomplete",
                        "reason": "deterministic goal contract satisfied",
                    }
                )
                return {
                    "progress": 100,
                    "should_stop": True,
                    "stop_reason": "goal_contract_autocomplete",
                }

        stall_info = self.executor._update_stall_state(
            job=self.job,
            state=self.state,
            progress=progress,
            action=action,
        )

        recovery_triggered = False
        if stall_info.get("should_recover"):
            recovery_budget = int(
                self.executor._get_stall_config(self.job).get("max_recovery_actions", 0)
            )
            used_recoveries = int(self.state.get("recovery_actions_used", 0) or 0)
            if (
                used_recoveries < recovery_budget
                and self.job.tool_calls_used < self.job.max_tool_calls
            ):
                recovery_action = self.executor._build_recovery_action(
                    self.job, self.state, exclude_tool=(action or {}).get("tool")
                )
                if recovery_action:
                    effective_recovery = self.executor._apply_default_scope_to_action(
                        dict(recovery_action), self.job
                    )
                    rec_req_params = (
                        recovery_action.get("params")
                        if isinstance(recovery_action.get("params"), dict)
                        else {}
                    )
                    rec_eff_params = (
                        effective_recovery.get("params")
                        if isinstance(effective_recovery.get("params"), dict)
                        else {}
                    )
                    self.executor._append_scope_event(
                        self.state,
                        {
                            "type": "tool_scope",
                            "timestamp": datetime.utcnow().isoformat(),
                            "iteration": int(self.job.iteration or 0),
                            "tool": str(recovery_action.get("tool") or ""),
                            "requested_source_id": str(
                                rec_req_params.get("source_id") or ""
                            ).strip()
                            or None,
                            "effective_source_id": str(
                                rec_eff_params.get("source_id") or ""
                            ).strip()
                            or None,
                            "scope_source": self.executor._resolve_scope_source(
                                self.job
                            ),
                            "recovery": True,
                        },
                    )
                    recovery_result = await self.executor.action_service.act(
                        self.executor,
                        self.job,
                        recovery_action,
                        self.state,
                        self.db,
                    )
                    self.state["actions_taken"].append(
                        {
                            "action": recovery_action,
                            "result": recovery_result,
                            "iteration": self.job.iteration,
                        }
                    )
                    self.executor._append_scope_event(
                        self.state,
                        {
                            "type": "tool_result_scope",
                            "timestamp": datetime.utcnow().isoformat(),
                            "iteration": int(self.job.iteration or 0),
                            "tool": str(recovery_action.get("tool") or ""),
                            "success": bool(recovery_result.get("success", False)),
                            "blocked_by_scope_guard": bool(
                                recovery_result.get("scope_guard")
                            ),
                            "error": str(recovery_result.get("error") or "")[:260]
                            if recovery_result.get("error")
                            else "",
                            "recovery": True,
                        },
                    )
                    recovery_triggered = True
                    self.job.tool_calls_used += 1
                    self.state["recovery_actions_used"] = used_recoveries + 1
                    if recovery_result.get("findings"):
                        self.state["findings"].extend(recovery_result["findings"])
                    if recovery_result.get("artifacts"):
                        self.state["artifacts"].extend(recovery_result["artifacts"])
                    self.executor._record_tool_outcome(
                        self.state, recovery_action, recovery_result
                    )
                    self.executor._update_skill_profile_metrics(
                        self.state, recovery_action, recovery_result
                    )
                    self.executor._apply_recovery_post_action_updates(
                        job=self.job,
                        state=self.state,
                        recovery_action=recovery_action,
                        recovery_result=recovery_result,
                    )
                    self.job.add_log_entry(
                        {
                            "phase": "stall_recovery",
                            "trigger_reason": stall_info.get("reason"),
                            "recovery_action": recovery_action.get("tool"),
                            "recovery_success": bool(recovery_result.get("success")),
                            "forced_exploration": bool(
                                self.state.get(
                                    "last_recovery_was_forced_exploration", False
                                )
                            ),
                            "forced_exploration_attempts": int(
                                self.state.get("forced_exploration_attempts", 0) or 0
                            ),
                            "forced_exploration_successes": int(
                                self.state.get("forced_exploration_successes", 0) or 0
                            ),
                            "forced_exploration_failures": int(
                                self.state.get("forced_exploration_failures", 0) or 0
                            ),
                            "forced_exploration_used": int(
                                self.state.get("forced_exploration_used", 0) or 0
                            ),
                            "tool_cooldown_blocks": int(
                                self.state.get("tool_cooldown_blocks", 0) or 0
                            ),
                        }
                    )
                    self.state["stalled_iterations"] = max(
                        0, int(self.state.get("stalled_iterations", 0)) - 1
                    )

        if stall_info.get("should_stop") and not recovery_triggered:
            logger.info(
                f"Job {self.job.id} stopping due to stall: {stall_info.get('reason')}"
            )
            self.job.add_log_entry(
                {"phase": "voluntary_stop", "reason": stall_info.get("reason")}
            )
            return {
                "progress": progress,
                "should_stop": True,
                "stop_reason": str(stall_info.get("reason") or ""),
            }

        return {"progress": progress, "should_stop": False}

    async def on_iteration_complete(
        self,
        observation: Dict[str, Any],
        decision: Dict[str, Any],
        action_bundle: Dict[str, Any],
        evaluation: Dict[str, Any],
    ) -> None:
        action = action_bundle.get("action")
        verification_action = action_bundle.get("verification_action")
        verification_result = action_bundle.get("verification_result")
        summarize_action = action_bundle.get("summarize_action")
        summarize_result = action_bundle.get("summarize_result")
        progress = int((evaluation or {}).get("progress", 0) or 0)

        findings_count = len(self.state["findings"])
        self.job.add_log_entry(
            {
                "phase": "iteration_complete",
                "action": action.get("tool") if isinstance(action, dict) else None,
                "progress": progress,
                "findings_count": findings_count,
                "verify_tool": verification_action.get("tool")
                if isinstance(verification_action, dict)
                else None,
                "verify_success": bool(
                    (verification_result or {}).get("success", False)
                )
                if isinstance(verification_result, dict)
                else None,
                "summarize_tool": summarize_action.get("tool")
                if isinstance(summarize_action, dict)
                else None,
                "summarize_success": bool(
                    (summarize_result or {}).get("success", False)
                )
                if isinstance(summarize_result, dict)
                else None,
                "plan_step_index": int(self.state.get("plan_step_index", 0) or 0),
                "plan_steps_total": len(self.state.get("execution_plan", []) or []),
                "stalled_iterations": int(self.state.get("stalled_iterations", 0) or 0),
                "repeated_action_iterations": int(
                    self.state.get("repeated_action_iterations", 0) or 0
                ),
                "counterfactual_candidates": self.counterfactual_candidates[:5],
                "selection_explainability": self.selection_explainability,
            }
        )

        await self.executor.trigger_progress_chain(
            self.job, progress, findings_count, self.db
        )

        if self.job.iteration % 5 == 0:
            self.executor._persist_runtime_execution_strategy(self.job, self.state)
            await self.executor._save_checkpoint(self.job, self.state, self.db)

        if self.progress_callback:
            exec_runtime = self.executor._get_execution_graph_runtime_snapshot(
                self.state
            )
            scope_runtime = {
                "resolved_scope_id": self.executor._resolve_default_source_scope(
                    self.job
                ),
                "scope_source": self.executor._resolve_scope_source(self.job),
                "events": (
                    self.state.get("scope_events")
                    if isinstance(self.state.get("scope_events"), list)
                    else []
                )[-25:],
            }
            await self.progress_callback(
                {
                    "job_id": str(self.job.id),
                    "iteration": self.job.iteration,
                    "progress": progress,
                    "phase": self.job.current_phase,
                    "phase_details": self.job.phase_details,
                    "execution_graph_runtime": exec_runtime,
                    "scope_observability_runtime": scope_runtime,
                }
            )

        await self.db.commit()

    async def on_iteration_error(self, exc: Exception) -> bool:
        logger.error(f"Error in iteration {self.job.iteration}: {exc}")
        self.job.add_log_entry({"phase": "error", "error": str(exc)})
        self.job.error_count += 1
        self.job.last_error_at = datetime.utcnow()
        if self.job.error_count >= 5:
            self.job.error = f"Too many errors: {exc}"
            return False
        return True

    async def build_run_result(self) -> Dict[str, Any]:
        return await self.executor._finalize_job(self.job, self.state, self.db)


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
        self.arxiv_search_service = self.arxiv_service
        self._document_service = None
        self.vector_store = VectorStoreService()
        self.decision_parser = AgentDecisionParser(self.llm_service)
        self.planner = AgentExecutionPlanner(self.llm_service)
        self.tool_registry = AgentToolRegistry(
            [
                build_autonomous_document_provider(self),
                build_autonomous_research_provider(self),
                build_autonomous_data_analysis_provider(self),
                build_autonomous_memory_provider(self),
                build_autonomous_workflow_provider(self),
                build_autonomous_reasoning_provider(self),
                build_autonomous_collaboration_provider(self),
                build_autonomous_workspace_read_provider(self),
                build_autonomous_workspace_mutation_provider(self),
                build_autonomous_symbol_retrieval_provider(self),
                build_autonomous_document_authoring_provider(self),
                build_autonomous_observability_provider(self),
                build_autonomous_output_state_provider(self),
                build_autonomous_web_research_provider(self),
                build_autonomous_notification_visualization_provider(self),
                build_autonomous_kg_provider(self),
                build_autonomous_scheduling_provider(self),
                build_autonomous_media_provider(self),
                build_autonomous_snapshot_provider(self),
                build_autonomous_project_bootstrap_provider(self),
            ]
        )
        self.research_runner_service = AgentResearchRunnerService()
        self.latex_runner_service = AgentLatexRunnerService()
        self.experiment_runner_service = AgentExperimentRunnerService()
        self.coding_runner_service = AgentCodingRunnerService()
        self.ingestion_demo_runner_service = AgentIngestionDemoRunnerService()
        self.deterministic_runner_registry = build_deterministic_runner_registry(self)
        self.observation_service = AgentObservationService()
        self.thinking_service = AgentThinkingService()
        self.action_service = AgentActionService()
        self.progress_evaluation_service = AgentProgressEvaluationService()
        self.runtime_policy_service = AgentRuntimePolicyService()
        self.skill_profile_service = AgentSkillProfileService()
        self.goal_contract_service = AgentGoalContractService()
        self.chain_orchestration_service = AgentChainOrchestrationService()
        self.follow_up_job_service = AgentFollowUpJobService()
        self.scientific_validation_service = AgentScientificValidationService()
        self.checkpoint_service = AgentCheckpointService()
        self._workspace_manager = None
        self._memory_service = None
        self._symbol_index_service = None
        # Store for job-specific data (findings, reading lists, etc.)
        self._job_findings: Dict[str, List[Dict[str, Any]]] = {}
        # Store for data analysis tools instances per job
        self._data_analysis_tools: Dict[str, DataAnalysisTools] = {}

    @property
    def document_service(self):
        if self._document_service is None:
            from app.services.document_service import DocumentService

            self._document_service = DocumentService()
        return self._document_service

    @property
    def workspace_manager(self):
        if self._workspace_manager is None:
            from app.services.coding_workspace_manager import CodingWorkspaceManager

            self._workspace_manager = CodingWorkspaceManager()
        return self._workspace_manager

    @property
    def memory_service(self):
        if self._memory_service is None:
            from app.services.memory_service import MemoryService

            self._memory_service = MemoryService()
        return self._memory_service

    @property
    def symbol_index_service(self):
        if self._symbol_index_service is None:
            from app.services.repo_symbol_index_service import RepoSymbolIndexService

            self._symbol_index_service = RepoSymbolIndexService()
        return self._symbol_index_service

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
        cooldown_seconds = _opt_int(
            "llm_unhealthy_cooldown_seconds", "cooldown_seconds"
        )

        if (
            not tier
            and not fallback_tiers
            and timeout_seconds is None
            and max_tokens_cap is None
            and cooldown_seconds is None
        ):
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
        result = await db.execute(select(AgentJob).where(AgentJob.id == job_id))
        job = result.scalar_one_or_none()

        if not job:
            return {"error": "Job not found", "status": "failed"}

        if job.status not in [
            AgentJobStatus.PENDING.value,
            AgentJobStatus.RUNNING.value,
        ]:
            return {
                "error": f"Job cannot be executed in status: {job.status}",
                "status": job.status,
            }

        # Load user settings
        user_settings = await self._load_user_settings(job.user_id, db)

        # Load agent definition if specified
        agent_def = None
        if job.agent_definition_id:
            agent_result = await db.execute(
                select(AgentDefinition).where(
                    AgentDefinition.id == job.agent_definition_id
                )
            )
            agent_def = agent_result.scalar_one_or_none()

        # Update job status
        job.status = AgentJobStatus.RUNNING.value
        job.started_at = job.started_at or datetime.utcnow()
        job.last_activity_at = datetime.utcnow()
        await db.commit()

        try:
            det = (job.config or {}).get("deterministic_runner")
            (
                handled,
                deterministic_result,
            ) = await self.deterministic_runner_registry.try_execute(
                det,
                job=job,
                db=db,
                progress_callback=progress_callback,
            )

            if handled:
                # Ensure chained jobs trigger even for deterministic runners.
                event = (
                    "complete"
                    if job.status == AgentJobStatus.COMPLETED.value
                    else "fail"
                )
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
        finally:
            # Persist workspace artifacts to MinIO before cleanup
            try:
                for _wid, _ws in list(self.workspace_manager._workspaces.items()):
                    if _ws.owner_job_id and _ws.owner_job_id != str(job_id):
                        continue
                    existing_workspace_ids = {
                        str(row.get("workspace_id") or "")
                        for row in (
                            job.output_artifacts
                            if isinstance(job.output_artifacts, list)
                            else []
                        )
                        if isinstance(row, dict)
                    }
                    if str(_ws.workspace_id) in existing_workspace_ids:
                        continue
                    persist_result = await self.workspace_manager.persist_workspace(
                        workspace=_ws,
                        job_id=str(job_id),
                        user_id=str(job.user_id),
                    )
                    if persist_result.get("manifest"):
                        if job.output_artifacts is None:
                            job.output_artifacts = []
                        job.output_artifacts.append(persist_result["manifest"])
                        flag_modified(job, "output_artifacts")
                        await db.commit()
            except Exception as persist_err:
                logger.warning(
                    f"Workspace persistence error for job {job_id}: {persist_err}"
                )
            # Clean up temp directories
            try:
                self.workspace_manager.cleanup_all()
            except Exception as cleanup_err:
                logger.warning(
                    f"Workspace cleanup error for job {job_id}: {cleanup_err}"
                )

    async def _run_ai_hub_scientist(
        self,
        *,
        job: AgentJob,
        db: AsyncSession,
        progress_callback: Optional[callable],
    ) -> Dict[str, Any]:
        """Compatibility wrapper around the extracted deterministic runner service."""
        return await self.research_runner_service.run_ai_hub_scientist(
            self,
            job=job,
            db=db,
            progress_callback=progress_callback,
        )

    async def _run_research_inbox_monitor(
        self,
        *,
        job: AgentJob,
        db: AsyncSession,
        progress_callback: Optional[callable],
    ) -> Dict[str, Any]:
        """Compatibility wrapper around the extracted deterministic runner service."""
        return await self.research_runner_service.run_research_inbox_monitor(
            self,
            job=job,
            db=db,
            progress_callback=progress_callback,
        )

    async def _run_code_patch_proposer(
        self,
        *,
        job: AgentJob,
        db: AsyncSession,
        progress_callback: Optional[callable],
    ) -> Dict[str, Any]:
        """Compatibility wrapper around the extracted deterministic runner service."""
        return await self.coding_runner_service.run_code_patch_proposer(
            self,
            job=job,
            db=db,
            progress_callback=progress_callback,
        )

    async def _run_research_engineer_scientist(
        self,
        *,
        job: AgentJob,
        db: AsyncSession,
        progress_callback: Optional[callable],
    ) -> Dict[str, Any]:
        """Compatibility wrapper around the extracted deterministic runner service."""
        return await self.research_runner_service.run_research_engineer_scientist(
            self,
            job=job,
            db=db,
            progress_callback=progress_callback,
        )

    async def _run_domain_research_orchestrator(
        self,
        *,
        job: AgentJob,
        db: AsyncSession,
        progress_callback: Optional[callable],
    ) -> Dict[str, Any]:
        """Compatibility wrapper around the extracted deterministic runner service."""
        return await self.research_runner_service.run_domain_research_orchestrator(
            self,
            job=job,
            db=db,
            progress_callback=progress_callback,
        )

    async def _create_domain_research_note(
        self,
        *,
        db: AsyncSession,
        job: AgentJob,
        title: str,
        content_markdown: str,
        tags: list[str],
        source_document_ids: list[Any],
        attribution: dict[str, Any],
        structured_payload: Optional[dict[str, Any]] = None,
    ):
        from app.models.research_note import ResearchNote

        def _json_safe(value: Any) -> Any:
            if isinstance(value, UUID):
                return str(value)
            if isinstance(value, datetime):
                return value.isoformat()
            if isinstance(value, dict):
                return {str(key): _json_safe(val) for key, val in value.items()}
            if isinstance(value, list):
                return [_json_safe(item) for item in value]
            if isinstance(value, tuple):
                return [_json_safe(item) for item in value]
            return value

        doc_ids: list[str] = []
        for raw in source_document_ids:
            try:
                doc_ids.append(str(UUID(str(raw))))
            except Exception:
                continue
        note = ResearchNote(
            user_id=job.user_id,
            title=title[:500],
            content_markdown=(content_markdown or "").strip()[:120000],
            tags=[str(tag).strip() for tag in tags if str(tag).strip()][:20] or None,
            source_document_ids=doc_ids or None,
            attribution=_json_safe(attribution)
            if isinstance(attribution, dict)
            else None,
            structured_payload=_json_safe(structured_payload)
            if isinstance(structured_payload, dict)
            else None,
        )
        db.add(note)
        await db.flush()
        return note

    async def _create_domain_research_follow_up_job(
        self,
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
        """Compatibility wrapper around the extracted orchestration service."""
        return await self.follow_up_job_service.create_domain_research_follow_up_job(
            self,
            db=db,
            job=job,
            domain=domain,
            objective=objective,
            customer_context=customer_context,
            track_type=track_type,
            source_scope=source_scope,
            top_idea=top_idea,
            docs=docs,
            repo_documents=repo_documents,
            papers=papers,
            repo_source_ids=repo_source_ids,
            benchmark_queries=benchmark_queries,
            automation_profile=automation_profile,
            automation_policy=automation_policy,
            sandbox_profile_id=sandbox_profile_id,
            profile_id=profile_id,
        )

    def _scientific_validation_context_key(
        self,
        *,
        profile_id: Optional[str] = None,
        portfolio_id: Optional[str] = None,
        hypothesis_id: Optional[str] = None,
    ) -> str:
        """Compatibility wrapper around the extracted orchestration service."""
        return self.scientific_validation_service.scientific_validation_context_key(
            self,
            profile_id=profile_id,
            portfolio_id=portfolio_id,
            hypothesis_id=hypothesis_id,
        )

    async def _update_scientific_validation_summary_links(
        self,
        *,
        db: AsyncSession,
        profile_id: Optional[str] = None,
        portfolio_id: Optional[str] = None,
        run_id: Optional[str] = None,
        run_record: Optional[dict[str, Any]] = None,
    ) -> None:
        """Compatibility wrapper around the extracted orchestration service."""
        return await self.scientific_validation_service.update_scientific_validation_summary_links(
            self,
            db=db,
            profile_id=profile_id,
            portfolio_id=portfolio_id,
            run_id=run_id,
            run_record=run_record,
        )

    async def _create_scientific_validation_run(
        self,
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
        """Compatibility wrapper around the extracted orchestration service."""
        return (
            await self.scientific_validation_service.create_scientific_validation_run(
                self,
                db=db,
                parent_job=parent_job,
                experiment_plan=experiment_plan,
                track_type=track_type,
                objective=objective,
                hypothesis_title=hypothesis_title,
                hypothesis_text=hypothesis_text,
                validation_policy=validation_policy,
                sandbox_profile_id=sandbox_profile_id,
                repo_source_ids=repo_source_ids,
                benchmark_queries=benchmark_queries,
                supporting_evidence=supporting_evidence,
                supporting_sources=supporting_sources,
                profile_id=profile_id,
                portfolio_id=portfolio_id,
                hypothesis_id=hypothesis_id,
                originating_job_id=originating_job_id,
            )
        )

    async def _run_research_fleet_orchestrator(
        self,
        *,
        job: AgentJob,
        db: AsyncSession,
        progress_callback: Optional[callable],
    ) -> Dict[str, Any]:
        """Compatibility wrapper around the extracted deterministic runner service."""
        return await self.research_runner_service.run_research_fleet_orchestrator(
            self,
            job=job,
            db=db,
            progress_callback=progress_callback,
        )

    async def _run_research_engineer_paper_update(
        self,
        *,
        job: AgentJob,
        db: AsyncSession,
        progress_callback: Optional[callable],
    ) -> Dict[str, Any]:
        """Compatibility wrapper around the extracted deterministic runner service."""
        return await self.research_runner_service.run_research_engineer_paper_update(
            self,
            job=job,
            db=db,
            progress_callback=progress_callback,
        )

    async def _run_latex_citation_sync(
        self,
        *,
        job: AgentJob,
        db: AsyncSession,
        progress_callback: Optional[callable],
    ) -> Dict[str, Any]:
        """Compatibility wrapper around the extracted deterministic runner service."""
        return await self.latex_runner_service.run_latex_citation_sync(
            self,
            job=job,
            db=db,
            progress_callback=progress_callback,
        )

    async def _run_latex_compile_project(
        self,
        *,
        job: AgentJob,
        db: AsyncSession,
        progress_callback: Optional[callable],
    ) -> Dict[str, Any]:
        """Compatibility wrapper around the extracted deterministic runner service."""
        return await self.latex_runner_service.run_latex_compile_project(
            self,
            job=job,
            db=db,
            progress_callback=progress_callback,
        )

    async def _run_latex_publish_project(
        self,
        *,
        job: AgentJob,
        db: AsyncSession,
        progress_callback: Optional[callable],
    ) -> Dict[str, Any]:
        """Compatibility wrapper around the extracted deterministic runner service."""
        return await self.latex_runner_service.run_latex_publish_project(
            self,
            job=job,
            db=db,
            progress_callback=progress_callback,
        )

    async def _run_latex_apply_unified_diff(
        self,
        *,
        job: AgentJob,
        db: AsyncSession,
        progress_callback: Optional[callable],
    ) -> Dict[str, Any]:
        """Compatibility wrapper around the extracted deterministic runner service."""
        return await self.latex_runner_service.run_latex_apply_unified_diff(
            self,
            job=job,
            db=db,
            progress_callback=progress_callback,
        )

    async def _run_latex_reviewer_critic(
        self,
        *,
        job: AgentJob,
        db: AsyncSession,
        progress_callback: Optional[callable],
    ) -> Dict[str, Any]:
        """Compatibility wrapper around the extracted deterministic runner service."""
        return await self.latex_runner_service.run_latex_reviewer_critic(
            self,
            job=job,
            db=db,
            progress_callback=progress_callback,
        )

    async def _run_experiment_plan_generate(
        self,
        *,
        job: AgentJob,
        db: AsyncSession,
        progress_callback: Optional[callable],
    ) -> Dict[str, Any]:
        """Compatibility wrapper around the extracted deterministic runner service."""
        return await self.experiment_runner_service.run_experiment_plan_generate(
            self,
            job=job,
            db=db,
            progress_callback=progress_callback,
        )

    async def _run_experiment_loop_seed(
        self,
        *,
        job: AgentJob,
        db: AsyncSession,
        progress_callback: Optional[callable],
    ) -> Dict[str, Any]:
        """Compatibility wrapper around the extracted deterministic runner service."""
        return await self.experiment_runner_service.run_experiment_loop_seed(
            self,
            job=job,
            db=db,
            progress_callback=progress_callback,
        )

    async def _run_experiment_decide_next(
        self,
        *,
        job: AgentJob,
        db: AsyncSession,
        progress_callback: Optional[callable],
    ) -> Dict[str, Any]:
        """Compatibility wrapper around the extracted deterministic runner service."""
        return await self.experiment_runner_service.run_experiment_decide_next(
            self,
            job=job,
            db=db,
            progress_callback=progress_callback,
        )

    async def _run_experiment_persist_results(
        self,
        *,
        job: AgentJob,
        db: AsyncSession,
        progress_callback: Optional[callable],
    ) -> Dict[str, Any]:
        """Compatibility wrapper around the extracted deterministic runner service."""
        return await self.experiment_runner_service.run_experiment_persist_results(
            self,
            job=job,
            db=db,
            progress_callback=progress_callback,
        )

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
                        text = str(
                            row.get("title")
                            or row.get("summary")
                            or row.get("message")
                            or row.get("insight")
                            or row.get("content")
                            or ""
                        ).strip()
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

        def _extract_paths(results: Dict[str, Any]) -> List[str]:
            if not isinstance(results, dict):
                return []
            buckets: List[Any] = []
            for key in (
                "file_paths",
                "suspect_files",
                "touched_files",
                "modified_files",
                "changed_files",
                "impacted_files",
            ):
                value = results.get(key)
                if isinstance(value, list):
                    buckets.extend(value)
            code_exec = (
                results.get("code_patch_execution")
                if isinstance(results.get("code_patch_execution"), dict)
                else {}
            )
            workspace = (
                code_exec.get("workspace")
                if isinstance(code_exec.get("workspace"), dict)
                else {}
            )
            for key in ("modified_files", "changed_files", "added_files"):
                value = workspace.get(key)
                if isinstance(value, list):
                    buckets.extend(value)
            out: List[str] = []
            seen: set[str] = set()
            for raw in buckets:
                path = str(raw or "").replace("\\", "/").strip().lstrip("/")
                while path.startswith("./"):
                    path = path[2:]
                if not path or ":" in path:
                    continue
                parts = [seg for seg in path.split("/") if seg not in {"", ".", ".."}]
                normalized = "/".join(parts)[:500]
                key = normalized.lower()
                if not normalized or key in seen:
                    continue
                seen.add(key)
                out.append(normalized)
                if len(out) >= 12:
                    break
            return out

        def _extract_commands(results: Dict[str, Any]) -> List[str]:
            if not isinstance(results, dict):
                return []
            buckets: List[Any] = []
            for key in ("commands", "verification_commands"):
                value = results.get(key)
                if isinstance(value, list):
                    buckets.extend(value)
            experiment = (
                results.get("experiment_run")
                if isinstance(results.get("experiment_run"), dict)
                else {}
            )
            for key in ("verification_commands", "commands", "failed_commands"):
                value = experiment.get(key)
                if isinstance(value, list):
                    buckets.extend(value)
            out: List[str] = []
            seen: set[str] = set()
            for raw in buckets:
                command = _norm_text(raw)[:500]
                key = command.lower()
                if not command or key in seen:
                    continue
                seen.add(key)
                out.append(command)
                if len(out) >= 8:
                    break
            return out

        def _path_cluster_keys(path: str) -> List[str]:
            normalized = str(path or "").replace("\\", "/").strip().strip("/")
            if not normalized:
                return []
            parts = [
                segment
                for segment in normalized.split("/")
                if segment not in {"", ".", ".."}
            ]
            if not parts:
                return []
            keys: List[str] = ["/".join(parts).lower()]
            if len(parts) >= 2:
                keys.append("/".join(parts[-2:]).lower())
            keys.append(parts[-1].lower())
            return [
                key for idx, key in enumerate(keys) if key and key not in keys[:idx]
            ]

        def _path_cluster_label(path: str) -> str:
            normalized = str(path or "").replace("\\", "/").strip().strip("/")
            if not normalized:
                return ""
            parts = [
                segment
                for segment in normalized.split("/")
                if segment not in {"", ".", ".."}
            ]
            if not parts:
                return ""
            return "/".join(parts[-2:]) if len(parts) >= 2 else parts[-1]

        def _canonical_command(command: str) -> str:
            lowered = _norm_text(command).lower()
            if not lowered:
                return ""
            lowered = re.sub(
                r"^\s*(ci=true|node_env=\S+|pythonunbuffered=\S+)\s+", "", lowered
            )
            return lowered.strip()

        sibling_jobs = payload.get("sibling_jobs")
        if not isinstance(sibling_jobs, list):
            sibling_jobs = []
        coding_swarm_enabled = bool(payload.get("coding_swarm_enabled")) or (
            str(payload.get("coding_swarm_profile") or "").strip().lower()
            == "bug_triage"
        )
        coding_harness_enabled = bool(payload.get("coding_harness_enabled"))
        fallback_paths = (
            [
                str(p).strip()
                for p in (payload.get("file_paths") or [])
                if str(p).strip()
            ]
            if isinstance(payload.get("file_paths"), list)
            else []
        )
        fallback_commands = (
            [str(c).strip() for c in (payload.get("commands") or []) if str(c).strip()]
            if isinstance(payload.get("commands"), list)
            else []
        )
        confidence_threshold = float(
            payload.get("coding_swarm_confidence_threshold") or 0.70
        )
        tiebreaker_threshold = float(
            payload.get("coding_swarm_tiebreaker_threshold") or 0.50
        )
        expected = int(payload.get("expected_siblings", 0) or 0)
        if expected <= 0:
            expected = len(sibling_jobs)
        terminal_count = int(payload.get("terminal_siblings", 0) or 0)
        if terminal_count <= 0:
            terminal_statuses = {
                AgentJobStatus.COMPLETED.value,
                AgentJobStatus.FAILED.value,
                AgentJobStatus.CANCELLED.value,
            }
            terminal_count = len(
                [
                    s
                    for s in sibling_jobs
                    if str((s or {}).get("status") or "") in terminal_statuses
                ]
            )

        support_map: Dict[str, Dict[str, Any]] = {}
        role_summaries: List[Dict[str, Any]] = []
        sibling_status: List[Dict[str, Any]] = []
        roles_ordered: List[str] = []
        completed_count = 0
        failed_roles: List[str] = []
        ranked_candidates: List[Dict[str, Any]] = []
        winning_candidate: Optional[Dict[str, Any]] = None
        role_file_hints: List[Dict[str, Any]] = []
        role_command_hints: List[Dict[str, Any]] = []

        for row in sibling_jobs:
            if not isinstance(row, dict):
                continue
            role = _norm_text(row.get("role") or row.get("name") or "unknown_role")[
                :120
            ]
            status = _norm_text(row.get("status") or "unknown").lower()
            normalized_role = self._normalize_role_token(role)
            if role and role not in roles_ordered:
                roles_ordered.append(role)
            if status == AgentJobStatus.COMPLETED.value:
                completed_count += 1
            if status in {AgentJobStatus.FAILED.value, AgentJobStatus.CANCELLED.value}:
                failed_roles.append(role or "unknown_role")

            row_results = (
                row.get("results") if isinstance(row.get("results"), dict) else {}
            )
            points = _extract_points(row_results)
            candidate_paths = _extract_paths(row_results)
            candidate_commands = _extract_commands(row_results)
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
            if coding_swarm_enabled:
                harness_evidence = (
                    row_results.get("coding_harness")
                    if isinstance(row_results.get("coding_harness"), dict)
                    else {}
                )
                verification_eligible = bool(
                    harness_evidence.get("completion_eligible", False)
                )
                candidate_snapshot = (
                    harness_evidence.get("candidate_snapshot")
                    if isinstance(harness_evidence.get("candidate_snapshot"), dict)
                    else None
                )
                promotion_eligible = not coding_harness_enabled or (
                    normalized_role == "coder"
                    and verification_eligible
                    and bool(candidate_snapshot)
                )
                role_bonus = {
                    "coder": 0.18,
                    "critic": 0.14,
                    "verifier": 0.16,
                }.get(normalized_role, 0.08)
                completion_bonus = (
                    0.25 if status == AgentJobStatus.COMPLETED.value else 0.0
                )
                score = (
                    completion_bonus
                    + role_bonus
                    + min(0.25, len(points) * 0.04)
                    + min(0.16, len(candidate_paths) * 0.04)
                    + (
                        0.18
                        if coding_harness_enabled and verification_eligible
                        else 0.0
                    )
                )
                candidate = {
                    "job_id": str(row.get("job_id") or ""),
                    "role": role,
                    "normalized_role": normalized_role,
                    "status": status,
                    "score": round(score, 4),
                    "suspect_files": candidate_paths[:8],
                    "recommended_commands": candidate_commands[:6],
                    "verification_eligible": verification_eligible,
                    "verification_state": str(
                        harness_evidence.get("verification_state") or "unknown"
                    ),
                    "modified_files": (
                        harness_evidence.get("modified_files")
                        if isinstance(harness_evidence.get("modified_files"), list)
                        else []
                    )[:12],
                    "candidate_snapshot": deepcopy(candidate_snapshot),
                }
                ranked_candidates.append(candidate)
                if promotion_eligible:
                    if winning_candidate is None or float(
                        candidate.get("score") or 0.0
                    ) > float(winning_candidate.get("score") or 0.0):
                        winning_candidate = candidate
                if candidate_paths:
                    role_file_hints.append(
                        {
                            "role": role,
                            "paths": candidate_paths[:8],
                        }
                    )
                if candidate_commands:
                    role_command_hints.append(
                        {
                            "role": role,
                            "commands": candidate_commands[:6],
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
        support_rows.sort(
            key=lambda r: (
                -int(r.get("support_count", 0) or 0),
                str(r.get("finding") or ""),
            )
        )

        consensus = [
            r for r in support_rows if int(r.get("support_count", 0) or 0) >= 2
        ][:10]
        singletons = [
            r for r in support_rows if int(r.get("support_count", 0) or 0) <= 1
        ][:10]

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

        file_cluster_support: Dict[str, Dict[str, Any]] = {}
        for row in role_file_hints:
            role = str(row.get("role") or "").strip() or "unknown_role"
            seen_role_clusters: set[str] = set()
            for raw_path in row.get("paths") or []:
                path = str(raw_path or "").strip()
                if not path:
                    continue
                label = _path_cluster_label(path)
                for cluster_key in _path_cluster_keys(path):
                    if cluster_key in seen_role_clusters:
                        continue
                    seen_role_clusters.add(cluster_key)
                    slot = file_cluster_support.get(cluster_key)
                    if not isinstance(slot, dict):
                        slot = {
                            "cluster": label or cluster_key,
                            "roles": set(),
                            "support_count": 0,
                        }
                    roles_set = slot.get("roles")
                    if not isinstance(roles_set, set):
                        roles_set = set()
                    roles_set.add(role)
                    slot["roles"] = roles_set
                    slot["support_count"] = len(roles_set)
                    file_cluster_support[cluster_key] = slot

        command_support: Dict[str, Dict[str, Any]] = {}
        for row in role_command_hints:
            role = str(row.get("role") or "").strip() or "unknown_role"
            seen_role_commands: set[str] = set()
            for raw_command in row.get("commands") or []:
                command = str(raw_command or "").strip()
                canonical = _canonical_command(command)
                if not canonical or canonical in seen_role_commands:
                    continue
                seen_role_commands.add(canonical)
                slot = command_support.get(canonical)
                if not isinstance(slot, dict):
                    slot = {"command": command, "roles": set(), "support_count": 0}
                roles_set = slot.get("roles")
                if not isinstance(roles_set, set):
                    roles_set = set()
                roles_set.add(role)
                slot["roles"] = roles_set
                slot["support_count"] = len(roles_set)
                command_support[canonical] = slot

        top_file_cluster = None
        if file_cluster_support:
            top_file_cluster = max(
                file_cluster_support.values(),
                key=lambda item: (
                    int(item.get("support_count") or 0),
                    str(item.get("cluster") or ""),
                ),
            )
        top_command_cluster = None
        if command_support:
            top_command_cluster = max(
                command_support.values(),
                key=lambda item: (
                    int(item.get("support_count") or 0),
                    str(item.get("command") or ""),
                ),
            )

        file_convergence_support = int(
            (top_file_cluster or {}).get("support_count") or 0
        )
        command_convergence_support = int(
            (top_command_cluster or {}).get("support_count") or 0
        )
        file_converged = file_convergence_support >= 2
        command_converged = command_convergence_support >= 2
        if coding_swarm_enabled and role_file_hints and not file_converged:
            conflicts.append(
                {
                    "type": "suspect_file_disagreement",
                    "description": "Roles disagree on the primary suspect file cluster.",
                    "roles": [
                        str(row.get("role") or "")
                        for row in role_file_hints[:8]
                        if str(row.get("role") or "").strip()
                    ],
                }
            )
        if coding_swarm_enabled and role_command_hints and not command_converged:
            conflicts.append(
                {
                    "type": "command_disagreement",
                    "description": "Roles disagree on the strongest reproduction or verification command.",
                    "roles": [
                        str(row.get("role") or "")
                        for row in role_command_hints[:8]
                        if str(row.get("role") or "").strip()
                    ],
                }
            )

        coverage = float(min(1.0, float(len(sibling_jobs)) / float(max(1, expected))))
        completion = float(
            min(1.0, float(completed_count) / float(max(1, len(sibling_jobs))))
        )
        agreement = 0.0
        if consensus:
            agreement = float(
                sum(
                    min(
                        1.0,
                        float(int(r.get("support_count", 0) or 0))
                        / float(max(1, len(sibling_jobs))),
                    )
                    for r in consensus
                )
            )
            agreement = max(0.0, min(1.0, agreement / float(max(1, len(consensus)))))
        overall = max(
            0.0, min(1.0, (0.35 * coverage) + (0.35 * completion) + (0.3 * agreement))
        )

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
        promotion_reason = ""
        recommended_commands: List[str] = []
        candidate_paths: List[Any] = []
        review_state = "informational"
        review_reason = ""
        review_required = False
        tiebreaker_attempted = bool(payload.get("tie_breaker_attempted"))
        tie_breaker_job_id = str(payload.get("tie_breaker_job_id") or "").strip()
        tie_breaker_source_job_id = str(
            payload.get("tie_breaker_source_job_id") or ""
        ).strip()
        if coding_swarm_enabled:
            ranked_candidates.sort(
                key=lambda item: (
                    -float(item.get("score") or 0.0),
                    str(item.get("role") or ""),
                )
            )
            candidate_paths = ranked_candidates[:6]
            if winning_candidate:
                recommended_commands.extend(
                    [
                        str(cmd).strip()
                        for cmd in (winning_candidate.get("recommended_commands") or [])
                        if str(cmd).strip()
                    ]
                )
            recommended_commands.extend(fallback_commands)
            dedup_commands: List[str] = []
            seen_commands: set[str] = set()
            for command in recommended_commands:
                key = command.lower()
                if not command or key in seen_commands:
                    continue
                seen_commands.add(key)
                dedup_commands.append(command)
                if len(dedup_commands) >= 8:
                    break
            recommended_commands = dedup_commands

            if not candidate_paths and fallback_paths:
                candidate_paths = [
                    {
                        "job_id": "",
                        "role": "Config scope",
                        "status": "configured",
                        "score": 0.0,
                        "suspect_files": fallback_paths[:8],
                    }
                ]
            verification_guardrail_met = bool(
                winning_candidate
                and (
                    not coding_harness_enabled
                    or (
                        winning_candidate.get("verification_eligible")
                        and winning_candidate.get("candidate_snapshot")
                    )
                )
            )
            guardrails_met = (
                bool(winning_candidate)
                and file_converged
                and command_converged
                and verification_guardrail_met
            )
            if coding_harness_enabled and not winning_candidate:
                review_state = "needs_review"
                review_required = True
                review_reason = (
                    "No mutation-owner result included both changed files and "
                    "successful verification evidence."
                )
                promotion_reason = review_reason
            elif (
                overall >= confidence_threshold and winning_candidate and guardrails_met
            ):
                promotion_reason = (
                    f"Auto-promote {str(winning_candidate.get('role') or 'top candidate')} at swarm confidence "
                    f"{overall:.2f}."
                )
                review_state = "auto_promoted"
                action_plan.insert(
                    0,
                    {
                        "priority": "high",
                        "action": f"Auto-promote winning coding slice: {str(winning_candidate.get('role') or '')}",
                        "rationale": promotion_reason,
                    },
                )
            elif (
                overall >= confidence_threshold
                and winning_candidate
                and not guardrails_met
            ):
                review_state = "needs_review"
                review_required = True
                review_reason = (
                    "Confidence cleared the promotion threshold, but file-cluster and command convergence "
                    "did not both meet the promotion guardrail."
                )
                promotion_reason = review_reason
            elif overall >= tiebreaker_threshold:
                if tiebreaker_attempted:
                    review_state = "insufficient_swarm_consensus"
                    review_required = True
                    review_reason = (
                        f"Confidence {overall:.2f} remained below auto-promotion threshold {confidence_threshold:.2f} "
                        "after a verifier tie-break."
                    )
                    promotion_reason = review_reason
                else:
                    review_state = "tie_break_needed"
                    review_reason = (
                        f"Confidence {overall:.2f} is below auto-promotion threshold {confidence_threshold:.2f}; "
                        "launch a verifier tie-break before operator review."
                    )
                    promotion_reason = review_reason
            else:
                review_state = "consensus_failed"
                review_required = True
                review_reason = (
                    f"Confidence {overall:.2f} is too low for automatic repair handoff."
                )
                promotion_reason = review_reason
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
            "fan_in_group_id": str(
                fan_in_group_id or payload.get("swarm_fan_in_group_id") or ""
            ),
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
            "winning_slice_id": str((winning_candidate or {}).get("job_id") or ""),
            "winning_role": str((winning_candidate or {}).get("role") or ""),
            "winning_candidate_snapshot": deepcopy(
                (winning_candidate or {}).get("candidate_snapshot")
            ),
            "promotion_reason": promotion_reason,
            "review_state": review_state,
            "review_reason": review_reason,
            "review_required": review_required,
            "coding_harness_enabled": coding_harness_enabled,
            "verification_guardrail_met": bool(
                winning_candidate
                and (
                    not coding_harness_enabled
                    or (
                        winning_candidate.get("verification_eligible")
                        and winning_candidate.get("candidate_snapshot")
                    )
                )
            ),
            "tie_breaker_attempted": tiebreaker_attempted,
            "tie_breaker_job_id": tie_breaker_job_id,
            "tie_breaker_source_job_id": tie_breaker_source_job_id,
            "file_converged": file_converged,
            "file_convergence_support": file_convergence_support,
            "top_file_cluster": (
                {
                    "cluster": str(top_file_cluster.get("cluster") or ""),
                    "support_count": int(top_file_cluster.get("support_count") or 0),
                    "roles": sorted(
                        [
                            str(role)
                            for role in (top_file_cluster.get("roles") or set())
                            if str(role).strip()
                        ]
                    )[:10],
                }
                if isinstance(top_file_cluster, dict)
                else None
            ),
            "command_converged": command_converged,
            "command_convergence_support": command_convergence_support,
            "top_command_cluster": (
                {
                    "command": str(top_command_cluster.get("command") or ""),
                    "support_count": int(top_command_cluster.get("support_count") or 0),
                    "roles": sorted(
                        [
                            str(role)
                            for role in (top_command_cluster.get("roles") or set())
                            if str(role).strip()
                        ]
                    )[:10],
                }
                if isinstance(top_command_cluster, dict)
                else None
            ),
            "candidate_paths": candidate_paths[:6],
            "recommended_commands": recommended_commands[:8],
            "generated_at": datetime.utcnow().isoformat(),
        }

    def _compose_swarm_rerun_payload(
        self,
        base_payload: Dict[str, Any],
        *,
        tie_breaker_job: AgentJob,
        tie_breaker_source_job_id: str = "",
    ) -> Dict[str, Any]:
        payload = dict(base_payload or {})
        sibling_jobs = (
            payload.get("sibling_jobs")
            if isinstance(payload.get("sibling_jobs"), list)
            else []
        )
        out: List[Dict[str, Any]] = []
        for row in sibling_jobs:
            if not isinstance(row, dict):
                continue
            if str(row.get("job_id") or "") == str(tie_breaker_job.id):
                continue
            out.append(dict(row))
        cfg = tie_breaker_job.config if isinstance(tie_breaker_job.config, dict) else {}
        out.append(
            {
                "job_id": str(tie_breaker_job.id),
                "name": str(tie_breaker_job.name or "")[:200],
                "status": str(tie_breaker_job.status or ""),
                "is_terminal": str(tie_breaker_job.status or "")
                in {
                    AgentJobStatus.COMPLETED.value,
                    AgentJobStatus.FAILED.value,
                    AgentJobStatus.CANCELLED.value,
                },
                "progress": int(tie_breaker_job.progress or 0),
                "role": str(cfg.get("swarm_role") or "Tie-breaker Verifier"),
                "results": tie_breaker_job.results
                if isinstance(tie_breaker_job.results, dict)
                else {},
            }
        )
        payload["sibling_jobs"] = out
        payload["expected_siblings"] = max(
            int(payload.get("expected_siblings", 0) or 0), len(out)
        )
        payload["terminal_siblings"] = len(
            [
                row
                for row in out
                if str((row or {}).get("status") or "")
                in {
                    AgentJobStatus.COMPLETED.value,
                    AgentJobStatus.FAILED.value,
                    AgentJobStatus.CANCELLED.value,
                }
            ]
        )
        payload["tie_breaker_attempted"] = True
        payload["tie_breaker_job_id"] = str(tie_breaker_job.id)
        payload["tie_breaker_source_job_id"] = str(tie_breaker_source_job_id or "")
        return payload

    async def _launch_bug_triage_swarm_tie_breaker_job(
        self,
        *,
        fan_in_job: AgentJob,
        db: AsyncSession,
        merged: Dict[str, Any],
        swarm_payload: Dict[str, Any],
    ) -> Optional[AgentJob]:
        cfg = fan_in_job.config if isinstance(fan_in_job.config, dict) else {}
        from app.services.agent_coding_harness_service import (
            agent_coding_harness_service,
        )

        verifier_config = agent_coding_harness_service.get_role_catalog()["verifier"][
            "config"
        ]
        parent_root_id_raw = (
            cfg.get("swarm_parent_job_id") or fan_in_job.root_job_id or fan_in_job.id
        )
        try:
            parent_root_id = UUID(str(parent_root_id_raw))
        except Exception:
            parent_root_id = fan_in_job.root_job_id or fan_in_job.id

        failure_symptom = str(cfg.get("failure_symptom") or "").strip()
        disagreement_summary = [
            str(conflict.get("description") or conflict.get("type") or "").strip()
            for conflict in (
                merged.get("conflicts")
                if isinstance(merged.get("conflicts"), list)
                else []
            )
            if isinstance(conflict, dict)
            and str(conflict.get("description") or conflict.get("type") or "").strip()
        ][:4]
        candidate_rows = (
            merged.get("candidate_paths")
            if isinstance(merged.get("candidate_paths"), list)
            else []
        )
        top_candidates = []
        candidate_snapshots: List[Dict[str, Any]] = []
        for row in candidate_rows[:3]:
            if not isinstance(row, dict):
                continue
            suspect_files = [
                str(path).strip()
                for path in (row.get("suspect_files") or [])
                if str(path).strip()
            ]
            top_candidates.append(
                {
                    "role": str(row.get("role") or "").strip() or "candidate",
                    "suspect_files": suspect_files[:4],
                    "recommended_commands": [
                        str(cmd).strip()
                        for cmd in (row.get("recommended_commands") or [])
                        if str(cmd).strip()
                    ][:3],
                    "snapshot_id": str(
                        (
                            row.get("candidate_snapshot")
                            if isinstance(row.get("candidate_snapshot"), dict)
                            else {}
                        ).get("snapshot_id")
                        or ""
                    ),
                }
            )
            candidate_snapshot = row.get("candidate_snapshot")
            if isinstance(candidate_snapshot, dict) and str(
                candidate_snapshot.get("snapshot_id") or ""
            ):
                candidate_snapshots.append(deepcopy(candidate_snapshot))

        tie_breaker_goal = (
            "Tie-breaker verifier for bug triage swarm.\n"
            f"Original goal: {str(fan_in_job.goal or '').strip()[:1200]}\n"
            f"Failure symptom: {failure_symptom[:800]}\n\n"
            "Review the competing swarm paths, challenge weak assumptions, and identify the strongest "
            "reproduction command and suspect file cluster.\n"
            f"Disagreement summary: {json.dumps(disagreement_summary, ensure_ascii=False)}\n"
            f"Top candidates: {json.dumps(top_candidates, ensure_ascii=False)}"
        )

        rerun_group_id = hashlib.sha256(
            f"swarm_fan_in_rerun:{fan_in_job.id}".encode("utf-8")
        ).hexdigest()[:16]
        rerun_child = {
            "name": "Bug Triage Swarm Tie-Break Fan-in",
            "description": "Auto-generated fan-in rerun after verifier tie-break.",
            "job_type": "synthesis",
            "goal": (
                "Re-run coding swarm fan-in after the verifier tie-break and either auto-promote "
                "the winning slice or pause for operator review."
            ),
            "config": {
                "origin": "swarm_fan_in_rerun_aggregator",
                "deterministic_runner": "swarm_fan_in_aggregate",
                "swarm_fan_in_group_id": rerun_group_id,
                "swarm_parent_job_id": str(parent_root_id),
                "coding_swarm_enabled": True,
                "coding_swarm_profile": str(cfg.get("coding_swarm_profile") or "")
                .strip()
                .lower()
                or "bug_triage",
                "coding_swarm_confidence_threshold": cfg.get(
                    "coding_swarm_confidence_threshold"
                ),
                "coding_swarm_tiebreaker_threshold": cfg.get(
                    "coding_swarm_tiebreaker_threshold"
                ),
                "swarm_child_jobs_enabled": False,
                "auto_subgoal_child_jobs_enabled": False,
                "tie_breaker_attempted": True,
                "tie_breaker_source_job_id": str(fan_in_job.id),
                "swarm_rerun_base_payload": deepcopy(swarm_payload),
                "source_id": str(cfg.get("source_id") or ""),
                "failure_symptom": failure_symptom,
                "error_output": str(cfg.get("error_output") or "").strip(),
                "scope": str(cfg.get("scope") or "auto").strip().lower() or "auto",
                "search_query": str(cfg.get("search_query") or "").strip(),
                "file_paths": cfg.get("file_paths")
                if isinstance(cfg.get("file_paths"), list)
                else [],
                "commands": cfg.get("commands")
                if isinstance(cfg.get("commands"), list)
                else [],
                "candidate_snapshots": candidate_snapshots,
                "apply_patch_to_kb": False,
                "apply_patch_to_kb_confirm": False,
            },
            "max_iterations": max(6, int((fan_in_job.max_iterations or 20) * 0.4)),
            "max_tool_calls": max(8, int((fan_in_job.max_tool_calls or 50) * 0.4)),
            "max_llm_calls": max(6, int((fan_in_job.max_llm_calls or 30) * 0.4)),
            "max_runtime_minutes": max(
                10, int((fan_in_job.max_runtime_minutes or 60) * 0.35)
            ),
        }

        tie_breaker = AgentJob(
            name=f"Tie-breaker Verifier - {str(fan_in_job.name or '')[:72]}",
            description="Auto-launched verifier tie-breaker for bug triage swarm.",
            job_type="analysis",
            goal=tie_breaker_goal[:2400],
            config={
                **dict(verifier_config),
                "origin": "swarm_tie_breaker_verifier",
                "swarm_role": "Tie-breaker Verifier",
                "swarm_role_key": "verifier_tiebreaker",
                "agent_role": "verifier",
                "swarm_parent_job_id": str(parent_root_id),
                "tie_breaker_source_job_id": str(fan_in_job.id),
                "create_workspace_from_source": True,
                "prefer_sources": ["documents"],
                "emit_execution_plan": True,
                "source_id": str(cfg.get("source_id") or ""),
                "failure_symptom": failure_symptom,
                "error_output": str(cfg.get("error_output") or "").strip(),
                "scope": str(cfg.get("scope") or "auto").strip().lower() or "auto",
                "search_query": str(cfg.get("search_query") or "").strip(),
                "file_paths": cfg.get("file_paths")
                if isinstance(cfg.get("file_paths"), list)
                else [],
                "commands": cfg.get("commands")
                if isinstance(cfg.get("commands"), list)
                else [],
                "candidate_snapshots": candidate_snapshots,
                "coding_workspace_session_id": str(
                    cfg.get("coding_workspace_session_id") or ""
                ).strip(),
                "coding_swarm_enabled": True,
                "coding_swarm_profile": str(cfg.get("coding_swarm_profile") or "")
                .strip()
                .lower()
                or "bug_triage",
                "apply_patch_to_kb": False,
                "apply_patch_to_kb_confirm": False,
                "auto_subgoal_child_jobs_enabled": False,
                "swarm_child_jobs_enabled": False,
            },
            user_id=fan_in_job.user_id,
            status=AgentJobStatus.PENDING.value,
            parent_job_id=fan_in_job.id,
            root_job_id=parent_root_id,
            chain_depth=int(fan_in_job.chain_depth or 0) + 1,
            chain_config={
                "trigger_condition": ChainTriggerCondition.ON_ANY_END.value,
                "inherit_results": True,
                "inherit_config": False,
                "child_jobs": [rerun_child],
            },
            max_iterations=max(6, int((fan_in_job.max_iterations or 20) * 0.4)),
            max_tool_calls=max(8, int((fan_in_job.max_tool_calls or 50) * 0.4)),
            max_llm_calls=max(6, int((fan_in_job.max_llm_calls or 30) * 0.4)),
            max_runtime_minutes=max(
                10, int((fan_in_job.max_runtime_minutes or 60) * 0.35)
            ),
            enable_memory=False,
            results=(
                {
                    "swarm_collaboration": deepcopy(
                        (fan_in_job.results or {}).get("swarm_collaboration")
                    )
                }
                if isinstance(
                    (fan_in_job.results or {}).get("swarm_collaboration"), dict
                )
                else None
            ),
        )
        db.add(tie_breaker)
        await db.flush()
        return tie_breaker

    async def _launch_bug_triage_swarm_repair_job(
        self,
        *,
        fan_in_job: AgentJob,
        db: AsyncSession,
        merged: Dict[str, Any],
        candidate_job_id: str = "",
        manual_promotion: bool = False,
    ) -> Optional[AgentJob]:
        from app.services.agent_job_templates import (
            REPO_BUG_TRIAGE_REPAIR_TEMPLATE_ID,
            get_builtin_agent_job_template,
        )

        template = get_builtin_agent_job_template(REPO_BUG_TRIAGE_REPAIR_TEMPLATE_ID)
        if template is None:
            return None

        cfg = fan_in_job.config if isinstance(fan_in_job.config, dict) else {}
        parent_root_id_raw = (
            cfg.get("swarm_parent_job_id") or fan_in_job.root_job_id or fan_in_job.id
        )
        try:
            parent_root_id = UUID(str(parent_root_id_raw))
        except Exception:
            parent_root_id = fan_in_job.root_job_id or fan_in_job.id
        candidate_rows = (
            merged.get("candidate_paths")
            if isinstance(merged.get("candidate_paths"), list)
            else []
        )
        selected_candidate = None
        if candidate_job_id:
            for row in candidate_rows:
                if (
                    isinstance(row, dict)
                    and str(row.get("job_id") or "").strip()
                    == str(candidate_job_id).strip()
                ):
                    selected_candidate = row
                    break
        if selected_candidate is None and candidate_rows:
            selected_candidate = (
                candidate_rows[0] if isinstance(candidate_rows[0], dict) else None
            )
        commands = []
        if isinstance(selected_candidate, dict):
            commands.extend(
                [
                    str(c).strip()
                    for c in (selected_candidate.get("recommended_commands") or [])
                    if str(c).strip()
                ]
            )
        commands.extend(
            [
                str(c).strip()
                for c in (merged.get("recommended_commands") or [])
                if str(c).strip()
            ]
        )
        file_paths: List[str] = []
        candidate_iterable = (
            [selected_candidate] if isinstance(selected_candidate, dict) else []
        )
        candidate_iterable.extend(
            [
                row
                for row in candidate_rows
                if isinstance(row, dict) and row is not selected_candidate
            ]
        )
        for row in candidate_iterable:
            if not isinstance(row, dict):
                continue
            file_paths.extend(
                [
                    str(p).strip()
                    for p in (row.get("suspect_files") or [])
                    if str(p).strip()
                ]
            )
        dedup_file_paths: List[str] = []
        seen_paths: set[str] = set()
        for path in file_paths + (
            [str(p).strip() for p in (cfg.get("file_paths") or []) if str(p).strip()]
            if isinstance(cfg.get("file_paths"), list)
            else []
        ):
            key = path.lower()
            if not path or key in seen_paths:
                continue
            seen_paths.add(key)
            dedup_file_paths.append(path)
            if len(dedup_file_paths) >= 12:
                break

        from app.services.agent_coding_harness_service import (
            agent_coding_harness_service,
        )

        patcher_config = agent_coding_harness_service.get_role_catalog()["patcher"][
            "config"
        ]
        child_config = {
            **dict(template.default_config or {}),
            **dict(patcher_config),
        }
        candidate_snapshot = (
            selected_candidate.get("candidate_snapshot")
            if isinstance(selected_candidate, dict)
            and isinstance(selected_candidate.get("candidate_snapshot"), dict)
            else merged.get("winning_candidate_snapshot")
            if isinstance(merged.get("winning_candidate_snapshot"), dict)
            else None
        )
        child_config.update(
            {
                "source_id": str(cfg.get("source_id") or ""),
                "failure_symptom": str(cfg.get("failure_symptom") or "").strip(),
                "error_output": str(cfg.get("error_output") or "").strip(),
                "scope": str(cfg.get("scope") or "auto").strip().lower() or "auto",
                "search_query": str(cfg.get("search_query") or "").strip(),
                "commands": commands[:6],
                "file_paths": dedup_file_paths[:12],
                "apply_patch_to_kb": False,
                "apply_patch_to_kb_confirm": False,
                "launch_mode": "bug_triage_swarm_repair_handoff",
                "relaunch_from_job_id": str(parent_root_id),
                "coding_workspace_session_id": str(
                    cfg.get("coding_workspace_session_id")
                    or (candidate_snapshot or {}).get("session_id")
                    or ""
                ).strip(),
                "candidate_snapshot": deepcopy(candidate_snapshot),
                "swarm_handoff": {
                    "fan_in_job_id": str(fan_in_job.id),
                    "swarm_parent_job_id": str(parent_root_id),
                    "winning_slice_id": str(merged.get("winning_slice_id") or ""),
                    "winning_role": str(merged.get("winning_role") or ""),
                    "confidence": merged.get("confidence") or {},
                    "promotion_reason": (
                        f"Manual promotion from swarm candidate {str(selected_candidate.get('role') or '').strip() or str(candidate_job_id)[:8]}."
                        if manual_promotion
                        else str(merged.get("promotion_reason") or "")
                    ),
                    "manual_promotion": manual_promotion,
                    "candidate_snapshot": deepcopy(candidate_snapshot),
                },
            }
        )
        child = AgentJob(
            name=f"Bug Triage Repair - {str(fan_in_job.name or '')[:80]}",
            description="Auto-launched repair chain from bug triage swarm fan-in.",
            job_type=template.job_type,
            goal=(
                f"Use the promoted bug-triage swarm evidence to propose and verify the minimal repair.\n"
                f"Original goal: {str(fan_in_job.goal or '').strip()[:1200]}"
            ).strip(),
            config=child_config,
            user_id=fan_in_job.user_id,
            status=AgentJobStatus.PENDING.value,
            parent_job_id=fan_in_job.id,
            root_job_id=parent_root_id,
            chain_depth=int(fan_in_job.chain_depth or 0) + 1,
            chain_config=deepcopy(template.default_chain_config),
            max_iterations=template.default_max_iterations,
            max_tool_calls=template.default_max_tool_calls,
            max_llm_calls=template.default_max_llm_calls,
            max_runtime_minutes=template.default_max_runtime_minutes,
            enable_memory=False,
            results=(
                {
                    "swarm_collaboration": deepcopy(
                        (fan_in_job.results or {}).get("swarm_collaboration")
                    )
                }
                if isinstance(
                    (fan_in_job.results or {}).get("swarm_collaboration"), dict
                )
                else None
            ),
        )
        db.add(child)
        await db.flush()
        return child

    def _build_swarm_backlog_collaboration(
        self, fan_in_job: AgentJob
    ) -> dict[str, Any]:
        results = fan_in_job.results if isinstance(fan_in_job.results, dict) else {}
        raw = (
            results.get("swarm_collaboration")
            if isinstance(results.get("swarm_collaboration"), dict)
            else {}
        )
        shared_with: list[str] = []
        seen: set[str] = set()
        for raw_value in (
            raw.get("shared_with_user_ids")
            if isinstance(raw.get("shared_with_user_ids"), list)
            else []
        ):
            try:
                value = str(UUID(str(raw_value))).strip()
            except Exception:
                continue
            if not value or value in seen:
                continue
            seen.add(value)
            shared_with.append(value)
        assigned_user_id = str(raw.get("assigned_user_id") or "").strip() or None
        if assigned_user_id and assigned_user_id not in shared_with:
            shared_with.append(assigned_user_id)
        visibility = (
            "shared"
            if bool(raw.get("shared_review")) or bool(shared_with)
            else "private"
        )
        return {
            "owner_user_id": str(raw.get("owner_user_id") or fan_in_job.user_id),
            "visibility": visibility,
            "shared_with_user_ids": shared_with,
            "assigned_user_id": assigned_user_id,
            "assigned_by_user_id": str(raw.get("assigned_by_user_id") or "").strip()
            or None,
            "assigned_at": str(raw.get("assigned_at") or "").strip() or None,
            "note": str(raw.get("review_note") or "").strip() or None,
        }

    async def _find_existing_swarm_backlog_item(
        self,
        *,
        fan_in_job: AgentJob,
        db: AsyncSession,
    ) -> Optional[Any]:
        from app.models.coding_backlog import CodingBacklogItem

        rows = (
            (
                await db.execute(
                    select(CodingBacklogItem).where(
                        CodingBacklogItem.user_id == fan_in_job.user_id
                    )
                )
            )
            .scalars()
            .all()
        )
        target_job_id = str(fan_in_job.id)
        for item in rows:
            lineage = item.lineage if isinstance(item.lineage, dict) else {}
            if (
                str(lineage.get("originating_swarm_job_id") or "").strip()
                == target_job_id
            ):
                return item
        return None

    async def _auto_route_swarm_to_backlog(
        self,
        *,
        fan_in_job: AgentJob,
        db: AsyncSession,
        merged: Dict[str, Any],
    ) -> Optional[Any]:
        from app.models.coding_backlog import CodingBacklogItem

        existing = await self._find_existing_swarm_backlog_item(
            fan_in_job=fan_in_job, db=db
        )
        if existing is not None:
            merged["backlog_item_id"] = str(existing.id)
            merged["backlog_route_mode"] = str(
                (
                    (existing.lineage or {})
                    if isinstance(existing.lineage, dict)
                    else {}
                ).get("originating_swarm_route_mode")
                or "manual"
            )
            merged["backlog_auto_route_suppressed_reason"] = "existing_backlog_link"
            return existing

        cfg = fan_in_job.config if isinstance(fan_in_job.config, dict) else {}
        quick_start = (
            cfg.get("quick_start") if isinstance(cfg.get("quick_start"), dict) else {}
        )
        candidate_paths = (
            merged.get("candidate_paths")
            if isinstance(merged.get("candidate_paths"), list)
            else []
        )
        top_candidate = (
            candidate_paths[0]
            if candidate_paths and isinstance(candidate_paths[0], dict)
            else {}
        )
        preset_key = (
            str(
                quick_start.get("preset_key")
                or cfg.get("coding_swarm_preset_key")
                or ""
            )
            .strip()
            .lower()
            or "bug_triage_swarm"
        )
        preset_label = {
            "build_break_swarm": "Build Break Swarm",
            "frontend_regression_swarm": "Frontend Regression Swarm",
        }.get(preset_key, "Bug Triage Swarm")
        suspect_files: list[str] = []
        for row in candidate_paths[:4]:
            if not isinstance(row, dict):
                continue
            suspect_files.extend(
                [
                    str(value).strip()
                    for value in (row.get("suspect_files") or [])
                    if str(value).strip()
                ]
            )
        dedup_files: list[str] = []
        seen_files: set[str] = set()
        for path in suspect_files:
            key = path.lower()
            if key in seen_files:
                continue
            seen_files.add(key)
            dedup_files.append(path)
            if len(dedup_files) >= 12:
                break

        collaboration = self._build_swarm_backlog_collaboration(fan_in_job)
        now = datetime.utcnow()
        source_uuid = None
        try:
            if str(cfg.get("source_id") or "").strip():
                source_uuid = UUID(str(cfg.get("source_id")))
        except Exception:
            source_uuid = None
        item = CodingBacklogItem(
            user_id=fan_in_job.user_id,
            source_id=source_uuid,
            title=f"{preset_label} review - {str(fan_in_job.name or 'autonomous job')[:72]}",
            portfolio_goal=str(
                fan_in_job.goal
                or "Review coding swarm findings and implement the best repair path"
            ).strip()[:2000],
            status="draft",
            priority=50,
            scope=str(cfg.get("scope") or "auto").strip().lower() or "auto",
            failure_symptom=str(cfg.get("failure_symptom") or "").strip() or None,
            error_output=str(cfg.get("error_output") or "").strip() or None,
            file_paths=dedup_files[:12],
            commands=[
                str(value).strip()
                for value in (merged.get("recommended_commands") or [])
                if str(value).strip()
            ][:6],
            auto_apply_enabled=True,
            require_patch_pr=False,
            visibility=str(collaboration.get("visibility") or "private"),
            shared_with_user_ids=list(collaboration.get("shared_with_user_ids") or [])
            or None,
            assigned_user_id=UUID(str(collaboration.get("assigned_user_id")))
            if str(collaboration.get("assigned_user_id") or "").strip()
            else None,
            assigned_by_user_id=UUID(str(collaboration.get("assigned_by_user_id")))
            if str(collaboration.get("assigned_by_user_id") or "").strip()
            else None,
            assigned_at=datetime.fromisoformat(str(collaboration.get("assigned_at")))
            if str(collaboration.get("assigned_at") or "").strip()
            else None,
            collaboration=collaboration,
            policy={
                "max_auto_retries": 1,
                "max_files_touched": 3,
                "blocked_path_prefixes": [],
                "require_experiments_ok": True,
                "confidence_threshold": 0.55,
            },
            lineage={
                "originating_swarm_job_id": str(fan_in_job.id),
                "originating_swarm_preset": preset_key,
                "originating_swarm_review_reason": str(
                    merged.get("review_reason") or merged.get("promotion_reason") or ""
                ).strip()
                or None,
                "originating_swarm_candidate_job_id": str(
                    top_candidate.get("job_id") or ""
                ).strip()
                or None,
                "originating_swarm_candidate_role": str(
                    top_candidate.get("role") or ""
                ).strip()
                or None,
                "originating_swarm_candidate_index": 0,
                "originating_swarm_route_mode": "auto",
                "originating_swarm_auto_routed_at": now.isoformat(),
            },
            decomposition={
                "strategy": "portfolio_goal",
                "planned_slices": [],
                "active_slice_id": None,
                "completed_slices": [],
                "failed_slices": [],
                "promotion_decisions": [],
                "backlog_timeline": [
                    {
                        "at": now.isoformat(),
                        "actor": "system",
                        "action": "auto_routed_from_swarm",
                        "new_status": "draft",
                        "note": str(
                            merged.get("review_reason")
                            or "Auto-routed unresolved coding swarm into backlog."
                        )[:5000],
                        "metadata": {
                            "swarm_job_id": str(fan_in_job.id),
                            "preset_key": preset_key,
                        },
                    }
                ],
                "lineage_summary": {
                    "repair_job_count": 0,
                    "apply_job_count": 0,
                    "patch_pr_count": 0,
                    "proposal_count": 0,
                    "operator_action_count": 0,
                },
                "portfolio_progress": {
                    "total_slices": 0,
                    "pending_slices": 0,
                    "completed_slices": 0,
                    "failed_slices": 0,
                    "auto_applied_slices": 0,
                    "proposal_only_slices": 0,
                },
            },
            child_job_ids=[],
            latest_summary={
                "status": "draft",
                "note": "Auto-routed from unresolved coding swarm.",
                "route_mode": "auto",
                "portfolio_progress": {
                    "total_slices": 0,
                    "pending_slices": 0,
                    "completed_slices": 0,
                    "failed_slices": 0,
                    "auto_applied_slices": 0,
                    "proposal_only_slices": 0,
                },
            },
        )
        db.add(item)
        await db.flush()
        merged["backlog_item_id"] = str(item.id)
        merged["backlog_route_mode"] = "auto"
        merged["backlog_auto_routed_at"] = now.isoformat()
        return item

    async def _run_swarm_fan_in_aggregate(
        self,
        *,
        job: AgentJob,
        db: AsyncSession,
        progress_callback: Optional[callable],
    ) -> Dict[str, Any]:
        """Compatibility wrapper around the extracted deterministic runner service."""
        return await self.research_runner_service.run_swarm_fan_in_aggregate(
            self,
            job=job,
            db=db,
            progress_callback=progress_callback,
        )

    async def _run_experiment_runner(
        self,
        *,
        job: AgentJob,
        db: AsyncSession,
        progress_callback: Optional[callable],
    ) -> Dict[str, Any]:
        """Compatibility wrapper around the extracted deterministic runner service."""
        return await self.experiment_runner_service.run_experiment_runner(
            self,
            job=job,
            db=db,
            progress_callback=progress_callback,
        )

    async def _run_code_patch_apply_to_kb(
        self,
        *,
        job: AgentJob,
        db: AsyncSession,
        progress_callback: Optional[callable],
    ) -> Dict[str, Any]:
        """Compatibility wrapper around the extracted deterministic runner service."""
        return await self.coding_runner_service.run_code_patch_apply_to_kb(
            self,
            job=job,
            db=db,
            progress_callback=progress_callback,
        )

    async def _run_coding_backlog_orchestrator(
        self,
        *,
        job: AgentJob,
        db: AsyncSession,
        progress_callback: Optional[callable],
    ) -> Dict[str, Any]:
        """Compatibility wrapper around the extracted deterministic runner service."""
        return await self.coding_runner_service.run_coding_backlog_orchestrator(
            self,
            job=job,
            db=db,
            progress_callback=progress_callback,
        )

    async def _run_arxiv_inbox_extract_repos(
        self,
        *,
        job: AgentJob,
        db: AsyncSession,
        progress_callback: Optional[callable],
    ) -> Dict[str, Any]:
        """Compatibility wrapper around the extracted deterministic runner service."""
        return await self.ingestion_demo_runner_service.run_arxiv_inbox_extract_repos(
            self,
            job=job,
            db=db,
            progress_callback=progress_callback,
        )

    async def _run_git_repo_ingest_wait(
        self,
        *,
        job: AgentJob,
        db: AsyncSession,
        progress_callback: Optional[callable],
    ) -> Dict[str, Any]:
        """Compatibility wrapper around the extracted deterministic runner service."""
        return await self.ingestion_demo_runner_service.run_git_repo_ingest_wait(
            self,
            job=job,
            db=db,
            progress_callback=progress_callback,
        )

    async def _run_generated_project_demo_check(
        self,
        *,
        job: AgentJob,
        db: AsyncSession,
        progress_callback: Optional[callable],
    ) -> Dict[str, Any]:
        """Compatibility wrapper around the extracted deterministic runner service."""
        return (
            await self.ingestion_demo_runner_service.run_generated_project_demo_check(
                self,
                job=job,
                db=db,
                progress_callback=progress_callback,
            )
        )

    async def _run_paper_algorithm_project(
        self,
        *,
        job: AgentJob,
        db: AsyncSession,
        progress_callback: Optional[callable],
    ) -> Dict[str, Any]:
        """Compatibility wrapper around the extracted deterministic runner service."""
        return await self.ingestion_demo_runner_service.run_paper_algorithm_project(
            self,
            job=job,
            db=db,
            progress_callback=progress_callback,
        )

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

        # Load checkpoint if resuming
        checkpoint = await self._load_latest_checkpoint(job.id, db)
        if checkpoint:
            logger.info(f"Resuming job {job.id} from iteration {checkpoint.iteration}")
        state = initialize_runtime_state(checkpoint.state if checkpoint else None)
        recovered_completion = agent_execution_journal_service.recover_completed_action(
            state=state
        )
        if recovered_completion:
            job.add_log_entry(
                {
                    "phase": "execution_journal_recovered_result",
                    "invocation_id": state.get(
                        "execution_journal_recovered_invocation_id"
                    ),
                }
            )
        reconciliation = agent_execution_journal_service.reconcile_interrupted(
            job=job, state=state
        )
        if reconciliation:
            job.status = AgentJobStatus.PAUSED.value
            job.current_phase = "awaiting_reconciliation"
            job.phase_details = str(reconciliation.get("message") or "")[:280]
            job.add_log_entry(
                {
                    "phase": "execution_reconciliation",
                    "checkpoint": reconciliation,
                }
            )
            agent_execution_journal_service._sync_job_summary(job, state)
            await self.checkpoint_service.save_checkpoint(
                job=job,
                state=state,
                db=db,
                reason="execution_reconciliation",
            )
            return {
                "status": job.status,
                "progress": int(state.get("goal_progress", 0) or 0),
                "results": job.results if isinstance(job.results, dict) else {},
                "iterations": job.iteration,
                "tool_calls": job.tool_calls_used,
                "llm_calls": job.llm_calls_used,
                "checkpoint": reconciliation,
            }

        # Resolve selection policy assignment once (deterministic; reused in ranking and telemetry).
        try:
            self._resolve_tool_selection_mode(
                job, state=state, selection_cfg=self._get_tool_selection_config(job)
            )
        except Exception:
            pass

        # Load deployment-level customer profile + optional per-job customer context.
        # This is a lightweight, stable signal used to tailor the research loop.
        if (
            state.get("customer_profile") is None
            and not (state.get("customer_context") or "").strip()
        ):
            try:
                from app.core.feature_flags import get_str as get_feature_str
                from app.schemas.customer_profile import CustomerProfile

                raw_profile = await get_feature_str("ai_hub_customer_profile")
                customer_profile = None
                if raw_profile:
                    try:
                        customer_profile = CustomerProfile.model_validate(
                            json.loads(raw_profile)
                        )
                    except Exception:
                        customer_profile = None

                customer_context = str(
                    (job.config or {}).get("customer_context") or ""
                ).strip()
                if not customer_context and customer_profile and customer_profile.notes:
                    customer_context = str(customer_profile.notes).strip()

                state["customer_profile"] = (
                    customer_profile.model_dump() if customer_profile else None
                )
                state["customer_context"] = customer_context
            except Exception:
                # Do not fail the job if the customer profile isn't configured.
                state["customer_profile"] = None
                state["customer_context"] = str(
                    (job.config or {}).get("customer_context") or ""
                ).strip()

        # Resolve skill profile once per run (role-aware prompt/tool constraints).
        try:
            skill_profile = self._resolve_agent_skill_profile(job, state=state)
            state["skill_profile"] = skill_profile
            if not isinstance(
                state.get("skill_profile_metrics"), dict
            ) or not state.get("skill_profile_metrics"):
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

        # Resolve execution mode once per run.
        try:
            execution_mode = self._resolve_execution_mode(job, state=state)
            job.add_log_entry(
                {
                    "phase": "execution_mode_resolved",
                    "execution_mode": execution_mode,
                }
            )
        except Exception as e:
            logger.warning(f"Failed resolving execution mode for job {job.id}: {e}")

        # Build project profile once so planning/actions can follow repo-specific structure.
        try:
            cfg = job.config if isinstance(job.config, dict) else {}
            auto_bootstrap = self._coerce_bool(
                cfg.get("project_bootstrap_auto", True), default=True
            )
            force_bootstrap = self._coerce_bool(
                cfg.get("project_bootstrap_force", False), default=False
            )
            has_profile = isinstance(state.get("project_profile"), dict) and bool(
                state.get("project_profile")
            )
            if auto_bootstrap and (force_bootstrap or not has_profile):
                source_id = self._resolve_default_source_scope(job)
                max_files = int(cfg.get("project_bootstrap_max_files", 400) or 400)
                profile = await build_project_profile(
                    job,
                    db,
                    source_id=source_id,
                    max_files=max_files,
                )
                if (
                    isinstance(profile, dict)
                    and int(profile.get("sampled_files", 0) or 0) > 0
                ):
                    state["project_profile"] = profile
                    job.add_log_entry(
                        {
                            "phase": "project_profile_bootstrap",
                            "source_id": profile.get("source_id"),
                            "sampled_files": int(profile.get("sampled_files", 0) or 0),
                            "detected_stack": profile.get("detected_stack", []),
                        }
                    )
        except Exception as e:
            logger.warning(f"Failed project bootstrap for job {job.id}: {e}")

        # Resolve memory persistence policy once per run.
        try:
            state["memory_extraction_policy"] = self._resolve_memory_extraction_policy(
                job
            )
        except Exception:
            state["memory_extraction_policy"] = {}

        # Inject relevant memories if enabled (with optional per-role overrides).
        memory_runtime = self._resolve_memory_runtime_config(job, state)
        state["memory_runtime"] = (
            memory_runtime if isinstance(memory_runtime, dict) else {}
        )
        if bool(memory_runtime.get("enabled", False)):
            try:
                memories = await agent_job_memory_service.get_relevant_memories_for_job(
                    job=job,
                    user_id=str(job.user_id),
                    db=db,
                    limit=(
                        int(memory_runtime.get("limit"))
                        if isinstance(memory_runtime.get("limit"), int)
                        else None
                    ),
                    memory_types_override=(
                        memory_runtime.get("memory_types")
                        if isinstance(memory_runtime.get("memory_types"), list)
                        else None
                    ),
                    include_chat_memory_override=(
                        bool(memory_runtime.get("include_chat_memory"))
                        if isinstance(memory_runtime.get("include_chat_memory"), bool)
                        else None
                    ),
                )
                if memories:
                    state[
                        "memory_context"
                    ] = agent_job_memory_service.format_memories_for_job_context(
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
                        feedback_learning = (
                            agent_job_memory_service.extract_feedback_learning_signals(
                                memories=memories,
                                job_type=str(job.job_type or ""),
                                role=str(
                                    (state.get("skill_profile") or {}).get("role") or ""
                                ),
                            )
                        )
                        state["feedback_learning"] = (
                            feedback_learning
                            if isinstance(feedback_learning, dict)
                            else {}
                        )
                    except Exception:
                        state["feedback_learning"] = {}
                    job.add_log_entry(
                        {
                            "phase": "memory_injection",
                            "memories_injected": len(memories),
                            "memory_types": list(set(m.memory_type for m in memories)),
                            "memory_profile": str(memory_runtime.get("profile") or "")
                            or None,
                            "memory_limit": memory_runtime.get("limit"),
                            "memory_role": str(memory_runtime.get("role") or ""),
                            "feedback_signals": (
                                {
                                    "feedback_count": int(
                                        (state.get("feedback_learning") or {}).get(
                                            "feedback_count", 0
                                        )
                                        or 0
                                    ),
                                    "preferred_tools": (
                                        (state.get("feedback_learning") or {}).get(
                                            "preferred_tools"
                                        )
                                        or []
                                    )[:5],
                                    "discouraged_tools": (
                                        (state.get("feedback_learning") or {}).get(
                                            "discouraged_tools"
                                        )
                                        or []
                                    )[:5],
                                }
                                if isinstance(state.get("feedback_learning"), dict)
                                else {}
                            ),
                        }
                    )
            except Exception as e:
                logger.warning(f"Failed to inject memories for job {job.id}: {e}")

        # Load cross-job tool priors once per execution.
        try:
            priors = await self._load_tool_priors(job, db)
            if priors:
                state["tool_priors"] = priors
                job.add_log_entry(
                    {
                        "phase": "tool_priors_loaded",
                        "tools": len(priors),
                    }
                )
        except Exception as e:
            logger.warning(f"Failed loading tool priors for job {job.id}: {e}")

        adapter = _AutonomousRuntimeAdapter(
            executor=self,
            job=job,
            agent_def=agent_def,
            user_settings=user_settings,
            state=state,
            db=db,
            start_time=start_time,
            max_runtime=max_runtime,
            progress_callback=progress_callback,
        )
        return await AgentRuntimeRunner().run(adapter)

    async def _observe(
        self,
        job: AgentJob,
        state: Dict[str, Any],
        db: AsyncSession,
    ) -> Dict[str, Any]:
        """Compatibility wrapper around the extracted observation service."""
        return await self.observation_service.observe(self, job, state, db)

    async def _think(
        self,
        job: AgentJob,
        agent_def: Optional[AgentDefinition],
        state: Dict[str, Any],
        observation: Dict[str, Any],
        user_settings: Optional[UserLLMSettings],
        db: AsyncSession,
    ) -> Dict[str, Any]:
        """Compatibility wrapper around the extracted thinking service."""
        return await self.thinking_service.think(
            self,
            job,
            agent_def,
            state,
            observation,
            user_settings,
            db,
        )

    def _parse_decision_response(
        self,
        raw_response: Any,
        job: AgentJob,
        state: Dict[str, Any],
        available_tools: List[str],
    ) -> Dict[str, Any]:
        """Compatibility wrapper around the extracted thinking service."""
        return self.thinking_service.parse_decision_response(
            self,
            raw_response=raw_response,
            job=job,
            state=state,
            available_tools=available_tools,
        )

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

        fence_match = re.search(
            r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL
        )
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
                        candidate = text[start : idx + 1]
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

    def _get_tool_fallback_policies(self) -> Dict[str, Dict[str, Dict[str, str]]]:
        """Expose tool fallback policies to extracted runtime services."""
        return _TOOL_FALLBACK_POLICIES

    def _resolve_default_source_scope(self, job: AgentJob) -> Optional[str]:
        """
        Resolve a default document source scope from job config/inherited data.

        This keeps autonomous actions anchored to the intended project/repository
        without requiring every LLM action to repeat `source_id`.
        """
        cfg = job.config if isinstance(job.config, dict) else {}

        for key in ("source_id", "target_source_id"):
            value = str(cfg.get(key) or "").strip()
            if value:
                return value

        inherited = cfg.get("inherited_data")
        if not isinstance(inherited, dict):
            return None
        parent_results = inherited.get("parent_results")
        if not isinstance(parent_results, dict):
            return None

        for bucket_key in ("repo_ingest", "generated_project", "code_patch"):
            bucket = parent_results.get(bucket_key)
            if not isinstance(bucket, dict):
                continue
            value = str(
                bucket.get("source_id") or bucket.get("target_source_id") or ""
            ).strip()
            if value:
                return value
        return None

    def _apply_default_scope_to_action(
        self,
        action: Dict[str, Any],
        job: AgentJob,
    ) -> Dict[str, Any]:
        """Inject default source scope for search-like tools when omitted."""
        if not isinstance(action, dict):
            return action
        tool = str(action.get("tool") or "").strip()
        if not tool:
            return action

        scoped_tools = {
            "search_documents",
            "search_with_filters",
            "get_knowledge_base_stats",
            "create_synthesis_document",
            "add_to_reading_list",
            "get_reading_lists",
            "save_research_finding",
            "create_document_from_text",
            "project_bootstrap",
        }
        if tool not in scoped_tools:
            return action

        params = action.get("params")
        if not isinstance(params, dict):
            params = {}

        source_id = str(params.get("source_id") or "").strip()
        if not source_id:
            default_source = self._resolve_default_source_scope(job)
            if default_source:
                params["source_id"] = default_source
        action["params"] = params
        return action

    def _get_scope_guard_config(self, job: AgentJob) -> Dict[str, Any]:
        """Get normalized source-scope guard settings."""
        cfg = job.config if isinstance(job.config, dict) else {}

        def _as_list(value: Any) -> List[str]:
            if isinstance(value, list):
                return [str(x).strip() for x in value if str(x).strip()]
            if isinstance(value, str):
                return [str(x).strip() for x in value.split(",") if str(x).strip()]
            return []

        default_write_tools = [
            "create_synthesis_document",
            "create_document_from_text",
            "save_research_finding",
            "add_to_reading_list",
        ]
        configured = _as_list(cfg.get("scope_guard_write_tools"))
        write_tools = configured if configured else default_write_tools

        return {
            "enabled": self._coerce_bool(
                cfg.get("scope_guard_enabled", True), default=True
            ),
            "enforce": self._coerce_bool(
                cfg.get("scope_guard_enforce", True), default=True
            ),
            "allow_cross_source": self._coerce_bool(
                cfg.get("scope_guard_allow_cross_source", False), default=False
            ),
            "allow_param_override": self._coerce_bool(
                cfg.get("scope_guard_allow_param_override", True), default=True
            ),
            "write_tools": write_tools,
        }

    def _validate_action_scope(
        self, job: AgentJob, action: Dict[str, Any]
    ) -> Optional[str]:
        """Return guard violation message when an action attempts cross-scope writes."""
        if not isinstance(action, dict):
            return None
        cfg = self._get_scope_guard_config(job)
        if not bool(cfg.get("enabled", True)):
            return None

        default_source = self._resolve_default_source_scope(job)
        if not default_source:
            return None

        tool = str(action.get("tool") or "").strip()
        write_tools = set(
            [str(x).strip() for x in (cfg.get("write_tools") or []) if str(x).strip()]
        )
        if tool not in write_tools:
            return None

        params = action.get("params")
        if not isinstance(params, dict):
            params = {}

        allow_cross_param = self._coerce_bool(
            params.get("allow_cross_scope"), default=False
        )
        if allow_cross_param and bool(cfg.get("allow_param_override", True)):
            return None
        if bool(cfg.get("allow_cross_source", False)):
            return None

        action_source = str(params.get("source_id") or "").strip()
        if action_source and action_source != default_source:
            return (
                f"Scope guard blocked cross-source write for tool '{tool}': "
                f"action source_id={action_source}, default source_id={default_source}"
            )
        return None

    def _resolve_scope_source(self, job: AgentJob) -> str:
        """Explain where the effective default source scope was derived from."""
        cfg = job.config if isinstance(job.config, dict) else {}
        if str(cfg.get("source_id") or "").strip():
            return "config.source_id"
        if str(cfg.get("target_source_id") or "").strip():
            return "config.target_source_id"

        inherited = cfg.get("inherited_data")
        if not isinstance(inherited, dict):
            return "none"
        parent_results = inherited.get("parent_results")
        if not isinstance(parent_results, dict):
            return "none"

        for bucket_key in ("repo_ingest", "generated_project", "code_patch"):
            bucket = parent_results.get(bucket_key)
            if not isinstance(bucket, dict):
                continue
            if str(bucket.get("source_id") or "").strip():
                return f"inherited_data.parent_results.{bucket_key}.source_id"
            if str(bucket.get("target_source_id") or "").strip():
                return f"inherited_data.parent_results.{bucket_key}.target_source_id"
        return "none"

    def _append_scope_event(
        self, state: Dict[str, Any], event: Dict[str, Any], *, max_events: int = 200
    ) -> None:
        """Append bounded scope telemetry event to state."""
        if not isinstance(event, dict):
            return
        events = state.get("scope_events")
        if not isinstance(events, list):
            events = []
        events.append(event)
        state["scope_events"] = events[-max(1, min(max_events, 2000)) :]

    def _append_step_event(
        self, state: Dict[str, Any], event: Dict[str, Any], *, max_events: int = 800
    ) -> None:
        """Append bounded per-step audit event."""
        if not isinstance(event, dict):
            return
        rows = state.get("step_events")
        if not isinstance(rows, list):
            rows = []
        row = dict(event)
        row.setdefault("at", datetime.utcnow().isoformat())
        rows.append(row)
        state["step_events"] = rows[-max(50, min(max_events, 5000)) :]

    def _append_job_result_step_event(
        self, job: AgentJob, event: Dict[str, Any], *, max_events: int = 300
    ) -> None:
        """Append a bounded step event directly into persisted job results."""
        if not isinstance(event, dict):
            return
        results = job.results if isinstance(job.results, dict) else {}
        execution = (
            results.get("execution_strategy")
            if isinstance(results.get("execution_strategy"), dict)
            else {}
        )
        rows = (
            execution.get("step_events")
            if isinstance(execution.get("step_events"), list)
            else []
        )
        row = dict(event)
        row.setdefault("at", datetime.utcnow().isoformat())
        rows.append(row)
        execution["step_events"] = rows[-max(50, min(max_events, 2000)) :]
        results["execution_strategy"] = execution
        job.results = results

    def _sync_runtime_execution_strategy(
        self,
        job: AgentJob,
        state: Dict[str, Any],
        execution: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Sync live execution diagnostics into persisted execution_strategy results."""
        runtime = self._get_execution_graph_runtime_snapshot(state)
        execution = execution if isinstance(execution, dict) else {}
        execution["step_events"] = (
            state.get("step_events")
            if isinstance(state.get("step_events"), list)
            else []
        )[-300:]
        execution["execution_graph_runtime"] = {
            **runtime,
            "nodes": (
                state.get("execution_graph_nodes")
                if isinstance(state.get("execution_graph_nodes"), list)
                else []
            )[-100:],
            "edges": (
                state.get("execution_graph_edges")
                if isinstance(state.get("execution_graph_edges"), list)
                else []
            )[-200:],
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
        }
        execution["scope_observability_runtime"] = {
            "resolved_scope_id": self._resolve_default_source_scope(job),
            "scope_source": self._resolve_scope_source(job),
            "events": (
                state.get("scope_events")
                if isinstance(state.get("scope_events"), list)
                else []
            )[-50:],
        }
        return execution

    def _persist_runtime_execution_strategy(
        self, job: AgentJob, state: Dict[str, Any]
    ) -> None:
        """Persist synced runtime execution diagnostics into job.results."""
        results = job.results if isinstance(job.results, dict) else {}
        execution = (
            results.get("execution_strategy")
            if isinstance(results.get("execution_strategy"), dict)
            else {}
        )
        results["execution_strategy"] = self._sync_runtime_execution_strategy(
            job, state, execution
        )
        job.results = results

    def _append_execution_graph_node(
        self, state: Dict[str, Any], node: Dict[str, Any], *, max_nodes: int = 500
    ) -> None:
        """Append bounded execution-graph node telemetry."""
        if not isinstance(node, dict):
            return
        nodes = state.get("execution_graph_nodes")
        if not isinstance(nodes, list):
            nodes = []
        nodes.append(node)
        state["execution_graph_nodes"] = nodes[-max(20, min(max_nodes, 5000)) :]

    def _append_execution_graph_edge(
        self, state: Dict[str, Any], edge: Dict[str, Any], *, max_edges: int = 1000
    ) -> None:
        """Append bounded execution-graph edge telemetry."""
        if not isinstance(edge, dict):
            return
        edges = state.get("execution_graph_edges")
        if not isinstance(edges, list):
            edges = []
        edges.append(edge)
        state["execution_graph_edges"] = edges[-max(40, min(max_edges, 10000)) :]

    def _build_execution_graph_stats(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Build compact DAG-style statistics for execution graph telemetry."""
        return agent_execution_graph.build_stats(nodes, edges)

    def _build_execution_graph_health(
        self, dag_stats: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Classify graph runtime quality into compact UI-friendly health status."""
        return agent_execution_graph.build_health(dag_stats)

    def _build_execution_graph_recommendations(
        self, health: Dict[str, Any]
    ) -> List[str]:
        """Create short remediation hints based on graph health signals."""
        return agent_execution_graph.build_recommendations(health)

    def _get_execution_graph_runtime_snapshot(
        self, state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build live execution-graph diagnostics for in-loop planning."""
        return agent_execution_graph.build_runtime_snapshot(state)

    def _has_graph_recovery_pressure(
        self,
        state: Dict[str, Any],
        *,
        verification_debt_threshold: int = 2,
        severity_threshold: int = 20,
    ) -> bool:
        """Return whether graph health indicates rescue/recovery pressure."""
        return agent_execution_graph.has_recovery_pressure(
            state,
            verification_debt_threshold=verification_debt_threshold,
            severity_threshold=severity_threshold,
        )

    def _format_execution_graph_for_prompt(self, state: Dict[str, Any]) -> str:
        """Render compact live execution-graph diagnostics for the planner prompt."""
        runtime = (
            state.get("execution_graph_runtime")
            if isinstance(state.get("execution_graph_runtime"), dict)
            else self._get_execution_graph_runtime_snapshot(state)
        )
        return agent_prompt_sections.format_execution_graph(runtime)

    def _annotate_execution_plan_graph(
        self, plan: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Inject stable step IDs and dependency edges into plan steps."""
        if not isinstance(plan, list):
            return []
        out: List[Dict[str, Any]] = []
        prev_step_id: Optional[str] = None
        for idx, row in enumerate(plan):
            if not isinstance(row, dict):
                continue
            step = dict(row)
            step_id = str(step.get("step_id") or "").strip() or f"step_{idx + 1}"
            step["step_id"] = step_id
            step["node_type"] = "act"
            depends_on = step.get("depends_on")
            if isinstance(depends_on, list):
                deps = [str(x).strip() for x in depends_on if str(x).strip()]
            else:
                deps = []
            if not deps and prev_step_id:
                deps = [prev_step_id]
            step["depends_on"] = deps
            out.append(step)
            prev_step_id = step_id
        return out

    def _select_verification_commands_from_profile(
        self,
        profile: Optional[Dict[str, Any]],
        *,
        max_commands: int = 3,
    ) -> List[str]:
        """
        Pick verification/test commands from an inferred project profile.

        Prioritizes test-oriented commands from `suggested_commands`, then falls back
        to stack-specific test commands.
        """
        if not isinstance(profile, dict):
            return []

        max_commands = max(1, min(int(max_commands or 3), 6))
        command_groups = (
            profile.get("command_groups")
            if isinstance(profile.get("command_groups"), dict)
            else {}
        )
        grouped_primary = (
            command_groups.get("test")
            if isinstance(command_groups.get("test"), list)
            else []
        )
        grouped_fallback = (
            command_groups.get("test_fallback")
            if isinstance(command_groups.get("test_fallback"), list)
            else []
        )
        selected: List[str] = []
        for bucket in (grouped_primary, grouped_fallback):
            for raw in bucket:
                cmd = str(raw or "").strip()
                if not cmd or cmd in selected:
                    continue
                selected.append(cmd)
                if len(selected) >= max_commands:
                    return selected[:max_commands]

        suggested = (
            profile.get("suggested_commands")
            if isinstance(profile.get("suggested_commands"), list)
            else []
        )
        for raw in suggested:
            cmd = str(raw or "").strip()
            if not cmd:
                continue
            lower = cmd.lower()
            if (
                " test" in lower
                or lower.startswith("test")
                or "pytest" in lower
                or "unittest" in lower
                or "go test" in lower
                or "dotnet test" in lower
                or "cargo test" in lower
                or "make test" in lower
            ):
                if cmd not in selected:
                    selected.append(cmd)
            if len(selected) >= max_commands:
                return selected[:max_commands]

        stacks = (
            profile.get("detected_stack")
            if isinstance(profile.get("detected_stack"), list)
            else []
        )
        stack_defaults: List[str] = []
        if "python" in stacks:
            stack_defaults.append("python -m pytest -q")
        if "node" in stacks or "typescript" in stacks or "javascript" in stacks:
            stack_defaults.append("npm test")
        if "go" in stacks:
            stack_defaults.append("go test ./...")
        if "dotnet" in stacks:
            stack_defaults.append("dotnet test")
        if "rust" in stacks:
            stack_defaults.append("cargo test")

        for cmd in stack_defaults:
            if cmd not in selected:
                selected.append(cmd)
            if len(selected) >= max_commands:
                break
        return selected[:max_commands]

    def _get_bootstrap_and_fallback_commands_from_profile(
        self,
        profile: Optional[Dict[str, Any]],
        *,
        primary_commands: Optional[List[str]] = None,
        max_install: int = 3,
        max_fallback: int = 3,
    ) -> Dict[str, List[str]]:
        """Return install/bootstrap and fallback verification commands from a project profile."""
        if not isinstance(profile, dict):
            return {"install": [], "fallback": []}

        primary = [str(x).strip() for x in (primary_commands or []) if str(x).strip()]
        command_groups = (
            profile.get("command_groups")
            if isinstance(profile.get("command_groups"), dict)
            else {}
        )
        install = (
            command_groups.get("install")
            if isinstance(command_groups.get("install"), list)
            else []
        )
        fallback = (
            command_groups.get("test_fallback")
            if isinstance(command_groups.get("test_fallback"), list)
            else []
        )

        install_out: List[str] = []
        for raw in install:
            cmd = str(raw or "").strip()
            if cmd and cmd not in install_out:
                install_out.append(cmd)
            if len(install_out) >= max(1, min(int(max_install or 3), 6)):
                break

        fallback_out: List[str] = []
        for raw in fallback:
            cmd = str(raw or "").strip()
            if not cmd or cmd in primary or cmd in fallback_out:
                continue
            fallback_out.append(cmd)
            if len(fallback_out) >= max(1, min(int(max_fallback or 3), 6)):
                break

        return {"install": install_out, "fallback": fallback_out}

    def _should_bootstrap_after_verification_failure(
        self, run: Optional[Dict[str, Any]]
    ) -> bool:
        """Heuristic: decide whether a failed verification run suggests missing environment/tooling."""
        if not isinstance(run, dict):
            return False
        if bool(run.get("ok")):
            return False

        try:
            exit_code = (
                int(run.get("exit_code")) if run.get("exit_code") is not None else None
            )
        except Exception:
            exit_code = None
        if exit_code == 127:
            return True

        stderr = str(run.get("stderr") or "").lower()
        stdout = str(run.get("stdout") or "").lower()
        text = f"{stderr}\n{stdout}"
        hints = [
            "command not found",
            "no module named",
            "cannot find module",
            "module not found",
            "pytest: not found",
            "poetry: not found",
            "npm: not found",
            "yarn: not found",
            "pnpm: not found",
            "dotnet: not found",
            "go: not found",
            "cargo: not found",
            "could not find a version that satisfies the requirement",
            "executable file not found",
        ]
        return any(hint in text for hint in hints)

    def _summarize_experiment_run_phases(
        self, runs: Optional[List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Summarize experiment phases so retry/bootstrap behavior is visible without parsing raw runs."""
        if not isinstance(runs, list):
            return {
                "phases": [],
                "verification_phases": [],
                "final_phase": "",
                "final_ok": False,
                "failed_commands": [],
            }

        phases: List[str] = []
        verification_phases: List[str] = []
        final_phase = ""
        final_ok = False
        failed_commands: List[str] = []
        verification_phase_set = {"primary", "retry_primary", "fallback"}

        for run in runs:
            if not isinstance(run, dict):
                continue
            phase = str(run.get("phase") or "").strip()
            command = str(run.get("command") or "").strip()
            if phase:
                if phase not in phases:
                    phases.append(phase)
                if phase in verification_phase_set and phase not in verification_phases:
                    verification_phases.append(phase)
                    final_phase = phase
            if not bool(run.get("ok")) and command:
                failed_commands.append(command)

        if final_phase:
            final_phase_runs = [
                run
                for run in runs
                if isinstance(run, dict)
                and str(run.get("phase") or "").strip() == final_phase
            ]
            final_ok = bool(final_phase_runs) and all(
                bool(run.get("ok")) for run in final_phase_runs
            )

        return {
            "phases": phases,
            "verification_phases": verification_phases,
            "final_phase": final_phase,
            "final_ok": final_ok,
            "failed_commands": failed_commands[:6],
        }

    def _extract_latest_failed_command_output(
        self, experiment_run: Optional[Dict[str, Any]]
    ) -> str:
        if not isinstance(experiment_run, dict):
            return ""
        runs = (
            experiment_run.get("runs")
            if isinstance(experiment_run.get("runs"), list)
            else []
        )
        for run in reversed(runs):
            if not isinstance(run, dict) or bool(run.get("ok")):
                continue
            text = str(run.get("stderr") or run.get("stdout") or "").strip()
            if text:
                return text[:4000]
        return ""

    def _build_code_patch_execution_recovery(
        self,
        *,
        job: AgentJob,
        experiment_run: Optional[Dict[str, Any]] = None,
        existing_recovery: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        existing_recovery = (
            existing_recovery if isinstance(existing_recovery, dict) else {}
        )
        experiment_run = experiment_run if isinstance(experiment_run, dict) else {}
        failed_commands = [
            str(cmd).strip()
            for cmd in (
                experiment_run.get("failed_commands")
                if isinstance(experiment_run.get("failed_commands"), list)
                else existing_recovery.get("last_failed_commands")
                if isinstance(existing_recovery.get("last_failed_commands"), list)
                else []
            )
            if str(cmd).strip()
        ][:6]
        final_phase = str(experiment_run.get("final_phase") or "").strip().lower()
        run_ok = experiment_run.get("ok")
        can_resume = bool(existing_recovery.get("can_resume_verification"))
        if (
            not can_resume
            and str(job.status or "").lower() == AgentJobStatus.PAUSED.value
        ):
            can_resume = final_phase in {
                "primary",
                "retry_primary",
                "fallback",
            } or bool(failed_commands)
        recovery_state = (
            str(existing_recovery.get("recovery_state") or "").strip().lower()
        )
        if not recovery_state:
            if failed_commands and run_ok is False:
                recovery_state = "verification_failed"
            elif can_resume:
                recovery_state = "needs_operator_retry"
            else:
                recovery_state = "relaunch_available"
        retry_reason = str(existing_recovery.get("retry_reason") or "").strip()
        if not retry_reason and failed_commands:
            retry_reason = "Verification failed and needs a refined retry."
        suggested_operator_actions = [
            str(action).strip()
            for action in (
                existing_recovery.get("suggested_operator_actions")
                if isinstance(existing_recovery.get("suggested_operator_actions"), list)
                else []
            )
            if str(action).strip()
        ]
        if not suggested_operator_actions:
            if failed_commands:
                suggested_operator_actions.append("retry_with_refined_plan")
            if can_resume:
                suggested_operator_actions.append("resume_verification")
            suggested_operator_actions.append("relaunch_clean_run")
        return {
            "recovery_state": recovery_state,
            "last_failed_commands": failed_commands,
            "retry_reason": retry_reason or None,
            "resume_hint": (
                str(existing_recovery.get("resume_hint") or "").strip()
                or (
                    "Resume verification from the paused job state."
                    if can_resume
                    else None
                )
            ),
            "suggested_operator_actions": suggested_operator_actions,
            "can_retry_with_refined_plan": bool(
                existing_recovery.get(
                    "can_retry_with_refined_plan", bool(failed_commands)
                )
            ),
            "can_resume_verification": can_resume,
            "latest_failed_output": self._extract_latest_failed_command_output(
                experiment_run
            ),
        }

    def _normalize_causal_experiment_plan(
        self,
        payload: Dict[str, Any],
        *,
        max_hypotheses: int = 4,
        max_experiments: int = 6,
    ) -> Dict[str, Any]:
        """Normalize causal experiment planner output into stable schema."""
        return agent_plan_normalization.normalize_causal_experiment_plan(
            payload,
            max_hypotheses=max_hypotheses,
            max_experiments=max_experiments,
        )

    def _fallback_causal_experiment_plan(
        self,
        job: AgentJob,
        *,
        max_hypotheses: int = 3,
        max_experiments: int = 4,
    ) -> Dict[str, Any]:
        """Deterministic fallback when LLM causal planning is unavailable."""
        return agent_plan_normalization.fallback_causal_experiment_plan(
            str(getattr(job, "goal", "") or ""),
            max_hypotheses=max_hypotheses,
            max_experiments=max_experiments,
        )

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

        findings = (
            state.get("findings") if isinstance(state.get("findings"), list) else []
        )
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
            hypotheses["source"] = str(
                hypotheses.get("source") or ("llm" if used_llm else "fallback")
            )
            state["causal_experiment_plan"] = hypotheses
            job.add_log_entry(
                {
                    "phase": "causal_experiment_plan_generated",
                    "hypotheses": len(
                        hypotheses.get("hypotheses", [])
                        if isinstance(hypotheses.get("hypotheses"), list)
                        else []
                    ),
                    "experiments": len(
                        hypotheses.get("experiments", [])
                        if isinstance(hypotheses.get("experiments"), list)
                        else []
                    ),
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
        state.setdefault("execution_plan_version", 1)
        state.setdefault("plan_replan_count", 0)
        state.setdefault("plan_completed", False)
        execution_mode = self._resolve_execution_mode(job, state=state)
        plan_then_act_enabled = self._coerce_bool(
            cfg.get("plan_then_act_enabled"), default=True
        )
        if execution_mode == "plan_and_execute":
            plan_then_act_enabled = True
        if not plan_then_act_enabled:
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
            state["execution_plan"] = self._annotate_execution_plan_graph(plan)
            state["plan_step_index"] = 0
            state["plan_completed"] = False
            if isinstance(state["execution_plan"][0], dict):
                state["execution_plan"][0]["status"] = "in_progress"
            first_step = (
                state["execution_plan"][0]
                if isinstance(state["execution_plan"], list) and state["execution_plan"]
                else {}
            )
            self._append_step_event(
                state,
                {
                    "type": "plan_initialized",
                    "plan_steps_total": len(state["execution_plan"])
                    if isinstance(state.get("execution_plan"), list)
                    else 0,
                    "plan_step_id": str(
                        (
                            first_step.get("step_id")
                            if isinstance(first_step, dict)
                            else ""
                        )
                        or ""
                    )
                    or None,
                    "plan_step_index": 0,
                    "execution_mode": execution_mode,
                },
            )
        return used_llm

    def _apply_revised_plan(
        self, state: Dict[str, Any], revised: ExecutionPlan
    ) -> None:
        """Merge a revised execution plan into state, preserving completed steps."""
        old_plan = state.get("execution_plan") or []
        completed = [
            s for s in old_plan if isinstance(s, dict) and s.get("status") == "done"
        ]
        new_steps = [s.model_dump() for s in revised.steps]
        merged = completed + [s for s in new_steps if s.get("status") != "done"]
        state["execution_plan"] = self._annotate_execution_plan_graph(merged)
        state["plan_step_index"] = len(completed)
        state["execution_plan_version"] = revised.version
        state["plan_replan_count"] = revised.replan_count
        merged_plan = ExecutionPlan(
            steps=[
                PlanStep.model_validate(step)
                for step in merged
                if isinstance(step, dict)
            ],
            subgoals=list(revised.subgoals),
            version=revised.version,
            replan_count=revised.replan_count,
            last_replanned_at=revised.last_replanned_at,
        )
        state["plan_progress"] = AgentExecutionPlanner.compute_plan_progress(
            merged_plan
        )
        if revised.subgoals:
            state["subgoals"] = [sg.model_dump() for sg in revised.subgoals]
        # Mark first pending step as in_progress
        for step in state["execution_plan"]:
            if isinstance(step, dict) and step.get("status") == "pending":
                step["status"] = "in_progress"
                break
        state["plan_completed"] = self._is_execution_plan_complete(state)

    def _normalize_execution_plan(
        self,
        payload: Dict[str, Any],
        max_steps: int = 6,
    ) -> List[Dict[str, Any]]:
        """Normalize planner output into stable step objects."""
        return agent_plan_normalization.normalize_execution_plan(payload, max_steps)

    def _fallback_execution_plan(
        self, job: AgentJob, max_steps: int = 6
    ) -> List[Dict[str, Any]]:
        """Create a deterministic fallback plan when LLM planning is unavailable."""
        return agent_plan_normalization.fallback_execution_plan(
            str(getattr(job, "job_type", "") or ""), max_steps
        )

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
            parts = [
                p.strip()
                for p in re.split(r"[.;]|(?:\s+and\s+)|(?:\s+then\s+)|,", goal)
                if p.strip()
            ]
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

        trigger = (
            str(
                cfg.get(
                    "swarm_trigger_condition", ChainTriggerCondition.ON_COMPLETE.value
                )
                or ChainTriggerCondition.ON_COMPLETE.value
            )
            .strip()
            .lower()
        )
        if trigger not in {
            ChainTriggerCondition.ON_COMPLETE.value,
            ChainTriggerCondition.ON_ANY_END.value,
            ChainTriggerCondition.ON_PROGRESS.value,
            ChainTriggerCondition.ON_FINDINGS.value,
            ChainTriggerCondition.ON_FAIL.value,
        }:
            trigger = ChainTriggerCondition.ON_COMPLETE.value

        return {
            "enabled": bool(cfg.get("swarm_child_jobs_enabled", False)),
            "max_agents": _as_int("swarm_max_agents", 4, 1, 12),
            "roles": roles,
            "inherit_results": bool(cfg.get("swarm_inherit_results", True)),
            "inherit_config": bool(cfg.get("swarm_inherit_config", False)),
            "trigger_condition": trigger,
            "max_iterations_ratio": _as_float(
                "swarm_child_max_iterations_ratio", 0.45, 0.1, 1.0
            ),
            "max_tool_calls_ratio": _as_float(
                "swarm_child_max_tool_calls_ratio", 0.45, 0.1, 1.0
            ),
            "max_llm_calls_ratio": _as_float(
                "swarm_child_max_llm_calls_ratio", 0.45, 0.1, 1.0
            ),
            "max_runtime_ratio": _as_float(
                "swarm_child_max_runtime_ratio", 0.5, 0.1, 1.0
            ),
            "min_iterations": _as_int("swarm_child_min_iterations", 6, 1, 100),
            "min_tool_calls": _as_int("swarm_child_min_tool_calls", 8, 1, 200),
            "min_llm_calls": _as_int("swarm_child_min_llm_calls", 6, 1, 200),
            "min_runtime_minutes": _as_int(
                "swarm_child_min_runtime_minutes", 10, 1, 240
            ),
            "goal_prefix": str(cfg.get("swarm_goal_prefix", "Swarm role")).strip()[:80],
            "fan_in_enabled": bool(cfg.get("swarm_fan_in_enabled", True)),
            "fan_in_name": str(cfg.get("swarm_fan_in_name", "Swarm Synthesis")).strip()[
                :120
            ],
            "fan_in_job_type": str(
                cfg.get("swarm_fan_in_job_type", "synthesis") or "synthesis"
            )
            .strip()
            .lower(),
            "fan_in_trigger_condition": str(
                cfg.get(
                    "swarm_fan_in_trigger_condition",
                    ChainTriggerCondition.ON_ANY_END.value,
                )
                or ChainTriggerCondition.ON_ANY_END.value
            )
            .strip()
            .lower(),
        }

    def _ensure_swarm_chain_config(self, job: AgentJob, state: Dict[str, Any]) -> None:
        """Create a swarm of specialized child jobs when enabled and no chain exists yet."""
        swarm_cfg = self._get_swarm_config(job)
        cfg = job.config if isinstance(job.config, dict) else {}
        if not bool(swarm_cfg.get("enabled", False)):
            return
        if bool(state.get("swarm_chain_configured", False)):
            return

        chain = job.chain_config if isinstance(job.chain_config, dict) else {}
        existing_children = chain.get("child_jobs")
        if isinstance(existing_children, list) and existing_children:
            state["swarm_chain_configured"] = True
            state["swarm_child_jobs_count"] = len(existing_children)
            chain_data_existing = (
                chain.get("chain_data")
                if isinstance(chain.get("chain_data"), dict)
                else {}
            )
            state["swarm_fan_in_enabled"] = bool(
                chain_data_existing.get("swarm_fan_in_enabled", False)
            )
            state["swarm_fan_in_group_id"] = str(
                chain_data_existing.get("swarm_fan_in_group_id") or ""
            )
            self._append_step_event(
                state,
                {
                    "type": "swarm_chain_reused",
                    "iteration": int(job.iteration or 0),
                    "child_jobs_count": len(existing_children),
                    "fan_in_enabled": bool(state.get("swarm_fan_in_enabled", False)),
                    "fan_in_group_id": str(state.get("swarm_fan_in_group_id") or ""),
                },
            )
            return

        coding_swarm_enabled = bool(
            cfg.get("coding_swarm_enabled")
            or str(cfg.get("launch_mode") or "").strip().lower()
            == "quick_start_bug_triage_swarm"
            or str(
                (cfg.get("quick_start") or {}).get("profile")
                if isinstance(cfg.get("quick_start"), dict)
                else ""
            )
            .strip()
            .lower()
            == "bug_triage_swarm"
        )
        if coding_swarm_enabled:
            from app.services.agent_coding_harness_service import (
                agent_coding_harness_service,
            )
            from app.services.agent_coding_workspace_session_service import (
                agent_coding_workspace_session_service,
            )

            role_templates = agent_coding_harness_service.get_role_catalog()
            role_template_aliases = agent_coding_harness_service.role_aliases()
            default_roles: List[Any] = [
                "reproducer",
                "root_cause",
                "patcher",
                "verifier",
            ]
            fallback_role_key = "reproducer"
        else:
            role_templates = {
                "researcher": {
                    "name": "Researcher",
                    "job_type": "research",
                    "objective": "Gather high-signal evidence from papers and internal knowledge sources.",
                    "agent_role": "researcher",
                    "config": {
                        "prefer_sources": ["documents", "arxiv"],
                        "max_documents": 10,
                        "max_papers": 8,
                    },
                },
                "researcher_documents": {
                    "name": "Knowledge Researcher",
                    "job_type": "research",
                    "objective": "Focus on internal documents and existing knowledge-base evidence.",
                    "agent_role": "researcher_documents",
                    "config": {
                        "prefer_sources": ["documents"],
                        "max_documents": 14,
                        "max_papers": 2,
                    },
                },
                "researcher_arxiv": {
                    "name": "Literature Researcher",
                    "job_type": "research",
                    "objective": "Focus on external paper discovery and validation.",
                    "agent_role": "researcher_arxiv",
                    "config": {
                        "prefer_sources": ["arxiv"],
                        "max_documents": 4,
                        "max_papers": 12,
                    },
                },
                "analyst": {
                    "name": "Analyst",
                    "job_type": "analysis",
                    "objective": "Compare sources, identify gaps/contradictions, and stress-test assumptions.",
                    "agent_role": "critic",
                    "config": {"prefer_sources": ["documents", "arxiv"]},
                },
                "critic": {
                    "name": "Critic",
                    "job_type": "analysis",
                    "objective": "Challenge assumptions and identify evidence gaps before synthesis.",
                    "agent_role": "critic",
                    "config": {"prefer_sources": ["documents", "arxiv"]},
                },
                "synthesizer": {
                    "name": "Synthesizer",
                    "job_type": "synthesis",
                    "objective": "Produce concise synthesis with traceable evidence and clear next actions.",
                    "agent_role": "synthesizer",
                    "config": {"prefer_sources": ["documents"]},
                },
                "monitor": {
                    "name": "Monitor",
                    "job_type": "monitor",
                    "objective": "Track updates and ingest newly relevant sources for the topic.",
                    "agent_role": "verifier",
                    "config": {"prefer_sources": ["arxiv", "documents"]},
                },
                "verifier": {
                    "name": "Verifier",
                    "job_type": "analysis",
                    "objective": "Verify evidence quality, consistency, and confidence before final decisions.",
                    "agent_role": "verifier",
                    "config": {"prefer_sources": ["documents", "arxiv"]},
                },
                "knowledge_expander": {
                    "name": "Knowledge Expander",
                    "job_type": "knowledge_expansion",
                    "objective": "Find adjacent concepts and add structured knowledge links.",
                    "agent_role": "researcher",
                    "config": {"prefer_sources": ["documents", "arxiv"]},
                },
            }
            role_template_aliases = {
                "research": "researcher",
                "researcher_docs": "researcher_documents",
                "document_researcher": "researcher_documents",
                "docs_researcher": "researcher_documents",
                "knowledge_researcher": "researcher_documents",
                "literature_researcher": "researcher_arxiv",
                "paper_researcher": "researcher_arxiv",
                "arxiv_researcher": "researcher_arxiv",
                "reviewer": "critic",
                "validator": "verifier",
                "qa": "verifier",
                "checker": "verifier",
                "writer": "synthesizer",
                "aggregator": "synthesizer",
                "synth": "synthesizer",
            }
            default_roles = [
                "researcher_documents",
                "researcher_arxiv",
                "analyst",
            ]
            fallback_role_key = "researcher"
        roles_raw = swarm_cfg.get("roles")
        if not isinstance(roles_raw, list) or not roles_raw:
            roles_raw = default_roles

        max_agents = int(swarm_cfg.get("max_agents", 4) or 4)
        max_agents = max(1, min(max_agents, 12))
        parent_goal = str(job.goal or "").strip()[:1600]
        fan_in_enabled = bool(swarm_cfg.get("fan_in_enabled", True))
        fan_in_trigger = (
            str(
                swarm_cfg.get(
                    "fan_in_trigger_condition", ChainTriggerCondition.ON_ANY_END.value
                )
                or ChainTriggerCondition.ON_ANY_END.value
            )
            .strip()
            .lower()
        )
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
            int(
                (job.max_iterations or 20)
                * float(swarm_cfg.get("max_iterations_ratio", 0.45) or 0.45)
            ),
        )
        child_max_tool_calls = max(
            int(swarm_cfg.get("min_tool_calls", 8) or 8),
            int(
                (job.max_tool_calls or 50)
                * float(swarm_cfg.get("max_tool_calls_ratio", 0.45) or 0.45)
            ),
        )
        child_max_llm_calls = max(
            int(swarm_cfg.get("min_llm_calls", 6) or 6),
            int(
                (job.max_llm_calls or 30)
                * float(swarm_cfg.get("max_llm_calls_ratio", 0.45) or 0.45)
            ),
        )
        child_max_runtime = max(
            int(swarm_cfg.get("min_runtime_minutes", 10) or 10),
            int(
                (job.max_runtime_minutes or 60)
                * float(swarm_cfg.get("max_runtime_ratio", 0.5) or 0.5)
            ),
        )

        allowed_job_types = {
            "research",
            "monitor",
            "analysis",
            "synthesis",
            "knowledge_expansion",
            "custom",
            "data_analysis",
        }
        fan_in_job_type = (
            str(swarm_cfg.get("fan_in_job_type", "synthesis") or "synthesis")
            .strip()
            .lower()
        )
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
            role_template_key = "researcher"

            if isinstance(raw, dict):
                role_key = (
                    str(
                        raw.get("role")
                        or raw.get("type")
                        or raw.get("name")
                        or "researcher"
                    )
                    .strip()
                    .lower()
                )
                role_key = role_key.replace("-", "_").replace(" ", "_")
                role_key = re.sub(
                    r"_+", "_", re.sub(r"[^a-z0-9_]+", "_", role_key)
                ).strip("_")
                role_template_key = (
                    role_key
                    if role_key in role_templates
                    else role_template_aliases.get(role_key, fallback_role_key)
                )
                tpl = role_templates.get(
                    role_template_key, role_templates[fallback_role_key]
                )
                role_name = str(
                    raw.get("name") or tpl.get("name") or "Researcher"
                ).strip()
                role_objective = str(
                    raw.get("objective") or tpl.get("objective") or ""
                ).strip()
                role_job_type = (
                    str(raw.get("job_type") or tpl.get("job_type") or job.job_type)
                    .strip()
                    .lower()
                )
                role_agent_role = (
                    str(
                        raw.get("agent_role")
                        or tpl.get("agent_role")
                        or role_template_key
                    )
                    .strip()
                    .lower()
                )
                role_cfg = dict(
                    tpl.get("config") if isinstance(tpl.get("config"), dict) else {}
                )
                if isinstance(raw.get("config"), dict):
                    role_cfg.update(raw.get("config") or {})
            else:
                role_token = str(raw or "").strip()
                if not role_token:
                    continue
                role_key = role_token.lower().replace("-", "_").replace(" ", "_")
                if ":" in role_key:
                    role_key, role_tag = [p.strip() for p in role_key.split(":", 1)]
                role_key = re.sub(
                    r"_+", "_", re.sub(r"[^a-z0-9_]+", "_", role_key)
                ).strip("_")
                role_template_key = (
                    role_key
                    if role_key in role_templates
                    else role_template_aliases.get(role_key, fallback_role_key)
                )
                tpl = role_templates.get(
                    role_template_key, role_templates[fallback_role_key]
                )
                role_name = str(tpl.get("name") or "Researcher").strip()
                role_objective = str(tpl.get("objective") or "").strip()
                role_job_type = str(tpl.get("job_type") or job.job_type).strip().lower()
                role_agent_role = (
                    str(tpl.get("agent_role") or role_template_key).strip().lower()
                )
                role_cfg = dict(
                    tpl.get("config") if isinstance(tpl.get("config"), dict) else {}
                )
                if role_tag:
                    role_name = f"{role_name} ({role_tag[:40]})"
                    role_objective = f"{role_objective} Focus tag: {role_tag[:120]}."

            if role_job_type not in allowed_job_types:
                role_job_type = str(job.job_type or "research")

            role_name = role_name[:120] if role_name else f"Role {idx}"
            role_names.append(role_name)
            goal_prefix = str(
                swarm_cfg.get("goal_prefix", "Swarm role") or "Swarm role"
            ).strip()[:80]
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
                        **(
                            agent_coding_workspace_session_service.child_session_config(
                                job,
                                role=role_template_key,
                                role_index=idx,
                            )
                            if coding_swarm_enabled
                            else {}
                        ),
                        "origin": "swarm_child_agent",
                        "swarm_role": role_name[:120],
                        "swarm_role_key": role_template_key[:80],
                        "agent_role": role_agent_role[:80],
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

        fan_in_group_id = hashlib.sha256(
            f"swarm_fan_in:{job.id}:{max_agents}".encode("utf-8")
        ).hexdigest()[:16]
        fan_in_template: Optional[Dict[str, Any]] = None
        if fan_in_enabled:
            coding_swarm_profile = (
                str(cfg.get("coding_swarm_profile") or "").strip().lower()
            )
            if coding_swarm_enabled and not coding_swarm_profile:
                coding_swarm_profile = "bug_triage"
            fan_in_name = str(
                swarm_cfg.get("fan_in_name", "Swarm Synthesis") or "Swarm Synthesis"
            ).strip()[:120]
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
                    "coding_swarm_enabled": coding_swarm_enabled,
                    "coding_swarm_profile": coding_swarm_profile or None,
                    "coding_harness_enabled": bool(
                        cfg.get("coding_harness_enabled", False)
                    ),
                    "coding_harness_version": str(
                        (
                            cfg.get("coding_harness")
                            if isinstance(cfg.get("coding_harness"), dict)
                            else {}
                        ).get("version")
                        or cfg.get("coding_harness_version")
                        or ""
                    ).strip()
                    or None,
                    "coding_workspace_session_id": str(
                        cfg.get("coding_workspace_session_id") or ""
                    ).strip()
                    or None,
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
                    "config": dict(
                        fan_in_template.get("config")
                        if isinstance(fan_in_template.get("config"), dict)
                        else {}
                    ),
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
        merged.setdefault(
            "trigger_condition",
            str(
                swarm_cfg.get("trigger_condition")
                or ChainTriggerCondition.ON_COMPLETE.value
            ),
        )
        merged.setdefault(
            "inherit_results", bool(swarm_cfg.get("inherit_results", True))
        )
        merged.setdefault(
            "inherit_config", bool(swarm_cfg.get("inherit_config", False))
        )
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
        self._append_step_event(
            state,
            {
                "type": "swarm_roles_configured",
                "iteration": int(job.iteration or 0),
                "child_jobs_count": len(child_jobs),
                "roles": role_names[:max_agents],
                "fan_in_enabled": fan_in_enabled,
                "fan_in_group_id": state.get("swarm_fan_in_group_id"),
                "trigger_condition": str(merged.get("trigger_condition") or ""),
            },
        )

    def _ensure_subgoal_chain_config(
        self, job: AgentJob, state: Dict[str, Any]
    ) -> None:
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
            "uncertainty_top_gap_threshold": _as_float(
                "critic_uncertainty_top_gap_threshold", 0.05, 0.0, 2.0
            ),
            "uncertainty_min_candidates": _as_int(
                "critic_uncertainty_min_candidates", 2, 2, 20
            ),
            "uncertainty_max_age_iterations": _as_int(
                "critic_uncertainty_max_age_iterations", 2, 1, 50
            ),
            "uncertainty_min_iterations_since_last": _as_int(
                "critic_uncertainty_min_iterations_since_last", 1, 1, 50
            ),
            "uncertainty_stage_schedule_enabled": bool(
                cfg.get("critic_uncertainty_stage_schedule_enabled", True)
            ),
            "uncertainty_mode_schedule_enabled": bool(
                cfg.get("critic_uncertainty_mode_schedule_enabled", True)
            ),
            "uncertainty_stage_multiplier_discovery": _as_float(
                "critic_uncertainty_stage_multiplier_discovery", 1.3, 0.1, 5.0
            ),
            "uncertainty_stage_multiplier_consolidation": _as_float(
                "critic_uncertainty_stage_multiplier_consolidation", 1.0, 0.1, 5.0
            ),
            "uncertainty_stage_multiplier_finish": _as_float(
                "critic_uncertainty_stage_multiplier_finish", 0.8, 0.1, 5.0
            ),
            "uncertainty_stage_multiplier_rescue": _as_float(
                "critic_uncertainty_stage_multiplier_rescue", 1.2, 0.1, 5.0
            ),
            "uncertainty_mode_multiplier_baseline": _as_float(
                "critic_uncertainty_mode_multiplier_baseline", 0.9, 0.1, 5.0
            ),
            "uncertainty_mode_multiplier_adaptive": _as_float(
                "critic_uncertainty_mode_multiplier_adaptive", 1.0, 0.1, 5.0
            ),
            "uncertainty_mode_multiplier_thompson": _as_float(
                "critic_uncertainty_mode_multiplier_thompson", 1.15, 0.1, 5.0
            ),
            "uncertainty_threshold_min": _as_float(
                "critic_uncertainty_threshold_min", 0.005, 0.0, 2.0
            ),
            "uncertainty_threshold_max": _as_float(
                "critic_uncertainty_threshold_max", 0.5, 0.0, 2.0
            ),
            "max_notes": _as_int("critic_max_notes", 6, 1, 20),
            "force_pivot_on_high": bool(cfg.get("critic_force_pivot_on_high", True)),
            "force_min_confidence": _as_float(
                "critic_force_min_confidence", 0.6, 0.0, 1.0
            ),
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
                "discovery": float(
                    cfg.get("uncertainty_stage_multiplier_discovery", 1.3) or 1.3
                ),
                "consolidation": float(
                    cfg.get("uncertainty_stage_multiplier_consolidation", 1.0) or 1.0
                ),
                "finish": float(
                    cfg.get("uncertainty_stage_multiplier_finish", 0.8) or 0.8
                ),
                "rescue": float(
                    cfg.get("uncertainty_stage_multiplier_rescue", 1.2) or 1.2
                ),
            }
            threshold *= float(stage_multipliers.get(stage, 1.0))

        mode = str(state.get("tool_selection_effective_mode") or "").strip().lower()
        if mode not in {"baseline", "adaptive", "thompson"}:
            mode = (
                str(
                    self._get_tool_selection_config(job).get("policy_mode", "adaptive")
                    or "adaptive"
                )
                .strip()
                .lower()
            )
            if mode not in {"baseline", "adaptive", "thompson"}:
                mode = "adaptive"
        if bool(cfg.get("uncertainty_mode_schedule_enabled", True)):
            mode_multipliers = {
                "baseline": float(
                    cfg.get("uncertainty_mode_multiplier_baseline", 0.9) or 0.9
                ),
                "adaptive": float(
                    cfg.get("uncertainty_mode_multiplier_adaptive", 1.0) or 1.0
                ),
                "thompson": float(
                    cfg.get("uncertainty_mode_multiplier_thompson", 1.15) or 1.15
                ),
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
        by_stall = bool(cfg.get("on_stall", True)) and int(
            state.get("stalled_iterations", 0) or 0
        ) >= int(cfg.get("stall_threshold", 2))
        by_graph = bool(
            cfg.get("on_stall", True)
        ) and self._has_graph_recovery_pressure(
            state,
            verification_debt_threshold=int(cfg.get("stall_threshold", 2) or 2),
            severity_threshold=20,
        )
        by_uncertainty = False
        uncertainty_gap: Optional[float] = None
        uncertainty_threshold: Optional[float] = None
        uncertainty_stage = (
            str(state.get("tool_selection_goal_stage") or "").strip().lower()
        )
        uncertainty_mode = (
            str(state.get("tool_selection_effective_mode") or "").strip().lower()
        )
        uncertainty_candidates = 0
        if bool(cfg.get("on_uncertainty", True)):
            min_since_last = int(
                cfg.get("uncertainty_min_iterations_since_last", 1) or 1
            )
            if (iteration - last_iter) >= min_since_last:
                rows = state.get("counterfactual_last")
                min_candidates = int(cfg.get("uncertainty_min_candidates", 2) or 2)
                uncertainty_candidates = len(rows) if isinstance(rows, list) else 0
                if uncertainty_candidates >= min_candidates:
                    max_age = int(cfg.get("uncertainty_max_age_iterations", 2) or 2)
                    last_cf_iteration = int(
                        state.get("counterfactual_last_iteration", 0) or 0
                    )
                    fresh_enough = (
                        True
                        if last_cf_iteration <= 0
                        else (iteration - last_cf_iteration) <= max_age
                    )
                    if fresh_enough:
                        uncertainty_gap = self._counterfactual_top_score_gap(state)
                        (
                            uncertainty_threshold,
                            uncertainty_stage,
                            uncertainty_mode,
                        ) = self._effective_uncertainty_gap_threshold(job, state, cfg)
                        if (
                            uncertainty_gap is not None
                            and uncertainty_gap <= uncertainty_threshold
                        ):
                            by_uncertainty = True

        triggered = by_interval or by_stall or by_graph or by_uncertainty
        if not triggered:
            return False

        trigger_reason = "interval"
        if by_stall:
            trigger_reason = "stall"
        if by_graph:
            trigger_reason = "graph"
        if by_uncertainty:
            trigger_reason = "uncertainty"

        trigger_payload: Dict[str, Any] = {
            "iteration": iteration,
            "reason": trigger_reason,
            "by_interval": bool(by_interval),
            "by_stall": bool(by_stall),
            "by_graph": bool(by_graph),
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
        available_tools = self._get_tools_for_job_type(
            job.job_type, job.config, profile=profile
        )
        recent_actions = (
            state.get("actions_taken", [])
            if isinstance(state.get("actions_taken"), list)
            else []
        )
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
            str(t).strip() for t in rec_tools if str(t).strip() in set(available_tools)
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
            "trajectory_assessment": str(
                payload.get("trajectory_assessment") or ""
            ).strip()[:350],
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
        available = set(
            self._get_tools_for_job_type(job.job_type, job.config, profile=profile)
        )
        combined_stats = self._merge_tool_stats(
            state.get("tool_priors")
            if isinstance(state.get("tool_priors"), dict)
            else {},
            state.get("tool_stats")
            if isinstance(state.get("tool_stats"), dict)
            else {},
        )
        exclude = str(exclude_tool or "").strip()
        findings = (
            state.get("findings", []) if isinstance(state.get("findings"), list) else []
        )
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

        if severity != "high" or confidence < float(
            cfg.get("force_min_confidence", 0.6)
        ):
            return decision

        current_action = (
            decision.get("action") if isinstance(decision.get("action"), dict) else {}
        )
        current_tool = str(current_action.get("tool") or "").strip()
        recommended = (
            note.get("recommended_tools")
            if isinstance(note.get("recommended_tools"), list)
            else []
        )
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
        decision[
            "reasoning"
        ] = f"{reasoning[:350]} Critic override applied (high risk): {pivot_txt[:220]}".strip()
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
        mode = (
            str(state.get("tool_selection_effective_mode") or "adaptive")
            .strip()
            .lower()
        )
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
        return agent_tool_scoring.normalize_tool_stats_map(raw)

    def _merge_tool_stats(
        self,
        *stats_maps: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """Merge multiple tool stat maps by summing success/failure counts."""
        return agent_tool_scoring.merge_tool_stats(*stats_maps)

    def _tool_success_ratio(self, stat: Dict[str, Any]) -> float:
        """Compute smoothed success ratio for a tool stat."""
        return agent_tool_scoring.tool_success_ratio(stat)

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
            "exploration_enabled": bool(
                cfg.get("tool_selection_exploration_enabled", True)
            ),
            "exploration_bonus": _as_float(
                "tool_selection_exploration_bonus", 0.15, 0.0, 2.0
            ),
            "cold_start_bonus": _as_float(
                "tool_selection_cold_start_bonus", 0.05, 0.0, 1.0
            ),
            "min_trials": _as_int("tool_selection_min_trials", 3, 0, 100),
            "failure_penalty": _as_float(
                "tool_selection_failure_penalty", 0.08, 0.0, 1.0
            ),
            "thompson_alpha_prior": _as_float(
                "tool_selection_thompson_alpha_prior", 1.0, 0.1, 100.0
            ),
            "thompson_beta_prior": _as_float(
                "tool_selection_thompson_beta_prior", 1.0, 0.1, 100.0
            ),
            "thompson_temperature": _as_float(
                "tool_selection_thompson_temperature", 1.0, 0.1, 5.0
            ),
            "ab_test_enabled": bool(cfg.get("tool_selection_ab_test_enabled", False)),
            "ab_test_split": _as_float("tool_selection_ab_test_split", 0.5, 0.0, 1.0),
            "ab_test_variant_a": _as_mode(
                "tool_selection_ab_test_variant_a", "adaptive"
            ),
            "ab_test_variant_b": _as_mode(
                "tool_selection_ab_test_variant_b", "thompson"
            ),
            "live_fallback_enabled": bool(
                cfg.get("tool_selection_live_fallback_enabled", True)
            ),
            "live_fallback_min_samples": _as_int(
                "tool_selection_live_fallback_min_samples", 8, 1, 10_000
            ),
            "live_fallback_min_success_rate": _as_float(
                "tool_selection_live_fallback_min_success_rate", 0.2, 0.0, 1.0
            ),
            "live_fallback_to_mode": _as_mode(
                "tool_selection_live_fallback_to_mode", "adaptive"
            ),
            "live_fallback_reset_enabled": bool(
                cfg.get("tool_selection_live_fallback_reset_enabled", True)
            ),
            "live_fallback_reset_min_samples": _as_int(
                "tool_selection_live_fallback_reset_min_samples", 10, 1, 10_000
            ),
            "live_fallback_reset_min_success_rate": _as_float(
                "tool_selection_live_fallback_reset_min_success_rate", 0.55, 0.0, 1.0
            ),
            "stage_schedule_enabled": bool(
                cfg.get("tool_selection_stage_schedule_enabled", False)
            ),
            "stage_discovery_mode": _as_mode(
                "tool_selection_stage_discovery_mode", "thompson"
            ),
            "stage_consolidation_mode": _as_mode(
                "tool_selection_stage_consolidation_mode", "adaptive"
            ),
            "stage_finish_mode": _as_mode(
                "tool_selection_stage_finish_mode", "baseline"
            ),
            "stage_rescue_mode": _as_mode(
                "tool_selection_stage_rescue_mode", "adaptive"
            ),
            "stage_rescue_stall_threshold": _as_int(
                "tool_selection_stage_rescue_stall_threshold", 3, 1, 100
            ),
            "stage_finish_progress": _as_int(
                "tool_selection_stage_finish_progress", 80, 10, 100
            ),
            "stage_discovery_progress": _as_int(
                "tool_selection_stage_discovery_progress", 35, 0, 90
            ),
            "family_diversification_enabled": bool(
                cfg.get("tool_selection_family_diversification_enabled", True)
            ),
            "family_diversification_window": _as_int(
                "tool_selection_family_diversification_window", 6, 1, 100
            ),
            "family_diversification_bonus": _as_float(
                "tool_selection_family_diversification_bonus", 0.08, 0.0, 1.0
            ),
            "family_diversification_target_unique": _as_int(
                "tool_selection_family_diversification_target_unique", 3, 1, 20
            ),
            "feedback_learning_enabled": bool(
                cfg.get("feedback_learning_enabled", True)
            ),
            "feedback_learning_weight": _as_float(
                "feedback_learning_weight", 0.08, 0.0, 0.6
            ),
            "feedback_learning_max_abs_bias": _as_float(
                "feedback_learning_max_abs_bias", 0.3, 0.0, 1.0
            ),
        }

    def _stable_fraction(self, key: str) -> float:
        """Map a key to stable [0,1) fraction."""
        return agent_tool_scoring.stable_fraction(key)

    def _derive_goal_stage(
        self,
        state: Dict[str, Any],
        selection_cfg: Dict[str, Any],
    ) -> str:
        """Derive a coarse execution stage for policy scheduling."""
        progress = int(state.get("goal_progress", 0) or 0)
        stalled = int(state.get("stalled_iterations", 0) or 0)
        findings = len(
            state.get("findings", []) if isinstance(state.get("findings"), list) else []
        )

        rescue_threshold = int(
            selection_cfg.get("stage_rescue_stall_threshold", 3) or 3
        )
        finish_progress = int(selection_cfg.get("stage_finish_progress", 80) or 80)
        discovery_progress = int(
            selection_cfg.get("stage_discovery_progress", 35) or 35
        )
        graph_pressure = self._has_graph_recovery_pressure(
            state,
            verification_debt_threshold=max(1, rescue_threshold),
            severity_threshold=20,
        )

        if stalled >= rescue_threshold or graph_pressure:
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
        """Compatibility wrapper around runtime policy service."""
        return self.runtime_policy_service.resolve_tool_selection_mode(
            self,
            job,
            state=state,
            selection_cfg=selection_cfg,
        )

    def _maybe_reset_live_mode_override(
        self,
        job: AgentJob,
        state: Dict[str, Any],
        selection_cfg: Dict[str, Any],
    ) -> None:
        """Clear an existing fallback override when the override mode recovers."""
        if not bool(selection_cfg.get("live_fallback_reset_enabled", True)):
            return
        current_override = (
            str(state.get("tool_selection_mode_override") or "").strip().lower()
        )
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
        min_samples = int(
            selection_cfg.get("live_fallback_reset_min_samples", 10) or 10
        )
        if samples < min_samples:
            return
        success_rate = float(success) / float(max(1, samples))
        min_rate = float(
            selection_cfg.get("live_fallback_reset_min_success_rate", 0.55) or 0.55
        )
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

        fallback_mode = (
            str(selection_cfg.get("live_fallback_to_mode", "adaptive") or "adaptive")
            .strip()
            .lower()
        )
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
        min_rate = float(
            selection_cfg.get("live_fallback_min_success_rate", 0.2) or 0.2
        )
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

        top_score = _as_float(
            top_row.get("priority_score") if isinstance(top_row, dict) else 0.0
        )
        selected_score = _as_float(
            selected_row.get("priority_score")
            if isinstance(selected_row, dict)
            else 0.0
        )
        runner_score = _as_float(
            runner_row.get("priority_score") if isinstance(runner_row, dict) else 0.0
        )

        return {
            "selected_tool": tool,
            "effective_mode": str(state.get("tool_selection_effective_mode") or ""),
            "goal_stage": str(state.get("tool_selection_goal_stage") or ""),
            "mode_override": str(state.get("tool_selection_mode_override") or ""),
            "selected_rank": int(selected_row.get("rank", 0) or 0)
            if isinstance(selected_row, dict)
            else 0,
            "selected_score": round(selected_score, 6),
            "top_tool": str(top_row.get("tool") or "")
            if isinstance(top_row, dict)
            else "",
            "top_score": round(top_score, 6),
            "score_gap_to_top": round(top_score - selected_score, 6),
            "score_gap_top_vs_runner_up": round(top_score - runner_score, 6),
            "candidate_count": len(ranked),
            "fallback_event_count": len(
                state.get("tool_selection_fallback_events", [])
                if isinstance(state.get("tool_selection_fallback_events"), list)
                else []
            ),
        }

    def _tool_observation_count(self, stat: Dict[str, Any]) -> int:
        """Return total observed outcomes for a tool."""
        return agent_tool_scoring.tool_observation_count(stat)

    def _tool_family(self, tool: str) -> str:
        """Map a tool to a coarse family for diversification incentives."""
        return agent_tool_scoring.tool_family(tool)

    def _family_diversification_bonus(
        self,
        tool: str,
        *,
        state: Optional[Dict[str, Any]],
        selection_cfg: Optional[Dict[str, Any]],
    ) -> float:
        """Boost underrepresented tool families based on recent action history."""
        return agent_tool_scoring.family_diversification_bonus(
            tool, state=state, selection_cfg=selection_cfg
        )

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
        """Score a tool for adaptive selection."""
        return agent_tool_scoring.tool_priority_score(
            stat,
            total_trials=total_trials,
            selection_cfg=selection_cfg,
            mode=mode,
            tool_name=tool_name,
            job_id=str(getattr(job, "id", "") or "") if job is not None else "",
            iteration=int(getattr(job, "iteration", 0) or 0) if job is not None else 0,
            state=state,
            context_tag=context_tag,
        )

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
        cfg = self._get_tool_selection_config(job)
        mode, _assignment = self._resolve_tool_selection_mode(
            job, state=state, selection_cfg=cfg
        )
        return agent_tool_scoring.rank_tools_for_selection(
            tools,
            combined_stats,
            selection_cfg=cfg,
            mode=mode,
            job_id=str(getattr(job, "id", "") or ""),
            iteration=int(getattr(job, "iteration", 0) or 0),
            state=state,
            context_tag=context_tag,
        )

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
        available = self._get_tools_for_job_type(
            job.job_type, job.config, profile=profile
        )
        if not available:
            return []

        combined_stats = self._merge_tool_stats(
            state.get("tool_priors")
            if isinstance(state.get("tool_priors"), dict)
            else {},
            state.get("tool_stats")
            if isinstance(state.get("tool_stats"), dict)
            else {},
        )
        ranked = self._rank_tools_for_selection(
            job,
            available,
            combined_stats,
            state=state,
            context_tag=context_tag or "counterfactual",
        )
        cfg = self._get_tool_selection_config(job)
        mode = (
            str(
                state.get("tool_selection_effective_mode")
                or cfg.get("policy_mode")
                or "adaptive"
            )
            .strip()
            .lower()
        )
        total_trials = sum(
            self._tool_observation_count(combined_stats.get(t, {})) for t in available
        )
        selected = str(selected_tool or "").strip()
        top_k = max(1, min(int(limit or 3), 10))

        out: List[Dict[str, Any]] = []
        for idx, tool in enumerate(ranked[:top_k], start=1):
            stat = (
                combined_stats.get(tool, {}) if isinstance(combined_stats, dict) else {}
            )
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
        tools = sorted(
            [t for t, s in stats.items() if self._tool_observation_count(s) > 0]
        )
        if not tools:
            return {
                "steps": 0,
                "seed": seed,
                "tools": [],
                "modes": {},
            }

        total_steps = max(10, min(int(steps or 200), 50_000))
        modes = (
            policy_modes
            if isinstance(policy_modes, list) and policy_modes
            else ["baseline", "adaptive", "thompson"]
        )
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
                total_trials = sum(
                    self._tool_observation_count(sim_stats[t]) for t in tools
                )
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
                draw_seed = int(
                    hashlib.sha256(draw_key.encode("utf-8")).hexdigest()[:16], 16
                )
                rng = random.Random(draw_seed)
                reward = rng.random() < chosen_rate
                slot = sim_stats.get(chosen) or {
                    "success": 0,
                    "failure": 0,
                    "last_error": "",
                }
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
                "realized_regret_vs_best": max(
                    0.0, best_rate - (float(successes) / float(max(1, total_steps)))
                ),
                "cumulative_expected_regret": cumulative_expected_regret,
                "mean_expected_regret": cumulative_expected_regret
                / float(max(1, total_steps)),
                "unique_tools_selected": len(selected_tools),
                "selection_counts": selection_counts,
            }

        comparison: List[Dict[str, Any]] = []
        for mode, stats_out in out_modes.items():
            comparison.append(
                {
                    "mode": mode,
                    "mean_reward": float(stats_out.get("mean_reward", 0.0) or 0.0),
                    "realized_regret_vs_best": float(
                        stats_out.get("realized_regret_vs_best", 0.0) or 0.0
                    ),
                    "cumulative_expected_regret": float(
                        stats_out.get("cumulative_expected_regret", 0.0) or 0.0
                    ),
                    "mean_expected_regret": float(
                        stats_out.get("mean_expected_regret", 0.0) or 0.0
                    ),
                    "unique_tools_selected": int(
                        stats_out.get("unique_tools_selected", 0) or 0
                    ),
                }
            )
        comparison.sort(
            key=lambda r: (
                -float(r.get("mean_reward", 0.0) or 0.0),
                float(r.get("cumulative_expected_regret", 0.0) or 0.0),
            )
        )

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
            "every_n_stalled_iterations": _as_int(
                "tool_selection_forced_exploration_every_n", 2, 1, 20
            ),
            "min_stalled_iterations": _as_int(
                "tool_selection_forced_exploration_min_stalled", 2, 1, 50
            ),
            "max_observations": _as_int(
                "tool_selection_forced_exploration_max_observations", 2, 0, 100
            ),
            "max_failures_per_tool": _as_int(
                "tool_selection_forced_exploration_max_failures", 8, 0, 100
            ),
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
            "cooldown_iterations": _as_int(
                "tool_selection_cooldown_iterations", 2, 1, 30
            ),
            "forced_only": bool(cfg.get("tool_selection_cooldown_forced_only", True)),
            "on_failure_extra_iterations": _as_int(
                "tool_selection_cooldown_failure_extra_iterations", 2, 0, 30
            ),
            "on_success_shorten_by": _as_int(
                "tool_selection_cooldown_success_shorten_by", 1, 0, 30
            ),
        }

    def _is_tool_in_cooldown(
        self,
        tool: str,
        cooldowns: Dict[str, Any],
        current_iteration: int,
    ) -> bool:
        """Return true if a tool is still under cooldown at current iteration."""
        return agent_tool_scoring.is_tool_in_cooldown(
            tool, cooldowns, current_iteration
        )

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
            state["forced_exploration_successes"] = (
                int(state.get("forced_exploration_successes", 0) or 0) + 1
            )
        else:
            state["forced_exploration_failures"] = (
                int(state.get("forced_exploration_failures", 0) or 0) + 1
            )

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
        return ((stalled >= min_stalled) and (stalled % cadence == 0)) or (
            (repeated >= min_stalled) and (repeated % cadence == 0)
        )

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
        default_source = self._resolve_default_source_scope(job)

        def _with_source(p: Dict[str, Any]) -> Dict[str, Any]:
            out = dict(p)
            if default_source and not str(out.get("source_id") or "").strip():
                out["source_id"] = default_source
            return out

        if t == "search_documents":
            return {
                "tool": t,
                "params": _with_source({"query": goal[:200], "limit": 10}),
                "purpose": purpose,
            }
        if t == "project_bootstrap":
            return {
                "tool": t,
                "params": _with_source({"max_files": 400}),
                "purpose": purpose,
            }
        if t == "search_arxiv":
            return {
                "tool": t,
                "params": {"query": goal[:140], "max_results": 8},
                "purpose": purpose,
            }
        if t == "search_with_filters":
            return {
                "tool": t,
                "params": _with_source({"query": goal[:200], "limit": 20}),
                "purpose": purpose,
            }
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
        return agent_tool_scoring.apply_decay_to_prior_counts(
            success_count,
            failure_count,
            updated_at,
            now=now,
            enabled=enabled,
            half_life_days=half_life_days,
            min_factor=min_factor,
        )

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
                .order_by(
                    desc(AgentToolPrior.updated_at), desc(AgentToolPrior.success_count)
                )
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
                AgentJob.status.in_(
                    [AgentJobStatus.COMPLETED.value, AgentJobStatus.FAILED.value]
                ),
            )
            .order_by(desc(AgentJob.completed_at), desc(AgentJob.created_at))
            .limit(lookback_jobs)
        )
        rows = result.scalars().all()

        aggregated: Dict[str, Dict[str, Any]] = {}
        for prev in rows:
            prev_results = prev.results if isinstance(prev.results, dict) else {}
            strategy = (
                prev_results.get("execution_strategy")
                if isinstance(prev_results.get("execution_strategy"), dict)
                else {}
            )
            stats = (
                strategy.get("tool_stats")
                if isinstance(strategy.get("tool_stats"), dict)
                else {}
            )
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

    def _resolve_execution_mode(
        self,
        job: AgentJob,
        *,
        state: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Resolve execution mode for the autonomous loop."""
        cfg = job.config if isinstance(job.config, dict) else {}
        quick_start = (
            cfg.get("quick_start") if isinstance(cfg.get("quick_start"), dict) else {}
        )
        mode_raw = (
            cfg.get("execution_mode")
            or quick_start.get("execution_mode")
            or (
                "plan_and_execute"
                if self._coerce_bool(cfg.get("plan_and_execute_enabled"), default=False)
                else ""
            )
        )
        token = str(mode_raw or "").strip().lower().replace("-", "_").replace(" ", "_")
        if token in {
            "plan_and_execute",
            "plan_then_act",
            "plan_execute",
            "planner_executor",
        }:
            mode = "plan_and_execute"
        else:
            mode = "adaptive"
        if isinstance(state, dict):
            state["execution_mode"] = mode
        return mode

    def _is_execution_plan_complete(self, state: Dict[str, Any]) -> bool:
        """Return True when every plan step is marked done."""
        plan = state.get("execution_plan")
        if not isinstance(plan, list) or not plan:
            return False
        for row in plan:
            if not isinstance(row, dict):
                continue
            if str(row.get("status") or "").strip().lower() != "done":
                return False
        return True

    def _collect_recent_document_ids(
        self, state: Dict[str, Any], limit: int = 8
    ) -> List[str]:
        """Collect deduplicated document ids from findings/action history."""
        out: List[str] = []
        max_items = max(1, min(int(limit or 8), 30))
        findings = (
            state.get("findings") if isinstance(state.get("findings"), list) else []
        )
        for row in findings:
            if not isinstance(row, dict):
                continue
            did = str(row.get("document_id") or row.get("id") or "").strip()
            if did and did not in out:
                out.append(did)
                if len(out) >= max_items:
                    return out

        actions_taken = (
            state.get("actions_taken")
            if isinstance(state.get("actions_taken"), list)
            else []
        )
        for row in reversed(actions_taken[-40:]):
            if not isinstance(row, dict):
                continue
            result = row.get("result") if isinstance(row.get("result"), dict) else {}
            artifacts = (
                result.get("artifacts")
                if isinstance(result.get("artifacts"), list)
                else []
            )
            for art in artifacts:
                if not isinstance(art, dict):
                    continue
                if str(art.get("type") or "").strip().lower() != "document":
                    continue
                did = str(art.get("document_id") or art.get("id") or "").strip()
                if did and did not in out:
                    out.append(did)
                    if len(out) >= max_items:
                        return out
        return out

    def _enforce_plan_step_action(
        self,
        job: AgentJob,
        state: Dict[str, Any],
        action: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """In plan-and-execute mode, bias action to current step suggested tools."""
        if not isinstance(action, dict) or not action:
            return action
        if self._resolve_execution_mode(job, state=state) != "plan_and_execute":
            return action

        plan = (
            state.get("execution_plan")
            if isinstance(state.get("execution_plan"), list)
            else []
        )
        if not plan:
            return action
        idx = int(state.get("plan_step_index", 0) or 0)
        idx = max(0, min(idx, len(plan) - 1))
        step = plan[idx] if isinstance(plan[idx], dict) else {}
        suggested = (
            step.get("suggested_tools")
            if isinstance(step.get("suggested_tools"), list)
            else []
        )
        if not suggested:
            return action

        profile = (
            state.get("skill_profile")
            if isinstance(state.get("skill_profile"), dict)
            else self._resolve_agent_skill_profile(job, state=state)
        )
        available = set(
            self._get_tools_for_job_type(job.job_type, job.config, profile=profile)
        )
        ordered_suggested: List[str] = []
        seen: set[str] = set()
        for raw in suggested:
            tool = str(raw or "").strip()
            if not tool or tool in seen:
                continue
            seen.add(tool)
            if tool in available:
                ordered_suggested.append(tool)

        if not ordered_suggested:
            return action

        selected = str(action.get("tool") or "").strip()
        if selected in set(ordered_suggested):
            return action

        doc_ids = self._collect_recent_document_ids(state, limit=8)
        for candidate in ordered_suggested:
            replacement = self._build_action_for_tool(
                tool=candidate,
                job=job,
                doc_ids=doc_ids,
                purpose=f"Follow current plan step suggested tool: {candidate}",
            )
            if not replacement:
                continue
            self._append_step_event(
                state,
                {
                    "type": "plan_action_adjusted",
                    "iteration": int(job.iteration or 0),
                    "plan_step_id": str(
                        step.get("step_id") or f"step_{idx + 1}"
                    ).strip(),
                    "plan_step_index": int(idx),
                    "from_tool": selected or None,
                    "to_tool": candidate,
                },
            )
            return replacement

        return action

    def _is_step_exit_criteria_satisfied(
        self,
        step: Dict[str, Any],
        action: Optional[Dict[str, Any]],
        action_result: Optional[Dict[str, Any]],
        *,
        progress_delta: int = 0,
    ) -> Tuple[bool, str]:
        """Best-effort gate for plan step completion using step exit criteria text."""
        if not isinstance(step, dict):
            return True, ""

        criteria_raw = str(step.get("exit_criteria") or "").strip()
        if not criteria_raw:
            return True, ""

        criteria = criteria_raw.lower()
        action_tool = str((action or {}).get("tool") or "").strip().lower()
        success = bool((action_result or {}).get("success", False))
        findings_count = len((action_result or {}).get("findings") or [])
        artifacts_count = len((action_result or {}).get("artifacts") or [])
        if not success:
            return False, "action_not_successful"

        unmet: List[str] = []
        wants_findings = any(
            token in criteria
            for token in [
                "finding",
                "evidence",
                "insight",
                "fact",
                "source",
                "collect",
                "gather",
                "retrieve",
                "search",
            ]
        )
        if wants_findings and findings_count <= 0 and int(progress_delta or 0) < 2:
            unmet.append("findings")

        wants_output = any(
            token in criteria
            for token in [
                "summary",
                "synthesis",
                "report",
                "artifact",
                "document",
                "write",
                "draft",
            ]
        )
        output_tools = {
            "create_synthesis_document",
            "create_document_from_text",
            "write_progress_report",
        }
        if wants_output and artifacts_count <= 0 and action_tool not in output_tools:
            unmet.append("output_artifact")

        wants_analysis = any(
            token in criteria
            for token in [
                "analy",
                "compare",
                "validate",
                "verify",
                "review",
                "critique",
                "gap",
                "contradiction",
                "risk",
            ]
        )
        analysis_tools = {
            "compare_documents",
            "compare_methodologies",
            "identify_research_gaps",
            "build_research_graph",
        }
        if (
            wants_analysis
            and findings_count <= 0
            and action_tool not in analysis_tools
            and int(progress_delta or 0) < 3
        ):
            unmet.append("analysis_signal")

        if unmet:
            return False, ",".join(unmet[:3])
        return True, ""

    def _advance_execution_plan_state(
        self,
        state: Dict[str, Any],
        action: Optional[Dict[str, Any]],
        action_result: Optional[Dict[str, Any]],
        previous_progress: int,
        current_progress: int,
        iteration: int = 0,
    ) -> None:
        """Advance plan step when a meaningful action completes."""
        plan = state.get("execution_plan")
        if not isinstance(plan, list) or not plan:
            return
        state.setdefault("plan_completed", False)

        idx = int(state.get("plan_step_index", 0) or 0)
        idx = max(0, min(idx, len(plan) - 1))
        delta = int(current_progress or 0) - int(previous_progress or 0)

        mode = str(state.get("execution_mode") or "adaptive").strip().lower()
        action_tool = str((action or {}).get("tool") or "").strip()
        action_success = bool((action_result or {}).get("success"))
        findings_count = len((action_result or {}).get("findings") or [])
        artifacts_count = len((action_result or {}).get("artifacts") or [])

        should_advance = False
        if delta >= 4:
            should_advance = True
        elif action_success and findings_count > 0:
            should_advance = True
        elif action_success and artifacts_count > 0:
            should_advance = True
        elif action_success and action_tool in {
            "create_synthesis_document",
            "create_document_from_text",
            "write_progress_report",
        }:
            should_advance = True

        # Keep at least one chance for the current step before advancing solely on small wins.
        step = plan[idx] if isinstance(plan[idx], dict) else {}
        if bool((action_result or {}).get("deferred_external")):
            if isinstance(step, dict):
                step["status"] = "waiting_external"
                data = (action_result or {}).get("data")
                if isinstance(data, dict) and data.get("outbox_id"):
                    step["external_outbox_id"] = str(data["outbox_id"])
            return
        completions = (
            int(step.get("completions", 0) or 0) if isinstance(step, dict) else 0
        )
        if (
            should_advance
            and delta < 4
            and completions < 1
            and mode != "plan_and_execute"
        ):
            should_advance = False
        if mode == "plan_and_execute" and not action_success:
            should_advance = False
        if should_advance and mode == "plan_and_execute":
            criteria_ok, criteria_reason = self._is_step_exit_criteria_satisfied(
                step if isinstance(step, dict) else {},
                action,
                action_result,
                progress_delta=delta,
            )
            if not criteria_ok:
                should_advance = False
                if isinstance(step, dict):
                    blocked_iter = int(
                        step.get("exit_criteria_blocked_iteration", -1) or -1
                    )
                    if blocked_iter != int(iteration or 0):
                        self._append_step_event(
                            state,
                            {
                                "type": "step_exit_not_met",
                                "iteration": int(iteration or 0),
                                "plan_step_id": str(
                                    step.get("step_id") or f"step_{idx + 1}"
                                ).strip(),
                                "plan_step_index": int(idx),
                                "reason": criteria_reason,
                                "exit_criteria": str(step.get("exit_criteria") or "")[
                                    :220
                                ],
                                "tool": action_tool or None,
                            },
                        )
                    step["exit_criteria_blocked_iteration"] = int(iteration or 0)
            elif isinstance(step, dict) and "exit_criteria_blocked_iteration" in step:
                step.pop("exit_criteria_blocked_iteration", None)

        if isinstance(step, dict):
            step["completions"] = completions + 1
            if step.get("status") != "done":
                step["status"] = "in_progress"

        if not should_advance:
            return

        if isinstance(step, dict):
            step["status"] = "done"
            self._append_step_event(
                state,
                {
                    "type": "step_completed",
                    "iteration": int(iteration or 0),
                    "plan_step_id": str(
                        step.get("step_id") or f"step_{idx + 1}"
                    ).strip(),
                    "plan_step_index": int(idx),
                    "tool": action_tool or None,
                    "progress_before": int(previous_progress or 0),
                    "progress_after": int(current_progress or 0),
                },
            )
        next_idx = min(len(plan) - 1, idx + 1)
        state["plan_step_index"] = next_idx
        if (
            next_idx != idx
            and isinstance(plan[next_idx], dict)
            and plan[next_idx].get("status") != "done"
        ):
            plan[next_idx]["status"] = "in_progress"
            self._append_step_event(
                state,
                {
                    "type": "step_started",
                    "iteration": int(iteration or 0),
                    "plan_step_id": str(
                        plan[next_idx].get("step_id") or f"step_{next_idx + 1}"
                    ).strip(),
                    "plan_step_index": int(next_idx),
                    "triggered_by_step_id": str(
                        step.get("step_id") or f"step_{idx + 1}"
                    ).strip()
                    if isinstance(step, dict)
                    else None,
                },
            )
        if next_idx == idx and self._is_execution_plan_complete(state):
            state["plan_completed"] = True
            self._append_step_event(
                state,
                {
                    "type": "plan_completed",
                    "iteration": int(iteration or 0),
                    "plan_steps_total": len(plan),
                },
            )

        # Keep subgoal index aligned with plan progress when possible.
        subgoals = state.get("subgoals")
        if isinstance(subgoals, list) and subgoals:
            sidx = int(state.get("subgoal_index", 0) or 0)
            sidx = max(0, min(sidx, len(subgoals) - 1))
            if isinstance(subgoals[sidx], dict):
                subgoals[sidx]["status"] = "done"
            next_sidx = min(len(subgoals) - 1, sidx + 1)
            state["subgoal_index"] = next_sidx
            if (
                next_sidx != sidx
                and isinstance(subgoals[next_sidx], dict)
                and subgoals[next_sidx].get("status") != "done"
            ):
                subgoals[next_sidx]["status"] = "in_progress"

    def _get_execution_graph_config(self, job: AgentJob) -> Dict[str, Any]:
        """Get normalized plan->act->verify->summarize settings."""
        cfg = job.config if isinstance(job.config, dict) else {}

        def _as_list(value: Any) -> List[str]:
            if isinstance(value, list):
                return [str(x).strip() for x in value if str(x).strip()]
            if isinstance(value, str):
                return [str(x).strip() for x in value.split(",") if str(x).strip()]
            return []

        verify_tools_default = [
            "create_synthesis_document",
            "create_document_from_text",
            "save_research_finding",
            "add_to_reading_list",
            "create_knowledge_base_entry",
        ]
        verify_tools = _as_list(cfg.get("execution_graph_verify_on_tools"))
        if not verify_tools:
            verify_tools = verify_tools_default

        summarize_every_raw = cfg.get("execution_graph_summarize_every_n_iterations", 1)
        try:
            summarize_every = int(summarize_every_raw or 1)
        except Exception:
            summarize_every = 1

        return {
            "enabled": self._coerce_bool(
                cfg.get("execution_graph_enabled", True), default=True
            ),
            "verify_enabled": self._coerce_bool(
                cfg.get("execution_graph_verify_enabled", True), default=True
            ),
            "summarize_enabled": self._coerce_bool(
                cfg.get("execution_graph_summarize_enabled", True), default=True
            ),
            "verify_on_tools": verify_tools,
            "summarize_every_n_iterations": max(1, min(summarize_every, 20)),
        }

    def _build_verification_action(
        self,
        job: AgentJob,
        primary_action: Dict[str, Any],
        primary_result: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Build a deterministic verification action for a successful write-style action."""
        tool = str((primary_action or {}).get("tool") or "").strip()
        if not tool or not bool((primary_result or {}).get("success", False)):
            return None

        source_id = str(
            ((primary_action or {}).get("params") or {}).get("source_id") or ""
        ).strip() or self._resolve_default_source_scope(job)
        data = (
            primary_result.get("data")
            if isinstance(primary_result.get("data"), dict)
            else {}
        )
        artifacts = (
            primary_result.get("artifacts")
            if isinstance(primary_result.get("artifacts"), list)
            else []
        )

        def _doc_id_from_result() -> Optional[str]:
            did = str(data.get("document_id") or "").strip()
            if did:
                return did
            for art in artifacts:
                if not isinstance(art, dict):
                    continue
                if str(art.get("type") or "").strip() == "document":
                    candidate = str(
                        art.get("id") or art.get("document_id") or ""
                    ).strip()
                    if candidate:
                        return candidate
            return None

        if tool in {"create_document_from_text", "create_synthesis_document"}:
            did = _doc_id_from_result()
            if did:
                return {
                    "tool": "get_document_details",
                    "params": {"document_id": did},
                    "purpose": f"Verify persisted document output for {tool}.",
                }
            return {
                "tool": "get_research_findings",
                "params": {"limit": 20},
                "purpose": f"Verify evidence state after {tool}.",
            }

        if tool == "save_research_finding":
            category = str(
                ((primary_action or {}).get("params") or {}).get("category") or ""
            ).strip()
            params: Dict[str, Any] = {"limit": 20}
            if category:
                params["category"] = category
            return {
                "tool": "get_research_findings",
                "params": params,
                "purpose": "Verify that the finding is queryable.",
            }

        if tool == "add_to_reading_list":
            list_name = str(
                ((primary_action or {}).get("params") or {}).get("list_name") or ""
            ).strip()
            params: Dict[str, Any] = {"include_items": True}
            if list_name:
                params["list_name"] = list_name
            if source_id:
                params["source_id"] = source_id
            return {
                "tool": "get_reading_lists",
                "params": params,
                "purpose": "Verify reading-list update was persisted.",
            }

        if tool == "create_knowledge_base_entry":
            params: Dict[str, Any] = {"recent_limit": 10}
            if source_id:
                params["source_id"] = source_id
            return {
                "tool": "get_knowledge_base_stats",
                "params": params,
                "purpose": "Verify knowledge-base entry impact in source stats.",
            }

        return None

    def _build_summarize_action(
        self,
        job: AgentJob,
        state: Dict[str, Any],
        primary_action: Dict[str, Any],
        primary_result: Dict[str, Any],
        verification_action: Optional[Dict[str, Any]],
        verification_result: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Build a deterministic summarize node action."""
        if not bool((primary_result or {}).get("success", False)):
            return None

        action_tool = str((primary_action or {}).get("tool") or "").strip()
        verify_tool = str((verification_action or {}).get("tool") or "").strip()
        verify_ok = bool((verification_result or {}).get("success", False))
        findings_total = len(
            state.get("findings", []) if isinstance(state.get("findings"), list) else []
        )

        summary = (
            f"Execution graph summary (iteration {int(job.iteration or 0)}): "
            f"action={action_tool} success=true; "
            f"verify={verify_tool or 'none'} success={str(verify_ok).lower()}; "
            f"findings_total={findings_total}."
        )
        return {
            "tool": "write_progress_report",
            "params": {
                "summary": summary,
                "completed_tasks": [f"act:{action_tool}"]
                + ([f"verify:{verify_tool}"] if verify_tool else []),
                "pending_tasks": [],
                "key_findings": [],
                "blockers": []
                if verify_ok or not verify_tool
                else [f"verification failed for {verify_tool}"],
                "next_steps": ["Continue next planned step with scoped evidence."],
            },
            "purpose": "Summarize act+verify node outcomes for deterministic traceability.",
        }

    def _format_execution_plan_for_prompt(self, state: Dict[str, Any]) -> str:
        """Render current plan context for decision prompts."""
        return agent_prompt_sections.format_execution_plan(state)

    def _format_causal_experiment_plan_for_prompt(self, state: Dict[str, Any]) -> str:
        """Render causal experiment context for research decisions."""
        return agent_prompt_sections.format_causal_experiment_plan(state)

    def _format_subgoals_for_prompt(self, state: Dict[str, Any]) -> str:
        """Render subgoal context for prompts."""
        return agent_prompt_sections.format_subgoals(state)

    def _format_critic_for_prompt(self, state: Dict[str, Any]) -> str:
        """Render the latest critic guidance for prompts."""
        return agent_prompt_sections.format_critic(state)

    def _format_tool_stats_for_prompt(self, state: Dict[str, Any]) -> str:
        """Render per-tool outcomes as prompt hints."""
        return agent_prompt_sections.format_tool_stats(state)

    def _normalize_role_token(self, value: Any) -> str:
        token = str(value or "").strip().lower()
        token = token.replace("-", "_").replace(" ", "_")
        token = re.sub(r"[^a-z0-9_]+", "_", token)
        token = re.sub(r"_+", "_", token).strip("_")
        if not token:
            return ""

        alias_map = {
            "research": "researcher",
            "researcher": "researcher",
            "researcher_documents": "researcher",
            "researcher_docs": "researcher",
            "researcher_arxiv": "researcher",
            "knowledge_researcher": "researcher",
            "document_researcher": "researcher",
            "docs_researcher": "researcher",
            "literature_researcher": "researcher",
            "paper_researcher": "researcher",
            "arxiv_researcher": "researcher",
            "critic": "critic",
            "reviewer": "critic",
            "analyst": "critic",
            "synth": "synthesizer",
            "synthesizer": "synthesizer",
            "writer": "synthesizer",
            "aggregator": "synthesizer",
            "verify": "verifier",
            "verifier": "verifier",
            "validator": "verifier",
            "qa": "verifier",
            "monitor": "verifier",
            "reproducer": "verifier",
            "repro": "verifier",
            "root_cause": "critic",
            "rootcause": "critic",
            "patcher": "coder",
            "repairer": "coder",
            "implementer": "coder",
            "primary_implementer": "coder",
        }
        if token in alias_map:
            return alias_map[token]

        parts = [p for p in token.split("_") if p]
        if "analyst" in parts or "critic" in parts or "reviewer" in parts:
            return "critic"
        if (
            any(p.startswith("synth") for p in parts)
            or "aggregator" in parts
            or "writer" in parts
        ):
            return "synthesizer"
        if (
            "monitor" in parts
            or "qa" in parts
            or any(p.startswith("verif") for p in parts)
            or any(p.startswith("validat") for p in parts)
            or any(p.startswith("repro") for p in parts)
        ):
            return "verifier"
        if "root" in parts or "cause" in parts:
            return "critic"
        if (
            "researcher" in parts
            or "research" in parts
            or "arxiv" in parts
            or "literature" in parts
        ):
            return "researcher"
        if "patch" in parts or "repair" in parts or "implementer" in parts:
            return "coder"
        return token

    def _normalize_memory_types(self, value: Any) -> List[str]:
        """Normalize memory type settings from config."""
        raw: List[str] = []
        if isinstance(value, list):
            raw = [str(v).strip().lower() for v in value if str(v).strip()]
        elif isinstance(value, str):
            raw = [str(v).strip().lower() for v in value.split(",") if str(v).strip()]

        allowed = {
            "finding",
            "insight",
            "pattern",
            "lesson",
            "fact",
            "preference",
            "context",
            "summary",
        }
        out: List[str] = []
        for mem_type in raw:
            if mem_type in allowed and mem_type not in out:
                out.append(mem_type)
        return out[:12]

    def _resolve_memory_runtime_config(
        self, job: AgentJob, state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Resolve per-job memory injection settings, including role-specific overrides."""
        cfg = job.config if isinstance(job.config, dict) else {}
        memory_cfg = cfg.get("memory") if isinstance(cfg.get("memory"), dict) else {}

        profile = (
            state.get("skill_profile")
            if isinstance(state.get("skill_profile"), dict)
            else self._resolve_agent_skill_profile(job, state=state)
        )
        role = self._normalize_role_token((profile or {}).get("role"))

        enabled = bool(job.enable_memory)
        if isinstance(memory_cfg, dict) and "enabled" in memory_cfg:
            enabled = self._coerce_bool(memory_cfg.get("enabled"), default=enabled)

        limit: Optional[int] = None
        try:
            if memory_cfg.get("max_memories") is not None:
                limit = int(memory_cfg.get("max_memories"))
        except Exception:
            limit = None
        if limit is not None:
            limit = max(1, min(limit, 50))

        memory_types = self._normalize_memory_types(memory_cfg.get("memory_types"))
        include_chat: Optional[bool] = None
        if isinstance(memory_cfg, dict) and "include_chat_memory" in memory_cfg:
            include_chat = self._coerce_bool(
                memory_cfg.get("include_chat_memory"), default=True
            )

        role_profiles = (
            memory_cfg.get("role_profiles")
            if isinstance(memory_cfg.get("role_profiles"), dict)
            else {}
        )
        role_cfg = (
            role_profiles.get(role) if isinstance(role_profiles.get(role), dict) else {}
        )
        if not role_cfg and role_profiles:
            for key, value in role_profiles.items():
                if not isinstance(value, dict):
                    continue
                if self._normalize_role_token(key) == role:
                    role_cfg = value
                    break
        if role_cfg:
            if "enabled" in role_cfg:
                enabled = self._coerce_bool(role_cfg.get("enabled"), default=enabled)
            try:
                if role_cfg.get("max_memories") is not None:
                    role_limit = int(role_cfg.get("max_memories"))
                    limit = max(1, min(role_limit, 50))
            except Exception:
                pass
            role_types = self._normalize_memory_types(role_cfg.get("memory_types"))
            if role_types:
                memory_types = role_types
            if "include_chat_memory" in role_cfg:
                include_chat = self._coerce_bool(
                    role_cfg.get("include_chat_memory"), default=True
                )

        return {
            "enabled": bool(enabled),
            "limit": limit,
            "memory_types": memory_types,
            "include_chat_memory": include_chat,
            "role": role,
            "profile": str(memory_cfg.get("profile") or "").strip().lower(),
        }

    def _resolve_memory_extraction_policy(self, job: AgentJob) -> Dict[str, Any]:
        """Resolve which terminal statuses should trigger memory extraction."""
        cfg = job.config if isinstance(job.config, dict) else {}
        memory_cfg = cfg.get("memory") if isinstance(cfg.get("memory"), dict) else {}

        raw_statuses = memory_cfg.get("extract_on_statuses")
        statuses_raw: List[str] = []
        if isinstance(raw_statuses, list):
            statuses_raw = [
                str(v).strip().lower() for v in raw_statuses if str(v).strip()
            ]
        elif isinstance(raw_statuses, str):
            statuses_raw = [
                str(v).strip().lower()
                for v in raw_statuses.split(",")
                if str(v).strip()
            ]

        allowed_statuses = {"completed", "failed", "cancelled"}
        statuses: List[str] = []
        for status in statuses_raw:
            if status in allowed_statuses and status not in statuses:
                statuses.append(status)
        if not statuses:
            statuses = ["completed"]
            if self._coerce_bool(memory_cfg.get("extract_on_failure"), default=True):
                statuses.append("failed")

        failed_types = self._normalize_memory_types(
            memory_cfg.get("failed_extraction_types")
        )
        if not failed_types:
            failed_types = ["pattern", "lesson", "insight"]
        completed_types = self._normalize_memory_types(
            memory_cfg.get("completed_extraction_types")
        )

        return {
            "extract_on_statuses": statuses,
            "failed_extraction_types": failed_types[:12],
            "completed_extraction_types": completed_types[:12],
        }

    def _resolve_agent_skill_profile(
        self,
        job: AgentJob,
        *,
        state: Optional[Dict[str, Any]] = None,
        override_role: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Compatibility wrapper around skill profile service."""
        return self.skill_profile_service.resolve_agent_skill_profile(
            self,
            job,
            state=state,
            override_role=override_role,
        )

    def _format_skill_profile_for_prompt(self, state: Dict[str, Any]) -> str:
        """Render active role profile for the planner prompt."""
        return agent_prompt_sections.format_skill_profile(state)

    def _format_feedback_learning_for_prompt(self, state: Dict[str, Any]) -> str:
        """Render compact human-feedback guidance for prompt conditioning."""
        return agent_prompt_sections.format_feedback_learning(state)

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
        return agent_tool_scoring.feedback_tool_bias(
            tool_name, state, weight=weight, max_abs=max_abs, enabled=enabled
        )

    def _update_skill_profile_metrics(
        self,
        state: Dict[str, Any],
        action: Optional[Dict[str, Any]],
        action_result: Optional[Dict[str, Any]],
    ) -> None:
        """Track role-specific execution metrics for observability."""
        if not isinstance(state, dict) or not isinstance(action, dict):
            return
        profile = (
            state.get("skill_profile")
            if isinstance(state.get("skill_profile"), dict)
            else {}
        )
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
                counters["evidence_actions"] = (
                    int(counters.get("evidence_actions", 0) or 0) + 1
                )
            counters["evidence_findings"] = (
                int(counters.get("evidence_findings", 0) or 0) + findings_count
            )
        elif role == "critic":
            if tool in {
                "compare_documents",
                "compare_methodologies",
                "identify_research_gaps",
                "build_research_graph",
            }:
                counters["challenge_actions"] = (
                    int(counters.get("challenge_actions", 0) or 0) + 1
                )
            risk_count = 0
            if isinstance(findings, list):
                for item in findings:
                    if not isinstance(item, dict):
                        continue
                    cat = str(item.get("category") or "").strip().lower()
                    if cat in {"contradiction", "gap", "risk"}:
                        risk_count += 1
            counters["risk_findings"] = (
                int(counters.get("risk_findings", 0) or 0) + risk_count
            )
        elif role == "synthesizer":
            if family == "synthesis":
                counters["synthesis_actions"] = (
                    int(counters.get("synthesis_actions", 0) or 0) + 1
                )
            counters["artifacts_created"] = (
                int(counters.get("artifacts_created", 0) or 0) + artifacts_count
            )
        elif role == "verifier":
            if family in {"analysis", "retrieval"}:
                counters["verification_actions"] = (
                    int(counters.get("verification_actions", 0) or 0) + 1
                )
            if not success:
                counters["failed_checks"] = (
                    int(counters.get("failed_checks", 0) or 0) + 1
                )

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
        """Build the full thinking prompt (stable prefix + volatile context).

        Legacy combined form. Cache-aware callers use
        `_build_thinking_prompt_stable` (system prompt — byte-stable across
        iterations, so provider prompt caches hit) plus
        `_build_thinking_prompt_volatile` (per-iteration context, placed in
        the user message after the cached prefix).
        """
        stable = self._build_thinking_prompt_stable(
            job, agent_def, state, profile=profile
        )
        volatile = self._build_thinking_prompt_volatile(job, state)
        if volatile:
            return f"{stable}\n\n{volatile}"
        return stable

    def _build_thinking_prompt_stable(
        self,
        job: AgentJob,
        agent_def: Optional[AgentDefinition],
        state: Dict[str, Any],
        profile: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Stable per-job system prompt; must stay byte-identical across iterations."""
        base_prompt = f"""You are an autonomous agent executing a background job.

Job Type: {job.job_type}
Job Name: {job.name}

GOAL:
{job.goal}

"""
        inherited_data = (
            (job.config or {}).get("inherited_data")
            if isinstance(job.config, dict)
            else None
        )
        if isinstance(inherited_data, dict) and inherited_data:
            parent_results = (
                inherited_data.get("parent_results")
                if isinstance(inherited_data.get("parent_results"), dict)
                else None
            )
            parent_findings = (
                inherited_data.get("parent_findings")
                if isinstance(inherited_data.get("parent_findings"), list)
                else []
            )

            base_prompt += "\nINHERITED DATA (from parent job):\n"
            if parent_results:
                summary = str(parent_results.get("summary") or "").strip()
                research_bundle = (
                    parent_results.get("research_bundle")
                    if isinstance(parent_results.get("research_bundle"), dict)
                    else None
                )
                if summary:
                    base_prompt += f"- Parent summary: {summary[:600]}\n"
                if research_bundle:
                    top_docs = (
                        research_bundle.get("top_documents")
                        if isinstance(research_bundle.get("top_documents"), list)
                        else []
                    )
                    top_papers = (
                        research_bundle.get("top_papers")
                        if isinstance(research_bundle.get("top_papers"), list)
                        else []
                    )
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
        default_source_scope = self._resolve_default_source_scope(job)
        if default_source_scope:
            base_prompt += "\nPROJECT SCOPE:\n"
            base_prompt += f"- Default source_id: {default_source_scope}\n"
            base_prompt += (
                "- Keep retrieval scoped to this source unless broadening scope is explicitly needed.\n"
                "- If you broaden scope, justify why project-only evidence is insufficient.\n"
            )
        project_profile_context = format_project_profile_for_prompt(state)
        if project_profile_context:
            base_prompt += f"\n{project_profile_context}\n"
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

        # Add handoff contract if present (from create_handoff in parent)
        handoff_contract = (
            (job.config or {}).get("handoff_contract")
            if isinstance(job.config, dict)
            else None
        )
        if isinstance(handoff_contract, dict):
            base_prompt += "\nHANDOFF CONTRACT (from parent agent):\n"
            ctx = str(handoff_contract.get("context", ""))[:1000]
            if ctx:
                base_prompt += f"- Context: {ctx}\n"
            outputs = handoff_contract.get("expected_outputs", [])
            if isinstance(outputs, list) and outputs:
                base_prompt += (
                    f"- Expected outputs: {', '.join(str(o) for o in outputs[:10])}\n"
                )
            base_prompt += (
                "- You MUST produce results that satisfy the expected outputs.\n"
            )

        active_profile = (
            profile
            if isinstance(profile, dict)
            else self._resolve_agent_skill_profile(job, state=state)
        )
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

        try:
            from app.services.agent_coding_harness_service import (
                agent_coding_harness_service,
            )

            coding_harness_context = agent_coding_harness_service.format_prompt_context(
                job, state
            )
        except Exception:
            coding_harness_context = ""
        if coding_harness_context:
            base_prompt += f"""

{coding_harness_context}
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

RESPONSE FORMAT:
{AgentDecisionParser.get_schema_for_prompt()}
"""
        return base_prompt

    def _build_thinking_prompt_volatile(
        self,
        job: AgentJob,
        state: Dict[str, Any],
    ) -> str:
        """Per-iteration execution context; changes every iteration.

        Kept out of the system prompt so the stable prefix (and the provider
        prompt cache keyed on it) survives across iterations.
        """
        parts: List[str] = []

        compressed_history = state.get("compressed_history", "")
        if compressed_history:
            parts.append(
                f"COMPRESSED HISTORY (summary of earlier iterations):\n{compressed_history}"
            )

        focus_directive = state.get("focus_directive", "")
        if focus_directive:
            parts.append(f"FOCUS DIRECTIVE (set by agent):\n{focus_directive}")

        for formatter in (
            self._format_causal_experiment_plan_for_prompt,
            self._format_execution_plan_for_prompt,
            self._format_execution_graph_for_prompt,
            self._format_subgoals_for_prompt,
            self._format_critic_for_prompt,
            self._format_tool_stats_for_prompt,
        ):
            try:
                text = formatter(state)
            except Exception:
                text = ""
            if text:
                parts.append(str(text))

        if not parts:
            return ""
        return "CURRENT EXECUTION CONTEXT:\n\n" + "\n\n".join(parts)

    def _get_tools_for_job_type(
        self,
        job_type: str,
        config: Optional[Dict[str, Any]],
        profile: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """Get available tools based on job type."""
        # Base tools available to all autonomous jobs
        base_tools = [
            "search_documents",
            "get_document_details",
            "read_document_content",
            "save_research_finding",
            "get_research_findings",
            "write_progress_report",
            "suggest_next_action",
            "search_with_filters",
            "project_bootstrap",
            # Structured reasoning (available to all job types)
            "reflect",
            "hypothesize",
            "weigh_evidence",
            "critique_plan",
            # Multi-agent coordination
            "delegate_subtask",
            "wait_for_subtask",
            "share_findings",
            "request_review",
            # Code execution
            "execute_python",
            # Memory (available to all job types)
            "create_memory",
            "search_memories",
            "recall_memories",
            "get_memory_stats",
            # Workflow orchestration (available to all job types)
            "list_available_workflows",
            "execute_workflow",
            "get_workflow_status",
            # Agent-to-agent communication
            "send_message_to_agent",
            "read_agent_messages",
            # Research
            "search_web",
            "fetch_url_content",
            # Notification/alerting
            "send_notification",
            "send_email_alert",
            # Data visualization
            "create_chart",
            "render_diagram",
            # Knowledge graph (read)
            "query_kg_entities",
            "get_entity_context",
            "query_kg_graph",
            # Scheduling
            "schedule_job",
            "cancel_scheduled_job",
            # Document authoring (read)
            "list_documents_by_tag",
            # Self-reflection
            "get_job_history",
            "get_job_metrics",
            # Tool analytics
            "get_tool_usage_stats",
            "get_tool_failure_analysis",
            # Batch processing
            "batch_search",
            "batch_summarize",
            # Conditional execution
            "evaluate_condition",
            "count_findings",
            "check_goal_status",
            # Context window management
            "compress_history",
            "summarize_findings",
            # Agent collaboration
            "create_handoff",
            "get_sibling_status",
            "broadcast_to_siblings",
            # Prompt template management
            "switch_strategy",
            "set_focus_directive",
            "get_available_strategies",
            # Output formatting
            "format_as_table",
            "format_as_report",
            "set_output_schema",
            # Multi-modal ingestion
            "transcribe_document",
            "analyze_image",
            "get_media_info",
            # Workspace snapshots
            "capture_snapshot",
            "compare_snapshots",
            "detect_drift",
        ]

        # Type-specific tools
        type_tools = {
            "research": [
                "search_arxiv",
                "summarize_document",
                "find_similar_documents",
                "get_knowledge_base_stats",
                "add_to_reading_list",
                "get_reading_lists",
                "extract_paper_insights",
                "find_related_papers",
                "build_research_graph",
                "compare_methodologies",
                "identify_research_gaps",
                "create_synthesis_document",
                "generate_research_presentation",
                "ingest_paper_by_id",
                "batch_ingest_papers",
                "analyze_document_cluster",
                "create_knowledge_base_entry",
                "create_document_from_text",
                "summarize_url",
                "create_kg_entity",
                "create_kg_relationship",
                "merge_documents",
            ],
            "monitor": [
                "search_arxiv",
                "search_documents",
                "get_knowledge_base_stats",
                "monitor_arxiv_topic",
                "ingest_paper_by_id",
                "add_to_reading_list",
                "get_reading_lists",
            ],
            "analysis": [
                "search_documents",
                "get_document_details",
                "summarize_document",
                "find_similar_documents",
                "compare_documents",
                "extract_paper_insights",
                "compare_methodologies",
                "analyze_document_cluster",
                "build_research_graph",
                "identify_research_gaps",
                "create_synthesis_document",
                "create_document_from_text",
                # Coding tools available in analysis
                "clone_and_index_repo",
                "browse_repo_files",
                "read_file",
                "write_file",
                "apply_patch",
                "run_command",
                "search_code",
                "get_workspace_status",
                "create_workspace_checkpoint",
                "list_workspace_checkpoints",
                "restore_workspace_checkpoint",
                "hydrate_candidate_snapshot",
                "persist_durable_workspace_checkpoint",
                "list_durable_workspace_checkpoints",
                "restore_durable_workspace_checkpoint",
                "retrieve_repo_symbols",
                "get_symbol_context",
                "find_tests_for_symbol",
                "get_workspace_artifact_url",
                "summarize_url",
                "create_kg_entity",
                "create_kg_relationship",
                "merge_documents",
            ],
            "synthesis": [
                "search_documents",
                "get_document_details",
                "summarize_document",
                "generate_diagram",
                "create_synthesis_document",
                "generate_research_presentation",
                "create_knowledge_base_entry",
                "link_entities",
                "create_document_from_text",
                # Document authoring tools available in synthesis
                "plan_document",
                "write_section",
                "revise_section",
                "assemble_document",
                "export_document",
                "insert_figure",
                "merge_documents",
            ],
            "coding": [
                "clone_and_index_repo",
                "browse_repo_files",
                "read_file",
                "write_file",
                "apply_patch",
                "run_command",
                "search_code",
                "get_workspace_status",
                "create_workspace_checkpoint",
                "list_workspace_checkpoints",
                "restore_workspace_checkpoint",
                "hydrate_candidate_snapshot",
                "persist_durable_workspace_checkpoint",
                "list_durable_workspace_checkpoints",
                "restore_durable_workspace_checkpoint",
                "search_documents",
                "get_document_details",
                "read_document_content",
                # Symbol-aware code retrieval
                "retrieve_repo_symbols",
                "get_symbol_context",
                "find_tests_for_symbol",
                # Workspace artifact access
                "get_workspace_artifact_url",
            ],
            "document_authoring": [
                "plan_document",
                "write_section",
                "revise_section",
                "assemble_document",
                "export_document",
                "insert_figure",
                "search_documents",
                "get_document_details",
                "read_document_content",
                "summarize_document",
                "create_document_from_text",
            ],
            "knowledge_expansion": [
                "search_arxiv",
                "search_documents",
                "find_similar_documents",
                "get_knowledge_base_stats",
                "ingest_paper_by_id",
                "batch_ingest_papers",
                "find_related_papers",
                "build_research_graph",
                "link_entities",
                "create_knowledge_base_entry",
            ],
            "custom": [
                # Custom jobs get most tools
                "search_arxiv",
                "summarize_document",
                "find_similar_documents",
                "add_to_reading_list",
                "extract_paper_insights",
                "create_synthesis_document",
                "create_document_from_text",
            ],
            "data_analysis": [
                # Data analysis, ETL, and visualization tools
                "load_csv_data",
                "load_json_data",
                "create_dataset",
                "list_datasets",
                "describe_dataset",
                "query_data",
                "filter_data",
                "aggregate_data",
                "join_datasets",
                "transform_data",
                "detect_anomalies",
                "calculate_correlations",
                "create_chart",
                "create_correlation_heatmap",
                "create_flowchart",
                "create_sequence_diagram",
                "create_er_diagram",
                "create_architecture_diagram",
                "create_drawio_diagram",
                "create_gantt_chart",
                "export_dataset_csv",
                "export_dataset_json",
                "search_documents",
                "get_document_details",
                "read_document_content",
                # Code execution tools for data jobs
                "execute_data_pipeline",
                "write_and_run_script",
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
            "project_bootstrap",
            # Structured reasoning tools
            "reflect",
            "hypothesize",
            "weigh_evidence",
            "critique_plan",
            # Multi-agent coordination tools
            "delegate_subtask",
            "wait_for_subtask",
            "share_findings",
            "request_review",
            # Code execution tools
            "execute_python",
            "execute_data_pipeline",
            "write_and_run_script",
            # Coding workspace tools
            "clone_and_index_repo",
            "browse_repo_files",
            "read_file",
            "write_file",
            "apply_patch",
            "run_command",
            "search_code",
            "get_workspace_status",
            "create_workspace_checkpoint",
            "list_workspace_checkpoints",
            "restore_workspace_checkpoint",
            "hydrate_candidate_snapshot",
            "persist_durable_workspace_checkpoint",
            "list_durable_workspace_checkpoints",
            "restore_durable_workspace_checkpoint",
            # Document authoring tools
            "plan_document",
            "write_section",
            "revise_section",
            "assemble_document",
            "export_document",
            "insert_figure",
            # Workspace artifact retrieval
            "get_workspace_artifact_url",
            # Memory tools
            "create_memory",
            "search_memories",
            "recall_memories",
            "get_memory_stats",
            # Symbol-aware code retrieval
            "retrieve_repo_symbols",
            "get_symbol_context",
            "find_tests_for_symbol",
            # Workflow orchestration
            "list_available_workflows",
            "execute_workflow",
            "get_workflow_status",
            # Agent-to-agent communication
            "send_message_to_agent",
            "read_agent_messages",
            # Research
            "search_web",
            "summarize_url",
            "fetch_url_content",
            # Notification/alerting
            "send_notification",
            "send_email_alert",
            # Data visualization
            "create_chart",
            "render_diagram",
            # Knowledge graph
            "query_kg_entities",
            "get_entity_context",
            "create_kg_entity",
            "create_kg_relationship",
            "query_kg_graph",
            # Scheduling
            "schedule_job",
            "cancel_scheduled_job",
            # Document authoring
            "list_documents_by_tag",
            "merge_documents",
            # Self-reflection
            "get_job_history",
            "get_job_metrics",
            # Tool analytics
            "get_tool_usage_stats",
            "get_tool_failure_analysis",
            # Batch processing
            "batch_search",
            "batch_summarize",
            # Conditional execution
            "evaluate_condition",
            "count_findings",
            "check_goal_status",
            # Context window management
            "compress_history",
            "summarize_findings",
            # Agent collaboration
            "create_handoff",
            "get_sibling_status",
            "broadcast_to_siblings",
            # Prompt template management
            "switch_strategy",
            "set_focus_directive",
            "get_available_strategies",
            # Output formatting
            "format_as_table",
            "format_as_report",
            "set_output_schema",
            # Multi-modal ingestion
            "transcribe_document",
            "analyze_image",
            "get_media_info",
            # Workspace snapshots
            "capture_snapshot",
            "compare_snapshots",
            "detect_drift",
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
        preferred = [
            t for t in _as_list(role_profile.get("preferred_tools")) if t in proposed
        ]
        discouraged = [
            t for t in _as_list(role_profile.get("discouraged_tools")) if t in proposed
        ]
        if blocked:
            proposed = [t for t in proposed if t not in blocked]

        preferred_seen = set()
        preferred_ordered: List[str] = []
        for t in preferred:
            if t not in preferred_seen and t in proposed:
                preferred_seen.add(t)
                preferred_ordered.append(t)

        discouraged_set = set(discouraged)
        middle = [
            t for t in proposed if t not in preferred_seen and t not in discouraged_set
        ]
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
        preferred = set(
            [
                str(x).strip()
                for x in (role_profile.get("preferred_tools") or [])
                if str(x).strip()
            ]
        )
        discouraged = set(
            [
                str(x).strip()
                for x in (role_profile.get("discouraged_tools") or [])
                if str(x).strip()
            ]
        )
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
            "max_iterations_without_progress": _as_int(
                "stall_max_iterations_without_progress", 4, 1, 50
            ),
            "max_repeated_actions": _as_int("stall_max_repeated_actions", 3, 2, 50),
            "hard_stop_iterations": _as_int("stall_hard_stop_iterations", 8, 2, 200),
            "max_recovery_actions": _as_int("stall_max_recovery_actions", 3, 0, 50),
            "graph_recovery_enabled": bool(
                cfg.get("stall_graph_recovery_enabled", True)
            ),
            "graph_recovery_verification_debt": _as_int(
                "stall_graph_recovery_verification_debt", 2, 1, 100
            ),
            "graph_recovery_severity": _as_int(
                "stall_graph_recovery_severity", 20, 1, 100
            ),
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
            raw.get(
                "required_finding_types",
                cfg.get("goal_contract_required_finding_types", []),
            )
        )
        required_artifact_types = _as_str_list(
            raw.get(
                "required_artifact_types",
                cfg.get("goal_contract_required_artifact_types", []),
            )
        )
        required_result_keys = _as_str_list(
            raw.get(
                "required_result_keys",
                cfg.get("goal_contract_required_result_keys", []),
            )
        )

        return {
            "enabled": bool(enabled),
            "min_progress": _as_int(
                raw.get("min_progress", cfg.get("goal_contract_min_progress", 100)),
                100,
                0,
                100,
            ),
            "min_findings": _as_int(
                raw.get("min_findings", cfg.get("goal_contract_min_findings", 0)),
                0,
                0,
                100_000,
            ),
            "min_artifacts": _as_int(
                raw.get("min_artifacts", cfg.get("goal_contract_min_artifacts", 0)),
                0,
                0,
                100_000,
            ),
            "required_finding_types": required_finding_types[:24],
            "required_artifact_types": required_artifact_types[:24],
            "required_result_keys": required_result_keys[:24],
            "auto_complete_when_satisfied": self._coerce_bool(
                raw.get(
                    "auto_complete_when_satisfied",
                    cfg.get("goal_contract_auto_complete_when_satisfied", True),
                ),
                default=True,
            ),
            "strict_completion": self._coerce_bool(
                raw.get(
                    "strict_completion",
                    cfg.get("goal_contract_strict_completion", False),
                ),
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
        """Compatibility wrapper around goal-contract service."""
        return self.goal_contract_service.evaluate_goal_contract(
            self,
            job,
            state,
            include_result_keys=include_result_keys,
        )

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
            raw.get(
                "enabled", cfg.get("approval_checkpoints_enabled", enabled_default)
            ),
            default=enabled_default,
        )
        tools = _as_str_list(raw.get("tools", cfg.get("approval_checkpoint_tools", [])))
        iterations = _as_int_list(
            raw.get("iterations", cfg.get("approval_checkpoint_iterations", []))
        )
        try:
            progress_at_or_above = int(
                raw.get(
                    "progress_at_or_above",
                    cfg.get("approval_checkpoint_progress_at_or_above", -1),
                )
            )
        except Exception:
            progress_at_or_above = -1
        progress_at_or_above = max(-1, min(progress_at_or_above, 100))

        return {
            "enabled": bool(enabled),
            "tools": tools[:40],
            "iterations": iterations[:200],
            "progress_at_or_above": progress_at_or_above,
            "once_per_checkpoint": self._coerce_bool(
                raw.get(
                    "once_per_checkpoint",
                    cfg.get("approval_checkpoint_once_per_checkpoint", True),
                ),
                default=True,
            ),
            "message_prefix": str(
                raw.get(
                    "message_prefix",
                    cfg.get(
                        "approval_checkpoint_message_prefix",
                        "Approval required before autonomous action",
                    ),
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
        watch_tools = set(
            str(x).strip() for x in (cfg.get("tools") or []) if str(x).strip()
        )
        if watch_tools and tool in watch_tools:
            reasons.append(f"tool:{tool}")
            reason_keys.append(f"tool:{tool}")

        iteration = int(getattr(job, "iteration", 0) or 0)
        watch_iterations = set(
            int(x) for x in (cfg.get("iterations") or []) if isinstance(x, int)
        )
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
                "params": action.get("params")
                if isinstance(action.get("params"), dict)
                else {},
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

    def _build_executive_digest(
        self, job: AgentJob, state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compatibility wrapper around goal-contract service digest builder."""
        return self.goal_contract_service.build_executive_digest(self, job, state)

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
        stable_params = {
            k: v for k, v in params.items() if not str(k).startswith("_fallback_")
        }
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
            state["stalled_iterations"] = (
                int(state.get("stalled_iterations", 0) or 0) + 1
            )
        else:
            state["stalled_iterations"] = 0

        sig = self._action_signature(action)
        if delta > min_delta:
            # Forward progress clears repetition pressure.
            state["repeated_action_iterations"] = 1 if sig else 0
            state["last_action_signature"] = sig
        elif sig:
            if sig == state.get("last_action_signature"):
                state["repeated_action_iterations"] = (
                    int(state.get("repeated_action_iterations", 0) or 0) + 1
                )
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
        runtime_graph = self._get_execution_graph_runtime_snapshot(state)
        graph_health = (
            runtime_graph.get("graph_health")
            if isinstance(runtime_graph.get("graph_health"), dict)
            else {}
        )
        verification_attempts = int(runtime_graph.get("verification_attempts", 0) or 0)
        verification_successes = int(
            runtime_graph.get("verification_successes", 0) or 0
        )
        verification_debt = max(0, verification_attempts - verification_successes)
        graph_severity = int(graph_health.get("severity_score", 0) or 0)
        graph_reasons = [
            str(x).strip()
            for x in (graph_health.get("reasons") or [])
            if str(x).strip()
        ]
        graph_recovery = bool(cfg.get("graph_recovery_enabled", True)) and (
            verification_debt
            >= int(cfg.get("graph_recovery_verification_debt", 2) or 2)
            or graph_severity >= int(cfg.get("graph_recovery_severity", 20) or 20)
        )
        should_recover = (
            stalled >= int(cfg["max_iterations_without_progress"])
            or repeated >= int(cfg["max_repeated_actions"])
            or graph_recovery
        )
        should_stop = stalled >= int(cfg["hard_stop_iterations"]) or repeated >= int(
            cfg["hard_stop_iterations"]
        )

        reason_parts = []
        if stalled:
            reason_parts.append(f"stalled_iterations={stalled}")
        if repeated:
            reason_parts.append(f"repeated_action_iterations={repeated}")
        if delta <= min_delta:
            reason_parts.append(f"progress_delta={delta}")
        if graph_recovery:
            reason_parts.append(f"verification_debt={verification_debt}")
            reason_parts.append(f"graph_severity={graph_severity}")
            if graph_reasons:
                reason_parts.append(f"graph_reasons={','.join(graph_reasons[:4])}")

        return {
            "enabled": True,
            "should_recover": should_recover,
            "should_stop": should_stop,
            "reason": ", ".join(reason_parts),
            "stalled_iterations": stalled,
            "repeated_action_iterations": repeated,
            "verification_debt": verification_debt,
            "graph_severity": graph_severity,
            "graph_reasons": graph_reasons[:8],
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
        available = set(
            self._get_tools_for_job_type(job.job_type, job.config, profile=profile)
        )
        findings = (
            state.get("findings", []) if isinstance(state.get("findings"), list) else []
        )
        recent_actions = (
            state.get("actions_taken", [])
            if isinstance(state.get("actions_taken"), list)
            else []
        )
        current_stats = (
            state.get("tool_stats") if isinstance(state.get("tool_stats"), dict) else {}
        )
        prior_stats = (
            state.get("tool_priors")
            if isinstance(state.get("tool_priors"), dict)
            else {}
        )
        combined_stats = self._merge_tool_stats(prior_stats, current_stats)
        forced_cfg = self._get_forced_exploration_config(job)
        forced_tools = (
            set(forced_cfg.get("tools", []))
            if isinstance(forced_cfg.get("tools"), list)
            else set()
        )
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
                if (
                    bool(cooldown_cfg.get("forced_only", True))
                    and forced_tools
                    and tool not in forced_tools
                ):
                    apply_cooldown = False
                if apply_cooldown and self._is_tool_in_cooldown(
                    tool, cooldowns, cur_iter
                ):
                    state["tool_cooldown_blocks"] = (
                        int(state.get("tool_cooldown_blocks", 0) or 0) + 1
                    )
                    return False
            tstats = (
                combined_stats.get(tool) if isinstance(combined_stats, dict) else None
            )
            if isinstance(tstats, dict):
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
        has_papers = any(
            isinstance(f, dict) and f.get("type") == "paper" for f in findings
        )

        runtime_graph = self._get_execution_graph_runtime_snapshot(state)
        graph_health = (
            runtime_graph.get("graph_health")
            if isinstance(runtime_graph.get("graph_health"), dict)
            else {}
        )
        graph_reasons = {
            str(x).strip()
            for x in (graph_health.get("reasons") or [])
            if str(x).strip()
        }
        substantive_graph_reasons = {
            reason for reason in graph_reasons if reason != "empty_graph"
        }
        verification_attempts = int(runtime_graph.get("verification_attempts", 0) or 0)
        verification_successes = int(
            runtime_graph.get("verification_successes", 0) or 0
        )
        has_verification_debt = verification_attempts > verification_successes

        # Periodically force exploration of under-sampled tools to avoid local optima.
        if self._should_force_exploration(job, state):
            state["forced_exploration_attempts"] = (
                int(state.get("forced_exploration_attempts", 0) or 0) + 1
            )
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
                state["forced_exploration_used"] = (
                    int(state.get("forced_exploration_used", 0) or 0) + 1
                )
                state["last_recovery_was_forced_exploration"] = True
                forced_tool = str(forced.get("tool") or "").strip()
                if forced_tool:
                    history = state.get("forced_exploration_history")
                    if not isinstance(history, list):
                        history = []
                    history.append(
                        {
                            "iteration": cur_iter,
                            "tool": forced_tool,
                            "success": None,
                        }
                    )
                    state["forced_exploration_history"] = history[-20:]
                    if bool(cooldown_cfg.get("enabled", True)):
                        until = cur_iter + int(
                            cooldown_cfg.get("cooldown_iterations", 2) or 2
                        )
                        prior_until = int(cooldowns.get(forced_tool, 0) or 0)
                        cooldowns[forced_tool] = max(prior_until, until)
                        state["tool_cooldowns"] = cooldowns
                return forced

        # Prioritize the latest critic recommendation when viable.
        critic_notes = (
            state.get("critic_notes")
            if isinstance(state.get("critic_notes"), list)
            else []
        )
        if critic_notes and isinstance(critic_notes[-1], dict):
            rec_tools = (
                critic_notes[-1].get("recommended_tools")
                if isinstance(critic_notes[-1].get("recommended_tools"), list)
                else []
            )
            rec_action = self._build_action_from_recommended_tools(
                job=job,
                state=state,
                recommended_tools=[str(t).strip() for t in rec_tools if str(t).strip()],
                exclude_tool=exclude,
            )
            if rec_action and _can_use(str(rec_action.get("tool") or "").strip()):
                return rec_action

        if substantive_graph_reasons or has_verification_debt:
            if not isinstance(state.get("project_profile"), dict) or not state.get(
                "project_profile"
            ):
                if _can_use("project_bootstrap"):
                    action = self._build_action_for_tool(
                        tool="project_bootstrap",
                        job=job,
                        doc_ids=doc_ids,
                        purpose="Recover from degraded execution graph by rebuilding project context.",
                    )
                    if action:
                        return action

            if (
                has_verification_debt
                and has_documents
                and _can_use("read_document_content")
            ):
                action = self._build_action_for_tool(
                    tool="read_document_content",
                    job=job,
                    doc_ids=doc_ids,
                    purpose="Recover from failed verification by re-reading source evidence.",
                )
                if action:
                    return action

            if (
                "cycle_detected" in substantive_graph_reasons
                or "long_critical_path" in substantive_graph_reasons
            ) and _can_use("suggest_next_action"):
                action = self._build_action_for_tool(
                    tool="suggest_next_action",
                    job=job,
                    doc_ids=doc_ids,
                    purpose="Recover from unhealthy execution graph by replanning the next step.",
                )
                if action:
                    return action

        # Then prefer current plan step suggested tools.
        plan = (
            state.get("execution_plan")
            if isinstance(state.get("execution_plan"), list)
            else []
        )
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
        """Compatibility wrapper around the extracted action service."""
        return await self.action_service.act(self, job, action, state, db)

    def _score_research_evidence_quality(
        self,
        findings: List[Dict[str, Any]],
        target_docs: int,
        target_papers: int,
    ) -> float:
        """Compatibility wrapper around the extracted progress service."""
        return self.progress_evaluation_service.score_research_evidence_quality(
            findings=findings,
            target_docs=target_docs,
            target_papers=target_papers,
        )

    async def _evaluate_progress(
        self,
        job: AgentJob,
        state: Dict[str, Any],
        user_settings: Optional[UserLLMSettings],
        db: AsyncSession,
    ) -> int:
        """Compatibility wrapper around the extracted progress service."""
        return await self.progress_evaluation_service.evaluate_progress(
            self,
            job,
            state,
            user_settings,
            db,
        )

    async def _finalize_job(
        self,
        job: AgentJob,
        state: Dict[str, Any],
        db: AsyncSession,
    ) -> Dict[str, Any]:
        """Compatibility wrapper around the extracted job finalizer helper."""
        return await finalize_job(self, job, state, db)

    async def _evaluate_swarm_fan_in_gate(
        self,
        parent_job: AgentJob,
        db: AsyncSession,
    ) -> Dict[str, Any]:
        """Compatibility wrapper around the extracted orchestration service."""
        return await self.chain_orchestration_service.evaluate_swarm_fan_in_gate(
            self,
            parent_job,
            db,
        )

    async def _build_swarm_sibling_payload(
        self,
        parent_job: AgentJob,
        db: AsyncSession,
    ) -> Dict[str, Any]:
        """Compatibility wrapper around the extracted orchestration service."""
        return await self.chain_orchestration_service.build_swarm_sibling_payload(
            self,
            parent_job,
            db,
        )

    async def _trigger_chained_jobs(
        self,
        parent_job: AgentJob,
        event: str,
        db: AsyncSession,
        value: int = 0,
    ) -> List[str]:
        """Compatibility wrapper around the extracted orchestration service."""
        return await self.chain_orchestration_service.trigger_chained_jobs(
            self,
            parent_job,
            event,
            db,
            value,
        )

    async def _create_chained_job(
        self,
        parent_job: AgentJob,
        child_config: Dict[str, Any],
        db: AsyncSession,
    ) -> Optional[AgentJob]:
        """Compatibility wrapper around the extracted orchestration service."""
        return await self.chain_orchestration_service.create_chained_job(
            self,
            parent_job,
            child_config,
            db,
        )

    async def trigger_progress_chain(
        self,
        job: AgentJob,
        progress: int,
        findings_count: int,
        db: AsyncSession,
    ) -> List[str]:
        """Compatibility wrapper around the extracted orchestration service."""
        return await self.chain_orchestration_service.trigger_progress_chain(
            self,
            job,
            progress,
            findings_count,
            db,
        )

    async def _save_checkpoint(
        self,
        job: AgentJob,
        state: Dict[str, Any],
        db: AsyncSession,
    ) -> None:
        """Compatibility wrapper around checkpoint service."""
        try:
            from app.services.agent_coding_durable_checkpoint_service import (
                agent_coding_durable_checkpoint_service,
            )

            await agent_coding_durable_checkpoint_service.persist(
                self,
                job,
                state,
                label=f"Runtime iteration {int(job.iteration or 0)}",
                reason="runtime_checkpoint",
                db=db,
            )
        except Exception as exc:
            logger.warning(
                f"Failed to persist durable coding checkpoint for job {job.id}: {exc}"
            )
        await self.checkpoint_service.save_checkpoint(job=job, state=state, db=db)

    async def _load_latest_checkpoint(
        self,
        job_id: UUID,
        db: AsyncSession,
    ) -> Optional[AgentJobCheckpoint]:
        """Compatibility wrapper around checkpoint service."""
        return await self.checkpoint_service.load_latest_checkpoint(
            job_id=job_id, db=db
        )

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
            .where(
                AgentJob.id == job_id, AgentJob.status == AgentJobStatus.RUNNING.value
            )
            .values(status=AgentJobStatus.PAUSED.value)
        )
        await db.commit()
        return result.rowcount > 0

    async def resume_job(self, job_id: UUID, db: AsyncSession) -> bool:
        """Resume a paused job."""
        result = await db.execute(
            update(AgentJob)
            .where(
                AgentJob.id == job_id, AgentJob.status == AgentJobStatus.PAUSED.value
            )
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
                AgentJob.status.in_(
                    [
                        AgentJobStatus.PENDING.value,
                        AgentJobStatus.RUNNING.value,
                        AgentJobStatus.PAUSED.value,
                    ]
                ),
            )
            .values(status=AgentJobStatus.CANCELLED.value)
        )
        await db.commit()
        return result.rowcount > 0
