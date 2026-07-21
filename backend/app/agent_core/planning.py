"""Planning primitives extracted from the app service layer."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from loguru import logger
from pydantic import BaseModel, Field, field_validator

from app.services.agent_decision_parser import extract_first_json_object


class PlanStep(BaseModel):
    """A single step in the execution plan."""

    step_id: str = Field(default_factory=lambda: f"s-{uuid.uuid4().hex[:8]}")
    title: str = Field(..., max_length=220)
    objective: str = Field(default="", max_length=500)
    exit_criteria: str = Field(default="", max_length=300)
    suggested_tools: List[str] = Field(default_factory=list)
    status: str = "pending"
    depends_on: List[str] = Field(default_factory=list)
    parallel_group: Optional[str] = None
    progress: int = 0

    @field_validator("status", mode="before")
    @classmethod
    def validate_status(cls, v: Any) -> str:
        v = str(v or "pending").strip().lower()
        if v not in {"pending", "in_progress", "done", "skipped"}:
            return "pending"
        return v

    @field_validator("progress", mode="before")
    @classmethod
    def clamp_progress(cls, v: Any) -> int:
        try:
            return max(0, min(100, int(v)))
        except (TypeError, ValueError):
            return 0


class Subgoal(BaseModel):
    """A decomposed subgoal with explicit success criteria."""

    id: str = Field(default_factory=lambda: f"sg-{uuid.uuid4().hex[:8]}")
    title: str = Field(..., max_length=220)
    success_criteria: str = Field(default="", max_length=400)
    status: str = "pending"
    linked_step_ids: List[str] = Field(default_factory=list)
    progress: int = 0

    @field_validator("progress", mode="before")
    @classmethod
    def clamp_progress(cls, v: Any) -> int:
        try:
            return max(0, min(100, int(v)))
        except (TypeError, ValueError):
            return 0


class ExecutionPlan(BaseModel):
    """Full execution plan with metadata."""

    steps: List[PlanStep]
    subgoals: List[Subgoal] = Field(default_factory=list)
    version: int = 1
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_replanned_at: Optional[str] = None
    replan_count: int = 0


_PLAN_GENERATION_PROMPT = """\
You are a planning assistant for an autonomous AI agent. Given a goal,
generate a concrete execution plan as a JSON object.

Goal: {goal}

Job type: {job_type}
Available tools: {tools}

{context}

Respond with ONLY a JSON object:
{{
    "steps": [
        {{
            "title": "Short step title",
            "objective": "What this step achieves",
            "exit_criteria": "How to know this step is done",
            "suggested_tools": ["tool1", "tool2"],
            "depends_on": []
        }}
    ],
    "subgoals": [
        {{
            "title": "High-level subgoal",
            "success_criteria": "How to know this subgoal is met",
            "linked_step_ids": []
        }}
    ]
}}

Rules:
- Generate at most {max_steps} steps.
- Steps should be concrete and actionable.
- Use tool names from the available tools list only.
- Order steps logically; use depends_on for non-sequential dependencies.
- Subgoals group related steps into higher-level objectives.
"""

_REPLAN_PROMPT = """\
You are replanning for an autonomous AI agent. The current plan needs revision.

Original goal: {goal}
Replan trigger: {trigger_reason}

Completed steps and outcomes:
{completed_summary}

Remaining steps (to revise):
{remaining_summary}

Findings so far ({findings_count} total):
{findings_summary}

Available tools: {tools}

Generate a REVISED plan that:
1. Does NOT include already-completed steps.
2. Adjusts remaining steps based on what we learned.
3. May add or remove steps as needed.

Respond with ONLY a JSON object matching the same schema as before:
{{
    "steps": [ ... ],
    "subgoals": [ ... ]
}}

Generate at most {max_steps} remaining steps.
"""


class AgentExecutionPlanner:
    """Manages plan lifecycle for an agent runtime."""

    def __init__(self, llm_service: Any):
        self.llm_service = llm_service

    async def create_plan(
        self,
        job: Any,
        observation: Dict[str, Any],
        user_settings: Any,
        available_tools: List[str],
        routing: Optional[Dict[str, Any]] = None,
        max_steps: int = 6,
    ) -> ExecutionPlan:
        context_parts: List[str] = []
        if observation:
            obs_summary = json.dumps(observation, default=str)[:1500]
            context_parts.append(f"Current observation:\n{obs_summary}")

        prompt = _PLAN_GENERATION_PROMPT.format(
            goal=str(getattr(job, "goal", "") or ""),
            job_type=str(getattr(job, "job_type", "custom") or "custom"),
            tools=", ".join(available_tools[:60]),
            context="\n".join(context_parts),
            max_steps=max_steps,
        )

        try:
            response = await self.llm_service.generate_response(
                system_prompt="You are a planning assistant. Respond with valid JSON only.",
                user_message=prompt,
                user_settings=user_settings,
                routing=routing,
            )
            plan = self._parse_plan_response(str(response or ""), available_tools, max_steps)
            if plan:
                plan.steps = self.annotate_dependencies(plan.steps)
                return plan
        except Exception as exc:
            logger.error(f"Plan creation LLM call failed: {exc}")

        return self._fallback_plan(job)

    def evaluate_replan_triggers(
        self,
        job: Any,
        state: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        cfg = config or (getattr(job, "config", None) or {})
        if not cfg.get("replan_enabled", True):
            return None
        if state.get("plan_replan_count", 0) >= cfg.get("replan_max_replans", 3):
            return None

        iteration = getattr(job, "iteration", 0) or 0
        if state.get("stalled_iterations", 0) >= cfg.get("replan_stall_threshold", 3):
            return "stall_detected"
        interval = cfg.get("replan_interval_iterations", 8)
        if interval > 0 and iteration > 0 and iteration % interval == 0:
            return "periodic_interval"

        failure_streak = cfg.get("replan_on_tool_failure_streak", 2)
        recent_actions = state.get("actions_taken", [])[-failure_streak:]
        if len(recent_actions) >= failure_streak and all(
            isinstance(a, dict) and not a.get("success", True) for a in recent_actions
        ):
            return "tool_failure_streak"

        plan_progress = state.get("plan_progress", 0)
        goal_progress = state.get("goal_progress", 0)
        divergence_threshold = cfg.get("replan_progress_divergence_threshold", 30)
        if plan_progress > 80 and (plan_progress - goal_progress) > divergence_threshold:
            return "progress_divergence"
        return None

    async def replan(
        self,
        job: Any,
        state: Dict[str, Any],
        observation: Dict[str, Any],
        user_settings: Any,
        available_tools: List[str],
        routing: Optional[Dict[str, Any]] = None,
        trigger_reason: str = "",
        max_steps: int = 6,
    ) -> ExecutionPlan:
        old_plan = state.get("execution_plan") or []
        completed = [s for s in old_plan if isinstance(s, dict) and s.get("status") == "done"]
        remaining = [s for s in old_plan if isinstance(s, dict) and s.get("status") != "done"]
        findings = state.get("findings", [])

        prompt = _REPLAN_PROMPT.format(
            goal=str(getattr(job, "goal", "") or ""),
            trigger_reason=trigger_reason,
            completed_summary=json.dumps(completed[-10:], default=str)[:2000],
            remaining_summary=json.dumps(remaining[:10], default=str)[:1500],
            findings_count=len(findings),
            findings_summary=json.dumps(
                [
                    {"title": f.get("title", ""), "category": f.get("category", "")}
                    for f in findings[-10:]
                    if isinstance(f, dict)
                ],
                default=str,
            )[:1500],
            tools=", ".join(available_tools[:60]),
            max_steps=max_steps,
        )
        replan_count = state.get("plan_replan_count", 0) + 1

        try:
            response = await self.llm_service.generate_response(
                system_prompt="You are a replanning assistant. Respond with valid JSON only.",
                user_message=prompt,
                user_settings=user_settings,
                routing=routing,
            )
            plan = self._parse_plan_response(str(response or ""), available_tools, max_steps)
            if plan:
                plan.version = state.get("execution_plan_version", 1) + 1
                plan.replan_count = replan_count
                plan.last_replanned_at = datetime.now(timezone.utc).isoformat()
                plan.steps = self.annotate_dependencies(plan.steps)
                return plan
        except Exception as exc:
            logger.error(f"Replan LLM call failed: {exc}")

        return ExecutionPlan(
            steps=[PlanStep.model_validate(s) for s in remaining if isinstance(s, dict)],
            version=state.get("execution_plan_version", 1) + 1,
            replan_count=replan_count,
            last_replanned_at=datetime.now(timezone.utc).isoformat(),
        )

    def decompose_into_subgoals(
        self,
        plan: ExecutionPlan,
        goal: str,
        goal_criteria: Optional[Dict[str, Any]] = None,
    ) -> List[Subgoal]:
        if goal_criteria and isinstance(goal_criteria, dict):
            criteria_list = goal_criteria.get("criteria", [])
            if isinstance(criteria_list, list) and criteria_list:
                subgoals: List[Subgoal] = []
                for i, criterion in enumerate(criteria_list):
                    if isinstance(criterion, dict):
                        subgoals.append(
                            Subgoal(
                                title=str(criterion.get("name", f"Criterion {i + 1}")),
                                success_criteria=str(criterion.get("description", "")),
                            )
                        )
                    else:
                        subgoals.append(
                            Subgoal(title=str(criterion)[:220], success_criteria=str(criterion))
                        )
                return subgoals

        subgoals: List[Subgoal] = []
        current_group: List[PlanStep] = []
        current_family = ""
        for step in plan.steps:
            family = self._tool_family(step.suggested_tools)
            if family != current_family and current_group:
                subgoals.append(self._subgoal_from_steps(current_group))
                current_group = []
            current_family = family
            current_group.append(step)

        if current_group:
            subgoals.append(self._subgoal_from_steps(current_group))
        return subgoals

    @staticmethod
    def annotate_dependencies(steps: List[PlanStep]) -> List[PlanStep]:
        if not steps or any(s.depends_on for s in steps):
            return steps
        for i in range(1, len(steps)):
            steps[i].depends_on = [steps[i - 1].step_id]
        return steps

    @staticmethod
    def get_eligible_steps(plan: ExecutionPlan) -> List[PlanStep]:
        done_ids = {s.step_id for s in plan.steps if s.status == "done"}
        return [
            step
            for step in plan.steps
            if step.status == "pending" and all(dep in done_ids for dep in step.depends_on)
        ]

    @staticmethod
    def update_step_progress(
        plan: ExecutionPlan,
        step_id: str,
        progress: int,
        status: Optional[str] = None,
    ) -> None:
        for step in plan.steps:
            if step.step_id == step_id:
                step.progress = max(0, min(100, progress))
                if status:
                    step.status = status
                break

    @staticmethod
    def compute_plan_progress(plan: ExecutionPlan) -> int:
        if not plan.steps:
            return 0
        return min(100, sum(step.progress for step in plan.steps) // len(plan.steps))

    def _parse_plan_response(
        self,
        text: str,
        available_tools: List[str],
        max_steps: int,
    ) -> Optional[ExecutionPlan]:
        payload = extract_first_json_object(text)
        if not isinstance(payload, dict):
            return None

        raw_steps = payload.get("steps")
        if not isinstance(raw_steps, list) or not raw_steps:
            return None

        tools_set = set(available_tools)
        steps: List[PlanStep] = []
        for raw in raw_steps[:max_steps]:
            if not isinstance(raw, dict):
                continue
            try:
                suggested = raw.get("suggested_tools", [])
                if isinstance(suggested, list):
                    raw["suggested_tools"] = [
                        t for t in suggested if isinstance(t, str) and t in tools_set
                    ]
                steps.append(PlanStep.model_validate(raw))
            except Exception:
                continue

        if not steps:
            return None

        subgoals: List[Subgoal] = []
        for raw_subgoal in payload.get("subgoals", []) if isinstance(payload.get("subgoals", []), list) else []:
            if isinstance(raw_subgoal, dict):
                try:
                    subgoals.append(Subgoal.model_validate(raw_subgoal))
                except Exception:
                    continue

        return ExecutionPlan(steps=steps, subgoals=subgoals)

    @staticmethod
    def _fallback_plan(job: Any) -> ExecutionPlan:
        goal = str(getattr(job, "goal", "") or "Achieve the stated objective")
        return ExecutionPlan(
            steps=[
                PlanStep(
                    title="Gather information",
                    objective=f"Search for relevant information: {goal[:200]}",
                    exit_criteria="Found relevant documents or data",
                    suggested_tools=["search_documents"],
                ),
                PlanStep(
                    title="Analyze findings",
                    objective="Review and analyze gathered information",
                    exit_criteria="Key findings identified",
                    suggested_tools=["read_document_content", "summarize_document"],
                ),
                PlanStep(
                    title="Synthesize results",
                    objective="Compile findings into a coherent result",
                    exit_criteria="Final result produced",
                    suggested_tools=["save_research_finding", "create_synthesis_document"],
                ),
            ]
        )

    @staticmethod
    def _tool_family(tools: List[str]) -> str:
        if not tools:
            return "general"
        search_tools = {"search_documents", "search_arxiv", "search_with_filters", "web_scrape"}
        doc_tools = {"read_document_content", "get_document_details", "summarize_document", "find_similar_documents"}
        synthesis_tools = {"create_synthesis_document", "save_research_finding", "compare_methodologies"}
        code_tools = {"execute_python", "execute_data_pipeline", "write_and_run_script"}

        tools_set = set(tools)
        if tools_set & search_tools:
            return "search"
        if tools_set & doc_tools:
            return "document"
        if tools_set & synthesis_tools:
            return "synthesis"
        if tools_set & code_tools:
            return "code"
        return "general"

    @staticmethod
    def _subgoal_from_steps(steps: List[PlanStep]) -> Subgoal:
        return Subgoal(
            title=" + ".join(step.title for step in steps)[:220],
            success_criteria=(
                "; ".join(step.exit_criteria for step in steps if step.exit_criteria)[:400]
                or "All linked steps completed"
            ),
            linked_step_ids=[step.step_id for step in steps],
        )
