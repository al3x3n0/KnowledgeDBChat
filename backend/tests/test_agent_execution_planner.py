"""Tests for agent_execution_planner module."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock

from app.services.agent_execution_planner import (
    AgentExecutionPlanner,
    ExecutionPlan,
    PlanStep,
    Subgoal,
)


# ---------------------------------------------------------------------------
# PlanStep model
# ---------------------------------------------------------------------------

class TestPlanStepModel:
    def test_valid_step(self):
        step = PlanStep(title="Search papers", objective="Find relevant papers")
        assert step.title == "Search papers"
        assert step.status == "pending"
        assert step.progress == 0

    def test_progress_clamping(self):
        step = PlanStep(title="Test", progress=150)
        assert step.progress == 100

    def test_negative_progress(self):
        step = PlanStep(title="Test", progress=-10)
        assert step.progress == 0

    def test_invalid_status_defaults(self):
        step = PlanStep(title="Test", status="invalid")
        assert step.status == "pending"

    def test_step_id_auto_generated(self):
        step = PlanStep(title="Test")
        assert step.step_id.startswith("s-")


# ---------------------------------------------------------------------------
# ExecutionPlan model
# ---------------------------------------------------------------------------

class TestExecutionPlanModel:
    def test_valid_plan(self):
        plan = ExecutionPlan(
            steps=[PlanStep(title="Step 1"), PlanStep(title="Step 2")]
        )
        assert len(plan.steps) == 2
        assert plan.version == 1
        assert plan.replan_count == 0


# ---------------------------------------------------------------------------
# AgentExecutionPlanner
# ---------------------------------------------------------------------------

class TestAgentExecutionPlanner:
    def setup_method(self):
        self.llm = AsyncMock()
        self.planner = AgentExecutionPlanner(self.llm)
        self.tools = ["search_documents", "read_document_content", "summarize_document", "save_research_finding"]

    # ------------------------------------------------------------------
    # create_plan
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_create_plan_from_llm(self):
        self.llm.generate_response.return_value = json.dumps({
            "steps": [
                {"title": "Search", "objective": "Find papers", "suggested_tools": ["search_documents"]},
                {"title": "Read", "objective": "Read papers", "suggested_tools": ["read_document_content"]},
            ]
        })
        job = MagicMock()
        job.goal = "Research quantum computing"
        job.job_type = "research"
        job.config = {}

        plan = await self.planner.create_plan(
            job=job, observation={}, user_settings=None, available_tools=self.tools,
        )
        assert len(plan.steps) >= 2
        assert plan.steps[0].title == "Search"

    @pytest.mark.asyncio
    async def test_create_plan_fallback_on_error(self):
        self.llm.generate_response.side_effect = Exception("LLM down")
        job = MagicMock()
        job.goal = "Research topic"
        job.job_type = "research"
        job.config = {}

        plan = await self.planner.create_plan(
            job=job, observation={}, user_settings=None, available_tools=self.tools,
        )
        # Should return fallback plan
        assert len(plan.steps) == 3
        assert plan.steps[0].title == "Gather information"

    @pytest.mark.asyncio
    async def test_create_plan_filters_unavailable_tools(self):
        self.llm.generate_response.return_value = json.dumps({
            "steps": [
                {"title": "Step", "suggested_tools": ["search_documents", "nonexistent_tool"]},
            ]
        })
        job = MagicMock()
        job.goal = "Test"
        job.job_type = "custom"
        job.config = {}

        plan = await self.planner.create_plan(
            job=job, observation={}, user_settings=None, available_tools=self.tools,
        )
        assert "nonexistent_tool" not in plan.steps[0].suggested_tools

    # ------------------------------------------------------------------
    # evaluate_replan_triggers
    # ------------------------------------------------------------------

    def test_no_trigger_by_default(self):
        job = MagicMock()
        job.iteration = 3
        job.config = {}
        state = {"stalled_iterations": 0, "plan_replan_count": 0, "actions_taken": []}
        assert self.planner.evaluate_replan_triggers(job, state) is None

    def test_stall_trigger(self):
        job = MagicMock()
        job.iteration = 5
        job.config = {}
        state = {"stalled_iterations": 3, "plan_replan_count": 0}
        assert self.planner.evaluate_replan_triggers(job, state) == "stall_detected"

    def test_periodic_trigger(self):
        job = MagicMock()
        job.iteration = 8
        job.config = {}
        state = {"stalled_iterations": 0, "plan_replan_count": 0, "actions_taken": []}
        assert self.planner.evaluate_replan_triggers(job, state) == "periodic_interval"

    def test_tool_failure_streak_trigger(self):
        job = MagicMock()
        job.iteration = 5
        job.config = {}
        state = {
            "stalled_iterations": 0,
            "plan_replan_count": 0,
            "actions_taken": [{"success": False}, {"success": False}],
        }
        assert self.planner.evaluate_replan_triggers(job, state) == "tool_failure_streak"

    def test_progress_divergence_trigger(self):
        job = MagicMock()
        job.iteration = 5
        job.config = {}
        state = {
            "stalled_iterations": 0,
            "plan_replan_count": 0,
            "plan_progress": 90,
            "goal_progress": 30,
            "actions_taken": [],
        }
        assert self.planner.evaluate_replan_triggers(job, state) == "progress_divergence"

    def test_max_replans_respected(self):
        job = MagicMock()
        job.iteration = 8
        job.config = {}
        state = {"stalled_iterations": 5, "plan_replan_count": 3}
        assert self.planner.evaluate_replan_triggers(job, state) is None

    def test_replan_disabled(self):
        job = MagicMock()
        job.iteration = 8
        job.config = {"replan_enabled": False}
        state = {"stalled_iterations": 5, "plan_replan_count": 0}
        assert self.planner.evaluate_replan_triggers(job, state, config=job.config) is None

    # ------------------------------------------------------------------
    # replan
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_replan_preserves_completed(self):
        self.llm.generate_response.return_value = json.dumps({
            "steps": [
                {"title": "New step", "objective": "Do something new"},
            ]
        })
        job = MagicMock()
        job.goal = "Test"
        job.config = {}
        state = {
            "execution_plan": [
                {"title": "Done step", "status": "done"},
                {"title": "Old pending", "status": "pending"},
            ],
            "execution_plan_version": 1,
            "plan_replan_count": 0,
            "findings": [],
        }

        plan = await self.planner.replan(
            job=job, state=state, observation={},
            user_settings=None, available_tools=self.tools,
            trigger_reason="stall_detected",
        )
        assert plan.version == 2
        assert plan.replan_count == 1

    # ------------------------------------------------------------------
    # decompose_into_subgoals
    # ------------------------------------------------------------------

    def test_decompose_from_criteria(self):
        plan = ExecutionPlan(steps=[PlanStep(title="Step 1")])
        subgoals = self.planner.decompose_into_subgoals(
            plan, "Test goal",
            goal_criteria={"criteria": [
                {"name": "Find papers", "description": "At least 5 papers"},
                {"name": "Analyze", "description": "Identify key themes"},
            ]},
        )
        assert len(subgoals) == 2
        assert subgoals[0].title == "Find papers"

    def test_decompose_automatic(self):
        plan = ExecutionPlan(steps=[
            PlanStep(title="Search", suggested_tools=["search_documents"]),
            PlanStep(title="Read", suggested_tools=["read_document_content"]),
            PlanStep(title="Synthesize", suggested_tools=["save_research_finding"]),
        ])
        subgoals = self.planner.decompose_into_subgoals(plan, "Research")
        assert len(subgoals) >= 1

    # ------------------------------------------------------------------
    # Dependency annotation
    # ------------------------------------------------------------------

    def test_annotate_sequential_dependencies(self):
        steps = [PlanStep(title="A"), PlanStep(title="B"), PlanStep(title="C")]
        annotated = AgentExecutionPlanner.annotate_dependencies(steps)
        assert annotated[0].depends_on == []
        assert annotated[1].depends_on == [annotated[0].step_id]
        assert annotated[2].depends_on == [annotated[1].step_id]

    def test_annotate_preserves_explicit_deps(self):
        steps = [
            PlanStep(title="A"),
            PlanStep(title="B", depends_on=["custom-dep"]),
        ]
        annotated = AgentExecutionPlanner.annotate_dependencies(steps)
        assert annotated[1].depends_on == ["custom-dep"]

    # ------------------------------------------------------------------
    # get_eligible_steps
    # ------------------------------------------------------------------

    def test_eligible_steps_respects_deps(self):
        s1 = PlanStep(step_id="s1", title="A", status="done")
        s2 = PlanStep(step_id="s2", title="B", depends_on=["s1"])
        s3 = PlanStep(step_id="s3", title="C", depends_on=["s2"])
        plan = ExecutionPlan(steps=[s1, s2, s3])
        eligible = AgentExecutionPlanner.get_eligible_steps(plan)
        assert len(eligible) == 1
        assert eligible[0].step_id == "s2"

    # ------------------------------------------------------------------
    # Progress
    # ------------------------------------------------------------------

    def test_compute_plan_progress(self):
        plan = ExecutionPlan(steps=[
            PlanStep(title="A", progress=100),
            PlanStep(title="B", progress=50),
            PlanStep(title="C", progress=0),
        ])
        assert AgentExecutionPlanner.compute_plan_progress(plan) == 50

    def test_compute_plan_progress_empty(self):
        plan = ExecutionPlan(steps=[])
        assert AgentExecutionPlanner.compute_plan_progress(plan) == 0

    def test_update_step_progress(self):
        step = PlanStep(step_id="s1", title="A")
        plan = ExecutionPlan(steps=[step])
        AgentExecutionPlanner.update_step_progress(plan, "s1", 75, "in_progress")
        assert plan.steps[0].progress == 75
        assert plan.steps[0].status == "in_progress"
