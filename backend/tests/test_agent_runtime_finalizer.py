"""Tests for autonomous runtime finalization helpers."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from app.models.agent_job import AgentJobStatus
from app.services import agent_runtime_finalizer
from app.services.agent_runtime_finalizer import finalize_job


class _DummyJob:
    def __init__(self, *, status: str = "running", goal_progress: int = 100) -> None:
        self.id = "job-1"
        self.status = status
        self.progress = 0
        self.iteration = 3
        self.tool_calls_used = 4
        self.llm_calls_used = 2
        self.memory_injection_count = 0
        self.memories_created_count = 0
        self.results = {}
        self.output_artifacts = []
        self.config = {}
        self.job_type = "analysis"
        self.goal = "Summarize dataset"
        self.user_id = "user-1"
        self.completed_at = None
        self.error_count = 0
        self.error = None
        self.enable_memory = False
        self.max_runtime_minutes = 60
        self.goal_progress = goal_progress
        self.log_entries = []

    def is_resource_limited(self):
        return False, ""

    def add_log_entry(self, entry):
        self.log_entries.append(entry)


class _DummyExecutor:
    def __init__(self) -> None:
        self._data_analysis_tools = {}
        self.persist_tool_priors_calls = 0
        self.trigger_calls = []

    def _evaluate_goal_contract(self, job, state):
        return {"enabled": True, "satisfied": True, "missing": [], "contract": {}, "metrics": {}}

    def _resolve_default_source_scope(self, job):
        return "source-1"

    def _resolve_scope_source(self, job):
        return "job"

    def _build_execution_graph_stats(self, nodes, edges):
        return {"nodes": len(nodes), "edges": len(edges)}

    def _build_execution_graph_health(self, stats):
        return {"status": "healthy", "stats": stats}

    def _build_execution_graph_recommendations(self, health):
        return [{"kind": "noop", "status": health.get("status")}]

    def _get_approval_checkpoint_config(self, job):
        return {"enabled": False}

    def _get_execution_graph_config(self, job):
        return {"enabled": False}

    def _get_scope_guard_config(self, job):
        return {"enabled": False}

    def _get_tool_selection_config(self, job):
        return {"mode": "adaptive"}

    def _get_forced_exploration_config(self, job):
        return {"enabled": False}

    def _get_tool_cooldown_config(self, job):
        return {"enabled": False}

    def _resolve_memory_extraction_policy(self, job):
        return {"extract_on_statuses": []}

    def _build_executive_digest(self, job, state):
        return {"summary": "ok"}

    async def _persist_tool_priors(self, job, state, db):
        self.persist_tool_priors_calls += 1

    async def _trigger_chained_jobs(self, job, event, db):
        self.trigger_calls.append(event)


class _DummyDb:
    def __init__(self) -> None:
        self.commit = AsyncMock()


@pytest.mark.asyncio
async def test_finalize_job_completed_path_updates_results(monkeypatch):
    monkeypatch.setattr(
        agent_runtime_finalizer.agent_job_memory_service,
        "extract_memories_from_job",
        AsyncMock(return_value=[]),
    )
    executor = _DummyExecutor()
    job = _DummyJob(status="running")
    state = {
        "goal_progress": 100,
        "findings": [{"id": "f1", "type": "document", "title": "Doc"}],
        "actions_taken": [{"tool": "search_documents"}],
        "artifacts": [{"type": "report", "title": "Artifact"}],
        "execution_plan": [{"title": "step"}],
        "step_events": [],
        "tool_stats": {},
        "tool_priors": {},
        "execution_graph_nodes": [],
        "execution_graph_edges": [],
        "scope_events": [],
        "scope_guard_events": [],
        "skill_profile": {"role": "researcher"},
        "skill_profile_metrics": {},
        "memory_runtime": {},
        "memory_extraction_policy": {"extract_on_statuses": []},
        "memory_extraction": {},
        "injected_memories": [],
    }
    db = _DummyDb()

    result = await finalize_job(executor, job, state, db)

    assert result["status"] == AgentJobStatus.COMPLETED.value
    assert job.status == AgentJobStatus.COMPLETED.value
    assert job.progress == 100
    assert job.results["findings_count"] == 1
    assert job.results["execution_strategy"]["execution_mode"] == "adaptive"
    assert job.results["executive_digest"] == {"summary": "ok"}
    assert executor.persist_tool_priors_calls == 1
    assert executor.trigger_calls == ["complete"]
    assert db.commit.await_count >= 1


@pytest.mark.asyncio
async def test_finalize_job_paused_path_preserves_paused_status(monkeypatch):
    monkeypatch.setattr(
        agent_runtime_finalizer.agent_job_memory_service,
        "extract_memories_from_job",
        AsyncMock(return_value=[]),
    )
    executor = _DummyExecutor()
    job = _DummyJob(status=AgentJobStatus.PAUSED.value, goal_progress=40)
    state = {
        "goal_progress": 40,
        "findings": [],
        "actions_taken": [],
        "artifacts": [],
        "execution_plan": [],
        "step_events": [],
        "tool_stats": {},
        "tool_priors": {},
        "execution_graph_nodes": [],
        "execution_graph_edges": [],
        "scope_events": [],
        "scope_guard_events": [],
        "skill_profile": {},
        "skill_profile_metrics": {},
        "memory_runtime": {},
        "memory_extraction_policy": {"extract_on_statuses": []},
        "memory_extraction": {},
        "injected_memories": [],
    }
    db = _DummyDb()

    result = await finalize_job(executor, job, state, db)

    assert result["status"] == AgentJobStatus.PAUSED.value
    assert job.status == AgentJobStatus.PAUSED.value
    assert executor.trigger_calls == []
