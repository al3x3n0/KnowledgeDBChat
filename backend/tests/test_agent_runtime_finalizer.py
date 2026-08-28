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
        return {
            "enabled": True,
            "satisfied": True,
            "missing": [],
            "contract": {},
            "metrics": {},
        }

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
        "actions_taken": [
            {
                "action": {
                    "tool": "search_documents",
                    "params": {"query": "private query"},
                },
                "result": {"success": True, "findings": [{"id": "f1"}]},
                "iteration": 1,
                "node": "act",
            }
        ],
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
    assert job.results["actions"] == [
        {
            "tool": "search_documents",
            "success": True,
            "iteration": 1,
            "node": "act",
        }
    ]
    assert "private query" not in repr(job.results["actions"])
    assert job.results["execution_strategy"]["execution_mode"] == "adaptive"
    assert job.results["executive_digest"] == {"summary": "ok"}
    assert job.results["evaluation_outcome"]["schema_version"] == 3
    assert job.results["evaluation_outcome"]["status"] == "completed"
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


class _ExpiringJob(_DummyJob):
    """A job whose attribute reads fail the way an expired ORM object's do.

    A rollback expires every object in the session; reading one of those
    attributes is IO, which raises MissingGreenlet when it happens outside an
    awaitable context. A refresh clears the condition.
    """

    def __init__(self, **kwargs) -> None:
        self._expired = False
        super().__init__(**kwargs)

    @property
    def status(self):
        if self._expired:
            raise RuntimeError(
                "greenlet_spawn has not been called; can't call await_only() here"
            )
        return self._status

    @status.setter
    def status(self, value):
        self._status = value

    def expire(self):
        self._expired = True

    def reload(self):
        self._expired = False


class _RecoveringDb(_DummyDb):
    def __init__(self) -> None:
        super().__init__()
        self.rollback = AsyncMock()
        self.refreshed = []

    async def refresh(self, obj):
        self.refreshed.append(obj)
        if hasattr(obj, "reload"):
            obj.reload()


class _MemoryExtractingExecutor(_DummyExecutor):
    def _resolve_memory_extraction_policy(self, job):
        return {"extract_on_statuses": [AgentJobStatus.COMPLETED.value]}


@pytest.mark.asyncio
async def test_finalize_job_triggers_chain_after_memory_extraction_expires_job(
    monkeypatch,
):
    """A failed memory extraction must not cost the job its chained jobs."""
    executor = _MemoryExtractingExecutor()
    job = _ExpiringJob(status="running")
    job.enable_memory = True

    async def _extract_then_expire(**_kwargs):
        # What the service does on an LLM failure: roll the session back, which
        # leaves every ORM object in it expired.
        job.expire()
        return []

    monkeypatch.setattr(
        agent_runtime_finalizer.agent_job_memory_service,
        "extract_memories_from_job",
        _extract_then_expire,
    )

    state = {
        "goal_progress": 100,
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
        "memory_extraction": {},
        "injected_memories": [],
    }
    db = _RecoveringDb()

    result = await finalize_job(executor, job, state, db)

    assert job in db.refreshed
    assert executor.trigger_calls == ["complete"]
    assert result["status"] == AgentJobStatus.COMPLETED.value


@pytest.mark.asyncio
async def test_a_failing_conclusion_does_not_cost_the_job_its_chain(monkeypatch):
    """The same shape as the memory-extraction bug: an optional step that
    raises inside finalization would skip the chain trigger after it."""
    monkeypatch.setattr(
        agent_runtime_finalizer.agent_job_memory_service,
        "extract_memories_from_job",
        AsyncMock(return_value=[]),
    )

    async def _exploding_conclusion(*args, **kwargs):
        raise RuntimeError("synthesis blew up")

    monkeypatch.setattr(
        agent_runtime_finalizer, "synthesize_conclusion", _exploding_conclusion
    )

    executor = _DummyExecutor()
    job = _DummyJob(status="running")
    state = {
        "goal_progress": 100,
        "findings": [{"type": "codegen_measurement", "title": "kernel @ -O3"}],
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

    result = await finalize_job(executor, job, state, _DummyDb())

    assert result["status"] == AgentJobStatus.COMPLETED.value
    assert executor.trigger_calls == ["complete"], "the chain must still fire"
    assert job.results["conclusion"]["generated_by"] == "error"
    assert "synthesis blew up" in job.results["conclusion"]["gaps"][0]


def _state_with(progress: int, findings=None):
    """The minimum state finalize_job reads, at a chosen progress."""
    return {
        "goal_progress": progress,
        "findings": findings if findings is not None else [],
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
        "skill_profile": {"role": "researcher"},
        "skill_profile_metrics": {},
        "memory_runtime": {},
        "memory_extraction_policy": {"extract_on_statuses": []},
        "memory_extraction": {},
        "injected_memories": [],
    }


class _UnmetContractExecutor(_DummyExecutor):
    """A contract that is enabled and not satisfied."""

    def _evaluate_goal_contract(self, job, state):
        return {
            "enabled": True,
            "satisfied": False,
            "missing": ["finding_type:mechanism_comparison>=2"],
            "contract": {},
            "metrics": {},
        }


@pytest.mark.asyncio
class TestAnUnmetContractIsNotAFinishedGoal:
    """A live run measured nothing, recalled fifty numbers other runs had
    measured, and was filed `completed` at 100% with its contract reporting
    `mechanism_comparison>=2` still missing.

    The contract already gates the two places a run declares victory --
    `goal_achieved` and the autocomplete when satisfied. `goal_progress` was a
    third road to the same verdict: the progress evaluator is an LLM judgement
    free to return 100, and the finalizer read >= 100 as done without ever
    consulting the contract it evaluates a few lines above.
    """

    async def _finalize(self, monkeypatch, executor, job, state):
        monkeypatch.setattr(
            agent_runtime_finalizer.agent_job_memory_service,
            "extract_memories_from_job",
            AsyncMock(return_value=[]),
        )
        return await finalize_job(executor, job, state, _DummyDb())

    async def test_progress_100_does_not_finish_an_unmet_contract(self, monkeypatch):
        job = _DummyJob(status="running")
        await self._finalize(
            monkeypatch, _UnmetContractExecutor(), job, _state_with(100)
        )

        assert job.progress != 100

    async def test_the_missing_requirement_is_named_in_the_log(self, monkeypatch):
        """Written into the results is where it already was, and where nothing
        read it. It has to be in a field a reader sees."""
        job = _DummyJob(status="running")
        await self._finalize(
            monkeypatch, _UnmetContractExecutor(), job, _state_with(100)
        )

        entries = [
            e for e in job.log_entries if e.get("phase") == "completed_contract_unmet"
        ]
        assert entries, "an unmet contract must say so in the log"
        assert "mechanism_comparison" in str(entries[0]["missing"])

    async def test_a_satisfied_contract_still_completes_at_100(self, monkeypatch):
        """The control. _DummyExecutor's contract is satisfied, so nothing
        about this path changes."""
        job = _DummyJob(status="running")
        await self._finalize(monkeypatch, _DummyExecutor(), job, _state_with(100))

        assert job.status == AgentJobStatus.COMPLETED.value
        assert job.progress == 100
        assert not [
            e for e in job.log_entries if e.get("phase") == "completed_contract_unmet"
        ]

    async def test_an_errored_run_is_still_a_failure(self, monkeypatch):
        """An unmet contract must not upgrade a failing run to completed."""
        job = _DummyJob(status="running")
        job.error_count = 6
        await self._finalize(
            monkeypatch, _UnmetContractExecutor(), job, _state_with(20)
        )

        assert job.status == AgentJobStatus.FAILED.value

    async def test_a_paused_run_stays_paused(self, monkeypatch):
        job = _DummyJob(status=AgentJobStatus.PAUSED.value)
        await self._finalize(
            monkeypatch, _UnmetContractExecutor(), job, _state_with(100)
        )

        assert job.status == AgentJobStatus.PAUSED.value

    async def test_a_cancelled_run_stays_cancelled(self, monkeypatch):
        job = _DummyJob(status=AgentJobStatus.CANCELLED.value)
        await self._finalize(
            monkeypatch, _UnmetContractExecutor(), job, _state_with(100)
        )

        assert job.status == AgentJobStatus.CANCELLED.value

    async def test_a_disabled_contract_does_not_gate_anything(self, monkeypatch):
        """Most jobs have no contract. They must finish exactly as before."""

        class _NoContract(_DummyExecutor):
            def _evaluate_goal_contract(self, job, state):
                return {"enabled": False, "satisfied": True, "missing": []}

        job = _DummyJob(status="running")
        await self._finalize(monkeypatch, _NoContract(), job, _state_with(100))

        assert job.status == AgentJobStatus.COMPLETED.value
        assert job.progress == 100
