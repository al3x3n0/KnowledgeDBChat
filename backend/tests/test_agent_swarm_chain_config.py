"""Direct tests for the extracted swarm chain configuration.

No session, LLM or logging involved: the functions read a job plus its state
and plan the swarm child jobs, so they run here without an executor. The
step-event appender is injected, and these tests capture it to assert on it.
"""

from uuid import uuid4

from app.models.agent_job import AgentJob, AgentJobStatus
from app.services.agent_swarm_chain_config import (
    ensure_swarm_chain_config,
    get_swarm_config,
)


def _job(**config) -> AgentJob:
    return AgentJob(
        id=uuid4(),
        name="Swarm Test",
        goal="Investigate a flaky parser",
        job_type="research",
        user_id=uuid4(),
        status=AgentJobStatus.RUNNING.value,
        config=config,
        max_iterations=20,
        max_tool_calls=20,
        max_llm_calls=20,
        max_runtime_minutes=30,
    )


def _state() -> dict:
    return {"subgoals": [], "step_events": []}


class _Events(list):
    """Captures the injected step-event appender's calls."""

    def __call__(self, _state, event, **_kwargs):
        self.append(event)


def test_swarm_config_defaults_to_disabled():
    assert get_swarm_config(_job())["enabled"] is False


def test_swarm_config_exposes_normalized_budget_ratios():
    config = get_swarm_config(_job(swarm_child_jobs_enabled=True, swarm_max_agents=3))

    assert config["enabled"] is True
    assert config["max_agents"] == 3
    assert 0.0 < config["max_iterations_ratio"] <= 1.0
    assert config["trigger_condition"] == "on_complete"


def test_disabled_swarm_configures_no_children():
    job, state, events = _job(), _state(), _Events()

    ensure_swarm_chain_config(job, state, append_step_event=events)

    assert not (job.chain_config or {}).get("child_jobs")
    assert state.get("swarm_chain_configured") is not True
    assert events == []


def test_enabled_swarm_plans_children_and_records_the_decision():
    job = _job(swarm_child_jobs_enabled=True, swarm_max_agents=3)
    state, events = _state(), _Events()

    ensure_swarm_chain_config(job, state, append_step_event=events)

    children = (job.chain_config or {}).get("child_jobs") or []
    assert len(children) == 3
    assert state["swarm_chain_configured"] is True
    assert state["swarm_child_jobs_count"] == 3
    assert [e.get("type") for e in events] == ["swarm_roles_configured"]


def test_children_get_distinct_roles():
    job = _job(swarm_child_jobs_enabled=True, swarm_max_agents=3)
    state = _state()

    ensure_swarm_chain_config(job, state, append_step_event=_Events())

    roles = [c["config"].get("swarm_role") for c in job.chain_config["child_jobs"]]
    assert len(set(roles)) == len(roles), "each swarm agent should get its own role"
    assert state["swarm_roles_assigned"]


def test_max_agents_bounds_the_swarm():
    job = _job(swarm_child_jobs_enabled=True, swarm_max_agents=1)

    ensure_swarm_chain_config(job, _state(), append_step_event=_Events())

    assert len(job.chain_config["child_jobs"]) == 1


def test_children_get_a_fraction_of_the_parent_budget():
    job = _job(swarm_child_jobs_enabled=True, swarm_max_agents=2)

    ensure_swarm_chain_config(job, _state(), append_step_event=_Events())

    for child in job.chain_config["child_jobs"]:
        assert child["max_iterations"] <= job.max_iterations
        assert child["max_llm_calls"] <= job.max_llm_calls


def test_configuring_twice_does_not_duplicate_children():
    job = _job(swarm_child_jobs_enabled=True, swarm_max_agents=3)
    state = _state()

    ensure_swarm_chain_config(job, state, append_step_event=_Events())
    first = list(job.chain_config["child_jobs"])
    ensure_swarm_chain_config(job, state, append_step_event=_Events())

    assert job.chain_config["child_jobs"] == first


def test_explicit_roles_are_honoured():
    job = _job(
        swarm_child_jobs_enabled=True,
        swarm_max_agents=2,
        swarm_roles=["researcher_arxiv", "analyst"],
    )

    ensure_swarm_chain_config(job, _state(), append_step_event=_Events())

    children = job.chain_config["child_jobs"]
    assert len(children) == 2
    assert all(c["config"].get("swarm_role") for c in children)
