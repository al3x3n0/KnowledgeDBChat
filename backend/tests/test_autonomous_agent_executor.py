from datetime import datetime, timedelta, timezone
from uuid import uuid4

from app.models.agent_job import AgentJob, AgentJobStatus
from app.services.autonomous_agent_executor import AutonomousAgentExecutor


def _make_job(config=None) -> AgentJob:
    return AgentJob(
        name="Executor Test",
        goal="Improve retrieval quality for knowledge base questions",
        job_type="research",
        user_id=uuid4(),
        status=AgentJobStatus.RUNNING.value,
        config=config or {},
        max_iterations=20,
        max_tool_calls=20,
        max_llm_calls=20,
        max_runtime_minutes=30,
    )


def _make_state() -> dict:
    return {
        "findings": [],
        "actions_taken": [],
        "goal_progress": 0,
        "execution_plan": [],
        "plan_step_index": 0,
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
        "goal_contract_last": {},
        "goal_contract_satisfied_iteration": 0,
        "approval_checkpoint_pending": None,
        "approval_checkpoint_events": [],
        "approval_checkpoint_seen": [],
    }


def test_parse_decision_response_handles_markdown_wrapped_json():
    executor = AutonomousAgentExecutor()
    job = _make_job()
    state = _make_state()
    available_tools = executor._get_tools_for_job_type(job.job_type, job.config)

    raw = """```json
{
  "goal_achieved": false,
  "should_stop": false,
  "reasoning": "Need more internal context before synthesis",
  "assessment": 35,
  "action": {
    "tool": "search_documents",
    "params": {"query": "retrieval quality", "limit": 5},
    "purpose": "Find stronger evidence"
  }
}
```"""
    decision = executor._parse_decision_response(raw, job, state, available_tools)

    assert decision["goal_achieved"] is False
    assert decision["should_stop"] is False
    assert decision["action"] is not None
    assert decision["action"]["tool"] == "search_documents"


def test_parse_decision_response_recovers_from_invalid_tool():
    executor = AutonomousAgentExecutor()
    job = _make_job()
    state = _make_state()
    available_tools = executor._get_tools_for_job_type(job.job_type, job.config)

    raw = """{
      "goal_achieved": false,
      "should_stop": false,
      "reasoning": "Try a tool",
      "action": {"tool": "non_existent_tool", "params": {"foo": "bar"}}
    }"""
    decision = executor._parse_decision_response(raw, job, state, available_tools)

    assert decision["goal_achieved"] is False
    assert decision["should_stop"] is False
    assert decision["action"] is not None
    assert decision["action"]["tool"] in set(available_tools)


def test_update_stall_state_triggers_recovery_and_stop_thresholds():
    executor = AutonomousAgentExecutor()
    job = _make_job(
        config={
            "stall_detection_enabled": True,
            "stall_min_progress_delta": 0,
            "stall_max_iterations_without_progress": 2,
            "stall_max_repeated_actions": 2,
            "stall_hard_stop_iterations": 3,
            "stall_max_recovery_actions": 1,
        }
    )
    state = _make_state()
    action = {"tool": "search_documents", "params": {"query": "retrieval quality"}}

    first = executor._update_stall_state(job, state, progress=10, action=action)
    second = executor._update_stall_state(job, state, progress=10, action=action)
    third = executor._update_stall_state(job, state, progress=10, action=action)

    assert first["should_recover"] is False
    assert second["should_recover"] is True
    assert third["should_stop"] is True


def test_fallback_execution_plan_produces_multiple_steps():
    executor = AutonomousAgentExecutor()
    job = _make_job(config={"plan_then_act_enabled": True, "plan_max_steps": 5})

    plan = executor._fallback_execution_plan(job, max_steps=5)

    assert isinstance(plan, list)
    assert len(plan) >= 3
    assert all(isinstance(step, dict) for step in plan)
    assert all(step.get("title") for step in plan)


def test_record_tool_outcome_tracks_success_and_failure():
    executor = AutonomousAgentExecutor()
    state = _make_state()
    state["tool_stats"] = {}

    executor._record_tool_outcome(
        state=state,
        action={"tool": "search_documents", "params": {"query": "quality"}},
        action_result={"success": True},
    )
    executor._record_tool_outcome(
        state=state,
        action={"tool": "search_documents", "params": {"query": "quality"}},
        action_result={"success": False, "error": "timeout"},
    )

    stats = state["tool_stats"]["search_documents"]
    assert stats["success"] == 1
    assert stats["failure"] == 1
    assert "timeout" in stats.get("last_error", "")


def test_research_evidence_quality_prefers_richer_findings():
    executor = AutonomousAgentExecutor()
    sparse = [{"type": "document", "id": "doc-1"}]
    rich = [
        {"type": "document", "id": "doc-1", "score": 0.92},
        {"type": "paper", "arxiv_id": "2401.00001", "authors": ["A"], "published": "2024-01-01"},
        {"type": "paper", "arxiv_id": "2401.00002", "authors": ["B"], "published": "2024-01-02"},
        {"type": "insight", "category": "key_insight"},
    ]

    sparse_score = executor._score_research_evidence_quality(sparse, target_docs=8, target_papers=8)
    rich_score = executor._score_research_evidence_quality(rich, target_docs=8, target_papers=8)

    assert 0.0 <= sparse_score <= 1.0
    assert 0.0 <= rich_score <= 1.0
    assert rich_score > sparse_score


def test_ensure_subgoals_uses_execution_plan_titles():
    executor = AutonomousAgentExecutor()
    job = _make_job(config={"subgoal_decomposition_enabled": True, "max_subgoals": 4})
    state = _make_state()
    state["execution_plan"] = [
        {"title": "Collect internal docs", "status": "pending"},
        {"title": "Validate with papers", "status": "pending"},
    ]

    executor._ensure_subgoals(job, state)

    assert len(state["subgoals"]) == 2
    assert state["subgoals"][0]["title"] == "Collect internal docs"
    assert state["subgoals"][0]["status"] == "in_progress"
    assert state["subgoal_index"] == 0


def test_should_run_critic_by_interval_and_stall():
    executor = AutonomousAgentExecutor()
    job = _make_job(
        config={
            "critic_enabled": True,
            "critic_every_n_iterations": 4,
            "critic_on_stall": True,
            "critic_stall_threshold": 2,
        }
    )
    state = _make_state()

    job.iteration = 4
    state["last_critic_iteration"] = 0
    state["stalled_iterations"] = 0
    assert executor._should_run_critic(job, state) is True

    state["last_critic_iteration"] = 4
    state["stalled_iterations"] = 0
    assert executor._should_run_critic(job, state) is False

    state["stalled_iterations"] = 3
    assert executor._should_run_critic(job, state) is True


def test_recovery_action_prefers_critic_recommendation_when_usable():
    executor = AutonomousAgentExecutor()
    job = _make_job()
    state = _make_state()
    state["critic_notes"] = [{"recommended_tools": ["summarize_document"]}]
    state["findings"] = [{"type": "document", "id": "doc-123"}]

    action = executor._build_recovery_action(job, state, exclude_tool="search_documents")

    assert action is not None
    assert action["tool"] == "summarize_document"
    assert action["params"]["document_id"] == "doc-123"


def test_ensure_subgoal_chain_config_creates_child_jobs():
    executor = AutonomousAgentExecutor()
    job = _make_job(config={"auto_subgoal_child_jobs_enabled": True, "auto_subgoal_child_jobs_max": 2})
    state = _make_state()
    state["subgoals"] = [
        {"title": "Scope", "status": "in_progress"},
        {"title": "Collect evidence", "status": "pending"},
        {"title": "Draft synthesis", "status": "pending"},
    ]

    executor._ensure_subgoal_chain_config(job, state)

    assert state["subgoal_chain_configured"] is True
    assert isinstance(job.chain_config, dict)
    children = job.chain_config.get("child_jobs")
    assert isinstance(children, list)
    assert len(children) == 2
    assert "Subgoal:" in children[0]["goal"]


def test_ensure_swarm_chain_config_creates_specialized_child_jobs():
    executor = AutonomousAgentExecutor()
    job = _make_job(
        config={
            "swarm_child_jobs_enabled": True,
            "swarm_max_agents": 3,
            "swarm_roles": ["researcher_documents", "researcher_arxiv", "analyst"],
            "swarm_inherit_config": False,
            "swarm_inherit_results": True,
        }
    )
    state = _make_state()

    executor._ensure_swarm_chain_config(job, state)

    assert state["swarm_chain_configured"] is True
    assert state["swarm_child_jobs_count"] == 3
    assert len(state["swarm_roles_assigned"]) == 3
    assert state["swarm_fan_in_enabled"] is True
    assert state["swarm_fan_in_group_id"]
    assert isinstance(job.chain_config, dict)
    assert job.chain_config.get("chain_data", {}).get("source") == "swarm_child_jobs"
    assert job.chain_config.get("chain_data", {}).get("swarm_fan_in_enabled") is True
    assert job.chain_config.get("inherit_results") is True
    assert job.chain_config.get("inherit_config") is False

    children = job.chain_config.get("child_jobs")
    assert isinstance(children, list)
    assert len(children) == 3
    assert children[0]["config"]["origin"] == "swarm_child_agent"
    assert children[0]["config"]["swarm_role_index"] == 1
    assert children[0]["config"]["swarm_child_jobs_enabled"] is False
    assert children[0]["config"]["auto_subgoal_child_jobs_enabled"] is False
    assert isinstance(children[0].get("chain_config"), dict)
    assert children[0]["chain_config"]["chain_data"]["source"] == "swarm_fan_in"
    fan_in_child = children[0]["chain_config"]["child_jobs"][0]
    assert fan_in_child["config"]["origin"] == "swarm_fan_in_aggregator"
    assert fan_in_child["config"]["deterministic_runner"] == "swarm_fan_in_aggregate"
    assert fan_in_child["config"]["swarm_fan_in_group_id"] == state["swarm_fan_in_group_id"]


def test_swarm_chain_config_takes_precedence_over_subgoal_child_jobs():
    executor = AutonomousAgentExecutor()
    job = _make_job(
        config={
            "swarm_child_jobs_enabled": True,
            "swarm_max_agents": 2,
            "swarm_roles": ["researcher_documents", "analyst"],
            "auto_subgoal_child_jobs_enabled": True,
            "auto_subgoal_child_jobs_max": 4,
        }
    )
    state = _make_state()
    state["subgoals"] = [
        {"title": "Scope", "status": "in_progress"},
        {"title": "Collect evidence", "status": "pending"},
        {"title": "Draft synthesis", "status": "pending"},
    ]

    executor._ensure_swarm_chain_config(job, state)
    executor._ensure_subgoal_chain_config(job, state)

    children = job.chain_config.get("child_jobs")
    assert isinstance(children, list)
    assert len(children) == 2
    assert job.chain_config.get("chain_data", {}).get("source") == "swarm_child_jobs"
    assert state["swarm_chain_configured"] is True
    assert state["subgoal_chain_configured"] is True


def test_swarm_chain_config_can_disable_fan_in():
    executor = AutonomousAgentExecutor()
    job = _make_job(
        config={
            "swarm_child_jobs_enabled": True,
            "swarm_fan_in_enabled": False,
            "swarm_max_agents": 2,
            "swarm_roles": ["researcher_documents", "analyst"],
        }
    )
    state = _make_state()

    executor._ensure_swarm_chain_config(job, state)

    children = job.chain_config.get("child_jobs")
    assert isinstance(children, list)
    assert len(children) == 2
    assert state["swarm_fan_in_enabled"] is False
    assert state["swarm_fan_in_group_id"] == ""
    assert job.chain_config.get("chain_data", {}).get("swarm_fan_in_enabled") is False
    assert all("chain_config" not in c for c in children)


def test_build_swarm_fan_in_result_aggregates_consensus_and_conflicts():
    executor = AutonomousAgentExecutor()
    payload = {
        "swarm_parent_job_id": "parent-1",
        "expected_siblings": 3,
        "terminal_siblings": 3,
        "sibling_jobs": [
            {
                "job_id": "j1",
                "role": "Knowledge Researcher",
                "status": "completed",
                "progress": 100,
                "results": {
                    "findings": [{"title": "Prioritize internal docs for baseline facts"}],
                    "summary": "Internal docs show repeated bottleneck in ingestion.",
                },
            },
            {
                "job_id": "j2",
                "role": "Literature Researcher",
                "status": "completed",
                "progress": 100,
                "results": {
                    "findings": [{"title": "Prioritize internal docs for baseline facts"}],
                    "research": {"top_insights": ["Benchmark against arXiv baselines for coverage"]},
                },
            },
            {
                "job_id": "j3",
                "role": "Analyst",
                "status": "failed",
                "progress": 70,
                "results": {
                    "summary": "Potential contradiction in metric definitions across sources.",
                },
            },
        ],
    }

    merged = executor._build_swarm_fan_in_result(payload, fan_in_group_id="group-123")

    assert merged["fan_in_group_id"] == "group-123"
    assert merged["expected_siblings"] == 3
    assert merged["received_siblings"] == 3
    assert merged["terminal_siblings"] == 3
    assert merged["confidence"]["overall"] >= 0.0
    assert isinstance(merged["consensus_findings"], list)
    assert merged["consensus_findings"]
    assert merged["consensus_findings"][0]["support_count"] >= 2
    assert isinstance(merged["conflicts"], list)
    assert any(c.get("type") == "execution_divergence" for c in merged["conflicts"])
    assert isinstance(merged["action_plan"], list)
    assert merged["action_plan"]


def test_maybe_apply_critic_pivot_override_forces_recommended_tool():
    executor = AutonomousAgentExecutor()
    job = _make_job(config={"critic_force_pivot_on_high": True, "critic_force_min_confidence": 0.5})
    state = _make_state()
    state["critic_notes"] = [
        {
            "severity": "high",
            "confidence": 0.9,
            "pivot": "Shift back to internal document evidence",
            "recommended_tools": ["search_documents"],
        }
    ]

    decision = {
        "goal_achieved": False,
        "should_stop": False,
        "reasoning": "Continue as planned.",
        "action": {"tool": "search_arxiv", "params": {"query": "x"}},
    }

    updated = executor._maybe_apply_critic_pivot_override(job, state, decision)

    assert updated["action"]["tool"] == "search_documents"
    assert "Critic override applied" in updated.get("reasoning", "")


def test_merge_tool_stats_sums_success_and_failure():
    executor = AutonomousAgentExecutor()
    merged = executor._merge_tool_stats(
        {"search_documents": {"success": 2, "failure": 1}},
        {"search_documents": {"success": 3, "failure": 4}, "search_arxiv": {"success": 1, "failure": 0}},
    )

    assert merged["search_documents"]["success"] == 5
    assert merged["search_documents"]["failure"] == 5
    assert merged["search_arxiv"]["success"] == 1


def test_build_action_from_recommended_tools_uses_priors():
    executor = AutonomousAgentExecutor()
    job = _make_job()
    state = _make_state()
    state["tool_priors"] = {
        "search_documents": {"success": 8, "failure": 1},
        "search_arxiv": {"success": 0, "failure": 7},
    }

    action = executor._build_action_from_recommended_tools(
        job=job,
        state=state,
        recommended_tools=["search_arxiv", "search_documents"],
        exclude_tool=None,
    )

    assert action is not None
    assert action["tool"] == "search_documents"


def test_recovery_action_avoids_historically_bad_tool():
    executor = AutonomousAgentExecutor()
    job = _make_job()
    state = _make_state()
    state["tool_priors"] = {
        "search_documents": {"success": 0, "failure": 10},
        "search_arxiv": {"success": 4, "failure": 1},
    }
    state["findings"] = []  # no documents yet

    action = executor._build_recovery_action(job, state)

    assert action is not None
    assert action["tool"] == "search_arxiv"


def test_get_tool_prior_decay_config_clamps_values():
    executor = AutonomousAgentExecutor()
    job = _make_job(
        config={
            "tool_prior_decay_enabled": True,
            "tool_prior_half_life_days": -10,
            "tool_prior_decay_min_factor": 5,
        }
    )

    cfg = executor._get_tool_prior_decay_config(job)

    assert cfg["enabled"] is True
    assert cfg["half_life_days"] == 1.0
    assert cfg["min_factor"] == 1.0


def test_apply_decay_to_prior_counts_respects_disabled_flag():
    executor = AutonomousAgentExecutor()
    now = datetime(2026, 2, 6, 12, 0, 0)
    updated = now - timedelta(days=90)

    s, f = executor._apply_decay_to_prior_counts(
        success_count=10,
        failure_count=6,
        updated_at=updated,
        now=now,
        enabled=False,
    )

    assert s == 10
    assert f == 6


def test_apply_decay_to_prior_counts_applies_half_life():
    executor = AutonomousAgentExecutor()
    now = datetime(2026, 2, 6, 12, 0, 0)
    updated = now - timedelta(days=45)

    s, f = executor._apply_decay_to_prior_counts(
        success_count=20,
        failure_count=10,
        updated_at=updated,
        now=now,
        enabled=True,
        half_life_days=45.0,
        min_factor=0.01,
    )

    assert s == 10
    assert f == 5


def test_apply_decay_to_prior_counts_handles_timezone_aware_timestamps():
    executor = AutonomousAgentExecutor()
    now_utc = datetime(2026, 2, 6, 12, 0, 0, tzinfo=timezone.utc)
    # Same instant as 2026-02-06 12:00:00+00:00.
    updated_same_instant = datetime(2026, 2, 6, 7, 0, 0, tzinfo=timezone(timedelta(hours=-5)))

    s, f = executor._apply_decay_to_prior_counts(
        success_count=8,
        failure_count=4,
        updated_at=updated_same_instant,
        now=now_utc,
        enabled=True,
        half_life_days=45.0,
        min_factor=0.01,
    )

    assert s == 8
    assert f == 4


def test_build_action_from_recommended_tools_exploration_prefers_under_sampled_tool():
    executor = AutonomousAgentExecutor()
    job = _make_job(
        config={
            "tool_selection_exploration_enabled": True,
            "tool_selection_exploration_bonus": 0.8,
            "tool_selection_cold_start_bonus": 0.2,
            "tool_selection_min_trials": 10,
            "tool_selection_failure_penalty": 0.02,
        }
    )
    state = _make_state()
    state["tool_priors"] = {
        "search_documents": {"success": 20, "failure": 5},
        "search_arxiv": {"success": 2, "failure": 0},
    }

    action = executor._build_action_from_recommended_tools(
        job=job,
        state=state,
        recommended_tools=["search_documents", "search_arxiv"],
        exclude_tool=None,
    )

    assert action is not None
    assert action["tool"] == "search_arxiv"


def test_build_action_from_recommended_tools_without_exploration_prefers_best_ratio():
    executor = AutonomousAgentExecutor()
    job = _make_job(
        config={
            "tool_selection_exploration_enabled": False,
            "tool_selection_exploration_bonus": 0.8,
            "tool_selection_cold_start_bonus": 0.2,
            "tool_selection_min_trials": 10,
        }
    )
    state = _make_state()
    state["tool_priors"] = {
        "search_documents": {"success": 20, "failure": 5},
        "search_arxiv": {"success": 2, "failure": 0},
    }

    action = executor._build_action_from_recommended_tools(
        job=job,
        state=state,
        recommended_tools=["search_documents", "search_arxiv"],
        exclude_tool=None,
    )

    assert action is not None
    assert action["tool"] == "search_documents"


def test_should_force_exploration_uses_stall_cadence():
    executor = AutonomousAgentExecutor()
    job = _make_job(
        config={
            "tool_selection_forced_exploration_enabled": True,
            "tool_selection_forced_exploration_every_n": 2,
            "tool_selection_forced_exploration_min_stalled": 2,
        }
    )
    state = _make_state()
    state["stalled_iterations"] = 1
    state["repeated_action_iterations"] = 1
    assert executor._should_force_exploration(job, state) is False

    state["stalled_iterations"] = 2
    state["repeated_action_iterations"] = 0
    assert executor._should_force_exploration(job, state) is True

    state["stalled_iterations"] = 3
    state["repeated_action_iterations"] = 0
    assert executor._should_force_exploration(job, state) is False


def test_recovery_action_forced_exploration_prefers_under_sampled_tool():
    executor = AutonomousAgentExecutor()
    job = _make_job(
        config={
            "tool_selection_forced_exploration_enabled": True,
            "tool_selection_forced_exploration_every_n": 1,
            "tool_selection_forced_exploration_min_stalled": 1,
            "tool_selection_forced_exploration_max_observations": 2,
            "tool_selection_forced_exploration_tools": ["search_arxiv", "search_documents"],
            "tool_selection_cooldown_enabled": True,
            "tool_selection_cooldown_iterations": 2,
            "tool_selection_cooldown_forced_only": True,
        }
    )
    job.iteration = 5
    state = _make_state()
    state["stalled_iterations"] = 2
    state["tool_priors"] = {
        "search_documents": {"success": 12, "failure": 2},
        "search_arxiv": {"success": 1, "failure": 0},
    }
    state["findings"] = []

    action = executor._build_recovery_action(job, state)

    assert action is not None
    assert action["tool"] == "search_arxiv"
    assert state["forced_exploration_attempts"] == 1
    assert state["forced_exploration_used"] == 1
    assert state["tool_cooldowns"]["search_arxiv"] >= 7

    action2 = executor._build_recovery_action(job, state)
    assert action2 is not None
    assert action2["tool"] == "search_documents"
    assert state["forced_exploration_attempts"] == 2
    assert state["forced_exploration_used"] == 1
    assert state["tool_cooldown_blocks"] >= 1


def test_recovery_action_without_forced_exploration_uses_default_priority():
    executor = AutonomousAgentExecutor()
    job = _make_job(
        config={
            "tool_selection_forced_exploration_enabled": False,
            "tool_selection_forced_exploration_tools": ["search_arxiv", "search_documents"],
        }
    )
    state = _make_state()
    state["stalled_iterations"] = 2
    state["tool_priors"] = {
        "search_documents": {"success": 12, "failure": 2},
        "search_arxiv": {"success": 1, "failure": 0},
    }
    state["findings"] = []

    action = executor._build_recovery_action(job, state)

    assert action is not None
    assert action["tool"] == "search_documents"


def test_apply_recovery_post_action_updates_extends_cooldown_on_forced_failure():
    executor = AutonomousAgentExecutor()
    job = _make_job(
        config={
            "tool_selection_cooldown_enabled": True,
            "tool_selection_cooldown_failure_extra_iterations": 3,
            "tool_selection_cooldown_success_shorten_by": 1,
        }
    )
    job.iteration = 6
    state = _make_state()
    state["last_recovery_was_forced_exploration"] = True
    state["tool_cooldowns"] = {"search_arxiv": 8}
    state["forced_exploration_history"] = [{"iteration": 6, "tool": "search_arxiv", "success": None}]

    executor._apply_recovery_post_action_updates(
        job=job,
        state=state,
        recovery_action={"tool": "search_arxiv", "params": {}},
        recovery_result={"success": False, "error": "timeout"},
    )

    assert state["forced_exploration_failures"] == 1
    assert state["forced_exploration_successes"] == 0
    assert state["tool_cooldowns"]["search_arxiv"] == 11
    assert state["forced_exploration_history"][-1]["success"] is False


def test_apply_recovery_post_action_updates_shortens_cooldown_on_forced_success():
    executor = AutonomousAgentExecutor()
    job = _make_job(
        config={
            "tool_selection_cooldown_enabled": True,
            "tool_selection_cooldown_failure_extra_iterations": 3,
            "tool_selection_cooldown_success_shorten_by": 2,
        }
    )
    job.iteration = 6
    state = _make_state()
    state["last_recovery_was_forced_exploration"] = True
    state["tool_cooldowns"] = {"search_arxiv": 10}
    state["forced_exploration_history"] = [{"iteration": 6, "tool": "search_arxiv", "success": None}]

    executor._apply_recovery_post_action_updates(
        job=job,
        state=state,
        recovery_action={"tool": "search_arxiv", "params": {}},
        recovery_result={"success": True},
    )

    assert state["forced_exploration_successes"] == 1
    assert state["forced_exploration_failures"] == 0
    assert state["tool_cooldowns"]["search_arxiv"] == 8
    assert state["forced_exploration_history"][-1]["success"] is True


def test_resolve_tool_selection_mode_ab_split_selects_variant_a_and_b():
    executor = AutonomousAgentExecutor()

    job_a = _make_job(
        config={
            "tool_selection_policy_mode": "adaptive",
            "tool_selection_ab_test_enabled": True,
            "tool_selection_ab_test_split": 1.0,
            "tool_selection_ab_test_variant_a": "baseline",
            "tool_selection_ab_test_variant_b": "thompson",
        }
    )
    state_a = _make_state()
    mode_a, assignment_a = executor._resolve_tool_selection_mode(job_a, state=state_a)
    assert mode_a == "baseline"
    assert assignment_a["variant"] == "A"
    assert state_a["tool_selection_effective_mode"] == "baseline"

    job_b = _make_job(
        config={
            "tool_selection_policy_mode": "adaptive",
            "tool_selection_ab_test_enabled": True,
            "tool_selection_ab_test_split": 0.0,
            "tool_selection_ab_test_variant_a": "baseline",
            "tool_selection_ab_test_variant_b": "thompson",
        }
    )
    state_b = _make_state()
    mode_b, assignment_b = executor._resolve_tool_selection_mode(job_b, state=state_b)
    assert mode_b == "thompson"
    assert assignment_b["variant"] == "B"
    assert state_b["tool_selection_effective_mode"] == "thompson"


def test_rank_tools_for_selection_thompson_mode_is_deterministic_for_same_state():
    executor = AutonomousAgentExecutor()
    job = _make_job(config={"tool_selection_policy_mode": "thompson"})
    job.iteration = 7
    state = _make_state()
    state["forced_exploration_attempts"] = 2
    combined = {
        "search_documents": {"success": 12, "failure": 4},
        "search_arxiv": {"success": 2, "failure": 0},
        "summarize_document": {"success": 1, "failure": 2},
    }
    tools = ["search_documents", "search_arxiv", "summarize_document"]

    ranked1 = executor._rank_tools_for_selection(
        job,
        tools,
        combined,
        state=state,
        context_tag="unit_test",
    )
    ranked2 = executor._rank_tools_for_selection(
        job,
        tools,
        combined,
        state=state,
        context_tag="unit_test",
    )

    assert ranked1 == ranked2
    assert set(ranked1) == set(tools)
    assert state["tool_selection_effective_mode"] == "thompson"


def test_simulate_tool_selection_replay_returns_metrics_for_requested_modes():
    executor = AutonomousAgentExecutor()
    tool_stats = {
        "search_documents": {"success": 20, "failure": 10},
        "search_arxiv": {"success": 6, "failure": 2},
        "summarize_document": {"success": 1, "failure": 3},
    }

    replay = executor.simulate_tool_selection_replay(
        tool_stats,
        steps=120,
        policy_modes=["baseline", "thompson"],
        seed=123,
    )

    assert replay["steps"] == 120
    assert "baseline" in replay["modes"]
    assert "thompson" in replay["modes"]
    assert replay["modes"]["baseline"]["steps"] == 120
    assert replay["modes"]["thompson"]["steps"] == 120
    assert replay["modes"]["baseline"]["unique_tools_selected"] >= 1
    assert replay["modes"]["thompson"]["unique_tools_selected"] >= 1
    assert replay["best_possible_mean_reward"] >= 0.0
    assert isinstance(replay["comparison"], list)
    assert replay["comparison"]
    assert "cumulative_expected_regret" in replay["comparison"][0]


def test_live_mode_guardrail_falls_back_to_configured_mode():
    executor = AutonomousAgentExecutor()
    job = _make_job(
        config={
            "tool_selection_policy_mode": "thompson",
            "tool_selection_live_fallback_enabled": True,
            "tool_selection_live_fallback_min_samples": 3,
            "tool_selection_live_fallback_min_success_rate": 0.5,
            "tool_selection_live_fallback_to_mode": "adaptive",
        }
    )
    state = _make_state()
    state["tool_selection_mode_metrics"] = {
        "thompson": {"success": 0, "failure": 4},
    }
    job.iteration = 9

    mode, assignment = executor._resolve_tool_selection_mode(job, state=state)

    assert mode == "adaptive"
    assert assignment["mode"] == "adaptive"
    assert state["tool_selection_mode_override"] == "adaptive"
    assert state["tool_selection_fallback_events"]
    assert state["tool_selection_fallback_events"][-1]["from_mode"] == "thompson"


def test_build_counterfactual_candidates_returns_ranked_scored_tools():
    executor = AutonomousAgentExecutor()
    job = _make_job(config={"tool_selection_policy_mode": "baseline"})
    state = _make_state()
    state["tool_priors"] = {
        "search_documents": {"success": 9, "failure": 1},
        "search_arxiv": {"success": 2, "failure": 2},
        "summarize_document": {"success": 1, "failure": 3},
    }

    candidates = executor._build_counterfactual_candidates(
        job=job,
        state=state,
        selected_tool="search_documents",
        limit=3,
        context_tag="unit_counterfactual",
    )

    assert len(candidates) >= 1
    assert len(candidates) <= 3
    assert candidates[0]["tool"] == "search_documents"
    assert isinstance(candidates[0]["priority_score"], float)
    assert any(bool(c.get("selected")) for c in candidates)


def test_goal_stage_schedule_changes_mode_by_progress_and_stall():
    executor = AutonomousAgentExecutor()
    job = _make_job(
        config={
            "tool_selection_policy_mode": "adaptive",
            "tool_selection_stage_schedule_enabled": True,
            "tool_selection_stage_discovery_mode": "thompson",
            "tool_selection_stage_consolidation_mode": "adaptive",
            "tool_selection_stage_finish_mode": "baseline",
            "tool_selection_stage_rescue_mode": "adaptive",
            "tool_selection_stage_finish_progress": 80,
            "tool_selection_stage_discovery_progress": 35,
            "tool_selection_stage_rescue_stall_threshold": 3,
        }
    )
    state = _make_state()

    state["goal_progress"] = 10
    state["stalled_iterations"] = 0
    state["findings"] = []
    mode1, _ = executor._resolve_tool_selection_mode(job, state=state)
    assert mode1 == "thompson"
    assert state["tool_selection_goal_stage"] == "discovery"

    state["goal_progress"] = 55
    state["findings"] = [{"type": "document", "id": "d1"}, {"type": "document", "id": "d2"}, {"type": "paper", "id": "p1"}]
    state["stalled_iterations"] = 0
    mode2, _ = executor._resolve_tool_selection_mode(job, state=state)
    assert mode2 == "adaptive"
    assert state["tool_selection_goal_stage"] == "consolidation"

    state["goal_progress"] = 60
    state["stalled_iterations"] = 4
    mode3, _ = executor._resolve_tool_selection_mode(job, state=state)
    assert mode3 == "adaptive"
    assert state["tool_selection_goal_stage"] == "rescue"

    state["goal_progress"] = 90
    state["stalled_iterations"] = 0
    mode4, _ = executor._resolve_tool_selection_mode(job, state=state)
    assert mode4 == "baseline"
    assert state["tool_selection_goal_stage"] == "finish"


def test_live_mode_guardrail_reset_clears_override_after_recovery():
    executor = AutonomousAgentExecutor()
    job = _make_job(
        config={
            "tool_selection_policy_mode": "thompson",
            "tool_selection_live_fallback_enabled": True,
            "tool_selection_live_fallback_reset_enabled": True,
            "tool_selection_live_fallback_reset_min_samples": 5,
            "tool_selection_live_fallback_reset_min_success_rate": 0.7,
        }
    )
    state = _make_state()
    state["tool_selection_mode_override"] = "adaptive"
    state["tool_selection_mode_metrics"] = {
        "adaptive": {"success": 6, "failure": 2},
    }
    job.iteration = 11

    mode, assignment = executor._resolve_tool_selection_mode(job, state=state)

    assert mode == "thompson"
    assert assignment["mode"] == "thompson"
    assert state["tool_selection_mode_override"] == ""
    assert state["tool_selection_fallback_events"]
    assert state["tool_selection_fallback_events"][-1]["event"] == "reset_override"


def test_build_selection_explainability_includes_score_gaps_and_metadata():
    executor = AutonomousAgentExecutor()
    state = _make_state()
    state["tool_selection_effective_mode"] = "adaptive"
    state["tool_selection_goal_stage"] = "consolidation"
    state["tool_selection_mode_override"] = ""
    state["tool_selection_fallback_events"] = [{"event": "fallback"}]
    candidates = [
        {"rank": 1, "tool": "search_documents", "priority_score": 0.92},
        {"rank": 2, "tool": "search_arxiv", "priority_score": 0.81},
    ]

    expl = executor._build_selection_explainability(
        state=state,
        selected_tool="search_arxiv",
        candidates=candidates,
    )

    assert expl["selected_tool"] == "search_arxiv"
    assert expl["effective_mode"] == "adaptive"
    assert expl["goal_stage"] == "consolidation"
    assert expl["selected_rank"] == 2
    assert expl["top_tool"] == "search_documents"
    assert expl["score_gap_to_top"] > 0.0
    assert expl["fallback_event_count"] == 1


def test_rank_tools_for_selection_family_diversification_boosts_underrepresented_family():
    executor = AutonomousAgentExecutor()
    tools = ["search_documents", "create_document_from_text"]
    combined = {
        "search_documents": {"success": 8, "failure": 1},
        "create_document_from_text": {"success": 7, "failure": 2},
    }

    state = _make_state()
    state["actions_taken"] = [
        {"action": {"tool": "search_documents"}},
        {"action": {"tool": "search_with_filters"}},
        {"action": {"tool": "search_documents"}},
        {"action": {"tool": "find_similar_documents"}},
    ]

    job_without_diversification = _make_job(
        config={
            "tool_selection_policy_mode": "baseline",
            "tool_selection_family_diversification_enabled": False,
        }
    )
    ranked_without = executor._rank_tools_for_selection(
        job_without_diversification,
        tools,
        combined,
        state=state,
        context_tag="unit_family_diversification_off",
    )
    assert ranked_without[0] == "search_documents"

    job_with_diversification = _make_job(
        config={
            "tool_selection_policy_mode": "baseline",
            "tool_selection_family_diversification_enabled": True,
            "tool_selection_family_diversification_bonus": 0.4,
            "tool_selection_family_diversification_window": 4,
            "tool_selection_family_diversification_target_unique": 3,
        }
    )
    ranked_with = executor._rank_tools_for_selection(
        job_with_diversification,
        tools,
        combined,
        state=state,
        context_tag="unit_family_diversification_on",
    )
    assert ranked_with[0] == "create_document_from_text"


def test_should_run_critic_on_uncertainty_when_score_gap_is_small():
    executor = AutonomousAgentExecutor()
    job = _make_job(
        config={
            "critic_enabled": True,
            "critic_every_n_iterations": 99,
            "critic_on_stall": False,
            "critic_on_uncertainty": True,
            "critic_uncertainty_top_gap_threshold": 0.03,
            "critic_uncertainty_min_candidates": 2,
            "critic_uncertainty_max_age_iterations": 2,
            "critic_uncertainty_min_iterations_since_last": 1,
        }
    )
    state = _make_state()
    job.iteration = 6
    state["last_critic_iteration"] = 5
    state["counterfactual_last_iteration"] = 5
    state["counterfactual_last"] = [
        {"rank": 1, "tool": "search_documents", "priority_score": 0.81},
        {"rank": 2, "tool": "search_arxiv", "priority_score": 0.80},
    ]

    assert executor._should_run_critic(job, state) is True
    assert state["critic_last_trigger"]["reason"] == "uncertainty"
    assert state["critic_last_trigger"]["by_uncertainty"] is True
    assert state["critic_trigger_counts"]["uncertainty"] >= 1

    state["counterfactual_last"] = [
        {"rank": 1, "tool": "search_documents", "priority_score": 0.81},
        {"rank": 2, "tool": "search_arxiv", "priority_score": 0.40},
    ]
    assert executor._should_run_critic(job, state) is False


def test_should_run_critic_uncertainty_threshold_scales_by_stage_and_mode():
    executor = AutonomousAgentExecutor()
    job = _make_job(
        config={
            "critic_enabled": True,
            "critic_every_n_iterations": 99,
            "critic_on_stall": False,
            "critic_on_uncertainty": True,
            "critic_uncertainty_top_gap_threshold": 0.02,
            "critic_uncertainty_stage_schedule_enabled": True,
            "critic_uncertainty_mode_schedule_enabled": True,
            "critic_uncertainty_stage_multiplier_discovery": 2.0,
            "critic_uncertainty_stage_multiplier_finish": 0.5,
            "critic_uncertainty_mode_multiplier_thompson": 2.0,
            "critic_uncertainty_mode_multiplier_baseline": 0.5,
            "critic_uncertainty_threshold_min": 0.001,
            "critic_uncertainty_threshold_max": 0.5,
            "critic_uncertainty_min_candidates": 2,
            "critic_uncertainty_max_age_iterations": 2,
            "critic_uncertainty_min_iterations_since_last": 1,
        }
    )
    state = _make_state()
    job.iteration = 8
    state["last_critic_iteration"] = 7
    state["counterfactual_last_iteration"] = 7
    state["counterfactual_last"] = [
        {"rank": 1, "tool": "search_documents", "priority_score": 0.81},
        {"rank": 2, "tool": "search_arxiv", "priority_score": 0.76},
    ]

    state["tool_selection_goal_stage"] = "discovery"
    state["tool_selection_effective_mode"] = "thompson"
    assert executor._should_run_critic(job, state) is True
    assert state["critic_last_trigger"]["uncertainty_effective_threshold"] == 0.08

    state["tool_selection_goal_stage"] = "finish"
    state["tool_selection_effective_mode"] = "baseline"
    assert executor._should_run_critic(job, state) is False


def test_evaluate_goal_contract_reports_missing_and_satisfied_states():
    executor = AutonomousAgentExecutor()
    job = _make_job(
        config={
            "goal_contract_enabled": True,
            "goal_contract_min_progress": 70,
            "goal_contract_min_findings": 2,
            "goal_contract_required_finding_types": ["paper"],
            "goal_contract_required_artifact_types": ["document"],
        }
    )
    state = _make_state()
    state["goal_progress"] = 65
    state["findings"] = [{"type": "document", "id": "d1"}]
    state["artifacts"] = []

    unmet = executor._evaluate_goal_contract(job, state)
    assert unmet["enabled"] is True
    assert unmet["satisfied"] is False
    assert "progress>=70" in unmet["missing"]
    assert "findings>=2" in unmet["missing"]
    assert "finding_type:paper" in unmet["missing"]
    assert "artifact_type:document" in unmet["missing"]

    state["goal_progress"] = 85
    state["findings"] = [{"type": "document", "id": "d1"}, {"type": "paper", "arxiv_id": "2401.00001"}]
    state["artifacts"] = [{"type": "document", "id": "out-1"}]
    met = executor._evaluate_goal_contract(job, state)
    assert met["satisfied"] is True
    assert met["missing"] == []


def test_goal_contract_can_skip_result_key_checks_in_loop_mode():
    executor = AutonomousAgentExecutor()
    job = _make_job(
        config={
            "goal_contract_enabled": True,
            "goal_contract_min_progress": 100,
            "goal_contract_required_result_keys": ["executive_digest"],
        }
    )
    state = _make_state()
    state["goal_progress"] = 100

    loop_eval = executor._evaluate_goal_contract(job, state, include_result_keys=False)
    finalize_eval = executor._evaluate_goal_contract(job, state, include_result_keys=True)

    assert loop_eval["satisfied"] is True
    assert finalize_eval["satisfied"] is False
    assert "result_key:executive_digest" in finalize_eval["missing"]


def test_approval_checkpoint_triggers_for_tool_and_then_suppresses_repeats():
    executor = AutonomousAgentExecutor()
    job = _make_job(
        config={
            "approval_checkpoints_enabled": True,
            "approval_checkpoint_tools": ["create_document_from_text"],
            "approval_checkpoint_once_per_checkpoint": True,
        }
    )
    state = _make_state()
    action = {
        "tool": "create_document_from_text",
        "params": {"title": "Draft"},
        "purpose": "Persist current synthesis",
    }

    first = executor._evaluate_approval_checkpoint(job, state, action)
    second = executor._evaluate_approval_checkpoint(job, state, action)

    assert first["required"] is True
    assert first["checkpoint"]["action"]["tool"] == "create_document_from_text"
    assert any(r.startswith("tool:create_document_from_text") for r in first["checkpoint"]["reasons"])
    assert second["required"] is False


def test_build_executive_digest_includes_risks_contract_and_next_steps():
    executor = AutonomousAgentExecutor()
    job = _make_job(
        config={
            "goal_contract_enabled": True,
            "goal_contract_min_findings": 2,
        }
    )
    job.results = {"summary": "Partial research outcome", "research_bundle": {"next_steps": ["Validate metrics", "Run follow-up search"]}}
    state = _make_state()
    state["goal_progress"] = 60
    state["findings"] = [{"type": "document", "title": "Internal ingestion bottleneck"}]
    state["artifacts"] = [{"type": "note", "id": "a1"}]
    state["actions_taken"] = [
        {"action": {"tool": "search_documents"}, "result": {"success": True}},
        {"action": {"tool": "search_arxiv"}, "result": {"success": False, "error": "timeout"}},
    ]
    state["critic_notes"] = [{"severity": "high", "pivot": "Need stronger external baselines"}]

    digest = executor._build_executive_digest(job, state)

    assert digest["outcome"] == "Partial research outcome"
    assert digest["metrics"]["failed_actions"] == 1
    assert digest["key_findings"]
    assert digest["risks"]
    assert digest["goal_contract"]["enabled"] is True
    assert digest["goal_contract"]["satisfied"] is False
    assert digest["next_actions"][0] == "Validate metrics"
