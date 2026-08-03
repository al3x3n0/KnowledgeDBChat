import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from sqlalchemy import select

from app.models.agent_job import AgentJob, AgentJobStatus
from app.models.document import Document, DocumentSource
from app.models.domain_research_profile import DomainResearchProfile
from app.models.experiment import ExperimentPlan
from app.models.research_inbox import ResearchInboxItem
from app.models.research_note import ResearchNote
from app.models.research_portfolio import ResearchPortfolio
from app.services.agent_execution_planner import ExecutionPlan, PlanStep
from app.services.autonomous_agent_executor import AutonomousAgentExecutor
from app.services.project_profile_service import infer_project_profile_from_paths
from app.services.research_opportunity_service import (
    merge_operator_fields,
    normalize_research_opportunity,
)


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
        "step_events": [],
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


def test_parse_decision_response_injects_default_source_scope():
    executor = AutonomousAgentExecutor()
    scoped_source = str(uuid4())
    job = _make_job(config={"source_id": scoped_source})
    state = _make_state()
    available_tools = executor._get_tools_for_job_type(job.job_type, job.config)

    raw = """{
      "goal_achieved": false,
      "should_stop": false,
      "reasoning": "Search within project scope first",
      "action": {"tool": "search_documents", "params": {"query": "retrieval quality"}}
    }"""
    decision = executor._parse_decision_response(raw, job, state, available_tools)

    assert decision["action"] is not None
    assert decision["action"]["tool"] == "search_documents"


@pytest.mark.asyncio
async def test_domain_research_follow_up_job_inherits_policy_and_sandbox_context(
    db_session,
):
    executor = AutonomousAgentExecutor()
    parent_job = _make_job(
        {
            "automation_profile": "balanced",
            "automation_policy": {"follow_up_review_mode": "queue_for_approval"},
            "sandbox_profile_id": "scientific-compiler-sandbox",
        }
    )
    db_session.add(parent_job)
    await db_session.flush()

    child = await executor._create_domain_research_follow_up_job(
        db=db_session,
        job=parent_job,
        domain="Compiler optimization",
        objective="Validate the strongest compiler hypothesis",
        customer_context="bounded compiler research",
        track_type="compiler",
        source_scope="kb_plus_arxiv_plus_repo",
        top_idea={
            "title": "Vectorization regression",
            "hypothesis": "A pass ordering issue regressed vectorization",
        },
        docs=[],
        repo_documents=[],
        papers=[],
        repo_source_ids=["repo-source-1"],
        benchmark_queries=["compile time regression"],
        automation_profile="balanced",
        automation_policy={
            "follow_up_review_mode": "queue_for_approval",
            "auto_execute_validation_runs": False,
        },
        sandbox_profile_id="scientific-compiler-sandbox",
        profile_id="profile-compiler-1",
    )

    assert child is not None
    assert child.config["automation_profile"] == "balanced"
    assert (
        child.config["automation_policy"]["follow_up_review_mode"]
        == "queue_for_approval"
    )
    assert child.config["sandbox_profile_id"] == "scientific-compiler-sandbox"
    assert child.config["profile_id"] == "profile-compiler-1"
    assert (
        child.config["validation_policy"]["follow_up_review_mode"]
        == "queue_for_approval"
    )
    assert (
        child.config["domain_research_follow_up"]["sandbox_profile_id"]
        == "scientific-compiler-sandbox"
    )


def test_build_action_for_tool_inherits_project_source_scope():
    executor = AutonomousAgentExecutor()
    scoped_source = str(uuid4())
    job = _make_job(config={"target_source_id": scoped_source})

    action = executor._build_action_for_tool(
        "search_documents", job, purpose="scope-aware recovery"
    )

    assert action is not None
    assert action["tool"] == "search_documents"
    assert action["params"]["source_id"] == scoped_source


def test_build_thinking_prompt_includes_project_scope_guidance():
    executor = AutonomousAgentExecutor()
    scoped_source = str(uuid4())
    job = _make_job(config={"source_id": scoped_source})
    state = _make_state()
    observation = {"iteration": 1, "context": []}

    prompt = executor._build_thinking_prompt(
        job, agent_def=None, state=state, observation=observation
    )

    assert "PROJECT SCOPE" in prompt
    assert scoped_source in prompt


def test_infer_project_profile_from_paths_detects_stack_and_commands():
    paths = [
        "frontend/package.json",
        "frontend/src/App.tsx",
        "frontend/src/components/Button.tsx",
        "backend/pyproject.toml",
        "backend/app/main.py",
        "backend/tests/test_agent.py",
        "docker-compose.yml",
    ]

    profile = infer_project_profile_from_paths(paths)

    stacks = profile.get("detected_stack") or []
    commands = profile.get("suggested_commands") or []
    assert "node" in stacks or "typescript" in stacks
    assert "python" in stacks
    assert any("npm" in str(cmd) for cmd in commands)
    assert any("pytest" in str(cmd) for cmd in commands)


def test_build_thinking_prompt_includes_project_profile_context():
    executor = AutonomousAgentExecutor()
    job = _make_job(config={"source_id": str(uuid4())})
    state = _make_state()
    state["project_profile"] = {
        "source_name": "Repo Source",
        "source_type": "gitlab",
        "detected_stack": ["python", "node"],
        "suggested_commands": ["python -m pytest -q", "npm test"],
        "command_groups": {
            "install": ["npm install"],
            "test": ["python -m pytest -q"],
            "test_fallback": ["python3 -m pytest -q"],
        },
        "bootstrap_notes": ["Install dependencies before running tests."],
        "marker_files": ["backend/pyproject.toml", "frontend/package.json"],
        "test_paths": ["backend/tests/test_agent.py"],
    }
    observation = {"iteration": 1, "context": []}

    prompt = executor._build_thinking_prompt(
        job, agent_def=None, state=state, observation=observation
    )

    assert "PROJECT PROFILE" in prompt
    assert "Detected stack" in prompt
    assert "Suggested commands" in prompt
    assert "Preferred verification" in prompt
    assert "Verification fallback" in prompt
    assert "Bootstrap notes" in prompt


@pytest.mark.asyncio
async def test_execute_job_dispatches_known_deterministic_runner_without_autonomous_loop(
    db_session,
):
    executor = AutonomousAgentExecutor()
    job = _make_job(config={"deterministic_runner": "ai_hub_scientist"})
    job.status = AgentJobStatus.PENDING.value
    db_session.add(job)
    await db_session.commit()

    executor._workspace_manager = SimpleNamespace(
        _workspaces={},
        persist_workspace=AsyncMock(return_value={}),
        cleanup_all=lambda: None,
    )
    executor._load_user_settings = AsyncMock(return_value=None)
    executor._run_autonomous_loop = AsyncMock(
        return_value={"status": "completed", "path": "loop"}
    )
    executor._trigger_chained_jobs = AsyncMock()
    executor.deterministic_runner_registry = SimpleNamespace(
        try_execute=AsyncMock(
            return_value=(True, {"status": "completed", "path": "deterministic"})
        )
    )

    result = await executor.execute_job(job.id, db_session)

    assert result == {"status": "completed", "path": "deterministic"}
    executor._run_autonomous_loop.assert_not_awaited()
    executor._trigger_chained_jobs.assert_awaited_once()


@pytest.mark.asyncio
async def test_execute_job_falls_back_to_autonomous_loop_when_no_deterministic_runner_matches(
    db_session,
):
    executor = AutonomousAgentExecutor()
    job = _make_job(config={"deterministic_runner": "unknown_runner"})
    job.status = AgentJobStatus.PENDING.value
    db_session.add(job)
    await db_session.commit()

    executor._workspace_manager = SimpleNamespace(
        _workspaces={},
        persist_workspace=AsyncMock(return_value={}),
        cleanup_all=lambda: None,
    )
    executor._load_user_settings = AsyncMock(return_value=None)
    executor._run_autonomous_loop = AsyncMock(
        return_value={"status": "completed", "path": "loop"}
    )
    executor._trigger_chained_jobs = AsyncMock()
    executor.deterministic_runner_registry = SimpleNamespace(
        try_execute=AsyncMock(return_value=(False, None))
    )

    result = await executor.execute_job(job.id, db_session)

    assert result == {"status": "completed", "path": "loop"}
    executor._run_autonomous_loop.assert_awaited_once()
    executor._trigger_chained_jobs.assert_not_awaited()


def test_infer_project_profile_from_paths_scopes_commands_for_nested_repo_dirs():
    paths = [
        "frontend/package.json",
        "frontend/src/App.tsx",
        "frontend/src/__tests__/App.test.tsx",
        "backend/pyproject.toml",
        "backend/app/main.py",
        "backend/tests/test_agent.py",
        "Makefile",
    ]

    profile = infer_project_profile_from_paths(paths)

    commands = profile.get("suggested_commands") or []
    command_groups = profile.get("command_groups") or {}
    assert "CI=true npm --prefix frontend test -- --watchAll=false" in commands
    assert "python -m pytest -q backend/tests" in commands
    assert "make test" in commands
    assert "npm test" not in commands
    assert "npm --prefix frontend install" in (command_groups.get("install") or [])
    assert "python3 -m pytest -q backend/tests" in (
        command_groups.get("test_fallback") or []
    )


def test_infer_project_profile_from_paths_uses_poetry_and_yarn_fallbacks():
    paths = [
        "frontend/package.json",
        "frontend/yarn.lock",
        "frontend/src/App.tsx",
        "frontend/src/__tests__/App.test.tsx",
        "backend/pyproject.toml",
        "backend/poetry.lock",
        "backend/app/main.py",
        "backend/tests/test_agent.py",
    ]

    profile = infer_project_profile_from_paths(paths)

    command_groups = profile.get("command_groups") or {}
    assert "cd frontend && yarn install" in (command_groups.get("install") or [])
    assert "cd frontend && CI=true yarn test --watchAll=false" in (
        command_groups.get("test") or []
    )
    assert "cd backend && poetry install" in (command_groups.get("install") or [])
    assert "poetry run pytest -q backend/tests" in (command_groups.get("test") or [])
    assert "python -m pytest -q backend/tests" in (
        command_groups.get("test_fallback") or []
    )


@pytest.mark.asyncio
async def test_domain_research_orchestrator_creates_structured_memo_and_experiment_drafts(
    db_session, test_user, monkeypatch
):
    queued_jobs: list[tuple[str, str]] = []

    monkeypatch.setattr(
        "app.tasks.agent_job_tasks.execute_agent_job_task.delay",
        lambda job_id, user_id: queued_jobs.append((job_id, user_id)),
    )

    source = DocumentSource(
        name="Research Source",
        source_type="file",
        config={},
        is_active=True,
    )
    db_session.add(source)
    await db_session.flush()

    doc1 = Document(
        title="LLVM Pass Fusion Study",
        content="Fusion can reduce pipeline pressure and improve throughput.",
        content_hash="doc1",
        source_id=source.id,
        source_identifier="doc1",
        summary="Compiler pass fusion reduces overhead in hot loops.",
    )
    doc2 = Document(
        title="Branch Predictor Notes",
        content="Speculative misses dominate on irregular branch-heavy code.",
        content_hash="doc2",
        source_id=source.id,
        source_identifier="doc2",
        summary="Branch behavior limits IPC in branch-heavy kernels.",
    )
    repo_source = DocumentSource(
        name="Compiler Repo",
        source_type="github",
        config={},
        is_active=True,
    )
    db_session.add(repo_source)
    await db_session.flush()
    repo_doc = Document(
        title="vectorizer.cpp hotspot notes",
        content="Compile time regression appears in vectorizer scheduling and codegen handoff.",
        content_hash="repo-doc",
        source_id=repo_source.id,
        source_identifier="repo-doc",
        summary="Vectorizer scheduling hotspot tied to compile time regression.",
        file_path="llvm/lib/Transforms/Vectorize/vectorizer.cpp",
    )
    repo_pyproject = Document(
        title="pyproject.toml",
        content='[tool.pytest.ini_options]\naddopts = "-q"\n',
        content_hash="repo-pyproject",
        source_id=repo_source.id,
        source_identifier="pyproject.toml",
        summary="Pytest configuration for compiler validation repo.",
        file_path="pyproject.toml",
    )
    repo_test_doc = Document(
        title="tests/test_vectorizer.py",
        content="def test_vectorizer_regression():\n    assert True\n",
        content_hash="repo-test-doc",
        source_id=repo_source.id,
        source_identifier="tests/test_vectorizer.py",
        summary="Regression test entrypoint for vectorizer validation.",
        file_path="tests/test_vectorizer.py",
    )
    profile = DomainResearchProfile(
        user_id=test_user.id,
        title="Compiler Frontier",
        domain="Compiler and microarchitecture",
        objective="Find novel, testable optimization directions",
        status="running",
        automation_profile="max_autonomy",
        automation_policy={
            "follow_up_review_mode": "auto_launch_safe",
            "auto_launch_experiment_runs": True,
            "auto_execute_validation_runs": True,
            "experiment_readiness_threshold": 0.72,
        },
        source_scope="kb_plus_arxiv_plus_repo",
        track_type="compiler",
        research_mode="literature_to_hypothesis",
        monitor_queries=["llvm fusion", "branch predictor optimization"],
        repo_source_ids=[str(repo_source.id)],
        benchmark_queries=["compile time regression"],
        sandbox_profile_id="scientific-compiler-sandbox",
        validation_policy={
            "confidence_threshold": 0.68,
            "experiment_readiness_threshold": 0.72,
            "auto_execute_validation_runs": True,
            "auto_launch_experiment_runs": True,
        },
        scoring_policy={"minimum_subscore": 0.6, "minimum_supporting_sources": 2},
        selection_policy={"max_candidates": 8, "max_hypotheses": 2},
        latest_summary={"ranked_opportunities": ["Known stale idea"]},
    )
    job = _make_job(
        config={
            "profile_id": str(profile.id),
            "domain": profile.domain,
            "objective": profile.objective,
            "source_scope": profile.source_scope,
            "track_type": profile.track_type,
            "research_mode": profile.research_mode,
            "monitor_queries": profile.monitor_queries,
            "repo_source_ids": profile.repo_source_ids,
            "benchmark_queries": profile.benchmark_queries,
            "sandbox_profile_id": profile.sandbox_profile_id,
            "persist_artifacts": True,
            "auto_launch_follow_up": False,
            "auto_create_experiment_plans": True,
            "confidence_threshold": 0.7,
            "validation_policy": profile.validation_policy,
            "scoring_policy": profile.scoring_policy,
            "selection_policy": profile.selection_policy,
        }
    )
    job.user_id = test_user.id

    db_session.add_all(
        [doc1, doc2, repo_doc, repo_pyproject, repo_test_doc, profile, job]
    )
    await db_session.commit()
    await db_session.refresh(job)
    await db_session.refresh(profile)
    await db_session.refresh(doc1)
    await db_session.refresh(doc2)

    executor = AutonomousAgentExecutor()

    async def _fake_search(**kwargs):
        if kwargs.get("source_id") == str(repo_source.id):
            return (
                [
                    {
                        "id": str(repo_doc.id),
                        "title": repo_doc.title,
                        "summary": repo_doc.summary,
                        "snippet": repo_doc.content,
                        "source": repo_source.name,
                    }
                ],
                1,
                0.01,
            )
        return (
            [
                {"id": str(doc1.id), "title": doc1.title, "summary": doc1.summary},
                {"id": str(doc2.id), "title": doc2.title, "summary": doc2.summary},
            ],
            2,
            0.01,
        )

    async def _fake_arxiv_search(**kwargs):
        return SimpleNamespace(
            items=[
                {
                    "id": "arxiv:2501.12345",
                    "title": "Compiler Branch Scheduling on Modern CPUs",
                    "summary": "Scheduling around branch predictor behavior improves front-end efficiency.",
                    "published": "2026-03-20T00:00:00Z",
                }
            ]
        )

    async def _fake_generate_response(**kwargs):
        return """
        {
          "domain_summary": "New compiler and microarch evidence suggests a testable scheduling direction.",
          "discovered_signals": ["pass fusion", "branch predictor sensitivity"],
          "proposed_ideas": [
            {
              "title": "Speculation-aware pass fusion ordering",
              "hypothesis": "Fusion should be prioritized for branch-heavy kernels where front-end pressure dominates.",
              "opportunity": "Reduce front-end stalls by reordering fusion decisions with branch behavior in mind.",
              "supporting_evidence": ["LLVM Pass Fusion Study", "Branch Predictor Notes", "Compiler Branch Scheduling on Modern CPUs", "vectorizer.cpp hotspot notes"],
              "confidence": 0.86,
              "next_steps": ["Benchmark on branch-heavy kernels", "Measure branch misses and IPC delta"],
              "counterarguments": ["May overfit to branch-heavy workloads"]
            },
            {
              "title": "Known stale idea",
              "hypothesis": "Repeat the previous baseline comparison.",
              "opportunity": "Re-run a known study with minor changes.",
              "supporting_evidence": ["LLVM Pass Fusion Study"],
              "confidence": 0.75,
              "next_steps": ["Re-run the old experiment"]
            }
          ],
          "ranked_opportunities": ["Speculation-aware pass fusion ordering"],
          "open_questions": ["How stable is the effect across different predictor designs?"],
          "brief_markdown": "",
          "report_markdown": ""
        }
        """

    executor.search_service.search = _fake_search
    executor.arxiv_search_service.search = _fake_arxiv_search
    executor.llm_service.generate_response = _fake_generate_response

    result = await executor._run_domain_research_orchestrator(
        job=job, db=db_session, progress_callback=None
    )

    assert result["status"] == "completed"
    await db_session.refresh(job)
    await db_session.refresh(profile)

    notes = list(
        (
            await db_session.execute(
                select(ResearchNote).where(ResearchNote.user_id == test_user.id)
            )
        )
        .scalars()
        .all()
    )
    assert notes
    memo_note = next(
        note for note in notes if isinstance(note.structured_payload, dict)
    )
    assert memo_note.structured_payload["research_mode"] == "literature_to_hypothesis"
    assert memo_note.structured_payload["track_type"] == "compiler"
    assert len(memo_note.structured_payload["evidence_snapshot"]["repo_documents"]) == 1
    assert len(memo_note.structured_payload["hypotheses"]) == 1
    assert (
        memo_note.structured_payload["hypotheses"][0]["title"]
        == "Speculation-aware pass fusion ordering"
    )
    assert memo_note.structured_payload["hypotheses"][0]["track_fit_score"] > 0

    plans = list(
        (
            await db_session.execute(
                select(ExperimentPlan).where(ExperimentPlan.user_id == test_user.id)
            )
        )
        .scalars()
        .all()
    )
    assert len(plans) == 1
    assert plans[0].generator_details["source_hypothesis_id"] == "idea_1"
    assert plans[0].generator_details["generation_reason"] == "autonomous_research_memo"
    assert str(repo_source.id) in plans[0].generator_details["source_repo_ids"]

    from app.models.experiment import ExperimentRun

    runs = list(
        (
            await db_session.execute(
                select(ExperimentRun).where(ExperimentRun.user_id == test_user.id)
            )
        )
        .scalars()
        .all()
    )
    assert len(runs) == 1
    assert runs[0].status == "queued"
    assert (
        runs[0].config["scientific_validation"]["recipe_family"]
        == "compiler_validation"
    )
    assert (
        runs[0].config["scientific_validation"]["recipe_id"] == "compiler_validation_v1"
    )
    assert runs[0].config["scientific_validation"]["recipe_version"] == 1
    assert (
        runs[0].config["scientific_validation"]["sandbox_profile_id"]
        == "scientific-compiler-sandbox"
    )
    assert runs[0].config["scientific_validation"]["domain_research_profile_id"] == str(
        profile.id
    )
    assert runs[0].config["scientific_validation"]["blocked_reason_code"] is None
    assert runs[0].config["scientific_validation"]["capability_check"]["ok"] is True
    assert (
        runs[0].config["scientific_validation"]["profile_snapshot"]["id"]
        == "scientific-compiler-sandbox"
    )
    assert (
        runs[0].config["scientific_validation"]["profile_snapshot"]["created_at"]
        is not None
    )
    assert (
        runs[0].config["scientific_validation"]["recipe_snapshot"]["recipe_id"]
        == "compiler_validation_v1"
    )
    recipe_commands = runs[0].config["scientific_validation"]["recipe_snapshot"][
        "commands"
    ]
    assert recipe_commands
    assert any(
        str(command).startswith("python -m pytest -q tests")
        or str(command).startswith("pytest -q tests")
        for command in recipe_commands
    )
    assert runs[0].agent_job_id is not None
    assert queued_jobs and queued_jobs[0][1] == str(test_user.id)

    inbox_items = list(
        (
            await db_session.execute(
                select(ResearchInboxItem).where(
                    ResearchInboxItem.user_id == test_user.id
                )
            )
        )
        .scalars()
        .all()
    )
    assert len(inbox_items) == 1
    assert inbox_items[0].item_type == "hypothesis_memo"
    assert profile.latest_summary["review_item_id"] == str(inbox_items[0].id)
    assert profile.latest_summary["track_type"] == "compiler"
    assert (
        profile.latest_summary["effective_policy"]["follow_up_review_mode"]
        == "queue_for_approval"
    )
    assert profile.latest_summary["profile_config_revision"]
    assert "autonomy_state_counts" in profile.latest_summary
    assert profile.latest_summary["evidence_mix"]["repo_documents"] == 1
    assert profile.latest_validation_run_ids == [str(runs[0].id)]


def test_merge_operator_fields_preserves_domain_operator_state_and_appends_evidence():
    previous = normalize_research_opportunity(
        {
            "opportunity_id": "opp_sched",
            "canonical_key": "speculation_aware_pass_fusion_ordering",
            "title": "Speculation-aware pass fusion ordering",
            "decision_state": "suppressed",
            "decision_source": "operator",
            "operator_note": "Already triaged",
            "linked_experiment_plan_ids": ["plan-existing"],
            "linked_validation_run_ids": ["run-existing"],
            "child_job_ids": ["job-existing"],
            "supporting_evidence": ["Earlier evidence"],
            "next_steps": ["Review later"],
        }
    )
    current = normalize_research_opportunity(
        {
            "opportunity_id": "opp_sched",
            "canonical_key": "speculation_aware_pass_fusion_ordering",
            "title": "Speculation-aware pass fusion ordering",
            "hypothesis": "Updated fusion order for branch-heavy kernels.",
            "supporting_evidence": ["Fresh branch-heavy benchmark signal"],
            "next_steps": ["Benchmark branch-heavy kernels"],
            "source_job_ids": ["job-current"],
        }
    )

    merged = merge_operator_fields(current, previous)

    assert merged["decision_state"] == "suppressed"
    assert merged["decision_source"] == "operator"
    assert merged["operator_note"] == "Already triaged"
    assert merged["linked_experiment_plan_ids"] == ["plan-existing"]
    assert merged["linked_validation_run_ids"] == ["run-existing"]
    assert merged["child_job_ids"] == ["job-existing"]
    assert "Earlier evidence" in (merged.get("supporting_evidence") or [])
    assert "Fresh branch-heavy benchmark signal" in (
        merged.get("supporting_evidence") or []
    )
    assert "Review later" in (merged.get("next_steps") or [])
    assert "Benchmark branch-heavy kernels" in (merged.get("next_steps") or [])


@pytest.mark.asyncio
async def test_research_fleet_orchestrator_preserves_suppressed_operator_state_and_skips_auto_actions(
    db_session, test_user, monkeypatch
):
    note = ResearchNote(
        user_id=test_user.id,
        title="Portfolio Source Note",
        content_markdown="Research note for suppressed portfolio opportunity.",
    )
    profile = DomainResearchProfile(
        user_id=test_user.id,
        title="Compiler Frontier",
        domain="Compiler",
        objective="Find compiler opportunities",
        status="completed",
        source_scope="kb_plus_arxiv_plus_repo",
        track_type="compiler",
        research_mode="literature_to_hypothesis",
        latest_note_ids=[],
        latest_summary={
            "opportunities": [
                {
                    "opportunity_id": "opp_profile_portfolio",
                    "canonical_key": "compiler_hotspot",
                    "title": "Compiler hotspot",
                    "hypothesis": "Fresh evidence from a rerun",
                    "confidence": 0.96,
                    "novelty": 0.92,
                    "supporting_evidence": ["Fresh profile evidence"],
                    "next_steps": ["Run the new benchmark slice"],
                }
            ]
        },
    )
    portfolio = ResearchPortfolio(
        user_id=test_user.id,
        title="Scientific Fleet",
        objective="Track and validate scientific opportunities",
        status="running",
        linked_profile_ids=[],
        automation_policy={
            "auto_create_experiment_plans": True,
            "auto_launch_experiment_runs": True,
            "auto_launch_follow_up": True,
            "confidence_threshold": 0.2,
            "experiment_readiness_threshold": 0.2,
            "max_auto_follow_up_launches": 2,
        },
        sandbox_profile_id="scientific-generic-sandbox",
        opportunities=[
            {
                "opportunity_id": "opp_compiler_hotspot",
                "canonical_key": "compiler_hotspot",
                "title": "Compiler hotspot",
                "hypothesis": "Prior operator-reviewed opportunity",
                "decision_state": "suppressed",
                "decision_source": "operator",
                "operator_note": "Do not promote automatically",
                "linked_experiment_plan_ids": ["plan-existing"],
                "linked_validation_run_ids": ["run-existing"],
                "child_job_ids": ["job-existing"],
                "supporting_evidence": ["Earlier evidence"],
            }
        ],
        latest_summary={},
        latest_note_ids=[],
        latest_experiment_plan_ids=["plan-existing"],
        latest_validation_run_ids=["run-existing"],
        child_job_ids=["job-existing"],
    )
    db_session.add(note)
    await db_session.flush()
    profile.latest_note_ids = [str(note.id)]
    portfolio.linked_profile_ids = [str(profile.id)]
    db_session.add_all([profile, portfolio])
    await db_session.flush()
    job = _make_job(
        config={
            "research_portfolio_id": str(portfolio.id),
            "linked_profile_ids": [str(profile.id)],
            "automation_policy": portfolio.automation_policy,
            "sandbox_profile_id": portfolio.sandbox_profile_id,
        }
    )
    job.user_id = test_user.id
    db_session.add(job)
    await db_session.commit()
    await db_session.refresh(job)
    await db_session.refresh(portfolio)

    async def _unexpected_validation(*args, **kwargs):
        raise AssertionError(
            "suppressed opportunities should not auto-launch validation"
        )

    async def _unexpected_follow_up(*args, **kwargs):
        raise AssertionError(
            "suppressed opportunities should not auto-launch follow-up jobs"
        )

    monkeypatch.setattr(
        executor := AutonomousAgentExecutor(),
        "_create_scientific_validation_run",
        _unexpected_validation,
    )
    monkeypatch.setattr(
        executor, "_create_domain_research_follow_up_job", _unexpected_follow_up
    )

    result = await executor._run_research_fleet_orchestrator(
        job=job, db=db_session, progress_callback=None
    )

    assert result["status"] == "completed"
    await db_session.refresh(portfolio)

    opportunity = next(
        row
        for row in (portfolio.opportunities or [])
        if str((row or {}).get("canonical_key") or "") == "compiler_hotspot"
    )
    assert opportunity["decision_state"] == "suppressed"
    assert opportunity["decision_source"] == "operator"
    assert opportunity["operator_note"] == "Do not promote automatically"
    assert opportunity["linked_experiment_plan_ids"] == ["plan-existing"]
    assert opportunity["linked_validation_run_ids"] == ["run-existing"]
    assert opportunity["child_job_ids"] == ["job-existing"]
    assert str(profile.id) in (opportunity.get("source_profile_ids") or [])
    assert "Fresh profile evidence" in (opportunity.get("supporting_evidence") or [])
    assert portfolio.latest_summary["autonomy_mode"] == "balanced"
    assert (
        portfolio.latest_summary["autonomy_summary"]["suppressed_duplicates_count"] >= 0
    )
    assert portfolio.latest_summary["stage_counts"]["suppressed"] == 1


@pytest.mark.asyncio
async def test_research_fleet_orchestrator_does_not_duplicate_existing_runs_or_follow_up_jobs(
    db_session, test_user, monkeypatch
):
    note = ResearchNote(
        user_id=test_user.id,
        title="Accepted Portfolio Source Note",
        content_markdown="Research note for accepted portfolio opportunity.",
    )
    profile = DomainResearchProfile(
        user_id=test_user.id,
        title="Compiler Frontier",
        domain="Compiler",
        objective="Find compiler opportunities",
        status="completed",
        source_scope="kb_plus_arxiv_plus_repo",
        track_type="compiler",
        research_mode="literature_to_hypothesis",
        latest_note_ids=[],
        latest_summary={
            "opportunities": [
                {
                    "opportunity_id": "opp_profile_existing",
                    "canonical_key": "accepted_compiler_hotspot",
                    "title": "Accepted compiler hotspot",
                    "hypothesis": "Fresh evidence from a rerun",
                    "confidence": 0.97,
                    "novelty": 0.94,
                    "supporting_evidence": ["Accepted profile evidence"],
                    "next_steps": ["Keep tracking the benchmark delta"],
                }
            ]
        },
    )
    portfolio = ResearchPortfolio(
        user_id=test_user.id,
        title="Scientific Fleet",
        objective="Track and validate scientific opportunities",
        status="running",
        linked_profile_ids=[],
        automation_policy={
            "auto_create_experiment_plans": True,
            "auto_launch_experiment_runs": True,
            "auto_launch_follow_up": True,
            "confidence_threshold": 0.2,
            "experiment_readiness_threshold": 0.2,
            "max_auto_follow_up_launches": 2,
        },
        sandbox_profile_id="scientific-generic-sandbox",
        opportunities=[
            {
                "opportunity_id": "opp_existing_portfolio",
                "canonical_key": "accepted_compiler_hotspot",
                "title": "Accepted compiler hotspot",
                "hypothesis": "Prior accepted opportunity",
                "decision_state": "accepted",
                "decision_source": "operator",
                "linked_experiment_plan_ids": ["plan-existing"],
                "linked_validation_run_ids": ["run-existing"],
                "child_job_ids": ["job-existing"],
                "supporting_evidence": ["Earlier evidence"],
            }
        ],
        latest_summary={},
        latest_note_ids=[],
        latest_experiment_plan_ids=["plan-existing"],
        latest_validation_run_ids=["run-existing"],
        child_job_ids=["job-existing"],
    )
    db_session.add(note)
    await db_session.flush()
    profile.latest_note_ids = [str(note.id)]
    portfolio.linked_profile_ids = [str(profile.id)]
    db_session.add_all([profile, portfolio])
    await db_session.flush()
    job = _make_job(
        config={
            "research_portfolio_id": str(portfolio.id),
            "linked_profile_ids": [str(profile.id)],
            "automation_policy": portfolio.automation_policy,
            "sandbox_profile_id": portfolio.sandbox_profile_id,
        }
    )
    job.user_id = test_user.id
    db_session.add(job)
    await db_session.commit()
    await db_session.refresh(job)
    await db_session.refresh(portfolio)

    async def _unexpected_validation(*args, **kwargs):
        raise AssertionError(
            "accepted opportunities with linked runs should not auto-launch duplicate validation"
        )

    async def _unexpected_follow_up(*args, **kwargs):
        raise AssertionError(
            "accepted opportunities with child jobs should not auto-launch duplicate follow-up jobs"
        )

    executor = AutonomousAgentExecutor()
    monkeypatch.setattr(
        executor, "_create_scientific_validation_run", _unexpected_validation
    )
    monkeypatch.setattr(
        executor, "_create_domain_research_follow_up_job", _unexpected_follow_up
    )

    result = await executor._run_research_fleet_orchestrator(
        job=job, db=db_session, progress_callback=None
    )

    assert result["status"] == "completed"
    await db_session.refresh(portfolio)

    opportunity = next(
        row
        for row in (portfolio.opportunities or [])
        if str((row or {}).get("canonical_key") or "") == "accepted_compiler_hotspot"
    )
    assert opportunity["decision_state"] == "accepted"
    assert opportunity["linked_experiment_plan_ids"] == ["plan-existing"]
    assert opportunity["linked_validation_run_ids"] == ["run-existing"]
    assert opportunity["child_job_ids"] == ["job-existing"]
    assert opportunity["stage"] == "validating"
    assert portfolio.latest_summary["autonomy_mode"] == "balanced"


@pytest.mark.asyncio
async def test_research_fleet_orchestrator_records_transient_validation_skip_without_blocking_opportunity(
    db_session, test_user, monkeypatch
):
    note = ResearchNote(
        user_id=test_user.id,
        title="Transient Skip Note",
        content_markdown="Research note for transient skip opportunity.",
    )
    profile = DomainResearchProfile(
        user_id=test_user.id,
        title="Retrieval Frontier",
        domain="Retrieval",
        objective="Find retrieval opportunities",
        status="completed",
        source_scope="kb_plus_arxiv_plus_repo",
        track_type="generic",
        research_mode="literature_to_hypothesis",
        latest_note_ids=[],
        latest_summary={
            "opportunities": [
                {
                    "opportunity_id": "opp_transient_skip",
                    "canonical_key": "retrieval_eval_gap",
                    "title": "Retrieval eval gap",
                    "hypothesis": "Need another run after cooldown",
                    "confidence": 0.95,
                    "novelty": 0.9,
                    "supporting_evidence": ["Fresh profile evidence"],
                    "next_steps": ["Retry after cooldown"],
                }
            ]
        },
    )
    portfolio = ResearchPortfolio(
        user_id=test_user.id,
        title="Scientific Fleet",
        objective="Track and validate scientific opportunities",
        status="running",
        linked_profile_ids=[],
        automation_policy={
            "auto_create_experiment_plans": True,
            "auto_launch_experiment_runs": True,
            "auto_launch_follow_up": False,
            "confidence_threshold": 0.2,
            "experiment_readiness_threshold": 0.2,
            "max_auto_follow_up_launches": 2,
        },
        sandbox_profile_id="scientific-generic-sandbox",
        opportunities=[],
        latest_summary={},
        latest_note_ids=[],
        latest_experiment_plan_ids=[],
        latest_validation_run_ids=[],
        child_job_ids=[],
    )
    db_session.add(note)
    await db_session.flush()
    profile.latest_note_ids = [str(note.id)]
    portfolio.linked_profile_ids = [str(profile.id)]
    db_session.add_all([profile, portfolio])
    await db_session.flush()
    job = _make_job(
        config={
            "research_portfolio_id": str(portfolio.id),
            "linked_profile_ids": [str(profile.id)],
            "automation_policy": portfolio.automation_policy,
            "sandbox_profile_id": portfolio.sandbox_profile_id,
        }
    )
    job.user_id = test_user.id
    db_session.add(job)
    await db_session.commit()
    await db_session.refresh(job)
    await db_session.refresh(portfolio)

    async def _blocked_validation(*args, **kwargs):
        return {
            "run_id": "run-transient",
            "status": "blocked",
            "reason_code": "backoff_cooldown",
            "job_id": None,
        }

    executor = AutonomousAgentExecutor()
    monkeypatch.setattr(
        executor, "_create_scientific_validation_run", _blocked_validation
    )

    result = await executor._run_research_fleet_orchestrator(
        job=job, db=db_session, progress_callback=None
    )

    assert result["status"] == "completed"
    await db_session.refresh(portfolio)
    opportunity = next(
        row
        for row in (portfolio.opportunities or [])
        if str((row or {}).get("canonical_key") or "") == "retrieval_eval_gap"
    )
    assert opportunity["stage"] in {"accepted", "planned"}
    assert opportunity["last_skip_reason_code"] == "backoff_cooldown"
    assert (opportunity.get("linked_validation_run_ids") or []) == []
    assert (
        portfolio.latest_summary["autonomy_summary"]["skipped_opportunities_count"] == 1
    )
    assert (
        portfolio.latest_summary["skipped_opportunities"][0]["reason_code"]
        == "backoff_cooldown"
    )


@pytest.mark.asyncio
async def test_research_fleet_orchestrator_keeps_completed_opportunity_idle_until_evidence_changes(
    db_session, test_user, monkeypatch
):
    profile = DomainResearchProfile(
        user_id=test_user.id,
        title="Compiler Frontier",
        domain="Compiler",
        objective="Track compiler opportunities",
        status="completed",
        source_scope="kb_plus_arxiv_plus_repo",
        track_type="generic",
        research_mode="literature_to_hypothesis",
        latest_summary={
            "opportunities": [
                {
                    "opportunity_id": "opp_completed_hold",
                    "canonical_key": "compiler_hotspot",
                    "title": "Compiler hotspot",
                    "hypothesis": "No change since the last successful validation",
                    "confidence": 0.94,
                    "novelty": 0.81,
                    "supporting_evidence": ["Stable evidence"],
                    "next_steps": ["Wait for more evidence"],
                }
            ]
        },
    )
    portfolio = ResearchPortfolio(
        user_id=test_user.id,
        title="Scientific Fleet",
        objective="Track and validate scientific opportunities",
        status="running",
        linked_profile_ids=[],
        automation_profile="max_autonomy",
        automation_policy={
            "auto_create_experiment_plans": True,
            "auto_launch_experiment_runs": True,
            "auto_launch_follow_up": False,
            "confidence_threshold": 0.2,
            "experiment_readiness_threshold": 0.2,
            "max_auto_follow_up_launches": 2,
        },
        sandbox_profile_id="scientific-generic-sandbox",
        opportunities=[
            {
                "opportunity_id": "opp_completed_hold",
                "canonical_key": "compiler_hotspot",
                "title": "Compiler hotspot",
                "hypothesis": "No change since the last successful validation",
                "confidence": 0.94,
                "novelty": 0.81,
                "readiness": 0.894,
                "supporting_evidence": ["Stable evidence"],
                "stage": "completed",
                "autonomy_state": "completed_waiting_change",
                "last_decision_reason_code": "completed_current_evidence",
            }
        ],
        latest_summary={},
        latest_note_ids=[],
        latest_experiment_plan_ids=[],
        latest_validation_run_ids=[],
        child_job_ids=[],
    )
    portfolio.linked_profile_ids = [str(profile.id)]
    db_session.add_all([profile, portfolio])
    await db_session.flush()
    job = _make_job(
        config={
            "research_portfolio_id": str(portfolio.id),
            "linked_profile_ids": [str(profile.id)],
            "automation_policy": portfolio.automation_policy,
            "sandbox_profile_id": portfolio.sandbox_profile_id,
        }
    )
    job.user_id = test_user.id
    db_session.add(job)
    await db_session.commit()

    async def _unexpected_validation(*args, **kwargs):
        raise AssertionError(
            "completed opportunities with unchanged evidence should not relaunch validation"
        )

    executor = AutonomousAgentExecutor()
    monkeypatch.setattr(
        executor, "_create_scientific_validation_run", _unexpected_validation
    )

    result = await executor._run_research_fleet_orchestrator(
        job=job, db=db_session, progress_callback=None
    )

    assert result["status"] == "completed"
    await db_session.refresh(portfolio)
    opportunity = next(
        row
        for row in (portfolio.opportunities or [])
        if str((row or {}).get("canonical_key") or "") == "compiler_hotspot"
    )
    assert opportunity["stage"] == "completed"
    assert opportunity["autonomy_state"] == "completed_waiting_change"
    assert opportunity["last_decision_reason_code"] == "completed_current_evidence"
    assert (
        portfolio.latest_summary["autonomy_state_counts"]["completed_waiting_change"]
        == 1
    )


@pytest.mark.asyncio
async def test_research_fleet_orchestrator_reopens_completed_opportunity_when_evidence_changes(
    db_session, test_user, monkeypatch
):
    note = ResearchNote(
        user_id=test_user.id,
        title="Changed Evidence Note",
        content_markdown="New evidence for a previously completed opportunity.",
    )
    profile = DomainResearchProfile(
        user_id=test_user.id,
        title="Compiler Frontier",
        domain="Compiler",
        objective="Track compiler opportunities",
        status="completed",
        source_scope="kb_plus_arxiv_plus_repo",
        track_type="generic",
        research_mode="literature_to_hypothesis",
        latest_note_ids=[],
        latest_summary={
            "opportunities": [
                {
                    "opportunity_id": "opp_completed_reopen",
                    "canonical_key": "compiler_hotspot",
                    "title": "Compiler hotspot",
                    "hypothesis": "New evidence justifies another validation",
                    "confidence": 0.95,
                    "novelty": 0.84,
                    "supporting_evidence": ["Fresh benchmark delta"],
                    "next_steps": ["Run another validation"],
                }
            ]
        },
    )
    portfolio = ResearchPortfolio(
        user_id=test_user.id,
        title="Scientific Fleet",
        objective="Track and validate scientific opportunities",
        status="running",
        linked_profile_ids=[],
        automation_profile="max_autonomy",
        automation_policy={
            "auto_create_experiment_plans": True,
            "auto_launch_experiment_runs": True,
            "auto_launch_follow_up": False,
            "confidence_threshold": 0.2,
            "experiment_readiness_threshold": 0.2,
            "max_auto_follow_up_launches": 2,
        },
        sandbox_profile_id="scientific-generic-sandbox",
        opportunities=[
            {
                "opportunity_id": "opp_completed_reopen",
                "canonical_key": "compiler_hotspot",
                "title": "Compiler hotspot",
                "hypothesis": "Old evidence was already validated",
                "confidence": 0.95,
                "novelty": 0.84,
                "readiness": 0.911,
                "supporting_evidence": ["Older benchmark delta"],
                "stage": "completed",
                "autonomy_state": "completed_waiting_change",
                "last_decision_reason_code": "completed_current_evidence",
            }
        ],
        latest_summary={},
        latest_note_ids=[],
        latest_experiment_plan_ids=[],
        latest_validation_run_ids=[],
        child_job_ids=[],
    )
    db_session.add(note)
    await db_session.flush()
    profile.latest_note_ids = [str(note.id)]
    portfolio.linked_profile_ids = [str(profile.id)]
    db_session.add_all([profile, portfolio])
    await db_session.flush()
    job = _make_job(
        config={
            "research_portfolio_id": str(portfolio.id),
            "linked_profile_ids": [str(profile.id)],
            "automation_policy": portfolio.automation_policy,
            "sandbox_profile_id": portfolio.sandbox_profile_id,
        }
    )
    job.user_id = test_user.id
    db_session.add(job)
    await db_session.commit()

    async def _validation(*args, **kwargs):
        return {
            "run_id": "run-reopened",
            "status": "queued",
            "reason_code": None,
            "job_id": "job-reopened",
        }

    executor = AutonomousAgentExecutor()
    monkeypatch.setattr(executor, "_create_scientific_validation_run", _validation)

    result = await executor._run_research_fleet_orchestrator(
        job=job, db=db_session, progress_callback=None
    )

    assert result["status"] == "completed"
    await db_session.refresh(portfolio)
    opportunity = next(
        row
        for row in (portfolio.opportunities or [])
        if str((row or {}).get("canonical_key") or "") == "compiler_hotspot"
    )
    assert opportunity["stage"] == "validating"
    assert opportunity["autonomy_state"] == "active"
    assert opportunity["last_decision_type"] in {
        "validation_run_queued",
        "experiment_plan_created",
    }
    assert "run-reopened" in (
        portfolio.latest_summary["launched_validation_run_ids"] or []
    )


@pytest.mark.asyncio
async def test_research_fleet_orchestrator_queues_follow_up_approval_without_auto_launch(
    db_session, test_user, monkeypatch
):
    note = ResearchNote(
        user_id=test_user.id,
        title="Queued Follow-up Note",
        content_markdown="Research note for queued follow-up approval.",
    )
    profile = DomainResearchProfile(
        user_id=test_user.id,
        title="Generic Frontier",
        domain="Generic",
        objective="Find generic opportunities",
        status="completed",
        source_scope="kb_plus_arxiv",
        track_type="generic",
        research_mode="literature_to_hypothesis",
        latest_note_ids=[],
        latest_summary={
            "opportunities": [
                {
                    "opportunity_id": "opp_follow_up_queue",
                    "canonical_key": "generic_hotspot",
                    "title": "Generic hotspot",
                    "hypothesis": "Queue this for approval",
                    "confidence": 0.97,
                    "novelty": 0.88,
                    "supporting_evidence": ["Fresh evidence"],
                }
            ]
        },
    )
    portfolio = ResearchPortfolio(
        user_id=test_user.id,
        title="Scientific Fleet",
        objective="Track and validate scientific opportunities",
        status="running",
        linked_profile_ids=[],
        automation_profile="max_autonomy",
        automation_policy={
            "auto_create_experiment_plans": False,
            "auto_launch_experiment_runs": False,
            "auto_launch_follow_up": True,
            "follow_up_review_mode": "queue_for_approval",
            "confidence_threshold": 0.2,
            "experiment_readiness_threshold": 0.2,
            "max_auto_follow_up_launches": 2,
        },
        sandbox_profile_id="scientific-generic-sandbox",
        opportunities=[],
        latest_summary={},
        latest_note_ids=[],
        latest_experiment_plan_ids=[],
        latest_validation_run_ids=[],
        child_job_ids=[],
    )
    db_session.add(note)
    await db_session.flush()
    profile.latest_note_ids = [str(note.id)]
    portfolio.linked_profile_ids = [str(profile.id)]
    db_session.add_all([profile, portfolio])
    await db_session.flush()
    job = _make_job(
        config={
            "research_portfolio_id": str(portfolio.id),
            "linked_profile_ids": [str(profile.id)],
            "automation_policy": portfolio.automation_policy,
            "sandbox_profile_id": portfolio.sandbox_profile_id,
        }
    )
    job.user_id = test_user.id
    job.schedule_type = "continuous"
    job.next_run_at = datetime.utcnow() + timedelta(minutes=60)
    db_session.add(job)
    await db_session.commit()

    async def _unexpected_follow_up(*args, **kwargs):
        raise AssertionError("queue_for_approval should not auto-launch follow-up jobs")

    executor = AutonomousAgentExecutor()
    monkeypatch.setattr(
        executor, "_create_domain_research_follow_up_job", _unexpected_follow_up
    )

    result = await executor._run_research_fleet_orchestrator(
        job=job, db=db_session, progress_callback=None
    )

    assert result["status"] == "completed"
    await db_session.refresh(portfolio)
    opportunity = portfolio.opportunities[0]
    assert opportunity["follow_up_review_status"] == "pending_approval"
    assert (
        opportunity["follow_up_review_evidence_revision"]
        == opportunity["evidence_revision"]
    )
    assert (opportunity.get("child_job_ids") or []) == []
    assert (
        portfolio.latest_summary["scheduler_summary"][
            "pending_follow_up_approvals_count"
        ]
        == 1
    )
    assert (
        portfolio.latest_summary["pending_follow_up_approvals"][0]["reason_code"]
        == "follow_up_pending_approval"
    )


@pytest.mark.asyncio
async def test_research_fleet_orchestrator_records_manual_follow_up_recommendation_without_launch(
    db_session, test_user, monkeypatch
):
    note = ResearchNote(
        user_id=test_user.id,
        title="Manual Follow-up Note",
        content_markdown="Research note for manual follow-up recommendation.",
    )
    profile = DomainResearchProfile(
        user_id=test_user.id,
        title="Generic Frontier",
        domain="Generic",
        objective="Find generic opportunities",
        status="completed",
        source_scope="kb_plus_arxiv",
        track_type="generic",
        research_mode="literature_to_hypothesis",
        latest_note_ids=[],
        latest_summary={
            "opportunities": [
                {
                    "opportunity_id": "opp_follow_up_manual",
                    "canonical_key": "manual_hotspot",
                    "title": "Manual hotspot",
                    "hypothesis": "Recommend manually",
                    "confidence": 0.96,
                    "novelty": 0.87,
                    "supporting_evidence": ["Fresh evidence"],
                }
            ]
        },
    )
    portfolio = ResearchPortfolio(
        user_id=test_user.id,
        title="Scientific Fleet",
        objective="Track and validate scientific opportunities",
        status="running",
        linked_profile_ids=[],
        automation_profile="max_autonomy",
        automation_policy={
            "auto_create_experiment_plans": False,
            "auto_launch_experiment_runs": False,
            "auto_launch_follow_up": True,
            "follow_up_review_mode": "manual_only",
            "confidence_threshold": 0.2,
            "experiment_readiness_threshold": 0.2,
            "max_auto_follow_up_launches": 2,
        },
        sandbox_profile_id="scientific-generic-sandbox",
        opportunities=[],
        latest_summary={},
        latest_note_ids=[],
        latest_experiment_plan_ids=[],
        latest_validation_run_ids=[],
        child_job_ids=[],
    )
    db_session.add(note)
    await db_session.flush()
    profile.latest_note_ids = [str(note.id)]
    portfolio.linked_profile_ids = [str(profile.id)]
    db_session.add_all([profile, portfolio])
    await db_session.flush()
    job = _make_job(
        config={
            "research_portfolio_id": str(portfolio.id),
            "linked_profile_ids": [str(profile.id)],
            "automation_policy": portfolio.automation_policy,
            "sandbox_profile_id": portfolio.sandbox_profile_id,
        }
    )
    job.user_id = test_user.id
    job.schedule_type = "continuous"
    job.next_run_at = datetime.utcnow() + timedelta(minutes=60)
    db_session.add(job)
    await db_session.commit()

    async def _unexpected_follow_up(*args, **kwargs):
        raise AssertionError("manual_only should not auto-launch follow-up jobs")

    executor = AutonomousAgentExecutor()
    monkeypatch.setattr(
        executor, "_create_domain_research_follow_up_job", _unexpected_follow_up
    )

    result = await executor._run_research_fleet_orchestrator(
        job=job, db=db_session, progress_callback=None
    )

    assert result["status"] == "completed"
    await db_session.refresh(portfolio)
    opportunity = portfolio.opportunities[0]
    assert opportunity["follow_up_review_status"] == "manual_recommendation"
    assert (opportunity.get("child_job_ids") or []) == []
    assert (
        portfolio.latest_summary["scheduler_summary"][
            "manual_follow_up_recommendations_count"
        ]
        == 1
    )
    assert (
        portfolio.latest_summary["manual_follow_up_recommendations"][0]["reason_code"]
        == "manual_follow_up_recommendation"
    )


@pytest.mark.asyncio
async def test_research_fleet_orchestrator_preserves_rejected_follow_up_for_same_evidence(
    db_session, test_user, monkeypatch
):
    profile = DomainResearchProfile(
        user_id=test_user.id,
        title="Generic Frontier",
        domain="Generic",
        objective="Find generic opportunities",
        status="completed",
        source_scope="kb_plus_arxiv",
        track_type="generic",
        research_mode="literature_to_hypothesis",
        latest_summary={
            "opportunities": [
                {
                    "opportunity_id": "opp_follow_up_rejected",
                    "canonical_key": "rejected_hotspot",
                    "title": "Rejected hotspot",
                    "hypothesis": "Do not relaunch this follow-up yet",
                    "confidence": 0.96,
                    "novelty": 0.84,
                    "supporting_evidence": ["Stable evidence"],
                }
            ]
        },
    )
    portfolio = ResearchPortfolio(
        user_id=test_user.id,
        title="Scientific Fleet",
        objective="Track and validate scientific opportunities",
        status="running",
        linked_profile_ids=[],
        automation_profile="max_autonomy",
        automation_policy={
            "auto_create_experiment_plans": False,
            "auto_launch_experiment_runs": False,
            "auto_launch_follow_up": True,
            "follow_up_review_mode": "auto_launch_safe",
            "confidence_threshold": 0.2,
            "experiment_readiness_threshold": 0.2,
            "max_auto_follow_up_launches": 2,
        },
        sandbox_profile_id="scientific-generic-sandbox",
        opportunities=[
            {
                "opportunity_id": "opp_follow_up_rejected",
                "canonical_key": "rejected_hotspot",
                "title": "Rejected hotspot",
                "hypothesis": "Do not relaunch this follow-up yet",
                "confidence": 0.96,
                "novelty": 0.84,
                "readiness": 0.918,
                "supporting_evidence": ["Stable evidence"],
                "follow_up_review_status": "rejected",
                "follow_up_review_evidence_revision": "will_be_replaced",
            }
        ],
        latest_summary={},
        latest_note_ids=[],
        latest_experiment_plan_ids=[],
        latest_validation_run_ids=[],
        child_job_ids=[],
    )
    portfolio.linked_profile_ids = [str(profile.id)]
    db_session.add_all([profile, portfolio])
    await db_session.flush()
    portfolio.opportunities = [
        normalize_research_opportunity(portfolio.opportunities[0])
    ]
    portfolio.opportunities[0]["follow_up_review_status"] = "rejected"
    portfolio.opportunities[0][
        "follow_up_review_evidence_revision"
    ] = portfolio.opportunities[0]["evidence_revision"]
    await db_session.commit()

    job = _make_job(
        config={
            "research_portfolio_id": str(portfolio.id),
            "linked_profile_ids": [str(profile.id)],
            "automation_policy": portfolio.automation_policy,
            "sandbox_profile_id": portfolio.sandbox_profile_id,
        }
    )
    job.user_id = test_user.id
    job.schedule_type = "continuous"
    job.next_run_at = datetime.utcnow() + timedelta(minutes=60)
    db_session.add(job)
    await db_session.commit()

    async def _unexpected_follow_up(*args, **kwargs):
        raise AssertionError(
            "rejected follow-up for unchanged evidence should not relaunch"
        )

    executor = AutonomousAgentExecutor()
    monkeypatch.setattr(
        executor, "_create_domain_research_follow_up_job", _unexpected_follow_up
    )

    result = await executor._run_research_fleet_orchestrator(
        job=job, db=db_session, progress_callback=None
    )

    assert result["status"] == "completed"
    await db_session.refresh(portfolio)
    opportunity = portfolio.opportunities[0]
    assert opportunity["follow_up_review_status"] == "rejected"
    assert opportunity["last_decision_reason_code"] == "operator_rejected_follow_up"
    assert (
        portfolio.latest_summary["scheduler_summary"]["suppressed_relaunches_count"]
        >= 1
    )
    assert (
        portfolio.latest_summary["suppressed_relaunches"][0]["reason_code"]
        == "operator_rejected_follow_up"
    )


def test_select_verification_commands_from_profile_prefers_test_commands():
    executor = AutonomousAgentExecutor()
    profile = {
        "detected_stack": ["python", "node"],
        "suggested_commands": [
            "npm install",
            "npm run build",
            "npm test",
            "python -m pytest -q",
        ],
    }

    commands = executor._select_verification_commands_from_profile(
        profile, max_commands=3
    )

    assert "npm test" in commands
    assert "python -m pytest -q" in commands
    assert "npm install" not in commands


def test_select_verification_commands_from_profile_preserves_nested_repo_commands():
    executor = AutonomousAgentExecutor()
    profile = {
        "detected_stack": ["python", "node"],
        "command_groups": {
            "test": [
                "CI=true npm --prefix frontend test -- --watchAll=false",
                "python -m pytest -q backend/tests",
            ],
            "test_fallback": [
                "python3 -m pytest -q backend/tests",
            ],
        },
        "suggested_commands": [
            "CI=true npm --prefix frontend test -- --watchAll=false",
            "python -m pytest -q backend/tests",
        ],
    }

    commands = executor._select_verification_commands_from_profile(
        profile, max_commands=3
    )

    assert commands == [
        "CI=true npm --prefix frontend test -- --watchAll=false",
        "python -m pytest -q backend/tests",
        "python3 -m pytest -q backend/tests",
    ]


def test_select_verification_commands_from_profile_falls_back_to_stack_defaults():
    executor = AutonomousAgentExecutor()
    profile = {
        "detected_stack": ["go", "dotnet"],
        "suggested_commands": ["go build ./...", "dotnet build"],
    }

    commands = executor._select_verification_commands_from_profile(
        profile, max_commands=2
    )

    assert commands == ["go test ./...", "dotnet test"]


def test_get_bootstrap_and_fallback_commands_from_profile_skips_primary_duplicates():
    executor = AutonomousAgentExecutor()
    profile = {
        "command_groups": {
            "install": [
                "npm --prefix frontend install",
                "cd backend && poetry install",
            ],
            "test_fallback": [
                "CI=true npm --prefix frontend test -- --watchAll=false",
                "python3 -m pytest -q backend/tests",
            ],
        }
    }

    commands = executor._get_bootstrap_and_fallback_commands_from_profile(
        profile,
        primary_commands=[
            "CI=true npm --prefix frontend test -- --watchAll=false",
            "python -m pytest -q backend/tests",
        ],
        max_install=2,
        max_fallback=3,
    )

    assert commands["install"] == [
        "npm --prefix frontend install",
        "cd backend && poetry install",
    ]
    assert commands["fallback"] == [
        "python3 -m pytest -q backend/tests",
    ]


def test_should_bootstrap_after_verification_failure_detects_missing_toolchain_signals():
    executor = AutonomousAgentExecutor()

    assert executor._should_bootstrap_after_verification_failure(
        {"ok": False, "exit_code": 127, "stderr": "/bin/sh: pytest: command not found"}
    )
    assert executor._should_bootstrap_after_verification_failure(
        {
            "ok": False,
            "exit_code": 1,
            "stderr": "ModuleNotFoundError: No module named 'fastapi'",
        }
    )
    assert not executor._should_bootstrap_after_verification_failure(
        {"ok": False, "exit_code": 1, "stderr": "AssertionError: expected 2 == 3"}
    )


def test_summarize_experiment_run_phases_prefers_latest_verification_phase():
    executor = AutonomousAgentExecutor()

    summary = executor._summarize_experiment_run_phases(
        [
            {"command": "npm test", "phase": "primary", "ok": False},
            {"command": "npm install", "phase": "bootstrap", "ok": True},
            {"command": "npm test", "phase": "retry_primary", "ok": True},
        ]
    )

    assert summary["phases"] == ["primary", "bootstrap", "retry_primary"]
    assert summary["verification_phases"] == ["primary", "retry_primary"]
    assert summary["final_phase"] == "retry_primary"
    assert summary["final_ok"] is True
    assert summary["failed_commands"] == ["npm test"]


def test_summarize_experiment_run_phases_tracks_fallback_failure():
    executor = AutonomousAgentExecutor()

    summary = executor._summarize_experiment_run_phases(
        [
            {"command": "pytest -q", "phase": "primary", "ok": False},
            {"command": "python3 -m pytest -q", "phase": "fallback", "ok": False},
        ]
    )

    assert summary["phases"] == ["primary", "fallback"]
    assert summary["verification_phases"] == ["primary", "fallback"]
    assert summary["final_phase"] == "fallback"
    assert summary["final_ok"] is False
    assert summary["failed_commands"] == ["pytest -q", "python3 -m pytest -q"]


def test_resolve_default_source_scope_from_inherited_parent_results():
    executor = AutonomousAgentExecutor()
    inherited_source = str(uuid4())
    job = _make_job(
        config={
            "inherited_data": {
                "parent_results": {"repo_ingest": {"source_id": inherited_source}}
            }
        }
    )

    scoped = executor._resolve_default_source_scope(job)

    assert scoped == inherited_source


def test_get_tools_for_job_type_includes_project_bootstrap():
    executor = AutonomousAgentExecutor()
    tools = executor._get_tools_for_job_type("research", {"source_id": str(uuid4())})
    assert "project_bootstrap" in tools


def test_resolve_scope_source_prefers_canonical_source_id():
    executor = AutonomousAgentExecutor()
    job = _make_job(
        config={"source_id": str(uuid4()), "target_source_id": str(uuid4())}
    )

    source = executor._resolve_scope_source(job)

    assert source == "config.source_id"


def test_resolve_scope_source_uses_inherited_repo_ingest():
    executor = AutonomousAgentExecutor()
    job = _make_job(
        config={
            "inherited_data": {
                "parent_results": {"repo_ingest": {"source_id": str(uuid4())}}
            }
        }
    )

    source = executor._resolve_scope_source(job)

    assert source == "inherited_data.parent_results.repo_ingest.source_id"


def test_append_scope_event_keeps_bounded_history():
    executor = AutonomousAgentExecutor()
    state = _make_state()
    for i in range(12):
        executor._append_scope_event(
            state, {"type": "resolved_scope", "iteration": i}, max_events=5
        )

    events = state.get("scope_events") or []
    assert len(events) == 5
    assert events[-1].get("iteration") == 11


def test_build_verification_action_for_created_document_uses_document_details():
    executor = AutonomousAgentExecutor()
    job = _make_job(config={"source_id": str(uuid4())})
    primary_action = {
        "tool": "create_document_from_text",
        "params": {"source_id": str(uuid4())},
    }
    primary_result = {"success": True, "data": {"document_id": str(uuid4())}}

    verification = executor._build_verification_action(
        job, primary_action, primary_result
    )

    assert verification is not None
    assert verification["tool"] == "get_document_details"
    assert "document_id" in (verification.get("params") or {})


def test_build_verification_action_for_saved_finding_uses_findings_query():
    executor = AutonomousAgentExecutor()
    job = _make_job(config={"source_id": str(uuid4())})
    primary_action = {
        "tool": "save_research_finding",
        "params": {"category": "key_insight"},
    }
    primary_result = {"success": True, "data": {"finding_id": str(uuid4())}}

    verification = executor._build_verification_action(
        job, primary_action, primary_result
    )

    assert verification is not None
    assert verification["tool"] == "get_research_findings"
    assert verification["params"].get("category") == "key_insight"


def test_build_summarize_action_emits_progress_report():
    executor = AutonomousAgentExecutor()
    job = _make_job()
    state = _make_state()
    primary_action = {"tool": "create_synthesis_document", "params": {}}
    primary_result = {"success": True}
    verification_action = {
        "tool": "get_document_details",
        "params": {"document_id": str(uuid4())},
    }
    verification_result = {"success": True}

    summary_action = executor._build_summarize_action(
        job,
        state,
        primary_action,
        primary_result,
        verification_action,
        verification_result,
    )

    assert summary_action is not None
    assert summary_action["tool"] == "write_progress_report"
    assert "summary" in (summary_action.get("params") or {})


def test_annotate_execution_plan_graph_adds_step_ids_and_dependencies():
    executor = AutonomousAgentExecutor()
    raw_plan = [
        {"title": "Step A", "objective": "Do A", "status": "pending"},
        {"title": "Step B", "objective": "Do B", "status": "pending"},
        {
            "title": "Step C",
            "objective": "Do C",
            "status": "pending",
            "depends_on": ["custom_prev"],
        },
    ]

    plan = executor._annotate_execution_plan_graph(raw_plan)

    assert len(plan) == 3
    assert plan[0].get("step_id") == "step_1"
    assert plan[0].get("depends_on") == []
    assert plan[1].get("step_id") == "step_2"
    assert plan[1].get("depends_on") == ["step_1"]
    assert plan[2].get("depends_on") == ["custom_prev"]
    assert all(str(x.get("node_type") or "") == "act" for x in plan)


def test_build_execution_graph_stats_reports_dag_shape_and_critical_path():
    executor = AutonomousAgentExecutor()
    nodes = [
        {"id": "step_1", "type": "act", "success": True},
        {"id": "step_1.verify", "type": "verify", "success": True},
        {"id": "step_1.summarize", "type": "summarize", "success": True},
        {"id": "step_2", "type": "act", "success": False},
    ]
    edges = [
        {"from": "step_1", "to": "step_1.verify", "type": "verify_after"},
        {"from": "step_1.verify", "to": "step_1.summarize", "type": "summarize_after"},
        {"from": "step_1.summarize", "to": "step_2", "type": "next_step"},
    ]

    stats = executor._build_execution_graph_stats(nodes, edges)

    assert stats["total_nodes"] == 4
    assert stats["total_edges"] == 3
    assert stats["has_cycle"] is False
    assert stats["critical_path_length"] == 4
    assert stats["blocked_nodes"] == 1
    assert stats["root_nodes"] == 1
    assert stats["leaf_nodes"] == 1


def test_build_execution_graph_health_marks_critical_on_cycle():
    executor = AutonomousAgentExecutor()
    health = executor._build_execution_graph_health(
        {
            "total_nodes": 4,
            "blocked_nodes": 1,
            "has_cycle": True,
            "critical_path_length": 4,
            "orphan_nodes": 0,
        }
    )

    assert health["status"] == "critical"
    assert "cycle_detected" in (health.get("reasons") or [])


def test_build_execution_graph_health_marks_warning_for_blocked_ratio():
    executor = AutonomousAgentExecutor()
    health = executor._build_execution_graph_health(
        {
            "total_nodes": 8,
            "blocked_nodes": 3,
            "has_cycle": False,
            "critical_path_length": 7,
            "orphan_nodes": 0,
        }
    )

    assert health["status"] in {"warning", "critical"}
    assert "blocked_ratio" in " ".join([str(x) for x in (health.get("reasons") or [])])


def test_build_execution_graph_health_marks_ok_for_clean_graph():
    executor = AutonomousAgentExecutor()
    health = executor._build_execution_graph_health(
        {
            "total_nodes": 6,
            "blocked_nodes": 0,
            "has_cycle": False,
            "critical_path_length": 5,
            "orphan_nodes": 0,
        }
    )

    assert health["status"] == "ok"
    assert health["severity_score"] < 20


def test_build_execution_graph_recommendations_for_cycle():
    executor = AutonomousAgentExecutor()
    health = {
        "status": "critical",
        "reasons": ["cycle_detected", "high_blocked_ratio"],
        "severity_score": 90,
    }

    recs = executor._build_execution_graph_recommendations(health)

    assert isinstance(recs, list)
    assert recs
    joined = " ".join(recs).lower()
    assert "cycle" in joined or "cyclic" in joined
    assert "blocked" in joined or "failed" in joined


def test_build_execution_graph_recommendations_for_ok_graph():
    executor = AutonomousAgentExecutor()
    health = {
        "status": "ok",
        "reasons": [],
        "severity_score": 0,
    }

    recs = executor._build_execution_graph_recommendations(health)

    assert isinstance(recs, list)
    assert len(recs) >= 1
    assert "stable" in recs[0].lower() or "continue" in recs[0].lower()


def test_get_execution_graph_runtime_snapshot_tracks_live_health():
    executor = AutonomousAgentExecutor()
    state = _make_state()
    state["execution_graph_nodes"] = [
        {"id": "step_1", "type": "act", "success": True},
        {"id": "step_1.verify", "type": "verify", "success": False},
    ]
    state["execution_graph_edges"] = [
        {"from": "step_1", "to": "step_1.verify", "type": "verify_after"},
    ]
    state["verification_attempts"] = 1
    state["verification_successes"] = 0

    runtime = executor._get_execution_graph_runtime_snapshot(state)

    assert runtime["verification_attempts"] == 1
    assert runtime["verification_successes"] == 0
    assert runtime["dag_stats"]["total_nodes"] == 2
    assert isinstance(runtime["graph_health"], dict)
    assert isinstance(runtime["recommended_actions"], list)


def test_build_thinking_prompt_includes_execution_graph_context():
    executor = AutonomousAgentExecutor()
    job = _make_job(config={"source_id": str(uuid4())})
    state = _make_state()
    state["execution_graph_runtime"] = {
        "verification_attempts": 3,
        "verification_successes": 1,
        "summarization_attempts": 2,
        "summarization_successes": 2,
        "dag_stats": {
            "total_nodes": 5,
            "total_edges": 4,
            "critical_path_length": 4,
        },
        "graph_health": {
            "status": "warning",
            "severity_score": 20,
            "reasons": ["moderate_blocked_ratio"],
        },
        "recommended_actions": [
            "Review failed/blocked nodes and tighten tool params before retrying affected steps.",
        ],
    }
    observation = {"iteration": 2, "context": []}

    prompt = executor._build_thinking_prompt(
        job, agent_def=None, state=state, observation=observation
    )

    assert "EXECUTION GRAPH" in prompt
    assert "moderate_blocked_ratio" in prompt
    assert "Review failed/blocked nodes" in prompt


def test_validate_action_scope_blocks_cross_source_write_by_default():
    executor = AutonomousAgentExecutor()
    default_source = str(uuid4())
    other_source = str(uuid4())
    job = _make_job(config={"source_id": default_source})
    action = {
        "tool": "create_document_from_text",
        "params": {"title": "Note", "content": "Body", "source_id": other_source},
    }

    error = executor._validate_action_scope(job, action)

    assert error is not None
    assert "Scope guard blocked cross-source write" in error


def test_validate_action_scope_allows_override_when_enabled():
    executor = AutonomousAgentExecutor()
    default_source = str(uuid4())
    other_source = str(uuid4())
    job = _make_job(
        config={"source_id": default_source, "scope_guard_allow_param_override": True}
    )
    action = {
        "tool": "save_research_finding",
        "params": {
            "title": "Finding",
            "content": "X",
            "source_id": other_source,
            "allow_cross_scope": True,
        },
    }

    error = executor._validate_action_scope(job, action)

    assert error is None


def test_validate_action_scope_allows_when_guard_disabled():
    executor = AutonomousAgentExecutor()
    default_source = str(uuid4())
    other_source = str(uuid4())
    job = _make_job(config={"source_id": default_source, "scope_guard_enabled": False})
    action = {
        "tool": "create_synthesis_document",
        "params": {"title": "S", "topic": "T", "source_id": other_source},
    }

    error = executor._validate_action_scope(job, action)

    assert error is None


def test_parse_decision_response_scopes_create_synthesis_document():
    executor = AutonomousAgentExecutor()
    scoped_source = str(uuid4())
    job = _make_job(config={"target_source_id": scoped_source})
    state = _make_state()
    available_tools = executor._get_tools_for_job_type(job.job_type, job.config)

    raw = """{
      "goal_achieved": false,
      "should_stop": false,
      "reasoning": "Synthesize scoped findings",
      "action": {"tool": "create_synthesis_document", "params": {"title": "Scoped Summary", "topic": "Retrieval"}}
    }"""
    decision = executor._parse_decision_response(raw, job, state, available_tools)

    assert decision["action"] is not None
    assert decision["action"]["tool"] == "create_synthesis_document"
    assert decision["action"]["params"]["source_id"] == scoped_source


def test_parse_decision_response_scopes_create_document_from_text():
    executor = AutonomousAgentExecutor()
    scoped_source = str(uuid4())
    job = _make_job(config={"source_id": scoped_source})
    state = _make_state()
    available_tools = executor._get_tools_for_job_type(job.job_type, job.config)

    raw = """{
      "goal_achieved": false,
      "should_stop": false,
      "reasoning": "Write scoped report",
      "action": {"tool": "create_document_from_text", "params": {"title": "Report", "content": "Scoped output"}}
    }"""
    decision = executor._parse_decision_response(raw, job, state, available_tools)

    assert decision["action"] is not None
    assert decision["action"]["tool"] == "create_document_from_text"
    assert decision["action"]["params"]["source_id"] == scoped_source


def test_parse_decision_response_scopes_save_research_finding():
    executor = AutonomousAgentExecutor()
    scoped_source = str(uuid4())
    job = _make_job(config={"target_source_id": scoped_source})
    state = _make_state()
    available_tools = executor._get_tools_for_job_type(job.job_type, job.config)

    raw = """{
      "goal_achieved": false,
      "should_stop": false,
      "reasoning": "Persist key evidence",
      "action": {"tool": "save_research_finding", "params": {"title": "Key insight", "content": "Important detail"}}
    }"""
    decision = executor._parse_decision_response(raw, job, state, available_tools)

    assert decision["action"] is not None
    assert decision["action"]["tool"] == "save_research_finding"
    assert decision["action"]["params"]["source_id"] == scoped_source


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


def test_update_stall_state_triggers_recovery_from_graph_verification_debt():
    executor = AutonomousAgentExecutor()
    job = _make_job(
        config={
            "stall_detection_enabled": True,
            "stall_min_progress_delta": 5,
            "stall_max_iterations_without_progress": 4,
            "stall_max_repeated_actions": 4,
            "stall_hard_stop_iterations": 8,
            "stall_graph_recovery_enabled": True,
            "stall_graph_recovery_verification_debt": 2,
            "stall_graph_recovery_severity": 20,
        }
    )
    state = _make_state()
    state["execution_graph_nodes"] = [
        {"id": "step_1", "type": "act", "success": True},
        {"id": "step_1.verify", "type": "verify", "success": False},
    ]
    state["execution_graph_edges"] = [
        {"from": "step_1", "to": "step_1.verify", "type": "verify_after"},
    ]
    state["verification_attempts"] = 3
    state["verification_successes"] = 1
    action = {"tool": "create_document_from_text", "params": {"title": "Report"}}

    result = executor._update_stall_state(job, state, progress=25, action=action)

    assert result["should_recover"] is True
    assert result["should_stop"] is False
    assert result["verification_debt"] == 2
    assert "verification_debt=2" in result["reason"]


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
        {
            "type": "paper",
            "arxiv_id": "2401.00001",
            "authors": ["A"],
            "published": "2024-01-01",
        },
        {
            "type": "paper",
            "arxiv_id": "2401.00002",
            "authors": ["B"],
            "published": "2024-01-02",
        },
        {"type": "insight", "category": "key_insight"},
    ]

    sparse_score = executor._score_research_evidence_quality(
        sparse, target_docs=8, target_papers=8
    )
    rich_score = executor._score_research_evidence_quality(
        rich, target_docs=8, target_papers=8
    )

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


def test_should_run_critic_on_execution_graph_pressure():
    executor = AutonomousAgentExecutor()
    job = _make_job(
        config={
            "critic_enabled": True,
            "critic_every_n_iterations": 99,
            "critic_on_stall": True,
            "critic_stall_threshold": 2,
            "critic_on_uncertainty": False,
        }
    )
    state = _make_state()
    job.iteration = 5
    state["last_critic_iteration"] = 4
    state["stalled_iterations"] = 0
    state["execution_graph_nodes"] = [
        {"id": "step_1", "type": "act", "success": True},
        {"id": "step_1.verify", "type": "verify", "success": False},
    ]
    state["execution_graph_edges"] = [
        {"from": "step_1", "to": "step_1.verify", "type": "verify_after"},
    ]
    state["verification_attempts"] = 2
    state["verification_successes"] = 0

    assert executor._should_run_critic(job, state) is True
    assert state["critic_last_trigger"]["reason"] == "graph"
    assert state["critic_last_trigger"]["by_graph"] is True


def test_recovery_action_prefers_critic_recommendation_when_usable():
    executor = AutonomousAgentExecutor()
    job = _make_job()
    state = _make_state()
    state["critic_notes"] = [{"recommended_tools": ["summarize_document"]}]
    state["findings"] = [{"type": "document", "id": "doc-123"}]

    action = executor._build_recovery_action(
        job, state, exclude_tool="search_documents"
    )

    assert action is not None
    assert action["tool"] == "summarize_document"
    assert action["params"]["document_id"] == "doc-123"


def test_recovery_action_uses_project_bootstrap_when_graph_health_is_degraded():
    executor = AutonomousAgentExecutor()
    job = _make_job(config={"source_id": str(uuid4())})
    state = _make_state()
    state["execution_graph_nodes"] = [
        {"id": "step_1", "type": "act", "success": True},
        {"id": "step_1.verify", "type": "verify", "success": False},
    ]
    state["execution_graph_edges"] = [
        {"from": "step_1", "to": "step_1.verify", "type": "verify_after"},
    ]
    state["verification_attempts"] = 2
    state["verification_successes"] = 0

    action = executor._build_recovery_action(
        job, state, exclude_tool="search_documents"
    )

    assert action is not None
    assert action["tool"] == "project_bootstrap"


def test_recovery_action_replans_on_execution_graph_cycle_when_bootstrapped():
    executor = AutonomousAgentExecutor()
    job = _make_job(config={"source_id": str(uuid4())})
    state = _make_state()
    state["project_profile"] = {"source_id": str(uuid4()), "sampled_files": 10}
    state["execution_graph_nodes"] = [
        {"id": "step_1", "type": "act", "success": True},
        {"id": "step_2", "type": "act", "success": False},
    ]
    state["execution_graph_edges"] = [
        {"from": "step_1", "to": "step_2", "type": "next_step"},
        {"from": "step_2", "to": "step_1", "type": "next_step"},
    ]

    action = executor._build_recovery_action(
        job, state, exclude_tool="search_documents"
    )

    assert action is not None
    assert action["tool"] == "suggest_next_action"


def test_resolve_memory_runtime_config_applies_role_profile_overrides():
    executor = AutonomousAgentExecutor()
    job = _make_job(
        config={
            "memory": {
                "enabled": True,
                "max_memories": 8,
                "memory_types": ["finding", "insight"],
                "include_chat_memory": False,
                "role_profiles": {
                    "researcher": {
                        "max_memories": 15,
                        "memory_types": ["finding", "fact", "context"],
                        "include_chat_memory": True,
                    }
                },
            }
        }
    )
    state = _make_state()
    state["skill_profile"] = {"role": "researcher"}

    resolved = executor._resolve_memory_runtime_config(job, state)

    assert resolved["enabled"] is True
    assert resolved["role"] == "researcher"
    assert resolved["limit"] == 15
    assert resolved["include_chat_memory"] is True
    assert resolved["memory_types"] == ["finding", "fact", "context"]


def test_normalize_role_token_maps_common_swarm_aliases():
    executor = AutonomousAgentExecutor()

    assert executor._normalize_role_token("researcher_documents") == "researcher"
    assert executor._normalize_role_token("researcher_arxiv") == "researcher"
    assert executor._normalize_role_token("Knowledge Researcher") == "researcher"
    assert executor._normalize_role_token("Swarm Agent 2: Analyst") == "critic"
    assert executor._normalize_role_token("monitor") == "verifier"
    assert executor._normalize_role_token("synth") == "synthesizer"
    assert executor._normalize_role_token("reproducer") == "verifier"
    assert executor._normalize_role_token("root_cause") == "critic"
    assert executor._normalize_role_token("patcher") == "coder"


def test_resolve_agent_skill_profile_maps_role_aliases_deterministically():
    executor = AutonomousAgentExecutor()
    state = _make_state()
    cases = [
        ("researcher_documents", "researcher"),
        ("researcher_arxiv", "researcher"),
        ("analyst", "critic"),
        ("monitor", "verifier"),
        ("synth", "synthesizer"),
    ]

    for role_input, expected in cases:
        job = _make_job(config={"agent_role": role_input})
        profile = executor._resolve_agent_skill_profile(job, state=state)
        assert profile["role"] == expected


def test_resolve_memory_runtime_config_matches_alias_role_profile_keys():
    executor = AutonomousAgentExecutor()
    job = _make_job(
        config={
            "agent_role": "analyst",
            "memory": {
                "enabled": True,
                "max_memories": 6,
                "memory_types": ["finding"],
                "include_chat_memory": True,
                "role_profiles": {
                    "analyst": {
                        "max_memories": 11,
                        "memory_types": ["pattern", "lesson"],
                        "include_chat_memory": False,
                    }
                },
            },
        }
    )
    state = _make_state()

    resolved = executor._resolve_memory_runtime_config(job, state)

    assert resolved["role"] == "critic"
    assert resolved["limit"] == 11
    assert resolved["include_chat_memory"] is False
    assert resolved["memory_types"] == ["pattern", "lesson"]


def test_resolve_memory_extraction_policy_defaults_include_failed():
    executor = AutonomousAgentExecutor()
    job = _make_job(config={})

    policy = executor._resolve_memory_extraction_policy(job)

    assert policy["extract_on_statuses"] == ["completed", "failed"]
    assert policy["failed_extraction_types"] == ["pattern", "lesson", "insight"]
    assert policy["completed_extraction_types"] == []


def test_resolve_memory_extraction_policy_honors_statuses_and_type_overrides():
    executor = AutonomousAgentExecutor()
    job = _make_job(
        config={
            "memory": {
                "extract_on_statuses": ["completed"],
                "failed_extraction_types": ["lesson", "invalid", "pattern"],
                "completed_extraction_types": ["finding", "summary", "oops"],
            }
        }
    )

    policy = executor._resolve_memory_extraction_policy(job)

    assert policy["extract_on_statuses"] == ["completed"]
    assert policy["failed_extraction_types"] == ["lesson", "pattern"]
    assert policy["completed_extraction_types"] == ["finding", "summary"]


def test_resolve_execution_mode_normalizes_aliases():
    executor = AutonomousAgentExecutor()
    state = _make_state()
    job = _make_job(config={"execution_mode": "plan-then-act"})

    mode = executor._resolve_execution_mode(job, state=state)

    assert mode == "plan_and_execute"
    assert state["execution_mode"] == "plan_and_execute"


@pytest.mark.asyncio
async def test_ensure_execution_plan_initializes_state_once_in_plan_mode():
    executor = AutonomousAgentExecutor()
    executor.llm_service.generate_response = AsyncMock(
        return_value=json.dumps(
            {
                "plan_steps": [
                    {
                        "title": "Collect evidence",
                        "objective": "Search for relevant documents",
                        "exit_criteria": "Relevant evidence gathered",
                        "suggested_tools": ["search_documents"],
                    },
                    {
                        "title": "Summarize evidence",
                        "objective": "Convert evidence into a concise synthesis",
                        "exit_criteria": "A synthesis artifact exists",
                        "suggested_tools": ["create_document_from_text"],
                    },
                ]
            }
        )
    )

    job = _make_job(config={"execution_mode": "plan_and_execute"})
    state = _make_state()

    used_llm = await executor._ensure_execution_plan(
        job=job,
        agent_def=None,
        state=state,
        observation={"iteration": 1, "context": []},
        user_settings=None,
    )
    repeated = await executor._ensure_execution_plan(
        job=job,
        agent_def=None,
        state=state,
        observation={"iteration": 2, "context": []},
        user_settings=None,
    )

    assert used_llm is True
    assert repeated is False
    assert executor.llm_service.generate_response.await_count == 1
    assert state["plan_generation_attempted"] is True
    assert state["execution_plan_version"] == 1
    assert state["plan_replan_count"] == 0
    assert state["plan_completed"] is False
    assert state["plan_step_index"] == 0
    assert len(state["execution_plan"]) == 2
    assert state["execution_plan"][0]["status"] == "in_progress"
    assert state["execution_plan"][1]["status"] == "pending"
    assert state["execution_plan"][0]["step_id"].startswith("step_")


def test_is_execution_plan_complete_requires_all_steps_done():
    executor = AutonomousAgentExecutor()
    state = _make_state()
    state["execution_plan"] = [
        {"step_id": "step_1", "status": "done"},
        {"step_id": "step_2", "status": "pending"},
    ]

    assert executor._is_execution_plan_complete(state) is False

    state["execution_plan"][1]["status"] = "done"
    assert executor._is_execution_plan_complete(state) is True


def test_apply_revised_plan_preserves_completed_steps_and_recomputes_progress():
    executor = AutonomousAgentExecutor()
    state = _make_state()
    state["execution_plan"] = [
        {
            "step_id": "step_1",
            "title": "Collect evidence",
            "objective": "Search for relevant documents",
            "status": "done",
            "progress": 100,
            "suggested_tools": ["search_documents"],
        },
        {
            "step_id": "step_2",
            "title": "Old pending step",
            "objective": "Replace the remainder of the plan",
            "status": "pending",
            "progress": 0,
            "suggested_tools": ["read_document_content"],
        },
    ]
    state["plan_step_index"] = 1
    state["execution_plan_version"] = 1
    state["plan_replan_count"] = 0
    state["plan_completed"] = False

    revised = ExecutionPlan(
        steps=[
            PlanStep(
                title="Collect more evidence",
                objective="Search for stronger supporting evidence",
                suggested_tools=["search_documents"],
            ),
            PlanStep(
                title="Write synthesis",
                objective="Package the result into a reusable artifact",
                suggested_tools=["create_document_from_text"],
            ),
        ],
        version=2,
        replan_count=1,
    )

    executor._apply_revised_plan(state, revised)

    assert state["execution_plan_version"] == 2
    assert state["plan_replan_count"] == 1
    assert state["plan_step_index"] == 1
    assert state["plan_completed"] is False
    assert len(state["execution_plan"]) == 3
    assert state["execution_plan"][0]["title"] == "Collect evidence"
    assert state["execution_plan"][0]["status"] == "done"
    assert state["execution_plan"][1]["title"] == "Collect more evidence"
    assert state["execution_plan"][1]["status"] == "in_progress"
    assert state["execution_plan"][2]["title"] == "Write synthesis"
    assert state["plan_progress"] == 33


def test_enforce_plan_step_action_adjusts_to_suggested_tool_in_plan_mode():
    executor = AutonomousAgentExecutor()
    job = _make_job(config={"execution_mode": "plan_and_execute"})
    state = _make_state()
    state["execution_plan"] = [
        {
            "step_id": "step_1",
            "title": "Collect evidence",
            "status": "in_progress",
            "suggested_tools": ["search_documents", "read_document_content"],
        }
    ]
    state["plan_step_index"] = 0
    action = {
        "tool": "create_document_from_text",
        "params": {},
        "purpose": "skip ahead",
    }

    adjusted = executor._enforce_plan_step_action(job, state, action)

    assert isinstance(adjusted, dict)
    assert adjusted["tool"] == "search_documents"
    assert any(
        isinstance(row, dict) and row.get("type") == "plan_action_adjusted"
        for row in state.get("step_events", [])
    )


def test_advance_execution_plan_state_marks_plan_completed_for_last_step():
    executor = AutonomousAgentExecutor()
    state = _make_state()
    state["execution_mode"] = "plan_and_execute"
    state["execution_plan"] = [
        {
            "step_id": "step_1",
            "title": "Collect evidence",
            "status": "in_progress",
            "suggested_tools": ["search_documents"],
        }
    ]
    state["plan_step_index"] = 0

    executor._advance_execution_plan_state(
        state=state,
        action={"tool": "search_documents", "params": {}},
        action_result={"success": True, "findings": [{"title": "f1"}]},
        previous_progress=20,
        current_progress=28,
        iteration=3,
    )

    assert state["plan_completed"] is True
    assert state["execution_plan"][0]["status"] == "done"
    assert any(
        isinstance(row, dict) and row.get("type") == "plan_completed"
        for row in state.get("step_events", [])
    )


def test_advance_execution_plan_state_keeps_deferred_external_step_waiting():
    executor = AutonomousAgentExecutor()
    state = _make_state()
    state["execution_mode"] = "plan_and_execute"
    state["execution_plan"] = [
        {
            "step_id": "step_1",
            "title": "Run external compiler experiment",
            "status": "waiting_external",
            "external_outbox_id": "outbox-1",
        }
    ]
    state["plan_step_index"] = 0

    executor._advance_execution_plan_state(
        state=state,
        action={"tool": "enqueue_external_agent_call", "params": {}},
        action_result={
            "success": True,
            "deferred_external": True,
            "data": {"outbox_id": "outbox-1"},
            "artifacts": [{"type": "external_call_outbox"}],
        },
        previous_progress=20,
        current_progress=30,
        iteration=3,
    )

    assert state["plan_step_index"] == 0
    assert state["plan_completed"] is False
    assert state["execution_plan"][0]["status"] == "waiting_external"
    assert state["execution_plan"][0].get("completions", 0) == 0


def test_advance_execution_plan_state_blocks_when_exit_criteria_not_met_in_plan_mode():
    executor = AutonomousAgentExecutor()
    state = _make_state()
    state["execution_mode"] = "plan_and_execute"
    state["execution_plan"] = [
        {
            "step_id": "step_1",
            "title": "Produce synthesis output",
            "status": "in_progress",
            "exit_criteria": "Create a synthesis report artifact.",
            "suggested_tools": ["create_synthesis_document"],
        }
    ]
    state["plan_step_index"] = 0

    executor._advance_execution_plan_state(
        state=state,
        action={"tool": "search_documents", "params": {}},
        action_result={"success": True, "findings": [{"title": "f1"}], "artifacts": []},
        previous_progress=20,
        current_progress=28,
        iteration=4,
    )

    assert state["plan_step_index"] == 0
    assert state["plan_completed"] is False
    assert state["execution_plan"][0]["status"] == "in_progress"
    assert any(
        isinstance(row, dict) and row.get("type") == "step_exit_not_met"
        for row in state.get("step_events", [])
    )


def test_advance_execution_plan_state_allows_exit_criteria_when_write_output_tool_used():
    executor = AutonomousAgentExecutor()
    state = _make_state()
    state["execution_mode"] = "plan_and_execute"
    state["execution_plan"] = [
        {
            "step_id": "step_1",
            "title": "Produce synthesis output",
            "status": "in_progress",
            "exit_criteria": "Create a synthesis report artifact.",
            "suggested_tools": ["create_document_from_text"],
        }
    ]
    state["plan_step_index"] = 0

    executor._advance_execution_plan_state(
        state=state,
        action={"tool": "create_document_from_text", "params": {}},
        action_result={"success": True, "findings": [], "artifacts": []},
        previous_progress=30,
        current_progress=36,
        iteration=5,
    )

    assert state["execution_plan"][0]["status"] == "done"
    assert state["plan_completed"] is True


def test_ensure_subgoal_chain_config_creates_child_jobs():
    executor = AutonomousAgentExecutor()
    job = _make_job(
        config={
            "auto_subgoal_child_jobs_enabled": True,
            "auto_subgoal_child_jobs_max": 2,
        }
    )
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
    assert children[0]["config"]["swarm_role_key"] == "researcher_documents"
    assert children[0]["config"]["agent_role"] == "researcher_documents"
    assert children[0]["config"]["swarm_role_index"] == 1
    assert children[0]["config"]["swarm_child_jobs_enabled"] is False
    assert children[0]["config"]["auto_subgoal_child_jobs_enabled"] is False
    assert isinstance(children[0].get("chain_config"), dict)
    assert children[0]["chain_config"]["chain_data"]["source"] == "swarm_fan_in"
    fan_in_child = children[0]["chain_config"]["child_jobs"][0]
    assert fan_in_child["config"]["origin"] == "swarm_fan_in_aggregator"
    assert fan_in_child["config"]["deterministic_runner"] == "swarm_fan_in_aggregate"
    assert (
        fan_in_child["config"]["swarm_fan_in_group_id"]
        == state["swarm_fan_in_group_id"]
    )
    assert any(
        isinstance(row, dict) and row.get("type") == "swarm_roles_configured"
        for row in state.get("step_events", [])
    )


def test_ensure_swarm_chain_config_creates_coding_bug_triage_roles():
    executor = AutonomousAgentExecutor()
    job = _make_job(
        config={
            "launch_mode": "quick_start_bug_triage_swarm",
            "coding_swarm_enabled": True,
            "swarm_child_jobs_enabled": True,
            "swarm_max_agents": 4,
            "swarm_roles": ["reproducer", "root_cause", "patcher", "verifier"],
        }
    )
    state = _make_state()

    executor._ensure_swarm_chain_config(job, state)

    children = job.chain_config.get("child_jobs")
    assert isinstance(children, list)
    assert len(children) == 4
    assert children[0]["config"]["swarm_role_key"] == "reproducer"
    assert children[0]["config"]["agent_role"] == "verifier"
    assert children[1]["config"]["swarm_role_key"] == "root_cause"
    assert children[1]["config"]["agent_role"] == "critic"
    assert children[2]["config"]["swarm_role_key"] == "patcher"
    assert children[2]["config"]["agent_role"] == "coder"
    fan_in_child = children[0]["chain_config"]["child_jobs"][0]
    assert fan_in_child["config"]["coding_swarm_enabled"] is True
    assert fan_in_child["config"]["coding_swarm_profile"] == "bug_triage"


def test_ensure_swarm_chain_config_persists_normalized_agent_role_keys():
    executor = AutonomousAgentExecutor()
    job = _make_job(
        config={
            "swarm_child_jobs_enabled": True,
            "swarm_max_agents": 3,
            "swarm_roles": ["document_researcher", "critic", "validator"],
        }
    )
    state = _make_state()

    executor._ensure_swarm_chain_config(job, state)

    children = job.chain_config.get("child_jobs")
    assert isinstance(children, list)
    assert len(children) == 3
    assert children[0]["config"]["swarm_role_key"] == "researcher_documents"
    assert children[0]["config"]["agent_role"] == "researcher_documents"
    assert children[1]["config"]["swarm_role_key"] == "critic"
    assert children[1]["config"]["agent_role"] == "critic"
    assert children[2]["config"]["swarm_role_key"] == "verifier"
    assert children[2]["config"]["agent_role"] == "verifier"


def test_append_job_result_step_event_persists_rows():
    executor = AutonomousAgentExecutor()
    job = _make_job()
    job.results = {}

    executor._append_job_result_step_event(
        job,
        {
            "type": "chain_triggered",
            "iteration": 2,
            "created_jobs_count": 3,
        },
    )

    execution = (job.results or {}).get("execution_strategy", {})
    rows = execution.get("step_events")
    assert isinstance(rows, list)
    assert rows
    assert rows[-1]["type"] == "chain_triggered"
    assert rows[-1]["created_jobs_count"] == 3
    assert isinstance(rows[-1].get("at"), str)


def test_sync_runtime_execution_strategy_persists_live_graph_and_scope_state():
    executor = AutonomousAgentExecutor()
    job = _make_job(config={"source_id": str(uuid4())})
    state = _make_state()
    state["step_events"] = [{"type": "checkpoint_waiting", "iteration": 2}]
    state["execution_graph_nodes"] = [{"id": "step_1", "type": "act", "success": True}]
    state["execution_graph_edges"] = []
    state["verification_attempts"] = 1
    state["verification_successes"] = 0
    state["scope_events"] = [{"type": "resolved_scope", "iteration": 2}]

    execution = executor._sync_runtime_execution_strategy(job, state, {})

    assert isinstance(execution.get("step_events"), list)
    assert execution["step_events"][-1]["type"] == "checkpoint_waiting"
    runtime = execution.get("execution_graph_runtime") or {}
    assert runtime.get("verification_attempts") == 1
    assert isinstance(runtime.get("nodes"), list)
    scope = execution.get("scope_observability_runtime") or {}
    assert scope.get("resolved_scope_id") == job.config["source_id"]
    assert isinstance(scope.get("events"), list)


def test_persist_runtime_execution_strategy_updates_job_results():
    executor = AutonomousAgentExecutor()
    job = _make_job(config={"source_id": str(uuid4())})
    job.results = {"summary": "in-flight"}
    state = _make_state()
    state["execution_graph_nodes"] = [{"id": "step_1", "type": "act", "success": True}]
    state["verification_attempts"] = 2
    state["verification_successes"] = 1

    executor._persist_runtime_execution_strategy(job, state)

    execution = (job.results or {}).get("execution_strategy") or {}
    runtime = execution.get("execution_graph_runtime") or {}
    assert runtime.get("verification_attempts") == 2
    assert runtime.get("verification_successes") == 1
    assert isinstance(runtime.get("nodes"), list)


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
                    "findings": [
                        {"title": "Prioritize internal docs for baseline facts"}
                    ],
                    "summary": "Internal docs show repeated bottleneck in ingestion.",
                },
            },
            {
                "job_id": "j2",
                "role": "Literature Researcher",
                "status": "completed",
                "progress": 100,
                "results": {
                    "findings": [
                        {"title": "Prioritize internal docs for baseline facts"}
                    ],
                    "research": {
                        "top_insights": [
                            "Benchmark against arXiv baselines for coverage"
                        ]
                    },
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


def test_build_swarm_fan_in_result_for_coding_swarm_emits_winner_and_candidate_paths():
    executor = AutonomousAgentExecutor()
    payload = {
        "swarm_parent_job_id": "parent-coding",
        "expected_siblings": 4,
        "terminal_siblings": 4,
        "coding_swarm_enabled": True,
        "coding_swarm_profile": "bug_triage",
        "commands": ["CI=true npm --prefix frontend test -- --watchAll=false"],
        "sibling_jobs": [
            {
                "job_id": "j1",
                "role": "Reproducer",
                "status": "completed",
                "progress": 100,
                "results": {
                    "summary": "Reproduced the save failure",
                    "file_paths": ["frontend/src/pages/DocumentsPage.tsx"],
                },
            },
            {
                "job_id": "j2",
                "role": "Patcher",
                "status": "completed",
                "progress": 100,
                "results": {
                    "summary": "Likely fix is in the save handler",
                    "file_paths": [
                        "frontend/src/pages/DocumentsPage.tsx",
                        "frontend/src/services/api.ts",
                    ],
                    "commands": [
                        "CI=true npm --prefix frontend test -- --watchAll=false"
                    ],
                },
            },
            {
                "job_id": "j3",
                "role": "Verifier",
                "status": "completed",
                "progress": 100,
                "results": {
                    "summary": "Frontend save handler is the strongest candidate path",
                },
            },
            {
                "job_id": "j4",
                "role": "Root Cause Analyst",
                "status": "failed",
                "progress": 55,
                "results": {
                    "summary": "Root cause analysis incomplete",
                },
            },
        ],
    }

    merged = executor._build_swarm_fan_in_result(
        payload, fan_in_group_id="group-coding"
    )

    assert merged["fan_in_group_id"] == "group-coding"
    assert merged["winning_slice_id"] in {"j1", "j2", "j3"}
    assert merged["winning_role"]
    assert isinstance(merged["candidate_paths"], list)
    assert merged["candidate_paths"]
    assert isinstance(merged["recommended_commands"], list)
    assert merged["recommended_commands"] == [
        "CI=true npm --prefix frontend test -- --watchAll=false"
    ]
    assert merged["review_state"] == "tie_break_needed"
    assert merged["file_converged"] is True
    assert merged["command_converged"] is False


@pytest.mark.asyncio
async def test_run_swarm_fan_in_aggregate_launches_tie_breaker_for_medium_confidence(
    db_session, test_user, monkeypatch
):
    queued_jobs: list[tuple[str, str]] = []

    monkeypatch.setattr(
        "app.tasks.agent_job_tasks.execute_agent_job_task.delay",
        lambda job_id, user_id: queued_jobs.append((job_id, user_id)),
    )

    executor = AutonomousAgentExecutor()
    payload = {
        "swarm_parent_job_id": "parent-coding",
        "expected_siblings": 4,
        "terminal_siblings": 4,
        "coding_swarm_enabled": True,
        "coding_swarm_profile": "bug_triage",
        "commands": ["CI=true npm --prefix frontend test -- --watchAll=false"],
        "sibling_jobs": [
            {
                "job_id": "j1",
                "role": "Reproducer",
                "status": "completed",
                "progress": 100,
                "results": {
                    "summary": "Reproduced the save failure",
                    "file_paths": ["frontend/src/pages/DocumentsPage.tsx"],
                },
            },
            {
                "job_id": "j2",
                "role": "Patcher",
                "status": "completed",
                "progress": 100,
                "results": {
                    "summary": "Likely fix is in the save handler",
                    "file_paths": [
                        "frontend/src/pages/DocumentsPage.tsx",
                        "frontend/src/services/api.ts",
                    ],
                    "commands": [
                        "CI=true npm --prefix frontend test -- --watchAll=false"
                    ],
                },
            },
            {
                "job_id": "j3",
                "role": "Verifier",
                "status": "completed",
                "progress": 100,
                "results": {
                    "summary": "Frontend save handler is the strongest candidate path",
                },
            },
            {
                "job_id": "j4",
                "role": "Root Cause Analyst",
                "status": "failed",
                "progress": 55,
                "results": {
                    "summary": "Root cause analysis incomplete",
                },
            },
        ],
    }
    fan_in_job = AgentJob(
        name="Bug Swarm Fan-in",
        goal="Resolve frontend save failure",
        job_type="synthesis",
        user_id=test_user.id,
        status=AgentJobStatus.PENDING.value,
        config={
            "coding_swarm_enabled": True,
            "coding_swarm_profile": "bug_triage",
            "coding_swarm_confidence_threshold": 0.70,
            "coding_swarm_tiebreaker_threshold": 0.50,
            "inherited_data": {"swarm": payload},
        },
        max_iterations=12,
        max_tool_calls=16,
        max_llm_calls=12,
        max_runtime_minutes=20,
    )
    db_session.add(fan_in_job)
    await db_session.flush()

    async def _fake_launch_tie_breaker(*, fan_in_job, db, merged, swarm_payload):
        child = AgentJob(
            name="Tie-breaker Verifier",
            goal="Break the tie",
            job_type="analysis",
            user_id=test_user.id,
            status=AgentJobStatus.PENDING.value,
            parent_job_id=fan_in_job.id,
            root_job_id=fan_in_job.root_job_id or fan_in_job.id,
        )
        db.add(child)
        await db.flush()
        return child

    monkeypatch.setattr(
        executor, "_launch_bug_triage_swarm_tie_breaker_job", _fake_launch_tie_breaker
    )

    result = await executor._run_swarm_fan_in_aggregate(
        job=fan_in_job, db=db_session, progress_callback=None
    )

    assert result["status"] == "completed"
    assert fan_in_job.status == AgentJobStatus.COMPLETED.value
    fan_in = (fan_in_job.results or {}).get("swarm_fan_in") or {}
    assert fan_in["review_state"] == "tie_break_running"
    assert fan_in["tie_breaker_job_id"]
    assert queued_jobs


@pytest.mark.asyncio
async def test_run_swarm_fan_in_aggregate_pauses_after_unsuccessful_tie_break(
    db_session, test_user, monkeypatch
):
    queued_jobs: list[tuple[str, str]] = []

    monkeypatch.setattr(
        "app.tasks.agent_job_tasks.execute_agent_job_task.delay",
        lambda job_id, user_id: queued_jobs.append((job_id, user_id)),
    )

    executor = AutonomousAgentExecutor()
    payload = {
        "swarm_parent_job_id": "parent-coding",
        "expected_siblings": 4,
        "terminal_siblings": 4,
        "coding_swarm_enabled": True,
        "coding_swarm_profile": "bug_triage",
        "tie_breaker_attempted": True,
        "tie_breaker_job_id": "tb-1",
        "commands": ["CI=true npm --prefix frontend test -- --watchAll=false"],
        "sibling_jobs": [
            {
                "job_id": "j1",
                "role": "Reproducer",
                "status": "completed",
                "progress": 100,
                "results": {
                    "summary": "Reproduced the save failure",
                    "file_paths": ["frontend/src/pages/DocumentsPage.tsx"],
                },
            },
            {
                "job_id": "j2",
                "role": "Patcher",
                "status": "completed",
                "progress": 100,
                "results": {
                    "summary": "Likely fix is in the save handler",
                    "file_paths": [
                        "frontend/src/pages/DocumentsPage.tsx",
                        "frontend/src/services/api.ts",
                    ],
                    "commands": [
                        "CI=true npm --prefix frontend test -- --watchAll=false"
                    ],
                },
            },
            {
                "job_id": "j3",
                "role": "Verifier",
                "status": "completed",
                "progress": 100,
                "results": {
                    "summary": "Frontend save handler is the strongest candidate path",
                },
            },
            {
                "job_id": "j4",
                "role": "Root Cause Analyst",
                "status": "failed",
                "progress": 55,
                "results": {
                    "summary": "Root cause analysis incomplete",
                },
            },
        ],
    }
    fan_in_job = AgentJob(
        name="Bug Swarm Fan-in",
        goal="Resolve frontend save failure",
        job_type="synthesis",
        user_id=test_user.id,
        status=AgentJobStatus.PENDING.value,
        config={
            "coding_swarm_enabled": True,
            "coding_swarm_profile": "bug_triage",
            "coding_swarm_confidence_threshold": 0.70,
            "coding_swarm_tiebreaker_threshold": 0.50,
            "tie_breaker_attempted": True,
            "inherited_data": {"swarm": payload},
        },
        max_iterations=12,
        max_tool_calls=16,
        max_llm_calls=12,
        max_runtime_minutes=20,
    )
    db_session.add(fan_in_job)
    await db_session.flush()

    result = await executor._run_swarm_fan_in_aggregate(
        job=fan_in_job, db=db_session, progress_callback=None
    )

    assert result["status"] == "completed"
    assert fan_in_job.status == AgentJobStatus.PAUSED.value
    fan_in = (fan_in_job.results or {}).get("swarm_fan_in") or {}
    assert fan_in["review_state"] == "insufficient_swarm_consensus"
    assert not queued_jobs


def test_maybe_apply_critic_pivot_override_forces_recommended_tool():
    executor = AutonomousAgentExecutor()
    job = _make_job(
        config={"critic_force_pivot_on_high": True, "critic_force_min_confidence": 0.5}
    )
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
        {
            "search_documents": {"success": 3, "failure": 4},
            "search_arxiv": {"success": 1, "failure": 0},
        },
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
    updated_same_instant = datetime(
        2026, 2, 6, 7, 0, 0, tzinfo=timezone(timedelta(hours=-5))
    )

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
            "tool_selection_forced_exploration_tools": [
                "search_arxiv",
                "search_documents",
            ],
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
            "tool_selection_forced_exploration_tools": [
                "search_arxiv",
                "search_documents",
            ],
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
    state["forced_exploration_history"] = [
        {"iteration": 6, "tool": "search_arxiv", "success": None}
    ]

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
    state["forced_exploration_history"] = [
        {"iteration": 6, "tool": "search_arxiv", "success": None}
    ]

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
    state["findings"] = [
        {"type": "document", "id": "d1"},
        {"type": "document", "id": "d2"},
        {"type": "paper", "id": "p1"},
    ]
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


def test_resolve_tool_selection_mode_enters_rescue_on_execution_graph_pressure():
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
    state["goal_progress"] = 60
    state["stalled_iterations"] = 0
    state["findings"] = [
        {"type": "document", "id": "d1"},
        {"type": "document", "id": "d2"},
        {"type": "paper", "id": "p1"},
    ]
    state["execution_graph_nodes"] = [
        {"id": "step_1", "type": "act", "success": True},
        {"id": "step_1.verify", "type": "verify", "success": False},
    ]
    state["execution_graph_edges"] = [
        {"from": "step_1", "to": "step_1.verify", "type": "verify_after"},
    ]
    state["verification_attempts"] = 3
    state["verification_successes"] = 0

    mode, _ = executor._resolve_tool_selection_mode(job, state=state)

    assert mode == "adaptive"
    assert state["tool_selection_goal_stage"] == "rescue"


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
    state["findings"] = [
        {"type": "document", "id": "d1"},
        {"type": "paper", "arxiv_id": "2401.00001"},
    ]
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
    finalize_eval = executor._evaluate_goal_contract(
        job, state, include_result_keys=True
    )

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
    assert any(
        r.startswith("tool:create_document_from_text")
        for r in first["checkpoint"]["reasons"]
    )
    assert second["required"] is False


def test_build_executive_digest_includes_risks_contract_and_next_steps():
    executor = AutonomousAgentExecutor()
    job = _make_job(
        config={
            "goal_contract_enabled": True,
            "goal_contract_min_findings": 2,
        }
    )
    job.results = {
        "summary": "Partial research outcome",
        "research_bundle": {"next_steps": ["Validate metrics", "Run follow-up search"]},
    }
    state = _make_state()
    state["goal_progress"] = 60
    state["findings"] = [{"type": "document", "title": "Internal ingestion bottleneck"}]
    state["artifacts"] = [{"type": "note", "id": "a1"}]
    state["actions_taken"] = [
        {"action": {"tool": "search_documents"}, "result": {"success": True}},
        {
            "action": {"tool": "search_arxiv"},
            "result": {"success": False, "error": "timeout"},
        },
    ]
    state["critic_notes"] = [
        {"severity": "high", "pivot": "Need stronger external baselines"}
    ]

    digest = executor._build_executive_digest(job, state)

    assert digest["outcome"] == "Partial research outcome"
    assert digest["metrics"]["failed_actions"] == 1
    assert digest["key_findings"]
    assert digest["risks"]
    assert digest["goal_contract"]["enabled"] is True
    assert digest["goal_contract"]["satisfied"] is False
    assert digest["next_actions"][0] == "Validate metrics"
