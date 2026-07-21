from types import SimpleNamespace
from datetime import datetime, timedelta
import asyncio
from uuid import uuid4

from app.api.endpoints.agent_jobs import (
    _build_relaunch_children_counts,
    _build_relaunch_children_counts_for_user,
    _build_launch_mode_stats,
    _build_relaunch_lineage,
    _chain_definition_to_response,
    _build_chain_config_for_step,
    _append_launch_log_if_present,
    _build_launch_mode_counts,
    _build_quick_start_bug_triage_swarm_config,
    _build_quick_start_claude_backend_config,
    _build_quick_start_domain_research_config,
    _build_quick_start_bug_triage_swarm_relaunch_request,
    _build_quick_start_repo_bug_triage_config,
    _build_quick_start_role_workflow_config,
    _build_quick_start_relaunch_request,
    _build_quick_start_domain_research_relaunch_request,
    _build_quick_start_repo_bug_triage_relaunch_request,
    _build_quick_start_role_workflow_relaunch_request,
    _extract_launch_mode,
    _extract_relaunch_parent_job_id,
    _extract_source_id_from_config,
    _infer_coding_swarm_preset_key,
    _derive_repair_verification_status,
    _derive_swarm_outcome_case,
    _find_unsafe_commands,
    _is_none_launch_mode,
    _is_source_owned_by_user,
    _matches_launch_mode_filter,
    _normalize_scope_keys_deep,
    _swarm_confidence_bucket,
)
from app.schemas.agent_job import AgentJobQuickStartClaudeBackendRequest
from app.schemas.agent_job import AgentJobQuickStartBugTriageSwarmRequest
from app.schemas.agent_job import AgentJobQuickStartDomainResearchRequest
from app.schemas.agent_job import AgentJobQuickStartRepoBugTriageRequest
from app.schemas.agent_job import AgentJobQuickStartRoleWorkflowRequest


def test_is_source_owned_by_user_matches_requested_by_user_id():
    user_id = str(uuid4())
    source = SimpleNamespace(config={"requested_by_user_id": user_id})
    user = SimpleNamespace(id=user_id, username="alice")

    assert _is_source_owned_by_user(source, user) is True


def test_is_source_owned_by_user_matches_requested_by_username():
    source = SimpleNamespace(config={"requested_by": "alice"})
    user = SimpleNamespace(id=str(uuid4()), username="alice")

    assert _is_source_owned_by_user(source, user) is True


def test_is_source_owned_by_user_returns_false_when_mismatch():
    source = SimpleNamespace(config={"requested_by": "bob"})
    user = SimpleNamespace(id=str(uuid4()), username="alice")

    assert _is_source_owned_by_user(source, user) is False


def test_infer_coding_swarm_preset_key_prefers_quick_start_metadata():
    job = SimpleNamespace(
        config={
            "launch_mode": "quick_start_frontend_regression_swarm",
            "quick_start": {"preset_key": "frontend_regression_swarm"},
        }
    )

    assert _infer_coding_swarm_preset_key(job) == "frontend_regression_swarm"


def test_swarm_confidence_bucket_uses_expected_thresholds():
    assert _swarm_confidence_bucket(0.8) == "high"
    assert _swarm_confidence_bucket(0.55) == "medium"
    assert _swarm_confidence_bucket(0.2) == "low"


def test_derive_repair_verification_status_uses_recovery_state():
    job = SimpleNamespace(
        status="completed",
        execution_log=[],
        results={
            "code_patch_execution": {
                "recovery": {
                    "recovery_state": "verified_fix",
                    "retry_reason": "Verification succeeded against the promoted fix.",
                }
            }
        },
    )

    status, reason = _derive_repair_verification_status(job)

    assert status == "succeeded"
    assert reason == "Verification succeeded against the promoted fix."


def test_derive_swarm_outcome_case_prefers_verified_fix_over_review_state():
    swarm_job = SimpleNamespace(
        id="swarm-1",
        name="Frontend Regression Swarm",
        status="completed",
        created_at=datetime(2026, 3, 10, 0, 0, 0),
        completed_at=datetime(2026, 3, 10, 0, 5, 0),
        last_activity_at=datetime(2026, 3, 10, 0, 5, 0),
        config={
            "launch_mode": "quick_start_frontend_regression_swarm",
            "source_id": "repo-source-1",
            "quick_start": {
                "preset_key": "frontend_regression_swarm",
                "source_name": "Knowledge Repo",
            },
        },
        results={
            "swarm_fan_in": {
                "review_state": "auto_promoted",
                "promotion_reason": "Auto-promoted winning coding slice.",
                "confidence": {"overall": 0.82},
                "repair_chain_job_id": "repair-1",
            }
        },
    )
    repair_job = SimpleNamespace(
        id="repair-1",
        name="Repair Chain",
        status="completed",
        created_at=datetime(2026, 3, 10, 0, 20, 0),
        completed_at=datetime(2026, 3, 10, 0, 35, 0),
        last_activity_at=datetime(2026, 3, 10, 0, 35, 0),
        execution_log=[],
        results={
            "code_patch_execution": {
                "recovery": {
                    "recovery_state": "verified_fix",
                    "retry_reason": "Verification succeeded.",
                }
            }
        },
    )

    case = _derive_swarm_outcome_case(
        swarm_job,
        repair_jobs_by_id={"repair-1": repair_job},
        backlog_by_swarm_job_id={},
    )

    assert case.preset_key == "frontend_regression_swarm"
    assert case.promotion_mode == "auto"
    assert case.repair_job_id == "repair-1"
    assert case.verification_status == "succeeded"
    assert case.terminal_outcome == "verified_fix"


def test_build_quick_start_config_includes_launch_metadata():
    req = AgentJobQuickStartClaudeBackendRequest(
        goal="Fix backend tests",
        source_id="00000000-0000-0000-0000-000000000123",
        search_query="backend",
        commands=["python -m pytest -q"],
    )

    cfg = _build_quick_start_claude_backend_config(
        req,
        source_name="Repo Source",
        source_type="github",
    )

    assert cfg["source_id"] == "00000000-0000-0000-0000-000000000123"
    assert cfg["launch_mode"] == "quick_start_claude_backend"
    assert isinstance(cfg.get("quick_start"), dict)
    assert cfg["quick_start"]["profile"] == "claude_backend"
    assert cfg["quick_start"]["source_name"] == "Repo Source"
    assert cfg["quick_start"]["source_type"] == "github"


def test_build_quick_start_role_workflow_config_includes_memory_and_swarm_defaults():
    req = AgentJobQuickStartRoleWorkflowRequest(
        goal="Investigate quality regressions in retrieval",
        roles=["researcher_documents", "analyst", "synthesizer"],
        memory_profile="evidence",
        approval_mode="high_impact",
    )

    cfg = _build_quick_start_role_workflow_config(req)

    assert cfg["launch_mode"] == "quick_start_role_workflow"
    assert cfg["swarm_child_jobs_enabled"] is True
    assert cfg["swarm_inherit_config"] is True
    assert cfg["swarm_inherit_results"] is True
    assert cfg["swarm_fan_in_enabled"] is True
    assert cfg["quick_start"]["profile"] == "role_workflow"
    assert cfg["quick_start"]["memory_profile"] == "evidence"
    assert cfg["quick_start"]["execution_mode"] == "plan_and_execute"
    assert cfg["execution_mode"] == "plan_and_execute"
    assert cfg["memory"]["enabled"] is True
    assert cfg["memory"]["profile"] == "evidence"
    assert cfg["memory"]["extract_on_statuses"] == ["completed", "failed"]
    assert cfg["quick_start"]["extract_memory_on_failure"] is True
    assert cfg["enable_memory"] is True
    assert cfg["approval_checkpoints"]["enabled"] is True


def test_build_quick_start_domain_research_config_includes_launch_metadata():
    req = AgentJobQuickStartDomainResearchRequest(
        domain="Multimodal retrieval",
        objective="Rank evidence-backed opportunities",
        customer_context="Enterprise search",
        source_scope="kb_plus_arxiv_plus_repo",
        track_type="compiler",
        monitor_queries=["multimodal retrieval benchmarking"],
        repo_source_ids=["00000000-0000-0000-0000-000000000111"],
        benchmark_queries=["compile time regression", "vectorization benchmark"],
        sandbox_profile_id="scientific-compiler-sandbox",
        report_format="brief_and_report",
        persist_artifacts=True,
        auto_launch_follow_up=True,
        automation_profile="max_autonomy",
        automation_policy={
            "confidence_threshold": 0.77,
            "experiment_readiness_threshold": 0.82,
            "max_auto_follow_up_launches": 3,
            "auto_create_experiment_plans": True,
            "auto_launch_follow_up": True,
        },
    )

    cfg = _build_quick_start_domain_research_config(req)

    assert cfg["launch_mode"] == "quick_start_domain_research"
    assert cfg["deterministic_runner"] == "domain_research_orchestrator"
    assert cfg["domain_research_mode"] is True
    assert cfg["domain"] == "Multimodal retrieval"
    assert cfg["objective"] == "Rank evidence-backed opportunities"
    assert cfg["track_type"] == "compiler"
    assert cfg["research_mode"] == "literature_to_hypothesis"
    assert cfg["repo_source_ids"] == ["00000000-0000-0000-0000-000000000111"]
    assert cfg["benchmark_queries"] == ["compile time regression", "vectorization benchmark"]
    assert cfg["sandbox_profile_id"] == "scientific-compiler-sandbox"
    assert cfg["persist_target"] == "research_notes"
    assert cfg["automation_profile"] == "max_autonomy"
    assert cfg["auto_create_experiment_plans"] is True
    assert cfg["automation_policy"]["confidence_threshold"] == 0.77
    assert "validation_policy" not in cfg
    assert cfg["quick_start"]["profile"] == "domain_research"
    assert cfg["quick_start"]["research_mode"] == "literature_to_hypothesis"
    assert cfg["quick_start"]["source_scope"] == "kb_plus_arxiv_plus_repo"
    assert cfg["quick_start"]["track_type"] == "compiler"
    assert cfg["quick_start"]["sandbox_profile_id"] == "scientific-compiler-sandbox"
    assert cfg["quick_start"]["auto_launch_follow_up"] is True
    assert cfg["quick_start"]["auto_create_experiment_plans"] is True
    assert cfg["prefer_sources"] == ["documents", "arxiv", "repo"]


def test_build_quick_start_domain_research_config_does_not_seed_legacy_validation_policy_when_canonical_fields_exist():
    req = AgentJobQuickStartDomainResearchRequest(
        domain="Compiler",
        objective="Track compiler opportunities",
        automation_profile="balanced",
        automation_policy={"confidence_threshold": 0.81, "auto_launch_follow_up": False},
        start_immediately=True,
    )

    cfg = _build_quick_start_domain_research_config(req)

    assert cfg["automation_profile"] == "balanced"
    assert cfg["automation_policy"]["confidence_threshold"] == 0.81
    assert "validation_policy" not in cfg


def test_build_quick_start_repo_bug_triage_config_includes_launch_metadata():
    req = AgentJobQuickStartRepoBugTriageRequest(
        failure_symptom="Frontend save action returns 500",
        goal="Fix the regression and keep the form save flow working",
        source_id="00000000-0000-0000-0000-000000000123",
        scope="frontend",
        error_output="TypeError: undefined is not a function",
    )

    cfg = _build_quick_start_repo_bug_triage_config(
        req,
        source_name="Repo Source",
        source_type="github",
    )

    assert cfg["source_id"] == "00000000-0000-0000-0000-000000000123"
    assert cfg["launch_mode"] == "quick_start_repo_bug_triage"
    assert cfg["failure_symptom"] == "Frontend save action returns 500"
    assert cfg["scope"] == "frontend"
    assert cfg["error_output"] == "TypeError: undefined is not a function"
    assert isinstance(cfg.get("quick_start"), dict)
    assert cfg["quick_start"]["profile"] == "repo_bug_triage"
    assert cfg["quick_start"]["version"] == "v2"
    assert cfg["quick_start"]["source_name"] == "Repo Source"
    assert cfg["quick_start"]["source_type"] == "github"
    assert cfg["quick_start"]["autonomy_mode"] == "patch_proposal"
    assert cfg["quick_start"]["execution_depth"] == "workspace_planned"
    assert cfg["search_query"] == "frontend Frontend save action returns 500"


def test_build_quick_start_bug_triage_swarm_config_includes_swarm_defaults():
    req = AgentJobQuickStartBugTriageSwarmRequest(
        failure_symptom="Frontend save action returns 500",
        goal="Promote the strongest repair path",
        source_id="00000000-0000-0000-0000-000000000123",
        scope="frontend",
        max_agents=4,
        commands=["CI=true npm --prefix frontend test -- --watchAll=false"],
        file_paths=["frontend/src/pages/DocumentsPage.tsx"],
        error_output="TypeError: undefined is not a function",
    )

    cfg = _build_quick_start_bug_triage_swarm_config(
        req,
        source_name="Repo Source",
        source_type="github",
    )

    assert cfg["source_id"] == "00000000-0000-0000-0000-000000000123"
    assert cfg["launch_mode"] == "quick_start_bug_triage_swarm"
    assert cfg["scope"] == "frontend"
    assert cfg["swarm_child_jobs_enabled"] is True
    assert cfg["swarm_fan_in_enabled"] is True
    assert cfg["coding_swarm_enabled"] is True
    assert cfg["coding_swarm_auto_promote_best_slice"] is True
    assert cfg["coding_swarm_auto_launch_repair_chain"] is True
    assert cfg["swarm_roles"] == ["reproducer", "root_cause", "patcher", "verifier"]
    assert cfg["quick_start"]["profile"] == "bug_triage_swarm"
    assert cfg["quick_start"]["autonomy_mode"] == "max_autonomy"
    assert cfg["quick_start"]["max_agents"] == 4
    assert cfg["search_query"] == "frontend Frontend save action returns 500 bug symptom"


def test_build_quick_start_role_workflow_config_applies_memory_extraction_overrides():
    req = AgentJobQuickStartRoleWorkflowRequest(
        goal="Investigate quality regressions in retrieval",
        roles=["researcher_documents", "analyst", "synthesizer"],
        memory_profile="balanced",
        extract_memory_on_failure=False,
        memory_failed_types=["lesson", "pattern"],
        memory_completed_types=["finding", "summary"],
    )

    cfg = _build_quick_start_role_workflow_config(req)

    assert cfg["memory"]["extract_on_statuses"] == ["completed"]
    assert cfg["memory"]["extract_on_failure"] is False
    assert cfg["memory"]["failed_extraction_types"] == ["lesson", "pattern"]
    assert cfg["memory"]["completed_extraction_types"] == ["finding", "summary"]
    assert cfg["quick_start"]["extract_memory_on_failure"] is False


def test_build_quick_start_domain_research_relaunch_request_preserves_context():
    job = SimpleNamespace(
        id=uuid4(),
        name="Domain Research - Retrieval",
        goal="Research the domain 'retrieval'",
        config={
            "launch_mode": "quick_start_domain_research",
            "deterministic_runner": "domain_research_orchestrator",
            "domain_research_mode": True,
            "domain": "Retrieval",
            "objective": "Rank ideas",
            "customer_context": "Enterprise context",
            "source_scope": "kb_plus_arxiv_plus_repo",
            "track_type": "microarchitecture",
            "research_mode": "literature_to_hypothesis",
            "monitor_queries": ["retrieval latency"],
            "repo_source_ids": [str(uuid4())],
            "benchmark_queries": ["ipc stall"],
            "sandbox_profile_id": "scientific-microarchitecture-sandbox",
            "report_format": "brief_and_report",
            "scoring_policy": {"minimum_subscore": 0.6},
            "selection_policy": {"max_hypotheses": 3},
            "persist_artifacts": True,
            "auto_launch_follow_up": False,
            "auto_create_experiment_plans": True,
            "max_documents": 9,
            "max_papers": 7,
            "profile_id": str(uuid4()),
            "automation_profile": "max_autonomy",
            "automation_policy": {"confidence_threshold": 0.83, "experiment_readiness_threshold": 0.9},
            "validation_policy": {"confidence_threshold": 0.83, "experiment_readiness_threshold": 0.9},
            "confidence_threshold": 0.8,
        },
    )

    req = _build_quick_start_domain_research_relaunch_request(job)

    assert req is not None
    assert req.domain == "Retrieval"
    assert req.objective == "Rank ideas"
    assert req.customer_context == "Enterprise context"
    assert req.source_scope == "kb_plus_arxiv_plus_repo"
    assert req.track_type == "microarchitecture"
    assert req.research_mode == "literature_to_hypothesis"
    assert req.monitor_queries == ["retrieval latency"]
    assert req.benchmark_queries == ["ipc stall"]
    assert req.sandbox_profile_id == "scientific-microarchitecture-sandbox"
    assert req.scoring_policy == {"minimum_subscore": 0.6}
    assert req.selection_policy == {"max_hypotheses": 3}
    assert req.auto_launch_follow_up is False
    assert req.auto_create_experiment_plans is True
    assert req.automation_profile == "max_autonomy"
    assert req.automation_policy == {
        "confidence_threshold": 0.83,
        "experiment_readiness_threshold": 0.9,
        "max_auto_follow_up_launches": 2,
        "duplicate_window_items": 60,
        "auto_create_experiment_plans": True,
        "auto_launch_follow_up": True,
        "auto_execute_validation_runs": False,
        "max_concurrent_validation_runs": 1,
        "max_validation_runtime_minutes": 20,
        "max_validation_budget_per_run": 25.0,
        "follow_up_review_mode": "queue_for_approval",
        "validation_backoff_policy": {
            "max_consecutive_failures": 2,
            "cooldown_minutes": 180,
        },
        "auto_launch_experiment_runs": False,
    }
    assert req.validation_policy == {
        "confidence_threshold": 0.83,
        "experiment_readiness_threshold": 0.9,
        "max_auto_follow_up_launches": 2,
        "auto_create_experiment_plans": True,
        "auto_launch_follow_up": True,
        "auto_execute_validation_runs": False,
        "max_concurrent_validation_runs": 1,
        "max_validation_runtime_minutes": 20,
        "max_validation_budget_per_run": 25.0,
        "validation_backoff_policy": {
            "max_consecutive_failures": 2,
            "cooldown_minutes": 180,
        },
    }
    assert req.profile_id is not None
    assert req.config_overrides is not None
    assert req.config_overrides["relaunch_from_job_id"] == str(job.id)


def test_normalize_scope_keys_deep_rewrites_nested_target_source_ids():
    payload = {
        "target_source_id": "root",
        "nested": {"target_source_id": "n1"},
        "items": [
            {"target_source_id": "i1"},
            {"source_id": "i2", "target_source_id": "legacy"},
        ],
    }

    out = _normalize_scope_keys_deep(payload)

    assert out["source_id"] == "root"
    assert "target_source_id" not in out
    assert out["nested"]["source_id"] == "n1"
    assert "target_source_id" not in out["nested"]
    assert out["items"][0]["source_id"] == "i1"
    assert "target_source_id" not in out["items"][0]
    assert out["items"][1]["source_id"] == "i2"
    assert "target_source_id" not in out["items"][1]


def test_chain_definition_to_response_normalizes_nested_scope_keys():
    chain = SimpleNamespace(
        id=uuid4(),
        name="chain_a",
        display_name="Chain A",
        description=None,
        chain_steps=[
            {
                "step_name": "Step 1",
                "config": {
                    "target_source_id": "s1",
                    "nested": {"target_source_id": "n1"},
                },
            }
        ],
        default_settings={
            "target_source_id": "d1",
            "child": {"target_source_id": "d2"},
        },
        owner_user_id=uuid4(),
        is_system=False,
        is_active=True,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )

    res = _chain_definition_to_response(chain)

    step_cfg = res.chain_steps[0]["config"]
    assert step_cfg["source_id"] == "s1"
    assert "target_source_id" not in step_cfg
    assert step_cfg["nested"]["source_id"] == "n1"
    assert "target_source_id" not in step_cfg["nested"]
    assert res.default_settings is not None
    assert res.default_settings["source_id"] == "d1"
    assert "target_source_id" not in res.default_settings
    assert res.default_settings["child"]["source_id"] == "d2"
    assert "target_source_id" not in res.default_settings["child"]


def test_build_chain_config_for_step_normalizes_nested_scope_keys():
    chain = SimpleNamespace(
        id=uuid4(),
        chain_steps=[
            {
                "trigger_condition": "on_complete",
                "trigger_thresholds": {},
            },
            {
                "step_name": "Step 2",
                "job_type": "analysis",
                "goal_template": "Run {task}",
                "config": {
                    "target_source_id": "step2",
                    "nested": {"target_source_id": "step2-nested"},
                },
                "trigger_condition": "on_complete",
            },
        ],
    )
    defaults = {
        "target_source_id": "default-src",
        "inherited": {"target_source_id": "default-nested"},
    }

    cfg = _build_chain_config_for_step(chain, 0, {"task": "checks"}, defaults)

    assert isinstance(cfg, dict)
    child = (cfg.get("child_jobs") or [])[0]
    child_cfg = child["config"]
    assert child_cfg["source_id"] == "default-src"
    assert "target_source_id" not in child_cfg
    assert child_cfg["nested"]["source_id"] == "step2-nested"
    assert "target_source_id" not in child_cfg["nested"]
    assert child_cfg["inherited"]["source_id"] == "default-nested"
    assert "target_source_id" not in child_cfg["inherited"]


def test_extract_launch_mode_normalizes_case_and_whitespace():
    cfg = {"launch_mode": "  QUICK_START_CLAUDE_BACKEND "}

    assert _extract_launch_mode(cfg) == "quick_start_claude_backend"


def test_matches_launch_mode_filter_handles_empty_and_mismatch():
    cfg = {"launch_mode": "quick_start_claude_backend"}

    assert _matches_launch_mode_filter(cfg, "") is True
    assert _matches_launch_mode_filter(cfg, "quick_start_claude_backend") is True
    assert _matches_launch_mode_filter(cfg, "manual") is False


def test_matches_launch_mode_filter_supports_none_bucket():
    cfg_with_mode = {"launch_mode": "quick_start_claude_backend"}
    cfg_without_mode = {}

    assert _matches_launch_mode_filter(cfg_without_mode, "__none__") is True
    assert _matches_launch_mode_filter(cfg_without_mode, "none") is True
    assert _matches_launch_mode_filter(cfg_without_mode, "manual") is True
    assert _matches_launch_mode_filter(cfg_with_mode, "__none__") is False


def test_is_none_launch_mode_matches_manual_tokens():
    assert _is_none_launch_mode("") is True
    assert _is_none_launch_mode("manual") is True
    assert _is_none_launch_mode("none") is True
    assert _is_none_launch_mode("quick_start_claude_backend") is False


def test_build_launch_mode_counts_aggregates_non_empty_modes():
    cfgs = [
        {"launch_mode": "quick_start_claude_backend"},
        {"launch_mode": " quick_start_claude_backend "},
        {"launch_mode": "quick_start_repo_bug_triage"},
        {"launch_mode": "manual"},
        {"launch_mode": "none"},
        {},
        None,
    ]

    counts = _build_launch_mode_counts(cfgs)

    assert counts["quick_start_claude_backend"] == 2
    assert counts["quick_start_repo_bug_triage"] == 1
    assert "manual" not in counts


def test_build_launch_mode_stats_returns_counts_and_none_bucket():
    cfgs = [
        {"launch_mode": "quick_start_claude_backend"},
        {"launch_mode": "quick_start_repo_bug_triage"},
        {"launch_mode": "manual"},
        {"launch_mode": "none"},
        {},
        None,
    ]

    counts, none_count = _build_launch_mode_stats(cfgs)

    assert counts == {"quick_start_claude_backend": 1, "quick_start_repo_bug_triage": 1}
    assert none_count == 4


def test_append_launch_log_if_present_adds_log_entry():
    job = SimpleNamespace(
        config={
            "launch_mode": "quick_start_claude_backend",
            "quick_start": {"profile": "claude_backend", "version": "v1", "source_name": "Repo Source", "source_type": "github"},
        },
        execution_log=[],
        iteration=0,
    )

    def _add_log_entry(entry):
        if not isinstance(job.execution_log, list):
            job.execution_log = []
        job.execution_log.append(entry)

    job.add_log_entry = _add_log_entry

    changed = _append_launch_log_if_present(job)  # type: ignore[arg-type]

    assert changed is True
    assert len(job.execution_log) == 1
    row = job.execution_log[0]
    assert row.get("phase") == "launch"
    assert row.get("action") == "job_launch"
    result = row.get("result") or {}
    assert result.get("launch_mode") == "quick_start_claude_backend"
    assert result.get("source_id") is None
    assert result.get("search_query") is None
    assert int(result.get("commands_count") or 0) == 0
    assert int(result.get("file_paths_count") or 0) == 0
    assert result.get("relaunch_from_job_id") is None


def test_append_launch_log_if_present_keeps_relaunch_provenance():
    job = SimpleNamespace(
        config={
            "launch_mode": "quick_start_claude_backend",
            "relaunch_from_job_id": "00000000-0000-0000-0000-000000000abc",
            "quick_start": {"profile": "claude_backend", "version": "v1"},
        },
        execution_log=[],
        iteration=0,
    )

    def _add_log_entry(entry):
        if not isinstance(job.execution_log, list):
            job.execution_log = []
        job.execution_log.append(entry)

    job.add_log_entry = _add_log_entry

    changed = _append_launch_log_if_present(job)  # type: ignore[arg-type]
    assert changed is True
    result = (job.execution_log[0] or {}).get("result") or {}
    assert result.get("relaunch_from_job_id") == "00000000-0000-0000-0000-000000000abc"


def test_build_repo_bug_triage_relaunch_request_preserves_failure_context():
    job = SimpleNamespace(
        id="00000000-0000-0000-0000-000000000abc",
        name="Repo Bug Triage - 2026-03-23",
        goal="Triage and repair the reported frontend bug. Symptom: Save fails",
        config={
            "launch_mode": "quick_start_repo_bug_triage",
            "source_id": "00000000-0000-0000-0000-000000000111",
            "failure_symptom": "Save fails",
            "scope": "frontend",
            "search_query": "frontend Save fails",
            "commands": ["CI=true npm test -- --watchAll=false"],
            "file_paths": ["frontend/src/pages/DocumentsPage.tsx"],
            "error_output": "TypeError: save is undefined",
        },
    )

    request = _build_quick_start_repo_bug_triage_relaunch_request(job)

    assert request is not None
    assert str(request.source_id) == "00000000-0000-0000-0000-000000000111"
    assert request.failure_symptom == "Save fails"
    assert request.scope == "frontend"
    assert request.commands == ["CI=true npm test -- --watchAll=false"]
    assert request.file_paths == ["frontend/src/pages/DocumentsPage.tsx"]
    assert request.error_output == "TypeError: save is undefined"
    assert request.config_overrides is not None
    assert request.config_overrides.get("relaunch_from_job_id") == "00000000-0000-0000-0000-000000000abc"


def test_build_bug_triage_swarm_relaunch_request_preserves_failure_context():
    job = SimpleNamespace(
        id="00000000-0000-0000-0000-000000000bbb",
        name="Bug Triage Swarm - 2026-03-23",
        goal="Run a coding swarm for the bug",
        config={
            "launch_mode": "quick_start_bug_triage_swarm",
            "source_id": "00000000-0000-0000-0000-000000000111",
            "failure_symptom": "Save fails",
            "scope": "frontend",
            "search_query": "frontend Save fails",
            "commands": ["CI=true npm --prefix frontend test -- --watchAll=false"],
            "file_paths": ["frontend/src/pages/DocumentsPage.tsx"],
            "error_output": "TypeError: save is undefined",
            "quick_start": {"max_agents": 4},
            "coding_swarm_enabled": True,
        },
    )

    request = _build_quick_start_bug_triage_swarm_relaunch_request(job)

    assert request is not None
    assert str(request.source_id) == "00000000-0000-0000-0000-000000000111"
    assert request.failure_symptom == "Save fails"
    assert request.scope == "frontend"
    assert request.max_agents == 4
    assert request.commands == ["CI=true npm --prefix frontend test -- --watchAll=false"]
    assert request.file_paths == ["frontend/src/pages/DocumentsPage.tsx"]
    assert request.error_output == "TypeError: save is undefined"
    assert request.config_overrides is not None
    assert request.config_overrides.get("relaunch_from_job_id") == "00000000-0000-0000-0000-000000000bbb"


def test_build_repo_bug_triage_refined_retry_request_carries_recovery_context():
    job = SimpleNamespace(
        id="00000000-0000-0000-0000-000000000abc",
        name="Repo Bug Triage - 2026-03-23",
        goal="Triage and repair the reported frontend bug. Symptom: Save fails",
        status="failed",
        results={
            "code_patch_execution": {
                "recovery": {
                    "retry_reason": "Verification failed and needs a refined retry.",
                    "resume_hint": "Resume verification from the paused job state.",
                }
            },
            "experiment_run": {
                "ok": False,
                "final_phase": "fallback",
                "failed_commands": ["CI=true npm --prefix frontend test -- --watchAll=false"],
                "runs": [
                    {
                        "phase": "fallback",
                        "command": "CI=true npm --prefix frontend test -- --watchAll=false",
                        "ok": False,
                        "stderr": "TypeError: saveDocument is not a function",
                    }
                ],
            },
            "execution_strategy": {
                "execution_graph": {
                    "graph_health": {"reasons": ["fallback verification still failing"]},
                }
            },
        },
        config={
            "launch_mode": "quick_start_repo_bug_triage",
            "source_id": "00000000-0000-0000-0000-000000000111",
            "failure_symptom": "Save fails",
            "scope": "frontend",
            "search_query": "frontend Save fails",
            "commands": ["CI=true npm --prefix frontend test -- --watchAll=false"],
            "file_paths": ["frontend/src/pages/DocumentsPage.tsx"],
            "error_output": "TypeError: save is undefined",
        },
    )

    request = _build_quick_start_repo_bug_triage_relaunch_request(job, retry_strategy="refined_retry")

    assert request is not None
    assert request.config_overrides is not None
    assert request.config_overrides.get("relaunch_from_job_id") == "00000000-0000-0000-0000-000000000abc"
    recovery = request.config_overrides.get("coding_recovery") or {}
    assert recovery.get("strategy") == "refined_retry"
    assert recovery.get("retry_reason") == "Verification failed and needs a refined retry."
    assert recovery.get("last_failed_commands") == ["CI=true npm --prefix frontend test -- --watchAll=false"]
    assert request.error_output == "TypeError: saveDocument is not a function"


def test_append_launch_log_if_present_includes_scope_and_counts():
    job = SimpleNamespace(
        config={
            "launch_mode": "quick_start_claude_backend",
            "source_id": "00000000-0000-0000-0000-000000000111",
            "search_query": "backend api",
            "commands": ["pytest -q", "ruff check ."],
            "file_paths": ["backend/app/api/endpoints/agent_jobs.py"],
            "quick_start": {"profile": "claude_backend", "version": "v1"},
        },
        execution_log=[],
        iteration=0,
    )

    def _add_log_entry(entry):
        if not isinstance(job.execution_log, list):
            job.execution_log = []
        job.execution_log.append(entry)

    job.add_log_entry = _add_log_entry
    changed = _append_launch_log_if_present(job)  # type: ignore[arg-type]
    assert changed is True
    result = (job.execution_log[0] or {}).get("result") or {}
    assert result.get("source_id") == "00000000-0000-0000-0000-000000000111"
    assert result.get("search_query") == "backend api"
    assert int(result.get("commands_count") or 0) == 2
    assert int(result.get("file_paths_count") or 0) == 1


def test_find_unsafe_commands_detects_destructive_patterns():
    commands = [
        "python -m pytest -q",
        "rm -rf /tmp/workdir",
        "sudo reboot",
        "echo hello",
    ]

    blocked = _find_unsafe_commands(commands)

    assert "rm -rf /tmp/workdir" in blocked
    assert "sudo reboot" in blocked
    assert "python -m pytest -q" not in blocked


def test_extract_source_id_from_config_normalizes_legacy_key():
    cfg = {"target_source_id": "00000000-0000-0000-0000-000000000777"}
    assert _extract_source_id_from_config(cfg) == "00000000-0000-0000-0000-000000000777"


def test_build_quick_start_relaunch_request_builds_payload():
    job = SimpleNamespace(
        id="00000000-0000-0000-0000-000000000999",
        name="Backend Loop",
        goal="Fix flaky tests",
        config={
            "launch_mode": "quick_start_claude_backend",
            "source_id": "00000000-0000-0000-0000-000000000111",
            "search_query": "pytest",
            "commands": ["pytest -q", "pytest -q"],
            "file_paths": ["backend/app/api/endpoints/agent_jobs.py", "../bad/path"],
            "temperature": 0.2,
            "quick_start": {"profile": "claude_backend", "version": "v1"},
        },
    )

    req = _build_quick_start_relaunch_request(job)  # type: ignore[arg-type]

    assert req is not None
    assert str(req.source_id) == "00000000-0000-0000-0000-000000000111"
    assert req.goal == "Fix flaky tests"
    assert req.commands == ["pytest -q"]
    assert req.file_paths == ["backend/app/api/endpoints/agent_jobs.py"]
    assert isinstance(req.config_overrides, dict)
    assert req.config_overrides.get("temperature") == 0.2
    assert req.config_overrides.get("relaunch_from_job_id") == "00000000-0000-0000-0000-000000000999"
    assert "quick_start" not in req.config_overrides


def test_build_quick_start_relaunch_request_returns_none_for_non_quick_start():
    job = SimpleNamespace(
        name="Manual Job",
        goal="Do thing",
        config={"launch_mode": "manual", "source_id": "00000000-0000-0000-0000-000000000111"},
    )

    req = _build_quick_start_relaunch_request(job)  # type: ignore[arg-type]
    assert req is None


def test_build_quick_start_role_workflow_relaunch_request_builds_payload():
    job = SimpleNamespace(
        id="00000000-0000-0000-0000-000000000555",
        name="Role Workflow",
        goal="Investigate regression root causes",
        config={
            "launch_mode": "quick_start_role_workflow",
            "swarm_roles": ["researcher_documents", "analyst", "synthesizer"],
            "swarm_max_agents": 3,
            "memory": {
                "profile": "balanced",
                "extract_on_statuses": ["completed"],
                "failed_extraction_types": ["lesson", "pattern"],
                "completed_extraction_types": ["finding", "summary"],
            },
            "approval_checkpoints": {"enabled": True},
            "quick_start": {"profile": "role_workflow", "version": "v1", "execution_mode": "adaptive"},
            "temperature": 0.3,
        },
    )

    req = _build_quick_start_role_workflow_relaunch_request(job)  # type: ignore[arg-type]

    assert req is not None
    assert req.goal == "Investigate regression root causes"
    assert req.max_agents == 3
    assert req.roles == ["researcher_documents", "analyst", "synthesizer"]
    assert req.memory_profile == "balanced"
    assert req.approval_mode == "high_impact"
    assert req.execution_mode == "adaptive"
    assert req.extract_memory_on_failure is False
    assert req.memory_failed_types == ["lesson", "pattern"]
    assert req.memory_completed_types == ["finding", "summary"]
    assert isinstance(req.config_overrides, dict)
    assert req.config_overrides.get("temperature") == 0.3
    assert req.config_overrides.get("relaunch_from_job_id") == "00000000-0000-0000-0000-000000000555"


def test_build_quick_start_role_workflow_relaunch_request_returns_none_for_other_modes():
    job = SimpleNamespace(
        name="Manual Job",
        goal="Do thing",
        config={"launch_mode": "manual"},
    )

    req = _build_quick_start_role_workflow_relaunch_request(job)  # type: ignore[arg-type]
    assert req is None


def test_extract_relaunch_parent_job_id_parses_uuid():
    parent_id = str(uuid4())
    cfg = {"relaunch_from_job_id": parent_id}
    assert str(_extract_relaunch_parent_job_id(cfg)) == parent_id


def test_build_relaunch_children_counts_aggregates_by_parent():
    p1 = uuid4()
    p2 = uuid4()
    rows = [
        (uuid4(), {"relaunch_from_job_id": str(p1)}),
        (uuid4(), {"relaunch_from_job_id": str(p1)}),
        (uuid4(), {"relaunch_from_job_id": str(p2)}),
        (uuid4(), {"launch_mode": "quick_start_claude_backend"}),
    ]
    counts = _build_relaunch_children_counts(rows)  # type: ignore[arg-type]
    assert counts[p1] == 2
    assert counts[p2] == 1


def test_build_relaunch_children_counts_for_user_aggregates_sql_rows():
    p1 = uuid4()
    p2 = uuid4()
    rows = [
        (str(p1), 3),
        (str(p2), 1),
        ("not-a-uuid", 8),
        ("", 4),
        (None, 2),
    ]

    class _Result:
        def all(self):
            return rows

    class _Db:
        async def execute(self, _query):
            return _Result()

    counts = asyncio.run(
        _build_relaunch_children_counts_for_user(  # type: ignore[arg-type]
            _Db(),
            user_id=uuid4(),
        )
    )

    assert counts[p1] == 3
    assert counts[p2] == 1
    assert len(counts) == 2


def test_build_relaunch_lineage_returns_ancestors_and_descendants():
    root_id = uuid4()
    mid_id = uuid4()
    leaf_id = uuid4()
    side_id = uuid4()
    t0 = datetime.utcnow()

    root = SimpleNamespace(
        id=root_id,
        name="root",
        status="completed",
        created_at=t0,
        config={"launch_mode": "quick_start_claude_backend"},
    )
    mid = SimpleNamespace(
        id=mid_id,
        name="mid",
        status="completed",
        created_at=t0 + timedelta(minutes=1),
        config={"launch_mode": "quick_start_claude_backend", "relaunch_from_job_id": str(root_id)},
    )
    leaf = SimpleNamespace(
        id=leaf_id,
        name="leaf",
        status="running",
        created_at=t0 + timedelta(minutes=2),
        config={"launch_mode": "quick_start_claude_backend", "relaunch_from_job_id": str(mid_id)},
    )
    side = SimpleNamespace(
        id=side_id,
        name="side",
        status="failed",
        created_at=t0 + timedelta(minutes=3),
        config={"launch_mode": "quick_start_claude_backend", "relaunch_from_job_id": str(root_id)},
    )

    jobs_by_id = {root.id: root, mid.id: mid, leaf.id: leaf, side.id: side}
    lineage = _build_relaunch_lineage(mid, jobs_by_id)  # type: ignore[arg-type]

    assert lineage.job_id == mid_id
    assert lineage.root_job_id == root_id
    assert lineage.parent_job_id == root_id
    assert [n.id for n in lineage.ancestors] == [root_id]
    assert set(n.id for n in lineage.descendants) == {leaf_id}


def test_build_relaunch_lineage_latest_child_is_newest_by_created_at():
    root_id = uuid4()
    child_a_id = uuid4()
    child_b_id = uuid4()
    grandchild_old_id = uuid4()
    t0 = datetime.utcnow()

    root = SimpleNamespace(
        id=root_id,
        name="root",
        status="completed",
        created_at=t0,
        config={"launch_mode": "quick_start_claude_backend"},
    )
    child_a = SimpleNamespace(
        id=child_a_id,
        name="child-a",
        status="completed",
        created_at=t0 + timedelta(minutes=1),
        config={"launch_mode": "quick_start_claude_backend", "relaunch_from_job_id": str(root_id)},
    )
    child_b = SimpleNamespace(
        id=child_b_id,
        name="child-b-newest",
        status="completed",
        created_at=t0 + timedelta(minutes=3),
        config={"launch_mode": "quick_start_claude_backend", "relaunch_from_job_id": str(root_id)},
    )
    grandchild_old = SimpleNamespace(
        id=grandchild_old_id,
        name="grandchild-old",
        status="failed",
        created_at=t0 + timedelta(minutes=2),
        config={"launch_mode": "quick_start_claude_backend", "relaunch_from_job_id": str(child_a_id)},
    )

    jobs_by_id = {
        root.id: root,
        child_a.id: child_a,
        child_b.id: child_b,
        grandchild_old.id: grandchild_old,
    }
    lineage = _build_relaunch_lineage(root, jobs_by_id)  # type: ignore[arg-type]
    assert lineage.latest_child_job_id == child_b_id


def test_build_relaunch_lineage_respects_limits():
    root_id = uuid4()
    t0 = datetime.utcnow()
    root = SimpleNamespace(
        id=root_id,
        name="root",
        status="completed",
        created_at=t0,
        config={"launch_mode": "quick_start_claude_backend"},
    )
    c1 = SimpleNamespace(
        id=uuid4(),
        name="c1",
        status="completed",
        created_at=t0 + timedelta(minutes=1),
        config={"launch_mode": "quick_start_claude_backend", "relaunch_from_job_id": str(root_id)},
    )
    c2 = SimpleNamespace(
        id=uuid4(),
        name="c2",
        status="completed",
        created_at=t0 + timedelta(minutes=2),
        config={"launch_mode": "quick_start_claude_backend", "relaunch_from_job_id": str(c1.id)},
    )
    c3 = SimpleNamespace(
        id=uuid4(),
        name="c3",
        status="completed",
        created_at=t0 + timedelta(minutes=3),
        config={"launch_mode": "quick_start_claude_backend", "relaunch_from_job_id": str(c2.id)},
    )
    jobs_by_id = {root.id: root, c1.id: c1, c2.id: c2, c3.id: c3}

    lineage = _build_relaunch_lineage(root, jobs_by_id, max_descendants=2)  # type: ignore[arg-type]
    assert len(lineage.descendants) == 2
    assert lineage.descendants_truncated is True

    lineage2 = _build_relaunch_lineage(c3, jobs_by_id, max_ancestors=2)  # type: ignore[arg-type]
    assert len(lineage2.ancestors) == 2
    assert lineage2.ancestors_truncated is True
