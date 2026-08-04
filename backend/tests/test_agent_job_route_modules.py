"""Architecture regressions for modular autonomous-job API routers."""

import ast
from pathlib import Path

from fastapi.routing import APIRoute

from app.api.endpoints import agent_jobs
from app.modules.autonomy.api import (
    ai_hub_feedback,
    chain_definitions,
    chain_execution,
    checkpoint_follow_up_actions,
    checkpoint_job_actions,
    checkpoint_queue,
    decision_trace_actions,
    decision_trace_queries,
    decision_trace_reporting,
    decision_trace_views,
    domain_research_promotion,
    follow_up_queue_actions,
    job_actions,
    job_checkpoints,
    job_crud,
    job_exports,
    job_feedback,
    job_logs,
    job_memories,
    job_progress,
    job_queries,
    job_step_events,
    job_templates,
    quick_starts,
)
from app.modules.autonomy.api import relaunch_lineage as relaunch_lineage_routes
from app.modules.autonomy.api import swarm_analytics, swarm_outcomes
from app.modules.autonomy.application import (
    checkpoint_queue_composer,
    checkpoint_queue_inbox,
    checkpoint_queue_jobs,
    checkpoint_queue_monitors,
    checkpoint_queue_portfolios,
    checkpoint_queue_priority,
    checkpoint_queue_profiles,
    coding_swarm_relaunch,
    decision_trace_events,
    decision_trace_follow_up_targets,
    decision_trace_jobs,
    decision_trace_loader,
    decision_trace_monitors,
    decision_trace_opportunities,
    decision_trace_queue,
    decision_trace_validation,
    domain_research_promotion_seed,
    feedback_presenters,
    follow_up_inbox_relaunch,
    follow_up_learning_profiles,
    follow_up_policy,
    follow_up_queue_dispatcher,
    follow_up_queue_events,
    follow_up_queue_inbox,
    follow_up_queue_portfolios,
    follow_up_queue_profiles,
    follow_up_recommendations,
    job_action_checkpoint_decisions,
    job_action_checkpoint_resume,
    job_action_checkpoints,
    job_action_contracts,
    job_action_interventions,
    job_action_lifecycle,
    job_action_plan_state,
    job_action_recovery,
    job_action_state_machine,
    job_action_swarm,
    job_operator_events,
    job_presenters,
    memory_presenters,
    operator_queue_context,
    portfolio_queue_state,
    quick_start_builders,
    quick_start_relaunch,
    relaunch_dispatcher,
    relaunch_lineage,
    repair_verification,
    swarm_outcome_cases,
    swarm_summaries,
    template_recommendations,
)


def _route_signatures(router):
    return {
        (route.path, method)
        for route in router.routes
        if isinstance(route, APIRoute)
        for method in route.methods
    }


def test_chain_definition_router_owns_expected_contracts():
    signatures = _route_signatures(chain_definitions.router)

    assert signatures == {
        ("/chains", "GET"),
        ("/chains", "POST"),
        ("/chains/{chain_id}", "GET"),
        ("/chains/{chain_id}", "PATCH"),
        ("/chains/{chain_id}", "DELETE"),
    }


def test_agent_jobs_composes_chain_routes_before_dynamic_job_route():
    signatures = _route_signatures(agent_jobs.router)
    signature_rows = [
        (route.path, method)
        for route in agent_jobs.router.routes
        if isinstance(route, APIRoute)
        for method in route.methods
    ]
    paths = [
        route.path for route in agent_jobs.router.routes if isinstance(route, APIRoute)
    ]

    assert _route_signatures(chain_definitions.router) <= signatures
    for signature in _route_signatures(chain_definitions.router):
        assert signature_rows.count(signature) == 1
    for signature in _route_signatures(agent_jobs.chain_execution_api.router):
        assert signature_rows.count(signature) == 1
    for signature in _route_signatures(agent_jobs.quick_start_api.router):
        assert signature_rows.count(signature) == 1
    assert paths.index("/chains") < paths.index("/{job_id}")
    assert paths.index("/from-chain") < paths.index("/{job_id}")
    assert paths.index("/quick-start/domain-research") < paths.index("/{job_id}")


def test_agent_jobs_preserves_chain_handler_compatibility_exports():
    assert agent_jobs.list_chain_definitions is chain_definitions.list_chain_definitions
    assert (
        agent_jobs.create_chain_definition is chain_definitions.create_chain_definition
    )
    assert agent_jobs.get_chain_definition is chain_definitions.get_chain_definition
    assert (
        agent_jobs.update_chain_definition is chain_definitions.update_chain_definition
    )
    assert (
        agent_jobs.delete_chain_definition is chain_definitions.delete_chain_definition
    )
    assert (
        agent_jobs.create_job_from_chain
        is agent_jobs.chain_execution_api.create_job_from_chain
    )
    assert (
        agent_jobs.get_chain_status is agent_jobs.chain_execution_api.get_chain_status
    )
    assert (
        agent_jobs.save_job_as_chain_definition
        is agent_jobs.chain_execution_api.save_job_as_chain_definition
    )


def test_chain_router_does_not_import_legacy_agent_jobs_module():
    for module in (chain_definitions, chain_execution, quick_starts):
        source_path = Path(module.__file__)
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        imports = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        }

        assert "app.api.endpoints.agent_jobs" not in imports


def test_chain_execution_router_owns_expected_contracts():
    signatures = _route_signatures(agent_jobs.chain_execution_api.router)

    assert signatures == {
        ("/from-chain", "POST"),
        ("/{job_id}/chain-status", "GET"),
        ("/{job_id}/save-as-chain", "POST"),
    }


def test_quick_start_router_owns_expected_contracts():
    signatures = _route_signatures(agent_jobs.quick_start_api.router)

    assert signatures == {
        ("/quick-start/claude-backend", "POST"),
        ("/quick-start/domain-research", "POST"),
        ("/quick-start/repo-bug-triage", "POST"),
        ("/quick-start/bug-triage-swarm", "POST"),
        ("/quick-start/build-break-swarm", "POST"),
        ("/quick-start/frontend-regression-swarm", "POST"),
        ("/quick-start/role-workflow", "POST"),
    }


def test_agent_jobs_preserves_quick_start_handler_exports():
    assert (
        agent_jobs.quick_start_claude_backend_job
        is agent_jobs.quick_start_api.quick_start_claude_backend_job
    )
    assert (
        agent_jobs.quick_start_domain_research_job
        is agent_jobs.quick_start_api.quick_start_domain_research_job
    )
    assert (
        agent_jobs.quick_start_repo_bug_triage_job
        is agent_jobs.quick_start_api.quick_start_repo_bug_triage_job
    )
    assert (
        agent_jobs.quick_start_bug_triage_swarm_job
        is agent_jobs.quick_start_api.quick_start_bug_triage_swarm_job
    )
    assert (
        agent_jobs.quick_start_build_break_swarm_job
        is agent_jobs.quick_start_api.quick_start_build_break_swarm_job
    )
    assert (
        agent_jobs.quick_start_frontend_regression_swarm_job
        is agent_jobs.quick_start_api.quick_start_frontend_regression_swarm_job
    )
    assert (
        agent_jobs.quick_start_role_workflow_job
        is agent_jobs.quick_start_api.quick_start_role_workflow_job
    )


def test_agent_jobs_preserves_quick_start_builder_exports():
    assert (
        agent_jobs._build_quick_start_claude_backend_config
        is quick_start_builders.build_claude_backend_config
    )
    assert (
        agent_jobs._build_domain_research_goal
        is quick_start_builders.build_domain_research_goal
    )
    assert (
        agent_jobs._build_quick_start_domain_research_config
        is quick_start_builders.build_domain_research_config
    )
    assert (
        agent_jobs._build_repo_bug_triage_goal
        is quick_start_builders.build_repo_bug_triage_goal
    )
    assert (
        agent_jobs._build_quick_start_repo_bug_triage_config
        is quick_start_builders.build_repo_bug_triage_config
    )
    assert agent_jobs._coerce_bool is quick_start_builders.coerce_bool
    assert (
        agent_jobs._normalize_swarm_roles is quick_start_builders.normalize_swarm_roles
    )
    assert (
        agent_jobs._build_quick_start_role_workflow_config
        is quick_start_builders.build_role_workflow_config
    )


def test_quick_start_builders_respect_application_boundary():
    source_path = Path(quick_start_builders.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }

    assert "app.api.endpoints.agent_jobs" not in imports
    assert not any(
        module == "fastapi" or module.startswith("fastapi.") for module in imports
    )
    assert not any(
        module == "app.tasks" or module.startswith("app.tasks.") for module in imports
    )


def test_domain_research_promotion_seed_respects_application_boundary():
    source = Path(domain_research_promotion_seed.__file__).read_text(encoding="utf-8")
    assert "app.api.endpoints.agent_jobs" not in source
    assert "fastapi" not in source
    assert callable(domain_research_promotion_seed.build_domain_research_promotion_seed)


def test_agent_jobs_preserves_general_quick_start_relaunch_exports():
    assert (
        agent_jobs._build_quick_start_relaunch_request
        is quick_start_relaunch.build_claude_backend_relaunch_request
    )
    assert (
        agent_jobs._build_quick_start_domain_research_relaunch_request
        is quick_start_relaunch.build_domain_research_relaunch_request
    )
    assert (
        agent_jobs._build_quick_start_role_workflow_relaunch_request
        is quick_start_relaunch.build_role_workflow_relaunch_request
    )


def test_quick_start_relaunch_respects_application_boundary():
    source_path = Path(quick_start_relaunch.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }

    assert "app.api.endpoints.agent_jobs" not in imports
    assert not any(
        module == "fastapi" or module.startswith("fastapi.") for module in imports
    )
    assert not any(
        module == "app.tasks" or module.startswith("app.tasks.") for module in imports
    )
    assert not any(
        module == "app.models" or module.startswith("app.models.") for module in imports
    )


def test_agent_jobs_preserves_coding_swarm_relaunch_exports():
    assert (
        agent_jobs._build_quick_start_repo_bug_triage_relaunch_request
        is coding_swarm_relaunch.build_repo_bug_triage_relaunch_request
    )
    assert (
        agent_jobs._build_quick_start_bug_triage_swarm_relaunch_request
        is coding_swarm_relaunch.build_bug_triage_swarm_relaunch_request
    )
    assert (
        agent_jobs._build_quick_start_build_break_swarm_relaunch_request
        is coding_swarm_relaunch.build_build_break_swarm_relaunch_request
    )
    assert (
        agent_jobs._build_quick_start_frontend_regression_swarm_relaunch_request
        is coding_swarm_relaunch.build_frontend_regression_swarm_relaunch_request
    )
    assert (
        agent_jobs._build_quick_start_coding_swarm_relaunch_request
        is coding_swarm_relaunch.build_coding_swarm_relaunch_request
    )
    assert (
        agent_jobs._extract_repo_bug_triage_coding_recovery
        is coding_swarm_relaunch.extract_repo_bug_triage_coding_recovery
    )


def test_coding_swarm_relaunch_respects_application_boundary():
    source_path = Path(coding_swarm_relaunch.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }

    assert "app.api.endpoints.agent_jobs" not in imports
    assert not any(
        module == "fastapi" or module.startswith("fastapi.") for module in imports
    )
    assert not any(
        module == "app.tasks" or module.startswith("app.tasks.") for module in imports
    )
    assert not any(
        module == "app.models" or module.startswith("app.models.") for module in imports
    )


def test_relaunch_dispatcher_respects_application_boundary():
    source_path = Path(relaunch_dispatcher.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }

    assert "app.api.endpoints.agent_jobs" not in imports
    assert not any(
        module == "fastapi" or module.startswith("fastapi.") for module in imports
    )
    assert not any(
        module == "app.tasks" or module.startswith("app.tasks.") for module in imports
    )
    assert not any(
        module == "app.models" or module.startswith("app.models.") for module in imports
    )


def test_relaunch_lineage_router_owns_expected_contract():
    assert (
        "/{job_id}/relaunch-lineage",
        "GET",
    ) in _route_signatures(relaunch_lineage_routes.router)
    assert (
        agent_jobs.get_agent_job_relaunch_lineage
        is relaunch_lineage_routes.get_agent_job_relaunch_lineage
    )
    assert agent_jobs._build_relaunch_lineage is relaunch_lineage.build_lineage
    assert (
        agent_jobs._extract_relaunch_parent_job_id
        is relaunch_lineage.extract_parent_job_id
    )


def test_relaunch_lineage_respects_application_boundary():
    source_path = Path(relaunch_lineage.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }

    assert "app.api.endpoints.agent_jobs" not in imports
    assert not any(
        module == "fastapi" or module.startswith("fastapi.") for module in imports
    )
    assert not any(
        module == "app.tasks" or module.startswith("app.tasks.") for module in imports
    )
    assert not any(
        module == "app.models" or module.startswith("app.models.") for module in imports
    )


def test_agent_jobs_preserves_memory_presenter_exports():
    assert agent_jobs._to_int is memory_presenters.to_int
    assert agent_jobs._to_float is memory_presenters.to_float
    assert agent_jobs._to_string_list is memory_presenters.to_string_list
    assert agent_jobs._to_string is memory_presenters.to_string
    assert (
        agent_jobs._build_extract_job_memories_response
        is memory_presenters.build_extract_job_memories_response
    )
    assert (
        agent_jobs._build_job_memory_response
        is memory_presenters.build_job_memory_response
    )
    assert (
        agent_jobs._build_job_memories_list_response
        is memory_presenters.build_job_memories_list_response
    )
    assert (
        agent_jobs._build_memory_search_response
        is memory_presenters.build_memory_search_response
    )
    assert (
        agent_jobs._build_memory_stats_response
        is memory_presenters.build_memory_stats_response
    )
    assert (
        agent_jobs._build_memory_graph_response
        is memory_presenters.build_memory_graph_response
    )


def test_memory_presenters_respect_application_boundary():
    source_path = Path(memory_presenters.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }

    assert "app.api.endpoints.agent_jobs" not in imports
    assert not any(
        module == "fastapi" or module.startswith("fastapi.") for module in imports
    )
    assert not any(
        module == "app.tasks" or module.startswith("app.tasks.") for module in imports
    )
    assert not any(
        module == "app.models" or module.startswith("app.models.") for module in imports
    )


def test_job_memory_router_owns_expected_contracts():
    signatures = _route_signatures(job_memories.router)
    assert ("/{job_id}/memories", "GET") in signatures
    assert ("/{job_id}/memories", "POST") in signatures
    assert ("/{job_id}/memories", "DELETE") in signatures
    assert ("/{job_id}/memories/extract", "POST") in signatures
    assert ("/{job_id}/memories/graph", "GET") in signatures
    assert ("/memory/graph", "GET") in signatures
    assert ("/memory/stats", "GET") in signatures
    assert ("/memory/search", "GET") in signatures
    assert agent_jobs.get_job_memories is job_memories.get_job_memories
    assert agent_jobs.extract_job_memories is job_memories.extract_job_memories
    assert agent_jobs.create_job_memory is job_memories.create_job_memory
    assert agent_jobs.delete_job_memories is job_memories.delete_job_memories
    assert agent_jobs.get_task_memory_graph is job_memories.get_task_memory_graph
    assert agent_jobs.get_job_memory_graph is job_memories.get_job_memory_graph
    assert agent_jobs.get_memory_stats is job_memories.get_memory_stats
    assert agent_jobs.search_memories is job_memories.search_memories


def test_job_feedback_router_owns_expected_contracts():
    signatures = _route_signatures(job_feedback.router)
    assert ("/{job_id}/feedback", "POST") in signatures
    assert ("/{job_id}/feedback", "GET") in signatures
    assert ("/memory/feedback", "GET") in signatures
    assert (
        agent_jobs.create_agent_job_feedback is job_feedback.create_agent_job_feedback
    )
    assert agent_jobs.list_agent_job_feedback is job_feedback.list_agent_job_feedback
    assert agent_jobs.list_learning_feedback is job_feedback.list_learning_feedback
    assert agent_jobs._sanitize_tool_names is feedback_presenters.sanitize_tool_names
    assert (
        agent_jobs._memory_to_feedback_response
        is feedback_presenters.memory_to_feedback_response
    )


def test_ai_hub_feedback_router_owns_expected_contracts():
    signatures = _route_signatures(ai_hub_feedback.router)
    route = "/{job_id}/ai-hub/recommendation-feedback"
    assert (route, "GET") in signatures
    assert (route, "POST") in signatures
    assert (
        agent_jobs.list_ai_hub_recommendation_feedback
        is ai_hub_feedback.list_ai_hub_recommendation_feedback
    )
    assert (
        agent_jobs.create_ai_hub_recommendation_feedback
        is ai_hub_feedback.create_ai_hub_recommendation_feedback
    )


def test_job_crud_routers_own_expected_contracts_and_preserve_route_order():
    creation_signatures = _route_signatures(agent_jobs.job_creation_api.router)
    record_signatures = _route_signatures(agent_jobs.job_record_api.router)
    assert ("", "POST") in creation_signatures
    assert ("/from-template", "POST") in creation_signatures
    assert ("/{job_id}", "GET") in record_signatures
    assert ("/{job_id}", "PATCH") in record_signatures
    assert ("/{job_id}", "DELETE") in record_signatures
    assert agent_jobs.create_agent_job is agent_jobs.job_creation_api.create_agent_job
    assert (
        agent_jobs.create_job_from_template
        is agent_jobs.job_creation_api.create_job_from_template
    )
    assert agent_jobs.get_agent_job is agent_jobs.job_record_api.get_agent_job
    assert agent_jobs.update_agent_job is agent_jobs.job_record_api.update_agent_job
    assert agent_jobs.delete_agent_job is agent_jobs.job_record_api.delete_agent_job

    get_paths = [
        route.path
        for route in agent_jobs.router.routes
        if isinstance(route, APIRoute) and "GET" in route.methods
    ]
    assert get_paths.index("/templates") < get_paths.index("/{job_id}")
    assert callable(job_crud.build_job_creation_api)
    assert callable(job_crud.build_job_record_api)


def test_job_query_router_owns_expected_contracts_and_precedes_detail():
    signatures = _route_signatures(agent_jobs.job_query_api.router)
    assert ("", "GET") in signatures
    assert ("/stats", "GET") in signatures
    assert agent_jobs.list_agent_jobs is agent_jobs.job_query_api.list_agent_jobs
    assert agent_jobs.get_job_stats is agent_jobs.job_query_api.get_job_stats

    get_paths = [
        route.path
        for route in agent_jobs.router.routes
        if isinstance(route, APIRoute) and "GET" in route.methods
    ]
    assert get_paths.index("/stats") < get_paths.index("/{job_id}")
    assert callable(job_queries.build_job_query_api)


def test_job_template_router_owns_expected_contract_and_precedes_detail():
    signatures = _route_signatures(agent_jobs.job_template_api.router)
    assert ("/templates", "GET") in signatures
    assert (
        agent_jobs.list_job_templates is agent_jobs.job_template_api.list_job_templates
    )
    assert callable(job_templates.build_job_template_api)
    assert (
        agent_jobs._score_template_recommendation
        is template_recommendations.score_template_recommendation
    )

    get_paths = [
        route.path
        for route in agent_jobs.router.routes
        if isinstance(route, APIRoute) and "GET" in route.methods
    ]
    assert get_paths.index("/templates") < get_paths.index("/{job_id}")


def test_domain_research_promotion_router_owns_expected_contract():
    signatures = _route_signatures(agent_jobs.domain_research_promotion_api.router)
    assert ("/{job_id}/promote-domain-research", "POST") in signatures
    assert (
        agent_jobs.promote_domain_research_job
        is agent_jobs.domain_research_promotion_api.promote_domain_research_job
    )
    assert callable(domain_research_promotion.build_domain_research_promotion_api)


def test_swarm_analytics_router_owns_expected_contract():
    signatures = _route_signatures(agent_jobs.swarm_analytics_api.router)
    assert ("/swarm-analytics", "GET") in signatures
    assert (
        agent_jobs.get_swarm_analytics
        is agent_jobs.swarm_analytics_api.get_swarm_analytics
    )
    assert callable(swarm_analytics.build_swarm_analytics_api)


def test_swarm_outcomes_router_owns_expected_contract():
    signatures = _route_signatures(agent_jobs.swarm_outcomes_api.router)
    assert ("/swarm-outcomes", "GET") in signatures
    assert (
        agent_jobs.get_swarm_outcomes
        is agent_jobs.swarm_outcomes_api.get_swarm_outcomes
    )
    assert callable(swarm_outcomes.build_swarm_outcomes_api)


def test_checkpoint_queue_router_owns_expected_contract():
    signatures = _route_signatures(agent_jobs.checkpoint_queue_api.router)
    assert ("/checkpoint-queue", "GET") in signatures
    assert (
        agent_jobs.get_checkpoint_queue
        is agent_jobs.checkpoint_queue_api.get_checkpoint_queue
    )
    assert callable(checkpoint_queue.build_checkpoint_queue_api)


def test_checkpoint_queue_job_projection_respects_application_boundary():
    source_path = Path(checkpoint_queue_jobs.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }

    assert "app.api.endpoints.agent_jobs" not in imports
    assert not any(
        module == "fastapi" or module.startswith("fastapi.") for module in imports
    )
    assert callable(checkpoint_queue_jobs.build_job_checkpoint_queue_items)


def test_checkpoint_queue_priority_respects_application_boundary():
    source = Path(checkpoint_queue_priority.__file__).read_text(encoding="utf-8")
    assert "app.api.endpoints.agent_jobs" not in source
    assert "fastapi" not in source
    assert callable(checkpoint_queue_priority.queue_priority_fields)


def test_checkpoint_queue_composer_respects_application_boundary():
    source = Path(checkpoint_queue_composer.__file__).read_text(encoding="utf-8")
    assert "app.api.endpoints.agent_jobs" not in source
    assert "fastapi" not in source
    assert callable(checkpoint_queue_composer.compose_checkpoint_queue)
    assert callable(checkpoint_queue_composer.bind_checkpoint_queue_composer)


def test_checkpoint_queue_monitor_projection_respects_application_boundary():
    source_path = Path(checkpoint_queue_monitors.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }

    assert "app.api.endpoints.agent_jobs" not in imports
    assert not any(
        module == "fastapi" or module.startswith("fastapi.") for module in imports
    )
    assert callable(checkpoint_queue_monitors.build_monitor_checkpoint_queue_items)


def test_checkpoint_queue_inbox_projection_respects_application_boundary():
    source_path = Path(checkpoint_queue_inbox.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }

    assert "app.api.endpoints.agent_jobs" not in imports
    assert not any(
        module == "fastapi" or module.startswith("fastapi.") for module in imports
    )
    assert callable(checkpoint_queue_inbox.build_inbox_checkpoint_queue_items)


def test_checkpoint_queue_portfolio_projection_respects_application_boundary():
    source_path = Path(checkpoint_queue_portfolios.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }

    assert "app.api.endpoints.agent_jobs" not in imports
    assert not any(
        module == "fastapi" or module.startswith("fastapi.") for module in imports
    )
    assert callable(checkpoint_queue_portfolios.build_portfolio_checkpoint_queue_items)


def test_checkpoint_queue_profile_projection_respects_application_boundary():
    source_path = Path(checkpoint_queue_profiles.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }

    assert "app.api.endpoints.agent_jobs" not in imports
    assert not any(
        module == "fastapi" or module.startswith("fastapi.") for module in imports
    )
    assert callable(checkpoint_queue_profiles.build_profile_checkpoint_queue_items)


def test_follow_up_queue_inbox_handler_respects_application_boundary():
    source_path = Path(follow_up_queue_inbox.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }

    assert "app.api.endpoints.agent_jobs" not in imports
    assert not any(
        module == "fastapi" or module.startswith("fastapi.") for module in imports
    )
    assert callable(follow_up_queue_inbox.perform_inbox_follow_up_queue_action)


def test_follow_up_inbox_relaunch_respects_application_boundary():
    source = Path(follow_up_inbox_relaunch.__file__).read_text(encoding="utf-8")
    assert "app.api.endpoints.agent_jobs" not in source
    assert "fastapi" not in source
    assert callable(follow_up_inbox_relaunch.relaunch_inbox_follow_up)


def test_follow_up_learning_profiles_respects_application_boundary():
    source = Path(follow_up_learning_profiles.__file__).read_text(encoding="utf-8")
    assert "app.api.endpoints.agent_jobs" not in source
    assert "fastapi" not in source
    assert (
        agent_jobs._load_follow_up_learning_profile
        is follow_up_learning_profiles.load_follow_up_learning_profile
    )


def test_repair_verification_respects_application_boundary():
    source = Path(repair_verification.__file__).read_text(encoding="utf-8")
    assert "app.api.endpoints.agent_jobs" not in source
    assert "fastapi" not in source
    assert callable(repair_verification.derive_repair_verification_status)


def test_job_operator_events_respects_application_boundary():
    source = Path(job_operator_events.__file__).read_text(encoding="utf-8")
    assert "app.api.endpoints.agent_jobs" not in source
    assert "fastapi" not in source
    assert callable(job_operator_events.record_job_operator_event)


def test_job_action_interventions_respect_application_boundary():
    source = Path(job_action_interventions.__file__).read_text(encoding="utf-8")
    assert "app.api.endpoints.agent_jobs" not in source
    assert "fastapi" not in source
    assert (
        agent_jobs._append_operator_intervention
        is job_action_interventions.append_operator_intervention
    )


def test_job_action_checkpoints_respect_application_boundary():
    source = Path(job_action_checkpoints.__file__).read_text(encoding="utf-8")
    assert "app.api.endpoints.agent_jobs" not in source
    assert "fastapi" not in source
    assert (
        agent_jobs._approval_payload_from_results
        is job_action_checkpoints.approval_payload_from_results
    )
    assert (
        agent_jobs._normalize_checkpoint_action_patch
        is job_action_checkpoints.normalize_checkpoint_action_patch
    )
    assert (
        agent_jobs._apply_checkpoint_action_patch
        is job_action_checkpoints.apply_checkpoint_action_patch
    )
    assert (
        agent_jobs._append_approval_event
        is job_action_checkpoints.append_approval_event
    )
    assert agent_jobs._append_step_event is job_action_checkpoints.append_step_event
    assert (
        agent_jobs._sync_execution_strategy_state
        is job_action_checkpoints.sync_execution_strategy_state
    )


def test_job_action_plan_state_respects_application_boundary():
    source = Path(job_action_plan_state.__file__).read_text(encoding="utf-8")
    assert "app.api.endpoints.agent_jobs" not in source
    assert "fastapi" not in source
    assert (
        agent_jobs._set_current_plan_step_status
        is job_action_plan_state.set_current_plan_step_status
    )


def test_operator_queue_context_respects_application_boundary():
    source = Path(operator_queue_context.__file__).read_text(encoding="utf-8")
    assert "app.api.endpoints.agent_jobs" not in source
    assert "fastapi" not in source
    assert callable(operator_queue_context.clean_text_list)
    assert callable(operator_queue_context.build_operator_queue_context)
    assert agent_jobs._clean_queue_text_list is operator_queue_context.clean_text_list
    assert (
        agent_jobs._build_operator_queue_context
        is operator_queue_context.build_operator_queue_context
    )


def test_portfolio_queue_state_respects_application_boundary():
    source = Path(portfolio_queue_state.__file__).read_text(encoding="utf-8")
    assert "app.api.endpoints.agent_jobs" not in source
    assert "fastapi" not in source
    assert (
        agent_jobs._sync_portfolio_queue_state
        is portfolio_queue_state.sync_portfolio_queue_state
    )


def test_follow_up_queue_dispatcher_respects_application_boundary():
    source = Path(follow_up_queue_dispatcher.__file__).read_text(encoding="utf-8")
    assert "app.api.endpoints.agent_jobs" not in source
    assert "fastapi" not in source
    assert callable(follow_up_queue_dispatcher.dispatch_follow_up_queue_action)


def test_decision_trace_follow_up_targets_respect_application_boundary():
    source = Path(decision_trace_follow_up_targets.__file__).read_text(encoding="utf-8")
    assert "app.api.endpoints.agent_jobs" not in source
    assert "fastapi" not in source
    assert callable(decision_trace_follow_up_targets.resolve_follow_up_target)


def test_follow_up_queue_action_api_owns_http_error_boundary():
    source = Path(follow_up_queue_actions.__file__).read_text(encoding="utf-8")
    assert "app.api.endpoints.agent_jobs" not in source
    assert callable(follow_up_queue_actions.build_follow_up_queue_action_api)
    assert (
        agent_jobs._perform_follow_up_queue_action
        is agent_jobs.follow_up_queue_action_api.perform_follow_up_queue_action
    )


def test_follow_up_queue_events_respects_application_boundary():
    source = Path(follow_up_queue_events.__file__).read_text(encoding="utf-8")
    assert "app.api.endpoints.agent_jobs" not in source
    assert "fastapi" not in source
    assert callable(follow_up_queue_events.record_follow_up_queue_decision)


def test_follow_up_queue_portfolio_handler_respects_application_boundary():
    source_path = Path(follow_up_queue_portfolios.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }

    assert "app.api.endpoints.agent_jobs" not in imports
    assert not any(
        module == "fastapi" or module.startswith("fastapi.") for module in imports
    )
    assert callable(follow_up_queue_portfolios.perform_portfolio_follow_up_queue_action)


def test_follow_up_queue_profile_handler_respects_application_boundary():
    source_path = Path(follow_up_queue_profiles.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }

    assert "app.api.endpoints.agent_jobs" not in imports
    assert not any(
        module == "fastapi" or module.startswith("fastapi.") for module in imports
    )
    assert callable(follow_up_queue_profiles.perform_profile_follow_up_queue_action)


def test_follow_up_policy_respects_application_boundary():
    source_path = Path(follow_up_policy.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }

    assert "app.api.endpoints.agent_jobs" not in imports
    assert not any(
        module == "fastapi" or module.startswith("fastapi.") for module in imports
    )
    assert callable(follow_up_policy.apply_follow_up_policy_on_accept)


def test_monitor_decision_trace_projection_respects_application_boundary():
    source_path = Path(decision_trace_monitors.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert "app.api.endpoints.agent_jobs" not in imports
    assert not any(
        module == "fastapi" or module.startswith("fastapi.") for module in imports
    )
    assert callable(decision_trace_monitors.build_monitor_decision_trace)


def test_decision_trace_loader_respects_application_boundary():
    source_path = Path(decision_trace_loader.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert "app.api.endpoints.agent_jobs" not in imports
    assert not any(
        module == "fastapi" or module.startswith("fastapi.") for module in imports
    )
    assert not any(
        module == "app.tasks" or module.startswith("app.tasks.") for module in imports
    )
    assert callable(decision_trace_loader.load_derived_decision_trace_events)


def test_job_decision_trace_projection_respects_application_boundary():
    source = Path(decision_trace_jobs.__file__).read_text(encoding="utf-8")
    assert "app.api.endpoints.agent_jobs" not in source
    assert "fastapi" not in source
    assert callable(decision_trace_jobs.build_job_decision_trace)


def test_decision_trace_event_factory_respects_application_boundary():
    source = Path(decision_trace_events.__file__).read_text(encoding="utf-8")
    assert "app.api.endpoints.agent_jobs" not in source
    assert "fastapi" not in source
    assert callable(decision_trace_events.decision_trace_event_id)
    assert callable(decision_trace_events.build_decision_trace_event)


def test_queue_decision_trace_projection_respects_application_boundary():
    source = Path(decision_trace_queue.__file__).read_text(encoding="utf-8")
    assert "app.api.endpoints.agent_jobs" not in source
    assert "fastapi" not in source
    assert callable(decision_trace_queue.build_queue_decision_trace)


def test_validation_decision_trace_projection_respects_application_boundary():
    source_path = Path(decision_trace_validation.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert "app.api.endpoints.agent_jobs" not in imports
    assert not any(
        module == "fastapi" or module.startswith("fastapi.") for module in imports
    )
    assert callable(decision_trace_validation.build_validation_decision_trace)


def test_opportunity_decision_trace_policy_respects_application_boundary():
    source = Path(decision_trace_opportunities.__file__).read_text(encoding="utf-8")
    assert "app.api.endpoints.agent_jobs" not in source
    assert "fastapi" not in source
    assert callable(decision_trace_opportunities.classify_opportunity_event)
    assert callable(decision_trace_opportunities.build_opportunity_decision_trace)
    assert callable(decision_trace_opportunities.bind_opportunity_decision_trace)


def test_swarm_outcome_case_projection_respects_application_boundary():
    source = Path(swarm_outcome_cases.__file__).read_text(encoding="utf-8")
    assert "app.api.endpoints.agent_jobs" not in source
    assert "fastapi" not in source
    assert callable(swarm_outcome_cases.derive_swarm_outcome_case)


def test_swarm_summary_projection_respects_application_boundary():
    source = Path(swarm_summaries.__file__).read_text(encoding="utf-8")
    assert "app.api.endpoints.agent_jobs" not in source
    assert "fastapi" not in source
    assert callable(swarm_summaries.extract_swarm_summary)


def test_follow_up_recommendations_respect_application_boundary():
    source = Path(follow_up_recommendations.__file__).read_text(encoding="utf-8")
    assert "app.api.endpoints.agent_jobs" not in source
    assert "fastapi" not in source
    assert callable(follow_up_recommendations.score_follow_up_action)
    assert callable(follow_up_recommendations.build_follow_up_actions)


def test_job_presenter_respects_application_boundary():
    source = Path(job_presenters.__file__).read_text(encoding="utf-8")
    assert "app.api.endpoints.agent_jobs" not in source
    assert "fastapi" not in source
    assert callable(job_presenters.present_job)


def test_checkpoint_job_action_router_owns_expected_contract():
    signatures = _route_signatures(agent_jobs.checkpoint_job_action_api.router)
    assert ("/checkpoint-queue/bulk-action", "POST") in signatures
    assert (
        agent_jobs.checkpoint_queue_bulk_action
        is agent_jobs.checkpoint_job_action_api.checkpoint_queue_bulk_action
    )
    assert (
        agent_jobs._validate_bulk_queue_action
        is agent_jobs.checkpoint_job_action_api.validate_bulk_queue_action
    )
    assert (
        agent_jobs._job_matches_bulk_queue_item_type
        is agent_jobs.checkpoint_job_action_api.job_matches_bulk_queue_item_type
    )
    assert callable(checkpoint_job_actions.build_checkpoint_job_action_api)


def test_checkpoint_follow_up_action_router_owns_expected_contracts():
    signatures = _route_signatures(agent_jobs.checkpoint_follow_up_action_api.router)
    assert ("/checkpoint-queue/follow-up-action", "POST") in signatures
    assert ("/checkpoint-queue/follow-up-bulk-action", "POST") in signatures
    assert (
        agent_jobs.checkpoint_queue_follow_up_action
        is agent_jobs.checkpoint_follow_up_action_api.checkpoint_queue_follow_up_action
    )
    assert (
        agent_jobs.checkpoint_queue_bulk_follow_up_action
        is agent_jobs.checkpoint_follow_up_action_api.checkpoint_queue_bulk_follow_up_action
    )
    assert callable(checkpoint_follow_up_actions.build_checkpoint_follow_up_action_api)


def test_job_action_router_owns_expected_contract():
    signatures = _route_signatures(agent_jobs.job_action_api.router)
    assert ("/{job_id}/action", "POST") in signatures
    assert agent_jobs.job_action is agent_jobs.job_action_api.job_action
    assert callable(job_actions.build_job_action_api)


def test_job_action_state_machine_respects_application_boundary():
    source_path = Path(job_action_state_machine.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }

    assert (
        agent_jobs.perform_job_action_state_machine
        is job_action_state_machine.perform_job_action
    )
    assert isinstance(
        agent_jobs._JOB_ACTION_DEPENDENCIES,
        job_action_state_machine.JobActionDependencies,
    )
    assert "app.api.endpoints.agent_jobs" not in imports
    assert not any(
        module == "fastapi" or module.startswith("fastapi.") for module in imports
    )


def test_job_action_swarm_handler_respects_application_boundary():
    for module in (job_action_contracts, job_action_swarm):
        source_path = Path(module.__file__)
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        imports = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        }

        assert "app.api.endpoints.agent_jobs" not in imports
        assert not any(
            imported == "fastapi" or imported.startswith("fastapi.")
            for imported in imports
        )

    assert set(job_action_swarm.SWARM_ACTIONS) == {
        "assign_swarm_review",
        "clear_swarm_assignment",
        "update_swarm_review_note",
        "launch_tie_breaker",
        "promote_swarm_candidate",
    }
    assert callable(job_action_swarm.perform_swarm_action)


def test_job_action_recovery_handler_respects_application_boundary():
    source_path = Path(job_action_recovery.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }

    assert "app.api.endpoints.agent_jobs" not in imports
    assert not any(
        module == "fastapi" or module.startswith("fastapi.") for module in imports
    )
    assert set(job_action_recovery.RECOVERY_ACTIONS) == {"restart", "relaunch"}
    assert callable(job_action_recovery.perform_recovery_action)


def test_job_action_checkpoint_resume_respects_application_boundary():
    source_path = Path(job_action_checkpoint_resume.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }

    assert "app.api.endpoints.agent_jobs" not in imports
    assert not any(
        module == "fastapi" or module.startswith("fastapi.") for module in imports
    )
    assert callable(job_action_checkpoint_resume.perform_resume_action)


def test_job_action_checkpoint_decisions_respect_application_boundary():
    source_path = Path(job_action_checkpoint_decisions.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }

    assert "app.api.endpoints.agent_jobs" not in imports
    assert not any(
        module == "fastapi" or module.startswith("fastapi.") for module in imports
    )
    assert set(job_action_checkpoint_decisions.CHECKPOINT_DECISION_ACTIONS) == {
        "approve",
        "edit",
        "skip",
        "reject",
    }
    assert callable(job_action_checkpoint_decisions.perform_checkpoint_decision)


def test_job_action_lifecycle_handler_respects_application_boundary():
    source_path = Path(job_action_lifecycle.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }

    assert "app.api.endpoints.agent_jobs" not in imports
    assert not any(
        module == "fastapi" or module.startswith("fastapi.") for module in imports
    )
    assert set(job_action_lifecycle.LIFECYCLE_ACTIONS) == {
        "pause",
        "cancel",
        "generate_summary",
    }
    assert callable(job_action_lifecycle.perform_lifecycle_action)


def test_decision_trace_query_router_owns_expected_contract():
    signatures = _route_signatures(agent_jobs.decision_trace_query_api.router)
    assert ("/decision-trace", "GET") in signatures
    assert (
        agent_jobs.get_decision_trace
        is agent_jobs.decision_trace_query_api.get_decision_trace
    )
    assert callable(decision_trace_queries.build_decision_trace_query_api)


def test_job_export_router_owns_expected_contract():
    signatures = _route_signatures(agent_jobs.job_export_api.router)
    assert signatures == {("/{job_id}/export", "GET")}
    assert agent_jobs.export_job_results is agent_jobs.job_export_api.export_job_results
    source = Path(job_exports.__file__).read_text(encoding="utf-8")
    assert "app.api.endpoints.agent_jobs" not in source
    assert callable(job_exports.build_job_export_api)


def test_job_checkpoint_router_owns_expected_contract():
    signatures = _route_signatures(agent_jobs.job_checkpoint_api.router)
    assert signatures == {("/{job_id}/checkpoints", "GET")}
    assert (
        agent_jobs.get_job_checkpoints
        is agent_jobs.job_checkpoint_api.get_job_checkpoints
    )
    source = Path(job_checkpoints.__file__).read_text(encoding="utf-8")
    assert "app.api.endpoints.agent_jobs" not in source
    assert callable(job_checkpoints.build_job_checkpoint_api)


def test_job_log_router_owns_expected_contract():
    signatures = _route_signatures(agent_jobs.job_log_api.router)
    assert signatures == {("/{job_id}/log", "GET")}
    assert agent_jobs.get_job_log is agent_jobs.job_log_api.get_job_log
    source = Path(job_logs.__file__).read_text(encoding="utf-8")
    assert "app.api.endpoints.agent_jobs" not in source
    assert callable(job_logs.build_job_log_api)
    assert callable(job_logs.build_job_log_page)


def test_job_progress_router_owns_expected_contract():
    websocket_routes = [
        route
        for route in agent_jobs.job_progress_api.router.routes
        if route.path == "/{job_id}/progress"
    ]
    assert len(websocket_routes) == 1
    assert (
        agent_jobs.agent_job_progress_websocket
        is agent_jobs.job_progress_api.agent_job_progress_websocket
    )
    source = Path(job_progress.__file__).read_text(encoding="utf-8")
    assert "app.api.endpoints.agent_jobs" not in source
    assert callable(job_progress.build_job_progress_api)


def test_job_step_event_router_owns_expected_contract():
    signatures = _route_signatures(agent_jobs.job_step_event_api.router)
    assert signatures == {("/{job_id}/step-events", "GET")}
    assert (
        agent_jobs.get_job_step_events
        is agent_jobs.job_step_event_api.get_job_step_events
    )
    source = Path(job_step_events.__file__).read_text(encoding="utf-8")
    assert "app.api.endpoints.agent_jobs" not in source
    assert callable(job_step_events.build_job_step_event_api)
    assert callable(job_step_events.build_step_event_page)


def test_decision_trace_reporting_router_owns_expected_contracts():
    signatures = _route_signatures(agent_jobs.decision_trace_reporting_api.router)
    assert ("/decision-trace/export", "GET") in signatures
    assert ("/decision-trace/analytics", "GET") in signatures
    assert (
        agent_jobs.export_decision_trace
        is agent_jobs.decision_trace_reporting_api.export_decision_trace
    )
    assert (
        agent_jobs.get_decision_trace_analytics
        is agent_jobs.decision_trace_reporting_api.get_decision_trace_analytics
    )
    assert callable(decision_trace_reporting.build_decision_trace_reporting_api)


def test_decision_trace_action_router_owns_expected_contract():
    signatures = _route_signatures(agent_jobs.decision_trace_action_api.router)
    assert ("/decision-trace/{event_id}/action", "POST") in signatures
    assert (
        agent_jobs.act_on_decision_trace_event
        is agent_jobs.decision_trace_action_api.act_on_decision_trace_event
    )
    assert callable(decision_trace_actions.build_decision_trace_action_api)


def test_decision_trace_view_router_owns_expected_contracts():
    signatures = _route_signatures(agent_jobs.decision_trace_view_api.router)
    assert ("/decision-trace/views", "GET") in signatures
    assert ("/decision-trace/views", "POST") in signatures
    assert ("/decision-trace/views/{view_id}", "PATCH") in signatures
    assert ("/decision-trace/views/{view_id}", "DELETE") in signatures
    assert (
        agent_jobs.list_decision_trace_views
        is agent_jobs.decision_trace_view_api.list_decision_trace_views
    )
    assert (
        agent_jobs.create_decision_trace_view
        is agent_jobs.decision_trace_view_api.create_decision_trace_view
    )
    assert (
        agent_jobs.update_decision_trace_view
        is agent_jobs.decision_trace_view_api.update_decision_trace_view
    )
    assert (
        agent_jobs.delete_decision_trace_view
        is agent_jobs.decision_trace_view_api.delete_decision_trace_view
    )
    assert callable(decision_trace_views.build_decision_trace_view_api)


def test_feedback_presenters_respect_application_boundary():
    source_path = Path(feedback_presenters.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }

    assert "app.api.endpoints.agent_jobs" not in imports
    assert not any(
        module == "fastapi" or module.startswith("fastapi.") for module in imports
    )
    assert not any(
        module == "app.tasks" or module.startswith("app.tasks.") for module in imports
    )
    assert not any(
        module == "app.models" or module.startswith("app.models.") for module in imports
    )
