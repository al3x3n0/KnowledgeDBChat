"""Job-type tool policy.

Which tools a job type may use, and how tool selection is tuned for a job.
Pure policy tables extracted from AutonomousAgentExecutor: they read only the
job type and its config, so they are unit-tested directly.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.agent_core import tool_specs
from app.models.agent_job import AgentJob
from app.services.data_analysis_tools import exposed_data_analysis_tools


def get_tools_for_job_type(
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
        # Custom tools: create one, then call it in this run or a later one
        "create_custom_tool",
        "run_custom_tool",
        "list_custom_tools",
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
            # The dataset-backed charting tool, under the name dispatch routes
            # on. Absent from this list it was unreachable on the only job type
            # that can produce a dataset to chart.
            "create_chart_from_dataset",
            "create_correlation_heatmap",
            "create_flowchart",
            "create_sequence_diagram",
            "create_er_diagram",
            "create_architecture_diagram",
            "create_drawio_diagram",
            "create_gantt_chart",
            "create_pie_chart_diagram",
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
        "create_custom_tool",
        "run_custom_tool",
        "list_custom_tools",
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
    # Exposed names, not definition keys: dispatch routes on the exposed name,
    # so filtering against the raw keys silently dropped the renamed tool from
    # every proposal that asked for it.
    supported_tools.update(exposed_data_analysis_tools().keys())
    # The measurement family declares its own availability in
    # agent_core.tool_specs; listing it here as well was a second place to
    # forget it.
    supported_tools.update(tool_specs.spec_names())

    proposed = sorted(
        set(base_tools)
        | set(type_tools.get(job_type, []))
        | set(tool_specs.tools_for_job_type(job_type))
    )
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


def get_tool_selection_config(job: AgentJob) -> Dict[str, Any]:
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
        "failure_penalty": _as_float("tool_selection_failure_penalty", 0.08, 0.0, 1.0),
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
        "ab_test_variant_a": _as_mode("tool_selection_ab_test_variant_a", "adaptive"),
        "ab_test_variant_b": _as_mode("tool_selection_ab_test_variant_b", "thompson"),
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
        "stage_finish_mode": _as_mode("tool_selection_stage_finish_mode", "baseline"),
        "stage_rescue_mode": _as_mode("tool_selection_stage_rescue_mode", "adaptive"),
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
        "feedback_learning_enabled": bool(cfg.get("feedback_learning_enabled", True)),
        "feedback_learning_weight": _as_float(
            "feedback_learning_weight", 0.08, 0.0, 0.6
        ),
        "feedback_learning_max_abs_bias": _as_float(
            "feedback_learning_max_abs_bias", 0.3, 0.0, 1.0
        ),
    }
