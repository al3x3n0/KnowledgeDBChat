"""Unified tool catalog for agent core consumers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional


@dataclass(frozen=True)
class ToolMetadata:
    name: str
    description: str
    input_schema: Dict[str, Any]
    effects: str
    network: str
    cost_tier: str
    pii_risk: str


def _default_metadata(
    *, name: str, description: str, input_schema: Dict[str, Any]
) -> ToolMetadata:
    base = name.split("mcp:", 1)[1] if name.startswith("mcp:") else name

    # A tool that declares its own governance is believed. The name lists
    # below are the older form: a tool is classified by whether someone
    # remembered to add it, which is how 25 mutating tools came to be
    # classified read-safe by omission.
    from app.agent_core.tool_specs import spec_for

    spec = spec_for(base)
    if spec is not None:
        return ToolMetadata(
            name=name,
            description=description or spec.description,
            input_schema=input_schema or spec.parameters,
            effects=spec.effects,
            network=spec.network,
            cost_tier=spec.cost_tier,
            pii_risk=spec.pii_risk,
        )

    write_tools = {
        "delete_document",
        "batch_delete_documents",
        "update_document_tags",
        "create_document_from_text",
        "ingest_url",
        "merge_entities",
        "delete_entity",
        "rebuild_document_knowledge_graph",
        # Deletes a document's mentions and relationships before re-extracting.
        "build_research_graph",
        "run_custom_tool",
        # Creates a persistent, executable capability for this user.
        "create_custom_tool",
        "run_workflow",
        "docker_execute",
        "delegate_subtask",
        "share_findings",
        "request_review",
        "execute_data_pipeline",
        # Compile and run submitted code inside a Docker sandbox.
        "compile_c_snippet",
        "analyze_snippet_cycles",
        "profile_c_workload",
        "simulate_c_workload",
        "sample_hardware_counters",
        "measure_predictability",
        "select_counter_taps",
        "evaluate_predictor_design",
        "describe_model_parameters",
        "find_fusion_candidates",
        "cost_fusion_candidate",
        "verify_run_bundle",
        "record_prediction",
        "record_measurement",
        "axis_check",
        "axis_emit",
        "axis_prove",
        "benchmark_c_snippet",
        "write_and_run_script",
        "write_file",
        "apply_patch",
        "run_command",
        "export_document",
        "create_memory",
        "record_method",
        "execute_workflow",
        "send_message_to_agent",
        "send_notification",
        "send_email_alert",
        "create_chart",
        "render_diagram",
        "create_kg_entity",
        "create_kg_relationship",
        "schedule_job",
        "cancel_scheduled_job",
        "merge_documents",
        "create_handoff",
        "broadcast_to_siblings",
        "transcribe_document",
        # Added after an audit found 25 tools whose names imply mutation
        # classified read-safe by omission. The rule applied here is
        # fail-closed: a tool goes on this list unless it is demonstrably pure
        # computation, because under-classifying is the dangerous direction —
        # an enforced read-only policy would otherwise permit them.
        #
        # execute_python runs arbitrary code. It was classified read while its
        # sibling write_and_run_script was classified write, which is the exact
        # false assurance an allowed_effects gate would have inherited.
        "execute_python",
        # The rule: "write" means an effect outside the agent's own run — a
        # database row, a file, a network call, a queued job. Tools that only
        # mutate in-run state (set_focus_directive, save_research_finding,
        # write_section) stay read: an operator setting a read-only policy wants
        # to stop side effects on the world, not stop the agent steering itself.
        # Verified to reach outside the run:
        "create_synthesis_document",
        "create_knowledge_base_entry",
        "create_workflow_from_description",
        "add_to_reading_list",
        "link_entities",
        "export_data",
        "ingest_arxiv_papers",
        "ingest_paper_by_id",
        # Generation tools that persist an artifact rather than only returning
        # text. Classified write pending per-tool review; see the guard test.
        "generate_report",
        "generate_diagram",
        "generate_documentation",
        "generate_executive_summary",
        "generate_gitlab_architecture",
        "generate_literature_review_for_source",
        "generate_meeting_notes",
        "generate_research_presentation",
        "generate_slides_for_source",
        "generate_chart_data",
        # MCP-exposed tools that create persistent jobs. Classified read-safe by
        # the same omission that hid execute_python; this surface sat outside
        # the guard test until iter_mcp_tools was lifted to module level.
        "create_presentation",
        "create_repo_report",
    }
    network_tools = {
        "web_scrape",
        "ingest_url",
        "search_arxiv",
        "ingest_arxiv_papers",
        "literature_review_arxiv",
        "create_repo_report",
        "docker_execute",
        "clone_and_index_repo",
        "search_web",
        "fetch_url_content",
        "summarize_url",
        "render_diagram",
    }
    high_pii = {"docker_execute", "write_and_run_script"}
    medium_pii = {
        "web_scrape",
        "ingest_url",
        "run_custom_tool",
        "create_custom_tool",
        "execute_python",
        "execute_data_pipeline",
        "compile_c_snippet",
        "analyze_snippet_cycles",
        "profile_c_workload",
        "simulate_c_workload",
        "sample_hardware_counters",
        "measure_predictability",
        "select_counter_taps",
        "evaluate_predictor_design",
        "describe_model_parameters",
        "find_fusion_candidates",
        "cost_fusion_candidate",
        "verify_run_bundle",
        "record_prediction",
        "record_measurement",
        "axis_check",
        "axis_emit",
        "axis_prove",
        "benchmark_c_snippet",
        "run_command",
        "search_code",
    }
    high_cost = {
        "docker_execute",
        "execute_data_pipeline",
        "compile_c_snippet",
        "analyze_snippet_cycles",
        "profile_c_workload",
        "simulate_c_workload",
        "sample_hardware_counters",
        "measure_predictability",
        "select_counter_taps",
        "evaluate_predictor_design",
        "describe_model_parameters",
        "find_fusion_candidates",
        "cost_fusion_candidate",
        "verify_run_bundle",
        "record_prediction",
        "record_measurement",
        "axis_check",
        "axis_emit",
        "axis_prove",
        "benchmark_c_snippet",
        "write_and_run_script",
        "run_command",
    }
    medium_cost = {
        "generate_report",
        "create_repo_report",
        "create_presentation",
        "delegate_subtask",
        "request_review",
        "execute_python",
        "clone_and_index_repo",
        "export_document",
        "execute_workflow",
        "search_web",
        "summarize_url",
        "create_chart",
        "batch_search",
        "batch_summarize",
        "compress_history",
        "summarize_findings",
        "create_handoff",
        "transcribe_document",
        "analyze_image",
    }
    return ToolMetadata(
        name=name,
        description=description,
        input_schema=input_schema,
        effects="write" if base in write_tools else "read",
        network="egress" if base in network_tools else "none",
        cost_tier="high"
        if base in high_cost
        else ("medium" if base in medium_cost else "low"),
        pii_risk="high"
        if base in high_pii
        else ("medium" if base in medium_pii else "low"),
    )


def iter_builtin_tools() -> Iterable[ToolMetadata]:
    """Every tool a run can execute.

    There is one registry to read now. There were several, and the
    data-analysis tools were in a different one from this, so 21 tools an agent
    job could call had no metadata at all: invisible to the tool-policy UI,
    unclassifiable by effects, and denied as "unknown" by any policy carrying
    constraints. The guard written for exactly that failure could not see them
    either, because it derived its universe from the same partial registry.
    """
    try:
        from app.services.agent_tools import AGENT_TOOLS
    except Exception:
        AGENT_TOOLS = []

    for tool in AGENT_TOOLS or []:
        name = str(tool.get("name") or "").strip()
        if not name:
            continue
        yield _default_metadata(
            name=name,
            description=str(tool.get("description") or "").strip(),
            input_schema=tool.get("parameters")
            if isinstance(tool.get("parameters"), dict)
            else {},
        )


def get_tool_metadata(tool_name: str) -> Optional[ToolMetadata]:
    name = str(tool_name or "").strip()
    if not name:
        return None
    is_mcp = name.startswith("mcp:")
    base_name = name.split("mcp:", 1)[1].strip() if is_mcp else name

    for meta in iter_builtin_tools():
        if meta.name == base_name:
            if not is_mcp:
                return meta
            return ToolMetadata(
                name=f"mcp:{base_name}",
                description=meta.description,
                input_schema=meta.input_schema,
                effects=meta.effects,
                network=meta.network,
                cost_tier=meta.cost_tier,
                pii_risk=meta.pii_risk,
            )

    return iter_mcp_tools().get(base_name) if is_mcp else None


def iter_mcp_tools() -> Dict[str, ToolMetadata]:
    """Metadata for MCP-exposed tools, keyed by unprefixed name.

    Module level so the classification guard can enumerate this surface too.
    It previously lived inside get_tool_metadata, which put it out of reach of
    the test that checks tools are classified deliberately — and that is exactly
    where two mutating tools were found classified read-safe.
    """
    return {
        "search": _default_metadata(
            name="mcp:search",
            description="Semantic search over the knowledge base",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        ),
        "list_documents": _default_metadata(
            name="mcp:list_documents",
            description="List documents",
            input_schema={"type": "object", "properties": {}},
        ),
        "get_document": _default_metadata(
            name="mcp:get_document",
            description="Get document by id",
            input_schema={
                "type": "object",
                "properties": {"document_id": {"type": "string"}},
                "required": ["document_id"],
            },
        ),
        "list_sources": _default_metadata(
            name="mcp:list_sources",
            description="List document sources",
            input_schema={"type": "object", "properties": {}},
        ),
        "chat": _default_metadata(
            name="mcp:chat",
            description="Ask a question and get an answer grounded in the KB",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        ),
        "create_presentation": _default_metadata(
            name="mcp:create_presentation",
            description="Create a presentation job",
            input_schema={
                "type": "object",
                "properties": {"topic": {"type": "string"}},
                "required": ["topic"],
            },
        ),
        "create_repo_report": _default_metadata(
            name="mcp:create_repo_report",
            description="Create a repo report job",
            input_schema={
                "type": "object",
                "properties": {"repo_url": {"type": "string"}},
                "required": ["repo_url"],
            },
        ),
        "get_job_status": _default_metadata(
            name="mcp:get_job_status",
            description="Get status of a generation job",
            input_schema={
                "type": "object",
                "properties": {"job_id": {"type": "string"}},
                "required": ["job_id"],
            },
        ),
        "list_jobs": _default_metadata(
            name="mcp:list_jobs",
            description="List generation jobs",
            input_schema={"type": "object", "properties": {}},
        ),
        "docker_execute": _default_metadata(
            name="mcp:docker_execute",
            description="Execute a command inside a Docker container",
            input_schema={
                "type": "object",
                "properties": {
                    "image": {"type": "string"},
                    "command": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["image", "command"],
            },
        ),
    }
