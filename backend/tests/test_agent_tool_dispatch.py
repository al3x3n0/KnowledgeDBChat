"""Tests for the app-side tool provider registry."""

import pytest

from app.services.agent_tool_dispatch import (
    AgentToolExecutionContext,
    AgentToolRegistry,
    FunctionToolProvider,
    build_agent_service_analytics_content_provider,
    build_agent_service_chat_core_provider,
    build_agent_service_knowledge_graph_provider,
    build_agent_service_research_provider,
    build_agent_service_workflow_provider,
    build_autonomous_collaboration_provider,
    build_autonomous_data_analysis_provider,
    build_autonomous_document_authoring_provider,
    build_autonomous_document_provider,
    build_autonomous_kg_provider,
    build_autonomous_media_provider,
    build_autonomous_memory_provider,
    build_autonomous_notification_visualization_provider,
    build_autonomous_observability_provider,
    build_autonomous_output_state_provider,
    build_autonomous_project_bootstrap_provider,
    build_autonomous_reasoning_provider,
    build_autonomous_research_provider,
    build_autonomous_scheduling_provider,
    build_autonomous_snapshot_provider,
    build_autonomous_symbol_retrieval_provider,
    build_autonomous_web_research_provider,
    build_autonomous_workflow_provider,
    build_autonomous_workspace_mutation_provider,
    build_autonomous_workspace_read_provider,
)
from app.services.arxiv_search_service import ArxivSearchResult


@pytest.mark.asyncio
async def test_registry_executes_matching_provider():
    provider = FunctionToolProvider(
        name="test",
        modes={"chat"},
        handlers={"echo": lambda params, ctx: _echo(params, ctx)},
    )
    registry = AgentToolRegistry([provider])
    handled, result = await registry.try_execute(
        "echo",
        {"value": 1},
        AgentToolExecutionContext(mode="chat", db=None, service=None),
    )
    assert handled is True
    assert result == {"value": 1}


@pytest.mark.asyncio
async def test_registry_skips_provider_with_wrong_mode():
    provider = FunctionToolProvider(
        name="test",
        modes={"autonomous"},
        handlers={"echo": lambda params, ctx: _echo(params, ctx)},
    )
    registry = AgentToolRegistry([provider])
    handled, result = await registry.try_execute(
        "echo",
        {"value": 1},
        AgentToolExecutionContext(mode="chat", db=None, service=None),
    )
    assert handled is False
    assert result is None


def test_workflow_provider_resolves_chat_tools():
    provider = build_agent_service_workflow_provider(_DummyService())
    ctx = AgentToolExecutionContext(mode="chat", db=None, service=None, user_id="u")
    assert provider.can_handle("list_workflows", ctx) is True
    assert provider.can_handle("run_workflow", ctx) is True
    assert provider.can_handle("search_entities", ctx) is False


def test_kg_provider_resolves_chat_tools():
    provider = build_agent_service_knowledge_graph_provider(_DummyService())
    ctx = AgentToolExecutionContext(mode="chat", db=None, service=None, user_id="u")
    assert provider.can_handle("search_entities", ctx) is True
    assert provider.can_handle("merge_entities", ctx) is True
    assert provider.can_handle("run_workflow", ctx) is False


def test_research_provider_resolves_chat_tools():
    provider = build_agent_service_research_provider(_DummyService())
    ctx = AgentToolExecutionContext(mode="chat", db=None, service=None, user_id="u")
    assert provider.can_handle("search_arxiv", ctx) is True
    assert provider.can_handle("generate_literature_review_for_source", ctx) is True
    assert provider.can_handle("delegate_to_agent", ctx) is False


def test_analytics_content_provider_resolves_chat_tools():
    provider = build_agent_service_analytics_content_provider(_DummyService())
    ctx = AgentToolExecutionContext(mode="chat", db=None, service=None, user_id="u")
    assert provider.can_handle("get_trending_topics", ctx) is True
    assert provider.can_handle("generate_report", ctx) is True
    assert provider.can_handle("delegate_to_agent", ctx) is False


def test_chat_core_provider_resolves_remaining_chat_tools():
    provider = build_agent_service_chat_core_provider(_DummyService())
    ctx = AgentToolExecutionContext(mode="chat", db=None, service=None, user_id="u")
    assert provider.can_handle("request_file_upload", ctx) is True
    assert provider.can_handle("answer_question", ctx) is True
    assert provider.can_handle("delegate_to_agent", ctx) is True
    assert provider.can_handle("list_available_agents", ctx) is True
    assert provider.can_handle("search_documents", ctx) is False
    assert provider.can_handle("run_workflow", ctx) is False


def test_autonomous_research_provider_resolves_autonomous_tools():
    provider = build_autonomous_research_provider(_DummyService())
    ctx = AgentToolExecutionContext(
        mode="autonomous", db=None, service=None, user_id="u", job=_DummyJob(), state={}
    )
    assert provider.can_handle("search_arxiv", ctx) is True
    assert provider.can_handle("save_research_finding", ctx) is True
    assert provider.can_handle("execute_python", ctx) is False


@pytest.mark.asyncio
async def test_autonomous_search_arxiv_unwraps_search_result():
    """search_arxiv must read entries off ArxivSearchResult, not index it."""

    class _FakeArxivService:
        def __init__(self):
            self.calls = []

        async def search(self, **kwargs):
            self.calls.append(kwargs)
            return ArxivSearchResult(
                total_results=1,
                start=0,
                max_results=int(kwargs.get("max_results") or 10),
                items=[
                    {
                        "id": "2401.00001",
                        "title": "Speculative Decoding Survey",
                        "summary": "An abstract.",
                        "authors": ["A. Author"],
                        "published": "2024-01-01T00:00:00Z",
                    }
                ],
            )

    class _ArxivExecutor(_DummyService):
        arxiv_service = _FakeArxivService()

    executor = _ArxivExecutor()
    provider = build_autonomous_research_provider(executor)
    ctx = AgentToolExecutionContext(
        mode="autonomous", db=None, service=None, user_id="u", job=_DummyJob(), state={}
    )

    result = await provider.execute("search_arxiv", {"query": "speculative"}, ctx)

    assert result["success"] is True
    assert result["data"] == [
        {
            "id": "2401.00001",
            "title": "Speculative Decoding Survey",
            "summary": "An abstract.",
            "authors": ["A. Author"],
            "published": "2024-01-01T00:00:00Z",
        }
    ]
    assert [f["title"] for f in result["findings"]] == ["Speculative Decoding Survey"]
    assert [f["arxiv_id"] for f in result["findings"]] == ["2401.00001"]


def test_autonomous_data_analysis_provider_resolves_autonomous_tools():
    provider = build_autonomous_data_analysis_provider(_DummyService())
    ctx = AgentToolExecutionContext(
        mode="autonomous", db=None, service=None, user_id="u", job=_DummyJob(), state={}
    )
    assert provider.can_handle("load_csv_data", ctx) is True
    assert provider.can_handle("aggregate_data", ctx) is True
    assert provider.can_handle("create_chart", ctx) is True
    assert provider.can_handle("execute_python", ctx) is False
    assert provider.can_handle("render_diagram", ctx) is False


def test_autonomous_memory_provider_resolves_autonomous_tools():
    provider = build_autonomous_memory_provider(_DummyService())
    ctx = AgentToolExecutionContext(
        mode="autonomous", db=None, service=None, user_id="u", job=_DummyJob(), state={}
    )
    assert provider.can_handle("create_memory", ctx) is True
    assert provider.can_handle("get_memory_stats", ctx) is True
    assert provider.can_handle("execute_python", ctx) is False
    assert provider.can_handle("read_file", ctx) is False


def test_autonomous_workflow_provider_resolves_autonomous_tools():
    provider = build_autonomous_workflow_provider(_DummyService())
    ctx = AgentToolExecutionContext(
        mode="autonomous", db=None, service=None, user_id="u", job=_DummyJob(), state={}
    )
    assert provider.can_handle("list_available_workflows", ctx) is True
    assert provider.can_handle("execute_workflow", ctx) is True
    assert provider.can_handle("read_file", ctx) is False


def test_autonomous_reasoning_provider_resolves_autonomous_tools():
    provider = build_autonomous_reasoning_provider(_DummyService())
    ctx = AgentToolExecutionContext(
        mode="autonomous", db=None, service=None, user_id="u", job=_DummyJob(), state={}
    )
    assert provider.can_handle("reflect", ctx) is True
    assert provider.can_handle("critique_plan", ctx) is True
    assert provider.can_handle("execute_python", ctx) is False


def test_autonomous_collaboration_provider_resolves_autonomous_tools():
    provider = build_autonomous_collaboration_provider(_DummyService())
    ctx = AgentToolExecutionContext(
        mode="autonomous", db=None, service=None, user_id="u", job=_DummyJob(), state={}
    )
    assert provider.can_handle("delegate_subtask", ctx) is True
    assert provider.can_handle("read_agent_messages", ctx) is True
    assert provider.can_handle("execute_python", ctx) is False
    assert provider.can_handle("read_file", ctx) is False


def test_autonomous_workspace_read_provider_resolves_autonomous_tools():
    provider = build_autonomous_workspace_read_provider(_DummyService())
    ctx = AgentToolExecutionContext(
        mode="autonomous", db=None, service=None, user_id="u", job=_DummyJob(), state={}
    )
    assert provider.can_handle("clone_and_index_repo", ctx) is True
    assert provider.can_handle("browse_repo_files", ctx) is True
    assert provider.can_handle("read_file", ctx) is True
    assert provider.can_handle("search_code", ctx) is True
    assert provider.can_handle("get_workspace_status", ctx) is True
    assert provider.can_handle("list_workspace_checkpoints", ctx) is True
    assert provider.can_handle("list_durable_workspace_checkpoints", ctx) is True
    assert provider.can_handle("get_workspace_artifact_url", ctx) is True
    assert provider.can_handle("write_file", ctx) is False
    assert provider.can_handle("apply_patch", ctx) is False
    assert provider.can_handle("run_command", ctx) is False


def test_autonomous_workspace_mutation_provider_resolves_autonomous_tools():
    provider = build_autonomous_workspace_mutation_provider(_DummyService())
    ctx = AgentToolExecutionContext(
        mode="autonomous", db=None, service=None, user_id="u", job=_DummyJob(), state={}
    )
    assert provider.can_handle("execute_python", ctx) is True
    assert provider.can_handle("write_file", ctx) is True
    assert provider.can_handle("apply_patch", ctx) is True
    assert provider.can_handle("run_command", ctx) is True
    assert provider.can_handle("create_workspace_checkpoint", ctx) is True
    assert provider.can_handle("restore_workspace_checkpoint", ctx) is True
    assert provider.can_handle("hydrate_candidate_snapshot", ctx) is True
    assert provider.can_handle("persist_durable_workspace_checkpoint", ctx) is True
    assert provider.can_handle("restore_durable_workspace_checkpoint", ctx) is True
    assert provider.can_handle("project_bootstrap", ctx) is False
    assert provider.can_handle("search_web", ctx) is False
    assert provider.can_handle("create_chart", ctx) is False


def test_autonomous_symbol_retrieval_provider_resolves_autonomous_tools():
    provider = build_autonomous_symbol_retrieval_provider(_DummyService())
    ctx = AgentToolExecutionContext(
        mode="autonomous", db=None, service=None, user_id="u", job=_DummyJob(), state={}
    )
    assert provider.can_handle("retrieve_repo_symbols", ctx) is True
    assert provider.can_handle("get_symbol_context", ctx) is True
    assert provider.can_handle("find_tests_for_symbol", ctx) is True
    assert provider.can_handle("write_file", ctx) is False
    assert provider.can_handle("run_command", ctx) is False


def test_autonomous_document_authoring_provider_resolves_autonomous_tools():
    provider = build_autonomous_document_authoring_provider(_DummyService())
    ctx = AgentToolExecutionContext(
        mode="autonomous", db=None, service=None, user_id="u", job=_DummyJob(), state={}
    )
    assert provider.can_handle("plan_document", ctx) is True
    assert provider.can_handle("write_section", ctx) is True
    assert provider.can_handle("assemble_document", ctx) is True
    assert provider.can_handle("export_document", ctx) is True
    assert provider.can_handle("insert_figure", ctx) is True
    assert provider.can_handle("write_file", ctx) is False
    assert provider.can_handle("search_web", ctx) is False


def test_autonomous_document_provider_resolves_autonomous_tools():
    provider = build_autonomous_document_provider(_DummyService())
    ctx = AgentToolExecutionContext(
        mode="autonomous", db=None, service=None, user_id="u", job=_DummyJob(), state={}
    )
    assert provider.can_handle("search_documents", ctx) is True
    assert provider.can_handle("search_with_filters", ctx) is True
    assert provider.can_handle("web_scrape", ctx) is True
    assert provider.can_handle("ingest_url", ctx) is True
    assert provider.can_handle("get_document_details", ctx) is True
    assert provider.can_handle("read_document_content", ctx) is True
    assert provider.can_handle("summarize_document", ctx) is True
    assert provider.can_handle("find_similar_documents", ctx) is True
    assert provider.can_handle("get_knowledge_base_stats", ctx) is True
    assert provider.can_handle("create_document_from_text", ctx) is True
    assert provider.can_handle("list_documents_by_tag", ctx) is True
    assert provider.can_handle("merge_documents", ctx) is True
    assert provider.can_handle("search_web", ctx) is False
    assert provider.can_handle("send_notification", ctx) is False
    assert provider.can_handle("create_chart", ctx) is False


def test_autonomous_web_research_provider_resolves_autonomous_tools():
    provider = build_autonomous_web_research_provider(_DummyService())
    ctx = AgentToolExecutionContext(
        mode="autonomous", db=None, service=None, user_id="u", job=_DummyJob(), state={}
    )
    assert provider.can_handle("search_web", ctx) is True
    assert provider.can_handle("summarize_url", ctx) is True
    assert provider.can_handle("project_bootstrap", ctx) is False
    assert provider.can_handle("create_chart", ctx) is False


def test_autonomous_notification_visualization_provider_resolves_autonomous_tools():
    provider = build_autonomous_notification_visualization_provider(_DummyService())
    ctx = AgentToolExecutionContext(
        mode="autonomous", db=None, service=None, user_id="u", job=_DummyJob(), state={}
    )
    assert provider.can_handle("send_notification", ctx) is True
    assert provider.can_handle("create_chart", ctx) is True
    assert provider.can_handle("query_kg_entities", ctx) is False
    assert provider.can_handle("transcribe_document", ctx) is False


def test_autonomous_kg_provider_resolves_autonomous_tools():
    provider = build_autonomous_kg_provider(_DummyService())
    ctx = AgentToolExecutionContext(
        mode="autonomous", db=None, service=None, user_id="u", job=_DummyJob(), state={}
    )
    assert provider.can_handle("query_kg_entities", ctx) is True
    assert provider.can_handle("create_kg_relationship", ctx) is True
    assert provider.can_handle("send_notification", ctx) is False
    assert provider.can_handle("schedule_job", ctx) is False


def test_autonomous_scheduling_provider_resolves_autonomous_tools():
    provider = build_autonomous_scheduling_provider(_DummyService())
    ctx = AgentToolExecutionContext(
        mode="autonomous", db=None, service=None, user_id="u", job=_DummyJob(), state={}
    )
    assert provider.can_handle("schedule_job", ctx) is True
    assert provider.can_handle("cancel_scheduled_job", ctx) is True
    assert provider.can_handle("search_web", ctx) is False
    assert provider.can_handle("transcribe_document", ctx) is False


def test_autonomous_media_provider_resolves_autonomous_tools():
    provider = build_autonomous_media_provider(_DummyService())
    ctx = AgentToolExecutionContext(
        mode="autonomous", db=None, service=None, user_id="u", job=_DummyJob(), state={}
    )
    assert provider.can_handle("transcribe_document", ctx) is True
    assert provider.can_handle("get_media_info", ctx) is True
    assert provider.can_handle("schedule_job", ctx) is False
    assert provider.can_handle("capture_snapshot", ctx) is False


def test_autonomous_snapshot_provider_resolves_autonomous_tools():
    provider = build_autonomous_snapshot_provider(_DummyService())
    ctx = AgentToolExecutionContext(
        mode="autonomous", db=None, service=None, user_id="u", job=_DummyJob(), state={}
    )
    assert provider.can_handle("capture_snapshot", ctx) is True
    assert provider.can_handle("detect_drift", ctx) is True
    assert provider.can_handle("transcribe_document", ctx) is False
    assert provider.can_handle("project_bootstrap", ctx) is False


def test_autonomous_project_bootstrap_provider_resolves_autonomous_tools():
    provider = build_autonomous_project_bootstrap_provider(_DummyService())
    ctx = AgentToolExecutionContext(
        mode="autonomous", db=None, service=None, user_id="u", job=_DummyJob(), state={}
    )
    assert provider.can_handle("project_bootstrap", ctx) is True
    assert provider.can_handle("search_web", ctx) is False
    assert provider.can_handle("create_chart", ctx) is False


def test_autonomous_observability_provider_resolves_autonomous_tools():
    provider = build_autonomous_observability_provider(_DummyService())
    ctx = AgentToolExecutionContext(
        mode="autonomous", db=None, service=None, user_id="u", job=_DummyJob(), state={}
    )
    assert provider.can_handle("get_job_metrics", ctx) is True
    assert provider.can_handle("batch_search", ctx) is True
    assert provider.can_handle("evaluate_condition", ctx) is True
    assert provider.can_handle("summarize_findings", ctx) is True
    assert provider.can_handle("execute_python", ctx) is False
    assert provider.can_handle("search_web", ctx) is False
    assert provider.can_handle("create_chart", ctx) is False


def test_autonomous_output_state_provider_resolves_autonomous_tools():
    provider = build_autonomous_output_state_provider(_DummyService())
    ctx = AgentToolExecutionContext(
        mode="autonomous", db=None, service=None, user_id="u", job=_DummyJob(), state={}
    )
    assert provider.can_handle("create_handoff", ctx) is True
    assert provider.can_handle("switch_strategy", ctx) is True
    assert provider.can_handle("format_as_report", ctx) is True
    assert provider.can_handle("set_output_schema", ctx) is True
    assert provider.can_handle("execute_python", ctx) is False
    assert provider.can_handle("search_web", ctx) is False
    assert provider.can_handle("create_chart", ctx) is False


async def _echo(params, ctx):
    return {"value": params["value"]}


class _DummyService:
    async def _tool_answer_question(self, params, user_id, db):
        return {}

    async def _tool_list_available_agents(self, db):
        return {}

    async def _tool_delegate_to_agent(self, params, user_id, db):
        return {}

    async def _tool_search_entities(self, params, db):
        return {}

    async def _tool_get_entity_relationships(self, params, db):
        return {}

    async def _tool_find_documents_by_entity(self, params, db):
        return {}

    async def _tool_get_document_knowledge_graph(self, params, db):
        return {}

    async def _tool_get_global_knowledge_graph(self, params, db):
        return {}

    async def _tool_get_entity_mentions(self, params, db):
        return {}

    async def _tool_get_kg_stats(self, db):
        return {}

    async def _tool_rebuild_document_knowledge_graph(self, params, user_id, db):
        return {}

    async def _tool_merge_entities(self, params, user_id, db):
        return {}

    async def _tool_delete_entity(self, params, user_id, db):
        return {}

    async def _tool_generate_diagram(self, params, user_id, db):
        return {}

    async def _tool_run_workflow(self, params, user_id, db):
        return {}

    async def _tool_propose_workflow_from_description(self, params, user_id, db):
        return {}

    async def _tool_create_workflow_from_description(self, params, user_id, db):
        return {}

    async def _tool_list_workflows(self, params, user_id, db):
        return {}

    async def _tool_run_custom_tool(self, params, user_id, db):
        return {}

    async def _tool_list_custom_tools(self, params, user_id, db):
        return {}

    async def _tool_start_template_fill(self, params, user_id, db):
        return {}

    async def _tool_list_template_jobs(self, params, user_id, db):
        return {}

    async def _tool_get_template_job_status(self, params, user_id, db):
        return {}

    async def _tool_search_arxiv(self, params):
        return {}

    async def _tool_ingest_arxiv_papers(self, params, user_id, db):
        return {}

    async def _tool_literature_review_arxiv(self, params, user_id, db):
        return {}

    async def _tool_summarize_documents_in_source(self, params, user_id, db):
        return {}

    async def _tool_enrich_arxiv_metadata_for_source(self, params, user_id, db):
        return {}

    async def _tool_generate_literature_review_for_source(self, params, user_id, db):
        return {}

    async def _tool_generate_slides_for_source(self, params, user_id, db):
        return {}

    async def _tool_get_collection_statistics(self, params, db):
        return {}

    async def _tool_get_source_analytics(self, params, db):
        return {}

    async def _tool_get_trending_topics(self, params, db):
        return {}

    async def _tool_generate_chart_data(self, params, db):
        return {}

    async def _tool_export_data(self, params, db):
        return {}

    async def _tool_faceted_search(self, params, db):
        return {}

    async def _tool_get_search_suggestions(self, params, db):
        return {}

    async def _tool_get_related_searches(self, params, db):
        return {}

    async def _tool_draft_email(self, params, db):
        return {}

    async def _tool_generate_meeting_notes(self, params, db):
        return {}

    async def _tool_generate_documentation(self, params, db):
        return {}

    async def _tool_generate_executive_summary(self, params, db):
        return {}

    async def _tool_generate_report(self, params, db):
        return {}

    async def _tool_generate_gitlab_architecture(self, params, user_id, db):
        return {}


class _DummyJob:
    id = "job-1"
    goal = "research transformers"
    user_id = "user-1"
    iteration = 1
    progress = 10
    max_iterations = 10
    config = {}
