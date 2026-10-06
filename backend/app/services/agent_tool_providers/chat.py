"""Tool providers that answer chat (``AgentService``) calls.

Split out of ``agent_tool_dispatch``, which re-exports every name here.
"""

from __future__ import annotations

from typing import Any, Dict

from app.services.agent_tool_providers.base import (
    AgentToolExecutionContext,
    FunctionToolProvider,
)


def build_agent_service_document_provider(service: Any) -> FunctionToolProvider:
    """Document-domain tools for AgentService."""

    async def _search_documents(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        return await service._tool_search_documents(params, ctx.db)

    async def _get_document_details(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        return await service._tool_get_document_details(params, ctx.db)

    async def _summarize_document(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        return await service._tool_summarize_document(params, ctx.db)

    async def _delete_document(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        return await service._tool_delete_document(params, ctx.db)

    async def _list_recent_documents(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        return await service._tool_list_recent_documents(params, ctx.db)

    async def _list_document_sources(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        return await service._tool_list_document_sources(params, ctx.db)

    async def _list_documents_by_source(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        return await service._tool_list_documents_by_source(params, ctx.db)

    async def _web_scrape(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        return await service._tool_web_scrape(params, ctx.user_id, ctx.db)

    async def _create_document_from_text(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        return await service._tool_create_document_from_text(
            params, ctx.user_id, ctx.db
        )

    async def _ingest_url(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        return await service._tool_ingest_url(params, ctx.user_id, ctx.db)

    async def _find_similar_documents(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        return await service._tool_find_similar_documents(params, ctx.db)

    async def _search_documents_by_author(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        return await service._tool_search_documents_by_author(params, ctx.db)

    async def _update_document_tags(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        return await service._tool_update_document_tags(params, ctx.db)

    async def _get_knowledge_base_stats(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        return await service._tool_get_knowledge_base_stats(ctx.db)

    async def _batch_delete_documents(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        return await service._tool_batch_delete_documents(params, ctx.db)

    async def _batch_summarize_documents(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        return await service._tool_batch_summarize_documents(params, ctx.db)

    async def _search_by_tags(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        return await service._tool_search_by_tags(params, ctx.db)

    async def _list_all_tags(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        return await service._tool_list_all_tags(ctx.db)

    async def _compare_documents(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        return await service._tool_compare_documents(params, ctx.user_id, ctx.db)

    async def _read_document_content(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        return await service._tool_read_document_content(params, ctx.db)

    return FunctionToolProvider(
        name="agent_service_document_tools",
        modes={"chat"},
        handlers={
            "search_documents": _search_documents,
            "get_document_details": _get_document_details,
            "summarize_document": _summarize_document,
            "delete_document": _delete_document,
            "list_recent_documents": _list_recent_documents,
            "list_document_sources": _list_document_sources,
            "list_documents_by_source": _list_documents_by_source,
            "web_scrape": _web_scrape,
            "create_document_from_text": _create_document_from_text,
            "ingest_url": _ingest_url,
            "find_similar_documents": _find_similar_documents,
            "search_documents_by_author": _search_documents_by_author,
            "update_document_tags": _update_document_tags,
            "get_knowledge_base_stats": _get_knowledge_base_stats,
            "batch_delete_documents": _batch_delete_documents,
            "batch_summarize_documents": _batch_summarize_documents,
            "search_by_tags": _search_by_tags,
            "search_documents_by_tag": _search_by_tags,
            "list_all_tags": _list_all_tags,
            "compare_documents": _compare_documents,
            "read_document_content": _read_document_content,
        },
    )


def build_agent_service_knowledge_graph_provider(service: Any) -> FunctionToolProvider:
    """Knowledge-graph tools for AgentService."""

    async def _search_entities(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        return await service._tool_search_entities(params, ctx.db)

    async def _get_entity_relationships(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        return await service._tool_get_entity_relationships(params, ctx.db)

    async def _find_documents_by_entity(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        return await service._tool_find_documents_by_entity(params, ctx.db)

    async def _get_document_knowledge_graph(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        return await service._tool_get_document_knowledge_graph(params, ctx.db)

    async def _get_global_knowledge_graph(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        return await service._tool_get_global_knowledge_graph(params, ctx.db)

    async def _get_entity_mentions(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        return await service._tool_get_entity_mentions(params, ctx.db)

    async def _get_kg_stats(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        return await service._tool_get_kg_stats(ctx.db)

    async def _rebuild_document_knowledge_graph(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        return await service._tool_rebuild_document_knowledge_graph(
            params, ctx.user_id, ctx.db
        )

    async def _merge_entities(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        return await service._tool_merge_entities(params, ctx.user_id, ctx.db)

    async def _delete_entity(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        return await service._tool_delete_entity(params, ctx.user_id, ctx.db)

    return FunctionToolProvider(
        name="agent_service_knowledge_graph_tools",
        modes={"chat"},
        handlers={
            "search_entities": _search_entities,
            "get_entity_relationships": _get_entity_relationships,
            "find_documents_by_entity": _find_documents_by_entity,
            "get_document_knowledge_graph": _get_document_knowledge_graph,
            "get_global_knowledge_graph": _get_global_knowledge_graph,
            "get_entity_mentions": _get_entity_mentions,
            "get_kg_stats": _get_kg_stats,
            "rebuild_document_knowledge_graph": _rebuild_document_knowledge_graph,
            "merge_entities": _merge_entities,
            "delete_entity": _delete_entity,
        },
    )


def build_agent_service_workflow_provider(service: Any) -> FunctionToolProvider:
    """Workflow and custom-tool helpers for AgentService."""

    async def _generate_diagram(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        return await service._tool_generate_diagram(params, ctx.user_id, ctx.db)

    async def _run_workflow(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        return await service._tool_run_workflow(params, ctx.user_id, ctx.db)

    async def _propose_workflow_from_description(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        return await service._tool_propose_workflow_from_description(
            params, ctx.user_id, ctx.db
        )

    async def _create_workflow_from_description(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        return await service._tool_create_workflow_from_description(
            params, ctx.user_id, ctx.db
        )

    async def _list_workflows(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        return await service._tool_list_workflows(params, ctx.user_id, ctx.db)

    async def _run_custom_tool(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        return await service._tool_run_custom_tool(params, ctx.user_id, ctx.db)

    async def _list_custom_tools(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        return await service._tool_list_custom_tools(params, ctx.user_id, ctx.db)

    async def _start_template_fill(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        return await service._tool_start_template_fill(params, ctx.user_id, ctx.db)

    async def _list_template_jobs(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        return await service._tool_list_template_jobs(params, ctx.user_id, ctx.db)

    async def _get_template_job_status(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        return await service._tool_get_template_job_status(params, ctx.user_id, ctx.db)

    return FunctionToolProvider(
        name="agent_service_workflow_tools",
        modes={"chat"},
        handlers={
            "generate_diagram": _generate_diagram,
            "run_workflow": _run_workflow,
            "propose_workflow_from_description": _propose_workflow_from_description,
            "create_workflow_from_description": _create_workflow_from_description,
            "list_workflows": _list_workflows,
            "run_custom_tool": _run_custom_tool,
            "list_custom_tools": _list_custom_tools,
            "start_template_fill": _start_template_fill,
            "list_template_jobs": _list_template_jobs,
            "get_template_job_status": _get_template_job_status,
        },
    )


def build_agent_service_research_provider(service: Any) -> FunctionToolProvider:
    """Research and arXiv tools for AgentService."""

    async def _search_arxiv(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        return await service._tool_search_arxiv(params)

    async def _ingest_arxiv_papers(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        return await service._tool_ingest_arxiv_papers(params, ctx.user_id, ctx.db)

    async def _literature_review_arxiv(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        return await service._tool_literature_review_arxiv(params, ctx.user_id, ctx.db)

    async def _summarize_documents_in_source(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        return await service._tool_summarize_documents_in_source(
            params, ctx.user_id, ctx.db
        )

    async def _enrich_arxiv_metadata_for_source(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        return await service._tool_enrich_arxiv_metadata_for_source(
            params, ctx.user_id, ctx.db
        )

    async def _generate_literature_review_for_source(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        return await service._tool_generate_literature_review_for_source(
            params, ctx.user_id, ctx.db
        )

    async def _generate_slides_for_source(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        return await service._tool_generate_slides_for_source(
            params, ctx.user_id, ctx.db
        )

    return FunctionToolProvider(
        name="agent_service_research_tools",
        modes={"chat"},
        handlers={
            "search_arxiv": _search_arxiv,
            "ingest_arxiv_papers": _ingest_arxiv_papers,
            "literature_review_arxiv": _literature_review_arxiv,
            "summarize_documents_in_source": _summarize_documents_in_source,
            "enrich_arxiv_metadata_for_source": _enrich_arxiv_metadata_for_source,
            "generate_literature_review_for_source": _generate_literature_review_for_source,
            "generate_slides_for_source": _generate_slides_for_source,
        },
    )


def build_agent_service_analytics_content_provider(
    service: Any,
) -> FunctionToolProvider:
    """Analytics, search helpers, and content-generation tools for AgentService."""

    async def _get_collection_statistics(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        return await service._tool_get_collection_statistics(params, ctx.db)

    async def _get_source_analytics(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        return await service._tool_get_source_analytics(params, ctx.db)

    async def _get_trending_topics(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        return await service._tool_get_trending_topics(params, ctx.db)

    async def _generate_chart_data(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        return await service._tool_generate_chart_data(params, ctx.db)

    async def _export_data(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        return await service._tool_export_data(params, ctx.db)

    async def _faceted_search(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        return await service._tool_faceted_search(params, ctx.db)

    async def _get_search_suggestions(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        return await service._tool_get_search_suggestions(params, ctx.db)

    async def _get_related_searches(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        return await service._tool_get_related_searches(params, ctx.db)

    async def _draft_email(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        return await service._tool_draft_email(params, ctx.db)

    async def _generate_meeting_notes(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        return await service._tool_generate_meeting_notes(params, ctx.db)

    async def _generate_documentation(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        return await service._tool_generate_documentation(params, ctx.db)

    async def _generate_executive_summary(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        return await service._tool_generate_executive_summary(params, ctx.db)

    async def _generate_report(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        return await service._tool_generate_report(params, ctx.db)

    async def _generate_gitlab_architecture(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        return await service._tool_generate_gitlab_architecture(
            params, ctx.user_id, ctx.db
        )

    return FunctionToolProvider(
        name="agent_service_analytics_content_tools",
        modes={"chat"},
        handlers={
            "get_collection_statistics": _get_collection_statistics,
            "get_source_analytics": _get_source_analytics,
            "get_trending_topics": _get_trending_topics,
            "generate_chart_data": _generate_chart_data,
            "export_data": _export_data,
            "faceted_search": _faceted_search,
            "get_search_suggestions": _get_search_suggestions,
            "get_related_searches": _get_related_searches,
            "draft_email": _draft_email,
            "generate_meeting_notes": _generate_meeting_notes,
            "generate_documentation": _generate_documentation,
            "generate_executive_summary": _generate_executive_summary,
            "generate_report": _generate_report,
            "generate_gitlab_architecture": _generate_gitlab_architecture,
        },
    )


def build_agent_service_chat_core_provider(service: Any) -> FunctionToolProvider:
    """Remaining chat-only core tools for AgentService."""

    async def _request_file_upload(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        return {
            "action": "upload_requested",
            "message": "Please select a file to upload using the upload button.",
            "suggested_title": params.get("suggested_title"),
            "suggested_tags": params.get("suggested_tags", []),
        }

    async def _answer_question(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        return await service._tool_answer_question(params, ctx.user_id, ctx.db)

    async def _delegate_to_agent(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        return await service._tool_delegate_to_agent(params, ctx.user_id, ctx.db)

    async def _list_available_agents(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        return await service._tool_list_available_agents(ctx.db)

    return FunctionToolProvider(
        name="agent_service_chat_core_tools",
        modes={"chat"},
        handlers={
            "request_file_upload": _request_file_upload,
            "answer_question": _answer_question,
            "delegate_to_agent": _delegate_to_agent,
            "list_available_agents": _list_available_agents,
        },
    )
