"""App-layer tool provider registry for agent services."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Protocol

from sqlalchemy import select

from app.services import llm_structured
from app.services.data_analysis_tools import (
    DATA_ANALYSIS_EXPOSED_NAMES,
    DATA_ANALYSIS_TOOL_DEFINITIONS,
)


def _unimplemented_tool(tool_name: str) -> Dict[str, Any]:
    """Report a capability that does not exist, instead of faking success.

    These handlers used to return {"success": True, ...} with invented fields —
    "relationship_created": True from a function that created nothing,
    "Comparison would be generated here" as an actual result. The agent believed
    the work happened and recorded it as evidence, which corrupts every
    downstream claim built on it.

    Returning an error marks the call failed (success is derived as `not
    error`), so the loop's existing tool-failure handling takes over and the
    agent can pick a different route rather than proceeding on a fiction.
    """
    return {
        "error": (
            f"Tool '{tool_name}' is not implemented. It is advertised but has no "
            "behaviour behind it; do not retry, choose a different approach."
        ),
        "unimplemented": True,
    }


async def _load_documents_for_analysis(
    ctx: AgentToolExecutionContext, document_ids: Any, *, max_docs: int = 8
) -> tuple[list[Any], Optional[str]]:
    """Load documents by id for an analysis tool, or return why it cannot run.

    Bounded deliberately: these tools feed document text to a model, and an
    unbounded list would blow the context window rather than fail cleanly.
    """
    from uuid import UUID as _UUID

    from app.services.document_service import DocumentService

    if not isinstance(document_ids, list) or not document_ids:
        return [], "document_ids must be a non-empty list"

    service = DocumentService()
    documents: list[Any] = []
    missing: list[str] = []
    for raw in document_ids[:max_docs]:
        try:
            doc = await service.get_document(_UUID(str(raw)), ctx.db)
        except (TypeError, ValueError):
            missing.append(str(raw))
            continue
        if doc is None:
            missing.append(str(raw))
        else:
            documents.append(doc)

    if not documents:
        return [], f"No documents found for: {', '.join(missing) or document_ids}"
    return documents, None


def _document_excerpts(documents: list[Any], *, per_doc_chars: int = 4000) -> str:
    """Render documents for a prompt, truncated per document rather than overall
    so a long first document cannot crowd the others out entirely."""
    blocks = []
    for doc in documents:
        content = (getattr(doc, "content", "") or "")[:per_doc_chars]
        blocks.append(
            f"### {getattr(doc, 'title', 'Untitled')} (id={getattr(doc, 'id', '')})\n{content}"
        )
    return "\n\n".join(blocks)


_CLUSTER_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "common_themes": {"type": "array", "items": {"type": "string"}},
        "differences": {"type": "array", "items": {"type": "string"}},
        "patterns": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary"],
}

_METHODOLOGY_COMPARISON_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "comparisons": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "aspect": {"type": "string"},
                    "finding": {"type": "string"},
                },
                "required": ["aspect", "finding"],
            },
        },
        "shared_approaches": {"type": "array", "items": {"type": "string"}},
        "notable_differences": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary"],
}

_RESEARCH_GAPS_SCHEMA = {
    "type": "object",
    "properties": {
        "gaps_identified": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "gap": {"type": "string"},
                    "why_it_matters": {"type": "string"},
                    "supporting_evidence": {"type": "string"},
                },
                "required": ["gap"],
            },
        },
        "opportunities": {"type": "array", "items": {"type": "string"}},
        "evidence_sufficient": {"type": "boolean"},
    },
    "required": ["gaps_identified"],
}


@dataclass(slots=True)
class AgentToolExecutionContext:
    """Execution context for app-side tool providers."""

    mode: str
    db: Any
    service: Any
    user_id: Any = None
    job: Any = None
    state: Optional[Dict[str, Any]] = None
    idempotency_key: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


class AgentToolProvider(Protocol):
    @property
    def supported_tools(self) -> set[str]:
        ...

    def can_handle(self, tool_name: str, context: AgentToolExecutionContext) -> bool:
        ...

    async def execute(
        self,
        tool_name: str,
        params: Dict[str, Any],
        context: AgentToolExecutionContext,
    ) -> Any:
        ...


class FunctionToolProvider:
    """Simple provider backed by async callables."""

    def __init__(
        self,
        *,
        name: str,
        handlers: Dict[
            str, Callable[[Dict[str, Any], AgentToolExecutionContext], Awaitable[Any]]
        ],
        modes: Optional[Iterable[str]] = None,
    ) -> None:
        self.name = name
        self._handlers = dict(handlers)
        self._modes = set(modes or [])

    @property
    def supported_tools(self) -> set[str]:
        return set(self._handlers.keys())

    def can_handle(self, tool_name: str, context: AgentToolExecutionContext) -> bool:
        if self._modes and context.mode not in self._modes:
            return False
        return tool_name in self._handlers

    async def execute(
        self,
        tool_name: str,
        params: Dict[str, Any],
        context: AgentToolExecutionContext,
    ) -> Any:
        job_config = (
            context.job.config
            if isinstance(getattr(context.job, "config", None), dict)
            else {}
        )

        def _tool_set(value: Any) -> set[str]:
            if isinstance(value, list):
                return {str(item).strip() for item in value if str(item).strip()}
            if isinstance(value, str):
                return {item.strip() for item in value.split(",") if item.strip()}
            return set()

        allowed_tools = _tool_set(
            job_config.get("allowed_tools") or job_config.get("tool_allowlist")
        )
        blocked_tools = _tool_set(
            job_config.get("blocked_tools") or job_config.get("tool_denylist")
        )
        if tool_name in blocked_tools or (
            allowed_tools and tool_name not in allowed_tools
        ):
            return {
                "success": False,
                "error": (
                    f"Tool '{tool_name}' is not permitted by this agent's "
                    "enforced tool policy"
                ),
            }
        return await self._handlers[tool_name](params, context)


class AgentToolRegistry:
    """Resolves and executes tool providers."""

    def __init__(self, providers: Optional[Iterable[AgentToolProvider]] = None) -> None:
        self._providers = list(providers or [])

    def register(self, provider: AgentToolProvider) -> None:
        self._providers.append(provider)

    def resolve(
        self, tool_name: str, context: AgentToolExecutionContext
    ) -> Optional[AgentToolProvider]:
        for provider in self._providers:
            if provider.can_handle(tool_name, context):
                return provider
        return None

    async def try_execute(
        self,
        tool_name: str,
        params: Dict[str, Any],
        context: AgentToolExecutionContext,
    ) -> tuple[bool, Any]:
        provider = self.resolve(tool_name, context)
        if provider is None:
            return False, None
        await self._verify_instrument(provider, tool_name, context)
        result = await self._execute_maybe_replicated(
            provider, tool_name, params, context
        )
        result = self._check_measures_what_it_names(tool_name, params, result)
        await self._record_evidence(tool_name, params, result, context)
        return True, result

    @staticmethod
    async def _execute_maybe_replicated(
        provider: AgentToolProvider,
        tool_name: str,
        params: Dict[str, Any],
        context: AgentToolExecutionContext,
    ) -> Any:
        """Take a nondeterministic measurement several times, once.

        Only tools whose answers actually move are replicated -- callgrind
        counts, llvm-mca and gem5 are deterministic, and calling them three
        times buys the same number at three times the cost.

        A control call is never itself replicated: the controls already run a
        median over 31 rounds internally, and replicating them would multiply
        the cost of verifying the instrument by the cost of using it.
        """
        from loguru import logger

        from app.services import agent_measurement_replication as replication
        from app.services import agent_tool_controls as controls

        if not replication.is_replicated(tool_name):
            return await provider.execute(tool_name, params, context)
        if controls.is_control_call(params):
            return await provider.execute(tool_name, params, context)

        async def _once() -> Any:
            return await provider.execute(tool_name, dict(params), context)

        try:
            return await replication.run_replicated(_once, tool_name)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"Could not replicate {tool_name}: {exc}")
            return await provider.execute(tool_name, params, context)

    @staticmethod
    async def _verify_instrument(
        provider: AgentToolProvider,
        tool_name: str,
        context: AgentToolExecutionContext,
    ) -> None:
        """Run this tool's controls before its first use in the run.

        Here for the same reason evidence capture is here: every call passes
        through this point, so the run cannot use a measurement tool without
        first establishing that the tool works. A control the agent has to
        remember is a control that is missing from whichever run mattered.

        Only the *opening* half of the bracket can be automated -- nothing at
        call time knows which measurement is the last one. The closing half is
        the evaluate phase's job, and `validity.instruments_verified` refuses
        the run until it has happened.

        Never allowed to fail the call it precedes. A failing control does not
        stop the tool; it records that nothing the tool produces in this window
        is evidence, which the contract then acts on.
        """
        from loguru import logger

        from app.services import agent_tool_controls as controls

        if not controls.is_controlled(tool_name):
            return
        state = getattr(context, "state", None)
        if not isinstance(state, dict):
            return
        if not controls.needs_pre_control(state, tool_name):
            return

        async def _call(name: str, params: Dict[str, Any]) -> Any:
            return await provider.execute(name, params, context)

        try:
            verdicts = await controls.run_controls(
                _call, tool_name, state, when="before"
            )
            failed = [v for v in verdicts if not v.get("passed")]
            if failed:
                logger.warning(
                    f"Instrument control failed for {tool_name}: "
                    f"{failed[0].get('reason')}"
                )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"Could not run controls for {tool_name}: {exc}")

    @staticmethod
    def _check_measures_what_it_names(
        tool_name: str, params: Dict[str, Any], result: Any
    ) -> Any:
        """Ask whether the call measured the thing it named.

        The one failure controls and replication both miss, because it is
        neither broken nor noisy: a chain that reaches infinity is precise,
        stable, reproducible and about something else.

        Attached to the result and to its findings rather than raised. A tool
        that refused would strand a run mid-measurement over an analysis this
        module can only sometimes perform; the contract is where the judgement
        belongs.
        """
        from loguru import logger

        from app.services import agent_measurement_replication as replication
        from app.services import agent_measurement_sanity as sanity

        if not replication.is_replicated(tool_name):
            return result
        if not isinstance(result, dict):
            return result

        try:
            verdict = sanity.check(params, result)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"Could not sanity-check {tool_name}: {exc}")
            return result

        if not verdict.get("checked"):
            return result
        if not verdict.get("sound"):
            logger.warning(
                f"{tool_name} may not have measured what it named: "
                f"{'; '.join(verdict.get('problems') or [])[:300]}"
            )

        enriched = dict(result)
        data = (
            dict(enriched.get("data") or {})
            if isinstance(enriched.get("data"), dict)
            else {}
        )
        data["measurement_sanity"] = verdict
        enriched["data"] = data
        findings = enriched.get("findings")
        if isinstance(findings, list):
            enriched["findings"] = [
                {**f, "measurement_sanity": verdict} if isinstance(f, dict) else f
                for f in findings
            ]
        return enriched

    @staticmethod
    async def _record_evidence(
        tool_name: str,
        params: Dict[str, Any],
        result: Any,
        context: AgentToolExecutionContext,
    ) -> None:
        """Add this call to the run's bundle, as it happens.

        Here rather than in each tool because every call passes through this
        point: a bundle that depends on tools opting in is a bundle missing
        whichever tool was added last. Failures are recorded too -- a run that
        cited a measurement from a call that had failed is only visible if the
        failure is in the record.

        Never allowed to affect the call itself. A bundle is a description of
        the run, not a participant in it.
        """
        from loguru import logger

        from app.services import agent_evidence_bundle as bundle

        if tool_name not in bundle.EVIDENCE_TOOLS:
            return
        job_id = getattr(getattr(context, "job", None), "id", None)
        if not job_id:
            return
        try:
            image = ""
            if isinstance(result, dict):
                image = str((result.get("data") or {}).get("image") or "")
            if not image:
                image = str(params.get("image") or "")
            image_id = await bundle.resolve_image_id(image) if image else ""
            bundle.record_entry(
                job_id=str(job_id),
                tool=tool_name,
                params=params if isinstance(params, dict) else {"params": params},
                result=result,
                image_id=image_id,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"evidence bundle: skipped {tool_name}: {exc}")


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


def _tool_snapshot_context(ctx: Any, tool: str) -> Dict[str, Any]:
    """Attribute a tool's own LLM call to the job phase that made it.

    Without db and this context the snapshot recorder returns early, so these
    calls are absent from a run's export and its captured total falls short of
    the calls the job reports making.
    """
    job = getattr(ctx, "job", None)
    return {
        "job_id": str(getattr(job, "id", "") or "") or None,
        "iteration": int(getattr(job, "iteration", 0) or 0),
        "phase": f"tool:{tool}",
    }


def build_autonomous_research_provider(executor: Any) -> FunctionToolProvider:
    """Research-family tools for AutonomousAgentExecutor."""

    async def _arxiv_search(**kwargs: Any) -> List[Dict[str, Any]]:
        """Run an arXiv search and return its entries as plain dicts.

        ``ArxivSearchService.search`` returns an ``ArxivSearchResult`` dataclass;
        every caller here wants the entry list.
        """
        result = await executor.arxiv_service.search(**kwargs)
        items = getattr(result, "items", result) or []
        return [item for item in items if isinstance(item, dict)]

    async def _search_arxiv(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        job = ctx.job
        papers = await _arxiv_search(
            query=params.get("query", job.goal[:100] if job else ""),
            max_results=params.get("max_results", 10),
        )
        return {
            "success": True,
            "data": papers,
            "findings": [
                {
                    "type": "paper",
                    "title": paper.get("title"),
                    "id": paper.get("id"),
                    "arxiv_id": paper.get("id"),
                    "summary": paper.get("summary", "")[:500],
                    "authors": paper.get("authors", []),
                    "published": paper.get("published"),
                }
                for paper in papers[:10]
            ],
        }

    async def _save_research_finding(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        import uuid
        from datetime import datetime

        job = ctx.job
        state = ctx.state if isinstance(ctx.state, dict) else {}
        source_scope_id = str(params.get("source_id") or "").strip() or None
        finding = {
            "id": str(uuid.uuid4()),
            "title": params.get("title"),
            "content": params.get("content"),
            "category": params.get("category"),
            "source_document_ids": params.get("source_document_ids", []),
            "source_id": source_scope_id,
            "confidence": params.get("confidence", 0.8),
            "tags": params.get("tags", []),
            "created_at": datetime.utcnow().isoformat(),
        }

        job_id_str = str(job.id)
        if job_id_str not in executor._job_findings:
            executor._job_findings[job_id_str] = []
        executor._job_findings[job_id_str].append(finding)

        findings = state.get("findings")
        if not isinstance(findings, list):
            findings = []
            state["findings"] = findings
        findings.append(finding)

        return {
            "success": True,
            "data": {"finding_id": finding["id"]},
            "findings": [finding],
        }

    async def _get_research_findings(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        job = ctx.job
        findings = list(executor._job_findings.get(str(job.id), []))
        category = params.get("category")
        if category:
            findings = [
                finding for finding in findings if finding.get("category") == category
            ]
        min_confidence = params.get("min_confidence")
        if min_confidence:
            findings = [
                finding
                for finding in findings
                if finding.get("confidence", 0) >= min_confidence
            ]
        findings = findings[: params.get("limit", 50)]
        return {
            "success": True,
            "data": {"findings": findings, "total": len(findings)},
        }

    async def _ingest_paper_by_id(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        arxiv_id = params.get("arxiv_id")
        if not arxiv_id:
            return {"error": "Missing required parameter: arxiv_id"}
        papers = await _arxiv_search(query=f"id:{arxiv_id}", max_results=1)
        if not papers:
            return {"error": f"Paper {arxiv_id} not found"}
        paper = papers[0]
        return {
            "success": True,
            "data": paper,
            "findings": [
                {
                    "type": "paper_ingested",
                    "arxiv_id": arxiv_id,
                    "title": paper.get("title"),
                }
            ],
        }

    async def _batch_ingest_papers(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        import uuid

        job = ctx.job
        arxiv_ids = [
            x.strip()
            for x in (params.get("arxiv_ids") or [])
            if isinstance(x, str) and x.strip()
        ]
        search_queries = [
            x.strip()
            for x in (params.get("search_queries") or [])
            if isinstance(x, str) and x.strip()
        ]
        categories = [
            x.strip()
            for x in (params.get("categories") or [])
            if isinstance(x, str) and x.strip()
        ]
        max_results = max(1, min(int(params.get("max_results") or 25), 200))
        if not arxiv_ids and not search_queries and not categories:
            return {
                "error": "Provide at least one of: arxiv_ids, search_queries, categories"
            }

        display = params.get("display") or "Autonomous job import"
        source_name = f"ArXiv Import (Job {str(job.id)[:8]}) #{uuid.uuid4().hex[:6]}"
        cfg = {
            "paper_ids": arxiv_ids,
            "search_queries": search_queries,
            "categories": categories,
            "max_results": max_results,
            "requested_by_user_id": str(job.user_id),
            "requested_by": str(job.user_id),
            "display": display,
        }
        source = await executor.document_service.create_document_source(
            name=source_name,
            source_type="arxiv",
            config=cfg,
            db=ctx.db,
        )

        queued = False
        try:
            from app.tasks.ingestion_tasks import ingest_from_source

            ingest_from_source.delay(str(source.id))
            queued = True
        except Exception:
            queued = False

        return {
            "success": True,
            "data": {
                "source_id": str(source.id),
                "source_name": source.name,
                "queued": queued,
                "paper_ids_count": len(arxiv_ids),
                "search_queries_count": len(search_queries),
                "categories_count": len(categories),
                "max_results": max_results,
            },
            "findings": [
                {
                    "type": "arxiv_ingest_requested",
                    "source_id": str(source.id),
                    "queued": queued,
                }
            ],
            "artifacts": [
                {
                    "type": "document_source",
                    "id": str(source.id),
                    "name": source.name,
                    "source_type": "arxiv",
                },
                {
                    "type": "arxiv_ingest_requested",
                    "source_id": str(source.id),
                    "queued": queued,
                },
            ],
        }

    async def _monitor_arxiv_topic(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        topic = params.get("topic")
        papers = await _arxiv_search(
            query=params.get("query") or f"all:{topic}",
            max_results=params.get("max_results", 20),
            sort_by="submittedDate",
            sort_order="descending",
        )
        return {
            "success": True,
            "data": papers,
            "findings": [
                {
                    "type": "new_paper",
                    "title": paper.get("title"),
                    "arxiv_id": paper.get("id"),
                    "published": paper.get("published"),
                }
                for paper in papers[:10]
            ],
        }

    async def _find_related_papers(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from uuid import UUID

        from app.models.document import Document

        query = ""
        doc_id = params.get("document_id")
        arxiv_id = params.get("arxiv_id")
        if doc_id:
            doc_result = await ctx.db.execute(
                select(Document).where(Document.id == UUID(doc_id))
            )
            doc = doc_result.scalar_one_or_none()
            if doc:
                query = doc.title
        elif arxiv_id:
            papers = await _arxiv_search(query=f"id:{arxiv_id}", max_results=1)
            if papers:
                query = papers[0].get("title", "")

        if not query or not params.get("search_external", True):
            return {"error": "No query could be built"}

        related = await _arxiv_search(query=query, max_results=params.get("limit", 10))
        return {
            "success": True,
            "data": related,
            "findings": [
                {
                    "type": "related_paper",
                    "title": paper.get("title"),
                    "arxiv_id": paper.get("id"),
                }
                for paper in related
            ],
        }

    async def _extract_paper_insights(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        import json
        from uuid import UUID

        from app.models.document import Document

        job = ctx.job
        doc_id = params.get("document_id")
        if not doc_id:
            return {"error": "Missing required parameter: document_id"}
        doc_result = await ctx.db.execute(
            select(Document).where(Document.id == UUID(doc_id))
        )
        doc = doc_result.scalar_one_or_none()
        if not doc or not doc.content:
            return {"error": "Document not found or has no content"}

        prompt = f"""Extract key insights from this research paper.
Focus on: {', '.join(params.get('focus_areas', ['methodology', 'results', 'contributions']))}

Paper Title: {doc.title}
Content: {doc.content[:8000]}

Provide structured insights in JSON format:
{{
    "methodology": "...",
    "key_findings": ["..."],
    "contributions": ["..."],
    "limitations": ["..."],
    "future_work": ["..."]
}}"""
        try:
            response = await executor.llm_service.generate_response(
                system_prompt="You are a research paper analyst. Extract structured insights.",
                user_message=prompt,
                routing=executor._llm_routing_from_job_config(job.config),
                task_type="summarization",
                user_id=job.user_id,
                db=ctx.db,
                snapshot_context=_tool_snapshot_context(ctx, "extract_paper_insights"),
            )
            insights = json.loads(response)
            return {
                "success": True,
                "data": insights,
                "findings": [
                    {
                        "type": "paper_insights",
                        "document_id": doc_id,
                        "insights": insights,
                        "source_id": str(doc.source_id)
                        if getattr(doc, "source_id", None)
                        else None,
                    }
                ],
            }
        except Exception as exc:
            raw = response if "response" in locals() else str(exc)
            return {"success": True, "data": {"raw_analysis": raw}}

    async def _create_synthesis_document(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        import hashlib
        import uuid

        job = ctx.job
        title = params.get("title")
        topic = params.get("topic")
        document_ids = params.get("document_ids", [])
        persist = bool(params.get("persist")) or bool(
            (job.config or {}).get("persist_artifacts", False)
        )
        scoped_source_id = str(params.get("source_id") or "").strip()

        findings = list(executor._job_findings.get(str(job.id), []))
        if scoped_source_id:
            findings = [
                finding
                for finding in findings
                if not isinstance(finding, dict)
                or not str(finding.get("source_id") or "").strip()
                or str(finding.get("source_id") or "").strip() == scoped_source_id
            ]

        synthesis_content = f"# {title}\n\n## Research Topic\n{topic}\n\n"
        if findings:
            synthesis_content += "## Key Findings\n"
            for i, finding in enumerate(findings[:20], 1):
                synthesis_content += f"\n### {i}. {finding.get('title', 'Finding')}\n"
                synthesis_content += f"{finding.get('content', '')}\n"
                if finding.get("category"):
                    synthesis_content += f"*Category: {finding['category']}*\n"

        result = {
            "success": True,
            "data": {
                "title": title,
                "content": synthesis_content,
                "findings_included": len(findings),
            },
            "artifacts": [
                {
                    "type": "synthesis_document",
                    "title": title,
                    "content": synthesis_content,
                }
            ],
        }

        if persist and title and synthesis_content.strip():
            try:
                from app.models.document import Document

                notes_source = (
                    await executor.document_service._get_or_create_agent_notes_source(
                        ctx.db
                    )
                )
                content_hash = hashlib.sha256(
                    synthesis_content.encode("utf-8")
                ).hexdigest()
                doc = Document(
                    title=str(title).strip(),
                    content=synthesis_content,
                    content_hash=content_hash,
                    url=None,
                    file_path=None,
                    file_type="text/markdown",
                    file_size=len(synthesis_content.encode("utf-8")),
                    source_id=notes_source.id,
                    source_identifier=f"agent_synthesis:{uuid.uuid4().hex}",
                    author=None,
                    tags=["autonomous_job", "research"],
                    extra_metadata={
                        "origin": "autonomous_job",
                        "job_id": str(job.id),
                        "job_type": job.job_type,
                        "topic": topic,
                        "document_ids": document_ids,
                        "source_scope_id": scoped_source_id or None,
                    },
                    is_processed=False,
                )
                ctx.db.add(doc)
                await ctx.db.commit()
                await ctx.db.refresh(doc)
                try:
                    await executor.document_service.reprocess_document(
                        doc.id, ctx.db, user_id=job.user_id
                    )
                except Exception:
                    pass
                result["data"]["document_id"] = str(doc.id)
                result["artifacts"].append(
                    {"type": "document", "id": str(doc.id), "title": doc.title}
                )
            except Exception:
                pass

        return result

    async def _compare_methodologies(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        """Compare the methodologies described across several papers."""
        documents, error = await _load_documents_for_analysis(
            ctx, params.get("document_ids")
        )
        if error:
            return {"error": error}

        aspects = params.get("comparison_aspects")
        if not isinstance(aspects, list) or not aspects:
            aspects = ["approach", "results"]

        payload = await llm_structured.ask_for_json(
            executor.llm_service,
            schema=_METHODOLOGY_COMPARISON_SCHEMA,
            user_message=(
                "Compare the methodologies in these papers. For each aspect, say "
                "how the papers differ and what that implies. Return JSON with "
                "comparisons, shared_approaches, notable_differences and summary.\n"
                f"ASPECTS: {', '.join(str(a) for a in aspects)}\n\n"
                + _document_excerpts(documents)
            ),
            task_type="methodology_comparison",
            temperature=0.2,
            max_tokens=1800,
            user_id=ctx.user_id,
            db=ctx.db,
        )
        if payload is None:
            return {"error": "The model did not return a usable comparison"}
        payload["documents_compared"] = [str(d.id) for d in documents]
        return {"success": True, "data": payload}

    async def _identify_research_gaps(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        """Identify research gaps from the job's findings and named documents."""
        job = ctx.job
        findings = list(executor._job_findings.get(str(getattr(job, "id", "")), []))
        documents: list[Any] = []
        if isinstance(params.get("document_ids"), list) and params["document_ids"]:
            documents, error = await _load_documents_for_analysis(
                ctx, params.get("document_ids")
            )
            if error:
                return {"error": error}

        if not findings and not documents:
            return {
                "error": (
                    "No evidence to analyse: pass document_ids, or record "
                    "findings first with save_research_finding"
                )
            }

        topic = str(params.get("topic") or getattr(job, "goal", "") or "").strip()
        import json as _json

        evidence = _json.dumps(findings[:40], ensure_ascii=False, default=str)
        payload = await llm_structured.ask_for_json(
            executor.llm_service,
            schema=_RESEARCH_GAPS_SCHEMA,
            user_message=(
                "Identify research gaps and opportunities from the evidence "
                "below. A gap must be supported by what is present; say so "
                "explicitly when the evidence is too thin to support any.\n"
                f"TOPIC: {topic}\n\nFINDINGS:\n{evidence}\n\n"
                + (_document_excerpts(documents) if documents else "")
            ),
            task_type="research_gap_analysis",
            temperature=0.3,
            max_tokens=1500,
            user_id=ctx.user_id,
            db=ctx.db,
        )
        if payload is None:
            return {"error": "The model did not return a usable gap analysis"}
        payload["findings_analyzed"] = len(findings)
        payload["topic"] = topic
        return {"success": True, "data": payload}

    async def _add_to_reading_list(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from uuid import UUID

        from sqlalchemy import func
        from sqlalchemy.exc import IntegrityError

        from app.models.document import Document
        from app.models.reading_list import ReadingList, ReadingListItem

        job = ctx.job
        list_name = (params.get("list_name") or "").strip()
        items = params.get("items", []) or []
        scoped_source_id = str(params.get("source_id") or "").strip()
        scoped_source_uuid = None
        if scoped_source_id:
            try:
                scoped_source_uuid = UUID(scoped_source_id)
            except Exception:
                scoped_source_uuid = None
        if not list_name:
            return {"error": "Missing required parameter: list_name"}
        if not isinstance(items, list) or not items:
            return {"error": "Missing required parameter: items"}

        rl_res = await ctx.db.execute(
            select(ReadingList).where(
                ReadingList.user_id == job.user_id, ReadingList.name == list_name
            )
        )
        rl = rl_res.scalar_one_or_none()
        if not rl:
            rl = ReadingList(
                user_id=job.user_id,
                name=list_name,
                description=None,
                source_id=scoped_source_uuid,
            )
            ctx.db.add(rl)
            await ctx.db.flush()

        max_pos = int(
            (
                await ctx.db.execute(
                    select(func.max(ReadingListItem.position)).where(
                        ReadingListItem.reading_list_id == rl.id
                    )
                )
            ).scalar()
            or 0
        )
        added = 0
        skipped = 0
        warnings: list[str] = []

        for raw in items:
            if not isinstance(raw, dict):
                skipped += 1
                continue
            doc_id = raw.get("document_id")
            arxiv_id = raw.get("arxiv_id")
            notes = raw.get("notes")
            priority = int(raw.get("priority", 3) or 3)

            doc = None
            if doc_id:
                try:
                    doc = await ctx.db.get(Document, UUID(str(doc_id)))
                except Exception:
                    doc = None
            elif arxiv_id:
                arxiv_id = str(arxiv_id).strip()
                if arxiv_id.startswith("arxiv:"):
                    arxiv_id = arxiv_id.split("arxiv:", 1)[1].strip()
                if arxiv_id:
                    doc_res = await ctx.db.execute(
                        select(Document)
                        .where(Document.source_identifier == arxiv_id)
                        .limit(1)
                    )
                    doc = doc_res.scalar_one_or_none()

            if not doc:
                skipped += 1
                if arxiv_id:
                    warnings.append(f"Document not found for arXiv id: {arxiv_id}")
                elif doc_id:
                    warnings.append(f"Document not found for id: {doc_id}")
                continue
            if (
                scoped_source_uuid
                and getattr(doc, "source_id", None) != scoped_source_uuid
            ):
                skipped += 1
                warnings.append(
                    f"Document {doc.id} is outside scoped source {scoped_source_id}"
                )
                continue

            exists = await ctx.db.execute(
                select(func.count())
                .select_from(ReadingListItem)
                .where(
                    ReadingListItem.reading_list_id == rl.id,
                    ReadingListItem.document_id == doc.id,
                )
            )
            if int(exists.scalar() or 0) > 0:
                skipped += 1
                continue

            item = ReadingListItem(
                reading_list_id=rl.id,
                document_id=doc.id,
                status="to-read",
                priority=max(0, min(priority, 5)),
                position=max_pos + 1,
                notes=str(notes).strip()[:2000] if notes else None,
            )
            ctx.db.add(item)
            try:
                await ctx.db.flush()
            except IntegrityError:
                await ctx.db.rollback()
                skipped += 1
                continue
            max_pos += 1
            added += 1

        await ctx.db.commit()
        return {
            "success": True,
            "data": {
                "reading_list_id": str(rl.id),
                "list_name": rl.name,
                "items_added": added,
                "items_skipped": skipped,
                "warnings": warnings[:25],
            },
            "artifacts": [
                {
                    "type": "reading_list",
                    "id": str(rl.id),
                    "name": rl.name,
                    "items_added": added,
                }
            ],
        }

    async def _get_reading_lists(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from uuid import UUID

        from sqlalchemy import desc

        from app.models.document import Document
        from app.models.reading_list import ReadingList, ReadingListItem

        job = ctx.job
        list_name = (params.get("list_name") or "").strip()
        include_items = bool(params.get("include_items", True))
        scoped_source_id = str(params.get("source_id") or "").strip()
        scoped_source_uuid = None
        if scoped_source_id:
            try:
                scoped_source_uuid = UUID(scoped_source_id)
            except Exception:
                scoped_source_uuid = None

        q = (
            select(ReadingList)
            .where(ReadingList.user_id == job.user_id)
            .order_by(desc(ReadingList.updated_at))
        )
        if list_name:
            q = q.where(ReadingList.name == list_name)
        if scoped_source_uuid:
            q = q.where(ReadingList.source_id == scoped_source_uuid)

        lists = (await ctx.db.execute(q.limit(100))).scalars().all()
        payload = []
        for rl in lists:
            entry: dict[str, Any] = {
                "id": str(rl.id),
                "name": rl.name,
                "description": rl.description,
                "created_at": rl.created_at.isoformat() if rl.created_at else None,
                "updated_at": rl.updated_at.isoformat() if rl.updated_at else None,
            }
            if include_items:
                items_res = await ctx.db.execute(
                    select(ReadingListItem, Document.title)
                    .join(Document, Document.id == ReadingListItem.document_id)
                    .where(ReadingListItem.reading_list_id == rl.id)
                    .order_by(
                        ReadingListItem.position.asc(), ReadingListItem.created_at.asc()
                    )
                )
                entry["items"] = [
                    {
                        "id": str(item.id),
                        "document_id": str(item.document_id),
                        "document_title": title,
                        "status": item.status,
                        "priority": item.priority,
                        "position": item.position,
                        "notes": item.notes,
                        "created_at": item.created_at.isoformat()
                        if item.created_at
                        else None,
                    }
                    for item, title in items_res.all()
                ]
            payload.append(entry)
        return {
            "success": True,
            "data": {"reading_lists": payload, "total": len(payload)},
        }

    async def _write_progress_report(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from datetime import datetime

        job = ctx.job
        state = ctx.state if isinstance(ctx.state, dict) else {}
        report = {
            "summary": params.get("summary"),
            "completed_tasks": params.get("completed_tasks", []),
            "pending_tasks": params.get("pending_tasks", []),
            "key_findings": params.get("key_findings", []),
            "blockers": params.get("blockers", []),
            "next_steps": params.get("next_steps", []),
            "iteration": job.iteration,
            "progress": job.progress,
            "timestamp": datetime.utcnow().isoformat(),
        }
        reports = state.get("progress_reports")
        if not isinstance(reports, list):
            reports = []
            state["progress_reports"] = reports
        reports.append(report)
        return {
            "success": True,
            "data": report,
            "artifacts": [{"type": "progress_report", "report": report}],
        }

    async def _suggest_next_action(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        job = ctx.job
        state = ctx.state if isinstance(ctx.state, dict) else {}
        prompt = f"""Given the current research goal and progress, suggest the best next action.

Goal: {params.get('current_goal', job.goal)}
Progress so far: {params.get('progress_so_far', '')}
Findings count: {len(state.get('findings', []))}
Iteration: {job.iteration}/{job.max_iterations}

Available actions:
- Search for more papers on arXiv
- Analyze existing documents
- Synthesize findings
- Create a report
- Monitor for new papers

Suggest the single best next action and explain why."""
        try:
            suggestion = await executor.llm_service.generate_response(
                system_prompt="You are a research planning assistant.",
                user_message=prompt,
                routing=executor._llm_routing_from_job_config(job.config),
                task_type="research_engineer_scientist",
                user_id=job.user_id,
                db=ctx.db,
                snapshot_context=_tool_snapshot_context(ctx, "suggest_next_action"),
            )
            return {"success": True, "data": {"suggestion": suggestion}}
        except Exception as exc:
            return {"error": str(exc)}

    async def _generate_research_presentation(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        """Queue a real presentation job.

        Previously reported presentation_queued=True without queueing anything.
        """
        from app.models.presentation import PresentationJob
        from app.tasks.presentation_tasks import generate_presentation_task

        title = str(params.get("title") or "").strip()
        topic = str(params.get("topic") or "").strip()
        if not title or not topic:
            return {"error": "title and topic are required"}

        user_id = ctx.user_id or getattr(ctx.job, "user_id", None)
        if user_id is None:
            return {"error": "no user context for the presentation job"}

        try:
            slide_count = int(params.get("slide_count", 12) or 12)
        except (TypeError, ValueError):
            slide_count = 12
        slide_count = max(1, min(slide_count, 60))

        document_ids = params.get("source_document_ids")
        job_record = PresentationJob(
            user_id=user_id,
            title=title[:500],
            topic=topic[:500],
            source_document_ids=(
                [str(d) for d in document_ids] if isinstance(document_ids, list) else []
            ),
            slide_count=slide_count,
            style=str(params.get("style") or "professional"),
            include_diagrams=1 if params.get("include_diagrams", True) else 0,
            status="pending",
            progress=0,
        )
        ctx.db.add(job_record)
        await ctx.db.commit()
        await ctx.db.refresh(job_record)
        generate_presentation_task.delay(str(job_record.id), str(user_id))

        return {
            "success": True,
            "data": {
                "presentation_queued": True,
                "presentation_job_id": str(job_record.id),
                "title": job_record.title,
                "topic": job_record.topic,
                "slides": slide_count,
            },
        }

    async def _analyze_document_cluster(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        """Find common themes, differences and patterns across documents."""
        documents, error = await _load_documents_for_analysis(
            ctx, params.get("document_ids")
        )
        if error:
            return {"error": error}

        analysis_type = str(params.get("analysis_type") or "comprehensive").strip()
        payload = await llm_structured.ask_for_json(
            executor.llm_service,
            schema=_CLUSTER_ANALYSIS_SCHEMA,
            user_message=(
                "Analyse this set of documents as a cluster. Return JSON with "
                "common_themes, differences, patterns and a short summary.\n"
                f"ANALYSIS TYPE: {analysis_type}\n\n" + _document_excerpts(documents)
            ),
            task_type="document_cluster_analysis",
            temperature=0.2,
            max_tokens=1500,
            user_id=ctx.user_id,
            db=ctx.db,
        )
        if payload is None:
            return {"error": "The model did not return a usable cluster analysis"}
        payload["documents_analyzed"] = [str(d.id) for d in documents]
        return {"success": True, "data": payload}

    return FunctionToolProvider(
        name="autonomous_research_tools",
        modes={"autonomous"},
        handlers={
            "search_arxiv": _search_arxiv,
            "save_research_finding": _save_research_finding,
            "get_research_findings": _get_research_findings,
            "ingest_paper_by_id": _ingest_paper_by_id,
            "batch_ingest_papers": _batch_ingest_papers,
            "monitor_arxiv_topic": _monitor_arxiv_topic,
            "find_related_papers": _find_related_papers,
            "extract_paper_insights": _extract_paper_insights,
            "create_synthesis_document": _create_synthesis_document,
            "compare_methodologies": _compare_methodologies,
            "identify_research_gaps": _identify_research_gaps,
            "add_to_reading_list": _add_to_reading_list,
            "get_reading_lists": _get_reading_lists,
            "write_progress_report": _write_progress_report,
            "suggest_next_action": _suggest_next_action,
            "generate_research_presentation": _generate_research_presentation,
            "analyze_document_cluster": _analyze_document_cluster,
        },
    )


# The exposed names live with the definitions, in data_analysis_tools: the
# rename below is part of each tool's public name, and every surface that
# advertises or governs these tools has to agree with dispatch about it.


def build_autonomous_data_analysis_provider(executor: Any) -> FunctionToolProvider:
    """Data-analysis tools for AutonomousAgentExecutor."""

    def _get_tools(ctx: AgentToolExecutionContext) -> Any:
        from app.services.data_analysis_tools import DataAnalysisTools

        job = ctx.job
        job_id_str = str(job.id)
        if job_id_str not in executor._data_analysis_tools:
            executor._data_analysis_tools[job_id_str] = DataAnalysisTools(
                job_id=job_id_str,
                user_id=str(job.user_id),
            )
        return executor._data_analysis_tools[job_id_str]

    async def _execute(
        tool_name: str, params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        tools = _get_tools(ctx)
        if tool_name == "load_csv_data":
            tool_result = tools.load_csv_data(
                content=params.get("content", ""),
                name=params.get("name", "dataset"),
                delimiter=params.get("delimiter", ","),
                has_header=params.get("has_header", True),
            )
        elif tool_name == "load_json_data":
            tool_result = tools.load_json_data(
                content=params.get("content", ""),
                name=params.get("name", "dataset"),
            )
        elif tool_name == "create_dataset":
            tool_result = tools.create_dataset(
                data=params.get("data", {}),
                name=params.get("name", "dataset"),
            )
        elif tool_name == "list_datasets":
            tool_result = tools.list_datasets()
        elif tool_name == "describe_dataset":
            tool_result = tools.describe_dataset(dataset_id=params.get("dataset_id"))
        elif tool_name == "query_data":
            tool_result = tools.query_data(
                dataset_id=params.get("dataset_id"),
                query=params.get("query"),
            )
        elif tool_name == "filter_data":
            tool_result = tools.filter_data(
                dataset_id=params.get("dataset_id"),
                conditions=params.get("conditions", {}),
            )
        elif tool_name == "aggregate_data":
            tool_result = tools.aggregate_data(
                dataset_id=params.get("dataset_id"),
                group_by=params.get("group_by"),
                aggregations=params.get("aggregations"),
            )
        elif tool_name == "join_datasets":
            tool_result = tools.join_datasets(
                left_dataset_id=params.get("left_dataset_id"),
                right_dataset_id=params.get("right_dataset_id"),
                on=params.get("on"),
                left_on=params.get("left_on"),
                right_on=params.get("right_on"),
                how=params.get("how", "inner"),
            )
        elif tool_name == "transform_data":
            tool_result = tools.transform_data(
                dataset_id=params.get("dataset_id"),
                operations=params.get("operations", []),
            )
        elif tool_name == "detect_anomalies":
            tool_result = tools.detect_anomalies(
                dataset_id=params.get("dataset_id"),
                columns=params.get("columns"),
                method=params.get("method", "zscore"),
                threshold=params.get("threshold", 3.0),
            )
        elif tool_name == "calculate_correlations":
            tool_result = tools.calculate_correlations(
                dataset_id=params.get("dataset_id"),
                columns=params.get("columns"),
                method=params.get("method", "pearson"),
            )
        elif tool_name == "create_chart":
            tool_result = tools.create_chart(
                dataset_id=params.get("dataset_id"),
                chart_type=params.get("chart_type", "bar"),
                x_column=params.get("x_column"),
                y_columns=params.get("y_columns"),
                title=params.get("title", ""),
                config=params.get("config"),
            )
        elif tool_name == "create_correlation_heatmap":
            tool_result = tools.create_correlation_heatmap(
                dataset_id=params.get("dataset_id"),
                title=params.get("title", "Correlation Matrix"),
            )
        elif tool_name == "create_flowchart":
            tool_result = tools.create_flowchart(
                nodes=params.get("nodes", []),
                edges=params.get("edges", []),
                title=params.get("title", ""),
                direction=params.get("direction", "TD"),
            )
        elif tool_name == "create_sequence_diagram":
            tool_result = tools.create_sequence_diagram(
                participants=params.get("participants", []),
                messages=params.get("messages", []),
                title=params.get("title", ""),
            )
        elif tool_name == "create_er_diagram":
            tool_result = tools.create_er_diagram(
                entities=params.get("entities", []),
                relationships=params.get("relationships", []),
                title=params.get("title", ""),
            )
        elif tool_name == "create_architecture_diagram":
            tool_result = tools.create_architecture_diagram(
                components=params.get("components", []),
                connections=params.get("connections", []),
                title=params.get("title", ""),
                format=params.get("format", "auto"),
            )
        elif tool_name == "create_drawio_diagram":
            tool_result = tools.create_drawio_diagram(
                nodes=params.get("nodes", []),
                edges=params.get("edges", []),
                title=params.get("title", ""),
            )
        elif tool_name == "create_gantt_chart":
            tool_result = tools.create_gantt_chart(
                sections=params.get("sections", []),
                title=params.get("title", "Project Timeline"),
            )
        elif tool_name == "export_dataset_csv":
            tool_result = tools.export_dataset_csv(dataset_id=params.get("dataset_id"))
        elif tool_name == "export_dataset_json":
            tool_result = tools.export_dataset_json(dataset_id=params.get("dataset_id"))
        else:
            tool_result = {
                "success": False,
                "error": f"Unknown data analysis tool: {tool_name}",
            }

        result: Dict[str, Any] = {
            "success": tool_result.get("success", False),
            "data": tool_result,
        }
        if tool_result.get("success"):
            artifacts = []
            if tool_result.get("image_base64"):
                artifacts.append(
                    {
                        "type": "chart" if "chart" in tool_name else "diagram",
                        "tool": tool_name,
                        "image_base64": tool_result["image_base64"],
                        "mime_type": tool_result.get("mime_type", "image/png"),
                    }
                )
            if tool_result.get("mermaid_code"):
                artifacts.append(
                    {
                        "type": "diagram",
                        "format": "mermaid",
                        "tool": tool_name,
                        "code": tool_result["mermaid_code"],
                    }
                )
            if tool_result.get("xml"):
                artifacts.append(
                    {
                        "type": "diagram",
                        "format": "drawio",
                        "tool": tool_name,
                        "xml": tool_result["xml"],
                        "edit_url": tool_result.get("edit_url"),
                    }
                )
            if tool_result.get("dot_code"):
                artifacts.append(
                    {
                        "type": "diagram",
                        "format": "graphviz",
                        "tool": tool_name,
                        "code": tool_result["dot_code"],
                    }
                )
            if artifacts:
                result["artifacts"] = artifacts

            if tool_name in {
                "detect_anomalies",
                "calculate_correlations",
                "describe_dataset",
            }:
                result["findings"] = [
                    {
                        "type": "data_analysis",
                        "tool": tool_name,
                        "result": tool_result,
                    }
                ]

        return result

    handlers = {
        DATA_ANALYSIS_EXPOSED_NAMES.get(tool_name, tool_name): (
            lambda params, ctx, _tool_name=tool_name: _execute(_tool_name, params, ctx)
        )
        for tool_name in DATA_ANALYSIS_TOOL_DEFINITIONS
    }
    return FunctionToolProvider(
        name="autonomous_data_analysis_tools",
        modes={"autonomous"},
        handlers=handlers,
    )


def build_autonomous_memory_provider(executor: Any) -> FunctionToolProvider:
    """Memory tools for AutonomousAgentExecutor."""

    async def _record_method(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from app.services import agent_method_record

        job = ctx.job
        state = ctx.state if isinstance(ctx.state, dict) else {}
        findings = (
            state.get("findings") if isinstance(state.get("findings"), list) else []
        )
        available = sorted(
            {
                str(f.get("type")).strip()
                for f in findings
                if isinstance(f, dict) and str(f.get("type") or "").strip()
            }
        )

        try:
            record = agent_method_record.build_record(
                name=params.get("name"),
                procedure=params.get("procedure"),
                prevents=params.get("prevents"),
                derived_from=params.get("derived_from"),
                available_finding_types=available,
                applies_to=params.get("applies_to"),
                limits=str(params.get("limits") or ""),
            )
        except agent_method_record.MethodRecordError as exc:
            return {"error": str(exc)}

        content = agent_method_record.render(record)
        try:
            # Constructed directly rather than through MemoryCreate: that
            # schema's types exclude "pattern", and a method stored under a
            # type the job-memory filter does not inject would be written and
            # never recalled -- the one outcome that makes this tool pointless.
            from app.models.memory import ConversationMemory

            stored = ConversationMemory(
                user_id=job.user_id,
                job_id=getattr(job, "id", None),
                memory_type=agent_method_record.MEMORY_TYPE,
                content=content,
                # A method outranks an observation about one subject: it is
                # what a later job on a different subject can still use.
                importance_score=(
                    0.9 if record["status"] == agent_method_record.VALIDATED else 0.6
                ),
                tags=agent_method_record.tags_for(record),
                context={
                    "record": "method",
                    "status": record["status"],
                    "evidence": record["evidence"],
                },
            )
            ctx.db.add(stored)
            await ctx.db.commit()
            await ctx.db.refresh(stored)
        except Exception as exc:
            return {"error": f"Failed to record the method: {str(exc)[:200]}"}

        return {
            "success": True,
            "data": {
                "memory_id": str(stored.id),
                "name": record["name"],
                "status": record["status"],
                "evidence": record["evidence"],
                "note": (
                    "Stored where later jobs recall it. A method recorded "
                    "unvalidated stays that way until a run demonstrates it."
                ),
            },
            "findings": [
                {
                    "type": "method_recorded",
                    "title": f"Method: {record['name']} ({record['status']})",
                    "content": content,
                    "category": "insight",
                }
            ],
        }

    async def _create_memory(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from app.schemas.memory import MemoryCreate

        job = ctx.job
        content_str = str(params.get("content", "")).strip()
        if not content_str:
            return {"error": "content is required"}

        importance = max(0.0, min(1.0, float(params.get("importance", 0.5) or 0.5)))
        category = str(params.get("category", "fact") or "fact")
        metadata = (
            params.get("metadata") if isinstance(params.get("metadata"), dict) else None
        )
        tags = []
        if metadata and isinstance(metadata.get("tags"), list):
            metadata = dict(metadata)
            tags = metadata.pop("tags")
        try:
            memory_data = MemoryCreate(
                memory_type=category,
                content=content_str,
                importance_score=importance,
                context=metadata,
                tags=tags or None,
            )
            mem_resp = await executor.memory_service.create_memory(
                job.user_id, memory_data, ctx.db
            )
            return {
                "success": True,
                "data": {"memory_id": str(mem_resp.id), "content": content_str[:200]},
            }
        except Exception as exc:
            return {"error": f"Failed to create memory: {exc}"}

    async def _search_memories(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from app.schemas.memory import MemorySearchRequest

        job = ctx.job
        query_str = str(params.get("query", "")).strip()
        if not query_str:
            return {"error": "query is required"}

        limit = min(int(params.get("limit", 10) or 10), 50)
        cat_filter = params.get("category_filter")
        min_imp = params.get("min_importance")
        memory_types = [cat_filter] if cat_filter else None
        try:
            search_req = MemorySearchRequest(
                query=query_str,
                limit=limit,
                memory_types=memory_types,
                min_importance=float(min_imp) if min_imp is not None else None,
            )
            memories = await executor.memory_service.search_memories(
                job.user_id, search_req, ctx.db
            )
            return {
                "success": True,
                "data": {
                    "memories": [
                        {
                            "id": str(m.id),
                            "content": m.content,
                            "importance": m.importance_score,
                            "type": m.memory_type,
                        }
                        for m in memories
                    ],
                    "count": len(memories),
                },
            }
        except Exception as exc:
            return {"error": f"Memory search failed: {exc}"}

    async def _recall_memories(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from app.schemas.memory import MemorySearchRequest

        job = ctx.job
        topic = str(params.get("topic", "")).strip()
        if not topic:
            return {"error": "topic is required"}

        limit = min(int(params.get("limit", 10) or 10), 50)
        try:
            search_req = MemorySearchRequest(query=topic, limit=limit)
            memories = await executor.memory_service.search_memories(
                job.user_id, search_req, ctx.db
            )
            return {
                "success": True,
                "data": {
                    "memories": [
                        {
                            "id": str(m.id),
                            "content": m.content,
                            "importance": m.importance_score,
                            "type": m.memory_type,
                        }
                        for m in memories
                    ],
                    "count": len(memories),
                },
            }
        except Exception as exc:
            return {"error": f"Memory recall failed: {exc}"}

    async def _get_memory_stats(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        job = ctx.job
        try:
            stats = await executor.memory_service.get_memory_stats(job.user_id, ctx.db)
            return {
                "success": True,
                "data": {
                    "total_memories": stats.total_memories,
                    "memories_by_type": stats.memories_by_type,
                    "recent_memories": stats.recent_memories,
                },
            }
        except Exception as exc:
            return {"error": f"Failed to get memory stats: {exc}"}

    return FunctionToolProvider(
        name="autonomous_memory_tools",
        modes={"autonomous"},
        handlers={
            "create_memory": _create_memory,
            "record_method": _record_method,
            "search_memories": _search_memories,
            "recall_memories": _recall_memories,
            "get_memory_stats": _get_memory_stats,
        },
    )


def build_autonomous_workflow_provider(executor: Any) -> FunctionToolProvider:
    """Workflow orchestration tools for AutonomousAgentExecutor."""

    async def _list_available_workflows(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from sqlalchemy import select as _select
        from sqlalchemy.orm import selectinload as _selectinload

        from app.models.workflow import Workflow

        job = ctx.job
        try:
            is_active = params.get("is_active", True)
            wf_query = _select(Workflow).where(Workflow.user_id == job.user_id)
            if is_active is not None:
                wf_query = wf_query.where(Workflow.is_active == bool(is_active))
            wf_query = (
                wf_query.options(_selectinload(Workflow.nodes))
                .order_by(Workflow.updated_at.desc())
                .limit(20)
            )
            wf_result = await ctx.db.execute(wf_query)
            workflows = wf_result.scalars().all()
            return {
                "success": True,
                "data": {
                    "workflows": [
                        {
                            "id": str(wf.id),
                            "name": wf.name,
                            "description": wf.description or "",
                            "is_active": wf.is_active,
                            "node_count": len(wf.nodes) if wf.nodes else 0,
                        }
                        for wf in workflows
                    ],
                    "count": len(workflows),
                },
            }
        except Exception as exc:
            return {"error": f"Failed to list workflows: {exc}"}

    async def _execute_workflow(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from uuid import UUID as _UUID

        from app.models.user import User as _User
        from app.services.workflow_engine import WorkflowEngine

        job = ctx.job
        wf_id_str = str(params.get("workflow_id", "")).strip()
        if not wf_id_str:
            return {"error": "workflow_id is required"}
        try:
            user_obj = await ctx.db.get(_User, job.user_id)
            if not user_obj:
                return {"error": "Could not load user for workflow execution"}
            engine = WorkflowEngine(ctx.db, user_obj)
            execution = await engine.execute_workflow(
                workflow_id=_UUID(wf_id_str),
                trigger_type="agent_job",
                trigger_data=params.get("trigger_data")
                or {"source_job_id": str(job.id)},
                initial_context=params.get("inputs"),
            )
            return {
                "success": True,
                "data": {
                    "execution_id": str(execution.id),
                    "status": execution.status,
                    "workflow_id": wf_id_str,
                },
            }
        except Exception as exc:
            return {"error": f"Workflow execution failed: {exc}"}

    async def _get_workflow_status(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from uuid import UUID as _UUID

        from sqlalchemy import select as _select

        from app.models.workflow import WorkflowExecution

        exec_id_str = str(params.get("execution_id", "")).strip()
        if not exec_id_str:
            return {"error": "execution_id is required"}
        try:
            exec_result = await ctx.db.execute(
                _select(WorkflowExecution).where(
                    WorkflowExecution.id == _UUID(exec_id_str)
                )
            )
            execution = exec_result.scalar_one_or_none()
            if not execution:
                return {"error": f"Workflow execution {exec_id_str} not found"}
            return {
                "success": True,
                "data": {
                    "execution_id": str(execution.id),
                    "workflow_id": str(execution.workflow_id),
                    "status": execution.status,
                    "progress": execution.progress,
                    "error": execution.error,
                    "started_at": str(execution.started_at)
                    if execution.started_at
                    else None,
                    "completed_at": str(execution.completed_at)
                    if execution.completed_at
                    else None,
                },
            }
        except Exception as exc:
            return {"error": f"Failed to get workflow status: {exc}"}

    async def _enqueue_external_agent_call(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        import hashlib as _hashlib
        import json as _json
        from uuid import UUID as _UUID

        from app.models.agent_job import AgentJobStatus as _AgentJobStatus
        from app.models.user import User as _User
        from app.models.workflow import UserTool as _UserTool
        from app.services.agent_external_call_outbox_service import (
            AgentExternalCallOutboxError,
            agent_external_call_outbox_service,
        )
        from app.services.external_agent_gateway_service import (
            external_agent_gateway_service,
        )
        from app.services.tool_policy_engine import evaluate_tool_policy

        job = ctx.job
        try:
            tool_id = _UUID(str(params.get("tool_id") or "").strip())
        except (TypeError, ValueError):
            return {"error": "tool_id must be a valid external-agent connection ID"}
        capability = str(params.get("capability") or "").strip().lower()
        payload = params.get("payload")
        if not capability:
            return {"error": "capability is required"}
        if not isinstance(payload, dict):
            return {"error": "payload must be an object"}
        user = await ctx.db.get(_User, job.user_id)
        tool = await ctx.db.get(_UserTool, tool_id)
        if (
            user is None
            or tool is None
            or tool.user_id != job.user_id
            or tool.tool_type != "external_agent"
            or not bool(tool.is_enabled)
        ):
            return {"error": "Enabled external-agent connection was not found"}
        try:
            gateway_config = external_agent_gateway_service.validate_config(
                tool.config if isinstance(tool.config, dict) else {}
            )
        except Exception as exc:
            return {"error": f"External-agent connection is invalid: {exc}"}
        if capability not in set(gateway_config.get("capabilities") or []):
            return {"error": "Capability is not allowed by this connection"}
        decision = await evaluate_tool_policy(
            db=ctx.db,
            tool_name=f"user_tool:{tool.id}",
            tool_args={
                "capability": capability,
                "payload": payload,
                "agent_job_id": str(job.id),
                "delivery_mode": "transactional_outbox",
            },
            user=user,
        )
        if not decision.allowed:
            return {
                "error": decision.denied_reason
                or "External-agent call was denied by tool policy"
            }
        if decision.require_approval:
            return {
                "error": (
                    "External-agent call requires approval before it can be " "enqueued"
                ),
                "approval_required": True,
            }
        idempotency_key = str(
            params.get("idempotency_key") or ctx.idempotency_key or ""
        ).strip()
        if not idempotency_key:
            fingerprint = _json.dumps(
                {
                    "job_id": str(job.id),
                    "iteration": int(job.iteration or 0),
                    "tool_id": str(tool.id),
                    "capability": capability,
                    "payload": payload,
                },
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
            idempotency_key = _hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()
        state = ctx.state if isinstance(ctx.state, dict) else {}
        plan = state.get("execution_plan")
        plan_step_index = int(state.get("plan_step_index", 0) or 0)
        plan_step = None
        if isinstance(plan, list) and plan:
            plan_step_index = max(0, min(plan_step_index, len(plan) - 1))
            plan_step = plan[plan_step_index]
        plan_step_id = (
            str(plan_step.get("step_id") or f"step_{plan_step_index + 1}")
            if isinstance(plan_step, dict)
            else None
        )
        correlation = {
            "job_id": str(job.id),
            "iteration": int(job.iteration or 0),
            "plan_step_id": plan_step_id,
            "plan_step_index": plan_step_index if plan_step_id else None,
            "journal_idempotency_key": idempotency_key,
        }
        try:
            row, created = await agent_external_call_outbox_service.enqueue(
                db=ctx.db,
                job_id=job.id,
                user_id=job.user_id,
                tool_id=tool.id,
                capability=capability,
                payload=payload,
                idempotency_key=idempotency_key,
                max_attempts=params.get("max_attempts", 5),
                correlation=correlation,
            )
        except AgentExternalCallOutboxError as exc:
            return {"error": str(exc)}
        deferred = str(row.status) != "succeeded"
        if deferred:
            pending = state.setdefault("external_calls_pending", {})
            pending[str(row.id)] = {
                **correlation,
                "capability": capability,
                "status": str(row.status),
            }
            if isinstance(plan_step, dict):
                plan_step["status"] = "waiting_external"
                plan_step["external_outbox_id"] = str(row.id)
                plan_step["external_capability"] = capability
                plan_step["waiting_since_iteration"] = int(job.iteration or 0)
            job.status = _AgentJobStatus.PAUSED.value
            job.current_phase = "awaiting_external"
            job.phase_details = f"Waiting for external capability: {capability}"[:280]
        return {
            "success": True,
            "deferred_external": deferred,
            "correlation": correlation,
            "data": {
                "outbox_id": str(row.id),
                "status": str(row.status),
                "created": created,
                "idempotency_key": row.idempotency_key,
                "request_id": row.request_id,
                "response": row.response if not deferred else None,
            },
            "artifacts": [
                {
                    "type": "external_call_outbox",
                    "id": str(row.id),
                    "status": str(row.status),
                }
            ],
        }

    async def _get_external_call_status(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from uuid import UUID as _UUID

        from app.models.agent_external_call_outbox import AgentExternalCallOutbox

        try:
            outbox_id = _UUID(str(params.get("outbox_id") or "").strip())
        except (TypeError, ValueError):
            return {"error": "outbox_id must be a valid UUID"}
        row = await ctx.db.get(AgentExternalCallOutbox, outbox_id)
        if (
            row is None
            or row.user_id != ctx.job.user_id
            or (row.job_id is not None and row.job_id != ctx.job.id)
        ):
            return {"error": "External-call outbox row was not found"}
        return {
            "success": True,
            "data": {
                "outbox_id": str(row.id),
                "status": str(row.status),
                "attempts": int(row.attempts or 0),
                "max_attempts": int(row.max_attempts or 0),
                "next_attempt_at": (
                    row.next_attempt_at.isoformat()
                    if row.next_attempt_at is not None
                    else None
                ),
                "delivered_at": (
                    row.delivered_at.isoformat()
                    if row.delivered_at is not None
                    else None
                ),
                "correlated_at": (
                    row.correlated_at.isoformat()
                    if row.correlated_at is not None
                    else None
                ),
                "resume_enqueued_at": (
                    row.resume_enqueued_at.isoformat()
                    if row.resume_enqueued_at is not None
                    else None
                ),
                "error": str(row.error or "")[:1000] or None,
                "response": (
                    row.response
                    if row.status == "succeeded" and isinstance(row.response, dict)
                    else None
                ),
            },
        }

    return FunctionToolProvider(
        name="autonomous_workflow_tools",
        modes={"autonomous"},
        handlers={
            "list_available_workflows": _list_available_workflows,
            "execute_workflow": _execute_workflow,
            "get_workflow_status": _get_workflow_status,
            "enqueue_external_agent_call": _enqueue_external_agent_call,
            "get_external_call_status": _get_external_call_status,
        },
    )


def build_autonomous_reasoning_provider(executor: Any) -> FunctionToolProvider:
    """Structured reasoning tools for AutonomousAgentExecutor."""

    async def _reflect(params: Dict[str, Any], ctx: AgentToolExecutionContext) -> Any:
        from datetime import datetime

        job = ctx.job
        state = ctx.state if isinstance(ctx.state, dict) else {}
        reflections = state.get("reflections")
        if not isinstance(reflections, list):
            reflections = []
        entry = {
            "iteration": int(job.iteration or 0),
            "topic": str(params.get("topic", ""))[:300],
            "assessment": str(params.get("assessment", ""))[:500],
            "blind_spots": [
                str(b)[:200]
                for b in (params.get("blind_spots") or [])
                if isinstance(b, str)
            ][:10],
            "suggested_corrections": [
                str(c)[:200]
                for c in (params.get("suggested_corrections") or [])
                if isinstance(c, str)
            ][:10],
            "timestamp": datetime.utcnow().isoformat(),
        }
        reflections.append(entry)
        state["reflections"] = reflections[-50:]
        return {
            "success": True,
            "data": {"reflection_count": len(state["reflections"]), "recorded": entry},
        }

    async def _hypothesize(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from datetime import datetime

        state = ctx.state if isinstance(ctx.state, dict) else {}
        hypotheses = state.get("hypotheses")
        if not isinstance(hypotheses, list):
            hypotheses = []
        hyp_id = str(params.get("hypothesis_id") or "").strip()
        status = str(params.get("status") or "proposed").strip()
        if status not in {
            "proposed",
            "testing",
            "supported",
            "refuted",
            "inconclusive",
        }:
            status = "proposed"
        result: Dict[str, Any] = {}
        if hyp_id:
            updated = False
            for hypothesis in hypotheses:
                if isinstance(hypothesis, dict) and hypothesis.get("id") == hyp_id:
                    hypothesis["status"] = status
                    if params.get("rationale"):
                        hypothesis["rationale"] = str(params["rationale"])[:400]
                    if params.get("testable_predictions"):
                        hypothesis["testable_predictions"] = [
                            str(p)[:200] for p in params["testable_predictions"]
                        ][:10]
                    hypothesis["updated_at"] = datetime.utcnow().isoformat()
                    updated = True
                    result["data"] = {"hypothesis": hypothesis, "action": "updated"}
                    break
            if not updated:
                result["error"] = f"Hypothesis {hyp_id} not found"
                result["data"] = {
                    "available_ids": [
                        h.get("id") for h in hypotheses if isinstance(h, dict)
                    ]
                }
        else:
            hyp_id = f"h-{len(hypotheses) + 1}"
            entry = {
                "id": hyp_id,
                "hypothesis": str(params.get("hypothesis", ""))[:500],
                "rationale": str(params.get("rationale") or "")[:400],
                "testable_predictions": [
                    str(p)[:200] for p in (params.get("testable_predictions") or [])
                ][:10],
                "status": status,
                "created_at": datetime.utcnow().isoformat(),
            }
            hypotheses.append(entry)
            result["data"] = {"hypothesis": entry, "action": "created"}
        state["hypotheses"] = hypotheses[-30:]
        result["success"] = not result.get("error")
        return result

    async def _weigh_evidence(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from datetime import datetime

        state = ctx.state if isinstance(ctx.state, dict) else {}
        ledger = state.get("evidence_ledger")
        if not isinstance(ledger, list):
            ledger = []
        verdict = str(params.get("verdict") or "neutral").strip()
        if verdict not in {
            "strongly_supported",
            "weakly_supported",
            "neutral",
            "weakly_refuted",
            "strongly_refuted",
        }:
            verdict = "neutral"
        ev_for = params.get("evidence_for") or []
        ev_against = params.get("evidence_against") or []
        entry = {
            "claim": str(params.get("claim", ""))[:500],
            "hypothesis_id": str(params.get("hypothesis_id") or "").strip() or None,
            "evidence_for": [
                {
                    "statement": str(e.get("statement", ""))[:300],
                    "source_document_id": str(e.get("source_document_id") or ""),
                    "strength": max(0.0, min(1.0, float(e.get("strength", 0.5)))),
                }
                for e in ev_for
                if isinstance(e, dict)
            ][:10],
            "evidence_against": [
                {
                    "statement": str(e.get("statement", ""))[:300],
                    "source_document_id": str(e.get("source_document_id") or ""),
                    "strength": max(0.0, min(1.0, float(e.get("strength", 0.5)))),
                }
                for e in ev_against
                if isinstance(e, dict)
            ][:10],
            "verdict": verdict,
            "timestamp": datetime.utcnow().isoformat(),
        }
        for_score = (
            sum(e["strength"] for e in entry["evidence_for"])
            if entry["evidence_for"]
            else 0
        )
        against_score = (
            sum(e["strength"] for e in entry["evidence_against"])
            if entry["evidence_against"]
            else 0
        )
        entry["aggregate_score"] = round(for_score - against_score, 3)
        ledger.append(entry)
        state["evidence_ledger"] = ledger[-100:]
        hyp_id = entry.get("hypothesis_id")
        if hyp_id:
            for hypothesis in state.get("hypotheses") or []:
                if isinstance(hypothesis, dict) and hypothesis.get("id") == hyp_id:
                    if verdict in {"strongly_supported", "weakly_supported"}:
                        hypothesis["status"] = "supported"
                    elif verdict in {"strongly_refuted", "weakly_refuted"}:
                        hypothesis["status"] = "refuted"
                    break
        return {
            "success": True,
            "data": {"entry": entry, "ledger_size": len(state["evidence_ledger"])},
        }

    async def _critique_plan(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from datetime import datetime

        job = ctx.job
        state = ctx.state if isinstance(ctx.state, dict) else {}
        critiques = state.get("plan_critiques")
        if not isinstance(critiques, list):
            critiques = []
        severity = str(params.get("severity") or "moderate").strip()
        if severity not in {"minor", "moderate", "major"}:
            severity = "moderate"
        entry = {
            "iteration": int(job.iteration or 0),
            "plan_summary": str(params.get("plan_summary", ""))[:500],
            "weaknesses": [
                str(w)[:200]
                for w in (params.get("weaknesses") or [])
                if isinstance(w, str)
            ][:10],
            "missing_steps": [
                str(s)[:200]
                for s in (params.get("missing_steps") or [])
                if isinstance(s, str)
            ][:10],
            "assumptions_challenged": [
                str(a)[:200]
                for a in (params.get("assumptions_challenged") or [])
                if isinstance(a, str)
            ][:10],
            "severity": severity,
            "timestamp": datetime.utcnow().isoformat(),
        }
        critiques.append(entry)
        state["plan_critiques"] = critiques[-20:]
        if severity == "major":
            notes = state.get("critic_notes")
            if not isinstance(notes, list):
                notes = []
            notes.append(
                {
                    "trajectory_assessment": f"Plan critique (major): {entry['plan_summary'][:200]}",
                    "pivot": "; ".join(entry["weaknesses"][:3]),
                    "recommended_tools": [],
                    "source": "critique_plan_tool",
                }
            )
            state["critic_notes"] = notes[-6:]
        return {
            "success": True,
            "data": {
                "critique": entry,
                "critiques_count": len(state["plan_critiques"]),
            },
        }

    return FunctionToolProvider(
        name="autonomous_reasoning_tools",
        modes={"autonomous"},
        handlers={
            "reflect": _reflect,
            "hypothesize": _hypothesize,
            "weigh_evidence": _weigh_evidence,
            "critique_plan": _critique_plan,
        },
    )


def build_autonomous_collaboration_provider(executor: Any) -> FunctionToolProvider:
    """Collaboration tools for AutonomousAgentExecutor."""

    async def _delegate_subtask(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        import asyncio

        from app.models.agent_job import AgentJob, AgentJobStatus
        from app.tasks.agent_job_tasks import execute_agent_job_task

        job = ctx.job
        state = ctx.state if isinstance(ctx.state, dict) else {}
        chain_depth = int(getattr(job, "chain_depth", 0) or 0)
        if chain_depth >= 3:
            return {"error": "Maximum delegation depth (3) reached"}

        delegated_ids = state.get("delegated_subtask_ids")
        if not isinstance(delegated_ids, list):
            delegated_ids = []
        if len(delegated_ids) >= 5:
            return {"error": "Maximum child job budget (5) reached for this parent"}

        child_name = str(params.get("name", "Subtask"))[:200]
        child_goal = str(params.get("goal", ""))[:2000]
        child_type = str(params.get("job_type", "custom")).strip()
        if child_type not in {"research", "analysis", "synthesis", "custom"}:
            child_type = "custom"
        child_config = (
            params.get("config") if isinstance(params.get("config"), dict) else {}
        )
        share = params.get("share_findings", True)
        if not isinstance(share, bool):
            share = True
        remaining_iters = max(1, (job.max_iterations or 100) - (job.iteration or 0))
        child_max = min(int(params.get("max_iterations", 30) or 30), remaining_iters)

        if share:
            child_config["inherited_findings"] = (state.get("findings") or [])[-20:]

        try:
            child = AgentJob(
                name=child_name,
                description=f"Subtask delegated from {job.name}: {child_goal[:500]}",
                job_type=child_type,
                goal=child_goal,
                config=child_config,
                status=AgentJobStatus.PENDING.value,
                user_id=job.user_id,
                parent_job_id=job.id,
                chain_depth=chain_depth + 1,
                root_job_id=getattr(job, "root_job_id", None) or job.id,
                max_iterations=child_max,
                max_tool_calls=min(child_max * 5, job.max_tool_calls or 500),
                max_llm_calls=min(child_max * 3, job.max_llm_calls or 200),
                max_runtime_minutes=min(30, job.max_runtime_minutes or 60),
            )
            ctx.db.add(child)
            await ctx.db.flush()
            delegated_ids.append(str(child.id))
            state["delegated_subtask_ids"] = delegated_ids

            execute_agent_job_task.delay(str(child.id), str(job.user_id))

            result = {
                "success": True,
                "data": {
                    "child_job_id": str(child.id),
                    "name": child_name,
                    "status": "pending",
                    "max_iterations": child_max,
                },
            }

            if params.get("wait"):
                timeout = min(int(params.get("timeout_seconds", 60) or 60), 60)
                waited = 0
                while waited < timeout:
                    await asyncio.sleep(3)
                    waited += 3
                    await ctx.db.refresh(child)
                    if child.status in [
                        AgentJobStatus.COMPLETED.value,
                        AgentJobStatus.FAILED.value,
                        AgentJobStatus.CANCELLED.value,
                    ]:
                        result["data"]["status"] = child.status
                        result["data"]["results"] = (
                            child.results if isinstance(child.results, dict) else {}
                        )
                        state.setdefault("delegated_subtask_results", {})[
                            str(child.id)
                        ] = result["data"]["results"]
                        break
                else:
                    result["data"]["status"] = child.status
                    result["data"][
                        "note"
                    ] = "Timed out waiting; use wait_for_subtask to check later"

            try:
                await executor._save_checkpoint(job, state, ctx.db)
            except Exception:
                pass
            return result
        except Exception as exc:
            return {"error": f"Failed to create child job: {exc}"}

    async def _wait_for_subtask(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        import asyncio
        import uuid

        from app.models.agent_job import AgentJob, AgentJobStatus

        job = ctx.job
        state = ctx.state if isinstance(ctx.state, dict) else {}
        subtask_id = str(params.get("subtask_job_id") or "").strip()
        delegated_ids = state.get("delegated_subtask_ids")
        if not isinstance(delegated_ids, list):
            delegated_ids = []
        if subtask_id not in delegated_ids:
            return {"error": f"Job {subtask_id} is not a delegated subtask of this job"}

        cached = (state.get("delegated_subtask_results") or {}).get(subtask_id)
        if cached:
            return {
                "success": True,
                "data": {"status": "completed", "results": cached, "source": "cache"},
            }

        timeout = min(int(params.get("timeout_seconds", 30) or 30), 120)
        try:
            subtask_uuid = uuid.UUID(subtask_id)
            child_query = await ctx.db.execute(
                select(AgentJob).where(
                    AgentJob.id == subtask_uuid, AgentJob.parent_job_id == job.id
                )
            )
            child = child_query.scalar_one_or_none()
            if not child:
                return {"error": f"Child job {subtask_id} not found"}
            waited = 0
            while waited < timeout:
                if child.status in [
                    AgentJobStatus.COMPLETED.value,
                    AgentJobStatus.FAILED.value,
                    AgentJobStatus.CANCELLED.value,
                ]:
                    break
                await asyncio.sleep(3)
                waited += 3
                await ctx.db.refresh(child)
            child_results = child.results if isinstance(child.results, dict) else {}
            state.setdefault("delegated_subtask_results", {})[
                subtask_id
            ] = child_results
            return {
                "success": True,
                "data": {
                    "status": child.status,
                    "progress": child.progress,
                    "results": child_results,
                    "findings_count": len(child_results.get("findings", []))
                    if isinstance(child_results.get("findings"), list)
                    else 0,
                },
            }
        except Exception as exc:
            return {"error": f"Failed to check subtask: {exc}"}

    async def _share_findings(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        import uuid
        from datetime import datetime

        from sqlalchemy.orm.attributes import flag_modified

        from app.models.agent_job import AgentJob

        job = ctx.job
        state = ctx.state if isinstance(ctx.state, dict) else {}
        findings_to_share = params.get("findings") or []
        if not isinstance(findings_to_share, list) or not findings_to_share:
            return {"error": "No findings provided to share"}
        if not getattr(job, "parent_job_id", None):
            return {
                "error": "Cannot share findings: this job has no parent (no siblings)"
            }

        target_ids = params.get("target_job_ids") or []
        if not isinstance(target_ids, list):
            target_ids = []
        try:
            query = select(AgentJob).where(
                AgentJob.parent_job_id == job.parent_job_id, AgentJob.id != job.id
            )
            if target_ids:
                target_uuids = []
                for tid in target_ids:
                    try:
                        target_uuids.append(uuid.UUID(str(tid)))
                    except (ValueError, AttributeError):
                        pass
                if target_uuids:
                    query = query.where(AgentJob.id.in_(target_uuids))
            siblings_result = await ctx.db.execute(query)
            siblings = siblings_result.scalars().all()
            shared_count = 0
            for sibling in siblings:
                sib_results = (
                    sibling.results if isinstance(sibling.results, dict) else {}
                )
                shared = sib_results.get("shared_findings", [])
                if not isinstance(shared, list):
                    shared = []
                for finding in findings_to_share[:10]:
                    if isinstance(finding, dict):
                        shared.append(
                            {
                                "from_job_id": str(job.id),
                                "title": str(finding.get("title", ""))[:200],
                                "content": str(finding.get("content", ""))[:1000],
                                "category": str(finding.get("category", ""))[:100],
                                "shared_at": datetime.utcnow().isoformat(),
                            }
                        )
                sib_results["shared_findings"] = shared[-50:]
                sibling.results = sib_results
                flag_modified(sibling, "results")
                shared_count += 1
            await ctx.db.flush()
            try:
                await executor._save_checkpoint(job, state, ctx.db)
            except Exception:
                pass
            return {
                "success": True,
                "data": {
                    "siblings_updated": shared_count,
                    "findings_shared": len(findings_to_share[:10]),
                },
            }
        except Exception as exc:
            return {"error": f"Failed to share findings: {exc}"}

    async def _request_review(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from datetime import datetime

        from app.models.agent_job import AgentJob, AgentJobStatus
        from app.tasks.agent_job_tasks import execute_agent_job_task

        job = ctx.job
        state = ctx.state if isinstance(ctx.state, dict) else {}
        review_type = str(params.get("review_type") or "peer_agent").strip()
        content = str(params.get("content_to_review", ""))[:3000]
        criteria = [
            str(c)[:200]
            for c in (params.get("review_criteria") or [])
            if isinstance(c, str)
        ][:10]

        review_entry = {
            "type": review_type,
            "content": content[:500],
            "criteria": criteria,
            "timestamp": datetime.utcnow().isoformat(),
            "iteration": int(job.iteration or 0),
        }
        reviews = state.get("review_requests")
        if not isinstance(reviews, list):
            reviews = []
        reviews.append(review_entry)
        state["review_requests"] = reviews[-20:]

        if review_type == "human":
            state["approval_checkpoint_pending"] = {
                "type": "review_request",
                "content_to_review": content,
                "review_criteria": criteria,
                "requested_at": datetime.utcnow().isoformat(),
            }
            return {
                "success": True,
                "data": {
                    "action": "paused_for_human_review",
                    "checkpoint": state["approval_checkpoint_pending"],
                },
            }

        chain_depth = int(getattr(job, "chain_depth", 0) or 0)
        if chain_depth >= 3:
            return {
                "error": "Cannot spawn peer review: maximum delegation depth reached"
            }

        try:
            review_goal = f"Review the following content and provide feedback:\n\n{content[:1500]}"
            if criteria:
                review_goal += "\n\nEvaluate against these criteria:\n" + "\n".join(
                    f"- {c}" for c in criteria
                )
            child = AgentJob(
                name=f"Peer review for {job.name}"[:200],
                description="Peer review requested by sibling agent",
                job_type="analysis",
                goal=review_goal,
                config={"review_mode": True},
                status=AgentJobStatus.PENDING.value,
                user_id=job.user_id,
                parent_job_id=job.id,
                chain_depth=chain_depth + 1,
                root_job_id=getattr(job, "root_job_id", None) or job.id,
                max_iterations=10,
                max_tool_calls=30,
                max_llm_calls=15,
                max_runtime_minutes=15,
            )
            ctx.db.add(child)
            await ctx.db.flush()
            delegated_ids = state.get("delegated_subtask_ids")
            if not isinstance(delegated_ids, list):
                delegated_ids = []
            delegated_ids.append(str(child.id))
            state["delegated_subtask_ids"] = delegated_ids

            execute_agent_job_task.delay(str(child.id), str(job.user_id))

            try:
                await executor._save_checkpoint(job, state, ctx.db)
            except Exception:
                pass
            return {
                "success": True,
                "data": {
                    "action": "peer_review_spawned",
                    "review_job_id": str(child.id),
                },
            }
        except Exception as exc:
            return {"error": f"Failed to spawn peer review: {exc}"}

    async def _send_message_to_agent(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from datetime import datetime
        from uuid import UUID as _UUID

        from sqlalchemy.orm.attributes import flag_modified

        from app.models.agent_job import AgentJob

        job = ctx.job
        target_job_id_str = str(params.get("target_job_id", "")).strip()
        message_text = str(params.get("message", "")).strip()
        if not target_job_id_str:
            return {"error": "target_job_id is required"}
        if not message_text:
            return {"error": "message is required"}
        try:
            target_job = await ctx.db.get(AgentJob, _UUID(target_job_id_str))
            if not target_job:
                return {"error": f"Target job {target_job_id_str} not found"}
            if str(target_job.user_id) != str(job.user_id):
                return {"error": "Cannot send messages to jobs owned by other users"}
            target_results = (
                target_job.results if isinstance(target_job.results, dict) else {}
            )
            agent_msgs = target_results.get("agent_messages", [])
            if not isinstance(agent_msgs, list):
                agent_msgs = []
            category = str(params.get("category", ""))[:100].strip()
            agent_msgs.append(
                {
                    "from_job_id": str(job.id),
                    "from_job_name": job.name or "unknown",
                    "message": message_text[:2000],
                    "category": category,
                    "sent_at": datetime.utcnow().isoformat(),
                }
            )
            target_results["agent_messages"] = agent_msgs[-100:]
            target_job.results = target_results
            flag_modified(target_job, "results")
            await ctx.db.flush()
            return {
                "success": True,
                "data": {
                    "delivered": True,
                    "target_job_id": target_job_id_str,
                    "message_index": len(agent_msgs) - 1,
                },
            }
        except Exception as exc:
            return {"error": f"Failed to send message: {exc}"}

    async def _read_agent_messages(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        job = ctx.job
        try:
            job_results = job.results if isinstance(job.results, dict) else {}
            agent_msgs = job_results.get("agent_messages", [])
            if not isinstance(agent_msgs, list):
                agent_msgs = []
            shared = job_results.get("shared_findings", [])
            if not isinstance(shared, list):
                shared = []
            since = max(0, int(params.get("since_index", 0) or 0))
            return {
                "success": True,
                "data": {
                    "messages": agent_msgs[since:],
                    "total": len(agent_msgs),
                    "since_index": since,
                    "shared_findings_count": len(shared),
                },
            }
        except Exception as exc:
            return {"error": f"Failed to read messages: {exc}"}

    return FunctionToolProvider(
        name="autonomous_collaboration_tools",
        modes={"autonomous"},
        handlers={
            "delegate_subtask": _delegate_subtask,
            "wait_for_subtask": _wait_for_subtask,
            "share_findings": _share_findings,
            "request_review": _request_review,
            "send_message_to_agent": _send_message_to_agent,
            "read_agent_messages": _read_agent_messages,
        },
    )


def build_autonomous_workspace_read_provider(executor: Any) -> FunctionToolProvider:
    """Read-oriented workspace tools for AutonomousAgentExecutor."""

    async def _clone_and_index_repo(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        source_id = str(params.get("source_id") or "").strip()
        repo_url = str(params.get("repo_url") or "").strip()
        branch = str(params.get("branch") or "").strip() or None
        job = ctx.job
        state = ctx.state if isinstance(ctx.state, dict) else {}
        if not source_id and not repo_url:
            source_id = str((job.config or {}).get("source_id") or "").strip()
        if not source_id and not repo_url:
            return {"error": "Either source_id or repo_url is required"}
        try:
            if source_id:
                ws = await executor.workspace_manager.create_from_source(
                    source_id, ctx.db
                )
            else:
                from app.core.config import settings as app_settings
                from app.core.feature_flags import get_flag

                enabled = await get_flag("unsafe_code_execution_enabled")
                if enabled is None:
                    enabled = bool(
                        getattr(app_settings, "ENABLE_UNSAFE_CODE_EXECUTION", False)
                    )
                if not enabled:
                    return {"error": "Git clone requires unsafe_code_execution_enabled"}
                ws = await executor.workspace_manager.create_from_url(repo_url, branch)
            state["coding_workspace_id"] = ws.workspace_id
            ws.owner_job_id = str(job.id)
            ws.session_id = (
                str((job.config or {}).get("coding_workspace_session_id") or "").strip()
                or None
            )
            from app.services.agent_coding_harness_service import (
                agent_coding_harness_service,
            )

            instruction_context = (
                agent_coding_harness_service.discover_project_instructions(ws)
            )
            state["coding_harness_context"] = instruction_context
            baseline_checkpoint = None
            if bool((job.config or {}).get("coding_harness_may_mutate")):
                (
                    baseline_checkpoint,
                    checkpoint_error,
                ) = executor.workspace_manager.create_checkpoint(
                    ws,
                    label="Automatic baseline before mutation",
                    kind="baseline",
                )
                if checkpoint_error:
                    executor.workspace_manager.cleanup(ws.workspace_id)
                    state.pop("coding_workspace_id", None)
                    return {
                        "error": (
                            "Failed to create mandatory pre-mutation checkpoint: "
                            f"{checkpoint_error}"
                        )
                    }
                state["coding_pre_mutation_checkpoint_id"] = str(
                    baseline_checkpoint.get("checkpoint_id") or ""
                )
            restored_durable_checkpoint = None
            durable_checkpoint_id = str(
                state.get("coding_last_durable_checkpoint_id") or ""
            ).strip()
            if durable_checkpoint_id:
                try:
                    from app.services.agent_coding_durable_checkpoint_service import (
                        agent_coding_durable_checkpoint_service,
                    )

                    restored_durable_checkpoint = (
                        await agent_coding_durable_checkpoint_service.restore(
                            executor,
                            job,
                            state,
                            checkpoint_id=durable_checkpoint_id,
                        )
                    )
                except Exception as exc:
                    executor.workspace_manager.cleanup(ws.workspace_id)
                    state.pop("coding_workspace_id", None)
                    return {
                        "error": (
                            "Failed to restore durable coding session checkpoint: "
                            f"{exc}"
                        )
                    }
            return {
                "success": True,
                "data": {
                    "workspace_id": ws.workspace_id,
                    "files_count": len(ws.original_hashes),
                    "source": "kb_source" if source_id else "git_clone",
                    "instruction_files": [
                        str(item.get("path") or "")
                        for item in instruction_context.get("files", [])
                        if isinstance(item, dict)
                        and str(item.get("path") or "").strip()
                    ],
                    "baseline_checkpoint": baseline_checkpoint,
                    "restored_durable_checkpoint": restored_durable_checkpoint,
                },
            }
        except Exception as exc:
            return {"error": f"Failed to create workspace: {exc}"}

    async def _browse_repo_files(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        state = ctx.state if isinstance(ctx.state, dict) else {}
        ws = executor.workspace_manager.get_or_default(
            params.get("workspace_id"), state
        )
        if not ws:
            return {
                "error": "No active coding workspace. Use clone_and_index_repo first."
            }
        entries = executor.workspace_manager.browse_files(
            ws,
            path=str(params.get("path", ".") or "."),
            glob_pattern=params.get("glob_pattern"),
            max_results=min(int(params.get("max_results", 200) or 200), 500),
        )
        return {"success": True, "data": {"files": entries, "count": len(entries)}}

    async def _read_file(params: Dict[str, Any], ctx: AgentToolExecutionContext) -> Any:
        state = ctx.state if isinstance(ctx.state, dict) else {}
        ws = executor.workspace_manager.get_or_default(
            params.get("workspace_id"), state
        )
        if not ws:
            return {"error": "No active coding workspace"}
        path = str(params.get("path", "")).strip()
        if not path:
            return {"error": "path is required"}
        content, err = executor.workspace_manager.read_file(
            ws,
            path,
            start_line=params.get("start_line"),
            end_line=params.get("end_line"),
            max_chars=min(int(params.get("max_chars", 20000) or 20000), 50000),
        )
        if err:
            return {"error": err}
        return {
            "success": True,
            "data": {"path": path, "content": content, "length": len(content or "")},
        }

    async def _search_code(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        state = ctx.state if isinstance(ctx.state, dict) else {}
        ws = executor.workspace_manager.get_or_default(
            params.get("workspace_id"), state
        )
        if not ws:
            return {"error": "No active coding workspace"}
        pattern = str(params.get("pattern", "")).strip()
        if not pattern:
            return {"error": "pattern is required"}
        matches = executor.workspace_manager.search_code(
            ws,
            pattern,
            path=str(params.get("path", ".") or "."),
            file_glob=params.get("file_glob"),
            max_results=min(int(params.get("max_results", 50) or 50), 200),
            context_lines=min(int(params.get("context_lines", 2) or 2), 10),
        )
        return {"success": True, "data": {"matches": matches, "count": len(matches)}}

    async def _get_workspace_status(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        state = ctx.state if isinstance(ctx.state, dict) else {}
        ws = executor.workspace_manager.get_or_default(
            params.get("workspace_id"), state
        )
        if not ws:
            return {"error": "No active coding workspace"}
        status = executor.workspace_manager.get_status(ws)
        return {"success": True, "data": status}

    async def _list_workspace_checkpoints(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        state = ctx.state if isinstance(ctx.state, dict) else {}
        ws = executor.workspace_manager.get_or_default(
            params.get("workspace_id"), state
        )
        if not ws:
            return {"error": "No active coding workspace"}
        checkpoints = executor.workspace_manager.list_checkpoints(ws)
        return {
            "success": True,
            "data": {
                "workspace_id": ws.workspace_id,
                "checkpoints": checkpoints,
                "count": len(checkpoints),
            },
        }

    async def _list_durable_workspace_checkpoints(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        state = ctx.state if isinstance(ctx.state, dict) else {}
        from app.services.agent_coding_durable_checkpoint_service import (
            agent_coding_durable_checkpoint_service,
        )

        checkpoints = agent_coding_durable_checkpoint_service.list_checkpoints(
            ctx.job,
            state,
        )
        rows = [
            {
                key: item.get(key)
                for key in (
                    "checkpoint_id",
                    "session_id",
                    "workspace_state_digest",
                    "changes_summary",
                    "persistence_complete",
                    "label",
                    "reason",
                    "persisted_at",
                )
            }
            for item in checkpoints
        ]
        return {"success": True, "data": {"checkpoints": rows, "count": len(rows)}}

    async def _get_workspace_artifact_url(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        ws_job_id = str(params.get("job_id", "")).strip()
        ws_file_path = str(params.get("file_path", "")).strip()
        if not ws_job_id or not ws_file_path:
            return {"error": "job_id and file_path are required"}
        from app.services.storage_service import storage_service

        full_path = f"workspaces/{ws_job_id}/{ws_file_path}"
        try:
            await storage_service.initialize()
            url = await storage_service.get_presigned_download_url(full_path)
            return {"success": True, "data": {"url": url, "object_path": full_path}}
        except Exception as exc:
            return {"error": f"Failed to get download URL: {exc}"}

    return FunctionToolProvider(
        name="autonomous_workspace_read_tools",
        modes={"autonomous"},
        handlers={
            "clone_and_index_repo": _clone_and_index_repo,
            "browse_repo_files": _browse_repo_files,
            "read_file": _read_file,
            "search_code": _search_code,
            "get_workspace_status": _get_workspace_status,
            "list_workspace_checkpoints": _list_workspace_checkpoints,
            "list_durable_workspace_checkpoints": (_list_durable_workspace_checkpoints),
            "get_workspace_artifact_url": _get_workspace_artifact_url,
        },
    )


def build_autonomous_workspace_mutation_provider(executor: Any) -> FunctionToolProvider:
    """Workspace mutation and code-execution tools for AutonomousAgentExecutor."""

    async def _resolve_user(ctx: AgentToolExecutionContext) -> Any:
        from app.models.user import User

        job = ctx.job
        user_result = await ctx.db.execute(select(User).where(User.id == job.user_id))
        user = user_result.scalar_one_or_none()
        if not user:
            raise ValueError("User not found for code execution")
        return user

    async def _execute_python(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from datetime import datetime

        from app.services.custom_tool_service import CustomToolService

        state = ctx.state if isinstance(ctx.state, dict) else {}
        code = str(params.get("code", ""))
        timeout = min(int(params.get("timeout_seconds", 10) or 10), 30)
        if not code.strip():
            return {"error": "No code provided"}
        try:
            cts = CustomToolService()
            user = await _resolve_user(ctx)
            exec_result = await cts._execute_python(
                config={"code": code, "timeout_seconds": timeout},
                inputs={},
                user=user,
            )
            history = state.get("code_execution_history")
            if not isinstance(history, list):
                history = []
            history.append(
                {
                    "tool": "execute_python",
                    "success": True,
                    "code_preview": code[:200],
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )
            state["code_execution_history"] = history[-50:]
            return {"success": True, "data": exec_result}
        except Exception as exc:
            return {"error": f"Python execution failed: {exc}"}

    async def _execute_data_pipeline(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        import json
        from datetime import datetime

        from app.core.config import settings
        from app.services.custom_tool_service import CustomToolService

        state = ctx.state if isinstance(ctx.state, dict) else {}
        code = str(params.get("code", ""))
        timeout = min(int(params.get("timeout_seconds", 60) or 60), 300)
        input_data = (
            params.get("input_data")
            if isinstance(params.get("input_data"), dict)
            else {}
        )
        if not code.strip():
            return {"error": "No code provided"}
        try:
            cts = CustomToolService()
            user = await _resolve_user(ctx)
            if getattr(settings, "CUSTOM_TOOL_DOCKER_ENABLED", False):
                wrapper = (
                    "import json, sys\n"
                    "input_data = json.loads(sys.stdin.read()) if not sys.stdin.isatty() else {}\n"
                    f"{code}\n"
                    "if 'result' in dir():\n"
                    "    print(json.dumps(result, default=str))\n"
                )
                exec_result = await cts._execute_docker(
                    config={
                        "image": "python:3.11-slim",
                        "command": ["python", "-c", wrapper],
                        "timeout_seconds": timeout,
                        "memory_limit": "512m",
                        "network_enabled": False,
                    },
                    inputs={"stdin": json.dumps(input_data, default=str)},
                    user=user,
                )
            else:
                exec_result = await cts._execute_python(
                    config={
                        "code": f"input_data = {repr(input_data)}\n{code}",
                        "timeout_seconds": timeout,
                    },
                    inputs={},
                    user=user,
                )
            history = state.get("code_execution_history")
            if not isinstance(history, list):
                history = []
            history.append(
                {
                    "tool": "execute_data_pipeline",
                    "success": True,
                    "code_preview": code[:200],
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )
            state["code_execution_history"] = history[-50:]
            return {"success": True, "data": exec_result}
        except Exception as exc:
            return {"error": f"Data pipeline execution failed: {exc}"}

    async def _write_and_run_script(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        import json
        from datetime import datetime

        from app.core.config import settings
        from app.services.custom_tool_service import CustomToolService

        state = ctx.state if isinstance(ctx.state, dict) else {}
        script_name = str(params.get("script_name", "script.py"))[:100]
        script_content = str(params.get("script_content", ""))
        timeout = min(int(params.get("timeout_seconds", 120) or 120), 300)
        input_data = (
            params.get("input_data")
            if isinstance(params.get("input_data"), dict)
            else {}
        )
        requirements = params.get("requirements") or []
        arguments = params.get("arguments") or []
        if not isinstance(requirements, list):
            requirements = []
        if not isinstance(arguments, list):
            arguments = []

        safe_packages = {
            "pandas",
            "numpy",
            "scipy",
            "scikit-learn",
            "matplotlib",
            "seaborn",
            "networkx",
            "requests",
            "beautifulsoup4",
            "lxml",
            "pyyaml",
            "tabulate",
            "openpyxl",
            "xlsxwriter",
        }
        requirements = [
            r
            for r in requirements
            if isinstance(r, str) and r.strip().lower() in safe_packages
        ]

        if not script_content.strip():
            return {"error": "No script content provided"}
        if not getattr(settings, "CUSTOM_TOOL_DOCKER_ENABLED", False):
            return {
                "error": "Docker execution is not enabled; write_and_run_script requires Docker"
            }
        try:
            cts = CustomToolService()
            user = await _resolve_user(ctx)
            pip_cmd = (
                f"pip install -q {' '.join(requirements)} && " if requirements else ""
            )
            input_cmd = ""
            if input_data:
                input_cmd = f"echo '{json.dumps(input_data, default=str)}' > /workspace/input.json && "
            args_str = " ".join(str(arg) for arg in arguments[:10])
            exec_result = await cts._execute_docker(
                config={
                    "image": "python:3.11-slim",
                    "command": [
                        "bash",
                        "-c",
                        f"{pip_cmd}{input_cmd}python /workspace/{script_name} {args_str}",
                    ],
                    "timeout_seconds": timeout,
                    "memory_limit": "512m",
                    "network_enabled": False,
                },
                inputs={"stdin": script_content},
                user=user,
            )
            history = state.get("code_execution_history")
            if not isinstance(history, list):
                history = []
            history.append(
                {
                    "tool": "write_and_run_script",
                    "success": True,
                    "script_name": script_name,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )
            state["code_execution_history"] = history[-50:]
            return {"success": True, "data": exec_result}
        except Exception as exc:
            return {"error": f"Script execution failed: {exc}"}

    async def _write_file(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        state = ctx.state if isinstance(ctx.state, dict) else {}
        ws = executor.workspace_manager.get_or_default(
            params.get("workspace_id"), state
        )
        if not ws:
            return {"error": "No active coding workspace"}
        path = str(params.get("path", "")).strip()
        content = str(params.get("content", ""))
        if not path:
            return {"error": "path is required"}
        err = executor.workspace_manager.write_file(
            ws,
            path,
            content,
            create_dirs=params.get("create_dirs", True),
        )
        if err:
            return {"error": err}
        modified = state.get("coding_modified_files")
        if not isinstance(modified, list):
            modified = []
        if path not in modified:
            modified.append(path)
        state["coding_modified_files"] = modified[-200:]
        return {
            "success": True,
            "data": {"path": path, "bytes_written": len(content.encode("utf-8"))},
        }

    async def _create_workspace_checkpoint(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        state = ctx.state if isinstance(ctx.state, dict) else {}
        ws = executor.workspace_manager.get_or_default(
            params.get("workspace_id"), state
        )
        if not ws:
            return {"error": "No active coding workspace"}
        checkpoint, error = executor.workspace_manager.create_checkpoint(
            ws,
            label=str(params.get("label") or "").strip(),
            kind="manual",
        )
        if error:
            return {"error": error}
        state["coding_last_checkpoint_id"] = str(
            (checkpoint or {}).get("checkpoint_id") or ""
        )
        return {"success": True, "data": checkpoint}

    async def _restore_workspace_checkpoint(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        state = ctx.state if isinstance(ctx.state, dict) else {}
        ws = executor.workspace_manager.get_or_default(
            params.get("workspace_id"), state
        )
        if not ws:
            return {"error": "No active coding workspace"}
        checkpoint_id = str(params.get("checkpoint_id") or "").strip()
        if not checkpoint_id:
            return {"error": "checkpoint_id is required"}
        result, error = executor.workspace_manager.restore_checkpoint(
            ws,
            checkpoint_id,
            preserve_current=bool(params.get("preserve_current", True)),
        )
        if error:
            return {"error": error}
        status = (result or {}).get("status") or {}
        state["coding_modified_files"] = list(
            dict.fromkeys(
                [
                    *list(status.get("modified") or []),
                    *list(status.get("added") or []),
                    *list(status.get("deleted") or []),
                ]
            )
        )[:200]
        state["coding_last_restored_checkpoint_id"] = checkpoint_id
        return {"success": True, "data": result}

    async def _hydrate_candidate_snapshot(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        state = ctx.state if isinstance(ctx.state, dict) else {}
        ws = executor.workspace_manager.get_or_default(
            params.get("workspace_id"), state
        )
        if not ws:
            return {"error": "No active coding workspace"}
        config = ctx.job.config if isinstance(ctx.job.config, dict) else {}
        handoff = (
            config.get("swarm_handoff")
            if isinstance(config.get("swarm_handoff"), dict)
            else {}
        )
        configured_manifest = (
            config.get("candidate_snapshot")
            if isinstance(config.get("candidate_snapshot"), dict)
            else handoff.get("candidate_snapshot")
            if isinstance(handoff.get("candidate_snapshot"), dict)
            else None
        )
        configured_manifests = (
            config.get("candidate_snapshots")
            if isinstance(config.get("candidate_snapshots"), list)
            else []
        )
        requested_snapshot_id = str(params.get("snapshot_id") or "").strip()
        manifest = configured_manifest
        if requested_snapshot_id:
            if (
                isinstance(configured_manifest, dict)
                and str(configured_manifest.get("snapshot_id") or "")
                == requested_snapshot_id
            ):
                manifest = configured_manifest
            else:
                manifest = next(
                    (
                        item
                        for item in configured_manifests
                        if isinstance(item, dict)
                        and str(item.get("snapshot_id") or "") == requested_snapshot_id
                    ),
                    None,
                )
        elif not isinstance(manifest, dict) and len(configured_manifests) == 1:
            manifest = (
                configured_manifests[0]
                if isinstance(configured_manifests[0], dict)
                else None
            )
        if not isinstance(manifest, dict):
            return {
                "error": (
                    "No matching system-provided candidate snapshot is available; "
                    "supply snapshot_id when multiple candidates exist"
                )
            }
        result, error = await executor.workspace_manager.hydrate_candidate_snapshot(
            ws,
            manifest,
        )
        if error:
            return {"error": error}
        state["coding_hydrated_candidate_snapshot_id"] = str(
            manifest.get("snapshot_id") or ""
        )
        state["coding_modified_files"] = list(
            dict.fromkeys(
                [
                    *list((result or {}).get("hydrated_files") or []),
                    *list((result or {}).get("deleted_files") or []),
                ]
            )
        )[:200]
        return {"success": True, "data": result}

    async def _persist_durable_workspace_checkpoint(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        state = ctx.state if isinstance(ctx.state, dict) else {}
        from app.services.agent_coding_durable_checkpoint_service import (
            agent_coding_durable_checkpoint_service,
        )

        try:
            manifest = await agent_coding_durable_checkpoint_service.persist(
                executor,
                ctx.job,
                state,
                label=str(params.get("label") or "").strip(),
                reason="agent_requested",
                db=ctx.db,
            )
        except Exception as exc:
            return {"error": f"Failed to persist durable checkpoint: {exc}"}
        if not isinstance(manifest, dict):
            return {"error": "Durable checkpoint was not created"}
        return {
            "success": True,
            "data": {
                "checkpoint_id": str(manifest.get("checkpoint_id") or ""),
                "session_id": str(manifest.get("session_id") or ""),
                "workspace_state_digest": str(
                    manifest.get("workspace_state_digest") or ""
                ),
                "persistence_complete": bool(
                    manifest.get("persistence_complete", False)
                ),
                "changes_summary": manifest.get("changes_summary") or {},
            },
        }

    async def _restore_durable_workspace_checkpoint(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        state = ctx.state if isinstance(ctx.state, dict) else {}
        checkpoint_id = str(params.get("checkpoint_id") or "").strip()
        if not checkpoint_id:
            return {"error": "checkpoint_id is required"}
        from app.services.agent_coding_durable_checkpoint_service import (
            agent_coding_durable_checkpoint_service,
        )

        try:
            result = await agent_coding_durable_checkpoint_service.restore(
                executor,
                ctx.job,
                state,
                checkpoint_id=checkpoint_id,
            )
        except Exception as exc:
            return {"error": f"Failed to restore durable checkpoint: {exc}"}
        return {"success": True, "data": result}

    async def _apply_patch(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        state = ctx.state if isinstance(ctx.state, dict) else {}
        ws = executor.workspace_manager.get_or_default(
            params.get("workspace_id"), state
        )
        if not ws:
            return {"error": "No active coding workspace"}
        diff_text = str(params.get("diff", "")).strip()
        dry_run = bool(params.get("dry_run", False))
        if not diff_text:
            return {"error": "diff is required"}
        try:
            from app.services.code_patch_apply_service import CodePatchApplyService

            svc = CodePatchApplyService()
            file_diffs = svc.parse(diff_text)
            applied_files = []
            errors = []
            for file_diff in file_diffs:
                file_path = file_diff.path
                target = executor.workspace_manager.safe_resolve(ws, file_path)
                if not target or not target.is_file():
                    errors.append(f"File not found: {file_path}")
                    continue
                original = target.read_text(encoding="utf-8", errors="replace")
                new_text, _debug = svc.apply_to_text(original, file_diff)
                if not dry_run:
                    target.write_text(new_text, encoding="utf-8")
                applied_files.append(file_path)
            if not dry_run:
                modified = state.get("coding_modified_files")
                if not isinstance(modified, list):
                    modified = []
                for file_path in applied_files:
                    if file_path not in modified:
                        modified.append(file_path)
                state["coding_modified_files"] = modified[-200:]
            return {
                "success": True,
                "data": {
                    "applied_files": applied_files,
                    "errors": errors,
                    "dry_run": dry_run,
                    "files_count": len(applied_files),
                },
            }
        except Exception as exc:
            return {"error": f"Patch failed: {exc}"}

    async def _run_command(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        import asyncio
        import os
        from datetime import datetime

        from app.core.config import settings as app_settings
        from app.core.feature_flags import get_flag

        state = ctx.state if isinstance(ctx.state, dict) else {}
        ws = executor.workspace_manager.get_or_default(
            params.get("workspace_id"), state
        )
        if not ws:
            return {"error": "No active coding workspace"}
        command = str(params.get("command", "")).strip()
        if not command:
            return {"error": "command is required"}
        from app.services.agent_job_creation_service import agent_job_creation_service

        unsafe_commands = agent_job_creation_service.find_unsafe_commands([command])
        if unsafe_commands:
            return {
                "success": False,
                "error": "Command rejected by coding harness safety policy",
                "data": {"blocked_commands": unsafe_commands},
            }
        enabled = await get_flag("unsafe_code_execution_enabled")
        if enabled is None:
            enabled = bool(getattr(app_settings, "ENABLE_UNSAFE_CODE_EXECUTION", False))
        if not enabled:
            return {
                "error": "Shell execution requires unsafe_code_execution_enabled feature flag"
            }
        timeout = min(int(params.get("timeout_seconds", 30) or 30), 120)
        extra_env = params.get("env") if isinstance(params.get("env"), dict) else {}
        env = {**os.environ, **extra_env, "HOME": str(ws.base_path)}
        max_output = int(
            getattr(app_settings, "UNSAFE_CODE_EXEC_MAX_STDOUT_CHARS", 20000) or 20000
        )
        try:
            proc = await asyncio.wait_for(
                asyncio.create_subprocess_exec(
                    "/bin/sh",
                    "-lc",
                    command,
                    cwd=str(ws.base_path),
                    env=env,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                ),
                timeout=5,
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
            stdout_str = stdout_bytes.decode("utf-8", errors="replace")[:max_output]
            stderr_str = stderr_bytes.decode("utf-8", errors="replace")[:max_output]
            history = state.get("coding_command_history")
            if not isinstance(history, list):
                history = []
            history.append(
                {
                    "command": command[:200],
                    "exit_code": proc.returncode,
                    "stdout_preview": stdout_str[:200],
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )
            state["coding_command_history"] = history[-50:]
            command_succeeded = proc.returncode == 0
            result = {
                "success": command_succeeded,
                "data": {
                    "exit_code": proc.returncode,
                    "stdout": stdout_str,
                    "stderr": stderr_str,
                    "command": command[:200],
                },
            }
            if not command_succeeded:
                result["error"] = f"Command exited with status {proc.returncode}"
            elif bool((ctx.job.config or {}).get("coding_harness_may_mutate")):
                workspace_status = executor.workspace_manager.get_status(ws)
                if int(workspace_status.get("changes_count") or 0) > 0:
                    try:
                        from app.services.agent_coding_durable_checkpoint_service import (
                            agent_coding_durable_checkpoint_service,
                        )

                        durable_checkpoint = (
                            await agent_coding_durable_checkpoint_service.persist(
                                executor,
                                ctx.job,
                                state,
                                label=f"Verified by {command[:80]}",
                                reason="successful_verification",
                                db=ctx.db,
                            )
                        )
                        if isinstance(durable_checkpoint, dict):
                            result["data"]["durable_checkpoint_id"] = str(
                                durable_checkpoint.get("checkpoint_id") or ""
                            )
                    except Exception as checkpoint_exc:
                        result["data"]["durable_checkpoint_error"] = str(
                            checkpoint_exc
                        )[:500]
            return result
        except asyncio.TimeoutError:
            return {"error": f"Command timed out after {timeout}s"}
        except Exception as exc:
            return {"error": f"Command failed: {exc}"}

    async def _compile_c_snippet(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from app.services import agent_compiler_sandbox

        return await agent_compiler_sandbox.compile_c_snippet(
            code=str(params.get("code") or ""),
            flags=str(params.get("flags") or "-O2"),
            emit=str(params.get("emit") or "asm"),
            label=str(params.get("label") or ""),
        )

    async def _profile_c_workload(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from app.services import agent_profile_sandbox

        return await agent_profile_sandbox.profile_c_workload(
            code=str(params.get("code") or ""),
            flags=str(params.get("flags") or agent_profile_sandbox.DEFAULT_FLAGS),
            run_args=str(params.get("run_args") or ""),
            label=str(params.get("label") or ""),
            top_functions=min(int(params.get("top_functions", 8) or 8), 25),
            top_blocks=min(int(params.get("top_blocks", 5) or 5), 15),
        )

    def _recent_counter_sample(state: Any) -> Any:
        """The most recent successful counter sampling, series and all.

        The whole result rather than the series alone, because whether the
        trace changes regime part way through belongs to the trace and has to
        travel with it -- a window is only sound relative to the break it was
        or was not taken across.

        Read from the run rather than retyped by the model, for the reason
        _recent_hot_blocks exists: a trace is tens of counters by tens of
        intervals, and a truncated copy answers a question about different
        data than the one that was sampled.
        """
        actions = (
            (state or {}).get("actions_taken") if isinstance(state, dict) else None
        )
        if not isinstance(actions, list):
            return None
        for entry in reversed(actions):
            if not isinstance(entry, dict):
                continue
            action = (
                entry.get("action") if isinstance(entry.get("action"), dict) else {}
            )
            result = (
                entry.get("result") if isinstance(entry.get("result"), dict) else {}
            )
            if str(action.get("tool") or "") != "sample_hardware_counters":
                continue
            if not bool(result.get("success")):
                continue
            data = result.get("data") if isinstance(result.get("data"), dict) else {}
            series = data.get("series")
            if isinstance(series, dict) and series:
                return data
        return None

    def _counter_window(data: Any, params: Dict[str, Any]) -> Any:
        """The slice of a trace a caller asked for, and what it straddles."""
        from app.services import agent_trace_regime

        return agent_trace_regime.window(data, params.get("from_interval"))

    async def _measure_predictability(
        params: Dict[str, Any], context: AgentToolExecutionContext
    ) -> Dict[str, Any]:
        from app.services import agent_predictability

        sample = _recent_counter_sample(getattr(context, "state", None))
        series, window = _counter_window(sample, params) if sample else (None, {})
        if not series:
            return {
                "success": False,
                "error": (
                    "No counter trace in this run. Call sample_hardware_counters "
                    "first, with M5_SAMPLE() in the workload -- predictability is "
                    "a property of counters over time and cannot be read from a "
                    "run total."
                ),
            }

        result = agent_predictability.ceiling(
            series,
            str(params.get("target") or ""),
            bins=int(params.get("bins") or agent_predictability.DEFAULT_BINS),
        )
        if not result.get("measured"):
            return {"success": False, "error": result.get("refusal"), "data": result}

        return {
            "success": True,
            "data": result,
            "findings": [
                {
                    "type": "predictability_ceiling",
                    "subject": result["target"],
                    "title": (
                        f"{result['target']}: {result['best_counter_beyond_persistence_bits']} "
                        f"bits available beyond persistence over {result['intervals']} intervals"
                    ),
                    "target": result["target"],
                    "intervals": result["intervals"],
                    "target_entropy_bits": result["target_entropy_bits"],
                    "persistence_information_bits": result[
                        "persistence_information_bits"
                    ],
                    "best_counter_beyond_persistence_bits": result[
                        "best_counter_beyond_persistence_bits"
                    ],
                    **window,
                    "verdict": result["verdict"],
                }
            ],
        }

    async def _select_counter_taps(
        params: Dict[str, Any], context: AgentToolExecutionContext
    ) -> Dict[str, Any]:
        from app.services import agent_predictability

        sample = _recent_counter_sample(getattr(context, "state", None))
        series, window = _counter_window(sample, params) if sample else (None, {})
        if not series:
            return {
                "success": False,
                "error": (
                    "No counter trace in this run. Call sample_hardware_counters "
                    "first -- which counters to tap together is a question about "
                    "counters over time and cannot be read from a run total."
                ),
            }

        result = agent_predictability.select_taps(
            series,
            str(params.get("target") or ""),
            bins=int(params.get("bins") or agent_predictability.DEFAULT_BINS),
        )
        if not result.get("measured"):
            return {"success": False, "error": result.get("refusal"), "data": result}

        kept = result["taps"]
        return {
            "success": True,
            "data": result,
            "findings": [
                {
                    "type": "counter_tap_selection",
                    "subject": result["target"],
                    "title": (
                        f"{result['target']}: {result['recommended_taps']} tap(s) "
                        f"survive their own null of "
                        f"{result['max_taps_supported']} this trace can support"
                    ),
                    "target": result["target"],
                    "intervals": result["intervals"],
                    "recommended_taps": result["recommended_taps"],
                    "taps": kept,
                    "max_taps_supported": result["max_taps_supported"],
                    "total_beyond_persistence_bits": result["total_beyond_persistence"],
                    "total_at_full_depth_bits": result["total_at_full_depth"],
                    "selection": result["selection"],
                    **window,
                    "verdict": result["verdict"],
                }
            ],
        }

    async def _evaluate_predictor_design(
        params: Dict[str, Any], context: AgentToolExecutionContext
    ) -> Dict[str, Any]:
        from app.services import agent_predictor_design

        sample = _recent_counter_sample(getattr(context, "state", None))
        series, window = _counter_window(sample, params) if sample else (None, {})
        if not series:
            return {
                "success": False,
                "error": (
                    "No counter trace in this run. Call sample_hardware_counters "
                    "first -- a predictor is scored on intervals over time, and "
                    "there is nothing to hold out of a run total."
                ),
            }

        result = agent_predictor_design.evaluate(
            series,
            str(params.get("target") or ""),
            str(params.get("tap") or ""),
            bins=int(params.get("bins") or agent_predictor_design.DEFAULT_BINS),
            split=float(params.get("split") or agent_predictor_design.DEFAULT_SPLIT),
        )
        if not result.get("measured"):
            return {"success": False, "error": result.get("refusal"), "data": result}

        return {
            "success": True,
            "data": result,
            "findings": [
                {
                    "type": "predictor_design_result",
                    "subject": f"{result['target']} from {result['tap']}",
                    "title": (
                        f"{result['best_design']} gains "
                        f"{result['best_gain_over_persistence']:+.4f} over "
                        f"persistence on {result['scored_intervals']} held-out "
                        "intervals"
                    ),
                    "target": result["target"],
                    "tap": result["tap"],
                    "scored_intervals": result["scored_intervals"],
                    "persistence_accuracy": result["persistence_accuracy"],
                    "ceiling_accuracy": result["ceiling_accuracy"],
                    "best_design": result["best_design"],
                    "best_gain_over_persistence": result["best_gain_over_persistence"],
                    "share_of_headroom": result["best_share_of_headroom"],
                    "survives_null": result["survives_null"],
                    "ceiling_exceeded": result["ceiling_exceeded"],
                    "designs": result["designs"],
                    **window,
                    "verdict": result["verdict"],
                }
            ],
        }

    async def _sample_hardware_counters(
        params: Dict[str, Any], context: AgentToolExecutionContext
    ) -> Dict[str, Any]:
        from app.services import agent_gem5_sandbox

        return await agent_gem5_sandbox.sample_counters(
            code=str(params.get("code") or ""),
            flags=str(params.get("flags") or agent_gem5_sandbox.DEFAULT_FLAGS),
            cpu_type=str(params.get("cpu_type") or agent_gem5_sandbox.DEFAULT_CPU),
            label=str(params.get("label") or ""),
            max_counters=int(params.get("max_counters") or 60),
            language=str(params.get("language") or "c"),
            extra_files=(
                params.get("extra_files")
                if isinstance(params.get("extra_files"), dict)
                else None
            ),
            include_dirs=(
                params.get("include_dirs")
                if isinstance(params.get("include_dirs"), list)
                else None
            ),
            co_runner=str(params.get("co_runner") or ""),
            intends_alternating_phases=bool(params.get("intends_alternating_phases")),
        )

    async def _simulate_c_workload(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from app.services import agent_gem5_sandbox

        overrides = params.get("param_overrides")
        if isinstance(overrides, str):
            # A single assignment is the common case and arrives unwrapped.
            overrides = [overrides]

        return await agent_gem5_sandbox.simulate_c_workload(
            code=str(params.get("code") or ""),
            flags=str(params.get("flags") or agent_gem5_sandbox.DEFAULT_FLAGS),
            cpu_type=str(params.get("cpu_type") or agent_gem5_sandbox.DEFAULT_CPU),
            param_overrides=[str(x) for x in overrides]
            if isinstance(overrides, list)
            else None,
            run_args=str(params.get("run_args") or ""),
            label=str(params.get("label") or ""),
        )

    def _recent_hot_blocks(state: Any) -> Any:
        """The hot blocks from the most recent successful profile in this run.

        Tools that hand a large structure to the next tool should not make the
        model retype it: the copy is expensive, and a truncated one mines a
        different program than the one that was profiled.
        """
        actions = (
            (state or {}).get("actions_taken") if isinstance(state, dict) else None
        )
        if not isinstance(actions, list):
            return None
        for entry in reversed(actions):
            if not isinstance(entry, dict):
                continue
            action = (
                entry.get("action") if isinstance(entry.get("action"), dict) else {}
            )
            result = (
                entry.get("result") if isinstance(entry.get("result"), dict) else {}
            )
            if str(action.get("tool") or "") != "profile_c_workload":
                continue
            if not bool(result.get("success")):
                continue
            data = result.get("data") if isinstance(result.get("data"), dict) else {}
            blocks = data.get("hot_blocks")
            if isinstance(blocks, list) and blocks:
                return blocks
        return None

    async def _cost_fusion_candidate(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from app.services import agent_compiler_sandbox

        def _as_int(value: Any, default: int) -> int:
            try:
                return int(value)
            except (TypeError, ValueError):
                return default

        return await agent_compiler_sandbox.cost_fusion_candidate(
            pattern=str(params.get("pattern") or ""),
            cpu=str(params.get("cpu") or ""),
            copies=_as_int(params.get("copies"), 20),
            mode=str(params.get("mode") or "dependent"),
            label=str(params.get("label") or ""),
        )

    async def _find_fusion_candidates(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from app.services import isa_candidate_mining

        blocks = params.get("blocks")
        if isinstance(blocks, str):
            # A model asked for a large structure sends it as text. Parsing it
            # costs nothing and refusing it costs an iteration, which is what
            # happened: a live run serialised the profiler's blocks and was
            # told the field should be an array.
            try:
                blocks = json.loads(blocks)
            except (TypeError, ValueError):
                blocks = None
        if isinstance(blocks, dict):
            blocks = blocks.get("hot_blocks") if "hot_blocks" in blocks else [blocks]

        if not isinstance(blocks, list) or not blocks:
            # Copying kilobytes of disassembly from one tool call into the next
            # is work the run should not have to do by hand, and a truncated
            # copy would mine the wrong thing silently. Fall back to the
            # profile this run already produced.
            blocks = _recent_hot_blocks(ctx.state)

        if not isinstance(blocks, list) or not blocks:
            return {
                "error": (
                    "No hot blocks to mine. Run profile_c_workload first and "
                    "this tool will pick up its blocks automatically, or pass "
                    "`blocks` as objects with an `instructions` list of "
                    "assembly lines and an `executions` count. Mining source "
                    "text instead of a profiled run measures how often a "
                    "pattern is written, not how often it runs."
                )
            }

        def _as_int(value: Any, default: int, low: int, high: int) -> int:
            try:
                return max(low, min(int(value), high))
            except (TypeError, ValueError):
                return default

        ranked = isa_candidate_mining.mine_blocks(
            [b for b in blocks if isinstance(b, dict)],
            max_nodes=_as_int(params.get("max_instructions"), 3, 2, 6),
            max_inputs=_as_int(params.get("max_inputs"), 2, 1, 8),
            max_outputs=_as_int(params.get("max_outputs"), 1, 1, 4),
            min_dynamic=_as_int(params.get("min_executions"), 0, 0, 10**15),
        )
        if not ranked:
            return {
                "success": True,
                "data": {
                    "candidates": [],
                    "note": (
                        "No group of instructions in these blocks both passes "
                        "values between its members and fits the operand "
                        "budget. Widen max_instructions or max_inputs, or "
                        "check the blocks carry disassembly."
                    ),
                },
            }

        top = ranked[:25]
        best = top[0]
        return {
            "success": True,
            "data": {
                "candidates": top,
                "blocks_examined": len(blocks),
                "note": (
                    "Ranked by how often the containing block executed. This "
                    "says a shape is frequent, not that fusing it pays: cost "
                    "the sequence and its replacement with "
                    "analyze_snippet_cycles before proposing it, because "
                    "instruction count is not cycles."
                ),
            },
            "findings": [
                {
                    "type": "fusion_candidate",
                    "title": (
                        f"{' + '.join(best['mnemonics'])}: "
                        f"{best['dynamic_occurrences']:,} dynamic occurrences, "
                        f"{best['inputs']} in / {best['outputs']} out"
                    ),
                    "pattern": best["pattern"],
                    "dynamic_occurrences": best["dynamic_occurrences"],
                    "static_occurrences": best["static_occurrences"],
                    "example": best["example"],
                    "category": "insight",
                }
            ],
        }

    async def _describe_model_parameters(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from app.services import agent_gem5_sandbox

        op_classes = params.get("op_classes")
        if isinstance(op_classes, str):
            op_classes = [x.strip() for x in op_classes.split(",") if x.strip()]

        return await agent_gem5_sandbox.describe_model_parameters(
            cpu_type=str(params.get("cpu_type") or agent_gem5_sandbox.DEFAULT_CPU),
            op_classes=[str(x) for x in op_classes]
            if isinstance(op_classes, list)
            else None,
        )

    async def _verify_run_bundle(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from app.services import agent_evidence_bundle as bundle

        job_id = getattr(getattr(ctx, "job", None), "id", None)
        if not job_id:
            return {"error": "No job in context; there is no bundle to verify"}

        integrity = bundle.verify_integrity(str(job_id))
        if not integrity["entries"]:
            return {
                "error": (
                    "This run has recorded no evidence yet, so there is nothing "
                    "to verify. Run the measurements first."
                )
            }

        replay: Dict[str, Any] = {}
        if bool(params.get("replay", False)):

            async def execute(tool: str, tool_params: Dict[str, Any]) -> Any:
                _, result = await executor.tool_registry.try_execute(
                    tool, tool_params, ctx
                )
                return result

            replay = await bundle.replay_bundle(str(job_id), execute)

        summary = bundle.summarize(str(job_id))
        verdict = replay.get("verdict") if replay else "not replayed"
        return {
            "success": True,
            "data": {
                "bundle": summary,
                "integrity": integrity,
                "replay": replay,
                "note": (
                    "Integrity shows the artifacts are the ones this run "
                    "produced. Only a replay shows they can be produced again, "
                    "and it judges nothing that reports wall clock."
                ),
            },
            "findings": [
                {
                    "type": "bundle_verified",
                    "title": (
                        f"Evidence bundle: {summary['entries']} calls recorded, "
                        f"integrity {'intact' if integrity['intact'] else 'BROKEN'}, "
                        f"replay {verdict}"
                    ),
                    "intact": integrity["intact"],
                    "replay_verdict": verdict,
                    "entries": summary["entries"],
                }
            ],
        }

    async def _record_prediction(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from app.services import agent_calibration_service as calibration

        job = getattr(ctx, "job", None)
        # ctx.user_id is not populated in autonomous runs; the owner is the
        # job's user, which is how the other write tools resolve it.
        owner_id = getattr(job, "user_id", None) or ctx.user_id
        tags = params.get("methodology_tags")

        # What evidence actually exists in this run right now. A prediction
        # that cites a measurement it never obtained is the worst failure this
        # store can suffer: a run predicted from "llvm-mca reported 11.8 cycles
        # per iteration" while its only mca call had failed, and the real
        # answer -- 59.05 -- arrived three iterations later. The error column
        # caught the consequence and could not see the cause.
        state = ctx.state if isinstance(ctx.state, dict) else {}
        findings = (
            state.get("findings") if isinstance(state.get("findings"), list) else []
        )
        available = sorted(
            {
                str(f.get("type")).strip()
                for f in findings
                if isinstance(f, dict) and str(f.get("type") or "").strip()
            }
        )
        required = params.get("derived_from")
        required = (
            [str(r).strip() for r in required if str(r).strip()]
            if isinstance(required, list)
            else []
        )
        # Required, not optional. Left optional, the guard never fired: a run
        # that had just fabricated an llvm-mca result simply did not mention
        # what it derived from, and nothing asked. A prediction with no
        # measurement behind it is legitimate, but it has to say so.
        if not required:
            return {
                "error": (
                    "derived_from is required: list the finding types this "
                    "number comes from, e.g. ['cycle_model_measurement']. "
                    f"Findings available in this run: "
                    f"{', '.join(available) or 'none'}. If the prediction is a "
                    "judgement with no measurement behind it, pass ['none'] "
                    "and say so in the methodology."
                )
            }
        declared_guess = required == ["none"]
        if not declared_guess:
            from app.services import agent_evidence_citation

            resolved, missing = agent_evidence_citation.resolve_all(required, available)
            if missing:
                return {
                    "error": agent_evidence_citation.explain_unresolved(
                        missing, available
                    )
                }
            # Store the resolved type names rather than the prose the caller
            # wrote, so the record says which evidence it rests on.
            required = resolved
        try:
            # A savepoint, not the caller's transaction: a rejected insert
            # would otherwise poison the session the whole run shares.
            async with ctx.db.begin_nested():
                prediction = await calibration.record_prediction(
                    ctx.db,
                    subject=str(params.get("subject") or ""),
                    metric=str(params.get("metric") or ""),
                    predicted_value=float(params.get("predicted_value") or 0.0),
                    methodology=str(params.get("methodology") or ""),
                    prediction_basis=str(params.get("prediction_basis") or ""),
                    # Record what evidence was on hand when the claim was
                    # made, so a later reader can tell a derived prediction
                    # from a guess without taking the methodology text at its
                    # word.
                    methodology_tags=(
                        ([str(t) for t in tags] if isinstance(tags, list) else [])
                        + [f"evidence:{name}" for name in available]
                        + (["declared:no-measurement"] if declared_guess else [])
                    )
                    or None,
                    job_id=getattr(job, "id", None),
                    user_id=owner_id,
                )
            await ctx.db.commit()
        except calibration.CalibrationError as exc:
            return {"error": str(exc)}
        except Exception as exc:
            return {"error": f"Could not record the prediction: {str(exc)[:200]}"}

        return {
            "success": True,
            "data": {
                "prediction_id": str(prediction.id),
                "subject": prediction.subject,
                "metric": prediction.metric,
                "predicted_value": prediction.predicted_value,
                "note": (
                    "Recorded before the outcome is known. Settle it with "
                    "record_measurement once the referee has run."
                ),
            },
            "findings": [
                {
                    "type": "prediction_recorded",
                    "title": (
                        f"Predicted {prediction.metric}={prediction.predicted_value} "
                        f"for {prediction.subject}"
                    ),
                    "prediction_id": str(prediction.id),
                }
            ],
        }

    async def _record_measurement(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from uuid import UUID as _PredUUID

        from app.services import agent_calibration_service as calibration

        raw_id = str(params.get("prediction_id") or "").strip()
        try:
            prediction_id = _PredUUID(raw_id)
        except (ValueError, AttributeError, TypeError):
            return {
                "error": (
                    f"prediction_id should be a UUID, got {raw_id!r}; it is the id "
                    "record_prediction returned."
                )
            }
        try:
            async with ctx.db.begin_nested():
                settled = await calibration.record_measurement(
                    ctx.db,
                    prediction_id=prediction_id,
                    measured_value=float(params.get("measured_value") or 0.0),
                    measurement_source=str(params.get("measurement_source") or ""),
                    notes=str(params.get("notes") or ""),
                )
            await ctx.db.commit()
        except calibration.CalibrationError as exc:
            return {"error": str(exc)}
        except Exception as exc:
            return {"error": f"Could not record the measurement: {str(exc)[:200]}"}

        return {
            "success": True,
            "data": {
                "prediction_id": str(settled.id),
                "predicted_value": settled.predicted_value,
                "measured_value": settled.measured_value,
                "error_absolute": settled.error_absolute,
                "relative_error": settled.error_relative,
                "measurement_source": settled.measurement_source,
            },
            "findings": [
                {
                    "type": "prediction_settled",
                    "title": (
                        f"{settled.subject}: predicted {settled.predicted_value}, "
                        f"measured {settled.measured_value} "
                        f"({settled.measurement_source})"
                    ),
                    "relative_error": settled.error_relative,
                }
            ],
        }

    async def _calibration_report(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from app.services import agent_calibration_service as calibration

        try:
            report = await calibration.calibration_report(
                ctx.db,
                metric=str(params.get("metric") or "") or None,
                subject=str(params.get("subject") or "") or None,
                limit=min(int(params.get("limit", 50) or 50), 200),
            )
        except Exception as exc:
            return {
                "error": f"Could not read the calibration history: {str(exc)[:200]}"
            }

        return {"success": True, "data": report}

    async def _axis_check(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from app.services import agent_axis_sandbox

        return await agent_axis_sandbox.check_description(
            source=str(params.get("source") or "")
        )

    async def _axis_emit(params: Dict[str, Any], ctx: AgentToolExecutionContext) -> Any:
        from app.services import agent_axis_sandbox

        return await agent_axis_sandbox.emit_artifact(
            source=str(params.get("source") or ""),
            target=str(params.get("target") or ""),
        )

    async def _axis_prove(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from app.services import agent_axis_sandbox

        return await agent_axis_sandbox.prove_equivalence(
            source=str(params.get("source") or ""),
            obligation=str(params.get("obligation") or ""),
        )

    async def _analyze_snippet_cycles(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from app.services import agent_compiler_sandbox

        return await agent_compiler_sandbox.analyze_snippet_cycles(
            code=str(params.get("code") or ""),
            asm=str(params.get("asm") or ""),
            cpu=str(params.get("cpu") or ""),
            flags=str(params.get("flags") or "-O3"),
            target=str(
                params.get("target") or agent_compiler_sandbox.DEFAULT_ANALYSIS_TARGET
            ),
            iterations=params.get("iterations", 100),
            label=str(params.get("label") or ""),
        )

    async def _benchmark_c_snippet(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from app.services import agent_compiler_sandbox

        return await agent_compiler_sandbox.benchmark_c_snippet(
            code=str(params.get("code") or ""),
            flags=str(params.get("flags") or "-O2"),
            repeat=int(params.get("repeat") or 3),
            label=str(params.get("label") or ""),
        )

    async def _create_custom_tool(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        """Create a reusable tool owned by this user.

        Mirrors the validation on POST /user-tools: docker_container stays
        behind CUSTOM_TOOL_DOCKER_ENABLED, and workflow_runner is reserved for
        workflow synthesis, which fills in the workflow id it points at.
        """
        from app.models.workflow import UserTool
        from app.services.custom_tool_types import reject_custom_tool_type

        name = str(params.get("name") or "").strip()
        if not name:
            return {"error": "name is required"}
        tool_type = str(params.get("tool_type") or "").strip().lower()

        # An agent may not create a workflow_runner: that type points at a
        # workflow id which workflow synthesis fills in.
        rejection = reject_custom_tool_type(tool_type, include_workflow_runner=False)
        if rejection:
            return {"error": rejection}

        config = params.get("config")
        if not isinstance(config, dict) or not config:
            return {"error": "config is required and must be an object"}
        schema = params.get("parameters_schema")
        if not isinstance(schema, dict):
            schema = {"type": "object", "properties": {}}

        # ctx.user_id is not populated in autonomous runs; the owner is the
        # job's user, which is how the other write tools resolve it.
        owner_id = getattr(getattr(ctx, "job", None), "user_id", None) or ctx.user_id
        if owner_id is None:
            return {"error": "Cannot determine the owning user for the new tool"}

        existing = (
            await ctx.db.execute(
                select(UserTool).where(
                    UserTool.user_id == owner_id, UserTool.name == name
                )
            )
        ).scalar_one_or_none()
        if existing is not None:
            return {
                "error": (
                    f"A tool named {name!r} already exists. Choose another "
                    "name, or call it with run_custom_tool."
                )
            }

        tool = UserTool(
            user_id=owner_id,
            name=name,
            description=str(params.get("description") or "").strip() or None,
            tool_type=tool_type,
            parameters_schema=schema,
            config=config,
            is_enabled=True,
        )
        # A savepoint, not the caller's transaction: a rejected insert here
        # would otherwise poison the session the whole run shares, and one bad
        # tool definition would end the job rather than the action.
        try:
            async with ctx.db.begin_nested():
                ctx.db.add(tool)
            await ctx.db.commit()
        except Exception as exc:
            return {"error": f"Could not create the tool: {str(exc)[:200]}"}

        return {
            "success": True,
            "data": {
                "tool_id": str(tool.id),
                "name": tool.name,
                "tool_type": tool.tool_type,
            },
            "findings": [
                {
                    "type": "tool_created",
                    "title": f"Created custom tool {tool.name!r} ({tool.tool_type})",
                    "tool_id": str(tool.id),
                }
            ],
        }

    def _tool_owner(ctx: AgentToolExecutionContext) -> Any:
        """Autonomous runs carry the user on the job, not on the context."""
        return getattr(getattr(ctx, "job", None), "user_id", None) or ctx.user_id

    async def _run_custom_tool_autonomous(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        # AgentService is the chat-mode surface and is not reachable from the
        # autonomous executor; go to the same service it uses.
        from sqlalchemy import func

        from app.models.user import User
        from app.models.workflow import UserTool
        from app.services.custom_tool_service import CustomToolService

        owner_id = _tool_owner(ctx)
        if owner_id is None:
            return {"error": "Cannot determine the owning user for this tool"}
        tool_name = str(params.get("tool_name") or "").strip()
        if not tool_name:
            return {"error": "tool_name is required"}

        tool = (
            await ctx.db.execute(
                select(UserTool).where(
                    UserTool.user_id == owner_id,
                    func.lower(UserTool.name) == tool_name.lower(),
                )
            )
        ).scalar_one_or_none()
        if tool is None:
            return {"error": f"No custom tool named {tool_name!r} for this user"}
        if not tool.is_enabled:
            return {"error": f"Custom tool {tool_name!r} is disabled"}

        user = (
            await ctx.db.execute(select(User).where(User.id == owner_id))
        ).scalar_one_or_none()

        inputs = params.get("inputs")
        if not isinstance(inputs, dict):
            inputs = {}
        try:
            output = await CustomToolService().execute_tool(
                tool=tool, inputs=inputs, user=user, db=ctx.db
            )
        except Exception as exc:
            return {"error": f"Custom tool {tool_name!r} failed: {str(exc)[:300]}"}

        return {
            "success": True,
            "data": {"tool_name": tool.name, "output": output},
            "findings": [
                {
                    "type": "custom_tool_result",
                    "title": f"{tool.name}: {str(output)[:180]}",
                }
            ],
        }

    async def _list_custom_tools_autonomous(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from app.models.workflow import UserTool

        owner_id = _tool_owner(ctx)
        if owner_id is None:
            return {"error": "Cannot determine the owning user for this tool"}
        tools = (
            (await ctx.db.execute(select(UserTool).where(UserTool.user_id == owner_id)))
            .scalars()
            .all()
        )
        return {
            "success": True,
            "data": {
                "count": len(tools),
                "tools": [
                    {
                        "name": t.name,
                        "tool_type": t.tool_type,
                        "description": t.description,
                        "enabled": bool(t.is_enabled),
                        "parameters_schema": t.parameters_schema or {},
                    }
                    for t in tools
                ],
            },
        }

    return FunctionToolProvider(
        name="autonomous_workspace_mutation_tools",
        modes={"autonomous"},
        handlers={
            "execute_python": _execute_python,
            "create_custom_tool": _create_custom_tool,
            "run_custom_tool": _run_custom_tool_autonomous,
            "list_custom_tools": _list_custom_tools_autonomous,
            "compile_c_snippet": _compile_c_snippet,
            "analyze_snippet_cycles": _analyze_snippet_cycles,
            "profile_c_workload": _profile_c_workload,
            "simulate_c_workload": _simulate_c_workload,
            "describe_model_parameters": _describe_model_parameters,
            "find_fusion_candidates": _find_fusion_candidates,
            "cost_fusion_candidate": _cost_fusion_candidate,
            "verify_run_bundle": _verify_run_bundle,
            "record_prediction": _record_prediction,
            "record_measurement": _record_measurement,
            "calibration_report": _calibration_report,
            "axis_check": _axis_check,
            "axis_emit": _axis_emit,
            "axis_prove": _axis_prove,
            "benchmark_c_snippet": _benchmark_c_snippet,
            "sample_hardware_counters": _sample_hardware_counters,
            "measure_predictability": _measure_predictability,
            "select_counter_taps": _select_counter_taps,
            "evaluate_predictor_design": _evaluate_predictor_design,
            "execute_data_pipeline": _execute_data_pipeline,
            "write_and_run_script": _write_and_run_script,
            "write_file": _write_file,
            "apply_patch": _apply_patch,
            "run_command": _run_command,
            "create_workspace_checkpoint": _create_workspace_checkpoint,
            "restore_workspace_checkpoint": _restore_workspace_checkpoint,
            "hydrate_candidate_snapshot": _hydrate_candidate_snapshot,
            "persist_durable_workspace_checkpoint": (
                _persist_durable_workspace_checkpoint
            ),
            "restore_durable_workspace_checkpoint": (
                _restore_durable_workspace_checkpoint
            ),
        },
    )


def build_autonomous_symbol_retrieval_provider(executor: Any) -> FunctionToolProvider:
    """Symbol-aware retrieval tools for AutonomousAgentExecutor."""

    async def _retrieve_repo_symbols(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        import asyncio as _asyncio

        state = ctx.state if isinstance(ctx.state, dict) else {}
        query_str = str(params.get("query", "")).strip()
        if not query_str:
            return {"error": "query is required"}
        ws = executor.workspace_manager.get_or_default(
            params.get("workspace_id"), state
        )
        if not ws:
            return {
                "error": "No active coding workspace. Use clone_and_index_repo first."
            }
        lang_filter = params.get("language_filter")
        max_results = min(int(params.get("max_results", 20) or 20), 50)
        query_keywords = [
            t.strip()
            for t in query_str.replace("-", " ").replace("_", " ").split()
            if t.strip()
        ]
        try:
            retrieve_result = await _asyncio.to_thread(
                executor.symbol_index_service.retrieve,
                repo_root=ws.base_path,
                query_keywords=query_keywords,
                include_paths=[],
                max_symbols=max_results,
                max_snippets=min(max_results, 10),
            )
            if lang_filter:
                ext_map = {
                    "python": {".py"},
                    "typescript": {".ts", ".tsx"},
                    "javascript": {".js", ".jsx"},
                }
                allowed_exts = ext_map.get(lang_filter, set())
                if allowed_exts:
                    retrieve_result["symbol_matches"] = [
                        s
                        for s in retrieve_result.get("symbol_matches", [])
                        if any(
                            str(s.get("path", "")).endswith(ext) for ext in allowed_exts
                        )
                    ]
            return {"success": True, "data": retrieve_result}
        except Exception as exc:
            return {"error": f"Symbol retrieval failed: {exc}"}

    async def _get_symbol_context(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        import asyncio as _asyncio

        state = ctx.state if isinstance(ctx.state, dict) else {}
        symbol_name = str(params.get("symbol_name", "")).strip()
        file_path_param = str(params.get("file_path", "")).strip()
        if not symbol_name or not file_path_param:
            return {"error": "symbol_name and file_path are required"}
        ws = executor.workspace_manager.get_or_default(
            params.get("workspace_id"), state
        )
        if not ws:
            return {"error": "No active coding workspace"}
        try:
            retrieve_result = await _asyncio.to_thread(
                executor.symbol_index_service.retrieve,
                repo_root=ws.base_path,
                query_keywords=[symbol_name],
                include_paths=[file_path_param],
                max_symbols=20,
                max_snippets=10,
            )
            matches = [
                s
                for s in retrieve_result.get("symbol_matches", [])
                if s.get("path") == file_path_param
            ]
            exact = [s for s in matches if s.get("symbol") == symbol_name]
            target = exact[0] if exact else (matches[0] if matches else None)
            if not target:
                return {
                    "error": f"Symbol '{symbol_name}' not found in {file_path_param}"
                }
            code_content, _ = executor.workspace_manager.read_file(
                ws,
                file_path_param,
                start_line=max(1, target.get("start_line", 1) - 5),
                end_line=(target.get("end_line") or target.get("start_line", 1)) + 5,
                max_chars=10000,
            )
            related = [s for s in matches if s.get("symbol") != symbol_name][:5]
            return {
                "success": True,
                "data": {
                    "symbol": target,
                    "code_context": code_content,
                    "related_symbols": related,
                    "file_path": file_path_param,
                },
            }
        except Exception as exc:
            return {"error": f"Symbol context retrieval failed: {exc}"}

    async def _find_tests_for_symbol(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        import asyncio as _asyncio

        state = ctx.state if isinstance(ctx.state, dict) else {}
        symbol_name = str(params.get("symbol_name", "")).strip()
        if not symbol_name:
            return {"error": "symbol_name is required"}
        ws = executor.workspace_manager.get_or_default(
            params.get("workspace_id"), state
        )
        if not ws:
            return {"error": "No active coding workspace"}
        try:
            retrieve_result = await _asyncio.to_thread(
                executor.symbol_index_service.retrieve,
                repo_root=ws.base_path,
                query_keywords=[symbol_name, "test"],
                include_paths=[],
                max_symbols=30,
                max_snippets=10,
            )
            test_matches = list(retrieve_result.get("related_tests", []))
            for sym in retrieve_result.get("symbol_matches", []):
                path_lower = str(sym.get("path", "")).lower()
                if executor.symbol_index_service._looks_like_test(path_lower):
                    entry = {
                        "path": sym.get("path"),
                        "symbol": sym.get("symbol"),
                        "score": sym.get("score", 0),
                    }
                    if entry not in test_matches:
                        test_matches.append(entry)
            return {
                "success": True,
                "data": {
                    "tests": test_matches[:20],
                    "count": len(test_matches[:20]),
                    "symbol_searched": symbol_name,
                },
            }
        except Exception as exc:
            return {"error": f"Test search failed: {exc}"}

    return FunctionToolProvider(
        name="autonomous_symbol_retrieval_tools",
        modes={"autonomous"},
        handlers={
            "retrieve_repo_symbols": _retrieve_repo_symbols,
            "get_symbol_context": _get_symbol_context,
            "find_tests_for_symbol": _find_tests_for_symbol,
        },
    )


def build_autonomous_document_authoring_provider(executor: Any) -> FunctionToolProvider:
    """Document authoring tools for AutonomousAgentExecutor."""

    async def _plan_document(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        state = ctx.state if isinstance(ctx.state, dict) else {}
        title = str(params.get("title", "")).strip()
        sections = params.get("sections") or []
        if not title:
            return {"error": "title is required"}
        if not isinstance(sections, list) or not sections:
            return {"error": "At least one section is required"}

        plan_sections = []
        for section in sections[:30]:
            if not isinstance(section, dict):
                continue
            plan_sections.append(
                {
                    "id": str(section.get("id") or f"s-{len(plan_sections)+1}"),
                    "title": str(section.get("title", ""))[:200],
                    "description": str(section.get("description", ""))[:500],
                    "content": None,
                    "revision_count": 0,
                    "citations": [],
                    "figures": [],
                }
            )
        doc_ws = {
            "plan": {
                "title": title[:300],
                "abstract": str(params.get("abstract", ""))[:2000],
                "doc_type": str(params.get("doc_type", "research_report")),
                "style": str(params.get("style", "professional")),
                "sections": plan_sections,
            },
            "citations_registry": {},
            "assembled_markdown": None,
            "export_artifacts": [],
        }
        state["document_workspace"] = doc_ws
        return {
            "success": True,
            "data": {
                "title": title[:300],
                "sections_count": len(plan_sections),
                "section_ids": [section["id"] for section in plan_sections],
            },
        }

    async def _write_section(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        state = ctx.state if isinstance(ctx.state, dict) else {}
        doc_ws = state.get("document_workspace")
        if not doc_ws or not isinstance(doc_ws, dict) or not doc_ws.get("plan"):
            return {"error": "No document plan. Use plan_document first."}
        section_id = str(params.get("section_id", "")).strip()
        content = str(params.get("content", ""))
        if not section_id or not content:
            return {"error": "section_id and content are required"}

        section = None
        for section_row in doc_ws["plan"]["sections"]:
            if section_row["id"] == section_id:
                section = section_row
                break
        if not section:
            return {"error": f"Section '{section_id}' not found in document plan"}

        section["content"] = content
        citations = params.get("citations") or []
        if isinstance(citations, list):
            for citation in citations[:20]:
                if isinstance(citation, dict) and citation.get("ref_id"):
                    section["citations"].append(citation)
                    doc_ws["citations_registry"][citation["ref_id"]] = {
                        "document_id": str(citation.get("document_id", "")),
                        "title": str(citation.get("title", ""))[:200],
                        "excerpt": str(citation.get("excerpt", ""))[:500],
                    }
        return {
            "success": True,
            "data": {
                "section_id": section_id,
                "content_length": len(content),
                "citations_count": len(section["citations"]),
            },
        }

    async def _revise_section(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        state = ctx.state if isinstance(ctx.state, dict) else {}
        doc_ws = state.get("document_workspace")
        if not doc_ws or not isinstance(doc_ws, dict) or not doc_ws.get("plan"):
            return {"error": "No document plan"}
        section_id = str(params.get("section_id", "")).strip()
        new_content = str(params.get("new_content", ""))
        if not section_id or not new_content:
            return {"error": "section_id and new_content are required"}

        section = None
        for section_row in doc_ws["plan"]["sections"]:
            if section_row["id"] == section_id:
                section = section_row
                break
        if not section:
            return {"error": f"Section '{section_id}' not found"}

        section["content"] = new_content
        section["revision_count"] = section.get("revision_count", 0) + 1
        for citation in params.get("additional_citations") or []:
            if isinstance(citation, dict) and citation.get("ref_id"):
                section["citations"].append(citation)
                doc_ws["citations_registry"][citation["ref_id"]] = {
                    "document_id": str(citation.get("document_id", "")),
                    "title": str(citation.get("title", ""))[:200],
                    "excerpt": str(citation.get("excerpt", ""))[:500],
                }
        return {
            "success": True,
            "data": {
                "section_id": section_id,
                "revision_count": section["revision_count"],
                "content_length": len(new_content),
            },
        }

    async def _assemble_document(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        state = ctx.state if isinstance(ctx.state, dict) else {}
        doc_ws = state.get("document_workspace")
        if not doc_ws or not isinstance(doc_ws, dict) or not doc_ws.get("plan"):
            return {"error": "No document plan"}

        plan = doc_ws["plan"]
        include_toc = params.get("include_toc", True)
        include_refs = params.get("include_references", True)
        include_abstract = params.get("include_abstract", True)
        custom_order = params.get("section_order")

        sections = plan["sections"]
        if isinstance(custom_order, list) and custom_order:
            order_map = {section_id: i for i, section_id in enumerate(custom_order)}
            sections = sorted(
                sections, key=lambda section: order_map.get(section["id"], 999)
            )

        parts = [f"# {plan['title']}\n"]
        if include_abstract and plan.get("abstract"):
            parts.append(f"## Abstract\n\n{plan['abstract']}\n")

        if include_toc:
            toc_lines = ["## Table of Contents\n"]
            for idx, section in enumerate(sections, 1):
                toc_lines.append(f"{idx}. [{section['title']}](#{section['id']})")
            parts.append("\n".join(toc_lines) + "\n")

        written = 0
        skipped = 0
        for section in sections:
            if section.get("content"):
                parts.append(f"## {section['title']}\n\n{section['content']}\n")
                written += 1
            else:
                parts.append(f"## {section['title']}\n\n*[Section not yet written]*\n")
                skipped += 1

        if include_refs and doc_ws.get("citations_registry"):
            ref_lines = ["## References\n"]
            for ref_id, ref in sorted(doc_ws["citations_registry"].items()):
                ref_lines.append(f"- **{ref_id}**: {ref.get('title', 'Untitled')}")
            parts.append("\n".join(ref_lines) + "\n")

        assembled = "\n---\n\n".join(parts)
        doc_ws["assembled_markdown"] = assembled
        return {
            "success": True,
            "data": {
                "total_sections": len(sections),
                "sections_written": written,
                "sections_skipped": skipped,
                "total_length": len(assembled),
                "citations_count": len(doc_ws.get("citations_registry", {})),
            },
        }

    async def _export_document(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        import hashlib
        import re as _re

        from loguru import logger

        from app.schemas.presentation import PresentationOutline, SlideContent

        job = ctx.job
        state = ctx.state if isinstance(ctx.state, dict) else {}
        doc_ws = state.get("document_workspace")
        if not doc_ws or not doc_ws.get("assembled_markdown"):
            return {"error": "No assembled document. Use assemble_document first."}

        fmt = str(params.get("format", "")).strip().lower()
        if fmt not in {"docx", "pdf", "pptx", "latex"}:
            return {
                "error": f"Unsupported format: {fmt}. Use docx, pdf, pptx, or latex."
            }

        try:
            title = doc_ws["plan"]["title"]
            markdown = doc_ws["assembled_markdown"]
            if len(markdown) > 500_000:
                return {
                    "error": f"Document too large ({len(markdown)} chars). Max 500,000 chars."
                }

            file_bytes = None
            mime_type = ""
            if fmt == "docx":
                from app.services.docx_builder import (
                    DOCXBuilder,
                    markdown_to_content_items,
                )

                content_items = markdown_to_content_items(markdown)
                builder = DOCXBuilder()
                file_bytes = builder.build(title=title, content_items=content_items)
                mime_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            elif fmt == "pdf":
                from app.services.docx_builder import (
                    markdown_to_content_items as md_to_items,
                )
                from app.services.pdf_builder import PDFBuilder

                content_items = md_to_items(markdown)
                builder = PDFBuilder()
                file_bytes = builder.build(title=title, content_items=content_items)
                mime_type = "application/pdf"
            elif fmt == "pptx":
                from app.services.pptx_builder import PPTXBuilder

                slides = []
                sections = _re.split(r"^##\s+", markdown, flags=_re.MULTILINE)
                slide_num = 1
                for section in sections:
                    section = section.strip()
                    if not section:
                        continue
                    lines = section.split("\n", 1)
                    slide_title = lines[0].strip().lstrip("#").strip()
                    body = lines[1].strip() if len(lines) > 1 else ""
                    bullets = []
                    for body_line in body.split("\n"):
                        body_line = body_line.strip()
                        if body_line.startswith(("- ", "* ", "+ ")):
                            bullets.append(body_line[2:].strip())
                        elif body_line and not body_line.startswith("#"):
                            bullets.append(body_line)
                    slides.append(
                        SlideContent(
                            slide_number=slide_num,
                            slide_type="content",
                            title=slide_title,
                            content=bullets[:10],
                        )
                    )
                    slide_num += 1
                if not slides:
                    slides.append(
                        SlideContent(
                            slide_number=1,
                            slide_type="title",
                            title=title,
                            content=["Generated from document"],
                        )
                    )
                outline = PresentationOutline(title=title, slides=slides)
                builder = PPTXBuilder()
                file_bytes = builder.build(outline=outline)
                mime_type = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
            elif fmt == "latex":
                from app.services.latex_compiler_service import LatexCompilerService

                compile_result = LatexCompilerService.compile_to_pdf(
                    tex_source=markdown,
                    timeout_seconds=60,
                    max_source_chars=500000,
                )
                if compile_result.success:
                    file_bytes = compile_result.pdf_bytes
                    mime_type = "application/pdf"
                else:
                    return {
                        "error": f"LaTeX compilation failed: {compile_result.log[:500]}"
                    }

            artifact = {
                "type": "exported_document",
                "format": fmt,
                "title": title,
                "size_bytes": len(file_bytes),
                "mime_type": mime_type,
            }
            doc_ws.setdefault("export_artifacts", []).append(artifact)

            if params.get("persist_to_kb"):
                try:
                    from app.models.document import Document

                    doc = Document(
                        title=f"{title} ({fmt.upper()})",
                        content=markdown[:100000],
                        content_hash=hashlib.sha256(markdown.encode()).hexdigest(),
                        file_type=mime_type,
                        file_size=len(file_bytes),
                        extra_metadata={
                            "origin": "document_author",
                            "job_id": str(job.id),
                            "format": fmt,
                        },
                    )
                    ctx.db.add(doc)
                    await ctx.db.flush()
                    artifact["document_id"] = str(doc.id)
                except Exception as exc:
                    logger.warning(f"Failed to persist exported doc to KB: {exc}")

            return {"success": True, "data": artifact}
        except Exception as exc:
            logger.error(f"export_document ({fmt}) failed: {exc}")
            return {"error": f"Export failed: {exc}"}

    async def _insert_figure(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        state = ctx.state if isinstance(ctx.state, dict) else {}
        doc_ws = state.get("document_workspace")
        if not doc_ws or not isinstance(doc_ws, dict) or not doc_ws.get("plan"):
            return {"error": "No document plan"}

        section_id = str(params.get("section_id", "")).strip()
        figure_type = str(params.get("figure_type", "")).strip()
        caption = str(params.get("caption", ""))[:300]
        if not section_id or not figure_type:
            return {"error": "section_id and figure_type are required"}

        section = None
        for section_row in doc_ws["plan"]["sections"]:
            if section_row["id"] == section_id:
                section = section_row
                break
        if not section:
            return {"error": f"Section '{section_id}' not found"}

        figure_entry = {
            "type": figure_type,
            "caption": caption,
            "data": params.get("data")
            if isinstance(params.get("data"), dict)
            else None,
            "diagram_spec": str(params.get("diagram_spec", ""))[:5000] or None,
            "position": str(params.get("position", "inline")),
        }
        section.setdefault("figures", []).append(figure_entry)
        fig_md = f"\n\n*[Figure: {caption}]*\n"
        if section.get("content"):
            section["content"] += fig_md
        return {
            "success": True,
            "data": {
                "section_id": section_id,
                "figure_type": figure_type,
                "figures_count": len(section["figures"]),
            },
        }

    return FunctionToolProvider(
        name="autonomous_document_authoring_tools",
        modes={"autonomous"},
        handlers={
            "plan_document": _plan_document,
            "write_section": _write_section,
            "revise_section": _revise_section,
            "assemble_document": _assemble_document,
            "export_document": _export_document,
            "insert_figure": _insert_figure,
        },
    )


def build_autonomous_observability_provider(executor: Any) -> FunctionToolProvider:
    """Observability, analytics, and conditional tools for AutonomousAgentExecutor."""

    async def _get_job_history(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from app.models.agent_job import AgentJob as AgentJobModel

        job = ctx.job
        try:
            stmt = (
                select(AgentJobModel)
                .where(
                    AgentJobModel.user_id == job.user_id,
                    AgentJobModel.id != job.id,
                )
                .order_by(AgentJobModel.created_at.desc())
            )
            jt_filter = str(params.get("job_type", "")).strip()
            if jt_filter:
                stmt = stmt.where(AgentJobModel.job_type == jt_filter)
            status_filter = str(params.get("status", "")).strip()
            if status_filter:
                stmt = stmt.where(AgentJobModel.status == status_filter)
            limit = min(int(params.get("limit", 10) or 10), 50)
            stmt = stmt.limit(limit)

            past_jobs = (await ctx.db.execute(stmt)).scalars().all()
            return {
                "success": True,
                "data": {
                    "jobs": [
                        {
                            "id": str(j.id),
                            "goal": (j.goal or "")[:200],
                            "job_type": j.job_type,
                            "status": j.status,
                            "iteration": j.iteration,
                            "tool_calls_used": j.tool_calls_used,
                            "llm_calls_used": j.llm_calls_used,
                            "tokens_used": j.tokens_used,
                            "error": (j.error or "")[:200] if j.error else None,
                            "created_at": j.created_at.isoformat()
                            if j.created_at
                            else None,
                            "completed_at": j.completed_at.isoformat()
                            if j.completed_at
                            else None,
                            "duration_minutes": round(
                                (j.completed_at - j.started_at).total_seconds() / 60, 1
                            )
                            if j.started_at and j.completed_at
                            else None,
                        }
                        for j in past_jobs
                    ],
                    "count": len(past_jobs),
                },
            }
        except Exception as exc:
            return {"error": f"Failed to get job history: {exc}"}

    async def _get_job_metrics(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from uuid import UUID as _UUID

        from app.models.agent_job import AgentJob as AgentJobModel

        job = ctx.job
        try:
            target_id_str = str(params.get("job_id", "")).strip()
            if target_id_str:
                target_job = await ctx.db.get(AgentJobModel, _UUID(target_id_str))
            else:
                target_job = job
            if not target_job:
                return {"error": f"Job not found: {target_id_str}"}
            if target_job.user_id != job.user_id:
                return {"error": "Not authorized to view this job's metrics"}

            duration = None
            if target_job.started_at and target_job.completed_at:
                duration = round(
                    (target_job.completed_at - target_job.started_at).total_seconds()
                    / 60,
                    2,
                )
            tool_counts = {}
            if target_job.execution_log:
                for entry in target_job.execution_log:
                    tool = entry.get("action")
                    if tool:
                        tool_counts[tool] = tool_counts.get(tool, 0) + 1

            return {
                "success": True,
                "data": {
                    "id": str(target_job.id),
                    "goal": (target_job.goal or "")[:200],
                    "status": target_job.status,
                    "job_type": target_job.job_type,
                    "iterations": target_job.iteration,
                    "tool_calls_used": target_job.tool_calls_used,
                    "llm_calls_used": target_job.llm_calls_used,
                    "tokens_used": target_job.tokens_used,
                    "max_tool_calls": target_job.max_tool_calls,
                    "max_llm_calls": target_job.max_llm_calls,
                    "max_runtime_minutes": target_job.max_runtime_minutes,
                    "duration_minutes": duration,
                    "error_count": target_job.error_count,
                    "tool_usage_breakdown": tool_counts,
                    "created_at": target_job.created_at.isoformat()
                    if target_job.created_at
                    else None,
                    "started_at": target_job.started_at.isoformat()
                    if target_job.started_at
                    else None,
                    "completed_at": target_job.completed_at.isoformat()
                    if target_job.completed_at
                    else None,
                },
            }
        except ValueError:
            return {"error": "Invalid job_id format"}
        except Exception as exc:
            return {"error": f"Failed to get job metrics: {exc}"}

    async def _get_tool_usage_stats(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from datetime import datetime, timedelta, timezone

        from app.models.agent_job import AgentJob as AgentJobModel

        job = ctx.job
        try:
            days = min(int(params.get("days", 7) or 7), 30)
            tool_name_filter = str(params.get("tool_name", "")).strip() or None
            cutoff = datetime.now(timezone.utc) - timedelta(days=days)
            stmt = select(AgentJobModel).where(
                AgentJobModel.user_id == job.user_id,
                AgentJobModel.created_at >= cutoff,
                AgentJobModel.execution_log.isnot(None),
            )
            analyzed_jobs = (await ctx.db.execute(stmt)).scalars().all()

            tool_stats = {}
            for row in analyzed_jobs:
                for entry in row.execution_log or []:
                    tool = entry.get("action")
                    if not tool or (tool_name_filter and tool != tool_name_filter):
                        continue
                    stats = tool_stats.setdefault(
                        tool, {"calls": 0, "successes": 0, "failures": 0}
                    )
                    stats["calls"] += 1
                    if entry.get("error"):
                        stats["failures"] += 1
                    else:
                        stats["successes"] += 1
            sorted_tools = sorted(
                tool_stats.items(), key=lambda x: x[1]["calls"], reverse=True
            )
            return {
                "success": True,
                "data": {
                    "period_days": days,
                    "total_jobs_analyzed": len(analyzed_jobs),
                    "tools": [
                        {
                            "name": name,
                            "calls": stats["calls"],
                            "successes": stats["successes"],
                            "failures": stats["failures"],
                            "success_rate": round(
                                stats["successes"] / stats["calls"], 3
                            )
                            if stats["calls"] > 0
                            else 0.0,
                        }
                        for name, stats in sorted_tools[:50]
                    ],
                },
            }
        except Exception as exc:
            return {"error": f"Failed to get tool usage stats: {exc}"}

    async def _get_tool_failure_analysis(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from datetime import datetime, timedelta, timezone

        from app.models.agent_job import AgentJob as AgentJobModel

        job = ctx.job
        analysis_tool_name = str(params.get("tool_name", "")).strip()
        if not analysis_tool_name:
            return {"error": "tool_name is required"}
        try:
            days = min(int(params.get("days", 7) or 7), 30)
            cutoff = datetime.now(timezone.utc) - timedelta(days=days)
            stmt = select(AgentJobModel).where(
                AgentJobModel.user_id == job.user_id,
                AgentJobModel.created_at >= cutoff,
                AgentJobModel.execution_log.isnot(None),
            )
            analyzed_jobs = (await ctx.db.execute(stmt)).scalars().all()
            total_calls = 0
            errors = []
            for row in analyzed_jobs:
                for entry in row.execution_log or []:
                    if entry.get("action") != analysis_tool_name:
                        continue
                    total_calls += 1
                    if entry.get("error"):
                        errors.append(
                            {
                                "job_id": str(row.id),
                                "job_type": row.job_type,
                                "error": str(entry.get("error"))[:200],
                                "timestamp": entry.get("timestamp"),
                                "iteration": entry.get("iteration"),
                            }
                        )
            error_patterns = {}
            for err in errors:
                key = err["error"][:80]
                pattern = error_patterns.setdefault(key, {"count": 0, "examples": []})
                pattern["count"] += 1
                if len(pattern["examples"]) < 3:
                    pattern["examples"].append(err)
            sorted_patterns = sorted(
                error_patterns.items(), key=lambda x: x[1]["count"], reverse=True
            )
            return {
                "success": True,
                "data": {
                    "tool_name": analysis_tool_name,
                    "period_days": days,
                    "total_calls": total_calls,
                    "total_failures": len(errors),
                    "failure_rate": round(len(errors) / total_calls, 3)
                    if total_calls > 0
                    else 0.0,
                    "error_patterns": [
                        {
                            "pattern": pat,
                            "count": info["count"],
                            "examples": info["examples"],
                        }
                        for pat, info in sorted_patterns[:10]
                    ],
                    "recent_failures": errors[-10:],
                },
            }
        except Exception as exc:
            return {"error": f"Failed to analyze tool failures: {exc}"}

    async def _batch_search(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        queries_raw = params.get("queries")
        if not queries_raw or not isinstance(queries_raw, list) or not queries_raw:
            return {"error": "queries is required and must be a non-empty array"}
        try:
            queries = [str(q).strip() for q in queries_raw if str(q).strip()][:10]
            if not queries:
                return {"error": "No valid queries provided"}
            limit_per = min(int(params.get("limit_per_query", 5) or 5), 20)
            source_id_filter = str(params.get("source_id", "")).strip() or None
            dedup = params.get("deduplicate", True)
            if dedup is None:
                dedup = True
            all_results = []
            seen_ids = set()
            findings = []
            for query in queries:
                try:
                    results_list, total, _took = await executor.search_service.search(
                        query=query,
                        mode="smart",
                        page=1,
                        page_size=limit_per,
                        source_id=source_id_filter,
                        db=ctx.db,
                    )
                    query_results = []
                    for row in results_list:
                        doc_id = row.get("id")
                        if dedup and doc_id in seen_ids:
                            continue
                        if doc_id:
                            seen_ids.add(doc_id)
                        query_results.append(row)
                    all_results.append(
                        {"query": query, "results": query_results, "total": total}
                    )
                except Exception as exc:
                    all_results.append(
                        {
                            "query": query,
                            "results": [],
                            "total": 0,
                            "error": str(exc)[:200],
                        }
                    )
            for qr in all_results:
                for row in qr.get("results", [])[:5]:
                    findings.append(
                        {
                            "type": "document",
                            "title": row.get("title"),
                            "id": row.get("id"),
                            "score": row.get("relevance_score", row.get("score")),
                            "query": qr.get("query"),
                        }
                    )
            return {
                "success": True,
                "data": {
                    "queries_executed": len(queries),
                    "results": all_results,
                    "total_unique_documents": len(seen_ids),
                },
                "findings": findings,
            }
        except Exception as exc:
            return {"error": f"Failed to execute batch search: {exc}"}

    async def _batch_summarize(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from uuid import UUID as _UUID

        from app.models.document import Document

        job = ctx.job
        doc_ids_raw = params.get("document_ids")
        if not doc_ids_raw or not isinstance(doc_ids_raw, list) or not doc_ids_raw:
            return {"error": "document_ids is required and must be a non-empty array"}
        try:
            doc_ids = [str(d).strip() for d in doc_ids_raw if str(d).strip()][:20]
            generate_missing = bool(params.get("generate_missing", False))
            summaries = []
            for doc_id_str in doc_ids:
                try:
                    doc = await ctx.db.get(Document, _UUID(doc_id_str))
                    if not doc:
                        summaries.append(
                            {"document_id": doc_id_str, "status": "not_found"}
                        )
                        continue
                    if doc.summary:
                        summaries.append(
                            {
                                "document_id": doc_id_str,
                                "title": doc.title,
                                "summary": doc.summary,
                                "status": "available",
                            }
                        )
                    elif generate_missing:
                        try:
                            summary_text = (
                                await executor.document_service.summarize_document(
                                    doc.id, ctx.db, user_id=job.user_id
                                )
                            )
                            summaries.append(
                                {
                                    "document_id": doc_id_str,
                                    "title": doc.title,
                                    "summary": summary_text or "",
                                    "status": "generated",
                                }
                            )
                        except Exception as exc:
                            summaries.append(
                                {
                                    "document_id": doc_id_str,
                                    "title": doc.title,
                                    "status": "generation_failed",
                                    "error": str(exc)[:200],
                                }
                            )
                    else:
                        summaries.append(
                            {
                                "document_id": doc_id_str,
                                "title": doc.title,
                                "status": "no_summary",
                            }
                        )
                except Exception:
                    summaries.append({"document_id": doc_id_str, "status": "error"})
            available = sum(
                1 for s in summaries if s.get("status") in ("available", "generated")
            )
            return {
                "success": True,
                "data": {
                    "summaries": summaries,
                    "total_requested": len(doc_ids),
                    "available": available,
                    "missing": len(doc_ids) - available,
                },
            }
        except Exception as exc:
            return {"error": f"Failed to execute batch summarize: {exc}"}

    async def _evaluate_condition(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        state = ctx.state if isinstance(ctx.state, dict) else {}
        try:
            condition = str(params.get("condition", "")).strip()
            threshold = int(params.get("threshold", 1) or 1)
            data: Dict[str, Any]
            if condition == "findings_count":
                count = len(state.get("findings", []))
                data = {
                    "met": count >= threshold,
                    "actual": count,
                    "threshold": threshold,
                    "condition": condition,
                }
            elif condition == "findings_has_category":
                cat = str(params.get("category", "")).strip()
                matches = [
                    f for f in state.get("findings", []) if f.get("category") == cat
                ]
                data = {
                    "met": len(matches) >= threshold,
                    "actual": len(matches),
                    "category": cat,
                    "threshold": threshold,
                    "condition": condition,
                }
            elif condition == "documents_count":
                source_id = str(params.get("source_id", "")).strip() or None
                _, total, _ = await executor.search_service.search(
                    query="*",
                    mode="smart",
                    page=1,
                    page_size=1,
                    source_id=source_id,
                    db=ctx.db,
                )
                data = {
                    "met": total >= threshold,
                    "actual": total,
                    "threshold": threshold,
                    "condition": condition,
                }
            elif condition == "search_has_results":
                query = str(params.get("query", "")).strip()
                source_id = str(params.get("source_id", "")).strip() or None
                if not query:
                    return {
                        "error": "query parameter required for search_has_results condition"
                    }
                _, total, _ = await executor.search_service.search(
                    query=query,
                    mode="smart",
                    page=1,
                    page_size=1,
                    source_id=source_id,
                    db=ctx.db,
                )
                data = {
                    "met": total >= threshold,
                    "actual": total,
                    "query": query,
                    "threshold": threshold,
                    "condition": condition,
                }
            elif condition == "actions_count":
                count = len(state.get("actions_taken", []))
                data = {
                    "met": count >= threshold,
                    "actual": count,
                    "threshold": threshold,
                    "condition": condition,
                }
            elif condition == "progress_above":
                progress = state.get("goal_progress", 0)
                data = {
                    "met": progress >= threshold,
                    "actual": progress,
                    "threshold": threshold,
                    "condition": condition,
                }
            else:
                return {
                    "error": f"Unknown condition: {condition}. Valid: findings_count, findings_has_category, documents_count, search_has_results, actions_count, progress_above"
                }
            return {"success": True, "data": data}
        except Exception as exc:
            return {"error": f"Failed to evaluate condition: {exc}"}

    async def _count_findings(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        state = ctx.state if isinstance(ctx.state, dict) else {}
        try:
            findings = state.get("findings", [])
            min_conf = float(params.get("min_confidence", 0.0) or 0.0)
            cat_filter = str(params.get("category", "")).strip() or None
            filtered = [
                f
                for f in findings
                if float(f.get("confidence", 0.8) or 0.8) >= min_conf
            ]
            if cat_filter:
                filtered = [f for f in filtered if f.get("category") == cat_filter]
            by_category: dict[str, int] = {}
            for finding in filtered:
                category = str(finding.get("category", "uncategorized"))
                by_category[category] = by_category.get(category, 0) + 1
            return {
                "success": True,
                "data": {
                    "total": len(filtered),
                    "by_category": by_category,
                    "categories": list(by_category.keys()),
                },
            }
        except Exception as exc:
            return {"error": f"Failed to count findings: {exc}"}

    def _contract_status(state: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize the executor's most recent goal-contract evaluation.

        Reuses `goal_contract_last` rather than re-evaluating: the executor
        writes it every iteration, and a second evaluation here could disagree
        with the one that actually gates completion.
        """
        from app.services import agent_measurement_validity

        last = state.get("goal_contract_last")
        if not isinstance(last, dict) or not last.get("enabled"):
            return {"goal_contract_enabled": False}

        missing = last.get("missing") if isinstance(last.get("missing"), list) else []
        validity = (last.get("metrics") or {}).get("validity") or {}
        status: Dict[str, Any] = {
            "goal_contract_enabled": True,
            "goal_contract_satisfied": bool(last.get("satisfied")),
            "goal_contract_missing": [str(x) for x in missing][:10],
        }
        remedies = agent_measurement_validity.explain(
            missing, validity.get("details") or {}
        )
        if remedies:
            status["goal_contract_remedies"] = remedies[:4]
        return status

    async def _check_goal_status(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        job = ctx.job
        state = ctx.state if isinstance(ctx.state, dict) else {}
        try:
            exec_plan = state.get("execution_plan")
            plan_steps_total = (
                len(exec_plan.get("steps", [])) if isinstance(exec_plan, dict) else 0
            )
            return {
                "success": True,
                "data": {
                    "iteration": job.iteration,
                    "max_iterations": job.max_iterations,
                    "iterations_remaining": job.max_iterations - job.iteration,
                    "tool_calls_used": job.tool_calls_used,
                    "max_tool_calls": job.max_tool_calls,
                    "tool_calls_remaining": job.max_tool_calls - job.tool_calls_used,
                    "goal_progress": state.get("goal_progress", 0),
                    "findings_count": len(state.get("findings", [])),
                    "actions_count": len(state.get("actions_taken", [])),
                    "has_execution_plan": bool(exec_plan),
                    "plan_steps_completed": state.get("plan_step_index", 0),
                    "plan_steps_total": plan_steps_total,
                    # What the run still has to satisfy, from the executor's
                    # own last evaluation. Without this the contract is only
                    # discoverable by trying to finish and being refused,
                    # which wastes the iteration that discovers it.
                    **_contract_status(state),
                },
            }
        except Exception as exc:
            return {"error": f"Failed to check goal status: {exc}"}

    async def _compress_history(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        state = ctx.state if isinstance(ctx.state, dict) else {}
        job = ctx.job
        try:
            actions = state.get("actions_taken", [])
            keep_last = min(int(params.get("keep_last", 5) or 5), 20)
            if len(actions) <= keep_last:
                return {
                    "success": True,
                    "data": {
                        "message": "Not enough history to compress",
                        "actions_count": len(actions),
                    },
                }
            to_compress = actions[:-keep_last] if keep_last > 0 else list(actions)
            actions_text = ""
            for action in to_compress:
                tool = (
                    action.get("action", {}).get("tool", "unknown")
                    if isinstance(action.get("action"), dict)
                    else "unknown"
                )
                res_summary = ""
                act_result = action.get("result", {})
                if isinstance(act_result, dict):
                    if act_result.get("success"):
                        data_keys = (
                            list(act_result.get("data", {}).keys())
                            if isinstance(act_result.get("data"), dict)
                            else []
                        )
                        res_summary = f"success, data keys: {data_keys}"
                    else:
                        res_summary = (
                            f"failed: {str(act_result.get('error', ''))[:100]}"
                        )
                actions_text += f"- Iteration {action.get('iteration', '?')}: {tool} → {res_summary}\n"
            existing_compressed = state.get("compressed_history", "")
            compress_prompt = (
                "Summarize the following agent action history into a concise narrative (max 500 words).\n"
                "Focus on: what was discovered, what worked/failed, key decisions made, and current trajectory.\n\n"
            )
            if existing_compressed:
                compress_prompt += (
                    f"Previous compressed history:\n{existing_compressed}\n\n"
                )
            compress_prompt += f"New actions to compress:\n{actions_text}\n\nWrite a concise summary in past tense."
            user_settings = await executor._get_user_settings(job.user_id, ctx.db)
            summary_resp = await executor.llm_service.generate_response(
                system_prompt="You are a concise summarizer. Output only the summary, no preamble.",
                user_message=compress_prompt,
                user_settings=user_settings,
                db=ctx.db,
                snapshot_context=_tool_snapshot_context(ctx, "compress_history"),
            )
            summary_text = str(summary_resp or "").strip()[:2000]
            state["compressed_history"] = summary_text
            state["actions_taken"] = actions[-keep_last:] if keep_last > 0 else []
            return {
                "success": True,
                "data": {
                    "compressed_actions": len(to_compress),
                    "kept_actions": len(state["actions_taken"]),
                    "summary_length": len(summary_text),
                },
            }
        except Exception as exc:
            return {"error": f"Failed to compress history: {exc}"}

    async def _summarize_findings(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        import uuid
        from datetime import datetime

        state = ctx.state if isinstance(ctx.state, dict) else {}
        job = ctx.job
        try:
            findings = state.get("findings", [])
            cat_filter = str(params.get("category", "")).strip() or None
            consolidate = bool(params.get("consolidate", False))
            target = (
                [f for f in findings if f.get("category") == cat_filter]
                if cat_filter
                else list(findings)
            )
            if not target:
                return {
                    "success": True,
                    "data": {"message": "No findings to summarize", "count": 0},
                }
            findings_text = ""
            for finding in target:
                findings_text += f"- [{finding.get('category', 'general')}] {finding.get('title', 'Untitled')}: {str(finding.get('content', ''))[:300]}\n"
            synth_prompt = (
                f"Synthesize these {len(target)} research findings into a coherent summary (max 800 words).\n"
                "Group related findings, identify themes, note contradictions, and highlight the most important insights.\n\n"
                f"Findings:\n{findings_text}\n\nWrite a structured synthesis."
            )
            user_settings = await executor._get_user_settings(job.user_id, ctx.db)
            synthesis_resp = await executor.llm_service.generate_response(
                system_prompt="You are a research synthesizer. Output only the synthesis, no preamble.",
                user_message=synth_prompt,
                user_settings=user_settings,
                db=ctx.db,
                snapshot_context=_tool_snapshot_context(ctx, "summarize_findings"),
            )
            synthesis_text = str(synthesis_resp or "").strip()[:3000]
            out: Dict[str, Any] = {
                "success": True,
                "data": {
                    "synthesis": synthesis_text,
                    "findings_summarized": len(target),
                    "consolidated": consolidate,
                },
            }
            if consolidate:
                if cat_filter:
                    state["findings"] = [
                        f for f in findings if f.get("category") != cat_filter
                    ]
                else:
                    state["findings"] = []
                consolidated = {
                    "id": str(uuid.uuid4()),
                    "title": f"Synthesis: {cat_filter or 'all findings'} ({len(target)} items)",
                    "content": synthesis_text,
                    "category": "synthesis",
                    "confidence": 0.9,
                    "tags": ["synthesized", "compressed"],
                    "created_at": datetime.utcnow().isoformat(),
                }
                state["findings"].append(consolidated)
                executor._job_findings.setdefault(str(job.id), []).append(consolidated)
                out["findings"] = [consolidated]
            return out
        except Exception as exc:
            return {"error": f"Failed to summarize findings: {exc}"}

    return FunctionToolProvider(
        name="autonomous_observability_tools",
        modes={"autonomous"},
        handlers={
            "get_job_history": _get_job_history,
            "get_job_metrics": _get_job_metrics,
            "get_tool_usage_stats": _get_tool_usage_stats,
            "get_tool_failure_analysis": _get_tool_failure_analysis,
            "batch_search": _batch_search,
            "batch_summarize": _batch_summarize,
            "evaluate_condition": _evaluate_condition,
            "count_findings": _count_findings,
            "check_goal_status": _check_goal_status,
            "compress_history": _compress_history,
            "summarize_findings": _summarize_findings,
        },
    )


def build_autonomous_output_state_provider(executor: Any) -> FunctionToolProvider:
    """Output shaping, strategy, and handoff tools for AutonomousAgentExecutor."""

    async def _create_handoff(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from app.models.agent_job import AgentJob, AgentJobStatus
        from app.tasks.agent_job_tasks import execute_agent_job_task

        job = ctx.job
        state = ctx.state if isinstance(ctx.state, dict) else {}
        try:
            child_goal = str(params.get("goal", "")).strip()
            expected_outputs = params.get("expected_outputs", [])
            if not child_goal:
                return {"error": "goal parameter is required"}
            if not isinstance(expected_outputs, list) or not expected_outputs:
                return {
                    "error": "expected_outputs must be a non-empty array of strings"
                }
            chain_depth = int(getattr(job, "chain_depth", 0) or 0)
            if chain_depth >= 3:
                return {
                    "error": "Maximum chain depth (3) reached — cannot create further handoffs"
                }
            existing_children = state.get("delegated_subtask_ids", [])
            if len(existing_children) >= 5:
                return {"error": "Maximum child jobs (5) reached for this parent"}
            child_type = str(params.get("job_type", "research")).strip()
            if child_type not in {"research", "analysis", "synthesis", "custom"}:
                child_type = "research"
            child_max = min(int(params.get("max_iterations", 10) or 10), 20)
            share = params.get("share_findings", True)
            if share is None:
                share = True
            child_config: dict = {
                "handoff_contract": {
                    "from_job_id": str(job.id),
                    "from_job_name": job.name or "unknown",
                    "context": str(params.get("context", ""))[:2000],
                    "expected_outputs": [str(o)[:200] for o in expected_outputs[:10]],
                },
            }
            if share:
                child_config["inherited_findings"] = (state.get("findings") or [])[-20:]
            source_scope_id = executor._resolve_default_source_scope(job)
            if source_scope_id:
                child_config["default_source_id"] = source_scope_id
            child_name = f"Handoff from {job.name or 'parent'}: {child_goal[:80]}"
            child = AgentJob(
                name=child_name[:200],
                description=f"Structured handoff from {job.name}: {child_goal[:500]}",
                job_type=child_type,
                goal=child_goal[:2000],
                config=child_config,
                status=AgentJobStatus.PENDING.value,
                user_id=job.user_id,
                parent_job_id=job.id,
                chain_depth=chain_depth + 1,
                root_job_id=getattr(job, "root_job_id", None) or job.id,
                max_iterations=child_max,
                max_tool_calls=min(child_max * 5, job.max_tool_calls or 500),
                max_llm_calls=min(child_max * 3, job.max_llm_calls or 200),
                max_runtime_minutes=min(30, job.max_runtime_minutes or 60),
                results={},
            )
            ctx.db.add(child)
            await ctx.db.flush()
            state.setdefault("delegated_subtask_ids", []).append(str(child.id))
            execute_agent_job_task.delay(str(child.id), str(job.user_id))
            return {
                "success": True,
                "data": {
                    "child_job_id": str(child.id),
                    "child_name": child_name[:200],
                    "job_type": child_type,
                    "expected_outputs": [str(o)[:200] for o in expected_outputs[:10]],
                    "max_iterations": child_max,
                    "findings_shared": bool(share),
                },
            }
        except Exception as exc:
            return {"error": f"Failed to create handoff: {exc}"}

    async def _get_sibling_status(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from app.models.agent_job import AgentJob

        job = ctx.job
        try:
            if not job.parent_job_id:
                return {"error": "This job has no parent — no siblings exist"}
            include_findings = bool(params.get("include_findings", False))
            siblings_q = await ctx.db.execute(
                select(AgentJob).where(
                    AgentJob.parent_job_id == job.parent_job_id,
                    AgentJob.id != job.id,
                    AgentJob.user_id == job.user_id,
                )
            )
            sibling_data = []
            for sibling in siblings_q.scalars().all():
                entry: dict = {
                    "job_id": str(sibling.id),
                    "name": sibling.name,
                    "job_type": sibling.job_type,
                    "status": sibling.status,
                    "iteration": sibling.iteration,
                    "max_iterations": sibling.max_iterations,
                }
                if include_findings and isinstance(sibling.results, dict):
                    findings = sibling.results.get("findings", [])
                    if isinstance(findings, list):
                        entry["findings_count"] = len(findings)
                        entry["finding_titles"] = [
                            str(f.get("title", ""))[:100]
                            for f in findings[:10]
                            if isinstance(f, dict)
                        ]
                sibling_data.append(entry)
            return {
                "success": True,
                "data": {"siblings": sibling_data, "count": len(sibling_data)},
            }
        except Exception as exc:
            return {"error": f"Failed to get sibling status: {exc}"}

    async def _broadcast_to_siblings(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from datetime import datetime

        from sqlalchemy.orm.attributes import flag_modified

        from app.models.agent_job import AgentJob

        job = ctx.job
        try:
            message = str(params.get("message", "")).strip()
            if not message:
                return {"error": "message parameter is required"}
            if not job.parent_job_id:
                return {"error": "This job has no parent — no siblings to broadcast to"}
            category = str(params.get("category", "broadcast")).strip()[:100]
            msg_entry = {
                "from_job_id": str(job.id),
                "from_job_name": job.name or "unknown",
                "message": message[:2000],
                "category": category,
                "sent_at": datetime.utcnow().isoformat(),
                "broadcast": True,
            }
            siblings_q = await ctx.db.execute(
                select(AgentJob).where(
                    AgentJob.parent_job_id == job.parent_job_id,
                    AgentJob.id != job.id,
                    AgentJob.user_id == job.user_id,
                )
            )
            delivered = 0
            for sibling in siblings_q.scalars().all():
                s_results = sibling.results if isinstance(sibling.results, dict) else {}
                agent_msgs = s_results.get("agent_messages", [])
                if not isinstance(agent_msgs, list):
                    agent_msgs = []
                agent_msgs.append(msg_entry)
                s_results["agent_messages"] = agent_msgs[-100:]
                sibling.results = s_results
                flag_modified(sibling, "results")
                delivered += 1
            await ctx.db.flush()
            return {
                "success": True,
                "data": {"recipients": delivered, "message_length": len(message)},
            }
        except Exception as exc:
            return {"error": f"Failed to broadcast to siblings: {exc}"}

    async def _switch_strategy(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from datetime import datetime

        job = ctx.job
        state = ctx.state if isinstance(ctx.state, dict) else {}
        try:
            valid_roles = {
                "researcher",
                "critic",
                "synthesizer",
                "verifier",
                "coder",
                "author",
            }
            role = str(params.get("role", "")).strip().lower()
            if role not in valid_roles:
                return {
                    "error": f"Invalid role: {role}. Valid: {', '.join(sorted(valid_roles))}"
                }
            old_role = (
                state.get("skill_profile", {}).get("role", "unknown")
                if isinstance(state.get("skill_profile"), dict)
                else "unknown"
            )
            new_profile = executor._resolve_agent_skill_profile(
                job, state=state, override_role=role
            )
            state["skill_profile"] = new_profile
            state.setdefault("strategy_switches", []).append(
                {
                    "from": old_role,
                    "to": role,
                    "reason": str(params.get("reason", ""))[:500],
                    "iteration": job.iteration,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )
            return {
                "success": True,
                "data": {
                    "previous_role": old_role,
                    "new_role": role,
                    "display_name": new_profile.get("display_name", role),
                    "preferred_tools": new_profile.get("preferred_tools", [])[:5],
                },
            }
        except Exception as exc:
            return {"error": f"Failed to switch strategy: {exc}"}

    async def _set_focus_directive(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        state = ctx.state if isinstance(ctx.state, dict) else {}
        try:
            directive = str(params.get("directive", "")).strip()[:1000]
            if not directive:
                return {"error": "directive parameter is required"}
            append = bool(params.get("append", False))
            if append:
                existing = str(state.get("focus_directive", ""))
                state["focus_directive"] = (existing + "\n" + directive).strip()[:2000]
            else:
                state["focus_directive"] = directive
            return {
                "success": True,
                "data": {
                    "directive": state["focus_directive"],
                    "mode": "appended" if append else "replaced",
                },
            }
        except Exception as exc:
            return {"error": f"Failed to set focus directive: {exc}"}

    async def _get_available_strategies(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        job = ctx.job
        state = ctx.state if isinstance(ctx.state, dict) else {}
        try:
            strategies = []
            for role_name in [
                "researcher",
                "critic",
                "synthesizer",
                "verifier",
                "coder",
                "author",
            ]:
                profile = executor._resolve_agent_skill_profile(
                    job, state=state, override_role=role_name
                )
                strategies.append(
                    {
                        "role": role_name,
                        "display_name": profile.get("display_name", role_name),
                        "guidance": "; ".join(profile.get("prompt_directives", []))[
                            :300
                        ],
                        "preferred_tools": profile.get("preferred_tools", [])[:5],
                        "discouraged_tools": profile.get("discouraged_tools", []),
                    }
                )
            current = (
                state.get("skill_profile", {}).get("role", "researcher")
                if isinstance(state.get("skill_profile"), dict)
                else "researcher"
            )
            return {
                "success": True,
                "data": {"strategies": strategies, "current_role": current},
            }
        except Exception as exc:
            return {"error": f"Failed to get available strategies: {exc}"}

    async def _format_as_table(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        state = ctx.state if isinstance(ctx.state, dict) else {}
        try:
            title = str(params.get("title", "")).strip()
            source = str(params.get("source", "custom")).strip()
            columns = params.get("columns", [])
            rows = params.get("rows", [])
            if not title:
                return {"error": "title parameter is required"}
            if source == "findings":
                findings = state.get("findings", [])
                fields = params.get(
                    "finding_fields", ["title", "category", "confidence"]
                )
                if not isinstance(fields, list):
                    fields = ["title", "category", "confidence"]
                fields = [str(f).strip() for f in fields if str(f).strip()][:10]
                columns = fields
                rows = []
                for finding in findings:
                    if isinstance(finding, dict):
                        rows.append(
                            [str(finding.get(field, ""))[:200] for field in fields]
                        )
            elif (
                not isinstance(columns, list)
                or not columns
                or not isinstance(rows, list)
                or not rows
            ):
                return {"error": "columns and rows are required for custom tables"}
            md = f"## {title}\n\n"
            col_headers = [str(c) for c in columns]
            md += "| " + " | ".join(col_headers) + " |\n"
            md += "| " + " | ".join("---" for _ in col_headers) + " |\n"
            row_count = 0
            for row in rows[:100]:
                cells = [
                    str(c).replace("|", "\\|")[:200]
                    for c in (row if isinstance(row, list) else [])
                ]
                while len(cells) < len(col_headers):
                    cells.append("")
                cells = cells[: len(col_headers)]
                md += "| " + " | ".join(cells) + " |\n"
                row_count += 1
            state.setdefault("formatted_outputs", []).append(
                {
                    "type": "table",
                    "title": title,
                    "markdown": md,
                    "columns": col_headers,
                    "row_count": row_count,
                }
            )
            return {
                "success": True,
                "data": {
                    "markdown": md,
                    "row_count": row_count,
                    "columns": len(col_headers),
                },
                "artifacts": [{"type": "formatted_table", "title": title}],
            }
        except Exception as exc:
            return {"error": f"Failed to format as table: {exc}"}

    async def _format_as_report(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from loguru import logger

        from app.models.document import Document

        job = ctx.job
        state = ctx.state if isinstance(ctx.state, dict) else {}
        try:
            title = str(params.get("title", "")).strip()
            if not title:
                return {"error": "title parameter is required"}
            md = f"# {title}\n\n"
            exec_summary = str(params.get("executive_summary", "")).strip()
            if exec_summary:
                md += f"## Executive Summary\n\n{exec_summary[:3000]}\n\n"
            sections = params.get("sections", [])
            if isinstance(sections, list):
                for sec in sections[:20]:
                    if isinstance(sec, dict):
                        heading = str(sec.get("heading", "Section"))[:200]
                        content = str(sec.get("content", ""))[:5000]
                        md += f"## {heading}\n\n{content}\n\n"
            include_findings = params.get("include_findings", True)
            if include_findings is None:
                include_findings = True
            if include_findings:
                findings = state.get("findings", [])
                if findings:
                    md += "## Findings\n\n"
                    for i, finding in enumerate(findings[:30], 1):
                        if not isinstance(finding, dict):
                            continue
                        md += f"### {i}. {finding.get('title', 'Untitled')}\n\n"
                        md += f"{str(finding.get('content', ''))[:1000]}\n\n"
                        meta_parts = []
                        if finding.get("category"):
                            meta_parts.append(f"Category: {finding['category']}")
                        if finding.get("confidence"):
                            meta_parts.append(f"Confidence: {finding['confidence']}")
                        if meta_parts:
                            md += f"*{' | '.join(meta_parts)}*\n\n"
            include_progress = params.get("include_progress", True)
            if include_progress is None:
                include_progress = True
            if include_progress:
                reports = state.get("progress_reports", [])
                if reports:
                    md += "## Progress History\n\n"
                    for report in reports[-5:]:
                        if isinstance(report, dict):
                            md += f"### Iteration {report.get('iteration', '?')}\n\n"
                            if report.get("summary"):
                                md += f"{report['summary']}\n\n"
            md = md[:50000]
            state.setdefault("formatted_outputs", []).append(
                {"type": "report", "title": title, "markdown": md}
            )
            doc_id = None
            if params.get("persist", False):
                try:
                    notes_source = (
                        await executor.document_service._ensure_agent_notes_source(
                            ctx.db, job.user_id
                        )
                    )
                    doc = Document(
                        title=title[:500],
                        content=md,
                        file_type="text/markdown",
                        source_id=notes_source.id,
                        user_id=job.user_id,
                    )
                    ctx.db.add(doc)
                    await ctx.db.flush()
                    doc_id = str(doc.id)
                    state.setdefault("artifacts", []).append(
                        {"type": "document", "id": doc_id, "title": title[:500]}
                    )
                except Exception as doc_exc:
                    logger.warning(f"Failed to persist report document: {doc_exc}")
            return {
                "success": True,
                "data": {"markdown": md, "length": len(md), "document_id": doc_id},
                "artifacts": [{"type": "formatted_report", "title": title}],
            }
        except Exception as exc:
            return {"error": f"Failed to format as report: {exc}"}

    async def _set_output_schema(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        state = ctx.state if isinstance(ctx.state, dict) else {}
        try:
            schema = params.get("schema")
            if not isinstance(schema, dict) or not schema:
                return {"error": "schema must be a non-empty object"}
            merge = params.get("merge", True)
            if merge is None:
                merge = True
            if merge:
                existing = state.get("output_schema", {})
                if not isinstance(existing, dict):
                    existing = {}
                existing.update(schema)
                state["output_schema"] = existing
            else:
                state["output_schema"] = dict(schema)
            return {
                "success": True,
                "data": {
                    "schema_keys": list(state["output_schema"].keys()),
                    "total_keys": len(state["output_schema"]),
                    "mode": "merged" if merge else "replaced",
                },
            }
        except Exception as exc:
            return {"error": f"Failed to set output schema: {exc}"}

    return FunctionToolProvider(
        name="autonomous_output_state_tools",
        modes={"autonomous"},
        handlers={
            "create_handoff": _create_handoff,
            "get_sibling_status": _get_sibling_status,
            "broadcast_to_siblings": _broadcast_to_siblings,
            "switch_strategy": _switch_strategy,
            "set_focus_directive": _set_focus_directive,
            "get_available_strategies": _get_available_strategies,
            "format_as_table": _format_as_table,
            "format_as_report": _format_as_report,
            "set_output_schema": _set_output_schema,
        },
    )


def build_autonomous_web_research_provider(executor: Any) -> FunctionToolProvider:
    """External web research helpers for AutonomousAgentExecutor."""

    async def _search_web(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        import re as _re
        from urllib.parse import unquote

        import httpx

        query = str(params.get("query", "")).strip()
        if not query:
            return {"error": "query is required"}
        try:
            max_results = min(int(params.get("max_results", 5) or 5), 10)
            async with httpx.AsyncClient(
                timeout=15.0,
                headers={"User-Agent": "Mozilla/5.0 (compatible; KnowledgeDBChat/1.0)"},
                follow_redirects=True,
            ) as client:
                resp = await client.get(
                    "https://html.duckduckgo.com/html/", params={"q": query}
                )
                resp.raise_for_status()
            result_blocks = _re.findall(
                r'<a[^>]+class="result__a"[^>]+href="([^"]*)"[^>]*>(.*?)</a>.*?'
                r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>',
                resp.text,
                _re.DOTALL,
            )
            results_list = []
            for url_raw, title_raw, snippet_raw in result_blocks[:max_results]:
                title_clean = _re.sub(r"<[^>]+>", "", title_raw).strip()
                snippet_clean = _re.sub(r"<[^>]+>", "", snippet_raw).strip()
                url_match = _re.search(r"uddg=([^&]+)", url_raw)
                url_clean = unquote(url_match.group(1) if url_match else url_raw)
                if title_clean:
                    results_list.append(
                        {
                            "title": title_clean[:200],
                            "url": url_clean[:500],
                            "snippet": snippet_clean[:500],
                        }
                    )
            return {
                "success": True,
                "data": {
                    "query": query,
                    "results": results_list,
                    "count": len(results_list),
                },
            }
        except Exception as exc:
            return {"error": f"Web search failed: {exc}"}

    async def _scrape_one_page(url: str, max_chars: int) -> Dict[str, Any]:
        """Fetch a single page.

        WebScraperService exposes `scrape`, which crawls and returns a "pages"
        list; there is no scrape_url. Both handlers below want one page.
        """
        from app.services.web_scraper_service import WebScraperService

        scraper = WebScraperService()
        try:
            result = await scraper.scrape(
                url,
                follow_links=False,
                max_pages=1,
                include_links=False,
                max_content_chars=max_chars,
            )
        finally:
            await scraper.aclose()
        pages = (result or {}).get("pages") or []
        return pages[0] if isinstance(pages[0], dict) else {} if pages else {}

    async def _fetch_url_content(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        url = str(params.get("url", "")).strip()
        if not url:
            return {"error": "url is required"}
        try:
            max_chars = min(int(params.get("max_chars", 50000) or 50000), 100000)
            page = await _scrape_one_page(url, max_chars)
            content = str(page.get("content", ""))[:max_chars]
            title = str(page.get("title", ""))[:200]
            if not content.strip():
                return {"error": f"No content extracted from {url}"}
            return {
                "success": True,
                "data": {
                    "url": url,
                    "title": title,
                    "content": content,
                    "content_length": len(content),
                },
            }
        except Exception as exc:
            return {"error": f"Failed to fetch URL: {exc}"}

    async def _summarize_url(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        url = str(params.get("url", "")).strip()
        if not url:
            return {"error": "url is required"}
        try:
            page = await _scrape_one_page(url, 100000)
            text = str(page.get("content", ""))[:50000]
            if not text.strip():
                return {"error": f"No content extracted from {url}"}
            focus = str(params.get("focus", "")).strip()
            focus_clause = f" with focus on: {focus}" if focus else ""
            # LLMService has no `generate`; the text entry point is
            # generate_response(system_prompt=..., user_message=...).
            summary = await executor.llm_service.generate_response(
                system_prompt=(
                    "Summarize the web page content the user provides"
                    f"{focus_clause}. Be concise and extract key information."
                ),
                user_message=text[:30000],
                max_tokens=1000,
                db=ctx.db,
                snapshot_context=_tool_snapshot_context(ctx, "summarize_url"),
            )
            return {
                "success": True,
                "data": {
                    "url": url,
                    "summary": summary,
                    "content_length": len(text),
                    "focus": focus or None,
                },
            }
        except Exception as exc:
            return {"error": f"URL summarization failed: {exc}"}

    return FunctionToolProvider(
        name="autonomous_web_research_tools",
        modes={"autonomous"},
        handlers={
            "search_web": _search_web,
            "fetch_url_content": _fetch_url_content,
            "summarize_url": _summarize_url,
        },
    )


def build_autonomous_notification_visualization_provider(
    executor: Any,
) -> FunctionToolProvider:
    """Notification and standalone visualization tools for AutonomousAgentExecutor."""

    async def _send_notification(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from app.services.notification_service import NotificationService

        job = ctx.job
        notif_title = str(params.get("title", "")).strip()
        notif_message = str(params.get("message", "")).strip()
        if not notif_title:
            return {"error": "title is required"}
        if not notif_message:
            return {"error": "message is required"}
        try:
            ns = NotificationService()
            priority = str(params.get("priority", "normal")).strip().lower()
            if priority not in {"low", "normal", "high", "urgent"}:
                priority = "normal"
            notification = await ns.create_notification(
                db=ctx.db,
                user_id=job.user_id,
                notification_type="agent_job_alert",
                title=notif_title[:200],
                message=notif_message[:2000],
                priority=priority,
                related_entity_type="agent_job",
                related_entity_id=job.id,
                data={"source_job_id": str(job.id), "source_job_name": job.name or ""},
                action_url=str(params.get("action_url", "")).strip()[:500] or None,
                commit=False,
            )
            await ctx.db.flush()
            return {
                "success": True,
                "data": {
                    "notification_id": str(notification.id) if notification else None,
                    "delivered": notification is not None,
                    "priority": priority,
                },
            }
        except Exception as exc:
            return {"error": f"Failed to send notification: {exc}"}

    async def _send_email_alert(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from loguru import logger

        from app.services.notification_service import NotificationService

        job = ctx.job
        subject = str(params.get("subject", "")).strip()
        body = str(params.get("body", "")).strip()
        if not subject:
            return {"error": "subject is required"}
        if not body:
            return {"error": "body is required"}
        try:
            logger.info(
                f"Email alert requested by job {job.id} (no SMTP configured), falling back to notification"
            )
            ns = NotificationService()
            priority = str(params.get("priority", "normal")).strip().lower()
            if priority not in {"low", "normal", "high", "urgent"}:
                priority = "normal"
            notification = await ns.create_notification(
                db=ctx.db,
                user_id=job.user_id,
                notification_type="agent_job_alert",
                title=f"[Email] {subject[:180]}",
                message=body[:2000],
                priority=priority,
                related_entity_type="agent_job",
                related_entity_id=job.id,
                data={"intended_delivery": "email", "source_job_id": str(job.id)},
                commit=False,
            )
            await ctx.db.flush()
            return {
                "success": True,
                "data": {
                    "notification_id": str(notification.id) if notification else None,
                    "delivered": notification is not None,
                    "delivery_method": "in_app_notification",
                    "note": "SMTP not configured; delivered as in-app notification",
                },
            }
        except Exception as exc:
            return {"error": f"Failed to send email alert: {exc}"}

    async def _create_chart(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        import base64 as b64
        from uuid import uuid4 as _uuid4

        from loguru import logger

        from app.services.storage_service import storage_service
        from app.services.visualization_service import VisualizationService

        job = ctx.job
        chart_type = str(params.get("chart_type", "")).strip().lower()
        data = params.get("data")
        if not chart_type:
            return {"error": "chart_type is required"}
        if not data or not isinstance(data, dict):
            return {"error": "data is required and must be an object"}
        if chart_type not in {
            "bar",
            "line",
            "pie",
            "scatter",
            "histogram",
            "heatmap",
            "box",
            "area",
        }:
            return {
                "error": f"Invalid chart_type: {chart_type}. Must be bar, line, pie, scatter, histogram, heatmap, box, or area"
            }
        try:
            vs = VisualizationService()
            fmt = str(params.get("format", "png")).strip().lower()
            if fmt not in {"png", "svg"}:
                fmt = "png"
            config = {"format": fmt}
            for key in ("title", "x_label", "y_label"):
                val = str(params.get(key, "")).strip()
                if val:
                    config[key] = val
            chart_result = vs.create_chart(
                chart_type=chart_type, data=data, config=config
            )
            image_bytes = b64.b64decode(chart_result["image_base64"])
            object_path = f"agent_artifacts/{job.id}/charts/{_uuid4()}.{fmt}"
            await storage_service.initialize()
            await storage_service.upload_to_path(
                object_path, image_bytes, chart_result.get("mime_type", f"image/{fmt}")
            )
            url = await storage_service.get_presigned_download_url(object_path)
            return {
                "success": True,
                "data": {
                    "chart_type": chart_type,
                    "url": url,
                    "format": fmt,
                    "size_bytes": len(image_bytes),
                },
            }
        except Exception as exc:
            logger.error(f"create_chart failed: {exc}")
            return {"error": f"Failed to create chart: {exc}"}

    async def _render_diagram(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        import base64 as b64
        from uuid import uuid4 as _uuid4

        from loguru import logger

        from app.services.storage_service import storage_service

        job = ctx.job
        diagram_code = str(params.get("diagram_code", "")).strip()
        if not diagram_code:
            return {"error": "diagram_code is required"}
        try:
            diagram_type = str(params.get("diagram_type", "mermaid")).strip().lower()
            fmt = str(params.get("format", "png")).strip().lower()
            if fmt not in {"png", "svg"}:
                fmt = "png"
            mime = f"image/{fmt}" if fmt == "png" else "image/svg+xml"
            if diagram_type == "graphviz":
                from app.services.diagram_service import DiagramService

                ds = DiagramService()
                image_bytes = b64.b64decode(
                    ds._render_graphviz(diagram_code, {"output_format": fmt})
                )
            else:
                from app.services.mermaid_renderer import MermaidRenderer

                renderer = MermaidRenderer()
                if fmt == "svg":
                    svg_str = await renderer.render_to_svg(diagram_code)
                    image_bytes = (
                        svg_str.encode("utf-8") if isinstance(svg_str, str) else svg_str
                    )
                else:
                    image_bytes = await renderer.render_to_png(diagram_code)
            object_path = f"agent_artifacts/{job.id}/diagrams/{_uuid4()}.{fmt}"
            await storage_service.initialize()
            await storage_service.upload_to_path(object_path, image_bytes, mime)
            url = await storage_service.get_presigned_download_url(object_path)
            return {
                "success": True,
                "data": {
                    "url": url,
                    "diagram_type": diagram_type,
                    "format": fmt,
                    "size_bytes": len(image_bytes),
                },
            }
        except Exception as exc:
            logger.error(f"render_diagram failed: {exc}")
            return {"error": f"Failed to render diagram: {exc}"}

    return FunctionToolProvider(
        name="autonomous_notification_visualization_tools",
        modes={"autonomous"},
        handlers={
            "send_notification": _send_notification,
            "send_email_alert": _send_email_alert,
            "create_chart": _create_chart,
            "render_diagram": _render_diagram,
        },
    )


def build_autonomous_kg_provider(executor: Any) -> FunctionToolProvider:
    """Knowledge-graph and related placeholder research helpers for AutonomousAgentExecutor."""

    async def _build_research_graph(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        """Extract entities and relationships from documents into the graph.

        Re-extracts per document, so calling it twice does not double up: the
        rebuild clears that document's existing mentions and relationships
        first.
        """
        from app.services.knowledge_graph_service import KnowledgeGraphService

        documents, error = await _load_documents_for_analysis(
            ctx, params.get("document_ids"), max_docs=20
        )
        if error:
            return {"error": error}

        kg = KnowledgeGraphService()
        entities_found = 0
        relationships_found = 0
        failures: list[str] = []
        for doc in documents:
            try:
                result = await kg.rebuild_for_document(ctx.db, doc.id)
            except Exception as exc:  # noqa: BLE001 - reported per document
                failures.append(f"{doc.id}: {exc}")
                continue
            if isinstance(result, dict):
                # rebuild_for_document reports "mentions"; reading "entities"
                # found nothing every time, so a graph built from documents
                # carrying 52, 38 and 33 mentions reported zero and still
                # returned success.
                entities_found += int(
                    result.get("mentions") or result.get("entities") or 0
                )
                relationships_found += int(result.get("relationships") or 0)

        if failures and not entities_found and not relationships_found:
            return {"error": "Graph extraction failed: " + "; ".join(failures[:3])}

        return {
            "success": True,
            "data": {
                "documents_analyzed": len(documents),
                "focus": params.get("focus_on", ["methods", "concepts"]),
                "entities_found": entities_found,
                "mentions_found": entities_found,
                "relationships_found": relationships_found,
                "failed_documents": failures[:5],
            },
        }

    async def _link_entities(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        """Create a knowledge-graph relationship between two entities.

        Accepts entity UUIDs or names; names are resolved case-insensitively
        against canonical names, and an ambiguous name is reported rather than
        guessed at, since linking the wrong entities silently corrupts the graph.
        """
        from app.services.knowledge_graph_service import KnowledgeGraphService

        kg = KnowledgeGraphService()
        relation_type = str(params.get("relationship_type") or "").strip()
        if not relation_type:
            return {"error": "relationship_type is required"}

        async def _resolve(id_key: str, name_key: str, label: str):
            raw_id = str(params.get(id_key) or "").strip()
            if raw_id:
                return raw_id, None
            name = str(params.get(name_key) or "").strip()
            if not name:
                return None, f"{label} requires {id_key} or {name_key}"
            matches = await kg.entities(ctx.db, q=name, limit=25)
            exact = [
                e for e in matches if str(e.canonical_name).lower() == name.lower()
            ]
            candidates = exact or matches
            if not candidates:
                return None, f"No entity found matching {name!r}"
            if len(candidates) > 1:
                names = ", ".join(str(e.canonical_name) for e in candidates[:5])
                return None, f"{name!r} is ambiguous; candidates: {names}"
            return str(candidates[0].id), None

        source_id, error = await _resolve("source_entity_id", "source_name", "source")
        if error:
            return {"error": error}
        target_id, error = await _resolve("target_entity_id", "target_name", "target")
        if error:
            return {"error": error}
        if source_id == target_id:
            return {"error": "source and target resolve to the same entity"}

        try:
            confidence = float(params.get("confidence", 0.8) or 0.8)
        except (TypeError, ValueError):
            confidence = 0.8
        confidence = max(0.0, min(1.0, confidence))

        try:
            relationship = await kg.create_relationship(
                ctx.db,
                source_entity_id=source_id,
                target_entity_id=target_id,
                relation_type=relation_type,
                confidence=confidence,
                evidence=str(params.get("evidence") or "").strip() or None,
            )
        except ValueError as exc:
            return {"error": str(exc)}

        return {
            "success": True,
            "data": {
                "relationship_id": str(relationship.id),
                "source_entity_id": source_id,
                "target_entity_id": target_id,
                "relationship_type": relationship.relation_type,
                "confidence": relationship.confidence,
            },
        }

    async def _create_knowledge_base_entry(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        """Persist curated knowledge as a research note.

        Previously claimed entry_created=True and emitted an artifact that was
        never written anywhere.
        """
        from app.models.research_note import ResearchNote

        title = str(params.get("title") or "").strip()
        content = str(params.get("content") or "").strip()
        if not title or not content:
            return {"error": "title and content are required"}

        user_id = ctx.user_id or getattr(ctx.job, "user_id", None)
        if user_id is None:
            return {"error": "no user context for the knowledge base entry"}

        tags = params.get("tags")
        entry_type = str(params.get("entry_type") or "").strip()
        note = ResearchNote(
            user_id=user_id,
            title=title[:500],
            content_markdown=content[:120000],
            tags=(
                [str(t).strip() for t in tags if str(t).strip()][:20]
                if isinstance(tags, list)
                else ([entry_type] if entry_type else None)
            ),
        )
        ctx.db.add(note)
        await ctx.db.commit()
        await ctx.db.refresh(note)

        return {
            "success": True,
            "data": {
                "entry_created": True,
                "research_note_id": str(note.id),
                "title": note.title,
                "type": entry_type or None,
            },
        }

    async def _compare_documents(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        """Compare two documents.

        Delegates to the same implementation the interactive agent uses. This
        path previously returned an invented similarity_score of 0.0 with
        success=True, so the autonomous runner — the one nobody watches — was
        the only consumer getting a fabricated answer.
        """
        from app.services.agent_service import AgentService

        return await AgentService()._tool_compare_documents(params, ctx.user_id, ctx.db)

    async def _query_kg_entities(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from app.services.knowledge_graph_service import KnowledgeGraphService

        query = str(params.get("query", "")).strip()
        if not query:
            return {"error": "query is required"}
        try:
            kg = KnowledgeGraphService()
            limit = min(int(params.get("limit", 20) or 20), 100)
            entity_type = str(params.get("entity_type", "")).strip() or None
            entities = await kg.entities(ctx.db, q=query, limit=limit)
            if entity_type:
                entities = [e for e in entities if e.entity_type == entity_type]
            return {
                "success": True,
                "data": {
                    "query": query,
                    "entities": [
                        {
                            "id": str(e.id),
                            "canonical_name": e.canonical_name,
                            "entity_type": e.entity_type,
                            "description": e.description or "",
                        }
                        for e in entities
                    ],
                    "count": len(entities),
                },
            }
        except Exception as exc:
            return {"error": f"Failed to query KG entities: {exc}"}

    async def _get_entity_context(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from uuid import UUID as _UUID

        from app.services.knowledge_graph_service import KnowledgeGraphService

        entity_id = str(params.get("entity_id", "")).strip()
        if not entity_id:
            return {"error": "entity_id is required"}
        try:
            kg = KnowledgeGraphService()
            context = await kg.get_entity_context(
                [_UUID(entity_id)], ctx.db, max_relationships=30
            )
            return {"success": True, "data": context}
        except ValueError:
            return {"error": f"Invalid entity_id format: {entity_id}"}
        except Exception as exc:
            return {"error": f"Failed to get entity context: {exc}"}

    async def _create_kg_entity(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from app.models.knowledge_graph import Entity as KGEntity

        name = str(params.get("name", "")).strip()
        entity_type = str(params.get("entity_type", "")).strip().lower()
        if not name:
            return {"error": "name is required"}
        if not entity_type:
            return {"error": "entity_type is required"}
        try:
            entity = KGEntity(
                canonical_name=name[:512],
                entity_type=entity_type[:64],
                description=str(params.get("description", "")).strip() or None,
            )
            ctx.db.add(entity)
            await ctx.db.flush()
            return {
                "success": True,
                "data": {
                    "id": str(entity.id),
                    "canonical_name": entity.canonical_name,
                    "entity_type": entity.entity_type,
                    "description": entity.description or "",
                },
            }
        except Exception as exc:
            return {"error": f"Failed to create KG entity: {exc}"}

    async def _create_kg_relationship(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from app.services.knowledge_graph_service import KnowledgeGraphService

        source_id = str(params.get("source_entity_id", "")).strip()
        target_id = str(params.get("target_entity_id", "")).strip()
        relation_type = str(params.get("relation_type", "")).strip()
        if not source_id:
            return {"error": "source_entity_id is required"}
        if not target_id:
            return {"error": "target_entity_id is required"}
        if not relation_type:
            return {"error": "relation_type is required"}
        try:
            kg = KnowledgeGraphService()
            confidence = max(0.0, min(1.0, float(params.get("confidence", 0.8) or 0.8)))
            rel = await kg.create_relationship(
                db=ctx.db,
                source_entity_id=source_id,
                target_entity_id=target_id,
                relation_type=relation_type[:128],
                confidence=confidence,
                evidence=str(params.get("evidence", "")).strip() or None,
            )
            return {
                "success": True,
                "data": {
                    "id": str(rel.id),
                    "relation_type": rel.relation_type,
                    "source_entity_id": str(rel.source_entity_id),
                    "target_entity_id": str(rel.target_entity_id),
                    "confidence": rel.confidence,
                },
            }
        except Exception as exc:
            return {"error": f"Failed to create KG relationship: {exc}"}

    async def _query_kg_graph(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from app.services.knowledge_graph_service import KnowledgeGraphService

        try:
            kg = KnowledgeGraphService()
            graph = await kg.global_graph(
                db=ctx.db,
                entity_types=params.get("entity_types")
                if isinstance(params.get("entity_types"), list)
                else None,
                relation_types=params.get("relation_types")
                if isinstance(params.get("relation_types"), list)
                else None,
                min_confidence=float(params.get("min_confidence", 0.0) or 0.0),
                search=str(params.get("search", "")).strip() or None,
                limit_nodes=min(int(params.get("limit_nodes", 50) or 50), 200),
                limit_edges=min(int(params.get("limit_nodes", 50) or 50), 200) * 3,
            )
            return {"success": True, "data": graph}
        except Exception as exc:
            return {"error": f"Failed to query KG graph: {exc}"}

    return FunctionToolProvider(
        name="autonomous_kg_tools",
        modes={"autonomous"},
        handlers={
            "build_research_graph": _build_research_graph,
            "link_entities": _link_entities,
            "create_knowledge_base_entry": _create_knowledge_base_entry,
            "compare_documents": _compare_documents,
            "query_kg_entities": _query_kg_entities,
            "get_entity_context": _get_entity_context,
            "create_kg_entity": _create_kg_entity,
            "create_kg_relationship": _create_kg_relationship,
            "query_kg_graph": _query_kg_graph,
        },
    )


def build_autonomous_scheduling_provider(executor: Any) -> FunctionToolProvider:
    """Scheduling helpers for AutonomousAgentExecutor."""

    async def _schedule_job(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from datetime import datetime, timezone

        from app.models.agent_job import AgentJob as AgentJobModel

        job = ctx.job
        goal = str(params.get("goal", "")).strip()
        schedule_type = str(params.get("schedule_type", "")).strip().lower()
        if not goal:
            return {"error": "goal is required"}
        if schedule_type not in {"once", "recurring"}:
            return {"error": "schedule_type must be 'once' or 'recurring'"}
        try:
            job_type_param = str(params.get("job_type", "research")).strip().lower()
            config_param = (
                params.get("config") if isinstance(params.get("config"), dict) else {}
            )
            next_run = None
            cron_expr = None
            if schedule_type == "once":
                run_at = str(params.get("run_at", "")).strip()
                if not run_at:
                    return {"error": "run_at is required for schedule_type=once"}
                next_run = datetime.fromisoformat(run_at)
                if next_run.tzinfo is None:
                    next_run = next_run.replace(tzinfo=timezone.utc)
            else:
                from croniter import croniter

                cron_expr = str(params.get("cron", "")).strip()
                if not cron_expr:
                    return {"error": "cron is required for schedule_type=recurring"}
                if not croniter.is_valid(cron_expr):
                    return {"error": f"Invalid cron expression: {cron_expr}"}
                next_run = croniter(cron_expr, datetime.now(timezone.utc)).get_next(
                    datetime
                )
            new_job = AgentJobModel(
                user_id=job.user_id,
                goal=goal[:2000],
                job_type=job_type_param,
                schedule_type=schedule_type,
                schedule_cron=cron_expr,
                next_run_at=next_run,
                status="pending",
                config=config_param,
                parent_job_id=job.id,
            )
            ctx.db.add(new_job)
            await ctx.db.flush()
            return {
                "success": True,
                "data": {
                    "id": str(new_job.id),
                    "goal": new_job.goal,
                    "job_type": new_job.job_type,
                    "schedule_type": schedule_type,
                    "next_run_at": next_run.isoformat(),
                    "cron": cron_expr,
                },
            }
        except Exception as exc:
            return {"error": f"Failed to schedule job: {exc}"}

    async def _cancel_scheduled_job(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from uuid import UUID as _UUID

        from app.models.agent_job import AgentJob as AgentJobModel

        job = ctx.job
        cancel_job_id = str(params.get("job_id", "")).strip()
        if not cancel_job_id:
            return {"error": "job_id is required"}
        try:
            target = await ctx.db.get(AgentJobModel, _UUID(cancel_job_id))
            if not target:
                return {"error": f"Job not found: {cancel_job_id}"}
            if target.user_id != job.user_id:
                return {"error": "Not authorized to cancel this job"}
            if target.status == "running":
                return {"error": "Cannot cancel a currently running job"}
            target.status = "cancelled"
            target.next_run_at = None
            target.schedule_type = None
            await ctx.db.flush()
            return {
                "success": True,
                "data": {
                    "id": str(target.id),
                    "status": "cancelled",
                    "goal": target.goal,
                },
            }
        except ValueError:
            return {"error": f"Invalid job_id format: {cancel_job_id}"}
        except Exception as exc:
            return {"error": f"Failed to cancel job: {exc}"}

    return FunctionToolProvider(
        name="autonomous_scheduling_tools",
        modes={"autonomous"},
        handlers={
            "schedule_job": _schedule_job,
            "cancel_scheduled_job": _cancel_scheduled_job,
        },
    )


def build_autonomous_media_provider(executor: Any) -> FunctionToolProvider:
    """Media ingestion and analysis tools for AutonomousAgentExecutor."""

    async def _transcribe_document(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from uuid import UUID as _UUID

        from sqlalchemy.orm.attributes import flag_modified

        from app.models.document import Document as DocModel
        from app.tasks.transcription_tasks import transcribe_document as transcribe_task

        job = ctx.job
        doc_id = (params.get("document_id") or "").strip()
        if not doc_id:
            return {"error": "Missing required parameter: document_id"}
        try:
            doc_result = await ctx.db.execute(
                select(DocModel).where(
                    DocModel.id == _UUID(doc_id), DocModel.user_id == job.user_id
                )
            )
            doc = doc_result.scalar_one_or_none()
            if not doc:
                return {"error": f"Document {doc_id} not found"}
            if not doc.file_path:
                return {"error": "Document has no associated file"}
            meta = doc.extra_metadata or {}
            if meta.get("is_transcribed"):
                return {
                    "success": True,
                    "data": {
                        "document_id": doc_id,
                        "status": "already_transcribed",
                        "transcript_document_id": meta.get("transcript_document_id"),
                    },
                }
            if meta.get("is_transcribing"):
                return {
                    "success": True,
                    "data": {"document_id": doc_id, "status": "in_progress"},
                }
            ft = (doc.file_type or "").lower()
            from pathlib import Path as _Path

            ext = _Path(doc.file_path).suffix.lower()
            av_exts = {
                ".mp3",
                ".mp4",
                ".wav",
                ".m4a",
                ".ogg",
                ".flac",
                ".aac",
                ".avi",
                ".mkv",
                ".mov",
                ".webm",
                ".flv",
                ".wmv",
            }
            is_av = (
                any(ft.startswith(p) for p in ("audio/", "video/")) or ext in av_exts
            )
            if not is_av:
                return {"error": f"Document is not audio/video (type={ft}, ext={ext})"}
            doc.extra_metadata = {**meta, "is_transcribing": True}
            flag_modified(doc, "extra_metadata")
            await ctx.db.commit()
            celery_result = transcribe_task.delay(str(doc.id))
            return {
                "success": True,
                "data": {
                    "document_id": doc_id,
                    "status": "dispatched",
                    "task_id": celery_result.id,
                    "title": doc.title,
                },
                "findings": [
                    {
                        "type": "transcription_started",
                        "title": f"Transcription started for {doc.title}",
                        "document_id": doc_id,
                        "task_id": celery_result.id,
                    }
                ],
            }
        except Exception as exc:
            return {"error": f"Failed to transcribe document: {exc}"}

    async def _analyze_image(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        import base64
        from pathlib import Path as _Path
        from uuid import UUID as _UUID

        from app.core.config import settings as _settings
        from app.models.document import Document as DocModel
        from app.services.storage_service import storage_service as _storage

        job = ctx.job
        doc_id = (params.get("document_id") or "").strip()
        prompt_text = (
            params.get("prompt") or ""
        ).strip() or "Describe this image in detail, including any text, diagrams, charts, or notable visual elements."
        vision_model = (params.get("model") or "").strip()
        if not doc_id:
            return {"error": "Missing required parameter: document_id"}
        try:
            doc_result = await ctx.db.execute(
                select(DocModel).where(
                    DocModel.id == _UUID(doc_id), DocModel.user_id == job.user_id
                )
            )
            doc = doc_result.scalar_one_or_none()
            if not doc:
                return {"error": f"Document {doc_id} not found"}
            if not doc.file_path:
                return {"error": "Document has no associated file"}
            ft = (doc.file_type or "").lower()
            ext = _Path(doc.file_path).suffix.lower()
            image_types = {
                "image/png",
                "image/jpeg",
                "image/jpg",
                "image/gif",
                "image/webp",
                "image/bmp",
                "image/tiff",
            }
            image_exts = {
                ".png",
                ".jpg",
                ".jpeg",
                ".gif",
                ".webp",
                ".bmp",
                ".tiff",
                ".tif",
            }
            if ft not in image_types and ext not in image_exts:
                return {"error": f"Document is not an image (type={ft}, ext={ext})"}
            image_bytes = await _storage.get_file_content(doc.file_path)
            if not image_bytes:
                return {"error": "Failed to download image: empty content"}
            if len(image_bytes) > 20 * 1024 * 1024:
                return {
                    "error": f"Image too large ({len(image_bytes) // (1024*1024)}MB). Max 20MB."
                }
            if not vision_model:
                vision_model = getattr(_settings, "VISION_MODEL", "llava") or "llava"
            payload = {
                "model": vision_model,
                "prompt": prompt_text[:2000],
                "images": [base64.b64encode(image_bytes).decode("utf-8")],
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": 2048},
            }
            response = await executor.llm_service.client.post(
                f"{executor.llm_service.base_url}/api/generate",
                json=payload,
                timeout=120.0,
            )
            response.raise_for_status()
            analysis_text = (response.json().get("response") or "").strip()
            if not analysis_text:
                return {"error": "Vision model returned empty response"}
            return {
                "success": True,
                "data": {
                    "document_id": doc_id,
                    "title": doc.title,
                    "analysis": analysis_text[:5000],
                    "model": vision_model,
                    "prompt": prompt_text[:200],
                },
                "findings": [
                    {
                        "type": "image_analysis",
                        "title": f"Image analysis: {doc.title}",
                        "document_id": doc_id,
                        "content": analysis_text[:2000],
                        "model": vision_model,
                    }
                ],
            }
        except Exception as exc:
            error_msg = str(exc)
            if "404" in error_msg or "not found" in error_msg.lower():
                return {
                    "error": f"Vision model '{vision_model}' not available. Pull it with: ollama pull {vision_model}"
                }
            return {"error": f"Failed to analyze image: {error_msg}"}

    async def _get_media_info(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from pathlib import Path as _Path
        from uuid import UUID as _UUID

        from app.models.document import Document as DocModel

        job = ctx.job
        doc_id = (params.get("document_id") or "").strip()
        if not doc_id:
            return {"error": "Missing required parameter: document_id"}
        try:
            doc_result = await ctx.db.execute(
                select(DocModel).where(
                    DocModel.id == _UUID(doc_id), DocModel.user_id == job.user_id
                )
            )
            doc = doc_result.scalar_one_or_none()
            if not doc:
                return {"error": f"Document {doc_id} not found"}
            if not doc.file_path:
                return {"error": "Document has no associated file"}
            ft = (doc.file_type or "").lower()
            ext = _Path(doc.file_path).suffix.lower()
            media_info = {
                "document_id": doc_id,
                "title": doc.title,
                "file_type": doc.file_type,
                "file_size": doc.file_size,
            }
            av_exts = {
                ".mp3",
                ".mp4",
                ".wav",
                ".m4a",
                ".ogg",
                ".flac",
                ".aac",
                ".avi",
                ".mkv",
                ".mov",
                ".webm",
                ".flv",
                ".wmv",
            }
            image_exts = {
                ".png",
                ".jpg",
                ".jpeg",
                ".gif",
                ".webp",
                ".bmp",
                ".tiff",
                ".tif",
            }
            is_av = (
                any(ft.startswith(p) for p in ("audio/", "video/")) or ext in av_exts
            )
            is_image = ft.startswith("image/") or ext in image_exts
            if is_av:
                import json
                import os
                import subprocess
                import tempfile

                from app.services.storage_service import storage_service as _storage

                temp_path = None
                try:
                    tmp = tempfile.NamedTemporaryFile(
                        delete=False, suffix=ext or ".tmp"
                    )
                    temp_path = tmp.name
                    tmp.close()
                    await _storage.download_file(doc.file_path, temp_path)
                    probe_result = subprocess.run(
                        [
                            "ffprobe",
                            "-v",
                            "quiet",
                            "-print_format",
                            "json",
                            "-show_format",
                            "-show_streams",
                            temp_path,
                        ],
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    if probe_result.returncode == 0:
                        probe_data = json.loads(probe_result.stdout)
                        fmt = probe_data.get("format", {})
                        media_info["duration_seconds"] = float(fmt.get("duration", 0))
                        media_info["format_name"] = fmt.get("format_name")
                        media_info["bit_rate"] = int(fmt.get("bit_rate", 0) or 0)
                        for stream in probe_data.get("streams", []):
                            codec_type = stream.get("codec_type")
                            if codec_type == "video":
                                media_info["video_codec"] = stream.get("codec_name")
                                media_info["width"] = stream.get("width")
                                media_info["height"] = stream.get("height")
                                media_info["fps"] = stream.get("r_frame_rate")
                            elif codec_type == "audio":
                                media_info["audio_codec"] = stream.get("codec_name")
                                media_info["sample_rate"] = stream.get("sample_rate")
                                media_info["channels"] = stream.get("channels")
                    else:
                        media_info["probe_error"] = "ffprobe failed or not installed"
                    media_info["media_category"] = "audio_video"
                except Exception as probe_err:
                    media_info["probe_error"] = str(probe_err)
                    media_info["media_category"] = "audio_video"
                finally:
                    if temp_path and os.path.exists(temp_path):
                        os.unlink(temp_path)
            elif is_image:
                try:
                    from io import BytesIO

                    from PIL import Image

                    from app.services.storage_service import storage_service as _storage

                    image_bytes = await _storage.get_file_content(doc.file_path)
                    img = Image.open(BytesIO(image_bytes))
                    media_info["width"] = img.width
                    media_info["height"] = img.height
                    media_info["image_format"] = img.format
                    media_info["color_mode"] = img.mode
                    media_info["media_category"] = "image"
                except ImportError:
                    media_info["probe_error"] = "Pillow not installed"
                    media_info["media_category"] = "image"
                except Exception as img_err:
                    media_info["probe_error"] = str(img_err)
                    media_info["media_category"] = "image"
            else:
                media_info["media_category"] = "other"
            meta = doc.extra_metadata or {}
            media_info["is_transcribed"] = bool(meta.get("is_transcribed"))
            media_info["is_transcribing"] = bool(meta.get("is_transcribing"))
            media_info["transcript_document_id"] = meta.get("transcript_document_id")
            return {"success": True, "data": media_info}
        except Exception as exc:
            return {"error": f"Failed to get media info: {exc}"}

    return FunctionToolProvider(
        name="autonomous_media_tools",
        modes={"autonomous"},
        handlers={
            "transcribe_document": _transcribe_document,
            "analyze_image": _analyze_image,
            "get_media_info": _get_media_info,
        },
    )


def build_autonomous_snapshot_provider(executor: Any) -> FunctionToolProvider:
    """Workspace snapshot and drift-detection tools for AutonomousAgentExecutor."""

    async def _capture_snapshot(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        import re
        from datetime import datetime as _dt

        state = ctx.state if isinstance(ctx.state, dict) else {}
        snap_name = str(params.get("name", "")).strip()[:100]
        extra_keys = params.get("keys", [])
        if not snap_name:
            return {"error": "Missing required parameter: name"}
        if not re.match(r"^[a-zA-Z0-9_\-]+$", snap_name):
            return {
                "error": "Snapshot name must be alphanumeric with underscores/hyphens only"
            }
        doc_ids = {
            str(f.get("document_id") or f.get("source_id"))
            for f in state.get("findings", [])
            if f.get("document_id") or f.get("source_id")
        }
        snapshot = {
            "iteration": state.get("iteration", 0),
            "timestamp": _dt.utcnow().isoformat(),
            "findings_count": len(state.get("findings", [])),
            "actions_count": len(state.get("actions_taken", [])),
            "goal_progress": state.get("goal_progress", 0),
            "documents_found": len(doc_ids),
            "tool_stats": dict(state.get("tool_stats", {})),
            "stalled_iterations": state.get("stalled_iterations", 0),
            "artifacts_count": len(state.get("artifacts", [])),
            "formatted_outputs_count": len(state.get("formatted_outputs", [])),
            "focus_directive": state.get("focus_directive", ""),
            "skill_profile_role": (state.get("skill_profile") or {}).get("role", ""),
        }
        if isinstance(extra_keys, list) and extra_keys:
            custom = {}
            for key in extra_keys[:20]:
                key = str(key).strip()
                if key and key != "workspace_snapshots":
                    val = state.get(key)
                    if val is not None:
                        custom[key] = str(val)[:5000]
            if custom:
                snapshot["custom_keys"] = custom
        snapshots = state.setdefault("workspace_snapshots", {})
        if len(snapshots) >= 20 and snap_name not in snapshots:
            oldest = min(snapshots, key=lambda n: snapshots[n].get("iteration", 0))
            del snapshots[oldest]
        snapshots[snap_name] = snapshot
        return {
            "success": True,
            "data": {
                "name": snap_name,
                "iteration": snapshot["iteration"],
                "findings_count": snapshot["findings_count"],
                "actions_count": snapshot["actions_count"],
                "goal_progress": snapshot["goal_progress"],
                "documents_found": snapshot["documents_found"],
                "total_snapshots": len(snapshots),
            },
        }

    async def _compare_snapshots(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        state = ctx.state if isinstance(ctx.state, dict) else {}
        name_a = str(params.get("snapshot_a", "")).strip()
        name_b = str(params.get("snapshot_b", "")).strip()
        if not name_a or not name_b:
            return {"error": "Both snapshot_a and snapshot_b are required"}
        snapshots = state.get("workspace_snapshots", {})
        snap_a = snapshots.get(name_a)
        snap_b = snapshots.get(name_b)
        if not snap_a:
            return {"error": f"Snapshot '{name_a}' not found"}
        if not snap_b:
            return {"error": f"Snapshot '{name_b}' not found"}
        numeric_keys = [
            "findings_count",
            "actions_count",
            "goal_progress",
            "documents_found",
            "stalled_iterations",
            "artifacts_count",
            "formatted_outputs_count",
        ]
        diff = {}
        for key in numeric_keys:
            a_val = float(snap_a.get(key, 0) or 0)
            b_val = float(snap_b.get(key, 0) or 0)
            delta = b_val - a_val
            diff[key] = {
                "before": a_val,
                "after": b_val,
                "delta": delta,
                "direction": "increased"
                if delta > 0
                else ("decreased" if delta < 0 else "unchanged"),
            }
        for key in ["focus_directive", "skill_profile_role"]:
            a_val = str(snap_a.get(key, ""))
            b_val = str(snap_b.get(key, ""))
            diff[key] = {"before": a_val, "after": b_val, "changed": a_val != b_val}
        stats_a = snap_a.get("tool_stats", {})
        stats_b = snap_b.get("tool_stats", {})
        tools_added = set(stats_b.keys()) - set(stats_a.keys())
        tools_removed = set(stats_a.keys()) - set(stats_b.keys())
        diff["tool_stats"] = {
            "tools_added": sorted(list(tools_added)),
            "tools_removed": sorted(list(tools_removed)),
            "total_before": len(stats_a),
            "total_after": len(stats_b),
        }
        iter_a = snap_a.get("iteration", "?")
        iter_b = snap_b.get("iteration", "?")
        summary_parts = [f"Between iteration {iter_a} and {iter_b}:"]
        if diff["findings_count"]["delta"]:
            summary_parts.append(
                f"findings {'+' if diff['findings_count']['delta'] > 0 else ''}{int(diff['findings_count']['delta'])}"
            )
        if diff["goal_progress"]["delta"]:
            summary_parts.append(
                f"progress {'+' if diff['goal_progress']['delta'] > 0 else ''}{int(diff['goal_progress']['delta'])}%"
            )
        if tools_added:
            summary_parts.append(f"{len(tools_added)} new tools used")
        return {
            "success": True,
            "data": {
                "diff": diff,
                "summary": " ".join(summary_parts),
                "snapshot_a_iteration": iter_a,
                "snapshot_b_iteration": iter_b,
            },
        }

    async def _detect_drift(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        state = ctx.state if isinstance(ctx.state, dict) else {}
        baseline_name = str(params.get("baseline", "")).strip()
        custom_thresholds = params.get("thresholds", {})
        if not baseline_name:
            return {"error": "Missing required parameter: baseline"}
        snapshots = state.get("workspace_snapshots", {})
        baseline = snapshots.get(baseline_name)
        if not baseline:
            return {"error": f"Baseline snapshot '{baseline_name}' not found"}
        doc_ids = {
            str(f.get("document_id") or f.get("source_id"))
            for f in state.get("findings", [])
            if f.get("document_id") or f.get("source_id")
        }
        current = {
            "iteration": state.get("iteration", 0),
            "findings_count": len(state.get("findings", [])),
            "actions_count": len(state.get("actions_taken", [])),
            "goal_progress": state.get("goal_progress", 0),
            "documents_found": len(doc_ids),
            "stalled_iterations": state.get("stalled_iterations", 0),
            "artifacts_count": len(state.get("artifacts", [])),
        }
        thresholds = {
            "stalled_iterations": 2,
            "goal_progress_drop": 0,
            "findings_stale_iterations": 5,
            "tool_failure_rate": 0.5,
        }
        if isinstance(custom_thresholds, dict):
            for key, val in custom_thresholds.items():
                if key in thresholds:
                    try:
                        thresholds[key] = float(val)
                    except (TypeError, ValueError):
                        pass
        iterations_elapsed = current["iteration"] - baseline.get("iteration", 0)
        alerts = []
        if current["stalled_iterations"] > thresholds["stalled_iterations"]:
            alerts.append(
                {
                    "metric": "stalled_iterations",
                    "baseline_value": baseline.get("stalled_iterations", 0),
                    "current_value": current["stalled_iterations"],
                    "severity": "warning",
                    "message": f"Agent has stalled for {current['stalled_iterations']} iterations",
                }
            )
        progress_drop = baseline.get("goal_progress", 0) - current["goal_progress"]
        if progress_drop > thresholds["goal_progress_drop"]:
            alerts.append(
                {
                    "metric": "goal_progress",
                    "baseline_value": baseline.get("goal_progress", 0),
                    "current_value": current["goal_progress"],
                    "severity": "critical" if progress_drop > 20 else "warning",
                    "message": f"Goal progress dropped by {progress_drop}% since baseline",
                }
            )
        findings_delta = current["findings_count"] - baseline.get("findings_count", 0)
        if (
            findings_delta == 0
            and iterations_elapsed >= thresholds["findings_stale_iterations"]
        ):
            alerts.append(
                {
                    "metric": "findings_count",
                    "baseline_value": baseline.get("findings_count", 0),
                    "current_value": current["findings_count"],
                    "severity": "warning",
                    "message": f"No new findings in {iterations_elapsed} iterations",
                }
            )
        for tool, stats in state.get("tool_stats", {}).items():
            if isinstance(stats, dict):
                total = (stats.get("success", 0) or 0) + (stats.get("failure", 0) or 0)
                if total >= 3:
                    fail_rate = (stats.get("failure", 0) or 0) / total
                    if fail_rate > thresholds["tool_failure_rate"]:
                        alerts.append(
                            {
                                "metric": f"tool_failure:{tool}",
                                "baseline_value": 0,
                                "current_value": round(fail_rate, 2),
                                "severity": "warning",
                                "message": f"Tool '{tool}' failure rate is {round(fail_rate * 100)}%",
                            }
                        )
        if not alerts:
            alerts.append(
                {
                    "metric": "overall",
                    "baseline_value": baseline.get("goal_progress", 0),
                    "current_value": current["goal_progress"],
                    "severity": "info",
                    "message": f"No drift detected after {iterations_elapsed} iterations",
                }
            )
        severity_counts = {}
        for alert in alerts:
            severity_counts[alert["severity"]] = (
                severity_counts.get(alert["severity"], 0) + 1
            )
        summary = f"{len(alerts)} alert(s) after {iterations_elapsed} iterations"
        if severity_counts.get("critical"):
            summary += f" ({severity_counts['critical']} critical)"
        elif severity_counts.get("warning"):
            summary += f" ({severity_counts['warning']} warnings)"
        payload = {
            "success": True,
            "data": {
                "alerts": alerts,
                "metrics_compared": len(current),
                "iterations_elapsed": iterations_elapsed,
                "summary": summary,
            },
        }
        if any(a["severity"] in ("warning", "critical") for a in alerts):
            payload["findings"] = [
                {
                    "type": "drift_detected",
                    "title": f"Drift detected: {summary}",
                    "baseline": baseline_name,
                    "alert_count": len(alerts),
                    "severity_counts": severity_counts,
                }
            ]
        return payload

    return FunctionToolProvider(
        name="autonomous_snapshot_tools",
        modes={"autonomous"},
        handlers={
            "capture_snapshot": _capture_snapshot,
            "compare_snapshots": _compare_snapshots,
            "detect_drift": _detect_drift,
        },
    )


def build_autonomous_project_bootstrap_provider(executor: Any) -> FunctionToolProvider:
    """Project bootstrap tool for AutonomousAgentExecutor."""

    async def _project_bootstrap(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from app.services.project_profile_service import build_project_profile

        job = ctx.job
        state = ctx.state if isinstance(ctx.state, dict) else {}
        source_id = str(
            params.get("source_id") or ""
        ).strip() or executor._resolve_default_source_scope(job)
        max_files = int(params.get("max_files", 400) or 400)
        profile = await build_project_profile(
            job,
            ctx.db,
            source_id=source_id,
            max_files=max_files,
        )
        if not profile.get("sampled_files"):
            return {"error": "No repository-like files found to build project profile."}
        state["project_profile"] = profile
        return {
            "success": True,
            "data": profile,
            "findings": [
                {
                    "type": "project_profile",
                    "title": "Project bootstrap profile generated",
                    "source_id": profile.get("source_id"),
                    "detected_stack": profile.get("detected_stack", []),
                    "sampled_files": profile.get("sampled_files", 0),
                }
            ],
            "artifacts": [
                {
                    "type": "project_profile",
                    "source_id": profile.get("source_id"),
                    "sampled_files": profile.get("sampled_files", 0),
                    "detected_stack": profile.get("detected_stack", []),
                }
            ],
        }

    return FunctionToolProvider(
        name="autonomous_project_bootstrap_tools",
        modes={"autonomous"},
        handlers={
            "project_bootstrap": _project_bootstrap,
        },
    )


def build_autonomous_document_provider(executor: Any) -> FunctionToolProvider:
    """Document-domain tools for AutonomousAgentExecutor."""

    async def _search_documents(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        db = ctx.db
        job = ctx.job
        query = params.get("query", job.goal[:100] if job else "")
        limit = params.get("limit", 10)
        source_id = str(params.get("source_id") or "").strip() or None
        page_size = max(1, min(int(limit or 10), 100))
        results, _total, _took = await executor.search_service.search(
            query=query,
            mode="smart",
            page=1,
            page_size=page_size,
            source_id=source_id,
            db=db,
        )
        return {
            "success": True,
            "data": results,
            "findings": [
                {
                    "type": "document",
                    "title": row.get("title"),
                    "id": row.get("id"),
                    "score": row.get("relevance_score", row.get("score")),
                    "source": row.get("source"),
                    "source_id": row.get("source_id"),
                }
                for row in results[:10]
            ],
        }

    async def _search_with_filters(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        query = params.get("query", "")
        limit = params.get("limit", 20)
        source_id = str(params.get("source_id") or "").strip() or None
        file_type = str(params.get("file_type") or "").strip() or None
        mode = str(params.get("mode") or "smart").strip().lower() or "smart"
        page_size = max(1, min(int(limit or 20), 100))
        results, _total, _took = await executor.search_service.search(
            query=query,
            mode=mode,
            page=1,
            page_size=page_size,
            source_id=source_id,
            file_type=file_type,
            db=ctx.db,
        )
        return {"success": True, "data": results}

    async def _web_scrape(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from urllib.parse import urlparse

        from app.models.document import DocumentSource
        from app.services.web_scraper_service import WebScraperService

        url = params.get("url", "")
        if not url:
            return {"error": "Missing required parameter: url"}

        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
        allow_private = False
        if host:
            src_res = await ctx.db.execute(
                select(DocumentSource).where(
                    DocumentSource.source_type == "web",
                    DocumentSource.is_active.is_(True),
                )
            )
            sources = src_res.scalars().all()

            def host_matches(allowed: str) -> bool:
                allowed = (allowed or "").strip().lower()
                if not allowed:
                    return False
                return host == allowed or host.endswith("." + allowed)

            for source in sources:
                cfg = source.config or {}
                for domain in cfg.get("allowed_domains") or []:
                    if host_matches(domain):
                        allow_private = True
                        break
                if allow_private:
                    break
                for base in cfg.get("base_urls") or []:
                    try:
                        base_host = (urlparse(str(base)).hostname or "").lower()
                    except Exception:
                        base_host = ""
                    if base_host and host_matches(base_host):
                        allow_private = True
                        break
                if allow_private:
                    break

        scraper = WebScraperService(enforce_network_safety=True)
        try:
            scrape_result = await scraper.scrape(
                url,
                follow_links=bool(params.get("follow_links", False)),
                max_pages=int(params.get("max_pages", 1)),
                max_depth=int(params.get("max_depth", 0)),
                same_domain_only=bool(params.get("same_domain_only", True)),
                include_links=bool(params.get("include_links", True)),
                allow_private_networks=allow_private,
                max_content_chars=int(params.get("max_content_chars", 50_000)),
            )
            payload: Dict[str, Any] = {"success": True, "data": scrape_result}
            pages = (
                scrape_result.get("pages", [])
                if isinstance(scrape_result, dict)
                else []
            )
            if pages:
                payload["findings"] = [
                    {
                        "type": "web_page",
                        "title": page.get("title"),
                        "url": page.get("url"),
                        "content_preview": (page.get("content") or "")[:500],
                    }
                    for page in pages[:5]
                ]
            return payload
        finally:
            await scraper.aclose()

    async def _ingest_url(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from app.models.user import User
        from app.services.url_ingestion_service import UrlIngestionService

        job = ctx.job
        url = (params.get("url") or "").strip()
        if not url:
            return {"error": "Missing required parameter: url"}
        user_res = await ctx.db.execute(select(User).where(User.id == job.user_id))
        user = user_res.scalar_one_or_none()
        if not user:
            return {"error": "User not found"}
        service = UrlIngestionService()
        ingest = await service.ingest_url(
            db=ctx.db,
            user=user,
            url=url,
            title=params.get("title"),
            tags=params.get("tags"),
            follow_links=bool(params.get("follow_links", False)),
            max_pages=int(params.get("max_pages", 1)),
            max_depth=int(params.get("max_depth", 0)),
            same_domain_only=bool(params.get("same_domain_only", True)),
            one_document_per_page=bool(params.get("one_document_per_page", False)),
            allow_private_networks=bool(params.get("allow_private_networks", False)),
            max_content_chars=int(params.get("max_content_chars", 50_000)),
        )
        if ingest.get("error"):
            return {"error": ingest["error"]}
        return {"success": True, "data": ingest}

    async def _get_document_details(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from uuid import UUID

        from app.models.document import Document

        doc_id = params.get("document_id")
        if not doc_id:
            return {"error": "Missing required parameter: document_id"}
        doc_result = await ctx.db.execute(
            select(Document).where(Document.id == UUID(doc_id))
        )
        doc = doc_result.scalar_one_or_none()
        if not doc:
            return {"error": "Document not found"}
        return {
            "success": True,
            "data": {
                "id": str(doc.id),
                "title": doc.title,
                "source": doc.source,
                "file_type": doc.file_type,
                "author": doc.author,
                "summary": doc.summary,
                "has_content": bool(doc.content),
            },
        }

    async def _read_document_content(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from uuid import UUID

        from app.models.document import Document

        doc_id = params.get("document_id")
        max_length = params.get("max_length", 10000)
        if not doc_id:
            return {"error": "Missing required parameter: document_id"}
        doc_result = await ctx.db.execute(
            select(Document).where(Document.id == UUID(doc_id))
        )
        doc = doc_result.scalar_one_or_none()
        if not doc or not doc.content:
            return {"error": "Document not found or has no content"}
        return {
            "success": True,
            "data": {
                "id": str(doc.id),
                "title": doc.title,
                "content": doc.content[:max_length],
                "truncated": len(doc.content) > max_length,
            },
        }

    async def _summarize_document(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from uuid import UUID

        from app.models.document import Document

        doc_id = params.get("document_id")
        if not doc_id:
            return {"error": "Missing required parameter: document_id"}
        doc_result = await ctx.db.execute(
            select(Document).where(Document.id == UUID(doc_id))
        )
        doc = doc_result.scalar_one_or_none()
        if not doc:
            return {"error": "Document not found"}
        if doc.summary:
            return {
                "success": True,
                "data": {"summary": doc.summary},
                "findings": [
                    {
                        "type": "summary",
                        "document_id": doc_id,
                        "content": doc.summary[:500],
                        "source_id": str(doc.source_id)
                        if getattr(doc, "source_id", None)
                        else None,
                    }
                ],
            }
        return {"success": True, "data": {"status": "summarization_queued"}}

    async def _find_similar_documents(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from uuid import UUID

        from app.models.document import Document

        doc_id = params.get("document_id")
        limit = params.get("limit", 5)
        if not doc_id:
            return {"error": "Missing required parameter: document_id"}
        doc_result = await ctx.db.execute(
            select(Document).where(Document.id == UUID(doc_id))
        )
        doc = doc_result.scalar_one_or_none()
        if not doc or not doc.content:
            return {"error": "Document not found or has no content"}
        similar, _total, _took = await executor.search_service.search(
            query=doc.title + " " + (doc.summary or doc.content[:500]),
            mode="smart",
            page=1,
            page_size=max(1, min(int(limit or 5) + 1, 100)),
            source_id=str(doc.source_id) if getattr(doc, "source_id", None) else None,
            db=ctx.db,
        )
        similar = [row for row in similar if str(row.get("id")) != doc_id][:limit]
        return {
            "success": True,
            "data": similar,
            "findings": [
                {
                    "type": "similar_document",
                    "id": row.get("id"),
                    "title": row.get("title"),
                    "source_id": row.get("source_id"),
                }
                for row in similar
            ],
        }

    async def _get_knowledge_base_stats(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from collections import Counter
        from uuid import UUID

        from sqlalchemy import desc, func

        from app.models.document import Document, DocumentSource

        limit = int(params.get("recent_limit", 25) or 25)
        limit = max(1, min(limit, 100))
        source_id_raw = str(params.get("source_id") or "").strip()
        source_uuid = None
        if source_id_raw:
            try:
                source_uuid = UUID(source_id_raw)
            except Exception:
                source_uuid = None

        docs_count_query = select(func.count()).select_from(Document)
        if source_uuid:
            docs_count_query = docs_count_query.where(Document.source_id == source_uuid)
        total_docs = int((await ctx.db.execute(docs_count_query)).scalar() or 0)
        total_sources = (
            1
            if source_uuid
            else int(
                (
                    await ctx.db.execute(
                        select(func.count()).select_from(DocumentSource)
                    )
                ).scalar()
                or 0
            )
        )

        recent_query = (
            select(Document.id, Document.title, Document.created_at, Document.tags)
            .order_by(desc(Document.created_at))
            .limit(limit)
        )
        if source_uuid:
            recent_query = recent_query.where(Document.source_id == source_uuid)
        rows = (await ctx.db.execute(recent_query)).all()
        tag_counter: Counter[str] = Counter()
        for _, _, _, tags in rows:
            if isinstance(tags, list):
                tag_counter.update([str(tag).lower() for tag in tags if tag])

        return {
            "success": True,
            "data": {
                "documents_total": total_docs,
                "sources_total": total_sources,
                "source_id": str(source_uuid) if source_uuid else None,
                "recent_documents": [
                    {"id": str(doc_id), "title": title, "created_at": str(created_at)}
                    for doc_id, title, created_at, _ in rows
                ],
                "top_tags": [
                    {"tag": tag, "count": count}
                    for tag, count in tag_counter.most_common(10)
                ],
            },
        }

    async def _create_document_from_text(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        import hashlib
        import uuid

        from app.models.document import Document

        job = ctx.job
        title = (params.get("title") or "").strip()
        content = (params.get("content") or "").strip()
        tags = params.get("tags") or []
        source_scope_id = str(params.get("source_id") or "").strip() or None

        if not title:
            return {"error": "Title is required"}
        if not content:
            return {"error": "Content is required"}

        notes_source = (
            await executor.document_service._get_or_create_agent_notes_source(ctx.db)
        )
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        doc = Document(
            title=title,
            content=content,
            content_hash=content_hash,
            url=None,
            file_path=None,
            file_type="text/plain",
            file_size=len(content.encode("utf-8")),
            source_id=notes_source.id,
            source_identifier=f"agent_note:{uuid.uuid4().hex}",
            author=None,
            tags=tags if isinstance(tags, list) else None,
            extra_metadata={
                "origin": "autonomous_job",
                "job_id": str(job.id),
                "job_type": job.job_type,
                "source_scope_id": source_scope_id,
            },
            is_processed=False,
        )
        ctx.db.add(doc)
        await ctx.db.commit()
        await ctx.db.refresh(doc)

        try:
            await executor.document_service.reprocess_document(
                doc.id, ctx.db, user_id=job.user_id
            )
        except Exception:
            pass

        return {
            "success": True,
            "data": {
                "document_id": str(doc.id),
                "title": doc.title,
                "source_scope_id": source_scope_id,
            },
            "artifacts": [
                {
                    "type": "document",
                    "id": str(doc.id),
                    "title": doc.title,
                    "source_scope_id": source_scope_id,
                }
            ],
        }

    async def _list_documents_by_tag(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from app.models.document import Document

        tags_param = params.get("tags")
        if not tags_param or not isinstance(tags_param, list) or not tags_param:
            return {"error": "tags is required and must be a non-empty array"}
        match_all = bool(params.get("match_all", False))
        limit = min(int(params.get("limit", 20) or 20), 100)
        tags_set = set(str(tag).strip() for tag in tags_param if str(tag).strip())
        stmt = select(Document).where(Document.tags.isnot(None)).limit(500)
        docs = (await ctx.db.execute(stmt)).scalars().all()

        if match_all:
            matched = [doc for doc in docs if tags_set.issubset(set(doc.tags or []))]
        else:
            matched = [doc for doc in docs if tags_set & set(doc.tags or [])]
        matched = matched[:limit]
        return {
            "success": True,
            "data": {
                "tags": list(tags_set),
                "match_all": match_all,
                "documents": [
                    {
                        "id": str(doc.id),
                        "title": doc.title,
                        "tags": doc.tags or [],
                        "file_type": doc.file_type,
                        "summary": (doc.summary or "")[:200],
                        "created_at": doc.created_at.isoformat()
                        if doc.created_at
                        else None,
                    }
                    for doc in matched
                ],
                "count": len(matched),
            },
        }

    async def _merge_documents(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        import hashlib
        import uuid
        from uuid import UUID as _UUID

        from app.models.document import Document

        job = ctx.job
        doc_ids = params.get("document_ids")
        merge_title = str(params.get("title", "")).strip()
        if not doc_ids or not isinstance(doc_ids, list) or not doc_ids:
            return {"error": "document_ids is required and must be a non-empty array"}
        if not merge_title:
            return {"error": "title is required"}

        separator = str(params.get("separator", "\n\n---\n\n"))
        merge_tags = params.get("tags") if isinstance(params.get("tags"), list) else []
        sections = []
        source_ids = []
        for doc_id_str in doc_ids[:20]:
            try:
                doc_obj = await ctx.db.get(Document, _UUID(str(doc_id_str).strip()))
                if doc_obj and doc_obj.content:
                    sections.append(f"# {doc_obj.title}\n\n{doc_obj.content}")
                    source_ids.append(str(doc_obj.id))
            except Exception:
                continue
        if not sections:
            return {"error": "No valid documents with content found"}

        merged_content = separator.join(sections)
        if len(merged_content.encode("utf-8")) > 2_000_000:
            return {"error": "Merged content exceeds 2MB limit"}

        content_hash = hashlib.sha256(merged_content.encode("utf-8")).hexdigest()
        notes_source = (
            await executor.document_service._get_or_create_agent_notes_source(ctx.db)
        )
        new_doc = Document(
            title=merge_title,
            content=merged_content,
            content_hash=content_hash,
            file_type="text/plain",
            file_size=len(merged_content.encode("utf-8")),
            source_id=notes_source.id,
            source_identifier=f"agent_merge:{uuid.uuid4().hex}",
            tags=merge_tags,
            extra_metadata={
                "origin": "agent_merge",
                "source_document_ids": source_ids,
                "job_id": str(job.id),
            },
            is_processed=False,
        )
        ctx.db.add(new_doc)
        await ctx.db.commit()
        await ctx.db.refresh(new_doc)

        try:
            await executor.document_service.reprocess_document(
                new_doc.id, ctx.db, user_id=job.user_id
            )
        except Exception:
            pass

        return {
            "success": True,
            "data": {
                "document_id": str(new_doc.id),
                "title": new_doc.title,
                "source_count": len(source_ids),
                "content_length": len(merged_content),
            },
            "artifacts": [
                {"type": "document", "id": str(new_doc.id), "title": new_doc.title}
            ],
        }

    return FunctionToolProvider(
        name="autonomous_document_tools",
        modes={"autonomous"},
        handlers={
            "search_documents": _search_documents,
            "search_with_filters": _search_with_filters,
            "web_scrape": _web_scrape,
            "ingest_url": _ingest_url,
            "get_document_details": _get_document_details,
            "read_document_content": _read_document_content,
            "summarize_document": _summarize_document,
            "find_similar_documents": _find_similar_documents,
            "get_knowledge_base_stats": _get_knowledge_base_stats,
            "create_document_from_text": _create_document_from_text,
            "list_documents_by_tag": _list_documents_by_tag,
            "merge_documents": _merge_documents,
        },
    )
