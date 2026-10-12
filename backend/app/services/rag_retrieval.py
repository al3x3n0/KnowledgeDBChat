"""Retrieving what an answer is to be written from.

This is the retrieval half of a chat turn: rewrite the question, search (with
several phrasings when expansion is on), drop what is not relevant, pick the
passages that fit the context budget, and add what the knowledge graph says
about them. It was inline in `ChatService.generate_response`, so the chat
page was the only place that had it: the agent's `answer_question` tool and
the MCP chat tool each searched once and used whatever came back.

`retrieve` is that block moved, not rewritten, so the chat page behaves as it
did. It returns everything the turn went on to use, including the trace
payload that makes a retrieval inspectable afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.services.context_manager import ContextManager
from app.services.query_processor import QueryProcessor


@dataclass
class Retrieved:
    #: The question as it was searched for.
    search_query: str
    #: The query processor's reading of it (intent, key terms).
    processed_query: Dict[str, Any]
    #: The passages selected, best first, as the vector store returns them.
    results: List[Dict[str, Any]]
    #: Those passages as the text handed to the model.
    context: str
    #: What the knowledge graph adds about them; "" when nothing.
    kg_context: str = ""
    context_metrics: Dict[str, Any] = field(default_factory=dict)
    #: What happened, for the retrieval trace.
    trace: Dict[str, Any] = field(default_factory=dict)


async def retrieve(
    query: str,
    *,
    db: AsyncSession,
    vector_store: Any,
    query_processor: Any,
    context_manager: Any,
    llm_service: Any,
    user_settings: Optional[Any] = None,
    document_ids: Optional[List[str]] = None,
    max_results: Optional[int] = None,
) -> Retrieved:
    """Find and prepare the passages a question is answered from.

    `max_results` is how many passages the answer may draw on; the chat
    page's own `MAX_SEARCH_RESULTS` when not given.

    `document_ids` restricts the search to those documents, in the store
    itself: filtering the best passages from anywhere afterwards lets other
    documents crowd out the ones asked about.
    """
    max_results = int(max_results or settings.MAX_SEARCH_RESULTS)
    # Process query
    processed_query = query_processor.process_query(
        query,
        expand=settings.RAG_QUERY_EXPANSION_ENABLED,
        rewrite=True,
        normalize=True,
    )

    # Use processed query for search
    search_query = processed_query.get("processed", query)

    # Ensure vector store is initialized
    await vector_store.initialize(background=True)

    # Generate multi-queries if enabled for better recall
    all_search_results = []
    retrieval_trace_payload: dict[str, Any] = {
        "original_query": query,
        "processed_query": search_query,
        "query_processor": {
            "intent": processed_query.get("intent"),
            "key_terms": processed_query.get("key_terms"),
        },
    }
    if settings.RAG_QUERY_EXPANSION_ENABLED:
        try:
            query_variations = await query_processor.generate_multi_queries(
                search_query,
                llm_service=llm_service,
                num_queries=3,
                user_settings=user_settings,
            )

            # Search with all variations
            traces: list[dict[str, Any]] = []
            for q_var in query_variations:
                results, one_trace = await vector_store.search_with_trace(
                    query=q_var,
                    limit=max_results * 2,
                    document_ids=document_ids,
                )
                traces.append({"query": q_var, "trace": one_trace})
                all_search_results.extend(results)

            retrieval_trace_payload["multi_query"] = {
                "queries": query_variations,
                "traces": traces,
            }

            # Deduplicate by ID and merge scores
            seen_ids = {}
            for result in all_search_results:
                result_id = result.get("id")
                if result_id in seen_ids:
                    # Merge scores (take max)
                    seen_ids[result_id]["score"] = max(
                        seen_ids[result_id]["score"], result.get("score", 0.0)
                    )
                else:
                    seen_ids[result_id] = result

            search_results = list(seen_ids.values())
            # Sort by score
            search_results = sorted(
                search_results, key=lambda x: x.get("score", 0.0), reverse=True
            )
        except Exception as e:
            logger.warning(f"Multi-query generation failed: {e}, using single query")
            (
                search_results,
                one_trace,
            ) = await vector_store.search_with_trace(
                query=search_query,
                limit=max_results * 2,
                document_ids=document_ids,
            )
            retrieval_trace_payload["single_query"] = {
                "query": search_query,
                "trace": one_trace,
            }
    else:
        # Single query search
        search_results, one_trace = await vector_store.search_with_trace(
            query=search_query,
            limit=max_results * 2,  # Get more for filtering/reranking
            document_ids=document_ids,
        )
        retrieval_trace_payload["single_query"] = {
            "query": search_query,
            "trace": one_trace,
        }

    # Filter by relevance
    search_results = context_manager.filter_by_relevance(search_results)

    # Select most relevant parts
    search_results = context_manager.select_relevant_parts(
        search_results,
        query=search_query,
        max_results=max_results,
    )

    retrieval_trace_payload["selected_results"] = [
        {
            "id": r.get("id"),
            "score": r.get("score"),
            "document_id": (r.get("metadata") or {}).get("document_id"),
            "chunk_id": (r.get("metadata") or {}).get("chunk_id"),
            "title": (r.get("metadata") or {}).get("title"),
            "source": (r.get("metadata") or {}).get("source")
            or (r.get("metadata") or {}).get("source_type"),
        }
        for r in (search_results or [])[:max_results]
    ]

    # Build context with token management
    context = await context_manager.compress_context(
        search_results, llm_service=llm_service
    )

    # Get context metrics for logging
    context_metrics = context_manager.get_context_metrics(search_results)
    retrieval_trace_payload["context_metrics"] = context_metrics

    # Inject Knowledge Graph context if enabled
    kg_context = ""
    if settings.RAG_KG_CONTEXT_ENABLED:
        try:
            from app.services.knowledge_graph_service import KnowledgeGraphService

            kg_service = KnowledgeGraphService()

            # Prefer a grounded "context pack" from the retrieved chunks/docs.
            kg_pack = await kg_service.build_chat_context_pack(
                db=db,
                search_results=search_results,
                max_entities=settings.RAG_KG_MAX_ENTITIES,
                max_relationships=settings.RAG_KG_MAX_RELATIONSHIPS,
            )
            if kg_pack.get("kg_context"):
                kg_context = str(kg_pack["kg_context"])
                # Keep the trace compact: it's JSON, but can grow quickly.
                retrieval_trace_payload["kg_context_pack"] = {
                    "stats": kg_pack.get("stats"),
                    "entities": (kg_pack.get("entities") or [])[
                        : settings.RAG_KG_MAX_ENTITIES
                    ],
                    "relationships": (kg_pack.get("relationships") or [])[
                        : settings.RAG_KG_MAX_RELATIONSHIPS
                    ],
                    # Useful for debugging / UI inspection; keep bounded.
                    "kg_context": str(kg_context)[:8000],
                }

            # If the pack couldn't be built (e.g., no mentions in retrieved scope), fall back.
            if not kg_context:
                # Extract potential entity names from query and search results
                entity_names = context_manager.extract_entity_names_from_results(
                    search_results
                )

                # Also add key terms from the processed query
                key_terms = processed_query.get("key_terms", [])
                for term in key_terms:
                    if len(term) > 2 and term not in entity_names:
                        entity_names.append(term)

                # Search for matching entities in KG
                if entity_names:
                    entities = await kg_service.search_entities_by_names(
                        names=entity_names[:20],  # Limit search candidates
                        db=db,
                        limit=settings.RAG_KG_MAX_ENTITIES,
                    )

                    if entities:
                        # Get relationships between found entities
                        entity_ids = [e.id for e in entities]
                        kg_data = await kg_service.get_entity_context(
                            entity_ids=entity_ids,
                            db=db,
                            max_relationships=settings.RAG_KG_MAX_RELATIONSHIPS,
                        )

                        # Build KG context string
                        kg_context = context_manager.build_kg_context(
                            entities=kg_data["entities"],
                            relationships=kg_data["relationships"],
                        )
                        logger.debug(
                            f"KG context injected: {len(kg_data['entities'])} entities, {len(kg_data['relationships'])} relationships"
                        )
                        retrieval_trace_payload["kg_context_fallback"] = {
                            "entity_names": entity_names[:20],
                            "entities": len(kg_data.get("entities") or []),
                            "relationships": len(kg_data.get("relationships") or []),
                        }
        except Exception as e:
            logger.warning(f"KG context injection failed: {e}")
            # Continue without KG context - don't break the chat flow

    return Retrieved(
        search_query=search_query,
        processed_query=processed_query,
        results=search_results,
        context=context,
        kg_context=kg_context,
        context_metrics=context_metrics,
        trace=retrieval_trace_payload,
    )


# Neither holds state between calls.
_QUERY_PROCESSOR = QueryProcessor()
_CONTEXT_MANAGER = ContextManager()


async def retrieve_for_answer(
    query: str,
    *,
    db: AsyncSession,
    vector_store: Any,
    llm_service: Any,
    user_settings: Optional[Any] = None,
    document_ids: Optional[List[str]] = None,
    max_results: Optional[int] = None,
) -> Retrieved:
    """`retrieve`, for a caller that has no query processor of its own."""
    return await retrieve(
        query,
        document_ids=document_ids,
        max_results=max_results,
        db=db,
        vector_store=vector_store,
        query_processor=_QUERY_PROCESSOR,
        context_manager=_CONTEXT_MANAGER,
        llm_service=llm_service,
        user_settings=user_settings,
    )


def cited_sources(
    results: List[Dict[str, Any]], *, limit: int, excerpt_chars: int = 200
) -> List[Dict[str, Any]]:
    """One entry per document the passages came from, best first."""
    sources: List[Dict[str, Any]] = []
    seen = set()
    for result in results:
        metadata = result.get("metadata") or {}
        document_id = str(metadata.get("document_id") or "")
        if not document_id or document_id in seen:
            continue
        seen.add(document_id)
        content = result.get("content") or result.get("page_content") or ""
        sources.append(
            {
                "document_id": document_id,
                "title": metadata.get("title") or "Unknown",
                "relevance_score": round(float(result.get("score") or 0.0), 3),
                "excerpt": content[:excerpt_chars]
                + ("..." if len(content) > excerpt_chars else ""),
            }
        )
        if len(sources) >= limit:
            break
    return sources


def confidence_of(results: List[Dict[str, Any]]) -> str:
    """high / medium / low, from how well the selected passages matched."""
    scores = [float(r.get("score") or 0.0) for r in results]
    average = sum(scores) / len(scores) if scores else 0.0
    return "high" if average > 0.7 else "medium" if average > 0.4 else "low"


async def record_trace(
    db: AsyncSession,
    retrieved: Retrieved,
    *,
    query: str,
    user_id: Any,
    vector_store: Any,
    trace_type: str,
    session_id: Any = None,
    commit: bool = True,
) -> Any:
    """Store how an answer's passages were found, and return the row.

    Only the chat page did this, so an answer from the agent's tool or from
    MCP could not be inspected afterwards: there was no record of which
    passages it had been written from or why those. `trace_type` says which
    surface asked ("chat", "tool", "mcp").

    `commit=False` flushes instead, for a caller working inside somebody
    else's transaction: the row is written when that transaction commits.
    """
    from app.models.retrieval_trace import RetrievalTrace

    provider = getattr(vector_store, "provider", None)
    row = RetrievalTrace(
        user_id=user_id,
        session_id=session_id,
        trace_type=trace_type,
        query=query,
        processed_query=retrieved.search_query,
        provider=provider,
        settings_snapshot={
            "provider": provider,
            "hybrid_enabled": bool(
                getattr(settings, "RAG_HYBRID_SEARCH_ENABLED", False)
            ),
            "hybrid_alpha": float(getattr(settings, "RAG_HYBRID_SEARCH_ALPHA", 0.0)),
            "rerank_enabled": bool(getattr(settings, "RAG_RERANKING_ENABLED", False)),
            "rerank_model": getattr(settings, "RAG_RERANKING_MODEL", None),
            "mmr_enabled": bool(getattr(settings, "RAG_MMR_ENABLED", False)),
            "dedup_enabled": bool(
                getattr(settings, "RAG_DEDUPLICATION_ENABLED", False)
            ),
            "min_relevance": float(getattr(settings, "RAG_MIN_RELEVANCE_SCORE", 0.0)),
            "max_search_results": int(getattr(settings, "MAX_SEARCH_RESULTS", 0)),
        },
        trace=retrieved.trace,
    )
    db.add(row)
    if commit:
        await db.commit()
        await db.refresh(row)
    else:
        await db.flush()
    return row
