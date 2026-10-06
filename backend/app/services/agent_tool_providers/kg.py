"""Autonomous-job tools: the ``kg`` provider.

Split out of ``agent_tool_dispatch``, which re-exports every name here.
"""

from __future__ import annotations

from typing import Any, Dict

from app.services.agent_tool_providers.base import (
    AgentToolExecutionContext,
    FunctionToolProvider,
)
from app.services.agent_tool_providers.common import _load_documents_for_analysis


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
            "findings": [
                {
                    "type": "research_graph",
                    "documents_analyzed": len(documents),
                    "entities_found": entities_found,
                    "relationships_found": relationships_found,
                }
            ],
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
            "findings": [
                {
                    "type": "kb_entry",
                    "research_note_id": str(note.id),
                    "title": note.title,
                    "entry_type": entry_type or None,
                }
            ],
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
            entities = await kg.entities(
                ctx.db, q=query, limit=limit, entity_type=entity_type
            )
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
            if not context["entities"]:
                return {"error": f"Entity {entity_id} not found"}
            # Plain data: the service returns rows, which neither a model nor
            # the job's JSON results can hold.
            return {
                "success": True,
                "data": {
                    "entities": [
                        {
                            "id": str(e.id),
                            "canonical_name": e.canonical_name,
                            "entity_type": e.entity_type,
                            "description": e.description or "",
                        }
                        for e in context["entities"]
                    ],
                    "relationships": [
                        {
                            "id": str(r.id),
                            "relation_type": r.relation_type,
                            "source_entity_id": str(r.source_entity_id),
                            "target_entity_id": str(r.target_entity_id),
                            "confidence": r.confidence,
                            "evidence": r.evidence or "",
                        }
                        for r in context["relationships"]
                    ],
                },
            }
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
            # Committed, like the relationship beside it: the id is handed to
            # the model, and a later tool rolling this session back would
            # leave it holding the id of a row that no longer exists.
            await ctx.db.commit()
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
                relation_type=relation_type[:64],
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
                # An entity this run created has no mentions yet; the default
                # of 1 hid it from the graph it had just been added to.
                min_mentions=0,
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
