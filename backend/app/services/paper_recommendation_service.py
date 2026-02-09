"""
Related-paper recommendations using KG overlap + embedding similarity.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from loguru import logger
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document
from app.models.knowledge_graph import Entity, EntityMention, Relationship
from app.services.vector_store import vector_store_service


KG_REL_TYPES = ("uses_method", "evaluated_on", "targets_task")


class PaperRecommendationService:
    def __init__(self) -> None:
        self.vector_store = vector_store_service
        self._vector_store_initialized = False

    async def _ensure_vector_store_initialized(self) -> None:
        if not self._vector_store_initialized:
            await self.vector_store.initialize(background=True)
            self._vector_store_initialized = True

    async def related_documents(
        self,
        db: AsyncSession,
        document_id: UUID,
        *,
        limit: int = 8,
    ) -> List[Dict[str, Any]]:
        doc = await db.get(Document, document_id)
        if not doc:
            raise ValueError("Document not found")

        kg_hits, kg_entities = await self._kg_candidates(db, doc.id, limit=60)
        vec_hits = await self._vector_candidates(doc, limit=80)

        candidate_ids = set(kg_hits.keys()) | set(vec_hits.keys())
        candidate_ids.discard(doc.id)
        if not candidate_ids:
            return []

        docs_result = await db.execute(select(Document.id, Document.title).where(Document.id.in_(list(candidate_ids))))
        title_by_id = {row[0]: row[1] for row in docs_result.all()}

        max_kg = max((v for v, _ in kg_hits.values()), default=0)
        max_vec = max((v for v, _ in vec_hits.values()), default=0.0)
        max_kg = max(max_kg, 1)
        max_vec = max(max_vec, 1e-6)

        scored: List[Tuple[float, Dict[str, Any]]] = []
        for cid in candidate_ids:
            kg_count, common_ids = kg_hits.get(cid, (0, []))
            vec_score, best_chunk = vec_hits.get(cid, (0.0, None))
            combined = (0.6 * (kg_count / max_kg)) + (0.4 * (vec_score / max_vec))

            common_names = [kg_entities.get(eid) for eid in common_ids if eid in kg_entities]
            common_names = [n for n in common_names if n]

            scored.append(
                (
                    combined,
                    {
                        "document_id": str(cid),
                        "title": title_by_id.get(cid),
                        "score": combined,
                        "kg_overlap": int(kg_count),
                        "vector_score": float(vec_score),
                        "common_entities": common_names[:8],
                        "best_chunk_id": str(best_chunk) if best_chunk else None,
                    },
                )
            )

        scored.sort(key=lambda x: x[0], reverse=True)
        return [payload for _, payload in scored[:limit]]

    async def _kg_candidates(
        self, db: AsyncSession, document_id: UUID, *, limit: int
    ) -> Tuple[Dict[UUID, Tuple[int, List[UUID]]], Dict[UUID, str]]:
        rels_result = await db.execute(
            select(Relationship.target_entity_id)
            .where(
                Relationship.document_id == document_id,
                Relationship.relation_type.in_(KG_REL_TYPES),
                Relationship.inferred == True,
            )
        )
        entity_ids = [row[0] for row in rels_result.all() if row and row[0]]
        if not entity_ids:
            return {}, {}

        ents_result = await db.execute(select(Entity.id, Entity.canonical_name).where(Entity.id.in_(entity_ids)))
        entity_name = {row[0]: row[1] for row in ents_result.all()}

        mentions_result = await db.execute(
            select(EntityMention.document_id, EntityMention.entity_id, func.count(EntityMention.id).label("cnt"))
            .where(EntityMention.entity_id.in_(entity_ids))
            .group_by(EntityMention.document_id, EntityMention.entity_id)
        )

        by_doc: Dict[UUID, List[UUID]] = {}
        for doc_id, ent_id, _cnt in mentions_result.all():
            if not doc_id or not ent_id or doc_id == document_id:
                continue
            by_doc.setdefault(doc_id, [])
            if ent_id not in by_doc[doc_id]:
                by_doc[doc_id].append(ent_id)

        candidates = sorted(by_doc.items(), key=lambda kv: len(kv[1]), reverse=True)[:limit]
        out: Dict[UUID, Tuple[int, List[UUID]]] = {}
        for doc_id, ent_ids in candidates:
            out[doc_id] = (len(ent_ids), ent_ids)
        return out, entity_name

    async def _vector_candidates(
        self, doc: Document, *, limit: int
    ) -> Dict[UUID, Tuple[float, Optional[UUID]]]:
        await self._ensure_vector_store_initialized()
        query = (doc.title or "") + "\n" + ((doc.summary or "")[:800] if doc.summary else (doc.content or "")[:800])
        query = query.strip()
        if not query:
            return {}

        try:
            results = await self.vector_store.search(query=query, limit=limit)
        except Exception as exc:
            logger.debug(f"Vector recommendations failed: {exc}")
            return {}

        best: Dict[UUID, Tuple[float, Optional[UUID]]] = {}
        for r in results or []:
            meta = r.get("metadata", {}) or {}
            doc_id_raw = meta.get("document_id") or r.get("id")
            try:
                cand_id = UUID(str(doc_id_raw))
            except Exception:
                continue
            if cand_id == doc.id:
                continue
            score = float(r.get("score", 0.0) or 0.0)
            chunk_id = meta.get("chunk_id")
            try:
                chunk_uuid = UUID(str(chunk_id)) if chunk_id else None
            except Exception:
                chunk_uuid = None
            prev = best.get(cand_id)
            if not prev or score > prev[0]:
                best[cand_id] = (score, chunk_uuid)
        return best
