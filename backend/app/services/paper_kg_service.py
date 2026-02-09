"""
Utilities to map paper insights (structured extraction) into the knowledge graph.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, Optional
from uuid import UUID

from loguru import logger
from sqlalchemy import select, delete
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document, DocumentSource
from app.models.knowledge_graph import Entity, EntityMention, Relationship


REL_USES_METHOD = "uses_method"
REL_EVALUATED_ON = "evaluated_on"
REL_TARGETS_TASK = "targets_task"


def _normalize_list(values: Any) -> list[str]:
    if not values:
        return []
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list):
        return []
    out: list[str] = []
    for v in values:
        if isinstance(v, str):
            s = v.strip()
            if s:
                out.append(s[:512])
    # Preserve order, remove duplicates
    seen = set()
    uniq = []
    for s in out:
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(s)
    return uniq


def _arxiv_id_from_source_identifier(source_identifier: Optional[str]) -> Optional[str]:
    if not source_identifier:
        return None
    parts = source_identifier.strip().rstrip("/").split("/")
    if not parts:
        return None
    last = parts[-1]
    return last or None


class PaperKnowledgeGraphService:
    async def upsert_from_document(self, db: AsyncSession, document_id: UUID, force: bool = False) -> Dict[str, Any]:
        doc = await db.get(Document, document_id)
        if not doc:
            raise ValueError("Document not found")
        src = await db.get(DocumentSource, doc.source_id)
        if not src or src.source_type != "arxiv":
            return {"skipped": True, "reason": "not_arxiv"}

        meta = doc.extra_metadata or {}
        insights = meta.get("paper_insights") if isinstance(meta, dict) else None
        if not isinstance(insights, dict):
            return {"skipped": True, "reason": "no_paper_insights"}

        if force:
            await db.execute(
                delete(Relationship).where(
                    Relationship.document_id == doc.id,
                    Relationship.relation_type.in_([REL_USES_METHOD, REL_EVALUATED_ON, REL_TARGETS_TASK]),
                    Relationship.inferred == True,
                    Relationship.evidence == "paper_insights",
                )
            )
            await db.execute(
                delete(EntityMention).where(
                    EntityMention.document_id == doc.id,
                    EntityMention.chunk_id.is_(None),
                    EntityMention.sentence == "paper_insights",
                )
            )
            await db.flush()

        arxiv_id = _arxiv_id_from_source_identifier(doc.source_identifier)
        paper_key = arxiv_id or (doc.title or "paper")[:512]
        paper, _ = await self._get_or_create_entity(
            db,
            canonical_name=paper_key,
            entity_type="paper",
            description=(doc.title or None) if arxiv_id else None,
            properties={
                "document_id": str(doc.id),
                "title": doc.title,
                "arxiv_id": arxiv_id,
                "url": doc.url,
            },
        )

        methods = _normalize_list(insights.get("methods"))
        datasets = _normalize_list(insights.get("datasets"))
        tasks = _normalize_list(insights.get("tasks"))

        created_entities = 0
        created_mentions = 0
        created_relationships = 0

        async def link_many(items: Iterable[str], entity_type: str, relation_type: str) -> None:
            nonlocal created_entities, created_mentions, created_relationships
            for name in items:
                ent, created = await self._get_or_create_entity(db, canonical_name=name, entity_type=entity_type)
                if created:
                    created_entities += 1

                if await self._ensure_mention(db, ent.id, doc.id, text=name):
                    created_mentions += 1

                if await self._ensure_relationship(db, relation_type, paper.id, ent.id, doc.id):
                    created_relationships += 1

        await link_many(methods, "method", REL_USES_METHOD)
        await link_many(datasets, "dataset", REL_EVALUATED_ON)
        await link_many(tasks, "task", REL_TARGETS_TASK)

        await db.commit()
        return {
            "skipped": False,
            "document_id": str(doc.id),
            "paper_entity_id": str(paper.id),
            "entities_created": created_entities,
            "mentions_created": created_mentions,
            "relationships_created": created_relationships,
        }

    async def _get_or_create_entity(
        self,
        db: AsyncSession,
        canonical_name: str,
        entity_type: str,
        description: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
    ) -> tuple[Entity, bool]:
        q = await db.execute(
            select(Entity).where(Entity.canonical_name == canonical_name, Entity.entity_type == entity_type)
        )
        ent = q.scalar_one_or_none()
        if ent:
            return ent, False
        ent = Entity(
            canonical_name=canonical_name[:512],
            entity_type=entity_type[:64],
            description=description,
            properties=json.dumps(properties, ensure_ascii=False) if properties else None,
        )
        db.add(ent)
        await db.flush()
        return ent, True

    async def _ensure_mention(self, db: AsyncSession, entity_id: UUID, document_id: UUID, text: str) -> bool:
        q = await db.execute(
            select(EntityMention.id).where(
                EntityMention.entity_id == entity_id,
                EntityMention.document_id == document_id,
                EntityMention.chunk_id.is_(None),
                EntityMention.text == text[:512],
            )
        )
        if q.scalar_one_or_none():
            return False
        db.add(
            EntityMention(
                entity_id=entity_id,
                document_id=document_id,
                chunk_id=None,
                text=text[:512],
                start_pos=None,
                end_pos=None,
                sentence="paper_insights",
            )
        )
        await db.flush()
        return True

    async def _ensure_relationship(
        self,
        db: AsyncSession,
        relation_type: str,
        source_entity_id: UUID,
        target_entity_id: UUID,
        document_id: UUID,
    ) -> bool:
        exists = await db.execute(
            select(Relationship.id).where(
                Relationship.relation_type == relation_type[:64],
                Relationship.source_entity_id == source_entity_id,
                Relationship.target_entity_id == target_entity_id,
                Relationship.document_id == document_id,
            )
        )
        if exists.scalar_one_or_none():
            return False
        try:
            async with db.begin_nested():
                db.add(
                    Relationship(
                        relation_type=relation_type[:64],
                        confidence=0.7,
                        inferred=True,
                        source_entity_id=source_entity_id,
                        target_entity_id=target_entity_id,
                        document_id=document_id,
                        chunk_id=None,
                        evidence="paper_insights",
                    )
                )
                await db.flush()
            return True
        except IntegrityError:
            return False
        except Exception as exc:
            logger.debug(f"Failed to add KG relationship: {exc}")
            return False
