"""
Service for querying the knowledge graph (entities, mentions, relationships).

Includes methods for:
- Basic CRUD operations on entities
- Graph queries (per-document, global)
- RAG integration (entity context for chat)
- Admin operations (merge, delete, audit)
"""

from typing import List, Dict, Any, Optional, Set
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, or_, and_
from loguru import logger

from app.models.knowledge_graph import Entity, EntityMention, Relationship


class KnowledgeGraphService:
    async def stats(self, db: AsyncSession) -> Dict[str, Any]:
        ent_count = (await db.execute(select(func.count(Entity.id)))).scalar() or 0
        rel_count = (await db.execute(select(func.count(Relationship.id)))).scalar() or 0
        mention_count = (await db.execute(select(func.count(EntityMention.id)))).scalar() or 0
        return {
            "entities": ent_count,
            "relationships": rel_count,
            "mentions": mention_count,
        }

    async def entities(self, db: AsyncSession, q: Optional[str] = None, limit: int = 50, offset: int = 0) -> List[Entity]:
        stmt = select(Entity).order_by(Entity.updated_at.desc())
        if q:
            like = f"%{q}%"
            from sqlalchemy import or_
            stmt = stmt.where(or_(Entity.canonical_name.ilike(like), Entity.entity_type.ilike(like)))
        stmt = stmt.offset(offset).limit(limit)
        res = await db.execute(stmt)
        return res.scalars().all()

    async def entities_count(self, db: AsyncSession, q: Optional[str] = None) -> int:
        stmt = select(func.count(Entity.id))
        if q:
            like = f"%{q}%"
            from sqlalchemy import or_
            stmt = stmt.where(or_(Entity.canonical_name.ilike(like), Entity.entity_type.ilike(like)))
        res = await db.execute(stmt)
        return int(res.scalar() or 0)

    async def audit_logs(
        self,
        db: AsyncSession,
        action: Optional[str] = None,
        user_id: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ):
        from app.models.knowledge_graph import KGAuditLog
        from app.models.user import User
        stmt = (
            select(
                KGAuditLog.id,
                KGAuditLog.user_id,
                KGAuditLog.action,
                KGAuditLog.details,
                KGAuditLog.created_at,
                User.username.label("username"),
                User.full_name.label("full_name"),
            )
            .join(User, User.id == KGAuditLog.user_id, isouter=True)
            .order_by(KGAuditLog.created_at.desc())
        )
        from sqlalchemy import and_
        conditions = []
        if action:
            conditions.append(KGAuditLog.action == action)
        if user_id:
            from uuid import UUID as _UUID
            try:
                user_uuid = _UUID(user_id)
                conditions.append(KGAuditLog.user_id == user_uuid)
            except ValueError:
                pass  # Invalid UUID format, skip filter
        if date_from:
            from datetime import datetime
            try:
                df = datetime.fromisoformat(date_from)
                conditions.append(KGAuditLog.created_at >= df)
            except Exception:
                pass
        if date_to:
            from datetime import datetime
            try:
                dt = datetime.fromisoformat(date_to)
                conditions.append(KGAuditLog.created_at <= dt)
            except Exception:
                pass
        if conditions:
            stmt = stmt.where(and_(*conditions))
        total_stmt = select(func.count()).select_from(stmt.subquery())
        total = int((await db.execute(total_stmt)).scalar() or 0)
        stmt = stmt.offset(offset).limit(limit)
        rows = (await db.execute(stmt)).all()
        items = []
        for r in rows:
            items.append({
                "id": str(r.id),
                "user_id": str(r.user_id),
                "user_name": (r.full_name or r.username) if (r.full_name or r.username) else None,
                "action": r.action,
                "details": r.details,
                "created_at": r.created_at.isoformat() if r.created_at else "",
            })
        return {"items": items, "total": total, "limit": limit, "offset": offset}

    async def relationships_for_entity(self, db: AsyncSession, entity_id, limit: int = 100) -> List[Relationship]:
        stmt = select(Relationship).where(
            (Relationship.source_entity_id == entity_id) | (Relationship.target_entity_id == entity_id)
        ).order_by(Relationship.created_at.desc()).limit(limit)
        res = await db.execute(stmt)
        return res.scalars().all()

    async def graph_for_document(self, db: AsyncSession, document_id) -> Dict[str, Any]:
        # Nodes: entities mentioned in this doc; edges: relations in this doc
        ent_stmt = select(Entity).join(EntityMention, EntityMention.entity_id == Entity.id).where(EntityMention.document_id == document_id).distinct()
        rel_stmt = select(Relationship).where(Relationship.document_id == document_id)
        ents = (await db.execute(ent_stmt)).scalars().all()
        rels = (await db.execute(rel_stmt)).scalars().all()
        return {
            "nodes": [
                {
                    "id": str(e.id),
                    "name": e.canonical_name,
                    "type": e.entity_type,
                }
                for e in ents
            ],
            "edges": [
                {
                    "id": str(r.id),
                    "type": r.relation_type,
                    "source": str(r.source_entity_id),
                    "target": str(r.target_entity_id),
                    "confidence": r.confidence,
                    "evidence": r.evidence,
                    "chunk_id": str(r.chunk_id) if r.chunk_id else None,
                }
                for r in rels
            ],
        }

    async def mentions_for_entity(self, db: AsyncSession, entity_id, limit: int = 25, offset: int = 0):
        from app.models.knowledge_graph import EntityMention
        from app.models.document import Document
        stmt = (
            select(
                EntityMention.id,
                EntityMention.entity_id,
                EntityMention.document_id,
                EntityMention.chunk_id,
                EntityMention.text,
                EntityMention.sentence,
                EntityMention.start_pos,
                EntityMention.end_pos,
                EntityMention.created_at,
                Document.title.label("document_title"),
            )
            .join(Document, Document.id == EntityMention.document_id)
            .where(EntityMention.entity_id == entity_id)
            .order_by(EntityMention.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
        res = await db.execute(stmt)
        rows = res.all()
        # Convert Row objects to dicts
        results = []
        for r in rows:
            d = {
                "id": str(r.id),
                "entity_id": str(r.entity_id),
                "document_id": str(r.document_id),
                "document_title": r.document_title,
                "chunk_id": str(r.chunk_id) if r.chunk_id else None,
                "text": r.text,
                "sentence": r.sentence,
                "start_pos": r.start_pos,
                "end_pos": r.end_pos,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            results.append(d)
        return results

    async def mentions_count_for_entity(self, db: AsyncSession, entity_id) -> int:
        from app.models.knowledge_graph import EntityMention
        q = await db.execute(select(func.count(EntityMention.id)).where(EntityMention.entity_id == entity_id))
        return int(q.scalar() or 0)

    async def merge_entities(self, db: AsyncSession, source_id: str, target_id: str) -> dict:
        """Merge source entity into target entity.

        - Repoint mentions to target
        - Update relationships to target (deduplicate where unique key collides)
        - Delete source entity
        Returns counts of updated/deleted items.
        """
        from uuid import UUID as _UUID
        from app.models.knowledge_graph import Entity, EntityMention, Relationship

        if source_id == target_id:
            return {"mentions_updated": 0, "rels_updated": 0, "rels_deleted": 0, "entity_deleted": False}

        src = await db.get(Entity, _UUID(source_id))
        tgt = await db.get(Entity, _UUID(target_id))
        if not src or not tgt:
            raise ValueError("Source or target entity not found")

        # Update mentions
        from sqlalchemy import update
        res = await db.execute(update(EntityMention).where(EntityMention.entity_id == src.id).values(entity_id=tgt.id))
        mentions_updated = res.rowcount or 0

        # Handle relationships referencing source
        rels_q = await db.execute(
            select(Relationship).where(
                (Relationship.source_entity_id == src.id) | (Relationship.target_entity_id == src.id)
            )
        )
        rels = rels_q.scalars().all()
        rels_updated = 0
        rels_deleted = 0
        for r in rels:
            new_source = tgt.id if r.source_entity_id == src.id else r.source_entity_id
            new_target = tgt.id if r.target_entity_id == src.id else r.target_entity_id
            # Check if equivalent relation exists
            exists_q = await db.execute(
                select(Relationship.id).where(
                    (Relationship.relation_type == r.relation_type)
                    & (Relationship.source_entity_id == new_source)
                    & (Relationship.target_entity_id == new_target)
                    & (Relationship.document_id == r.document_id)
                )
            )
            exists_id = exists_q.scalar_one_or_none()
            if exists_id and exists_id != r.id:
                # Duplicate would be created; delete current
                await db.delete(r)
                rels_deleted += 1
            else:
                r.source_entity_id = new_source
                r.target_entity_id = new_target
                rels_updated += 1

        # Delete source entity
        await db.delete(src)
        await db.commit()
        return {
            "mentions_updated": mentions_updated,
            "rels_updated": rels_updated,
            "rels_deleted": rels_deleted,
            "entity_deleted": True,
        }

    async def delete_entity(self, db: AsyncSession, entity_id: str) -> dict:
        """Delete an entity and cascade mentions/relationships.

        Returns counts of affected rows.
        """
        from uuid import UUID as _UUID
        from app.models.knowledge_graph import Entity, EntityMention, Relationship

        ent = await db.get(Entity, _UUID(entity_id))
        if not ent:
            raise ValueError("Entity not found")

        # Count mentions
        m_count_q = await db.execute(select(func.count(EntityMention.id)).where(EntityMention.entity_id == ent.id))
        mentions = int(m_count_q.scalar() or 0)

        # Count relationships where entity participates
        r_count_q = await db.execute(
            select(func.count(Relationship.id)).where(
                (Relationship.source_entity_id == ent.id) | (Relationship.target_entity_id == ent.id)
            )
        )
        relationships = int(r_count_q.scalar() or 0)

        await db.delete(ent)
        await db.commit()
        return {"mentions_deleted": mentions, "relationships_deleted": relationships, "entity_deleted": True}

    async def rebuild_for_document(self, db: AsyncSession, document_id) -> Dict[str, Any]:
        """Delete KG items for a document and re-extract from its chunks."""
        # Delete existing mentions and relationships for the doc
        from sqlalchemy import delete
        await db.execute(delete(EntityMention).where(EntityMention.document_id == document_id))
        await db.execute(delete(Relationship).where(Relationship.document_id == document_id))
        await db.flush()

        # Re-extract for each chunk in batches to avoid memory issues
        from app.models.document import DocumentChunk, Document
        from app.services.knowledge_extraction import extractor

        doc = (await db.execute(select(Document).where(Document.id == document_id))).scalar_one_or_none()
        if not doc:
            return {"mentions": 0, "relationships": 0}

        total_m, total_r = 0, 0
        batch_size = 100
        offset = 0

        while True:
            chunk_query = (
                select(DocumentChunk)
                .where(DocumentChunk.document_id == document_id)
                .order_by(DocumentChunk.chunk_index)
                .offset(offset)
                .limit(batch_size)
            )
            chunks = (await db.execute(chunk_query)).scalars().all()
            if not chunks:
                break

            for ch in chunks:
                m, r = await extractor.index_chunk(db, doc, ch)
                total_m += m
                total_r += r

            offset += batch_size

        await db.commit()
        return {"mentions": total_m, "relationships": total_r}

    # ==================== RAG Integration Methods ====================

    async def search_entities_by_names(
        self,
        names: List[str],
        db: AsyncSession,
        entity_types: Optional[List[str]] = None,
        limit: int = 20
    ) -> List[Entity]:
        """Find entities matching given names (case-insensitive, partial match).

        Used for RAG context injection - finds relevant entities from user query.

        Args:
            names: List of entity name strings to search for
            db: Database session
            entity_types: Optional filter by entity types
            limit: Maximum entities to return

        Returns:
            List of matching Entity objects
        """
        if not names:
            return []

        # Build OR conditions for each name (partial match)
        name_conditions = []
        for name in names:
            if len(name) < 2:  # Skip very short strings
                continue
            like_pattern = f"%{name}%"
            name_conditions.append(Entity.canonical_name.ilike(like_pattern))

        if not name_conditions:
            return []

        stmt = select(Entity).where(or_(*name_conditions))

        # Filter by entity types if specified
        if entity_types:
            stmt = stmt.where(Entity.entity_type.in_(entity_types))

        stmt = stmt.limit(limit)
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def get_entity_context(
        self,
        entity_ids: List[UUID],
        db: AsyncSession,
        max_relationships: int = 30
    ) -> Dict[str, Any]:
        """Get entities with their relationships for RAG context building.

        Args:
            entity_ids: List of entity UUIDs to get context for
            db: Database session
            max_relationships: Maximum relationships to return

        Returns:
            Dict with 'entities' and 'relationships' lists
        """
        if not entity_ids:
            return {"entities": [], "relationships": []}

        # Get entities
        stmt = select(Entity).where(Entity.id.in_(entity_ids))
        result = await db.execute(stmt)
        entities = list(result.scalars().all())

        # Get relationships between these entities (both directions)
        entity_id_set = set(entity_ids)
        rel_stmt = (
            select(Relationship)
            .where(
                or_(
                    Relationship.source_entity_id.in_(entity_ids),
                    Relationship.target_entity_id.in_(entity_ids)
                )
            )
            .order_by(Relationship.confidence.desc())
            .limit(max_relationships * 2)  # Get more, then filter
        )
        rel_result = await db.execute(rel_stmt)
        all_rels = list(rel_result.scalars().all())

        # Prioritize relationships between the found entities
        internal_rels = []
        external_rels = []
        for r in all_rels:
            if r.source_entity_id in entity_id_set and r.target_entity_id in entity_id_set:
                internal_rels.append(r)
            else:
                external_rels.append(r)

        # Combine: internal first, then external up to limit
        relationships = internal_rels[:max_relationships]
        remaining = max_relationships - len(relationships)
        if remaining > 0:
            relationships.extend(external_rels[:remaining])

        # Load related entity names for external relationships
        external_entity_ids = set()
        for r in relationships:
            if r.source_entity_id not in entity_id_set:
                external_entity_ids.add(r.source_entity_id)
            if r.target_entity_id not in entity_id_set:
                external_entity_ids.add(r.target_entity_id)

        if external_entity_ids:
            ext_stmt = select(Entity).where(Entity.id.in_(external_entity_ids))
            ext_result = await db.execute(ext_stmt)
            entities.extend(ext_result.scalars().all())

        return {
            "entities": entities,
            "relationships": relationships
        }

    async def extract_entity_names_from_text(
        self,
        text: str,
        db: AsyncSession,
        limit: int = 10
    ) -> List[str]:
        """Extract potential entity names from text by matching against known entities.

        Uses simple word/phrase matching against existing entity names.
        Useful for quick entity lookup from user queries.

        Args:
            text: Text to extract entity names from
            db: Database session
            limit: Maximum entity names to return

        Returns:
            List of entity names found in the text
        """
        if not text or len(text) < 3:
            return []

        # Get candidate phrases (2-4 word sequences that might be entity names)
        import re
        words = re.findall(r'\b[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*\b', text)

        # Also get any quoted strings
        quoted = re.findall(r'"([^"]+)"', text)
        candidates = list(set(words + quoted))

        if not candidates:
            return []

        # Find matching entities
        found_names = []
        for candidate in candidates[:20]:  # Limit candidates to check
            if len(candidate) < 2:
                continue
            stmt = select(Entity.canonical_name).where(
                Entity.canonical_name.ilike(f"%{candidate}%")
            ).limit(3)
            result = await db.execute(stmt)
            for name in result.scalars().all():
                if name not in found_names:
                    found_names.append(name)
                    if len(found_names) >= limit:
                        return found_names

        return found_names

    # ==================== Global Graph Methods ====================

    async def global_graph(
        self,
        db: AsyncSession,
        entity_types: Optional[List[str]] = None,
        relation_types: Optional[List[str]] = None,
        min_confidence: float = 0.0,
        min_mentions: int = 1,
        limit_nodes: int = 300,
        limit_edges: int = 1000,
        search: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build global knowledge graph across all documents.

        Args:
            db: Database session
            entity_types: Filter by entity types (e.g., ['person', 'org'])
            relation_types: Filter by relationship types
            min_confidence: Minimum relationship confidence (0.0-1.0)
            min_mentions: Minimum entity mention count to include
            limit_nodes: Maximum nodes to return
            limit_edges: Maximum edges to return
            search: Optional search string for entity names

        Returns:
            Dict with 'nodes', 'edges', and 'metadata'
        """
        # Build entity query with mention count
        entity_subq = (
            select(
                Entity.id,
                Entity.canonical_name,
                Entity.entity_type,
                Entity.description,
                func.count(EntityMention.id).label("mention_count")
            )
            .outerjoin(EntityMention, EntityMention.entity_id == Entity.id)
            .group_by(Entity.id)
        )

        # Apply filters
        conditions = []
        if entity_types:
            conditions.append(Entity.entity_type.in_(entity_types))
        if search:
            conditions.append(Entity.canonical_name.ilike(f"%{search}%"))

        if conditions:
            entity_subq = entity_subq.where(and_(*conditions))

        # Filter by minimum mentions using HAVING
        entity_subq = entity_subq.having(func.count(EntityMention.id) >= min_mentions)

        # Order by mention count descending, limit
        entity_subq = entity_subq.order_by(func.count(EntityMention.id).desc()).limit(limit_nodes)

        # Execute entity query
        entity_result = await db.execute(entity_subq)
        entity_rows = entity_result.all()

        # Build nodes list and collect entity IDs
        nodes = []
        entity_ids: Set[UUID] = set()
        for row in entity_rows:
            entity_ids.add(row.id)
            nodes.append({
                "id": str(row.id),
                "name": row.canonical_name,
                "type": row.entity_type,
                "description": row.description,
                "mention_count": row.mention_count,
            })

        # Get relationships between found entities
        edges = []
        if entity_ids:
            rel_stmt = (
                select(Relationship)
                .where(
                    and_(
                        Relationship.source_entity_id.in_(entity_ids),
                        Relationship.target_entity_id.in_(entity_ids),
                        Relationship.confidence >= min_confidence
                    )
                )
            )
            if relation_types:
                rel_stmt = rel_stmt.where(Relationship.relation_type.in_(relation_types))

            rel_stmt = rel_stmt.order_by(Relationship.confidence.desc()).limit(limit_edges)

            rel_result = await db.execute(rel_stmt)
            relationships = rel_result.scalars().all()

            for r in relationships:
                edges.append({
                    "id": str(r.id),
                    "type": r.relation_type,
                    "source": str(r.source_entity_id),
                    "target": str(r.target_entity_id),
                    "confidence": r.confidence,
                    "evidence": r.evidence,
                    "chunk_id": str(r.chunk_id) if r.chunk_id else None,
                })

        # Get total counts for metadata
        total_entities = (await db.execute(select(func.count(Entity.id)))).scalar() or 0
        total_relationships = (await db.execute(select(func.count(Relationship.id)))).scalar() or 0

        # Types (for UI filter dropdowns). Keep it cheap by deriving from returned graph.
        returned_entity_types = sorted({n.get("type") for n in nodes if n.get("type")})
        returned_relation_types = sorted({e.get("type") for e in edges if e.get("type")})

        return {
            "nodes": nodes,
            "edges": edges,
            "metadata": {
                "total_entities": total_entities,
                "total_relationships": total_relationships,
                # Fields required by KGGlobalGraphMetadata schema
                "filtered_nodes": len(nodes),
                "filtered_edges": len(edges),
                "entity_types": returned_entity_types,
                "relation_types": returned_relation_types,

                # Backward/forward compatibility (ignored by response_model if not declared)
                "returned_entities": len(nodes),
                "returned_relationships": len(edges),
                "filters_applied": {
                    "entity_types": entity_types,
                    "relation_types": relation_types,
                    "min_confidence": min_confidence,
                    "min_mentions": min_mentions,
                    "search": search,
                },
            }
        }

    # =========================================================================
    # Relationship CRUD Operations
    # =========================================================================

    async def get_relationship(self, db: AsyncSession, rel_id: str) -> Optional[Relationship]:
        """Get a single relationship by ID."""
        from uuid import UUID as _UUID
        return await db.get(Relationship, _UUID(rel_id))

    async def get_relationship_detail(self, db: AsyncSession, rel_id: str) -> Optional[Dict[str, Any]]:
        """Get a relationship with source and target entity names."""
        from uuid import UUID as _UUID

        rel = await db.get(Relationship, _UUID(rel_id))
        if not rel:
            return None

        source_entity = await db.get(Entity, rel.source_entity_id)
        target_entity = await db.get(Entity, rel.target_entity_id)

        return {
            "id": str(rel.id),
            "relation_type": rel.relation_type,
            "source_entity_id": str(rel.source_entity_id),
            "target_entity_id": str(rel.target_entity_id),
            "source_entity_name": source_entity.canonical_name if source_entity else "Unknown",
            "target_entity_name": target_entity.canonical_name if target_entity else "Unknown",
            "confidence": rel.confidence,
            "evidence": rel.evidence,
            "document_id": str(rel.document_id) if rel.document_id else None,
            "chunk_id": str(rel.chunk_id) if rel.chunk_id else None,
            "is_manual": rel.document_id is None,
            "created_at": rel.created_at.isoformat() if rel.created_at else None,
        }

    async def create_relationship(
        self,
        db: AsyncSession,
        source_entity_id: str,
        target_entity_id: str,
        relation_type: str,
        confidence: float = 0.8,
        evidence: Optional[str] = None,
    ) -> Relationship:
        """Create a manual relationship between two entities.

        Manual relationships have no document_id (they're not extracted from text).
        """
        from uuid import UUID as _UUID, uuid4

        # Validate entities exist
        source = await db.get(Entity, _UUID(source_entity_id))
        if not source:
            raise ValueError(f"Source entity {source_entity_id} not found")

        target = await db.get(Entity, _UUID(target_entity_id))
        if not target:
            raise ValueError(f"Target entity {target_entity_id} not found")

        # Normalize relation type
        relation_type = relation_type.lower().replace(" ", "_").replace("-", "_")

        # Check for existing relationship (manual relationships have document_id=None)
        existing = await db.execute(
            select(Relationship).where(
                Relationship.relation_type == relation_type,
                Relationship.source_entity_id == source.id,
                Relationship.target_entity_id == target.id,
                Relationship.document_id.is_(None),
            )
        )
        if existing.scalar_one_or_none():
            raise ValueError(
                f"Manual relationship '{relation_type}' already exists between these entities"
            )

        rel = Relationship(
            id=uuid4(),
            relation_type=relation_type,
            source_entity_id=source.id,
            target_entity_id=target.id,
            confidence=confidence,
            evidence=evidence,
            document_id=None,  # Manual relationship
            chunk_id=None,
            inferred=False,
        )
        db.add(rel)
        await db.commit()
        await db.refresh(rel)
        return rel

    async def update_relationship(
        self,
        db: AsyncSession,
        rel_id: str,
        relation_type: Optional[str] = None,
        confidence: Optional[float] = None,
        evidence: Optional[str] = None,
    ) -> Optional[Relationship]:
        """Update a relationship's type, confidence, or evidence."""
        from uuid import UUID as _UUID

        rel = await db.get(Relationship, _UUID(rel_id))
        if not rel:
            return None

        changed = False

        if relation_type is not None:
            rel.relation_type = relation_type.lower().replace(" ", "_").replace("-", "_")
            changed = True

        if confidence is not None:
            rel.confidence = confidence
            changed = True

        if evidence is not None:
            rel.evidence = evidence
            changed = True

        if changed:
            await db.commit()
            await db.refresh(rel)

        return rel

    async def delete_relationship(self, db: AsyncSession, rel_id: str) -> bool:
        """Delete a relationship by ID."""
        from uuid import UUID as _UUID

        rel = await db.get(Relationship, _UUID(rel_id))
        if not rel:
            return False

        await db.delete(rel)
        await db.commit()
        return True

    async def list_relation_types(self, db: AsyncSession) -> List[str]:
        """Get all unique relation types in the system."""
        result = await db.execute(
            select(Relationship.relation_type)
            .distinct()
            .order_by(Relationship.relation_type)
        )
        return [row[0] for row in result.fetchall()]
