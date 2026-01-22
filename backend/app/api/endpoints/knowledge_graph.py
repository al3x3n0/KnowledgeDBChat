"""
Knowledge graph API endpoints.
"""

from typing import Optional, List
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.services.knowledge_graph_service import KnowledgeGraphService
from app.schemas.knowledge_graph import (
    KGStats, KGGraph, KGEntity, KGRelationship, KGChunk, KGMention,
    KGEntityDetail, KGEntityUpdate, KGAuditRecord, KGGlobalGraph,
    KGRelationshipCreate, KGRelationshipUpdate, KGRelationshipDetail,
)
from pydantic import BaseModel
from app.services.auth_service import require_admin
from app.models.document import DocumentChunk, Document
from sqlalchemy import select
from app.models.knowledge_graph import Entity
import json


router = APIRouter()


@router.get("/stats", response_model=KGStats)
async def kg_stats(db: AsyncSession = Depends(get_db)):
    svc = KnowledgeGraphService()
    return await svc.stats(db)


@router.get("/document/{document_id}/graph", response_model=KGGraph)
async def document_graph(document_id: str, db: AsyncSession = Depends(get_db)):
    try:
        svc = KnowledgeGraphService()
        return await svc.graph_for_document(db, document_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get graph: {e}")


@router.get("/global/graph", response_model=KGGlobalGraph)
async def global_graph(
    entity_types: Optional[str] = Query(None, description="Comma-separated entity types to include"),
    relation_types: Optional[str] = Query(None, description="Comma-separated relation types to include"),
    min_confidence: float = Query(0.0, ge=0, le=1, description="Minimum relationship confidence"),
    min_mentions: int = Query(1, ge=1, description="Minimum entity mention count"),
    limit_nodes: int = Query(300, ge=10, le=1000, description="Maximum nodes to return"),
    limit_edges: int = Query(1000, ge=10, le=5000, description="Maximum edges to return"),
    search: Optional[str] = Query(None, description="Search entities by name"),
    db: AsyncSession = Depends(get_db),
):
    """
    Get global knowledge graph across all documents.

    Returns nodes (entities) and edges (relationships) with filters for:
    - Entity types (person, org, technology, etc.)
    - Relation types (works_for, manages, uses, etc.)
    - Minimum confidence score
    - Minimum mention count
    - Text search on entity names
    """
    try:
        svc = KnowledgeGraphService()
        return await svc.global_graph(
            db=db,
            entity_types=entity_types.split(",") if entity_types else None,
            relation_types=relation_types.split(",") if relation_types else None,
            min_confidence=min_confidence,
            min_mentions=min_mentions,
            limit_nodes=limit_nodes,
            limit_edges=limit_edges,
            search=search,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get global graph: {e}")


@router.get("/audit")
async def get_audit_logs(
    action: Optional[str] = Query(default=None),
    user_id: Optional[str] = Query(default=None),
    date_from: Optional[str] = Query(default=None),
    date_to: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db),
    current_user=Depends(require_admin),
):
    try:
        svc = KnowledgeGraphService()
        return await svc.audit_logs(db, action=action, user_id=user_id, date_from=date_from, date_to=date_to, limit=limit, offset=offset)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get audit logs: {e}")


@router.get("/entities")
async def list_entities(
    q: Optional[str] = Query(default=None, description="Search query by name/type"),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    svc = KnowledgeGraphService()
    ents = await svc.entities(db, q=q, limit=limit, offset=offset)
    total = await svc.entities_count(db, q=q)
    # Pydantic model expects fields by alias mapping
    items = [KGEntity(id=str(e.id), canonical_name=e.canonical_name, entity_type=e.entity_type) for e in ents]
    return {"items": items, "total": total, "limit": limit, "offset": offset}


@router.get("/entity/{entity_id}/relationships", response_model=List[KGRelationship])
async def entity_relationships(entity_id: str, limit: int = Query(default=100, ge=1, le=500), db: AsyncSession = Depends(get_db)):
    svc = KnowledgeGraphService()
    rels = await svc.relationships_for_entity(db, entity_id, limit=limit)
    return [
        KGRelationship(
            id=str(r.id),
            type=r.relation_type,
            source=str(r.source_entity_id),
            target=str(r.target_entity_id),
            confidence=r.confidence or 0.0,
            evidence=getattr(r, 'evidence', None),
            chunk_id=str(getattr(r, 'chunk_id')) if getattr(r, 'chunk_id', None) else None,
        )
        for r in rels
    ]


@router.post("/document/{document_id}/rebuild")
async def rebuild_document_graph(document_id: str, db: AsyncSession = Depends(get_db)):
    try:
        svc = KnowledgeGraphService()
        result = await svc.rebuild_for_document(db, document_id)
        return {"message": "rebuild_complete", **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to rebuild: {e}")


@router.get("/chunk/{chunk_id}", response_model=KGChunk)
async def get_chunk(
    chunk_id: str,
    evidence: Optional[str] = Query(default=None, description="Evidence text to locate within the chunk"),
    db: AsyncSession = Depends(get_db)
):
    try:
        chunk = await db.get(DocumentChunk, UUID(chunk_id))
        if not chunk:
            raise HTTPException(status_code=404, detail="Chunk not found")
        # Load document title
        doc = (await db.execute(select(Document).where(Document.id == chunk.document_id))).scalar_one_or_none()
        content = chunk.content or ""
        match_start = None
        match_end = None
        if evidence:
            idx = content.find(evidence)
            if idx >= 0:
                match_start = idx
                match_end = idx + len(evidence)
        return KGChunk(
            id=str(chunk.id),
            document_id=str(chunk.document_id),
            document_title=doc.title if doc else None,
            chunk_index=chunk.chunk_index,
            content=content,
            match_start=match_start,
            match_end=match_end,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch chunk: {e}")


@router.get("/entity/{entity_id}/mentions")
async def entity_mentions(
    entity_id: str,
    limit: int = Query(default=25, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db)
):
    try:
        svc = KnowledgeGraphService()
        items = await svc.mentions_for_entity(db, entity_id, limit=limit, offset=offset)
        total = await svc.mentions_count_for_entity(db, entity_id)
        return {"items": items, "total": total, "limit": limit, "offset": offset}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get mentions: {e}")


class MergeEntitiesRequest(BaseModel):
    source_id: str
    target_id: str


@router.post("/entities/merge")
async def merge_entities(
    req: MergeEntitiesRequest,
    db: AsyncSession = Depends(get_db),
    current_user=Depends(require_admin),
):
    try:
        svc = KnowledgeGraphService()
        result = await svc.merge_entities(db, req.source_id, req.target_id)
        # Audit log
        try:
            from app.models.knowledge_graph import KGAuditLog
            log = KGAuditLog(
                user_id=current_user.id,
                action='merge_entities',
                details=json.dumps({"source_id": req.source_id, "target_id": req.target_id, **result})
            )
            db.add(log)
            await db.commit()
        except Exception:
            pass
        return {"message": "merged", **result}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to merge entities: {e}")


@router.get("/entity/{entity_id}", response_model=KGEntityDetail)
async def get_entity(entity_id: str, db: AsyncSession = Depends(get_db)):
    try:
        ent = await db.get(Entity, UUID(entity_id))
        if not ent:
            raise HTTPException(status_code=404, detail="Entity not found")
        return KGEntityDetail(
            id=str(ent.id),
            canonical_name=ent.canonical_name,
            entity_type=ent.entity_type,
            description=ent.description,
            properties=ent.properties,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get entity: {e}")


@router.patch("/entity/{entity_id}", response_model=KGEntityDetail)
async def update_entity(
    entity_id: str,
    data: KGEntityUpdate,
    db: AsyncSession = Depends(get_db),
    current_user=Depends(require_admin),
):
    try:
        ent = await db.get(Entity, UUID(entity_id))
        if not ent:
            raise HTTPException(status_code=404, detail="Entity not found")
        changed = False
        if data.canonical_name is not None:
            ent.canonical_name = data.canonical_name
            changed = True
        if data.entity_type is not None:
            ent.entity_type = data.entity_type
            changed = True
        if data.description is not None:
            ent.description = data.description
            changed = True
        if data.properties is not None:
            ent.properties = data.properties
            changed = True
        if changed:
            await db.commit()
            await db.refresh(ent)
            # Audit log
            try:
                from app.models.knowledge_graph import KGAuditLog
                log = KGAuditLog(
                    user_id=current_user.id,
                    action='update_entity',
                    details=json.dumps({"entity_id": entity_id, "changes": data.dict(exclude_unset=True)})
                )
                db.add(log)
                await db.commit()
            except Exception:
                pass
        return KGEntityDetail(
            id=str(ent.id),
            canonical_name=ent.canonical_name,
            entity_type=ent.entity_type,
            description=ent.description,
            properties=ent.properties,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update entity: {e}")


@router.delete("/entity/{entity_id}")
async def delete_entity(
    entity_id: str,
    confirm_name: str = Query(..., description="Type the entity canonical name to confirm"),
    db: AsyncSession = Depends(get_db),
    current_user=Depends(require_admin),
):
    try:
        ent = await db.get(Entity, UUID(entity_id))
        if not ent:
            raise HTTPException(status_code=404, detail="Entity not found")
        if ent.canonical_name != confirm_name:
            raise HTTPException(status_code=400, detail="Confirmation name does not match entity")
        svc = KnowledgeGraphService()
        result = await svc.delete_entity(db, entity_id)
        # Audit log
        try:
            from app.models.knowledge_graph import KGAuditLog
            log = KGAuditLog(
                user_id=current_user.id,
                action='delete_entity',
                details=json.dumps({"entity_id": entity_id, **result})
            )
            db.add(log)
            await db.commit()
        except Exception:
            pass
        return {"message": "deleted", **result}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete entity: {e}")


# =============================================================================
# Relationship CRUD Endpoints
# =============================================================================

@router.get("/relation-types")
async def list_relation_types(db: AsyncSession = Depends(get_db)):
    """List all unique relation types in the system."""
    svc = KnowledgeGraphService()
    types = await svc.list_relation_types(db)
    return {"types": types}


@router.get("/relationship/{rel_id}", response_model=KGRelationshipDetail)
async def get_relationship(
    rel_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get detailed information about a relationship."""
    svc = KnowledgeGraphService()
    detail = await svc.get_relationship_detail(db, rel_id)
    if not detail:
        raise HTTPException(status_code=404, detail="Relationship not found")
    return detail


@router.post("/relationship", response_model=KGRelationshipDetail)
async def create_relationship(
    data: KGRelationshipCreate,
    db: AsyncSession = Depends(get_db),
    current_user=Depends(require_admin),
):
    """Create a new manual relationship between two entities (admin only)."""
    try:
        svc = KnowledgeGraphService()
        rel = await svc.create_relationship(
            db=db,
            source_entity_id=data.source_entity_id,
            target_entity_id=data.target_entity_id,
            relation_type=data.relation_type,
            confidence=data.confidence,
            evidence=data.evidence,
        )

        # Audit log
        try:
            from app.models.knowledge_graph import KGAuditLog
            log = KGAuditLog(
                user_id=current_user.id,
                action='create_relationship',
                details=json.dumps({
                    "relationship_id": str(rel.id),
                    "source_entity_id": data.source_entity_id,
                    "target_entity_id": data.target_entity_id,
                    "relation_type": data.relation_type,
                })
            )
            db.add(log)
            await db.commit()
        except Exception:
            pass

        # Return detail
        detail = await svc.get_relationship_detail(db, str(rel.id))
        return detail

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create relationship: {e}")


@router.patch("/relationship/{rel_id}", response_model=KGRelationshipDetail)
async def update_relationship(
    rel_id: str,
    data: KGRelationshipUpdate,
    db: AsyncSession = Depends(get_db),
    current_user=Depends(require_admin),
):
    """Update a relationship's type, confidence, or evidence (admin only)."""
    try:
        svc = KnowledgeGraphService()

        # Get current state for audit
        old_rel = await svc.get_relationship(db, rel_id)
        if not old_rel:
            raise HTTPException(status_code=404, detail="Relationship not found")

        rel = await svc.update_relationship(
            db=db,
            rel_id=rel_id,
            relation_type=data.relation_type,
            confidence=data.confidence,
            evidence=data.evidence,
        )

        if not rel:
            raise HTTPException(status_code=404, detail="Relationship not found")

        # Audit log
        try:
            from app.models.knowledge_graph import KGAuditLog
            changes = data.dict(exclude_unset=True)
            log = KGAuditLog(
                user_id=current_user.id,
                action='update_relationship',
                details=json.dumps({
                    "relationship_id": rel_id,
                    "changes": changes,
                })
            )
            db.add(log)
            await db.commit()
        except Exception:
            pass

        # Return detail
        detail = await svc.get_relationship_detail(db, str(rel.id))
        return detail

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update relationship: {e}")


@router.delete("/relationship/{rel_id}")
async def delete_relationship(
    rel_id: str,
    confirm: bool = Query(False, description="Set to true to confirm deletion"),
    db: AsyncSession = Depends(get_db),
    current_user=Depends(require_admin),
):
    """Delete a relationship (admin only)."""
    try:
        svc = KnowledgeGraphService()

        # Get relationship details for response and audit
        detail = await svc.get_relationship_detail(db, rel_id)
        if not detail:
            raise HTTPException(status_code=404, detail="Relationship not found")

        if not confirm:
            return {
                "message": "Set confirm=true to delete this relationship",
                "relationship": detail,
            }

        deleted = await svc.delete_relationship(db, rel_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Relationship not found")

        # Audit log
        try:
            from app.models.knowledge_graph import KGAuditLog
            log = KGAuditLog(
                user_id=current_user.id,
                action='delete_relationship',
                details=json.dumps({
                    "relationship_id": rel_id,
                    "relation_type": detail["relation_type"],
                    "source_entity_id": detail["source_entity_id"],
                    "target_entity_id": detail["target_entity_id"],
                })
            )
            db.add(log)
            await db.commit()
        except Exception:
            pass

        return {"message": "Relationship deleted", "deleted": detail}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete relationship: {e}")
