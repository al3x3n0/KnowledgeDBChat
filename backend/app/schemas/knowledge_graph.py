"""
Pydantic schemas for knowledge graph APIs.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class KGEntity(BaseModel):
    id: str
    name: str = Field(alias="canonical_name")
    type: str = Field(alias="entity_type")

    class Config:
        populate_by_name = True


class KGRelationship(BaseModel):
    id: str
    type: str
    source: str
    target: str
    confidence: float
    evidence: Optional[str] = None
    chunk_id: Optional[str] = None


class KGRelationshipCreate(BaseModel):
    """Request schema for creating a relationship."""
    source_entity_id: str
    target_entity_id: str
    relation_type: str
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    evidence: Optional[str] = None


class KGRelationshipUpdate(BaseModel):
    """Request schema for updating a relationship."""
    relation_type: Optional[str] = None
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    evidence: Optional[str] = None


class KGRelationshipDetail(BaseModel):
    """Detailed relationship response with entity names."""
    id: str
    relation_type: str
    source_entity_id: str
    target_entity_id: str
    source_entity_name: str
    target_entity_name: str
    confidence: float
    evidence: Optional[str] = None
    document_id: Optional[str] = None
    chunk_id: Optional[str] = None
    is_manual: bool = False
    created_at: str


class KGStats(BaseModel):
    entities: int
    relationships: int
    mentions: int


class KGGraph(BaseModel):
    nodes: List[dict]
    edges: List[dict]


class KGChunk(BaseModel):
    id: str
    document_id: str
    document_title: Optional[str] = None
    chunk_index: int
    content: str
    match_start: Optional[int] = None
    match_end: Optional[int] = None


class KGMention(BaseModel):
    id: str
    entity_id: str
    document_id: str
    document_title: Optional[str] = None
    chunk_id: Optional[str] = None
    text: str
    sentence: Optional[str] = None
    start_pos: Optional[int] = None
    end_pos: Optional[int] = None
    created_at: Optional[str] = None


class KGEntityDetail(BaseModel):
    id: str
    canonical_name: str
    entity_type: str
    description: Optional[str] = None
    properties: Optional[str] = None


class KGEntityUpdate(BaseModel):
    canonical_name: Optional[str] = None
    entity_type: Optional[str] = None
    description: Optional[str] = None
    properties: Optional[str] = None


class KGAuditRecord(BaseModel):
    id: str
    user_id: str
    user_name: Optional[str] = None
    action: str
    details: Optional[str] = None
    created_at: str


class KGGlobalGraphMetadata(BaseModel):
    """Metadata for global graph response."""
    total_entities: int
    total_relationships: int
    filtered_nodes: int
    filtered_edges: int
    entity_types: List[str]
    relation_types: List[str]


class KGGlobalGraph(BaseModel):
    """Response model for global knowledge graph."""
    nodes: List[dict]
    edges: List[dict]
    metadata: KGGlobalGraphMetadata
