"""
Knowledge graph database models: entities, mentions, and relationships.
"""

from datetime import datetime
from typing import Optional
from sqlalchemy import (
    Column,
    String,
    Text,
    DateTime,
    ForeignKey,
    Float,
    Boolean,
    Index,
    UniqueConstraint,
    Integer,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
import uuid

from app.core.database import Base


class Entity(Base):
    """Canonical entity node in the knowledge graph."""

    __tablename__ = "kg_entities"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    canonical_name = Column(String(512), nullable=False)
    entity_type = Column(String(64), nullable=False)  # person, org, location, product, email, url, other
    description = Column(Text, nullable=True)
    properties = Column(Text, nullable=True)  # JSON string to avoid PG JSON dependency coupling
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    mentions = relationship("EntityMention", back_populates="entity", cascade="all, delete-orphan")
    outgoing = relationship("Relationship", back_populates="source_entity", foreign_keys="Relationship.source_entity_id", cascade="all, delete-orphan")
    incoming = relationship("Relationship", back_populates="target_entity", foreign_keys="Relationship.target_entity_id", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_kg_entities_type_name", "entity_type", "canonical_name"),
    )


class EntityMention(Base):
    """Occurrence of an entity in a document/chunk with offsets."""

    __tablename__ = "kg_entity_mentions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    entity_id = Column(UUID(as_uuid=True), ForeignKey("kg_entities.id", ondelete="CASCADE"), nullable=False)

    # Provenance
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    chunk_id = Column(UUID(as_uuid=True), ForeignKey("document_chunks.id", ondelete="CASCADE"), nullable=True)

    text = Column(String(512), nullable=False)
    start_pos = Column(Integer, nullable=True)
    end_pos = Column(Integer, nullable=True)
    sentence = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    entity = relationship("Entity", back_populates="mentions")

    __table_args__ = (
        Index("ix_kg_mentions_doc", "document_id"),
        Index("ix_kg_mentions_chunk", "chunk_id"),
    )


class Relationship(Base):
    """Directed relationship between two entities with provenance."""

    __tablename__ = "kg_relationships"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    relation_type = Column(String(64), nullable=False)  # works_for, mentions, part_of, references, etc.
    confidence = Column(Float, default=0.5)
    inferred = Column(Boolean, default=False)

    # Endpoints
    source_entity_id = Column(UUID(as_uuid=True), ForeignKey("kg_entities.id", ondelete="CASCADE"), nullable=False)
    target_entity_id = Column(UUID(as_uuid=True), ForeignKey("kg_entities.id", ondelete="CASCADE"), nullable=False)

    # Provenance
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    chunk_id = Column(UUID(as_uuid=True), ForeignKey("document_chunks.id", ondelete="CASCADE"), nullable=True)
    evidence = Column(Text, nullable=True)  # sentence or snippet supporting the relation
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    source_entity = relationship("Entity", foreign_keys=[source_entity_id], back_populates="outgoing")
    target_entity = relationship("Entity", foreign_keys=[target_entity_id], back_populates="incoming")

    __table_args__ = (
        Index("ix_kg_relations_type", "relation_type"),
        Index("ix_kg_relations_doc", "document_id"),
        UniqueConstraint("relation_type", "source_entity_id", "target_entity_id", "document_id", name="uq_kg_relation_once_per_doc"),
    )


class KGAuditLog(Base):
    """Audit log for KG admin actions."""

    __tablename__ = "kg_audit_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False)
    action = Column(String(64), nullable=False)  # merge_entities, update_entity, delete_entity
    details = Column(Text, nullable=True)  # JSON string
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    __table_args__ = (
        Index("ix_kg_audit_action", "action"),
        Index("ix_kg_audit_created", "created_at"),
    )
