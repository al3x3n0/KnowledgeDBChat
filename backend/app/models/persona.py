"""
Persona-related database models.
"""

from datetime import datetime
from typing import Optional, Dict, Any
from sqlalchemy import Column, String, Text, DateTime, Boolean, ForeignKey, JSON, Float, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
import uuid

from app.core.database import Base


class Persona(Base):
    """Persona model describing a human/agent referenced inside the platform."""

    __tablename__ = "personas"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Identity
    name = Column(String(255), nullable=False, index=True)
    platform_id = Column(String(255), nullable=True, unique=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)

    # Description
    description = Column(Text, nullable=True)
    extra_metadata = Column(JSON, nullable=True)
    avatar_url = Column(String(500), nullable=True)

    # Status
    is_active = Column(Boolean, nullable=False, default=True)
    is_system = Column(Boolean, nullable=False, default=False)

    # Timestamps
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="personas")
    owned_documents = relationship("Document", back_populates="owner_persona")
    persona_mentions = relationship("DocumentPersonaDetection", back_populates="persona", cascade="all, delete-orphan")
    edit_requests = relationship("PersonaEditRequest", back_populates="persona", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Persona(id={self.id}, name='{self.name}', platform_id='{self.platform_id}')>"


class DocumentPersonaDetection(Base):
    """Association model linking documents to personas with detection metadata."""

    __tablename__ = "document_persona_detections"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    persona_id = Column(UUID(as_uuid=True), ForeignKey("personas.id", ondelete="CASCADE"), nullable=False, index=True)

    # Detection context
    role = Column(String(50), nullable=False, default="detected")  # detected, diarized, owner, mentioned
    detection_type = Column(String(50), nullable=True)  # diarization, ner, manual
    confidence = Column(Float, nullable=True)
    start_time = Column(Float, nullable=True)  # Seconds for diarization context
    end_time = Column(Float, nullable=True)
    details = Column(JSON, nullable=True)

    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    # Relationships
    document = relationship("Document", back_populates="persona_detections")
    persona = relationship("Persona", back_populates="persona_mentions")

    __table_args__ = (
        UniqueConstraint("document_id", "persona_id", "role", "start_time", "end_time", name="uq_document_persona_detection"),
    )

    def __repr__(self) -> str:
        return f"<DocumentPersonaDetection(doc={self.document_id}, persona={self.persona_id}, role='{self.role}')>"


class PersonaEditRequest(Base):
    """Requests submitted by users to modify persona details."""

    __tablename__ = "persona_edit_requests"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    persona_id = Column(UUID(as_uuid=True), ForeignKey("personas.id", ondelete="CASCADE"), nullable=False, index=True)
    requested_by = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="SET NULL"), nullable=True)
    message = Column(Text, nullable=False)
    status = Column(String(32), nullable=False, default="pending")
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    resolved_at = Column(DateTime(timezone=True), nullable=True)

    persona = relationship("Persona", back_populates="edit_requests")
    requested_by_user = relationship("User", back_populates="persona_edit_requests")
    document = relationship("Document")

    def __repr__(self) -> str:
        return f"<PersonaEditRequest(persona={self.persona_id}, requested_by={self.requested_by}, status='{self.status}')>"
