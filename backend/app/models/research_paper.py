"""
Structured paper extraction models for arXiv-backed documents.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import JSON, Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.core.database import Base


class ResearchPaper(Base):
    __tablename__ = "research_papers"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    document_id = Column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    source_id = Column(
        UUID(as_uuid=True),
        ForeignKey("document_sources.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    arxiv_id = Column(String(128), nullable=False, index=True)
    title = Column(String(500), nullable=False)
    authors = Column(JSON, nullable=True)
    abstract = Column(Text, nullable=True)
    published_at = Column(DateTime(timezone=True), nullable=True)
    categories = Column(JSON, nullable=True)
    paper_url = Column(String(1000), nullable=True)
    pdf_url = Column(String(1000), nullable=True)

    extraction_status = Column(
        String(32), nullable=False, default="pending", index=True
    )
    extracted_at = Column(DateTime(timezone=True), nullable=True)
    extractor_version = Column(String(64), nullable=True)

    summary = Column(Text, nullable=True)
    mechanisms = Column(JSON, nullable=True)
    assumptions = Column(JSON, nullable=True)
    benchmarks = Column(JSON, nullable=True)
    metrics = Column(JSON, nullable=True)
    limitations = Column(JSON, nullable=True)
    raw_extraction_payload = Column(JSON, nullable=True)

    created_at = Column(
        DateTime(timezone=True), default=datetime.utcnow, nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    user = relationship("User", backref="research_papers")
    document = relationship("Document", backref="research_paper")
    source = relationship("DocumentSource", backref="research_papers")
    claims = relationship(
        "PaperClaim", back_populates="paper", cascade="all, delete-orphan"
    )
    extraction_jobs = relationship(
        "PaperExtractionJob", back_populates="paper", cascade="all, delete-orphan"
    )


class PaperClaim(Base):
    __tablename__ = "paper_claims"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    paper_id = Column(
        UUID(as_uuid=True),
        ForeignKey("research_papers.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    kind = Column(String(32), nullable=False, default="other")
    statement = Column(Text, nullable=False)
    mechanism = Column(String(255), nullable=True)
    target_layer = Column(String(32), nullable=False, default="unknown")
    conditions = Column(JSON, nullable=True)
    assumptions = Column(JSON, nullable=True)
    expected_effect = Column(Text, nullable=True)
    evidence_summary = Column(Text, nullable=True)
    confidence = Column(Float, nullable=True)
    tags = Column(JSON, nullable=True)
    rank = Column(Integer, nullable=True)

    created_at = Column(
        DateTime(timezone=True), default=datetime.utcnow, nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    paper = relationship("ResearchPaper", back_populates="claims")


class PaperExtractionJob(Base):
    __tablename__ = "paper_extraction_jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    document_id = Column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    source_id = Column(
        UUID(as_uuid=True),
        ForeignKey("document_sources.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    paper_id = Column(
        UUID(as_uuid=True),
        ForeignKey("research_papers.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    status = Column(String(32), nullable=False, default="pending", index=True)
    extractor_version = Column(String(64), nullable=True)
    error = Column(Text, nullable=True)
    request_payload = Column(JSON, nullable=True)
    result_summary = Column(JSON, nullable=True)

    created_at = Column(
        DateTime(timezone=True), default=datetime.utcnow, nullable=False
    )
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    updated_at = Column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    user = relationship("User", backref="paper_extraction_jobs")
    document = relationship("Document", backref="paper_extraction_jobs")
    source = relationship("DocumentSource", backref="paper_extraction_jobs")
    paper = relationship("ResearchPaper", back_populates="extraction_jobs")
