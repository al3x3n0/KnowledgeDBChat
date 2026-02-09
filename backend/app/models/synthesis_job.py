"""
Synthesis Job Model.

Tracks multi-document synthesis and report generation jobs.
"""

import enum
from datetime import datetime
from typing import Optional, List, Dict, Any
from uuid import uuid4

from sqlalchemy import (
    Column, String, Text, DateTime, ForeignKey, Integer,
    JSON, Enum, Boolean
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.core.database import Base


class SynthesisJobType(str, enum.Enum):
    """Type of synthesis job."""
    MULTI_DOC_SUMMARY = "multi_doc_summary"        # Summarize multiple documents
    COMPARATIVE_ANALYSIS = "comparative_analysis"  # Compare documents
    THEME_EXTRACTION = "theme_extraction"          # Extract common themes
    KNOWLEDGE_SYNTHESIS = "knowledge_synthesis"    # Synthesize into new knowledge
    RESEARCH_REPORT = "research_report"            # Generate research report
    EXECUTIVE_BRIEF = "executive_brief"            # Executive briefing
    GAP_ANALYSIS_HYPOTHESES = "gap_analysis_hypotheses"  # Identify gaps & propose testable hypotheses


class SynthesisJobStatus(str, enum.Enum):
    """Status of synthesis job."""
    PENDING = "pending"
    ANALYZING = "analyzing"       # Analyzing source documents
    SYNTHESIZING = "synthesizing" # Generating synthesis
    GENERATING = "generating"     # Generating output document
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SynthesisJob(Base):
    """
    Tracks document synthesis jobs.

    Supports multi-document summarization, comparative analysis,
    theme extraction, and report generation with export options.
    """
    __tablename__ = "synthesis_jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)

    # Job configuration
    job_type = Column(String(50), nullable=False, default=SynthesisJobType.MULTI_DOC_SUMMARY.value)
    title = Column(String(500), nullable=False)
    description = Column(Text, nullable=True)

    # Source documents (list of document IDs)
    document_ids = Column(JSON, nullable=False, default=list)
    # Optional search query to find additional documents
    search_query = Column(Text, nullable=True)
    # Topic/focus for the synthesis
    topic = Column(String(500), nullable=True)

    # Synthesis options
    options = Column(JSON, nullable=True, default=dict)
    # Options include:
    # - max_length: int (word count limit)
    # - include_citations: bool
    # - include_charts: bool
    # - include_recommendations: bool
    # - comparison_criteria: list[str] (for comparative analysis)
    # - theme_categories: list[str] (for theme extraction)
    # - output_sections: list[str] (custom sections)
    # - style: str (professional, technical, casual)
    # - language: str

    # Output configuration
    output_format = Column(String(20), nullable=False, default="markdown")  # markdown, docx, pdf, pptx
    output_style = Column(String(50), nullable=False, default="professional")

    # Job status
    status = Column(String(20), nullable=False, default=SynthesisJobStatus.PENDING.value)
    progress = Column(Integer, nullable=False, default=0)
    current_stage = Column(String(100), nullable=True)

    # Results
    result_content = Column(Text, nullable=True)  # Generated content (markdown)
    result_metadata = Column(JSON, nullable=True)  # Metadata about results
    # Metadata includes:
    # - word_count: int
    # - documents_analyzed: int
    # - themes_found: list[str]
    # - key_findings: list[str]
    # - citations: list[dict]

    # Artifacts (charts, diagrams generated)
    artifacts = Column(JSON, nullable=True, default=list)

    # Export file (if output_format is not markdown)
    file_path = Column(String(500), nullable=True)  # MinIO path
    file_size = Column(Integer, nullable=True)

    # Error tracking
    error = Column(Text, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    # Relationships
    user = relationship("User", back_populates="synthesis_jobs")

    def __repr__(self):
        return f"<SynthesisJob {self.id} [{self.job_type}] - {self.status}>"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "user_id": str(self.user_id),
            "job_type": self.job_type,
            "title": self.title,
            "description": self.description,
            "document_ids": self.document_ids,
            "search_query": self.search_query,
            "topic": self.topic,
            "options": self.options,
            "output_format": self.output_format,
            "output_style": self.output_style,
            "status": self.status,
            "progress": self.progress,
            "current_stage": self.current_stage,
            "result_content": self.result_content,
            "result_metadata": self.result_metadata,
            "artifacts": self.artifacts,
            "file_path": self.file_path,
            "file_size": self.file_size,
            "error": self.error,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }
