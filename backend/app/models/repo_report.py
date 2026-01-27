"""
Repository report job models for generating reports and presentations from GitHub/GitLab repos.
"""

from datetime import datetime
from uuid import uuid4

from sqlalchemy import Column, String, Integer, Text, DateTime, Boolean, ForeignKey, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.core.database import Base


class RepoReportJob(Base):
    """
    Represents a repository report or presentation generation job.

    Tracks the status and progress of generating reports (DOCX/PDF) or
    presentations (PPTX) from GitHub/GitLab repository data.

    Supports two modes:
    1. Synced DocumentSource - uses existing configured repository
    2. Ad-hoc URL - one-time analysis of a repository URL
    """
    __tablename__ = "repo_report_jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)

    # Source reference - either synced DocumentSource or ad-hoc URL
    source_id = Column(UUID(as_uuid=True), ForeignKey("document_sources.id", ondelete="SET NULL"), nullable=True)
    adhoc_url = Column(String(500), nullable=True)  # For one-time repository analysis
    adhoc_token = Column(String(500), nullable=True)  # Encrypted token for ad-hoc private repos

    # Repository information (populated during analysis)
    repo_name = Column(String(255), nullable=False)
    repo_url = Column(String(500), nullable=False)
    repo_type = Column(String(20), nullable=False)  # 'github', 'gitlab'

    # Output configuration
    output_format = Column(String(20), nullable=False)  # 'docx', 'pdf', 'pptx'
    title = Column(String(255), nullable=False)

    # Section selection - list of sections to include in the report/presentation
    # Options: overview, readme, file_structure, commits, issues, pull_requests,
    #          code_stats, contributors, architecture, technology_stack
    sections = Column(JSON, nullable=False)

    # PPTX-specific settings
    slide_count = Column(Integer, nullable=True)  # Target slide count for presentations
    include_diagrams = Column(Boolean, default=True)  # Include Mermaid diagrams

    # Style configuration
    style = Column(String(50), default="professional")  # professional, technical, casual, etc.
    custom_theme = Column(JSON, nullable=True)
    # Example custom_theme:
    # {
    #   "title_color": "#1a365d",
    #   "heading_color": "#2e86ab",
    #   "text_color": "#333333",
    #   "title_font": "Arial",
    #   "body_font": "Times New Roman",
    #   "title_size": 24,
    #   "body_size": 12
    # }

    # Cached analysis data
    # Stores RepoAnalysisResult as JSON to avoid re-fetching for format conversion
    analysis_data = Column(JSON, nullable=True)

    # Job status
    status = Column(String(50), default="pending")
    # Status values:
    # - pending: Job created, waiting to start
    # - analyzing: Fetching and analyzing repository data
    # - generating_insights: Running LLM for architecture/features/tech insights
    # - building: Building the document/presentation
    # - uploading: Uploading to MinIO
    # - completed: Successfully completed
    # - failed: Job failed
    # - cancelled: Job cancelled by user

    progress = Column(Integer, default=0)  # 0-100
    current_stage = Column(String(100), nullable=True)  # Human-readable stage description

    # Output file
    file_path = Column(String(500), nullable=True)  # MinIO path to generated file
    file_size = Column(Integer, nullable=True)  # File size in bytes

    # Error handling
    error = Column(Text, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    # Relationships
    user = relationship("User", back_populates="repo_report_jobs")
    source = relationship("DocumentSource", backref="repo_report_jobs")

    def __repr__(self):
        return f"<RepoReportJob {self.id} - {self.repo_name} ({self.status})>"

    @property
    def is_complete(self) -> bool:
        """Check if job has finished (completed or failed)."""
        return self.status in ("completed", "failed", "cancelled")

    @property
    def duration_seconds(self) -> float | None:
        """Calculate job duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
