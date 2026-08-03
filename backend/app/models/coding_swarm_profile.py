from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import UUID

from app.core.database import Base


class CodingSwarmProfile(Base):
    __tablename__ = "coding_swarm_profiles"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    source_id = Column(
        UUID(as_uuid=True),
        ForeignKey("document_sources.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    title = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    status = Column(String(24), nullable=False, default="active", index=True)
    preset_key = Column(String(48), nullable=False, index=True)
    scope_default = Column(String(32), nullable=False, default="auto")
    default_commands = Column(JSON, nullable=True)
    default_file_paths = Column(JSON, nullable=True)
    max_agents = Column(Integer, nullable=False, default=4)
    safe_command_policy = Column(String(32), nullable=False, default="standard")
    saved_search_query = Column(String(500), nullable=True)
    is_default = Column(Boolean, nullable=False, default=False)
    visibility = Column(String(24), nullable=False, default="private", index=True)
    shared_with_user_ids = Column(JSON, nullable=True)
    latest_job_id = Column(
        UUID(as_uuid=True),
        ForeignKey("agent_jobs.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    profile_metadata = Column(JSON, nullable=True)

    created_at = Column(
        DateTime(timezone=True), default=datetime.utcnow, nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )
