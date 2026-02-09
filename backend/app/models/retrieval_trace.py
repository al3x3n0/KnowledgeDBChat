"""
Retrieval traces (observability + provenance for RAG).

Stores the stages of retrieval (semantic, BM25, hybrid merge, rerank, etc.)
to support artifact reviews, debugging, and evaluation/regression.
"""

from __future__ import annotations

from datetime import datetime
import uuid

from sqlalchemy import Column, DateTime, ForeignKey, Index, JSON, String, Text
from sqlalchemy.dialects.postgresql import UUID

from app.core.database import Base


class RetrievalTrace(Base):
    __tablename__ = "retrieval_traces"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)

    # Optional linkage back to chat sessions/messages.
    session_id = Column(UUID(as_uuid=True), ForeignKey("chat_sessions.id", ondelete="SET NULL"), nullable=True, index=True)
    chat_message_id = Column(UUID(as_uuid=True), ForeignKey("chat_messages.id", ondelete="SET NULL"), nullable=True, index=True)

    trace_type = Column(String(32), nullable=False, default="chat")  # chat|tool|artifact

    query = Column(Text, nullable=False)
    processed_query = Column(Text, nullable=True)

    provider = Column(String(32), nullable=True)  # chroma|qdrant

    # Snapshot of relevant settings (hybrid enabled, alpha, rerank model, etc.)
    settings_snapshot = Column(JSON, nullable=True)

    # Full trace payload (stage candidates, timings, postprocessing).
    trace = Column(JSON, nullable=False, default=dict)

    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index("ix_retrieval_traces_user_created", "user_id", "created_at"),
        Index("ix_retrieval_traces_session_created", "session_id", "created_at"),
    )

