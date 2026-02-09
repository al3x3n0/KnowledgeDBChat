"""
LLM usage tracking.

Stores per-request token/latency metadata for LLM calls, grouped by provider/model/task type.
"""

from uuid import uuid4

from sqlalchemy import Column, String, DateTime, ForeignKey, Integer
from sqlalchemy.dialects.postgresql import UUID, JSON
from sqlalchemy.sql import func

from app.core.database import Base


class LLMUsageEvent(Base):
    __tablename__ = "llm_usage_events"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)

    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)

    provider = Column(String(32), nullable=False, index=True)  # ollama, deepseek, openai, custom
    model = Column(String(128), nullable=True, index=True)
    task_type = Column(String(32), nullable=True, index=True)  # chat, summarization, etc.

    prompt_tokens = Column(Integer, nullable=True)
    completion_tokens = Column(Integer, nullable=True)
    total_tokens = Column(Integer, nullable=True)

    input_chars = Column(Integer, nullable=True)
    output_chars = Column(Integer, nullable=True)

    latency_ms = Column(Integer, nullable=True)
    error = Column(String(255), nullable=True)

    extra = Column(JSON, nullable=True)  # provider-specific metadata (durations, stop_reason, etc.)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)

