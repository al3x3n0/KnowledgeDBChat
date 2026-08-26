"""
LLM call snapshots (replay/debug observability).

Stores the full request and response of individual LLM calls so a failed
agent iteration (or any LLM-backed flow) can be inspected and replayed
exactly. Opt-in via LLM_CALL_SNAPSHOT_ENABLED — payloads contain full prompt
text, so storage and data-sensitivity costs are real.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import JSON, Column, DateTime, ForeignKey, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID

from app.core.database import Base


class LLMCallSnapshot(Base):
    __tablename__ = "llm_call_snapshots"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    # Correlation back to the agent loop (nullable: non-agent calls too).
    job_id = Column(
        UUID(as_uuid=True),
        ForeignKey("agent_jobs.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    iteration = Column(Integer, nullable=True)
    # thinking | native_tool_loop | compaction | decision_repair | ...
    phase = Column(String(50), nullable=True)

    provider = Column(String(50), nullable=True)
    model = Column(String(200), nullable=True)
    task_type = Column(String(50), nullable=True)

    # Full request payload: {"messages": [...]} and/or
    # {"system_prompt": ..., "user_message": ...} plus tool/schema flags.
    request = Column(JSON, nullable=False, default=dict)

    response_text = Column(Text, nullable=True)
    tool_calls = Column(JSON, nullable=True)
    structured = Column(JSON, nullable=True)

    # What the model thought before it answered. Reasoning models return this
    # separately from the answer and charge it against max_tokens, so a run can
    # spend most of its budget here and show nothing for it -- which is exactly
    # how an empty completion happens. Recorded so a decision can be audited
    # against the thinking that produced it, rather than only the model's own
    # summary of that thinking.
    reasoning_text = Column(Text, nullable=True)
    reasoning_tokens = Column(Integer, nullable=True)

    error = Column(Text, nullable=True)
    latency_ms = Column(Integer, nullable=True)
    prompt_tokens = Column(Integer, nullable=True)
    completion_tokens = Column(Integer, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    __table_args__ = (
        Index("ix_llm_call_snapshots_job_created", "job_id", "created_at"),
    )
