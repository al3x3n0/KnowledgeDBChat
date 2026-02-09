"""
AI Hub recommendation feedback model.

Stores user feedback on AI Scientist recommendations (accept/reject + reason)
so future recommendations can be biased per customer profile.
"""

from __future__ import annotations

from datetime import datetime
import uuid

from sqlalchemy import Column, DateTime, ForeignKey, String, Text, Index, JSON
from sqlalchemy.dialects.postgresql import UUID

from app.core.database import Base


class AIHubRecommendationFeedback(Base):
    __tablename__ = "ai_hub_recommendation_feedback"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)

    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    agent_job_id = Column(UUID(as_uuid=True), ForeignKey("agent_jobs.id", ondelete="CASCADE"), nullable=True, index=True)

    customer_profile_id = Column(UUID(as_uuid=True), nullable=True, index=True)
    customer_profile_name = Column(String(200), nullable=True, index=True)
    workflow = Column(String(32), nullable=False, index=True)  # triage|extraction|literature

    item_type = Column(String(32), nullable=False, index=True)  # dataset_preset|eval_template
    item_id = Column(String(200), nullable=False, index=True)

    decision = Column(String(16), nullable=False, index=True)  # accept|reject
    reason = Column(Text, nullable=True)

    # Snapshots for later analysis/debugging
    customer_keywords = Column(JSON, nullable=True)  # list[str]

    __table_args__ = (
        Index(
            "ix_ai_hub_reco_feedback_profile_item",
            "customer_profile_id",
            "workflow",
            "item_type",
            "item_id",
        ),
    )
