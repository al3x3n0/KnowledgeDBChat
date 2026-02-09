"""
User secrets (vault) model.

Stores per-user secrets encrypted at rest. Secrets are never returned in plaintext
by list endpoints; plaintext is only returned on create/update operations if needed.
"""

from datetime import datetime
from uuid import uuid4

from sqlalchemy import Column, String, DateTime, ForeignKey, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func

from app.core.database import Base


class UserSecret(Base):
    __tablename__ = "user_secrets"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)

    name = Column(String(255), nullable=False)
    encrypted_value = Column(String, nullable=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    __table_args__ = (
        UniqueConstraint("user_id", "name", name="uq_user_secrets_user_name"),
    )

