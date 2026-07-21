from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import JSON, Boolean, Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.core.database import Base


class BenchmarkSuite(Base):
    __tablename__ = "benchmark_suites"

    id = Column(String(120), primary_key=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    name = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    track_type = Column(String(32), nullable=False, default="compiler", index=True)
    benchmark_family = Column(String(64), nullable=False, default="compiler_regression", index=True)
    suite_version = Column(Integer, nullable=False, default=1)
    tags = Column(JSON, nullable=False, default=list)
    metadata_json = Column(JSON, nullable=False, default=dict)
    enabled = Column(Boolean, nullable=False, default=True, index=True)
    system_managed = Column(Boolean, nullable=False, default=False, index=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    cases = relationship("BenchmarkCase", back_populates="suite", cascade="all, delete-orphan")
    baselines = relationship("BenchmarkBaseline", back_populates="suite", cascade="all, delete-orphan")

    def to_dict(self, *, include_cases: bool = True, include_baselines: bool = True) -> dict[str, Any]:
        return {
            "id": self.id,
            "user_id": str(self.user_id) if self.user_id else None,
            "name": self.name,
            "description": self.description,
            "track_type": self.track_type,
            "benchmark_family": self.benchmark_family,
            "suite_version": int(self.suite_version or 1),
            "tags": list(self.tags or []),
            "metadata": dict(self.metadata_json or {}),
            "enabled": bool(self.enabled),
            "system_managed": bool(self.system_managed),
            "created_at": self.created_at.isoformat() if isinstance(self.created_at, datetime) else None,
            "updated_at": self.updated_at.isoformat() if isinstance(self.updated_at, datetime) else None,
            "cases": [case.to_dict() for case in sorted(self.cases, key=lambda row: (int(row.rank or 0), row.name))] if include_cases else [],
            "baselines": [row.to_dict() for row in self.baselines] if include_baselines else [],
        }


class BenchmarkCase(Base):
    __tablename__ = "benchmark_cases"

    id = Column(String(120), primary_key=True)
    suite_id = Column(String(120), ForeignKey("benchmark_suites.id", ondelete="CASCADE"), nullable=False, index=True)
    name = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    rank = Column(Integer, nullable=False, default=0)
    source_ref = Column(String(500), nullable=True)
    benchmark_query = Column(String(500), nullable=True)
    compile_command_template = Column(Text, nullable=True)
    run_command_template = Column(Text, nullable=True)
    expected_artifacts = Column(JSON, nullable=False, default=list)
    metrics = Column(JSON, nullable=False, default=list)
    metadata_json = Column(JSON, nullable=False, default=dict)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    suite = relationship("BenchmarkSuite", back_populates="cases")
    baselines = relationship("BenchmarkBaseline", back_populates="case")

    def to_dict(self) -> dict[str, Any]:
        metadata = dict(self.metadata_json or {})
        return {
            "id": self.id,
            "suite_id": self.suite_id,
            "name": self.name,
            "description": self.description,
            "rank": int(self.rank or 0),
            "source_ref": self.source_ref,
            "benchmark_query": self.benchmark_query,
            "compile_command_template": self.compile_command_template,
            "run_command_template": self.run_command_template,
            "expected_artifacts": list(self.expected_artifacts or []),
            "metrics": list(self.metrics or []),
            "observability": dict(metadata.get("observability") or {}) if isinstance(metadata.get("observability"), dict) else {},
            "metadata": metadata,
            "created_at": self.created_at.isoformat() if isinstance(self.created_at, datetime) else None,
            "updated_at": self.updated_at.isoformat() if isinstance(self.updated_at, datetime) else None,
        }


class BenchmarkBaseline(Base):
    __tablename__ = "benchmark_baselines"

    id = Column(String(120), primary_key=True)
    suite_id = Column(String(120), ForeignKey("benchmark_suites.id", ondelete="CASCADE"), nullable=False, index=True)
    case_id = Column(String(120), ForeignKey("benchmark_cases.id", ondelete="SET NULL"), nullable=True, index=True)
    name = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    compiler_revision = Column(String(120), nullable=True)
    toolchain_id = Column(String(120), nullable=True)
    sandbox_profile_id = Column(String(120), nullable=True)
    measurements = Column(JSON, nullable=False, default=dict)
    environment_snapshot = Column(JSON, nullable=False, default=dict)
    enabled = Column(Boolean, nullable=False, default=True, index=True)
    system_managed = Column(Boolean, nullable=False, default=False, index=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    suite = relationship("BenchmarkSuite", back_populates="baselines")
    case = relationship("BenchmarkCase", back_populates="baselines")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "suite_id": self.suite_id,
            "case_id": self.case_id,
            "name": self.name,
            "description": self.description,
            "compiler_revision": self.compiler_revision,
            "toolchain_id": self.toolchain_id,
            "sandbox_profile_id": self.sandbox_profile_id,
            "measurements": dict(self.measurements or {}),
            "environment_snapshot": dict(self.environment_snapshot or {}),
            "enabled": bool(self.enabled),
            "system_managed": bool(self.system_managed),
            "created_at": self.created_at.isoformat() if isinstance(self.created_at, datetime) else None,
            "updated_at": self.updated_at.isoformat() if isinstance(self.updated_at, datetime) else None,
        }
