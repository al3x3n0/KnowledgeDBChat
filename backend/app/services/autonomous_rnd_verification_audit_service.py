"""Canonical, signed audit snapshots for autonomous R&D verification."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from datetime import datetime, timezone
from typing import Any, Dict, Mapping
from uuid import UUID

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from app.core.config import settings
from app.models.agent_job import AgentJob


class AutonomousRnDVerificationAuditService:
    """Build and verify strictly allowlisted verification audit envelopes."""

    schema_version = 1
    canonicalization = "json-sort-keys-compact-v1"
    signature_algorithm = "ed25519"
    signature_encoding = "hex"

    def build_signed_snapshot(
        self,
        *,
        registry_id: UUID,
        parent_job: AgentJob,
        lifecycle: Mapping[str, Any],
        task_id: str | None = None,
        status: str | None = None,
        generated_at: datetime | None = None,
    ) -> Dict[str, Any]:
        snapshot = self.build_snapshot(
            registry_id=registry_id,
            parent_job=parent_job,
            lifecycle=lifecycle,
            task_id=task_id,
            status=status,
            generated_at=generated_at,
        )
        canonical = self.canonical_bytes(snapshot)
        key = self._private_key()
        public_key = self._public_key_hex(key)
        return {
            "snapshot": snapshot,
            "integrity": {
                "canonicalization": self.canonicalization,
                "sha256": hashlib.sha256(canonical).hexdigest(),
                "signature_algorithm": self.signature_algorithm,
                "signature_encoding": self.signature_encoding,
                "signature": key.sign(canonical).hex(),
                "key_id": self.key_id,
                "public_key": public_key,
            },
        }

    def build_snapshot(
        self,
        *,
        registry_id: UUID,
        parent_job: AgentJob,
        lifecycle: Mapping[str, Any],
        task_id: str | None = None,
        status: str | None = None,
        generated_at: datetime | None = None,
    ) -> Dict[str, Any]:
        normalized_task_id = _text(task_id)
        normalized_status = (_text(status) or "").lower() or None
        events = [
            self._safe_event(event)
            for event in lifecycle.get("timeline", [])
            if isinstance(event, Mapping)
            and (
                not normalized_task_id
                or _text(event.get("task_id")) == normalized_task_id
            )
            and (
                not normalized_status
                or (_text(event.get("status")) or "").lower() == normalized_status
            )
        ]
        event_task_ids = {
            str(event["task_id"]) for event in events if event.get("task_id")
        }
        tasks = [
            self._safe_task(task)
            for task in lifecycle.get("tasks", [])
            if isinstance(task, Mapping)
            and (
                not normalized_task_id
                or _text(task.get("task_id")) == normalized_task_id
            )
            and (
                not normalized_status
                or self._task_has_status(task, normalized_status)
                or _text(task.get("task_id")) in event_task_ids
            )
        ]
        timestamp = generated_at or datetime.now(timezone.utc)
        return {
            "schema_version": self.schema_version,
            "report_type": "autonomous_rnd_verification_audit",
            "registry_id": str(registry_id),
            "generated_at": timestamp.isoformat(),
            "job_id": str(parent_job.id),
            "job_status": str(parent_job.status or ""),
            "filters": {
                "task_id": normalized_task_id,
                "status": normalized_status,
            },
            "summary": {
                "task_count": len(tasks),
                "timeline_event_count": len(events),
            },
            "tasks": tasks,
            "timeline": events,
        }

    def verify_envelope(
        self,
        envelope: Mapping[str, Any],
        *,
        trusted_public_keys: Mapping[str, str] | None = None,
    ) -> Dict[str, Any]:
        snapshot = envelope.get("snapshot")
        integrity = envelope.get("integrity")
        if not isinstance(snapshot, Mapping) or not isinstance(integrity, Mapping):
            return {"valid": False, "reason": "invalid_envelope"}
        if integrity.get("canonicalization") != self.canonicalization:
            return {"valid": False, "reason": "unsupported_canonicalization"}
        if integrity.get("signature_algorithm") != self.signature_algorithm:
            return {"valid": False, "reason": "unsupported_signature_algorithm"}
        if integrity.get("signature_encoding") != self.signature_encoding:
            return {"valid": False, "reason": "unsupported_signature_encoding"}

        key_id = _text(integrity.get("key_id"))
        public_key_hex = _text(integrity.get("public_key"))
        if not key_id or not public_key_hex:
            return {"valid": False, "reason": "missing_public_key"}
        if trusted_public_keys is not None:
            trusted_public_key = trusted_public_keys.get(key_id)
            if not trusted_public_key or not _constant_time_hex_equal(
                trusted_public_key, public_key_hex
            ):
                return {"valid": False, "reason": "untrusted_key"}

        canonical = self.canonical_bytes(snapshot)
        expected_hash = hashlib.sha256(canonical).hexdigest()
        if not _constant_time_hex_equal(
            str(integrity.get("sha256") or ""), expected_hash
        ):
            return {"valid": False, "reason": "sha256_mismatch"}
        try:
            public_key = Ed25519PublicKey.from_public_bytes(
                bytes.fromhex(public_key_hex)
            )
            public_key.verify(
                bytes.fromhex(str(integrity.get("signature") or "")),
                canonical,
            )
        except (InvalidSignature, ValueError):
            return {"valid": False, "reason": "signature_mismatch"}
        return {
            "valid": True,
            "reason": "verified",
            "job_id": _text(snapshot.get("job_id")),
            "registry_id": _text(snapshot.get("registry_id")),
            "sha256": expected_hash,
            "key_id": key_id,
        }

    @staticmethod
    def canonical_bytes(payload: Mapping[str, Any]) -> bytes:
        return json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")

    @property
    def key_id(self) -> str:
        return settings.AUTONOMOUS_RND_AUDIT_SIGNING_KEY_ID.strip()

    def active_public_key(self) -> Dict[str, Any]:
        public_key = self._public_key_hex(self._private_key())
        return self.public_key_metadata(
            key_id=self.key_id,
            public_key=public_key,
            status="active",
        )

    def public_key_metadata(
        self,
        *,
        key_id: str,
        public_key: str,
        status: str,
    ) -> Dict[str, Any]:
        return {
            "key_id": key_id,
            "algorithm": self.signature_algorithm,
            "encoding": self.signature_encoding,
            "public_key": public_key,
            "fingerprint_sha256": hashlib.sha256(bytes.fromhex(public_key)).hexdigest(),
            "status": "active",
        }

    def _private_key(self) -> Ed25519PrivateKey:
        encoded = settings.AUTONOMOUS_RND_AUDIT_SIGNING_PRIVATE_KEY
        if encoded:
            try:
                padded = encoded.strip() + "=" * (-len(encoded.strip()) % 4)
                seed = base64.urlsafe_b64decode(padded)
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    "AUTONOMOUS_RND_AUDIT_SIGNING_PRIVATE_KEY must be base64 encoded"
                ) from exc
            if len(seed) != 32:
                raise ValueError(
                    "AUTONOMOUS_RND_AUDIT_SIGNING_PRIVATE_KEY must decode to 32 bytes"
                )
        else:
            seed = hashlib.sha256(
                (
                    "knowledgeops:verification-audit:ed25519:v1:"
                    f"{settings.SECRET_KEY}"
                ).encode("utf-8")
            ).digest()
        return Ed25519PrivateKey.from_private_bytes(seed)

    @staticmethod
    def _public_key_hex(private_key: Ed25519PrivateKey) -> str:
        return (
            private_key.public_key()
            .public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw,
            )
            .hex()
        )

    @staticmethod
    def _task_has_status(task: Mapping[str, Any], status: str) -> bool:
        return any(
            (_text(task.get(field)) or "").lower() == status
            for field in (
                "evidence_status",
                "launch_status",
                "job_status",
                "approval_status",
                "reconciliation_status",
            )
        )

    @staticmethod
    def _safe_task(task: Mapping[str, Any]) -> Dict[str, Any]:
        budget = task.get("budget") if isinstance(task.get("budget"), Mapping) else {}
        return {
            "task_id": _text(task.get("task_id")),
            "evidence_id": _text(task.get("evidence_id")),
            "evidence_status": _text(task.get("evidence_status")),
            "priority": _text(task.get("priority")),
            "priority_score": _number(task.get("priority_score")),
            "required_checks": [
                str(item)
                for item in task.get("required_checks", [])
                if isinstance(item, str)
            ],
            "launch_status": _text(task.get("launch_status")),
            "job_status": _text(task.get("job_status")),
            "approval_status": _text(task.get("approval_status")),
            "reconciliation_status": _text(task.get("reconciliation_status")),
            "reconciliation_recorded_at": _text(task.get("reconciliation_recorded_at")),
            "experiment_plan_id": _text(task.get("experiment_plan_id")),
            "experiment_run_id": _text(task.get("experiment_run_id")),
            "agent_job_id": _text(task.get("agent_job_id")),
            "audit_id": _text(task.get("audit_id")),
            "budget": {
                "repeat_count": _number(budget.get("repeat_count")),
                "timeout_seconds": _number(budget.get("timeout_seconds")),
                "max_runtime_minutes": _number(budget.get("max_runtime_minutes")),
                "budget_limit": _number(budget.get("budget_limit")),
            },
        }

    @staticmethod
    def _safe_event(event: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            "event_id": _text(event.get("event_id")),
            "task_id": _text(event.get("task_id")),
            "event_type": _text(event.get("event_type")),
            "at": _text(event.get("at")),
            "actor": _text(event.get("actor")),
            "label": _text(event.get("label")),
            "status": _text(event.get("status")),
            "entity_type": _text(event.get("entity_type")),
            "entity_id": _text(event.get("entity_id")),
        }


def _text(value: Any) -> str | None:
    normalized = str(value or "").strip()
    return normalized or None


def _number(value: Any) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value


def _constant_time_hex_equal(left: str, right: str) -> bool:
    try:
        left_bytes = bytes.fromhex(str(left))
        right_bytes = bytes.fromhex(str(right))
    except ValueError:
        return False
    return hmac.compare_digest(left_bytes, right_bytes)


autonomous_rnd_verification_audit_service = AutonomousRnDVerificationAuditService()
