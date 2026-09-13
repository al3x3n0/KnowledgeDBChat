import base64
import uuid
from copy import deepcopy

from app.core.config import settings
from app.models.agent_job import AgentJob
from app.services.autonomous_rnd_verification_audit_service import (
    AutonomousRnDVerificationAuditService,
)
from scripts.verify_autonomous_rnd_audit import verify as verify_offline


def _job():
    return AgentJob(id=uuid.uuid4(), status="completed")


def _lifecycle():
    return {
        "tasks": [
            {
                "task_id": "verify-1",
                "evidence_status": "verified",
                "required_checks": ["reproduce"],
                "budget": {"repeat_count": 3},
            }
        ],
        "timeline": [
            {
                "event_id": "event-1",
                "task_id": "verify-1",
                "event_type": "reconciliation_recorded",
                "at": "2026-07-28T12:00:00+00:00",
                "actor": "system",
                "label": "Verification reconciled",
                "status": "verified",
            }
        ],
    }


def test_ed25519_envelope_verifies_and_rejects_tampering():
    service = AutonomousRnDVerificationAuditService()
    envelope = service.build_signed_snapshot(
        registry_id=uuid.uuid4(),
        parent_job=_job(),
        lifecycle=_lifecycle(),
    )
    active = service.active_public_key()

    result = service.verify_envelope(
        envelope,
        trusted_public_keys={active["key_id"]: active["public_key"]},
    )
    assert result["valid"] is True
    assert verify_offline(envelope, {"keys": [active]})["valid"] is True

    tampered = deepcopy(envelope)
    tampered["snapshot"]["job_status"] = "running"
    assert service.verify_envelope(tampered)["reason"] == "sha256_mismatch"
    assert (
        service.verify_envelope(envelope, trusted_public_keys={})["reason"]
        == "untrusted_key"
    )


def test_historical_public_key_remains_usable_after_rotation(monkeypatch):
    service = AutonomousRnDVerificationAuditService()
    original = service.build_signed_snapshot(
        registry_id=uuid.uuid4(),
        parent_job=_job(),
        lifecycle=_lifecycle(),
    )
    original_key_id = original["integrity"]["key_id"]
    original_public_key = original["integrity"]["public_key"]

    monkeypatch.setattr(
        settings,
        "AUTONOMOUS_RND_AUDIT_SIGNING_KEY_ID",
        "knowledgeops-ed25519-v2",
    )
    monkeypatch.setattr(
        settings,
        "AUTONOMOUS_RND_AUDIT_SIGNING_PRIVATE_KEY",
        base64.urlsafe_b64encode(bytes(range(32))).decode("ascii"),
    )
    rotated = service.build_signed_snapshot(
        registry_id=uuid.uuid4(),
        parent_job=_job(),
        lifecycle=_lifecycle(),
    )

    assert rotated["integrity"]["key_id"] == "knowledgeops-ed25519-v2"
    assert rotated["integrity"]["public_key"] != original_public_key
    trusted = {
        original_key_id: original_public_key,
        rotated["integrity"]["key_id"]: rotated["integrity"]["public_key"],
    }
    assert service.verify_envelope(original, trusted_public_keys=trusted)["valid"]
    assert service.verify_envelope(rotated, trusted_public_keys=trusted)["valid"]
