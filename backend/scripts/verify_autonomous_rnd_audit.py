#!/usr/bin/env python3
"""Verify an exported R&D audit snapshot against a trusted public-key registry."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import sys
from pathlib import Path
from typing import Any, Dict

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey


def canonical_bytes(payload: Dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def verify(envelope: Dict[str, Any], key_registry: Dict[str, Any]) -> Dict[str, Any]:
    snapshot = envelope.get("snapshot")
    integrity = envelope.get("integrity")
    if not isinstance(snapshot, dict) or not isinstance(integrity, dict):
        return {"valid": False, "reason": "invalid_envelope"}
    if integrity.get("canonicalization") != "json-sort-keys-compact-v1":
        return {"valid": False, "reason": "unsupported_canonicalization"}
    if integrity.get("signature_algorithm") != "ed25519":
        return {"valid": False, "reason": "unsupported_signature_algorithm"}
    if integrity.get("signature_encoding") != "hex":
        return {"valid": False, "reason": "unsupported_signature_encoding"}

    key_id = str(integrity.get("key_id") or "")
    public_key = str(integrity.get("public_key") or "")
    trusted = {
        str(item.get("key_id")): str(item.get("public_key"))
        for item in key_registry.get("keys", [])
        if isinstance(item, dict)
    }
    if key_id not in trusted or not hmac.compare_digest(trusted[key_id], public_key):
        return {"valid": False, "reason": "untrusted_key", "key_id": key_id or None}

    canonical = canonical_bytes(snapshot)
    expected_hash = hashlib.sha256(canonical).hexdigest()
    if not hmac.compare_digest(str(integrity.get("sha256") or ""), expected_hash):
        return {"valid": False, "reason": "sha256_mismatch", "key_id": key_id}
    try:
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_key)).verify(
            bytes.fromhex(str(integrity.get("signature") or "")),
            canonical,
        )
    except (InvalidSignature, ValueError):
        return {"valid": False, "reason": "signature_mismatch", "key_id": key_id}
    return {
        "valid": True,
        "reason": "verified",
        "registry_id": snapshot.get("registry_id"),
        "job_id": snapshot.get("job_id"),
        "sha256": expected_hash,
        "key_id": key_id,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path, help="Exported signed envelope JSON")
    parser.add_argument(
        "--keys",
        type=Path,
        required=True,
        help="Trusted JSON returned by the verification-audit-keys endpoint",
    )
    args = parser.parse_args()
    result = verify(
        json.loads(args.snapshot.read_text(encoding="utf-8")),
        json.loads(args.keys.read_text(encoding="utf-8")),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
