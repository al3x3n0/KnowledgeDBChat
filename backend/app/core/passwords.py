"""Hashing and checking a password. Kept out of ``AuthService`` so code that
provisions accounts can give one an unusable password without importing the
authentication service, which imports it."""

from __future__ import annotations

import hashlib
import secrets

import bcrypt
from loguru import logger

#: bcrypt reads at most this many bytes of a password.
_BCRYPT_LIMIT = 72


def _password_bytes(password: str | bytes) -> bytes:
    password_bytes = password.encode("utf-8") if isinstance(password, str) else password
    if len(password_bytes) > _BCRYPT_LIMIT:
        logger.debug(
            f"Password exceeds {_BCRYPT_LIMIT} bytes ({len(password_bytes)}), "
            "pre-hashing with SHA256"
        )
        password_bytes = hashlib.sha256(password_bytes).hexdigest().encode("utf-8")
    return password_bytes


def hash_password(password: str | bytes) -> str:
    """A bcrypt hash of this password."""
    return bcrypt.hashpw(_password_bytes(password), bcrypt.gensalt()).decode("utf-8")


def unusable_password_hash() -> str:
    """A hash of a random password nobody knows, for an account that signs in
    elsewhere. The column is required, and an empty or fixed value is one
    somebody could match."""
    return hash_password(secrets.token_urlsafe(32))


def verify_password(plain_password: str | bytes, hashed_password: str | bytes) -> bool:
    """Whether this password is the one hashed."""
    password_bytes = (
        plain_password.encode("utf-8")
        if isinstance(plain_password, str)
        else plain_password
    )
    hashed_bytes = (
        hashed_password.encode("utf-8")
        if isinstance(hashed_password, str)
        else hashed_password
    )
    if bcrypt.checkpw(password_bytes, hashed_bytes):
        return True
    # Hashed after the SHA-256 pre-hash, as hash_password does above the limit.
    if len(password_bytes) > _BCRYPT_LIMIT:
        password_hash = hashlib.sha256(password_bytes).hexdigest().encode("utf-8")
        return bcrypt.checkpw(password_hash, hashed_bytes)
    return False
