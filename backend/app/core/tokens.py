"""Issuing and reading the access tokens this API signs.

Seven places decoded the JWT by hand -- the auth dependency, the rate
limiter, two WebSocket helpers, the refresh route and the document download
routes (twice). They agreed on the signature and nothing else: none checked
the token's ``type``, so the day a second kind of token is signed with this
key (a share link, a reset link) each of them would accept it as a login.
``decode_access_token`` is the one reader, and
``tests/test_one_token_decoder.py`` refuses a second.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Optional

from jose import JWTError, jwt

from app.core.config import settings

ACCESS = "access"


def create_access_token(subject: Any) -> str:
    """A signed token naming ``subject`` (a user id), valid for the configured time."""
    expires = datetime.utcnow() + timedelta(
        minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
    )
    return jwt.encode(
        {"sub": str(subject), "exp": expires, "type": ACCESS},
        settings.SECRET_KEY,
        algorithm=settings.ALGORITHM,
    )


def decode_access_token(token: Optional[str]) -> Optional[str]:
    """The user id an access token names, or None.

    None for anything that is not a currently valid access token of ours:
    missing, malformed, expired, signed with another key, or of another type.
    Never raises, so a caller cannot forget a case.
    """
    if not isinstance(token, str):
        return None
    raw = token.strip()
    if raw.lower().startswith("bearer "):
        raw = raw[7:].strip()
    if not raw:
        return None
    try:
        payload = jwt.decode(raw, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
    except JWTError:
        return None
    except Exception:  # a token that is not even a string of the right shape
        return None
    if payload.get("type") != ACCESS:
        return None
    subject = payload.get("sub")
    return str(subject) if subject else None


__all__ = ["ACCESS", "create_access_token", "decode_access_token"]
