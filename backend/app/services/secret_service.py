"""
Secrets vault service.
"""

import base64
import hashlib
from typing import Optional

from cryptography.fernet import Fernet, InvalidToken

from app.core.config import settings


class SecretService:
    def __init__(self):
        self._fernet = Fernet(self._get_key())

    def _get_key(self) -> bytes:
        if settings.SECRETS_ENCRYPTION_KEY:
            return settings.SECRETS_ENCRYPTION_KEY.encode("utf-8")
        # Derive a stable 32-byte key from SECRET_KEY.
        digest = hashlib.sha256(settings.SECRET_KEY.encode("utf-8")).digest()
        return base64.urlsafe_b64encode(digest)

    def encrypt(self, value: str) -> str:
        token = self._fernet.encrypt(value.encode("utf-8"))
        return token.decode("utf-8")

    def decrypt(self, token: str) -> Optional[str]:
        try:
            raw = self._fernet.decrypt(token.encode("utf-8"))
            return raw.decode("utf-8")
        except InvalidToken:
            return None

