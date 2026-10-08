"""Access tokens are issued and read in one place, with a key nobody else has.

Seven places decoded the JWT by hand and agreed on the signature alone: none
checked the token's type, and the refresh route renewed a deactivated user's
token. And the key itself only ever produced a warning -- an unset
``SECRET_KEY`` in the production compose file arrived as an empty string.
"""

import ast
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from jose import jwt

from app.core import tokens
from app.core.config import (
    DEFAULT_SECRET_KEY,
    PLACEHOLDER_SECRET_KEYS,
    Settings,
    settings,
)

pytestmark = pytest.mark.unit

APP = Path(__file__).resolve().parents[1] / "app"


def _signed(payload: dict, key: str = None) -> str:
    return jwt.encode(payload, key or settings.SECRET_KEY, algorithm=settings.ALGORITHM)


class TestTheDecoder:
    def test_it_reads_what_it_issued(self):
        assert tokens.decode_access_token(tokens.create_access_token("user-1")) == (
            "user-1"
        )

    def test_a_bearer_prefix_is_accepted(self):
        token = tokens.create_access_token("user-1")
        assert tokens.decode_access_token(f"Bearer {token}") == "user-1"

    @pytest.mark.parametrize("token", [None, "", "   ", "not-a-token", "a.b.c", 12345])
    def test_garbage_is_none_not_an_exception(self, token):
        assert tokens.decode_access_token(token) is None

    def test_another_kind_of_token_is_not_a_login(self):
        future = datetime.utcnow() + timedelta(minutes=5)
        # Signed with our key, for our user -- but not an access token. Every
        # hand-rolled decoder accepted this.
        for kind in ("reset", "share", "refresh", None):
            payload = {"sub": "user-1", "exp": future}
            if kind:
                payload["type"] = kind
            assert tokens.decode_access_token(_signed(payload)) is None

    def test_an_expired_token_is_none(self):
        past = datetime.utcnow() - timedelta(minutes=1)
        assert (
            tokens.decode_access_token(
                _signed({"sub": "user-1", "exp": past, "type": "access"})
            )
            is None
        )

    def test_another_key_is_none(self):
        future = datetime.utcnow() + timedelta(minutes=5)
        forged = _signed(
            {"sub": "user-1", "exp": future, "type": "access"}, key="someone-elses-key"
        )
        assert tokens.decode_access_token(forged) is None

    def test_a_token_naming_nobody_is_none(self):
        future = datetime.utcnow() + timedelta(minutes=5)
        assert (
            tokens.decode_access_token(_signed({"exp": future, "type": "access"}))
            is None
        )


def test_nothing_else_decodes_or_signs_tokens():
    offenders = []
    for path in sorted(APP.rglob("*.py")):
        if path.relative_to(APP).as_posix() == "core/tokens.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("decode", "encode")
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "jwt"
            ):
                offenders.append(f"{path.relative_to(APP)}:{node.lineno}")
    assert (
        not offenders
    ), "Tokens are issued and read in app/core/tokens.py only: " + ", ".join(offenders)


class TestRefresh:
    def test_a_valid_token_is_renewed(self, client, auth_headers):
        response = client.post("/api/v1/auth/refresh", headers=auth_headers)
        assert response.status_code == 200, response.text
        assert tokens.decode_access_token(response.json()["access_token"])

    def test_a_bad_token_is_401_not_500(self, client):
        # The route raised 401 inside a try whose last handler caught
        # Exception, so its own refusals came back as "Token refresh failed".
        response = client.post(
            "/api/v1/auth/refresh", headers={"Authorization": "Bearer not-a-token"}
        )
        assert response.status_code == 401

    def test_a_deleted_users_token_is_401(self, client):
        ghost = tokens.create_access_token("00000000-0000-0000-0000-000000000000")
        response = client.post(
            "/api/v1/auth/refresh", headers={"Authorization": f"Bearer {ghost}"}
        )
        assert response.status_code == 401

    async def test_a_deactivated_users_token_is_not_renewed(
        self, client, auth_headers, test_user, db_session
    ):
        test_user.is_active = False
        await db_session.commit()
        response = client.post("/api/v1/auth/refresh", headers=auth_headers)
        assert response.status_code == 403


class TestTheSigningKey:
    def _settings(self, **overrides):
        return Settings(_env_file=None, **overrides)

    def test_an_empty_key_is_refused_even_in_debug(self):
        for key in ("", "   "):
            with pytest.raises(ValueError, match="SECRET_KEY is empty"):
                self._settings(SECRET_KEY=key, DEBUG=True)

    def test_the_placeholder_is_refused_outside_debug(self):
        for key in sorted(PLACEHOLDER_SECRET_KEYS):
            with pytest.raises(ValueError, match="placeholder"):
                self._settings(SECRET_KEY=key, DEBUG=False)

    def test_the_example_files_key_is_a_known_placeholder(self):
        # env.example is copied to .env by `make setup`; whatever it ships is
        # a key the whole internet has.
        example = (APP.parent / "env.example").read_text(encoding="utf-8")
        shipped = [
            line.split("=", 1)[1].strip()
            for line in example.splitlines()
            if line.startswith("SECRET_KEY=")
        ]
        assert shipped and set(shipped) <= PLACEHOLDER_SECRET_KEYS

    def test_the_placeholder_is_tolerated_in_debug(self):
        assert self._settings(SECRET_KEY=DEFAULT_SECRET_KEY, DEBUG=True).SECRET_KEY

    def test_a_real_key_starts(self):
        key = "k" * 48
        assert self._settings(SECRET_KEY=key, DEBUG=False).SECRET_KEY == key
