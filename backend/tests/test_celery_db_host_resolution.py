"""One DNS lookup per task, not per database operation.

Celery uses NullPool, so every operation opens a new connection and performs a
new DNS lookup. An agent job does hundreds. Under load Docker's embedded
resolver drops one, asyncpg raises `[Errno -3] Temporary failure in name
resolution`, SQLAlchemy poisons the session, and the job dies hundreds of
statements away from the cause -- reported as "This Session's transaction has
been rolled back", which names the wrong thing entirely.

Measured: 14 such failures in celery against 0 in the API, which pools its
connections and resolves once. Two of two substantive runs died this way.

The engine is already built fresh per task, so resolving the host there gives
the same reduction a connection pool would -- without a pool that nothing
disposes of, which is why NullPool is the default in the first place.
"""

import pytest

from app.core.database import _url_with_resolved_host

pytestmark = pytest.mark.unit


class TestItResolvesWhatItShould:
    def test_a_service_name_becomes_an_address(self, monkeypatch):
        monkeypatch.setattr(
            "socket.gethostbyname",
            lambda host: "10.1.2.3" if host == "postgres" else host,
        )

        out = _url_with_resolved_host("postgresql+asyncpg://u:p@postgres:5432/db")

        assert "@10.1.2.3:5432/db" in out

    def test_credentials_and_database_survive(self, monkeypatch):
        """A URL rewrite that dropped the password would fail every
        connection, which is a worse failure than the one being fixed."""
        monkeypatch.setattr("socket.gethostbyname", lambda host: "10.1.2.3")

        out = _url_with_resolved_host(
            "postgresql+asyncpg://user:secret@postgres:5432/knowledge_db"
        )

        # NOT "user:***": SQLAlchemy masks the password in __str__, so
        # accepting the masked form is what let a URL that authenticates as
        # three asterisks pass its own test.
        assert "user:secret" in out, "the real password must survive the rewrite"
        assert "***" not in out
        assert out.endswith("/knowledge_db")
        assert "postgresql+asyncpg" in out


class TestItLeavesAloneWhatItShould:
    def test_localhost_is_not_rewritten(self):
        """Nothing to gain, and a resolver that answers differently for
        loopback would be a change with no upside."""
        url = "postgresql+asyncpg://u:p@localhost:5432/db"

        assert _url_with_resolved_host(url) == url

    def test_a_literal_address_is_not_rewritten(self):
        url = "postgresql+asyncpg://u:p@127.0.0.1:5432/db"

        assert _url_with_resolved_host(url) == url

    def test_an_unresolvable_host_falls_back_to_the_name(self, monkeypatch):
        """asyncpg should still get its own attempt, with the resolver retries
        configured in compose. Failing here would turn a transient lookup
        problem into a hard startup error."""

        def _boom(host):
            raise OSError("no such host")

        monkeypatch.setattr("socket.gethostbyname", _boom)
        url = "postgresql+asyncpg://u:p@postgres:5432/db"

        assert _url_with_resolved_host(url) == url

    def test_a_malformed_url_does_not_raise(self):
        assert _url_with_resolved_host("not a url at all") == "not a url at all"
