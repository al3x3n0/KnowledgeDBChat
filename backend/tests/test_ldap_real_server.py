"""LDAP against a real directory server.

The other LDAP tests run on ldap3's in-memory server, which has no TLS, no
operational attributes (``entryUUID`` is whatever a test writes) and no
overlays (``memberOf`` is whatever a test writes). This file runs the same
code against a real slapd, and skips unless one is named. It needs a
certificate for ``localhost``: the image's bundled CA expired years ago, and
the client rightly refuses it ("certificate has expired").
``scripts/ldap_test_server.sh`` makes one and starts the server (CI's
``ldap-real-server`` job runs exactly that)::

    scripts/ldap_test_server.sh /tmp/ldap-certs
    export TEST_LDAP_CA_FILE=/tmp/ldap-certs/ca.crt
    TEST_LDAP_URI=ldaps://localhost:3636 pytest tests/test_ldap_real_server.py

``ldap://localhost:3389`` runs the same tests over StartTLS.

It writes under ``ou=kdbc-tests`` and removes that subtree before each test,
so point it at a disposable server only. ``TEST_LDAP_ADMIN_DN`` and
``TEST_LDAP_ADMIN_PASSWORD`` default to that image's administrator.
"""

import os
import ssl

import pytest

from app.core.config import settings
from app.modules.identity.application import directory_sync
from app.services.auth_service import AuthService
from app.services.ldap_service import LdapService, LdapUnavailable

URI = os.environ.get("TEST_LDAP_URI", "")
CA_FILE = os.environ.get("TEST_LDAP_CA_FILE", "")
BASE = os.environ.get("TEST_LDAP_BASE_DN", "dc=example,dc=com")
ADMIN_DN = os.environ.get("TEST_LDAP_ADMIN_DN", f"cn=admin,{BASE}")
ADMIN_PASSWORD = os.environ.get("TEST_LDAP_ADMIN_PASSWORD", "admin-secret")

pytestmark = pytest.mark.skipif(
    not URI, reason="TEST_LDAP_URI names no directory server to test against"
)

ROOT = f"ou=kdbc-tests,{BASE}"
PEOPLE = f"ou=People,{ROOT}"
STAFF = f"ou=Staff,{ROOT}"
GROUPS = f"ou=Groups,{ROOT}"
ADMINS = f"cn=admins,{GROUPS}"
EVERYONE = "(objectClass=inetOrgPerson)"


def _person(uid, name):
    return {
        "objectClass": ["inetOrgPerson"],
        "uid": uid,
        "cn": name,
        "sn": name.split()[-1],
        "displayName": name,
        "mail": f"{uid}@example.com",
        "userPassword": f"{uid}-secret",
    }


def _admin():
    """A connection as the directory's administrator, for arranging entries."""
    import ldap3

    tls = ldap3.Tls(validate=ssl.CERT_REQUIRED, ca_certs_file=CA_FILE or None)
    server = ldap3.Server(URI, use_ssl=URI.startswith("ldaps://"), tls=tls)
    conn = ldap3.Connection(
        server, user=ADMIN_DN, password=ADMIN_PASSWORD, raise_exceptions=True
    )
    conn.open()
    if not URI.startswith("ldaps://"):
        conn.start_tls()
    conn.bind()
    return conn


def _wipe(conn):
    import ldap3
    from ldap3.core.exceptions import LDAPNoSuchObjectResult

    try:
        conn.search(ROOT, "(objectClass=*)", ldap3.SUBTREE, attributes=[])
    except LDAPNoSuchObjectResult:
        return
    # Deepest first: a parent cannot be deleted before its children.
    for dn in sorted(
        (str(e.entry_dn) for e in conn.entries), key=lambda d: -d.count(",")
    ):
        conn.delete(dn)


@pytest.fixture
def server(monkeypatch):
    conn = _admin()
    _wipe(conn)
    conn.add(ROOT, ["organizationalUnit"], {"ou": "kdbc-tests"})
    for ou in ("People", "Staff", "Groups"):
        conn.add(f"ou={ou},{ROOT}", ["organizationalUnit"], {"ou": ou})
    for uid, name in (("alice", "Alice Adams"), ("bob", "Bob Brown")):
        attrs = _person(uid, name)
        conn.add(f"uid={uid},{PEOPLE}", attrs.pop("objectClass"), attrs)
    conn.add(
        ADMINS,
        ["groupOfUniqueNames"],
        {"cn": "admins", "uniqueMember": [f"uid=alice,{PEOPLE}"]},
    )

    for name, value in {
        "LDAP_ENABLED": True,
        "LDAP_URI": URI,
        "LDAP_BASE_DN": ROOT,
        "LDAP_BIND_DN": ADMIN_DN,
        "LDAP_BIND_PASSWORD": ADMIN_PASSWORD,
        "LDAP_USER_DN_TEMPLATE": None,
        "LDAP_USER_SEARCH_FILTER": "(uid={username})",
        "LDAP_IMPORT_FILTER": EVERYONE,
        "LDAP_USER_ATTRIBUTES": "uid,mail,cn,displayName,memberOf",
        "LDAP_UNIQUE_ID_ATTRIBUTE": "entryUUID",
        "LDAP_START_TLS": not URI.startswith("ldaps://"),
        "LDAP_ALLOW_PLAINTEXT": False,
        "LDAP_INSECURE_SKIP_TLS_VERIFY": False,
        "LDAP_CA_CERT_FILE": CA_FILE or None,
        "LDAP_ADMIN_GROUP_DNS": ADMINS,
        "LDAP_VIEWER_GROUP_DNS": None,
        "LDAP_GROUP_SEARCH_BASE": None,
        "LDAP_DEFAULT_EMAIL_DOMAIN": None,
        "LDAP_CREATE_USER_ON_LOGIN": True,
        "LDAP_SYNC_ON_LOGIN": True,
        "LDAP_LOCAL_FALLBACK_WHEN_UNAVAILABLE": False,
        "LDAP_LINK_LOCAL_USERS_BY_USERNAME": False,
        "LDAP_SYNC_MAX_DEACTIVATE_FRACTION": 0.5,
        "LDAP_SEARCH_PAGE_SIZE": 200,
    }.items():
        monkeypatch.setattr(settings, name, value, raising=False)
    yield conn
    _wipe(conn)
    conn.unbind()


@pytest.fixture
def ldap():
    return LdapService()


class TestTheConnectionIsVerified:
    def test_the_named_ca_is_trusted(self, server, ldap):
        assert ldap.authenticate("alice", "alice-secret") is not None

    def test_without_it_the_certificate_is_refused(self, server, ldap, monkeypatch):
        monkeypatch.setattr(settings, "LDAP_CA_CERT_FILE", None, raising=False)

        with pytest.raises(LdapUnavailable):
            ldap.authenticate("alice", "alice-secret")

    def test_a_name_the_certificate_does_not_carry_is_refused(
        self, server, ldap, monkeypatch
    ):
        # The same server and port, reached by an address the certificate
        # was not issued for.
        monkeypatch.setattr(
            settings, "LDAP_URI", URI.replace("localhost", "127.0.0.1"), raising=False
        )

        with pytest.raises(LdapUnavailable):
            ldap.authenticate("alice", "alice-secret")


class TestAuthenticate:
    def test_the_entry_carries_the_server_s_own_id_and_groups(self, server, ldap):
        alice = ldap.authenticate("alice", "alice-secret")

        assert alice.dn == f"uid=alice,{PEOPLE}"
        # Generated by slapd, not written by the test.
        assert alice.unique_id and len(alice.unique_id) == 36
        # Maintained by the memberOf overlay, not written by the test.
        assert ldap.map_role(alice.groups) == "admin"

    def test_a_wrong_password_and_an_unknown_name_are_both_no(self, server, ldap):
        assert ldap.authenticate("alice", "wrong") is None
        assert ldap.authenticate("nobody", "x") is None

    def test_a_wildcard_name_matches_nobody(self, server, ldap):
        assert ldap.authenticate("*", "alice-secret") is None

    def test_paging_reads_every_entry(self, server, ldap, monkeypatch):
        monkeypatch.setattr(settings, "LDAP_SEARCH_PAGE_SIZE", 1, raising=False)
        monkeypatch.setattr(settings, "LDAP_GROUP_SEARCH_BASE", GROUPS, raising=False)

        found = ldap.search_users(search_filter=EVERYONE, limit=50)

        assert {e.username for e in found} == {"alice", "bob"}

    def test_the_diagnostic_passes_every_step(self, server, ldap):
        report = ldap.diagnose("alice")

        assert report["ok"], report["steps"]
        assert report["user"]["role"] == "admin"


class TestAccounts:
    async def test_a_move_in_the_directory_keeps_the_account(self, server, db_session):
        auth = AuthService()
        first = await auth.authenticate_user("alice", "alice-secret", db_session)
        assert first.role == "admin"

        server.modify_dn(f"uid=alice,{PEOPLE}", "uid=alice", new_superior=STAFF)
        again = await auth.authenticate_user("alice", "alice-secret", db_session)

        assert again.id == first.id
        assert again.auth_metadata["dn"] == f"uid=alice,{STAFF}"

    async def test_the_sync_deactivates_someone_who_left_and_takes_a_lost_role(
        self, server, db_session
    ):
        auth = AuthService()
        await auth.authenticate_user("alice", "alice-secret", db_session)
        await auth.authenticate_user("bob", "bob-secret", db_session)

        server.delete(f"uid=bob,{PEOPLE}")
        import ldap3

        server.modify(
            ADMINS,
            {"uniqueMember": [(ldap3.MODIFY_REPLACE, [ADMIN_DN])]},
        )

        report = await directory_sync.sync(db_session)

        assert report.deactivated == ["bob"]
        assert report.role_changes == ["alice: admin -> user"]
        # And bob can no longer sign in, since his entry is gone.
        assert await auth.authenticate_user("bob", "bob-secret", db_session) is None
