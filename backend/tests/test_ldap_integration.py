"""LDAP: what the directory is asked, what its answers are allowed to do.

Runs against ldap3's in-memory server (``MOCK_SYNC``), so binds, searches and
filters are the real ones; only the socket is missing. LDAP had no tests at
all, which is how it shipped without certificate verification, with the
username interpolated into the search filter, and with a login path that
reactivated deactivated users.
"""

import ssl

import pytest
from ldap3 import MOCK_SYNC, OFFLINE_SLAPD_2_4, Connection, Server
from ldap3.core.exceptions import LDAPSocketOpenError

from app.core.config import settings
from app.models.user import User
from app.services import ldap_service as ldap_module
from app.services.auth_service import AuthService
from app.services.ldap_service import (
    LdapMisconfigured,
    LdapService,
    LdapUnavailable,
    normalize_dn,
)

pytestmark = pytest.mark.unit

BASE = "dc=example,dc=com"
SERVICE_DN = f"cn=svc,{BASE}"
ALICE_DN = f"uid=alice,ou=People,{BASE}"
ADMINS_DN = f"cn=admins,ou=Groups,{BASE}"


class Directory:
    """One in-memory LDAP server that every connection in a test shares."""

    def __init__(self):
        self.server = Server("mock", get_info=OFFLINE_SLAPD_2_4)
        self.down = False
        self.binds = []
        seed = self._connection(None, None)
        self.add = seed.strategy.add_entry
        self.add(
            SERVICE_DN,
            {"userPassword": "svc-secret", "objectClass": "person", "sn": "svc"},
        )
        self.add(
            ALICE_DN,
            {
                "userPassword": "alice-secret",
                "objectClass": "inetOrgPerson",
                "uid": "alice",
                "mail": "alice@example.com",
                "cn": "Alice Adams",
                "sn": "Adams",
                "displayName": "Alice Adams",
            },
        )
        self.add(
            f"uid=bob,ou=People,{BASE}",
            {
                "userPassword": "bob-secret",
                "objectClass": "inetOrgPerson",
                "uid": "bob",
                "mail": "bob@example.com",
                "cn": "Bob",
                "sn": "B",
            },
        )
        self.add(
            ADMINS_DN,
            {"objectClass": "groupOfNames", "cn": "admins", "member": [ALICE_DN]},
        )

    def _connection(self, bind_dn, password):
        return Connection(
            self.server,
            user=bind_dn,
            password=password,
            client_strategy=MOCK_SYNC,
            raise_exceptions=True,
        )

    def new_connection(self, bind_dn, password):
        if self.down:
            raise LDAPSocketOpenError("connection refused")
        self.binds.append(bind_dn)
        return self._connection(bind_dn, password)


@pytest.fixture
def directory(monkeypatch):
    directory = Directory()
    for name, value in {
        "LDAP_ENABLED": True,
        "LDAP_URI": "ldaps://directory.example.com",
        "LDAP_BASE_DN": BASE,
        "LDAP_BIND_DN": SERVICE_DN,
        "LDAP_BIND_PASSWORD": "svc-secret",
        "LDAP_USER_DN_TEMPLATE": None,
        "LDAP_USER_SEARCH_FILTER": "(uid={username})",
        "LDAP_USER_ATTRIBUTES": "uid,mail,cn,displayName,memberOf",
        "LDAP_START_TLS": False,
        "LDAP_ALLOW_PLAINTEXT": False,
        "LDAP_INSECURE_SKIP_TLS_VERIFY": False,
        "LDAP_CA_CERT_FILE": None,
        "LDAP_ADMIN_GROUP_DNS": None,
        "LDAP_VIEWER_GROUP_DNS": None,
        "LDAP_GROUP_SEARCH_BASE": None,
        "LDAP_DEFAULT_EMAIL_DOMAIN": None,
        "LDAP_CREATE_USER_ON_LOGIN": True,
        "LDAP_SYNC_ON_LOGIN": True,
        "LDAP_LOCAL_FALLBACK_WHEN_UNAVAILABLE": False,
        "LDAP_LINK_LOCAL_USERS_BY_USERNAME": False,
    }.items():
        monkeypatch.setattr(settings, name, value, raising=False)

    def _new_connection(self, bind_dn, password):
        return directory.new_connection(bind_dn, password)

    monkeypatch.setattr(LdapService, "_new_connection", _new_connection)
    return directory


@pytest.fixture
def ldap():
    return ldap_module.ldap_service


# ---------------------------------------------------------------------------
# Transport: the certificate is verified, and a password is not sent in clear
# ---------------------------------------------------------------------------


class TestTransport:
    def test_the_certificate_is_verified_by_default(self, directory, ldap):
        assert ldap._tls().validate == ssl.CERT_REQUIRED
        # What reaches ldap3: with tls=None it validated nothing.
        assert ldap._make_server().tls.validate == ssl.CERT_REQUIRED
        assert ldap.verifies_certificates() is True

    def test_starttls_is_verified_too(self, directory, ldap, monkeypatch):
        monkeypatch.setattr(settings, "LDAP_URI", "ldap://directory.example.com")
        monkeypatch.setattr(settings, "LDAP_START_TLS", True)
        assert ldap.transport() == "starttls"
        assert ldap._make_server().tls.validate == ssl.CERT_REQUIRED

    def test_skipping_verification_takes_the_flag(self, directory, ldap, monkeypatch):
        monkeypatch.setattr(settings, "LDAP_INSECURE_SKIP_TLS_VERIFY", True)
        assert ldap._tls().validate == ssl.CERT_NONE
        assert ldap.verifies_certificates() is False

    def test_a_private_ca_can_be_named(self, directory, ldap, monkeypatch, tmp_path):
        bundle = tmp_path / "ca.pem"
        bundle.write_text("-----BEGIN CERTIFICATE-----\n")
        monkeypatch.setattr(settings, "LDAP_CA_CERT_FILE", str(bundle))
        assert ldap._tls().ca_certs_file == str(bundle)

    def test_plaintext_is_refused_unless_allowed(self, directory, ldap, monkeypatch):
        monkeypatch.setattr(settings, "LDAP_URI", "ldap://directory.example.com")
        assert ldap.transport() == "plaintext"
        assert any("not encrypted" in p for p in ldap.problems())
        with pytest.raises(LdapMisconfigured):
            ldap.authenticate("alice", "alice-secret")
        assert directory.binds == [], "no password left the process"

        monkeypatch.setattr(settings, "LDAP_ALLOW_PLAINTEXT", True)
        assert ldap.problems() == []
        assert ldap.authenticate("alice", "alice-secret").username == "alice"


# ---------------------------------------------------------------------------
# Authentication: the directory's answers
# ---------------------------------------------------------------------------


class TestAuthenticate:
    def test_the_right_password_returns_the_entry(self, directory, ldap):
        user = ldap.authenticate("alice", "alice-secret")
        assert (user.username, user.email, user.dn) == (
            "alice",
            "alice@example.com",
            ALICE_DN,
        )
        assert user.full_name == "Alice Adams"

    def test_a_wrong_password_is_none(self, directory, ldap):
        assert ldap.authenticate("alice", "nope-nope") is None

    def test_an_unknown_user_is_none(self, directory, ldap):
        assert ldap.authenticate("nobody", "whatever") is None

    @pytest.mark.parametrize("password", ["", "   ", None])
    def test_a_blank_password_never_reaches_the_directory(
        self, directory, ldap, password
    ):
        # A bind with a DN and no password is an anonymous bind on many
        # servers -- and anonymous binds succeed.
        assert ldap.authenticate("alice", password) is None
        assert directory.binds == []

    @pytest.mark.parametrize(
        "username", ["*", "a*", "alice)(uid=*", "*)(objectClass=*", "alice\x00"]
    )
    def test_the_username_cannot_rewrite_the_filter(self, directory, ldap, username):
        # Interpolated raw, "*" matched every user and ")(" changed the query.
        assert ldap.authenticate(username, "alice-secret") is None

    def test_a_name_matching_two_entries_is_not_a_login(
        self, directory, ldap, monkeypatch
    ):
        monkeypatch.setattr(
            settings, "LDAP_USER_SEARCH_FILTER", "(|(uid={username})(uid=bob))"
        )
        assert ldap.authenticate("alice", "alice-secret") is None

    def test_the_dn_template_mode_binds_directly(self, directory, ldap, monkeypatch):
        monkeypatch.setattr(settings, "LDAP_BIND_DN", None)
        monkeypatch.setattr(
            settings, "LDAP_USER_DN_TEMPLATE", f"uid={{username}},ou=People,{BASE}"
        )
        assert ldap.authenticate("alice", "alice-secret").dn == ALICE_DN
        assert ldap.authenticate("alice", "wrong-one") is None
        # And a name cannot climb out of its RDN.
        assert ldap.authenticate(f"alice,ou=People,{BASE}", "alice-secret") is None

    def test_an_unreachable_directory_is_not_a_wrong_password(self, directory, ldap):
        directory.down = True
        with pytest.raises(LdapUnavailable):
            ldap.authenticate("alice", "alice-secret")

    def test_a_refused_service_account_is_our_problem(
        self, directory, ldap, monkeypatch
    ):
        monkeypatch.setattr(settings, "LDAP_BIND_PASSWORD", "wrong")
        with pytest.raises(LdapUnavailable, match="service account"):
            ldap.authenticate("alice", "alice-secret")


# ---------------------------------------------------------------------------
# Groups and roles
# ---------------------------------------------------------------------------


class TestRoles:
    def test_no_mapping_means_no_opinion(self, directory, ldap):
        assert ldap.map_role([ADMINS_DN]) is None

    def test_dns_are_compared_as_dns(self, directory, ldap, monkeypatch):
        monkeypatch.setattr(
            settings, "LDAP_ADMIN_GROUP_DNS", "CN=Admins, OU=Groups, DC=Example, DC=Com"
        )
        assert normalize_dn("CN=Admins, OU=Groups,DC=example,DC=com") == normalize_dn(
            ADMINS_DN
        )
        assert ldap.map_role([ADMINS_DN]) == "admin"
        assert ldap.map_role([f"cn=staff,ou=Groups,{BASE}"]) == "user"

    def test_several_group_dns_are_separated_by_semicolons(
        self, directory, ldap, monkeypatch
    ):
        # A DN is comma-separated, so a comma cannot separate DNs. These
        # settings were split on commas: no group ever matched.
        staff = f"cn=staff,ou=Groups,{BASE}"
        monkeypatch.setattr(settings, "LDAP_ADMIN_GROUP_DNS", f"{staff}; {ADMINS_DN}")
        monkeypatch.setattr(
            settings, "LDAP_VIEWER_GROUP_DNS", f"cn=guests,ou=Groups,{BASE}"
        )
        assert ldap.map_role([ADMINS_DN]) == "admin"
        assert ldap.map_role([f"cn=guests,ou=Groups,{BASE}"]) == "viewer"
        assert ldap.map_role([f"ou=Groups,{BASE}"]) == "user"  # a fragment is no group

    def test_groups_are_found_by_searching_for_the_member(
        self, directory, ldap, monkeypatch
    ):
        # No memberOf on the entry (OpenLDAP without the overlay).
        assert ldap.authenticate("alice", "alice-secret").groups == []
        monkeypatch.setattr(settings, "LDAP_GROUP_SEARCH_BASE", f"ou=Groups,{BASE}")
        monkeypatch.setattr(settings, "LDAP_ADMIN_GROUP_DNS", ADMINS_DN)
        user = ldap.authenticate("alice", "alice-secret")
        assert [normalize_dn(g) for g in user.groups] == [normalize_dn(ADMINS_DN)]
        assert ldap.map_role(user.groups) == "admin"
        assert ldap.map_role(ldap.authenticate("bob", "bob-secret").groups) == "user"


# ---------------------------------------------------------------------------
# The diagnostic an admin runs
# ---------------------------------------------------------------------------


class TestDiagnose:
    def test_a_working_setup_reports_each_step(self, directory, ldap):
        report = ldap.diagnose("alice")
        assert report["ok"] is True
        assert [s["step"] for s in report["steps"]] == [
            "enabled",
            "settings",
            "transport",
            "service_account",
            "user_lookup",
        ]
        assert report["user"]["email"] == "alice@example.com"
        assert report["user"]["role"] is None

    def test_it_says_which_step_failed(self, directory, ldap, monkeypatch):
        directory.down = True
        report = ldap.diagnose("alice")
        assert report["ok"] is False
        assert report["steps"][-1]["step"] == "service_account"
        assert "connection refused" in report["steps"][-1]["message"]

        directory.down = False
        report = ldap.diagnose("nobody")
        assert report["ok"] is False and report["steps"][-1]["step"] == "user_lookup"

        monkeypatch.setattr(settings, "LDAP_ENABLED", False)
        assert [s["step"] for s in ldap.diagnose()["steps"]] == ["enabled"]

    def test_a_user_who_could_not_be_given_an_account_is_reported(
        self, directory, ldap
    ):
        directory.add(
            f"uid=carol,ou=People,{BASE}",
            {"objectClass": "inetOrgPerson", "uid": "carol", "cn": "Carol", "sn": "C"},
        )
        report = ldap.diagnose("carol")
        assert report["ok"] is False
        assert report["steps"][-1]["step"] == "email"


# ---------------------------------------------------------------------------
# Login: what a directory answer is allowed to do to an account
# ---------------------------------------------------------------------------


async def _local(db, auth, username, email, *, password="local-secret", **fields):
    user = User(
        username=username,
        email=email,
        hashed_password=auth.hash_password(password),
        is_active=fields.pop("is_active", True),
        is_verified=True,
        role=fields.pop("role", "user"),
        auth_provider=fields.pop("auth_provider", "local"),
        **fields,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user


class TestLogin:
    async def test_a_directory_user_is_given_an_account(self, directory, db_session):
        auth = AuthService()
        user = await auth.authenticate_user("alice", "alice-secret", db_session)
        assert user is not None
        assert (user.username, user.email, user.auth_provider) == (
            "alice",
            "alice@example.com",
            "ldap",
        )
        assert user.auth_subject == ALICE_DN and user.role == "user"
        # And the second login is the same account.
        again = await auth.authenticate_user("alice", "alice-secret", db_session)
        assert again.id == user.id

    async def test_a_wrong_directory_password_is_refused(self, directory, db_session):
        assert (
            await AuthService().authenticate_user("alice", "nope-nope", db_session)
            is None
        )

    async def test_a_deactivated_user_stays_deactivated(self, directory, db_session):
        auth = AuthService()
        user = await _local(
            db_session,
            auth,
            "alice",
            "alice@example.com",
            auth_provider="ldap",
            is_active=False,
        )
        # The directory still accepts this password. It used to set is_active
        # back to true, so deactivating an LDAP user did nothing.
        assert await auth.authenticate_user("alice", "alice-secret", db_session) is None
        await db_session.refresh(user)
        assert user.is_active is False

    async def test_a_directory_no_is_final(self, directory, db_session):
        auth = AuthService()
        await _local(
            db_session,
            auth,
            "alice",
            "alice@example.com",
            auth_provider="ldap",
            password="old-local-password",
        )
        # The stored password used to be tried after every refusal -- so an
        # account disabled in the directory kept working with it.
        assert (
            await auth.authenticate_user("alice", "old-local-password", db_session)
            is None
        )

    async def test_an_outage_does_not_open_the_stored_password_by_default(
        self, directory, db_session, monkeypatch
    ):
        auth = AuthService()
        await _local(
            db_session,
            auth,
            "alice",
            "alice@example.com",
            auth_provider="ldap",
            password="old-local-password",
        )
        directory.down = True
        assert (
            await auth.authenticate_user("alice", "old-local-password", db_session)
            is None
        )
        monkeypatch.setattr(settings, "LDAP_LOCAL_FALLBACK_WHEN_UNAVAILABLE", True)
        user = await auth.authenticate_user("alice", "old-local-password", db_session)
        assert user is not None and user.username == "alice"
        # Still only for the right password.
        assert await auth.authenticate_user("alice", "wrong-one", db_session) is None

    async def test_a_local_account_signs_in_without_the_directory(
        self, directory, db_session
    ):
        auth = AuthService()
        await _local(db_session, auth, "dave", "dave@example.com")
        directory.down = True  # would raise if it were asked
        user = await auth.authenticate_user("dave", "local-secret", db_session)
        assert user.username == "dave" and user.auth_provider == "local"

    async def test_a_shared_username_does_not_take_over_a_local_account(
        self, directory, db_session, monkeypatch
    ):
        auth = AuthService()
        local = await _local(
            db_session, auth, "alice", "someone-else@example.org", role="admin"
        )
        # The directory's alice knows her own password, not the local one.
        assert await auth.authenticate_user("alice", "alice-secret", db_session) is None
        await db_session.refresh(local)
        assert local.auth_provider == "local" and local.role == "admin"

        monkeypatch.setattr(settings, "LDAP_LINK_LOCAL_USERS_BY_USERNAME", True)
        linked = await auth.authenticate_user("alice", "alice-secret", db_session)
        assert linked.id == local.id and linked.auth_provider == "ldap"

    async def test_matching_emails_link_and_keep_the_role(self, directory, db_session):
        auth = AuthService()
        local = await _local(
            db_session, auth, "alice", "Alice@Example.com", role="admin"
        )
        user = await auth.authenticate_user("alice", "alice-secret", db_session)
        assert user.id == local.id and user.auth_provider == "ldap"
        # No group mapping is configured: the role is not the directory's to
        # change. It used to become "user".
        assert user.role == "admin"

    async def test_a_configured_mapping_does_set_the_role(
        self, directory, db_session, monkeypatch
    ):
        monkeypatch.setattr(settings, "LDAP_GROUP_SEARCH_BASE", f"ou=Groups,{BASE}")
        monkeypatch.setattr(settings, "LDAP_ADMIN_GROUP_DNS", ADMINS_DN)
        auth = AuthService()
        assert (
            await auth.authenticate_user("alice", "alice-secret", db_session)
        ).role == "admin"
        assert (
            await auth.authenticate_user("bob", "bob-secret", db_session)
        ).role == "user"

    async def test_provisioning_can_be_turned_off(
        self, directory, db_session, monkeypatch
    ):
        monkeypatch.setattr(settings, "LDAP_CREATE_USER_ON_LOGIN", False)
        assert (
            await AuthService().authenticate_user("alice", "alice-secret", db_session)
            is None
        )

    async def test_a_user_with_no_email_is_refused_not_a_500(
        self, directory, db_session
    ):
        directory.add(
            f"uid=carol,ou=People,{BASE}",
            {
                "userPassword": "carol-secret",
                "objectClass": "inetOrgPerson",
                "uid": "carol",
                "cn": "Carol",
                "sn": "C",
            },
        )
        assert (
            await AuthService().authenticate_user("carol", "carol-secret", db_session)
            is None
        )

    async def test_an_email_another_account_holds_is_not_taken(
        self, directory, db_session
    ):
        auth = AuthService()
        await _local(db_session, auth, "squatter", "alice@example.com")
        ldap_alice = await _local(
            db_session, auth, "alice", "old-alice@example.com", auth_provider="ldap"
        )
        user = await auth.authenticate_user("alice", "alice-secret", db_session)
        assert user.id == ldap_alice.id
        assert user.email == "old-alice@example.com"


def test_ldap_is_not_called_on_the_event_loop():
    import inspect

    source = inspect.getsource(AuthService.authenticate_user)
    assert "authenticate_async(" in source
    assert "ldap_service.authenticate(" not in source


# ---------------------------------------------------------------------------
# The admin surface
# ---------------------------------------------------------------------------


class TestAdminEndpoints:
    def test_status_says_what_is_configured_and_what_is_wrong(
        self, directory, client, admin_headers, monkeypatch
    ):
        status = client.get("/api/v1/admin/ldap/status", headers=admin_headers).json()
        assert status["enabled"] and status["configured"]
        assert status["transport"] == "ldaps" and status["verifies_certificates"]
        assert status["has_service_account"] and status["problems"] == []
        assert status["role_mapping_configured"] is False

        monkeypatch.setattr(settings, "LDAP_URI", "ldap://directory.example.com")
        status = client.get("/api/v1/admin/ldap/status", headers=admin_headers).json()
        assert status["transport"] == "plaintext"
        assert any("not encrypted" in p for p in status["problems"])

    def test_the_test_route_looks_a_user_up_without_a_password(
        self, directory, client, admin_headers
    ):
        response = client.post(
            "/api/v1/admin/ldap/test", headers=admin_headers, json={"username": "alice"}
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["ok"] is True
        assert body["user"]["dn"] == ALICE_DN
        assert body["steps"][-1]["step"] == "user_lookup"

    def test_the_test_route_reports_an_outage_as_a_step(
        self, directory, client, admin_headers
    ):
        directory.down = True
        body = client.post(
            "/api/v1/admin/ldap/test", headers=admin_headers, json={}
        ).json()
        assert body["ok"] is False
        assert body["steps"][-1] == {
            "step": "service_account",
            "ok": False,
            "message": "LDAPSocketOpenError: connection refused",
        }

    def test_only_an_admin_may_ask(self, directory, client, auth_headers):
        for method, path in (("get", "status"), ("post", "test"), ("post", "import")):
            response = getattr(client, method)(
                f"/api/v1/admin/ldap/{path}",
                headers=auth_headers,
                **({} if method == "get" else {"json": {}}),
            )
            assert response.status_code == 403, (path, response.status_code)

    def test_a_dry_run_import_lists_the_directory_and_changes_nothing(
        self, directory, client, admin_headers
    ):
        response = client.post(
            "/api/v1/admin/ldap/import",
            headers=admin_headers,
            json={"search_filter": "(objectClass=inetOrgPerson)", "dry_run": True},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["created"] == 2 and body["errors"] == 0
        assert {row["username"] for row in body["rows"]} == {"alice", "bob"}

    def test_an_import_during_an_outage_says_so(self, directory, client, admin_headers):
        directory.down = True
        response = client.post(
            "/api/v1/admin/ldap/import", headers=admin_headers, json={"dry_run": True}
        )
        assert response.status_code == 502
        assert "could not be searched" in response.json()["detail"]
