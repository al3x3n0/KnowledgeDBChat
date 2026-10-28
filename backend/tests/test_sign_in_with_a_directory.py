"""Signing in where accounts come from a directory.

Three things a directory deployment got wrong at the sign-in page:

* anyone could register a local account, with an email nothing checked, and
  linking trusted a matching email -- so a stranger could register a
  directory person's name and email before their first sign-in, use the
  account, and have the directory hand it to them;
* "Sign up" was offered regardless;
* an unreachable directory answered "Incorrect username or password".

There is no mail to verify an address with, so a self-registered account
links only by an email an admin has vouched for (``POST /admin/ldap/vouch``).
"""

import pytest

from app.core.config import settings
from app.models.user import User
from app.modules.identity.application import directory_accounts, directory_import
from app.services.auth_service import AuthService

pytestmark = pytest.mark.unit


async def _registered(db, username, email, *, origin="self_registration"):
    return await AuthService().create_user(
        username=username,
        email=email,
        password="registered-password-1",
        db=db,
        origin=origin,
    )


class TestASelfRegisteredAccountIsNotLinkedUnvouched:
    async def test_not_by_the_matching_email_a_stranger_typed(
        self, directory, db_session
    ):
        squatter = await _registered(db_session, "alice", "alice@example.com")

        user = await AuthService().authenticate_user(
            "alice", "alice-secret", db_session
        )

        assert user is None
        await db_session.refresh(squatter)
        assert squatter.auth_provider == "local"

    async def test_nor_by_name_when_names_are_trusted(
        self, directory, db_session, monkeypatch
    ):
        monkeypatch.setattr(
            settings, "LDAP_LINK_LOCAL_USERS_BY_USERNAME", True, raising=False
        )
        await _registered(db_session, "alice", "someone@example.com")

        assert (
            await AuthService().authenticate_user("alice", "alice-secret", db_session)
            is None
        )

    async def test_an_account_an_admin_made_still_links_by_email(
        self, directory, db_session
    ):
        """The control: provenance decides, not the email alone."""
        made = await _registered(db_session, "alice", "alice@example.com", origin=None)

        user = await AuthService().authenticate_user(
            "alice", "alice-secret", db_session
        )

        assert user is not None and user.id == made.id
        assert user.auth_provider == "ldap"

    async def test_the_import_refuses_it_and_says_why(
        self, directory, ldap, db_session
    ):
        await _registered(db_session, "alice", "alice@example.com")
        entries = await ldap.search_users_async(
            search_filter="(objectClass=inetOrgPerson)", limit=10
        )

        result = await directory_import.import_entries(
            db_session, entries, dry_run=True
        )

        (row,) = [r for r in result.rows if r.username == "alice"]
        assert row.action == "refused" and "registered from the" in row.error

    async def test_registration_records_where_the_account_came_from(self, db_session):
        user = await _registered(db_session, "dora", "dora@example.com")

        assert directory_accounts.self_registered(user)
        assert not directory_accounts.self_registered(User(auth_metadata=None))


async def _vouched(db, user, *, by):
    return await directory_accounts.vouch(db, user.username, by=by)


class TestAVouchedForAccountLinksByItsEmail:
    async def test_it_links_once_an_admin_vouches(
        self, directory, db_session, admin_user
    ):
        registered = await _registered(db_session, "alice", "alice@example.com")
        await _vouched(db_session, registered, by=admin_user)

        user = await AuthService().authenticate_user(
            "alice", "alice-secret", db_session
        )

        assert user is not None and user.id == registered.id
        assert user.auth_provider == "ldap"

    async def test_never_by_name_alone(
        self, directory, db_session, admin_user, monkeypatch
    ):
        monkeypatch.setattr(
            settings, "LDAP_LINK_LOCAL_USERS_BY_USERNAME", True, raising=False
        )
        registered = await _registered(db_session, "alice", "someone@example.com")
        await _vouched(db_session, registered, by=admin_user)

        assert (
            await AuthService().authenticate_user("alice", "alice-secret", db_session)
            is None
        )

    async def test_a_vouch_does_not_cover_a_later_email(
        self, directory, db_session, admin_user
    ):
        registered = await _registered(db_session, "alice", "old@example.com")
        await _vouched(db_session, registered, by=admin_user)
        registered.email = "alice@example.com"
        await db_session.commit()

        assert (
            await AuthService().authenticate_user("alice", "alice-secret", db_session)
            is None
        )

    async def test_a_withdrawn_vouch_links_nothing(
        self, directory, db_session, admin_user
    ):
        registered = await _registered(db_session, "alice", "alice@example.com")
        await _vouched(db_session, registered, by=admin_user)
        await directory_accounts.vouch(
            db_session, "alice", by=admin_user, verified=False
        )

        await db_session.refresh(registered)
        assert registered.is_verified is False
        assert "vouched" not in registered.auth_metadata
        assert (
            await AuthService().authenticate_user("alice", "alice-secret", db_session)
            is None
        )

    async def test_the_vouch_records_who_and_which_address(
        self, db_session, admin_user
    ):
        registered = await _registered(db_session, "dora", "Dora@Example.com")

        user = await _vouched(db_session, registered, by=admin_user)

        vouched = user.auth_metadata["vouched"]
        assert vouched["email"] == "dora@example.com"
        assert vouched["by"] == admin_user.username
        assert user.auth_metadata["origin"] == "self_registration"

    async def test_a_directory_account_is_not_vouched_for(
        self, directory, db_session, admin_user
    ):
        user = await AuthService().authenticate_user(
            "alice", "alice-secret", db_session
        )

        with pytest.raises(directory_accounts.AccountRefused):
            await directory_accounts.vouch(db_session, user.username, by=admin_user)


class TestTheVouchRoute:
    def test_an_admin_vouches(self, client, admin_headers, test_user):
        response = client.post(
            "/api/v1/admin/ldap/vouch",
            headers=admin_headers,
            json={"username": test_user.username},
        )

        assert response.status_code == 200, response.text
        assert response.json()["is_verified"] is True

    def test_nobody_else_may(self, client, auth_headers, test_user):
        response = client.post(
            "/api/v1/admin/ldap/vouch",
            headers=auth_headers,
            json={"username": test_user.username},
        )

        assert response.status_code == 403

    def test_an_unknown_name_is_404(self, client, admin_headers):
        response = client.post(
            "/api/v1/admin/ldap/vouch",
            headers=admin_headers,
            json={"username": "nobody-here"},
        )

        assert response.status_code == 404


class TestRegistration:
    def test_off_by_default_where_ldap_is_on(self, client, monkeypatch):
        monkeypatch.setattr(settings, "LDAP_ENABLED", True, raising=False)
        monkeypatch.setattr(settings, "AUTH_SELF_REGISTRATION", None, raising=False)

        response = client.post(
            "/api/v1/auth/register",
            json={
                "username": "stranger",
                "email": "stranger@example.com",
                "password": "Stranger-pass-1",
            },
        )

        assert response.status_code == 403
        assert "directory" in response.json()["detail"]

    def test_on_by_default_without_ldap(self, client, monkeypatch):
        monkeypatch.setattr(settings, "LDAP_ENABLED", False, raising=False)
        monkeypatch.setattr(settings, "AUTH_SELF_REGISTRATION", None, raising=False)

        response = client.post(
            "/api/v1/auth/register",
            json={
                "username": "newcomer",
                "email": "newcomer@example.com",
                "password": "Newcomer-pass-1",
            },
        )

        assert response.status_code == 200, response.text

    def test_the_setting_decides_when_it_is_set(self, client, monkeypatch):
        monkeypatch.setattr(settings, "LDAP_ENABLED", True, raising=False)
        monkeypatch.setattr(settings, "AUTH_SELF_REGISTRATION", True, raising=False)

        body = client.get("/api/v1/auth/options").json()

        assert body["self_registration"] is True

    def test_the_sign_in_page_can_ask_without_signing_in(
        self, directory, client, monkeypatch
    ):
        monkeypatch.setattr(settings, "AUTH_SELF_REGISTRATION", None, raising=False)

        response = client.get("/api/v1/auth/options")

        assert response.status_code == 200
        assert response.json() == {"self_registration": False, "directory": True}


class TestAnOutageIsNotAWrongPassword:
    def test_the_directory_down_answers_503(self, directory, client):
        directory.down = True

        response = client.post(
            "/api/v1/auth/login", json={"username": "alice", "password": "x"}
        )

        assert response.status_code == 503
        assert "could not be reached" in response.json()["detail"]
        assert response.headers.get("retry-after") == "30"

    def test_a_wrong_password_is_still_401(self, directory, client):
        response = client.post(
            "/api/v1/auth/login", json={"username": "alice", "password": "wrong"}
        )

        assert response.status_code == 401
