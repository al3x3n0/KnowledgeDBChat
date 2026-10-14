"""Directory accounts between logins: one set of rules for login, import and sync.

Runs against the same in-memory directory as ``test_ldap_integration``, so the
searches, filters and paging are ldap3's own.
"""

import uuid

import pytest

from app.core.config import settings
from app.models.user import User
from app.modules.identity.application import (
    directory_accounts,
    directory_import,
    directory_sync,
    sync_status,
)
from app.services.auth_service import AuthService
from app.services.ldap_service import directory_id
from tests.ldap_fixtures import ADMINS_DN, ALICE_DN, BASE, SERVICE_DN

pytestmark = pytest.mark.unit

ALICE_ID = "11111111-1111-1111-1111-111111111111"
EVERYONE = "(objectClass=inetOrgPerson)"


def _admin_connection(directory):
    conn = directory._connection(SERVICE_DN, "svc-secret")
    conn.bind()
    return conn


def _move_alice(directory, new_dn, *, unique_id=ALICE_ID):
    conn = _admin_connection(directory)
    conn.delete(ALICE_DN)
    directory.add(
        new_dn,
        {
            "userPassword": "alice-secret",
            "objectClass": "inetOrgPerson",
            "uid": "alice",
            "mail": "alice@example.com",
            "cn": "Alice Adams",
            "sn": "Adams",
            **({"entryUUID": unique_id} if unique_id else {}),
        },
    )


def _remove(directory, dn):
    _admin_connection(directory).delete(dn)


@pytest.fixture
def ids(directory, monkeypatch):
    """Alice carries an entryUUID, as an OpenLDAP entry does."""
    _move_alice(directory, ALICE_DN)
    monkeypatch.setattr(
        settings, "LDAP_UNIQUE_ID_ATTRIBUTE", "entryUUID", raising=False
    )
    monkeypatch.setattr(settings, "LDAP_IMPORT_FILTER", EVERYONE, raising=False)
    monkeypatch.setattr(
        settings, "LDAP_SYNC_MAX_DEACTIVATE_FRACTION", 0.5, raising=False
    )
    return directory


async def _local(db, username, email, **fields):
    user = User(
        username=username,
        email=email,
        hashed_password=AuthService().hash_password("local-secret"),
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


async def _entries(ldap):
    return await ldap.search_users_async(search_filter=EVERYONE, limit=100)


# ---------------------------------------------------------------------------
# A stable id
# ---------------------------------------------------------------------------


class TestAStableId:
    async def test_an_account_records_the_entry_id_not_the_dn(self, ids, db_session):
        user = await AuthService().authenticate_user(
            "alice", "alice-secret", db_session
        )

        assert user.auth_subject == ALICE_ID
        assert user.auth_metadata["dn"] == ALICE_DN

    async def test_a_moved_entry_is_still_the_same_account(self, ids, ldap, db_session):
        first = await AuthService().authenticate_user(
            "alice", "alice-secret", db_session
        )
        moved = f"uid=alice,ou=Staff,{BASE}"
        _move_alice(ids, moved)

        again = await AuthService().authenticate_user(
            "alice", "alice-secret", db_session
        )

        assert again.id == first.id
        assert again.auth_metadata["dn"] == moved

    async def test_an_account_from_before_ids_is_found_by_its_dn(
        self, ids, ldap, db_session
    ):
        legacy = await _local(
            db_session,
            "alice-old-name",
            "elsewhere@example.com",
            auth_provider="ldap",
            auth_subject=ALICE_DN,
        )
        (alice,) = [e for e in await _entries(ldap) if e.username == "alice"]

        found = await directory_accounts.find_account(db_session, alice)

        assert found.id == legacy.id

    def test_active_directory_guid_bytes_and_text_read_the_same(self):
        guid = uuid.UUID("12345678-1234-5678-1234-567812345678")

        assert directory_id(guid.bytes_le) == str(guid)
        assert directory_id("{12345678-1234-5678-1234-567812345678}") == str(guid)
        assert directory_id([guid.bytes_le]) == str(guid)
        assert directory_id(None) is None and directory_id([]) is None


# ---------------------------------------------------------------------------
# Import, by the rules login uses
# ---------------------------------------------------------------------------


class TestImport:
    async def test_a_local_account_sharing_only_a_name_is_not_taken_over(
        self, ids, ldap, db_session
    ):
        local = await _local(db_session, "alice", "the-local-alice@example.com")

        result = await directory_import.import_entries(
            db_session, await _entries(ldap), dry_run=False
        )

        (row,) = [r for r in result.rows if r.username == "alice"]
        assert row.action == "refused"
        await db_session.refresh(local)
        assert local.auth_provider == "local"
        # The control: login refuses the same link.
        assert (
            await AuthService().authenticate_user("alice", "alice-secret", db_session)
            is None
        )

    async def test_a_local_account_with_the_same_email_is_linked(
        self, ids, ldap, db_session
    ):
        local = await _local(db_session, "alice", "alice@example.com", role="admin")

        result = await directory_import.import_entries(
            db_session, await _entries(ldap), dry_run=False
        )

        assert result.refused == 0
        await db_session.refresh(local)
        assert local.auth_provider == "ldap" and local.auth_subject == ALICE_ID
        # No mapping configured: the role is left alone.
        assert local.role == "admin"

    async def test_one_bad_entry_does_not_stop_the_rest(
        self, ids, ldap, db_session, monkeypatch
    ):
        ids.add(
            f"uid=nomail,ou=People,{BASE}",
            {"objectClass": "inetOrgPerson", "uid": "nomail", "cn": "N", "sn": "N"},
        )

        result = await directory_import.import_entries(
            db_session, await _entries(ldap), dry_run=False
        )

        assert result.errors == 1 and result.created == 2
        assert {r.username for r in result.rows if r.action == "created"} == {
            "alice",
            "bob",
        }

    async def test_a_dry_run_writes_nothing(self, ids, ldap, db_session):
        result = await directory_import.import_entries(
            db_session, await _entries(ldap), dry_run=True
        )

        assert result.created == 2
        assert (
            await directory_accounts.find_account(db_session, (await _entries(ldap))[0])
            is None
        )


class TestBulkSearchSeesTheSameGroupsAsLogin:
    async def test_groups_found_by_member_search_map_the_role(
        self, ids, ldap, monkeypatch
    ):
        monkeypatch.setattr(
            settings, "LDAP_GROUP_SEARCH_BASE", f"ou=Groups,{BASE}", raising=False
        )
        monkeypatch.setattr(settings, "LDAP_ADMIN_GROUP_DNS", ADMINS_DN, raising=False)

        (alice,) = [e for e in await _entries(ldap) if e.username == "alice"]

        # Alice's entry has no memberOf; the group lists her as a member.
        assert ldap.map_role(alice.groups) == "admin"

    async def test_paging_survives_the_group_searches_between_pages(
        self, ids, ldap, monkeypatch
    ):
        monkeypatch.setattr(
            settings, "LDAP_GROUP_SEARCH_BASE", f"ou=Groups,{BASE}", raising=False
        )
        monkeypatch.setattr(settings, "LDAP_SEARCH_PAGE_SIZE", 1, raising=False)

        found = await _entries(ldap)

        assert {e.username for e in found} == {"alice", "bob"}


# ---------------------------------------------------------------------------
# The sync
# ---------------------------------------------------------------------------


async def _provisioned(db, ldap):
    await directory_import.import_entries(db, await _entries(ldap), dry_run=False)
    users = {
        u.username: u for u in (await db.execute(User.__table__.select())).fetchall()
    }
    return users


async def _user(db, username):
    from sqlalchemy import select

    return (
        (await db.execute(select(User).where(User.username == username)))
        .scalars()
        .one()
    )


class TestSync:
    async def test_someone_who_left_is_deactivated_and_it_says_why(
        self, ids, ldap, db_session
    ):
        await _provisioned(db_session, ldap)
        _remove(ids, f"uid=bob,ou=People,{BASE}")

        report = await directory_sync.sync(db_session)

        bob = await _user(db_session, "bob")
        assert not bob.is_active
        assert bob.auth_metadata["deactivated_by"] == "directory_sync"
        assert report.deactivated == ["bob"] and report.not_found == ["bob"]
        assert (await _user(db_session, "alice")).is_active

    async def test_and_is_reactivated_when_they_are_back(self, ids, ldap, db_session):
        await _provisioned(db_session, ldap)
        _remove(ids, f"uid=bob,ou=People,{BASE}")
        await directory_sync.sync(db_session)
        ids.add(
            f"uid=bob,ou=People,{BASE}",
            {
                "objectClass": "inetOrgPerson",
                "uid": "bob",
                "mail": "bob@example.com",
                "cn": "Bob",
                "sn": "B",
            },
        )

        report = await directory_sync.sync(db_session)

        bob = await _user(db_session, "bob")
        assert bob.is_active and "deactivated_by" not in bob.auth_metadata
        assert report.reactivated == ["bob"]

    async def test_an_admin_s_deactivation_is_never_undone(self, ids, ldap, db_session):
        await _provisioned(db_session, ldap)
        bob = await _user(db_session, "bob")
        bob.is_active = False
        await db_session.commit()

        report = await directory_sync.sync(db_session)

        assert not (await _user(db_session, "bob")).is_active
        assert report.reactivated == []

    async def test_losing_the_admin_group_takes_the_role(
        self, ids, ldap, db_session, monkeypatch
    ):
        monkeypatch.setattr(
            settings, "LDAP_GROUP_SEARCH_BASE", f"ou=Groups,{BASE}", raising=False
        )
        monkeypatch.setattr(settings, "LDAP_ADMIN_GROUP_DNS", ADMINS_DN, raising=False)
        await _provisioned(db_session, ldap)
        assert (await _user(db_session, "alice")).role == "admin"
        conn = _admin_connection(ids)
        conn.delete(ADMINS_DN)
        ids.add(
            ADMINS_DN,
            {"objectClass": "groupOfNames", "cn": "admins", "member": [SERVICE_DN]},
        )

        report = await directory_sync.sync(db_session)

        assert (await _user(db_session, "alice")).role == "user"
        assert report.role_changes == ["alice: admin -> user"]

    async def test_an_empty_answer_deactivates_nobody(self, ids, ldap, db_session):
        await _provisioned(db_session, ldap)

        report = await directory_sync.sync(db_session, entries=[])

        assert report.held_back and "returned nobody" in report.held_back
        assert report.deactivated == [] and set(report.not_found) == {"alice", "bob"}
        assert (await _user(db_session, "bob")).is_active

    async def test_too_many_leaving_at_once_deactivates_nobody(
        self, ids, ldap, db_session, monkeypatch
    ):
        monkeypatch.setattr(
            settings, "LDAP_SYNC_MAX_DEACTIVATE_FRACTION", 0.2, raising=False
        )
        await _provisioned(db_session, ldap)
        _remove(ids, f"uid=bob,ou=People,{BASE}")

        report = await directory_sync.sync(db_session)

        # One of two is 50%, over the 20% allowed.
        assert report.held_back and "1 of 2" in report.held_back
        assert (await _user(db_session, "bob")).is_active

    async def test_a_dry_run_changes_nothing_and_says_what_would(
        self, ids, ldap, db_session
    ):
        await _provisioned(db_session, ldap)
        _remove(ids, f"uid=bob,ou=People,{BASE}")

        report = await directory_sync.sync(db_session, dry_run=True)

        assert report.dry_run and report.deactivated == ["bob"]
        assert (await _user(db_session, "bob")).is_active

    async def test_local_accounts_are_not_the_sync_s_business(
        self, ids, ldap, db_session
    ):
        await _provisioned(db_session, ldap)
        carol = await _local(db_session, "carol", "carol@example.com")

        report = await directory_sync.sync(db_session)

        await db_session.refresh(carol)
        assert carol.is_active and "carol" not in report.not_found
        assert report.managed_accounts == 2


class TestSchedule:
    def test_off_unless_an_interval_is_set(self, monkeypatch):
        monkeypatch.setattr(settings, "LDAP_SYNC_INTERVAL_MINUTES", 0, raising=False)

        assert sync_status.claim_due_run() is False

    def test_the_task_does_nothing_when_off(self, directory, monkeypatch):
        from app.tasks.identity_tasks import sync_directory_accounts

        monkeypatch.setattr(settings, "LDAP_SYNC_INTERVAL_MINUTES", 0, raising=False)

        assert sync_directory_accounts.run() == {"ran": False}

    def test_it_is_on_the_beat_schedule(self):
        from app.core.celery import celery_app

        tasks = {entry["task"] for entry in celery_app.conf.beat_schedule.values()}
        assert "app.tasks.identity_tasks.sync_directory_accounts" in tasks


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


class TestRoutes:
    def test_a_sync_from_the_page_is_a_dry_run_unless_told(
        self, ids, client, admin_headers
    ):
        response = client.post(
            "/api/v1/admin/ldap/sync", headers=admin_headers, json={}
        )

        assert response.status_code == 200, response.text
        assert response.json()["dry_run"] is True

    def test_only_an_admin_may_sync(self, ids, client, auth_headers):
        response = client.post("/api/v1/admin/ldap/sync", headers=auth_headers, json={})

        assert response.status_code == 403

    def test_status_says_how_accounts_are_matched_and_how_often_synced(
        self, ids, client, admin_headers
    ):
        body = client.get("/api/v1/admin/ldap/status", headers=admin_headers).json()

        assert body["unique_id_attribute"] == "entryUUID"
        assert body["sync_interval_minutes"] == 0

    async def test_a_directory_account_cannot_set_a_local_password(
        self, ids, db_session
    ):
        from fastapi import HTTPException

        from app.api.endpoints.users import change_password
        from app.schemas.auth import PasswordChange

        user = await _local(
            db_session, "dana", "dana@example.com", auth_provider="ldap"
        )
        before = user.hashed_password

        with pytest.raises(HTTPException) as refused:
            await change_password(
                PasswordChange(
                    current_password="local-secret", new_password="A-new-password-1"
                ),
                current_user=user,
                db=db_session,
            )

        assert refused.value.status_code == 400
        assert "managed by the directory" in refused.value.detail
        await db_session.refresh(user)
        assert user.hashed_password == before

    async def test_a_local_account_still_can(self, ids, db_session):
        from app.api.endpoints.users import change_password
        from app.schemas.auth import PasswordChange

        user = await _local(db_session, "erin", "erin@example.com")

        answer = await change_password(
            PasswordChange(
                current_password="local-secret", new_password="A-new-password-1"
            ),
            current_user=user,
            db=db_session,
        )

        assert answer == {"message": "Password updated successfully"}
