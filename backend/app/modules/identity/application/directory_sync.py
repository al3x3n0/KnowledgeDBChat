"""Reconciling directory-managed accounts with the directory, between logins.

An account learned what the directory thought of it only when its owner
signed in. Someone who left, was disabled, or was taken out of the admin
group kept their account, their role and their tokens for as long as they
did not sign in again -- which, having left, they would not.

The people who should have access are those ``LDAP_IMPORT_FILTER`` finds.
For each directory-managed account:

* **found**: the role follows the group mapping, and the email and name
  follow the directory; if the sync itself had deactivated the account, it
  is reactivated.
* **not found**: deactivated, with ``auth_metadata["deactivated_by"]``
  saying the sync did it. An account an admin deactivated is never
  reactivated by the sync, and local accounts are never touched.

A result that looks like a broken query is not acted on: no entries at all,
or more deactivations than ``LDAP_SYNC_MAX_DEACTIVATE_FRACTION`` of the
managed accounts. A wrong base DN or filter looks exactly like everyone
leaving at once, and acting on it would lock out the whole deployment.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Optional

from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.user import User
from app.modules.identity.application import directory_accounts
from app.services.ldap_service import ldap_service

#: People listed by name in a report.
MAX_NAMED = 200


@dataclass
class SyncReport:
    ran_at: str
    dry_run: bool
    #: Entries the directory returned for LDAP_IMPORT_FILTER.
    directory_entries: int = 0
    managed_accounts: int = 0
    unchanged: int = 0
    updated: list[str] = field(default_factory=list)
    role_changes: list[str] = field(default_factory=list)
    #: Active accounts the directory no longer lists.
    not_found: list[str] = field(default_factory=list)
    #: Deactivated by this run (or, in a dry run, that would be).
    deactivated: list[str] = field(default_factory=list)
    reactivated: list[str] = field(default_factory=list)
    #: Why nothing was deactivated although someone was missing, if so.
    held_back: Optional[str] = None
    #: Why the sync did not run at all, if so.
    error: Optional[str] = None

    def as_dict(self) -> dict:
        return asdict(self)


def _index(entries: list[Any]) -> dict[str, Any]:
    """Each entry under every key an account might hold for it."""
    keyed: dict[str, Any] = {}
    for entry in entries:
        for key in (getattr(entry, "unique_id", None), getattr(entry, "dn", None)):
            if key:
                keyed[f"subject:{str(key).lower()}"] = entry
        username = (getattr(entry, "username", None) or "").strip().lower()
        if username:
            keyed.setdefault(f"username:{username}", entry)
    return keyed


def _entry_for(user: User, keyed: dict[str, Any]) -> Optional[Any]:
    metadata = user.auth_metadata if isinstance(user.auth_metadata, dict) else {}
    for key in (user.auth_subject, metadata.get("dn")):
        if key and f"subject:{str(key).lower()}" in keyed:
            return keyed[f"subject:{str(key).lower()}"]
    return keyed.get(f"username:{(user.username or '').strip().lower()}")


def _deactivated_by_sync(user: User) -> bool:
    metadata = user.auth_metadata if isinstance(user.auth_metadata, dict) else {}
    return metadata.get("deactivated_by") == directory_accounts.DEACTIVATED_BY_SYNC


def _named(names: list[str], name: str) -> None:
    if len(names) < MAX_NAMED:
        names.append(name)


async def sync(
    db: AsyncSession,
    *,
    dry_run: bool = False,
    entries: Optional[list[Any]] = None,
) -> SyncReport:
    """Reconcile every directory-managed account with the directory.

    ``entries`` is the directory's answer when the caller already has it;
    otherwise it is read here. Raises what the directory raises when it
    cannot be read: an unreachable directory says nothing about who left.
    """
    report = SyncReport(ran_at=datetime.utcnow().isoformat(), dry_run=dry_run)
    if entries is None:
        search_filter = str(getattr(settings, "LDAP_IMPORT_FILTER", "") or "").strip()
        if not search_filter:
            report.error = "LDAP_IMPORT_FILTER is empty, so nobody would be found."
            return report
        entries = await ldap_service.search_users_async(
            search_filter=search_filter,
            limit=int(getattr(settings, "LDAP_SYNC_MAX_ENTRIES", 50000) or 50000),
        )
    report.directory_entries = len(entries)
    keyed = _index(entries)

    accounts = list(
        (
            await db.execute(
                select(User).where(User.auth_provider == directory_accounts.DIRECTORY)
            )
        )
        .scalars()
        .all()
    )
    report.managed_accounts = len(accounts)

    missing = []
    for user in accounts:
        entry = _entry_for(user, keyed)
        if entry is None:
            if user.is_active:
                missing.append(user)
            else:
                report.unchanged += 1
            continue

        before = (user.role, user.email, user.full_name, user.auth_subject)
        if not dry_run:
            await directory_accounts.provision(
                db, entry, user, update_profile=True, commit=False
            )
        mapped = ldap_service.map_role(getattr(entry, "groups", None))
        if mapped is not None and mapped != before[0]:
            _named(report.role_changes, f"{user.username}: {before[0]} -> {mapped}")

        if not user.is_active and _deactivated_by_sync(user):
            _named(report.reactivated, user.username)
            if not dry_run:
                metadata = dict(user.auth_metadata or {})
                metadata.pop("deactivated_by", None)
                metadata.pop("deactivated_at", None)
                user.auth_metadata = metadata
                user.is_active = True
        elif dry_run:
            report.unchanged += 1
        elif (user.role, user.email, user.full_name, user.auth_subject) != before:
            _named(report.updated, user.username)
        else:
            report.unchanged += 1

    limit = float(getattr(settings, "LDAP_SYNC_MAX_DEACTIVATE_FRACTION", 0.2) or 0)
    if missing and not entries:
        report.held_back = (
            "The directory returned nobody for LDAP_IMPORT_FILTER, which reads as "
            "a wrong filter or base DN rather than everyone leaving; nobody was "
            "deactivated."
        )
    elif missing and len(missing) > limit * max(1, len(accounts)):
        report.held_back = (
            f"{len(missing)} of {len(accounts)} directory-managed accounts were not "
            f"found, more than LDAP_SYNC_MAX_DEACTIVATE_FRACTION ({limit:.0%}) "
            "allows; nobody was deactivated. Check the filter and base DN, or "
            "raise the limit if this many people really left."
        )
    for user in missing:
        _named(report.not_found, user.username)
        if report.held_back:
            continue
        _named(report.deactivated, user.username)
        if dry_run:
            continue
        user.is_active = False
        user.auth_metadata = {
            **(user.auth_metadata if isinstance(user.auth_metadata, dict) else {}),
            "deactivated_by": directory_accounts.DEACTIVATED_BY_SYNC,
            "deactivated_at": report.ran_at,
        }
    if report.held_back:
        logger.warning(f"LDAP sync held back: {report.held_back}")

    if dry_run:
        await db.rollback()
    else:
        await db.commit()
    return report
