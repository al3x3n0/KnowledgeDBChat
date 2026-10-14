"""Which local account a directory entry is, and what the directory may change.

Login, the admin import and the scheduled sync each turn a directory entry
into an account. Login and import used to do it by different rules: login
refused to take over a local account that merely shared a username (its
emails had to match), while the import linked by username alone, so an entry
``uid=admin`` in the directory became the local ``admin``. Every caller now
asks this module.

The rules:

* **An entry is matched by what names it for good**: the account's
  ``auth_subject`` against the entry's unique id, then against its DN (rows
  written before unique ids were read, or a directory without one), and only
  then by username and email.
* **A local account is linked only when it is the same person**: already
  directory-managed, or the emails match, or
  ``LDAP_LINK_LOCAL_USERS_BY_USERNAME`` says a shared username is enough.
* **The directory never reactivates an account an admin deactivated.** The
  sync may reactivate one it deactivated itself, and records which it was.
* **The role follows the group mapping only when one is configured.**
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from loguru import logger
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import passwords
from app.core.config import settings
from app.models.user import User
from app.services.ldap_service import ldap_service

DIRECTORY = "ldap"

#: Written to ``auth_metadata["deactivated_by"]`` by the sync, so that it
#: reactivates only what it deactivated.
DEACTIVATED_BY_SYNC = "directory_sync"


class AccountRefused(Exception):
    """This entry may not sign in as, or be written to, a local account."""


def _subjects(entry: Any) -> list[str]:
    return [
        value
        for value in (getattr(entry, "unique_id", None), getattr(entry, "dn", None))
        if value
    ]


async def _first(db: AsyncSession, *conditions) -> Optional[User]:
    result = await db.execute(select(User).where(*conditions).limit(1))
    return result.scalars().first()


async def find_account(db: AsyncSession, entry: Any) -> Optional[User]:
    """The local account this directory entry is, if there is one."""
    subjects = _subjects(entry)
    if subjects:
        found = await _first(
            db, User.auth_provider == DIRECTORY, User.auth_subject.in_(subjects)
        )
        if found is not None:
            return found
    username = (getattr(entry, "username", None) or "").strip()
    if username:
        found = await _first(db, User.username == username)
        if found is not None:
            return found
    email = (getattr(entry, "email", None) or "").strip()
    if email:
        return await _first(db, func.lower(User.email) == email.lower())
    return None


#: ``auth_metadata["origin"]`` of an account made from the sign-in page.
SELF_REGISTRATION = "self_registration"


def self_registered(user: User) -> bool:
    metadata = user.auth_metadata if isinstance(user.auth_metadata, dict) else {}
    return metadata.get("origin") == SELF_REGISTRATION


def may_link(existing: User, entry: Any) -> bool:
    """Whether a directory entry may become this existing account.

    An account already directory-managed is the same person by construction.
    A local one only when its email matches the directory's (a shared
    username alone is not identity: anyone able to create a directory entry
    named "admin" would otherwise become the local admin), or when
    ``LDAP_LINK_LOCAL_USERS_BY_USERNAME`` says a name is enough.
    """
    if getattr(existing, "auth_provider", "local") == DIRECTORY:
        return True
    if self_registered(existing):
        # Anyone may register, and nothing checks the email they give: a
        # stranger could register a directory person's name and email before
        # that person first signs in, use the account, and have the
        # directory hand it to them. Not by email, not by name.
        return False
    if getattr(settings, "LDAP_LINK_LOCAL_USERS_BY_USERNAME", False):
        return True
    ours = (existing.email or "").strip().lower()
    theirs = (getattr(entry, "email", None) or "").strip().lower()
    return bool(ours) and ours == theirs


def _metadata(user: Optional[User], entry: Any) -> dict:
    kept = dict(user.auth_metadata or {}) if user is not None else {}
    if not isinstance(kept, dict):
        kept = {}
    # A new dict, assigned: the column is plain JSON.
    return {
        **kept,
        "dn": getattr(entry, "dn", None),
        "groups": list(getattr(entry, "groups", None) or []),
    }


async def provision(
    db: AsyncSession,
    entry: Any,
    existing: Optional[User] = None,
    *,
    default_role: str = "user",
    update_profile: Optional[bool] = None,
    commit: bool = True,
) -> User:
    """Create the account for this entry, or bring an existing one up to date.

    Never changes ``is_active``. ``update_profile`` (default
    ``LDAP_SYNC_ON_LOGIN``) says whether email and name follow the directory.
    """
    if update_profile is None:
        update_profile = bool(getattr(settings, "LDAP_SYNC_ON_LOGIN", True))
    mapped_role = ldap_service.map_role(getattr(entry, "groups", None))
    subject = getattr(entry, "subject", None) or getattr(entry, "dn", None)

    user = existing
    if user is None:
        username = (getattr(entry, "username", None) or "").strip()
        email = (getattr(entry, "email", None) or "").strip()
        if not username:
            raise ValueError("LDAP entry has no username attribute")
        if not email:
            raise ValueError(
                "LDAP user has no email and LDAP_DEFAULT_EMAIL_DOMAIN is not set"
            )
        user = User(
            username=username,
            email=email,
            full_name=getattr(entry, "full_name", None),
            # Never used: a directory-managed account is checked against the
            # directory.
            hashed_password=passwords.unusable_password_hash(),
            is_active=True,
            is_verified=True,
            role=mapped_role or default_role,
            auth_provider=DIRECTORY,
            auth_subject=subject,
            auth_metadata=_metadata(None, entry),
        )
        db.add(user)
    else:
        if update_profile:
            new_email = (getattr(entry, "email", None) or "").strip()
            if new_email and new_email.lower() != (user.email or "").lower():
                # Emails are unique: taking one another account holds would
                # fail the whole write at commit.
                holder = await _first(
                    db, func.lower(User.email) == new_email.lower(), User.id != user.id
                )
                if holder is None:
                    user.email = new_email
                else:
                    logger.warning(
                        f"LDAP email {new_email} for {user.username} belongs to "
                        f"another account ({holder.username}); kept the old one"
                    )
            if getattr(entry, "full_name", None):
                user.full_name = entry.full_name
        user.auth_provider = DIRECTORY
        user.auth_subject = subject
        user.auth_metadata = _metadata(user, entry)
        if mapped_role is not None:
            user.role = mapped_role
        user.is_verified = True
        user.updated_at = datetime.utcnow()

    if commit:
        await db.commit()
        await db.refresh(user)
    else:
        await db.flush()
    return user


async def account_for(
    db: AsyncSession,
    entry: Any,
    *,
    create: bool,
    default_role: str = "user",
    commit: bool = True,
) -> User:
    """The account this entry signs in as, created if ``create`` allows.

    Raises :class:`AccountRefused` when the account is deactivated, when it
    is a local account that is not provably this person, or when there is
    none and none may be created; ``ValueError`` when the entry lacks what an
    account needs.
    """
    existing = await find_account(db, entry)
    if existing is not None:
        if not existing.is_active:
            raise AccountRefused(f"the account '{existing.username}' is deactivated")
        if not may_link(existing, entry):
            if self_registered(existing):
                raise AccountRefused(
                    f"the local account '{existing.username}' was registered from "
                    "the sign-in page, and the directory never links those"
                )
            raise AccountRefused(
                f"the local account '{existing.username}' has the same name but "
                "a different email; not linked (set "
                "LDAP_LINK_LOCAL_USERS_BY_USERNAME to link by name)"
            )
    elif not create:
        raise AccountRefused("no account exists and creating one is turned off")
    return await provision(
        db, entry, existing, default_role=default_role, commit=commit
    )
