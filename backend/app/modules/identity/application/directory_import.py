"""An admin importing people from the directory, by the rules login uses.

Moved out of the admin endpoint, where it was 170 lines with its own
matching: it found an existing account by username and linked it whatever
its email, so a directory entry could take over a local account that login
would have refused to. It also wrote each person with a commit inside the
loop, so one failed write left the session unusable for the rest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.identity.application import directory_accounts
from app.services.ldap_service import ldap_service

#: Rows returned to the page; the counts cover everything.
MAX_ROWS = 200


@dataclass
class ImportRow:
    username: str
    email: Optional[str]
    full_name: Optional[str]
    dn: Optional[str]
    role: str
    #: created, updated, refused or error.
    action: str
    error: Optional[str] = None


@dataclass
class ImportResult:
    created: int = 0
    updated: int = 0
    refused: int = 0
    errors: int = 0
    rows: list[ImportRow] = field(default_factory=list)

    def add(self, row: ImportRow) -> None:
        if len(self.rows) < MAX_ROWS:
            self.rows.append(row)


def _row(entry: Any, *, role: str, action: str, error: Optional[str] = None):
    return ImportRow(
        username=getattr(entry, "username", "") or "",
        email=getattr(entry, "email", None),
        full_name=getattr(entry, "full_name", None),
        dn=getattr(entry, "dn", None),
        role=role,
        action=action,
        error=error,
    )


async def import_entries(
    db: AsyncSession,
    entries: list[Any],
    *,
    dry_run: bool,
    default_role: str = "user",
    overwrite_role: bool = False,
) -> ImportResult:
    """Create or update an account for each entry; with ``dry_run``, only say
    what would happen."""
    result = ImportResult()
    for entry in entries:
        mapped = ldap_service.map_role(getattr(entry, "groups", None))
        existing = await directory_accounts.find_account(db, entry)

        refusal = None
        if existing is not None and not directory_accounts.may_link(existing, entry):
            refusal = (
                f"the local account '{existing.username}' was registered from the "
                "sign-in page; not linked"
                if directory_accounts.self_registered(existing)
                else f"the local account '{existing.username}' has the same name "
                "but a different email; not linked"
            )
        if refusal:
            result.refused += 1
            result.add(_row(entry, role=existing.role, action="refused", error=refusal))
            continue

        if overwrite_role:
            role = default_role
        elif existing is not None:
            role = mapped or existing.role
        else:
            role = mapped or default_role
        action = "updated" if existing is not None else "created"

        if dry_run:
            setattr(result, action, getattr(result, action) + 1)
            result.add(_row(entry, role=role, action=action))
            continue

        try:
            # One person's failure is theirs alone: a savepoint, so the
            # session is still usable for the next entry.
            async with db.begin_nested():
                user = await directory_accounts.provision(
                    db,
                    entry,
                    existing,
                    default_role=default_role,
                    # An admin importing is asking for the directory's view.
                    update_profile=True,
                    commit=False,
                )
                if overwrite_role:
                    user.role = default_role
        except Exception as exc:  # noqa: BLE001 - reported per row
            result.errors += 1
            result.add(_row(entry, role=role, action="error", error=str(exc)))
            continue
        setattr(result, action, getattr(result, action) + 1)
        result.add(_row(entry, role=user.role, action=action))

    if not dry_run:
        await db.commit()
    return result
