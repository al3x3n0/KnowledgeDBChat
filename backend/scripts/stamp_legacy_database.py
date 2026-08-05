#!/usr/bin/env python
"""Bring a database built before Alembic was authoritative under Alembic control.

Databases created by ``Base.metadata.create_all`` — which used to be the
documented bootstrap — already have the full schema but no ``alembic_version``
table, so Alembic believes nothing has been applied. Running ``upgrade head``
against one fails immediately, because migration 0002 tries to create tables
that are already there.

Stamping is the right move, but it cannot be done with a bare ``alembic stamp``:
Alembic creates ``alembic_version`` at its default VARCHAR(32) and this repo's
revision ids are longer than that, so the stamp fails with a truncation error.
(Migration 0038a widens the column, but it never runs on a database that was
never migrated.) This script creates the table at the right width first, then
stamps.

    DATABASE_URL=postgresql+asyncpg://... python scripts/stamp_legacy_database.py

It refuses to touch a database that already has an ``alembic_version`` row, so
it is safe to run twice and cannot silently rewrite a real migration state.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from alembic.config import Config  # noqa: E402
from alembic.script import ScriptDirectory  # noqa: E402
from sqlalchemy import text  # noqa: E402
from sqlalchemy.ext.asyncio import create_async_engine  # noqa: E402

# Matches migration 0038a, which widened the column for long revision ids.
VERSION_NUM_LENGTH = 128


def _head_revision() -> str:
    config = Config(str(BACKEND_ROOT / "alembic.ini"))
    heads = ScriptDirectory.from_config(config).get_heads()
    if len(heads) != 1:
        sys.exit(f"Expected exactly one head revision, found {len(heads)}: {heads}")
    return heads[0]


async def _stamp(head: str) -> int:
    url = os.environ.get("DATABASE_URL", "")
    if not url:
        sys.exit("DATABASE_URL is required")
    engine = create_async_engine(url)
    try:
        async with engine.begin() as connection:
            existing = await connection.execute(
                text(
                    "SELECT to_regclass('public.alembic_version') IS NOT NULL AS present"
                )
            )
            if existing.scalar():
                current = (
                    await connection.execute(
                        text("SELECT version_num FROM alembic_version")
                    )
                ).scalar()
                if current:
                    print(
                        f"Database is already under Alembic control at {current}; "
                        "nothing to do. Use `alembic upgrade head` to apply new "
                        "migrations."
                    )
                    return 0
            else:
                await connection.execute(
                    text(
                        "CREATE TABLE alembic_version ("
                        f"version_num VARCHAR({VERSION_NUM_LENGTH}) NOT NULL, "
                        "CONSTRAINT alembic_version_pkc PRIMARY KEY (version_num))"
                    )
                )
            await connection.execute(
                text("INSERT INTO alembic_version (version_num) VALUES (:rev)"),
                {"rev": head},
            )
    finally:
        await engine.dispose()

    print(f"Stamped this database at {head}.")
    print(
        "Verify with: DATABASE_URL=... python scripts/check_schema_drift.py\n"
        "From here on, schema changes go through `alembic revision "
        "--autogenerate`."
    )
    return 0


def main() -> int:
    return asyncio.run(_stamp(_head_revision()))


if __name__ == "__main__":
    raise SystemExit(main())
