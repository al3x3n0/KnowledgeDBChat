#!/usr/bin/env python
"""Fail if the migration history no longer builds the schema the models describe.

Run this against a database that has just had ``alembic upgrade head`` applied.
It asks Alembic to autogenerate a diff against the SQLAlchemy models: an empty
diff means migrations and models agree, and anything else is drift — a model
changed without a migration, or a migration was written by hand and got it
wrong.

This exists because the schema used to come from ``Base.metadata.create_all``
plus hand-written startup DDL, which let the migration history fall 12 tables
and 46 columns behind reality without anything noticing.

    DATABASE_URL=postgresql+asyncpg://... python scripts/check_schema_drift.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from sqlalchemy.ext.asyncio import create_async_engine  # noqa: E402

import app.models  # noqa: E402,F401 - registers every model on the metadata
from alembic.autogenerate import compare_metadata  # noqa: E402
from alembic.migration import MigrationContext  # noqa: E402
from app.core.database import Base  # noqa: E402

# Objects Alembic cannot see in the models and would otherwise report forever.
# Keep this list short and justified; it is an escape hatch, not a dumping
# ground.
IGNORED_TABLES = {
    "alembic_version",
}


def _database_url() -> str:
    url = os.environ.get("DATABASE_URL", "")
    if not url:
        sys.exit("DATABASE_URL is required")
    # Use the same async driver the app runs on, so this needs no extra
    # dependency and cannot disagree with the app about how it connects.
    if "+" not in url.split("://", 1)[0]:
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return url.replace("+psycopg2", "+asyncpg")


def _relevant(diff) -> bool:
    """Filter out diffs about objects this check deliberately ignores."""
    entries = diff if isinstance(diff, list) else [diff]
    for entry in entries:
        if not isinstance(entry, tuple) or not entry:
            continue
        for part in entry[1:]:
            name = getattr(part, "name", None) or (
                part if isinstance(part, str) else None
            )
            if name in IGNORED_TABLES:
                return False
    return True


def _compare(connection) -> list:
    context = MigrationContext.configure(connection)
    return [d for d in compare_metadata(context, Base.metadata) if _relevant(d)]


async def _collect_diffs() -> list:
    engine = create_async_engine(_database_url())
    try:
        async with engine.connect() as connection:
            return await connection.run_sync(_compare)
    finally:
        await engine.dispose()


def main() -> int:
    diffs = asyncio.run(_collect_diffs())

    if not diffs:
        print("Schema matches the models: alembic upgrade head is authoritative.")
        return 0

    print(f"Schema drift detected: {len(diffs)} difference(s) between the")
    print("migration result and the models.\n")
    for diff in diffs:
        print(f"  - {diff}")
    print(
        "\nGenerate a migration for these changes:\n"
        '    alembic revision --autogenerate -m "describe the change"\n'
        "Do not add hand-written DDL at startup; that is what this check exists "
        "to prevent."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
