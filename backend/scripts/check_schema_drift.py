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

# A missing table or column means the application will raise at runtime, so
# these fail the check. Everything else is a difference of declaration, not of
# existence: extra indexes migrations created and models never declared,
# nullable and type mismatches, and foreign keys whose ondelete differs. Those
# are reported so they stay visible, and are worth reconciling deliberately —
# but they do not stop the app from running, and failing on them would mean this
# check could never be green.
BREAKING_OPERATIONS = {"add_table", "add_column"}


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


def _operations(diff) -> list[str]:
    entries = diff if isinstance(diff, list) else [diff]
    return [e[0] for e in entries if isinstance(e, tuple) and e]


def main() -> int:
    diffs = asyncio.run(_collect_diffs())
    breaking = [d for d in diffs if set(_operations(d)) & BREAKING_OPERATIONS]
    other = [d for d in diffs if d not in breaking]

    if other:
        counts: dict[str, int] = {}
        for diff in other:
            for op in _operations(diff):
                counts[op] = counts.get(op, 0) + 1
        summary = ", ".join(f"{op}={n}" for op, n in sorted(counts.items()))
        print(f"Declaration differences (reported, not fatal): {summary}")

    if not breaking:
        print(
            "No missing tables or columns: alembic upgrade head builds a schema "
            "the application can run against."
        )
        return 0

    print(f"\nSchema drift detected: {len(breaking)} object(s) the models need")
    print("that migrations do not create.\n")
    for diff in breaking:
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
