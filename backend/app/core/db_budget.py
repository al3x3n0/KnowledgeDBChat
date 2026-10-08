"""How many database connections the API may open, against what the server allows.

A pool is per process. Every gunicorn worker of every API replica holds its own,
so the connections the API can ask for are

    (DB_POOL_SIZE + DB_MAX_OVERFLOW) x replicas x workers

and nothing added that up. At the old defaults (20 + 40) the Helm chart's
2 replicas x 4 workers could request 480 connections from a Postgres whose
default ``max_connections`` is 100: under load the first thing to fail would
have been ``FATAL: sorry, too many clients already``, in whichever process
asked last. Celery workers add theirs on top (unpooled, one per open session).

``check`` is the arithmetic; ``warn_if_over_budget`` asks the server for its
real limit at startup and says so, with the numbers, when the configuration
cannot fit. It never blocks startup: a wrong estimate of the process count
must not take the API down. The Helm chart makes the same check at render
time, where it can refuse outright.
"""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger
from sqlalchemy import text

#: Left for Celery workers, migrations, psql and superuser slots.
RESERVED_CONNECTIONS = 40


@dataclass(frozen=True)
class Budget:
    per_process: int
    processes: int
    needed: int
    available: int

    @property
    def fits(self) -> bool:
        return self.needed <= self.available

    def describe(self) -> str:
        return (
            f"the API may open {self.needed} database connections "
            f"({self.per_process} per process x {self.processes} processes) and "
            f"the server leaves {self.available} for it"
        )


def check(
    *,
    pool_size: int,
    max_overflow: int,
    processes: int,
    max_connections: int,
    reserved: int = RESERVED_CONNECTIONS,
) -> Budget:
    """The API's worst-case connection demand against the server's limit."""
    per_process = max(0, int(pool_size)) + max(0, int(max_overflow))
    count = max(1, int(processes))
    return Budget(
        per_process=per_process,
        processes=count,
        needed=per_process * count,
        available=max(0, int(max_connections) - max(0, int(reserved))),
    )


async def warn_if_over_budget(engine, settings) -> Budget | None:
    """Compare the configured pools with the server's ``max_connections``.

    Returns the budget, or None when the server could not be asked (not
    Postgres, or unreachable -- startup continues either way).
    """
    try:
        async with engine.connect() as connection:
            if connection.dialect.name != "postgresql":
                return None
            limit = int(
                (await connection.execute(text("SHOW max_connections"))).scalar()
            )
    except Exception as exc:
        logger.debug(f"Could not read max_connections: {exc}")
        return None

    budget = check(
        pool_size=settings.DB_POOL_SIZE,
        max_overflow=settings.DB_MAX_OVERFLOW,
        processes=settings.DB_EXPECTED_API_PROCESSES,
        max_connections=limit,
    )
    if budget.fits:
        logger.info(f"Database connection budget: {budget.describe()}.")
    else:
        logger.warning(
            f"Database connection budget exceeded: {budget.describe()} "
            f"(max_connections={limit}, {RESERVED_CONNECTIONS} reserved for "
            "workers and tools). Under load some requests will fail with "
            "'too many clients'. Lower DB_POOL_SIZE / DB_MAX_OVERFLOW, raise "
            "max_connections, or put PgBouncer in front."
        )
    return budget


__all__ = ["RESERVED_CONNECTIONS", "Budget", "check", "warn_if_over_budget"]
