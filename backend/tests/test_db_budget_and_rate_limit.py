"""Two things that are per process and were configured as if they were not.

A connection pool belongs to one process, so the API's demand on Postgres is
the pool times every worker of every replica; and the rate limiter counted in
process memory, so each of those processes allowed the full limit.
"""

from pathlib import Path

import pytest
import yaml
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.core import db_budget, rate_limit
from app.core.config import Settings, settings

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
CHART = ROOT / "deploy" / "helm" / "knowledgedbchat"


class TestConnectionBudget:
    def test_the_old_defaults_did_not_fit_the_default_deployment(self):
        # 20 + 40 per process, 2 replicas x 4 workers, Postgres's default 100.
        budget = db_budget.check(
            pool_size=20, max_overflow=40, processes=8, max_connections=100
        )
        assert budget.needed == 480
        assert not budget.fits

    def test_the_shipped_defaults_fit_the_shipped_deployment(self):
        # Read from the files a deployment actually uses, so a change to one
        # that breaks the sum fails here rather than under load.
        values = yaml.safe_load((CHART / "values.yaml").read_text())
        backend = values["backend"]
        defaults = Settings.model_fields
        assert backend["dbPool"]["size"] == defaults["DB_POOL_SIZE"].default
        assert backend["dbPool"]["maxOverflow"] == defaults["DB_MAX_OVERFLOW"].default

        budget = db_budget.check(
            pool_size=backend["dbPool"]["size"],
            max_overflow=backend["dbPool"]["maxOverflow"],
            processes=backend["replicaCount"] * backend["workers"],
            max_connections=values["postgres"]["maxConnections"],
        )
        assert budget.fits, budget.describe()

    @pytest.mark.parametrize(
        "compose", ["docker-compose.yml", "docker-compose.prod.yml"]
    )
    def test_compose_raises_max_connections_to_what_the_chart_assumes(self, compose):
        services = yaml.safe_load((ROOT / compose).read_text())["services"]
        values = yaml.safe_load((CHART / "values.yaml").read_text())
        expected = f"max_connections={values['postgres']['maxConnections']}"
        assert expected in str(services["postgres"].get("command"))

    def test_a_single_process_is_the_floor(self):
        budget = db_budget.check(
            pool_size=10, max_overflow=20, processes=0, max_connections=300
        )
        assert budget.processes == 1 and budget.needed == 30

    async def test_sqlite_is_not_asked(self, db_session):
        engine = db_session.bind
        assert await db_budget.warn_if_over_budget(engine, settings) is None

    async def test_an_overcommitted_server_is_reported(self):
        class _Result:
            def scalar(self):
                return "100"

        class _Connection:
            dialect = type("D", (), {"name": "postgresql"})()

            async def execute(self, _statement):
                return _Result()

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_exc):
                return False

        class _Engine:
            def connect(self):
                return _Connection()

        config = type(
            "S",
            (),
            {"DB_POOL_SIZE": 20, "DB_MAX_OVERFLOW": 40, "DB_EXPECTED_API_PROCESSES": 8},
        )()
        budget = await db_budget.warn_if_over_budget(_Engine(), config)
        assert budget is not None and not budget.fits
        assert (budget.needed, budget.available) == (480, 60)

    async def test_an_unreachable_server_does_not_stop_startup(self):
        class _Engine:
            def connect(self):
                raise ConnectionRefusedError("no database")

        assert await db_budget.warn_if_over_budget(_Engine(), settings) is None


class TestRateLimitsAreShared:
    def test_limits_are_counted_in_redis_unless_told_otherwise(self, monkeypatch):
        monkeypatch.setattr(settings, "RATE_LIMIT_STORAGE_URL", None)
        assert rate_limit.rate_limit_storage_uri() == settings.REDIS_URL
        monkeypatch.setattr(settings, "RATE_LIMIT_STORAGE_URL", "memory://")
        assert rate_limit.rate_limit_storage_uri() == "memory://"

    def test_the_setting_defaults_to_shared_storage(self):
        assert Settings.model_fields["RATE_LIMIT_STORAGE_URL"].default is None

    def test_a_redis_that_is_down_degrades_instead_of_failing(self):
        # Port 1 refuses at once. The limiter must keep serving and keep
        # counting (in process), not answer 500 because its cache is away.
        limiter = Limiter(
            key_func=lambda request: "someone",
            storage_uri="redis://127.0.0.1:1/0",
            in_memory_fallback_enabled=True,
        )
        app = FastAPI()
        app.state.limiter = limiter
        app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

        @app.get("/limited")
        @limiter.limit("2/minute")
        async def limited(request: Request):
            return {"ok": True}

        client = TestClient(app)
        statuses = [client.get("/limited").status_code for _ in range(3)]
        assert statuses == [200, 200, 429]

    def test_the_application_limiter_has_the_fallback_on(self):
        assert rate_limit.limiter._in_memory_fallback_enabled is True
