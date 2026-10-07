"""The Redis cache holds JSON, and what the old format left behind is read safely.

Values were pickled. Reading a pickle runs what it describes, so anything able
to write to Redis could run code in every API and worker process.
"""

import json
import pickle

import pytest

from app.core import cache
from app.core.cache import CacheService
from app.utils import ingestion_state

pytestmark = pytest.mark.unit


class _Redis:
    def __init__(self):
        self.data = {}

    async def get(self, key):
        return self.data.get(key)

    async def set(self, key, value):
        self.data[key] = value

    async def setex(self, key, ttl, value):
        self.data[key] = value


@pytest.fixture
def redis(monkeypatch):
    fake = _Redis()

    async def _client():
        return fake

    monkeypatch.setattr(cache, "get_redis_client", _client)
    monkeypatch.setattr(ingestion_state, "get_redis_client", _client)
    return fake


@pytest.mark.parametrize(
    "value", [True, False, 0, 3, "deepseek-v4-pro", {"status": "done", "n": [1, 2]}]
)
async def test_values_round_trip_as_json(redis, value):
    service = CacheService()
    assert await service.set("k", value, ttl=60) is True
    assert json.loads(redis.data["k"]) == value
    assert await service.get("k") == value


async def test_a_value_json_cannot_hold_is_refused_not_coerced(redis):
    class Row:
        pass

    assert await CacheService().set("k", Row()) is False
    assert "k" not in redis.data


@pytest.mark.parametrize("value", [True, "balanced", {"a": [1, 2.5, None]}])
async def test_plain_data_pickled_by_the_old_code_is_still_read(redis, value):
    # Feature flags have no expiry: these must survive the format change.
    redis.data["k"] = pickle.dumps(value)
    assert await CacheService().get("k") == value


async def test_a_pickle_that_names_a_callable_is_not_executed(redis, tmp_path):
    marker = tmp_path / "ran"

    class Exploit:
        def __reduce__(self):
            import os

            return (os.system, (f"touch {marker}",))

    redis.data["k"] = pickle.dumps(Exploit())
    assert await CacheService().get("k") is None  # refused, reported as a miss
    assert not marker.exists()


async def test_the_task_id_a_sync_records_is_the_one_a_cancel_reads(redis):
    # sync_tasks wrote this key through the pickling cache while every reader
    # took the raw string, so a cancel got pickle bytes instead of a task id.
    await ingestion_state.set_ingestion_task_mapping("src-1", "task-abc")
    assert await ingestion_state.get_ingestion_task_mapping("src-1") == "task-abc"


def test_the_scheduled_scan_uses_the_shared_helpers():
    import inspect

    from app.tasks import sync_tasks

    source = inspect.getsource(sync_tasks)
    assert "set_ingestion_task_mapping(" in source
    assert "cache_service" not in source
