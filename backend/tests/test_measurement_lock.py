"""One wall-clock measurement at a time, per host.

A swarm's whole value is that its roles look independently. Two roles timing
the same kernel on the same CPUs at the same second are not independent: they
are each other's load. Measured, on the first swarm whose roles both got as
far as benchmarking -- both started at 13:51:13, both reported the host
`busy`/`saturated`, and their trials spread by 130% and 142%.

The risk in serialising is not the happy path but the failure paths. This runs
inside a tool an agent called; it must never be the reason a measurement is
lost, and it must never hand the machine to a second run while a first is
still timing.
"""

import asyncio

import pytest

from app.services import agent_measurement_lock as lock

pytestmark = pytest.mark.unit


class FakeRedis:
    """Enough of SET NX PX / GET / DELETE to exercise the lock."""

    def __init__(self, *, fail=False):
        self.store = {}
        self.fail = fail
        self.deleted = []

    async def set(self, key, value, nx=False, px=None):
        if self.fail:
            raise RuntimeError("redis is down")
        if nx and key in self.store:
            return None
        self.store[key] = value
        return True

    async def get(self, key):
        if self.fail:
            raise RuntimeError("redis is down")
        return self.store.get(key)

    async def delete(self, key):
        self.deleted.append(key)
        self.store.pop(key, None)
        return 1


@pytest.fixture
def redis(monkeypatch):
    fake = FakeRedis()

    async def _client():
        return fake

    monkeypatch.setattr(lock, "_client", _client)
    return fake


class TestItSerialises:
    async def test_an_uncontended_measurement_holds_the_lock(self, redis):
        async with lock.exclusive_measurement() as outcome:
            assert outcome.held is True
        assert outcome.waited_seconds < 1

    async def test_the_lock_is_released_afterwards(self, redis):
        async with lock.exclusive_measurement():
            pass
        async with lock.exclusive_measurement() as second:
            assert second.held is True, "the first run must not keep the machine"

    async def test_a_second_measurement_waits_for_the_first(self, redis):
        """The defect this exists for: two roles timing at the same second."""
        order = []

        async def role(name, hold):
            async with lock.exclusive_measurement(wait_seconds=5) as outcome:
                order.append(f"{name}:start:{outcome.held}")
                await asyncio.sleep(hold)
                order.append(f"{name}:end")

        await asyncio.gather(role("a", 0.3), role("b", 0.05))

        # Whoever went first finished before the other started.
        assert order[0].endswith(":start:True")
        assert order[1].endswith(":end")
        assert order[2].endswith(":start:True")
        assert order[3].endswith(":end")

    async def test_the_wait_is_reported(self, redis):
        async def holder():
            async with lock.exclusive_measurement():
                await asyncio.sleep(0.8)

        async def waiter():
            await asyncio.sleep(0.05)
            async with lock.exclusive_measurement(wait_seconds=5) as outcome:
                return outcome

        _, outcome = await asyncio.gather(holder(), waiter())
        assert outcome.held is True
        assert outcome.waited_seconds >= 0.5
        assert outcome.as_quality()["measurement_queue_wait_seconds"] >= 0.5


class TestItNeverLosesTheMeasurement:
    async def test_redis_being_down_does_not_stop_the_work(self, monkeypatch):
        async def _broken():
            raise RuntimeError("no redis here")

        monkeypatch.setattr(lock, "_client", _broken)

        ran = False
        async with lock.exclusive_measurement() as outcome:
            ran = True
        assert ran is True
        assert outcome.held is False
        assert "could not reach" in outcome.detail

    async def test_a_set_that_raises_does_not_stop_the_work(self, monkeypatch):
        fake = FakeRedis(fail=True)

        async def _client():
            return fake

        monkeypatch.setattr(lock, "_client", _client)

        async with lock.exclusive_measurement() as outcome:
            pass
        assert outcome.held is False

    async def test_running_out_of_patience_proceeds_unserialised(self, redis):
        """A timing with a caveat beats no timing."""

        async def holder():
            async with lock.exclusive_measurement():
                await asyncio.sleep(1.0)

        async def waiter():
            await asyncio.sleep(0.05)
            async with lock.exclusive_measurement(wait_seconds=0.2) as outcome:
                return outcome

        _, outcome = await asyncio.gather(holder(), waiter())
        assert outcome.held is False
        assert "may have shared" in outcome.detail

    async def test_disabled_does_not_touch_redis(self, redis):
        async with lock.exclusive_measurement(enabled=False) as outcome:
            assert outcome.held is True
        assert redis.store == {}


class TestItDoesNotStealTheMachine:
    async def test_it_only_releases_its_own_claim(self, redis):
        """If the TTL expired mid-measurement the key belongs to somebody else,
        and deleting it blindly would let a third run start timing while the
        second still believes it holds the machine."""
        key = f"{lock.KEY_PREFIX}:default"

        async with lock.exclusive_measurement():
            redis.store[key] = "someone-elses-token"

        assert redis.store[key] == "someone-elses-token"
        assert key not in redis.deleted

    async def test_a_missing_key_is_not_an_error(self, redis):
        key = f"{lock.KEY_PREFIX}:default"
        async with lock.exclusive_measurement():
            redis.store.pop(key, None)


class TestTheClaimOutlivesTheMeasurement:
    def test_ttl_exceeds_the_callers_timeout(self):
        """A claim expiring mid-measurement is worse than no claim: two runs
        would then time each other while both believe they hold the lock."""
        assert lock.ttl_for(120) > 120
        assert lock.ttl_for(600) > 600

    def test_a_missing_timeout_still_yields_a_claim(self):
        assert lock.ttl_for(None) > 0

    def test_a_nonsense_timeout_does_not_raise(self):
        assert lock.ttl_for("banana") > 0


class TestWhatTheResultRecords:
    async def test_a_serialised_measurement_says_so(self, redis):
        async with lock.exclusive_measurement() as outcome:
            pass
        assert outcome.as_quality()["serialized"] is True
        assert "serialization_note" not in outcome.as_quality()

    async def test_an_unserialised_one_carries_the_caveat(self, monkeypatch):
        async def _broken():
            raise RuntimeError("down")

        monkeypatch.setattr(lock, "_client", _broken)
        async with lock.exclusive_measurement() as outcome:
            pass
        quality = outcome.as_quality()
        assert quality["serialized"] is False
        assert quality["serialization_note"]


class TestDisabledDoesNotClaimProtection:
    """`held` means "go ahead"; `serialized` means "you had the machine". They
    are different questions, and conflating them made every measurement report
    `serialized: true` the moment the feature was switched off -- found in an
    A/B run whose control visibly shared the CPU (both roles `busy`, one with
    233% trial spread) while claiming to be serialised.
    """

    async def test_it_does_not_claim_the_measurement_was_serialised(self, redis):
        async with lock.exclusive_measurement(enabled=False) as outcome:
            pass
        assert outcome.held is True, "the work must still proceed"
        assert outcome.serialized is False
        assert outcome.as_quality()["serialized"] is False

    async def test_it_says_why(self, redis):
        async with lock.exclusive_measurement(enabled=False) as outcome:
            pass
        assert "switched off" in outcome.as_quality()["serialization_note"]

    async def test_an_enabled_uncontended_run_does_claim_it(self, redis):
        async with lock.exclusive_measurement() as outcome:
            pass
        assert outcome.serialized is True
        assert "serialization_note" not in outcome.as_quality()
