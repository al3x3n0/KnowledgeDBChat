"""API keys: creating one, presenting one, and its own rate limits.

The service had no test. Two things were wrong, each invisible on SQLite or
without a second request:

* `key_prefix` was VARCHAR(8) and the service stores 12 characters. Postgres
  refuses that, so no key could be created on a real deployment. This file is
  in the Postgres slice for that reason.
* every key carries a per-minute and a per-day limit, set in the form and
  shown in the list, and nothing read either.
"""

from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app.mcp import server as mcp_server
from app.mcp.auth import enforce_rate_limits, validate_api_key
from app.models.api_key import APIKey, APIKeyUsageLog
from app.services.api_key_service import api_key_service

pytestmark = pytest.mark.unit


async def _key(db, user, **fields):
    key, plain = await api_key_service.create_api_key(
        db=db, user_id=user.id, name="ci", **fields
    )
    return key, plain


def _request(plain, path="/api/v1/mcp/tools"):
    return SimpleNamespace(
        headers={"Authorization": f"Bearer {plain}", "User-Agent": "test"},
        query_params={},
        url=SimpleNamespace(path=path),
        method="GET",
        client=SimpleNamespace(host="127.0.0.1"),
    )


def test_the_prefix_the_service_writes_fits_its_column():
    _full, prefix, _hash = api_key_service.generate_key()

    assert len(prefix) <= APIKey.__table__.c.key_prefix.type.length


async def test_a_key_can_be_created_and_then_presented(db_session, test_user):
    key, plain = await _key(db_session, test_user, scopes=["read"])

    assert plain.startswith(key.key_prefix)
    assert key.key_hash != plain

    auth = await validate_api_key(plain, db_session)
    assert auth.user.id == test_user.id
    assert auth.has_scope("read") and not auth.has_scope("write")
    assert await validate_api_key(plain + "x", db_session) is None


async def test_a_revoked_or_expired_key_is_refused(db_session, test_user):
    revoked, revoked_plain = await _key(db_session, test_user)
    assert await api_key_service.revoke_api_key(db_session, revoked.id, test_user.id)
    expired, expired_plain = await _key(db_session, test_user, expires_in_days=1)
    expired.expires_at = datetime.utcnow() - timedelta(days=1)
    await db_session.commit()

    assert await validate_api_key(revoked_plain, db_session) is None
    assert await validate_api_key(expired_plain, db_session) is None
    # Turning a revoked key back on does not un-revoke it.
    await api_key_service.update_api_key(
        db_session, revoked.id, test_user.id, is_active=True
    )
    assert await validate_api_key(revoked_plain, db_session) is None


async def test_a_key_is_its_owners_to_read_and_revoke(
    db_session, test_user, admin_user
):
    key, _ = await _key(db_session, admin_user)

    assert await api_key_service.get_api_key(db_session, key.id, test_user.id) is None
    assert not await api_key_service.revoke_api_key(db_session, key.id, test_user.id)
    assert await api_key_service.list_api_keys(db_session, test_user.id) == []


async def test_requests_beyond_the_per_minute_limit_are_refused(db_session, test_user):
    key, plain = await _key(db_session, test_user, rate_limit_per_minute=2)

    for _ in range(2):
        await mcp_server.get_mcp_auth(_request(plain), db_session)

    with pytest.raises(HTTPException) as refused:
        await mcp_server.get_mcp_auth(_request(plain), db_session)
    assert refused.value.status_code == 429
    assert refused.value.headers["Retry-After"] == "60"
    assert "2 requests per minute" in refused.value.detail


async def test_the_limit_is_each_keys_own(db_session, test_user):
    busy, busy_plain = await _key(db_session, test_user, rate_limit_per_minute=1)
    _other, other_plain = await _key(db_session, test_user, rate_limit_per_minute=1)

    await mcp_server.get_mcp_auth(_request(busy_plain), db_session)
    with pytest.raises(HTTPException):
        await mcp_server.get_mcp_auth(_request(busy_plain), db_session)

    await mcp_server.get_mcp_auth(_request(other_plain), db_session)


async def test_requests_from_an_earlier_minute_do_not_count(db_session, test_user):
    key, plain = await _key(db_session, test_user, rate_limit_per_minute=1)
    db_session.add(
        APIKeyUsageLog(
            api_key_id=key.id,
            endpoint="/mcp/tools",
            method="GET",
            status_code=200,
            timestamp=datetime.utcnow() - timedelta(minutes=5),
        )
    )
    await db_session.commit()

    await mcp_server.get_mcp_auth(_request(plain), db_session)


async def test_the_per_day_limit_counts_the_whole_day(db_session, test_user):
    key, plain = await _key(
        db_session, test_user, rate_limit_per_minute=1000, rate_limit_per_day=3
    )
    for hours in (2, 5, 9):
        db_session.add(
            APIKeyUsageLog(
                api_key_id=key.id,
                endpoint="/mcp/tools",
                method="GET",
                status_code=200,
                timestamp=datetime.utcnow() - timedelta(hours=hours),
            )
        )
    await db_session.commit()

    with pytest.raises(HTTPException) as refused:
        await mcp_server.get_mcp_auth(_request(plain), db_session)
    assert "3 requests per day" in refused.value.detail
    assert refused.value.headers["Retry-After"] == "3600"


async def test_a_tool_call_is_counted_once(db_session, test_user):
    """The tool-call route logs itself, with its outcome; the dependency must
    not log it a second time."""
    key, plain = await _key(db_session, test_user)
    path = "/api/v1/mcp" + mcp_server.TOOL_CALL_PATH

    await mcp_server.get_mcp_auth(_request(plain, path=path), db_session)

    stats = await api_key_service.get_usage_stats(db_session, key.id)
    assert stats["total_requests"] == 0


async def test_a_key_with_no_limit_is_not_limited(db_session, test_user):
    key, _ = await _key(db_session, test_user)
    key.rate_limit_per_minute = None
    key.rate_limit_per_day = 0
    for _ in range(3):
        db_session.add(
            APIKeyUsageLog(
                api_key_id=key.id, endpoint="/x", method="GET", status_code=200
            )
        )
    await db_session.commit()

    await enforce_rate_limits(db_session, key)


async def test_usage_is_reported_for_the_period_asked(db_session, test_user):
    key, plain = await _key(db_session, test_user)
    await mcp_server.get_mcp_auth(_request(plain), db_session)
    await mcp_server.get_mcp_auth(_request(plain, "/api/v1/mcp/sources"), db_session)

    stats = await api_key_service.get_usage_stats(db_session, key.id, days=1)

    assert stats["total_requests"] == 2
    assert stats["lifetime_requests"] == 2
    assert {e["endpoint"] for e in stats["top_endpoints"]} == {
        "/api/v1/mcp/tools",
        "/api/v1/mcp/sources",
    }
