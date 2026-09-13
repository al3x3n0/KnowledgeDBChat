"""Comparing stored timestamps with now, across the tz-aware/naive boundary.

213 columns are DateTime(timezone=True) and 22 are bare DateTime, while
datetime.utcnow() — used in 554 places — is always naive. Comparing across that
boundary raises TypeError, which is how APIKey.is_valid() came to crash for any
key that had an expiry set: an auth-path call that raised instead of returning
a decision.
"""

from datetime import datetime, timedelta, timezone

import pytest

from app.models.api_key import APIKey
from app.utils.datetimes import age, as_aware_utc, is_past, utc_now

AWARE_FUTURE = datetime.now(timezone.utc) + timedelta(days=1)
AWARE_PAST = datetime.now(timezone.utc) - timedelta(days=1)
NAIVE_FUTURE = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(days=1)
NAIVE_PAST = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=1)


def test_utc_now_is_aware():
    assert utc_now().tzinfo is not None


def test_naive_values_are_read_as_utc_not_local():
    # Reading them as local time would shift every stored timestamp by the
    # deployment's offset; every writer here means UTC.
    naive = datetime(2026, 1, 1, 12, 0, 0)
    assert as_aware_utc(naive) == datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def test_aware_values_are_converted_not_relabelled():
    other_zone = timezone(timedelta(hours=5))
    value = datetime(2026, 1, 1, 17, 0, 0, tzinfo=other_zone)
    assert as_aware_utc(value) == datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def test_non_datetimes_are_not_datetimes():
    for value in (None, "2026-01-01", 0, object()):
        assert as_aware_utc(value) is None
        assert is_past(value) is False
        assert age(value) is None


@pytest.mark.parametrize(
    "value,expected",
    [
        (AWARE_PAST, True),
        (AWARE_FUTURE, False),
        (NAIVE_PAST, True),
        (NAIVE_FUTURE, False),
        (None, False),
    ],
)
def test_is_past_agrees_regardless_of_tzinfo(value, expected):
    assert is_past(value) is expected


def test_age_works_across_the_boundary():
    assert age(AWARE_PAST).days == 0 or age(AWARE_PAST).days == 1
    assert age(NAIVE_PAST) is not None
    assert age(AWARE_FUTURE).total_seconds() < 0


def _key(expires_at):
    key = APIKey()
    key.is_active = True
    key.revoked_at = None
    key.expires_at = expires_at
    return key


@pytest.mark.parametrize(
    "expires_at,valid",
    [
        (AWARE_FUTURE, True),
        (AWARE_PAST, False),
        (NAIVE_FUTURE, True),
        (NAIVE_PAST, False),
        (None, True),
    ],
)
def test_api_key_validity_never_raises_on_an_expiry(expires_at, valid):
    """The regression: an aware expires_at made this raise TypeError.

    Postgres returns aware values for expires_at, so this was every key that
    had an expiry configured — the check failed loudly instead of deciding.
    """
    assert _key(expires_at).is_valid() is valid


def test_api_key_still_honours_revocation_and_inactivity():
    revoked = _key(AWARE_FUTURE)
    revoked.revoked_at = AWARE_PAST
    assert revoked.is_valid() is False

    inactive = _key(AWARE_FUTURE)
    inactive.is_active = False
    assert inactive.is_valid() is False
