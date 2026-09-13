"""Comparing stored timestamps with "now", when the two disagree about tzinfo.

This codebase mixes: 213 columns are ``DateTime(timezone=True)`` and 22 are bare
``DateTime``. Postgres returns aware values for the first kind and naive for the
second, while ``datetime.utcnow()`` — used in 554 places — is always naive.
Comparing across that boundary raises TypeError, which is how
``APIKey.is_valid()`` came to crash for any key with an expiry set.

``utc_now`` replaces ``utcnow()`` (also deprecated since Python 3.12), and
``as_aware_utc`` normalizes a stored value so a comparison cannot depend on
which column it came from. Naive values are read as UTC, which is what every
writer in this codebase intends — ``utcnow()`` produces UTC and drops the label.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def utc_now() -> datetime:
    """Timezone-aware current UTC time.

    Prefer this to ``datetime.utcnow()``, which returns a naive value that then
    misreads as local time in any comparison that does carry a timezone.
    """
    return datetime.now(timezone.utc)


def as_aware_utc(value: Any) -> datetime | None:
    """Return ``value`` as an aware UTC datetime, or None if it is not one.

    Naive input is assumed to be UTC rather than local: everything in this
    codebase writes UTC, and guessing local time would shift stored timestamps
    by the deployment's offset.
    """
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def is_past(value: Any, *, now: datetime | None = None) -> bool:
    """True when ``value`` is a datetime at or before now.

    Returns False for None or a non-datetime, so callers can ask "has this
    expired / come due" without a separate presence check.
    """
    moment = as_aware_utc(value)
    if moment is None:
        return False
    return moment <= (as_aware_utc(now) or utc_now())


def age(value: Any, *, now: datetime | None = None):
    """Elapsed time since ``value``, or None when it is not a datetime.

    Avoids the ``utcnow() - stored`` subtraction that raises whenever the stored
    side happens to carry a timezone.
    """
    moment = as_aware_utc(value)
    if moment is None:
        return None
    return (as_aware_utc(now) or utc_now()) - moment
