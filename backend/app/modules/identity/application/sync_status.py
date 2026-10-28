"""Where the last directory sync's report is kept, and whether one is due.

In Redis: the report is for the admin page, and losing it costs a line of
text, not an account. Never raises; a store that cannot be reached reads as
"no report" and "due".
"""

from __future__ import annotations

import json
from typing import Any, Optional

from loguru import logger

from app.core.config import settings

LAST_REPORT_KEY = "ldap:directory_sync:last"
DUE_KEY = "ldap:directory_sync:due"


def _client():
    import redis

    return redis.from_url(settings.REDIS_URL, decode_responses=True)


# What an unset LDAP_SYNC_INTERVAL_MINUTES means while LDAP is on.
DEFAULT_INTERVAL_MINUTES = 60


def interval_minutes() -> int:
    """Minutes between scheduled syncs; 0 means none.

    Unset is hourly when LDAP is on: off by default, the only deprovisioning
    was someone's next sign-in, which a person who has left never makes. An
    explicit 0 still turns it off.
    """
    configured = getattr(settings, "LDAP_SYNC_INTERVAL_MINUTES", None)
    if configured is None:
        if not getattr(settings, "LDAP_ENABLED", False):
            return 0
        return DEFAULT_INTERVAL_MINUTES
    try:
        return max(0, int(configured))
    except (TypeError, ValueError):
        return 0


def claim_due_run() -> bool:
    """True once per interval, across every worker: the beat entry fires
    more often than the interval, and two workers must not both run it."""
    minutes = interval_minutes()
    if minutes <= 0:
        return False
    client = None
    try:
        client = _client()
        # A little under the interval, so a run is not skipped because the
        # beat fired a few seconds early.
        return bool(client.set(DUE_KEY, "1", nx=True, ex=max(60, minutes * 60 - 30)))
    except Exception as exc:
        logger.warning(f"LDAP sync: could not claim the run ({exc}); skipping")
        return False
    finally:
        if client is not None:
            client.close()


def save_report(report: dict) -> None:
    client = None
    try:
        client = _client()
        client.set(LAST_REPORT_KEY, json.dumps(report, default=str))
    except Exception as exc:
        logger.warning(f"LDAP sync: could not store the report: {exc}")
    finally:
        if client is not None:
            client.close()


def last_report() -> Optional[dict[str, Any]]:
    client = None
    try:
        client = _client()
        raw = client.get(LAST_REPORT_KEY)
        return json.loads(raw) if raw else None
    except Exception as exc:
        logger.debug(f"LDAP sync: could not read the last report: {exc}")
        return None
    finally:
        if client is not None:
            client.close()
