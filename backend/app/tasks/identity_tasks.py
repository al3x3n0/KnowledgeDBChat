"""Scheduled work on accounts: reconciling directory-managed ones."""

from __future__ import annotations

from typing import Any, Dict

from loguru import logger

from app.core.celery import celery_app
from app.core.database import create_celery_session
from app.tasks.job_support import run_async


@celery_app.task(name="app.tasks.identity_tasks.sync_directory_accounts")
def sync_directory_accounts() -> Dict[str, Any]:
    """Run the directory sync when ``LDAP_SYNC_INTERVAL_MINUTES`` says it is due."""
    from app.modules.identity.application import sync_status
    from app.services.ldap_service import ldap_service

    if not ldap_service.is_configured() or not sync_status.claim_due_run():
        return {"ran": False}
    return run_async(_sync())


async def _sync() -> Dict[str, Any]:
    from datetime import datetime

    from app.modules.identity.application import directory_sync, sync_status
    from app.services.ldap_service import LdapError

    async with create_celery_session()() as db:
        try:
            report = (await directory_sync.sync(db)).as_dict()
        except LdapError as exc:
            # The directory could not be read, which says nothing about who
            # left: nothing is changed, and the page says why.
            logger.warning(f"LDAP sync could not read the directory: {exc}")
            report = {
                "ran_at": datetime.utcnow().isoformat(),
                "dry_run": False,
                "error": f"The directory could not be read: {exc}",
            }
    sync_status.save_report(report)
    return {"ran": True, **report}
