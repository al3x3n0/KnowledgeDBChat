"""The one cleanup the export, presentation and repo-report tasks share.

Each task had its own copy, and all three deleted the row whatever
`StorageService.delete_file` answered. That method returns False rather than
raising, so a file MinIO kept lost the only row that pointed at it.
"""

import inspect
from datetime import datetime, timedelta
from uuid import uuid4

import pytest
from sqlalchemy import select

from app.models.export_job import ExportJob
from app.services.storage_service import StorageService
from app.tasks import export_tasks, presentation_tasks, repo_report_tasks
from app.tasks.job_support import prune_finished_jobs

pytestmark = pytest.mark.unit


class FakeStorage:
    """`StorageService.delete_file`: True unless the path is in `refuse`."""

    def __init__(self, refuse=()):
        self.refuse = set(refuse)
        self.deleted = []

    async def delete_file(self, *args, **kwargs):
        bound = inspect.signature(StorageService.delete_file).bind(
            None, *args, **kwargs
        )
        path = bound.arguments["object_path"]
        if path in self.refuse:
            return False
        self.deleted.append(path)
        return True


async def _export(db, user, *, status, age_days, file_path=None):
    job = ExportJob(
        id=uuid4(),
        user_id=user.id,
        export_type="document",
        output_format="pdf",
        source_type="document",
        source_id=uuid4(),
        title="An export",
        status=status,
        file_path=file_path,
        created_at=datetime.utcnow() - timedelta(days=age_days),
    )
    db.add(job)
    await db.commit()
    return job


async def _remaining(db):
    return {j.id for j in (await db.execute(select(ExportJob))).scalars().all()}


async def test_old_finished_jobs_go_and_their_files_with_them(db_session, test_user):
    old = await _export(
        db_session, test_user, status="completed", age_days=40, file_path="exports/a"
    )
    failed = await _export(db_session, test_user, status="failed", age_days=40)
    storage = FakeStorage()

    counts = await prune_finished_jobs(
        db_session, ExportJob, older_than_days=30, storage=storage, label="export"
    )

    assert counts == {"deleted": 2, "kept": 0}
    assert storage.deleted == ["exports/a"]
    assert not {old.id, failed.id} & await _remaining(db_session)


async def test_recent_and_unfinished_jobs_are_left_alone(db_session, test_user):
    recent = await _export(db_session, test_user, status="completed", age_days=3)
    running = await _export(db_session, test_user, status="processing", age_days=90)

    counts = await prune_finished_jobs(
        db_session,
        ExportJob,
        older_than_days=30,
        storage=FakeStorage(),
        label="export",
    )

    assert counts == {"deleted": 0, "kept": 0}
    assert {recent.id, running.id} <= await _remaining(db_session)


async def test_a_file_that_would_not_delete_keeps_its_row(db_session, test_user):
    stuck = await _export(
        db_session, test_user, status="completed", age_days=40, file_path="exports/b"
    )

    counts = await prune_finished_jobs(
        db_session,
        ExportJob,
        older_than_days=30,
        storage=FakeStorage(refuse={"exports/b"}),
        label="export",
    )

    assert counts == {"deleted": 0, "kept": 1}
    assert stuck.id in await _remaining(db_session)


@pytest.mark.parametrize("task", [export_tasks, presentation_tasks, repo_report_tasks])
def test_every_cleanup_task_uses_it(task):
    source = inspect.getsource(task)
    assert "prune_finished_jobs(" in source
    assert "delete_file(" not in source
