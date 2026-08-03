"""Focused tests for the modular autonomous-job export API."""

from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import HTTPException

from app.modules.autonomy.api.job_exports import build_job_export_api


class _ScalarResult:
    def __init__(self, value):
        self.value = value

    def scalar_one_or_none(self):
        return self.value


class _Database:
    def __init__(self, job):
        self.job = job
        self.execute_calls = 0

    async def execute(self, _statement):
        self.execute_calls += 1
        return _ScalarResult(self.job)


class _Exporter:
    def __init__(self):
        self.standard_call = None
        self.enhanced_call = None

    def export(self, **kwargs):
        self.standard_call = kwargs
        return b"standard-export"

    async def export_enhanced(self, **kwargs):
        self.enhanced_call = kwargs
        return b"enhanced-export"


@pytest.mark.asyncio
async def test_job_export_rejects_invalid_format_before_querying():
    database = _Database(job=None)
    api = build_job_export_api()

    with pytest.raises(HTTPException) as exc_info:
        await api.export_job_results(
            job_id=uuid4(),
            format="html",
            style="professional",
            include_log=False,
            include_metadata=True,
            enhance=False,
            db=database,
            current_user=SimpleNamespace(id=uuid4()),
        )

    assert exc_info.value.status_code == 400
    assert database.execute_calls == 0


@pytest.mark.asyncio
async def test_job_export_returns_standard_attachment_with_safe_filename():
    user = SimpleNamespace(id=uuid4())
    job = SimpleNamespace(id=uuid4(), user_id=user.id, name="Repair / Report!")
    database = _Database(job=job)
    exporter = _Exporter()
    api = build_job_export_api(exporter_factory=lambda **_kwargs: exporter)

    response = await api.export_job_results(
        job_id=job.id,
        format="pdf",
        style="technical",
        include_log=True,
        include_metadata=False,
        enhance=False,
        db=database,
        current_user=user,
    )

    assert response.body == b"standard-export"
    assert response.media_type == "application/pdf"
    assert response.headers["content-disposition"] == (
        'attachment; filename="Repair  Report_report.pdf"'
    )
    assert exporter.standard_call["job"] is job
    assert exporter.standard_call["include_log"] is True
    assert exporter.standard_call["include_metadata"] is False


@pytest.mark.asyncio
async def test_job_export_enhancement_receives_user_settings():
    user = SimpleNamespace(id=uuid4())
    job = SimpleNamespace(id=uuid4(), user_id=user.id, name="Enhanced")
    database = _Database(job=job)
    exporter = _Exporter()
    settings = object()

    async def load_settings(**kwargs):
        assert kwargs == {"db": database, "user_id": user.id}
        return settings

    api = build_job_export_api(
        exporter_factory=lambda **_kwargs: exporter,
        load_user_settings=load_settings,
    )
    response = await api.export_job_results(
        job_id=job.id,
        format="docx",
        style="casual",
        include_log=False,
        include_metadata=True,
        enhance=True,
        db=database,
        current_user=user,
    )

    assert response.body == b"enhanced-export"
    assert exporter.enhanced_call["user_settings"] is settings
    assert exporter.enhanced_call["user_id"] == user.id
