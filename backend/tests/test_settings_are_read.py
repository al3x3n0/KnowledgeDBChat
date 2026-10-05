"""A setting that nothing reads is a promise the deployment cannot keep.

`TRAINING_ENABLED` was documented as the gate on the training subsystem and
read nowhere, so setting it to false changed nothing. The concurrency limit
and two dataset limits beside it were the same: declared, defaulted,
documented, inert.

A field nothing reads may not simply be deleted -- a deployed `.env` that
still names it makes `Settings` refuse to load -- so the ones that remain are
listed here, and the list may only shrink.
"""

import re
from pathlib import Path
from uuid import uuid4

import pytest

from app.core.config import Settings

pytestmark = pytest.mark.unit

BACKEND = Path(__file__).resolve().parents[1]

#: `    NAME: type = default` inside the Settings class.
DECLARATION = re.compile(r"^\s{4}[A-Z][A-Z0-9_]*\s*:")

#: Declared and deliberately inert. Each is marked NOT READ in config.py with
#: the reason. Wiring one up means removing it from here.
DECLARED_BUT_NOT_READ = {
    "CONFLUENCE_URL",
    "CONFLUENCE_USER",
    "CONFLUENCE_API_TOKEN",
    "KG_EXTRACTION_BATCH_SIZE",
    "TRAINING_DEFAULT_BACKEND",
    "TRAINING_LOCAL_MAX_GPU_MEMORY_GB",
    "MODAL_API_KEY",
    "RUNPOD_API_KEY",
}


def _readers() -> str:
    """Everything that could read a setting, except its own declaration."""
    parts = []
    for path in (BACKEND / "app").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if path.name == "config.py" and path.parent.name == "core":
            # A field's own declaration and the comments around it are not a
            # use. Everything else in the module is: a property that derives
            # a list from a field reads it, and so does the logging setup.
            text = "\n".join(
                line
                for line in text.splitlines()
                if not DECLARATION.match(line) and not line.lstrip().startswith("#")
            )
        parts.append(text)
    for path in (BACKEND / "scripts").rglob("*.py"):
        parts.append(path.read_text(encoding="utf-8"))
    for path in BACKEND.glob("*.py"):
        parts.append(path.read_text(encoding="utf-8"))
    return "\n".join(parts)


def test_every_setting_is_read_or_admitted_inert():
    text = _readers()
    fields = list(Settings.model_fields)
    assert len(fields) > 100, "too few settings; this guard would pass vacuously"
    unread = sorted(
        name
        for name in fields
        if name not in text and name not in DECLARED_BUT_NOT_READ
    )
    assert not unread, (
        "Declared in config.py and read nowhere. Use it, or mark it NOT READ "
        "there and add it to DECLARED_BUT_NOT_READ:\n"
        + "\n".join(f"  - {name}" for name in unread)
    )


def test_the_inert_list_only_names_settings_that_are_still_inert():
    text = _readers()
    now_read = sorted(name for name in DECLARED_BUT_NOT_READ if name in text)
    assert not now_read, f"these are read now; remove them from the list: {now_read}"
    missing = sorted(DECLARED_BUT_NOT_READ - set(Settings.model_fields))
    assert not missing, f"listed but no longer declared: {missing}"


def test_training_routes_are_refused_when_training_is_disabled(
    client, auth_headers, monkeypatch
):
    from app.core.config import settings

    assert client.get("/api/v1/training/jobs", headers=auth_headers).status_code == 200
    monkeypatch.setattr(settings, "TRAINING_ENABLED", False, raising=False)
    for path in (
        "/api/v1/training/jobs",
        "/api/v1/training/datasets",
        "/api/v1/training/models",
    ):
        response = client.get(path, headers=auth_headers)
        assert response.status_code == 403, path
        assert "TRAINING_ENABLED" in response.json()["detail"]


class TestTheTrainingLimitsLimit:
    async def test_a_job_is_refused_when_the_machine_is_already_full(
        self, db_session, test_user, monkeypatch
    ):
        from app.core.config import settings
        from app.models.training_job import TrainingJob, TrainingJobStatus
        from app.services.training_service import TrainingService

        def job(status):
            return TrainingJob(
                user_id=test_user.id,
                name=f"job-{status}",
                training_method="lora",
                training_backend="local",
                base_model="m",
                # SQLite does not enforce the foreign key; the column only has
                # to be non-null for a row the start path will count.
                dataset_id=uuid4(),
                status=status,
            )

        running = job(TrainingJobStatus.TRAINING.value)
        waiting = job(TrainingJobStatus.PENDING.value)
        db_session.add_all([running, waiting])
        await db_session.commit()

        monkeypatch.setattr(settings, "TRAINING_MAX_CONCURRENT_JOBS", 1, raising=False)
        with pytest.raises(ValueError) as refused:
            await TrainingService().start_job(db_session, waiting.id)
        assert "TRAINING_MAX_CONCURRENT_JOBS" in str(refused.value)
        assert "1 training job(s) are already running" in str(refused.value)
