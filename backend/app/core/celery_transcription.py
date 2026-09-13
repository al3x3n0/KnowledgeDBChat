"""Celery configuration for the dedicated transcription worker.

Transcription is the only thing in this codebase that needs Whisper, librosa,
speechbrain and resemblyzer -- and, underneath them, numba and llvmlite, which
are 185 MB on their own. Every API container, every Celery container and every
replica of both carried that to run a feature one worker performs. This app,
and the image built from `Dockerfile.transcription-worker`, are where it lives
now.

Like `celery_latex`, this deliberately does not import the full include list
from `app.core.celery`: a worker imports every module it is told to include at
startup, and this one has exactly one job.
"""

from celery import Celery

from app.core.config import settings

celery_app = Celery(
    "knowledge_db_transcription",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=[
        "app.tasks.transcription_tasks",
    ],
)

TRANSCRIPTION_QUEUE = (
    getattr(settings, "TRANSCRIPTION_CELERY_QUEUE", "transcription") or "transcription"
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_routes={
        "app.tasks.transcription_tasks.transcribe_document": {
            "queue": TRANSCRIPTION_QUEUE
        },
    },
    task_annotations={
        # Model download plus a long media decode on CPU. The dispatchers size
        # each call from the file's real duration (services/media_probe.py);
        # this is the ceiling for a call that arrives without one.
        "app.tasks.transcription_tasks.transcribe_document": {
            "soft_time_limit": 5 * 60 * 60,
            "time_limit": 6 * 60 * 60,
        },
    },
    task_time_limit=6 * 60 * 60,
    task_soft_time_limit=5 * 60 * 60,
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
    #: Recycle a child that has grown past this, in KiB. This worker runs
    #: --concurrency=1, so the child is effectively the whole container and the
    #: guard has one job: recycle before the container is OOM-killed, which
    #: loses the task and the worker with it. Restarting between tasks is
    #: strictly better than being killed during one.
    #:
    #: 4 GiB is 75% of the container's 6g mem_limit, deliberately set
    #: from that ceiling rather than from the workload: Whisper holds a model whose size depends on WHISPER_MODEL, unmeasured here.
    #: The worker sits at ~353 MB idle, so this cannot fire on a healthy idle
    #: child. Celery checks it after a task returns, so a single task that
    #: balloons mid-run is still the container limit's problem, not this one.
    worker_max_memory_per_child=4608 * 1024,
)
