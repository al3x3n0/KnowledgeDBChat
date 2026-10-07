"""Agent jobs run on their own queue, and every deployment has a consumer for it.

A queue nothing consumes fails silently: the job is accepted and waits for
ever. The transcription and LaTeX queues have that contract on purpose (their
workers are optional); the agents queue must not, because agent jobs are the
platform's main workload. These read the compose files rather than trusting a
comment, since a service renamed or an overlay edited is all it takes.
(The Helm chart is checked by `make helm-validate`; its general worker takes
the agents queue whenever celeryAgents is disabled.)
"""

from pathlib import Path

import pytest
import yaml

from app.core.celery import celery_app
from app.core.config import settings

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
AGENT_TASK = "app.tasks.agent_job_tasks.execute_agent_job_task"


def _services(name: str) -> dict:
    path = ROOT / name
    if not path.exists():  # pragma: no cover - a checkout without deploy files
        pytest.skip(f"{name} not present")
    return (yaml.safe_load(path.read_text()) or {}).get("services") or {}


def _queues(command) -> list[str]:
    words = command if isinstance(command, list) else str(command or "").split()
    if "-Q" not in words:
        return ["celery"]  # the default queue
    return words[words.index("-Q") + 1].split(",")


def test_agent_jobs_are_routed_to_their_queue():
    route = celery_app.conf.task_routes[AGENT_TASK]
    assert route["queue"] == settings.AGENT_JOB_CELERY_QUEUE


@pytest.mark.parametrize("compose", ["docker-compose.yml", "docker-compose.prod.yml"])
def test_every_stack_consumes_the_agents_queue(compose):
    services = _services(compose)
    consumers = [
        name
        for name, service in services.items()
        if settings.AGENT_JOB_CELERY_QUEUE in _queues(service.get("command"))
        and "celery" in str(service.get("command"))
    ]
    assert consumers, f"nothing in {compose} consumes the agents queue"
    # And it is not the general worker: separating them is the point.
    assert "celery" not in consumers


@pytest.mark.parametrize(
    "overlay",
    ["docker-compose.docker-tools.yml", "docker-compose.override.example.yml"],
)
def test_an_overlay_gives_the_agents_worker_what_it_gives_the_general_one(overlay):
    # `extends` copies the base file's `celery`, not an overlay's changes to
    # it, so an overlay must name celery_agents too -- or agent jobs, the
    # thing that needs the LLM keys and the Docker daemon, run without them.
    services = _services(overlay)
    if "celery" not in services:
        pytest.skip(f"{overlay} does not configure the general worker")
    assert services.get("celery_agents") == services["celery"]
