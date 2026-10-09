"""The query the agent-worker autoscaler asks Postgres.

It lives in the Helm chart, where nothing type-checks it and nothing runs it
until a cluster does. A wrong count there is quiet in both directions: too
low removes pods that are running jobs, too high holds pods nobody needs.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import text

from app.models.agent_job import AgentJob, AgentJobStatus
from app.services import agent_job_user_cap
from app.tasks import agent_job_tasks
from tests.conftest import TEST_ON_POSTGRES

CHART = Path(__file__).resolve().parents[2] / "deploy/helm/knowledgedbchat"
QUERY = (CHART / "files/agent-worker-demand.sql").read_text()


def test_it_names_the_status_and_the_reason_the_code_writes():
    assert f"status = '{AgentJobStatus.PENDING.value}'" in QUERY
    assert f"<> '{agent_job_user_cap.QUEUE_REASON}'" in QUERY


def test_it_reads_the_reason_from_where_the_task_writes_it():
    job = AgentJob(results={})
    agent_job_tasks._write_scheduler_state(job, {"queue_reason": "anything"})
    path = "{execution_strategy,scheduler_state,queue_reason}"

    assert path in QUERY
    found = job.results
    for key in path.strip("{}").split(","):
        found = found[key]
    assert found == "anything"


def test_the_chart_asks_this_query_and_scales_per_pod_capacity():
    template = (CHART / "templates/workers/celery-agents-scaledobject.yaml").read_text()

    assert '.Files.Get "files/agent-worker-demand.sql"' in template
    assert "targetQueryValue: {{ .Values.celeryAgents.concurrency" in template
    # The password is read from the worker's environment, never written here.
    assert "passwordFromEnv: POSTGRES_PASSWORD" in template


def _job(status, **fields) -> AgentJob:
    return AgentJob(
        id=uuid4(),
        name="demand",
        goal="count me, or not",
        job_type="research",
        user_id=fields.pop("user_id"),
        status=status,
        config={},
        results=fields.pop("results", {}),
        **fields,
    )


@pytest.mark.skipif(not TEST_ON_POSTGRES, reason="the query is Postgres SQL")
@pytest.mark.asyncio
async def test_it_counts_jobs_a_worker_holds_and_jobs_that_could_start(
    db_session, test_user
):
    now = datetime.utcnow()
    soon, ago = now + timedelta(minutes=5), now - timedelta(minutes=5)
    held_by_cap = agent_job_tasks._write_scheduler_state(
        AgentJob(results={}), {"queue_reason": agent_job_user_cap.QUEUE_REASON}
    )
    user = test_user.id
    counted = [
        # A worker holds it.
        _job("running", user_id=user, execution_lease_expires_at=soon),
        # Finished, but its worker is still writing up: the slot is in use.
        _job("completed", user_id=user, execution_lease_expires_at=soon),
        # Queued and could start now.
        _job("pending", user_id=user),
        _job("pending", user_id=user, next_run_at=ago, schedule_type="recurring"),
        _job("pending", user_id=user, results=None),
    ]
    not_counted = [
        # Its worker died; the sweep will make it pending again.
        _job("running", user_id=user, execution_lease_expires_at=ago),
        # Waiting for its schedule.
        _job("pending", user_id=user, next_run_at=soon, schedule_type="recurring"),
        # Held by its owner's cap: more pods would not let it run.
        _job(
            "pending",
            user_id=user,
            results={"execution_strategy": {"scheduler_state": held_by_cap}},
        ),
        _job("paused", user_id=user),
        _job("completed", user_id=user),
        _job("failed", user_id=user),
    ]
    db_session.add_all(counted + not_counted)
    await db_session.commit()

    demand = (await db_session.execute(text(QUERY))).scalar()

    assert demand == len(counted)
