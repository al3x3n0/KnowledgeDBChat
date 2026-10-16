"""Where a run's time went: the runner's phase times, the profile, the route."""

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.agent_core.runtime import AgentRuntimeRunner
from app.models.agent_job import AgentJob
from app.services import agent_run_profile

pytestmark = pytest.mark.unit

T0 = datetime(2026, 10, 14, 12, 0, 0)


class SlowActAdapter:
    """One iteration whose act phase takes measurably longer than the rest."""

    def __init__(self):
        self.done = False
        self.bundle = None

    async def can_continue(self):
        return not self.done

    async def on_iteration_start(self):
        pass

    async def observe_phase(self):
        return {}

    async def think_phase(self, observation):
        return {}

    async def act_phase(self, decision):
        await asyncio.sleep(0.05)
        return {"action": {"tool": "simulate_c_workload"}}

    async def evaluate_phase(self, decision, bundle):
        return {}

    async def on_iteration_complete(self, observation, decision, bundle, evaluation):
        self.bundle = bundle
        self.done = True

    async def on_iteration_error(self, exc):
        return False

    async def build_run_result(self):
        return {}


async def test_the_runner_times_each_phase():
    adapter = SlowActAdapter()

    await AgentRuntimeRunner().run(adapter)

    phases = adapter.bundle["phase_ms"]
    assert set(phases) == {"observe", "think", "act", "evaluate"}
    assert phases["act"] >= 40 and phases["act"] > phases["think"]


def job(
    name, *, created, started=None, completed=None, parent=None, log=(), stage=None
):
    return SimpleNamespace(
        id=uuid4(),
        parent_job_id=parent.id if parent else None,
        root_job_id=None,
        name=name,
        status="completed" if completed else "running",
        config={"pipeline_stage": stage} if stage else {},
        created_at=created,
        started_at=started,
        completed_at=completed,
        execution_log=list(log),
    )


def iteration(n, at, tool, phase_ms=None, success=True):
    entry = {
        "phase": "iteration_complete",
        "iteration": n,
        "action": tool,
        "success": success,
        "timestamp": at.isoformat(),
    }
    if phase_ms is not None:
        entry["phase_ms"] = phase_ms
    return entry


class TestTheProfile:
    def run(self):
        planner = job(
            "planner",
            created=T0,
            started=T0 + timedelta(seconds=30),
            completed=T0 + timedelta(minutes=10),
            log=[
                iteration(1, T0 + timedelta(minutes=4), "search_documents"),
                iteration(
                    2,
                    T0 + timedelta(minutes=9),
                    "propose_pipeline",
                    {"observe": 1000, "think": 200000, "act": 60000, "evaluate": 500},
                ),
            ],
        )
        # Aware, as Postgres returns it: the profile must mix the two.
        stage = job(
            "stage",
            stage="implement",
            parent=planner,
            created=(T0 + timedelta(minutes=40)).replace(tzinfo=timezone.utc),
            started=(T0 + timedelta(minutes=41)).replace(tzinfo=timezone.utc),
            completed=(T0 + timedelta(minutes=61)).replace(tzinfo=timezone.utc),
            log=[
                iteration(
                    1,
                    T0 + timedelta(minutes=51),
                    "check_implementation",
                    {"observe": 2000, "think": 300000, "act": 240000, "evaluate": 1000},
                    success=False,
                ),
                iteration(
                    2,
                    T0 + timedelta(minutes=60),
                    "check_implementation",
                    {"observe": 2000, "think": 200000, "act": 300000, "evaluate": 1000},
                ),
            ],
        )
        return planner, stage

    def test_each_job_says_queued_ran_and_how_long_after_its_parent(self):
        planner, stage = self.run()

        profile = agent_run_profile.build([stage, planner], root_id=planner.id)

        first, second = profile["jobs"]
        assert first["name"] == "planner" and second["stage"] == "implement"
        assert first["queued_seconds"] == 30 and first["ran_seconds"] == 570
        # The plan waited 30 minutes between the planner ending and the stage.
        assert second["after_parent_seconds"] == 1800
        assert profile["wall_seconds"] == 61 * 60

    def test_iterations_are_spans_and_carry_their_phases(self):
        planner, stage = self.run()

        profile = agent_run_profile.build([planner, stage], root_id=planner.id)

        spans = [i["seconds"] for i in profile["jobs"][0]["iterations"]]
        assert spans == [210, 300]
        assert profile["jobs"][0]["iterations"][0]["phase_ms"] is None
        assert profile["iterations_without_phase_times"] == 1

    def test_the_totals_add_the_phases_and_rank_the_tools(self):
        planner, stage = self.run()

        totals = agent_run_profile.build([planner, stage], root_id=planner.id)["totals"]

        assert totals["phase_ms"]["think"] == 700000
        assert totals["tools"][0] == {
            "tool": "check_implementation",
            "calls": 2,
            "failures": 1,
            "act_ms": 540000,
        }
        assert totals["after_parent_seconds"] == 1800

    def test_a_running_job_is_measured_to_now(self):
        running = job("live", created=T0, started=T0 + timedelta(seconds=10))
        now = (T0 + timedelta(minutes=5)).replace(tzinfo=timezone.utc)

        profile = agent_run_profile.build([running], root_id=running.id, now=now)

        assert profile["jobs"][0]["ran_seconds"] == 290


class TestTheRoute:
    def _jobs(self, db_session, user):
        root = AgentJob(
            name="planner",
            goal="g",
            job_type="research",
            user_id=user.id,
            status="completed",
            started_at=T0,
            completed_at=T0 + timedelta(minutes=5),
            execution_log=[
                iteration(1, T0 + timedelta(minutes=2), "search_documents", {"act": 5})
            ],
        )
        return root

    async def test_the_owner_reads_the_whole_run(
        self, client, auth_headers, db_session, test_user
    ):
        root = self._jobs(db_session, test_user)
        db_session.add(root)
        await db_session.flush()
        child = AgentJob(
            name="stage",
            goal="g",
            job_type="research",
            user_id=test_user.id,
            status="running",
            parent_job_id=root.id,
            root_job_id=root.id,
            started_at=T0 + timedelta(minutes=6),
        )
        db_session.add(child)
        await db_session.commit()

        response = client.get(
            f"/api/v1/agent-jobs/{child.id}/profile", headers=auth_headers
        )

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["root_job_id"] == str(root.id)
        assert [j["name"] for j in body["jobs"]] == ["planner", "stage"]
        assert body["totals"]["tools"][0]["tool"] == "search_documents"

    async def test_a_stranger_is_told_nothing_exists(
        self, client, admin_user, db_session, test_user
    ):
        from app.services.auth_service import AuthService

        root = self._jobs(db_session, admin_user)
        db_session.add(root)
        await db_session.commit()
        headers = {
            "Authorization": f"Bearer {AuthService().create_access_token(test_user.id)}"
        }

        response = client.get(f"/api/v1/agent-jobs/{root.id}/profile", headers=headers)

        assert response.status_code == 404


def test_a_paused_job_ran_until_it_stopped_not_until_now():
    """Found live: a blocked stage read 9.6 hours running."""
    blocked = job(
        "stage",
        created=T0,
        started=T0 + timedelta(seconds=10),
        log=[iteration(1, T0 + timedelta(minutes=10), "get_research_findings")],
    )
    blocked.status = "paused"
    now = (T0 + timedelta(hours=10)).replace(tzinfo=timezone.utc)

    (profile,) = agent_run_profile.build([blocked], root_id=blocked.id, now=now)["jobs"]

    assert profile["ran_seconds"] == 590
    assert profile["paused_seconds"] == 10 * 3600 - 600


def test_a_run_ending_in_a_pause_ends_where_it_stopped():
    """Found live: the run's wall time kept growing behind a blocked stage."""
    blocked = job(
        "stage",
        created=T0,
        started=T0 + timedelta(seconds=10),
        log=[iteration(1, T0 + timedelta(minutes=10), "get_research_findings")],
    )
    blocked.status = "paused"
    now = (T0 + timedelta(hours=10)).replace(tzinfo=timezone.utc)

    profile = agent_run_profile.build([blocked], root_id=blocked.id, now=now)

    assert profile["wall_seconds"] == 600
