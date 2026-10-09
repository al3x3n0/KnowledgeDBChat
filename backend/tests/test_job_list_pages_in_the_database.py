"""The job list reads the page it returns, not every job the user has.

It loaded every one of the caller's jobs, whole -- results and execution log
included -- and sliced twenty out in Python. The log alone is 163 KB after 80
iterations, so the cost of opening the page grew with everything the user had
ever run.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from uuid import uuid4

import pytest
from sqlalchemy import event

from app.api.endpoints import agent_jobs
from app.models.agent_job import AgentJob, AgentJobStatus
from tests.conftest import test_engine


def _job(user_id, n: int, **kwargs) -> AgentJob:
    return AgentJob(
        id=uuid4(),
        name=f"job {n:02d}",
        goal="List me",
        job_type="research",
        user_id=user_id,
        status=kwargs.pop("status", AgentJobStatus.COMPLETED.value),
        config=kwargs.pop("config", {}),
        results=kwargs.pop("results", {}),
        execution_log=[{"phase": "acting", "iteration": i} for i in range(5)],
        created_at=datetime(2026, 9, 1) + timedelta(minutes=n),
        **kwargs,
    )


async def _list(db, user, **overrides):
    params = dict(
        status=None,
        job_type=None,
        launch_mode=None,
        relaunch_from_job_id=None,
        has_relaunch_children=None,
        swarm_only=False,
        swarm_min_consensus=0,
        visibility_scope="mine",
        sort_by="created_desc",
        page=1,
        page_size=20,
        db=db,
        current_user=user,
    )
    params.update(overrides)
    return await agent_jobs.list_agent_jobs(**params)


class _Selects:
    """The SELECTs that read job rows while the block runs."""

    def __enter__(self):
        self.statements = []

        def record(conn, cursor, statement, parameters, context, executemany):
            text = " ".join(statement.split()).lower()
            if text.startswith("select") and " from agent_jobs" in text:
                self.statements.append(text)

        self._record = record
        event.listen(test_engine.sync_engine, "before_cursor_execute", record)
        return self

    def __exit__(self, *exc):
        event.remove(test_engine.sync_engine, "before_cursor_execute", self._record)


@pytest.fixture
async def twenty_five_jobs(db_session, test_user):
    db_session.add_all([_job(test_user.id, n) for n in range(25)])
    db_session.add_all([_job(uuid4(), 100 + n) for n in range(3)])
    await db_session.commit()
    return test_user


@pytest.mark.asyncio
async def test_a_page_is_the_newest_jobs_and_the_total_is_all_of_them(
    db_session, twenty_five_jobs
):
    listed = await _list(db_session, twenty_five_jobs, page_size=10)

    assert [job.name for job in listed.jobs] == [
        f"job {n:02d}" for n in range(24, 14, -1)
    ]
    assert listed.total == 25
    assert listed.has_more is True


@pytest.mark.asyncio
async def test_the_last_page_holds_what_is_left(db_session, twenty_five_jobs):
    listed = await _list(db_session, twenty_five_jobs, page=3, page_size=10)

    assert [job.name for job in listed.jobs] == [
        f"job {n:02d}" for n in range(4, -1, -1)
    ]
    assert listed.total == 25
    assert listed.has_more is False


@pytest.mark.asyncio
async def test_a_filter_narrows_the_total_as_well_as_the_page(
    db_session, twenty_five_jobs
):
    db_session.add(_job(twenty_five_jobs.id, 50, status=AgentJobStatus.FAILED.value))
    await db_session.commit()

    listed = await _list(
        db_session, twenty_five_jobs, status=AgentJobStatus.FAILED.value
    )

    assert [job.name for job in listed.jobs] == ["job 50"]
    assert listed.total == 1


@pytest.mark.asyncio
async def test_only_the_page_is_read_and_never_the_log(db_session, twenty_five_jobs):
    with _Selects() as seen:
        await _list(db_session, twenty_five_jobs, page_size=10)

    whole_rows = [text for text in seen.statements if "agent_jobs.goal" in text]
    assert len(whole_rows) == 1
    assert " limit " in whole_rows[0]
    assert all("execution_log" not in text for text in seen.statements)


@pytest.mark.asyncio
async def test_a_swarm_sort_still_ranks_every_job(db_session, test_user):
    # Ranked on a number inside each row's results, which the database is not
    # asked to read: the best one must surface from beyond the first page.
    def swarm(confidence):
        return {
            "execution_strategy": {"swarm": {"enabled": True, "configured": True}},
            "swarm_fan_in": {"confidence": {"overall": confidence}},
        }

    db_session.add_all([_job(test_user.id, n, results=swarm(0.1)) for n in range(1, 6)])
    db_session.add(_job(test_user.id, 0, results=swarm(0.9)))
    await db_session.commit()

    listed = await _list(
        db_session, test_user, sort_by="swarm_confidence_desc", page_size=2
    )

    assert listed.total == 6
    assert listed.jobs[0].name == "job 00"


async def _person(db, name: str):
    from app.services.auth_service import AuthService

    user = await AuthService().create_user(
        username=name,
        email=f"{name}@example.com",
        password="testpassword123",
        full_name=name,
        db=db,
    )
    return user.id


class TestWhoseNamesTheListMayShow:
    """`list_collaboration_user_ids` runs on every list request.

    It read every job the caller owned, whole, and learned from each that the
    caller owns it. Narrowing that must not lose the people it did find.
    """

    @pytest.mark.asyncio
    async def test_somebody_who_shared_a_swarm_with_you(self, db_session, test_user):
        from app.services.collaboration_service import list_collaboration_user_ids

        owner = await _person(db_session, "owner")
        assigner = await _person(db_session, "assigner")
        stranger = await _person(db_session, "stranger")
        shared = _job(
            owner,
            1,
            results={
                "swarm_collaboration": {
                    "shared_with_user_ids": [str(test_user.id)],
                    "assigned_by_user_id": str(assigner),
                }
            },
        )
        private = _job(
            stranger, 2, results={"swarm_collaboration": {"shared_review": False}}
        )
        db_session.add_all([shared, private, _job(test_user.id, 3)])
        await db_session.commit()

        with _Selects() as seen:
            visible = await list_collaboration_user_ids(
                db_session, current_user=test_user
            )

        assert {test_user.id, owner, assigner} <= visible
        assert stranger not in visible
        assert all("agent_jobs.goal" not in text for text in seen.statements)

    @pytest.mark.asyncio
    async def test_somebody_you_shared_your_own_swarm_with(self, db_session, test_user):
        from app.services.collaboration_service import list_collaboration_user_ids

        colleague = await _person(db_session, "colleague")
        db_session.add(
            _job(
                test_user.id,
                1,
                results={
                    "swarm_collaboration": {"shared_with_user_ids": [str(colleague)]}
                },
            )
        )
        await db_session.commit()

        visible = await list_collaboration_user_ids(db_session, current_user=test_user)

        assert colleague in visible


class TestSharingIsAColumn:
    """`collaborator_index` is kept by the model, whoever writes `results`."""

    @pytest.mark.asyncio
    async def test_it_follows_the_collaboration_block(self, db_session, test_user):
        job = _job(test_user.id, 1)
        db_session.add(job)
        await db_session.commit()
        assert job.collaborator_index is None

        colleague, reviewer = uuid4(), uuid4()
        job.results = {
            "swarm_collaboration": {
                "assigned_user_id": str(reviewer),
                "shared_with_user_ids": [str(colleague)],
            }
        }
        await db_session.commit()
        await db_session.refresh(job)
        assert job.collaborator_index == f"|{reviewer}|{colleague}|"

        # Edited in place, as the executor does, not reassigned.
        job.results["swarm_collaboration"]["shared_with_user_ids"] = []
        await db_session.commit()
        await db_session.refresh(job)
        assert job.collaborator_index == f"|{reviewer}|"

        job.results = {"findings": []}
        await db_session.commit()
        await db_session.refresh(job)
        assert job.collaborator_index is None

    @pytest.mark.asyncio
    async def test_an_id_is_matched_whole(self, db_session, test_user):
        from app.models.agent_job import collaborator_index_for, collaborator_pattern

        index = collaborator_index_for(
            {"swarm_collaboration": {"shared_with_user_ids": ["ABC-123"]}}
        )
        assert index == "|abc-123|"
        assert collaborator_pattern("ABC-123") == "%|abc-123|%"
        # "abc-12" is not somebody the job is shared with.
        assert "|abc-12|" not in index

    @pytest.mark.asyncio
    async def test_a_block_naming_nobody_is_still_a_block(self):
        from app.models.agent_job import collaborator_index_for

        assert collaborator_index_for({"swarm_collaboration": {}}) == "|"
        assert collaborator_index_for({"swarm_collaboration": "yes"}) is None
        assert collaborator_index_for(None) is None

    @pytest.mark.asyncio
    async def test_the_lookup_reads_no_ones_results_to_find_them(
        self, db_session, test_user
    ):
        from app.services.collaboration_service import list_collaboration_user_ids

        # Mentions the caller's id in its results without sharing anything.
        db_session.add(_job(uuid4(), 1, results={"notes": f"asked by {test_user.id}"}))
        await db_session.commit()

        with _Selects() as seen:
            await list_collaboration_user_ids(db_session, current_user=test_user)

        assert seen.statements
        assert all("cast(agent_jobs.results" not in text for text in seen.statements)


class TestOtherPeoplesJobs:
    def _shared_with(self, user_id, n, owner):
        return _job(
            owner,
            n,
            config={"launch_mode": "bug_triage_swarm_repair_handoff"},
            results={"swarm_collaboration": {"shared_with_user_ids": [str(user_id)]}},
        )

    @pytest.mark.asyncio
    async def test_the_shared_view_reads_only_jobs_that_name_you(
        self, db_session, test_user
    ):
        owner = await _person(db_session, "owner")
        db_session.add_all([_job(owner, n) for n in range(10)])
        db_session.add(self._shared_with(test_user.id, 50, owner))
        await db_session.commit()

        with _Selects() as seen:
            listed = await _list(db_session, test_user, visibility_scope="shared")

        assert [job.name for job in listed.jobs] == ["job 50"]
        assert listed.total == 1
        whole_rows = [text for text in seen.statements if "agent_jobs.goal" in text]
        assert len(whole_rows) == 1
        assert "collaborator_index like" in whole_rows[0]

    @pytest.mark.asyncio
    async def test_the_all_view_is_yours_and_what_names_you(
        self, db_session, test_user
    ):
        owner = await _person(db_session, "owner")
        db_session.add_all(
            [
                _job(test_user.id, 1),
                _job(owner, 2),
                self._shared_with(test_user.id, 3, owner),
            ]
        )
        await db_session.commit()

        listed = await _list(db_session, test_user, visibility_scope="all")

        assert [job.name for job in listed.jobs] == ["job 03", "job 01"]

    @pytest.mark.asyncio
    async def test_naming_you_is_not_enough_without_the_rule(
        self, db_session, test_user
    ):
        # The column narrows the candidates; `is_job_visible` still decides.
        # A job that is not a coding swarm is not shown, shared or not.
        owner = await _person(db_session, "owner")
        db_session.add(
            _job(
                owner,
                1,
                results={
                    "swarm_collaboration": {"shared_with_user_ids": [str(test_user.id)]}
                },
            )
        )
        await db_session.commit()

        listed = await _list(db_session, test_user, visibility_scope="shared")

        assert listed.jobs == []

    @pytest.mark.asyncio
    async def test_an_admin_sees_everyones_and_the_database_pages_them(
        self, db_session, admin_user
    ):
        owner = await _person(db_session, "owner")
        db_session.add_all([_job(owner, n) for n in range(25)])
        await db_session.commit()

        with _Selects() as seen:
            listed = await _list(
                db_session, admin_user, visibility_scope="all", page_size=10
            )

        assert [job.name for job in listed.jobs] == [
            f"job {n:02d}" for n in range(24, 14, -1)
        ]
        assert listed.total == 25
        whole_rows = [text for text in seen.statements if "agent_jobs.goal" in text]
        assert len(whole_rows) == 1 and " limit " in whole_rows[0]
