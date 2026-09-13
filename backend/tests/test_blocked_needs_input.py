"""A run that concludes it cannot proceed stops, and can be corrected.

Burning the remaining iterations proving the same wall is not persistence, and
reporting COMPLETED for a contract nobody met is a lie the pipeline believes.
So it pauses, says what is missing, and waits for a person -- who may know the
one thing it lacked.
"""

import uuid
from datetime import datetime, timedelta

import pytest

from app.models.agent_job import AgentJob, AgentJobCheckpoint, AgentJobStatus
from app.services import agent_runtime_finalizer
from app.tasks.agent_job_tasks import WAITING_ON_A_PERSON


def _job(**kw):
    job = AgentJob(
        id=uuid.uuid4(),
        name="reproduce a paper's throughput claim",
        goal="reproduce it",
        job_type="research",
        status=AgentJobStatus.RUNNING.value,
        iteration=3,
        max_iterations=8,
        error_count=0,
        tool_calls_used=0,
        max_tool_calls=500,
        llm_calls_used=0,
        max_llm_calls=500,
        results={},
        execution_log=[],
        config={},
    )
    for k, v in kw.items():
        setattr(job, k, v)
    return job


def _executor(contract_eval):
    """The real executor, with only the contract verdict scripted.

    The finaliser reads a long tail of config and stat accessors off it to
    compile the results payload; stubbing them by hand produced a fake that
    drifted from the real one within two runs.
    """
    from app.services.autonomous_agent_executor import AutonomousAgentExecutor

    executor = AutonomousAgentExecutor()
    executor._evaluate_goal_contract = lambda job, state: dict(contract_eval)
    return executor


CONTRACT_UNMET = {
    "enabled": True,
    "satisfied": False,
    "missing": ["needs 2 findings of type throughput_bound, has 0"],
}


@pytest.mark.asyncio
class TestAStuckRunStopsInsteadOfFinishing:
    async def test_a_loop_that_gave_up_with_an_unmet_contract_is_blocked_not_done(
        self, db_session
    ):
        job = _job()
        state = {
            "goal_progress": 40,
            "loop_policy_stop_reason": "no_new_findings for 2 rounds",
        }

        await agent_runtime_finalizer.finalize_job(
            _executor(CONTRACT_UNMET), job, state, db_session
        )

        assert job.status == AgentJobStatus.PAUSED.value
        assert job.current_phase == "blocked_needs_input"
        blocked = job.results["blocked"]
        assert blocked["resumable"] is True
        assert "no_new_findings" in blocked["reason"]
        # What a person needs to know to help, not just that it stopped.
        assert blocked["missing"], "a blocked run must say what it lacked"

    async def test_a_run_that_insisted_on_stopping_is_blocked_too(self, db_session):
        """The path the first live run actually took.

        Told twice its contract was unmet, it stopped anyway -- and was filed
        `completed` with nothing measured, because only the loop-policy path
        was being checked.
        """
        job = _job()
        state = {
            "goal_progress": 25,
            "stopped_short_of_contract": True,
            "stopped_short_reason": "the repository path does not exist",
        }

        await agent_runtime_finalizer.finalize_job(
            _executor(CONTRACT_UNMET), job, state, db_session
        )

        assert job.status == AgentJobStatus.PAUSED.value
        assert job.current_phase == "blocked_needs_input"
        assert "does not exist" in job.results["blocked"]["reason"]

    async def test_a_stall_stop_is_blocked_even_though_it_touches_no_state(
        self, db_session
    ):
        """The stall detector logs a voluntary stop and writes no state flag."""
        job = _job()
        job.execution_log = [
            {"phase": "iteration_complete"},
            {"phase": "voluntary_stop", "reason": "no progress for 3 iterations"},
        ]

        await agent_runtime_finalizer.finalize_job(
            _executor(CONTRACT_UNMET), job, {"goal_progress": 25}, db_session
        )

        assert job.status == AgentJobStatus.PAUSED.value
        assert "no progress" in job.results["blocked"]["reason"]

    async def test_a_blocked_run_keeps_the_findings_it_did_produce(self, db_session):
        """Most of what the person answering it needs is what it already found."""
        job = _job()
        state = {
            "goal_progress": 25,
            "stopped_short_of_contract": True,
            "findings": [
                {"type": "build_result", "content": "cargo build succeeded"},
            ],
        }

        await agent_runtime_finalizer.finalize_job(
            _executor(CONTRACT_UNMET), job, state, db_session
        )

        assert job.status == AgentJobStatus.PAUSED.value
        assert job.results["blocked"]["resumable"] is True
        assert job.results.get("findings"), "a blocked run must not lose its work"

    async def test_a_run_that_simply_ran_out_of_iterations_still_completes(
        self, db_session
    ):
        """Only a run that gave up early is stuck. Exhausting the budget is
        an ordinary ending, and pausing every one of those would put a human
        in the loop of every short run."""
        job = _job()

        await agent_runtime_finalizer.finalize_job(
            _executor(CONTRACT_UNMET), job, state={"goal_progress": 40}, db=db_session
        )

        assert job.status == AgentJobStatus.COMPLETED.value
        assert "blocked" not in (job.results or {})


@pytest.mark.asyncio
class TestTheSweepLeavesHumanGatesAlone:
    """The sweep that resumes stalled jobs filtered on status alone.

    A stage paused for approval was therefore resumed five minutes later with
    nobody approving anything -- the gate opened itself -- and a run that had
    correctly concluded it was blocked went straight back into the wall.
    """

    async def _sweep(self, db_session, jobs):
        """Drive the sweep's body against a real session."""
        from sqlalchemy import and_, or_, select

        from app.tasks import agent_job_tasks

        for job in jobs:
            db_session.add(job)
        await db_session.commit()

        cutoff = datetime.utcnow() - timedelta(minutes=5)
        rows = (
            (
                await db_session.execute(
                    select(AgentJob).where(
                        and_(
                            AgentJob.status == AgentJobStatus.PAUSED.value,
                            or_(
                                AgentJob.last_activity_at < cutoff,
                                AgentJob.last_activity_at.is_(None),
                            ),
                        )
                    )
                )
            )
            .scalars()
            .all()
        )

        resumed = []
        for job in rows:
            if str(job.current_phase or "") in agent_job_tasks.WAITING_ON_A_PERSON:
                continue
            limited, _ = job.is_resource_limited()
            if limited:
                continue
            resumed.append(job.name)
        return resumed

    async def test_it_skips_a_job_waiting_for_approval(self, db_session, test_user):
        approval = _job(
            name="waiting-for-approval",
            status=AgentJobStatus.PAUSED.value,
            current_phase="awaiting_approval",
            user_id=test_user.id,
        )
        blocked = _job(
            name="blocked-on-a-bad-path",
            status=AgentJobStatus.PAUSED.value,
            current_phase="blocked_needs_input",
            user_id=test_user.id,
        )
        stalled = _job(
            name="genuinely-stalled",
            status=AgentJobStatus.PAUSED.value,
            current_phase="thinking",
            user_id=test_user.id,
        )

        resumed = await self._sweep(db_session, [approval, blocked, stalled])

        # The one nobody is waiting on is still recovered.
        assert resumed == ["genuinely-stalled"]

    def test_the_phases_that_mean_a_person_is_the_next_step(self):
        assert "awaiting_approval" in WAITING_ON_A_PERSON
        assert "blocked_needs_input" in WAITING_ON_A_PERSON


@pytest.mark.asyncio
class TestACorrectionReachesTheAgent:
    async def test_the_resume_note_is_injected_into_the_state_it_reads(
        self, db_session, test_user
    ):
        from app.modules.autonomy.application import job_action_checkpoint_resume

        job = _job(
            status=AgentJobStatus.PAUSED.value,
            current_phase="blocked_needs_input",
            user_id=test_user.id,
        )

        checkpoint = AgentJobCheckpoint(
            id=uuid.uuid4(),
            job_id=job.id,
            iteration=3,
            state={"iteration": 3, "findings": []},
        )
        started = []

        class _Deps:
            @staticmethod
            def approval_payload_from_results(results):
                return ({}, {}, None)

            @staticmethod
            async def load_latest_checkpoint(job_id, db):
                return checkpoint

            @staticmethod
            def append_operator_intervention(payload, **kw):
                payload.setdefault("interventions", []).append(kw)

            class execute_agent_job_task:
                @staticmethod
                def delay(job_id, user_id):
                    started.append(job_id)

        resumed = await job_action_checkpoint_resume.perform_resume_action(
            job,
            "the repo is at /srv/paper-code, not /tmp -- retry the clone there",
            deps=_Deps(),
            db=db_session,
            current_user=test_user,
        )

        clues = checkpoint.state["operator_clues"]
        assert len(clues) == 1
        assert "/srv/paper-code" in clues[0]["note"]
        # And on the job, which is what a run with no checkpoint reads.
        assert "/srv/paper-code" in job.config["operator_clues"][0]["note"]
        assert resumed.status == AgentJobStatus.PENDING.value
        # And it stops looking like it is waiting for someone.
        assert resumed.current_phase not in WAITING_ON_A_PERSON
        assert started, "resuming must actually restart the run"


@pytest.mark.asyncio
class TestACorrectionSurvivesWithoutACheckpoint:
    async def test_the_note_is_kept_on_the_job_when_no_checkpoint_exists(
        self, db_session, test_user
    ):
        """The first live run to block had zero checkpoint rows.

        A clue written only to a checkpoint that does not exist is dropped in
        silence -- indistinguishable, from the outside, from an agent that read
        the correction and ignored it.
        """
        from app.modules.autonomy.application import job_action_checkpoint_resume

        job = _job(
            status=AgentJobStatus.PAUSED.value,
            current_phase="blocked_needs_input",
            user_id=test_user.id,
        )

        class _Deps:
            @staticmethod
            def approval_payload_from_results(results):
                return ({}, {}, None)

            @staticmethod
            async def load_latest_checkpoint(job_id, db):
                return None

            @staticmethod
            def append_operator_intervention(payload, **kw):
                payload.setdefault("interventions", []).append(kw)

            class execute_agent_job_task:
                @staticmethod
                def delay(job_id, user_id):
                    pass

        await job_action_checkpoint_resume.perform_resume_action(
            job,
            "the repo is at /srv/paper-code",
            deps=_Deps(),
            db=db_session,
            current_user=test_user,
        )

        assert "/srv/paper-code" in job.config["operator_clues"][0]["note"]


class TestTheClueIsReadable:
    def test_operator_guidance_leads_the_thinking_prompt(self):
        from app.services.autonomous_agent_executor import AutonomousAgentExecutor

        executor = AutonomousAgentExecutor()
        prompt = executor._build_thinking_prompt_volatile(
            _job(),
            {
                "operator_clues": [
                    {"note": "the repo is at /srv/paper-code", "iteration": 3}
                ],
                "compressed_history": "tried to clone /tmp/paper, failed 3 times",
            },
        )

        assert "/srv/paper-code" in prompt
        # Ahead of the history it is there to correct.
        assert prompt.index("/srv/paper-code") < prompt.index("tried to clone")


class TestAGapIsNotAMeasurement:
    """A finding that reports the absence of the thing is not the thing.

    A live run asked for a throughput_bound from an actual benchmark. It
    correctly established that the repository did not exist and filed that
    conclusion honestly -- as a finding of type `throughput_bound`, category
    `gap`, metric null. Counting by type alone read it as the measurement and
    autocompleted the job at 100%, which is the exact outcome the blocked-run
    pause exists to prevent.
    """

    @staticmethod
    def _evaluate(findings):
        from app.services.agent_goal_contract_service import AgentGoalContractService
        from app.services.autonomous_agent_executor import AutonomousAgentExecutor

        job = _job(
            config={
                "goal_contract": {
                    "enabled": True,
                    "required_finding_types": {"throughput_bound": 1},
                }
            }
        )
        return AgentGoalContractService().evaluate_goal_contract(
            AutonomousAgentExecutor(),
            job,
            {"findings": findings, "goal_progress": 100, "artifacts": []},
        )

    def test_a_gap_finding_does_not_satisfy_the_type_it_is_filed_under(self):
        result = self._evaluate(
            [
                {
                    "type": "throughput_bound",
                    "category": "gap",
                    "title": "Missing repository blocks throughput_bound actual run",
                    "metrics": {"throughput_bound": None},
                }
            ]
        )

        assert result["satisfied"] is False
        assert "finding_type:throughput_bound" in result["missing"]

    def test_a_null_metric_under_its_own_type_does_not_count(self):
        result = self._evaluate(
            [{"type": "throughput_bound", "metrics": {"throughput_bound": None}}]
        )

        assert result["satisfied"] is False

    def test_a_real_measurement_still_counts(self):
        """The narrow rule must not swallow ordinary evidence."""
        result = self._evaluate(
            [
                {
                    "type": "throughput_bound",
                    "category": "measurement",
                    "metrics": {"throughput_bound": 1.42e8},
                }
            ]
        )

        assert result["satisfied"] is True

    def test_a_disappointing_measurement_is_still_a_measurement(self):
        """Prose is not the signal -- only the finding's own declaration is."""
        result = self._evaluate(
            [
                {
                    "type": "throughput_bound",
                    "title": "Throughput far below the paper's claim; a real gap",
                    "content": "We could not reach the claimed number.",
                    "metrics": {"throughput_bound": 3.0e6},
                }
            ]
        )

        assert result["satisfied"] is True


@pytest.mark.asyncio
class TestABlockedRunStartsNothingDownstream:
    async def test_a_blocked_stage_does_not_trigger_its_chain(self, db_session):
        """The whole point of stopping is that downstream work would be built
        on evidence the run never produced."""
        job = _job(
            chain_config={"trigger_condition": "on_complete"},
            chain_triggered=False,
        )
        state = {"goal_progress": 25, "stopped_short_of_contract": True}
        triggered = []

        executor = _executor(CONTRACT_UNMET)

        async def _record(j, event, db):
            triggered.append(event)

        executor._trigger_chained_jobs = _record

        await agent_runtime_finalizer.finalize_job(executor, job, state, db_session)

        assert job.status == AgentJobStatus.PAUSED.value
        assert triggered == [], "a blocked run must not start downstream stages"


@pytest.mark.asyncio
class TestACorrectionUndoesTheGivingUp:
    async def test_resuming_with_a_note_clears_the_state_that_stopped_it(
        self, db_session, test_user
    ):
        """Otherwise resume is a no-op that reads as a defiant agent.

        Measured on a live run: after `resumed` came skill_profile_resolved,
        execution_mode_resolved, memory_injection, tool_priors_loaded, then
        `loop_policy_stop` -- it did its setup and stopped before the first
        iteration, because the restored state still held the dry-round history
        that made it give up in the first place.
        """
        from app.modules.autonomy.application import job_action_checkpoint_resume
        from app.services import agent_loop_policy

        job = _job(
            status=AgentJobStatus.PAUSED.value,
            current_phase="blocked_needs_input",
            user_id=test_user.id,
            config={"loop_until": "no_new_findings", "loop_dry_rounds": 2},
        )
        checkpoint = AgentJobCheckpoint(
            id=uuid.uuid4(),
            job_id=job.id,
            iteration=5,
            state={
                "iteration": 5,
                # Three rounds that established nothing: exactly what
                # should_stop fires on.
                "loop_finding_counts": [20, 20, 20],
                "loop_policy_stop_reason": "2 consecutive rounds produced no new findings",
                "stopped_short_of_contract": True,
                "stalled_iterations": 3,
            },
        )

        class _Deps:
            @staticmethod
            def approval_payload_from_results(results):
                return ({}, {}, None)

            @staticmethod
            async def load_latest_checkpoint(job_id, db):
                return checkpoint

            @staticmethod
            def append_operator_intervention(payload, **kw):
                payload.setdefault("interventions", []).append(kw)

            class execute_agent_job_task:
                @staticmethod
                def delay(job_id, user_id):
                    pass

        # Before: the policy would stop this run on sight.
        stop_before, _ = agent_loop_policy.should_stop(job.config, checkpoint.state)
        assert stop_before is True

        await job_action_checkpoint_resume.perform_resume_action(
            job,
            "the repo is at /tmp/paper-code -- clone that and run the benchmark",
            deps=_Deps(),
            db=db_session,
            current_user=test_user,
        )

        stop_after, _ = agent_loop_policy.should_stop(job.config, checkpoint.state)
        assert stop_after is False, "a corrected run must get a real chance to work"
        for key in agent_loop_policy.GIVE_UP_STATE_KEYS:
            assert key not in checkpoint.state
        # The clue itself survives the clearing.
        assert checkpoint.state["operator_clues"][0]["note"]

    async def test_resuming_without_a_note_leaves_the_state_alone(
        self, db_session, test_user
    ):
        """No correction, no reason to think the run's conclusion was wrong."""
        from app.modules.autonomy.application import job_action_checkpoint_resume

        job = _job(
            status=AgentJobStatus.PAUSED.value,
            user_id=test_user.id,
            config={"loop_until": "no_new_findings"},
        )
        checkpoint = AgentJobCheckpoint(
            id=uuid.uuid4(),
            job_id=job.id,
            iteration=5,
            state={"loop_finding_counts": [20, 20, 20]},
        )

        class _Deps:
            @staticmethod
            def approval_payload_from_results(results):
                return ({}, {}, None)

            @staticmethod
            async def load_latest_checkpoint(job_id, db):
                return checkpoint

            @staticmethod
            def append_operator_intervention(payload, **kw):
                payload.setdefault("interventions", []).append(kw)

            class execute_agent_job_task:
                @staticmethod
                def delay(job_id, user_id):
                    pass

        await job_action_checkpoint_resume.perform_resume_action(
            job, None, deps=_Deps(), db=db_session, current_user=test_user
        )

        assert checkpoint.state["loop_finding_counts"] == [20, 20, 20]


class TestACorrectionOverridesTheGoal:
    """Where the correction sits decides whether it is acted on.

    Measured: a run blocked on a bad repository path was resumed with the right
    one in its volatile context. It carried the clue for three iterations and
    never used it -- the goal statement in the system prompt still named the
    path it had already proved did not exist, so the correction read as one
    more detail under the history that proved it.
    """

    @staticmethod
    def _stable(state):
        from app.services.autonomous_agent_executor import AutonomousAgentExecutor

        job = _job(goal="Clone /tmp/does-not-exist-at-all and benchmark it.")
        return AutonomousAgentExecutor()._build_thinking_prompt_stable(job, None, state)

    def test_the_correction_sits_with_the_goal_it_amends(self):
        prompt = self._stable(
            {"operator_clues": [{"note": "the repo is at /tmp/paper-code"}]}
        )

        assert "/tmp/paper-code" in prompt
        # Next to the goal, not appended at the end of a long prompt.
        assert prompt.index("/tmp/paper-code") - prompt.index("GOAL:") < 1200

    def test_it_says_the_correction_wins(self):
        prompt = self._stable(
            {"operator_clues": [{"note": "the repo is at /tmp/paper-code"}]}
        )

        assert "OVERRIDE" in prompt

    def test_a_run_with_no_corrections_is_unchanged(self):
        """The stable prompt keys the provider cache; an empty clue list must
        not perturb it."""
        assert self._stable({}) == self._stable({"operator_clues": []})
