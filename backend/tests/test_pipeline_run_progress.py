"""What a pipeline run looks like while it is still happening.

Restarting from a stage you cannot see is guesswork, and the reason the run
view existed at all was to answer "where do I restart". But the same view is
the only place a person can ask the more ordinary question -- how far through
is this, and is it still moving -- and it could not answer either one:

* it reported only stages that already had a job. At stage two of six that is
  a two-stage run, which reads as finished.
* it reported `status`, and a run parked on a checkpoint waiting for a person
  has exactly the status of one whose worker died.

Both are asserted here rather than in the endpoint tests because the
derivations are pure: a run's progress is a function of its jobs and the chain
configs they carry, and there is no row anywhere that stores it.
"""

import uuid

import pytest

from app.models.agent_job import AgentJob, AgentJobStatus
from app.services import agent_pipeline_restart as restart

pytestmark = pytest.mark.unit


def _job(
    stage,
    status=AgentJobStatus.COMPLETED.value,
    *,
    children=(),
    checkpoint=False,
    goal=None,
):
    job = AgentJob(
        id=uuid.uuid4(),
        name=f"p: {stage}",
        goal=goal or f"do {stage}",
        job_type="research",
        status=status,
        iteration=2,
        max_iterations=10,
        config={"pipeline": "p", "pipeline_stage": stage},
        results={"goal_contract": {"enabled": True, "satisfied": True}},
        execution_log=[],
    )
    if children:
        job.chain_config = {
            "trigger_condition": "on_approval" if checkpoint else "on_complete",
            "child_jobs": [
                {"config": {"pipeline_stage": c}, "goal": f"do {c}"} for c in children
            ],
        }
    job.user_id = uuid.uuid4()
    return job


def _plan(root, *entries):
    by_stage = {e.stage_id: e for e in entries}
    return restart.stage_plan(root, by_stage)


class TestAStageThatHasNotStartedIsStillPartOfTheRun:
    def test_the_unreached_stages_are_planned_from_the_chain(self):
        """`load_run` can only see stages that have a job, which is the wrong
        set for a progress view: it makes stage two of six look like the end."""
        head = _job("profile", children=("measure",))
        ids = [p.stage_id for p in _plan(head, restart.StageJob("profile", head))]

        assert ids == ["profile", "measure"], "the unstarted stage is in the plan"

    def test_a_planned_stage_carries_the_goal_it_was_given(self):
        """A stage id is a label. Someone reading a run they did not author
        needs the sentence that says what the stage is for."""
        head = _job("profile", children=("measure",))
        plan = _plan(head, restart.StageJob("profile", head))

        assert plan[1].goal == "do measure"

    def test_the_order_is_the_order_the_run_will_do_them_in(self):
        head = _job("profile", children=("measure",))
        measure = _job("measure", children=("attribute",))
        ids = [
            p.stage_id
            for p in _plan(
                head,
                restart.StageJob("profile", head),
                restart.StageJob("measure", measure),
            )
        ]

        assert ids == ["profile", "measure", "attribute"]

    def test_a_cycle_in_stored_json_does_not_hang_the_request(self):
        """chain_config is stored JSON. It should not contain a cycle, and a
        view that trusts it not to is one bad row away from a hung request."""
        head = _job("profile", children=("measure",))
        measure = _job("measure", children=("profile",))
        ids = [
            p.stage_id
            for p in _plan(
                head,
                restart.StageJob("profile", head),
                restart.StageJob("measure", measure),
            )
        ]

        assert ids == ["profile", "measure"]


class TestTheRunThatIsHappeningNotTheOnePlanned:
    def test_an_inserted_stage_appears_where_it_was_inserted(self):
        """A stage inserted into a running pipeline is recorded on its new
        parent and appears nowhere in the plan the root was bound with. Reading
        the root's copy would show a run that is not the one running."""
        head = _job("profile", children=("measure",))
        # What insert_stage_after does: the parent points at the new stage
        # alone, and the displaced child re-derives beneath it.
        head.chain_config = {
            "trigger_condition": "on_complete",
            "child_jobs": [
                {
                    "config": {"pipeline_stage": "convert"},
                    "goal": "do convert",
                    "chain_config": {
                        "trigger_condition": "on_complete",
                        "child_jobs": [
                            {
                                "config": {"pipeline_stage": "measure"},
                                "goal": "do measure",
                            }
                        ],
                    },
                }
            ],
        }
        ids = [p.stage_id for p in _plan(head, restart.StageJob("profile", head))]

        assert ids == ["profile", "convert", "measure"]

    def test_a_stage_with_a_job_nobody_planned_is_not_dropped(self):
        """It should not happen. If it does, dropping it hides the one stage
        someone opened the page to find."""
        head = _job("profile")
        orphan = _job("mystery")
        ids = [
            p.stage_id
            for p in _plan(
                head,
                restart.StageJob("profile", head),
                restart.StageJob("mystery", orphan),
            )
        ]

        assert "mystery" in ids

    def test_a_checkpoint_stage_says_so(self):
        """ "Completed and nothing happening" is the expected state of a
        checkpoint, not a stall, and only the chain knows which it is."""
        head = _job("profile", children=("measure",), checkpoint=True)
        plan = _plan(head, restart.StageJob("profile", head))

        assert plan[0].checkpoint is True
        assert plan[1].checkpoint is False


class TestOneWordForWhatTheRunIsDoing:
    def _stage(self, status, **kw):
        base = {
            "status": status,
            "contract_satisfied": True,
            "waiting_on_person": False,
        }
        base.update(kw)
        return base

    def test_a_run_waiting_on_a_person_is_not_a_run_that_died(self):
        """They are the same `status`. Told apart, one needs an approval and
        the other needs someone to find out what killed the worker."""
        assert (
            restart.run_status(
                [
                    self._stage(AgentJobStatus.COMPLETED.value, waiting_on_person=True),
                    self._stage(AgentJobStatus.PENDING.value),
                ]
            )
            == "waiting"
        )

    def test_running_beats_a_tail_of_pending_stages(self):
        assert (
            restart.run_status(
                [
                    self._stage(AgentJobStatus.COMPLETED.value),
                    self._stage(AgentJobStatus.RUNNING.value),
                    self._stage(AgentJobStatus.PENDING.value),
                ]
            )
            == "running"
        )

    def test_every_stage_completed_and_every_contract_met(self):
        assert (
            restart.run_status(
                [
                    self._stage(AgentJobStatus.COMPLETED.value),
                    self._stage(AgentJobStatus.COMPLETED.value),
                ]
            )
            == "completed"
        )

    def test_completed_without_the_evidence_is_not_completed(self):
        """A stage can finish without meeting its contract, and a run that
        reports success for evidence it never produced is the failure this
        whole subsystem exists to prevent."""
        assert (
            restart.run_status(
                [
                    self._stage(AgentJobStatus.COMPLETED.value),
                    self._stage(
                        AgentJobStatus.COMPLETED.value, contract_satisfied=False
                    ),
                ]
            )
            == "completed_unmet"
        )

    def test_a_failure_anywhere_is_a_failed_run(self):
        assert (
            restart.run_status(
                [
                    self._stage(AgentJobStatus.COMPLETED.value),
                    self._stage(AgentJobStatus.FAILED.value),
                    self._stage(AgentJobStatus.PENDING.value),
                ]
            )
            == "failed"
        )

    def test_nothing_started_yet_is_queued_not_running(self):
        assert (
            restart.run_status(
                [
                    self._stage(AgentJobStatus.PENDING.value),
                    self._stage(AgentJobStatus.PENDING.value),
                ]
            )
            == "pending"
        )


class TestARunRestingOnRejectedEvidence:
    """Advisory does not mean invisible.

    Every contract met, and a person rejected one of the results they were met
    with. Nothing was invalidated -- that was the decision -- but a run that
    reads as `completed` in that state is exactly the false clean bill this
    subsystem exists to prevent.
    """

    def _stage(self, **kw):
        base = {
            "status": AgentJobStatus.COMPLETED.value,
            "contract_satisfied": True,
            "waiting_on_person": False,
            "disputed": False,
        }
        base.update(kw)
        return base

    def test_a_finished_run_with_a_rejection_is_not_completed(self):
        assert (
            restart.run_status([self._stage(), self._stage(disputed=True)])
            == "completed_disputed"
        )

    def test_a_finished_run_with_none_is(self):
        assert restart.run_status([self._stage(), self._stage()]) == "completed"

    def test_an_unmet_contract_still_outranks_a_rejection(self):
        """Both are true; the contract is the more fundamental failure, and
        reporting the softer one would understate it."""
        assert (
            restart.run_status([self._stage(contract_satisfied=False, disputed=True)])
            == "completed_unmet"
        )
