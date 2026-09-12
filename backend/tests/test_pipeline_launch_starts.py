"""Launching a pipeline has to start it.

`/agent-pipelines/launch` created the head job and returned 201 with a stage
list and a cost estimate -- and queued nothing. Nothing sweeps PENDING either:
`process_scheduled_agent_jobs` only picks up jobs carrying a `schedule_type`
and a `next_run_at`, and a pipeline head has neither. Found in the live
database as a pipeline that had been pending for three days under a launch
that had reported success.
"""

import pytest

pytestmark = pytest.mark.unit

SPEC = {
    "name": "two-stage",
    "stages": [
        {
            "id": "find",
            "goal": "Ingest the paper describing the algorithm",
            "contract": {"required_finding_types": ["papers_ingested"]},
        },
        {
            "id": "specify",
            "goal": "Read the paper into an implementable specification",
            "depends_on": ["find"],
            "assumes": ["papers_ingested"],
            "contract": {"required_finding_types": ["algorithm_spec"]},
        },
    ],
}


class TestNothingElseWouldStartIt:
    def test_the_scheduler_sweep_cannot_pick_up_a_pipeline_head(self):
        """The reason the missing dispatch was silent rather than delayed.

        If some sweep eventually ran pending jobs, the bug would have been a
        late start. It is not: the only sweep filters on schedule_type, which
        a pipeline head does not set.
        """
        import inspect

        from app.tasks import agent_job_tasks

        source = inspect.getsource(agent_job_tasks.process_scheduled_agent_jobs)
        assert "schedule_type" in source
        assert "next_run_at" in source


class TestLaunchQueuesTheHeadJob:
    def test_it_dispatches_the_job_it_created(self, client, auth_headers, monkeypatch):
        dispatched = []

        class _Task:
            @staticmethod
            def delay(job_id, user_id):
                dispatched.append((job_id, user_id))

        monkeypatch.setattr(
            "app.api.endpoints.agent_pipelines.execute_agent_job_task", _Task
        )

        response = client.post(
            "/api/v1/agent-pipelines/launch",
            headers=auth_headers,
            json={"spec": SPEC},
        )

        assert response.status_code == 201, response.text
        job_id = response.json()["job_id"]
        assert dispatched, "a launched pipeline that is never queued never runs"
        assert dispatched[0][0] == job_id, "queued a different job than it reported"

    def test_a_refused_pipeline_queues_nothing(self, client, auth_headers, monkeypatch):
        """The control. Dispatching before validation would start work the
        endpoint is about to refuse."""
        dispatched = []

        class _Task:
            @staticmethod
            def delay(job_id, user_id):
                dispatched.append(job_id)

        monkeypatch.setattr(
            "app.api.endpoints.agent_pipelines.execute_agent_job_task", _Task
        )

        response = client.post(
            "/api/v1/agent-pipelines/launch",
            headers=auth_headers,
            json={"spec": {"name": "empty", "stages": []}},
        )

        assert response.status_code == 422
        assert dispatched == []


class TestAStageThatDeclaresNoLoopStillNoticesItIsStuck:
    """Measured on the live reproduction run.

    A `specify` stage was told by a lying ingestion tool that a paper was in
    the corpus. It searched for the paper, searched again, re-ran the
    ingestion, searched by title, searched by arXiv id -- and was still going
    at iteration 9 of 100, correctly refusing to invent a specification, with
    nothing in place to notice that no round had produced anything new.
    """

    @staticmethod
    def _bind(stage_extra=None):
        from app.services import agent_pipeline_binding as binding
        from app.services import agent_pipeline_spec as ps

        stage = {
            "id": "specify",
            "goal": "Read the paper into an implementable specification",
            "contract": {"required_finding_types": ["algorithm_spec"]},
        }
        stage.update(stage_extra or {})
        spec = ps.normalize({"name": "p", "stages": [stage]})
        return binding.bind(spec).roots[0]

    def test_it_gets_a_stop_condition_by_default(self):
        from app.services import agent_pipeline_binding as binding

        config = self._bind()["config"]

        assert config["loop_until"] == "no_new_findings"
        assert config["loop_dry_rounds"] == binding.DEFAULT_STAGE_DRY_ROUNDS

    def test_the_default_is_one_the_loop_policy_actually_honours(self):
        """`loop_until` is read by name; an unrecognised value is ignored, so
        a default nobody honours would look identical to no default."""
        from app.services import agent_loop_policy

        config = self._bind()["config"]

        assert config["loop_until"] in agent_loop_policy.KNOWN_POLICIES
        assert agent_loop_policy.policy_warning(config) == ""

        dry = config["loop_dry_rounds"]
        stuck = {"loop_finding_counts": [4] * (dry + 1)}
        stop, reason = agent_loop_policy.should_stop(config, stuck)
        assert stop is True and "no new findings" in reason

    def test_an_author_who_asks_for_patience_still_gets_it(self):
        """The control. The default must not override a declared loop."""
        job = self._bind({"loop": {"max_iterations": 6, "until": "contract_satisfied"}})

        assert job["config"]["loop_until"] == "contract_satisfied"
        assert job["max_iterations"] == 6
        assert "loop_dry_rounds" not in job["config"]

    def test_a_stage_still_making_progress_is_not_stopped(self):
        from app.services import agent_loop_policy

        config = self._bind()["config"]
        progressing = {"loop_finding_counts": [1, 2, 3, 4, 5]}

        stop, _ = agent_loop_policy.should_stop(config, progressing)
        assert stop is False
