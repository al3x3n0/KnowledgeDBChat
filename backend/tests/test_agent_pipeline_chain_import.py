"""Importing a saved chain as a pipeline, through the API.

The converter's own tests cover the mapping. These cover what the endpoints owe
a person doing the migration: a survey that says which chains will convert and
how much work each leaves behind, an import that copies rather than moves, and a
refusal that names the step rather than only failing.
"""

import asyncio

from sqlalchemy import select

from app.models.agent_job import AgentJobChainDefinition
from app.models.agent_pipeline import AgentPipeline

SURVEY = "/api/v1/agent-pipelines/import/chains"


def _chain(db, *, name, steps, description=None):
    """Save a chain and return its id, driving the async session explicitly."""

    async def _seed():
        row = AgentJobChainDefinition(
            name=name,
            display_name=name.replace("_", " ").title(),
            description=description,
            chain_steps=steps,
        )
        db.add(row)
        await db.commit()
        await db.refresh(row)
        return row.id

    return asyncio.get_event_loop().run_until_complete(_seed())


def _fetch(db, statement):
    async def _run():
        return (await db.execute(statement)).scalars().all()

    return asyncio.get_event_loop().run_until_complete(_run())


def _step(name, trigger="on_complete", **extra):
    return {
        "step_name": name,
        "job_type": "research",
        "goal_template": f"Do {name}",
        "trigger_condition": trigger,
        **extra,
    }


class TestSurvey:
    def test_says_which_chains_convert_and_which_do_not(
        self, client, auth_headers, db_session
    ):
        _chain(db_session, name="clean", steps=[_step("A"), _step("B")])
        _chain(
            db_session,
            name="monitor",
            steps=[_step("Watch", "on_findings"), _step("Alert")],
        )

        response = client.get(SURVEY, headers=auth_headers)
        assert response.status_code == 200

        by_name = {c["name"]: c for c in response.json()["candidates"]}
        assert by_name["clean"]["convertible"] is True
        assert by_name["monitor"]["convertible"] is False

    def test_counts_the_contracts_a_conversion_would_leave_to_write(
        self, client, auth_headers, db_session
    ):
        # The number is the point: a converted chain is faithful, not finished,
        # and this says how much finishing each one needs.
        _chain(db_session, name="four", steps=[_step(c) for c in "ABCD"])

        response = client.get(SURVEY, headers=auth_headers)
        candidate = next(
            c for c in response.json()["candidates"] if c["name"] == "four"
        )
        assert candidate["steps"] == 4
        assert candidate["contracts_to_write"] == 4

    def test_a_blocked_chain_names_the_step_and_the_reason(
        self, client, auth_headers, db_session
    ):
        _chain(
            db_session,
            name="blocked",
            steps=[_step("Watch", "on_findings"), _step("Alert")],
        )

        response = client.get(SURVEY, headers=auth_headers)
        candidate = next(
            c for c in response.json()["candidates"] if c["name"] == "blocked"
        )
        (blocker,) = candidate["blockers"]
        assert blocker["step"] == "Watch"
        assert blocker["trigger"] == "on_findings"
        assert blocker["reason"]
        # A blocked chain has no contract count to report.
        assert candidate["contracts_to_write"] == 0


class TestImport:
    def test_saves_the_chain_as_a_pipeline(self, client, auth_headers, db_session):
        chain_id = _chain(
            db_session, name="lit_review", steps=[_step("Find"), _step("Read")]
        )

        response = client.post(
            SURVEY, headers=auth_headers, json={"chain_id": str(chain_id)}
        )
        assert response.status_code == 201

        body = response.json()
        assert body["name"] == "lit_review"
        assert [s["id"] for s in body["spec"]["stages"]] == ["find", "read"]
        assert body["spec"]["stages"][1]["depends_on"] == ["find"]

    def test_the_saved_pipeline_records_that_it_does_not_yet_check(
        self, client, auth_headers, db_session
    ):
        # Empty contracts mean it cannot run yet. Saving it anyway — and saying
        # so — is what lets someone repair it instead of starting over.
        chain_id = _chain(db_session, name="wip", steps=[_step("A")])

        response = client.post(
            SURVEY, headers=auth_headers, json={"chain_id": str(chain_id)}
        )
        assert response.json()["last_check_valid"] == "invalid"

    def test_the_chain_is_left_alone(self, client, auth_headers, db_session):
        # Copy, don't move: the pipeline cannot run until its contracts are
        # written, so removing the chain now would take away the working thing.
        chain_id = _chain(db_session, name="keep_me", steps=[_step("A")])

        client.post(SURVEY, headers=auth_headers, json={"chain_id": str(chain_id)})

        still_there = _fetch(
            db_session,
            select(AgentJobChainDefinition).where(
                AgentJobChainDefinition.id == chain_id
            ),
        )
        assert still_there != []

    def test_a_chain_that_cannot_convert_is_refused_with_its_blocker(
        self, client, auth_headers, db_session
    ):
        chain_id = _chain(
            db_session,
            name="monitor",
            steps=[_step("Watch", "on_findings"), _step("Alert")],
        )

        response = client.post(
            SURVEY, headers=auth_headers, json={"chain_id": str(chain_id)}
        )
        # 422, not 400: the request is well formed; the chain is what cannot be
        # expressed as a pipeline.
        assert response.status_code == 422

        detail = response.json()["detail"]
        assert detail["chain"] == "monitor"
        assert detail["blockers"][0]["trigger"] == "on_findings"

    def test_nothing_is_saved_when_the_chain_is_refused(
        self, client, auth_headers, db_session
    ):
        chain_id = _chain(
            db_session, name="monitor2", steps=[_step("Watch", "on_findings")]
        )

        client.post(SURVEY, headers=auth_headers, json={"chain_id": str(chain_id)})

        saved = _fetch(
            db_session, select(AgentPipeline).where(AgentPipeline.name == "monitor2")
        )
        assert saved == []

    def test_an_unknown_chain_is_a_404(self, client, auth_headers):
        response = client.post(
            SURVEY,
            headers=auth_headers,
            json={"chain_id": "00000000-0000-0000-0000-000000000000"},
        )
        assert response.status_code == 404

    def test_a_name_can_be_given_when_the_chain_name_is_taken(
        self, client, auth_headers, db_session
    ):
        chain_id = _chain(db_session, name="dup", steps=[_step("A")])

        first = client.post(
            SURVEY, headers=auth_headers, json={"chain_id": str(chain_id)}
        )
        assert first.status_code == 201

        clash = client.post(
            SURVEY, headers=auth_headers, json={"chain_id": str(chain_id)}
        )
        assert clash.status_code == 409

        renamed = client.post(
            SURVEY,
            headers=auth_headers,
            json={"chain_id": str(chain_id), "name": "dup (second try)"},
        )
        assert renamed.status_code == 201
        assert renamed.json()["name"] == "dup (second try)"
