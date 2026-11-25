"""A synthesis that draws on an autonomous run.

The point of the link is that a measurement reaches the document as a number
rather than as a memory of one. So the assertions here are about the numbers
surviving the trip, and about a run belonging to someone else not being
readable through a synthesis.
"""

import uuid

import pytest

from app.models.agent_job import AgentJob
from app.models.agent_retraction import RetractionKind
from app.services import agent_retraction_service
from app.services.synthesis_service import synthesis_service

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("run_count", [25, 500])
def test_campaign_prompt_keeps_evidence_from_every_run_within_its_budget(run_count):
    runs = [
        {
            "metadata": {"run_name": "Recorded experiment " * 12},
            "recorded_findings": f"- measured cycles={index}\n"
            + "More measurements.\n" * 500,
        }
        for index in range(run_count)
    ]

    context = synthesis_service._prepare_campaign_run_context(runs)

    assert len(context) <= 60000
    assert context.count("[Source: Run ") == run_count
    for index in range(run_count):
        assert f"- measured cycles={index}\n" in context


async def _make_run(
    db, user_id, *, name="throughput study", findings=None, goal="Measure it"
):
    job = AgentJob(
        id=uuid.uuid4(),
        name=name,
        goal=goal,
        job_type="experiment",
        user_id=user_id,
        status="completed",
        iteration=7,
        results={"findings": findings if findings is not None else []},
    )
    db.add(job)
    await db.commit()
    return job


class TestAgentRunsAsSources:
    @pytest.mark.asyncio
    async def test_reproduction_conditions_and_blockers_survive_report_context(
        self, db_session, test_user
    ):
        job = await _make_run(
            db_session,
            test_user.id,
            findings=[
                {"type": "simulated_measurement", "title": f"Trial {i}", "cycles": i}
                for i in range(55)
            ]
            + [
                {
                    "type": "reproduction_verdict",
                    "subject": "Streaming kernel",
                    "verdict": "incomparable",
                    "claimed_value": 40,
                    "claimed_unit": "%",
                    "measured_value": 1.4,
                    "measured_unit": "×",
                    "claimed_conditions": {"input_size": "64"},
                    "measured_conditions": {"input_size": "128"},
                    "measurement_source": "gem5, five trials",
                    "claim_location": "Table 2",
                    "blockers": ["The input sizes differ."],
                }
            ],
        )

        sources = await synthesis_service._load_agent_runs(
            db_session, [str(job.id)], test_user.id
        )
        content = synthesis_service._prepare_document_context(sources)
        assert "(incomparable)" in content
        assert "Paper claim: 40 %" in content
        assert "Run measurement: 1.4 ×" in content
        assert "Paper conditions: input_size=64" in content
        assert "Run conditions: input_size=128" in content
        assert "Table 2" in content
        assert "gem5, five trials" in content
        assert "The input sizes differ." in content

    @pytest.mark.asyncio
    async def test_context_excludes_inherited_recalled_and_disputed_evidence(
        self, db_session, test_user
    ):
        job = await _make_run(
            db_session,
            test_user.id,
            findings=[
                {"type": "benchmark_measurement", "title": "Rejected timing"},
                {
                    "type": "benchmark_measurement",
                    "title": "Other stage timing",
                    "inherited": True,
                },
                {
                    "type": "benchmark_measurement",
                    "title": "Old timing",
                    "recalled": True,
                },
                {"type": "benchmark_measurement", "title": "Fresh timing"},
            ],
        )
        await agent_retraction_service.retract(
            db_session,
            user_id=test_user.id,
            kind=RetractionKind.FINDING,
            ref=agent_retraction_service.finding_ref(job.id, 0),
            reason="Bad timing setup",
        )
        await db_session.commit()

        (source,) = await synthesis_service._load_agent_runs(
            db_session, [str(job.id)], test_user.id
        )
        assert "Fresh timing" in source["content"]
        assert "Rejected timing" not in source["content"]
        assert "Other stage timing" not in source["content"]
        assert "Old timing" not in source["content"]

    @pytest.mark.asyncio
    async def test_a_runs_measurements_reach_the_document(self, db_session, test_user):
        job = await _make_run(
            db_session,
            test_user.id,
            findings=[
                {
                    "type": "benchmark_measurement",
                    "title": "dotprod_bench_O2 @ clang -O2: fastest 15 ms of 5 trials",
                    "subject": "dotprod_bench_O2",
                    "fastest_ms": 15,
                    "all_ms": [54, 18, 16, 15, 30],
                }
            ],
        )

        sources = await synthesis_service._load_agent_runs(
            db_session, [str(job.id)], test_user.id
        )

        assert len(sources) == 1
        content = sources[0]["content"]
        # The title alone was never the problem; the numbers recorded in
        # fields are what a document needs to cite.
        assert "fastest_ms=15" in content
        assert "all_ms=15..54 (n=5)" in content
        assert "Measure it" in content

    @pytest.mark.asyncio
    async def test_provenance_travels_with_the_source(self, db_session, test_user):
        job = await _make_run(
            db_session,
            test_user.id,
            findings=[{"type": "insight", "title": "Bottleneck is upstream of L2"}],
        )

        sources = await synthesis_service._load_agent_runs(
            db_session, [str(job.id)], test_user.id
        )

        meta = sources[0]["metadata"]
        assert meta["source_kind"] == "agent_run"
        assert meta["agent_job_id"] == str(job.id)
        assert meta["finding_count"] == 1
        assert sources[0]["id"] == str(job.id)

    @pytest.mark.asyncio
    async def test_a_run_with_no_findings_is_not_a_source(self, db_session, test_user):
        # An empty source in a prompt reads as evidence that there was nothing
        # to find, which is not the same as a run that recorded nothing.
        job = await _make_run(db_session, test_user.id, findings=[])

        sources = await synthesis_service._load_agent_runs(
            db_session, [str(job.id)], test_user.id
        )

        assert sources == []

    @pytest.mark.asyncio
    async def test_another_users_run_is_not_readable(
        self, db_session, test_user, admin_user
    ):
        job = await _make_run(
            db_session,
            admin_user.id,
            findings=[{"type": "insight", "title": "Someone else's finding"}],
        )

        sources = await synthesis_service._load_agent_runs(
            db_session, [str(job.id)], test_user.id
        )

        assert sources == []

    @pytest.mark.asyncio
    async def test_retrieval_hits_are_not_cited_as_the_runs_own(
        self, db_session, test_user
    ):
        # A real run recorded twelve findings: eleven documents it had read and
        # one measurement. Cite all twelve and the document is written as
        # though the run produced a literature review.
        job = await _make_run(
            db_session,
            test_user.id,
            findings=[
                {
                    "type": "document",
                    "title": "OptimizeIR Product Roadmap",
                    "score": 0.52,
                },
                {
                    "type": "paper",
                    "title": "R-KV: Redundancy-aware KV Cache",
                    "score": 0.44,
                },
                {
                    "type": "simulated_measurement",
                    "title": "stream_sum_u32 @ O3CPU",
                    "cycles": 56026,
                    "ipc": 0.1168,
                },
            ],
        )

        sources = await synthesis_service._load_agent_runs(
            db_session, [str(job.id)], test_user.id
        )

        content = sources[0]["content"]
        assert "cycles=56026" in content
        assert "OptimizeIR Product Roadmap" not in content
        assert "R-KV" not in content
        # Both counts are kept: what the run recorded, and what it established.
        assert sources[0]["metadata"]["finding_count"] == 3
        assert sources[0]["metadata"]["cited_finding_count"] == 1

    @pytest.mark.asyncio
    async def test_a_run_that_only_read_things_is_not_a_source(
        self, db_session, test_user
    ):
        job = await _make_run(
            db_session,
            test_user.id,
            findings=[{"type": "document", "title": "Something it read", "score": 0.9}],
        )

        sources = await synthesis_service._load_agent_runs(
            db_session, [str(job.id)], test_user.id
        )

        assert sources == []

    @pytest.mark.asyncio
    async def test_a_malformed_id_does_not_sink_the_job(self, db_session, test_user):
        good = await _make_run(
            db_session,
            test_user.id,
            findings=[{"type": "insight", "title": "A real finding"}],
        )

        sources = await synthesis_service._load_agent_runs(
            db_session, ["not-a-uuid", str(good.id), str(uuid.uuid4())], test_user.id
        )

        assert len(sources) == 1
        assert sources[0]["id"] == str(good.id)
