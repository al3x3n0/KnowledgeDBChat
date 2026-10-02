"""`reflect`, `hypothesize`, `weigh_evidence` and `critique_plan`, run for real.

Each test calls the handler `build_autonomous_reasoning_provider` registers
and judges it on the run state it leaves behind and, where something reads
that state, on what the reader then shows the model. The executor is the real
`AutonomousAgentExecutor`, so the prompts checked here are the prompts a run
is given.
"""

from datetime import datetime
from uuid import uuid4

import pytest

from app.agent_core.tool_specs import agent_ops
from app.models.agent_job import AgentJob
from app.services import agent_prompt_sections
from app.services.agent_tool_dispatch import (
    AgentToolExecutionContext,
    build_autonomous_reasoning_provider,
)
from app.services.autonomous_agent_executor import AutonomousAgentExecutor

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def executor():
    return AutonomousAgentExecutor()


def _job(iteration=3):
    return AgentJob(
        id=uuid4(),
        user_id=uuid4(),
        name="Prefetcher study",
        goal="Find out whether stride prefetching helps pointer chasing",
        job_type="research",
        status="running",
        config={},
        iteration=iteration,
        max_iterations=20,
    )


async def _call(executor, tool, params, state=None, job=None):
    """Run one tool; returns (result, state)."""
    state = {} if state is None else state
    job = job or _job()
    provider = build_autonomous_reasoning_provider(executor)
    ctx = AgentToolExecutionContext(
        mode="autonomous",
        db=None,
        service=None,
        user_id=str(job.user_id),
        job=job,
        state=state,
    )
    return await provider._handlers[tool](params, ctx), state


def _prompt(executor, state, job=None):
    """Everything the next thinking step is shown, stable and volatile."""
    job = job or _job()
    return "\n".join(
        [
            executor._build_thinking_prompt_stable(job, None, state),
            executor._build_thinking_prompt_volatile(job, state) or "",
        ]
    )


def _refused(result):
    return bool(result.get("error")) and not result.get("success")


def _spec(name):
    return next(spec for spec in agent_ops.SPECS if spec.name == name)


def test_the_provider_answers_exactly_the_four_reasoning_tools(executor):
    provider = build_autonomous_reasoning_provider(executor)
    assert set(provider._handlers) == {
        "reflect",
        "hypothesize",
        "weigh_evidence",
        "critique_plan",
    }


class TestReflect:
    async def test_it_records_the_reflection(self, executor):
        result, state = await _call(
            executor,
            "reflect",
            {
                "topic": "search strategy",
                "assessment": "Only arXiv has been searched",
                "blind_spots": ["vendor white papers"],
                "suggested_corrections": ["search the web too"],
            },
            job=_job(iteration=4),
        )

        assert result["success"] is True
        assert result["data"]["reflection_count"] == 1
        entry = state["reflections"][0]
        assert result["data"]["recorded"] == entry
        assert entry["iteration"] == 4
        assert entry["topic"] == "search strategy"
        assert entry["assessment"] == "Only arXiv has been searched"
        assert entry["blind_spots"] == ["vendor white papers"]
        assert entry["suggested_corrections"] == ["search the web too"]
        datetime.fromisoformat(entry["timestamp"])

    async def test_reflections_accumulate_in_order(self, executor):
        state = {}
        await _call(executor, "reflect", {"topic": "a", "assessment": "1"}, state)
        result, _ = await _call(
            executor, "reflect", {"topic": "b", "assessment": "2"}, state
        )
        assert result["data"]["reflection_count"] == 2
        assert [r["topic"] for r in state["reflections"]] == ["a", "b"]

    async def test_text_and_lists_are_capped(self, executor):
        _, state = await _call(
            executor,
            "reflect",
            {
                "topic": "t" * 400,
                "assessment": "a" * 900,
                "blind_spots": ["b" * 300] * 15,
                "suggested_corrections": ["c" * 300] * 15,
            },
        )
        entry = state["reflections"][0]
        assert len(entry["topic"]) == 300
        assert len(entry["assessment"]) == 500
        assert len(entry["blind_spots"]) == 10
        assert len(entry["suggested_corrections"]) == 10
        assert {len(b) for b in entry["blind_spots"]} == {200}
        assert {len(c) for c in entry["suggested_corrections"]} == {200}

    async def test_only_the_newest_fifty_are_kept(self, executor):
        state = {}
        for i in range(55):
            result, _ = await _call(
                executor, "reflect", {"topic": f"topic {i}", "assessment": "x"}, state
            )
        assert result["data"]["reflection_count"] == 50
        assert state["reflections"][0]["topic"] == "topic 5"
        assert state["reflections"][-1]["topic"] == "topic 54"

    @pytest.mark.xfail(
        strict=True,
        reason="reflect ignores its required parameters: an empty call is "
        "recorded as a reflection with blank topic and assessment",
    )
    async def test_a_reflection_without_topic_or_assessment_is_refused(self, executor):
        result, state = await _call(executor, "reflect", {})
        assert _spec("reflect").parameters["required"] == ["topic", "assessment"]
        assert _refused(result)
        assert not state.get("reflections")

    @pytest.mark.xfail(
        strict=True,
        reason="state['reflections'] is written and never read: no prompt, "
        "result or transcript shows a reflection again",
    )
    async def test_a_reflection_is_available_for_future_reference(self, executor):
        _, state = await _call(
            executor,
            "reflect",
            {
                "topic": "evidence quality",
                "assessment": "Three of four sources are the same preprint",
            },
        )
        assert "same preprint" in _prompt(executor, state)


class TestHypothesize:
    async def test_it_creates_a_hypothesis(self, executor):
        result, state = await _call(
            executor,
            "hypothesize",
            {
                "hypothesis": "Stride prefetching does not help pointer chasing",
                "rationale": "Addresses are data dependent",
                "testable_predictions": ["pfUseful stays near zero"],
            },
        )

        assert result["success"] is True
        assert result["data"]["action"] == "created"
        entry = state["hypotheses"][0]
        assert result["data"]["hypothesis"] == entry
        assert entry["id"] == "h-1"
        assert entry["status"] == "proposed"
        assert entry["hypothesis"].startswith("Stride prefetching")
        assert entry["rationale"] == "Addresses are data dependent"
        assert entry["testable_predictions"] == ["pfUseful stays near zero"]
        datetime.fromisoformat(entry["created_at"])

    async def test_ids_count_up_and_a_declared_status_is_kept(self, executor):
        state = {}
        await _call(executor, "hypothesize", {"hypothesis": "first"}, state)
        result, _ = await _call(
            executor,
            "hypothesize",
            {"hypothesis": "second", "status": "testing"},
            state,
        )
        assert result["data"]["hypothesis"]["id"] == "h-2"
        assert result["data"]["hypothesis"]["status"] == "testing"

    async def test_an_existing_hypothesis_is_updated_in_place(self, executor):
        state = {}
        await _call(
            executor,
            "hypothesize",
            {"hypothesis": "first", "rationale": "guess"},
            state,
        )
        result, _ = await _call(
            executor,
            "hypothesize",
            {
                "hypothesis": "first",
                "hypothesis_id": "h-1",
                "status": "supported",
                "rationale": "measured",
                "testable_predictions": ["p1"],
            },
            state,
        )

        assert result["success"] is True
        assert result["data"]["action"] == "updated"
        assert len(state["hypotheses"]) == 1
        entry = state["hypotheses"][0]
        assert entry["status"] == "supported"
        assert entry["rationale"] == "measured"
        assert entry["testable_predictions"] == ["p1"]
        assert entry["hypothesis"] == "first"
        datetime.fromisoformat(entry["updated_at"])

    async def test_updating_an_unknown_hypothesis_is_refused(self, executor):
        state = {}
        await _call(executor, "hypothesize", {"hypothesis": "first"}, state)
        result, _ = await _call(
            executor,
            "hypothesize",
            {"hypothesis": "x", "hypothesis_id": "h-99", "status": "refuted"},
            state,
        )

        assert result["success"] is False
        assert "h-99" in result["error"]
        assert result["data"]["available_ids"] == ["h-1"]
        assert len(state["hypotheses"]) == 1
        assert state["hypotheses"][0]["status"] == "proposed"

    async def test_text_and_predictions_are_capped(self, executor):
        _, state = await _call(
            executor,
            "hypothesize",
            {
                "hypothesis": "h" * 900,
                "rationale": "r" * 900,
                "testable_predictions": ["p" * 300] * 15,
            },
        )
        entry = state["hypotheses"][0]
        assert len(entry["hypothesis"]) == 500
        assert len(entry["rationale"]) == 400
        assert len(entry["testable_predictions"]) == 10
        assert {len(p) for p in entry["testable_predictions"]} == {200}

    async def test_only_the_newest_thirty_are_kept(self, executor):
        state = {}
        for i in range(33):
            await _call(executor, "hypothesize", {"hypothesis": f"claim {i}"}, state)
        assert len(state["hypotheses"]) == 30
        assert state["hypotheses"][-1]["hypothesis"] == "claim 32"

    @pytest.mark.xfail(
        strict=True,
        reason="hypothesize derives a new id from the list length, which stops "
        "growing at the 30-entry cap: every hypothesis after the 30th is 'h-31'",
    )
    async def test_ids_stay_unique_past_the_cap(self, executor):
        state = {}
        for i in range(33):
            await _call(executor, "hypothesize", {"hypothesis": f"claim {i}"}, state)
        ids = [h["id"] for h in state["hypotheses"]]
        assert len(set(ids)) == len(ids)

    @pytest.mark.xfail(
        strict=True,
        reason="hypothesize ignores its required parameter: with no hypothesis "
        "it stores an empty one and reports success",
    )
    async def test_a_hypothesis_without_a_statement_is_refused(self, executor):
        result, state = await _call(executor, "hypothesize", {"rationale": "why"})
        assert _spec("hypothesize").parameters["required"] == ["hypothesis"]
        assert _refused(result)
        assert not state.get("hypotheses")

    @pytest.mark.xfail(
        strict=True,
        reason="an update that names no status resets the hypothesis to "
        "'proposed', discarding a supported/refuted verdict",
    )
    async def test_an_update_without_a_status_keeps_the_status(self, executor):
        state = {}
        await _call(
            executor,
            "hypothesize",
            {"hypothesis": "first", "status": "supported"},
            state,
        )
        await _call(
            executor,
            "hypothesize",
            {"hypothesis": "first", "hypothesis_id": "h-1", "rationale": "more"},
            state,
        )
        assert state["hypotheses"][0]["rationale"] == "more"
        assert state["hypotheses"][0]["status"] == "supported"

    @pytest.mark.xfail(
        strict=True,
        reason="a status outside the declared enum is silently replaced by "
        "'proposed' and written, instead of being refused",
    )
    async def test_an_unknown_status_is_refused(self, executor):
        state = {}
        await _call(
            executor,
            "hypothesize",
            {"hypothesis": "first", "status": "refuted"},
            state,
        )
        result, _ = await _call(
            executor,
            "hypothesize",
            {"hypothesis": "first", "hypothesis_id": "h-1", "status": "confirmed"},
            state,
        )
        assert _refused(result)
        assert state["hypotheses"][0]["status"] == "refuted"

    @pytest.mark.xfail(
        strict=True,
        reason="state['hypotheses'] is written and never read: the model is "
        "never shown its tracked hypotheses or their ids again",
    )
    async def test_tracked_hypotheses_are_shown_to_the_next_step(self, executor):
        _, state = await _call(
            executor,
            "hypothesize",
            {"hypothesis": "Stride prefetching does not help pointer chasing"},
        )
        assert "does not help pointer chasing" in _prompt(executor, state)


class TestWeighEvidence:
    async def test_it_records_a_scored_entry(self, executor):
        result, state = await _call(
            executor,
            "weigh_evidence",
            {
                "claim": "Stride prefetching helps array scans",
                "evidence_for": [
                    {
                        "statement": "2.1x on the scan kernel",
                        "source_document_id": "doc-1",
                        "strength": 0.9,
                    },
                    {"statement": "pfUseful is high"},
                ],
                "evidence_against": [{"statement": "one noisy run", "strength": 0.3}],
                "verdict": "weakly_supported",
            },
        )

        assert result["success"] is True
        assert result["data"]["ledger_size"] == 1
        entry = state["evidence_ledger"][0]
        assert result["data"]["entry"] == entry
        assert entry["claim"] == "Stride prefetching helps array scans"
        assert entry["verdict"] == "weakly_supported"
        assert entry["hypothesis_id"] is None
        assert entry["evidence_for"] == [
            {
                "statement": "2.1x on the scan kernel",
                "source_document_id": "doc-1",
                "strength": 0.9,
            },
            # strength defaults to the midpoint when not given
            {
                "statement": "pfUseful is high",
                "source_document_id": "",
                "strength": 0.5,
            },
        ]
        assert entry["evidence_against"][0]["strength"] == 0.3
        assert entry["aggregate_score"] == pytest.approx(0.9 + 0.5 - 0.3)
        datetime.fromisoformat(entry["timestamp"])

    async def test_strength_is_clamped_to_the_unit_interval(self, executor):
        _, state = await _call(
            executor,
            "weigh_evidence",
            {
                "claim": "c",
                "verdict": "neutral",
                "evidence_for": [{"statement": "a", "strength": 7}],
                "evidence_against": [{"statement": "b", "strength": -2}],
            },
        )
        entry = state["evidence_ledger"][0]
        assert entry["evidence_for"][0]["strength"] == 1.0
        assert entry["evidence_against"][0]["strength"] == 0.0
        assert entry["aggregate_score"] == 1.0

    async def test_no_evidence_scores_zero(self, executor):
        _, state = await _call(
            executor, "weigh_evidence", {"claim": "c", "verdict": "neutral"}
        )
        entry = state["evidence_ledger"][0]
        assert entry["evidence_for"] == [] and entry["evidence_against"] == []
        assert entry["aggregate_score"] == 0

    @pytest.mark.parametrize(
        "verdict, status",
        [
            ("strongly_supported", "supported"),
            ("weakly_supported", "supported"),
            ("strongly_refuted", "refuted"),
            ("weakly_refuted", "refuted"),
            ("neutral", "testing"),
        ],
    )
    async def test_a_verdict_settles_the_linked_hypothesis(
        self, executor, verdict, status
    ):
        state = {}
        await _call(
            executor, "hypothesize", {"hypothesis": "h", "status": "testing"}, state
        )
        await _call(executor, "hypothesize", {"hypothesis": "other"}, state)
        _, state = await _call(
            executor,
            "weigh_evidence",
            {"claim": "c", "hypothesis_id": "h-1", "verdict": verdict},
            state,
        )

        assert state["evidence_ledger"][0]["hypothesis_id"] == "h-1"
        assert state["hypotheses"][0]["status"] == status
        assert state["hypotheses"][1]["status"] == "proposed"

    async def test_text_and_evidence_are_capped(self, executor):
        _, state = await _call(
            executor,
            "weigh_evidence",
            {
                "claim": "c" * 900,
                "verdict": "neutral",
                "evidence_for": [{"statement": "s" * 500}] * 14,
                "evidence_against": [{"statement": "s" * 500}] * 14,
            },
        )
        entry = state["evidence_ledger"][0]
        assert len(entry["claim"]) == 500
        assert len(entry["evidence_for"]) == 10
        assert len(entry["evidence_against"]) == 10
        assert len(entry["evidence_for"][0]["statement"]) == 300

    async def test_only_the_newest_hundred_entries_are_kept(self, executor):
        state = {}
        for i in range(103):
            result, _ = await _call(
                executor,
                "weigh_evidence",
                {"claim": f"claim {i}", "verdict": "neutral"},
                state,
            )
        assert result["data"]["ledger_size"] == 100
        assert state["evidence_ledger"][0]["claim"] == "claim 3"

    @pytest.mark.xfail(
        strict=True,
        reason="weigh_evidence ignores its required parameters: with no claim "
        "and no verdict it records a blank 'neutral' entry and reports success",
    )
    async def test_evidence_without_claim_or_verdict_is_refused(self, executor):
        result, state = await _call(executor, "weigh_evidence", {})
        assert _spec("weigh_evidence").parameters["required"] == ["claim", "verdict"]
        assert _refused(result)
        assert not state.get("evidence_ledger")

    @pytest.mark.xfail(
        strict=True,
        reason="a verdict outside the declared enum is silently recorded as "
        "'neutral', so the linked hypothesis is never settled and nothing says so",
    )
    async def test_an_unknown_verdict_is_refused(self, executor):
        state = {}
        await _call(executor, "hypothesize", {"hypothesis": "h"}, state)
        result, _ = await _call(
            executor,
            "weigh_evidence",
            {"claim": "c", "hypothesis_id": "h-1", "verdict": "supported"},
            state,
        )
        assert _refused(result)

    @pytest.mark.xfail(
        strict=True,
        reason="a non-numeric strength makes the handler raise ValueError from "
        "float() instead of answering with an error or the default",
    )
    async def test_a_non_numeric_strength_does_not_crash_the_tool(self, executor):
        result, _ = await _call(
            executor,
            "weigh_evidence",
            {
                "claim": "c",
                "verdict": "neutral",
                "evidence_for": [{"statement": "s", "strength": "high"}],
            },
        )
        assert isinstance(result, dict)

    @pytest.mark.xfail(
        strict=True,
        reason="evidence linked to a hypothesis id that does not exist is "
        "accepted without a word, where hypothesize refuses the same id",
    )
    async def test_evidence_for_an_unknown_hypothesis_is_flagged(self, executor):
        state = {}
        await _call(executor, "hypothesize", {"hypothesis": "h"}, state)
        result, _ = await _call(
            executor,
            "weigh_evidence",
            {"claim": "c", "hypothesis_id": "h-99", "verdict": "strongly_refuted"},
            state,
        )
        assert "h-99" in str(result.get("error") or result.get("warning") or "")

    @pytest.mark.xfail(
        strict=True,
        reason="state['evidence_ledger'] is written and never read: the "
        "'running ledger' is not shown to the model or kept in the results",
    )
    async def test_the_ledger_is_shown_to_the_next_step(self, executor):
        _, state = await _call(
            executor,
            "weigh_evidence",
            {
                "claim": "Stride prefetching helps array scans",
                "verdict": "strongly_supported",
            },
        )
        assert "helps array scans" in _prompt(executor, state)


class TestCritiquePlan:
    async def test_it_records_the_critique(self, executor):
        result, state = await _call(
            executor,
            "critique_plan",
            {
                "plan_summary": "Search, read, summarise",
                "weaknesses": ["no measurement"],
                "missing_steps": ["run the kernel"],
                "assumptions_challenged": ["the paper's numbers transfer"],
                "severity": "minor",
            },
            job=_job(iteration=6),
        )

        assert result["success"] is True
        assert result["data"]["critiques_count"] == 1
        entry = state["plan_critiques"][0]
        assert result["data"]["critique"] == entry
        assert entry["iteration"] == 6
        assert entry["plan_summary"] == "Search, read, summarise"
        assert entry["weaknesses"] == ["no measurement"]
        assert entry["missing_steps"] == ["run the kernel"]
        assert entry["assumptions_challenged"] == ["the paper's numbers transfer"]
        assert entry["severity"] == "minor"
        datetime.fromisoformat(entry["timestamp"])

    async def test_severity_defaults_to_moderate(self, executor):
        _, state = await _call(
            executor, "critique_plan", {"plan_summary": "p", "weaknesses": ["w"]}
        )
        assert state["plan_critiques"][0]["severity"] == "moderate"

    async def test_a_major_critique_becomes_critic_feedback_in_the_prompt(
        self, executor
    ):
        _, state = await _call(
            executor,
            "critique_plan",
            {
                "plan_summary": "Summarise without measuring",
                "weaknesses": ["no baseline", "one kernel", "no spread", "fourth"],
                "severity": "major",
            },
        )

        note = state["critic_notes"][-1]
        assert note["source"] == "critique_plan_tool"
        rendered = agent_prompt_sections.format_critic(state)
        assert "Summarise without measuring" in rendered
        assert "no baseline; one kernel; no spread" in rendered
        assert "fourth" not in rendered
        assert "Summarise without measuring" in _prompt(executor, state)

    @pytest.mark.parametrize("severity", ["minor", "moderate"])
    async def test_a_lesser_critique_is_not_critic_feedback(self, executor, severity):
        _, state = await _call(
            executor,
            "critique_plan",
            {"plan_summary": "p", "weaknesses": ["w"], "severity": severity},
        )
        assert not state.get("critic_notes")
        assert agent_prompt_sections.format_critic(state) == ""

    async def test_critic_feedback_keeps_only_the_newest_six(self, executor):
        state = {
            "critic_notes": [{"trajectory_assessment": f"old {i}"} for i in range(6)]
        }
        await _call(
            executor,
            "critique_plan",
            {"plan_summary": "new plan", "weaknesses": ["w"], "severity": "major"},
            state,
        )
        assert len(state["critic_notes"]) == 6
        assert state["critic_notes"][0]["trajectory_assessment"] == "old 1"
        assert "new plan" in state["critic_notes"][-1]["trajectory_assessment"]

    async def test_text_and_lists_are_capped(self, executor):
        _, state = await _call(
            executor,
            "critique_plan",
            {
                "plan_summary": "p" * 900,
                "weaknesses": ["w" * 300] * 15,
                "missing_steps": ["m" * 300] * 15,
                "assumptions_challenged": ["a" * 300] * 15,
            },
        )
        entry = state["plan_critiques"][0]
        assert len(entry["plan_summary"]) == 500
        for key in ("weaknesses", "missing_steps", "assumptions_challenged"):
            assert len(entry[key]) == 10
            assert {len(item) for item in entry[key]} == {200}

    async def test_only_the_newest_twenty_are_kept(self, executor):
        state = {}
        for i in range(23):
            result, _ = await _call(
                executor,
                "critique_plan",
                {"plan_summary": f"plan {i}", "weaknesses": ["w"]},
                state,
            )
        assert result["data"]["critiques_count"] == 20
        assert state["plan_critiques"][0]["plan_summary"] == "plan 3"

    @pytest.mark.xfail(
        strict=True,
        reason="critique_plan ignores its required parameters: an empty call "
        "is recorded as a critique with no plan and no weaknesses",
    )
    async def test_a_critique_without_plan_or_weaknesses_is_refused(self, executor):
        result, state = await _call(executor, "critique_plan", {})
        assert _spec("critique_plan").parameters["required"] == [
            "plan_summary",
            "weaknesses",
        ]
        assert _refused(result)
        assert not state.get("plan_critiques")

    @pytest.mark.xfail(
        strict=True,
        reason="a severity outside the declared enum is silently recorded as "
        "'moderate', so a critique meant as severe never reaches the critic",
    )
    async def test_an_unknown_severity_is_refused(self, executor):
        result, _ = await _call(
            executor,
            "critique_plan",
            {"plan_summary": "p", "weaknesses": ["w"], "severity": "critical"},
        )
        assert _refused(result)

    @pytest.mark.xfail(
        strict=True,
        reason="state['plan_critiques'] is written and never read: a minor or "
        "moderate critique, and every missing step, is shown to nobody",
    )
    async def test_a_moderate_critique_is_shown_to_the_next_step(self, executor):
        _, state = await _call(
            executor,
            "critique_plan",
            {
                "plan_summary": "Summarise without measuring",
                "weaknesses": ["no baseline"],
                "missing_steps": ["run the kernel on gem5"],
            },
        )
        assert "run the kernel on gem5" in _prompt(executor, state)
