"""An unreachable model provider stops the run; it is not "recovered" from.

Measured: DeepSeek answered "Connection error" from iteration 8 of a run, and
the loop spent seven more iterations on fallback searches and verifications
before pausing as "no new findings" -- blaming the work for an outage.
"""

import pytest

from app.services import agent_thinking_service as thinking
from tests.test_golden_agent_tasks import ScriptedLLM, run_golden_job

OUTAGE = "LLM service error: deepseek request error: Connection error."


class UnreachableLLM(ScriptedLLM):
    def __init__(self):
        super().__init__([])
        self.decision_calls = 0

    async def generate_response(self, **kwargs):
        message = str(kwargs.get("user_message") or kwargs.get("prompt") or "")
        if self._is_decision_prompt(message):
            self.decision_calls += 1
            raise RuntimeError(OUTAGE)
        return await super().generate_response(**kwargs)

    async def generate_structured(self, **kwargs):
        self.decision_calls += 1
        raise RuntimeError(OUTAGE)


def test_the_provider_errors_are_recognised_and_parse_errors_are_not():
    assert thinking.provider_unreachable(RuntimeError(OUTAGE))
    assert thinking.provider_unreachable(
        RuntimeError("Failed to generate response: Request error: ")
    )
    assert thinking.provider_unreachable(RuntimeError("HTTP 503 Service Unavailable"))
    assert not thinking.provider_unreachable(ValueError("Invalid JSON in decision"))


@pytest.mark.asyncio
async def test_an_outage_stops_the_run_without_fallback_actions(
    db_session, monkeypatch
):
    slept = []

    async def no_wait(seconds):
        slept.append(seconds)

    monkeypatch.setattr(thinking.asyncio, "sleep", no_wait)

    import tests.test_golden_agent_tasks as golden

    llm = UnreachableLLM()
    monkeypatch.setattr(golden, "ScriptedLLM", lambda decisions: llm)
    run = await run_golden_job(db_session, decisions=[], max_iterations=6)

    assert slept == list(thinking.PROVIDER_BACKOFF_SECONDS)
    assert run.actions.calls == [] or all(
        c.get("tool") in ("write_progress_report",) for c in run.actions.calls
    )
    logged = [
        e
        for e in (run.job.execution_log or [])
        if e.get("phase") == "provider_unreachable"
    ]
    assert logged and "provider could not be reached" in logged[-1]["reason"]
    assert run.executor is not None


@pytest.mark.asyncio
async def test_an_outage_is_not_argued_with_by_an_unmet_contract(
    db_session, monkeypatch
):
    async def no_wait(seconds):
        return None

    monkeypatch.setattr(thinking.asyncio, "sleep", no_wait)
    import tests.test_golden_agent_tasks as golden

    llm = UnreachableLLM()
    monkeypatch.setattr(golden, "ScriptedLLM", lambda decisions: llm)
    run = await run_golden_job(
        db_session,
        decisions=[],
        max_iterations=6,
        config={
            "goal_contract": {
                "min_progress": 0,
                "required_finding_types": {"restructuring_result": 1},
            }
        },
    )
    phases = [e.get("phase") for e in (run.job.execution_log or [])]
    assert "provider_unreachable" in phases
    assert "voluntary_stop_blocked" not in phases
    # One iteration's worth of attempts, not three blocked stops' worth.
    # Stopped in the iteration the outage began, and paused -- resumable --
    # rather than filed completed or failed.
    assert run.job.iteration == 1
    assert "blocked_needs_input" in phases
    assert run.job.status == "paused"
