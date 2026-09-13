"""Golden-task regression suite for the autonomous agent loop.

Runs the REAL ``_run_autonomous_loop`` (observe→think→act→evaluate, budgets,
checkpoints, goal contracts, finalizer) end-to-end against in-memory SQLite,
with exactly two seams scripted:

- ``ScriptedLLM``: serves queued decision JSON only to decision-shaped
  prompts (identified by the decision schema in the message); every other
  LLM use (planning, critic, summaries, memory extraction) gets a harmless
  generic reply so it can never drain the script.
- ``ScriptedActionService``: canned per-tool results through the same
  ``act()`` seam the runtime uses.

These are behavioral contracts: if a prompt/parser/loop change breaks goal
completion, budget stops, malformed-output recovery, contract enforcement,
or tool-failure resilience, this suite fails.
"""

import copy
import json
from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.models.agent_job import AgentJob, AgentJobStatus
from app.services.autonomous_agent_executor import AutonomousAgentExecutor
from app.utils.exceptions import LLMServiceError


@pytest.fixture(autouse=True)
def _no_redis_feature_flags(monkeypatch):
    """Serve feature-flag defaults without Redis (no localhost:6379 retries)."""

    async def _default(key, default=None):
        return default

    monkeypatch.setattr("app.core.feature_flags.get_str", _default)
    monkeypatch.setattr("app.core.feature_flags.get_bool", _default, raising=False)


@pytest.fixture(autouse=True)
def _scripted_memory_service_llm(monkeypatch):
    """Point the memory-service singleton at a harmless scripted LLM.

    The finalizer's memory extraction uses this module-level singleton (not
    the executor's llm_service); a real call here would hit the network and,
    on failure, roll back the session mid-finalization.
    """
    from app.services.agent_job_memory_service import agent_job_memory_service

    monkeypatch.setattr(agent_job_memory_service, "llm_service", ScriptedLLM([]))


@pytest.fixture(autouse=True)
def _no_celery_dispatch(monkeypatch):
    """Record (instead of dispatch) Celery jobs queued by chain triggers."""
    from app.tasks.agent_job_tasks import execute_agent_job_task

    dispatched = []
    monkeypatch.setattr(
        execute_agent_job_task, "delay", lambda *args, **kwargs: dispatched.append(args)
    )
    return dispatched


STOP_DECISION = json.dumps(
    {
        "goal_achieved": False,
        "should_stop": True,
        "stop_reason": "Golden script exhausted",
        "reasoning": "No more scripted decisions.",
        "action": None,
    }
)


def _is_subsequence(needle, haystack):
    """True if all needle items appear in haystack in order (gaps allowed)."""
    it = iter(haystack)
    return all(item in it for item in needle)


def decision(
    tool=None,
    params=None,
    goal_achieved=False,
    should_stop=False,
    reasoning="scripted",
    assessment=None,
):
    action = (
        {"tool": tool, "params": params or {}, "purpose": "scripted"} if tool else None
    )
    return json.dumps(
        {
            "goal_achieved": goal_achieved,
            "should_stop": should_stop,
            "stop_reason": "",
            "reasoning": reasoning,
            "assessment": assessment,
            "action": action,
        }
    )


class ScriptedLLM:
    """Queue-backed decision source; generic text for all other prompts."""

    def __init__(self, decisions):
        self.queue = list(decisions)
        self.decision_calls = 0
        self.other_calls = 0

    @staticmethod
    def _is_decision_prompt(message: str) -> bool:
        return "goal_achieved" in message

    async def generate_response(self, **kwargs):
        message = str(
            kwargs.get("user_message")
            or kwargs.get("prompt")
            or kwargs.get("query")
            or ""
        )
        if self._is_decision_prompt(message):
            self.decision_calls += 1
            return self.queue.pop(0) if self.queue else STOP_DECISION
        self.other_calls += 1
        return "OK."

    async def generate_structured(self, **kwargs):
        # Golden runs exercise the prompted-text path deterministically.
        raise LLMServiceError("structured path disabled in golden harness")


class ScriptedActionService:
    """Canned per-tool results through the runtime's act() seam."""

    def __init__(self, results=None):
        self.results = results or {}
        self.calls = []

    async def act(self, executor, job, action, state, db):
        self.calls.append(copy.deepcopy(action))
        result = self.results.get(str(action.get("tool") or ""))
        if callable(result):
            result = result(action)
        if result is None:
            result = {"success": True, "data": {"ok": True}}
        return copy.deepcopy(result)

    @property
    def tools_called(self):
        return [str(a.get("tool") or "") for a in self.calls]


async def run_golden_job(
    db_session,
    *,
    decisions,
    tool_results=None,
    config=None,
    job_type="research",
    max_iterations=6,
):
    executor = AutonomousAgentExecutor()
    llm = ScriptedLLM(decisions)
    actions = ScriptedActionService(tool_results)
    executor.llm_service = llm
    executor.decision_parser.llm_service = llm
    executor.action_service = actions

    job = AgentJob(
        name="Golden Task",
        goal="Find retrieval-quality evidence in the knowledge base",
        job_type=job_type,
        user_id=uuid4(),
        status=AgentJobStatus.RUNNING.value,
        config=config or {},
        max_iterations=max_iterations,
        max_tool_calls=20,
        max_llm_calls=40,
        max_runtime_minutes=10,
    )
    db_session.add(job)
    await db_session.commit()

    result = await executor._run_autonomous_loop(job, None, None, db_session, None)
    return SimpleNamespace(
        executor=executor, job=job, result=result, llm=llm, actions=actions
    )


SEARCH_RESULT = {
    "success": True,
    "data": {"documents": [{"id": "doc-1", "title": "Reranking basics", "score": 0.9}]},
}
FINDING_RESULT = {
    "success": True,
    "findings": [
        {
            "title": "Reranking improves precision",
            "category": "insight",
            "content": "Cross-encoder reranking lifted top-5 precision.",
        }
    ],
}


@pytest.mark.asyncio
async def test_golden_research_job_completes_goal(db_session):
    run = await run_golden_job(
        db_session,
        decisions=[
            decision(tool="search_documents", params={"query": "reranking"}),
            decision(
                tool="save_research_finding", params={"title": "t", "content": "c"}
            ),
            decision(goal_achieved=True, reasoning="Evidence gathered and saved."),
        ],
        tool_results={
            "search_documents": SEARCH_RESULT,
            "save_research_finding": FINDING_RESULT,
        },
    )

    assert run.job.status == AgentJobStatus.COMPLETED.value
    # Scripted tools must run in order; the loop may interleave its own
    # support actions (progress reports, verification) between them.
    assert _is_subsequence(
        ["search_documents", "save_research_finding"], run.actions.tools_called
    ), run.actions.tools_called
    assert run.job.tool_calls_used >= 2
    assert run.job.iteration <= 4
    assert run.llm.queue == []  # every scripted decision consumed, in order
    results = run.job.results if isinstance(run.job.results, dict) else {}
    assert results, "finalizer must persist results"


@pytest.mark.asyncio
async def test_golden_job_stops_at_max_iterations(db_session):
    # Script always continues; the loop's iteration budget must stop it.
    run = await run_golden_job(
        db_session,
        decisions=[
            decision(tool="search_documents", params={"query": f"q{i}"})
            for i in range(10)
        ],
        tool_results={"search_documents": SEARCH_RESULT},
        max_iterations=3,
    )

    assert run.job.iteration == 3
    assert run.job.status != AgentJobStatus.RUNNING.value
    # One scripted search per iteration (support actions may add more calls).
    assert run.actions.tools_called.count("search_documents") == 3


@pytest.mark.asyncio
async def test_golden_job_recovers_from_malformed_llm_output(db_session):
    run = await run_golden_job(
        db_session,
        decisions=[
            "I think we should probably search the knowledge base first!",
            decision(tool="search_documents", params={"query": "reranking"}),
            decision(goal_achieved=True),
        ],
        tool_results={"search_documents": SEARCH_RESULT},
    )

    assert run.job.status == AgentJobStatus.COMPLETED.value
    assert "search_documents" in run.actions.tools_called
    metrics = run.executor.decision_parser.metrics
    assert metrics["parse_retry"] >= 1  # the garbage response forced a retry


@pytest.mark.asyncio
async def test_golden_goal_contract_blocks_false_completion(db_session):
    # Iteration 1 claims success with zero findings; the contract must block
    # it and keep the loop running until a finding exists.
    run = await run_golden_job(
        db_session,
        decisions=[
            decision(goal_achieved=True, reasoning="premature claim"),
            decision(
                tool="save_research_finding", params={"title": "t", "content": "c"}
            ),
            decision(goal_achieved=True, reasoning="now with evidence"),
        ],
        tool_results={"save_research_finding": FINDING_RESULT},
        config={
            "goal_contract": {
                "enabled": True,
                "min_findings": 1,
                "min_progress": 0,
            }
        },
    )

    assert run.job.status == AgentJobStatus.COMPLETED.value
    # The premature claim must not have ended the run before the finding.
    assert "save_research_finding" in run.actions.tools_called
    assert run.job.iteration >= 2


@pytest.mark.asyncio
async def test_golden_job_survives_tool_failure(db_session):
    run = await run_golden_job(
        db_session,
        decisions=[
            decision(tool="search_documents", params={"query": "broken"}),
            decision(tool="search_documents", params={"query": "retry works"}),
            decision(goal_achieved=True),
        ],
        tool_results={
            "search_documents": (
                lambda action: {"success": False, "error": "backend unavailable"}
                if action["params"].get("query") == "broken"
                else SEARCH_RESULT
            ),
        },
    )

    assert run.job.status == AgentJobStatus.COMPLETED.value
    assert run.actions.tools_called.count("search_documents") == 2
    assert run.job.error_count in (0, 1)  # tool failure must not fail the job


@pytest.mark.asyncio
async def test_golden_repeated_identical_failure_escalates_to_a_protocol(db_session):
    """The real loop must tell a run to stop retrying what cannot work.

    One run called the compiler with an unsupported flag four times, reading a
    message that could not help because no retry could have fixed it. By the
    third identical failure the run's own record should show the escalation.
    """
    params = {"code": "int main(void){return 0;}", "flags": "-O3 -march=native"}
    failure = {
        "success": False,
        "error": "Compilation failed: clang: error: unsupported argument 'native'",
    }

    run = await run_golden_job(
        db_session,
        decisions=[
            decision(tool="compile_c_snippet", params=params),
            decision(tool="compile_c_snippet", params=params),
            decision(tool="compile_c_snippet", params=params),
            decision(goal_achieved=True, reasoning="done"),
        ],
        tool_results={"compile_c_snippet": failure},
        max_iterations=6,
    )

    ledger = (run.job.results or {}).get("actions") or []
    repeats = [
        row
        for row in ledger
        if row.get("tool") == "compile_c_snippet" and row.get("repeat_attempt")
    ]
    assert repeats, "a thrice-repeated identical failure left no trace in the ledger"
    assert max(row["repeat_attempt"] for row in repeats) >= 3
    assert any(row.get("diagnosis_escalated") for row in repeats)
    assert all(row.get("failure_class") == "compilation" for row in repeats)


@pytest.mark.asyncio
async def test_golden_varied_retries_are_not_flagged_as_repeats(db_session):
    """Changing the call between attempts is the wanted behaviour."""
    failure = {"success": False, "error": "Compilation failed: some error"}

    run = await run_golden_job(
        db_session,
        decisions=[
            decision(tool="compile_c_snippet", params={"code": "a", "flags": "-O1"}),
            decision(tool="compile_c_snippet", params={"code": "a", "flags": "-O2"}),
            decision(tool="compile_c_snippet", params={"code": "a", "flags": "-O3"}),
            decision(goal_achieved=True, reasoning="done"),
        ],
        tool_results={"compile_c_snippet": failure},
        max_iterations=6,
    )

    ledger = (run.job.results or {}).get("actions") or []
    assert not [row for row in ledger if row.get("repeat_attempt")]


@pytest.mark.asyncio
async def test_golden_a_voluntary_stop_cannot_skip_the_contract(db_session):
    """Deciding the answer is "no" is a reason to stop, not a reason to skip
    settling the prediction that produced it.

    A live run stopped at iteration 6 with its predictions unsettled and no
    method recorded, and still reported completed: the contract gated
    goal_achieved and left this path open.
    """
    run = await run_golden_job(
        db_session,
        decisions=[
            decision(tool="search_documents", params={"query": "x"}),
            decision(should_stop=True, reasoning="nothing worth proposing"),
            decision(tool="search_documents", params={"query": "y"}),
            decision(goal_achieved=True, reasoning="done"),
        ],
        tool_results={"search_documents": SEARCH_RESULT},
        config={
            "goal_contract": {
                "enabled": True,
                "min_progress": 0,
                "required_finding_types": ["never_produced"],
            }
        },
        max_iterations=6,
    )

    assert run.job.iteration > 2, "the run stopped despite an unmet contract"
    blocked = [
        entry
        for entry in run.job.execution_log or []
        if entry.get("phase") == "voluntary_stop_blocked"
    ]
    contract = (run.job.results or {}).get("goal_contract") or {}
    assert contract.get("satisfied") is False
    assert blocked or contract.get("stopped_short"), (
        "a voluntary stop under an unmet contract must be blocked or recorded "
        "as stopping short"
    )


@pytest.mark.asyncio
async def test_golden_an_insistent_stop_is_honoured_and_recorded(db_session):
    """The contract holds a run to its requirements; it must not trap one
    whose tools have genuinely stopped working."""
    run = await run_golden_job(
        db_session,
        decisions=[decision(should_stop=True, reasoning="cannot proceed")] * 6,
        tool_results={},
        config={
            "goal_contract": {
                "enabled": True,
                "min_progress": 0,
                "required_finding_types": ["never_produced"],
            }
        },
        max_iterations=8,
    )

    contract = (run.job.results or {}).get("goal_contract") or {}
    assert contract.get("satisfied") is False
    assert (
        contract.get("stopped_short") is True
    ), "an insistent stop must be honoured, and recorded as short of contract"


@pytest.mark.asyncio
async def test_golden_a_stop_with_no_contract_is_left_alone(db_session):
    run = await run_golden_job(
        db_session,
        decisions=[decision(should_stop=True, reasoning="done here")],
        tool_results={},
        max_iterations=6,
    )

    assert run.job.iteration == 1, "an ungated run should stop when it says so"
