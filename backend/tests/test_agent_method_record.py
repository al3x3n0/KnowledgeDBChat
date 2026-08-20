"""Methods must be structured, evidence-backed, and readable again later.

The knowledge worth keeping from an investigation is often not about the
subject but about how to investigate it: that instruction latency cannot be
measured through C expressions because the compiler reshapes them, that a
timing harness needs an operation whose answer is known in advance. That
knowledge existed here only as prose a human wrote down.
"""

from __future__ import annotations

import pytest

from app.services import agent_method_record as method

CHAINS = {
    "name": "measure latency with inline-asm dependent chains",
    "procedure": [
        "Emit a chain where each instruction reads the previous one's result.",
        "Anchor wall clock to cycles with a dependent integer add.",
        "Reject any run whose anchor residual is not 1.00 within tolerance.",
    ],
    "prevents": (
        "The compiler vectorises or hoists a C loop, so the measurement is of "
        "different code than the one being reasoned about."
    ),
}


def test_a_method_needs_evidence_that_exists_in_the_run():
    with pytest.raises(method.MethodRecordError) as excinfo:
        method.build_record(
            **CHAINS,
            derived_from=["cycle_model_measurement"],
            available_finding_types=["dynamic_profile"],
        )

    assert "no such finding exists" in str(excinfo.value)
    assert "dynamic_profile" in str(excinfo.value), "must say what is available"


def test_a_method_backed_by_a_real_finding_is_validated():
    record = method.build_record(
        **CHAINS,
        derived_from=["cycle_model_measurement"],
        available_finding_types=["cycle_model_measurement", "dynamic_profile"],
    )

    assert record["status"] == method.VALIDATED
    assert record["evidence"] == ["cycle_model_measurement"]


def test_an_untested_method_may_be_recorded_but_is_marked():
    record = method.build_record(
        **CHAINS,
        derived_from=["none"],
        available_finding_types=[],
        limits="Not yet demonstrated on vector instructions.",
    )

    assert record["status"] == method.UNVALIDATED
    assert record["evidence"] == []
    assert "unvalidated" in method.render(record)


def test_a_method_without_steps_is_refused():
    """An opinion is not something a later run can follow."""
    with pytest.raises(method.MethodRecordError) as excinfo:
        method.build_record(
            name="be careful with timing",
            procedure=[],
            prevents="bad measurements",
            derived_from=["none"],
            available_finding_types=[],
        )

    assert "procedure is required" in str(excinfo.value)


def test_a_method_must_say_what_it_prevents():
    """Without it a reader cannot tell whether their situation is the same."""
    with pytest.raises(method.MethodRecordError) as excinfo:
        method.build_record(
            name="use inline asm",
            procedure=["write inline asm"],
            prevents="   ",
            derived_from=["none"],
            available_finding_types=[],
        )

    assert "prevents is required" in str(excinfo.value)


def test_a_procedure_written_as_text_is_accepted():
    """Models write numbered blocks when not handed a list."""
    record = method.build_record(
        name="two-point slope",
        procedure="1. Run at N iterations.\n2. Run at 3N.\n3. Take the slope.",
        prevents="Simulator startup is counted as workload cost.",
        derived_from=["none"],
        available_finding_types=[],
    )

    assert record["procedure"] == [
        "Run at N iterations.",
        "Run at 3N.",
        "Take the slope.",
    ]


def test_a_record_survives_the_round_trip():
    """Recall puts these back as plain text; the structure must be readable."""
    record = method.build_record(
        **CHAINS,
        derived_from=["cycle_model_measurement"],
        available_finding_types=["cycle_model_measurement"],
        applies_to=["benchmark_c_snippet", "microarchitecture"],
        limits="Assumes the anchor instruction retires one per cycle.",
    )

    restored = method.parse(method.render(record))

    assert restored["name"] == record["name"]
    assert restored["procedure"] == record["procedure"]
    assert restored["prevents"] == record["prevents"]
    assert restored["applies_to"] == record["applies_to"]
    assert restored["limits"] == record["limits"]
    assert restored["status"] == method.VALIDATED
    assert restored["evidence"] == ["cycle_model_measurement"]


def test_an_unvalidated_record_round_trips_as_unvalidated():
    """Losing this on recall is how an untested method becomes authoritative."""
    record = method.build_record(
        **CHAINS, derived_from=["none"], available_finding_types=[]
    )

    assert method.parse(method.render(record))["status"] == method.UNVALIDATED


def test_ordinary_memories_are_not_read_as_methods():
    assert method.parse("The reranker lifted precision by 8%.") is None


def test_tags_make_a_method_findable_by_what_it_applies_to():
    record = method.build_record(
        **CHAINS,
        derived_from=["none"],
        available_finding_types=[],
        applies_to=["simulate_c_workload"],
    )

    tags = method.tags_for(record)

    assert "method" in tags
    assert "simulate_c_workload" in tags
    assert method.UNVALIDATED in tags


@pytest.mark.asyncio
async def test_the_tool_stores_a_method_and_refuses_a_fabricated_one(db_session):
    """The agent-facing path, against the real memory store."""
    from types import SimpleNamespace
    from uuid import uuid4

    from app.services.agent_tool_dispatch import AgentToolExecutionContext
    from app.services.autonomous_agent_executor import AutonomousAgentExecutor

    executor = AutonomousAgentExecutor()
    job = SimpleNamespace(id=uuid4(), user_id=uuid4(), config={})
    state = {"findings": [{"type": "cycle_model_measurement", "title": "fsqrt 10.1"}]}
    ctx = AgentToolExecutionContext(
        mode="autonomous",
        db=db_session,
        service=None,
        user_id=job.user_id,
        job=job,
        state=state,
    )
    provider = executor.tool_registry.resolve("record_method", ctx)

    stored = await provider.execute(
        "record_method",
        {
            **CHAINS,
            "derived_from": ["cycle_model_measurement"],
            "applies_to": ["benchmark_c_snippet"],
        },
        ctx,
    )

    assert stored["success"] is True
    assert stored["data"]["status"] == method.VALIDATED
    assert stored["findings"][0]["type"] == "method_recorded"

    refused = await provider.execute(
        "record_method",
        {**CHAINS, "derived_from": ["simulated_measurement"]},
        ctx,
    )

    assert "error" in refused
    assert "no such finding exists" in refused["error"]


@pytest.mark.asyncio
async def test_a_recorded_method_reaches_a_later_job(db_session):
    """Recording is pointless unless the next job inherits it.

    A method stored under a type the job-memory filter does not inject would
    be written and never recalled, which is indistinguishable from not having
    recorded it at all.
    """
    from types import SimpleNamespace
    from uuid import uuid4

    from app.models.agent_job import AgentJob, AgentJobStatus
    from app.services.agent_job_memory_service import agent_job_memory_service
    from app.services.agent_tool_dispatch import AgentToolExecutionContext
    from app.services.autonomous_agent_executor import AutonomousAgentExecutor

    user_id = uuid4()
    executor = AutonomousAgentExecutor()
    writing_job = SimpleNamespace(id=uuid4(), user_id=user_id, config={})
    ctx = AgentToolExecutionContext(
        mode="autonomous",
        db=db_session,
        service=None,
        user_id=user_id,
        job=writing_job,
        state={"findings": [{"type": "cycle_model_measurement"}]},
    )
    provider = executor.tool_registry.resolve("record_method", ctx)
    written = await provider.execute(
        "record_method",
        {
            **CHAINS,
            "derived_from": ["cycle_model_measurement"],
            "applies_to": ["benchmark_c_snippet"],
        },
        ctx,
    )
    assert written["success"] is True

    later_job = AgentJob(
        name="Later job",
        goal="Measure instruction latency on this host with inline assembly chains",
        job_type="research",
        user_id=user_id,
        status=AgentJobStatus.RUNNING.value,
        config={},
        max_iterations=3,
        max_tool_calls=5,
        max_llm_calls=5,
        max_runtime_minutes=5,
    )
    db_session.add(later_job)
    await db_session.commit()

    recalled = await agent_job_memory_service.get_relevant_memories_for_job(
        later_job, str(user_id), db_session, limit=10
    )

    contents = [m.content for m in recalled]
    assert any(c.startswith("METHOD:") for c in contents), (
        "the recorded method was not recalled for a later job on the same "
        f"subject; got {len(recalled)} memories"
    )
    restored = method.parse(next(c for c in contents if c.startswith("METHOD:")))
    assert restored["status"] == method.VALIDATED
    assert restored["procedure"], "the procedure must survive to be followable"
