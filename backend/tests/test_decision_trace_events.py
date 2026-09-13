"""Contract tests for the shared decision-trace event factory."""

from datetime import datetime

from app.modules.autonomy.application.decision_trace_events import (
    DecisionTraceEventDependencies,
    build_decision_trace_event,
    decision_trace_event_id,
)


def _normalize_context(**kwargs):
    kwargs["domain"] = str(kwargs.get("domain") or "").strip() or None
    return kwargs


def test_decision_trace_event_id_is_stable_and_suffix_sensitive():
    event_time = datetime(2026, 3, 16, 10, 0, 0)

    first = decision_trace_event_id("job", "job-1", "pause", event_time, "0")
    second = decision_trace_event_id("job", "job-1", "pause", event_time, "0")
    distinct = decision_trace_event_id("job", "job-1", "pause", event_time, "1")

    assert first == second
    assert first != distinct


def test_build_decision_trace_event_normalizes_and_isolates_payload():
    scheduler_state = {"queue_reason": "execution_failure", "nested": {"count": 1}}
    event = build_decision_trace_event(
        event_type="",
        event_time=datetime(2026, 3, 16, 10, 0, 0),
        source_kind=" job ",
        source_id=" job-1 ",
        source_label=" Recovery job ",
        decision_type="job_recovery_queued",
        summary="",
        scheduler_state=scheduler_state,
        deep_link={"target_tab": "queue", "params": {"job": "job-1"}},
        is_derived=True,
        operator_context={"domain": " Compiler "},
        deps=DecisionTraceEventDependencies(
            build_operator_context=_normalize_context,
        ),
    )
    scheduler_state["nested"]["count"] = 2

    assert event.event_type == "job_recovery_queued"
    assert event.source_kind == "job"
    assert event.source_id == "job-1"
    assert event.summary == "Autonomy event"
    assert event.record_origin == "derived"
    assert event.domain == "Compiler"
    assert event.deep_link is not None
    assert event.deep_link.target_tab == "queue"
    assert event.scheduler_state["nested"]["count"] == 1
