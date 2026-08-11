"""Tests for the agent job transcript builder."""

from types import SimpleNamespace

from app.services.agent_job_transcript_service import (
    build_actions,
    build_conversation,
    build_job_transcript,
    build_reasoning,
)


def _checkpoint(state):
    return SimpleNamespace(state=state, iteration=1, phase="thinking")


def _action(tool, key, *, purpose="because", ok=True):
    return {
        "iteration": 1,
        "node": "act",
        "action": {
            "tool": tool,
            "purpose": purpose,
            "params": {"query": "x"},
            "_idempotency_key": key,
        },
        "result": {"success": ok, "data": {"hits": 3}},
    }


def _job(**kwargs):
    base = {
        "id": "job-1",
        "name": "Test Job",
        "job_type": "analysis",
        "goal": "Do the thing",
        "status": "completed",
        "iteration": 2,
        "llm_calls_used": 4,
        "tool_calls_used": 2,
        "created_at": "2026-08-11",
        "completed_at": "2026-08-11",
        "results": {"ok": True},
    }
    base.update(kwargs)
    return SimpleNamespace(**base)


def test_actions_are_not_repeated_across_checkpoints():
    """Each checkpoint holds the whole accumulated state, not just its delta."""
    first = _checkpoint({"actions_taken": [_action("search", "k1")]})
    second = _checkpoint(
        {"actions_taken": [_action("search", "k1"), _action("read", "k2")]}
    )

    actions = build_actions([first, second])

    assert [a["tool"] for a in actions] == ["search", "read"]


def test_action_carries_the_agents_stated_reason():
    actions = build_actions(
        [_checkpoint({"actions_taken": [_action("search", "k1", purpose="find refs")]})]
    )

    assert actions[0]["reason"] == "find refs"
    assert actions[0]["success"] is True


def test_actions_without_an_idempotency_key_are_still_deduplicated():
    entry = {"iteration": 1, "action": {"tool": "search", "params": {"q": 1}}}
    actions = build_actions([_checkpoint({"actions_taken": [entry, dict(entry)]})])

    assert len(actions) == 1


def test_malformed_state_is_tolerated():
    assert (
        build_actions([_checkpoint(None), _checkpoint({"actions_taken": "nope"})]) == []
    )
    assert build_actions([_checkpoint({"actions_taken": [None, {}]})]) == []


def test_reasoning_comes_from_the_latest_checkpoint():
    reasoning = build_reasoning(
        [
            _checkpoint({"progress_history": [10], "compressed_history": "early"}),
            _checkpoint({"progress_history": [10, 80], "compressed_history": "late"}),
        ]
    )

    assert reasoning["compressed_history"] == "late"
    assert reasoning["progress_history"] == [10, 80]


def _snapshot(**kwargs):
    base = {
        "iteration": 1,
        "phase": "thinking",
        "provider": "deepseek",
        "model": "deepseek-chat",
        "task_type": "chat",
        "latency_ms": 1200,
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "error": None,
        "request": {"system_prompt": "SYS", "user_message": "USR"},
        "response_text": "REPLY",
        "structured": None,
    }
    base.update(kwargs)
    return SimpleNamespace(**base)


def test_prompts_are_withheld_unless_requested():
    calls = build_conversation([_snapshot()], include_prompts=False)

    assert calls[0]["model"] == "deepseek-chat"
    assert "system_prompt" not in calls[0]
    assert "response" not in calls[0]


def test_prompts_are_included_on_request():
    calls = build_conversation([_snapshot()], include_prompts=True)

    assert calls[0]["system_prompt"] == "SYS"
    assert calls[0]["user_message"] == "USR"
    assert calls[0]["response"] == "REPLY"


def test_message_style_requests_are_rendered_as_a_dialogue():
    snapshot = _snapshot(
        request={
            "messages": [
                {"role": "system", "content": "S"},
                {"role": "user", "content": "U"},
            ]
        }
    )

    calls = build_conversation([snapshot], include_prompts=True)

    assert [m["role"] for m in calls[0]["messages"]] == ["system", "user"]


def test_structured_replies_stand_in_for_missing_text():
    snapshot = _snapshot(response_text=None, structured={"action": "search"})

    calls = build_conversation([snapshot], include_prompts=True)

    assert "search" in calls[0]["response"]


def test_long_text_is_truncated_rather_than_returned_whole():
    snapshot = _snapshot(response_text="x" * 50000)

    calls = build_conversation([snapshot], include_prompts=True)

    assert len(calls[0]["response"]) < 50000
    assert calls[0]["response"].endswith("[truncated]")


def test_deterministic_run_is_labelled_rather_than_looking_empty():
    """A runner that makes no model calls has no conversation to lose."""
    transcript = build_job_transcript(_job(llm_calls_used=0), [], [])

    assert (
        transcript["conversation"]["availability"] == "no_llm_calls_deterministic_run"
    )
    assert transcript["conversation"]["calls"] == []


def test_llm_run_without_snapshots_is_distinguished_from_a_deterministic_one():
    transcript = build_job_transcript(_job(llm_calls_used=5), [], [])

    assert (
        transcript["conversation"]["availability"] == "not_captured_snapshots_disabled"
    )


def test_captured_run_reports_its_calls():
    transcript = build_job_transcript(_job(), [], [_snapshot()], include_prompts=True)

    assert transcript["conversation"]["availability"] == "captured"
    assert transcript["conversation"]["calls"][0]["response"] == "REPLY"


def test_transcript_carries_job_identity_and_results():
    transcript = build_job_transcript(_job(), [], [])

    assert transcript["job"]["name"] == "Test Job"
    assert transcript["job"]["goal"] == "Do the thing"
    assert transcript["results"] == {"ok": True}
