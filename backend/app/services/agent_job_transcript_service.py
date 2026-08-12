"""Assemble a readable record of what an agent job did.

Three sources, each answering a different question:

* the job row — what it was asked to do and how it ended
* its checkpoints — which tools it chose, why, and what came back
* its LLM snapshots — the verbatim prompts and replies, when the run was made
  with LLM_CALL_SNAPSHOT_ENABLED

A job driven by a deterministic runner makes no model calls at all, so an empty
conversation there is a fact about the run rather than missing data. The
``conversation`` block says which case a reader is looking at.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List

# Prompts and replies are unbounded; a transcript is for reading, not replay.
MAX_TEXT_CHARS = 20000
MAX_RESULT_LEAF_CHARS = 2000


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _clip(value: Any, limit: int) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    text = text or ""
    return text if len(text) <= limit else text[:limit] + "…[truncated]"


def _shrink(value: Any, budget: int) -> Any:
    """Shrink a result to fit without breaking its structure.

    Truncating serialized JSON leaves a string that no longer parses, so a
    reader analysing the export sees a compiled snippet's codegen counts as
    absent rather than large. Shorten the long leaves instead and keep the
    shape intact.
    """
    if isinstance(value, dict):
        return {key: _shrink(item, budget) for key, item in value.items()}
    if isinstance(value, list):
        return [_shrink(item, budget) for item in value[:20]]
    if isinstance(value, str) and len(value) > budget:
        return value[:budget] + f"…[{len(value) - budget} more chars]"
    return value


def _action_key(action: Dict[str, Any]) -> str:
    """Identify an action so the same one is not listed once per checkpoint."""
    explicit = action.get("_idempotency_key")
    if explicit:
        return str(explicit)
    return json.dumps(
        {k: v for k, v in action.items() if k != "_idempotency_key"},
        sort_keys=True,
        default=str,
    )[:400]


def build_actions(checkpoints: Iterable[Any]) -> List[Dict[str, Any]]:
    """Flatten checkpoint state into one ordered, de-duplicated action list.

    Checkpoints carry the whole accumulated state, so the same action appears in
    every later checkpoint.
    """
    actions: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for checkpoint in checkpoints:
        state = _as_dict(getattr(checkpoint, "state", None))
        for entry in state.get("actions_taken") or []:
            entry = _as_dict(entry)
            action = _as_dict(entry.get("action"))
            if not action:
                continue
            key = _action_key(action)
            if key in seen:
                continue
            seen.add(key)
            result = _as_dict(entry.get("result"))
            row = {
                "iteration": entry.get("iteration"),
                "node": entry.get("node"),
                "tool": action.get("tool"),
                "reason": action.get("purpose"),
                "params": action.get("params"),
                "success": result.get("success", result.get("ok")),
                "error": result.get("error"),
                "result": _shrink(result, MAX_RESULT_LEAF_CHARS),
            }
            # Say when a different tool actually ran. A successful fallback
            # rewrites the result to success, so without this a reader sees the
            # requested tool and a tick and has no way to tell them apart.
            requested = str(result.get("primary_tool") or "").strip()
            executed = str(result.get("tool") or "").strip()
            if requested and executed and requested != executed:
                row["substituted"] = True
                row["requested_tool"] = requested
                row["executed_tool"] = executed
                row["primary_error"] = str(result.get("primary_error") or "")[:300]
            actions.append(row)
    return actions


def build_reasoning(checkpoints: Iterable[Any]) -> Dict[str, Any]:
    """Pull the model-written commentary out of the latest checkpoint."""
    latest: Dict[str, Any] = {}
    for checkpoint in checkpoints:
        state = _as_dict(getattr(checkpoint, "state", None))
        if state:
            latest = state
    return {
        "critic_notes": latest.get("critic_notes") or [],
        "compressed_history": latest.get("compressed_history"),
        "progress_history": latest.get("progress_history") or [],
        "findings": latest.get("findings") or [],
    }


def build_conversation(
    snapshots: Iterable[Any],
    *,
    include_prompts: bool = False,
) -> List[Dict[str, Any]]:
    """Render captured LLM calls, optionally with their full prompt text."""
    calls: List[Dict[str, Any]] = []
    for snapshot in snapshots:
        request = _as_dict(getattr(snapshot, "request", None))
        call: Dict[str, Any] = {
            "iteration": getattr(snapshot, "iteration", None),
            "phase": getattr(snapshot, "phase", None),
            "provider": getattr(snapshot, "provider", None),
            "model": getattr(snapshot, "model", None),
            "task_type": getattr(snapshot, "task_type", None),
            "latency_ms": getattr(snapshot, "latency_ms", None),
            "prompt_tokens": getattr(snapshot, "prompt_tokens", None),
            "completion_tokens": getattr(snapshot, "completion_tokens", None),
            "error": getattr(snapshot, "error", None),
        }
        if include_prompts:
            messages = request.get("messages")
            if isinstance(messages, list) and messages:
                call["messages"] = [
                    {
                        "role": _as_dict(m).get("role"),
                        "content": _clip(_as_dict(m).get("content"), MAX_TEXT_CHARS),
                    }
                    for m in messages
                ]
            else:
                call["system_prompt"] = _clip(
                    request.get("system_prompt") or "", MAX_TEXT_CHARS
                )
                call["user_message"] = _clip(
                    request.get("user_message") or "", MAX_TEXT_CHARS
                )
            reply = getattr(snapshot, "response_text", None)
            if not reply and getattr(snapshot, "structured", None) is not None:
                reply = json.dumps(snapshot.structured, default=str)
            call["response"] = _clip(reply or "", MAX_TEXT_CHARS)
        calls.append(call)
    return calls


def build_job_transcript(
    job: Any,
    checkpoints: Iterable[Any],
    snapshots: Iterable[Any],
    *,
    include_prompts: bool = False,
) -> Dict[str, Any]:
    """Assemble the full transcript payload for one job."""
    checkpoints = list(checkpoints)
    snapshots = list(snapshots)
    actions = build_actions(checkpoints)
    llm_calls_used = int(getattr(job, "llm_calls_used", 0) or 0)

    if llm_calls_used == 0:
        availability = "no_llm_calls_deterministic_run"
    elif snapshots:
        availability = "captured"
    else:
        availability = "not_captured_snapshots_disabled"

    return {
        "job": {
            "id": str(getattr(job, "id", "") or ""),
            "name": getattr(job, "name", None),
            "job_type": getattr(job, "job_type", None),
            "goal": getattr(job, "goal", None),
            "status": getattr(job, "status", None),
            "iterations": getattr(job, "iteration", None),
            "llm_calls_used": llm_calls_used,
            "tool_calls_used": getattr(job, "tool_calls_used", None),
            "created_at": str(getattr(job, "created_at", "") or "") or None,
            "completed_at": str(getattr(job, "completed_at", "") or "") or None,
        },
        "actions": actions,
        "reasoning": build_reasoning(checkpoints),
        "conversation": {
            "availability": availability,
            "include_prompts": bool(include_prompts),
            "calls": build_conversation(snapshots, include_prompts=include_prompts),
        },
        "results": getattr(job, "results", None),
    }
