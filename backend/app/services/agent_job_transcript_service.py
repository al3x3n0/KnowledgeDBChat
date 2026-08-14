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
from typing import Any, Dict, Iterable, List, Optional

# Prompts and replies are unbounded; a transcript is for reading, not replay.
MAX_TEXT_CHARS = 20000
MAX_RESULT_LEAF_CHARS = 2000


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _clip(value: Any, limit: int) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    text = text or ""
    return text if len(text) <= limit else text[:limit] + "…[truncated]"


MAX_LIST_ITEMS = 20


def _shrink(value: Any, budget: int, stats: Optional[Dict[str, int]] = None) -> Any:
    """Shrink a result to fit without breaking its structure or hiding the cut.

    Truncating serialized JSON leaves a string that no longer parses, so a
    reader analysing the export sees a compiled snippet's codegen counts as
    absent rather than large. Shorten the long leaves instead, keep the shape
    intact, and say where something was dropped: a silently shortened list
    reads as the whole list.
    """

    def _count(key: str) -> None:
        if stats is not None:
            stats[key] = stats.get(key, 0) + 1

    if isinstance(value, dict):
        return {key: _shrink(item, budget, stats) for key, item in value.items()}
    if isinstance(value, list):
        kept = [_shrink(item, budget, stats) for item in value[:MAX_LIST_ITEMS]]
        dropped = len(value) - len(kept)
        if dropped > 0:
            _count("lists_shortened")
            kept.append(f"…[{dropped} more items omitted]")
        return kept
    if isinstance(value, str) and len(value) > budget:
        _count("strings_shortened")
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


def build_actions(
    checkpoints: Iterable[Any],
    stats_out: Optional[Dict[str, int]] = None,
) -> List[Dict[str, Any]]:
    """Flatten checkpoint state into one ordered, de-duplicated action list.

    Checkpoints carry the whole accumulated state, so the same action appears in
    every later checkpoint.

    Pass ``stats_out`` to learn what was dropped on the way. Without it a reader
    cannot tell a run that took three actions from one whose entries were
    discarded as malformed.
    """
    stats: Dict[str, int] = stats_out if stats_out is not None else {}
    actions: List[Dict[str, Any]] = []
    seen: set[str] = set()

    def _count(key: str) -> None:
        stats[key] = stats.get(key, 0) + 1

    for checkpoint in checkpoints:
        state = _as_dict(getattr(checkpoint, "state", None))
        for entry in state.get("actions_taken") or []:
            _count("entries_seen")
            entry = _as_dict(entry)
            action = _as_dict(entry.get("action"))
            if not action:
                _count("entries_skipped_no_action")
                continue
            key = _action_key(action)
            if key in seen:
                _count("entries_deduplicated")
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
                "result": _shrink(result, MAX_RESULT_LEAF_CHARS, stats),
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


# A report is read, not replayed: keep the reason and the parameters legible
# without carrying a whole tool result onto the page.
TOOL_LOG_PURPOSE_CHARS = 300
TOOL_LOG_PARAM_CHARS = 200
TOOL_LOG_ERROR_CHARS = 400


def build_tool_log(checkpoints: Iterable[Any]) -> List[Dict[str, Any]]:
    """List every tool the job ran, in order, compact enough for a document.

    Tool calls live in checkpoint state, not in ``job.execution_log`` — that
    column only carries phase markers — so a report built from the job row
    alone cannot say what the agent actually did.
    """
    rows: List[Dict[str, Any]] = []
    for index, action in enumerate(build_actions(checkpoints), start=1):
        row = {
            "index": index,
            "iteration": action.get("iteration"),
            "node": action.get("node"),
            "tool": str(action.get("tool") or "unknown"),
            "purpose": _clip(action.get("reason") or "", TOOL_LOG_PURPOSE_CHARS),
            "params": _clip(action.get("params") or {}, TOOL_LOG_PARAM_CHARS),
            "success": action.get("success"),
            "error": _clip(action.get("error") or "", TOOL_LOG_ERROR_CHARS),
        }
        if action.get("substituted"):
            row["substituted"] = True
            row["requested_tool"] = action.get("requested_tool")
            row["executed_tool"] = action.get("executed_tool")
        rows.append(row)
    return rows


def summarize_tool_log(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Tally a tool log: totals, per-tool counts, and the calls that failed.

    ``success`` is whatever the tool result reported, so a missing value is
    counted as unknown rather than folded into either side — a run whose
    results carried no status should not read as a run that succeeded.
    """
    rows = list(rows)
    per_tool: Dict[str, Dict[str, int]] = {}
    succeeded = failed = unknown = 0
    failures: List[Dict[str, Any]] = []

    for row in rows:
        name = str(row.get("tool") or "unknown")
        tally = per_tool.setdefault(
            name, {"tool": name, "calls": 0, "ok": 0, "failed": 0}
        )
        tally["calls"] += 1
        success = row.get("success")
        if success is True:
            succeeded += 1
            tally["ok"] += 1
        elif success is False:
            failed += 1
            tally["failed"] += 1
            failures.append(row)
        else:
            unknown += 1

    return {
        "total": len(rows),
        "succeeded": succeeded,
        "failed": failed,
        "unknown": unknown,
        "substituted": sum(1 for row in rows if row.get("substituted")),
        "per_tool": sorted(
            per_tool.values(), key=lambda item: (-item["calls"], item["tool"])
        ),
        "failures": failures,
    }


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
    build_stats: Dict[str, int] = {}
    actions = build_actions(checkpoints, build_stats)
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
        # What this transcript does not contain. A diagnostic that quietly drops
        # data reads as evidence that there was none: reading an export whose
        # long results had been cut, I concluded a tool had measured nothing
        # when it had measured a great deal.
        "completeness": {
            "actions_listed": len(actions),
            "action_entries_seen": build_stats.get("entries_seen", 0),
            "action_entries_deduplicated": build_stats.get("entries_deduplicated", 0),
            "action_entries_skipped_no_action": build_stats.get(
                "entries_skipped_no_action", 0
            ),
            "results_with_shortened_text": build_stats.get("strings_shortened", 0),
            "results_with_shortened_lists": build_stats.get("lists_shortened", 0),
            "llm_calls_reported_by_job": llm_calls_used,
            "llm_calls_captured": len(snapshots),
            "llm_calls_not_captured": max(0, llm_calls_used - len(snapshots)),
            "prompts_included": bool(include_prompts),
            "checkpoints_read": len(checkpoints),
        },
        "results": getattr(job, "results", None),
    }
