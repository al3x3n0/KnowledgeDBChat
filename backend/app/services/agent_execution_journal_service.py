"""Durable, append-only execution journal for autonomous agent tool calls."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from uuid import uuid4

from sqlalchemy.orm.attributes import flag_modified

_EMPTY_HASH = "0" * 64
_SENSITIVE_KEY_PARTS = (
    "api_key",
    "authorization",
    "cookie",
    "credential",
    "password",
    "secret",
    "token",
)


class AgentExecutionJournalService:
    """Record durable tool intent/result pairs and detect interrupted calls."""

    def _config(self, job: Any) -> Dict[str, Any]:
        config = job.config if isinstance(job.config, dict) else {}
        try:
            max_entries = int(config.get("execution_journal_max_entries", 500) or 500)
        except (TypeError, ValueError):
            max_entries = 500
        return {
            "enabled": bool(config.get("execution_journal_enabled", True)),
            "max_entries": max(50, min(max_entries, 2000)),
        }

    @staticmethod
    def _canonical_hash(payload: Dict[str, Any]) -> str:
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), default=str
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _redact(self, value: Any) -> tuple[Any, bool]:
        redacted = False
        if isinstance(value, dict):
            output: Dict[str, Any] = {}
            for key, item in value.items():
                normalized = str(key).lower()
                if any(part in normalized for part in _SENSITIVE_KEY_PARTS):
                    output[str(key)] = "[REDACTED]"
                    redacted = True
                else:
                    output[str(key)], child_redacted = self._redact(item)
                    redacted = redacted or child_redacted
            return output, redacted
        if isinstance(value, list):
            output_list = []
            for item in value:
                clean, child_redacted = self._redact(item)
                output_list.append(clean)
                redacted = redacted or child_redacted
            return output_list, redacted
        return value, False

    def _append(
        self,
        *,
        job: Any,
        state: Dict[str, Any],
        event_type: str,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        config = self._config(job)
        rows = (
            list(state.get("execution_journal") or [])
            if isinstance(state.get("execution_journal"), list)
            else []
        )
        cursor = (
            dict(state.get("execution_journal_cursor") or {})
            if isinstance(state.get("execution_journal_cursor"), dict)
            else {}
        )
        sequence = int(cursor.get("sequence", 0) or 0) + 1
        previous_hash = str(cursor.get("entry_hash") or _EMPTY_HASH)
        entry = {
            "event_id": str(uuid4()),
            "sequence": sequence,
            "event_type": event_type,
            "job_id": str(job.id),
            "iteration": int(job.iteration or 0),
            "phase": str(job.current_phase or ""),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "previous_hash": previous_hash,
            **payload,
        }
        entry["entry_hash"] = self._canonical_hash(entry)
        rows.append(entry)
        state["execution_journal"] = rows[-config["max_entries"] :]
        state["execution_journal_cursor"] = {
            "sequence": sequence,
            "event_id": entry["event_id"],
            "entry_hash": entry["entry_hash"],
            "event_type": event_type,
        }
        return entry

    async def begin_tool_call(
        self,
        *,
        executor: Any,
        job: Any,
        state: Dict[str, Any],
        action: Dict[str, Any],
        db: Any,
    ) -> Optional[Dict[str, Any]]:
        """Persist tool intent before dispatching a potentially side-effecting call."""
        if not self._config(job)["enabled"]:
            return None
        execution_lease = getattr(executor, "execution_lease", None)
        if execution_lease is not None:
            from app.services.agent_execution_lease_service import (
                agent_execution_lease_service,
            )

            await agent_execution_lease_service.assert_owned(
                db=db,
                lease=execution_lease,
            )
        clean_params, contains_redactions = self._redact(
            action.get("params") if isinstance(action.get("params"), dict) else {}
        )
        raw_action_hash = self._canonical_hash(
            {
                "tool": str(action.get("tool") or ""),
                "params": (
                    action.get("params")
                    if isinstance(action.get("params"), dict)
                    else {}
                ),
            }
        )
        supplied_key = str(action.get("_idempotency_key") or "").strip().lower()
        if len(supplied_key) == 64 and all(
            char in "0123456789abcdef" for char in supplied_key
        ):
            idempotency_key = supplied_key
        else:
            idempotency_key = hashlib.sha256(
                (
                    f"agent-job:{job.id}:iteration:{int(job.iteration or 0)}:"
                    f"{raw_action_hash}"
                ).encode("utf-8")
            ).hexdigest()
        action["_idempotency_key"] = idempotency_key
        invocation_id = str(uuid4())
        action_payload = {
            "tool": str(action.get("tool") or ""),
            "purpose": str(action.get("purpose") or "")[:500],
            "params": clean_params,
            "_idempotency_key": idempotency_key,
        }
        entry = self._append(
            job=job,
            state=state,
            event_type="tool_intent",
            payload={
                "invocation_id": invocation_id,
                "action": action_payload,
                "action_hash": self._canonical_hash(action_payload),
                "idempotency_key": idempotency_key,
                "contains_redactions": contains_redactions,
                "retryable_from_journal": not contains_redactions,
                "status": "started",
            },
        )
        state["execution_journal_pending"] = {
            "invocation_id": invocation_id,
            "intent_event_id": entry["event_id"],
            "intent_sequence": entry["sequence"],
            "action": action_payload,
            "contains_redactions": contains_redactions,
            "retryable_from_journal": not contains_redactions,
            "idempotency_key": idempotency_key,
            "started_at": entry["timestamp"],
        }
        await executor.checkpoint_service.save_checkpoint(
            job=job,
            state=state,
            db=db,
            reason="tool_intent",
        )
        return entry

    async def complete_tool_call(
        self,
        *,
        executor: Any,
        job: Any,
        state: Dict[str, Any],
        intent: Optional[Dict[str, Any]],
        result: Any,
        db: Any,
    ) -> Optional[Dict[str, Any]]:
        """Persist the compact outcome corresponding to a durable tool intent."""
        if intent is None or not self._config(job)["enabled"]:
            return None
        normalized = result if isinstance(result, dict) else {}
        invocation_id = str(intent.get("invocation_id") or "")
        if isinstance(result, dict):
            result["_journal_invocation_id"] = invocation_id
        artifact_refs = []
        for artifact in normalized.get("artifacts") or []:
            if isinstance(artifact, dict):
                artifact_refs.append(
                    {
                        key: artifact.get(key)
                        for key in ("id", "type", "path", "uri", "checkpoint_id")
                        if artifact.get(key) is not None
                    }
                )
        entry = self._append(
            job=job,
            state=state,
            event_type="tool_result",
            payload={
                "invocation_id": invocation_id,
                "intent_event_id": str(intent.get("event_id") or ""),
                "tool": str(((intent.get("action") or {}).get("tool") or "")),
                "idempotency_key": str(intent.get("idempotency_key") or ""),
                "status": "completed",
                "success": bool(normalized.get("success", False)),
                "error": str(normalized.get("error") or "")[:500] or None,
                "artifact_refs": artifact_refs[:50],
            },
        )
        state["execution_journal_last_completion"] = {
            "invocation_id": invocation_id,
            "action": dict(intent.get("action") or {}),
            "result": {
                "success": bool(normalized.get("success", False)),
                "error": str(normalized.get("error") or "")[:500] or None,
                "artifacts": artifact_refs[:50],
                "_journal_invocation_id": invocation_id,
            },
            "iteration": int(job.iteration or 0),
            "event_id": entry["event_id"],
        }
        state["execution_journal_pending"] = None
        self._sync_job_summary(job, state)
        await executor.checkpoint_service.save_checkpoint(
            job=job,
            state=state,
            db=db,
            reason="tool_result",
        )
        return entry

    def recover_completed_action(self, *, state: Dict[str, Any]) -> bool:
        """Rehydrate a completed call if the worker died before phase bookkeeping."""
        completion = state.get("execution_journal_last_completion")
        if not isinstance(completion, dict) or not completion:
            return False
        invocation_id = str(completion.get("invocation_id") or "")
        if not invocation_id:
            return False
        actions = (
            state.get("actions_taken")
            if isinstance(state.get("actions_taken"), list)
            else []
        )
        for row in actions:
            result = row.get("result") if isinstance(row, dict) else {}
            if (
                isinstance(result, dict)
                and str(result.get("_journal_invocation_id") or "") == invocation_id
            ):
                return False
        actions.append(
            {
                "action": dict(completion.get("action") or {}),
                "result": dict(completion.get("result") or {}),
                "iteration": int(completion.get("iteration", 0) or 0),
                "node": "recovered_tool_result",
                "journal_recovered": True,
            }
        )
        state["actions_taken"] = actions
        state["execution_journal_recovered_invocation_id"] = invocation_id
        return True

    def reconcile_interrupted(
        self, *, job: Any, state: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Turn an unresolved intent into an explicit operator reconciliation."""
        pending = state.get("execution_journal_pending")
        if not isinstance(pending, dict) or not pending:
            return None
        action = (
            dict(pending.get("action") or {})
            if isinstance(pending.get("action"), dict)
            else {}
        )
        entry = self._append(
            job=job,
            state=state,
            event_type="tool_interrupted",
            payload={
                "invocation_id": str(pending.get("invocation_id") or ""),
                "intent_event_id": str(pending.get("intent_event_id") or ""),
                "tool": str(action.get("tool") or ""),
                "status": "outcome_unknown",
                "retryable_from_journal": bool(
                    pending.get("retryable_from_journal", False)
                ),
            },
        )
        state["execution_journal_pending"] = None
        reconciliation = {
            "checkpoint_type": "execution_reconciliation",
            "checkpoint_id": f"reconcile:{pending.get('invocation_id')}",
            "iteration": int(job.iteration or 0),
            "action": action,
            "message": (
                "The worker stopped after persisting tool intent but before recording "
                "its result. Confirm the external effect before retrying or skip it."
            ),
            "reasons": ["interrupted_tool_outcome_unknown"],
            "journal_event_id": entry["event_id"],
            "invocation_id": str(pending.get("invocation_id") or ""),
            "retryable_from_journal": bool(
                pending.get("retryable_from_journal", False)
            ),
            "requires_action_edit": bool(pending.get("contains_redactions", False)),
        }
        state["execution_reconciliation_pending"] = reconciliation
        state["approval_checkpoint_pending"] = reconciliation
        events = (
            list(state.get("approval_checkpoint_events") or [])
            if isinstance(state.get("approval_checkpoint_events"), list)
            else []
        )
        events.append(reconciliation)
        state["approval_checkpoint_events"] = events[-20:]
        self._sync_job_summary(job, state)
        return reconciliation

    @staticmethod
    def _sync_job_summary(job: Any, state: Dict[str, Any]) -> None:
        results = dict(job.results or {}) if isinstance(job.results, dict) else {}
        cursor = (
            dict(state.get("execution_journal_cursor") or {})
            if isinstance(state.get("execution_journal_cursor"), dict)
            else {}
        )
        results["execution_journal"] = {
            "cursor": cursor,
            "entries_retained": len(state.get("execution_journal") or []),
            "reconciliation_pending": bool(
                state.get("execution_reconciliation_pending")
            ),
        }
        reconciliation = state.get("execution_reconciliation_pending")
        if isinstance(reconciliation, dict) and reconciliation:
            strategy = (
                dict(results.get("execution_strategy") or {})
                if isinstance(results.get("execution_strategy"), dict)
                else {}
            )
            approvals = (
                dict(strategy.get("approval_checkpoints") or {})
                if isinstance(strategy.get("approval_checkpoints"), dict)
                else {}
            )
            approvals["pending"] = reconciliation
            approvals["events"] = (
                state.get("approval_checkpoint_events")
                if isinstance(state.get("approval_checkpoint_events"), list)
                else []
            )[-20:]
            strategy["approval_checkpoints"] = approvals
            results["execution_strategy"] = strategy
            results["approval_checkpoint"] = reconciliation
        job.results = results
        if hasattr(job, "_sa_instance_state"):
            flag_modified(job, "results")


agent_execution_journal_service = AgentExecutionJournalService()
