"""Automatic context compaction for autonomous agent jobs.

The agent-invoked ``compress_history`` tool already lets the model summarize
older iterations into ``state["compressed_history"]`` — but it only runs if
the model remembers to call it. This service applies the same contract
automatically: when the serialized iteration state crosses a size threshold,
older ``actions_taken`` entries are summarized (fast LLM tier) and folded
into ``compressed_history``, keeping only the most recent actions verbatim.

Triggered at the start of the think phase (see ``AgentThinkingService``), so
compacted state feeds the very next prompt build. If the summarization LLM
call fails, a deterministic bullet-list digest is used instead — compaction
never blocks the iteration.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from loguru import logger

from app.core.config import settings

_SUMMARY_MAX_CHARS = 2000

_SUMMARY_SYSTEM_PROMPT = (
    "You are a concise summarizer. Output only the summary, no preamble."
)


class AgentContextCompactionService:
    """Threshold-triggered history compaction (auto ``compress_history``)."""

    def resolve_config(self, job: Any) -> Dict[str, Any]:
        """Resolve compaction settings: job config overrides globals."""
        enabled = bool(settings.AGENT_AUTO_COMPACTION_ENABLED)
        threshold_chars = int(settings.AGENT_AUTO_COMPACTION_THRESHOLD_CHARS)
        keep_recent = int(settings.AGENT_AUTO_COMPACTION_KEEP_RECENT_ACTIONS)
        min_between = int(settings.AGENT_AUTO_COMPACTION_MIN_ITERATIONS_BETWEEN)

        config = job.config if isinstance(getattr(job, "config", None), dict) else {}
        raw = config.get("auto_compaction")
        if isinstance(raw, bool):
            enabled = raw
        elif isinstance(raw, dict):
            enabled = bool(raw.get("enabled", True))
            try:
                threshold_chars = int(raw.get("threshold_chars", threshold_chars))
                keep_recent = int(raw.get("keep_recent_actions", keep_recent))
                min_between = int(raw.get("min_iterations_between", min_between))
            except Exception:
                pass
        return {
            "enabled": enabled,
            "threshold_chars": max(1000, threshold_chars),
            "keep_recent_actions": max(1, min(keep_recent, 20)),
            "min_iterations_between": max(1, min_between),
        }

    @staticmethod
    def estimate_context_chars(state: Dict[str, Any]) -> int:
        """Rough serialized size of the state that feeds prompt assembly."""
        total = 0
        for key in ("actions_taken", "findings", "compressed_history", "memory_context"):
            value = state.get(key)
            if value is None:
                continue
            if isinstance(value, str):
                total += len(value)
                continue
            try:
                total += len(json.dumps(value, default=str))
            except Exception:
                total += len(str(value))
        return total

    async def maybe_compact(
        self,
        executor: Any,
        job: Any,
        state: Dict[str, Any],
        db: Any,
    ) -> bool:
        """Compact older history if the threshold is crossed. Returns True on compact."""
        cfg = self.resolve_config(job)
        if not cfg["enabled"]:
            return False

        actions = state.get("actions_taken")
        if not isinstance(actions, list):
            return False
        keep_recent = cfg["keep_recent_actions"]
        if len(actions) <= keep_recent:
            return False

        last = state.get("auto_compaction_last")
        if isinstance(last, dict):
            last_iteration = last.get("iteration")
            if isinstance(last_iteration, int):
                if int(job.iteration or 0) - last_iteration < cfg["min_iterations_between"]:
                    return False

        estimated = self.estimate_context_chars(state)
        if estimated < cfg["threshold_chars"]:
            return False

        to_compress = actions[:-keep_recent]
        actions_digest = self._build_actions_digest(to_compress)
        existing = str(state.get("compressed_history") or "").strip()

        summary = await self._summarize(
            executor, job, db, actions_digest=actions_digest, existing=existing
        )

        state["compressed_history"] = summary[:_SUMMARY_MAX_CHARS]
        state["actions_taken"] = actions[-keep_recent:]
        state["auto_compaction_last"] = {
            "iteration": int(job.iteration or 0),
            "compacted_actions": len(to_compress),
            "estimated_chars_before": estimated,
            "estimated_chars_after": self.estimate_context_chars(state),
        }
        try:
            job.add_log_entry(
                {
                    "phase": "auto_compaction",
                    "compacted_actions": len(to_compress),
                    "estimated_chars_before": estimated,
                }
            )
        except Exception:
            pass
        logger.info(
            f"Auto-compacted job {getattr(job, 'id', '?')}: "
            f"{len(to_compress)} actions, ~{estimated} chars before"
        )
        return True

    @staticmethod
    def _build_actions_digest(actions: List[Any]) -> str:
        """Deterministic bullet digest of actions (same shape as the tool's)."""
        lines: List[str] = []
        for action in actions:
            if not isinstance(action, dict):
                continue
            act = action.get("action") if isinstance(action.get("action"), dict) else {}
            tool = str(act.get("tool") or "unknown")
            result = action.get("result") if isinstance(action.get("result"), dict) else {}
            if result.get("success"):
                data = result.get("data")
                data_keys = list(data.keys()) if isinstance(data, dict) else []
                outcome = f"success, data keys: {data_keys}"
            elif result:
                outcome = f"failed: {str(result.get('error', ''))[:100]}"
            else:
                outcome = "no result"
            lines.append(f"- Iteration {action.get('iteration', '?')}: {tool} → {outcome}")
        return "\n".join(lines)

    async def _summarize(
        self,
        executor: Any,
        job: Any,
        db: Any,
        *,
        actions_digest: str,
        existing: str,
    ) -> str:
        """LLM summary on the fast tier; deterministic digest on failure."""
        prompt = (
            "Summarize the following agent action history into a concise "
            "narrative (max 500 words).\n"
            "Focus on: what was discovered, what worked/failed, key decisions "
            "made, and current trajectory.\n\n"
        )
        if existing:
            prompt += f"Previous compressed history:\n{existing}\n\n"
        prompt += (
            f"New actions to compress:\n{actions_digest}\n\n"
            "Write a concise summary in past tense."
        )

        try:
            user_settings = None
            get_settings = getattr(executor, "_get_user_settings", None)
            if callable(get_settings):
                user_settings = await get_settings(job.user_id, db)

            routing: Optional[Dict[str, Any]] = None
            get_routing = getattr(executor, "_llm_routing_from_job_config", None)
            if callable(get_routing):
                routing = get_routing(getattr(job, "config", None))
            routing = dict(routing) if isinstance(routing, dict) else {}
            routing["llm_tier"] = "fast"

            response = await executor.llm_service.generate_response(
                system_prompt=_SUMMARY_SYSTEM_PROMPT,
                user_message=prompt,
                user_settings=user_settings,
                routing=routing,
                task_type="summarization",
                user_id=getattr(job, "user_id", None),
                db=db,
                snapshot_context={
                    "job_id": str(getattr(job, "id", "") or "") or None,
                    "iteration": int(getattr(job, "iteration", 0) or 0),
                    "phase": "compaction",
                },
            )
            summary = str(response or "").strip()
            if summary:
                return summary
        except Exception as exc:
            logger.warning(f"Auto-compaction summary LLM call failed: {exc}")

        # Deterministic fallback: keep the digest itself (with prior summary)
        # so information survives even without an LLM summary.
        fallback = actions_digest
        if existing:
            fallback = f"{existing}\n\nEarlier actions (digest):\n{actions_digest}"
        return fallback


context_compaction_service = AgentContextCompactionService()
