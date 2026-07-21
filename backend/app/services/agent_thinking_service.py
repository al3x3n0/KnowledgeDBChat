"""Thinking phase helpers extracted from AutonomousAgentExecutor."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from loguru import logger


class AgentThinkingService:
    """Reason about next actions for the autonomous runtime."""

    async def think(
        self,
        executor: Any,
        job: Any,
        agent_def: Optional[Any],
        state: Dict[str, Any],
        observation: Dict[str, Any],
        user_settings: Optional[Any],
        db: Any,
    ) -> Dict[str, Any]:
        """Decide the next action based on goal and current state."""
        # Automatic context compaction runs before prompt assembly so a
        # compacted history feeds this very iteration's prompts.
        try:
            from app.services.agent_context_compaction import context_compaction_service

            await context_compaction_service.maybe_compact(executor, job, state, db)
        except Exception as exc:
            logger.debug(f"Auto-compaction skipped: {exc}")

        profile = (
            state.get("skill_profile")
            if isinstance(state.get("skill_profile"), dict)
            else executor._resolve_agent_skill_profile(job, state=state)
        )
        # Cache-friendly prompt split: the system prompt carries only the
        # byte-stable per-job prefix (so provider prompt caches hit across
        # iterations); all per-iteration context rides in the user message.
        build_stable = getattr(executor, "_build_thinking_prompt_stable", None)
        build_volatile = getattr(executor, "_build_thinking_prompt_volatile", None)
        volatile_context = ""
        if callable(build_stable) and callable(build_volatile):
            system_prompt = build_stable(job, agent_def, state, profile=profile)
            volatile_context = build_volatile(job, state) or ""
        else:
            system_prompt = executor._build_thinking_prompt(job, agent_def, state, observation, profile=profile)
        available_tools = executor._get_tools_for_job_type(job.job_type, job.config, profile=profile)

        user_message = f"""
{volatile_context}

Current iteration: {job.iteration}/{job.max_iterations}
Current progress: {state.get('goal_progress', 0)}%
Tool calls used: {job.tool_calls_used}/{job.max_tool_calls}
LLM calls used: {job.llm_calls_used}/{job.max_llm_calls}

Recent actions: {self._clip(json.dumps(state.get('actions_taken', [])[-3:], default=str), 4000)}

Current observation:
{self._clip(json.dumps(observation, default=str), 6000)}

Total findings so far: {len(state.get('findings', []))}

Based on the goal and current progress, decide:
1. Is the goal achieved? If so, explain why.
2. Should we stop for another reason? (e.g., no more progress possible)
3. If continuing, what is the next action to take?

Respond in JSON format:
{{
    "goal_achieved": true/false,
    "should_stop": true/false,
    "stop_reason": "reason if stopping",
    "reasoning": "your reasoning about current progress and next steps",
    "assessment": "assessment of goal completion (0-100%)",
    "action": {{
        "tool": "tool_name",
        "params": {{}},
        "purpose": "why this action"
    }} or null if stopping
}}
"""

        routing = executor._llm_routing_from_job_config(job.config)

        snapshot_context = {
            "job_id": str(getattr(job, "id", "") or "") or None,
            "iteration": int(job.iteration or 0),
            "phase": "thinking",
        }

        try:
            response: Optional[str] = None
            try:
                response = await self._maybe_run_native_tool_loop(
                    executor,
                    job,
                    state,
                    system_prompt=system_prompt,
                    user_message=user_message,
                    available_tools=available_tools,
                    user_settings=user_settings,
                    routing=routing,
                    db=db,
                )
            except Exception as exc:
                logger.warning(
                    f"Native tool loop failed; falling back to standard think: {exc}"
                )
            if not response:
                response = await self._generate_decision_text(
                    executor,
                    system_prompt=system_prompt,
                    user_message=user_message,
                    user_settings=user_settings,
                    routing=routing,
                    snapshot_context=snapshot_context,
                )

            decision = await executor.decision_parser.parse_with_retry(
                raw_response=str(response or ""),
                available_tools=available_tools,
                job=job,
                state=state,
                user_settings=user_settings,
                system_prompt=system_prompt,
                user_message=user_message,
                routing=routing,
            )

            state["decision_parse_metrics"] = executor.decision_parser.metrics

            if decision is not None:
                result = decision.model_dump()
                if isinstance(result.get("action"), dict):
                    result["action"] = executor._apply_default_scope_to_action(result["action"], job)
                if result.get("action") is None and not result.get("goal_achieved") and not result.get("should_stop"):
                    recovery = executor._build_recovery_action(job, state)
                    if recovery:
                        result["action"] = recovery
                        result["reasoning"] = (
                            f"{str(result.get('reasoning') or '').strip()[:360]} "
                            "Auto-selected recovery action."
                        ).strip()
                    else:
                        result["should_stop"] = True
                        result["stop_reason"] = "No valid action available for continuation"
                return result

            return self.parse_decision_response(
                executor,
                raw_response=response,
                job=job,
                state=state,
                available_tools=available_tools,
            )
        except Exception as exc:
            logger.error(f"Error in thinking phase: {exc}")
            recovery_action = executor._build_recovery_action(job, state)
            return {
                "goal_achieved": False,
                "should_stop": recovery_action is None,
                "stop_reason": f"Thinking error: {exc}" if recovery_action is None else "",
                "reasoning": str(exc),
                "action": recovery_action,
            }

    @staticmethod
    def _clip(text: str, max_chars: int) -> str:
        """Hard cap for serialized prompt payloads (huge tool results etc.)."""
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + " ... [truncated]"

    def _resolve_native_loop_config(self, job: Any) -> Dict[str, Any]:
        """Resolve native tool-loop settings: job config overrides globals."""
        from app.core.config import settings

        enabled = bool(settings.AGENT_NATIVE_TOOL_LOOP_ENABLED)
        max_tool_calls = int(settings.AGENT_NATIVE_TOOL_LOOP_MAX_TOOL_CALLS)
        max_llm_calls = int(settings.AGENT_NATIVE_TOOL_LOOP_MAX_LLM_CALLS)

        config = job.config if isinstance(getattr(job, "config", None), dict) else {}
        raw = config.get("native_tool_loop")
        if isinstance(raw, bool):
            enabled = raw
        elif isinstance(raw, dict):
            enabled = bool(raw.get("enabled", True))
            try:
                max_tool_calls = int(raw.get("max_tool_calls", max_tool_calls))
                max_llm_calls = int(raw.get("max_llm_calls", max_llm_calls))
            except Exception:
                pass
        return {
            "enabled": enabled,
            "max_tool_calls": max(1, max_tool_calls),
            "max_llm_calls": max(1, max_llm_calls),
        }

    async def _maybe_run_native_tool_loop(
        self,
        executor: Any,
        job: Any,
        state: Dict[str, Any],
        *,
        system_prompt: str,
        user_message: str,
        available_tools: List[str],
        user_settings: Optional[Any],
        routing: Optional[Dict[str, Any]],
        db: Any,
    ) -> Optional[str]:
        """Run the native tool-calling loop if enabled; return decision text.

        Returns ``None`` when the loop is disabled, has no budget/tools, or
        produced nothing — the caller then uses the standard prompted path.
        Gated tools (approval checkpoints, dangerous tools) are not executed
        here; they come back as the decision's action so the act phase applies
        its normal approval machinery.
        """
        cfg = self._resolve_native_loop_config(job)
        if not cfg["enabled"]:
            return None

        remaining_tools = int(job.max_tool_calls or 0) - int(job.tool_calls_used or 0)
        remaining_llm = int(job.max_llm_calls or 0) - int(job.llm_calls_used or 0)
        # Leave one tool slot for the act phase and cap by remaining budgets.
        max_tool_calls = min(cfg["max_tool_calls"], remaining_tools - 1)
        max_llm_calls = min(cfg["max_llm_calls"], max(1, remaining_llm))
        if max_tool_calls <= 0:
            return None

        from app.services.agent_tools import get_tool_by_name

        tool_defs = [t for t in (get_tool_by_name(n) for n in available_tools) if t]
        if not tool_defs:
            return None

        from app.core.config import settings as app_settings

        dangerous = set(app_settings.AGENT_DANGEROUS_TOOLS or [])

        def _should_defer(action: Dict[str, Any]) -> bool:
            tool = str(action.get("tool") or "")
            if tool in dangerous:
                return True
            try:
                gate = executor._evaluate_approval_checkpoint(job, state, action)
                return bool(gate.get("required"))
            except Exception:
                return False

        async def _execute(name: str, params: Dict[str, Any]) -> Any:
            action = {
                "tool": name,
                "params": params or {},
                "purpose": "Native tool loop information gathering",
            }
            try:
                action = executor._apply_default_scope_to_action(action, job)
            except Exception:
                pass
            result = await executor.action_service.act(executor, job, action, state, db)
            if not isinstance(result, dict):
                result = {"success": True, "data": result}
            job.tool_calls_used = int(job.tool_calls_used or 0) + 1
            actions_taken = state.get("actions_taken")
            if isinstance(actions_taken, list):
                actions_taken.append(
                    {
                        "action": action,
                        "result": result,
                        "iteration": job.iteration,
                        "node": "native_think",
                    }
                )
            if result.get("findings"):
                state.setdefault("findings", []).extend(result["findings"])
            if result.get("artifacts"):
                state.setdefault("artifacts", []).extend(result["artifacts"])
            return result

        from app.services.agent_decision_parser import AgentDecision
        from app.services.agent_native_tool_loop import native_tool_loop_service

        native_user_message = (
            f"{user_message}\n\n"
            "You may call the available tools directly to gather information "
            "before deciding. When you have enough information, stop calling "
            "tools and respond with the final JSON decision."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": native_user_message},
        ]

        loop_result = await native_tool_loop_service.run(
            llm_service=executor.llm_service,
            messages=messages,
            tools=tool_defs,
            execute_tool=_execute,
            should_defer=_should_defer,
            final_response_schema=AgentDecision.model_json_schema(),
            user_settings=user_settings,
            routing=routing,
            user_id=getattr(job, "user_id", None),
            db=db,
            max_tool_calls=max_tool_calls,
            max_llm_calls=max_llm_calls,
            snapshot_context={
                "job_id": str(getattr(job, "id", "") or "") or None,
                "iteration": int(job.iteration or 0),
                "phase": "native_tool_loop",
            },
        )

        # The runtime adapter adds one llm call after think; account for extras.
        job.llm_calls_used = int(job.llm_calls_used or 0) + max(
            0, loop_result.llm_calls_used - 1
        )
        state["native_tool_loop_last"] = {
            "iteration": int(job.iteration or 0),
            "stop_reason": loop_result.stop_reason,
            "tool_calls": loop_result.tool_calls_executed,
            "llm_calls": loop_result.llm_calls_used,
            "steps": loop_result.steps[-10:],
        }

        if loop_result.pending_action is not None:
            return json.dumps(
                {
                    "goal_achieved": False,
                    "should_stop": False,
                    "stop_reason": "",
                    "reasoning": (
                        "Native tool loop deferred a gated tool call for the "
                        "act phase's approval machinery."
                    ),
                    "action": {
                        **loop_result.pending_action,
                        "purpose": "Deferred from native tool loop for approval",
                    },
                },
                default=str,
            )

        if loop_result.structured is not None:
            return json.dumps(loop_result.structured, default=str)
        return loop_result.final_text or None

    async def _generate_decision_text(
        self,
        executor: Any,
        *,
        system_prompt: str,
        user_message: str,
        user_settings: Optional[Any],
        routing: Optional[Dict[str, Any]],
        snapshot_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Produce the raw decision text for parsing.

        Prefers the native structured-output path (schema-constrained JSON via
        `LLMService.generate_structured`), which eliminates most malformed-JSON
        retries. Falls back to the legacy prompted-text path when the
        structured path is unavailable or fails.
        """
        try:
            from app.services.agent_decision_parser import AgentDecision

            completion = await executor.llm_service.generate_structured(
                system_prompt=system_prompt,
                user_message=user_message,
                response_schema=AgentDecision.model_json_schema(),
                user_settings=user_settings,
                routing=routing,
                snapshot_context=snapshot_context,
            )
            if completion is not None:
                if getattr(completion, "structured", None) is not None:
                    return json.dumps(completion.structured, default=str)
                text = str(getattr(completion, "text", "") or "")
                if text:
                    return text
        except Exception as exc:
            logger.debug(f"Structured decision path unavailable, using prompted text: {exc}")

        response = await executor.llm_service.generate_response(
            system_prompt=system_prompt,
            user_message=user_message,
            user_settings=user_settings,
            routing=routing,
            snapshot_context=snapshot_context,
        )
        return str(response or "")

    def parse_decision_response(
        self,
        executor: Any,
        *,
        raw_response: Any,
        job: Any,
        state: Dict[str, Any],
        available_tools: List[str],
    ) -> Dict[str, Any]:
        """Parse and normalize LLM decision payload with resilient JSON extraction."""
        text = str(raw_response or "")
        payload = self._extract_first_json_object(text)
        if not isinstance(payload, dict):
            recovery = executor._build_recovery_action(job, state)
            return {
                "goal_achieved": False,
                "should_stop": recovery is None,
                "stop_reason": "Model response did not contain a valid JSON decision" if recovery is None else "",
                "reasoning": text[:500] if text else "Model returned an empty decision",
                "assessment": None,
                "action": recovery,
            }

        goal_achieved = self._coerce_bool(payload.get("goal_achieved"), default=False)
        should_stop = self._coerce_bool(payload.get("should_stop"), default=False)
        reasoning = str(payload.get("reasoning") or "").strip()
        stop_reason = str(payload.get("stop_reason") or "").strip()
        assessment = payload.get("assessment")

        action = self._normalize_decision_action(payload.get("action"), available_tools)
        if isinstance(action, dict):
            action = executor._apply_default_scope_to_action(action, job)
        if action is None and not goal_achieved and not should_stop:
            action = executor._build_recovery_action(job, state)
            if action:
                reasoning = f"{reasoning[:360]} Auto-selected deterministic recovery action.".strip()
            else:
                should_stop = True
                stop_reason = stop_reason or "No valid action available for continuation"

        if should_stop and not stop_reason:
            stop_reason = "Model requested stop"

        return {
            "goal_achieved": goal_achieved,
            "should_stop": should_stop,
            "stop_reason": stop_reason,
            "reasoning": reasoning[:800] if reasoning else (text[:500] if text else ""),
            "assessment": assessment,
            "action": action,
        }

    def _extract_first_json_object(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract the first valid JSON object from plain text or fenced markdown."""
        if not text:
            return None

        stripped = text.strip()
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
        if fence_match:
            fenced = fence_match.group(1).strip()
            try:
                parsed = json.loads(fenced)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass

        for start in [i for i, ch in enumerate(text) if ch == "{"]:
            depth = 0
            in_string = False
            escaped = False
            for idx in range(start, len(text)):
                ch = text[idx]
                if in_string:
                    if escaped:
                        escaped = False
                    elif ch == "\\":
                        escaped = True
                    elif ch == '"':
                        in_string = False
                    continue

                if ch == '"':
                    in_string = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = text[start : idx + 1]
                        try:
                            parsed = json.loads(candidate)
                            if isinstance(parsed, dict):
                                return parsed
                        except Exception:
                            break
        return None

    def _normalize_decision_action(
        self,
        action: Any,
        available_tools: List[str],
    ) -> Optional[Dict[str, Any]]:
        """Normalize action payload and reject unavailable tools."""
        if action is None:
            return None
        if isinstance(action, str):
            action = {"tool": action, "params": {}}
        if not isinstance(action, dict):
            return None

        tool = str(action.get("tool") or "").strip()
        if not tool or tool not in set(available_tools):
            return None

        params = action.get("params")
        if not isinstance(params, dict):
            params = {}

        purpose = str(action.get("purpose") or "").strip()
        return {"tool": tool, "params": params, "purpose": purpose[:300]}

    def _coerce_bool(self, value: Any, default: bool = False) -> bool:
        """Coerce flexible model outputs to booleans."""
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "yes", "1", "y"}:
                return True
            if lowered in {"false", "no", "0", "n"}:
                return False
        return default
