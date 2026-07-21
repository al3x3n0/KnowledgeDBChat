"""Native tool-calling loop for agent phases.

Runs a bounded multi-turn conversation in which the LLM invokes tools through
the provider's native tool-calling API (``LLMService.generate_structured``)
instead of emitting one prompted-JSON action per executor iteration. Tool
execution is delegated to a caller-supplied callback so the existing dispatch
stack (``AgentActionService`` → ``AgentToolRegistry`` → policy/audit) stays in
the path.

Safety contract:
- Every tool call the model emits receives a tool-result message (real result,
  placeholder, or budget notice) so provider conversation invariants hold.
- Tools the caller wants gated (approval checkpoints, dangerous tools) are
  *deferred*: the loop stops and returns the call as ``pending_action`` for
  the caller to route through its normal approval machinery.
- Hard caps on LLM rounds and tool executions; repeated identical calls are
  answered with a placeholder instead of re-executed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional

from loguru import logger

ExecuteTool = Callable[[str, Dict[str, Any]], Awaitable[Any]]
ShouldDefer = Callable[[Dict[str, Any]], bool]

_REPEAT_PLACEHOLDER = (
    "This exact tool call was already executed in this loop; the result has "
    "not changed. Use the earlier result or choose a different call."
)
_BUDGET_PLACEHOLDER = (
    "Tool budget for this loop is exhausted. Do not call more tools; produce "
    "your final response now."
)


@dataclass
class NativeToolLoopResult:
    final_text: str = ""
    structured: Optional[Dict[str, Any]] = None
    pending_action: Optional[Dict[str, Any]] = None
    steps: List[Dict[str, Any]] = field(default_factory=list)
    tool_calls_executed: int = 0
    llm_calls_used: int = 0
    stop_reason: str = "completed"


class NativeToolLoopService:
    """Bounded native tool-calling loop over ``LLMService.generate_structured``."""

    async def run(
        self,
        *,
        llm_service: Any,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        execute_tool: ExecuteTool,
        should_defer: Optional[ShouldDefer] = None,
        final_response_schema: Optional[Dict[str, Any]] = None,
        user_settings: Any = None,
        routing: Optional[Dict[str, Any]] = None,
        task_type: str = "chat",
        user_id: Any = None,
        db: Any = None,
        max_tool_calls: int = 5,
        max_llm_calls: int = 6,
        max_result_chars: int = 4000,
        max_repeated_calls: int = 3,
        snapshot_context: Optional[Dict[str, Any]] = None,
    ) -> NativeToolLoopResult:
        result = NativeToolLoopResult()
        convo: List[Dict[str, Any]] = list(messages)
        executed_signatures: Dict[str, int] = {}
        repeated_total = 0
        budget_exhausted = False

        async def _llm(
            *, with_tools: bool, response_schema: Optional[Dict[str, Any]] = None
        ):
            completion = await llm_service.generate_structured(
                messages=convo,
                tools=tools if with_tools else None,
                response_schema=response_schema,
                user_settings=user_settings,
                routing=routing,
                task_type=task_type,
                user_id=user_id,
                db=db,
                snapshot_context=snapshot_context,
            )
            result.llm_calls_used += 1
            return completion

        while True:
            if result.llm_calls_used >= max_llm_calls:
                result.stop_reason = "max_llm_calls"
                break

            completion = await _llm(with_tools=True)

            if not completion.tool_calls:
                result.final_text = completion.text
                result.structured = completion.structured
                result.stop_reason = "completed"
                break

            convo.append(
                {
                    "role": "assistant",
                    "content": completion.text or "",
                    "tool_calls": [
                        {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                        for tc in completion.tool_calls
                    ],
                }
            )

            for tc in completion.tool_calls:
                action = {"tool": tc.name, "params": tc.arguments or {}}

                if should_defer is not None and should_defer(dict(action)):
                    result.pending_action = action
                    result.stop_reason = "deferred_action"
                    return result

                if result.tool_calls_executed >= max_tool_calls:
                    budget_exhausted = True
                    self._append_tool_message(convo, tc, _BUDGET_PLACEHOLDER)
                    continue

                signature = self._signature(tc.name, tc.arguments)
                if executed_signatures.get(signature):
                    repeated_total += 1
                    self._append_tool_message(convo, tc, _REPEAT_PLACEHOLDER)
                    continue

                try:
                    tool_result = await execute_tool(tc.name, tc.arguments or {})
                except Exception as exc:  # noqa: BLE001 - tool errors feed the model
                    logger.warning(f"Native tool loop: '{tc.name}' raised: {exc}")
                    tool_result = {"success": False, "error": str(exc)[:500]}

                executed_signatures[signature] = 1
                result.tool_calls_executed += 1
                success = (
                    bool(tool_result.get("success", True))
                    if isinstance(tool_result, dict)
                    else True
                )
                result.steps.append(
                    {
                        "tool": tc.name,
                        "params": tc.arguments or {},
                        "success": success,
                        "error": (
                            str(tool_result.get("error") or "")[:260]
                            if isinstance(tool_result, dict) and tool_result.get("error")
                            else None
                        ),
                    }
                )
                self._append_tool_message(
                    convo, tc, self._serialize_result(tool_result, max_result_chars)
                )

            if budget_exhausted:
                result.stop_reason = "max_tool_calls"
                break
            if repeated_total >= max_repeated_calls:
                result.stop_reason = "repeated_tool_calls"
                break

        # Finalize: guarantee a schema-shaped answer when one was requested and
        # the loop ended without one (budget stop, repeats, or free-text final).
        if (
            final_response_schema is not None
            and result.structured is None
            and result.llm_calls_used < max_llm_calls + 1
        ):
            convo.append(
                {
                    "role": "user",
                    "content": (
                        "Tool use is finished. Produce your final response now "
                        "as a single JSON object matching the required schema."
                    ),
                }
            )
            try:
                completion = await _llm(
                    with_tools=False, response_schema=final_response_schema
                )
                result.structured = completion.structured
                if completion.text:
                    result.final_text = completion.text
            except Exception as exc:  # noqa: BLE001 - caller falls back on empty
                logger.warning(f"Native tool loop finalization failed: {exc}")

        if result.structured is not None and not result.final_text:
            result.final_text = json.dumps(result.structured, default=str)
        return result

    @staticmethod
    def _append_tool_message(convo: List[Dict[str, Any]], tc: Any, content: str) -> None:
        convo.append(
            {
                "role": "tool",
                "tool_call_id": tc.id,
                "name": tc.name,
                "content": content,
            }
        )

    @staticmethod
    def _signature(name: str, arguments: Optional[Dict[str, Any]]) -> str:
        try:
            args = json.dumps(arguments or {}, sort_keys=True, default=str)
        except Exception:
            args = str(arguments)
        return f"{name}:{args}"

    @staticmethod
    def _serialize_result(tool_result: Any, max_chars: int) -> str:
        if isinstance(tool_result, str):
            payload = tool_result
        else:
            try:
                payload = json.dumps(tool_result, default=str)
            except Exception:
                payload = str(tool_result)
        if len(payload) > max_chars:
            payload = payload[:max_chars] + " ... [truncated]"
        return payload


native_tool_loop_service = NativeToolLoopService()
