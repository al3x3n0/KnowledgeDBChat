"""
Agent Decision Parser.

Robust parsing, validation, retry, and repair of LLM decision responses
for the autonomous agent executor. Extracts structured decisions from
free-form LLM output using Pydantic validation with graceful fallbacks.
"""

import json
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class AgentActionDecision(BaseModel):
    """Validated action payload from an LLM decision."""

    tool: str = Field(..., min_length=1)
    params: Dict[str, Any] = Field(default_factory=dict)
    purpose: str = Field(default="", max_length=300)

    @field_validator("params", mode="before")
    @classmethod
    def coerce_params(cls, v: Any) -> Dict[str, Any]:
        if isinstance(v, dict):
            return v
        return {}

    @field_validator("purpose", mode="before")
    @classmethod
    def truncate_purpose(cls, v: Any) -> str:
        return str(v or "")[:300]


class AgentDecision(BaseModel):
    """Validated decision response from the LLM thinking phase."""

    goal_achieved: bool = False
    should_stop: bool = False
    stop_reason: str = ""
    reasoning: str = ""
    assessment: Optional[Any] = None
    action: Optional[AgentActionDecision] = None

    @field_validator("goal_achieved", "should_stop", mode="before")
    @classmethod
    def coerce_bool(cls, v: Any) -> bool:
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return bool(v)
        if isinstance(v, str):
            lowered = v.strip().lower()
            if lowered in {"true", "yes", "1", "y"}:
                return True
            if lowered in {"false", "no", "0", "n"}:
                return False
        return False

    @field_validator("reasoning", mode="before")
    @classmethod
    def truncate_reasoning(cls, v: Any) -> str:
        return str(v or "").strip()[:800]

    @field_validator("stop_reason", mode="before")
    @classmethod
    def truncate_stop_reason(cls, v: Any) -> str:
        return str(v or "").strip()[:400]

    @field_validator("action", mode="before")
    @classmethod
    def coerce_action(cls, v: Any) -> Optional[Dict[str, Any]]:
        if v is None:
            return None
        if isinstance(v, AgentActionDecision):
            return v.model_dump()
        if isinstance(v, str):
            return {"tool": v, "params": {}}
        if isinstance(v, dict):
            return v
        return None


# ---------------------------------------------------------------------------
# JSON extraction helpers (extracted from executor)
# ---------------------------------------------------------------------------

def extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
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

    fence_match = re.search(
        r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL
    )
    if fence_match:
        fenced = fence_match.group(1).strip()
        try:
            parsed = json.loads(fenced)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    # Balanced-brace extraction for responses with commentary before/after JSON.
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
        # Only try the first opening brace that doesn't parse; skip the rest
        # to avoid wasting time on large texts.
        if depth != 0:
            continue
    return None


# ---------------------------------------------------------------------------
# Decision JSON schema for prompt embedding
# ---------------------------------------------------------------------------

DECISION_JSON_SCHEMA = """\
You MUST respond with a single JSON object (no other text) matching this schema:
{
    "goal_achieved": <boolean>,
    "should_stop": <boolean>,
    "stop_reason": "<string, required if should_stop is true>",
    "reasoning": "<string, your reasoning about current progress and next steps>",
    "assessment": "<0-100, percentage of goal completion>",
    "action": {
        "tool": "<tool_name from the available tools list>",
        "params": { <tool-specific parameters> },
        "purpose": "<why this action advances the goal>"
    }
}
Set "action" to null if goal_achieved or should_stop is true."""


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class AgentDecisionParser:
    """Parse, validate, retry, and repair LLM decision responses."""

    def __init__(self, llm_service: Any):
        self.llm_service = llm_service
        self._metrics: Dict[str, int] = {
            "parse_success": 0,
            "parse_retry": 0,
            "parse_repair": 0,
            "parse_failure": 0,
        }

    @property
    def metrics(self) -> Dict[str, int]:
        return dict(self._metrics)

    def reset_metrics(self) -> None:
        for k in self._metrics:
            self._metrics[k] = 0

    @staticmethod
    def get_schema_for_prompt() -> str:
        """Return the JSON schema description to embed in system prompts."""
        return DECISION_JSON_SCHEMA

    # ------------------------------------------------------------------
    # Core parse
    # ------------------------------------------------------------------

    def parse(
        self,
        raw_response: str,
        available_tools: List[str],
    ) -> Tuple[Optional[AgentDecision], str]:
        """
        Attempt to parse raw LLM text into AgentDecision.

        Returns ``(decision, "")`` on success or ``(None, error_message)``
        on failure.
        """
        payload = extract_first_json_object(raw_response)
        if not isinstance(payload, dict):
            return None, "No valid JSON object found in response"

        try:
            decision = AgentDecision.model_validate(payload)
        except Exception as exc:
            return None, f"JSON schema validation failed: {exc}"

        decision = self.validate_action_tool(decision, available_tools)
        self._metrics["parse_success"] += 1
        return decision, ""

    # ------------------------------------------------------------------
    # Parse with retry + repair
    # ------------------------------------------------------------------

    async def parse_with_retry(
        self,
        raw_response: str,
        available_tools: List[str],
        job: Any,
        state: Dict[str, Any],
        user_settings: Any,
        system_prompt: str,
        user_message: str,
        routing: Optional[Dict[str, Any]] = None,
        max_retries: int = 2,
    ) -> Optional[AgentDecision]:
        """
        Parse with up to *max_retries* attempts.

        Retry 1: re-prompt the LLM with an explicit correction message.
        Retry 2: attempt LLM-assisted JSON repair on the broken text.
        Final fallback: returns ``None`` (caller should build recovery action).
        """
        # --- Attempt 0: direct parse ---
        decision, error = self.parse(raw_response, available_tools)
        if decision is not None:
            return decision

        logger.warning(f"Decision parse failed (attempt 0): {error}")

        for attempt in range(1, max_retries + 1):
            self._metrics["parse_retry"] += 1

            if attempt == 1:
                # Retry 1: re-prompt with correction
                correction_msg = (
                    f"Your previous response was not valid JSON. {error}\n\n"
                    f"Please respond with ONLY a JSON object matching this exact schema:\n"
                    f"{DECISION_JSON_SCHEMA}\n\n"
                    f"Original request:\n{user_message}"
                )
                try:
                    response = await self.llm_service.generate_response(
                        system_prompt=system_prompt,
                        user_message=correction_msg,
                        user_settings=user_settings,
                        routing=routing,
                    )
                    decision, error = self.parse(str(response or ""), available_tools)
                    if decision is not None:
                        return decision
                    logger.warning(f"Decision parse failed (retry {attempt}): {error}")
                except Exception as exc:
                    logger.error(f"LLM retry call failed: {exc}")

            elif attempt == 2:
                # Retry 2: LLM-assisted JSON repair
                repaired = await self.repair_json(
                    raw_response, error, user_settings, routing
                )
                if repaired:
                    decision, error = self.parse(repaired, available_tools)
                    if decision is not None:
                        return decision
                    logger.warning(f"Decision parse failed after repair: {error}")

        self._metrics["parse_failure"] += 1
        return None

    # ------------------------------------------------------------------
    # JSON repair via fast LLM
    # ------------------------------------------------------------------

    async def repair_json(
        self,
        broken_text: str,
        error_message: str,
        user_settings: Any,
        routing: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Use a fast LLM call to fix almost-valid JSON.

        Returns the repaired JSON string or ``None`` on failure.
        """
        self._metrics["parse_repair"] += 1
        # Prefer fast tier for repair
        repair_routing = dict(routing or {})
        repair_routing["llm_tier"] = "fast"

        snippet = broken_text[:2000]
        repair_prompt = (
            "Fix the following malformed JSON. The error was:\n"
            f"{error_message}\n\n"
            "Return ONLY the corrected JSON object, nothing else.\n\n"
            f"{snippet}"
        )
        try:
            response = await self.llm_service.generate_response(
                system_prompt="You are a JSON repair assistant. Return only valid JSON.",
                user_message=repair_prompt,
                user_settings=user_settings,
                routing=repair_routing,
            )
            return str(response or "")
        except Exception as exc:
            logger.error(f"JSON repair LLM call failed: {exc}")
            return None

    # ------------------------------------------------------------------
    # Tool validation
    # ------------------------------------------------------------------

    @staticmethod
    def validate_action_tool(
        decision: AgentDecision,
        available_tools: List[str],
    ) -> AgentDecision:
        """Clear action if tool is not in *available_tools*."""
        if decision.action is None:
            return decision
        tools_set = set(available_tools)
        if decision.action.tool not in tools_set:
            logger.info(
                f"Clearing unavailable tool '{decision.action.tool}' from decision"
            )
            decision.action = None
        return decision
