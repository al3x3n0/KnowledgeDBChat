"""Drive one DeepSeek Harness session and report it as JSON lines.

Runs under the harness's OWN interpreter (`AGENT_DEEPSEEK_HARNESS_PYTHON`),
never imported by the app: the SDK requires pydantic>=2.12 and the backend
pins an older one, so the two cannot share a virtualenv. Standard library and
the SDK only -- importing anything from `app` here would fail there.

Input, one JSON object on stdin:
    model, reasoning_effort, max_tokens, base_url, patch, dsh_home, cwd,
    prompt, system_prompt, timeout_seconds

Output, one JSON object per line on stdout:
    {"event": "assistant", "usage": {...}}     each model turn, as it happens
    {"event": "result", ...}                   once, at the end
    {"event": "error", "error": "..."}         instead of a result

The API key arrives in the environment (DEEPSEEK_API_KEY), which the caller
has already reduced to an allowlist; the SDK copies that environment into the
runtime it starts.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Dict


def _emit(payload: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, default=str) + "\n")
    sys.stdout.flush()


def _usage(event: Dict[str, Any]) -> Dict[str, Any]:
    data = event.get("data") if isinstance(event.get("data"), dict) else {}
    usage = data.get("usage")
    return usage if isinstance(usage, dict) else {}


def main() -> int:
    try:
        spec = json.loads(sys.stdin.read() or "{}")
        from deepseek_harness import DeepSeekHarness, DeepSeekHarnessConfig
    except Exception as exc:  # noqa: BLE001 - reported to the caller
        _emit({"event": "error", "error": f"{type(exc).__name__}: {exc}"})
        return 2

    totals: Dict[str, float] = {}
    counts = {"assistant_messages": 0, "tool_calls": 0, "turns": 0}
    # Why each turn ended, and any event that reports an error. A session
    # once ended with finish_reason "error" and nothing else to go on.
    reasons: list = []
    errors: list = []

    def on_notification(notification: Any) -> None:
        if getattr(notification, "method", "") != "session.event":
            return
        event = (notification.payload or {}).get("event")
        if not isinstance(event, dict):
            return
        kind = event.get("type")
        if kind == "assistant/message":
            usage = _usage(event)
            counts["assistant_messages"] += 1
            for key, value in usage.items():
                if isinstance(value, (int, float)):
                    totals[key] = totals.get(key, 0) + value
            _emit({"event": "assistant", "usage": usage})
        elif kind == "tool/call":
            counts["tool_calls"] += 1
        elif kind == "turn/end":
            counts["turns"] += 1
            data = event.get("data") if isinstance(event.get("data"), dict) else {}
            reasons.append(data.get("reason"))
        if "error" in str(kind) or (
            isinstance(event.get("data"), dict) and "error" in event["data"]
        ):
            errors.append(
                {
                    "type": kind,
                    "data": json.dumps(event.get("data"), default=str)[:1500],
                }
            )

    config = DeepSeekHarnessConfig(
        model=spec.get("model") or "deepseek-v4-pro",
        reasoning_effort=spec.get("reasoning_effort") or None,
        max_tokens=spec.get("max_tokens") or None,
        base_url=spec.get("base_url") or None,
        profile="sdk-minimal",
        patches=(spec["patch"],),
        dsh_home=spec["dsh_home"],
        cwd=spec["cwd"],
        env={
            "DSH_TELEMETRY_MODE": "DISABLED",
            "DSH_SYSTEM_PROMPT": spec.get("system_prompt") or "",
        },
        request_timeout_seconds=spec.get("timeout_seconds") or None,
    )
    try:
        with DeepSeekHarness(config) as harness:
            result = harness.run(spec["prompt"], on_notification=on_notification)
    except Exception as exc:  # noqa: BLE001 - reported to the caller
        _emit(
            {
                "event": "error",
                "error": f"{type(exc).__name__}: {exc}"[:2000],
                "counts": counts,
                "usage": totals,
                "turn_end_reasons": reasons[-5:],
                "errors": errors[-5:],
            }
        )
        return 1
    _emit(
        {
            "event": "result",
            "final_response": (result.final_response or "")[:8000],
            "finish_reason": result.finish_reason,
            "counts": counts,
            "usage": totals,
            "turn_end_reasons": reasons[-5:],
            "errors": errors[-5:],
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
