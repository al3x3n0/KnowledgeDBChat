"""Mutate execution-plan step state for operator checkpoint actions."""

from datetime import datetime
from typing import Any, Callable


def set_current_plan_step_status(
    state: dict | None,
    *,
    status: str,
    advance_next: bool = False,
    utcnow: Callable[[], datetime] = datetime.utcnow,
) -> dict[str, Any]:
    """Update the current plan step and optionally advance to the next step."""
    payload = state if isinstance(state, dict) else {}
    plan = (
        payload.get("execution_plan")
        if isinstance(payload.get("execution_plan"), list)
        else []
    )
    if not plan:
        return {"step_id": "", "plan_step_index": -1}

    index = int(payload.get("plan_step_index", 0) or 0)
    index = max(0, min(index, len(plan) - 1))
    step = plan[index] if isinstance(plan[index], dict) else {}
    step_id = str(step.get("step_id") or f"step_{index + 1}").strip()
    step["status"] = str(status).strip()[:40] or "pending"
    step["updated_at"] = utcnow().isoformat()
    plan[index] = step

    if advance_next:
        next_index = min(len(plan) - 1, index + 1)
        payload["plan_step_index"] = next_index
        if (
            next_index != index
            and isinstance(plan[next_index], dict)
            and str(plan[next_index].get("status") or "") != "done"
        ):
            plan[next_index]["status"] = "in_progress"
            plan[next_index]["updated_at"] = utcnow().isoformat()

    payload["execution_plan"] = plan
    return {"step_id": step_id, "plan_step_index": index}
