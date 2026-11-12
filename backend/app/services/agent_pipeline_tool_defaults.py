"""Stage argument defaults, merged before approval, scope checks and dispatch."""

from copy import deepcopy
from typing import Any, Dict


def apply_defaults(action: Dict[str, Any], job: Any) -> Dict[str, Any]:
    config = getattr(job, "config", None)
    if not isinstance(config, dict) or not config.get("pipeline_stage"):
        return action
    defaults = config.get("pipeline_tool_defaults")
    values = defaults.get(action.get("tool")) if isinstance(defaults, dict) else None
    params = action.get("params", {})
    if not isinstance(values, dict) or not values or not isinstance(params, dict):
        return action
    # Explicit arguments win, including False, zero, empty strings and null.
    # Copy nested values: a handler must not mutate the next call's defaults.
    return {**action, "params": deepcopy({**values, **params})}
