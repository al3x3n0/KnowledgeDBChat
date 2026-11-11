"""Read-only previews against recorded results, using the runtime resolver."""

import json
from copy import deepcopy
from itertools import islice
from typing import Any, Dict

from app.services.agent_pipeline_restart import StageJob
from app.services.agent_pipeline_spec import PipelineStage
from app.services.agent_pipeline_tool_bindings import (
    resolve_reference,
    validate_bound_value,
)
from app.services.agent_tool_validation import coerce_tool_params, validate_tool_params

MAX_ITEMS = 100
MAX_VALUE_CHARS = 1000


def value_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    return "array" if isinstance(value, list) else "object"


def preview(
    stage: PipelineStage, latest: Dict[str, StageJob], tools: Dict[str, Any]
) -> Dict[str, Any]:
    inputs = {
        name: entry.job.results if isinstance(entry.job.results, dict) else {}
        for name, entry in latest.items()
        if name in stage.depends_on
    }
    items = []
    count = sum(len(refs) for refs in stage.tool_bindings.values())
    for tool, references in stage.tool_bindings.items():
        if len(items) >= MAX_ITEMS:
            break
        references = dict(islice(references.items(), MAX_ITEMS - len(items)))
        metadata = tools.get(tool)
        values, errors = {}, {}
        for argument, reference in references.items():
            try:
                if metadata is None:
                    raise ValueError(f"Tool {tool!r} is unavailable")
                values[argument] = resolve_reference(
                    stage.depends_on, inputs, reference
                )
                validate_bound_value(tool, argument, values[argument])
            except (ValueError, TypeError) as error:
                errors[argument] = str(error)
        params = deepcopy({**stage.tool_defaults.get(tool, {}), **values})
        if metadata is not None:
            coerce_tool_params(tool, params)
        for argument, reference in references.items():
            if len(items) >= MAX_ITEMS:
                break
            ref = reference if isinstance(reference, dict) else {}
            source = ref.get("stage") if isinstance(ref.get("stage"), str) else ""
            entry = latest.get(source) if source in stage.depends_on else None
            row = {
                "tool": tool,
                "argument": argument,
                "source_stage": source,
                "path": ref.get("path") if isinstance(ref.get("path"), str) else "",
                "source_job_id": str(entry.job.id) if entry else None,
                "source_status": str(entry.job.status) if entry else None,
                "source_contract_satisfied": entry.contract_met if entry else None,
                "status": "invalid",
            }
            try:
                if argument in errors:
                    raise ValueError(errors[argument])
                if argument not in params:
                    raise ValueError("Argument removed by runtime normalization")
                value = params[argument]
                invalid = validate_tool_params(
                    tool,
                    {argument: value},
                    schema=metadata["parameters"],
                    partial=True,
                    reject_unknown=True,
                )
                if invalid:
                    raise ValueError(invalid)
                encoded = json.dumps(value, ensure_ascii=False, default=str, indent=2)
                row.update(
                    status="resolved",
                    value_type=value_type(value),
                    value_json=encoded[:MAX_VALUE_CHARS],
                    truncated=len(encoded) > MAX_VALUE_CHARS,
                )
            except (ValueError, TypeError) as error:
                row["error"] = str(error)[:MAX_VALUE_CHARS]
            items.append(row)
    return {"items": items, "omitted": max(0, count - len(items))}
