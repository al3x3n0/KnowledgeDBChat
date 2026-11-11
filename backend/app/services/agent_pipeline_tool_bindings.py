"""Resolve explicit tool inputs from a dependency's recorded results."""

import re
from copy import deepcopy
from typing import Any, Dict, Optional, Sequence

SOURCE_SCOPED_TOOLS = frozenset(
    {
        "search_documents",
        "search_with_filters",
        "get_knowledge_base_stats",
        "create_synthesis_document",
        "add_to_reading_list",
        "get_reading_lists",
        "save_research_finding",
        "create_document_from_text",
        "project_bootstrap",
    }
)


def validate_bound_value(tool: str, argument: str, value: Any) -> None:
    """Refuse an empty scope before runtime can replace it with a default."""
    if (
        tool in SOURCE_SCOPED_TOOLS
        and argument == "source_id"
        and not str(value or "").strip()
    ):
        raise ValueError("bound source scope is empty")


def valid_pointer(path: Any) -> bool:
    return (
        isinstance(path, str)
        and (not path or path.startswith("/"))
        and not re.search(r"~(?![01])", path)
    )


def pointer_value(value: Any, path: str) -> Any:
    if not valid_pointer(path):
        raise ValueError(
            "path must be a JSON Pointer, such as /structured_output/value"
        )
    if not path:
        return value
    for raw in path[1:].split("/"):
        key = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(value, dict) and key in value:
            value = value[key]
        elif (
            isinstance(value, list)
            and re.fullmatch(r"0|[1-9][0-9]*", key)
            and int(key) < len(value)
        ):
            value = value[int(key)]
        else:
            raise ValueError(f"result path {path!r} does not exist")
    return value


def resolve_reference(
    dependencies: Sequence[str], inputs: Dict[str, Any], reference: Any
) -> Any:
    if not isinstance(reference, dict) or set(reference) != {"stage", "path"}:
        raise ValueError("binding needs stage and path")
    stage = reference["stage"]
    if stage not in dependencies:
        raise ValueError("source must be a declared dependency")
    if stage not in inputs:
        raise ValueError(f"dependency {stage!r} has no recorded results")
    return pointer_value(inputs[stage], reference["path"])


def apply_bindings(action: Dict[str, Any], job: Any) -> Dict[str, Any]:
    config = getattr(job, "config", None)
    if not isinstance(config, dict) or not config.get("pipeline_stage"):
        return action
    bindings = config.get("pipeline_tool_bindings") or {}
    references = (
        bindings.get(action.get("tool")) if isinstance(bindings, dict) else None
    )
    if not isinstance(references, dict) or not references:
        return action
    prepared = deepcopy(action)
    prepared.pop("_pipeline_binding_error", None)
    params = prepared.get("params", {})
    if not isinstance(params, dict):
        prepared["_pipeline_binding_error"] = "Bound tool arguments must be an object"
        return prepared
    prepared["params"] = params
    inputs = config.get("pipeline_input_results") or {}
    for argument, reference in references.items():
        try:
            params[argument] = deepcopy(
                resolve_reference(
                    config.get("pipeline_depends_on") or [], inputs, reference
                )
            )
            validate_bound_value(action.get("tool"), argument, params[argument])
        except (ValueError, TypeError) as error:
            prepared[
                "_pipeline_binding_error"
            ] = f"Cannot bind {action.get('tool')}.{argument}: {error}"
            return prepared
    return prepared


def dependency_results(
    parent: Any,
    child_config: Dict[str, Any],
    sibling_payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Inputs declared by this child, retaining lineage through inserted stages."""
    parent_config = parent.config if isinstance(parent.config, dict) else {}
    if not child_config.get("pipeline_stage") or child_config.get(
        "pipeline"
    ) != parent_config.get("pipeline"):
        return {}
    dependencies = child_config.get("pipeline_depends_on") or []
    # An inserted stage can sit between a consumer and the stage its explicit
    # bindings still name. Carry only those declared inputs through that hop;
    # do not retarget bindings to the insertion's different output schema.
    inherited = {
        **(parent_config.get("pipeline_forward_results") or {}),
        **(parent_config.get("pipeline_input_results") or {}),
    }
    inputs = {
        name: deepcopy(value)
        for name, value in inherited.items()
        if name in dependencies
    }
    for sibling in (sibling_payload or {}).get("sibling_jobs") or []:
        stage = sibling.get("pipeline_stage")
        if stage in dependencies and sibling.get("pipeline") == child_config.get(
            "pipeline"
        ):
            inputs[stage] = deepcopy(sibling.get("results") or {})
    stage = parent_config.get("pipeline_stage")
    if stage in dependencies:
        inputs[stage] = deepcopy(parent.results or {})
    return inputs
