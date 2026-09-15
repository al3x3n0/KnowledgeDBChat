"""Tool definitions for the agentic chat system.

The definitions themselves live in ``app.agent_core.tool_specs``, one
declaration per tool, and this module is the view of them the rest of the
application already expected: a flat list of schemas plus the helpers built on
it. Nothing is declared here any more, so nothing here can disagree with the
catalog, the policy or the evidence map.
"""

from typing import Any, Dict, List, Optional, Sequence

from app.agent_core import tool_specs

#: Every tool a model may be offered, in domain order.
AGENT_TOOLS: List[Dict[str, Any]] = tool_specs.schemas()


def get_tools_description(extra: Optional[Sequence[Dict[str, Any]]] = None) -> str:
    """Generate a text description of available tools for the LLM prompt.

    ``extra`` is what the *caller's* plugins contribute. It is a parameter
    rather than a lookup because `AgentService` is a module-level singleton:
    caching one user's contributed tools on it would offer them to the next
    person who opened a chat.
    """
    descriptions = []
    for tool in list(AGENT_TOOLS) + list(extra or []):
        # Read defensively. Every built-in schema is hand-written here and has
        # a type and a description on every property; a *contributed* one is
        # whatever its author typed, and the first tool missing a key would
        # otherwise raise while building the prompt -- taking down the whole
        # turn, for every tool, because one plugin omitted a field.
        schema = tool.get("parameters") or {}
        params = schema.get("properties") or {}
        required = set(schema.get("required") or [])
        param_list = []
        for name, info in params.items():
            info = info if isinstance(info, dict) else {}
            kind = info.get("type") or "any"
            note = info.get("description") or "(no description)"
            param_list.append(
                f"  - {name} ({kind}{'*' if name in required else ''}): {note}"
            )

        tool_desc = f"""Tool: {tool.get('name', '')}
Description: {tool.get('description') or '(no description)'}
Parameters:
{chr(10).join(param_list) if param_list else '  (no parameters)'}"""
        descriptions.append(tool_desc)

    return "\n\n".join(descriptions)


def get_tool_by_name(name: str) -> Dict[str, Any] | None:
    """Get a tool definition by name."""
    for tool in AGENT_TOOLS:
        if tool["name"] == name:
            return tool
    return None


def validate_tool_params(tool_name: str, params: Dict[str, Any]) -> tuple[bool, str]:
    """
    Validate parameters for a tool call.

    Returns:
        Tuple of (is_valid, error_message)
    """
    tool = get_tool_by_name(tool_name)
    if not tool:
        return False, f"Unknown tool: {tool_name}"

    required_params = tool["parameters"].get("required", [])
    for param in required_params:
        if param not in params:
            return False, f"Missing required parameter: {param}"

    return True, ""
