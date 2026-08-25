"""Tool definitions for the agentic chat system.

The definitions themselves live in ``app.agent_core.tool_specs``, one
declaration per tool, and this module is the view of them the rest of the
application already expected: a flat list of schemas plus the helpers built on
it. Nothing is declared here any more, so nothing here can disagree with the
catalog, the policy or the evidence map.
"""

from typing import Any, Dict, List

from app.agent_core import tool_specs

#: Every tool a model may be offered, in domain order.
AGENT_TOOLS: List[Dict[str, Any]] = tool_specs.schemas()


def get_tools_description() -> str:
    """Generate a text description of available tools for the LLM prompt."""
    descriptions = []
    for tool in AGENT_TOOLS:
        params = tool["parameters"]["properties"]
        param_list = []
        for name, info in params.items():
            required = name in tool["parameters"].get("required", [])
            param_str = f"  - {name} ({info['type']}{'*' if required else ''}): {info['description']}"
            param_list.append(param_str)

        tool_desc = f"""Tool: {tool['name']}
Description: {tool['description']}
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
