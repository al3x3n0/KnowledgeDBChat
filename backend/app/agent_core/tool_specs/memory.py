"""Job memory: recording what a run learned, and recalling it later.

Generated from the literals these tools were declared as; the descriptions
and parameter schemas are the original text.
"""

from __future__ import annotations

from app.agent_core.tool_specs.spec import ToolSpec

SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="create_memory",
        description="Store a persistent memory for the current user. Use this to save important facts, insights, decisions, or context that should be available to future jobs.",
        parameters={
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "Memory content (concise, factual, and actionable)",
                },
                "importance": {
                    "type": "number",
                    "description": "Importance score 0.0-1.0 (default 0.5)",
                },
                "category": {
                    "type": "string",
                    "enum": [
                        "fact",
                        "preference",
                        "context",
                        "summary",
                        "goal",
                        "constraint",
                    ],
                    "description": "Memory category (default: fact)",
                },
                "metadata": {
                    "type": "object",
                    "description": "Optional metadata key-value pairs",
                },
            },
            "required": ["content"],
        },
        effects="write",
    ),
    ToolSpec(
        name="search_memories",
        description="Search the user's stored memories using semantic similarity. Returns ranked results matching the query.",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query for finding relevant memories",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results (default 10, max 50)",
                },
                "category_filter": {
                    "type": "string",
                    "enum": [
                        "fact",
                        "preference",
                        "context",
                        "summary",
                        "goal",
                        "constraint",
                    ],
                    "description": "Filter by memory category (optional)",
                },
                "min_importance": {
                    "type": "number",
                    "description": "Minimum importance score filter (0.0-1.0)",
                },
            },
            "required": ["query"],
        },
    ),
    ToolSpec(
        name="recall_memories",
        description="Recall memories related to a topic using broad semantic matching. Similar to search_memories but without filters, useful for open-ended context gathering.",
        parameters={
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "Topic to recall memories about",
                },
                "limit": {"type": "integer", "description": "Max results (default 10)"},
            },
            "required": ["topic"],
        },
    ),
    ToolSpec(
        name="get_memory_stats",
        description="Get statistics about the user's memory store including counts by type, recent activity, and most accessed memories.",
        parameters={"type": "object", "properties": {}, "required": []},
    ),
)
