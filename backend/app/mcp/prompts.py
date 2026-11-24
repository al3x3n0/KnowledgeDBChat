"""Research prompts for the resources an MCP profile can share."""

import json

from sqlalchemy.ext.asyncio import AsyncSession

from app.mcp.auth import MCPAuthContext
from app.mcp.resources import NOTE_URI_PREFIX, list_profile_resources
from app.schemas.mcp_prompt import (
    COMPARE_RESEARCH_NOTES,
    PROMPT_INPUT_PATTERN,
    RESEARCH_BRIEF,
    prompt_input_names,
)

MAX_ARGUMENT_LENGTH = 400
MAX_RENDERED_INSTRUCTIONS = 24_000

PROMPTS = {
    RESEARCH_BRIEF: {
        "name": RESEARCH_BRIEF,
        "title": "Research brief",
        "description": "Summarize evidence on a topic using the profile’s research context.",
        "arguments": [
            {
                "name": "topic",
                "description": "Research topic or question",
                "required": True,
            }
        ],
    },
    COMPARE_RESEARCH_NOTES: {
        "name": COMPARE_RESEARCH_NOTES,
        "title": "Compare research notes",
        "description": "Compare findings, methods, and gaps across the profile’s research notes.",
        "arguments": [
            {
                "name": "focus",
                "description": "Optional question for the comparison",
                "required": False,
            }
        ],
    },
}


def _prompt_uris(name: str, resources: list[dict]) -> list[str]:
    if name == COMPARE_RESEARCH_NOTES:
        notes = [r["uri"] for r in resources if r["uri"].startswith(NOTE_URI_PREFIX)]
        return notes if len(notes) >= 2 else []
    return [r["uri"] for r in resources]


async def list_profile_prompts(auth: MCPAuthContext, db: AsyncSession) -> list[dict]:
    resources = await list_profile_resources(auth, db)
    prompts = [
        prompt for name, prompt in PROMPTS.items() if _prompt_uris(name, resources)
    ]
    if auth.has_scope("read"):
        prompts.extend(
            _custom_prompt_metadata(prompt) for prompt in auth.api_key.mcp_prompts or []
        )
    return prompts


def _custom_prompt_metadata(prompt: dict) -> dict:
    metadata = {"name": prompt["name"], "title": prompt["title"]}
    if prompt.get("use_inputs", False):
        metadata["arguments"] = [
            {
                "name": name,
                "description": name.replace("_", " ").capitalize(),
                "required": True,
            }
            for name in prompt_input_names(prompt["instructions"])
        ]
    return metadata


def _render_custom_instructions(prompt: dict, arguments: dict) -> str | None:
    instructions = prompt["instructions"]
    if not prompt.get("use_inputs", False):
        return instructions if not arguments else None
    if arguments.keys() != set(prompt_input_names(instructions)):
        return None
    if any(
        not isinstance(value, str)
        or len(value) > MAX_ARGUMENT_LENGTH
        or not value.strip()
        for value in arguments.values()
    ):
        return None

    rendered = PROMPT_INPUT_PATTERN.sub(
        lambda match: arguments[match.group(1)].strip(), instructions
    )
    return rendered if len(rendered) <= MAX_RENDERED_INSTRUCTIONS else None


def _prompt_result(
    instructions: str, uris: list[str], description: str | None = None
) -> dict:
    if uris:
        references = "\n".join(f"- {uri}" for uri in uris)
        instructions += f"\n\nUse these MCP resources as context:\n{references}"
    result = {
        "messages": [
            {"role": "user", "content": {"type": "text", "text": instructions}}
        ],
    }
    if description is not None:
        result["description"] = description
    return result


async def get_profile_prompt(
    auth: MCPAuthContext, db: AsyncSession, name: str, arguments: dict
) -> dict | None:
    prompt = PROMPTS.get(name)
    if prompt is None:
        if not auth.has_scope("read"):
            return None
        for custom in auth.api_key.mcp_prompts or []:
            if custom["name"] == name:
                instructions = _render_custom_instructions(custom, arguments)
                if instructions is None:
                    return None
                uris = (
                    [r["uri"] for r in await list_profile_resources(auth, db)]
                    if custom["include_resources"]
                    else []
                )
                return _prompt_result(instructions, uris)
        return None
    argument = prompt["arguments"][0]
    key = argument["name"]
    if arguments.keys() - {key}:
        return None
    value = arguments.get(key, "")
    if not isinstance(value, str) or len(value) > MAX_ARGUMENT_LENGTH:
        return None
    value = value.strip()
    if argument["required"] and not value:
        return None

    uris = _prompt_uris(name, await list_profile_resources(auth, db))
    if not uris:
        return None
    if name == RESEARCH_BRIEF:
        instructions = (
            f"Prepare a research brief on this topic: {json.dumps(value, ensure_ascii=False)}. "
            "Summarize the main findings, cite the resources that support each claim, "
            "and identify gaps or questions that need more evidence."
        )
    else:
        instructions = (
            "Compare the findings and methods in these research notes. "
            "Identify where the notes agree, where they differ, and which gaps "
            "could explain the differences. Cite the notes for each comparison."
        )
        if value:
            instructions += (
                f" Focus the comparison on: {json.dumps(value, ensure_ascii=False)}."
            )

    return _prompt_result(instructions, uris, prompt["description"])
