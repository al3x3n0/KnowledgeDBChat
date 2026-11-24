"""Owner-authored prompt configuration for custom MCP profiles."""

import re

from pydantic import BaseModel, ConfigDict, Field, model_validator

RESEARCH_BRIEF = "research_brief"
COMPARE_RESEARCH_NOTES = "compare_research_notes"
BUILT_IN_PROMPT_NAMES = frozenset({RESEARCH_BRIEF, COMPARE_RESEARCH_NOTES})
MAX_PROMPT_INPUTS = 10
PROMPT_INPUT_PATTERN = re.compile(
    r"(?<!\{)\{\{[ \t]*([a-z][a-z0-9_]{0,47})[ \t]*\}\}(?!\})"
)


def prompt_input_names(instructions: str) -> list[str]:
    return list(dict.fromkeys(PROMPT_INPUT_PATTERN.findall(instructions)))


class MCPCustomPrompt(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    name: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,47}$")
    title: str = Field(min_length=1, max_length=100)
    instructions: str = Field(min_length=1, max_length=8000)
    include_resources: bool = True
    use_inputs: bool = False

    @model_validator(mode="after")
    def validate_inputs(self) -> "MCPCustomPrompt":
        if self.use_inputs:
            count = len(prompt_input_names(self.instructions))
            if not 1 <= count <= MAX_PROMPT_INPUTS:
                raise ValueError(
                    "Use between one and ten named inputs in the instructions"
                )
        return self
