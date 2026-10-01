"""Sandbox skills: packaged procedures a run loads and carries out.

Every other sandbox tool here is one fixed operation -- compile this, benchmark
that. A skill is a procedure plus the sandbox it runs in, written as data by a
person, a drafter or an earlier run. These four tools are how a run finds one,
reads it, executes it and proposes another.

Declared once and offered to every job type, like the measurement tools: a
skill is not owned by a kind of job. `run_sandbox_skill` declares no `produces`
because what it yields depends on the skill -- the evidence map carries that as
a rule about the `skill_` namespace instead (see `agent_evidence_map`).
"""

from __future__ import annotations

from app.agent_core.tool_specs.spec import ToolSpec

SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="list_sandbox_skills",
        description="List the sandbox skills available to this run. A skill is a "
        "packaged procedure for one kind of sandboxed work -- a toolchain, a "
        "measurement, an analysis -- with the image it runs in and the result "
        "it yields. Check here before improvising sandbox work from scratch: "
        "a skill has already been shown to run, and its result satisfies a "
        "contract requiring skill_<name>.",
        parameters={"type": "object", "properties": {}, "required": []},
    ),
    ToolSpec(
        name="load_sandbox_skill",
        description="Read one sandbox skill: its procedure, the helper files it "
        "places in the sandbox, the fields its result must have, and whether "
        "the skill supplies its own judge. Call this before run_sandbox_skill "
        "-- the procedure is the part that says what to run.",
        parameters={
            "type": "object",
            "properties": {
                "skill": {
                    "type": "string",
                    "description": "The skill's id, as list_sandbox_skills gives it.",
                },
            },
            "required": ["skill"],
        },
    ),
    ToolSpec(
        name="run_sandbox_skill",
        description="Run a shell command inside a skill's sandbox. The working "
        "directory persists between your calls in this run and is shared by "
        "every skill you use, so a procedure can be several commands: write a "
        "file, build, run, measure. In a pipeline, a later stage starts with a "
        "copy of the files the stage before it left; the result lists what is "
        "in the directory. "
        "The skill's helper files are at ./skill/ and are restored before every "
        "call. There is no network. Pass collect_result=true on the call that "
        "should count: the result is then read from ./result.json (written by "
        "the skill's judge if it has one, otherwise by your command), checked "
        "against the fields the skill declares, and recorded as a "
        "skill_<name> finding. You cannot supply result.json as a file; it "
        "has to come from what ran.",
        parameters={
            "type": "object",
            "properties": {
                "skill": {
                    "type": "string",
                    "description": "The skill's id.",
                },
                "command": {
                    "type": "string",
                    "description": (
                        "What to run, as /bin/sh would read it. Runs in the "
                        "working directory."
                    ),
                },
                "files": {
                    "type": "object",
                    "description": (
                        "Files to write into the working directory first, as "
                        "{relative_path: text}. Not under skill/, and not "
                        "result.json."
                    ),
                },
                "collect_result": {
                    "type": "boolean",
                    "description": (
                        "Collect and record the result after this command. "
                        "Leave false while you are still building or exploring."
                    ),
                },
                "label": {
                    "type": "string",
                    "description": (
                        "What this result is about -- the kernel, the input, "
                        "the configuration -- so two results from one skill "
                        "can be told apart."
                    ),
                },
                "timeout_seconds": {
                    "type": "integer",
                    "description": "Shorter than the skill's own limit, if you want.",
                },
            },
            "required": ["skill", "command"],
        },
        effects="write",
        cost_tier="high",
        typical_seconds=120,
        consumes="the skill's id and a shell command; load_sandbox_skill gives "
        "the procedure that says which.",
    ),
    ToolSpec(
        name="propose_sandbox_skill",
        description="Propose a new sandbox skill from a procedure that worked in "
        "THIS run, so later runs can follow it instead of rediscovering it. It "
        "is stored as a draft for a person to check and activate: it is not "
        "available to this run, and proposing one is not evidence of anything. "
        "Propose only what you actually ran -- the control you give is "
        "executed before anyone may activate the skill.",
        parameters={
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": (
                        "Short lowercase id, letters/digits/underscore. The "
                        "evidence is named skill_<id>."
                    ),
                },
                "name": {"type": "string"},
                "description": {
                    "type": "string",
                    "description": "WHEN to use it, in one or two sentences.",
                },
                "image": {
                    "type": "string",
                    "description": (
                        "The sandbox image, spelled exactly as an existing "
                        "skill or tool result reported it."
                    ),
                },
                "procedure": {
                    "type": "string",
                    "description": (
                        "The steps, concrete enough to follow without "
                        "rediscovering them, including what goes wrong."
                    ),
                },
                "files": {
                    "type": "object",
                    "description": "Helper files, as {relative_path: text}.",
                },
                "result_fields": {
                    "type": "object",
                    "description": (
                        "The fields result.json must have, as {name: type} "
                        "with type one of number, string, boolean, array, "
                        "object."
                    ),
                },
                "judge_command": {
                    "type": "string",
                    "description": (
                        "Optional. A command that computes result.json from "
                        "what a run left behind, so the run is not the author "
                        "of its own result."
                    ),
                },
                "control_command": {
                    "type": "string",
                    "description": (
                        "The smallest command that should work and leave a "
                        "valid result.json. Run before the skill can be "
                        "activated."
                    ),
                },
                "control_files": {
                    "type": "object",
                    "description": "Files the control needs, as {relative_path: text}.",
                },
                "why": {
                    "type": "string",
                    "description": "What this run did that makes it worth keeping.",
                },
            },
            "required": [
                "id",
                "name",
                "description",
                "image",
                "procedure",
                "result_fields",
                "control_command",
            ],
        },
        effects="write",
    ),
)
