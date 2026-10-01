"""Sandbox skills: a sandboxed capability written as data.

The risk in letting a skill be authored -- by a person, a drafter or a run --
is that each of the ways it can be wrong fails quietly: a result nothing
checks, a skill active against content its control never saw, a contract
requiring evidence no skill of this user yields. These tests pin the rules that
make each of those loud.

The Docker daemon is replaced by a stand-in that acts on the run directory, so
everything except the container itself is the real code.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.agent_core import tool_specs
from app.services import (
    agent_evidence_map,
    agent_pipeline_draft,
    agent_pipeline_spec,
    agent_sandbox_skill_tools,
    sandbox_skill_image_service,
    sandbox_skill_manifest,
    sandbox_skill_runtime,
    sandbox_skill_service,
)
from app.services.sandbox_skill_image_service import ImageError
from app.services.sandbox_skill_manifest import SkillError

IMAGE = "ghcr.io/al3x3n0/kdbc-compiler-research:latest"


def manifest(**overrides):
    base = {
        "id": "loop_trip",
        "name": "Loop trip counts",
        "description": "Count how many times each loop in a C kernel runs.",
        "image": IMAGE,
        "procedure": "Write kernel.c, build it, run skill/count.sh.",
        "files": {"count.sh": "#!/bin/sh\necho counting\n"},
        "result": {"fields": {"loops": "number", "hottest": "string"}},
        "control": {"command": "sh skill/count.sh"},
    }
    base.update(overrides)
    return base


def valid(**overrides):
    return sandbox_skill_manifest.validate_skill(
        manifest(**overrides), known_images=[IMAGE]
    )


# --------------------------------------------------------------- the validator


@pytest.mark.parametrize(
    "broken, fragment",
    [
        ({"id": "Loop-Trip"}, "lowercase letters"),
        ({"description": "short"}, "WHEN to use"),
        ({"procedure": ""}, "procedure is required"),
        ({"image": "docker.io/library/ubuntu"}, "may not be used"),
        ({"result": {"fields": {}}}, "result.fields is required"),
        ({"result": {"fields": {"n": "integer"}}}, "must be one of"),
        ({"control": None}, "control is required"),
        ({"control": {"command": ""}}, "control.command is required"),
        ({"files": {"../escape.sh": "x"}}, "not allowed"),
        ({"files": {"/abs.sh": "x"}}, "not allowed"),
        ({"files": {"result.json": "{}"}}, "may not include result.json"),
        ({"timeout_seconds": 2}, "must be between"),
    ],
)
def test_a_skill_is_refused_with_the_reason(broken, fragment):
    with pytest.raises(SkillError) as refused:
        valid(**broken)
    assert fragment in str(refused.value)


def test_a_refused_image_names_the_ones_that_are_allowed():
    """A caller told only that its choice was invalid will guess again."""
    with pytest.raises(SkillError) as refused:
        valid(image="kdbc-compiler-research")
    assert IMAGE in str(refused.value)


def test_a_misspelt_key_is_refused_rather_than_dropped():
    """`judge_cmd` dropped in silence is a skill running with no judge while
    its author believes it has one."""
    with pytest.raises(SkillError) as refused:
        valid(judge_cmd="python3 skill/judge.py")
    assert "judge_cmd" in str(refused.value)
    assert "judge_command" in str(refused.value)


def test_the_hash_ignores_key_order_and_notices_content():
    a = valid()
    b = sandbox_skill_manifest.validate_skill(
        dict(reversed(list(manifest().items()))), known_images=[IMAGE]
    )
    assert sandbox_skill_manifest.content_hash(
        a
    ) == sandbox_skill_manifest.content_hash(b)
    c = valid(procedure="Something else entirely.")
    assert sandbox_skill_manifest.content_hash(
        a
    ) != sandbox_skill_manifest.content_hash(c)


@pytest.mark.parametrize(
    "payload, fragment",
    [
        ([1, 2], "must be a JSON object"),
        ({"loops": 3}, "no 'hottest'"),
        ({"loops": "3", "hottest": "a"}, "should be a number"),
        ({"loops": True, "hottest": "a"}, "should be a number"),
        ({"loops": float("nan"), "hottest": "a"}, "not finite"),
    ],
)
def test_a_result_is_checked_against_the_declared_fields(payload, fragment):
    problems = sandbox_skill_manifest.result_problems(valid(), payload)
    assert any(fragment in p for p in problems), problems


def test_a_result_with_every_field_has_no_problems():
    assert not sandbox_skill_manifest.result_problems(
        valid(), {"loops": 3, "hottest": "inner", "extra": [1]}
    )


# ------------------------------------------------------------------ the runner


class FakeSandbox:
    """Stands in for `docker run`: maps a command to something done in /work."""

    def __init__(self):
        self.calls = []
        self.behaviours = {}

    def on(self, command, action):
        self.behaviours[command] = action

    async def __call__(self, script, workdir, *, image, timeout_seconds):
        self.calls.append((script, image, timeout_seconds))
        action = self.behaviours.get(script)
        if action is None:
            return 0, "", ""
        return action(Path(workdir))


def writes_result(payload, *, code=0):
    def _action(workdir: Path):
        (workdir / "result.json").write_text(json.dumps(payload))
        return code, "ok", ""

    return _action


@pytest.fixture
def sandbox(monkeypatch, tmp_path):
    fake = FakeSandbox()
    monkeypatch.setattr(sandbox_skill_runtime, "_run_in_sandbox", fake)
    monkeypatch.setattr(sandbox_skill_runtime, "_execution_enabled", lambda: True)
    monkeypatch.setattr(
        sandbox_skill_runtime.tempfile, "gettempdir", lambda: str(tmp_path)
    )
    return fake


async def test_nothing_runs_when_execution_is_disabled(monkeypatch, tmp_path):
    monkeypatch.setattr(sandbox_skill_runtime, "_execution_enabled", lambda: False)
    run = await sandbox_skill_runtime.execute(valid(), workdir=tmp_path, command="true")
    assert not run.ran
    assert "ENABLE_UNSAFE_CODE_EXECUTION" in run.error


async def test_an_image_that_is_not_allowed_never_starts(sandbox, tmp_path):
    run = await sandbox_skill_runtime.execute(
        valid(), workdir=tmp_path, command="true", image_allowed=False
    )
    assert not run.ran and not sandbox.calls
    assert "not allowlisted" in run.error


async def test_a_container_that_never_started_is_not_a_failed_command(
    sandbox, tmp_path
):
    """Exit 125 is docker's own failure. Reporting it as the command's exit
    code sends a run to rewrite code that never executed."""
    sandbox.on("true", lambda _w: (125, "", "Unable to find image"))
    run = await sandbox_skill_runtime.execute(valid(), workdir=tmp_path, command="true")
    assert not run.ran
    assert "never ran" in run.error or "could not start" in run.error


async def test_the_skills_files_are_placed_and_restored_every_call(sandbox, tmp_path):
    def vandalise(workdir: Path):
        (workdir / "skill" / "count.sh").write_text("echo tampered")
        return 0, "", ""

    sandbox.on("first", vandalise)
    await sandbox_skill_runtime.execute(valid(), workdir=tmp_path, command="first")
    assert (tmp_path / "skill" / "count.sh").read_text() == "echo tampered"
    await sandbox_skill_runtime.execute(valid(), workdir=tmp_path, command="second")
    assert "counting" in (tmp_path / "skill" / "count.sh").read_text()


async def test_a_stale_result_cannot_be_collected(sandbox, tmp_path):
    """The result of an earlier command is not the result of this one."""
    (tmp_path / "result.json").write_text(json.dumps({"loops": 9, "hottest": "x"}))
    run = await sandbox_skill_runtime.execute(
        valid(), workdir=tmp_path, command="does nothing", collect_result=True
    )
    assert run.result is None
    assert any("left no result.json" in p for p in run.result_problems)


async def test_a_valid_result_is_accepted_and_says_who_wrote_it(sandbox, tmp_path):
    sandbox.on("measure", writes_result({"loops": 4, "hottest": "inner"}))
    run = await sandbox_skill_runtime.execute(
        valid(), workdir=tmp_path, command="measure", collect_result=True
    )
    assert run.result == {"loops": 4, "hottest": "inner"}
    assert run.judged_by == "command"


async def test_a_judge_replaces_whatever_the_run_wrote(sandbox, tmp_path):
    """With a judge, the run is not the author of its own evidence."""
    sandbox.on("measure", writes_result({"loops": 999, "hottest": "made up"}))
    sandbox.on("judge", writes_result({"loops": 4, "hottest": "inner"}))
    run = await sandbox_skill_runtime.execute(
        valid(judge_command="judge"),
        workdir=tmp_path,
        command="measure",
        collect_result=True,
    )
    assert run.result == {"loops": 4, "hottest": "inner"}
    assert run.judged_by == "judge_command"


async def test_a_judge_that_writes_nothing_leaves_no_result(sandbox, tmp_path):
    """The run's own file was discarded before the judge ran; it must not
    come back when the judge has nothing to say."""
    sandbox.on("measure", writes_result({"loops": 999, "hottest": "made up"}))
    run = await sandbox_skill_runtime.execute(
        valid(judge_command="judge"),
        workdir=tmp_path,
        command="measure",
        collect_result=True,
    )
    assert run.result is None and run.judged_by == ""


async def test_a_failed_command_collects_nothing(sandbox, tmp_path):
    sandbox.on("measure", writes_result({"loops": 4, "hottest": "inner"}, code=2))
    run = await sandbox_skill_runtime.execute(
        valid(), workdir=tmp_path, command="measure", collect_result=True
    )
    assert run.result is None
    assert "exited 2" in run.result_problems[0]


@pytest.mark.parametrize(
    "files, fragment",
    [
        ({"result.json": "{}"}, "has to be produced by the command"),
        ({"skill/count.sh": "x"}, "holds the skill's own files"),
        ({"../x": "x"}, "not allowed"),
        ({"a.c": 3}, "must be text"),
    ],
)
def test_a_run_cannot_supply_its_own_result_or_overwrite_the_skill(files, fragment):
    assert fragment in sandbox_skill_runtime.reject_caller_files(files)


async def test_a_dry_run_passes_only_with_a_valid_result(sandbox):
    sandbox.on("sh skill/count.sh", writes_result({"loops": 1, "hottest": "a"}))
    outcome = await sandbox_skill_runtime.dry_run(valid(), image_allowed=True)
    assert outcome["ok"] and outcome["ran"]

    sandbox.on("sh skill/count.sh", writes_result({"loops": 1}))
    outcome = await sandbox_skill_runtime.dry_run(valid(), image_allowed=True)
    assert not outcome["ok"] and outcome["ran"]
    assert "hottest" in outcome["detail"]


async def test_a_dry_run_that_could_not_run_says_so(monkeypatch):
    monkeypatch.setattr(sandbox_skill_runtime, "_execution_enabled", lambda: False)
    outcome = await sandbox_skill_runtime.dry_run(valid(), image_allowed=True)
    assert not outcome["ok"] and not outcome["ran"]


# --------------------------------------------------------------- the lifecycle


@pytest.fixture
def allow_image(monkeypatch):
    monkeypatch.setattr(
        sandbox_skill_service.agent_sandbox_runtime, "allowed_images", lambda: [IMAGE]
    )


async def test_a_skill_is_a_draft_however_it_was_created(
    db_session, test_user, allow_image
):
    skill = await sandbox_skill_service.create_skill(
        db_session, user_id=test_user.id, raw=manifest()
    )
    assert skill.status == "draft"
    assert not await sandbox_skill_service.active_skills(db_session, test_user.id)


async def test_activation_is_refused_until_the_control_has_passed(
    db_session, test_user, allow_image, sandbox
):
    skill = await sandbox_skill_service.create_skill(
        db_session, user_id=test_user.id, raw=manifest()
    )
    with pytest.raises(SkillError) as refused:
        await sandbox_skill_service.activate_skill(db_session, skill)
    assert "has not passed its control run" in str(refused.value)

    sandbox.on("sh skill/count.sh", writes_result({"loops": 1, "hottest": "a"}))
    outcome = await sandbox_skill_service.dry_run_skill(db_session, skill)
    assert outcome["ok"]
    await sandbox_skill_service.activate_skill(db_session, skill)
    active = await sandbox_skill_service.active_skills(db_session, test_user.id)
    assert [s.slug for s in active] == ["loop_trip"]


async def test_editing_an_active_skill_returns_it_to_draft(
    db_session, test_user, allow_image, sandbox
):
    """A verified skill that is edited is a different skill."""
    sandbox.on("sh skill/count.sh", writes_result({"loops": 1, "hottest": "a"}))
    skill = await sandbox_skill_service.create_skill(
        db_session, user_id=test_user.id, raw=manifest()
    )
    await sandbox_skill_service.dry_run_skill(db_session, skill)
    await sandbox_skill_service.activate_skill(db_session, skill)

    await sandbox_skill_service.update_skill(
        db_session, skill, manifest(procedure="A different procedure.")
    )
    assert skill.status == "draft"
    assert not sandbox_skill_service.is_verified(skill)
    assert not await sandbox_skill_service.active_skills(db_session, test_user.id)
    with pytest.raises(SkillError) as refused:
        await sandbox_skill_service.activate_skill(db_session, skill)
    assert "since it was last edited" in str(refused.value)


async def test_saving_without_changing_anything_keeps_it_active(
    db_session, test_user, allow_image, sandbox
):
    sandbox.on("sh skill/count.sh", writes_result({"loops": 1, "hottest": "a"}))
    skill = await sandbox_skill_service.create_skill(
        db_session, user_id=test_user.id, raw=manifest()
    )
    await sandbox_skill_service.dry_run_skill(db_session, skill)
    await sandbox_skill_service.activate_skill(db_session, skill)
    await sandbox_skill_service.update_skill(db_session, skill, manifest())
    assert skill.status == "active"


async def test_a_control_that_starts_failing_deactivates_the_skill(
    db_session, test_user, allow_image, sandbox
):
    sandbox.on("sh skill/count.sh", writes_result({"loops": 1, "hottest": "a"}))
    skill = await sandbox_skill_service.create_skill(
        db_session, user_id=test_user.id, raw=manifest()
    )
    await sandbox_skill_service.dry_run_skill(db_session, skill)
    await sandbox_skill_service.activate_skill(db_session, skill)

    sandbox.on("sh skill/count.sh", lambda _w: (1, "", "count.sh: not found"))
    await sandbox_skill_service.dry_run_skill(db_session, skill)
    assert skill.status == "draft" and skill.verified_hash is None


async def test_an_unreachable_sandbox_does_not_revoke_a_verdict(
    db_session, test_user, allow_image, sandbox, monkeypatch
):
    """A daemon that is down says nothing about the skill either way."""
    sandbox.on("sh skill/count.sh", writes_result({"loops": 1, "hottest": "a"}))
    skill = await sandbox_skill_service.create_skill(
        db_session, user_id=test_user.id, raw=manifest()
    )
    await sandbox_skill_service.dry_run_skill(db_session, skill)
    await sandbox_skill_service.activate_skill(db_session, skill)

    monkeypatch.setattr(sandbox_skill_runtime, "_execution_enabled", lambda: False)
    outcome = await sandbox_skill_service.dry_run_skill(db_session, skill)
    assert not outcome["ok"]
    assert skill.status == "active" and sandbox_skill_service.is_verified(skill)


async def test_the_id_cannot_change_and_cannot_be_duplicated(
    db_session, test_user, allow_image
):
    skill = await sandbox_skill_service.create_skill(
        db_session, user_id=test_user.id, raw=manifest()
    )
    with pytest.raises(SkillError) as refused:
        await sandbox_skill_service.update_skill(
            db_session, skill, manifest(id="renamed")
        )
    assert "skill_loop_trip" in str(refused.value)
    with pytest.raises(SkillError) as refused:
        await sandbox_skill_service.create_skill(
            db_session, user_id=test_user.id, raw=manifest()
        )
    assert "already have a skill" in str(refused.value)


async def test_one_users_skill_is_not_another_users(
    db_session, test_user, admin_user, allow_image, sandbox
):
    sandbox.on("sh skill/count.sh", writes_result({"loops": 1, "hottest": "a"}))
    skill = await sandbox_skill_service.create_skill(
        db_session, user_id=test_user.id, raw=manifest()
    )
    await sandbox_skill_service.dry_run_skill(db_session, skill)
    await sandbox_skill_service.activate_skill(db_session, skill)
    assert await sandbox_skill_service.get_active(db_session, test_user.id, "loop_trip")
    assert not await sandbox_skill_service.get_active(
        db_session, admin_user.id, "loop_trip"
    )
    assert await sandbox_skill_service.unmet_skill_evidence(
        db_session, admin_user.id, ["skill_loop_trip", "benchmark_measurement"]
    ) == ["skill_loop_trip"]


# ------------------------------------------------- evidence, and pipelines


def test_no_builtin_evidence_lives_in_the_skill_namespace():
    """The prefix is only a namespace while nothing first-party occupies it."""
    taken = [
        produced
        for spec in tool_specs.all_specs()
        for produced in spec.produces
        if produced.startswith(agent_evidence_map.SKILL_EVIDENCE_PREFIX)
    ]
    assert not taken
    assert (
        agent_evidence_map.SKILL_EVIDENCE_PREFIX
        == sandbox_skill_manifest.EVIDENCE_PREFIX
    )


def test_skill_evidence_is_planned_through_the_skill_runner():
    assert agent_evidence_map.producers_of("skill_loop_trip") == ["run_sandbox_skill"]
    assert agent_evidence_map.chain_for(["skill_loop_trip"], job_type="research") == [
        "run_sandbox_skill"
    ]
    assert not agent_evidence_map.unobtainable(["skill_loop_trip"])
    # The bare prefix names no skill, and is not evidence of anything.
    assert agent_evidence_map.unobtainable(["skill_"]) == ["skill_"]


def test_the_prompt_names_the_skill_not_just_the_tool():
    """The tool is the same for every skill; the skill is the argument that
    decides what a run gets."""
    lines = agent_evidence_map.describe_chain(
        ["skill_loop_trip", "skill_other"], job_type="research"
    )
    assert len(lines) == 2
    assert "skill='loop_trip'" in lines[0] and "skill_loop_trip" in lines[0]
    assert "load_sandbox_skill" in lines[0]


def _pipeline(required):
    return agent_pipeline_spec.normalize(
        {
            "name": "p",
            "stages": [
                {
                    "id": "measure",
                    "goal": "Count the loop trips of the kernel.",
                    "contract": {"required_finding_types": list(required)},
                }
            ],
        }
    )


def test_a_stage_may_require_skill_evidence():
    assert not agent_pipeline_spec.validate(_pipeline(["skill_loop_trip"]))
    plan = agent_pipeline_spec.plan(_pipeline(["skill_loop_trip"]))
    assert "run_sandbox_skill" in plan.stages[0].tools


def test_a_stage_requiring_a_skill_nobody_activated_is_a_problem():
    """The static checker accepts the whole namespace; this is the half that
    knows which skills exist."""
    pipeline = _pipeline(["skill_loop_trip"])
    problems = agent_pipeline_draft.skill_problems(pipeline, [])
    assert problems and "no active sandbox skill 'loop_trip'" in problems[0]
    assert not agent_pipeline_draft.skill_problems(pipeline, ["skill_loop_trip"])


# ------------------------------------------------------------ what a run sees


@pytest.fixture
async def active_skill(db_session, test_user, allow_image, sandbox):
    sandbox.on("sh skill/count.sh", writes_result({"loops": 1, "hottest": "a"}))
    skill = await sandbox_skill_service.create_skill(
        db_session, user_id=test_user.id, raw=manifest()
    )
    await sandbox_skill_service.dry_run_skill(db_session, skill)
    return await sandbox_skill_service.activate_skill(db_session, skill)


def ctx_for(db_session, user, job_id=None):
    job = SimpleNamespace(
        id=job_id or uuid4(), user_id=user.id, name="study", config={}
    )
    return SimpleNamespace(db=db_session, job=job, user_id=user.id, state={})


async def test_a_run_lists_and_loads_only_active_skills(
    db_session, test_user, active_skill
):
    ctx = ctx_for(db_session, test_user)
    listed = await agent_sandbox_skill_tools.list_sandbox_skills({}, ctx)
    assert [s["skill"] for s in listed["data"]["skills"]] == ["loop_trip"]
    assert listed["data"]["skills"][0]["produces"] == "skill_loop_trip"

    loaded = await agent_sandbox_skill_tools.load_sandbox_skill(
        {"skill": "skill_loop_trip"}, ctx
    )
    assert loaded["data"]["procedure"].startswith("Write kernel.c")
    assert "skill/count.sh" in loaded["data"]["files"]
    assert loaded["data"]["judged_by"] == "command"


async def test_an_unknown_skill_names_the_ones_that_exist(
    db_session, test_user, active_skill
):
    result = await agent_sandbox_skill_tools.run_sandbox_skill(
        {"skill": "nope", "command": "true"}, ctx_for(db_session, test_user)
    )
    assert "loop_trip" in result["error"]


async def test_a_collected_result_is_recorded_as_the_skills_evidence(
    db_session, test_user, active_skill, sandbox
):
    sandbox.on("measure", writes_result({"loops": 4, "hottest": "inner", "type": "x"}))
    result = await agent_sandbox_skill_tools.run_sandbox_skill(
        {
            "skill": "loop_trip",
            "command": "measure",
            "collect_result": True,
            "label": "blur",
        },
        ctx_for(db_session, test_user),
    )
    assert result["success"]
    (finding,) = result["findings"]
    # A result field called `type` must not rename the evidence.
    assert finding["type"] == "skill_loop_trip"
    assert finding["loops"] == 4 and finding["subject"] == "blur"
    assert finding["judged_by"] == "command"


async def test_exploring_records_nothing(db_session, test_user, active_skill, sandbox):
    sandbox.on("measure", writes_result({"loops": 4, "hottest": "inner"}))
    result = await agent_sandbox_skill_tools.run_sandbox_skill(
        {"skill": "loop_trip", "command": "measure"}, ctx_for(db_session, test_user)
    )
    assert result["success"] and "findings" not in result


async def test_a_result_that_fails_its_shape_is_an_error_not_a_finding(
    db_session, test_user, active_skill, sandbox
):
    sandbox.on("measure", writes_result({"loops": "many"}))
    result = await agent_sandbox_skill_tools.run_sandbox_skill(
        {"skill": "loop_trip", "command": "measure", "collect_result": True},
        ctx_for(db_session, test_user),
    )
    assert "findings" not in result
    assert "hottest" in result["error"] and "should be a number" in result["error"]


async def test_the_working_directory_persists_between_calls(
    db_session, test_user, active_skill, sandbox
):
    seen = []
    sandbox.on("write", lambda w: ((w / "built").write_text("x"), (0, "", ""))[1])
    sandbox.on("read", lambda w: (seen.append((w / "built").exists()), (0, "", ""))[1])
    ctx = ctx_for(db_session, test_user)
    await agent_sandbox_skill_tools.run_sandbox_skill(
        {"skill": "loop_trip", "command": "write"}, ctx
    )
    await agent_sandbox_skill_tools.run_sandbox_skill(
        {"skill": "loop_trip", "command": "read"}, ctx
    )
    assert seen == [True]


async def test_a_run_proposes_a_draft_and_nothing_more(
    db_session, test_user, allow_image
):
    ctx = ctx_for(db_session, test_user, job_id=None)
    ctx.job.id = None  # no agent_jobs row in this test database
    result = await agent_sandbox_skill_tools.propose_sandbox_skill(
        {
            "id": "ir_stats",
            "name": "IR statistics",
            "description": "Count instructions per opcode in LLVM IR.",
            "image": IMAGE,
            "procedure": "Emit IR, then run skill/stats.py on it.",
            "result_fields": {"instructions": "number"},
            "control_command": "python3 skill/stats.py",
            "why": "Did this by hand three times.",
        },
        ctx,
    )
    assert result["success"] and result["data"]["status"] == "draft"
    assert "findings" not in result
    (skill,) = await sandbox_skill_service.list_skills(db_session, test_user.id)
    assert skill.origin == "agent" and skill.status == "draft"
    assert not await sandbox_skill_service.active_skills(db_session, test_user.id)


async def test_a_refused_proposal_hands_back_the_refusal(
    db_session, test_user, allow_image
):
    ctx = ctx_for(db_session, test_user)
    ctx.job.id = None
    result = await agent_sandbox_skill_tools.propose_sandbox_skill(
        {
            "id": "ir_stats",
            "name": "IR statistics",
            "description": "Count instructions per opcode in LLVM IR.",
            "image": "ubuntu",
            "procedure": "p",
            "result_fields": {"instructions": "number"},
            "control_command": "true",
        },
        ctx,
    )
    assert IMAGE in result["error"]


def test_every_skill_tool_is_declared_and_offered_to_every_job_type():
    for name in (
        "list_sandbox_skills",
        "load_sandbox_skill",
        "run_sandbox_skill",
        "propose_sandbox_skill",
    ):
        spec = tool_specs.spec_for(name)
        assert spec is not None and spec.job_types is None
        assert spec.produces == ()


# --------------------------------------------------------------------- images


@pytest.fixture
def allow_base(monkeypatch):
    monkeypatch.setattr(
        sandbox_skill_image_service.agent_sandbox_runtime,
        "allowed_images",
        lambda: [IMAGE],
    )


@pytest.mark.parametrize(
    "dockerfile, fragment",
    [
        ("RUN echo hi\n", "exactly one FROM"),
        ("FROM ubuntu:24.04\nRUN true\n", "not an image this server allows"),
        (f"FROM {IMAGE}\nFROM {IMAGE}\n", "exactly one FROM"),
        (f"FROM {IMAGE} AS build\n", "one image and nothing else"),
        (f"FROM {IMAGE}\nCOPY a b\n", "nothing to copy from"),
        (f'FROM {IMAGE}\nENTRYPOINT ["x"]\n', "/bin/sh -lc"),
    ],
)
def test_a_dockerfile_is_refused_with_the_reason(allow_base, dockerfile, fragment):
    with pytest.raises(ImageError) as refused:
        sandbox_skill_image_service.validate_dockerfile(dockerfile)
    assert fragment in str(refused.value)


def test_a_dockerfile_extending_an_allowed_image_is_accepted(allow_base):
    dockerfile = (
        f"# comment\nFROM {IMAGE}\nUSER root\n"
        "RUN apt-get update && \\\n    apt-get install -y z3\n"
    )
    assert sandbox_skill_image_service.validate_dockerfile(dockerfile) == IMAGE


def test_an_edited_dockerfile_is_a_different_image():
    a = sandbox_skill_image_service.tag_for("z3", "FROM a\nRUN x\n")
    b = sandbox_skill_image_service.tag_for("z3", "FROM a\nRUN y\n")
    assert a != b and a.startswith("kdbc-skill/z3:")


async def test_a_proposed_image_is_not_usable_until_built(
    db_session, test_user, allow_base, monkeypatch
):
    dockerfile = f"FROM {IMAGE}\nRUN true\n"
    image = await sandbox_skill_image_service.propose_image(
        db_session, user_id=test_user.id, slug="z3", dockerfile=dockerfile
    )
    assert image.status == "proposed"
    assert image.tag not in await sandbox_skill_service.known_images(db_session)

    # Building is refused while the deployment has not enabled it.
    with pytest.raises(ImageError) as refused:
        await sandbox_skill_image_service.mark_building(db_session, image, test_user.id)
    assert "SANDBOX_SKILL_IMAGE_BUILD_ENABLED" in str(refused.value)

    monkeypatch.setattr(sandbox_skill_image_service, "build_enabled", lambda: True)

    async def fake_build(tag, text, timeout):
        return 0, "Successfully built"

    monkeypatch.setattr(sandbox_skill_image_service, "_docker_build", fake_build)
    await sandbox_skill_image_service.mark_building(db_session, image, test_user.id)
    await sandbox_skill_image_service.build_image(db_session, image)
    assert image.status == "built"
    assert image.tag in await sandbox_skill_service.known_images(db_session)


async def test_a_failed_build_keeps_its_log_and_stays_unusable(
    db_session, test_user, allow_base, monkeypatch
):
    monkeypatch.setattr(sandbox_skill_image_service, "build_enabled", lambda: True)

    async def fake_build(tag, text, timeout):
        return 1, "E: Unable to locate package z33"

    monkeypatch.setattr(sandbox_skill_image_service, "_docker_build", fake_build)
    image = await sandbox_skill_image_service.propose_image(
        db_session,
        user_id=test_user.id,
        slug="z3",
        dockerfile=f"FROM {IMAGE}\nRUN apt-get install -y z33\n",
    )
    await sandbox_skill_image_service.build_image(db_session, image)
    assert image.status == "failed" and "z33" in image.build_log
    assert image.tag not in await sandbox_skill_service.known_images(db_session)


# ------------------------------------------------------------------------ API


def test_the_api_keeps_a_skill_a_draft_until_its_control_passes(
    client, auth_headers, allow_image, sandbox
):
    created = client.post(
        "/api/v1/sandbox-skills", json={"manifest": manifest()}, headers=auth_headers
    )
    assert created.status_code == 201, created.text
    skill = created.json()
    assert skill["status"] == "draft" and skill["produces"] == "skill_loop_trip"
    assert not skill["verified"]

    refused = client.post(
        f"/api/v1/sandbox-skills/{skill['id']}/activate", headers=auth_headers
    )
    assert refused.status_code == 409
    assert "control" in refused.json()["detail"]

    sandbox.on("sh skill/count.sh", writes_result({"loops": 1, "hottest": "a"}))
    ran = client.post(
        f"/api/v1/sandbox-skills/{skill['id']}/dry-run", headers=auth_headers
    )
    assert ran.status_code == 200 and ran.json()["ok"], ran.text
    assert ran.json()["skill"]["verified"]

    activated = client.post(
        f"/api/v1/sandbox-skills/{skill['id']}/activate", headers=auth_headers
    )
    assert activated.status_code == 200 and activated.json()["status"] == "active"

    listing = client.get("/api/v1/sandbox-skills", headers=auth_headers).json()
    assert [s["slug"] for s in listing["items"]] == ["loop_trip"]
    assert IMAGE in listing["images"]


def test_the_api_refuses_a_bad_skill_with_the_reason(client, auth_headers, allow_image):
    response = client.post(
        "/api/v1/sandbox-skills",
        json={"manifest": manifest(control=None)},
        headers=auth_headers,
    )
    assert response.status_code == 422
    assert "control is required" in response.json()["detail"]


def test_another_users_skill_does_not_exist(
    client, auth_headers, admin_headers, allow_image
):
    created = client.post(
        "/api/v1/sandbox-skills", json={"manifest": manifest()}, headers=auth_headers
    ).json()
    assert (
        client.get(
            f"/api/v1/sandbox-skills/{created['id']}", headers=admin_headers
        ).status_code
        == 404
    )


def test_only_an_administrator_may_build_an_image(
    client, auth_headers, allow_image, monkeypatch
):
    monkeypatch.setattr(
        sandbox_skill_image_service.agent_sandbox_runtime,
        "allowed_images",
        lambda: [IMAGE],
    )
    proposed = client.post(
        "/api/v1/sandbox-skills/images",
        json={"slug": "z3", "dockerfile": f"FROM {IMAGE}\nRUN true\n"},
        headers=auth_headers,
    )
    assert proposed.status_code == 201, proposed.text
    built = client.post(
        f"/api/v1/sandbox-skills/images/{proposed.json()['id']}/build",
        headers=auth_headers,
    )
    assert built.status_code == 403
