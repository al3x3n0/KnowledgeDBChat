"""A skill's judge can advance the workspace before its result is checkpointed."""

import json
import subprocess
from copy import deepcopy
from types import SimpleNamespace

import pytest

from app.services import agent_sandbox_runtime as sandbox
from app.services import kubernetes_sandbox as runner
from app.services import sandbox_skill_runtime as skills
from app.services.sandbox_invocation import invocation
from tests.test_kubernetes_sandbox import IMAGE, FakeAPI

pytestmark = pytest.mark.unit


@pytest.fixture
async def judged(monkeypatch, tmp_path):
    monkeypatch.setattr(sandbox, "execution_enabled", lambda: True)
    monkeypatch.setattr(skills, "_execution_enabled", lambda: True)
    config = runner.Configuration("experiments", "data", tmp_path / "records")
    api = FakeAPI(config)
    executed = []

    def execute(name):
        directory = config.root / name
        request = json.loads((directory / "control" / "request.json").read_text())
        executed.append(request["script"])
        result = subprocess.run(
            ["/bin/sh", "-c", request["script"]],
            cwd=directory / "work",
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        (directory / "io" / "stdout").write_bytes(result.stdout)
        (directory / "io" / "stderr").write_bytes(result.stderr)

    async def remote(script, workdir, **kwargs):
        return await runner.run(
            script, workdir, memory="2048m", cpus="2", config=config, api=api, **kwargs
        )

    monkeypatch.setattr(api, "write_outputs", execute)
    monkeypatch.setattr(skills, "_run_in_sandbox", remote)
    workdir = tmp_path / "shared-skill"
    workdir.mkdir()
    (workdir / "input.txt").write_text("original input")
    manifest = {
        "image": IMAGE,
        "timeout_seconds": 30,
        "files": {
            "judge.sh": "test -f candidate.bin; printf checked > candidate.bin; "
            "echo judged > judge-report.txt; printf '{\"score\":7}' > result.json"
        },
        "judge_command": "sh skill/judge.sh",
        "result": {"fields": {"score": "number"}},
    }
    command = "printf candidate > candidate.bin; printf '{\"score\":999}' > result.json"

    async def run(reference=None, *, command_override=None, key="same-action"):
        with invocation(
            "agent", key, defer=True, resume_reference=reference
        ) as context:
            result = await skills.execute(
                manifest,
                workdir=workdir,
                command=command_override or command,
                collect_result=True,
            )
            return result, context.runs

    first, references = await run()
    assert first.result == {"score": 7} and first.judged_by == "judge_command"
    return SimpleNamespace(
        run=run,
        api=api,
        workdir=workdir,
        config=config,
        executed=executed,
        reference=json.loads(json.dumps(references[-1])),
        references=references,
    )


async def test_resume_after_judge_restore_reuses_both_jobs(judged):
    s = judged
    assert (s.workdir / "candidate.bin").read_text() == "checked"
    # The worker dies before recording the tool result. The saved continuation
    # still names the judge, while its outputs are already in the shared folder.
    result, references = await s.run(s.reference)
    assert result.result == {"score": 7} and result.judged_by == "judge_command"
    assert [r["name"] for r in references] == [r["name"] for r in s.references]
    assert s.api.posts == 2 and len(s.executed) == 2
    assert (s.workdir / "judge-report.txt").read_text() == "judged\n"


async def test_interrupted_earlier_step_restore_can_replay_mixed_known_versions(
    judged, monkeypatch
):
    from pathlib import Path

    from app.services import kubernetes_sandbox_artifacts as artifacts

    class WorkerInterrupted(BaseException):
        pass

    s = judged
    original = artifacts.os.replace

    def interrupted(source, target):
        original(source, target)
        if Path(target) == s.workdir / "candidate.bin":
            raise WorkerInterrupted()

    with monkeypatch.context() as patch:
        patch.setattr(artifacts.os, "replace", interrupted)
        with pytest.raises(WorkerInterrupted):
            await s.run(s.reference)
    # Candidate is already rewound, while the later judge's report is still
    # present. Both versions are recorded for this same action/workspace.
    assert (s.workdir / "candidate.bin").read_text() == "candidate"
    assert (s.workdir / "judge-report.txt").exists()
    result, _ = await s.run(s.reference)
    assert result.result == {"score": 7} and s.api.posts == 2


@pytest.mark.parametrize("change", ["input", "extra", "command"])
async def test_a_later_step_does_not_authorize_unrecorded_changes(judged, change):
    s = judged
    options = {}
    if change == "input":
        (s.workdir / "input.txt").write_text("unrecorded input")
    elif change == "extra":
        (s.workdir / "external.txt").write_text("unrecorded file")
    else:
        options["command_override"] = "echo changed command"
    with pytest.raises(ValueError, match="different inputs or settings"):
        await s.run(s.reference, **options)
    assert s.api.posts == 2


@pytest.mark.parametrize(
    "field", ["name", "namespace", "fingerprint", "uid", "step", "workspace"]
)
async def test_replay_cannot_borrow_another_continuations_workspace(judged, field):
    reference = deepcopy(judged.reference)
    reference[field] = 3 if field in ("step", "workspace") else "different"
    with pytest.raises(ValueError, match="continuation reference"):
        await judged.run(reference)
    assert judged.api.posts == 2


async def test_advanced_workspace_requires_a_trusted_continuation(judged):
    with pytest.raises(ValueError, match="different inputs or settings"):
        await judged.run()
    assert judged.api.posts == 2


async def test_resume_derives_the_record_path_from_identity(judged):
    reference = dict(judged.reference, path="/not/a/trusted/record/path")
    result, _ = await judged.run(reference)
    assert result.result == {"score": 7} and judged.api.posts == 2


async def test_reference_cannot_be_reused_by_a_different_action(judged):
    with pytest.raises(ValueError, match="continuation reference"):
        await judged.run(judged.reference, key="different-action")
    assert judged.api.posts == 2


async def test_a_different_working_directory_cannot_authorize_changed_inputs(judged):
    s = judged
    first, second = [s.workdir.parent / name for name in ("first", "second")]
    for directory in (first, second):
        directory.mkdir()
        (directory / "input.txt").write_text(directory.name)

    async def run(directory, command):
        return await runner.run(
            command,
            str(directory),
            image=IMAGE,
            timeout_seconds=30,
            memory="2048m",
            cpus="2",
            config=s.config,
            api=s.api,
        )

    with invocation("agent", "separate-workspaces") as context:
        await run(first, "printf done-one > input.txt")
        await run(second, "printf done-two > input.txt")
        reference = deepcopy(context.runs[-1])
    # A value recorded for the other directory is not a valid input change.
    (first / "input.txt").write_text("done-two")
    with invocation("agent", "separate-workspaces", resume_reference=reference):
        with pytest.raises(ValueError, match="different inputs or settings"):
            await run(first, "printf done-one > input.txt")
    assert s.api.posts == 4


async def test_a_completed_step_can_replay_an_empty_workspace(judged):
    s = judged
    directory = s.workdir.parent / "empty"
    directory.mkdir()
    (directory / "input.txt").write_text("remove me")
    for _ in range(2):
        with invocation("agent", "empty-workspace"):
            assert (
                await runner.run(
                    "rm input.txt",
                    str(directory),
                    image=IMAGE,
                    timeout_seconds=30,
                    memory="2048m",
                    cpus="2",
                    config=s.config,
                    api=s.api,
                )
            )[0] == 0
        assert list(directory.iterdir()) == []
    assert s.api.posts == 3
