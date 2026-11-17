"""Job lifecycle, artifact isolation, and recovery at the Kubernetes boundary."""

import asyncio
import copy
import json
import resource
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from app.services import agent_sandbox_runtime as sandbox
from app.services import kubernetes_sandbox as runner
from app.services import kubernetes_sandbox_artifacts as artifacts
from app.services import kubernetes_sandbox_launcher as launcher
from app.services import sandbox_result_cache
from app.services.sandbox_invocation import current, invocation

pytestmark = pytest.mark.unit
IMAGE = "registry.example/tool@sha256:" + "a" * 64


class FakeAPI:
    def __init__(self, config):
        self.config = config
        self.jobs = {}
        self.posts = 0
        self.deletes = []
        self.delete_requests = []
        self.delete_pending = False
        self.complete = True
        self.exit_code = 0
        self.reason = "Completed"
        self.deadline = False
        self.conflict = False
        self.created = asyncio.Event()

    def response(self, method, path, code, data):
        return httpx.Response(
            code, json=data, request=httpx.Request(method, "https://cluster" + path)
        )

    async def request(self, method, path, **kwargs):
        if path.endswith("/pods"):
            uid = next(iter(self.jobs.values()))["metadata"]["uid"]
            return self.response(
                method,
                path,
                200,
                {
                    "items": [
                        {
                            "metadata": {
                                "labels": {"batch.kubernetes.io/controller-uid": uid}
                            },
                            "status": {
                                "containerStatuses": [
                                    {
                                        "name": "sandbox",
                                        "state": {
                                            "terminated": {
                                                "exitCode": self.exit_code,
                                                "reason": self.reason,
                                            }
                                        },
                                    }
                                ]
                            },
                        }
                    ]
                },
            )
        if method == "POST":
            self.posts += 1
            job = copy.deepcopy(kwargs["json"])
            name = job["metadata"]["name"]
            job["metadata"]["uid"] = "job-uid"
            self.jobs[name] = job
            self.created.set()
            self.write_outputs(name)
            return self.response(
                method, path, 409 if self.conflict else 201, self.job(name)
            )
        name = path.rsplit("/", 1)[-1]
        if method == "DELETE":
            self.delete_requests.append(kwargs["json"])
            expected_uid = kwargs["json"].get("preconditions", {}).get("uid")
            if name in self.jobs and expected_uid != self.jobs[name]["metadata"]["uid"]:
                return self.response(method, path, 409, {})
            self.deletes.append(name)
            if not self.delete_pending:
                self.jobs.pop(name, None)
            return self.response(method, path, 202 if self.delete_pending else 200, {})
        if name not in self.jobs:
            return self.response(method, path, 404, {})
        return self.response(method, path, 200, self.job(name))

    def write_outputs(self, name):
        directory = self.config.root / name
        (directory / "work" / "result.txt").write_text("measured result")
        (directory / "io" / "stdout").write_text("simulation done")
        (directory / "io" / "stderr").write_text("diagnostic")

    def job(self, name):
        job = copy.deepcopy(self.jobs[name])
        if self.complete:
            job["status"] = {
                "conditions": [
                    {
                        "type": "Complete"
                        if self.exit_code == 0 and not self.deadline
                        else "Failed",
                        "status": "True",
                        "reason": "DeadlineExceeded" if self.deadline else "",
                    }
                ]
            }
        return job


@pytest.fixture
def setup(monkeypatch, tmp_path):
    monkeypatch.setattr(sandbox, "execution_enabled", lambda: True)
    monkeypatch.setattr(sandbox, "allowed_images", lambda: [IMAGE])
    config = runner.Configuration(
        "experiments", "data", tmp_path / "persistent", poll_seconds=0.001
    )
    return config, FakeAPI(config), tmp_path


def workspace(root, name="source", code="input"):
    directory = root / name
    directory.mkdir()
    (directory / "input.txt").write_text(code)
    return directory


async def run(config, api, directory, **kwargs):
    return await runner.run(
        "echo simulation",
        str(directory),
        image=IMAGE,
        timeout_seconds=30,
        memory="2048m",
        cpus="2",
        config=config,
        api=api,
        **kwargs,
    )


async def test_completed_step_restores_outputs_without_resubmitting(setup):
    config, api, root = setup
    with invocation("agent", "intent") as first:
        assert await run(config, api, workspace(root)) == (
            0,
            "simulation done",
            "diagnostic",
        )
    api.jobs.clear()  # Kubernetes TTL cleanup does not erase recorded results.
    destination = workspace(root, "retry")
    with invocation("agent", "intent") as second:
        assert (await run(config, api, destination))[0] == 0
    assert (destination / "result.txt").read_text() == "measured result"
    assert first.runs == second.runs
    assert api.posts == 1


async def test_retry_after_outputs_restored_but_before_journal_completion(setup):
    config, api, root = setup
    directory = workspace(root)
    with invocation("agent", "intent"):
        await run(config, api, directory)
    # The workspace already contains outputs on a shared skill volume.
    with invocation("agent", "intent"):
        await run(config, api, directory)
    assert api.posts == 1


async def test_projected_service_account_token_is_reloaded(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "SERVICE_ACCOUNT", tmp_path)
    headers = []

    async def respond(request):
        headers.append(request.headers["Authorization"])
        return httpx.Response(200, json={})

    api = runner.KubernetesAPI.__new__(runner.KubernetesAPI)
    api.client = httpx.AsyncClient(
        base_url="https://cluster", transport=httpx.MockTransport(respond)
    )
    try:
        (tmp_path / "token").write_text("first")
        await api.request("GET", "/jobs")
        (tmp_path / "token").write_text("rotated")
        await api.request("GET", "/jobs")
    finally:
        await api.close()
    assert headers == ["Bearer first", "Bearer rotated"]


def test_skill_workspaces_and_python_call_names_survive_worker_replacement(
    monkeypatch, tmp_path
):
    from app.services import python_tool_container, sandbox_skill_runtime

    monkeypatch.setattr(sandbox.settings, "SANDBOX_EXECUTION_BACKEND", "kubernetes")
    monkeypatch.setattr(
        sandbox.settings, "KUBERNETES_SANDBOX_ROOT", str(tmp_path / "records")
    )
    assert sandbox_skill_runtime.root_dir() == tmp_path / "sandbox-workspaces"
    first = python_tool_container._prepare(
        workspace(tmp_path), "output=1", {}, "intent:1"
    )
    second = python_tool_container._prepare(
        workspace(tmp_path, "replacement"), "output=1", {}, "intent:1"
    )
    assert first.name == second.name


async def test_python_tool_recovery_reuses_remote_job_with_a_new_local_directory(
    setup, monkeypatch
):
    from app.services import python_tool_container

    config, api, root = setup
    monkeypatch.setattr(sandbox.settings, "SANDBOX_EXECUTION_BACKEND", "kubernetes")
    monkeypatch.setattr(sandbox.settings, "CUSTOM_TOOL_PYTHON_IMAGE", IMAGE)
    monkeypatch.setattr(sandbox.settings, "KUBERNETES_SANDBOX_ROOT", str(config.root))

    def outputs(name):
        api.write_outputs_original(name)
        call = next((config.root / name / "work" / ".pytool").iterdir())
        (call / "output.json").write_text(
            json.dumps({"ok": True, "value": {"score": 42}})
        )

    api.write_outputs_original = api.write_outputs
    monkeypatch.setattr(api, "write_outputs", outputs)

    async def remote(script, workdir, *, image, timeout_seconds):
        return await runner.run(
            script,
            workdir,
            image=image,
            timeout_seconds=timeout_seconds,
            memory="2048m",
            cpus="2",
            config=config,
            api=api,
        )

    monkeypatch.setattr(sandbox, "run_in_sandbox", remote)
    for _ in range(2):
        with invocation("agent", "intent"):
            assert await python_tool_container.run("output={'score':42}", {}) == {
                "score": 42
            }
    assert api.posts == 1


async def test_lost_worker_leaves_job_running_and_retry_reattaches(setup):
    config, api, root = setup
    api.complete = False

    async def first_worker():
        with invocation("agent", "intent"):
            await run(config, api, workspace(root))

    task = asyncio.create_task(first_worker())
    await api.created.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert api.deletes == []
    api.complete = True
    with invocation("agent", "intent"):
        assert (await run(config, api, workspace(root, "replacement")))[0] == 0
    assert api.posts == 1


async def test_explicit_owner_cancellation_deletes_job(setup):
    config, api, root = setup
    checks = 0

    async def cancelled():
        nonlocal checks
        checks += 1
        return checks > 1

    with invocation("agent", "intent", cancelled=cancelled):
        with pytest.raises(runner.SandboxJobCancelled):
            await run(config, api, workspace(root))
    assert len(api.deletes) == 1
    assert api.delete_requests[0]["preconditions"] == {"uid": "job-uid"}


@pytest.mark.parametrize("changed", ["uid", "fingerprint", "during_delete"])
async def test_cancellation_cannot_delete_a_replacement_job(
    setup, monkeypatch, changed
):
    config, api, _ = setup
    api.jobs["step"] = {
        "metadata": {
            "uid": "original",
            "annotations": {"kdbc.io/request-sha256": "fingerprint"},
        }
    }
    if changed == "uid":
        api.jobs["step"]["metadata"]["uid"] = "replacement"
    elif changed == "fingerprint":
        api.jobs["step"]["metadata"]["annotations"]["kdbc.io/request-sha256"] = "other"
    else:
        actual = api.request

        async def replace_before_delete(method, path, **kwargs):
            if method == "DELETE":
                api.jobs["step"]["metadata"]["uid"] = "replacement"
            return await actual(method, path, **kwargs)

        monkeypatch.setattr(api, "request", replace_before_delete)
    with pytest.raises(ValueError, match="identity changed"):
        await runner.cancel(
            api, config, "step", fingerprint="fingerprint", uid="original"
        )
    assert "step" in api.jobs and api.deletes == []


async def test_cancellation_reports_removal_only_after_foreground_deletion(setup):
    config, api, _ = setup
    api.jobs["step"] = {
        "metadata": {
            "uid": "original",
            "annotations": {"kdbc.io/request-sha256": "fingerprint"},
        }
    }
    api.delete_pending = True
    assert not await runner.cancel(api, config, "step", fingerprint="fingerprint")
    assert api.delete_requests[0]["preconditions"] == {"uid": "original"}
    api.jobs.clear()
    assert await runner.cancel(api, config, "step", fingerprint="fingerprint")
    assert len(api.deletes) == 1


async def test_retry_with_changed_inputs_is_rejected(setup):
    config, api, root = setup
    with invocation("agent", "intent"):
        await run(config, api, workspace(root))
    with invocation("agent", "intent"):
        with pytest.raises(ValueError, match="different inputs"):
            await run(config, api, workspace(root, "changed", "new input"))
    assert api.posts == 1


async def test_concurrent_submission_conflict_reattaches(setup):
    config, api, root = setup
    api.conflict = True
    with invocation("agent", "intent"):
        assert (await run(config, api, workspace(root)))[0] == 0
    assert api.posts == 1


async def test_missing_job_is_not_silently_executed_twice(setup):
    config, api, root = setup
    api.complete = False

    async def submit():
        with invocation("agent", "intent"):
            await run(config, api, workspace(root))

    task = asyncio.create_task(submit())
    await api.created.wait()
    # Let the runner record the Job UID before losing its worker.
    await asyncio.sleep(0.02)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    api.jobs.clear()
    with invocation("agent", "intent"):
        with pytest.raises(RuntimeError, match="disappeared"):
            await run(config, api, workspace(root, "retry"))
    assert api.posts == 1


async def test_deadline_is_reported_as_timeout(setup):
    config, api, root = setup
    api.deadline = True
    with pytest.raises(asyncio.TimeoutError):
        await run(config, api, workspace(root))


async def test_oom_and_failed_program_are_not_reported_as_success(setup):
    config, api, root = setup
    api.exit_code, api.reason = 137, "OOMKilled"
    result = await run(config, api, workspace(root))
    assert result[0] == 137
    assert "memory limit" in result[2]


async def test_disabled_execution_and_nonallowlisted_images_never_submit(
    setup, monkeypatch
):
    config, api, root = setup
    directory = workspace(root)
    monkeypatch.setattr(sandbox.settings, "SANDBOX_EXECUTION_BACKEND", "kubernetes")
    monkeypatch.setattr(sandbox, "execution_enabled", lambda: False)
    with pytest.raises(ValueError, match="ENABLE_UNSAFE"):
        await sandbox.run_in_sandbox(
            "true", str(directory), image=IMAGE, timeout_seconds=30
        )
    monkeypatch.setattr(sandbox, "execution_enabled", lambda: True)
    monkeypatch.setattr(sandbox, "allowed_images", lambda: [])
    with pytest.raises(ValueError, match="not allowlisted"):
        await sandbox.run_in_sandbox(
            "true", str(directory), image=IMAGE, timeout_seconds=30
        )
    assert api.posts == 0


def test_manifest_confines_experiment_and_profiles_resources(setup):
    config, _, _ = setup
    config.profiles = {
        "default": {"imagePullSecrets": [{"name": "registry"}]},
        "simulation": {
            "nodeSelector": {"workload": "simulation"},
            "runtimeClassName": "sandboxed",
            "extraResources": {"nvidia.com/gpu": "1"},
        },
    }
    config.image_profiles = {IMAGE: "simulation"}
    request = {
        "image": IMAGE,
        "timeout_seconds": 600,
        "resources": runner.resources("2g", "2", config),
    }
    job = runner.manifest("step", request, "fingerprint", config)
    pod = job["spec"]["template"]["spec"]
    container = pod["containers"][0]
    assert job["spec"]["backoffLimit"] == 0
    assert job["spec"]["activeDeadlineSeconds"] == 600
    assert "ownerReferences" not in job["metadata"]
    assert pod["automountServiceAccountToken"] is False
    assert pod["securityContext"]["runAsNonRoot"] is True
    assert container["securityContext"]["capabilities"]["drop"] == ["ALL"]
    assert container["securityContext"]["readOnlyRootFilesystem"] is True
    assert container["securityContext"]["allowPrivilegeEscalation"] is False
    assert not any("secret" in volume for volume in pod["volumes"])
    assert pod["runtimeClassName"] == "sandboxed"
    assert pod["imagePullSecrets"] == [{"name": "registry"}]
    assert container["resources"]["limits"]["nvidia.com/gpu"] == "1"
    assert all(
        m.get("subPath", "").startswith("kubernetes-sandbox/step/")
        for m in container["volumeMounts"]
        if m["name"] == "artifacts"
    )
    assert (
        "app.kubernetes.io/instance"
        not in job["spec"]["template"]["metadata"]["labels"]
    )


@pytest.mark.parametrize(
    "memory,cpus", [("1g", "NaN"), ("1g", "9"), ("999g", "1"), ("-1g", "1")]
)
def test_resource_limits_are_enforced_before_submission(setup, memory, cpus):
    with pytest.raises(ValueError):
        runner.resources(memory, cpus, setup[0])


def test_artifact_symlinks_and_oversized_inputs_are_rejected(tmp_path):
    directory = workspace(tmp_path)
    (directory / "escape").symlink_to(tmp_path)
    with pytest.raises(ValueError, match="regular files"):
        artifacts.files(directory, 1024)
    (directory / "escape").unlink()
    with pytest.raises(ValueError, match="size"):
        artifacts.files(directory, 1)


def test_logs_are_bounded_and_cannot_follow_symlinks(tmp_path):
    io = tmp_path / "io"
    io.mkdir()
    (io / "stdout").write_text("0123456789")
    assert artifacts.log(tmp_path, "stdout", 4) == "0123\n[output truncated]"
    (io / "stderr").symlink_to(io / "stdout")
    with pytest.raises(OSError):
        artifacts.log(tmp_path, "stderr", 4)


async def test_dispatch_selects_kubernetes_without_a_docker_client(monkeypatch):
    monkeypatch.setattr(sandbox.settings, "SANDBOX_EXECUTION_BACKEND", "kubernetes")
    monkeypatch.setattr(sandbox, "execution_enabled", lambda: True)
    monkeypatch.setattr(sandbox, "allowed_images", lambda: [IMAGE])

    async def remote(*args, **kwargs):
        return 0, "remote", ""

    monkeypatch.setattr(runner, "run", remote)
    assert await sandbox.run_in_sandbox(
        "true", "/w", image=IMAGE, timeout_seconds=10
    ) == (0, "remote", "")
    assert sandbox.could_not_run(125, "program failed", IMAGE) == ""
    assert await sandbox_result_cache.image_id("image:mutable") is None
    assert await sandbox_result_cache.image_id(IMAGE) == IMAGE


async def test_cache_hits_keep_following_step_identities_stable(setup, monkeypatch):
    config, api, root = setup
    monkeypatch.setattr(sandbox.settings, "SANDBOX_EXECUTION_BACKEND", "kubernetes")
    monkeypatch.setattr(sandbox_result_cache, "enabled", lambda: True)
    monkeypatch.setattr(sandbox_result_cache, "snapshot", lambda _: {})
    monkeypatch.setattr(
        sandbox_result_cache, "_restore", lambda *args: (0, "cached", "")
    )
    with invocation("agent", "intent") as context:
        result = await sandbox_result_cache.run(
            "simulate", str(workspace(root)), image=IMAGE, timeout_seconds=30
        )
        assert result == (0, "cached", "", True)
        assert context.calls == 1
        assert context.workspace_key(str(root / "another-workspace")) == 2
        assert context.next_key() == "agent:intent:2"


async def test_cached_step_preserves_inputs_for_the_next_kubernetes_job(
    setup, monkeypatch
):
    config, api, root = setup
    monkeypatch.setattr(sandbox.settings, "SANDBOX_EXECUTION_BACKEND", "kubernetes")
    monkeypatch.setattr(sandbox.settings, "SANDBOX_RESULT_CACHE_ENABLED", True)
    monkeypatch.setattr(
        sandbox.settings, "SANDBOX_RESULT_CACHE_DIR", str(root / "cache")
    )

    def execute(name):
        directory = config.root / name
        request = json.loads((directory / "control" / "request.json").read_text())
        result = subprocess.run(
            ["/bin/sh", "-c", request["script"]],
            cwd=directory / "work",
            capture_output=True,
            check=False,
        )
        api.exit_code = result.returncode
        (directory / "io" / "stdout").write_bytes(result.stdout)
        (directory / "io" / "stderr").write_bytes(result.stderr)

    async def remote(script, workdir, **kwargs):
        return await runner.run(script, workdir, config=config, api=api, **kwargs)

    monkeypatch.setattr(api, "write_outputs", execute)
    monkeypatch.setattr(sandbox, "run_in_sandbox", remote)
    script = "rm stale.txt; printf '#!/bin/sh\\necho ready\\n' > generated.sh; chmod 755 generated.sh"
    for attempt in range(2):
        directory = workspace(root, f"attempt-{attempt}")
        (directory / "stale.txt").write_text("old measurement")
        with invocation("agent", "same-action") as context:
            cached = await sandbox_result_cache.run(
                script, str(directory), image=IMAGE, timeout_seconds=30
            )
            assert cached[0] == 0 and cached[3] is bool(attempt)
            assert await remote(
                "test ! -e stale.txt && ./generated.sh",
                str(directory),
                image=IMAGE,
                timeout_seconds=30,
                memory="2048m",
                cpus="2",
            ) == (0, "ready\n", "")
            assert context.calls == 2
    assert api.posts == 2


async def test_action_context_exposes_job_references_to_existing_journal(
    setup, monkeypatch
):
    from app.services.agent_action_service import AgentActionService
    from app.services.agent_execution_journal_service import (
        agent_execution_journal_service,
    )

    config, api, root = setup
    recorded = []

    async def begin(**kwargs):
        kwargs["action"]["_idempotency_key"] = "intent"
        return {"idempotency_key": "intent"}

    async def complete(**kwargs):
        recorded.append(kwargs["result"])

    service = AgentActionService()

    async def act(*args):
        assert current.get().key == "intent"
        await run(config, api, workspace(root))
        return {"success": True}

    monkeypatch.setattr(agent_execution_journal_service, "begin_tool_call", begin)
    monkeypatch.setattr(agent_execution_journal_service, "complete_tool_call", complete)
    monkeypatch.setattr(service, "_act_unjournaled", act)
    executor = SimpleNamespace(
        _apply_default_scope_to_action=lambda action, job: action
    )
    result = await service.act(
        executor,
        SimpleNamespace(id="agent", status="running"),
        {},
        {},
        SimpleNamespace(refresh=AsyncMock()),
    )
    assert recorded[0]["artifacts"][0]["type"] == "kubernetes_sandbox"
    assert result["sandbox_jobs"][0]["namespace"] == "experiments"
    assert current.get() is None


@pytest.mark.parametrize(
    "script,timeout,expected",
    [
        ("echo out; echo err >&2; echo artifact > result.txt", 5, 0),
        ("sleep 10", 1, 124),
        ("exit 42", 5, 42),
    ],
)
def test_actual_launcher_captures_exit_logs_artifacts_and_deadline(
    tmp_path, script, timeout, expected
):
    control, output, work = [tmp_path / name for name in ("control", "output", "work")]
    for directory in (control, output, work):
        directory.mkdir()
    (control / "request.json").write_text(
        json.dumps(
            {
                "script": script,
                "timeout_seconds": timeout,
                # The local developer UID already owns unrelated processes.
                # Keep its existing process limit for this child-process test.
                "pids": resource.getrlimit(resource.RLIMIT_NPROC)[0],
                "max_file_bytes": 1024 * 1024,
            }
        )
    )
    module_path = str(Path(launcher.__file__).resolve())
    command = (
        "import importlib.util,sys; "
        "s=importlib.util.spec_from_file_location('launcher',sys.argv[1]); "
        "m=importlib.util.module_from_spec(s);s.loader.exec_module(m); "
        "sys.exit(m.main(*sys.argv[2:]))"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            command,
            module_path,
            str(control),
            str(output),
            str(work),
        ],
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == expected, result.stderr.decode()
    if expected == 0:
        assert (output / "stdout").read_text().strip() == "out"
        assert (output / "stderr").read_text().strip() == "err"
        assert (work / "result.txt").read_text().strip() == "artifact"
