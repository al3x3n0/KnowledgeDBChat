"""Deterministic sandbox runs replayed from disk, and the runs that never are.

The stand-in for the sandbox runs the script with ``/bin/sh`` in the working
directory and counts its calls; the cache, its keys, its tarballs and its
trimming are the real ones, in a temporary directory.
"""

import asyncio
import io
import json
import tarfile
from pathlib import Path

import pytest

from app.core.config import settings
from app.services import agent_evidence_bundle, agent_sandbox_runtime
from app.services import sandbox_result_cache as cache

pytestmark = pytest.mark.unit

SCRIPT = "mkdir -p m5out && cat workload.c | wc -c > m5out/stats.txt && echo done"


@pytest.fixture
def sandbox(monkeypatch, tmp_path):
    calls = []
    image = {"id": "sha256:aaa"}

    async def run(script, workdir, *, image, timeout_seconds, memory, cpus):
        calls.append(script)
        process = await asyncio.create_subprocess_exec(
            "/bin/sh",
            "-c",
            script,
            cwd=workdir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, err = await process.communicate()
        return process.returncode, out.decode(), err.decode()

    async def image_id(name):
        return image["id"]

    monkeypatch.setattr(agent_sandbox_runtime, "run_in_sandbox", run)
    monkeypatch.setattr(cache, "image_id", image_id)
    monkeypatch.setattr(settings, "SANDBOX_RESULT_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(settings, "SANDBOX_RESULT_CACHE_ENABLED", True)
    return {"calls": calls, "image": image, "tmp": tmp_path}


def workdir(tmp, name, source="int main(){}"):
    path = tmp / name
    path.mkdir()
    (path / "workload.c").write_text(source)
    return str(path)


async def simulate(path, script=SCRIPT):
    return await cache.run(script, path, image="gem5", timeout_seconds=60)


class TestReplay:
    async def test_the_same_run_is_read_back_with_its_files(self, sandbox):
        first = await simulate(workdir(sandbox["tmp"], "a"))
        second_dir = workdir(sandbox["tmp"], "b")
        second = await simulate(second_dir)

        assert len(sandbox["calls"]) == 1
        assert first[:3] == second[:3] and (first[3], second[3]) == (False, True)
        assert Path(second_dir, "m5out", "stats.txt").read_text().strip() == "12"

    async def test_a_changed_input_is_a_different_run(self, sandbox):
        await simulate(workdir(sandbox["tmp"], "a"))
        await simulate(workdir(sandbox["tmp"], "b", "int main(){return 1;}"))

        assert len(sandbox["calls"]) == 2

    async def test_a_rebuilt_image_is_a_different_simulator(self, sandbox):
        await simulate(workdir(sandbox["tmp"], "a"))
        sandbox["image"]["id"] = "sha256:bbb"
        await simulate(workdir(sandbox["tmp"], "b"))

        assert len(sandbox["calls"]) == 2

    async def test_a_changed_script_is_a_different_run(self, sandbox):
        await simulate(workdir(sandbox["tmp"], "a"))
        await simulate(workdir(sandbox["tmp"], "b"), SCRIPT + " && true")

        assert len(sandbox["calls"]) == 2


class TestWhatIsNeverReplayed:
    async def test_a_failure_is_not_stored(self, sandbox):
        await simulate(workdir(sandbox["tmp"], "a"), "exit 125")
        await simulate(workdir(sandbox["tmp"], "b"), "exit 125")

        assert len(sandbox["calls"]) == 2

    async def test_an_image_the_daemon_cannot_name_is_not_cached(self, sandbox):
        sandbox["image"]["id"] = None
        await simulate(workdir(sandbox["tmp"], "a"))
        await simulate(workdir(sandbox["tmp"], "b"))

        assert len(sandbox["calls"]) == 2

    async def test_bypassed_runs_for_real(self, sandbox):
        await simulate(workdir(sandbox["tmp"], "a"))
        with cache.bypassed():
            again = await simulate(workdir(sandbox["tmp"], "b"))

        assert len(sandbox["calls"]) == 2 and again[3] is False

    async def test_disabled_runs_for_real(self, sandbox, monkeypatch):
        monkeypatch.setattr(settings, "SANDBOX_RESULT_CACHE_ENABLED", False)
        await simulate(workdir(sandbox["tmp"], "a"))
        await simulate(workdir(sandbox["tmp"], "b"))

        assert len(sandbox["calls"]) == 2

    def test_no_wall_clock_tool_uses_the_cache(self):
        """Two honest timings disagree; replaying one reports a run nobody made.

        Core's users only: the gem5 modules that use the cache live in the
        uarch_lab research program, which pins its own.
        """
        app = Path(__file__).resolve().parents[1] / "app" / "services"
        users = {
            p.name
            for p in app.rglob("*.py")
            if "sandbox_result_cache" in p.read_text(encoding="utf-8")
        }

        assert users == {
            "sandbox_result_cache.py",
            "agent_evidence_bundle.py",
        }


class TestTheStore:
    async def test_a_member_outside_the_working_directory_is_skipped(self, sandbox):
        path = workdir(sandbox["tmp"], "a")
        before = cache.snapshot(path)
        key = cache.cache_key("sha256:aaa", SCRIPT, "2g", "2", before)
        entry = cache._entry(key)
        entry.mkdir(parents=True)
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
            data = b"owned"
            info = tarfile.TarInfo("../escaped.txt")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        (entry / "outputs.tar.gz").write_bytes(buffer.getvalue())
        (entry / "meta.json").write_text(
            json.dumps({"returncode": 0, "stdout": "", "stderr": ""})
        )

        cache._restore(entry, path)

        assert not (sandbox["tmp"] / "escaped.txt").exists()

    async def test_the_least_recently_used_go_first(self, sandbox, monkeypatch):
        for name in ("a", "b", "c"):
            await simulate(workdir(sandbox["tmp"], name, f"/* {name} */"))
        entries = sorted(cache.root().glob("??/*/meta.json"))
        assert len(entries) == 3
        monkeypatch.setattr(settings, "SANDBOX_RESULT_CACHE_MAX_MB", 0)

        cache.trim()

        assert list(cache.root().glob("??/*/meta.json")) == []


class TestEvidenceReplay:
    async def test_a_bundle_replay_never_reads_the_cache(self, monkeypatch, tmp_path):
        seen = []

        async def execute(tool, params):
            seen.append(cache._BYPASS.get())
            return {"success": True}

        directory = tmp_path / "job"
        (directory / "call").mkdir(parents=True)
        (directory / "call" / "params.json").write_text("{}")
        monkeypatch.setattr(
            agent_evidence_bundle,
            "read_manifest",
            lambda job_id, root=None: [
                {
                    "sequence": 1,
                    "tool": "simulate_c_workload",
                    "succeeded": True,
                    "artifact_dir": "call",
                }
            ],
        )
        monkeypatch.setattr(
            agent_evidence_bundle, "bundle_dir", lambda job_id, root=None: directory
        )

        await agent_evidence_bundle.replay_bundle("job", execute)

        assert seen == [True]

    def test_a_replayed_result_hashes_like_a_computed_one(self):
        computed = {"data": {"cycles": 10, "replayed_from_cache": False}}
        replayed = {"data": {"cycles": 10, "replayed_from_cache": True}}

        assert agent_evidence_bundle.canonical_digest(
            computed
        ) == agent_evidence_bundle.canonical_digest(replayed)
