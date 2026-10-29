"""A candidate is timed against a baseline beside it, and may be "cannot tell".

Two separate benchmarks are taken at different moments on a host whose speed
moves (repeated runs here varied by up to 44%), and a campaign concluded from
one trial of a candidate with no baseline at all. `compare_c_snippets` times
both in one container, alternating, and measures the host's noise from the
baseline's own trials.
"""

import shutil
import subprocess
import uuid

import pytest

from app.services import agent_compiler_sandbox as sandbox
from app.services import agent_paired_timing as paired

pytestmark = pytest.mark.unit


class TestTheVerdict:
    def test_a_clear_win(self):
        verdict = paired.judge(
            [300, 305, 298, 302, 301, 299], [600, 610, 605, 598, 602, 604]
        )
        assert verdict["verdict"] == "faster"
        assert 1.95 < verdict["speedup"] < 2.05
        assert verdict["faster_rounds"] == 6

    def test_a_clear_loss(self):
        verdict = paired.judge([900] * 6, [450, 452, 449, 451, 450, 448])
        assert verdict["verdict"] == "slower"

    def test_a_difference_inside_the_hosts_noise_is_not_one(self):
        """The baseline against itself moves 30%; a 10% gap is not evidence."""
        base = [500, 650, 520, 680, 510, 660, 530, 670]
        cand = [b * 0.9 for b in base]
        verdict = paired.judge(cand, base)
        assert verdict["verdict"] == "indistinguishable"
        assert "within the host's own noise" in verdict["reason"]
        assert verdict["noise_band"] > 0.2

    def test_rounds_that_disagree_do_not_carry_a_verdict(self):
        base = [500] * 8
        cand = [300, 300, 300, 300, 700, 700, 700, 300]
        verdict = paired.judge(cand, base)
        assert verdict["verdict"] == "indistinguishable"
        assert "disagree" in verdict["reason"]

    def test_too_few_rounds_cannot_be_judged(self):
        verdict = paired.judge([300, 300], [600, 600])
        assert verdict["verdict"] == "cannot_tell"
        assert "at least 5" in verdict["reason"]

    def test_a_workload_at_the_noise_floor_cannot_be_judged(self):
        verdict = paired.judge([20, 22, 19, 21, 20], [30, 45, 28, 50, 31])
        assert verdict["verdict"] == "cannot_tell"
        assert "250 ms" in verdict["reason"]

    def test_identical_trials_still_get_a_minimum_noise_band(self):
        """Two trials agreeing to the millisecond is luck, not precision."""
        verdict = paired.judge([505] * 6, [500] * 6)
        assert verdict["noise_band"] >= 0.02
        assert verdict["verdict"] == "indistinguishable"

    def test_the_order_alternates(self):
        assert paired.interleaved_order(4) == ["ab", "ba", "ab", "ba"]


def _stdout(a, b, *, load="2.0", cpus="8"):
    lines = []
    for x, y in zip(a, b):
        lines += [f"__a_ms__ {x}", f"__b_ms__ {y}"]
    lines += [
        "__out_a__",
        "checksum 42",
        "__out_end__",
        "__out_b__",
        "checksum 42",
        "__out_end__",
        f"__loadavg__ {load}",
        f"__cpus__ {cpus}",
    ]
    return "\n".join(lines)


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setattr(sandbox, "_execution_enabled", lambda: True)
    monkeypatch.setattr(sandbox, "_allowed_images", lambda: [sandbox.DEFAULT_IMAGE])


@pytest.mark.asyncio
class TestTheTool:
    async def _compare(self, monkeypatch, returncode=0, stdout="", stderr=""):
        seen = {}

        async def _fake_run(script, workdir, *, image, timeout_seconds):
            seen["script"] = script
            return returncode, stdout, stderr

        monkeypatch.setattr(sandbox, "_run", _fake_run)
        result = await sandbox.compare_c_snippets(
            code="int fastmod(void){return 1;} int main(void){return 0;}",
            baseline_code="int plainmod(void){return 1;} int main(void){return 0;}",
            label="fastmod",
            baseline_label="plain %",
            rounds=6,
        )
        return result, seen

    async def test_both_arms_and_the_comparison_are_recorded(
        self, monkeypatch, enabled
    ):
        result, seen = await self._compare(
            monkeypatch,
            stdout=_stdout(
                [300, 302, 299, 301, 300, 303], [600, 605, 598, 602, 601, 604]
            ),
        )

        assert result["success"]
        measurement_a, measurement_b, comparison = result["findings"]
        assert measurement_a["type"] == measurement_b["type"] == "benchmark_measurement"
        assert {measurement_a["subject"], measurement_b["subject"]} == {
            "fastmod",
            "plain %",
        }
        assert measurement_a["source"].startswith("int fastmod")
        assert measurement_b["source_sha256"] != measurement_a["source_sha256"]
        assert comparison["type"] == "benchmark_comparison"
        assert comparison["verdict"] == "faster"
        assert "2.00x" in comparison["title"]
        assert result["data"]["candidate_stdout"] == "checksum 42"

        script = seen["script"]
        assert script.index("arm a; arm b") < script.index("arm b; arm a")
        assert "exit 90" in script and "exit 92" in script

    async def test_a_baseline_that_does_not_compile_says_which(
        self, monkeypatch, enabled
    ):
        result, _ = await self._compare(
            monkeypatch, returncode=92, stderr="main.c:1: error: expected ';'"
        )
        assert not result["success"]
        assert result["error"].startswith("The baseline did not compile")

    async def test_a_program_that_fails_is_not_a_timing(self, monkeypatch, enabled):
        result, _ = await self._compare(monkeypatch, returncode=91, stderr="b exited 3")
        assert not result["success"]
        assert "b exited 3" in result["error"]

    async def test_same_labels_are_told_apart(self, monkeypatch, enabled):
        async def _fake_run(script, workdir, *, image, timeout_seconds):
            return 0, _stdout([300] * 5, [300] * 5), ""

        monkeypatch.setattr(sandbox, "_run", _fake_run)
        result = await sandbox.compare_c_snippets(
            code="int main(void){return 0;}",
            baseline_code="int main(void){return 1-1;}",
            label="kernel",
            baseline_label="kernel",
        )
        a, b, _ = result["findings"]
        assert a["subject"] != b["subject"], "two subjects, as a baseline requires"

    async def test_disabled_execution_runs_nothing(self, monkeypatch):
        monkeypatch.setattr(sandbox, "_execution_enabled", lambda: False)
        result = await sandbox.compare_c_snippets(code="x", baseline_code="y")
        assert "disabled" in result["error"]


BASE_IMAGE = "ghcr.io/al3x3n0/kdbc-sandbox-base:latest"


def _have_image() -> bool:
    if not shutil.which("docker"):
        return False
    probe = subprocess.run(
        ["docker", "image", "inspect", BASE_IMAGE],
        capture_output=True,
        timeout=20,
    )
    return probe.returncode == 0


@pytest.mark.skipif(not _have_image(), reason=f"needs Docker and {BASE_IMAGE}")
def test_the_script_alternates_and_times_on_a_real_shell(tmp_path):
    """The shell is the part a stand-in cannot check. Each "source" here is a
    script and the "build" copies it to prog, so the real loop, timer, early
    stop and output capture run in the sandbox's own shell."""
    for arm, seconds, text in (("a", "0.05", "fast"), ("b", "0.15", "slow")):
        (tmp_path / arm).mkdir()
        (tmp_path / arm / "main.c").write_text(
            f"#!/bin/sh\nsleep {seconds}\necho {text}\n", encoding="utf-8"
        )
    script = sandbox._paired_script("cp main.c prog && chmod +x prog", 6, 60)
    name = f"paired-test-{uuid.uuid4().hex[:8]}"
    run = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--name",
            name,
            "--network",
            "none",
            "-v",
            f"{tmp_path}:/work",
            "-w",
            "/work",
            BASE_IMAGE,
            "sh",
            "-c",
            script,
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert run.returncode == 0, run.stderr

    order = [
        line[2]
        for line in run.stdout.splitlines()
        if line.startswith(("__a_ms__ ", "__b_ms__ "))
    ]
    assert order == ["a", "b", "b", "a", "a", "b", "b", "a", "a", "b", "b", "a"]
    a = [int(x.split()[1]) for x in run.stdout.splitlines() if x.startswith("__a_ms__")]
    b = [int(x.split()[1]) for x in run.stdout.splitlines() if x.startswith("__b_ms__")]
    assert all(x < y for x, y in zip(a, b))
    assert "__out_a__\nfast" in run.stdout and "__out_b__\nslow" in run.stdout


@pytest.mark.skipif(not _have_image(), reason=f"needs Docker and {BASE_IMAGE}")
def test_a_failing_program_stops_the_script(tmp_path):
    for arm, body in (("a", "exit 0"), ("b", "exit 3")):
        (tmp_path / arm).mkdir()
        (tmp_path / arm / "main.c").write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
    script = sandbox._paired_script("cp main.c prog && chmod +x prog", 5, 60)
    run = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "-v",
            f"{tmp_path}:/work",
            "-w",
            "/work",
            BASE_IMAGE,
            "sh",
            "-c",
            script,
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert run.returncode == 91
    assert "b exited 3" in run.stderr
