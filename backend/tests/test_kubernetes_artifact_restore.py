"""Interrupted restores must preserve replayable input/output file versions."""

import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from pathlib import Path
from threading import Event

import pytest

from app.services import kubernetes_sandbox_artifacts as artifacts

pytestmark = pytest.mark.unit
LIMIT = 1024 * 1024


@pytest.fixture
def completed(tmp_path):
    root, destination = tmp_path / "records", tmp_path / "workspace"
    destination.mkdir()
    for name, value in {
        "input.txt": "before",
        "keep.txt": "keep",
        "obsolete.txt": "old",
    }.items():
        (destination / name).write_text(value)
    request = {"script": "update input and remove obsolete"}
    directory, record = artifacts.prepare(root, "step", destination, request, LIMIT)
    (directory / "work" / "input.txt").write_text("after")
    (directory / "work" / "obsolete.txt").unlink()
    (directory / "work" / "generated.txt").write_text("new")
    record.update(
        status="completed", outputs=artifacts.files(directory / "work", LIMIT)
    )
    artifacts.save(directory / "record.json", record)
    return root, destination, directory, record, request


def test_failed_staging_copy_leaves_the_original_workspace_untouched(
    completed, monkeypatch
):
    _, destination, directory, _, _ = completed
    before = artifacts.files(destination, LIMIT)

    def interrupted_copy(source, target, **kwargs):
        target = Path(target)
        target.mkdir(parents=True, exist_ok=True)
        (target / "input.txt").write_text("partial bytes")
        raise OSError("copy interrupted")

    monkeypatch.setattr(artifacts.shutil, "copytree", interrupted_copy)
    with pytest.raises(OSError, match="copy interrupted"):
        artifacts.restore(directory, destination, LIMIT)
    assert artifacts.files(destination, LIMIT) == before


def test_recorded_outputs_are_verified_before_replacing_inputs(completed):
    _, destination, directory, record, _ = completed
    before = artifacts.files(destination, LIMIT)
    (directory / "work" / "input.txt").write_text("unrecorded change")
    with pytest.raises(ValueError, match="recorded outputs"):
        artifacts.restore(directory, destination, LIMIT, expected=record["outputs"])
    assert artifacts.files(destination, LIMIT) == before


def test_killed_worker_leaves_only_known_file_versions_and_can_resume(completed):
    root, destination, directory, record, request = completed
    script = """
import os, sys
from pathlib import Path
from app.services import kubernetes_sandbox_artifacts as artifacts
directory, destination = map(Path, sys.argv[1:])
original = artifacts.os.replace
def interrupted(source, target):
    original(source, target)
    if Path(target) == destination / 'input.txt':
        os._exit(77)
artifacts.os.replace = interrupted
artifacts.restore(directory, destination, 1024 * 1024)
"""
    worker = subprocess.run(
        [sys.executable, "-c", script, str(directory), str(destination)],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        check=False,
    )
    assert worker.returncode == 77, worker.stderr
    current = artifacts.files(destination, LIMIT)
    for name, value in current.items():
        assert value in (record["inputs"].get(name), record["outputs"].get(name))
    _, recovered = artifacts.prepare(root, "step", destination, request, LIMIT)
    artifacts.restore(directory, destination, LIMIT, expected=recovered["outputs"])
    assert artifacts.files(destination, LIMIT) == record["outputs"]
    assert not (destination / "obsolete.txt").exists()


def test_input_snapshot_waits_for_a_restore_in_the_same_workspace(
    completed, monkeypatch
):
    root, destination, directory, record, request = completed
    entered, release, attempting = Event(), Event(), Event()
    copytree = artifacts.shutil.copytree

    def paused_copy(source, target, **kwargs):
        if Path(source) == directory / "work":
            entered.set()
            assert release.wait(timeout=5)
        return copytree(source, target, **kwargs)

    monkeypatch.setattr(artifacts.shutil, "copytree", paused_copy)

    def prepare():
        attempting.set()
        return artifacts.prepare(root, "step", destination, request, LIMIT)

    with ThreadPoolExecutor(max_workers=2) as workers:
        restoring = workers.submit(artifacts.restore, directory, destination, LIMIT)
        try:
            assert entered.wait(timeout=5)
            preparing = workers.submit(prepare)
            assert attempting.wait(timeout=5)
            with pytest.raises(TimeoutError):
                preparing.result(timeout=0.05)
        finally:
            release.set()
        restoring.result(timeout=5)
        preparing.result(timeout=5)
    assert artifacts.files(destination, LIMIT) == record["outputs"]


@pytest.mark.parametrize("file_to_directory", [False, True])
def test_restore_replays_file_and_directory_type_changes(tmp_path, file_to_directory):
    root, destination = tmp_path / "records", tmp_path / "workspace"
    destination.mkdir()
    if file_to_directory:
        (destination / "node").write_text("old")
    else:
        (destination / "node").mkdir()
        (destination / "node" / "old.txt").write_text("old")
    request = {"script": "change node type"}
    directory, record = artifacts.prepare(root, "step", destination, request, LIMIT)
    node = directory / "work" / "node"
    if file_to_directory:
        node.unlink()
        node.mkdir()
        (node / "new.txt").write_text("new")
    else:
        (node / "old.txt").unlink()
        node.rmdir()
        node.write_text("new")
    record.update(
        status="completed", outputs=artifacts.files(directory / "work", LIMIT)
    )
    artifacts.save(directory / "record.json", record)
    artifacts.restore(directory, destination, LIMIT, expected=record["outputs"])
    assert artifacts.files(destination, LIMIT) == record["outputs"]
    artifacts.prepare(root, "step", destination, request, LIMIT)
