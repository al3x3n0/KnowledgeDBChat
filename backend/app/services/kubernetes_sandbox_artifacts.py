"""Persistent, per-invocation workspaces shared by workers and Kubernetes Jobs."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import stat
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from app.services.sandbox_workspace import open_to_sandbox_user

MAX_ENTRIES = 20_000


def files(root: Path, limit: int) -> dict:
    """Hash bounded regular files; never follow links out of a sandbox."""
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Sandbox workspace must be a real directory")
    result, size = {}, 0
    for index, path in enumerate(root.rglob("*")):
        if index >= MAX_ENTRIES:
            raise ValueError(
                "Sandbox artifact entry count exceeds the configured limit"
            )
        mode = path.lstat().st_mode
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            raise ValueError(
                "Sandbox artifacts must contain only directories and regular files"
            )
        size += path.stat().st_size
        if size > limit:
            raise ValueError("Sandbox artifact size exceeds the configured limit")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        result[path.relative_to(root).as_posix()] = {
            "sha256": digest.hexdigest(),
            "mode": stat.S_IMODE(mode) & 0o777,
        }
    return result


def save(path: Path, record: dict) -> None:
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(record, stream, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


@contextmanager
def _workspace_lock(root: Path, workspace: Path):
    key = hashlib.sha256(str(workspace.resolve()).encode()).hexdigest()
    with (root / f".workspace-{key}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def prepare(
    root: Path,
    name: str,
    source: Path,
    request: dict,
    limit: int,
    *,
    replay_states: tuple = (),
    workspace_key: int | None = None,
) -> tuple:
    """Publish inputs before submitting a Job; concurrent callers share a record."""
    if source.resolve().is_relative_to(root.resolve()) or root.resolve().is_relative_to(
        source.resolve()
    ):
        raise ValueError(
            "Sandbox source workspace must not overlap the persistent artifact root"
        )
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if root.is_symlink():
        raise ValueError("Sandbox artifact root must not be a symlink")
    with _workspace_lock(root, source):
        return _prepare_locked(
            root,
            name,
            source,
            request,
            limit,
            replay_states=replay_states,
            workspace_key=workspace_key,
        )


def _prepare_locked(
    root,
    name,
    source,
    request,
    limit,
    *,
    replay_states,
    workspace_key,
) -> tuple:
    manifest = files(source, limit)
    request_hash = hashlib.sha256(
        json.dumps(request, sort_keys=True).encode()
    ).hexdigest()
    fingerprint = hashlib.sha256(
        json.dumps({"request": request, "inputs": manifest}, sort_keys=True).encode()
    ).hexdigest()
    directory = root / name
    with (root / f".{name}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if directory.exists():
            record = json.loads((directory / "record.json").read_text())
            if record["fingerprint"] != fingerprint:
                # A worker may have restored outputs before dying between the
                # tool return and its journal completion. Skill setup also
                # rewrites its control files and removes its last result.
                inputs = record["inputs"]
                variants = (
                    [record["outputs"]]
                    if isinstance(record.get("outputs"), dict)
                    else []
                )
                if record["status"] in ("completed", "failed"):
                    # Only a validated continuation can supply snapshots from
                    # a later step of this same action. Never use live files
                    # from the remote workspace as replay authorization.
                    variants.extend(replay_states)
                # Atomic file installs may leave a mix of recorded versions
                # after a worker exits, including during an earlier-step replay.
                states = [inputs, *variants]
                known = (
                    bool(variants)
                    and all(
                        any(entry == state.get(path) for state in states)
                        for path, entry in manifest.items()
                    )
                    and all(
                        path in manifest or any(path not in state for state in variants)
                        for path in inputs
                    )
                )
                if record.get("request_hash") != request_hash or not known:
                    raise ValueError(
                        "Sandbox invocation was retried with different inputs or settings"
                    )
            return directory, record
        temporary = Path(tempfile.mkdtemp(prefix=".prepare-", dir=root))
        try:
            shutil.copytree(source, temporary / "work", symlinks=True)
            shutil.copytree(source, temporary / "inputs", symlinks=True)
            if (
                files(temporary / "work", limit) != manifest
                or files(temporary / "inputs", limit) != manifest
            ):
                raise ValueError("Sandbox inputs changed while being snapshotted")
            control = temporary / "control"
            control.mkdir(mode=0o755)
            (temporary / "io").mkdir(mode=0o777)
            (temporary / "io").chmod(0o777)
            shutil.copyfile(
                Path(__file__).with_name("kubernetes_sandbox_launcher.py"),
                control / "launcher.py",
            )
            (control / "launcher.py").chmod(0o644)
            (control / "request.json").write_text(json.dumps(request))
            (control / "request.json").chmod(0o644)
            open_to_sandbox_user(str(temporary / "work"))
            # The kubelet mounts subdirectories directly; only it traverses
            # this parent. The experiment never receives the metadata mount.
            record: dict[str, Any] = {
                "version": 1,
                "name": name,
                "fingerprint": fingerprint,
                "status": "prepared",
                "inputs": manifest,
                "request_hash": request_hash,
                "workspace": workspace_key,
            }
            save(temporary / "record.json", record)
            os.rename(temporary, directory)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    return directory, record


def restore(
    directory: Path, destination: Path, limit: int, *, expected: dict | None = None
) -> None:
    """Stage verified outputs, then atomically install each complete file."""
    source = directory / "work"
    if destination.is_symlink() or not destination.is_dir():
        raise ValueError("Sandbox output destination must be a real directory")
    root = directory.parent.resolve()
    if destination.resolve().is_relative_to(root) or root.is_relative_to(
        destination.resolve()
    ):
        raise ValueError(
            "Sandbox output destination must not overlap the artifact root"
        )
    with _workspace_lock(root, destination):
        outputs = files(source, limit)
        if expected is not None and outputs != expected:
            raise ValueError("Sandbox workspace no longer matches its recorded outputs")
        before = files(destination, limit)
        with tempfile.TemporaryDirectory(
            prefix=".kdbc-restore-", dir=destination.parent
        ) as temporary:
            # Keep the wrapper's fresh timestamp: source directory timestamps
            # can be old enough for the workspace retention sweep to prune.
            staging = Path(temporary) / "contents"
            if Path(temporary).stat().st_dev != destination.stat().st_dev:
                raise ValueError(
                    "Sandbox output staging must use the workspace filesystem"
                )
            shutil.copytree(source, staging, dirs_exist_ok=True, symlinks=True)
            if files(staging, limit) != outputs:
                raise ValueError("Sandbox recorded outputs changed while being staged")
            if files(destination, limit) != before:
                raise ValueError("Sandbox workspace changed during output staging")
            directories = sorted(
                (p.relative_to(staging) for p in staging.rglob("*") if p.is_dir()),
                key=lambda p: len(p.parts),
            )
            for relative in directories:
                target = destination / relative
                if target.is_file():
                    target.unlink()
                target.mkdir(exist_ok=True)
            for relative in sorted(outputs):
                target = destination / relative
                if target.is_dir():
                    shutil.rmtree(target)
                os.replace(staging / relative, target)
            # Defer ordinary deletions until all new files are in place. A
            # killed worker leaves whole, known versions rather than missing
            # required inputs or partially copied bytes.
            for relative in before.keys() - outputs.keys():
                target = destination / relative
                if target.is_file():
                    target.unlink()
            desired = set(directories)
            for path in sorted(
                destination.rglob("*"), key=lambda p: len(p.parts), reverse=True
            ):
                if path.is_dir() and path.relative_to(destination) not in desired:
                    path.rmdir()
            for relative in reversed(directories):
                shutil.copystat(staging / relative, destination / relative)
            shutil.copystat(staging, destination)
            if files(destination, limit) != outputs:
                raise ValueError(
                    "Restored sandbox workspace does not match recorded outputs"
                )


def log(directory: Path, name: str, limit: int) -> str:
    path = directory / "io" / name
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        return ""
    with os.fdopen(descriptor, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError("Sandbox log must be a regular file")
        data = stream.read(limit + 1)
    text = data[:limit].decode("utf-8", "replace")
    return text + ("\n[output truncated]" if len(data) > limit else "")
