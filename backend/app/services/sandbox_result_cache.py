"""Replaying a deterministic sandbox run instead of repeating it.

A gem5 simulation of the same binary, configuration and image produces the
same cycles every time; a study that simulates its baseline arm again, a
restarted stage, a swarm role re-deriving its partner's number, each pays
minutes for an answer that is already known. ucb-bar/chia keeps a node's
result on disk under a tag and replays it on request ("caching and bypass"),
and computes its expensive ground truth (Verilator golden runs) once. This is
that idea where it is safe here:

  OPT-IN PER CALL SITE, AND ONLY FOR DETERMINISTIC WORK. A wall-clock
  benchmark is never cached -- two honest runs disagree, and replaying one
  would report a measurement nobody took. The call sites are the gem5
  simulations, whose output is simulated cycles.

  KEYED ON EVERYTHING THE RESULT DEPENDS ON. The image's content id (not its
  tag: a rebuilt image is a different simulator, which is how a fix stayed
  "absent" two rebuilds after it was made), the script, the resource limits,
  and every file in the working directory by content hash. A change to any of
  them is a different key. A run whose inputs cannot be read, or are larger
  than `MAX_INPUT_BYTES`, is simply not cached.

  ONLY A SUCCESS IS STORED. A timeout, a daemon that is not listening, or an
  image that is missing is a fact about that moment, and replaying it would
  make an outage permanent.

  REPLAY IS SAID, AND VERIFICATION BYPASSES IT. The caller gets `hit` and
  puts it on the result (`replayed_from_cache`), and an evidence-bundle
  replay runs under `bypassed()`: replaying a bundle exists to show the
  simulator reproduces the number, and reading the number back from a cache
  would show nothing.

The cache is a directory of entries (`meta.json` and the files the run wrote,
as a tarball), shared by every process that mounts it -- on the stack, the
sandbox work volume -- and trimmed oldest-used-first to
`SANDBOX_RESULT_CACHE_MAX_MB`. Nothing in it is trusted beyond being data: a
tarball member that is not a plain file under the working directory is
skipped.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import io
import json
import os
import shutil
import tarfile
import tempfile
import time
import uuid
from contextvars import ContextVar
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

from loguru import logger

from app.core.config import settings
from app.services import agent_sandbox_runtime

ROOT_NAME = "kdbc-sandbox-result-cache"
#: Larger inputs are not hashed on every call; such a run is not cached.
MAX_INPUT_BYTES = 64 * 1024 * 1024
#: A run that wrote more than this is not stored.
MAX_OUTPUT_BYTES = 256 * 1024 * 1024
#: How long an image's id is believed in this process. Short, because
#: `make sandbox-load` replaces an image under a running worker.
IMAGE_ID_TTL_SECONDS = 60.0

_BYPASS: ContextVar[bool] = ContextVar("sandbox_result_cache_bypass", default=False)
_image_ids: Dict[str, Tuple[float, str]] = {}


@contextlib.contextmanager
def bypassed() -> Iterator[None]:
    """Run everything inside without reading or writing the cache."""
    token = _BYPASS.set(True)
    try:
        yield
    finally:
        _BYPASS.reset(token)


def root() -> Path:
    configured = str(getattr(settings, "SANDBOX_RESULT_CACHE_DIR", "") or "").strip()
    return Path(configured) if configured else Path(tempfile.gettempdir()) / ROOT_NAME


def enabled() -> bool:
    return bool(getattr(settings, "SANDBOX_RESULT_CACHE_ENABLED", True)) and not (
        _BYPASS.get()
    )


async def image_id(image: str) -> Optional[str]:
    """The image's content id from the daemon the sandbox uses, or None."""
    now = time.monotonic()
    known = _image_ids.get(image)
    if known and now - known[0] < IMAGE_ID_TTL_SECONDS:
        return known[1]
    try:
        process = await asyncio.create_subprocess_exec(
            "docker",
            "image",
            "inspect",
            "--format",
            "{{.Id}}",
            image,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        try:
            out, _ = await asyncio.wait_for(process.communicate(), timeout=30)
        except asyncio.TimeoutError:
            process.kill()
            return None
    except (FileNotFoundError, OSError):
        return None
    found = (out or b"").decode().strip()
    if process.returncode != 0 or not found:
        return None
    _image_ids[image] = (now, found)
    return found


def snapshot(workdir: str) -> Optional[Dict[str, str]]:
    """Every regular file under `workdir` by content hash, or None if too big."""
    hashes: Dict[str, str] = {}
    total = 0
    base = Path(workdir)
    for path in sorted(base.rglob("*")):
        if path.is_symlink() or not path.is_file():
            continue
        total += path.stat().st_size
        if total > MAX_INPUT_BYTES:
            return None
        hashes[path.relative_to(base).as_posix()] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    return hashes


def cache_key(
    image: str, script: str, memory: str, cpus: str, inputs: Dict[str, str]
) -> str:
    material = json.dumps(
        {
            "image": image,
            "script": script,
            "memory": memory,
            "cpus": cpus,
            "inputs": inputs,
        },
        sort_keys=True,
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _entry(key: str) -> Path:
    return root() / key[:2] / key


def _restore(entry: Path, workdir: str) -> Optional[Tuple[int, str, str]]:
    meta_path = entry / "meta.json"
    if not meta_path.is_file():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    base = Path(workdir).resolve()
    archive = entry / "outputs.tar.gz"
    if archive.is_file():
        with tarfile.open(archive, "r:gz") as tar:
            for member in tar.getmembers():
                target = (base / member.name).resolve()
                if not member.isfile() or base not in target.parents:
                    continue
                source = tar.extractfile(member)
                if source is None:
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read())
    os.utime(meta_path)  # most recently used, for trimming
    return int(meta["returncode"]), str(meta["stdout"]), str(meta["stderr"])


def _store(
    key: str,
    workdir: str,
    before: Dict[str, str],
    result: Tuple[int, str, str],
) -> None:
    after = snapshot(workdir)
    if after is None:
        return
    written = [p for p, digest in after.items() if before.get(p) != digest]
    size = sum(Path(workdir, p).stat().st_size for p in written)
    if size > MAX_OUTPUT_BYTES:
        return
    entry = _entry(key)
    if entry.exists():
        return
    staging = root() / f".staging-{uuid.uuid4().hex}"
    staging.mkdir(parents=True)
    try:
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
            for path in written:
                tar.add(Path(workdir, path), arcname=path, recursive=False)
        (staging / "outputs.tar.gz").write_bytes(buffer.getvalue())
        returncode, stdout, stderr = result
        (staging / "meta.json").write_text(
            json.dumps(
                {
                    "returncode": returncode,
                    "stdout": stdout,
                    "stderr": stderr,
                    "files": written,
                    "stored_at": time.time(),
                }
            ),
            encoding="utf-8",
        )
        entry.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.rename(staging, entry)
        except OSError:
            pass  # another process stored the same key first
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    trim()


def trim() -> None:
    """Delete the least recently used entries until the cache fits its budget."""
    budget = int(getattr(settings, "SANDBOX_RESULT_CACHE_MAX_MB", 2048)) * 1024 * 1024
    entries = []
    total = 0
    for meta in root().glob("??/*/meta.json"):
        entry = meta.parent
        size = sum(f.stat().st_size for f in entry.iterdir() if f.is_file())
        entries.append((meta.stat().st_mtime, size, entry))
        total += size
    for _used, size, entry in sorted(entries):
        if total <= budget:
            break
        shutil.rmtree(entry, ignore_errors=True)
        total -= size


async def run(
    script: str,
    workdir: str,
    *,
    image: str,
    timeout_seconds: int,
    memory: str = agent_sandbox_runtime.DEFAULT_MEMORY,
    cpus: str = agent_sandbox_runtime.DEFAULT_CPUS,
) -> Tuple[int, str, str, bool]:
    """`run_in_sandbox`, answered from the cache when the same run was made.

    Returns the usual (returncode, stdout, stderr) and whether it was a
    replay. Anything wrong with the cache itself runs the real thing.
    """
    key = None
    before: Optional[Dict[str, str]] = None
    if enabled():
        try:
            identity = await image_id(image)
            before = await asyncio.to_thread(snapshot, workdir) if identity else None
            if identity and before is not None:
                key = cache_key(identity, script, memory, cpus, before)
                hit = await asyncio.to_thread(_restore, _entry(key), workdir)
                if hit is not None:
                    return (*hit, True)
        except Exception as exc:
            logger.warning(f"sandbox result cache unreadable, running: {exc}")
            key = None

    result = await agent_sandbox_runtime.run_in_sandbox(
        script,
        workdir,
        image=image,
        timeout_seconds=timeout_seconds,
        memory=memory,
        cpus=cpus,
    )
    if key is not None and before is not None and result[0] == 0:
        try:
            await asyncio.to_thread(_store, key, workdir, before, result)
        except Exception as exc:
            logger.warning(f"sandbox result not cached: {exc}")
    return (*result, False)
