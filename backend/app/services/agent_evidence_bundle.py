"""Assemble a run's reproducibility bundle while the run happens.

A bundle built at the end depends on two things that have both failed in
practice: the run surviving to write it, and the agent remembering to. This one
is a byproduct of tool execution instead. Every evidence-producing call appends
its inputs, its outputs and the image it ran in, as it happens, whether or not
the agent ever thinks about the bundle.

Failed calls are recorded too. A run that fabricated a measurement -- citing
"llvm-mca reported 11.8 cycles per iteration" from a call that had failed --
would show that plainly here: the failure is in the manifest, and no artifact
carries the number the prediction claimed.

Everything is content-hashed and the image is pinned by id rather than tag, so
a bundle says which bytes produced which result and in what, not merely that
some version of something once ran.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

BUNDLE_ROOT = Path("/app/data/agent-bundles")
MANIFEST_NAME = "manifest.jsonl"
MAX_ARTIFACT_BYTES = 2_000_000

# Tools whose results are evidence about the machine rather than about the
# agent's reading. A bundle of every search result would bury the measurements
# that a reviewer has to check.
EVIDENCE_TOOLS = {
    "compile_c_snippet",
    "benchmark_c_snippet",
    "analyze_snippet_cycles",
    "profile_c_workload",
    "simulate_c_workload",
    "axis_check",
    "axis_emit",
    "axis_prove",
    "execute_python",
    "run_command",
    "write_and_run_script",
}

# Result fields big enough to deserve their own file, so a reviewer can diff an
# assembly listing or a proof obligation directly instead of extracting it from
# JSON.
SPILL_FIELDS = ("output", "artifact", "assembly", "stdout", "compiler_stderr")

_image_ids: Dict[str, str] = {}


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


async def resolve_image_id(image: str) -> str:
    """Pin the image by content id, since a tag moves and a bundle must not.

    Returns "" when the id cannot be read; an empty pin is honest, a guessed
    one is not.
    """
    if image in _image_ids:
        return _image_ids[image]
    try:
        process = await asyncio.create_subprocess_exec(
            "docker",
            "image",
            "inspect",
            "--format",
            "{{.Id}}",
            image,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(process.communicate(), timeout=15)
        identifier = (stdout or b"").decode().strip()
    except Exception:  # pragma: no cover - defensive
        identifier = ""
    if identifier:
        _image_ids[image] = identifier
    return identifier


def bundle_dir(job_id: str, root: Optional[Path] = None) -> Path:
    return (root or BUNDLE_ROOT) / str(job_id)


def _next_sequence(directory: Path) -> int:
    manifest = directory / MANIFEST_NAME
    if not manifest.exists():
        return 1
    with manifest.open() as handle:
        return sum(1 for line in handle if line.strip()) + 1


def record_entry(
    *,
    job_id: str,
    tool: str,
    params: Dict[str, Any],
    result: Any,
    image_id: str = "",
    root: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """Append one call to the bundle. Never raises; a bundle is not the run."""
    try:
        directory = bundle_dir(job_id, root)
        artifacts = directory / "artifacts"
        artifacts.mkdir(parents=True, exist_ok=True)

        sequence = _next_sequence(directory)
        entry_dir = artifacts / f"{sequence:04d}-{tool}"
        entry_dir.mkdir(parents=True, exist_ok=True)

        params_bytes = json.dumps(
            params, indent=2, default=str, sort_keys=True
        ).encode()
        (entry_dir / "params.json").write_bytes(params_bytes[:MAX_ARTIFACT_BYTES])

        payload = result if isinstance(result, dict) else {"result": result}
        data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
        spilled: List[str] = []
        for field in SPILL_FIELDS:
            value = data.get(field)
            if isinstance(value, str) and len(value) > 2000:
                name = f"{field}.txt"
                (entry_dir / name).write_text(
                    value[:MAX_ARTIFACT_BYTES], encoding="utf-8"
                )
                spilled.append(name)

        result_bytes = json.dumps(
            payload, indent=2, default=str, sort_keys=True
        ).encode()
        (entry_dir / "result.json").write_bytes(result_bytes[:MAX_ARTIFACT_BYTES])

        entry = {
            "sequence": sequence,
            "recorded_at": datetime.utcnow().isoformat(),
            "tool": tool,
            # Recorded rather than inferred from the presence of an error: a
            # reviewer counting successes must not have to guess the rule.
            "succeeded": bool(isinstance(result, dict) and result.get("success")),
            "error": (
                str(result.get("error"))[:300]
                if isinstance(result, dict) and result.get("error")
                else ""
            ),
            "image_id": image_id,
            "params_sha256": _digest(params_bytes),
            "result_sha256": _digest(result_bytes),
            "artifact_dir": str(entry_dir.relative_to(directory)),
            "spilled_files": spilled,
        }
        with (directory / MANIFEST_NAME).open("a") as handle:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
        return entry
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(f"evidence bundle: could not record {tool}: {exc}")
        return None


def read_manifest(job_id: str, root: Optional[Path] = None) -> List[Dict[str, Any]]:
    manifest = bundle_dir(job_id, root) / MANIFEST_NAME
    if not manifest.exists():
        return []
    entries = []
    with manifest.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except ValueError:
                continue
    return entries


def summarize(job_id: str, root: Optional[Path] = None) -> Dict[str, Any]:
    """Describe the bundle: what ran, what failed, and in which images."""
    entries = read_manifest(job_id, root)
    tools: Dict[str, int] = {}
    for entry in entries:
        name = str(entry.get("tool") or "?")
        tools[name] = tools.get(name, 0) + 1
    failed = [e for e in entries if not e.get("succeeded")]
    images = sorted(
        {str(e.get("image_id") or "") for e in entries if e.get("image_id")}
    )
    return {
        "job_id": str(job_id),
        "path": str(bundle_dir(job_id, root)),
        "entries": len(entries),
        "succeeded": len(entries) - len(failed),
        # Kept, not filtered: a bundle showing only what worked describes a run
        # that did not happen.
        "failed": len(failed),
        "tools": dict(sorted(tools.items())),
        "image_ids": images,
        "unpinned_entries": sum(1 for e in entries if not e.get("image_id")),
    }
