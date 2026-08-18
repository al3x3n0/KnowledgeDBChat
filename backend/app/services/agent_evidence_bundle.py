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
            # The result with volatile fields removed: what a repeat run has
            # to match, since a differing timestamp is not a differing result.
            "canonical_sha256": canonical_digest(payload),
            "artifact_dir": str(entry_dir.relative_to(directory)),
            "spilled_files": spilled,
        }
        with (directory / MANIFEST_NAME).open("a") as handle:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
        # Refreshed per entry so a run that dies partway still leaves a bundle
        # that says what it is and can check itself.
        write_verifier(job_id, root)
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


VERIFIER_NAME = "verify.py"
README_NAME = "README.md"

VERIFIER_SOURCE = '''#!/usr/bin/env python3
"""Check this bundle against its own manifest.

Two different questions, and this answers only the first:

  integrity   -- are the artifacts the ones this run produced?
  reproduction -- does re-running the recorded calls produce them again?

Integrity needs nothing but python3 and these files. Reproduction needs the
images the manifest pins and the tools that made the calls; the manifest holds
what that requires -- tool, parameters, image id, and the hash to compare
against.

Exit status is 0 when every artifact matches, 1 otherwise.
"""

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest = HERE / "manifest.jsonl"
    if not manifest.exists():
        print("no manifest.jsonl: this is not a bundle")
        return 1

    entries = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    checked = mismatched = missing = 0
    for entry in entries:
        directory = HERE / entry["artifact_dir"]
        for name, expected in (
            ("params.json", entry.get("params_sha256")),
            ("result.json", entry.get("result_sha256")),
        ):
            path = directory / name
            if not path.exists():
                print(f"MISSING  #{entry['sequence']:>4} {entry['tool']}/{name}")
                missing += 1
                continue
            actual = digest(path)
            checked += 1
            if actual != expected:
                print(f"CHANGED  #{entry['sequence']:>4} {entry['tool']}/{name}")
                print(f"           recorded {expected}")
                print(f"           actual   {actual}")
                mismatched += 1

    succeeded = sum(1 for e in entries if e.get("succeeded"))
    unpinned = sum(1 for e in entries if not e.get("image_id"))
    print()
    print(f"calls recorded    : {len(entries)} ({succeeded} succeeded, {len(entries) - succeeded} failed)")
    print(f"artifacts checked : {checked}")
    print(f"unpinned entries  : {unpinned}")
    if mismatched or missing:
        print(f"FAILED: {mismatched} changed, {missing} missing")
        return 1
    print("OK: every artifact matches the manifest")
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''

README_TEMPLATE = """# Evidence bundle for job {job_id}

Written while the run happened, one entry per evidence-producing tool call, so
this describes what actually ran rather than what was reconstructed afterwards.

## What is here

- `manifest.jsonl` — one line per call, in order: the tool, whether it
  succeeded, the image it ran in (pinned by id, since a tag moves), and the
  sha256 of its recorded parameters and result.
- `artifacts/NNNN-<tool>/params.json` — exactly what the call was given.
- `artifacts/NNNN-<tool>/result.json` — exactly what it returned.
- `artifacts/NNNN-<tool>/*.txt` — large outputs given their own file, such as
  an assembly listing or a proof obligation, so they can be read and diffed
  directly.

Failed calls are here too. A bundle showing only what worked would describe a
run that did not happen, and a claim resting on a measurement that never
succeeded is only detectable if the failure was kept.

## Checking it

```
python3 {verifier}
```

That recomputes every artifact hash and compares it with the manifest. It
proves the artifacts are the ones this run produced; it does not re-execute
anything, so it does not prove they can be produced again. Re-execution needs
the pinned images and the tools that made the calls -- the manifest records
which image id each call used, and the hash any repeat run has to match.

## Summary at time of writing

- calls recorded: {entries} ({succeeded} succeeded, {failed} failed)
- tools: {tools}
- images: {images}
- entries with no image pin: {unpinned}
"""


def write_verifier(job_id: str, root: Optional[Path] = None) -> Optional[Path]:
    """Refresh the bundle's verifier and README.

    Rewritten on every entry rather than at the end, so a bundle is complete
    and self-describing at any moment -- including when a run dies partway,
    which is exactly when its evidence matters most.
    """
    try:
        directory = bundle_dir(job_id, root)
        if not directory.exists():
            return None
        verifier = directory / VERIFIER_NAME
        verifier.write_text(VERIFIER_SOURCE, encoding="utf-8")
        verifier.chmod(0o755)

        summary = summarize(job_id, root)
        (directory / README_NAME).write_text(
            README_TEMPLATE.format(
                job_id=job_id,
                verifier=VERIFIER_NAME,
                entries=summary["entries"],
                succeeded=summary["succeeded"],
                failed=summary["failed"],
                tools=", ".join(f"{k} x{v}" for k, v in summary["tools"].items())
                or "none",
                images=", ".join(summary["image_ids"]) or "none pinned",
                unpinned=summary["unpinned_entries"],
            ),
            encoding="utf-8",
        )
        return verifier
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(f"evidence bundle: could not write verifier: {exc}")
        return None


def verify_integrity(job_id: str, root: Optional[Path] = None) -> Dict[str, Any]:
    """Recompute artifact hashes and compare them with the manifest."""
    directory = bundle_dir(job_id, root)
    entries = read_manifest(job_id, root)
    checked = 0
    changed: List[str] = []
    missing: List[str] = []
    for entry in entries:
        entry_dir = directory / str(entry.get("artifact_dir") or "")
        for name, expected in (
            ("params.json", entry.get("params_sha256")),
            ("result.json", entry.get("result_sha256")),
        ):
            path = entry_dir / name
            label = f"{entry.get('sequence')}/{name}"
            if not path.exists():
                missing.append(label)
                continue
            checked += 1
            if _digest(path.read_bytes()) != expected:
                changed.append(label)
    return {
        "entries": len(entries),
        "artifacts_checked": checked,
        "changed": changed,
        "missing": missing,
        "intact": not changed and not missing,
    }


# Fields that differ between two identical runs and say nothing about whether
# the result reproduced.
VOLATILE_FIELDS = {
    "timestamp",
    "recorded_at",
    "_journal_invocation_id",
    "url",
    "elapsed_ms",
}

# Tools whose output is not expected to repeat exactly. A benchmark reports
# wall clock; two honest runs of it disagree, and calling that a reproduction
# failure would train a reader to ignore the report. Their entries are replayed
# and reported, but not judged by hash.
TIMING_DEPENDENT_TOOLS = {
    "benchmark_c_snippet",
    "execute_python",
    "run_command",
    "write_and_run_script",
}


def canonicalize(payload: Any) -> Any:
    """Drop the fields that vary between two identical runs."""
    if isinstance(payload, dict):
        return {
            key: canonicalize(value)
            for key, value in sorted(payload.items())
            if key not in VOLATILE_FIELDS
        }
    if isinstance(payload, list):
        return [canonicalize(item) for item in payload]
    return payload


def canonical_digest(result: Any) -> str:
    payload = result if isinstance(result, dict) else {"result": result}
    return _digest(
        json.dumps(canonicalize(payload), sort_keys=True, default=str).encode()
    )


async def replay_bundle(
    job_id: str,
    execute,
    *,
    root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Re-run each recorded call and report whether its result came back.

    ``execute`` is an async callable taking (tool, params) and returning the
    tool's result, so the caller supplies the machinery rather than this module
    reaching for it.

    Reproduction is judged on the canonical result -- the recorded one with
    volatile fields removed -- because a timestamp differing is not a failure
    to reproduce. Timing-dependent tools are replayed and reported but never
    judged: two honest runs of a benchmark disagree, and a report that called
    that a failure would teach a reader to ignore it.
    """
    entries = read_manifest(job_id, root)
    directory = bundle_dir(job_id, root)
    reproduced: List[int] = []
    differed: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    errored: List[Dict[str, Any]] = []

    for entry in entries:
        sequence = int(entry.get("sequence") or 0)
        tool = str(entry.get("tool") or "")
        if not entry.get("succeeded"):
            # Replaying a failure proves nothing about the evidence, and a
            # preflight rejection would just be rejected again.
            skipped.append(
                {"sequence": sequence, "tool": tool, "why": "did not succeed"}
            )
            continue
        params_path = directory / str(entry.get("artifact_dir") or "") / "params.json"
        if not params_path.exists():
            errored.append(
                {"sequence": sequence, "tool": tool, "why": "params missing"}
            )
            continue
        try:
            params = json.loads(params_path.read_text())
        except ValueError as exc:
            errored.append(
                {"sequence": sequence, "tool": tool, "why": f"params unreadable: {exc}"}
            )
            continue

        if tool in TIMING_DEPENDENT_TOOLS:
            skipped.append(
                {
                    "sequence": sequence,
                    "tool": tool,
                    "why": "reports wall clock; two honest runs disagree",
                }
            )
            continue

        try:
            result = await execute(tool, params)
        except Exception as exc:  # pragma: no cover - defensive
            errored.append({"sequence": sequence, "tool": tool, "why": str(exc)[:200]})
            continue

        expected = entry.get("canonical_sha256")
        actual = canonical_digest(result)
        if not expected:
            skipped.append(
                {
                    "sequence": sequence,
                    "tool": tool,
                    "why": "recorded before canonical hashing existed",
                }
            )
        elif actual == expected:
            reproduced.append(sequence)
        else:
            differed.append(
                {
                    "sequence": sequence,
                    "tool": tool,
                    "recorded": expected,
                    "actual": actual,
                }
            )

    judged = len(reproduced) + len(differed)
    return {
        "job_id": str(job_id),
        "entries": len(entries),
        "judged": judged,
        "reproduced": len(reproduced),
        "differed": differed,
        # Reported rather than hidden: a replay that judged three of twenty
        # calls should not read as a bundle that reproduced.
        "skipped": skipped,
        "errored": errored,
        "verdict": (
            "reproduced"
            if judged and not differed and not errored
            else "differed"
            if differed
            else "inconclusive"
        ),
    }
