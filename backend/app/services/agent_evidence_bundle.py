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
    # The baseline a tuning claim is measured against. A bundle that records
    # the tuned run but not the model it started from cannot show what the
    # tuning changed.
    "describe_model_parameters",
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
_image_details: Dict[str, Dict[str, Any]] = {}

IMAGES_NAME = "images.json"

# Where each sandbox image comes from, so a pinned id is something a reader can
# act on rather than an opaque hash. Rebuilding is not the same as loading the
# exact image -- packages move -- which is why the export path exists too.
IMAGE_ORIGINS = {
    "kdbc-compiler-research": {
        "dockerfile": "deploy/sandbox-images/compiler-research/Dockerfile",
        "context": "that directory",
    },
    "kdbc-microarch-research": {
        "dockerfile": "deploy/sandbox-images/microarch-research/Dockerfile",
        "context": "that directory",
    },
    "kdbc-profiling-research": {
        "dockerfile": "deploy/sandbox-images/profiling-research/Dockerfile",
        "context": "that directory",
    },
    "kdbc-axis-research": {
        "dockerfile": "deploy/sandbox-images/axis-research/Dockerfile",
        "context": "the AXIS repository, not this one",
    },
    "kdbc-gem5-research": {
        "dockerfile": "deploy/sandbox-images/README.md (built from gem5 source)",
        "context": "see the README: the published gem5 image is not used",
    },
}


def image_origin(reference: str) -> Dict[str, str]:
    """How to rebuild an image, when it is one of ours."""
    for name, origin in IMAGE_ORIGINS.items():
        if name in (reference or ""):
            return dict(origin)
    return {}


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


async def describe_image(image: str) -> Dict[str, Any]:
    """Read what a bundle needs to say about the image a call ran in.

    The id pins it, the size says what obtaining it costs, and the origin says
    how to get it. An empty result is honest; a guessed one is not.
    """
    if image in _image_details:
        return _image_details[image]
    details: Dict[str, Any] = {}
    try:
        process = await asyncio.create_subprocess_exec(
            "docker",
            "image",
            "inspect",
            "--format",
            "{{.Id}}\t{{.Size}}\t{{.Created}}",
            image,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(process.communicate(), timeout=15)
        parts = (stdout or b"").decode().strip().split("\t")
        if parts and parts[0]:
            details = {
                "reference": image,
                "id": parts[0],
                "size_bytes": int(parts[1])
                if len(parts) > 1 and parts[1].isdigit()
                else 0,
                "created": parts[2] if len(parts) > 2 else "",
                "origin": image_origin(image),
            }
    except Exception:  # pragma: no cover - defensive
        details = {}
    if details:
        _image_details[image] = details
        _image_ids[image] = details["id"]
    return details


async def resolve_image_id(image: str) -> str:
    """Pin the image by content id, since a tag moves and a bundle must not."""
    if image in _image_ids:
        return _image_ids[image]
    details = await describe_image(image)
    return str(details.get("id") or "")


def write_images_manifest(job_id: str, root: Optional[Path] = None) -> Optional[Path]:
    """Describe every image this bundle depends on, and how to obtain it."""
    try:
        directory = bundle_dir(job_id, root)
        if not directory.exists():
            return None
        used = {
            str(e.get("image_id"))
            for e in read_manifest(job_id, root)
            if e.get("image_id")
        }
        images = [dict(d) for d in _image_details.values() if d.get("id") in used]
        for image in images:
            image["obtain"] = (
                "docker load -i images/<file>.tar (exact), or rebuild from "
                f"{image['origin']['dockerfile']} using {image['origin']['context']} "
                "(equivalent, not identical: packages move)"
                if image.get("origin")
                else "not a known project image; obtain it from wherever it came from"
            )
        payload = {
            "images": images,
            # Named rather than implied: a bundle whose images cannot be
            # obtained is checkable but not reproducible, and should say so.
            "note": (
                "These ids are what the calls ran in. Verifying artifact "
                "integrity needs none of them; replaying the calls needs all "
                "of them, and they are not included here -- see the README."
            ),
            "unknown_image_ids": sorted(used - {str(i.get("id")) for i in images}),
        }
        path = directory / IMAGES_NAME
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return path
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(f"evidence bundle: could not write images manifest: {exc}")
        return None


async def export_images(
    job_id: str,
    destination: Path,
    *,
    root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Save the images this bundle used, so it can be replayed elsewhere.

    Deliberately not automatic: these images run to hundreds of megabytes each,
    and writing them beside every bundle would make the evidence too heavy to
    keep. Exporting is a decision about shipping a bundle, taken once.
    """
    destination.mkdir(parents=True, exist_ok=True)
    used = {
        str(e.get("image_id")) for e in read_manifest(job_id, root) if e.get("image_id")
    }
    exported: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []

    for details in _image_details.values():
        if details.get("id") not in used:
            continue
        reference = str(details.get("reference") or details["id"])
        safe = reference.replace("/", "_").replace(":", "_")
        target = destination / f"{safe}.tar"
        try:
            process = await asyncio.create_subprocess_exec(
                "docker",
                "save",
                "-o",
                str(target),
                reference,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await asyncio.wait_for(process.communicate(), timeout=1800)
            if process.returncode != 0 or not target.exists():
                failed.append(
                    {"reference": reference, "why": (stderr or b"").decode()[:200]}
                )
                continue
            exported.append(
                {
                    "reference": reference,
                    "id": details["id"],
                    "file": target.name,
                    "bytes": target.stat().st_size,
                    "sha256": _digest(target.read_bytes()),
                }
            )
        except Exception as exc:  # pragma: no cover - defensive
            failed.append({"reference": reference, "why": str(exc)[:200]})

    (destination / "load.sh").write_text(
        "#!/bin/sh\n"
        "# Load the images this bundle's calls ran in, then replay it.\n"
        "set -e\n"
        + "".join(f'docker load -i "$(dirname "$0")/{e["file"]}"\n' for e in exported),
        encoding="utf-8",
    )
    (destination / "load.sh").chmod(0o755)
    (destination / "exported.json").write_text(
        json.dumps({"exported": exported, "failed": failed}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {
        "destination": str(destination),
        "exported": exported,
        "failed": failed,
        "total_bytes": sum(e["bytes"] for e in exported),
    }


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
        write_images_manifest(job_id, root)
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

## Reproducing it elsewhere

`images.json` lists every image these calls ran in, pinned by id. The images
themselves are not here: they run to hundreds of megabytes each, and a bundle
carrying them would be too heavy to keep. Two ways to obtain them:

- **exact** — if the bundle was shipped with an `images/` directory, run
  `sh images/load.sh`, which loads the saved images by id. Replaying then
  compares like with like.
- **equivalent** — rebuild from the Dockerfile each entry names. That gives an
  image that does the same job, not the same image: base layers and packages
  move, so a replay may differ for reasons that have nothing to do with the
  result being studied.

Without either, this bundle can still be checked for integrity. It cannot be
replayed, and it says so rather than implying otherwise.

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
