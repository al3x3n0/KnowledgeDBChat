#!/usr/bin/env python3
"""Re-establish every claim in this bundle from its source, or say why it cannot.

The point of the bundle is that nothing here has to be taken on trust. The
`.axisl` is the only source: the SMT semantics are regenerated from it and
hashed, and each obligation is re-solved. A recorded verdict that no longer
reproduces is a failure, and so is a missing tool -- reported as inconclusive
rather than quietly skipped, because "we could not check" and "it checks out"
must never print the same way.

    python3 verify.py            # regenerate, re-solve, compare
    python3 verify.py --hashes   # print current hashes, for updating MANIFEST

Needs `axis` and `z3` on PATH. Exit status is 0 only when every obligation
reproduced its recorded verdict.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "MANIFEST.json"
SOURCE = HERE / "smlalb.axisl"
TIMEOUT_S = 600


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def emit_semantics() -> bytes:
    """Regenerate the SMT semantics from the description."""
    result = subprocess.run(
        ["axis", "emit-smt2", str(SOURCE)],
        capture_output=True,
        timeout=TIMEOUT_S,
    )
    if result.returncode != 0:
        raise RuntimeError(f"axis emit-smt2 failed: {result.stderr.decode()[:400]}")
    return result.stdout


def solve(semantics: bytes, obligation: Path) -> str:
    """Run one obligation against freshly emitted semantics."""
    combined = semantics + b"\n" + obligation.read_bytes()
    result = subprocess.run(
        ["z3", "-in", f"-T:{TIMEOUT_S}"],
        input=combined,
        capture_output=True,
        timeout=TIMEOUT_S + 60,
    )
    for line in result.stdout.decode().splitlines():
        token = line.strip()
        if token in ("unsat", "sat", "unknown"):
            return token
    detail = (result.stdout.decode() + result.stderr.decode()).strip()[:160]
    return f"no verdict: {detail}"


def missing_tools() -> list[str]:
    return [tool for tool in ("axis", "z3") if shutil.which(tool) is None]


def main() -> int:
    manifest = json.loads(MANIFEST.read_text())

    if "--hashes" in sys.argv:
        print(f"  source sha256: {sha256(SOURCE.read_bytes())}")
        for entry in manifest["obligations"]:
            path = HERE / entry["file"]
            print(f"  {entry['file']}: {sha256(path.read_bytes())}")
        try:
            print(f"  semantics sha256: {sha256(emit_semantics())}")
        except Exception as exc:  # the point of --hashes is to help, not to gate
            print(f"  semantics: could not emit ({exc})")
        return 0

    failures: list[str] = []
    inconclusive: list[str] = []

    # 1. The source itself, before anything derived from it.
    actual = sha256(SOURCE.read_bytes())
    if actual != manifest["source_sha256"]:
        failures.append(
            f"{SOURCE.name} does not match the manifest\n"
            f"      recorded {manifest['source_sha256']}\n"
            f"      actual   {actual}"
        )
        print("FAIL  the description has changed; nothing below would mean anything")
        print(f"  - {failures[0]}")
        return 1
    print(f"ok    {SOURCE.name} matches the manifest")

    # 2. Each obligation, unchanged, and re-solved against fresh semantics.
    absent = missing_tools()
    if absent:
        print(f"INCONCLUSIVE  not installed: {', '.join(absent)}")
        print("  The recorded verdicts were NOT re-checked. Install them and re-run.")
        return 2

    try:
        semantics = emit_semantics()
    except Exception as exc:
        print(f"FAIL  could not regenerate semantics from {SOURCE.name}: {exc}")
        return 1

    emitted = sha256(semantics)
    if emitted != manifest["semantics_sha256"]:
        # Not fatal by itself: a newer axis may emit the same meaning in a
        # different shape. The obligations below are what settle it, so this is
        # reported and the run continues.
        print(
            "note  emitted semantics differ from the recorded hash "
            f"(recorded {manifest['semantics_sha256'][:16]}..., "
            f"got {emitted[:16]}...); the obligations still decide"
        )
    else:
        print("ok    emitted semantics match the manifest byte for byte")

    for entry in manifest["obligations"]:
        path = HERE / entry["file"]
        if not path.exists():
            failures.append(f"{entry['file']} is missing")
            continue
        if sha256(path.read_bytes()) != entry["sha256"]:
            failures.append(f"{entry['file']} has been modified since it was recorded")
            continue

        verdict = solve(semantics, path)
        expected = entry["expect"]
        if verdict == expected:
            print(f"ok    {entry['file']}: {verdict} ({entry['claim']})")
        elif verdict == "unknown":
            inconclusive.append(f"{entry['file']} returned unknown (expected {expected})")
        else:
            failures.append(
                f"{entry['file']} returned {verdict}, recorded as {expected}"
            )

    print()
    if failures:
        print(f"FAILED  {len(failures)} claim(s) did not reproduce:")
        for line in failures:
            print(f"  - {line}")
        return 1
    if inconclusive:
        print(f"INCONCLUSIVE  {len(inconclusive)} obligation(s) could not be decided:")
        for line in inconclusive:
            print(f"  - {line}")
        return 2
    print("All claims reproduced from source.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
