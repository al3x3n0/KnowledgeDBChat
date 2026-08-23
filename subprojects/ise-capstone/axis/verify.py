#!/usr/bin/env python3
"""Re-establish every claim in these bundles from source, or say why it cannot.

The point of a bundle is that nothing in it has to be taken on trust. Each
`.axisl` is the only source: the SMT semantics are regenerated from it and
hashed, and every obligation is re-solved. A recorded verdict that no longer
reproduces is a failure, and so is a missing tool -- reported as inconclusive
rather than quietly skipped, because "we could not check" and "it checks out"
must never print the same way.

    python3 verify.py                  # every bundle under bundles/
    python3 verify.py smlalb fselgt    # only the named ones
    python3 verify.py --hashes smlalb  # current hashes, for updating MANIFEST

Each bundle is a directory under `bundles/` holding one `.axisl`, a
`MANIFEST.json`, and a `proof/` of obligations. Needs `axis` and `z3` on PATH.
Exit 0 only when every obligation reproduced its recorded verdict; 1 if any
failed; 2 if anything could not be decided.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUNDLES = HERE / "bundles"

# Generous, because an obligation over IEEE-754 semantics can be far slower
# than the same shape over bit-vectors -- an `fma` obligation here runs orders
# of magnitude longer than the integer ones.
TIMEOUT_S = 1800


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def emit_semantics(source: Path) -> bytes:
    result = subprocess.run(
        ["axis", "emit-smt2", str(source)], capture_output=True, timeout=TIMEOUT_S
    )
    if result.returncode != 0:
        raise RuntimeError(f"axis emit-smt2 failed: {result.stderr.decode()[:400]}")
    return result.stdout


def solve(semantics: bytes, obligation: Path) -> str:
    combined = semantics + b"\n" + obligation.read_bytes()
    try:
        result = subprocess.run(
            ["z3", "-in", f"-T:{TIMEOUT_S}"],
            input=combined,
            capture_output=True,
            timeout=TIMEOUT_S + 120,
        )
    except subprocess.TimeoutExpired:
        return "timeout"
    for line in result.stdout.decode().splitlines():
        token = line.strip()
        if token in ("unsat", "sat", "unknown", "timeout"):
            return token
    detail = (result.stdout.decode() + result.stderr.decode()).strip()[:160]
    return f"no verdict: {detail}"


def bundle_dirs(names: list[str]) -> list[Path]:
    if names:
        return [BUNDLES / name for name in names]
    return sorted(p for p in BUNDLES.iterdir() if (p / "MANIFEST.json").exists())


def check_bundle(path: Path) -> tuple[list[str], list[str]]:
    """Returns (failures, inconclusive) for one bundle."""
    failures: list[str] = []
    inconclusive: list[str] = []
    manifest = json.loads((path / "MANIFEST.json").read_text())
    source = path / manifest["source"]

    print(f"\n{path.name}  -- {manifest['candidate']}")

    actual = sha256(source.read_bytes())
    if actual != manifest["source_sha256"]:
        print(f"  FAIL  {source.name} does not match the manifest")
        print(f"        recorded {manifest['source_sha256']}")
        print(f"        actual   {actual}")
        return ([f"{path.name}: the description has changed"], [])
    print(f"  ok    {source.name} matches the manifest")

    try:
        semantics = emit_semantics(source)
    except Exception as exc:
        return ([f"{path.name}: could not regenerate semantics: {exc}"], [])

    if sha256(semantics) != manifest["semantics_sha256"]:
        # Not fatal alone: a newer axis may emit the same meaning differently.
        # The obligations are what settle it, so report and carry on.
        print("  note  emitted semantics differ from the recorded hash; "
              "the obligations still decide")
    else:
        print("  ok    emitted semantics match the manifest byte for byte")

    for entry in manifest["obligations"]:
        obligation = path / entry["file"]
        if not obligation.exists():
            failures.append(f"{path.name}/{entry['file']} is missing")
            continue
        if sha256(obligation.read_bytes()) != entry["sha256"]:
            failures.append(f"{path.name}/{entry['file']} has been modified")
            continue

        verdict = solve(semantics, obligation)
        if verdict == entry["expect"]:
            print(f"  ok    {entry['file']}: {verdict} -- {entry['claim']}")
        elif verdict in ("unknown", "timeout"):
            print(f"  ????  {entry['file']}: {verdict} (expected {entry['expect']})")
            inconclusive.append(f"{path.name}/{entry['file']} returned {verdict}")
        else:
            print(f"  FAIL  {entry['file']}: {verdict} (recorded {entry['expect']})")
            failures.append(
                f"{path.name}/{entry['file']} returned {verdict}, "
                f"recorded as {entry['expect']}"
            )
    return (failures, inconclusive)


def main() -> int:
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    want_hashes = "--hashes" in sys.argv

    if want_hashes:
        for path in bundle_dirs(argv):
            manifest = json.loads((path / "MANIFEST.json").read_text())
            source = path / manifest["source"]
            print(f"{path.name}:")
            print(f"  source_sha256:    {sha256(source.read_bytes())}")
            for entry in manifest["obligations"]:
                print(f"  {entry['file']}: {sha256((path / entry['file']).read_bytes())}")
            try:
                print(f"  semantics_sha256: {sha256(emit_semantics(source))}")
            except Exception as exc:
                print(f"  semantics: could not emit ({exc})")
        return 0

    absent = [tool for tool in ("axis", "z3") if shutil.which(tool) is None]
    if absent:
        print(f"INCONCLUSIVE  not installed: {', '.join(absent)}")
        print("  Nothing was re-checked. Install them and re-run.")
        return 2

    failures: list[str] = []
    inconclusive: list[str] = []
    for path in bundle_dirs(argv):
        bundle_failures, bundle_inconclusive = check_bundle(path)
        failures += bundle_failures
        inconclusive += bundle_inconclusive

    print()
    if failures:
        print(f"FAILED  {len(failures)} claim(s) did not reproduce:")
        for line in failures:
            print(f"  - {line}")
        return 1
    if inconclusive:
        print(f"INCONCLUSIVE  {len(inconclusive)} obligation(s) undecided:")
        for line in inconclusive:
            print(f"  - {line}")
        return 2
    print("All claims reproduced from source.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
