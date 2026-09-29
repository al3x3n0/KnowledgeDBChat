"""Scan C sources for places a known optimisation applies, and for new ones.

The passes in `/opt/llvm-passes` were reachable from a shell and from nowhere
else: they run under `opt`, and every tool here runs `clang`. So the scanner
that found 179 hoistable divisions in raylib was a capability one person could
use, which is the shape of gap this project keeps finding in its own code. This
is the tool that closes it.

TWO ANSWERS, KEPT APART.

`opportunities` are sites a pass in this image already handles, so the only
question is how many and where. Each names the pass that applies, taken from
`agent_toolchains.LLVM_PASSES` rather than restated, so a suggestion cannot
name a pass that is not there.

`shapes` is a tally of what feeds each expensive operation. A frequent shape
with no pass is a candidate for writing one -- which is where
loop-invariant-reciprocal came from, and it is the half that finds the next
optimisation rather than counting the last.

WHAT THIS DOES NOT MEASURE. Every count is static: how often a pattern is
written, not how often it runs. This project has already had a candidate
ranking invert when dynamic counts replaced source occurrences, so the result
is a list of places to look. Ranking needs a profile, and the caveat travels
on the finding because that is what survives into the record.
"""

from __future__ import annotations

import asyncio
import re
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from app.services import agent_sandbox_runtime
from app.services.agent_toolchains import LLVM_PASSES

DEFAULT_IMAGE = "ghcr.io/al3x3n0/kdbc-compiler-research:latest"
DEFAULT_TIMEOUT_SECONDS = 300
SCANNER = "/opt/llvm-passes/OptScan.so"

#: One translation unit can be large; a whole codebase passed as text cannot.
#: The cap refuses an absurd payload rather than defining what may be scanned.
MAX_TOTAL_CHARS = 4_000_000
MAX_SOURCES = 64

#: A source name is written into the sandbox and interpolated into a shell
#: command, so it is a bare filename and nothing else.
SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]{1,80}\.c$")
SAFE_FLAGS = re.compile(r"^[-A-Za-z0-9_=+., /]*$")

#: Which pass handles which opportunity the scanner reports. Read from the
#: declaration the tool descriptions use, so a suggestion cannot name a pass
#: this image does not carry.
_BY_NAME = {p.name: p for p in LLVM_PASSES}


def _preflight(
    sources: Dict[str, str], flags: str, image: str
) -> Optional[Dict[str, Any]]:
    if not sources:
        return {"error": "sources is required: a mapping of filename to C source"}
    if len(sources) > MAX_SOURCES:
        return {"error": f"at most {MAX_SOURCES} sources per scan, got {len(sources)}"}
    total = sum(len(v or "") for v in sources.values())
    if total > MAX_TOTAL_CHARS:
        return {
            "error": f"sources total {total} characters, over the {MAX_TOTAL_CHARS} cap"
        }
    for name in sources:
        if not SAFE_NAME.match(str(name or "")):
            return {
                "error": (
                    f"source name {name!r} is not usable: give a bare .c filename "
                    "with no directory part, e.g. 'raymath.c'"
                )
            }
    if not SAFE_FLAGS.match(flags or ""):
        return {"error": f"flags contain unsupported characters: {flags!r}"}
    if not agent_sandbox_runtime.execution_enabled():
        return {
            "error": "Sandboxed execution is disabled (ENABLE_UNSAFE_CODE_EXECUTION is false)."
        }
    if image not in agent_sandbox_runtime.allowed_images():
        return {
            "error": (
                f"Image {image} is not allowlisted. Allowed: "
                f"{', '.join(agent_sandbox_runtime.allowed_images()) or 'none'}"
            )
        }
    return None


def parse_records(text: str) -> Dict[str, Any]:
    """Turn the scanner's tab-separated output into the shape a caller reads."""
    known: Counter = Counter()
    declined: Counter = Counter()
    where: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    groups: Counter = Counter()
    shapes: Dict[str, Counter] = defaultdict(Counter)
    hoistable = 0
    loop_divisions = 0

    for line in (text or "").splitlines():
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 4 or parts[0] != "OPTSCAN":
            continue
        kind = parts[1]
        if kind == "known" and len(parts) >= 5:
            opp, fn, count = parts[2], parts[3], int(parts[4])
            known[opp] += count
            where[opp].append({"function": fn, "sites": count})
            for extra in parts[5:]:
                if extra.startswith("group_size="):
                    groups[int(extra.split("=", 1)[1])] += count
        elif kind == "declined" and len(parts) >= 5:
            declined[parts[2]] += int(parts[4])
        elif kind == "candidate" and len(parts) >= 5:
            hoistable += int(parts[4])
            for extra in parts[5:]:
                if extra.startswith("of_loop_divisions="):
                    loop_divisions += int(extra.split("=", 1)[1])
            if int(parts[4]):
                where["loop-invariant-reciprocal"].append(
                    {"function": parts[3], "sites": int(parts[4])}
                )
                known["loop-invariant-reciprocal"] += int(parts[4])
        elif kind == "shape" and len(parts) >= 5:
            shapes[parts[2]][parts[3]] += int(parts[4])

    return {
        "known": dict(known),
        "declined": dict(declined),
        "where": {
            k: sorted(v, key=lambda d: -d["sites"])[:12] for k, v in where.items()
        },
        "divisor_group_sizes": {str(k): v for k, v in sorted(groups.items())},
        "loop_divisions": loop_divisions,
        "loop_invariant_divisions": hoistable,
        "shapes": {k: dict(v.most_common(10)) for k, v in shapes.items()},
    }


def _suggestions(parsed: Dict[str, Any]) -> List[Dict[str, Any]]:
    """One entry per opportunity, naming the pass and whether it is safe."""
    out: List[Dict[str, Any]] = []
    for opp, sites in sorted(parsed["known"].items(), key=lambda kv: -kv[1]):
        spec = _BY_NAME.get(opp)
        out.append(
            {
                "pass": opp,
                "sites": sites,
                "declined": parsed["declined"].get(opp, 0),
                "flag": f"-fpass-plugin={spec.path}" if spec else None,
                "value_preserving": spec.value_preserving if spec else None,
                "top_functions": parsed["where"].get(opp, [])[:6],
            }
        )
    return out


async def scan_for_optimizations(
    *,
    sources: Dict[str, str],
    flags: str = "-O1",
    label: str = "",
    image: str = DEFAULT_IMAGE,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    """Report where known optimisations apply, and what shapes have no pass."""
    blocked = _preflight(sources, flags, image)
    if blocked:
        return blocked

    with tempfile.TemporaryDirectory(prefix="optscan_") as workdir:
        for name, text in sources.items():
            Path(workdir, name).write_text(text or "", encoding="utf-8")
        return await _scan_in(
            workdir,
            list(sources),
            flags,
            label=label,
            image=image,
            timeout_seconds=timeout_seconds,
        )


def _scan_script(paths: List[str], flags: str) -> str:
    """Compile each path to IR and run the scanner over it.

    Each file's IR goes beside it as <path>.ll, and a failure records the
    first compiler error for that file rather than one shared log: across a
    real repository, "3 of 12 failed" is only actionable with each reason.
    """
    listed = " ".join(f"'{p}'" for p in paths)
    return (
        "ok=0; "
        f"for f in {listed}; do "
        f'  if clang {flags} -S -emit-llvm -o "$f.ll" "$f" 2>err.txt; then '
        "    ok=$((ok+1)); "
        f"    opt -load-pass-plugin={SCANNER} -passes=opt-scan "
        '      -disable-output "$f.ll" 2>&1 | grep "^OPTSCAN" || true; '
        '  else echo "OPTSCAN_META\tfailed_file\t$f\t$(grep -m1 error: err.txt | cut -c1-240)"; fi; '
        "done; "
        'echo "OPTSCAN_META\tcompiled\t$ok"'
    )


async def _scan_in(
    workdir: str,
    paths: List[str],
    flags: str,
    *,
    label: str,
    image: str,
    timeout_seconds: int,
) -> Dict[str, Any]:
    try:
        _, stdout, _ = await agent_sandbox_runtime.run_in_sandbox(
            _scan_script(paths, flags),
            workdir,
            image=image,
            timeout_seconds=timeout_seconds,
        )
    except asyncio.TimeoutError:
        return {"error": f"scan timed out after {timeout_seconds}s"}
    except FileNotFoundError:
        return {"error": "Docker is not available to this process"}
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(f"scan_for_optimizations failed: {exc}")
        return {"error": f"scan failed: {exc}"}

    compiled = 0
    failed: List[str] = []
    failure_reasons: Dict[str, str] = {}
    for line in (stdout or "").splitlines():
        if line.startswith("OPTSCAN_META\tcompiled\t"):
            compiled = int(line.rsplit("\t", 1)[1] or 0)
        elif line.startswith("OPTSCAN_META\tfailed_file\t"):
            parts = line.split("\t")
            failed.append(parts[2])
            failure_reasons[parts[2]] = parts[3] if len(parts) > 3 else ""

    if not compiled:
        # Coverage of zero is not a scan with no findings, and must not read as
        # one: a caller told "0 opportunities" would conclude the code is clean.
        return {
            "success": False,
            "error": (
                f"none of the {len(paths)} sources compiled, so nothing was "
                "scanned. "
                + "; ".join(f"{k}: {v}" for k, v in list(failure_reasons.items())[:4])
            ),
        }

    parsed = parse_records(stdout)
    suggestions = _suggestions(parsed)
    caveat = (
        "Counts are static: how often a pattern is written, not how often it "
        "runs. Treat this as a list of places to look. Ranking them needs a "
        "profile, and a hot loop with one site beats a cold function with ten."
    )
    subject = (label or "").strip() or f"{compiled} source(s)"

    return {
        "success": True,
        "data": {
            "modules_compiled": compiled,
            "modules_failed": failed,
            "failure_reasons": failure_reasons,
            "coverage": f"{compiled} of {len(paths)}",
            "suggestions": suggestions,
            "loop_divisions": parsed["loop_divisions"],
            "loop_invariant_divisions": parsed["loop_invariant_divisions"],
            "divisor_group_sizes": parsed["divisor_group_sizes"],
            "shapes_without_a_pass": parsed["shapes"],
            "counts_are_static": caveat,
        },
        "findings": [
            {
                "type": "optimization_opportunity",
                "subject": subject,
                "title": (
                    f"{sum(parsed['known'].values())} site(s) across "
                    f"{compiled} module(s): "
                    + ", ".join(f"{s['sites']} {s['pass']}" for s in suggestions[:4])
                    if suggestions
                    else f"no known opportunity in {compiled} module(s)"
                ),
                "sites": parsed["known"],
                "coverage": f"{compiled} of {len(paths)}",
                "counts_are_static": caveat,
            }
        ],
    }


#: A repository-relative source path: segments of safe characters, no `..`,
#: no leading slash. It is quoted into a shell loop, so nothing else passes.
SAFE_REPO_PATH = re.compile(
    r"^(?!.*(^|/)\.\.(/|$))[A-Za-z0-9_.+-]+(/[A-Za-z0-9_.+-]+)*$"
)
#: A scan of a whole repository compiles each file; a timeout sized for one
#: pasted file is not enough for dozens.
WORKSPACE_TIMEOUT_SECONDS = 900


async def scan_workspace(
    *,
    root: str,
    paths: List[str],
    include_dirs: Optional[List[str]] = None,
    flags: str = "-O1",
    label: str = "",
    image: str = DEFAULT_IMAGE,
    timeout_seconds: int = WORKSPACE_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    """Scan files of a cloned repository, with its headers where they live.

    Pasting sources cannot scan a real codebase: raylib's translation units
    include raylib.h, rlgl.h, config.h and a dozen headers under external/,
    11 MB in all, and the text form takes bare .c names under a 4 MB cap. Here
    the workspace is copied whole into the sandbox's directory -- the daemon
    resolves mounts on the host, where only that directory is shared -- and
    each named file is compiled in place, so its includes resolve as they do
    in the repository's own build.
    """
    import shutil

    base = Path(root or "")
    if not root or not base.is_dir():
        return {
            "error": "the workspace is not available; run clone_and_index_repo first"
        }
    if not paths:
        return {"error": "paths is required: repository-relative .c files to scan"}
    if len(paths) > MAX_SOURCES:
        return {"error": f"at most {MAX_SOURCES} paths per scan, got {len(paths)}"}
    for rel in list(paths) + list(include_dirs or []):
        if not SAFE_REPO_PATH.match(str(rel or "")):
            return {"error": f"path {rel!r} is not a plain repository-relative path"}
    # A directory means the C files directly in it. A run passed ["src"] and
    # was told "not in the workspace: src" -- false, since src is there -- and
    # spent an iteration checking the workspace. Not recursive: src/external/
    # holds vendored libraries nobody asked to scan.
    expanded: List[str] = []
    missing: List[str] = []
    for rel in paths:
        target = base / rel
        if target.is_dir():
            found = sorted(
                f.relative_to(base).as_posix()
                for f in target.glob("*.c")
                if f.is_file()
            )
            if not found:
                return {
                    "error": f"{rel!r} is a directory with no .c files directly in it"
                }
            expanded.extend(found)
        elif target.is_file():
            expanded.append(rel)
        else:
            missing.append(rel)
    if missing:
        return {
            "error": (
                "no such file or directory in the workspace: " + ", ".join(missing[:8])
            )
        }
    paths = list(dict.fromkeys(expanded))
    if len(paths) > MAX_SOURCES:
        return {
            "error": f"the paths expand to {len(paths)} files; at most {MAX_SOURCES}"
        }
    if not SAFE_FLAGS.match(flags or ""):
        return {"error": f"flags contain unsupported characters: {flags!r}"}
    if not agent_sandbox_runtime.execution_enabled():
        return {
            "error": "Sandboxed execution is disabled (ENABLE_UNSAFE_CODE_EXECUTION is false)."
        }
    if image not in agent_sandbox_runtime.allowed_images():
        return {"error": agent_sandbox_runtime.image_not_allowlisted(image)}

    includes = " ".join(f"-I{d}" for d in (include_dirs or []))
    with tempfile.TemporaryDirectory(prefix="optscan_ws_") as workdir:
        shutil.copytree(
            base,
            workdir,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns(".git"),
        )
        return await _scan_in(
            workdir,
            list(paths),
            f"{flags} {includes}".strip(),
            label=label,
            image=image,
            timeout_seconds=timeout_seconds,
        )
