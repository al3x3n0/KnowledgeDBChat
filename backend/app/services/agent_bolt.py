"""Optimise a whole linked executable with BOLT, and judge each configuration.

Object-file binary mode (`agent_binary_rewrite`) can swap one function
because the linker is still there to prefer a strong symbol. A linked
executable has no linker left. What can still change is its LAYOUT: which
blocks fall through, which functions sit together, what is hot and what is
cold. BOLT rewrites exactly that, from a profile, and this module puts its
output through the same gates every other candidate here passes.

THE THREE ARMS.
  orig     the program as linked (non-PIE, --emit-relocs; see below)
  cand     BOLT with a configuration under test
  ceiling  BOLT with the standard recipe. A configuration that only matches
           the recipe everyone uses has not found anything about THIS program,
           the way a source rewrite that only matches -O3 is a flag. The
           verdict for that is `compiler_already_can`, read as "the standard
           recipe already can".

All three share ONE profile, and the measured binary is the profiled binary:
stage one returns the executable's bytes and the next stage reuses them rather
than rebuilding, because BOLT matches a profile to code by address and a
rebuild that differs by a byte silently misapplies it.

HELD-OUT TIMING. By default the profile comes from every input except the
timed one. A layout tuned to the exact run it is then timed on measures its
own training data; with a single input there is no choice, and the result
says so.

TWO BUILD REQUIREMENTS, BOTH MEASURED ON LUA 5.4.6:
  --emit-relocs, or BOLT cannot move functions at all;
  non-PIE. Debian's clang defaults to PIE, and instrumenting a PIE Lua failed
  on luaV_execute -- the hottest function -- with "unable to get new address
  corresponding to input address".

WHAT IT CANNOT CLAIM. BOLT reports how branches changed (Lua: taken branches
-93.2%), which is the mechanism, not the result; only the timing is the
result, and that was ~4% on Lua, near this host's noise. Both are reported, and
the verdict rests on the timing.
"""

from __future__ import annotations

import asyncio
import re
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from app.services import agent_sandbox_runtime
from app.services.agent_optscan import SAFE_NAME, SAFE_REPO_PATH
from app.services.agent_restructure import (
    DEFAULT_TIMEOUT_SECONDS,
    DEFAULT_TOLERANCE,
    DEFAULT_TRIALS,
    MAX_TRIALS,
    Arm,
    check_inputs,
    package,
    run_comparison,
    sandbox_blocked,
)

BOLT_IMAGE = "ghcr.io/al3x3n0/kdbc-bolt-research:latest"
STANDARD_RECIPE = (
    "-reorder-blocks=ext-tsp -reorder-functions=cdsort "
    "-split-functions -split-all-cold -icf=1"
)
DEFAULT_BUILD_FLAGS = "-O2"
#: Build flags and libraries are interpolated into a clang command line.
SAFE_BUILD_FLAGS = re.compile(r"^[-A-Za-z0-9_=+., /]*$")
#: Program arguments, the same way. `-` is allowed on its own: it is how an
#: interpreter is told to read its script from stdin.
SAFE_RUN_ARGS = re.compile(r"^[A-Za-z0-9_.=/+-]*( [A-Za-z0-9_.=/+-]+)*$")
MAX_PROFILE_BYTES = 4_000_000

#: The BOLT options a configuration may use, each with the values it accepts
#: (None: a flag with no value). An allowlist rather than a denylist: options
#: are interpolated into a shell command, and several BOLT options write files
#: or change what is written where.
_WORD = r"[A-Za-z0-9_+.-]{1,32}"
_NUM = r"\d{1,6}"
ALLOWED_OPTIONS: Dict[str, Optional[str]] = {
    "reorder-blocks": r"none|reverse|normal|branch-predictor|cache|cluster-shuffle|ext-tsp",
    "reorder-functions": r"none|exec-count|hfsort|hfsort\+|cdsort|pettis-hansen|random",
    "split-functions": None,
    "split-all-cold": None,
    "split-eh": None,
    "split-strategy": r"profile2|cdsplit|random2|randomN|all",
    "split-threshold": _NUM,
    "icf": r"0|1|none|all|safe",
    "peepholes": r"none|double-jumps|tailcall-traps|useless-branches|all",
    "plt": r"none|hot|all",
    "frame-opt": r"none|hot|all",
    "align-functions": _NUM,
    "align-blocks": None,
    "block-alignment": _NUM,
    "inline-small-functions": None,
    "simplify-conditional-tail-calls": None,
    "sctc-mode": r"always|preserve|heuristic",
    "eliminate-unreachable": None,
    "jump-tables": r"none|basic|move|split|aggressive",
    "indirect-call-promotion": r"none|calls|jump-tables|all",
    "indirect-call-promotion-topn": _NUM,
    "tail-duplication": r"none|aggressive|moderate|cache",
    "reorder-functions-use-hot-size": None,
    "cg-use-split-hot-size": None,
    "strip-rep-ret": None,
    "lite": r"0|1",
    "use-old-text": None,
    "min-branch-clusters": None,
    "cdsort-cache-entries": _NUM,
    "ext-tsp-forward-weight-cond": r"\d{1,3}(\.\d{1,4})?",
}


def check_options(options: str) -> Tuple[Optional[str], str]:
    """Refuse anything outside the allowlist, naming what was wrong."""
    tokens = (options or "").split()
    if not tokens:
        return "options is empty: give at least one BOLT option", ""
    kept: List[str] = []
    for token in tokens:
        m = re.fullmatch(r"--?([a-z][a-z0-9+-]*)(?:=(" + _WORD + r"))?", token)
        if not m:
            return (
                f"{token!r} is not a BOLT option of the form -name or -name=value",
                "",
            )
        name, value = m.group(1), m.group(2)
        if name not in ALLOWED_OPTIONS:
            return (
                f"-{name} is not an option this tool allows. Allowed: "
                + ", ".join(f"-{k}" for k in sorted(ALLOWED_OPTIONS)),
                "",
            )
        pattern = ALLOWED_OPTIONS[name]
        if pattern is None and value is not None and value not in ("0", "1"):
            return f"-{name} takes no value (or =0/=1), got {value!r}", ""
        if pattern is not None and (value is None or not re.fullmatch(pattern, value)):
            return f"-{name} needs a value matching {pattern}, got {value!r}", ""
        kept.append(f"-{name}" + (f"={value}" if value is not None else ""))
    return None, " ".join(kept)


# --------------------------------------------------------------------------- #
# Stage one: build, instrument, profile, and the standard recipe's statistics.
# --------------------------------------------------------------------------- #


def _profile_script(build: str, run_args: str, profile_inputs: List[int]) -> str:
    runs = "; ".join(
        f"./prog.inst {run_args} <in_{i}.txt >/dev/null 2>&1 || "
        f'echo "__profile_run_failed__ {i} $?"'
        for i in profile_inputs
    )
    return (
        f"{{ {build}; }} >build.log 2>&1 || "
        "{ echo __build_failed__; head -c 4000 build.log; exit 0; }; "
        "llvm-bolt prog -instrument -instrumentation-file=/work/prof "
        "-instrumentation-file-append-pid -o prog.inst >inst.log 2>&1 || "
        "{ echo __instrument_failed__; grep -m4 -E 'ERROR|error' inst.log; exit 0; }; "
        f"{runs}; "
        "ls prof.*.fdata >/dev/null 2>&1 || { echo __no_profile__; exit 0; }; "
        "merge-fdata prof.*.fdata >prof.fdata 2>merge.log || "
        "{ echo __merge_failed__; head -c 2000 merge.log; exit 0; }; "
        f"llvm-bolt prog -data=prof.fdata {STANDARD_RECIPE} -dyno-stats "
        "-o prog.std >std.log 2>&1; "
        'echo "__std_rc__ $?"; '
        "echo __size__ $(stat -c %s prog) $(stat -c %s prog.std 2>/dev/null || echo 0); "
        "echo __stdlog__; grep -E 'BOLT-(INFO|WARNING|ERROR)|^ +[0-9]+ : ' std.log "
        "| head -c 30000; echo __stdlog_end__"
    )


def parse_dyno(log: str) -> Dict[str, Dict[str, Any]]:
    """BOLT's dyno-stats: each metric before, and after with its change.

    BOLT prints the table twice, before and after, the second with a
    percentage. The first occurrence of a metric is the input binary.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for m in re.finditer(
        r"^\s+(\d+) : ([a-z][a-z ()/-]+?)(?: \(([-+]?[\d.]+)%\))?\s*$", log or "", re.M
    ):
        value, name, change = int(m.group(1)), m.group(2).strip(), m.group(3)
        entry = out.setdefault(name, {})
        if "before" not in entry:
            entry["before"] = value
        else:
            entry["after"] = value
            if change is not None:
                entry["change_pct"] = float(change)
    return out


def hot_functions(fdata: str, top: int = 15) -> List[Dict[str, Any]]:
    """Branch executions per function, from BOLT's profile.

    A branch record is `1 <from_fn> <off> 1 <to_fn> <off> <mispreds> <count>`;
    counts are attributed to the function the branch leaves. Enough to tell a
    proposer where the time goes, without handing it the raw profile.
    """
    counts: Counter = Counter()
    for line in (fdata or "").splitlines():
        parts = line.split()
        if len(parts) == 8 and parts[0] == "1" and parts[-1].isdigit():
            counts[parts[1]] += int(parts[-1])
    total = sum(counts.values()) or 1
    return [
        {"function": fn, "branch_executions": n, "share": round(n / total, 3)}
        for fn, n in counts.most_common(top)
    ]


def _unoptimised(log: str) -> Optional[str]:
    m = re.search(
        r"BOLT-INFO: (\d+) functions with profile could not be optimized", log or ""
    )
    return m.group(1) if m else None


async def build_and_profile(
    *,
    run_args: str,
    inputs: List[str],
    profile_inputs: List[int],
    sources: Optional[Dict[str, str]] = None,
    root: str = "",
    paths: Optional[List[str]] = None,
    include_dirs: Optional[List[str]] = None,
    build_flags: str = DEFAULT_BUILD_FLAGS,
    libs: str = "-lm",
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    """Stage one. Returns the binary, its profile, and what the recipe does."""
    if not SAFE_BUILD_FLAGS.match(build_flags or "") or not SAFE_BUILD_FLAGS.match(
        libs or ""
    ):
        return {"error": "build_flags and libs contain unsupported characters"}
    if not SAFE_RUN_ARGS.match(run_args or ""):
        return {"error": f"run_args contain unsupported characters: {run_args!r}"}
    blocked = sandbox_blocked(BOLT_IMAGE)
    if blocked:
        return {"error": blocked}

    if sources:
        bad = [n for n in sources if not SAFE_NAME.match(str(n))]
        if bad:
            return {"error": f"source names must be bare .c filenames: {bad[:4]}"}
        compile_list = list(sources)
    else:
        if not root or not Path(root).is_dir():
            return {"error": "give sources, or a workspace with paths"}
        if not paths:
            return {"error": "paths is required with a workspace: the .c files to link"}
        for rel in list(paths) + list(include_dirs or []):
            if not SAFE_REPO_PATH.match(str(rel)):
                return {
                    "error": f"path {rel!r} is not a plain repository-relative path"
                }
        missing = [p for p in paths if not Path(root, p).is_file()]
        if missing:
            return {"error": f"not in the workspace: {', '.join(missing[:8])}"}
        compile_list = list(paths)

    includes = " ".join(f"-I{d}" for d in (include_dirs or []))
    build = (
        f"clang {build_flags} {includes} -fno-pie -no-pie -Wl,--emit-relocs "
        f"-o prog {' '.join(compile_list)} {libs}"
    )
    with tempfile.TemporaryDirectory(prefix="bolt_profile_") as workdir:
        if sources:
            for name, text in sources.items():
                Path(workdir, name).write_text(text or "", encoding="utf-8")
        else:
            shutil.copytree(
                root, workdir, dirs_exist_ok=True, ignore=shutil.ignore_patterns(".git")
            )
        for i, text in enumerate(inputs):
            Path(workdir, f"in_{i}.txt").write_text(text, encoding="utf-8")
        try:
            _, stdout, _ = await agent_sandbox_runtime.run_in_sandbox(
                _profile_script(build, run_args, profile_inputs),
                workdir,
                image=BOLT_IMAGE,
                timeout_seconds=timeout_seconds,
            )
        except asyncio.TimeoutError:
            return {"error": f"build and profiling timed out after {timeout_seconds}s"}
        except FileNotFoundError:
            return {"error": "Docker is not available to this process"}

        for marker, what in (
            ("__build_failed__", "the program did not build"),
            ("__instrument_failed__", "BOLT could not instrument the program"),
            ("__merge_failed__", "the profiles could not be merged"),
        ):
            if marker in stdout:
                return {
                    "error": f"{what}: " + stdout.split(marker, 1)[1].strip()[:3000]
                }
        if "__no_profile__" in stdout:
            failed = re.findall(r"__profile_run_failed__ (\d+) (\d+)", stdout)
            return {
                "error": (
                    "the instrumented program wrote no profile"
                    + (f"; profiling runs failed: {failed}" if failed else "")
                )
            }
        prog = Path(workdir, "prog").read_bytes()
        fdata = Path(workdir, "prof.fdata").read_text(
            encoding="utf-8", errors="replace"
        )

    if len(fdata) > MAX_PROFILE_BYTES:
        return {"error": f"the profile is {len(fdata)} bytes, over {MAX_PROFILE_BYTES}"}
    std_log = stdout.split("__stdlog__", 1)[-1].split("__stdlog_end__", 1)[0]
    sizes = re.search(r"__size__ (\d+) (\d+)", stdout)
    rc = re.search(r"__std_rc__ (\d+)", stdout)
    return {
        "prog": prog,
        "fdata": fdata,
        "hot_functions": hot_functions(fdata),
        "standard_recipe": {
            "ok": bool(rc and rc.group(1) == "0"),
            "dyno_stats": parse_dyno(std_log),
            "functions_not_optimised": _unoptimised(std_log),
            "size_bytes": {
                "before": int(sizes.group(1)) if sizes else None,
                "after": int(sizes.group(2)) if sizes else None,
            },
            "errors": re.findall(r"BOLT-ERROR[^\n]*", std_log)[:4],
        },
        "profile_failures": re.findall(r"__profile_run_failed__ (\d+) (\d+)", stdout),
    }


# --------------------------------------------------------------------------- #
# Stage two: judge a configuration against the original and the recipe.
# --------------------------------------------------------------------------- #


def _bolt_build(binary: str, options: str) -> str:
    """Run BOLT; on failure put its own error first, where the report reads."""
    return (
        f"llvm-bolt prog -data=prof.fdata {options} -dyno-stats -o {binary} "
        f">bolt_{binary}.log 2>&1 || "
        f"{{ grep -m4 -E 'BOLT-ERROR|error|Error' bolt_{binary}.log | sed 's/^/error: /'; false; }}"
    )


async def judge_configuration(
    *,
    profiled: Dict[str, Any],
    options: str,
    inputs: List[str],
    run_args: str = "",
    baseline_options: str = "",
    bench_input: int = 0,
    trials: int = DEFAULT_TRIALS,
    label: str = "",
    rationale: str = "",
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    """Equivalence on every input, then timing against original and recipe.

    `baseline_options`, when given, makes the baseline arm BOLT with those
    options instead of the original binary -- one configuration measured
    against another.
    """
    problem, safe_options = check_options(options)
    if problem:
        # The caller's mistake, not the sandbox's: shaped like a build failure
        # so a repair loop hands it back rather than giving up.
        return {
            "success": True,
            "data": {"verdict": "did_not_compile", "compile_errors": problem},
            "findings": [],
        }
    safe_baseline = ""
    if baseline_options:
        problem, safe_baseline = check_options(baseline_options)
        if problem:
            return {"error": f"baseline_options: {problem}"}

    arms = [
        Arm(
            "orig",
            _bolt_build("orig", safe_baseline) if safe_baseline else "cp prog orig",
        ),
        Arm("cand", _bolt_build("cand", safe_options)),
        Arm("ceiling", _bolt_build("ceiling", STANDARD_RECIPE), differential=False),
    ]
    result = await run_comparison(
        files={"prog": profiled["prog"], "prof.fdata": profiled["fdata"]},
        prep="chmod +x prog",
        arms=arms,
        inputs=inputs,
        bench_input=bench_input,
        trials=max(3, min(int(trials or DEFAULT_TRIALS), MAX_TRIALS)),
        tolerance=DEFAULT_TOLERANCE,
        image=BOLT_IMAGE,
        timeout_seconds=timeout_seconds,
        run_args=run_args,
        collect=("bolt_cand.log",),
    )
    cand_log = (result.pop("collected", None) or {}).get("bolt_cand.log", "")
    extra: Dict[str, Any] = {
        "options": safe_options,
        "standard_recipe": STANDARD_RECIPE,
        "profiled_on_inputs": profiled.get("profiled_on"),
        "timed_on_input": bench_input,
    }
    if safe_baseline:
        extra["baseline_options"] = safe_baseline
    if cand_log:
        extra["dyno_stats"] = parse_dyno(cand_log)
        extra["functions_not_optimised"] = _unoptimised(cand_log)
    if bench_input in (profiled.get("profiled_on") or []):
        extra["timed_on_profiled_input"] = True
    packaged = package(
        result,
        kind="binary_layout_result",
        label=label or safe_options,
        invariant=rationale,
        value_preserving=True,
        n_inputs=len(inputs),
        ceiling_flags=f"the standard BOLT recipe ({STANDARD_RECIPE})",
        extra=extra,
    )
    data = packaged.get("data")
    if isinstance(data, dict) and extra.get("timed_on_profiled_input"):
        data.setdefault("notes", []).append(
            "Timed on an input the profile was taken from: a layout tuned to "
            "its own benchmark. Give more inputs so one can be held out."
        )
    return packaged


def default_profile_inputs(n_inputs: int, bench_input: int) -> List[int]:
    """Every input except the timed one; all of them when there is only one."""
    held_out = [i for i in range(n_inputs) if i != bench_input]
    return held_out or list(range(n_inputs))


async def optimize_executable(
    *,
    options: str,
    inputs: List[str],
    run_args: str = "",
    sources: Optional[Dict[str, str]] = None,
    root: str = "",
    paths: Optional[List[str]] = None,
    include_dirs: Optional[List[str]] = None,
    build_flags: str = DEFAULT_BUILD_FLAGS,
    libs: str = "-lm",
    profile_inputs: Optional[List[int]] = None,
    bench_input: int = 0,
    trials: int = DEFAULT_TRIALS,
    label: str = "",
    rationale: str = "",
) -> Dict[str, Any]:
    """Build, profile, and judge one BOLT configuration."""
    problem, cleaned = check_inputs(inputs, bench_input)
    if problem:
        return {"error": problem}
    chosen = (
        profile_inputs
        if profile_inputs
        else default_profile_inputs(len(cleaned), bench_input)
    )
    if any(not 0 <= i < len(cleaned) for i in chosen):
        return {
            "error": f"profile_inputs {chosen} are not indices into {len(cleaned)} inputs"
        }
    profiled = await build_and_profile(
        run_args=run_args,
        inputs=cleaned,
        profile_inputs=chosen,
        sources=sources,
        root=root,
        paths=paths,
        include_dirs=include_dirs,
        build_flags=build_flags,
        libs=libs,
    )
    if profiled.get("error"):
        return {"success": False, "error": profiled["error"]}
    profiled["profiled_on"] = chosen
    result = await judge_configuration(
        profiled=profiled,
        options=options,
        inputs=cleaned,
        run_args=run_args,
        bench_input=bench_input,
        trials=trials,
        label=label,
        rationale=rationale,
    )
    data = result.get("data")
    if isinstance(data, dict):
        data["hot_functions"] = profiled["hot_functions"][:8]
    logger.debug(
        f"optimize_executable: {label or options} -> {data and data.get('verdict')}"
    )
    return result


# --------------------------------------------------------------------------- #
# A model proposes configurations for THIS program; each is judged above.
# --------------------------------------------------------------------------- #

BOLT_SCHEMA = {
    "type": "object",
    "properties": {
        "proposals": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "idea": {"type": "string"},
                    "invariant": {"type": "string"},
                    "why_compiler_cannot": {"type": "string"},
                    "value_preserving": {"type": "boolean"},
                    "options": {"type": "string"},
                },
                "required": ["name", "idea", "invariant", "options"],
            },
        }
    },
    "required": ["proposals"],
}


def _bolt_system_prompt() -> str:
    allowed = "\n".join(
        f"  -{name}" + (f"=<{pattern}>" if pattern else "")
        for name, pattern in sorted(ALLOWED_OPTIONS.items())
    )
    return (
        "You tune BOLT, LLVM's post-link optimiser, for ONE program, from its "
        "profile. The standard recipe is:\n  " + STANDARD_RECIPE + "\n"
        "Your configuration is measured against the unoptimised binary AND "
        "against that recipe. Only one that beats the recipe has found "
        "something about this program -- matching it is reported as such.\n\n"
        "Reason from the profile you are given: where the branch executions "
        "concentrate, how many functions carry the profile, what the recipe "
        "already achieved. Examples of program-specific reasoning: an "
        "interpreter dispatch loop suits aggressive block reordering and may "
        "suffer from splitting; a profile concentrated in a few functions "
        "suits function reordering less than block layout; indirect calls "
        "through a small set of targets suit indirect-call promotion.\n\n"
        "Allowed options, and nothing else (anything else is refused):\n"
        + allowed
        + "\n\nReturn ONE proposal in 'proposals' with: name (kebab-case), "
        "idea (what the configuration does, two sentences), invariant (what "
        "about THIS profile it relies on), why_compiler_cannot (why the "
        "standard recipe misses it), options (the BOLT options, space-"
        "separated). Output JSON only."
    )


async def propose_bolt_configurations(
    *,
    inputs: List[str],
    run_args: str = "",
    sources: Optional[Dict[str, str]] = None,
    root: str = "",
    paths: Optional[List[str]] = None,
    include_dirs: Optional[List[str]] = None,
    build_flags: str = DEFAULT_BUILD_FLAGS,
    libs: str = "-lm",
    profile_inputs: Optional[List[int]] = None,
    bench_input: int = 0,
    count: int = 3,
    focus: str = "",
    label: str = "",
    user_id: Any = None,
    db: Any = None,
) -> Dict[str, Any]:
    """Profile once, ask for configurations one at a time, judge each."""
    from app.services import agent_restructure_proposer as proposer

    problem, cleaned = check_inputs(inputs, bench_input)
    if problem:
        return {"error": problem}
    chosen = (
        profile_inputs
        if profile_inputs
        else default_profile_inputs(len(cleaned), bench_input)
    )
    if any(not 0 <= i < len(cleaned) for i in chosen):
        return {
            "error": f"profile_inputs {chosen} are not indices into {len(cleaned)} inputs"
        }
    profiled = await build_and_profile(
        run_args=run_args,
        inputs=cleaned,
        profile_inputs=chosen,
        sources=sources,
        root=root,
        paths=paths,
        include_dirs=include_dirs,
        build_flags=build_flags,
        libs=libs,
    )
    if profiled.get("error"):
        return {"success": False, "error": profiled["error"]}
    profiled["profiled_on"] = chosen
    recipe = profiled["standard_recipe"]
    hot = "\n".join(
        f"  {h['function']}: {h['branch_executions']} branch executions ({h['share']:.1%})"
        for h in profiled["hot_functions"]
    )
    dyno = "\n".join(
        f"  {k}: {v.get('before')} -> {v.get('after')}"
        + (f" ({v['change_pct']:+}%)" if "change_pct" in v else "")
        for k, v in list(recipe["dyno_stats"].items())[:14]
    )
    message = (
        "Propose ONE BOLT configuration for this program.\n\n"
        + (f"About the program: {focus}\n\n" if focus else "")
        + f"=== hottest functions, by branch executions in the profile ===\n{hot}\n\n"
        f"=== what the standard recipe did (dyno stats, before -> after) ===\n{dyno}\n"
        f"functions with profile the recipe could not optimise: "
        f"{recipe['functions_not_optimised']}\n"
        f"binary size before -> after: {recipe['size_bytes']}\n"
    )
    subject = (label or "").strip() or "executable"

    async def evaluate(
        proposal: Dict[str, Any],
        baseline: Optional[Dict[str, Any]] = None,
        trials: int = DEFAULT_TRIALS,
    ) -> Dict[str, Any]:
        return await judge_configuration(
            profiled=profiled,
            options=proposal["options"],
            inputs=cleaned,
            run_args=run_args,
            baseline_options=baseline["options"] if baseline else "",
            bench_input=bench_input,
            trials=trials,
            label=f"{subject}/{proposal['name']}"
            + (f" over {baseline['name']}" if baseline else ""),
            rationale=proposal.get("invariant", ""),
        )

    out = await proposer._propose_and_judge(
        system=_bolt_system_prompt(),
        message=message,
        schema=BOLT_SCHEMA,
        body_key="options",
        count=max(1, min(int(count or 3), 5)),
        evaluate=evaluate,
        user_id=user_id,
        db=db,
    )
    result = proposer._as_result(out, subject, "BOLT")
    if result.get("success"):
        result["data"]["hot_functions"] = profiled["hot_functions"][:8]
        result["data"]["standard_recipe"] = recipe
        result["data"]["profiled_on_inputs"] = chosen
        result["data"]["timed_on_input"] = bench_input
    return result
