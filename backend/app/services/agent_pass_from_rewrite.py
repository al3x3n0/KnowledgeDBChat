"""Turn a winning rewrite into a compiler pass, and check the pass earns it.

`propose_restructurings` finds rewrites of one kernel. A rewrite helps one
file; a pass helps every file with the same shape, which is the step from a
finding to a tool. This module takes that step and, as everywhere here, keeps
the judging away from whoever did the writing.

WHAT "EARNS IT" MEANS, AS FOUR SEPARATE QUESTIONS.

  1. Does the pass FIRE on the kernel under `clang -fpass-plugin`? This repo
     has already shipped two plugins that registered only an `opt -passes=`
     parser: the flag was accepted, the build succeeded, and nothing changed.
     The kernel is compiled to IR with and without the plugin; identical IR is
     `did_not_fire`, and nothing is timed.
  2. Is the program the same? The same differential run every rewrite gets:
     identical output on every input.
  3. How much of the rewrite's gain does it RECOVER? The hand rewrite is timed
     in the same interleaved run, so `recovered` compares numbers taken
     together. A pass recovering 10% of a 7x rewrite has found a shadow of the
     idea, and should say so rather than report "faster".
  4. Does it DECLINE where its precondition fails? The proposer supplies a
     `must_decline` case -- code that looks similar and where the transform
     would be wrong. A pass that changes it gets the verdict `overreaches`,
     whatever it did for speed: one case cannot prove a pass sound, but a pass
     that fails it is certainly unsound.

NOT EVERY REWRITE IS A PASS. An application-specific rewrite may rely on a
fact the IR does not carry -- "m never changes within a call" is visible in IR
as loop-invariance, but "these particles never leave the box" is not visible
anywhere. The proposer is asked to say `expressible: false` in that case, and
that answer is a result rather than a failure: it is what separates
optimisations a compiler could learn from ones only the application's author
can make.
"""

from __future__ import annotations

import asyncio
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from app.services import agent_sandbox_runtime
from app.services.agent_compiler_sandbox import _clean_flags
from app.services.agent_pass_builder import (
    SAFE_PASS_NAME,
    _deltas,
    normalised,
    opcode_counts,
)
from app.services.agent_restructure import (
    DEFAULT_FLAGS,
    DEFAULT_TIMEOUT_SECONDS,
    DEFAULT_TOLERANCE,
    DEFAULT_TRIALS,
    MAX_SOURCE_CHARS,
    MAX_TRIALS,
    VALUE_CHANGING_TOLERANCE,
    Arm,
    check_inputs,
    clamp_trials,
    package,
    run_comparison,
    sandbox_blocked,
)

#: clang, opt, g++ and the LLVM headers together live only here.
PASS_IMAGE = "ghcr.io/al3x3n0/kdbc-pass-dev:latest"

_BUILD_PLUGIN = (
    'g++ -fPIC -shared -o pass.so pass.cpp $("$LLVM_CONFIG" --cxxflags) -fno-rtti'
)


def _check_names(pass_name: str, source: str) -> Optional[str]:
    if not SAFE_PASS_NAME.match(pass_name or ""):
        return (
            f"pass_name {pass_name!r} is not usable: the lowercase name the "
            "plugin registers, e.g. 'uchar-trig-table'"
        )
    if not (source or "").strip():
        return "pass_source is required: the C++ of an LLVM 14 pass plugin"
    if len(source) > MAX_SOURCE_CHARS:
        return f"pass_source exceeds {MAX_SOURCE_CHARS} characters"
    return None


def parse_firing(stdout: str) -> Dict[str, Any]:
    """Read the firing probe's output into what a caller acts on."""
    text = stdout or ""
    if "__build_failed__" in text:
        return {
            "stage": "build",
            "errors": text.split("__build_failed__", 1)[1].strip()[:8000],
        }
    if "__pass_crashed__" in text:
        # The plugin built and clang died running it: a segfault or an
        # assertion inside the pass, which is a bug in the pass and not in the
        # kernel.
        return {
            "stage": "run",
            "errors": text.split("__pass_crashed__", 1)[1].strip()[:6000],
        }
    if "__verify_failed__" in text:
        detail = text.split("__verify_failed__", 1)[1].strip()[:4000]
        if re.search(r"unknown pass name|unknown function pass", detail, re.I):
            return {"stage": "unregistered", "errors": detail}
        return {"stage": "invalid_ir", "errors": detail}
    plain = _section(text, "__plain__", "__passed__")
    passed = _section(text, "__passed__", "__end__")
    fired = normalised(plain) != normalised(passed)
    return {
        "stage": "ok",
        "fired": fired,
        "ir_deltas": _deltas(opcode_counts(plain), opcode_counts(passed)),
        "pass_stderr": _section(text, "__stderr__", "__stderr_end__").strip()[:3000],
        "plain_ir": plain,
    }


def _section(text: str, start: str, end: str) -> str:
    a = text.find(start)
    if a < 0:
        return ""
    b = text.find(end, a + len(start))
    return text[a + len(start) : b if b > a else len(text)]


def _firing_script(flags: str, sources: List[str], pass_name: str) -> str:
    """Build the plugin once, then compile each source with and without it.

    Each source is also run through `opt -passes=<name>,verify` directly. Release
    clang does not verify IR, and later passes can paper over a malformed
    instruction: a fastmod pass emitted `lshr i128 %x, i64 64`, the final IR
    verified clean, and the only symptom was a wrong answer -- so two repair
    attempts were told "diverged" and never why. The verifier right after the
    pass names the instruction. `opt` refusing the name is also caught here: a
    plugin whose parsing callback and pass_name disagree.
    """
    parts = [
        f"{_BUILD_PLUGIN} 2>build_err.txt || "
        "{ echo __build_failed__; head -c 8000 build_err.txt; exit 0; }"
    ]
    for src in sources:
        stem = src[:-2]
        parts.append(
            f"echo '__source__ {src}'; "
            f"clang {flags} -S -emit-llvm -o {stem}.plain.ll {src} 2>/dev/null; "
            f"clang {flags} -fpass-plugin=./pass.so -S -emit-llvm "
            f"-o {stem}.passed.ll {src} 2>{stem}.err || "
            f"{{ echo __pass_crashed__; head -c 6000 {stem}.err; exit 0; }}; "
            f"opt -load-pass-plugin=./pass.so -passes='{pass_name},verify' "
            f"-disable-output {stem}.plain.ll >{stem}.verify 2>&1 || "
            f"{{ echo __verify_failed__; head -c 4000 {stem}.verify; exit 0; }}; "
            f"echo __plain__; cat {stem}.plain.ll; echo __passed__; "
            f"cat {stem}.passed.ll; echo __end__; "
            f"echo __stderr__; head -c 3000 {stem}.err; echo __stderr_end__"
        )
    return "; ".join(parts)


async def probe_firing(
    *,
    pass_source: str,
    pass_name: str,
    sources: Dict[str, str],
    flags: str,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    """Build the plugin and report, per source, whether it changed the IR."""
    with tempfile.TemporaryDirectory(prefix="passfire_") as workdir:
        Path(workdir, "pass.cpp").write_text(pass_source, encoding="utf-8")
        for name, text in sources.items():
            Path(workdir, name).write_text(text, encoding="utf-8")
        try:
            _, stdout, _ = await agent_sandbox_runtime.run_in_sandbox(
                _firing_script(flags, list(sources), pass_name),
                workdir,
                image=PASS_IMAGE,
                timeout_seconds=timeout_seconds,
            )
        except asyncio.TimeoutError:
            return {"stage": "sandbox", "errors": f"timed out after {timeout_seconds}s"}
        except FileNotFoundError:
            return {
                "stage": "sandbox",
                "errors": "Docker is not available to this process",
            }

    if "__build_failed__" in stdout:
        return {"_all": parse_firing(stdout)}
    out: Dict[str, Any] = {}
    chunks = stdout.split("__source__ ")[1:]
    for chunk in chunks:
        name, _, body = chunk.partition("\n")
        out[name.strip().strip("'")] = parse_firing(body)
    return out


async def evaluate_pass_on_kernel(
    *,
    pass_source: str,
    pass_name: str,
    kernel: str,
    driver: str,
    inputs: List[str],
    rewrite_kernel: str = "",
    must_decline: str = "",
    value_preserving: bool = True,
    precondition: str = "",
    flags: str = DEFAULT_FLAGS,
    bench_input: int = 0,
    trials: int = DEFAULT_TRIALS,
    label: str = "",
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    """Fire, equivalence, speed, recovery against the rewrite, and declining."""
    problem = _check_names(pass_name, pass_source)
    if problem:
        return {"error": problem}
    if not (kernel or "").strip() or "main(" not in (driver or "").replace(" ", ""):
        return {"error": "kernel (no main) and driver (with main) are required"}
    problem, cleaned = check_inputs(inputs, bench_input)
    if problem:
        return {"error": problem}
    safe = _clean_flags(flags or DEFAULT_FLAGS)
    if safe is None:
        return {"error": f"flags contain unsupported characters: {flags!r}"}
    if re.search(r"(^|\s)-O0(\s|$)", safe):
        return {
            "error": (
                "flags include -O0, where clang runs no pass pipeline and a "
                "plugin's extension-point callbacks never fire"
            )
        }
    blocked = sandbox_blocked(PASS_IMAGE)
    if blocked:
        return {"error": blocked}

    probe_sources = {"kernel.c": kernel}
    if must_decline.strip():
        probe_sources["decline.c"] = must_decline
    firing = await probe_firing(
        pass_source=pass_source, pass_name=pass_name, sources=probe_sources, flags=safe
    )
    if "_all" in firing:
        return _early("did_not_compile", firing["_all"]["errors"], pass_name, label)
    on_kernel = firing.get("kernel.c") or {}
    if on_kernel.get("stage") == "sandbox":
        return {"success": False, "error": on_kernel["errors"]}
    if on_kernel.get("stage") == "unregistered":
        return _early(
            "unregistered",
            f"`opt -passes={pass_name}` does not know that name: the string given "
            "to registerPipelineParsingCallback and pass_name differ. "
            + on_kernel["errors"][:1500],
            pass_name,
            label,
        )
    if on_kernel.get("stage") == "invalid_ir":
        return _early(
            "invalid_ir",
            "the pass produced IR the verifier rejects -- this is what it said, "
            "run directly after the pass: " + on_kernel["errors"],
            pass_name,
            label,
        )
    if on_kernel.get("stage") == "run":
        return _early(
            "pass_crashed",
            "clang died running the pass on the kernel: " + on_kernel["errors"],
            pass_name,
            label,
        )
    if not on_kernel.get("fired"):
        return _early(
            "did_not_fire",
            (
                "the plugin built and loaded under -fpass-plugin and left the "
                "kernel's IR identical. Either it registers no extension-point "
                "callback (registerPipelineParsingCallback alone is only for "
                "`opt -passes=`), or its match condition does not see the IR "
                "as it looks at that point in the pipeline."
            ),
            pass_name,
            label,
            extra={"ir_at_flags": on_kernel.get("plain_ir", "")[:12000]},
        )

    decline = firing.get("decline.c")
    decline_report: Optional[Dict[str, Any]] = None
    if decline is not None:
        decline_report = {
            "fired": bool(decline.get("fired")),
            "stage": decline.get("stage"),
            "ir_deltas": decline.get("ir_deltas"),
        }

    arms = [
        Arm(
            "orig",
            f"clang {safe} -c kernel.c -o kernel.o && clang {safe} -o orig driver.o kernel.o -lm",
        ),
        Arm(
            "cand",
            f"{_BUILD_PLUGIN} && clang {safe} -fpass-plugin=./pass.so -c kernel.c "
            f"-o kernel_pass.o && clang {safe} -o cand driver.o kernel_pass.o -lm",
        ),
    ]
    if rewrite_kernel.strip():
        arms.append(
            Arm(
                "rewrite",
                f"clang {safe} -c rewrite.c -o rewrite.o && clang {safe} -o rewrite driver.o rewrite.o -lm",
                differential=False,
            )
        )
    ceiling_flags = "-O3" if value_preserving else "-O3 -ffast-math"
    arms.append(
        Arm(
            "ceiling",
            f"clang {safe} {ceiling_flags} -c kernel.c -o kernel_ceil.o && "
            f"clang {safe} -o ceiling driver.o kernel_ceil.o -lm",
            differential=False,
        )
    )
    files = {"kernel.c": kernel, "driver.c": driver, "pass.cpp": pass_source}
    if rewrite_kernel.strip():
        files["rewrite.c"] = rewrite_kernel
    result = await run_comparison(
        files=files,
        prep=f"clang {safe} -c driver.c -o driver.o",
        arms=arms,
        inputs=cleaned,
        bench_input=bench_input,
        trials=clamp_trials(trials),
        tolerance=DEFAULT_TOLERANCE if value_preserving else VALUE_CHANGING_TOLERANCE,
        image=PASS_IMAGE,
        timeout_seconds=timeout_seconds,
    )

    extra: Dict[str, Any] = {
        "pass_name": pass_name,
        "ir_deltas": on_kernel.get("ir_deltas"),
    }
    recovery = recovered_share(result.get("timing") or {})
    if recovery is not None:
        extra["recovered"] = recovery
    if decline_report is not None:
        extra["must_decline"] = decline_report
    packaged = package(
        result,
        kind="pass_evaluation",
        label=label or pass_name,
        invariant=precondition,
        value_preserving=value_preserving,
        n_inputs=len(cleaned),
        ceiling_flags=ceiling_flags,
        extra=extra,
    )
    _annotate(packaged, recovery, decline_report)
    if decline_report and decline_report.get("fired"):
        # Outranks every speed verdict. Measured: a pass widened to match any
        # integer index was "faster" on the kernel, whose indices are bytes,
        # and indexed past the end of its 256-entry table on a 16-bit one.
        # The kernel's inputs cannot catch that; the decline case did.
        _override(packaged, "overreaches")
    return packaged


def _override(packaged: Dict[str, Any], verdict: str) -> None:
    data = packaged.get("data")
    if isinstance(data, dict):
        data["speed_verdict"] = data.get("verdict")
        data["verdict"] = verdict
    for finding in packaged.get("findings") or []:
        finding["verdict"] = verdict
        finding["win"] = 1 if verdict == "faster" else 0
        finding["title"] = f"{finding.get('subject')}: {verdict}"


def recovered_share(timing: Dict[str, Any]) -> Optional[float]:
    """The share of the hand rewrite's gain the pass reproduces.

    Measured on the fastest trials of one interleaved run. None when there is
    no rewrite arm, or when the rewrite itself gained nothing to recover.
    """
    fastest = timing.get("arms_fastest_ms") or {}
    orig, cand, rewrite = (
        fastest.get("orig"),
        fastest.get("cand"),
        fastest.get("rewrite"),
    )
    if not (orig and cand and rewrite) or rewrite >= orig:
        return None
    rewrite_gain = orig / rewrite - 1
    pass_gain = orig / cand - 1
    return round(pass_gain / rewrite_gain, 3)


def _annotate(
    packaged: Dict[str, Any],
    recovery: Optional[float],
    decline: Optional[Dict[str, Any]],
) -> None:
    data = packaged.get("data")
    if not isinstance(data, dict):
        return
    notes = list(data.get("notes") or [])
    if recovery is not None:
        notes.append(
            f"The pass recovers {round(recovery * 100)}% of the hand rewrite's "
            "gain, measured in the same run."
            + (" Most of the idea is not in the pass." if recovery < 0.5 else "")
        )
    if decline and decline.get("fired"):
        notes.append(
            "The pass CHANGED the must_decline case, where its precondition "
            "does not hold. Treat it as unsound until that is explained."
        )
    if notes:
        data["notes"] = notes
    for finding in packaged.get("findings") or []:
        finding["recovered"] = recovery
        finding["fired_on_must_decline"] = bool(decline and decline.get("fired"))


def _early(
    verdict: str,
    detail: str,
    pass_name: str,
    label: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """A verdict reached before anything was run: nothing to time."""
    subject = (label or "").strip() or pass_name
    data: Dict[str, Any] = {
        "verdict": verdict,
        "detail": detail,
        "pass_name": pass_name,
    }
    if extra:
        data.update(extra)
    return {
        "success": True,
        "data": data,
        "findings": [
            {
                "type": "pass_evaluation",
                "subject": subject,
                "title": f"{subject}: {verdict}",
                "verdict": verdict,
            }
        ],
    }


# --------------------------------------------------------------------------- #
# Writing the pass: a model, repaired against the evidence above.
# --------------------------------------------------------------------------- #

MAX_ATTEMPTS = 3
#: Verdicts a further attempt can plausibly fix. `unresolved`/`slower` are not
#: here: a pass that is correct and not faster has answered the question.
REPAIRABLE = (
    "did_not_compile",
    "did_not_fire",
    "pass_crashed",
    "diverged",
    "crashed",
    "overreaches",
    "invalid_ir",
    "unregistered",
)

PASS_SCHEMA = {
    "type": "object",
    "properties": {
        "expressible": {"type": "boolean"},
        "reason": {"type": "string"},
        "precondition_in_ir": {"type": "string"},
        "pass_name": {"type": "string"},
        "pass_source": {"type": "string"},
        "must_decline": {"type": "string"},
        "value_preserving": {"type": "boolean"},
    },
    "required": ["expressible", "reason"],
}

PASS_SYSTEM_PROMPT = """\
You turn a hand optimisation of one C kernel into an LLVM 14 pass plugin
(new pass manager) that performs the same transformation on ANY code with the
same shape.

First decide whether that is possible. A pass may only rely on what the IR
itself establishes: integer widths (an i8 has 256 values), loop invariance,
constant operands, the absence of side effects, known library semantics. If
the hand rewrite relies on a fact the IR does not carry -- about the
application's data, about how callers behave -- set expressible=false and say
exactly which fact, in `reason`. That is a useful answer, not a failure.

If it is expressible:
- pass_name: lowercase kebab-case.
- pass_source: complete C++. A PassInfoMixin function pass, and an
  llvmGetPassPluginInfo that registers BOTH registerPipelineParsingCallback
  (for `opt -passes=<pass_name>`) AND an extension-point callback such as
  registerPeepholeEPCallback -- without the second, clang -fpass-plugin loads
  the plugin and runs nothing. Built with
  g++ -fPIC -shared $(llvm-config --cxxflags) -fno-rtti; no exceptions, no
  iostream. llvm/IR/PatternMatch.h is available. Constants may be computed
  at compile time with <cmath>.
- precondition_in_ir: the IR condition the pass checks before rewriting.
- must_decline: a small C file (no main) with a function that looks similar
  but where the precondition does NOT hold, so the pass must leave it alone.
  It is compiled with the pass; if the pass changes it, the pass is rejected.
- value_preserving: false only if floating-point results may change.

You are shown the kernel's IR at the flags it will be compiled with. Match
the IR as it is, not the C. Output JSON only."""


async def _kernel_ir(kernel: str, flags: str) -> str:
    with tempfile.TemporaryDirectory(prefix="kernel_ir_") as workdir:
        Path(workdir, "kernel.c").write_text(kernel, encoding="utf-8")
        try:
            _, stdout, _ = await agent_sandbox_runtime.run_in_sandbox(
                f"clang {flags} -S -emit-llvm -o - kernel.c",
                workdir,
                image=PASS_IMAGE,
                timeout_seconds=120,
            )
        except Exception as exc:
            logger.info(f"kernel IR unavailable: {exc}")
            return ""
    return stdout[:20000]


def _evidence(result: Dict[str, Any]) -> str:
    data = result.get("data") or {}
    verdict = data.get("verdict")
    if verdict in ("did_not_compile", "pass_crashed", "invalid_ir", "unregistered"):
        return f"It {verdict}:\n{(data.get('detail') or '')[:6000]}"
    if verdict == "did_not_fire":
        return (
            f"It did not fire. {data.get('detail', '')}\n"
            f"The kernel's IR at these flags:\n{(data.get('ir_at_flags') or '')[:8000]}"
        )
    if verdict == "overreaches":
        return (
            "It CHANGED the must_decline case, where its precondition does not "
            f"hold (deltas {data.get('must_decline', {}).get('ir_deltas')}). "
            "Tighten the match condition. The must_decline case is fixed now; "
            "changing it will not be accepted."
        )
    problem = (data.get("equivalence") or {}).get("first_problem") or {}
    return (
        f"The compiled program {verdict} on input #{problem.get('input')}: "
        f"{problem.get('detail')}\noriginal printed: "
        f"{problem.get('expected_excerpt', '')!r}\nwith the pass: "
        f"{problem.get('actual_excerpt', '')!r}"
    )


async def synthesize_pass_from_rewrite(
    *,
    kernel: str,
    rewrite_kernel: str,
    driver: str,
    inputs: List[str],
    idea: str = "",
    invariant: str = "",
    flags: str = DEFAULT_FLAGS,
    bench_input: int = 0,
    label: str = "",
    user_id: Any = None,
    db: Any = None,
) -> Dict[str, Any]:
    """Ask for a pass that generalises `rewrite_kernel`, and judge it."""
    from app.services.agent_restructure_proposer import _call

    if not (rewrite_kernel or "").strip():
        return {
            "error": "rewrite_kernel is required: the hand-optimised kernel to generalise"
        }
    problem, cleaned = check_inputs(inputs, bench_input)
    if problem:
        return {"error": problem}
    safe = _clean_flags(flags or DEFAULT_FLAGS)
    if safe is None:
        return {"error": f"flags contain unsupported characters: {flags!r}"}
    blocked = sandbox_blocked(PASS_IMAGE)
    if blocked:
        return {"error": blocked}

    ir = await _kernel_ir(kernel, safe)
    message = "Generalise this hand optimisation into a pass.\n\n" + (
        f"The idea: {idea}\n" if idea else ""
    ) + (
        f"What the rewrite relies on: {invariant}\n" if invariant else ""
    ) + f"\n=== original kernel.c ===\n{kernel}\n\n" f"=== hand-optimised kernel.c ===\n{rewrite_kernel}\n\n" + (
        f"=== original kernel IR at {safe} ===\n{ir}\n" if ir else ""
    )
    subject = (label or "").strip() or "rewrite"
    attempts: List[Dict[str, Any]] = []
    frozen_decline: Optional[str] = None
    result: Dict[str, Any] = {}
    ask = message

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            payload = await _call(
                PASS_SYSTEM_PROMPT, ask, PASS_SCHEMA, user_id=user_id, db=db
            )
        except Exception as exc:  # pragma: no cover - provider failure
            return {"success": False, "error": f"the model could not be reached: {exc}"}
        if payload.get("_unparsed"):
            attempts.append(
                {
                    "attempt": attempt,
                    "verdict": "unparsed",
                    "detail": payload["_unparsed"],
                }
            )
            continue
        if payload.get("expressible") is False:
            reason = str(payload.get("reason") or "")[:1500]
            attempts.append({"attempt": attempt, "verdict": "not_expressible"})
            return {
                "success": True,
                "data": {
                    "verdict": "not_expressible",
                    "reason": reason,
                    "attempts": attempts,
                    "how_to_read": (
                        "The rewrite relies on a fact the IR does not carry, so "
                        "no sound pass can make it. It stays an "
                        "application-level change."
                    ),
                },
                "findings": [
                    {
                        "type": "pass_evaluation",
                        "subject": subject,
                        "title": f"{subject}: not expressible as a pass",
                        "verdict": "not_expressible",
                        "reason": reason,
                    }
                ],
            }
        pass_name = str(payload.get("pass_name") or "").strip()
        pass_source = str(payload.get("pass_source") or "")
        if frozen_decline is None:
            frozen_decline = str(payload.get("must_decline") or "")
        result = await evaluate_pass_on_kernel(
            pass_source=pass_source,
            pass_name=pass_name,
            kernel=kernel,
            driver=driver,
            inputs=cleaned,
            rewrite_kernel=rewrite_kernel,
            must_decline=frozen_decline,
            value_preserving=payload.get("value_preserving") is not False,
            precondition=str(payload.get("precondition_in_ir") or "")[:600],
            flags=safe,
            bench_input=bench_input,
            label=f"{subject}/{pass_name or 'pass'}",
        )
        if result.get("error") and not result.get("data"):
            # A refused call (a bad pass name, a sandbox fault) is evidence the
            # model can act on only when it is about its own output.
            attempts.append(
                {"attempt": attempt, "verdict": "refused", "detail": result["error"]}
            )
            ask = f"{message}\n\nYour last answer was refused: {result['error']}\nFix that."
            continue
        data = result.get("data") or {}
        verdict = data.get("verdict")
        if verdict == "unresolved":
            # A correct pass whose gain sits inside the noise gets one bigger
            # sample before the verdict stands. Measured: a fastmod pass read
            # unresolved at 7 trials (109% spread on a host whose load the VM
            # cannot see) and faster at 15, 2.48x on the median.
            first = data.get("timing")
            result = await evaluate_pass_on_kernel(
                pass_source=pass_source,
                pass_name=pass_name,
                kernel=kernel,
                driver=driver,
                inputs=cleaned,
                rewrite_kernel=rewrite_kernel,
                must_decline=frozen_decline,
                value_preserving=payload.get("value_preserving") is not False,
                precondition=str(payload.get("precondition_in_ir") or "")[:600],
                flags=safe,
                bench_input=bench_input,
                trials=MAX_TRIALS,
                label=f"{subject}/{pass_name or 'pass'}",
            )
            data = result.setdefault("data", {})
            data["first_measurement"] = first
            verdict = data.get("verdict")
        attempts.append(
            {"attempt": attempt, "verdict": verdict, "pass_name": pass_name}
        )
        data["pass_source"] = pass_source
        data["must_decline_source"] = frozen_decline
        if verdict not in REPAIRABLE:
            break
        ask = (
            f"{message}\n\n=== your pass ({pass_name}) ===\n{pass_source}\n\n"
            f"=== what happened ===\n{_evidence(result)}\n\n"
            "Return the whole answer again, fixed."
        )

    if not result:
        return {
            "success": False,
            "error": "no attempt produced a pass to judge: "
            + "; ".join(str(a.get("detail")) for a in attempts),
        }
    data = result.setdefault("data", {})
    data["attempts"] = attempts
    return result
