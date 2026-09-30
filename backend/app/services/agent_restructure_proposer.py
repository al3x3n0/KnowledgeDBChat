"""Ask a model for optimisations no compiler may make, then refuse to believe it.

The scanner finds shapes a written pass handles; `shapes_without_a_pass` is
where the next pass comes from. Both are bounded by what is true of every
program. This asks for the other kind -- changes that are only valid, or only
worth it, because of something true of THIS application: a parameter that is
invariant for a whole call, a value range the inputs never leave, a layout that
suits how the data is actually walked, work the code repeats that it could
keep. A compiler is forbidden from assuming any of those; a reader of the code
is not.

The model proposes; it does not judge. Every proposal goes through
`agent_restructure.evaluate_restructuring` (or the binary variant), which runs
it against the original on the caller's inputs and times it against the
original AND the original rebuilt at -O3 -- so an "idea" that is really a flag
is labelled `compiler_already_can`, and one that is really a bug is labelled
`diverged` and never timed. What the model says about its own proposal
(`invariant`, `why_compiler_cannot`) is carried beside the verdict as a claim,
never as evidence.

ONE REPAIR, NOT A LOOP. A proposal that does not compile, crashes or diverges
goes back once with the evidence -- compiler errors verbatim, or the input on
which it disagreed. A second failure is recorded as the verdict. Iterating
until something passes would select for whatever slips past these particular
inputs, which is the opposite of what a differential check is for.

THE MODEL IS SHOWN THE COMPILER'S OWN OUTPUT. Without it, the cheapest
proposal is the one the compiler already made -- unroll this, vectorise that --
and the ceiling arm would reject it after a full timing run. Showing the -O2
listing up front spends tokens to save containers, and says plainly what
"original" has to beat.
"""

from __future__ import annotations

import asyncio
import base64
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from app.services import agent_sandbox_runtime
from app.services.agent_binary_rewrite import (
    ABI_NOTE,
    disassemble_symbol,
    evaluate_binary_rewrite,
)
from app.services.agent_compiler_sandbox import _clean_flags
from app.services.agent_restructure import (
    DEFAULT_FLAGS,
    DEFAULT_IMAGE,
    DEFAULT_TRIALS,
    MAX_TRIALS,
    MIN_RESOLVABLE,
    check_inputs,
    evaluate_restructuring,
    sandbox_blocked,
)
from app.services.agent_toolchains import LLVM_PASSES

MAX_PROPOSALS = 5
DEFAULT_PROPOSALS = 3
MAX_LISTING_CHARS = 16_000
#: Every proposal carries a whole kernel or a whole assembly file, so three of
#: them run to thousands of tokens. At a provider's default cap the JSON is cut
#: mid-string, parses as nothing, and read as "the model had no ideas". A
#: reasoning model spends from the same budget before it writes a word: at
#: 16k one returned zero characters with stop_reason=length.
MAX_OUTPUT_TOKENS = 32_000
#: A repair gets the evidence and one more chance; see the module docstring.
REPAIRABLE = ("did_not_compile", "crashed", "diverged")

#: Best first. `faster` ranks by its margin over the compiler's ceiling, since
#: beating -O2 alone is table stakes.
RANK = {
    "faster": 0,
    "faster_than_original": 1,
    "compiler_already_can": 2,
    "unresolved": 3,
    "slower": 4,
    "diverged": 5,
    "crashed": 6,
    "did_not_compile": 7,
    "baseline_broken": 8,
}

_PROPOSAL_FIELDS = {
    "name": {"type": "string"},
    "idea": {"type": "string"},
    "invariant": {"type": "string"},
    "why_compiler_cannot": {"type": "string"},
    "value_preserving": {"type": "boolean"},
}

SOURCE_SCHEMA = {
    "type": "object",
    "properties": {
        "proposals": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {**_PROPOSAL_FIELDS, "kernel": {"type": "string"}},
                "required": list(_PROPOSAL_FIELDS) + ["kernel"],
            },
        }
    },
    "required": ["proposals"],
}

BINARY_SCHEMA = {
    "type": "object",
    "properties": {
        "proposals": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {**_PROPOSAL_FIELDS, "assembly": {"type": "string"}},
                "required": list(_PROPOSAL_FIELDS) + ["assembly"],
            },
        }
    },
    "required": ["proposals"],
}


def _known_passes() -> str:
    return "\n".join(f"- {p.name}: {p.summary}" for p in LLVM_PASSES)


_COMMON_RULES = """\
What counts as ORIGINAL here:
- It exploits something true of THIS application that a compiler may not
  assume about arbitrary code: a parameter invariant across a call or a loop,
  a value range the inputs stay in, a shape or size the data always has, work
  repeated across calls or iterations with the same result, an access pattern
  that a different layout or traversal order would serve better, an algorithm
  that is asymptotically or constant-factor better for these inputs.
- It is NOT something the compiler already did (read its listing below:
  vectorised loops, unrolling, LICM, strength reduction by constants and
  inlining are taken), and NOT something a flag gives (-O3, -ffast-math,
  -funroll-loops). Those will be measured against and labelled as flags.
- It is NOT one of the research passes this platform already has:
{passes}

Every proposal is compiled, run against the original on the caller's inputs,
and must print IDENTICAL output (or, if you set value_preserving=false, output
within a 1e-6 relative tolerance). Then it is timed against the original and
against the original at -O3. Proposals that are wrong are caught; do not
propose anything you would not bet is correct.

For each proposal:
- name: short kebab-case name.
- idea: two or three sentences, concrete.
- invariant: the fact about this application it relies on. If it relies on
  none -- it is valid for every input -- say "none".
- why_compiler_cannot: the specific reason clang cannot do this itself.
- value_preserving: false only if floating-point results may change.

Propose DIFFERENT ideas, not variations of one, and apply each ONE idea to
the ORIGINAL kernel on its own. Do not carry one proposal's idea into another:
each is measured separately, and a proposal containing two ideas cannot say
which one paid -- it will be measured against the best of the others, and
credited only with what it adds. Output JSON only."""


def _source_system_prompt() -> str:
    return (
        "You are a performance engineer proposing application-specific "
        "optimisations and restructurings of one C file (the kernel).\n\n"
        + _COMMON_RULES.format(passes=_known_passes())
        + "\n\n- kernel: the COMPLETE replacement kernel C file. Keep every "
        "external function the driver calls, with the same signature. The "
        "driver is not yours to change and is compiled separately. C11, clang, "
        "aarch64; <stdint.h>, <string.h>, <math.h>, <arm_neon.h> are available."
    )


def _binary_system_prompt() -> str:
    return (
        "You are a performance engineer rewriting ONE function of a compiled "
        "program in aarch64 assembly. There is no source: you have its "
        "disassembly with relocations, and a driver showing how it is called.\n\n"
        + _COMMON_RULES.format(passes=_known_passes())
        + "\n\n- assembly: a COMPLETE GNU-as file, assembled by clang, that "
        "defines the function and nothing else global. "
        + ABI_NOTE
        + " Symbols the original references (see relocations) may be "
        "referenced by name. Put the function in .text, and align loops."
    )


def _payload(completion: Any) -> Dict[str, Any]:
    # One parser for every provider shape; see plugin_author_service._payload.
    from app.services.plugin_author_service import _payload as parse

    return parse(completion)


def _lenient_json(text: str) -> Dict[str, Any]:
    """Parse a reply whose strings hold raw newlines and tabs.

    Every proposal carries a whole source file inside a JSON string, and
    models write it with literal newlines as often as with `\\n`. Strict JSON
    rejects those control characters, so a 4,117-character reply full of
    usable proposals parsed as nothing. `strict=False` accepts them and
    nothing else.
    """
    import json

    body = text.strip()
    if body.startswith("```"):
        body = body.strip("`")
        if body[:4].lower() == "json":
            body = body[4:]
    start, end = body.find("{"), body.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        parsed = json.loads(body[start : end + 1], strict=False)
    except ValueError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _json_error(text: str) -> str:
    """The decoder's own complaint, with the text around where it stopped."""
    import json

    start = text.find("{")
    try:
        json.loads(text[start:] if start >= 0 else text, strict=False)
    except ValueError as exc:
        pos = getattr(exc, "pos", None)
        near = (
            f" near {text[max(0, (pos or 0) + max(start, 0) - 40):(pos or 0) + max(start, 0) + 40]!r}"
            if pos is not None
            else ""
        )
        return f"{getattr(exc, 'msg', exc)}{near}"
    return "parsed, but not as an object"


async def _call(
    system: str,
    message: str,
    schema: Dict[str, Any],
    *,
    user_id: Any,
    db: Any,
) -> Dict[str, Any]:
    from app.services.llm_service import LLMService

    completion = await LLMService().generate_structured(
        system_prompt=system,
        user_message=message,
        response_schema=schema,
        task_type="deep",
        max_tokens=MAX_OUTPUT_TOKENS,
        user_id=user_id,
        db=db,
    )
    payload = _payload(completion)
    if not payload:
        payload = _lenient_json(str(getattr(completion, "text", "") or ""))
    if not payload:
        # Say which failure it was: a reply cut at the token cap and a reply
        # that was never JSON need different remedies.
        stop = getattr(completion, "stop_reason", None)
        text = str(getattr(completion, "text", "") or "")
        return {
            "_unparsed": (
                f"reply of {len(text)} characters did not parse as JSON "
                f"(stop_reason={stop!r}; {_json_error(text)})"
                + (
                    "; it was cut at the output-token cap"
                    if stop in ("length", "max_tokens")
                    else ""
                )
            )
        }
    return payload


async def compiler_listing(
    kernel: str, flags: str, image: str = DEFAULT_IMAGE, timeout_seconds: int = 120
) -> str:
    """What the compiler already made of the kernel, for the proposer to beat."""
    with tempfile.TemporaryDirectory(prefix="listing_") as workdir:
        Path(workdir, "kernel.c").write_text(kernel, encoding="utf-8")
        try:
            _, stdout, _ = await agent_sandbox_runtime.run_in_sandbox(
                f"clang {flags} -S -fno-asynchronous-unwind-tables -o - kernel.c "
                "| grep -vE '^\\s*\\.(file|ident|addrsig|section\\s+\\.note|cfi_)'",
                workdir,
                image=image,
                timeout_seconds=timeout_seconds,
            )
        except Exception as exc:
            logger.info(f"compiler listing unavailable: {exc}")
            return ""
    if len(stdout) > MAX_LISTING_CHARS:
        return stdout[:MAX_LISTING_CHARS] + "\n... [listing cut]"
    return stdout


def _usable(p: Any, body_key: str) -> Optional[Dict[str, Any]]:
    if not isinstance(p, dict) or not str(p.get(body_key) or "").strip():
        return None
    return {
        "name": str(p.get("name") or "unnamed")[:60],
        "idea": str(p.get("idea") or "")[:1200],
        "invariant": str(p.get("invariant") or "")[:600],
        "why_compiler_cannot": str(p.get("why_compiler_cannot") or "")[:600],
        "value_preserving": p.get("value_preserving") is not False,
        body_key: str(p[body_key]),
    }


#: What models call the body when they do not use the schema's name.
_BODY_ALIASES = {
    "kernel": ("kernel", "code", "source", "c_source", "candidate"),
    "assembly": ("assembly", "asm", "code", "source", "replacement_asm"),
    "options": ("options", "bolt_options", "flags", "configuration"),
}


def _first_proposal(payload: Any, body_key: str) -> Optional[Dict[str, Any]]:
    """The first usable proposal, in whichever shape the reply took.

    Asked for exactly one, a model often returns the proposal itself rather
    than a one-element `proposals` list -- measured, two of three binary calls
    came back that way and both were reported as "no proposal".
    """
    if not isinstance(payload, dict):
        return None
    raw = payload.get("proposals")
    items = raw if isinstance(raw, list) else [raw] if isinstance(raw, dict) else []
    for item in items + [payload]:
        if not isinstance(item, dict):
            continue
        body = next(
            (
                item[k]
                for k in _BODY_ALIASES[body_key]
                if str(item.get(k) or "").strip()
            ),
            None,
        )
        usable = _usable({**item, body_key: body}, body_key) if body else None
        if usable:
            return usable
    return None


def _repair_message(
    original_message: str,
    proposal: Dict[str, Any],
    body_key: str,
    result: Dict[str, Any],
) -> str:
    data = result.get("data") or {}
    verdict = data.get("verdict")
    if verdict == "did_not_compile":
        evidence = f"It did not build:\n{data.get('compile_errors', '')}"
    else:
        problem = (data.get("equivalence") or {}).get("first_problem") or {}
        evidence = (
            f"It {verdict} on input #{problem.get('input')}: {problem.get('detail')}\n"
            f"original printed: {problem.get('expected_excerpt', '')!r}\n"
            f"yours printed:    {problem.get('actual_excerpt', '')!r}"
        )
    return (
        f"{original_message}\n\nYour proposal '{proposal['name']}' was checked. "
        f"{evidence}\n\nFix it -- or, if the idea is unsound for these inputs, "
        f"say so by returning it unchanged. Return exactly one proposal in "
        f"'proposals', the complete {body_key} again.\n\nThe proposal was:\n"
        f"{proposal[body_key]}"
    )


def _summarise(
    proposal: Dict[str, Any], body_key: str, result: Dict[str, Any], repaired: bool
) -> Dict[str, Any]:
    data = result.get("data") or {}
    timing = data.get("timing") or {}
    return {
        "name": proposal["name"],
        "idea": proposal["idea"],
        "invariant": proposal["invariant"],
        "why_compiler_cannot": proposal["why_compiler_cannot"],
        "value_preserving": proposal["value_preserving"],
        "verdict": data.get("verdict") or ("error" if result.get("error") else None),
        "speedup": timing.get("speedup"),
        "median_speedup": timing.get("median_speedup"),
        "speedup_over_ceiling": timing.get("speedup_over_ceiling"),
        "bit_identical": (data.get("equivalence") or {}).get("bit_identical"),
        "repaired": repaired,
        "notes": data.get("notes") or [],
        # `detail` is where a broken harness explains itself. It was missing
        # here, so a run was told "baseline_broken" with problem null -- a
        # driver that used a type only the kernel defined -- and went
        # browsing the repository instead of fixing the driver.
        "problem": data.get("compile_errors")
        or (data.get("equivalence") or {}).get("first_problem")
        or data.get("detail")
        or (data.get("equivalence") or {}).get("detail")
        or result.get("error"),
        "warnings": timing.get("warnings") or [],
        body_key: proposal[body_key],
    }


async def _remeasure_if_hinted(
    result: Dict[str, Any], proposal: Dict[str, Any], evaluate
) -> Dict[str, Any]:
    """One bigger sample for an equivalent candidate the noise swallowed.

    Only when the fastest trial hints at a gain: re-timing a candidate that
    was never faster would just spend time. Measured on raylib's
    ImageBlurGaussian: three bit-identical proposals, all unresolved at 7
    trials, the best at 1.53x fastest and 1.20x median -- the same shape a
    fastmod pass showed before resolving at 15 trials to 2.48x.
    """
    data = result.get("data") or {}
    speedup = (data.get("timing") or {}).get("speedup") or 0
    if data.get("verdict") != "unresolved" or speedup <= 1 + MIN_RESOLVABLE:
        return result
    again = await evaluate(proposal, None, MAX_TRIALS)
    again_data = again.get("data")
    if isinstance(again_data, dict):
        again_data["first_measurement"] = data.get("timing")
        return again
    return result


async def _attribute(judged: List[Dict[str, Any]], evaluate) -> List[Dict[str, Any]]:
    """Credit each winner only with what it adds over the best winner.

    Asked for three different ideas, a model returned three kernels that all
    carried the same lookup table, named for what else they did: "branchless
    masking" was 6.3x faster than the original, and the table was 7.2x on its
    own. Every verdict was true and every name was a misattribution. So each
    other winner is re-run with the best one as its baseline: if it is not
    faster than that, the idea it is named for added nothing that was
    measured.
    """
    winners = [j for j in judged if j.get("verdict") == "faster"]
    findings: List[Dict[str, Any]] = []
    if len(winners) < 2:
        return findings
    best = winners[0]
    for other in winners[1:]:
        result = await evaluate(other, best)
        data = result.get("data") or {}
        verdict = data.get("verdict")
        timing = data.get("timing") or {}
        other["over_best"] = {
            "baseline": best["name"],
            "verdict": verdict,
            "speedup": timing.get("speedup"),
        }
        if verdict != "faster":
            other["notes"] = list(other.get("notes") or []) + [
                # Two cases the measurement cannot tell apart, so neither is
                # claimed: it carried the best one's idea (three "different"
                # proposals once all held one lookup table), or it is a
                # different idea that is simply not as good (a LUT at 1.51x
                # beside NEON at 2.53x). Either way, it adds nothing on top.
                f"Measured against {best['name']} it is {verdict}, so it adds "
                f"nothing on top of {best['name']}: either it carries the same "
                "idea, or it is a different one that is not as good. Check "
                "its source to tell which."
            ]
        findings.extend(result.get("findings") or [])
    # Winners that add nothing of their own rank after winners that do.
    judged.sort(
        key=lambda j: (
            _rank_key(j)[0],
            (j.get("over_best") or {}).get("verdict") not in (None, "faster"),
            _rank_key(j)[1],
        )
    )
    return findings


def _rank_key(entry: Dict[str, Any]):
    return (
        RANK.get(entry.get("verdict"), 9),
        -(entry.get("speedup_over_ceiling") or entry.get("speedup") or 0),
    )


async def _propose_and_judge(
    *,
    system: str,
    message: str,
    schema: Dict[str, Any],
    body_key: str,
    count: int,
    evaluate,
    user_id: Any,
    db: Any,
) -> Dict[str, Any]:
    # One proposal per call, each told what is already taken. Asking for
    # three at once put three whole files in one reply, and for assembly a
    # reasoning model spent the entire 32k budget before writing a character.
    # It also makes independence structural rather than requested: each call
    # starts from the original and is told which ideas to stay away from.
    proposals: List[Dict[str, Any]] = []
    failures: List[str] = []
    for _ in range(count):
        taken = "".join(f"\n- {p['name']}: {p['idea'][:200]}" for p in proposals)
        ask = message + (
            f"\n\nAlready proposed -- propose something DIFFERENT:{taken}"
            if taken
            else ""
        )
        try:
            payload = await _call(system, ask, schema, user_id=user_id, db=db)
        except Exception as exc:  # pragma: no cover - provider failure
            failures.append(f"the model could not be reached: {exc}")
            break
        found = _first_proposal(payload, body_key)
        if found:
            proposals.append(found)
        else:
            failures.append(
                str(
                    payload.get("_unparsed")
                    or f"no proposal with a non-empty {body_key}; the reply's "
                    f"keys were {sorted(payload)[:12]}"
                )
            )
    if not proposals:
        return {
            "error": "the model returned no usable proposals: " + "; ".join(failures)
        }

    judged: List[Dict[str, Any]] = []
    findings: List[Dict[str, Any]] = []
    for proposal in proposals:
        result = await evaluate(proposal)
        repaired = False
        verdict = (result.get("data") or {}).get("verdict")
        if verdict in REPAIRABLE:
            try:
                fix = await _call(
                    system,
                    _repair_message(message, proposal, body_key, result),
                    schema,
                    user_id=user_id,
                    db=db,
                )
                fixed = _first_proposal(fix, body_key)
            except Exception as exc:  # pragma: no cover - provider failure
                logger.info(f"repair call failed: {exc}")
                fixed = None
            if fixed and fixed[body_key].strip() != proposal[body_key].strip():
                fixed["name"] = proposal["name"]
                proposal, repaired = fixed, True
                result = await evaluate(proposal)
        result = await _remeasure_if_hinted(result, proposal, evaluate)
        if result.get("error") and not result.get("data"):
            # A sandbox failure says nothing about the proposal; stop rather
            # than spend the remaining proposals against a broken harness.
            judged.append(_summarise(proposal, body_key, result, repaired))
            break
        judged.append(_summarise(proposal, body_key, result, repaired))
        findings.extend(result.get("findings") or [])
        if (result.get("data") or {}).get("verdict") == "baseline_broken":
            # Every other proposal would be judged against the same broken
            # baseline; the caller has to fix the harness first.
            break

    judged.sort(key=_rank_key)
    findings.extend(await _attribute(judged, evaluate))
    tally: Dict[str, int] = {}
    for entry in judged:
        tally[str(entry["verdict"])] = tally.get(str(entry["verdict"]), 0) + 1
    out = {"proposals": judged, "verdicts": tally, "findings": findings}
    if failures:
        out["proposal_failures"] = failures
    return out


async def propose_restructurings(
    *,
    kernel: str,
    driver: str,
    inputs: List[str],
    focus: str = "",
    count: int = DEFAULT_PROPOSALS,
    flags: str = DEFAULT_FLAGS,
    bench_input: int = 0,
    label: str = "",
    user_id: Any = None,
    db: Any = None,
) -> Dict[str, Any]:
    """Ask for application-specific rewrites of `kernel`, and judge each one."""
    problem, cleaned = check_inputs(inputs, bench_input)
    if problem:
        return {"error": problem}
    if not (kernel or "").strip() or not (driver or "").strip():
        return {"error": "kernel and driver are both required"}
    safe = _clean_flags(flags or DEFAULT_FLAGS)
    if safe is None:
        return {"error": f"flags contain unsupported characters: {flags!r}"}
    blocked = sandbox_blocked(DEFAULT_IMAGE)
    if blocked:
        return {"error": blocked}
    count = max(1, min(int(count or DEFAULT_PROPOSALS), MAX_PROPOSALS))

    listing = await compiler_listing(kernel, safe)
    message = (
        "Propose ONE application-specific optimisation of this kernel.\n\n"
        + (f"Where to look: {focus}\n\n" if focus else "")
        + f"=== kernel.c ===\n{kernel}\n\n=== driver.c (fixed; shows how the "
        f"kernel is called and what inputs look like) ===\n{driver}\n\n"
        f"=== an input (bench) ===\n{cleaned[bench_input][:2000]}\n\n"
        + (
            f"=== what clang {safe} already made of kernel.c ===\n{listing}\n"
            if listing
            else ""
        )
    )
    subject = (label or "").strip() or "kernel"

    async def evaluate(
        proposal: Dict[str, Any],
        baseline: Optional[Dict[str, Any]] = None,
        trials: int = DEFAULT_TRIALS,
    ) -> Dict[str, Any]:
        return await evaluate_restructuring(
            kernel=baseline["kernel"] if baseline else kernel,
            candidate=proposal["kernel"],
            driver=driver,
            inputs=cleaned,
            value_preserving=proposal["value_preserving"],
            invariant=proposal["invariant"],
            flags=safe,
            bench_input=bench_input,
            trials=trials,
            label=f"{subject}/{proposal['name']}"
            + (f" over {baseline['name']}" if baseline else ""),
        )

    out = await _propose_and_judge(
        system=_source_system_prompt(),
        message=message,
        schema=SOURCE_SCHEMA,
        body_key="kernel",
        count=count,
        evaluate=evaluate,
        user_id=user_id,
        db=db,
    )
    return _as_result(out, subject, "source")


async def propose_binary_rewrites(
    *,
    symbol: str,
    driver: str,
    inputs: List[str],
    object_b64: str = "",
    kernel: str = "",
    focus: str = "",
    count: int = DEFAULT_PROPOSALS,
    flags: str = DEFAULT_FLAGS,
    bench_input: int = 0,
    label: str = "",
    user_id: Any = None,
    db: Any = None,
) -> Dict[str, Any]:
    """Ask for machine-code rewrites of one symbol, from its disassembly alone.

    When `kernel` is given it is compiled to the object and then withheld from
    the model: the point of this mode is what can be done without source, and
    a proposer that has read the C is not testing that.
    """
    problem, cleaned = check_inputs(inputs, bench_input)
    if problem:
        return {"error": problem}
    count = max(1, min(int(count or DEFAULT_PROPOSALS), MAX_PROPOSALS))
    if kernel and not object_b64:
        # Compile once here so every evaluation links the SAME object the
        # model read -- recompiling per evaluation would be the same bytes
        # today and a different object the day the toolchain changes.
        built = await _compile_object(kernel, flags)
        if "error" in built:
            return built
        object_b64, kernel = built["object_b64"], ""

    listing = await disassemble_symbol(
        symbol=symbol, object_b64=object_b64, flags=flags
    )
    if listing.get("error"):
        return {"error": listing["error"]}
    info = listing["data"]
    message = (
        f"Propose ONE rewrite of `{symbol}` in aarch64 assembly.\n\n"
        + (f"Where to look: {focus}\n\n" if focus else "")
        + f"=== disassembly of {symbol} (with relocations) ===\n{info['listing']}\n\n"
        f"=== symbols the object defines ===\n"
        + "\n".join(info["defined_symbols"])
        + "\n\n"
        f"=== driver.c (fixed; shows how {symbol} is called) ===\n{driver}\n\n"
        f"=== an input (bench) ===\n{cleaned[bench_input][:2000]}\n"
    )
    subject = (label or "").strip() or symbol

    async def evaluate(
        proposal: Dict[str, Any],
        baseline: Optional[Dict[str, Any]] = None,
        trials: int = DEFAULT_TRIALS,
    ) -> Dict[str, Any]:
        return await evaluate_binary_rewrite(
            symbol=symbol,
            replacement_asm=proposal["assembly"],
            baseline_asm=baseline["assembly"] if baseline else "",
            driver=driver,
            inputs=cleaned,
            object_b64=object_b64,
            value_preserving=proposal["value_preserving"],
            invariant=proposal["invariant"],
            flags=flags,
            bench_input=bench_input,
            trials=trials,
            label=f"{subject}/{proposal['name']}"
            + (f" over {baseline['name']}" if baseline else ""),
        )

    out = await _propose_and_judge(
        system=_binary_system_prompt(),
        message=message,
        schema=BINARY_SCHEMA,
        body_key="assembly",
        count=count,
        evaluate=evaluate,
        user_id=user_id,
        db=db,
    )
    result = _as_result(out, subject, "binary")
    if result.get("success"):
        result["data"]["original_instructions"] = info["instructions"]
        result["data"]["calls_through_symbol_in_object"] = info[
            "calls_through_symbol_in_object"
        ]
    return result


async def _compile_object(kernel: str, flags: str) -> Dict[str, str]:
    # A dict either way: success and failure were both strings, and the
    # object's base64 came back to the caller as an error message.
    safe = _clean_flags(flags or DEFAULT_FLAGS)
    if safe is None:
        return {"error": f"flags contain unsupported characters: {flags!r}"}
    blocked = sandbox_blocked(DEFAULT_IMAGE)
    if blocked:
        return {"error": blocked}
    with tempfile.TemporaryDirectory(prefix="object_") as workdir:
        Path(workdir, "kernel.c").write_text(kernel, encoding="utf-8")
        try:
            rc, _, stderr = await agent_sandbox_runtime.run_in_sandbox(
                f"clang {safe} -c kernel.c -o kernel.o",
                workdir,
                image=DEFAULT_IMAGE,
                timeout_seconds=120,
            )
        except (asyncio.TimeoutError, FileNotFoundError) as exc:
            return {
                "error": f"could not compile the kernel to an object: {exc.__class__.__name__}"
            }
        obj = Path(workdir, "kernel.o")
        if rc != 0 or not obj.exists():
            return {"error": "kernel did not compile: " + (stderr or "")[:1500]}
        return {"object_b64": base64.b64encode(obj.read_bytes()).decode("ascii")}


def _as_result(out: Dict[str, Any], subject: str, mode: str) -> Dict[str, Any]:
    if out.get("error"):
        return {"success": False, "error": out["error"]}
    proposals = out["proposals"]
    broken = next((p for p in proposals if p["verdict"] == "baseline_broken"), None)
    if broken is not None:
        # Nothing was judged: every candidate would be compared with an
        # original that does not build or run. That is the whole answer, so
        # it is the headline rather than one proposal's verdict.
        return {
            "success": False,
            "error": (
                "the ORIGINAL kernel does not build or run with this driver and "
                "these inputs, so no proposal was judged. Fix the harness: "
                f"{broken.get('problem') or 'no detail was reported'}"
            ),
            "data": {"verdicts": out["verdicts"], "proposals": proposals},
        }
    winners = [p for p in proposals if p["verdict"] == "faster"]
    borrowed = [
        p
        for p in winners
        if (p.get("over_best") or {}).get("verdict") not in (None, "faster")
    ]
    headline = (
        f"{len(winners)} of {len(proposals)} {mode} proposal(s) for {subject} "
        "beat both the original and the compiler ceiling"
        + (
            f"; best: {winners[0]['name']} at {winners[0]['speedup']}x"
            if winners
            else ""
        )
        + (
            f"; {len(borrowed)} of those add nothing measurable over it"
            if borrowed
            else ""
        )
    )
    return {
        "success": True,
        "data": {
            "summary": headline,
            "verdicts": out["verdicts"],
            **(
                {"proposal_failures": out["proposal_failures"]}
                if out.get("proposal_failures")
                else {}
            ),
            "proposals": proposals,
            "how_to_read": (
                "Proposals are the model's; verdicts are measurements. "
                "'faster' means identical output on every input and a win over "
                "the original and over the compiler at -O3 that exceeds the "
                "trial noise. 'invariant' is what the proposal relies on and "
                "was checked only on the inputs given -- widen them before "
                "adopting one."
            ),
        },
        "findings": out["findings"],
    }
