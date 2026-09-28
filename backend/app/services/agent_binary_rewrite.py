"""Judge a rewrite of one function in machine code, where there is no source.

`agent_restructure` compares two kernels written in C. This is the same
judgement one level down: the application arrives as a relocatable object (or
C that is compiled to one and then treated as if its source were lost), and a
candidate is hand-written assembly for ONE symbol in it.

HOW THE REPLACEMENT IS SPLICED IN. The original object's copy of the symbol is
made weak (`llvm-objcopy --weaken-symbol`), and the replacement is assembled as
a strong definition beside it; the linker prefers the strong one. Everything
else in the object -- other functions, data, relocations -- is linked exactly
as it came, so the only difference between the two programs is that function.

THE TRAP THIS HAS TO CATCH. If the replacement does not define the symbol
globally -- a missing `.globl`, a typo in the name, a local label -- the link
still succeeds, silently using the weakened original, and the "candidate" is
the original measured twice. It would come back equivalent and unresolved,
which reads as "correct and no faster" when the truth is "never ran". So the
replacement's own symbol table is checked before anything is linked, and a
replacement that does not export the symbol is refused by name.

WHAT IT CANNOT SEE. A caller inside the object that INLINED the function keeps
the original body; only calls that go through the symbol are replaced. The
number of such calls inside the object is reported, so a caller can tell a
function the object calls through its symbol from one it has inlined away.
The driver always calls through the symbol, so what is timed is the rewrite.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from app.services import agent_sandbox_runtime
from app.services.agent_compiler_sandbox import _clean_flags
from app.services.agent_restructure import (
    DEFAULT_FLAGS,
    DEFAULT_IMAGE,
    DEFAULT_TIMEOUT_SECONDS,
    DEFAULT_TOLERANCE,
    DEFAULT_TRIALS,
    MAX_SOURCE_CHARS,
    MAX_TRIALS,
    VALUE_CHANGING_TOLERANCE,
    Arm,
    check_inputs,
    package,
    run_comparison,
    sandbox_blocked,
)

#: Interpolated into shell commands and objcopy arguments: a C identifier.
SAFE_SYMBOL = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
MAX_OBJECT_BYTES = 8 * 1024 * 1024
#: The disassembly a proposer reads. A hot function is rarely longer; one that
#: is will be cut, and the cut is marked so nobody mistakes it for the end.
MAX_LISTING_CHARS = 40_000

#: The ABI a replacement must keep, stated once for the proposer and the tool
#: description. The sandbox is aarch64; a rewrite that clobbers a callee-saved
#: register passes a leaf-function test and corrupts its caller's state later.
ABI_NOTE = (
    "aarch64 AAPCS64: arguments in x0-x7 / v0-v7, result in x0 or v0; x19-x28, "
    "x29, x30, sp and the low 64 bits of v8-v15 must be preserved; export the "
    "symbol with `.globl <name>` and `.type <name>, %function`."
)


def _decode_object(object_b64: str) -> Any:
    try:
        raw = base64.b64decode(object_b64 or "", validate=True)
    except (binascii.Error, ValueError):
        return "object_b64 is not valid base64"
    if len(raw) > MAX_OBJECT_BYTES:
        return f"object exceeds {MAX_OBJECT_BYTES} bytes"
    if raw[:4] != b"\x7fELF":
        return "object is not an ELF file; give a relocatable .o built for aarch64"
    return raw


def _object_source(object_b64: str, kernel: str, flags: str) -> Any:
    """Where orig.o comes from: the caller's bytes, or their C compiled here.

    Returns (files, prep_line) or an error string.
    """
    if object_b64 and kernel:
        return "give object_b64 or kernel, not both"
    if object_b64:
        raw = _decode_object(object_b64)
        if isinstance(raw, str):
            return raw
        return {"orig.o": raw}, "true"
    if kernel:
        if len(kernel) > MAX_SOURCE_CHARS:
            return f"kernel exceeds {MAX_SOURCE_CHARS} characters"
        return {"kernel.c": kernel}, f"clang {flags} -c kernel.c -o orig.o"
    return "object_b64 (a relocatable ELF, base64) or kernel (C to compile to one) is required"


async def disassemble_symbol(
    *,
    symbol: str,
    object_b64: str = "",
    kernel: str = "",
    flags: str = DEFAULT_FLAGS,
    image: str = DEFAULT_IMAGE,
    timeout_seconds: int = 120,
) -> Dict[str, Any]:
    """The machine code of one function, with relocations, and its neighbours.

    Relocations are what make the listing usable as a starting point: without
    them a `bl` or an `adrp` shows an address inside an unlinked object, and
    the name it refers to -- which a replacement has to reference -- is lost.
    """
    if not SAFE_SYMBOL.match(symbol or ""):
        return {"error": f"symbol {symbol!r} is not a C identifier"}
    safe = _clean_flags(flags or DEFAULT_FLAGS)
    if safe is None:
        return {"error": f"flags contain unsupported characters: {flags!r}"}
    source = _object_source(object_b64, kernel, safe)
    if isinstance(source, str):
        return {"error": source}
    files, prep = source
    blocked = sandbox_blocked(image)
    if blocked:
        return {"error": blocked}

    script = (
        f"{{ {prep}; }} 2>prep.log || {{ cat prep.log; echo '__prep_failed__'; exit 0; }}; "
        "echo '__nm__'; llvm-nm --defined-only orig.o; echo '__nm_end__'; "
        f"echo '__asm__'; llvm-objdump -dr --no-show-raw-insn "
        f"--disassemble-symbols={symbol} orig.o | sed -n '/^Disassembly/,$p'; "
        "echo '__asm_end__'; "
        f"echo \"__calls__ $(llvm-objdump -dr orig.o | grep -cE 'R_AARCH64_(CALL|JUMP)26[[:space:]]+{symbol}$')\""
    )
    with tempfile.TemporaryDirectory(prefix="disasm_") as workdir:
        for name, content in files.items():
            target = Path(workdir, name)
            if isinstance(content, bytes):
                target.write_bytes(content)
            else:
                target.write_text(content, encoding="utf-8")
        try:
            _, stdout, stderr = await agent_sandbox_runtime.run_in_sandbox(
                script, workdir, image=image, timeout_seconds=timeout_seconds
            )
        except asyncio.TimeoutError:
            return {"error": f"disassembly timed out after {timeout_seconds}s"}
        except FileNotFoundError:
            return {"error": "Docker is not available to this process"}

    if "__prep_failed__" in stdout:
        return {
            "error": "kernel did not compile: "
            + stdout.split("__prep_failed__")[0][-1500:]
        }
    symbols = _between(stdout, "__nm__", "__nm_end__")
    listing = _between(stdout, "__asm__", "__asm_end__").strip()
    defined = [
        line.split()[-1] for line in symbols.splitlines() if len(line.split()) >= 3
    ]
    if symbol not in defined:
        return {
            "error": (
                f"{symbol} is not defined in the object. Defined symbols: "
                + ", ".join(defined[:40])
            )
        }
    truncated = len(listing) > MAX_LISTING_CHARS
    calls = re.search(r"__calls__ (\d+)", stdout)
    return {
        "success": True,
        "data": {
            "symbol": symbol,
            "listing": listing[:MAX_LISTING_CHARS]
            + ("\n... [listing cut here; the function continues]" if truncated else ""),
            "instructions": sum(
                1
                for line in listing.splitlines()
                if re.match(r"^\s+[0-9a-f]+:\s", line)
            ),
            "defined_symbols": symbols.strip().splitlines()[:80],
            "calls_through_symbol_in_object": int(calls.group(1)) if calls else None,
            "abi": ABI_NOTE,
        },
    }


def _spliced_build(stem: str, binary: str, symbol: str, flags: str) -> str:
    """Link the driver against orig.o with `symbol` taken from `<stem>.s`.

    The export check is part of the build, so its failure is a did_not_compile
    naming the defect -- not a link that quietly succeeds against the weakened
    original. `false`, not `exit 1`: this runs inside a `{ ...; }` group in the
    shared script, where `exit` ends the whole script -- the log is never
    printed and every later arm silently never runs.
    """
    return (
        f"clang -c {stem}.s -o {stem}.o && "
        f"{{ llvm-nm --defined-only {stem}.o | grep -qE '^[0-9a-f]+ T {symbol}$' || "
        f"{{ echo 'error: {stem}.s does not define a GLOBAL function {symbol} "
        f"(needs .globl {symbol}); linking would silently keep the original'; false; }}; }} && "
        f"llvm-objcopy --weaken-symbol={symbol} orig.o weakened_{stem}.o && "
        f"clang {flags} -o {binary} driver.o weakened_{stem}.o {stem}.o -lm"
    )


def _between(text: str, start: str, end: str) -> str:
    a = text.find(start)
    b = text.find(end, a + len(start)) if a >= 0 else -1
    return text[a + len(start) : b] if a >= 0 and b > a else ""


async def evaluate_binary_rewrite(
    *,
    symbol: str,
    replacement_asm: str,
    driver: str,
    inputs: List[str],
    object_b64: str = "",
    kernel: str = "",
    baseline_asm: str = "",
    value_preserving: bool = True,
    invariant: str = "",
    flags: str = DEFAULT_FLAGS,
    bench_input: int = 0,
    trials: int = DEFAULT_TRIALS,
    label: str = "",
    image: str = DEFAULT_IMAGE,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    """Is the object with `symbol` replaced the same program, and faster?

    `baseline_asm`, when given, replaces the symbol in the baseline arm too, so
    one rewrite can be measured against another rather than only against the
    compiler's code.
    """
    if not SAFE_SYMBOL.match(symbol or ""):
        return {"error": f"symbol {symbol!r} is not a C identifier"}
    if not (replacement_asm or "").strip():
        return {
            "error": "replacement_asm is required: assembly defining "
            + (symbol or "the symbol")
        }
    if len(replacement_asm) > MAX_SOURCE_CHARS or len(driver or "") > MAX_SOURCE_CHARS:
        return {
            "error": f"replacement_asm and driver are capped at {MAX_SOURCE_CHARS} characters"
        }
    if "main(" not in (driver or "").replace(" ", ""):
        return {
            "error": "driver must define main: it reads stdin, calls the symbol and prints the result"
        }
    problem, cleaned = check_inputs(inputs, bench_input)
    if problem:
        return {"error": problem}
    safe = _clean_flags(flags or DEFAULT_FLAGS)
    if safe is None:
        return {"error": f"flags contain unsupported characters: {flags!r}"}
    source = _object_source(object_b64, kernel, safe)
    if isinstance(source, str):
        return {"error": source}
    files, obj_prep = source
    blocked = sandbox_blocked(image)
    if blocked:
        return {"error": blocked}

    files = {**files, "driver.c": driver, "replacement.s": replacement_asm}
    if baseline_asm:
        files["baseline.s"] = baseline_asm
    arms = [
        Arm(
            "orig",
            _spliced_build("baseline", "orig", symbol, safe)
            if baseline_asm
            else f"clang {safe} -o orig driver.o orig.o -lm",
        ),
        Arm("cand", _spliced_build("replacement", "cand", symbol, safe)),
    ]
    result = await run_comparison(
        files=files,
        prep=f"{obj_prep} && clang {safe} -c driver.c -o driver.o",
        arms=arms,
        inputs=cleaned,
        bench_input=bench_input,
        trials=max(3, min(int(trials or DEFAULT_TRIALS), MAX_TRIALS)),
        tolerance=DEFAULT_TOLERANCE if value_preserving else VALUE_CHANGING_TOLERANCE,
        image=image,
        timeout_seconds=timeout_seconds,
    )
    return package(
        result,
        kind="binary_rewrite_result",
        label=label or symbol,
        invariant=invariant,
        value_preserving=value_preserving,
        n_inputs=len(cleaned),
        extra={
            "symbol": symbol,
            "scope": (
                "only calls through the symbol are replaced; any copy the "
                "object inlined into another function keeps the original body"
            ),
        },
    )
