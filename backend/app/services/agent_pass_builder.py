"""Build an LLVM pass plugin and report what it actually did.

Every plugin in this repo was written by hand and compiled outside any tool,
which made "synthesize a pass" the one step of scan -> suggest -> synthesize
that no agent could take. This is that step.

WHAT IT REPORTS, AND WHY EACH IS SEPARATE. A pass can fail in four ways that
look alike from outside and need different fixes:

  compiled=False   the C++ is wrong. The compiler's own errors come back
                   verbatim, because they are the remedy.
  registered=False it built, but `opt -passes=<name>` does not know that name,
                   so the plugin's registration callback and the name asked for
                   disagree.
  fired=False      it built, it is registered, and the IR is byte-identical
                   afterwards. This is the failure worth naming loudest: a pass
                   that loads and silently does nothing looks exactly like a
                   pass that works, and this project shipped one. Two plugins
                   registered only a `-passes=` parser, so `clang
                   -fpass-plugin=` accepted the flag, compiled without error,
                   and changed nothing.
  fired=True       and then the opcode deltas say what changed, which is what
                   turns "it ran" into "it did the intended thing".

It does NOT say whether the pass is correct or worth anything. Those are
measurements -- differential execution and a cycle count -- and they belong to
the tools that already do them.
"""

from __future__ import annotations

import asyncio
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger

from app.services import agent_sandbox_runtime

#: Headers and g++ live here and nowhere else: the compiler-research image
#: ships the plugins but not the 500 MB that builds them.
DEFAULT_IMAGE = "ghcr.io/al3x3n0/kdbc-pass-dev:latest"
DEFAULT_TIMEOUT_SECONDS = 300

MAX_SOURCE_CHARS = 400_000
#: The name is interpolated into an `opt -passes=` argument.
SAFE_PASS_NAME = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
SAFE_FLAGS = re.compile(r"^[-A-Za-z0-9_=+., /]*$")

_INSTR = re.compile(r"^\s+(?:%\S+\s+=\s+)?([a-z][a-z0-9._]*)\s", re.M)
#: Opcodes alone cannot see a rewrite that keeps the opcode. Replacing a libm
#: sqrt with llvm.sqrt is call -> call, so the first version of this reported no
#: change for a pass whose own output said it had rewritten three sites.
_CALLEE = re.compile(r"call[^@\n]*(@[\w.]+)")


def opcode_counts(ir: str) -> Dict[str, int]:
    """A rough instruction census, enough to say what a pass changed."""
    skip = {"define", "declare", "attributes", "source_filename", "target", "ret"}
    counts: Counter = Counter()
    for m in _INSTR.finditer(ir or ""):
        op = m.group(1)
        if op not in skip:
            counts[op] += 1
    for m in _CALLEE.finditer(ir or ""):
        counts["call " + m.group(1)] += 1
    return dict(counts)


#: Lines that echo the input filename. baseline.ll and after.ll are produced
#: from differently named inputs by construction, so `; ModuleID = ...` always
#: differs and a byte comparison called every pass "fired" -- including one that
#: returns PreservedAnalyses::all() and touches nothing.
_ECHOES_FILENAME = re.compile(r"^(; ModuleID =|source_filename =).*$", re.M)


def normalised(ir: str) -> str:
    """The IR with the bits that name the file it came from removed."""
    return _ECHOES_FILENAME.sub("", ir or "").strip()


def _deltas(before: Dict[str, int], after: Dict[str, int]) -> Dict[str, int]:
    keys = set(before) | set(after)
    return {
        k: after.get(k, 0) - before.get(k, 0)
        for k in sorted(keys)
        if after.get(k, 0) != before.get(k, 0)
    }


def _preflight(
    source: str, pass_name: str, test_code: str, flags: str, image: str
) -> Optional[Dict[str, Any]]:
    if not (source or "").strip():
        return {"error": "source is required: the C++ of an LLVM pass plugin"}
    if len(source) > MAX_SOURCE_CHARS:
        return {"error": f"source exceeds {MAX_SOURCE_CHARS} characters"}
    if not SAFE_PASS_NAME.match(pass_name or ""):
        return {
            "error": (
                f"pass_name {pass_name!r} is not usable: give the lowercase name "
                "the plugin registers with registerPipelineParsingCallback, "
                "e.g. 'sqrt-errno-elision'"
            )
        }
    if not (test_code or "").strip():
        return {
            "error": (
                "test_code is required: a pass that builds is not a pass that "
                "works, and without an input there is no way to tell whether it "
                "fired"
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


async def build_llvm_pass(
    *,
    source: str,
    pass_name: str,
    test_code: str,
    flags: str = "-O1",
    label: str = "",
    image: str = DEFAULT_IMAGE,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    """Compile a pass plugin, run it on test code, and say what changed."""
    blocked = _preflight(source, pass_name, test_code, flags, image)
    if blocked:
        return blocked

    with tempfile.TemporaryDirectory(prefix="passbuild_") as workdir:
        Path(workdir, "pass.cpp").write_text(source, encoding="utf-8")
        Path(workdir, "test.c").write_text(test_code, encoding="utf-8")
        script = (
            'g++ -fPIC -shared -o pass.so pass.cpp $("$LLVM_CONFIG" --cxxflags) '
            "  -fno-rtti 2>build_err.txt; "
            'if [ $? -ne 0 ]; then echo "PB\tcompiled\t0"; '
            '  echo "PB\tbuild_err"; cat build_err.txt; exit 0; fi; '
            'echo "PB\tcompiled\t1"; '
            f"clang {flags} -S -emit-llvm -o before.ll test.c 2>test_err.txt || "
            '  { echo "PB\ttest_code_err"; cat test_err.txt; exit 0; }; '
            # opt reprints the module whatever it is asked to do, so the
            # baseline is opt with no pass. Comparing against clang's own
            # output reported a do-nothing pass as having fired -- the exact
            # failure this tool exists to catch.
            "opt -passes=verify -S -o baseline.ll before.ll 2>/dev/null; "
            f"opt -load-pass-plugin=./pass.so -passes={pass_name} "
            "  -S -o after.ll baseline.ll 2>pass_out.txt; "
            'echo "PB\topt_rc\t$?"; '
            'echo "PB\tpass_out"; cat pass_out.txt; '
            'echo "PB\tbefore"; cat baseline.ll 2>/dev/null; '
            'echo "PB\tafter"; cat after.ll 2>/dev/null'
        )
        try:
            _rc, stdout, stderr = await agent_sandbox_runtime.run_in_sandbox(
                script, workdir, image=image, timeout_seconds=timeout_seconds
            )
        except asyncio.TimeoutError:
            return {"error": f"pass build timed out after {timeout_seconds}s"}
        except FileNotFoundError:
            return {"error": "Docker is not available to this process"}
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"build_llvm_pass failed: {exc}")
            return {"error": f"pass build failed: {exc}"}
    never_ran = agent_sandbox_runtime.could_not_run(_rc, stderr, image)
    if never_ran:
        return {"success": False, "error": never_ran}

    sections: Dict[str, list] = {}
    current = None
    compiled = False
    opt_rc = None
    for line in (stdout or "").splitlines():
        if line.startswith("PB\t"):
            parts = line.split("\t")
            tag = parts[1]
            if tag == "compiled":
                compiled = parts[2] == "1"
                continue
            if tag == "opt_rc":
                opt_rc = int(parts[2] or 1)
                continue
            current = tag
            sections[current] = []
            continue
        if current:
            sections[current].append(line)

    join = lambda k: "\n".join(sections.get(k, []))  # noqa: E731

    if not compiled:
        return {
            "success": False,
            "compiled": False,
            "error": "the pass did not compile",
            "build_errors": join("build_err")[:8000],
        }
    if "test_code_err" in sections:
        return {
            "success": False,
            "compiled": True,
            "error": "the pass built, but the test code did not compile",
            "test_code_errors": join("test_code_err")[:4000],
        }

    pass_out = join("pass_out")
    unknown_name = "unknown pass" in pass_out.lower()
    if opt_rc not in (0, None) and not unknown_name:
        # opt knew the name and died running it: report_fatal_error, an
        # assertion, a segfault. Calling that "unregistered" sent the run to
        # fix a registration that was fine.
        return {
            "success": False,
            "compiled": True,
            "registered": True,
            "verdict": "pass_crashed",
            "error": (
                f"`opt -passes={pass_name}` found the pass and crashed running "
                f"it (exit {opt_rc}); its output says where."
            ),
            "opt_output": pass_out[:4000],
        }
    registered = opt_rc == 0 and not unknown_name
    if not registered:
        return {
            "success": False,
            "compiled": True,
            "registered": False,
            "error": (
                f"the plugin built, but `opt -passes={pass_name}` did not accept "
                "that name. The name registered by "
                "registerPipelineParsingCallback and the name asked for here "
                "must be the same string."
            ),
            "opt_output": pass_out[:4000],
        }

    before, after = join("before"), join("after")
    fired = normalised(before) != normalised(after)
    deltas = _deltas(opcode_counts(before), opcode_counts(after))
    subject = (label or "").strip() or pass_name

    return {
        "success": True,
        "compiled": True,
        "registered": True,
        "fired": fired,
        "data": {
            "pass_name": pass_name,
            "opcode_deltas": deltas,
            "pass_output": pass_out[:4000],
            "note": (
                "The pass ran and changed the IR."
                if fired
                else (
                    "The pass built and is registered, and left the IR "
                    "byte-identical. That is not the same as having no work to "
                    "do: check the match condition against this test input "
                    "before concluding the input was simply unaffected. A pass "
                    "that loads and silently does nothing is indistinguishable "
                    "from one that works until something compares the output."
                )
            ),
            "what_this_does_not_say": (
                "whether the transformation is correct, and whether it is worth "
                "anything. Those need differential execution and a cycle count."
            ),
        },
        "findings": [
            {
                "type": "pass_built",
                "subject": subject,
                "title": (
                    f"{pass_name}: built, registered, "
                    + (
                        "changed "
                        + ", ".join(f"{k} {v:+d}" for k, v in list(deltas.items())[:5])
                        if fired
                        else "made no change to the test input"
                    )
                ),
                "fired": fired,
                "opcode_deltas": deltas,
            }
        ],
    }
