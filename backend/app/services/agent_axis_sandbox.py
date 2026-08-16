"""Elaborate and prove instruction-set extension descriptions with AXIS.

AXIS is an architecture description language: one `.axisl` source describes a
proposed extension, and the toolchain generates the collateral a proposal needs
-- encoder, decoder, executable semantics, compiler patterns, a golden
reference model, and SMT-LIB bit-vector semantics.

That matters because it changes what a proposal *is*. Hand-written assembly
patches and per-candidate compiler edits are not reviewable, not regenerable,
and not checkable; a description that elaborates into all of them is all three,
and it is what makes an evidence bundle reproducible from a single file.

It also supplies the strongest gate available here. Cycle counts say a
sequence is faster; an SMT proof says the replacement computes the same thing
for every input. A candidate that fails the proof is wrong no matter what the
simulator measured.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger

from app.services import agent_sandbox_runtime

DEFAULT_IMAGE = "ghcr.io/al3x3n0/kdbc-axis-research:latest"
DEFAULT_TIMEOUT_SECONDS = 180
# A real ISA description is large: the AArch64 translation seed alone is 510KB.
# The cap is here to refuse an absurd payload, not to define what an
# architecture description may contain.
MAX_SOURCE_CHARS = 4_000_000
MAX_OUTPUT_CHARS = 60_000

# Emit targets the tool exposes. Restricted to a known set because the target
# is interpolated into a shell command, and named explicitly so the catalog
# documents what a caller can actually ask for.
EMIT_TARGETS = {
    # Structure and identity
    "json": "emit-json",
    "bundle-manifest": "emit-bundle-manifest",
    "legality-json": "emit-legality-json",
    # Encoding and decoding: what puts the instruction into a binary
    "encode-c": "emit-encode-c",
    "encode-json": "emit-encode-json",
    "decode-c": "emit-decode-c",
    "decode-json": "emit-decode-json",
    "roundtrip-json": "emit-roundtrip-json",
    "asm-disasm-json": "emit-asm-disasm-json",
    # Semantics: what a simulator and a reference model need
    "semantics-c": "emit-semantics-c",
    "semantics-rust": "emit-semantics-rust",
    "semantics-json": "emit-semantics-json",
    "sim-c": "emit-sim-c",
    "exec-c": "emit-exec-c",
    "exec-python": "emit-exec-python",
    "golden-python": "emit-golden-python",
    "smt2": "emit-smt2",
    # Compiler collateral. The TableGen backend emits RISC-V instruction
    # formats, so check it before relying on it for another target.
    "tablegen": "emit-tablegen",
    "llvm-ir": "emit-llvm-ir",
    "llvm-patterns": "emit-llvm-patterns",
    "llvm-intrinsics": "emit-llvm-intrinsics",
    "intrinsics": "emit-intrinsics",
    # Hardware
    "pyrtl": "emit-pyrtl",
}

SOLVER_VERDICTS = ("unsat", "sat", "unknown")


def _preflight(source: str, image: str) -> Optional[Dict[str, Any]]:
    if not (source or "").strip():
        return {"error": "source is required"}
    if len(source) > MAX_SOURCE_CHARS:
        return {"error": f"source exceeds {MAX_SOURCE_CHARS} characters"}
    if not agent_sandbox_runtime.execution_enabled():
        return {
            "error": (
                "Sandboxed execution is disabled on this server "
                "(ENABLE_UNSAFE_CODE_EXECUTION is false)."
            )
        }
    if image not in agent_sandbox_runtime.allowed_images():
        return {
            "error": (
                f"Image {image} is not allowlisted. Allowed: "
                f"{', '.join(agent_sandbox_runtime.allowed_images()) or 'none'}"
            )
        }
    return None


def _first_line(text: str) -> str:
    return next(
        (line.strip() for line in (text or "").splitlines() if line.strip()), ""
    )


async def check_description(
    *,
    source: str,
    image: str = DEFAULT_IMAGE,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    """Validate an AXIS description, reporting what is wrong with it."""
    blocked = _preflight(source, image)
    if blocked:
        return blocked

    with tempfile.TemporaryDirectory(prefix="axis_check_") as workdir:
        Path(workdir, "model.axisl").write_text(source, encoding="utf-8")
        try:
            returncode, stdout, stderr = await agent_sandbox_runtime.run_in_sandbox(
                "axis check model.axisl",
                workdir,
                image=image,
                timeout_seconds=timeout_seconds,
            )
        except TimeoutError:
            return {"error": f"axis check timed out after {timeout_seconds}s"}
        except FileNotFoundError:
            return {"error": "Docker is not available to this process"}
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"axis check failed: {exc}")
            return {"error": f"axis check failed: {exc}"}

    if returncode != 0:
        # AXIS already said what is wrong and where; repeating "check failed"
        # and filing the reason elsewhere means it may never be read.
        detail = _first_line(stderr) or _first_line(stdout)
        return {
            "success": False,
            "error": f"AXIS rejected the description: {detail[:500]}",
            "stderr": stderr[:MAX_OUTPUT_CHARS],
        }
    return {
        "success": True,
        "data": {"result": _first_line(stdout) or "ok"},
        "findings": [
            {
                "type": "axis_description_valid",
                "title": "AXIS description passes check",
            }
        ],
    }


async def emit_artifact(
    *,
    source: str,
    target: str,
    image: str = DEFAULT_IMAGE,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    """Generate one artifact from a description: decoder, semantics, SMT, ..."""
    blocked = _preflight(source, image)
    if blocked:
        return blocked

    requested = str(target or "").strip().lower()
    command = EMIT_TARGETS.get(requested)
    if command is None:
        return {
            "error": (
                f"Unknown emit target: {target!r}. Available: "
                f"{', '.join(sorted(EMIT_TARGETS))}"
            )
        }

    with tempfile.TemporaryDirectory(prefix="axis_emit_") as workdir:
        Path(workdir, "model.axisl").write_text(source, encoding="utf-8")
        try:
            returncode, stdout, stderr = await agent_sandbox_runtime.run_in_sandbox(
                f"axis {command} model.axisl",
                workdir,
                image=image,
                timeout_seconds=timeout_seconds,
            )
        except TimeoutError:
            return {"error": f"axis {command} timed out after {timeout_seconds}s"}
        except FileNotFoundError:
            return {"error": "Docker is not available to this process"}
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"axis {command} failed: {exc}")
            return {"error": f"axis {command} failed: {exc}"}

    if returncode != 0:
        detail = _first_line(stderr) or _first_line(stdout)
        return {
            "success": False,
            "error": f"AXIS could not emit {requested}: {detail[:500]}",
            "stderr": stderr[:MAX_OUTPUT_CHARS],
        }

    truncated = len(stdout) > MAX_OUTPUT_CHARS
    return {
        "success": True,
        "data": {
            "target": requested,
            "command": command,
            "artifact": stdout[:MAX_OUTPUT_CHARS],
            # Say so rather than hand back a clipped artifact that looks whole:
            # a truncated decoder still parses as text and compiles as nothing.
            "truncated": truncated,
            "artifact_chars": len(stdout),
        },
        "findings": [
            {
                "type": "axis_artifact",
                "title": f"AXIS emitted {requested} ({len(stdout)} chars)",
                "target": requested,
                "truncated": truncated,
            }
        ],
    }


def parse_solver_verdict(output: str) -> str:
    """Read z3's answer, ignoring anything it printed around it."""
    for line in (output or "").splitlines():
        candidate = line.strip().lower()
        if candidate in SOLVER_VERDICTS:
            return candidate
    return "error"


async def prove_equivalence(
    *,
    source: str,
    obligation: str,
    image: str = DEFAULT_IMAGE,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    """Discharge an SMT obligation against a description's own semantics.

    The obligation is appended to the semantics AXIS emits, so it can call the
    generated functions by name. State it as the *negation* of the claim and
    end with (check-sat): `unsat` then means no counterexample exists over all
    inputs, which is the only verdict that proves anything. `sat` returns a
    counterexample, and is a result worth having -- it is the candidate being
    wrong before it reaches a simulator.
    """
    blocked = _preflight(source, image)
    if blocked:
        return blocked
    if not (obligation or "").strip():
        return {
            "error": (
                "obligation is required: assert the negation of the claim and "
                "end with (check-sat), so that unsat means proved."
            )
        }
    if "(check-sat)" not in obligation:
        return {
            "error": "obligation must end with (check-sat) or nothing is asked of the solver"
        }

    with tempfile.TemporaryDirectory(prefix="axis_prove_") as workdir:
        Path(workdir, "model.axisl").write_text(source, encoding="utf-8")
        Path(workdir, "obligation.smt2").write_text(obligation, encoding="utf-8")
        script = (
            "axis emit-smt2 model.axisl > semantics.smt2 2>emit_err.txt || "
            "{ cat emit_err.txt >&2; exit 90; }; "
            "cat semantics.smt2 obligation.smt2 > query.smt2; "
            "z3 -T:%d query.smt2" % max(1, min(int(timeout_seconds) - 5, 600))
        )
        try:
            returncode, stdout, stderr = await agent_sandbox_runtime.run_in_sandbox(
                script, workdir, image=image, timeout_seconds=timeout_seconds
            )
        except TimeoutError:
            return {"error": f"proof attempt timed out after {timeout_seconds}s"}
        except FileNotFoundError:
            return {"error": "Docker is not available to this process"}
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"axis proof failed: {exc}")
            return {"error": f"proof attempt failed: {exc}"}

    if returncode == 90:
        return {
            "success": False,
            "error": f"AXIS could not emit semantics: {_first_line(stderr)[:400]}",
            "stderr": stderr[:MAX_OUTPUT_CHARS],
        }

    verdict = parse_solver_verdict(stdout)
    proved = verdict == "unsat"
    if verdict == "error":
        return {
            "success": False,
            "error": (
                "The solver returned no verdict; the obligation probably does "
                "not typecheck against the emitted semantics."
            ),
            "stdout": stdout[:MAX_OUTPUT_CHARS],
            "stderr": stderr[:MAX_OUTPUT_CHARS],
        }

    return {
        "success": True,
        "data": {
            "verdict": verdict,
            "proved": proved,
            "counterexample": stdout[:MAX_OUTPUT_CHARS] if verdict == "sat" else "",
            "note": (
                "unsat: no counterexample exists, the claim holds for all inputs. "
                "sat: the claim is false and the model above is a counterexample. "
                "unknown: the solver gave up; the claim is neither proved nor "
                "disproved."
            ),
        },
        "findings": [
            {
                "type": "axis_equivalence_proof",
                "title": (
                    "Equivalence proved for all inputs"
                    if proved
                    else f"Equivalence not proved (solver said {verdict})"
                ),
                "verdict": verdict,
                "proved": proved,
            }
        ],
    }
