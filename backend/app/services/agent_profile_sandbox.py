"""Build a workload, run it, and report which instructions actually executed.

Static occurrence answers "what does the source say"; this answers "what does
the machine do", which is the question instruction-set extension work is
actually asking. A sequence appearing ten thousand times in cold code is worth
less than one appearing twice in an inner loop, and only a dynamic profile can
tell those apart.

Callgrind counts every instruction exactly and needs no performance counters,
which is what makes this possible in a sandbox that drops the privileges a PMU
would require. The cost is speed: instrumented execution runs roughly fifty
times slower, so the workload passed here should be a kernel with a bounded
input, not an application.
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from app.services import agent_sandbox_runtime, callgrind_profile

DEFAULT_IMAGE = "ghcr.io/al3x3n0/kdbc-profiling-research:latest"
DEFAULT_TIMEOUT_SECONDS = 600
DEFAULT_FLAGS = "-O3 -g"
MAX_CODE_CHARS = 40_000
MAX_OUTPUT_CHARS = 8_000
MAX_BLOCK_INSTRUCTIONS = 40

# Arguments are interpolated into a shell command, so restrict them to what a
# benchmark's own arguments look like rather than quoting arbitrary text.
SAFE_RUN_ARGS = re.compile(r"^[-A-Za-z0-9_=+.,/ ]*$")
SAFE_FLAGS = re.compile(r"^[-A-Za-z0-9_=+., /]*$")


def _preflight(code: str, image: str) -> Optional[Dict[str, Any]]:
    if not (code or "").strip():
        return {"error": "code is required"}
    if len(code) > MAX_CODE_CHARS:
        return {"error": f"code exceeds {MAX_CODE_CHARS} characters"}
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


def summarize_functions(
    profile: callgrind_profile.Profile, limit: int
) -> List[Dict[str, Any]]:
    """Rank functions by instructions executed, with their share of the run."""
    total = profile.total or 1
    return [
        {
            "function": name,
            "instructions": cost,
            "share": round(cost / total, 4),
        }
        for name, cost in profile.hottest_functions(limit)
    ]


def summarize_blocks(
    profile: callgrind_profile.Profile,
    listing: Dict[int, str],
    limit: int,
) -> List[Dict[str, Any]]:
    """Describe the hottest straight-line runs, with their instructions."""
    blocks = callgrind_profile.hot_blocks(profile, listing, limit=limit)
    total = profile.total or 1
    described = []
    for block in blocks:
        rows = block["listing"][:MAX_BLOCK_INSTRUCTIONS]
        described.append(
            {
                "start": f"0x{block['start']:x}",
                "instructions": block["instructions"],
                "executions": block["executions"],
                "instruction_cost": block["instruction_cost"],
                "share": round(int(block["instruction_cost"]) / total, 4),
                # The instruction text is the point: this is what a candidate
                # fused operation would be built from.
                "disassembly": [row["text"] for row in rows],
                "disassembly_truncated": len(block["listing"]) > len(rows),
            }
        )
    return described


async def profile_c_workload(
    *,
    code: str,
    flags: str = DEFAULT_FLAGS,
    run_args: str = "",
    label: str = "",
    top_functions: int = 8,
    top_blocks: int = 5,
    image: str = DEFAULT_IMAGE,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    """Compile a self-contained C program, run it, and profile what executed."""
    blocked = _preflight(code, image)
    if blocked:
        return blocked

    safe_flags = (flags or DEFAULT_FLAGS).strip()
    if not SAFE_FLAGS.match(safe_flags):
        return {"error": f"flags contain unsupported characters: {flags!r}"}
    if "-g" not in safe_flags.split():
        # Without debug info the profile still counts instructions, but nothing
        # can be attributed to a function, which is most of its value.
        safe_flags = f"{safe_flags} -g"
    arguments = (run_args or "").strip()
    if not SAFE_RUN_ARGS.match(arguments):
        return {"error": f"run_args contain unsupported characters: {run_args!r}"}

    with tempfile.TemporaryDirectory(prefix="profile_workload_") as workdir:
        Path(workdir, "workload.c").write_text(code, encoding="utf-8")
        script = (
            f"clang {safe_flags} -o workload workload.c -lm 2>compile_err.txt || "
            "{ cat compile_err.txt >&2; exit 90; }; "
            "valgrind --tool=callgrind --dump-instr=yes "
            "--callgrind-out-file=cg.out ./workload "
            f"{arguments} >run_out.txt 2>run_err.txt || "
            "{ tail -20 run_err.txt >&2; exit 91; }; "
            "objdump -d workload > workload.dis; "
            "cat run_out.txt"
        )
        try:
            returncode, stdout, stderr = await agent_sandbox_runtime.run_in_sandbox(
                script, workdir, image=image, timeout_seconds=timeout_seconds
            )
        except TimeoutError:
            return {
                "error": (
                    f"Profiling timed out after {timeout_seconds}s. Instrumented "
                    "execution is ~50x slower than native; reduce the workload's "
                    "input size."
                )
            }
        except FileNotFoundError:
            return {"error": "Docker is not available to this process"}
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"profile_c_workload failed: {exc}")
            return {"error": f"Profiling failed: {exc}"}

        if returncode == 90:
            from app.services.agent_compiler_sandbox import explain_compiler_failure

            return {
                "success": False,
                "error": explain_compiler_failure(stderr),
                "compiler_stderr": stderr[:MAX_OUTPUT_CHARS],
            }
        if returncode == 91:
            return {
                "success": False,
                "error": "The workload ran but exited non-zero under callgrind",
                "stderr": stderr[:MAX_OUTPUT_CHARS],
            }
        if returncode != 0:
            return {
                "success": False,
                "error": f"Profiling failed with exit code {returncode}",
                "stderr": stderr[:MAX_OUTPUT_CHARS],
            }

        profile_path = Path(workdir, "cg.out")
        listing_path = Path(workdir, "workload.dis")
        if not profile_path.exists():
            return {
                "success": False,
                "error": "callgrind produced no profile",
                "stderr": stderr[:MAX_OUTPUT_CHARS],
            }
        with profile_path.open() as handle:
            profile = callgrind_profile.parse(handle)
        listing = (
            callgrind_profile.parse_disassembly(listing_path.read_text())
            if listing_path.exists()
            else {}
        )

    functions = summarize_functions(profile, top_functions)
    blocks = summarize_blocks(profile, listing, top_blocks)
    hottest = functions[0] if functions else {}
    subject = (label or "").strip() or "workload"

    return {
        "success": True,
        "data": {
            "image": image,
            "subject": subject,
            "flags": safe_flags,
            "instructions_executed": profile.total,
            "functions": functions,
            "hot_blocks": blocks,
            "program_output": stdout[:MAX_OUTPUT_CHARS],
            "note": (
                "Instruction counts are exact and dynamic: this is what "
                "executed, not what appears in the source. Timing is not "
                "measured here -- callgrind instruments rather than times."
            ),
        },
        "findings": [
            {
                "type": "dynamic_profile",
                "subject": subject,
                "title": (
                    f"{subject}: {profile.total} instructions executed, "
                    f"hottest {hottest.get('function', 'unknown')} at "
                    f"{round(float(hottest.get('share', 0)) * 100, 1)}%"
                ),
                "instructions_executed": profile.total,
                "hottest_function": hottest.get("function"),
                "hottest_share": hottest.get("share"),
                "hot_block_executions": blocks[0]["executions"] if blocks else None,
            }
        ],
    }
