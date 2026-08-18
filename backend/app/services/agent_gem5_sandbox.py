"""Measure a workload in a simulated core, as the referee for a prediction.

llvm-mca estimates how a sequence issues on a scheduling model with a warm
front end and no cache misses. gem5 executes it in a modelled core with caches,
a branch predictor and a memory hierarchy, and reports what that took. The
difference between those two numbers is the thing worth knowing: it is how a
prediction gets scored rather than merely defended.

Simulation is slow -- an out-of-order core runs on the order of 100k
instructions a second -- so what belongs here is a kernel with a bounded input,
after llvm-mca has already thrown out the candidates that were never going to
pay. The tool says that rather than letting it be discovered as a timeout.
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger

from app.services import agent_sandbox_runtime, gem5_stats

DEFAULT_IMAGE = "ghcr.io/al3x3n0/kdbc-gem5-research:latest"
GEM5_BINARY = "/opt/gem5/build/ARM/gem5.opt"
GEM5_SE_CONFIG = "/opt/gem5/configs/deprecated/example/se.py"
DEFAULT_TIMEOUT_SECONDS = 1800
DEFAULT_FLAGS = "-O3 -static"
DEFAULT_CPU = "O3CPU"
MAX_CODE_CHARS = 40_000
MAX_OUTPUT_CHARS = 8_000

# Simulated core models. Restricted because the value is interpolated into a
# shell command, and named so the catalog documents the real choice: AtomicCPU
# counts instructions quickly and tells you nothing about timing, O3CPU models
# an out-of-order pipeline and is the one a performance claim needs.
CPU_TYPES = {
    "O3CPU": "an out-of-order pipeline; use this for any timing claim",
    "MinorCPU": "an in-order pipeline",
    "TimingSimpleCPU": "simple timing, memory latencies but no pipeline",
    "AtomicSimpleCPU": "fastest, no timing model at all",
}

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


async def simulate_c_workload(
    *,
    code: str,
    flags: str = DEFAULT_FLAGS,
    cpu_type: str = DEFAULT_CPU,
    run_args: str = "",
    label: str = "",
    image: str = DEFAULT_IMAGE,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    """Compile a self-contained C program and run it in a simulated core."""
    blocked = _preflight(code, image)
    if blocked:
        return blocked

    model = str(cpu_type or DEFAULT_CPU).strip()
    if model not in CPU_TYPES:
        return {
            "error": (
                f"Unknown cpu_type {cpu_type!r}. Available: "
                + ", ".join(f"{name} ({why})" for name, why in CPU_TYPES.items())
            )
        }
    safe_flags = (flags or DEFAULT_FLAGS).strip()
    if not SAFE_FLAGS.match(safe_flags):
        return {"error": f"flags contain unsupported characters: {flags!r}"}
    if "-static" not in safe_flags.split():
        # Syscall-emulation mode has no dynamic loader; a dynamically linked
        # binary fails inside the simulator with an error about the interpreter
        # that says nothing about how to fix it.
        safe_flags = f"{safe_flags} -static"
    arguments = (run_args or "").strip()
    if not SAFE_RUN_ARGS.match(arguments):
        return {"error": f"run_args contain unsupported characters: {run_args!r}"}

    with tempfile.TemporaryDirectory(prefix="gem5_workload_") as workdir:
        Path(workdir, "workload.c").write_text(code, encoding="utf-8")
        options = f" --options='{arguments}'" if arguments else ""
        script = (
            f"gcc {safe_flags} -o workload workload.c -lm 2>compile_err.txt || "
            "{ cat compile_err.txt >&2; exit 90; }; "
            f"{GEM5_BINARY} --outdir=m5out {GEM5_SE_CONFIG} "
            f"--cmd=./workload{options} --cpu-type={model} --caches --l2cache "
            "> gem5.log 2>&1 || { tail -25 gem5.log >&2; exit 91; }; "
            "tail -5 gem5.log"
        )
        try:
            returncode, stdout, stderr = await agent_sandbox_runtime.run_in_sandbox(
                script,
                workdir,
                image=image,
                timeout_seconds=timeout_seconds,
                memory="4096m",
                cpus="2",
            )
        except TimeoutError:
            return {
                "error": (
                    f"Simulation timed out after {timeout_seconds}s. An "
                    "out-of-order model runs on the order of 100k instructions "
                    "a second; shrink the workload's input."
                )
            }
        except FileNotFoundError:
            return {"error": "Docker is not available to this process"}
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"simulate_c_workload failed: {exc}")
            return {"error": f"Simulation failed: {exc}"}

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
                "error": "gem5 exited non-zero; see stderr for its last output",
                "stderr": stderr[:MAX_OUTPUT_CHARS],
            }
        if returncode != 0:
            return {
                "success": False,
                "error": f"Simulation failed with exit code {returncode}",
                "stderr": stderr[:MAX_OUTPUT_CHARS],
            }

        stats_path = Path(workdir, "m5out", "stats.txt")
        if not stats_path.exists():
            return {
                "success": False,
                "error": "gem5 produced no stats.txt",
                "stdout": stdout[:MAX_OUTPUT_CHARS],
            }
        with stats_path.open() as handle:
            stats = gem5_stats.parse(handle)

    summary = gem5_stats.summarize(stats)
    subject = (label or "").strip() or "workload"
    source = f"gem5 {model}"

    return {
        "success": True,
        "data": {
            "subject": subject,
            "cpu_type": model,
            "flags": safe_flags,
            **summary,
            "measurement_source": source,
            "note": (
                "Simulated, not run on hardware: these are the modelled core's "
                "cycles. Compare cycles rather than sim_seconds, which also "
                "depends on the clock the configuration assigned."
            ),
        },
        "findings": [
            {
                "type": "simulated_measurement",
                "subject": subject,
                "title": (
                    f"{subject} @ {model}: {summary['cycles']} cycles, "
                    f"{summary['instructions']} instructions, IPC {summary['ipc']}"
                ),
                "cpu_type": model,
                "cycles": summary["cycles"],
                "instructions": summary["instructions"],
                "ipc": summary["ipc"],
                "measurement_source": source,
            }
        ],
    }
