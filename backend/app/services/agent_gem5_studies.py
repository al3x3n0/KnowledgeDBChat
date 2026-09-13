"""Studies that take more than one simulation to answer.

`simulate_mechanism` answers "is this mechanism worth anything here". Three
questions come before and after it, and none of them fit a single pair:

- **What is limiting this kernel at all?** `explain_bottleneck` reads it off
  the statistics a run already writes. In a live job the agent had a working
  way to measure a mechanism and no way to choose one, and spent two of nine
  iterations on document search before trying a prefetcher.
- **How much is there to win?** `measure_headroom` makes one structure
  effectively infinite and reports the cycles that recovers. That bounds any
  mechanism aimed at it, before one is designed. Measured on a strided scan:
  widening the issue queue recovered 11.8%, and widening the reorder buffer
  recovered nothing -- the reorder buffer was never binding, and no amount of
  cleverness aimed at it could have paid.
- **Does it keep winning?** `sweep_mechanism` turns two points into a curve, so
  saturation is visible instead of assumed, and `evaluate_across_kernels`
  reports a mechanism's distribution rather than its best case. One workload is
  how an evaluation gets overturned.

All of them share `run_configs`: one compile, one container, one binary behind
every number being compared.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Sequence

from app.services import gem5_bottleneck
from app.services.agent_gem5_mechanism import (
    DEFAULT_FLAGS,
    DEFAULT_IMAGE,
    SandboxRunFailed,
    run_configs,
    stats_identical,
)

logger = logging.getLogger(__name__)

MAX_SWEEP_POINTS = 8
MAX_KERNELS = 6


# ---------------------------------------------------------------------------
# Idealisations.
#
# Each makes one structure large enough to stop being the limit. They are not
# proposals -- a 512-entry issue queue is not buildable at this frequency --
# and the number one produces is a ceiling on what any mechanism aimed at that
# structure could recover, never a speedup anyone can have.
# ---------------------------------------------------------------------------
IDEALISATIONS: Dict[str, Dict[str, Any]] = {
    "issue_queue": {
        "config": {"cpu_params": {"instQueues[*].numEntries": 512}},
        "means": "every issue queue in the core widened to 512 entries",
    },
    "reorder_buffer": {
        "config": {"cpu_params": {"numROBEntries": 1024}},
        "means": "the reorder buffer widened to 1024 entries",
    },
    "load_queue": {
        "config": {"cpu_params": {"LQEntries": 128}},
        "means": "the load queue widened to 128 entries",
    },
    "store_queue": {
        "config": {"cpu_params": {"SQEntries": 128}},
        "means": "the store queue widened to 128 entries",
    },
    "physical_registers": {
        "config": {"cpu_params": {"numPhysIntRegs": 1024, "numPhysFloatRegs": 1024}},
        "means": "1024 physical registers of each class",
    },
    "pipeline_width": {
        "config": {
            "cpu_params": {
                "fetchWidth": 16,
                "dispatchWidth": 16,
                "issueWidth": 16,
                "commitWidth": 16,
            }
        },
        "means": "every pipeline stage widened to 16",
    },
    "l1d_capacity": {
        "config": {"caches": {"l1d": {"size": "16MiB", "assoc": 16}}},
        "means": "an L1 data cache large enough to hold the working set",
    },
    "l1i_capacity": {
        "config": {"caches": {"l1i": {"size": "16MiB", "assoc": 16}}},
        "means": "an L1 instruction cache large enough to hold the footprint",
    },
    "l2_capacity": {
        "config": {"caches": {"l2": {"size": "256MiB", "assoc": 16}}},
        "means": "an L2 large enough that nothing reaches memory twice",
    },
    # Not an idealisation and labelled as one nowhere: gem5 has no perfect
    # branch predictor, so this substitutes the largest one it ships. The gap
    # it shows is a LOWER bound on what prediction is costing, which is the
    # opposite direction from every other entry here -- hence `bounds`.
    "branch_prediction": {
        "config": {"branch_pred": {"conditional": "TAGE_SC_L_64KB"}},
        "means": "the largest predictor this build ships (TAGE_SC_L_64KB)",
        "bounds": "lower",
    },
}


#: What `explain_bottleneck` calls a structure, mapped to what
#: `measure_headroom` calls it. The attribution prints
#: `backpressure.dominant = "IQ"` beside `headroom_target = "issue_queue"`, and
#: a live run passed the first of those. Two names for one structure in one
#: result is the tool's fault, not the caller's.
TARGET_ALIASES = {
    "IQ": "issue_queue",
    "ROB": "reorder_buffer",
    "LQ": "load_queue",
    "SQ": "store_queue",
}


def resolve_target(name: str) -> str:
    """The idealisation a caller means, however the result spelled it."""
    cleaned = str(name or "").strip()
    if cleaned in IDEALISATIONS:
        return cleaned
    if cleaned in TARGET_ALIASES:
        return TARGET_ALIASES[cleaned]
    lowered = cleaned.lower()
    for known in IDEALISATIONS:
        if lowered == known.lower():
            return known
    for label, known in TARGET_ALIASES.items():
        if lowered == label.lower():
            return known
    return cleaned


def _merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    """Overlay one configuration onto another, recursing into nested dicts."""
    merged = dict(base or {})
    for key, value in (overlay or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _set_path(config: Dict[str, Any], path: str, value: Any) -> Dict[str, Any]:
    """A copy of `config` with one dotted path set.

    Intermediate levels are created as needed, so a sweep over
    `caches.l2.prefetcher.params.degree` works against a configuration that
    named the prefetcher as a bare string -- which is how the tool documents
    it, and therefore how it will usually arrive.
    """
    import copy

    out = copy.deepcopy(config or {})
    parts = path.split(".")
    node: Any = out
    for i, part in enumerate(parts[:-1]):
        nxt = node.get(part)
        if isinstance(nxt, str):
            # "prefetcher": "StridePrefetcher" becoming a parameterised form.
            nxt = {"class": nxt}
        if not isinstance(nxt, dict):
            nxt = {}
        node[part] = nxt
        node = nxt
    node[parts[-1]] = value
    return out


def _cycles(run: Dict[str, Any]) -> float:
    return float(run["stats"].get("system.cpu.numCycles") or 0.0)


def _speedup(baseline: float, variant: float) -> Optional[float]:
    return (baseline / variant) if baseline > 0 and variant > 0 else None


def _geomean(values: Sequence[float]) -> Optional[float]:
    usable = [v for v in values if v and v > 0]
    if not usable:
        return None
    product = 1.0
    for value in usable:
        product *= value
    return product ** (1.0 / len(usable))


# ---------------------------------------------------------------------------
# What is limiting this kernel.
# ---------------------------------------------------------------------------
async def explain_bottleneck(
    *,
    code: str,
    config: Optional[Dict[str, Any]] = None,
    flags: str = DEFAULT_FLAGS,
    run_args: str = "",
    label: str = "",
    image: str = DEFAULT_IMAGE,
    timeout_seconds: Optional[int] = None,
) -> Dict[str, Any]:
    """Run the kernel once and say what the cycles were spent waiting on."""
    if not (code or "").strip():
        return {"success": False, "error": "code is required"}
    try:
        runs = await run_configs(
            code=code,
            configs={"run": config or {}},
            flags=flags,
            run_args=run_args,
            image=image,
            timeout_seconds=timeout_seconds,
        )
    except SandboxRunFailed as failure:
        return failure.detail

    stats = runs["run"]["stats"]
    attribution = gem5_bottleneck.attribute(stats)
    subject = (label or "").strip() or "kernel"
    top = (attribution["signals"] or [{}])[0]
    return {
        "success": True,
        "label": subject,
        "configuration": runs["run"]["manifest"],
        **attribution,
        "findings": [
            {
                "type": "bottleneck_attribution",
                "subject": subject,
                "title": (
                    f"{subject}: {attribution['cycles']:.0f} cycles at IPC "
                    f"{attribution['ipc']}, strongest signal "
                    f"{top.get('signal', 'none')}"
                ),
                "cycles": attribution["cycles"],
                "ipc": attribution["ipc"],
                "top_signal": top.get("signal"),
                "top_signal_evidence": top.get("evidence"),
                "headroom_target": top.get("headroom_target"),
                "measurement_source": "gem5 stats attribution",
            }
        ],
    }


# ---------------------------------------------------------------------------
# How much is there to win.
# ---------------------------------------------------------------------------
async def measure_headroom(
    *,
    code: str,
    targets: Sequence[str],
    config: Optional[Dict[str, Any]] = None,
    flags: str = DEFAULT_FLAGS,
    run_args: str = "",
    label: str = "",
    image: str = DEFAULT_IMAGE,
    timeout_seconds: Optional[int] = None,
) -> Dict[str, Any]:
    """Idealise each named structure in turn and report what that recovers."""
    if not (code or "").strip():
        return {"success": False, "error": "code is required"}
    wanted = [resolve_target(t) for t in (targets or []) if str(t).strip()]
    if not wanted:
        return {
            "success": False,
            "error": (
                "targets is required. Available: "
                + ", ".join(sorted(IDEALISATIONS))
                + ". explain_bottleneck names the one worth trying first."
            ),
        }
    unknown = [t for t in wanted if t not in IDEALISATIONS]
    if unknown:
        return {
            "success": False,
            "error": (
                f"Unknown target(s) {', '.join(unknown)}. Available: "
                + ", ".join(sorted(IDEALISATIONS))
            ),
        }
    if len(wanted) > MAX_SWEEP_POINTS:
        return {
            "success": False,
            "error": (
                f"{len(wanted)} targets is more than {MAX_SWEEP_POINTS}; each "
                "is a full simulation and they run one after another."
            ),
        }

    base = config or {}
    configs = {"baseline": base}
    for target in wanted:
        configs[f"ideal_{target}"] = _merge(base, IDEALISATIONS[target]["config"])

    try:
        runs = await run_configs(
            code=code,
            configs=configs,
            flags=flags,
            run_args=run_args,
            image=image,
            timeout_seconds=timeout_seconds,
        )
    except SandboxRunFailed as failure:
        return failure.detail

    baseline_cycles = _cycles(runs["baseline"])
    results = []
    for target in wanted:
        run = runs[f"ideal_{target}"]
        cycles = _cycles(run)
        ideal = IDEALISATIONS[target]
        results.append(
            {
                "target": target,
                "means": ideal["means"],
                "cycles": cycles,
                "cycles_recovered": baseline_cycles - cycles,
                "headroom": _speedup(baseline_cycles, cycles),
                "headroom_percent": (
                    100.0 * (baseline_cycles - cycles) / baseline_cycles
                    if baseline_cycles
                    else None
                ),
                "bounds": ideal.get("bounds", "upper"),
                # Where the limit went once this one stopped binding. The most
                # useful line in the whole study: on a strided scan, widening
                # the issue queue moved the limit straight to the store queue.
                "next_limit": (
                    gem5_bottleneck.backpressure(run["stats"]).get("dominant")
                ),
            }
        )
    results.sort(key=lambda r: r["cycles_recovered"], reverse=True)

    subject = (label or "").strip() or "kernel"
    best = results[0]
    return {
        "success": True,
        "label": subject,
        "baseline_cycles": baseline_cycles,
        "baseline_limit": gem5_bottleneck.backpressure(runs["baseline"]["stats"]).get(
            "dominant"
        ),
        "results": results,
        "interpretation": (
            "Each number is what the kernel would gain if that structure "
            "stopped being a limit entirely. An idealised structure is not "
            "buildable, so these are ceilings on what any mechanism aimed "
            "there could be worth -- a target with near-zero headroom cannot "
            "pay however it is implemented, which is worth knowing before "
            "designing one. `next_limit` names what binds once it is removed."
        ),
        "findings": [
            {
                "type": "headroom_bound",
                "subject": f"{subject}: {r['target']}",
                "title": (
                    f"{subject}: idealising {r['target']} recovers "
                    f"{r['headroom_percent']:.1f}% "
                    f"({r['cycles_recovered']:.0f} cycles), next limit "
                    f"{r['next_limit'] or 'none'}"
                ),
                "target": r["target"],
                "headroom": r["headroom"],
                "headroom_percent": r["headroom_percent"],
                "baseline_cycles": baseline_cycles,
                "bounds": r["bounds"],
                "next_limit": r["next_limit"],
                "measurement_source": "gem5 idealised limit study",
            }
            for r in results
        ],
        "next_study": (
            f"{best['target']} has the most to give ("
            f"{best['headroom_percent']:.1f}%). A real mechanism aimed at it "
            "can be measured with simulate_mechanism and compared against "
            "this ceiling."
        ),
    }


# ---------------------------------------------------------------------------
# Does it keep winning as you turn it up.
# ---------------------------------------------------------------------------
async def sweep_mechanism(
    *,
    code: str,
    variant: Dict[str, Any],
    vary: str,
    values: Sequence[Any],
    baseline: Optional[Dict[str, Any]] = None,
    flags: str = DEFAULT_FLAGS,
    run_args: str = "",
    label: str = "",
    image: str = DEFAULT_IMAGE,
    timeout_seconds: Optional[int] = None,
) -> Dict[str, Any]:
    """Run one configuration at several settings and return the curve."""
    if not (code or "").strip():
        return {"success": False, "error": "code is required"}
    if not isinstance(variant, dict) or not variant:
        return {"success": False, "error": "variant must be a configuration object"}
    if not (vary or "").strip():
        return {
            "success": False,
            "error": (
                "vary is required: the dotted path to sweep, e.g. "
                "'caches.l2.prefetcher.params.degree' or 'caches.l2.size'."
            ),
        }
    points = list(values or [])
    if len(points) < 2:
        return {
            "success": False,
            "error": (
                "A sweep needs at least two values; one point is a comparison, "
                "and simulate_mechanism already does that."
            ),
        }
    if len(points) > MAX_SWEEP_POINTS:
        return {
            "success": False,
            "error": (
                f"{len(points)} values is more than {MAX_SWEEP_POINTS}; each "
                "is a full simulation and they run one after another."
            ),
        }

    configs: Dict[str, Dict[str, Any]] = {"baseline": baseline if baseline else {}}
    names = []
    for index, value in enumerate(points):
        name = f"point{index}"
        names.append(name)
        configs[name] = _set_path(variant, vary, value)

    try:
        runs = await run_configs(
            code=code,
            configs=configs,
            flags=flags,
            run_args=run_args,
            image=image,
            timeout_seconds=timeout_seconds,
        )
    except SandboxRunFailed as failure:
        return failure.detail

    baseline_cycles = _cycles(runs["baseline"])
    curve = []
    for name, value in zip(names, points):
        cycles = _cycles(runs[name])
        curve.append(
            {
                "value": value,
                "cycles": cycles,
                "speedup": _speedup(baseline_cycles, cycles),
            }
        )

    best = max(curve, key=lambda p: p["speedup"] or 0.0)
    saturation = _saturation_point(curve)
    monotonic = _is_monotonic(curve)
    subject = (label or "").strip() or vary
    return {
        "success": True,
        "label": subject,
        "varied": vary,
        "baseline_cycles": baseline_cycles,
        "curve": curve,
        "best": best,
        "saturates_at": saturation,
        "monotonic": monotonic,
        "interpretation": (
            "A curve says what two points cannot: where the setting stops "
            "paying, and whether it ever turns around. A mechanism reported "
            "at its best point alone is a best case, not a result."
        ),
        "findings": [
            {
                "type": "mechanism_sweep",
                "subject": subject,
                "title": (
                    f"{subject}: best {best['speedup']:.4f}x at {best['value']}"
                    # "Saturating" is the wrong word for a curve that turns
                    # around, and a measured one did: degree 1..8 climbed to
                    # 1.90x and degree 16 fell back to 1.67x. Calling that
                    # saturation would tell a reader the setting stopped
                    # helping, when it started hurting.
                    + (
                        f", {'peaking' if not monotonic else 'saturating'} "
                        f"at {saturation}"
                        if saturation is not None
                        else ""
                    )
                    + ("" if monotonic else "; the curve turns around")
                ),
                "varied": vary,
                "curve": curve,
                "best_value": best["value"],
                "best_speedup": best["speedup"],
                "saturates_at": saturation,
                "monotonic": monotonic,
                "measurement_source": "gem5 parameter sweep",
            }
        ],
    }


def _is_monotonic(curve: Sequence[Dict[str, Any]]) -> bool:
    speedups = [p["speedup"] for p in curve if p["speedup"]]
    return all(b >= a for a, b in zip(speedups, speedups[1:]))


def _saturation_point(
    curve: Sequence[Dict[str, Any]], epsilon: float = 0.01
) -> Optional[Any]:
    """The first setting past which nothing more is gained.

    Reported rather than inferred by the reader, because the interesting case
    is the one where the curve is flat long before the largest value tried --
    paying area for a setting that stopped helping is the mistake this catches.
    """
    best = max((p["speedup"] or 0.0) for p in curve) if curve else 0.0
    if best <= 0:
        return None
    for point in curve:
        if (point["speedup"] or 0.0) >= best * (1.0 - epsilon):
            return point["value"]
    return None


# ---------------------------------------------------------------------------
# Does it keep winning on other work.
# ---------------------------------------------------------------------------
async def evaluate_across_kernels(
    *,
    kernels: Sequence[Dict[str, str]],
    variant: Dict[str, Any],
    baseline: Optional[Dict[str, Any]] = None,
    flags: str = DEFAULT_FLAGS,
    label: str = "",
    image: str = DEFAULT_IMAGE,
    timeout_seconds: Optional[int] = None,
) -> Dict[str, Any]:
    """Measure one mechanism on several kernels and report the distribution."""
    from app.services.agent_gem5_mechanism import MECHANISM_KEYS, find_confounds

    listed = [k for k in (kernels or []) if isinstance(k, dict) and k.get("code")]
    if len(listed) < 2:
        return {
            "success": False,
            "error": (
                "At least two kernels are needed; one kernel is what "
                "simulate_mechanism already measures, and a single-workload "
                "result is the one that gets overturned."
            ),
        }
    if len(listed) > MAX_KERNELS:
        return {
            "success": False,
            "error": f"{len(listed)} kernels is more than {MAX_KERNELS}.",
        }

    if baseline is None:
        import copy

        baseline = copy.deepcopy(variant)
        for level_spec in (baseline.get("caches") or {}).values():
            if isinstance(level_spec, dict):
                for key in MECHANISM_KEYS:
                    level_spec.pop(key, None)
        baseline.pop("branch_pred", None)

    confounds = find_confounds(baseline, variant)
    if confounds:
        return {
            "success": False,
            "error": (
                "The two arms differ in more than the mechanism: "
                + "; ".join(confounds)
            ),
            "confounds": confounds,
        }

    per_kernel = []
    for index, kernel in enumerate(listed):
        name = str(kernel.get("name") or f"kernel{index}")
        try:
            runs = await run_configs(
                code=str(kernel["code"]),
                configs={"baseline": baseline, "variant": variant},
                flags=flags,
                run_args=str(kernel.get("run_args") or ""),
                image=image,
                timeout_seconds=timeout_seconds,
            )
        except SandboxRunFailed as failure:
            per_kernel.append(
                {"kernel": name, "error": failure.detail.get("error"), "speedup": None}
            )
            continue
        base_cycles = _cycles(runs["baseline"])
        var_cycles = _cycles(runs["variant"])
        per_kernel.append(
            {
                "kernel": name,
                "baseline_cycles": base_cycles,
                "variant_cycles": var_cycles,
                "speedup": _speedup(base_cycles, var_cycles),
                "identical_stats": stats_identical(
                    runs["baseline"]["stats"], runs["variant"]["stats"]
                ),
            }
        )

    measured = [k for k in per_kernel if k.get("speedup")]
    if not measured:
        return {
            "success": False,
            "error": "No kernel produced a usable comparison.",
            "per_kernel": per_kernel,
        }

    speedups = [k["speedup"] for k in measured]
    worst = min(measured, key=lambda k: k["speedup"])
    best = max(measured, key=lambda k: k["speedup"])
    regressions = [k["kernel"] for k in measured if k["speedup"] < 1.0]
    subject = (label or "").strip() or "mechanism"
    return {
        "success": True,
        "label": subject,
        "per_kernel": per_kernel,
        "geomean_speedup": _geomean(speedups),
        "worst": worst,
        "best": best,
        "regressions": regressions,
        "kernels_measured": len(measured),
        "interpretation": (
            "The geometric mean is the summary; the worst case is the one that "
            "decides whether a mechanism ships. A mechanism that helps every "
            "kernel but one is a different proposal from one that helps them "
            "all, and a single-kernel result cannot tell them apart."
        ),
        "findings": [
            {
                "type": "mechanism_evaluation",
                "subject": subject,
                "title": (
                    f"{subject}: geomean {_geomean(speedups):.4f}x over "
                    f"{len(measured)} kernels, worst {worst['speedup']:.4f}x "
                    f"on {worst['kernel']}"
                    + (
                        f", regressed on {', '.join(regressions)}"
                        if regressions
                        else ""
                    )
                ),
                "geomean_speedup": _geomean(speedups),
                "worst_speedup": worst["speedup"],
                "worst_kernel": worst["kernel"],
                "best_speedup": best["speedup"],
                "regressions": regressions,
                "kernels_measured": len(measured),
                "measurement_source": "gem5 multi-kernel evaluation",
            }
        ],
    }
