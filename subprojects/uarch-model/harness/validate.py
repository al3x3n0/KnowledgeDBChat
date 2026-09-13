"""Test the latency fit on a mixed sequence it was never fitted to.

The fit sets one opLat per instruction from a benchmark containing only that
instruction. This asks the question that actually matters: does getting the
individual latencies right predict a sequence that mixes them? A dependent
chain through six different instructions has one answer per model, and no
single op class dominates it.

The kernel is inline assembly so that the natively compiled binary and the
simulated one contain the same instructions. A plain C version of this check
had clang vectorising and gcc not, and the resulting 20x disagreement said
nothing about any model.
"""

from __future__ import annotations

import asyncio
import json
import sys

sys.path.insert(0, "/app/data/uarch")

from chains import (  # noqa: E402
    INSTRUCTIONS,
    MIXED_SEQUENCE,
    build_mixed_source,
    build_source,
    frequency_ghz,
)

from app.services import agent_compiler_sandbox as native  # noqa: E402
from app.services import agent_gem5_sandbox as gem5  # noqa: E402

sys.path.insert(0, "/app/data/uarch")
from tune import build_overrides  # noqa: E402

UNROLL = 8
OPS_PER_ITERATION = UNROLL * len(MIXED_SEQUENCE)
SMALL, LARGE = 2_000, 6_000
NATIVE_ITERATIONS = 200_000


async def simulate(iterations: int, overrides: list, tag: str):
    result = await gem5.simulate_c_workload(
        code=build_mixed_source(iterations=iterations, unroll=UNROLL, timed=False),
        cpu_type="O3CPU",
        param_overrides=overrides,
        label=tag,
        timeout_seconds=1500,
    )
    if not result.get("success"):
        return None, str(result.get("error"))[:150]
    return result["data"].get("cycles"), None


async def predicted_cycles_per_op(overrides: list, tag: str):
    small, error = await simulate(SMALL, overrides, f"{tag}-small")
    if error:
        return None, error
    large, error = await simulate(LARGE, overrides, f"{tag}-large")
    if error:
        return None, error
    if large <= small:
        return None, "more iterations did not cost more cycles"
    return (large - small) / ((LARGE - SMALL) * OPS_PER_ITERATION), None


async def measured_cycles_per_op():
    anchor = next(i for i in INSTRUCTIONS if i.name == "add")
    result = await native.benchmark_c_snippet(
        code=build_source(anchor, ways=1), flags="-O1", repeat=9, label="anchor"
    )
    samples = (result.get("data", {}).get("reported_metrics") or {}).get("ns_per_op") or []
    if not samples:
        return None, None, "anchor produced no samples"
    ghz = frequency_ghz(min(samples))

    result = await native.benchmark_c_snippet(
        code=build_mixed_source(iterations=NATIVE_ITERATIONS, unroll=UNROLL),
        flags="-O1",
        repeat=9,
        label="mixed-m3",
    )
    if not result.get("success"):
        return None, ghz, str(result.get("error"))[:150]
    samples = (result.get("data", {}).get("reported_metrics") or {}).get("ns_per_op") or []
    if not samples:
        return None, ghz, "kernel printed no ns_per_op"
    return min(samples) * ghz, ghz, None


async def main() -> None:
    truth = json.load(open("/app/data/uarch/m3-truth.json"))["per_instruction"]
    overrides, notes = build_overrides(truth)

    measured, ghz, error = await measured_cycles_per_op()
    if error:
        print(f"held-out kernel could not be measured: {error}")
        return
    # The fit's own prediction, from summing the measured latencies. If the
    # chain is purely dependent this is what any latency model must produce,
    # and comparing it to the measurement checks the kernel before either
    # simulator is blamed for missing it.
    per_iteration = sum(
        truth[name]["latency_cycles"]
        for name in ("fmul_s", "fadd_s", "fsqrt_s", "fmul_s", "fadd_s", "fmadd_s")
    )
    from_latencies = per_iteration / len(MIXED_SEQUENCE)

    print(f"held-out mixed chain: {len(MIXED_SEQUENCE)} instructions, dependent")
    print(f"  measured on the M3        {measured:6.2f} cycles/op  (at {ghz:.2f} GHz)")
    print(f"  sum of measured latencies {from_latencies:6.2f} cycles/op")

    stock, error = await predicted_cycles_per_op([], "stock")
    if error:
        print(f"  stock model failed: {error}")
        return
    tuned, error = await predicted_cycles_per_op(overrides, "tuned")
    if error:
        print(f"  tuned model failed: {error}")
        return

    stock_error = (stock - measured) / measured
    tuned_error = (tuned - measured) / measured
    print(f"  stock O3CPU               {stock:6.2f} cycles/op  error {stock_error*100:+.0f}%")
    print(f"  tuned O3CPU               {tuned:6.2f} cycles/op  error {tuned_error*100:+.0f}%")
    improved = abs(tuned_error) < abs(stock_error)
    print(
        "\n"
        + (
            "the fit generalises: tuning improved a sequence it never saw"
            if improved
            else "the fit does not generalise: tuning did not improve the held-out sequence"
        )
    )
    json.dump(
        {
            "kernel": "dependent mixed chain (fmul, fadd, fsqrt, fmul, fadd, fmadd)",
            "overrides": overrides,
            "tuned_from": notes,
            "m3_cycles_per_op": round(measured, 3),
            "m3_frequency_ghz": round(ghz, 3),
            "sum_of_measured_latencies": round(from_latencies, 3),
            "stock_cycles_per_op": round(stock, 3),
            "tuned_cycles_per_op": round(tuned, 3),
            "stock_relative_error": round(stock_error, 3),
            "tuned_relative_error": round(tuned_error, 3),
            "fit_generalises": improved,
        },
        open("/app/data/uarch/validation.json", "w"),
        indent=2,
        sort_keys=True,
    )
    print("written to data/uarch/validation.json")


asyncio.run(main())
