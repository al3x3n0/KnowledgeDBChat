"""Fit gem5's op latencies to the M3, then test the fit on a kernel it never saw.

Fitting is the easy half and proves almost nothing: setting each opLat to the
latency measured for that instruction will of course reproduce those
measurements. The question worth answering is whether per-instruction tuning
makes the model better at the thing a model is actually for -- predicting a
kernel nobody tuned against.

So the microbenchmarks are the training set and a vec3-normalize loop is the
held-out one. It mixes multiply, add, sqrt and divide with loads and stores,
and none of the load/store or issue-width parameters are touched by the fit,
so it can genuinely come out worse.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys

sys.path.insert(0, "/app/data/uarch")

from chains import INSTRUCTIONS, build_source, frequency_ghz  # noqa: E402

from app.services import agent_compiler_sandbox as native  # noqa: E402
from app.services import agent_gem5_sandbox as gem5  # noqa: E402

# opClass -> parameter path, read out of a config.ini gem5 wrote itself. The
# flattened names config.ini prints ("FUList03.opList4") are not addressable;
# the vectors need indices ("FUList[3].opList[4]"), which cost an hour to find.
FU_PATH = {
    "IntMult": "system.cpu[0].instQueues[0].fuPool.FUList[1].opList[0]",
    "FloatAdd": "system.cpu[0].instQueues[0].fuPool.FUList[2].opList[0]",
    "FloatMult": "system.cpu[0].instQueues[0].fuPool.FUList[3].opList[0]",
    "FloatMultAcc": "system.cpu[0].instQueues[0].fuPool.FUList[3].opList[1]",
    "FloatDiv": "system.cpu[0].instQueues[0].fuPool.FUList[3].opList[3]",
    "FloatSqrt": "system.cpu[0].instQueues[0].fuPool.FUList[3].opList[4]",
    "SimdFloatMultAcc": "system.cpu[0].instQueues[0].fuPool.FUList[5].opList[20]",
}

# Which measured instruction constrains which op class.
INSTRUCTION_OP_CLASS = {
    "mul": "IntMult",
    "fadd_s": "FloatAdd",
    "fmul_s": "FloatMult",
    "fmadd_s": "FloatMultAcc",
    "fdiv_s": "FloatDiv",
    "fsqrt_s": "FloatSqrt",
    "fmla_v": "SimdFloatMultAcc",
}

HELD_OUT = r"""
#include <stdio.h>
#define N 512
static float x[N], y[N], z[N];
int main(void) {
    for (int i = 0; i < N; i++) { x[i]=i*0.5f+1; y[i]=i*0.25f+1; z[i]=i*0.125f+1; }
    for (int r = 0; r < REPS; r++)
        for (int i = 0; i < N; i++) {
            float d = x[i]*x[i] + y[i]*y[i] + z[i]*z[i];
            float k = 1.0f / __builtin_sqrtf(d);
            x[i] *= k; y[i] *= k; z[i] *= k;
        }
    printf("out=%d\n", (int)x[3]);
    return 0;
}
"""

SMALL_REPS, LARGE_REPS = 20, 60
ELEMENTS = 512


def build_overrides(truth: dict) -> list:
    """One -P assignment per op class whose latency the M3 disagrees with."""
    overrides, notes = [], []
    for instruction, op_class in INSTRUCTION_OP_CLASS.items():
        measured = truth.get(instruction, {}).get("latency_cycles")
        if measured is None:
            continue
        # opLat is an integer count of cycles, so a measurement of 4.29 can
        # only ever become 4. That rounding is a real limit on how well any
        # gem5 model can match silicon, not a shortcut taken here.
        value = max(1, round(measured))
        overrides.append(f"{FU_PATH[op_class]}.opLat={value}")
        notes.append(f"{op_class}={value} (measured {measured})")
    return overrides, notes


async def simulate(code: str, reps: int, overrides: list, label: str):
    result = await gem5.simulate_c_workload(
        code=code.replace("REPS", str(reps)),
        cpu_type="O3CPU",
        param_overrides=overrides,
        label=label,
        timeout_seconds=1500,
    )
    if not result.get("success"):
        return None, str(result.get("error"))[:150]
    return result["data"].get("cycles"), None


async def kernel_cycles_per_element(overrides: list, tag: str):
    small, error = await simulate(HELD_OUT, SMALL_REPS, overrides, f"{tag}-small")
    if error:
        return None, error
    large, error = await simulate(HELD_OUT, LARGE_REPS, overrides, f"{tag}-large")
    if error:
        return None, error
    if large <= small:
        return None, "more work did not cost more cycles"
    return (large - small) / ((LARGE_REPS - SMALL_REPS) * ELEMENTS), None


async def measure_on_m3():
    """Cycles per element on the real machine, anchored the same way as before."""
    anchor = next(i for i in INSTRUCTIONS if i.name == "add")
    result = await native.benchmark_c_snippet(
        code=build_source(anchor, ways=1), flags="-O1", repeat=7, label="anchor"
    )
    samples = (result.get("data", {}).get("reported_metrics") or {}).get("ns_per_op") or []
    if not samples:
        return None, None, "anchor produced no samples"
    ghz = frequency_ghz(min(samples))

    source = (
        "#include <time.h>\n"
        + HELD_OUT.replace("REPS", "2000").replace(
            "    for (int r = 0; r < 2000; r++)",
            "    struct timespec t0,t1; clock_gettime(CLOCK_MONOTONIC,&t0);\n"
            "    for (int r = 0; r < 2000; r++)",
        ).replace(
            '    printf("out=%d\\n", (int)x[3]);',
            "    clock_gettime(CLOCK_MONOTONIC,&t1);\n"
            "    double ns=(t1.tv_sec-t0.tv_sec)*1e9+(t1.tv_nsec-t0.tv_nsec);\n"
            '    printf("out=%d\\n", (int)x[3]);\n'
            '    printf("ns_per_element=%.6f\\n", ns/(2000.0*512));',
        )
    )
    result = await native.benchmark_c_snippet(
        code=source, flags="-O3 -lm", repeat=7, label="heldout-m3"
    )
    if not result.get("success"):
        return None, ghz, str(result.get("error"))[:150]
    samples = (result.get("data", {}).get("reported_metrics") or {}).get(
        "ns_per_element"
    ) or []
    if not samples:
        return None, ghz, "kernel printed no ns_per_element"
    return min(samples) * ghz, ghz, None


async def main() -> None:
    truth = json.load(open("/app/data/uarch/m3-truth.json"))["per_instruction"]
    overrides, notes = build_overrides(truth)
    print("tuned op latencies, fitted to the M3 microbenchmarks:")
    for note in notes:
        print(f"  {note}")

    measured, ghz, error = await measure_on_m3()
    if error:
        print(f"\nheld-out kernel could not be measured on the M3: {error}")
        return
    print(f"\nheld-out kernel on the M3: {measured:.2f} cycles/element (at {ghz:.2f} GHz)")

    stock, error = await kernel_cycles_per_element([], "stock")
    if error:
        print(f"stock model failed: {error}")
        return
    tuned, error = await kernel_cycles_per_element(overrides, "tuned")
    if error:
        print(f"tuned model failed: {error}")
        return

    stock_error = (stock - measured) / measured
    tuned_error = (tuned - measured) / measured
    print(f"stock O3CPU:  {stock:.2f} cycles/element   error {stock_error*100:+.0f}%")
    print(f"tuned O3CPU:  {tuned:.2f} cycles/element   error {tuned_error*100:+.0f}%")
    verdict = (
        "tuning helped on a kernel it was not fitted to"
        if abs(tuned_error) < abs(stock_error)
        else "tuning did NOT help; the fit does not generalise"
    )
    print(f"\n{verdict}")
    json.dump(
        {
            "overrides": overrides,
            "held_out_kernel": "vec3 normalize, 512 elements",
            "m3_cycles_per_element": round(measured, 3),
            "m3_frequency_ghz": round(ghz, 3),
            "stock_cycles_per_element": round(stock, 3),
            "tuned_cycles_per_element": round(tuned, 3),
            "stock_relative_error": round(stock_error, 3),
            "tuned_relative_error": round(tuned_error, 3),
        },
        open("/app/data/uarch/tuning.json", "w"),
        indent=2,
        sort_keys=True,
    )
    print("written to data/uarch/tuning.json")


asyncio.run(main())
