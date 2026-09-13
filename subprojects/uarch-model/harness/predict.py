"""Score a simulated core model against what the host silicon actually does.

The comparison is deliberately not "read opLat out of the config and compare
it to a measurement". gem5's ARM decoder assigns scalar FP instructions to op
classes whose names do not match the mnemonics, so a config-reading comparison
would be checking my guess about that mapping rather than the model. Running
the identical benchmark through the model and asking what it cost sidesteps
the question entirely -- and it is also the prediction the model would really
be used to make.

Simulator startup dominates a short run, so each benchmark is simulated at two
iteration counts and the *slope* between them is taken. Whatever the static
binary spends reaching main cancels exactly, without having to be measured.
"""

from __future__ import annotations

import asyncio
import json
import sys

sys.path.insert(0, "/app/data/uarch")

from chains import INSTRUCTIONS, build_source, unroll_for  # noqa: E402

from app.services import agent_gem5_sandbox as gem5  # noqa: E402

MODELS = ["O3CPU", "NeoverseV2"]
SMALL, LARGE = 2_000, 6_000
STUDIED = ["add", "mul", "fadd_s", "fmul_s", "fmadd_s", "fdiv_s", "fsqrt_s", "fmla_v"]


async def cycles_at(instruction, iterations: int, model: str):
    source = build_source(instruction, ways=1, iterations=iterations, timed=False)
    result = await gem5.simulate_c_workload(
        code=source,
        cpu_type=model,
        label=f"{instruction.name}-{model}-{iterations}",
        timeout_seconds=1500,
    )
    if not result.get("success"):
        return None, result.get("error")
    return result["data"].get("cycles"), None


async def predict(instruction, model: str) -> dict:
    row = {"instruction": instruction.name, "model": model}
    small, error = await cycles_at(instruction, SMALL, model)
    if error:
        return {**row, "error": str(error)[:160]}
    large, error = await cycles_at(instruction, LARGE, model)
    if error:
        return {**row, "error": str(error)[:160]}
    ops_per_iteration = unroll_for(instruction, 1)
    delta_ops = (LARGE - SMALL) * ops_per_iteration
    row["cycles_small"], row["cycles_large"] = small, large
    if large <= small:
        # More work costing no more cycles means the loop was optimised away or
        # the stat is not what it claims. It has happened before, and reporting
        # the resulting number as a latency would be worse than reporting
        # nothing.
        return {**row, "error": "more iterations did not cost more cycles"}
    row["predicted_cycles_per_op"] = round((large - small) / delta_ops, 3)
    return row


async def main() -> None:
    truth = json.load(open("/app/data/uarch/m3-truth.json"))["per_instruction"]
    studied = [i for i in INSTRUCTIONS if i.name in STUDIED]
    results = []
    for model in MODELS:
        print(f"\n=== {model} vs Apple M3 ===", flush=True)
        print(f"{'instruction':<12}{'model':>9}{'M3':>9}{'M3 +-':>8}{'error':>10}")
        print("-" * 48)
        for instruction in studied:
            row = await predict(instruction, model)
            measured = truth.get(instruction.name, {})
            row["measured_cycles_per_op"] = measured.get("latency_cycles")
            row["measured_spread"] = measured.get("latency_spread")
            if "error" in row:
                print(f"{instruction.name:<12}  {row['error'][:44]}", flush=True)
                results.append(row)
                continue
            predicted, actual = row["predicted_cycles_per_op"], row["measured_cycles_per_op"]
            row["relative_error"] = round((predicted - actual) / actual, 3)
            # A disagreement smaller than the measurement's own run-to-run
            # spread is not evidence the model is right; it is evidence this
            # bench cannot tell. Saying so is the point of carrying the spread.
            row["within_noise"] = abs(row["relative_error"]) <= (row["measured_spread"] or 0)
            results.append(row)
            print(
                f"{instruction.name:<12}{predicted:>9.2f}{actual:>9.2f}"
                f"{(row['measured_spread'] or 0)*100:>7.0f}%"
                f"{row['relative_error']*100:>9.0f}%"
                f"{'  (within noise)' if row['within_noise'] else ''}",
                flush=True,
            )

    scored = [r for r in results if "relative_error" in r]
    with open("/app/data/uarch/calibration.json", "w") as handle:
        json.dump({"models": MODELS, "results": results}, handle, indent=2, sort_keys=True)
    print("\nwritten to data/uarch/calibration.json")
    for model in MODELS:
        rows = [r for r in scored if r["model"] == model]
        if not rows:
            continue
        errors = [abs(r["relative_error"]) for r in rows]
        signed = [r["relative_error"] for r in rows]
        print(
            f"{model:<12} mean |error| {sum(errors)/len(errors)*100:5.0f}%   "
            f"range {min(signed)*100:+.0f}%..{max(signed)*100:+.0f}%   "
            f"n={len(rows)}"
        )


asyncio.run(main())
