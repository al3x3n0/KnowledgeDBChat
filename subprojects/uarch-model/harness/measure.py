"""Measure instruction latency and throughput on this machine's silicon.

Run inside the backend container, where the sandbox tools are available:

    docker compose exec -T -e PYTHONPATH=/app backend \
        python /app/data/uarch/measure.py

Everything is wall clock, so the numbers are only as good as the quietest
trial: each benchmark is run several times and the fastest is kept, which
biases toward a warm performance core rather than an efficiency core or a
frequency-scaled one. That is a proxy, not a guarantee, and the report says so.
"""

from __future__ import annotations

import asyncio
import json
import sys

sys.path.insert(0, "/app/data/uarch")

from chains import (  # noqa: E402
    INSTRUCTIONS,
    usable_ways,
    build_source,
    cycles_per_op,
    frequency_ghz,
    is_chain_limited,
)

from app.services import agent_compiler_sandbox as sandbox  # noqa: E402

TRIALS = 9
THROUGHPUT_WAYS = 24


async def run_one(instruction, ways: int) -> dict:
    source = build_source(instruction, ways=ways)
    result = await sandbox.benchmark_c_snippet(
        code=source,
        flags="-O1",  # the work is inline asm; -O1 keeps the loop honest
        repeat=TRIALS,
        label=f"{instruction.name}-x{ways}",
    )
    if not result.get("success"):
        return {"instruction": instruction.name, "ways": ways, "error": result.get("error")}
    metrics = result["data"].get("reported_metrics") or {}
    samples = metrics.get("ns_per_op") or []
    if not samples:
        return {
            "instruction": instruction.name,
            "ways": ways,
            "error": "the benchmark printed no ns_per_op",
        }
    return {
        "instruction": instruction.name,
        "ways": ways,
        # Minimum across trials: the fastest run is the one least disturbed by
        # scheduling, other load and frequency scaling.
        "ns_per_op": min(samples),
        "ns_per_op_all": samples,
    }


async def main() -> None:
    anchor = next(i for i in INSTRUCTIONS if i.name == "add")
    # Run the anchor repeatedly. Its spread is the measurement floor: nothing
    # derived from it can be trusted more tightly than the anchor varies, and
    # an earlier run reported a 1.47-cycle dependent add -- impossible by
    # construction, and the honest signal that the timings were noise.
    anchors = []
    for _ in range(3):
        result = await run_one(anchor, 1)
        if "error" in result:
            print(json.dumps(result, indent=2))
            return
        anchors.extend(result["ns_per_op_all"])
    best, worst = min(anchors), max(anchors)
    ghz = frequency_ghz(best)
    spread = (worst - best) / best
    print(f"frequency anchor: dependent add, {len(anchors)} samples")
    print(f"  fastest {best:.4f} ns/op, slowest {worst:.4f} ns/op, spread {spread*100:.0f}%")
    print(f"  -> assuming 1 cycle/op, the core ran at {ghz:.3f} GHz on its best sample")
    if ghz < 2.5:
        print("  !! an M3 performance core is near 4 GHz. A figure this low means")
        print("     the work landed on an efficiency core, was throttled, or the")
        print("     chain is not retiring one per cycle. Treat what follows as")
        print("     ratios between instructions, not absolute cycle counts.")
    print()

    print(f"{'instruction':<12}{'latency':>10}{'recip thr':>12}{'per cycle':>11}  note")
    print("-" * 60)
    rows = []
    for instruction in INSTRUCTIONS:
        ways = usable_ways(instruction, THROUGHPUT_WAYS)
        latency = await run_one(instruction, 1)
        throughput = await run_one(instruction, ways)
        row = {"instruction": instruction.name}
        if "error" in latency or "error" in throughput:
            row["error"] = latency.get("error") or throughput.get("error")
            rows.append(row)
            print(f"{instruction.name:<12}  {str(row['error'])[:50]}")
            continue
        row["latency_cycles"] = round(cycles_per_op(latency["ns_per_op"], ghz), 2)
        row["recip_throughput_cycles"] = round(
            cycles_per_op(throughput["ns_per_op"], ghz), 3
        )
        row["ns_per_op_latency"] = latency["ns_per_op"]
        row["throughput_ways"] = ways
        chain_limited = is_chain_limited(
            row["latency_cycles"], row["recip_throughput_cycles"], ways
        )
        row["throughput_chain_limited"] = chain_limited
        rows.append(row)
        note = "chain-limited, NOT the core" if chain_limited else ""
        print(
            f"{instruction.name:<12}{row['latency_cycles']:>10.2f}"
            f"{row['recip_throughput_cycles']:>12.3f}"
            f"{1 / max(row['recip_throughput_cycles'], 1e-9):>11.2f}  {note}"
        )

    out = sys.argv[1] if len(sys.argv) > 1 else "/app/data/uarch/measured.json"
    suspect = [r["instruction"] for r in rows if r.get("throughput_chain_limited")]
    report = {
        "frequency_ghz": round(ghz, 4),
        "anchor_residual_cycles": next(
            (r.get("latency_cycles") for r in rows if r["instruction"] == "add"), None
        ),
        "throughput_ways": THROUGHPUT_WAYS,
        "trials": TRIALS,
        "chain_limited": suspect,
        "instructions": rows,
    }
    with open(out, "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print(f"\nwritten to {out}")
    if suspect:
        print(f"\n{len(suspect)} throughput result(s) are chain-limited and must not")
        print(f"be read as core throughput: {', '.join(suspect)}")
        print(f"Raise THROUGHPUT_WAYS above {THROUGHPUT_WAYS} for those.")


asyncio.run(main())
