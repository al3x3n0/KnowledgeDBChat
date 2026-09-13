"""Measure latencies and the mixed chain on one self-anchored scale.

Cycles are wall clock times a frequency, and the frequency is whatever the core
happened to be running at. Measuring the anchor in one process and the target
in another therefore compares two different frequencies -- which is what the
subproject's apparent 26% measurement bias turned out to be, near exactly:
1.328 of frequency ratio against 1.349 of apparent error. Timing both in the
same process makes the result a ratio and the frequency divides out.

The anchor is measured against itself first. Both halves are then the same
instruction, so the answer must be 1.00 by construction, and whatever it
actually reads is this host's measurement floor. Anything smaller than that
floor cannot be claimed from these numbers however many trials are run.
"""
import asyncio, json, sys
sys.path.insert(0, "/app/data/uarch")
from chains import INSTRUCTIONS, MIXED_SEQUENCE, build_paired_source
from app.services import agent_compiler_sandbox as native

STUDIED = ["add", "fmul_s", "fadd_s", "fsqrt_s", "fmadd_s"]
TRIALS = 7


async def measure(label, source):
    r = await native.benchmark_c_snippet(code=source, flags="-O1", repeat=TRIALS, label=label)
    if not r.get("success"):
        return None, str(r.get("error"))[:140]
    d = r["data"]
    m = d.get("reported_metrics") or {}
    vals = m.get("cycles_per_op") or []
    if not vals:
        return None, "no cycles_per_op reported"
    # Median: the ratio is already frequency-independent, so the fastest trial
    # has no special claim and the middle one is less swayed by one bad round.
    vals = sorted(vals)
    return {
        "cycles_per_op": vals[len(vals) // 2],
        "all": vals,
        "environment": d.get("measurement_environment"),
        "load_per_cpu": d.get("load_per_cpu"),
    }, None


ANCHOR_LOW, ANCHOR_HIGH = 0.90, 1.15


async def main():
    # The anchor is the one measurement whose answer is known in advance, so
    # it is the only thing that can say whether the rest is worth reading.
    # Checked first and refused outright: a host that cannot retire a
    # dependent integer add in one cycle is not measuring instructions.
    anchor = next(i for i in INSTRUCTIONS if i.name == "add")
    got, err = await measure("anchor-check", build_paired_source(anchor))
    if err:
        print("anchor could not be measured:", err); return
    residual = got["cycles_per_op"]
    print(f"anchor check: dependent add reads {residual:.2f} cycles/op "
          f"(must be 1.00), host {got['environment']}")
    if not (ANCHOR_LOW <= residual <= ANCHOR_HIGH):
        print(f"\nREFUSING to report latencies: the anchor is off by "
              f"{abs(residual - 1) * 100:.0f}%, so every cycle count derived "
              f"from it would be wrong by about the same factor.")
        print("The host is too contended for wall-clock measurement right now.")
        return
    print()

    out = {}
    print(f"{'instruction':<12}{'cycles/op':>11}{'spread':>9}  env")
    print("-" * 45)
    for name in STUDIED:
        instruction = next(i for i in INSTRUCTIONS if i.name == name)
        got, err = await measure(name, build_paired_source(instruction))
        if err:
            print(f"{name:<12}  {err}"); continue
        spread = (max(got["all"]) - min(got["all"])) / min(got["all"])
        out[name] = got["cycles_per_op"]
        print(f"{name:<12}{got['cycles_per_op']:>11.2f}{spread*100:>8.0f}%  {got['environment']}")

    got, err = await measure("mixed", build_paired_source(sequence=MIXED_SEQUENCE))
    if err:
        print("mixed failed:", err); return
    measured = got["cycles_per_op"]
    predicted = sum(
        out[n] for n in ("fmul_s", "fadd_s", "fsqrt_s", "fmul_s", "fadd_s", "fmadd_s")
        if n in out
    ) / len(MIXED_SEQUENCE)
    print(f"\nheld-out mixed chain")
    print(f"  measured directly            {measured:6.2f} cycles/op")
    print(f"  predicted from the latencies {predicted:6.2f} cycles/op")
    print(f"  disagreement                 {(predicted-measured)/measured*100:+6.1f}%")
    json.dump({"latencies": out, "mixed_measured": measured,
               "mixed_predicted": predicted}, open("/app/data/uarch/selfanchored.json", "w"),
              indent=2, sort_keys=True)

asyncio.run(main())
