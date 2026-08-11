"""Turn raw timings into the microarchitectural claims under test."""
import json, subprocess, sys

def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)

# perf exits 0 even when the PMU is absent and it prints "<not supported>",
# so require an actual cycle count before claiming counters are usable.
_perf = sh("perf stat -e cycles true")
_perf_text = (_perf.stdout or "") + (_perf.stderr or "")
out = {
    "perf_counters_usable": (
        _perf.returncode == 0
        and "not supported" not in _perf_text
        and "not permitted" not in _perf_text
    ),
    "perf_raw": _perf_text.strip()[:200],
}

# Validate the instrument before trusting its timings. clang will happily
# turn the conditional accumulate into predicated NEON, leaving no branch to
# mispredict; a timing difference measured in that state would be noise
# dressed up as a result.
sh("clang -O2 -S -o branch.s branch.c")
asm = open("branch.s").read() if __import__("os").path.exists("branch.s") else ""
vector_ops = sum(asm.count(t) for t in ("uaddw", "addp"))
acc_calls = asm.count("bl\tacc") + asm.count("bl      acc")
out["instrument"] = {
    "vector_ops_in_hot_loop": vector_ops,
    "opaque_calls_present": acc_calls > 0,
    "branch_actually_present": vector_ops == 0 and acc_calls > 0,
}

r = sh("./branch")
if r.returncode == 0:
    t_rand, t_sorted = (float(x) for x in r.stdout.split())
    out["branch"] = {
        "random_s": round(t_rand, 6),
        "sorted_s": round(t_sorted, 6),
        "speedup_sorted_over_random": round(t_rand / t_sorted, 3),
        # Same instructions, same data, only order differs: a real gap is
        # branch misprediction cost.
        # Only meaningful if a branch survived compilation.
        "misprediction_effect_detected": (
            t_rand / t_sorted > 1.15
            and out["instrument"]["branch_actually_present"]
        ),
    }
else:
    out["branch"] = {"error": r.stderr.strip()[:200]}

r = sh("./cache")
if r.returncode == 0:
    pts = json.loads(r.stdout)["points"]
    out["cache"] = {"points": pts}
    lat = [p["ns_per_access"] for p in pts]
    # A hierarchy shows up as latency growing with working-set size; report the
    # largest step between adjacent sizes as the clearest level boundary.
    steps = [
        {"from_kb": pts[i]["kb"], "to_kb": pts[i + 1]["kb"],
         "ratio": round(lat[i + 1] / lat[i], 3)}
        for i in range(len(pts) - 1)
    ]
    biggest = max(steps, key=lambda s: s["ratio"]) if steps else None
    out["cache"]["largest_latency_step"] = biggest
    out["cache"]["min_ns"] = round(min(lat), 3)
    out["cache"]["max_ns"] = round(max(lat), 3)
    out["cache"]["hierarchy_detected"] = bool(lat and max(lat) / min(lat) > 2.0)
else:
    out["cache"] = {"error": r.stderr.strip()[:200]}

json.dump(out, sys.stdout, indent=2)
print()
