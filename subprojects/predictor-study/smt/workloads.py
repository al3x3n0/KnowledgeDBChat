import asyncio
from app.services import agent_gem5_sandbox as g
from app.services import agent_predictability as pred

# Primary: the thread whose progress a scheduler hint would predict. Its own
# work is steady, so any variation in its progress comes from contention.
PRIMARY = r'''
#include <stdio.h>
#define N 2048
static float a[N];
int main(void) {
    for (int i = 0; i < N; i++) a[i] = i * 0.5f + 1.0f;
    volatile float s = 0.0f;
    for (int frame = 0; frame < 400; frame++) {
        for (int r = 0; r < 8; r++)
            for (int i = 0; i < N; i++) s += a[i] * 1.0000001f;
        M5_SAMPLE();
    }
    printf("p=%f\n", (double)s);
    return 0;
}
'''

# Co-runner: alternates between cache-resident and memory-hostile phases, so
# the pressure it puts on the shared core changes over time. Long phases, so
# there is structure a hint could in principle exploit.
CO = r'''
#include <stdio.h>
#define BIG (1<<20)
#define SMALL 1024
static long big[BIG];
static long small[SMALL];
int main(void) {
    for (long i = 0; i < BIG; i++) big[i] = i;
    for (long i = 0; i < SMALL; i++) small[i] = i;
    volatile long s = 0;
    for (int phase = 0; phase < 200; phase++) {
        if ((phase / 10) % 2 == 0) {
            for (int r = 0; r < 40; r++) for (long i = 0; i < SMALL; i++) s += small[i];
        } else {
            for (long i = 0; i < BIG; i += 8) s += big[i];
        }
    }
    printf("c=%ld\n", (long)s);
    return 0;
}
'''

async def main():
    r = await g.sample_counters(code=PRIMARY, co_runner=CO, cpu_type="O3CPU",
                                label="SMT: steady primary, phasing co-runner",
                                max_counters=50, timeout_seconds=3600)
    if not r.get("success"):
        print("FAILED:", str(r.get("error"))[:400]); return
    d = r["data"]
    print(f"intervals={d['intervals']} smt={d['smt']} co-runner active in {d['co_runner_active_intervals']}")
    if "WARNING" in str(d.get("note")):
        print("note:", str(d["note"]).split("WARNING",1)[1][:200])
    import json
    json.dump(d["series"], open("/app/smt_series.json", "w"))
    print("series saved for re-analysis without re-simulating")
    for target in ("derived.thread0_ipc",):
        c = pred.ceiling(d["series"], target)
        print(f"\n-- target {target}")
        if not c.get("measured"):
            print("   refused:", str(c.get("refusal"))[:180]); continue
        n = c.get("null") or {}
        print(f"   entropy {c['target_entropy_bits']} | persistence {c['persistence_information_bits']} "
              f"| best beyond {c['best_counter_beyond_persistence_bits']} | null p95 {n.get('null_p95')}")
        print(f"   survives null: {c['survives_null']}")
        for row in c["counters"][:3]:
            mark = "*" if row.get("above_null_p95") else " "
            print(f"    {mark} {row['counter'][:42]:42s} beyond={row['information_beyond_persistence']:.3f}")
        t = pred.select_taps(d["series"], target)
        print(f"   -- combinations: depth {t.get('max_taps_supported')} | "
              f"reached {t.get('total_beyond_persistence')} | null {t.get('null_p95')} | "
              f"survives {t.get('survives_null')}")
        for step in t.get("selection", []):
            print(f"      +{step['tap'][:40]:40s} total={step['total_beyond_persistence']:.3f} added={step['added']:.3f}")

asyncio.run(main())

