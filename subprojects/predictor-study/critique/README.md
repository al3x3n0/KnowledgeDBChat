# An agent's own workload, and what a critic found in it

`agent-workload.c` was written by the agent, not by a human. Given a bare goal
— counters, next-interval cycles, "a workload that alternates between a
cache-resident phase and a memory-heavy phase" — it chose
`sample_hardware_counters` unaided and wrote this to feed it.

It is a competent program. A 64 KB array for cache residency, an xorshift
walking 8192 lines across 256 MB for the memory phase, `M5_SAMPLE()` after
each. The tool sizes are right and it compiles and runs.

It also does not answer the question, and the run spent its simulation budget
finding that out the slow way.

## What three adversarial lenses found, before any simulation

**It does not alternate.** `main()` runs 100 `cache_phase()` calls and then 100
`mem_phase()` calls. There is a single cache-to-memory transition and no
memory-to-cache transition, so for nearly every sample the next interval is the
same phase type as the current one. That is two experiments spliced, which is
the defect `validity.traces_one_regime` exists to refuse — the harness would
have caught it *after* the simulation.

**The first sample of each block is unrepresentative.** `cache_array` is never
touched before the first sampled `cache_phase`, so that interval runs cold; the
first `mem_phase` includes the 256 MB `malloc` and the cold misses for every
line it touches.

**The memory phase stops being a memory phase.** The xorshift is re-seeded to
the same constant on every call, so the same 8192 lines — 512 KB — are touched
each time. After the first invocation they are resident in L2, and every later
"memory-heavy" interval is cache-warm and short.

The third one is the reason this directory exists. A human had already read
this file closely and found only the first defect. The critic found all three
in one pass, for the price of three model calls against a simulation budget of
tens of minutes.

## What it cost to make the critic report them

Two shape failures in the reading code, both worth recording because they are
the same failure the study keeps hitting:

The provider returned `{"concern": ..., "location": ...}` where the schema asked
for `summary`/`why_it_matters`/`remedy`/`severity`. The first parser required
the declared names, discarded a correct and precisely-located finding, and
reported **0 concerns** — which reads as a clean design rather than as a parse
failure. Then a later call returned a bare list instead of the declared object,
which at least raised.

Schema-constrained output from this provider is a **request, not a guarantee**.
The parser is tolerant of shape first and retries second, in that order: a good
answer in the wrong shape must not cost another call.

And an absent `severity` is now `unrated` rather than defaulted to `serious`.
All four live concerns arrived unrated; defaulting them made "0 blocking" read
as a reviewer declining to escalate when it had never rated anything.

## And what a deterministic pass measured, without reading it

The critic reasons about an artifact and can be wrong. A second check runs the
workload in gem5's cheapest model -- `AtomicSimpleCPU`, no timing at all, so
its cycles are meaningless and its instruction counts are exact -- and compares
what comes out against what the estimator downstream needs. It therefore
refuses rather than advises.

Against this same file, in **116 seconds**:

```
202 intervals, 180,234,817 instructions
projected 1802s of out-of-order simulation, past the 1800s timeout
work changes level 17.33x at interval 101
```

Both refusals are correct. The blocked-phase boundary is at interval 101,
exactly where the hundred cache phases end. The cost projection is marginal --
1802 against 1800 is 0.1% over, and the figure is order-of-magnitude rather
than a measurement -- but the run it predicted was indeed still going after
fifteen minutes when it was checked.

The two checks corroborate from independent directions and neither found
everything. The critic *reasoned* that `mem_phase` stops being memory-heavy
after its first call; the preflight *measured* that the memory phase executes
17x fewer instructions than the cache phase. One by reading, one by counting.

Every check in the deterministic pass is a property of the trace the design
will produce -- interval count, variation, regime structure, cost -- rather
than a rule about how the workload was written. That is why it catches traps
nobody enumerated: the four this study lost time to were a trace too short, two
regimes spliced, a target too flat to bin, and phases blocked instead of
interleaved, and all four are numbers here.

It does not close the gap. A design whose *cycles* vary while its instruction
counts do not will pass and may still be flat, because the cheap model has no
timing. It narrows it.
