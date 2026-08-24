# S1: how much signal is in the counters, before designing a predictor

The customer question is whether a lightweight ML predictor, implemented
entirely in hardware and fed from PMU taps, could produce useful dynamic hints
— an SMT scheduling hint being the named example. S1 asks the question that
comes before any design: **is the information there at all?**

The answer is a property of the workload and the counter set, not of the
predictor, so it can be measured first. A study that skips it can spend months
designing predictors for signal that was never present.

## What was built

`sample_hardware_counters` runs a workload under gem5 and returns every scalar
counter **sampled over time** — the workload calls `M5_SAMPLE()` and each
interval reports the counts since the previous call. That is the shape a
hardware predictor reads; run totals have no time axis and cannot train or
evaluate one.

`measure_predictability` then asks, for a target counter, how much the others
say about its **next** interval.

The headline is deliberately not the obvious number. A counter can look highly
informative and be worthless, because the target's own previous value already
told you the same thing — programs run in phases, so almost everything is
autocorrelated. The reported figure is **information beyond persistence**: what
a counter adds over predicting the same as last interval. A predictor that
cannot beat last-value is not worth a transistor.

## Results

All at 401 intervals, target `system.cpu.numCycles`, 3 bins, 50 counters,
gem5 O3CPU. Bits.

| workload | entropy | persistence | best counter beyond | null p95 | survives null |
|---|---|---|---|---|---|
| positive control, 16-interval phases | 1.542 | 0.727 | 0.227 | 0.036 | **yes, 6.3×** |
| hash-varied object count | 1.585 | 0.588 | 0.055 | 0.050 | marginal, 1.1× |
| irregular object count (xorshift) | 1.585 | 0.005 | 0.038 | 0.055 | **no** |

Read together:

**Where a workload has phases, persistence is most of the answer.** In the
positive control it carries 0.727 of 1.542 bits — 47% — and the best counter
adds 0.227 on top. That 0.227 is the entire budget a learned predictor is
competing for, and it is an *upper bound* that no design reaches.

**Where the cost driver is outside the machine, the counters know nothing.**
With object count drawn from a PRNG, persistence carries essentially zero
(0.005) and no counter recovers any of the remaining 1.58 bits. That is the
expected and correct answer: nothing in microarchitectural state can see a
decision the program has not made yet.

The practical shape of the customer's question follows from those two rows. An
SMT hint is worth building only where the workload has phase structure — and
exactly there, last-value already captures most of it.

## The null, and why the first answer was wrong

An earlier run of this study reported **0.33 bits available beyond persistence**
on a real kernel, with `system.cpu.cpi` as the best counter. That was noise, and
it is worth recording how it survived a first reading.

Conditional mutual information is **positively biased with small samples**.
Conditioning on two discretised variables splits the trace across bins² cells,
and sparse cells manufacture apparent structure. Measured directly: on 50
pure-noise counters at 65 intervals, the estimator reports **0.31 bits**. That
is more than the real kernel produced.

So every estimate is now placed against a permutation null — same trace length,
same bins, same marginals, no relationship.

The first null was itself wrong, in a way worth naming. It was **per-counter**,
and a trace carries tens of counters: comparing each against a 95th percentile
means one in twenty clears it by chance, so fifty counters guarantee two or
three "findings" on structureless data. The null is now over the **maximum
across counters** — how large is the best of fifty when none is related. That
moved the threshold from 0.26 to 0.42 and flipped the result from surviving to
not.

**Trace length was the whole problem.** The null p95 falls from **0.42 at 65
intervals to 0.036–0.055 at 401**. Any S1 result on a short trace is bias.

## A workload defect found by the longer trace

The second row above — hash-varied object count — was written to look like
realistic frame-to-frame scene variation. It is not. Its consecutive object
counts differ by exactly two values (`{241, 945}` for one modulus, `{2097,
1585}` for another), a two-state alternation that persistence tracks easily.
That is why persistence scores 0.588 there.

At 65 intervals the structure was invisible; at 401 it dominated. The row is
kept because the lesson is the point: **a workload written to look irregular
was not**, and only the longer trace exposed it. The xorshift row is the honest
version.

## Under SMT: the customer's actual question

The named example was a scheduling hint, which means two threads competing and
a predictor deciding which to favour. That is now runnable, and the answer has
a shape.

Getting there needed one piece of knowledge the tool now carries. O3CPU's
physical register files are sized for one thread, and running two panics with
"Not enough physical registers" — **one register class at a time**, so a caller
discovering it pays a simulator startup per panic. Six overrides are applied
automatically; they are a structural requirement for the run to start, not a
microarchitectural claim.

Primary: a steady FP loop, identical work every interval, so any variation in
its progress comes from contention. Co-runner: alternating cache-resident and
memory-hostile phases, ten intervals each. 401 intervals, co-runner active in
all of them.

| target | entropy | persistence | best beyond | null p95 | margin | top counter |
|---|---|---|---|---|---|---|
| thread-0 IPC | 1.585 | 0.843 | 0.106 | 0.053 | 2.0× | `iew.iqFullEvents` |
| thread-1 IPC | 1.585 | 0.832 | 0.070 | 0.055 | 1.3× | `simInsts` |
| cycles | 1.585 | 0.793 | 0.132 | 0.055 | 2.4× | `iew.iqFullEvents` |

**The top tap is issue-queue-full events.** That is the microarchitectural
signature of SMT contention rather than a counter that happened to correlate:
when the co-runner floods the issue queue, the other thread stalls. It is the
counter an architect would have named.

**What it says about a design.** The thread's own recent IPC carries 53% of the
entropy, and one contention tap adds about 14% of what persistence leaves. So
the shape indicated is **last-value plus one issue-queue-pressure tap** — a
saturating counter and a single signal, not a perceptron. That is a smaller
design than the question implied, and the measurement is the argument for it.

**Instructions is the wrong progress metric**, found by measuring it: a thread
doing identical work each interval commits a constant instruction count
whatever the contention, and the target came back with entropy 0.0 — nothing to
predict. Under SMT the cycle count is shared by both threads, so it is not
thread progress either. Per-thread IPC is now derived and returned for exactly
this reason.

Caveats, all real: the co-runner's phases are planted, one counter at a time
rather than combinations, three bins, a generic O3CPU rather than a named
AArch64 core, and a synthetic primary.

## What this does and does not establish

**Established.** The method detects planted structure at 6.3× its own null and
correctly rejects noise. Trace length must be in the hundreds of intervals.
Where phases exist, persistence dominates and the headroom for a learned
predictor is roughly 0.23 bits of 1.54.

**Not established.** No real corpus has been run, and the reason has now been
pinned down rather than guessed at.

The multi-file plumbing exists — `sample_hardware_counters` takes `language`,
`extra_files` and `include_dirs`, mirroring `profile_c_workload`, and the Godot
header closure was computed (21 headers, via `clang++ -MM`) and staged. The
blocker is one level below: **the gem5 image has no C++ compiler and no static
libstdc++.** Only `gcc` is installed. `profile_c_workload` runs C++ because it
uses a different image.

So a C++ corpus cannot be counter-sampled until g++ and libstdc++-static are
added to the gem5 image. The tool now refuses C++ with that reason rather than
surfacing `g++: not found`, which is an error about a missing binary when the
situation is that this image cannot build C++ at all.

Until then, nothing here should be read as a statement about real programs, and
a C workload must not be read as standing in for the C++ corpus.

Also untested: targets other than cycles (stalls, misses, per-thread progress —
the last being what an SMT hint would actually predict), counter *combinations*
rather than one at a time, and anything under SMT, which this gem5 build
supports but which has not been exercised.
