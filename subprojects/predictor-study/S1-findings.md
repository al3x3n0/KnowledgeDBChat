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

**These three rows are unverified against the defect found later** (see
*The trace was two experiments*). Each workload initialises before it settles,
which is the shape that inflates persistence, and the persistence column is
what carries the conclusion here. They cannot be re-checked: neither the traces
nor the workload sources survive — the study was driven from standalone scripts
rather than agent jobs, so nothing was bundled. Re-running them means writing
new workloads from these descriptions, whose numbers would not be comparable to
this table. Read the persistence column as an upper bound of unknown tightness.

Read together, and subject to that:

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
memory-hostile phases, ten intervals each. 401 intervals.

### The trace was two experiments

The co-runner was active in every interval, which the tool checked and
reported. It had not finished *initialising*: it spends its first ~104
intervals filling a 1M-element array, during which the primary runs
effectively unopposed at 131k cycles per interval against 509k afterwards — a
3.9× level shift. The presence check could not see this, because presence was
never the question.

Splicing those two regimes together makes the trace trivially predictable at
the join: almost every interval is in the same regime as the one before it, so
**persistence absorbs the splice and reports it as structure the workload does
not have.** Both segments, same trace, re-analysed:

| target | segment | entropy | persistence | best beyond | null p95 | margin | top counter |
|---|---|---|---|---|---|---|---|
| thread-0 IPC | all 401 | 1.585 | 0.843 | 0.106 | 0.053 | 2.0× | `iew.iqFullEvents` |
| thread-0 IPC | steady 297 | 1.585 | **0.405** | 0.113 | 0.087 | 1.3× | `iew.iqFullEvents` |
| thread-1 IPC | all 401 | 1.585 | 0.832 | 0.070 | 0.055 | 1.3× | `simInsts` |
| thread-1 IPC | steady 297 | 1.584 | **0.220** | 0.124 | 0.084 | 1.5× | `tol2bus.transDist::CleanEvict` |
| cycles | all 401 | 1.585 | 0.793 | 0.132 | 0.055 | 2.4× | `iew.iqFullEvents` |
| cycles | steady 297 | 1.585 | **0.409** | 0.134 | 0.080 | 1.7× | `iew.iqFullEvents` |

**Persistence roughly halves on every target.** More than half of what the
first reading called predictable structure was the trace announcing that solo
intervals stay solo.

**The tap contribution survives intact** — 0.106 → 0.113, 0.070 → 0.124,
0.132 → 0.134. `iqFullEvents` was never measuring the startup transient, which
is the one thing the first reading got right for the right reason.

**Margins fall, because the null rises on a shorter segment.** Thread-0 at 1.3×
is now marginal rather than comfortable. Nothing here is refuted, but less is
established than the first table implied.

**One counter identity changes.** Thread-1's top tap was `simInsts` across the
splice and is `tol2bus.transDist::CleanEvict` in steady state. `simInsts`
distinguished "co-runner still initialising" from "co-runner running" — it was
reading the break, not the contention. That row is retracted.

The sampler now detects this and reports it, and a goal contract may declare
`validity.traces_one_regime` to refuse a finding measured across a break. The
measurement tools take `from_interval` so a run can study one side.

## Which counters, together

One tap at a time cannot answer the design question, because PMU taps cost
wires and picking the best of fifty and then measuring it is the
multiple-comparisons trap. Greedy forward selection from persistence, at the
depth the trace supports (2 taps at 3 bins and 297 intervals), with the null
running the *same selection* on permuted counters so its threshold contains the
advantage selection confers — and each tap judged on **its own increment**,
because that is what its area buys:

| target | tap 1 | added | null | tap 2 | added | null | keep |
|---|---|---|---|---|---|---|---|
| thread-0 IPC | `iew.iqFullEvents` | 0.113 | 0.086 | `dram.rank0.pwrStateTime::IDLE` | 0.150 | 0.163 | **1** |
| thread-1 IPC | `tol2bus::CleanEvict` | 0.124 | 0.088 | `iew.dispatchStatus::unblocking` | 0.104 | 0.176 | **1** |
| cycles | `iew.iqFullEvents` | 0.134 | 0.091 | `dram.rank0.preBackEnergy` | 0.170 | 0.157 | 2, at 1.08× |

Scored on the cumulative total instead, thread-0's second tap reads 0.178
against a full-depth null of 0.127 and a two-tap design looks justified. Scored
on its increment it is a DRAM idle-power counter that won an auction among
fifty. The cycles row is the only one where a second tap survives at all, and
at 1.08× it is not a design.

Whether a *third* tap would add anything is unmeasured, not answered: 297
intervals cannot condition on one more.

## What a buildable predictor gets

A ceiling says what is available, not what a design reaches. The predictor is
run and scored against its own ceiling on intervals it was never warmed on —
contiguous split, never random, because adjacent intervals are near-identical
and a random split puts each scored row's twin in the warm-up.

For thread-0 IPC (148 warm, 148 scored). Persistence is 65.5% correct; the best
a table on (last IPC, `iqFullEvents`) could do is 68.9%:

| design | state | accuracy | vs persistence |
|---|---|---|---|
| bimodal on last value | 9 b | 62.8% | −2.7 |
| last value + tap | 18 b | 57.4% | −8.1 |
| last value + tap, with hysteresis | 27 b | 65.5% | ±0.0 |
| **last value + tap, per-level counters** | **54 b** | **68.9%** | **+3.4** |

**The mechanism decides this, not the information.** A single counter per cell
is ordinal: it travels through the middle level to get from low to high, so
where a cell's majority is weak it never arrives and the table degenerates into
predicting the same as last interval — indistinguishable, from outside, from a
tap carrying nothing. Per-level counters reach the majority and hit the ceiling
exactly, against a null of +1.3.

**So the answer for thread-0 is 54 bits of state**: nine cells, three two-bit
counters each, reading the thread's own last IPC level and one
issue-queue-pressure tap. It captures 100% of what that feature pair supports,
which means a learned model on those features is competing for nothing.

The other two targets do not land the same way. Cycles reaches its ceiling with
27 bits. **Thread-1 has no buildable design here**: its best gain is +2.7
against a null of +3.4, so on this feature pair nothing beats last-value.

## What this does and does not establish

**Established.** The method rejects noise — demonstrated directly, on fifty
pure-noise counters and on a permuted null under every estimate. Trace length
must be in the hundreds of intervals. A trace must be checked for regime
changes before anything is measured across it. Under SMT, for thread-0
progress, one issue-queue-pressure tap carries real signal and a 54-bit table
extracts all of it that the feature pair holds.

That the method *detects* planted structure rests on the positive control's
6.3× margin, which shares the unverified status of its row. The SMT results are
the stronger evidence for detection, because there the tap contribution was
re-derived after removing a regime break and held (0.106 → 0.113) while
persistence halved.

**Retracted.** Thread-1's top counter (`simInsts`) was reading the co-runner's
initialisation, not contention. The claim that a thread's own recent IPC
carries 53% of the entropy holds only across the splice; in steady state it is
26%, which leaves *more* for a predictor to do than was reported, not less.

**Not established.** No real corpus has been run. The multi-file plumbing
exists — `sample_hardware_counters` takes `language`, `extra_files` and
`include_dirs`, and the Godot header closure was computed (21 headers, via
`clang++ -MM`) and staged. The blocker is one level below: **the gem5 image has
no C++ compiler and no static libstdc++.** Only `gcc` is installed;
`profile_c_workload` runs C++ because it uses a different image. The tool now
refuses C++ with that reason rather than surfacing `g++: not found`.

Until then nothing here is a statement about real programs, and a C workload
must not be read as standing in for the C++ corpus.

The three non-SMT rows above are also unverified against the regime defect and
cannot be re-checked, because neither their traces nor their workload sources
survive. That is a process failure worth naming: the study ran from standalone
scripts rather than agent jobs, so the reproducibility bundles that exist for
exactly this purpose captured none of it. The SMT results could be corrected
only because one trace happened to be kept by hand.

Also untested: targets beyond the three here, more than two taps, bin counts
other than three, and a named AArch64 core rather than a generic O3CPU.
