# uarch-model

Agentic generation, tuning and calibration of microarchitecture models.

## Why this exists

Comparing an llvm-mca estimate against a gem5 measurement produced errors of
−22.7%, +20.1%, +41.6% and −83.4% across four kernels — sign-flipping, so no
correction factor exists. The cause was not subtle: mca was modelling
**neoverse-n1**, gem5 was running its **generic O3CPU**, and the silicon
underneath is an **Apple M3**. Three different cores.

An instruction-set extension proposal rests on a claim of the form "this
sequence costs N cycles and the replacement costs M". That claim is worth
nothing without a model of a specific core that has been shown to predict that
core. Building and validating such a model is a separate problem from proposing
extensions, and this is where it lives.

## The loop

1. **Generate** a candidate model — a gem5 core config (`configs/common/cores/arm/`
   ships `neoverse_v2.py`, `O3_ARM_v7a.py`, `HPI.py` with per-op-class `opLat`,
   functional units, widths and queue depths), or an LLVM scheduling model for
   mca.
2. **Predict** a microbenchmark's cost under that model.
3. **Measure** the same benchmark on ground truth.
4. **Score** the error in the calibration store, which requires a prediction to
   be recorded before its measurement and to cite the evidence it derives from.
5. **Tune** parameters and repeat, holding out kernels the tuning never saw.

## Ground truth

The **Apple M3** performance cores in this machine, measured by wall clock.
That is the only silicon available here, so it is what the method can be proven
against. It is not a customer target: the deliverable is a validated method
plus an M3 model, and pointing it at another core needs that core's silicon,
its vendor simulator, or its published characteristics.

## How the measurements work

Microbenchmarks are **inline assembly chains**, not C expressions, so the
compiler cannot vectorise, unroll or hoist away the thing being measured. An
earlier attempt at exactly that failed twice: clang folded a loop to a closed
form, and later hoisted a pure call out of its scaling loop so a workload
scaled 100× returned byte-identical cycle counts.

- **Latency**: a chain where each instruction depends on the previous. Time per
  instruction is its latency.
- **Throughput**: several independent chains interleaved so latency is hidden.
  Time per instruction is its reciprocal throughput.
- **Frequency anchor**: a dependent chain of `add xN, xN, #1`, which retires one
  per cycle on any sane core, converts wall clock into cycles. Without an
  anchor every measurement would be in nanoseconds and no model comparison
  would be possible.

## Results so far

Measured on the host, three runs, median with run-to-run spread
(`results/m3-truth.json`). Latencies in cycles, anchored as above:

| | add | mul | fadd_s | fmul_s | fmadd_s | fdiv_s | fsqrt_s | fadd_v | fmla_v |
|---|---|---|---|---|---|---|---|---|---|
| latency | 1.01 | 2.78 | 2.62 | 3.75 | 4.14 | 8.00 | 10.14 | 3.03 | 4.29 |
| spread | 19% | 17% | 44% | 10% | 9% | 12% | 13% | 15% | 19% |

The `add` row is the check on the rest: a dependent integer add is one cycle by
construction, so its 1.01 says the anchor and the derived cycles agree. One of
the three runs returned 0.83, which is impossible, and is the rejection
criterion for a run rather than a number to average in.

**The shipped gem5 models are far from this silicon** (`results/calibration.json`).
Each benchmark was simulated at two iteration counts and the slope taken, which
recovers each model's `opLat` exactly — 24.00, 12.00, 5.00 — confirming the
method reads the parameter it aims at:

| | O3CPU | NeoverseV2 | M3 |
|---|---|---|---|
| fsqrt_s | 24 | 33 | **10.1** |
| fdiv_s | 12 | 12 | **8.0** |
| fmla_v | 1 | 1 | **4.3** |
| add | 1 | 2 | **1.0** |

Mean absolute error: **40% for O3CPU, 77% for NeoverseV2**. A one-cycle vector
FMA does not exist in silicon, and no real Neoverse core takes two cycles for an
integer add. This is the concrete reason the earlier mca-versus-gem5 comparison
could not be interpreted: neither side was modelling the machine underneath.

**Tuning generalises, partially** (`results/validation.json`). Setting each
`opLat` to its measured value and testing on a *held-out* dependent chain of six
mixed instructions:

| | cycles/op | error |
|---|---|---|
| M3 measured | 3.34 | — |
| stock O3CPU | 6.83 | +105% |
| tuned O3CPU | 4.67 | +40% |

Halving the error on a sequence the fit never saw is real, but the residual
looked like a measurement bias and was not one.

**The apparent 26% bias was anchor drift, and it is now understood.** Summing
the measured latencies predicted 4.50 cycles/op for a chain the M3 ran at 3.34.
A dependent chain cannot beat the sum of its own latencies, so the
per-instruction numbers looked about 26% high. They were not: cycles are wall
clock times an assumed frequency, the latencies were taken with the anchor
reading 2.35 GHz and the chain with it reading 1.77 GHz, and the two factors
match almost exactly — 1.349 of apparent error against 1.328 of frequency.
Nothing was biased; two measurements were simply on different scales.

`harness/selfanchor.py` is the fix: it times the anchor and the target in the
same process, so the result is a ratio and the frequency divides out.

**What that then exposed is the real limit.** Run the anchor against *itself* —
the same instruction on both sides, so the answer must be 1.00 by construction
— and this host returns 0.87, while reporting its load as quiet. A 13%
disagreement between two identical measurements in one process is the floor:
no effect smaller than that can be claimed from wall clock here, whatever
statistic is used and however many trials are run.

Two statistics were tried and both fail their own sanity check in opposite
directions under contention: pooling every round gives a defensible 0.97 for
the anchor but 400–800% spreads across trials, while taking the least disturbed
round reports a dependent integer add at 0.32 cycles, which is impossible —
the minimum ratio prefers whichever round had the slowest *anchor*. The harness
now refuses to report latencies at all when the anchor check fails, rather than
returning numbers that look like measurements.

Getting a clean set needs a quiet machine, which is a scheduling problem rather
than an engineering one. The competing workload on this host is described
below.

## Doing this from an agent

The loop above no longer needs a human driving the simulator. `describe_model_parameters`
returns a model's op-class latencies and core widths together with the exact
parameter paths that set them, and `simulate_c_workload` takes those paths back
as `param_overrides`. So "measure the silicon, then tune the model to match" is
a sequence of tool calls rather than an afternoon of reading `config.ini`.

Two things the tool reports that are easy to get wrong by hand:

- **An op class can live in several issue queues.** NeoverseV2 carries
  `FloatSqrt` in two, so tuning it means setting both paths; setting one leaves
  the model half-tuned and the error unexplained. Each op class therefore
  reports a list of `parameters`, not a single path.
- **The paths gem5 prints cannot be assigned to.** `config.ini` shows
  `FUList03.opList4`, while the flag needs `FUList[3].opList[4]`, and a vector
  printed bare (`FUList` in NeoverseV2) still needs `[0]`. Get it wrong and gem5
  raises `KeyError` from inside its own Python, naming neither the flag nor the
  path; the sandbox now translates both failures into the indexing rule.

## Two traps that cost most of the time here

- **gem5's NeoverseV2, ex5_big and ex5_LITTLE deadlock on scalar `fmadd`.**
  Their functional-unit pools declare `SimdFloatMultAcc` but not
  `FloatMultAcc`, so the instruction can never issue and the model waits
  forever rather than failing. Measured: a program whose only unusual
  instruction is one `fmadd` finishes in 2s on O3CPU and never on NeoverseV2.
  Compilers contract `a*b+c` into `fmadd` by default, so this hits ordinary
  floating-point code. Every benchmark here hung at first, for a reason that
  had nothing to do with what any of them measured — the wall-clock epilogue
  `(t1-t0)*1e9 + ns` compiles to exactly one `fmadd`. `simulate_c_workload`
  now checks the compiled binary and refuses with this explanation.
- **The native and simulated toolchains are different compilers.** The
  benchmark tool uses clang, the gem5 tool uses gcc. On a plain C kernel clang
  vectorised and gcc did not, and the two "identical" kernels disagreed by 20x
  — a difference that says nothing about any model. Validation kernels are
  inline assembly for this reason, not only for the optimiser.

## What this cannot do here, stated plainly

- **No performance counters.** macOS does not expose the PMU without
  entitlements, and this runs in a VM besides. Everything is wall clock, so a
  parameter only becomes measurable if a benchmark can be built where it
  dominates the time.
- **Noise and DVFS.** The M3 scales frequency and has 4 performance plus 4
  efficiency cores; a container may be scheduled on either. Measurements use
  many trials and take the minimum, which biases toward a warm performance core
  — the best available proxy, not a guarantee.
- **Identifiability.** Many parameter sets reproduce the same aggregate
  timings. Where a fit is not unique the model must say so rather than present
  one solution as the model.
- **Overfitting.** A model tuned on microbenchmarks that fails on real kernels
  is worse than no model, because it fails with authority. Validation kernels
  are held out of tuning and scored separately.
