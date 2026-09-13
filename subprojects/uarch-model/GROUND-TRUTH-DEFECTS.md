# Two defects in the M3 ground truth, and why the held-out validation proved nothing

Found while validating a new cycle model against this subproject's measurements.
Both defects produce confident, plausible numbers, which is why neither was
visible until something was checked against something else.

## 1. Most of the chains overflow to infinity

A dependent floating-point chain changes the value it carries. Starting from
1.0, the chains used here reach `inf` in single precision almost immediately,
and every iteration after that times exceptional-value arithmetic rather than
the instruction named.

| chain | stays finite | reaches `inf` at iteration | share of a 100,000-iteration run at `inf` |
|---|---|---|---|
| mixed held-out kernel | **no** | 4 | 99.996% |
| `fadd_s` / `fadd_v` | **no** | 128 | 99.87% |
| `fmadd_s` / `fmla_v` | **no** | 8 | 99.99% |
| `fmul_s` | yes | — | — |
| `fsqrt_s` | yes | — | — |
| `fdiv_s` | yes | — | — |

The three stable ones are stable by luck, not design: 1.0 is a fixed point of
`x*x`, `sqrt(x)` and `x/x`. Nothing chose them for that, which is exactly why
this has to be checked per sequence instead of reasoned about once.

So **four of the nine classes in `results/m3-truth.json`** — `fadd_s`,
`fadd_v`, `fmadd_s`, `fmla_v` — and the held-out kernel in
`results/validation.json` are measurements of infinity arithmetic. Whether an
M3 costs `inf + inf` the same as `2.0 + 2.0` is unknown and is not the point:
the measurement does not establish what it claims, and the two must not be
assumed equal without checking.

`add` and `mul` are integer and unaffected.

### What this invalidates

- The four affected per-class latencies and throughputs.
- The held-out validation of the gem5 tuning. `validation.json` reports
  `tuned_relative_error: 0.398` against a kernel that is at infinity from its
  fourth iteration, so that number describes neither the tuning nor the kernel.
- The same four classes in the `aha-cycle-arm` core configuration, which was
  built from this table.

### The check

`harness/control.py::stays_finite` walks the chain in **single precision** over
a horizon comparable to the real loop. Both qualifiers were learned the hard
way here: an eight-iteration walk reports `fadd` stable when it overflows at
128, and a double-precision walk reports it stable at any horizon because the
overflow is at 2¹²⁸ and a double does not reach infinity until 2¹⁰²⁴. A check
less faithful than the thing it checks passes everything.

## 2. The host cannot currently measure anything

A **null control** — two identical dependent add chains, different registers,
one program — must read a ratio of 1.0.

```
A then B:  ratio 0.60      B then A:  ratio 0.49
```

Whichever block runs first is about twice as slow, in both orderings, so the
harness is measuring position rather than instructions. Absolute times also
moved 10× between two runs in the same program.

The cause is not subtle: **load average 173 on 8 cores**, with 12 containers
running. At 21× oversubscription no wall-clock ratio survives.

This is a stronger statement than the existing note that measurements were
taken at 1.08 per CPU with 22% trial spread. That is a noisy measurement. This
is no measurement at all, and the null control is what distinguishes them —
`harness/control.py::null_control` refuses the host rather than returning a
number.

## 3. A null control is not enough — and stopping the stack is not enough

Two further things were learned trying to re-take the measurements.

**The statistic mattered more than the machine.** Pooling totals across rounds
gave a null control of 0.53. ABBA ordering, which cancels linear drift, made it
*worse* and more variable (0.61 to 1.16), so the disturbance is bursty rather
than a trend — a sum lets one preempted block dominate. Taking the **median of
per-round ratios** over 31 rounds took the null control to 1.0000 repeatedly.
The original per-instruction measurements pooled totals, so they carry the
position bias as well as defect 1.

**A control on identical chains cannot detect a host that cannot measure.**
With the compose stack stopped, the null control read 1.0000 four times running
while a control whose answer is a known ratio of **2.0** read 2.50, then 2.00,
then 3.47. A disturbance that hits two identical chains cancels; one that hits
two different chains does not. `harness/control.py::scale_control` is that
second gate, and `preflight` now requires both.

So the re-measurement attempted here is **not published**. Every class was timed
once, and a control with a known answer was off by up to 74% in the same
session. The numbers it produced (`fdiv_s` at 36 cycles, `fsqrt_s` at 40) are
not credible and are recorded here only as an example of what an ungated
harness will hand you.

**Stopping the compose stack was necessary and not sufficient.** It took the
null control from 0.09 to ~0.97, but load stayed near 200 on 8 cores with the
Docker VM alone at 79% CPU, plus dotnet, another AI CLI and unrelated
containers. The machine has to be genuinely idle.

## 4. Re-taken, on a machine that finally went quiet

Load fell to 0.42 per CPU roughly half an hour after the compose stack stopped,
and both gates passed: null control **1.0000**, scale control **1.9979**. The
corrected table is `results/m3-truth-v2.json`.

Three runs were taken and **one was discarded**. The scale control run
immediately after run C read **2.2012** against a required 2.000, and C's data
shows exactly the damage that predicts — `fsqrt_s` at 16.91 where the two
accepted runs read 10.01 and 10.43, `fdiv_s` at 10.13 against 8.01 twice. The
control caught a bad run on its own evidence, which is the point of bracketing
a measurement rather than only preceding it.

**Latencies reproduce and land on integers**, which is what a correct
measurement of a real pipeline should do — 8 of 9 within 5% across the two
accepted runs:

| class | new | old | change | was measured on `inf` |
|---|---|---|---|---|
| `add` | 1.00 | 1.01 | −1.0% | no |
| `mul` | 3.18 | 2.78 | +14.4% | no (least reproducible, 12%) |
| `fadd_s` | 2.81 | 2.62 | +7.3% | **yes** |
| `fmul_s` | 4.00 | 3.75 | +6.7% | no |
| `fmadd_s` | 4.01 | 4.14 | −3.1% | **yes** |
| `fdiv_s` | 8.01 | 8.00 | +0.1% | no |
| `fsqrt_s` | 10.22 | 10.14 | +0.8% | no |
| `fadd_v` | 2.79 | 3.03 | −7.9% | **yes** |
| `fmla_v` | 4.05 | 4.29 | −5.6% | **yes** |

The four infinity-affected classes moved by −8% to +7%. `fmul_s` moved +6.7%
despite never having been at infinity — that is the pooling bias of defect 3,
not defect 1.

**Reciprocal throughput still does not reproduce**: 12% to 55% run-to-run
spread on a host whose controls passed, only **1 of 9** within 5%. It is
recorded in v2 and marked unusable. Either the 16-way independent chain is too
narrow, or loop overhead dominates for the cheap operations, or both. Until
that is fixed, issue capacity in the `aha-cycle-arm` model is **Assumed**
rather than inferred from measurement, and its provenance says so.

## What this does *not* invalidate

The `aha-cycle-arm` model reproduces its inputs exactly, and that check is
unaffected: it is arithmetic over the configuration, not a timing.

What cannot yet be said is whether the model is *right*, because the only
held-out kernel available is defect 1 and re-measuring is blocked by defect 2.
The earlier reading — that the model lands +39.9% on the held-out kernel,
essentially identical to tuned gem5's +39.8% — is real but uninterpretable:
both consume the same latency table, and that table is partly measurements of
infinity.

## Re-taking the ground truth

1. Run `harness/control.py::preflight` first, and stop if it refuses. It checks
   load, the null control, and the finiteness of the specific sequence.
2. Use **value-stable chains**: keep the dependence but not the value, with
   neutral operands (`fmul s0, s0, s1` with `s1 = 1.0`; `fadd s0, s0, s2` with
   `s2 = 0.0`). The chain stays dependent and the value never moves.
3. Pin those operand registers with `register float one asm("s1")` and pass
   them as **inputs**. Putting them in the clobber list says "destroyed", which
   is the opposite, and leaves them holding garbage — measured here as a chain
   that silently carried 0.0 throughout.
4. Take it on a quiet machine. Stop the compose stack first.
