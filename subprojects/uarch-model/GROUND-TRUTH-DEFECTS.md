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
