# SMLALB: a widening multiply-accumulate from signed bytes

A worked proposal, from a dynamic profile of real code to a machine-checked
proof, with a bundle that re-establishes every claim from source.

The bundle is `axis/`. `python3 axis/verify.py` regenerates the semantics from
the description and re-solves every obligation; it needs `axis` and `z3` on
PATH and exits non-zero if anything fails to reproduce.

## Where the candidate came from

Not from a wish list. `find_fusion_candidates` built the data-flow graph of each
hot block in a callgrind profile of the PCE INT8 multi-head attention kernel
(`xnn_transformer_mha_i8` and its softmax, extracted verbatim), enumerated
connected convex subgraphs within a 2-in/1-out operand budget, and ranked them
by the block's measured execution count.

`sxtl → smlal` came out at **1,572,864 dynamic occurrences**, with a bounded
saving of **5.0 cycles** each at neoverse-n1. Sign-extend packed signed bytes to
halfwords, then multiply halfword lanes, widen each product to 32 bits and
accumulate: two instructions and an intermediate register to move data up two
width steps and multiply once.

**Arm already ships SDOT** (v8.4-A), the four-way-reduction relative of this
shape. That is the most useful thing about this result rather than a
disappointment: there is published ground truth that a byte-input widening MAC
is worth building, and the pipeline arrived in that neighbourhood from a profile
without being told what to look for.

## What was formalized

`axis/smlalb.axisl` describes four instructions: the proposed `SMLALB`, the two
constituents `SXTL_B_H` and `SMLAL_H_S` it would replace, and a deliberately
wrong variant. AXIS elaborates one description into encoder, decoder, semantics,
compiler patterns and SMT-LIB, so the proposal has one source of truth and its
artifacts regenerate rather than being hand-written per candidate.

Lanes are carried in a 128-bit register as 4×32, each input lane's significant
value in its low 8 bits.

## The three obligations

Run against the semantics AXIS emits, in `QF_BV`, by z3 4.16.0.

| Obligation | Verdict | What it settles |
|---|---|---|
| `lane_reference.smt2` | **unsat** | The description means what it claims, lane by lane |
| `fusion_equivalence.smt2` | **unsat** | The fused form equals the pair it replaces, for every input |
| `negative_control.smt2` | **sat** | A wrongly-widened variant is caught |

Each asserts the negation of its claim, so `unsat` means no input distinguishes
the two — a proof over all 2³⁸⁴ inputs, not a benchmark over a sample.

**Obligation 1 exists because I got it wrong first.** The lane reference is
written independently of the `.axisl`, in explicit `extract`/`sign_extend`
rather than the shift pair the description uses, so agreement means the two say
the same thing instead of that the same expression was written twice. That
caught a real error: a literal in a vector expression is one 128-bit constant
sliced per lane, so a plain `24` shifts lane 0 by 24 and **lanes 1–3 by zero**.
Visible in the emitted SMT as `(bvlshr (_ bv24 128) (_ bv32 128))`. The shift
amount has to be 24 splatted into every lane, which is why the constants in the
source look absurd. Re-running obligation 1 against the version I nearly shipped
returns `sat`.

**Obligation 3 exists because a proof that cannot fail is not evidence.**
`SMLALB_ZEXT_BUG` is the same instruction with unsigned widening — the mistake
an INT8 pipeline could plausibly make, and one a benchmark on mostly-positive
activations would very likely miss. z3 returns a counterexample:

```
vn = ...000000f0   lane 0 byte 0xf0 = -16 signed, 240 unsigned
vm = ...000000d3   lane 0 byte 0xd3 = -45 signed, 211 unsigned
```

Signed: −16 × −45 = **720**. Unsigned: 240 × 211 = **50640**.

## What this does and does not establish

**Established.** The fused instruction computes exactly what `sxtl`+`smlal`
compute, for every input, with no overflow or truncation difference from
collapsing the two widening steps. That is the correctness half of a proposal,
and it is settled rather than argued.

**Not established, and each matters:**

- **The saving is an upper bound, not a prediction.** 5.0 cycles is the
  floor-to-ceiling bound from `cost_fusion_candidate` at neoverse-n1 — the
  fused form still has to do the slowest constituent's work, and at best
  replaces the whole sequence. It is not a measurement of an instruction that
  does not exist.
- **neoverse-n1 is not PCE hardware.** The cost model is a stand-in for the
  target, and on the mca-vs-gem5 comparison in this project mca neither
  predicted cost nor preserved ranking across four kernels.
- **The proof covers arithmetic, not packing.** NEON's byte packing and the
  register-file plumbing `sxtl` also performs are not modelled. The arithmetic
  is where the risk of fusing lives; the encoding is a separate question and no
  claim is made about it.
- **The latency, area and power figures in the `.axisl` are placeholders.**
  4 cycles, 3000 µm², 4 mW are not measured or synthesised. They are structural
  fields AXIS requires, and nothing here derives from them.
- **There is no denominator.** 1,572,864 occurrences × 5.0 cycles is up to 7.9M
  cycles, but no whole-program speedup is claimed, because the program's total
  cycle count under the same conditions was never measured.
- **One workload shape.** A single attention kernel at one size.

## Traps worth carrying forward

- **A stale `target/release` binary sent me after a bug that did not exist.**
  Vector shifts were rejected as "not lane-local" while `LANE_PARALLEL` in the
  source plainly listed all eight spellings. `cargo build --release` fixed it.
  Rebuild before reporting a tool bug.
- **AXIS refuses `sext` under a vector type**, correctly — its from-width is a
  property of the whole value, not of a lane. Per-lane widening is a shift pair.
  The refusal is the tool doing its job, not an obstacle.
- **Vector literals are whole-register constants.** See obligation 1 above. Any
  immediate in a lane-wise expression must be splatted, and the only reliable
  way to know it was is to prove against an independent reference.

## Reproducing

```
cd subprojects/ise-capstone/axis
python3 verify.py            # regenerate semantics, re-solve all obligations
python3 verify.py --hashes   # current hashes, for updating MANIFEST.json
```

Exit `0` all reproduced, `1` something failed, `2` inconclusive — a missing
tool is reported as inconclusive and never as success, because "we could not
check" and "it checks out" must not print the same way.
