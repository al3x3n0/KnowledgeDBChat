# Two more candidates, formalized and proven

The same path as [SMLALB](smlalb-proposal.md), walked twice more: mined from a
profile, described in AXIS, proven against the sequence it replaces, and
bundled so the claims re-establish themselves from source.

```
cd subprojects/ise-capstone/axis
python3 verify.py            # all three bundles, nine obligations
python3 verify.py fselgt     # just one
```

| Bundle | Candidate | Corpus | Occurrences | Bounded saving |
|---|---|---|---|---|
| `smlalb` | `sxtl`+`smlal` | PCE INT8 attention | 1,572,864 | 5.0 cyc |
| `fcvtmlab` | `sxtl`+`scvtf`+`fmla` | PCE INT8 attention | 786,432 | 13.0 cyc |
| `fselgt` | `fcmgt`+`bit` | Godot 4.4.1 `core/math` | 98,280 | 8.0 cyc |

## FSELGT — branchless select, and why NaN is the whole question

Godot's hottest fusable shape is compare-then-bitwise-insert: `fcmgt` builds a
lane mask of all-ones or all-zeros from a float compare, `bit` inserts under
that mask. It is how `MIN`/`MAX`/`CLAMP` compile without a branch, and it comes
from `AABB::expand_to` and `Vector3`.

The fused form is proven equal to the pair for every input, NaN included
(`unsat`).

The interesting result is the negative control. `FSELGT_NAN_BUG` is what a
reasonable engineer reaches for first: instead of testing `fm < fn1`, logically
negate `fn1 <= fm`. On ordinary numbers the two agree exactly. z3 returns:

```
fm  = 0xfffbca24  = NaN
fn1 = 0xf8ad01db  = -2.807e+34
fd  = 0x5117466f  =  4.061e+10
```

Every ordered compare is false when an operand is NaN, so negating one yields
true and the wrong operand is selected. Both forms cost the same and no
benchmark over real geometry would plausibly generate the NaN that separates
them. **Nothing except a proof distinguishes these two implementations** — which
is the clearest argument in this project for why the formal step earns its
place alongside the cycle counts.

### The control I nearly got wrong

The first version of that control used `(not i32 (fcmp_ole f32 fn1 fm))`. It
returned `sat`, as intended — but decoding the witness showed `fn1=0.0`,
`fm=1.0`, `fd=2.0`, with **no NaN anywhere**. `not` in AXIS is bitwise, so
`(not i32 1)` is `0xFFFFFFFE`, which `select` reads as true: the "negated"
predicate was true whenever the original was, and the control was failing for a
mundane reason that had nothing to do with the claim.

Had I not decoded the counterexample, the write-up would have asserted a NaN
result that the proof never produced. A `sat` verdict is not evidence for
whatever story you had in mind; only the witness is.

## FCVTMLAB — and where to state an obligation

INT8 attention dequantizes: load a signed byte, widen, convert to float,
multiply-accumulate. Three instructions to get one byte into an FP MAC.

All three forms end in the same `fma` over the same `fm` and `fd`, and differ
only in how the float operand is produced. Since `fma` is a function of its
arguments, proving the operands equal settles the whole instruction by
substitution — so the obligation is stated at the conversion.

That is not only tidier, it is the difference between a proof and no proof:

| Obligation | z3 verdict | Time |
|---|---|---|
| Whole instruction, three symbolic operands under `fma` | `timeout` | 100 s (limit) |
| Conversion only | `unsat` | < 1 s |

The undecomposed form is over an IEEE-754 `fma` with everything free, and z3
does not decide it in any time I was willing to spend. **Where a proof reduces
to the part that actually differs, reduce it** — and say in the bundle that you
did, which `MANIFEST.json` records under `proof_structure`.

The control catches unsigned conversion, with witness `0xfa`: −6 signed, 250
unsigned.

## What these two do and do not establish

Same shape of limits as SMLALB, plus one that is new and matters more here.

**Both are modelled one lane at a time, and the vector form is not proven.**
AXIS's lane-parallel float set is `fadd`/`fsub`/`fmul`/`fdiv`/`fsqrt` only.
`fcvt_s2f` and `fma` are not in it, and float *compares* are excluded
deliberately — a NEON lane compare writes a mask, not the 1/0 a scalar `fcmp`
returns, so spelling one as a lane-wise compare would quietly mean the wrong
thing. Both candidates are therefore inexpressible in vector form today, and
nothing here claims the 4-lane instruction is proven. SMLALB, being integer, is.

That is a coverage boundary of the formalism rather than a defect of the
candidates, and it is worth knowing before promising a proof for an arbitrary
mined pattern: **the two highest-value candidates by bounded saving both fall
outside the lane-wise proof ladder.**

Unchanged from SMLALB and still true of both: the savings are upper bounds and
not predictions; neoverse-n1 is not the target hardware; the latency, area and
power fields are placeholders nothing derives from; there is no denominator for
a whole-program speedup; and each rests on one workload shape.

## Traps added to the list

- **A `sat` verdict is not evidence for the story you expected.** Decode the
  witness. See the bitwise-`not` control above.
- **`not` is bitwise, not logical.** Negating a 0/1 predicate needs
  `(icmp_eq i32 p 0)`.
- **State the obligation where the difference is.** An FP obligation that will
  not terminate whole may be instant decomposed, and the decomposition is often
  a sounder statement of the claim anyway.
