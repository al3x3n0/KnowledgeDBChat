# Fusion candidates in the PCE INT8 attention kernel

First end-to-end run of the pipeline on code nobody wrote for the demo:
profile a real workload, mine its hot blocks for shapes one instruction could
encode, cost each against a named core, and rank by bounded benefit.

## Corpus

`KevinAI/pce/software`, INT8 multi-head attention. The kernels are used
verbatim — `xnn_transformer_mha_i8` from `xnn-sdk/src/xnn_transformer.c` and
the `xnn_ai_softmax_f32` it calls — extracted programmatically rather than
retyped, so what was profiled is the code that really runs. The four
`xnn_overlay_*` calls are PCE hardware hooks with no meaning on aarch64 and are
stubbed; they sit inside the attention loop, so a profile on the real target
would carry their cost and this one does not.

Workload: sequence 64, head dim 64, 4 heads, 3 repetitions.

## What the profile found

`xnn_transformer_mha_i8` accounts for **86.5%** of instructions, `expf` 6.1%.
Five hot blocks, the largest two running 393,216 times each.

## Candidates, ranked by bounded benefit

Costed with `llvm-mca -mcpu=neoverse-n1`, dependent chains:

| candidate | costs now | floor | best saving | occurrences | up to |
|---|---|---|---|---|---|
| `sxtl` → `scvtf` → `fmla` | 23.1 | `scvtf` 10.1 | 13.0 | 786,432 | 10.2M cycles |
| `sxtl` → `smlal` | 8.1 | `sxtl` 3.1 | 5.0 | 1,572,864 | 7.9M cycles |
| `sxtl` → `sxtl` → `scvtf` | 16.1 | `scvtf` 10.1 | 6.0 | 786,432 | 4.7M cycles |
| `sxtl` → `sxtl` → `smlal` | 8.1 | `sxtl` 3.1 | 5.0 | 786,432 | 3.9M cycles |

Cycles are per occurrence. The saving is a range whose floor is the slowest
operation the fused form still has to perform: it cannot beat that, and it must
beat the sequence to be worth building. No stand-in instruction is invented,
because the answer would then depend on which one was picked.

## The result worth noticing

`sxtl` → `smlal` is sign-extend followed by widening multiply-accumulate, which
is what the byte-wise dot product in the QK loop compiles to. That is the
pattern **Arm added `SDOT` for in v8.4-A**. The miner reached it from a profile
of this kernel, without being told what to look for, which is the first
evidence that the candidate-finding step recovers extensions that turned out to
be worth building in reality.

The top-ranked candidate is the int8-to-float weighted accumulate — extend,
convert, multiply-add — which is the `V` half of attention rather than the `QK`
half, and has no single instruction in AArch64 today.

## What these numbers are not

- **They are not the PCE.** The costs are llvm-mca's model of Neoverse N1,
  because that is a core model available here. A claim about the PCE needs the
  PCE's own model, and the gap between two cores has already been measured on
  this project at 40–77% per instruction.
- **They are upper bounds**, not predictions. The real saving depends on what
  the fused instruction would actually cost, which nothing here can measure.
- **One workload shape.** Sequence 64 / dim 64 / 4 heads. The ranking may move
  with shape; longer sequences weight the QK loop more heavily.
- **No denominator.** These are cycles saved, not a speedup: the profile counts
  instructions rather than cycles, so what fraction of runtime they represent
  is not established here.

## Reproducing

The workload is assembled by `tools/` from the kernel sources; the chain is
`profile_c_workload` → `find_fusion_candidates` → `cost_fusion_candidate`, all
available to an agent. `candidate-coster` renders any of these patterns back to
assembly for independent checking.
