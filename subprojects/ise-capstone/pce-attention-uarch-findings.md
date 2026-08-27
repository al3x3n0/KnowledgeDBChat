# What limits PCE INT8 attention, and what could be done about it

The earlier study of this kernel asked which instruction sequences recur on
its hot path. This one asks a different question of the same code: what is the
core actually waiting on, and how much would any mechanism aimed at that be
worth. Both use `xnn_transformer_mha_i8` and the `xnn_ai_softmax_f32` it
calls, sliced out of `xnn-sdk` by brace matching rather than retyped; the four
`xnn_overlay_*` PCE hardware hooks are stubbed. Shape is sequence 64 / dim 64
/ 4 heads, the same as before, so the two studies are about one workload.

Model: gem5 25.1 generic O3CPU, 64 KiB L1D, 2 MiB L2, 2 GHz. Every number is a
ratio inside that model. It is not this silicon and not PCE: measured against
an Apple M3 this model is about 40% off per instruction, so nothing here is a
claim about hardware.

## The kernel is load-queue limited

12,867,523 cycles, 16,350,365 instructions, IPC 1.27.

Structure-full events, counted at rename:

| structure | events |
|---|---|
| load queue | 9,342,457 |
| issue queue | 351,136 |
| store queue | 47,124 |
| reorder buffer | 9,145 |

The load queue by a factor of twenty-six. Decode is blocked 75.1% of cycles,
rename 0.58% -- the stall is recorded upstream of where it originates, which
is what made the first ranking of this kernel wrong.

## How much is winnable

Each row idealises one structure and reports what that recovers. An idealised
structure is not buildable; these are ceilings.

| idealised | recovered | next limit |
|---|---|---|
| load queue (128 entries) | **4.68%** (602,034 cyc) | issue queue |
| branch prediction (TAGE_SC_L_64KB) | 0.40% | load queue |
| store queue | 0.23% | load queue |
| pipeline width (16-wide) | 0.08% | load queue |
| issue queue (512 entries) | 0.02% | load queue |
| L1D capacity | 0.00% | load queue |
| L1I capacity | 0.00% | load queue |
| physical registers | 0.00% | load queue |

The load queue is worth more than everything else combined, and removing it
moves the limit to the issue queue rather than to memory.

## No shipped mechanism captures it

| mechanism | speedup |
|---|---|
| L2 StridePrefetcher | 1.0037x |
| L1D StridePrefetcher | 1.0036x |

Both fired -- they passed the activation gate -- and both delivered 0.37%,
which is what an L1D capacity headroom of 0.00% predicts. This kernel is not
memory-capacity limited, so prefetching cannot pay however it is configured.

That is the result worth carrying forward. gem5's shipped mechanism classes
are prefetchers, replacement policies and branch predictors; the limit here is
a load queue, and none of them addresses it. Capturing any of the 4.68% needs
a microarchitectural change written rather than composed.

## What this is not

- **4.68% is a ceiling, not a proposal.** A 128-entry load queue is not free
  and may not be buildable at this frequency; the number says only that no
  mechanism aimed elsewhere can beat it.
- **One shape.** Sequence 64 / dim 64 / 4 heads. Longer sequences weight the
  QK loop differently and the ranking may move.
- **One core model.** A generic O3CPU, not PCE and not any shipped silicon.
- **The load queue is where the stall lands, not necessarily its root cause.**
  Attribution ranked this kernel wrong once already, in the other direction.
