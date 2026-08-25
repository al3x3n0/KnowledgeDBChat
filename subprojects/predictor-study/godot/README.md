# The first real corpus, and what it measured

`trace.json` is 60 counters over 401 intervals from **Godot 4.4.1's own
`core/math`** — `Vector3::normalized()` and `AABB::expand_to()`, both
`_FORCE_INLINE_`, so the code under simulation is the engine's, not a
reimplementation of it. `header-closure.txt` is the 22-header closure that
compiles, computed with `g++ -MM`.

Everything before this ran on workloads written for the study, which is a study
about the study. This is not that.

## Two things the earlier attempt did not reach

`g++` was the known blocker. Two more sat behind it:

**`platform_config.h` does not exist in the source tree** — SCons generates it
per platform. But `platform/linuxbsd/platform_config.h` ships in the repo, so
adding that directory to the include path is the whole fix.

**`AABB::get_volume()` is not inline.** It lives in `aabb.cpp`. The math under
test is header-only; convenience accessors around it are not, and calling one
silently turns a header-only corpus into a build of Godot.

## The measurement, and why it is a negative result

Object count is constant every frame, so the instruction count does not vary by
construction. Only the geometry moves, along a trajectory, the way a scene
passes a camera. Anything the counters see therefore comes from the machine's
response to real data — branch outcomes inside `expand_to`, the
reciprocal-sqrt path in `normalized` — and not from a pattern planted in a loop
bound. S1's second workload was planted that way and persistence tracked it
trivially.

The first reading looked like a find:

```
entropy 1.563   persistence 1.387   best beyond 0.146   null 0.035   survives, 4.2x
top counter: branchPred.squashes_0::DirectCond
```

Branch mispredictions on direct conditional branches — mechanistically exactly
what `AABB::expand_to` should produce, six data-dependent comparisons per call.
It is the counter an architect would have named, which is the same weak check
that made the SMT result credible.

**It is an artifact.** Cycles per interval run 18,445 to 18,646 from the 5th to
the 95th percentile: a relative spread of **1.08%**. The workload is flat.
Three quantile bins split that 200-cycle jitter band into three "levels", and
the estimator then found real structure in what the quantiser invented. A
predictor getting every interval right would be predicting a 1% difference in
cycles, which changes no design.

The tell was in the design evaluation, not the information one: predicting the
same bin as last interval scored **0%** correct, because the labels alternate
0,1,0,1 — which is what binning a flat signal produces.

So: **Godot's inline vector math has nothing worth predicting at this
granularity.** That is a useful answer to the customer's question, and it is
the third time in this study a number looked real and was not — after a trace
too short to estimate on, and a trace that was two experiments spliced.

`agent_predictability` now reports the target's relative spread with every
estimate and says this in the verdict.

## Re-deriving it

No simulator needed; the trace is the input.

```python
import json
from app.services import agent_predictability as pred

series = json.load(open("trace.json"))
r = pred.ceiling(series, "system.cpu.numCycles")
r["target_relative_spread"]   # 0.0108
print(r["verdict"])
```
