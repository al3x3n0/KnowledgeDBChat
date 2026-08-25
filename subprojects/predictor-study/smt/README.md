# The SMT trace, kept

`trace.json` is the counter trace behind every SMT number in `../S1-findings.md`:
52 counters that vary, 401 intervals, gem5 O3CPU with two threads.
`workloads.py` is the primary and co-runner that produced it, and the driver
that ran them.

**Why this is here at all.** The three non-SMT rows in the findings cannot be
re-checked against the regime defect, because neither their traces nor their
workloads survive — that study ran from standalone scripts, so the
reproducibility bundles built for exactly this purpose captured none of it.
The SMT results could be corrected only because this one file happened to be
kept by hand. Keeping it by hand is not a system, so it is kept here instead.

Re-deriving the corrected table needs no simulator — the trace is the input:

Run it from `backend/` with that on the path — the driver and this snippet
both import the services directly.

```python
import json

from app.services import agent_predictability as pred
from app.services import agent_gem5_sandbox as gem5
from app.services import agent_trace_regime as regime

series = json.load(open("trace.json"))
break_at = gem5.find_regime_change(series["system.cpu.numCycles"])["at_interval"]
steady, _ = regime.window({"series": series, "regime_change": {"at_interval": break_at}}, break_at)
pred.ceiling(steady, "derived.thread0_ipc")["persistence_information_bits"]
```

`break_at` is 104. Measuring from 0 instead reproduces the first, wrong reading
— persistence 0.843 rather than 0.405 — which is the point of keeping both.
