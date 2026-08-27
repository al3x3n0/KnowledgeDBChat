# A mechanism that gem5 does not ship, written and measured

First use of the plugin path: a replacement policy invented rather than
chosen, compiled to a shared object and loaded into gem5 at run time. It does
not work, and the reason it does not work is the interesting part.

## The mechanism

Reuse-protected LRU. A line touched `protect_after_touches` times becomes
protected and is evicted only when the set holds no unprotected line;
insertion resets the count, so a line that was protected before eviction does
not arrive protected. ~110 lines against `gem5_rp_plugin_abi.h`.

## The workload, and why this one

A 3 MiB array swept repeatedly, one access per 64-byte line, against a 2 MiB
L2 — a working set 1.5x the cache, which is the textbook pathological case for
LRU: the line about to be needed is always the one just evicted.

**Chosen because it separates two policies gem5 already ships.** Two earlier
workloads did not, and returned LRURP, BRRIPRP and the new policy with
bit-identical L2 counts. A replacement-policy study needs a workload proven to
distinguish two shipped policies before a new one is added to it, or the
result is about the workload.

| policy | cycles | vs LRU | L2 misses | L2 hits |
|---|---|---|---|---|
| LRURP | 10,786,145 | 1.0000 | 295,656 | 15 |
| BRRIPRP | 10,086,750 | **1.0693** | 274,963 | 20,708 |
| ReuseProtectedRP (compiled in) | 10,786,283 | 0.9999 | 295,656 | 15 |
| PluginRP → the same policy, loaded | 10,786,283 | 0.9999 | 295,656 | 15 |

BRRIP separates from LRU by 6.9%, so the workload discriminates. The new
policy does not separate from LRU at all — same misses, same 15 hits.

## Why it fails, which is worth more than the number

**The pathology denies the policy the observation it learns from.** Protection
is earned by a second touch while resident. On a cyclic scan over a working
set larger than the cache, no line is ever touched twice while resident: every
access misses, `reset` puts the count back to 1, and nothing ever reaches the
threshold. The policy degenerates to exactly LRU, which is what the 15 hits
say.

BRRIP wins here precisely because it does not learn. It assumes distant
re-reference on insertion and keeps a fraction of the working set by default,
so it never needs to observe the reuse that LRU's own behaviour prevents.

That generalises: **a reuse-learning replacement policy cannot repair a
thrashing pattern, because thrashing is the absence of observable reuse.** Such
a policy is for workloads with a hot subset that survives long enough to be
touched twice — which is a different study, and needs a workload that
separates on that axis rather than this one.

## What this is not

- **Not a verdict on the mechanism.** One workload, chosen to be hostile to
  exactly this design. It says where the mechanism cannot help.
- **Not a claim about silicon.** gem5's generic O3CPU, 2 MiB L2, 8-way.
- The 178-cycle difference from LRU (0.002%) is the policy's own bookkeeping,
  not a benefit.

## The control that matters

The same algorithm compiled into gem5 and loaded as a plugin produced the same
cycles and the same L2 counts. The plugin boundary changes nothing; what
changes is that the mechanism cost 941 ms to build instead of 7m49s.
