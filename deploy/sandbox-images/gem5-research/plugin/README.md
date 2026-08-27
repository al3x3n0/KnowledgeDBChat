# Loading a gem5 mechanism at run time

gem5 has no plugin interface. A mechanism is compiled in, and on this machine
adding one costs **7m49s** of rebuild — which cannot be done while the
application stack is running, because the final link needs more memory than
the Docker VM has left (`collect2: fatal error: ld terminated with signal 9`).
That is a poor loop for research whose whole point is trying mechanisms.

These four files add one SimObject to gem5, once, and turn every later
mechanism into a `g++ -shared`. Measured: **941 ms**, no link step, and
therefore no memory wall — mechanisms can be built while the stack runs.

Two extension points are covered. Each is one SimObject compiled into gem5
once, and one header a plugin includes.

| file | what it is |
|---|---|
| `gem5_rp_plugin_abi.h` | the replacement-policy contract. Mentions no gem5 type, so a plugin needs this header and nothing else |
| `plugin_rp.hh` / `.cc` | `PluginRP`, compiled in once. dlopens a library and forwards the five virtual calls |
| `reuse_protected_plugin.cc` | worked example: reuse-protected LRU, ~110 lines |
| `apply.py` | registers `PluginRP` in `ReplacementPolicies.py` and `SConscript` |
| `gem5_pf_plugin_abi.h` | the prefetcher contract |
| `plugin_prefetcher.hh` / `.cc` | `PluginPrefetcher`, deriving from `Queued` |
| `tagged_plugin.cc` | worked example: next-N-lines, a deliberate reimplementation of gem5's own `TaggedPrefetcher` |
| `apply_prefetcher.py` | registers `PluginPrefetcher` in `Prefetcher.py` and `SConscript` |

**The prefetcher shim derives from `Queued`, not from `Base`.** The queue, the
throttling and the page-crossing checks stay gem5's, and the plugin is asked
only the question that is actually the algorithm: given this access, which
addresses would you fetch? A prefetcher that reimplements the queue measures
the harness — at gem5's default L1 `mshrs=4` a stride prefetcher issues 35 of
503,959 identified candidates and reads as no prefetcher at all, which is a
property of the queue and of no algorithm.

## Why it works

`gem5.opt` is a PIE exporting **110,182 dynamic symbols**, so a shared object
loaded into it resolves gem5's symbols at load time — verified by `dlopen`ing
a policy built against gem5's own headers, which resolved all 15 of the gem5
symbols it referenced.

The ABI deliberately does not rely on that. Crossing the boundary with gem5
types would tie every plugin to one build of gem5 and bring the rebuild back
by another route; crossing it with plain structs means a plugin compiles in a
second, in an image carrying no gem5 source, and keeps working across gem5
upgrades until the ABI version changes.

## Applying it

```sh
docker cp plugin/gem5_rp_plugin_abi.h  <ctr>:/src/gem5/src/mem/cache/replacement_policies/
docker cp plugin/plugin_rp.hh          <ctr>:/src/gem5/src/mem/cache/replacement_policies/
docker cp plugin/plugin_rp.cc          <ctr>:/src/gem5/src/mem/cache/replacement_policies/
docker cp plugin/apply.py              <ctr>:/tmp/ && docker exec <ctr> python3 /tmp/apply.py
docker exec <ctr> sh -c 'cd /src/gem5 && scons build/ARM/gem5.opt -j2 --ignore-style --linker=lld'
```

Stop the application stack first. The link needs the memory.

## Writing a plugin

```sh
g++ -shared -fPIC -O2 -std=c++17 -I <dir with the abi header> -o mine.so mine.cc
```

and select it:

```json
{"caches": {"l2": {"replacement_policy": {"class": "PluginRP", "params": {
    "library": "/work/mine.so", "config": "protect_after_touches=4"}}}}}
```

## What is checked, and what is not

`PluginRP` refuses a missing library, a missing entry symbol, an ABI version
mismatch, a null hook in the table, and a victim index outside the candidate
set — the last because evicting something arbitrary would turn a plugin bug
into a silently wrong measurement rather than an error.

It does not sandbox the plugin. The plugin runs in gem5's address space and a
bad one can corrupt or crash the simulation. That is acceptable where gem5
already runs inside the sandbox container; it is not a boundary to trust with
anything else.

**The null control to run after any change here:** a plugin must reproduce
the mechanism it mirrors, byte for byte.

- *Replacement policy* — the same algorithm compiled in and loaded gave
  11,081,822 cycles, 301,786 L2 misses and 56 hits both ways, against LRU's
  11,082,471 / 301,803 / 39. The policy acts; the boundary changes nothing.
- *Prefetcher* — `tagged_plugin.cc` against gem5's own `TaggedPrefetcher` at
  degree 4: **byte-identical statistics**, 856,605 cycles and 128,707 issued
  both ways, against 1,460,876 with no prefetcher. Worth 1.7054x, and the
  plugin earns exactly what the shipped one earns.

Writing a novel algorithm first leaves nothing to check the boundary against
but an opinion. Mirror something shipped, prove they agree, then innovate.
