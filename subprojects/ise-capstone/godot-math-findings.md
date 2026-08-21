# Fusion candidates in Godot's vector math

Second corpus for the pipeline, and the first that is a shipping game engine.

## Corpus

Godot 4.4.1, `core/math`. `Vector3`, `Basis`, `Transform3D` and `AABB` are used
verbatim from the engine — the driver around them transforms a batch of
vertices and grows a bounding box, which is the loop an engine spends its time
in: skinning, culling, particle updates.

Their inline methods link with no engine core behind them (25 headers). Two
calls are deliberately avoided: `Basis(axis, angle)` and `AABB::get_volume()`
are out-of-line in `.cpp` and would drag the string and error machinery into
the link. Neither is in the loop being measured.

Workload: 4096 vertices × 24 frames.

## What the profile found

**82.5%** of instructions in the transform loop. The remainder is mostly the
dynamic linker (6.6% `_dl_lookup_symbol_x`, 4.3% `do_lookup_x`), which is
visible only because the run is short.

## Candidates, ranked by bounded benefit

`llvm-mca -mcpu=neoverse-n1`, dependent chains:

| candidate | costs now | floor | best saving | occurrences | up to |
|---|---|---|---|---|---|
| `fcmgt` → `bit` → `fsub` | 13.1 | `fcmgt` 5.1 | 8.0 | 98,280 | 786,240 |
| `fmov` → `fmul` | 8.1 | `fmul` 5.1 | 3.0 | 196,632 | 589,896 |
| `bit` → `fsub` | 8.1 | `fsub` 5.1 | 3.0 | 98,280 | 294,840 |
| `fcmgt` → `bif` | 8.1 | `fcmgt` 5.1 | 3.0 | 98,280 | 294,840 |

## What the top candidate is

`fcmgt` → `bit` is a lane-wise compare feeding a bitwise insert: branchless
select, which is how min, max and clamp are written for vectors. It comes from
`AABB::expand_to` and the componentwise min/max in `Vector3`. The `fsub` that
follows is the extent calculation. A single "clamp" or "select-and-subtract"
instruction is what this pattern is asking for, and at 98,280 occurrences in a
24-frame run it is not a rare path.

That the compare-select pair is the hottest fusable shape in an engine's
geometry code is a more interesting result than the arithmetic ones: it is
about control flow expressed as data, which is where SIMD ISAs spend their
encoding space.

## What these numbers are not

- **Neoverse N1, not any shipping console or phone core.** Cycle counts are a
  property of a specific model, and this project has measured 40–77% error
  between core models on the same code.
- **Upper bounds.** The floor is the slowest operation the fused form must
  still perform; the real saving depends on what the fused instruction would
  cost, which nothing here can measure.
- **The mined shape does not record vector width.** The engine's code uses
  `.2s` here and the costing renders `.4s`, so the per-occurrence cycles are
  the right shape at a possibly different width.
- **One workload.** A batch vertex transform, not a running game. A real frame
  mixes physics, scripting and rendering, and the ranking would move.
- **No denominator.** Cycles saved, not a speedup: the profile counts
  instructions, so what fraction of frame time this represents is not
  established here.

## Two bugs this corpus found

Header-only C++ math exercises the profiler in ways C does not, and it found
two defects that silently produced wrong hot blocks — both fixed, both with the
evidence in `callgrind_profile.py`:

- Instruction costs from every binary object went into one address map, though
  a PIE executable and each shared library all start near zero.
- `fi=`/`fe=` records mark entering and leaving an inlined header **inside** a
  function, and the parser treated them as a reason to forget the current
  instruction address. Every following relative position resolved to nothing:
  of 8,336,032 instructions, only 447k kept an address and the hot loop was not
  among them. After the fix the program's own object accounts for 6,877,132 —
  82.5%, matching what the function-level profile independently reported.
