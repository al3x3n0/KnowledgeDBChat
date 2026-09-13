# idiom-miner

Counts code idioms across a source tree, with provenance. This is the
**frequency** half of deciding whether an instruction-set extension is worth
proposing; the **per-occurrence saving** half is measured separately, against a
core model (`analyze_snippet_cycles`, which wraps `llvm-mca`).

```
cargo build --release
./target/release/idiom-miner \
    --root /path/to/checkout \
    --patterns patterns/aarch64-candidates.txt \
    --commit $(git -C /path/to/checkout rev-parse HEAD) \
    --samples 3 > frequency.json
```

## What a count means, and what it does not

It reports **static occurrence**: how often a token sequence appears in the
source. That is not how often it executes. A pattern occurring ten thousand
times in cold code is worth less than one occurring twice in an inner loop, and
nothing here can tell those apart. The JSON says so in `corpus.counts`, and any
report built on it must repeat the distinction rather than quietly promote one
to the other.

Ranking candidates by frequency alone produces confident nonsense. Measured on
2026-08-14: folding pointer bumps into post-indexed `ldp`/`stp` in a saxpy loop
removed 300 instructions and saved **zero** cycles, because the loop was not
instruction-bound. Frequency says where to look; the core model says whether
there is anything there.

## Patterns

`name: pattern`, one per line, `#` for comments. A pattern is tokens separated
by spaces: `$id` is any identifier, `$num` any numeric literal, `$any` any
single token, and anything else must match a token exactly.

```
indexed_product: $id [ $id ] * $id [ $id ]      # matches a[i] * b[i]
rsqrt_idiom: 1.0f / sqrtf ( $any
```

Matching is over tokens rather than raw text, so formatting does not matter and
matches inside comments and string literals do not count. It is not a C++
parser: no macro expansion, no template instantiation, no build-configuration
awareness. A match means "this token sequence occurs in the source".

## Properties it is built to have

- **Deterministic.** Same tree, same output, byte for byte, independent of
  thread count — verified in both directions. Bundles that cannot be re-run to
  the same bytes are not evidence.
- **No dependencies.** Nothing is fetched at build time, so the binary can be
  rebuilt from a bundle years later without a working crates registry.
- **Loud about what it skipped.** `files_skipped` reports files that were too
  large, unreadable, or not UTF-8. Silent skipping reads as absence of matches.

## Scale

2500 files / 20 MB / 1.8M tokens of C and C++ headers in 228 ms on 8 cores
(~89 MB/s). An Unreal-sized tree is seconds, not hours.
