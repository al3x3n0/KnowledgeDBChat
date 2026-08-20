# candidate-coster

Renders a mined instruction pattern as the assembly `llvm-mca` has to cost.

Benefit is frequency times per-occurrence saving. `find_fusion_candidates`
measures the frequency; this produces the input for the other half.

```
candidate-coster emit --pattern "fsqrt fdiv | 0>1" --fused-as fsqrt \
    --mode dependent --copies 10 --out-dir /tmp/cand
```

The `--pattern` spelling is the miner's own, so a candidate can be passed
straight from one tool to the other. `0>1` means the first instruction's result
feeds the second.

Measured end to end on the reciprocal-square-root candidate, costed with
`llvm-mca -mcpu=neoverse-n1`:

| | cycles/iteration | instructions |
|---|---|---|
| baseline, `fsqrt` + `fdiv` | 340.03 | 20 |
| fused stand-in, `fsqrt` | 170.03 | 10 |

which is 17 cycles per occurrence, against a measured 65,536 executions of the
block it was mined from.

## What it assumes, and why that matters

**The fused instruction does not exist, so its cost is a stand-in.** `llvm-mca`
can only cost instructions its scheduling model knows. `--fused-as` names a
real instruction to stand for the proposed one, which makes the result "this
sequence costs as much as N of those". That is an assumption about the
instruction being proposed, not a measurement of it, and the tool says so in
every report rather than letting the number travel alone.

**Dependent and independent copies answer different questions.** Chained copies
measure the dependence chain, which is what a loop-carried computation runs
into; independent copies measure how many the machine overlaps, which is what
an unrolled loop runs into. They disagree by a lot. `--mode` is explicit
because picking the wrong one silently answers the wrong question — the same
mistake as fencing the wrong loop, which cost this project a 3.4x error once
already.

**Fewer instructions is not fewer cycles.** Folding pointer bumps into
post-indexed addressing in a saxpy loop here removed 300 instructions and saved
zero cycles; IPC just fell. A candidate that shortens the listing is not
thereby cheaper, which is the whole reason this emits assembly for a model to
cost rather than counting instructions itself.

## Region markers

The region is fenced with `# LLVM-MCA-BEGIN` / `# LLVM-MCA-END` and the file
ends with a newline. Both are required and neither failure is legible: an
unfenced estimate silently covers the whole file, and without the trailing
newline `llvm-mca` reads the final directive short — `# LLVM-MCA-END loop`
became region `loo` — then complains about region markers without naming a
cause. Four tool calls went into finding that the first time.

## Building

No dependencies, for the same reason as `idiom-miner`: this ships inside
reproducible evidence bundles, and a build that reaches the network for crates
is a build a third party cannot repeat years later.

```
cargo build --release
cargo test
```

## Not yet agent-callable

`analyze_snippet_cycles` already costs assembly from a job, but this binary is
not in a sandbox image, so an agent cannot generate the snippets itself and
still hand-writes them — which is exactly the step that produced a 40% error
once. Putting it in `deploy/sandbox-images/compiler-research` is the remaining
wiring.
