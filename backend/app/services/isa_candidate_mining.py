"""Find instruction sequences worth fusing, from the code that actually ran.

Costing a proposed instruction has been possible here for a while; proposing
one has not. That step was done by a human reading disassembly, which does not
scale to an engine-sized tree and cannot be defended -- "these three
instructions look fusable" is an opinion until something counts how often the
pair really occurs and on which path.

This mines candidates from hot blocks: build the data-flow graph of a basic
block, enumerate the connected subgraphs a real instruction slot could
encode, and count how often each shape occurs weighted by how often its block
executed. Static frequency is not dynamic hotness -- a pattern appearing 400
times in cold setup code is worth nothing -- so occurrences carry the block's
measured execution count and rank by that.

What it deliberately does not do is decide anything. A candidate here is a
claim that a shape is frequent, not that fusing it pays: whether it pays needs
llvm-mca on the sequence and its replacement, and the two traps already
recorded in this project (instruction count is not cycles; scope changes the
number by 3.4x) both apply.

The constraints that matter are architectural. A fused instruction has to
encode its operands, so a subgraph is only a candidate if it reads few enough
external registers and writes few enough results -- two in and one out is the
usual budget for a 32-bit encoding. It must also be *convex*: if a value
leaves the group and comes back, the group cannot execute as one instruction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple

# Instructions whose effect is not captured by register data flow alone.
# Excluded from candidates rather than modelled: fusing across a branch or a
# barrier is not a question this can answer.
CONTROL_FLOW = frozenset(
    {
        "b",
        "bl",
        "blr",
        "br",
        "ret",
        "cbz",
        "cbnz",
        "tbz",
        "tbnz",
        "svc",
        "hvc",
        "brk",
        "hlt",
        "dmb",
        "dsb",
        "isb",
        "yield",
    }
)
CONDITIONAL_BRANCH = re.compile(r"^b\.[a-z]+$")

# Instructions that write memory: their "result" is not a register, so a naive
# def/use reading makes them look like free-standing leaves that can be pulled
# into any group. Their ordering against loads is not modelled here.
STORES = frozenset(
    {"str", "strb", "strh", "stur", "sturb", "sturh", "stp", "stlr", "stxr"}
)
LOADS = frozenset(
    {"ldr", "ldrb", "ldrh", "ldrsw", "ldrsb", "ldrsh", "ldur", "ldp", "ldar", "ldxr"}
)

# Pair forms name two registers before the address, so the usual "first
# operand is the destination" rule credits only one of them.
PAIR_FORMS = frozenset({"ldp", "stp", "ldnp", "stnp"})

# Registers that are not data flow between instructions in the ordinary sense.
IGNORED_OPERANDS = frozenset({"sp", "wsp", "xzr", "wzr", "pc", "lr"})

# Instructions whose first operand is read, not written. Tested by name rather
# than by prefix: a prefix rule for "cmp" misses `fcmp`, which then looked like
# it defined its first register and invented a data-flow edge that is not
# there. A candidate built on a false edge is a shape the code never had.
NO_RESULT = frozenset(
    {"cmp", "cmn", "tst", "fcmp", "fcmpe", "ccmp", "ccmn", "fccmp", "fccmpe"}
)

_REGISTER = re.compile(
    r"""^(?:
        (?P<gp>[wx](?P<gpn>\d{1,2}))            |   # w3 / x3
        (?P<fp>[bhsdq](?P<fpn>\d{1,2}))         |   # s0 / d0 / q0
        (?P<vec>v(?P<vecn>\d{1,2}))                 # v0.4s
    )$""",
    re.VERBOSE,
)


def normalize_register(token: str) -> Optional[str]:
    """Map an operand to the physical register it touches, or None.

    AArch64 names one register many ways: w3 and x3 are the same 64-bit
    register seen at two widths, and s0, d0, q0 and v0 are all the same vector
    register. A data-flow graph that treats them as different registers simply
    loses edges -- silently, and in exactly the FP code this is aimed at.
    """
    text = str(token or "").strip().lower().rstrip(",")
    text = text.split(".")[0]  # v0.4s -> v0
    text = text.strip("[]!")
    if not text or text in IGNORED_OPERANDS:
        return None
    match = _REGISTER.match(text)
    if not match:
        return None
    if match.group("gp"):
        return f"x{int(match.group('gpn'))}"
    if match.group("fp"):
        return f"v{int(match.group('fpn'))}"
    return f"v{int(match.group('vecn'))}"


@dataclass(frozen=True)
class Instruction:
    """One decoded instruction: what it is, what it reads, what it writes."""

    index: int
    mnemonic: str
    defs: Tuple[str, ...]
    uses: Tuple[str, ...]
    text: str
    is_memory: bool = False


def parse_instruction(line: str, index: int = 0) -> Optional[Instruction]:
    """Decode one line of AArch64 assembly into defs and uses.

    Returns None for anything that is not an instruction -- labels, directives,
    comments -- so a caller can feed it raw disassembly.
    """
    text = str(line or "").split("//")[0].split(";")[0].strip()
    if not text or text.startswith((".", "#", "@")) or text.endswith(":"):
        return None
    # Two layouts reach this, and telling them apart matters more than it
    # looks: objdump writes "4005a0:\t91000400 \tadd\tx0, x0, #1" while the
    # compiler writes "\tadd\tx0, x0, #1". Reading the last tab-separated
    # field works for the first and silently returns the *operands* for the
    # second, so every mnemonic came back as a register name.
    if "\t" in text:
        parts = [p for p in text.split("\t") if p.strip()]
        if len(parts) >= 3 and re.match(r"^[0-9a-f]+:$", parts[0].strip()):
            text = " ".join(parts[2:])
        else:
            text = " ".join(parts)
    text = text.strip()
    if not text:
        return None

    pieces = text.replace(",", " , ").split()
    mnemonic = pieces[0].lower()
    if not re.match(r"^[a-z][a-z0-9._]*$", mnemonic):
        return None

    operand_text = text[len(pieces[0]) :].strip()
    operands = [o.strip() for o in operand_text.split(",")] if operand_text else []

    is_store = mnemonic in STORES
    is_load = mnemonic in LOADS
    defs: List[str] = []
    uses: List[str] = []

    for position, operand in enumerate(operands):
        registers = [
            normalize_register(token)
            for token in re.findall(
                r"[a-z]?\d+(?:\.\w+)?|\b[a-z]{1,2}\d{1,2}\b", operand
            )
        ]
        registers = [r for r in registers if r]
        if not registers:
            continue
        # The first operand is the destination for everything except stores,
        # comparisons and branches -- and a store's first operand is the value
        # it reads, which is the case a def/use rule gets backwards.
        writes_first = not (
            is_store
            or mnemonic in CONTROL_FLOW
            or CONDITIONAL_BRANCH.match(mnemonic)
            or mnemonic in NO_RESULT
        )
        destination_slots = 2 if mnemonic in PAIR_FORMS else 1
        if position < destination_slots and writes_first:
            defs.append(registers[0])
            uses.extend(registers[1:])  # "ldr x0, [x0, #8]" reads x0 as well
        else:
            uses.extend(registers)

    # A pre/post-indexed memory operand writes its *base* register back, and
    # the base is the one inside the brackets. Taking the first use instead
    # credits the write to the value being stored: `stp x29, x30, [sp, #-32]!`
    # was reported as defining x29.
    # Pre-indexed writes back with "!", post-indexed with the offset outside
    # the brackets: `ldp x1, x2, [x8], #16` updates x8 and carries no "!".
    post_indexed = re.search(r"\]\s*,\s*#", text) is not None
    if (is_load or is_store) and ("!" in text or post_indexed):
        bracketed = re.search(r"\[([^\]]*)\]", text)
        base = None
        if bracketed:
            for token in re.findall(r"\b[a-z]{1,2}\d{1,2}\b", bracketed.group(1)):
                base = normalize_register(token)
                if base:
                    break
        if base and base not in defs:
            defs.append(base)

    return Instruction(
        index=index,
        mnemonic=mnemonic,
        defs=tuple(dict.fromkeys(defs)),
        uses=tuple(dict.fromkeys(uses)),
        text=text,
        is_memory=is_load or is_store,
    )


def parse_block(lines: Iterable[str]) -> List[Instruction]:
    """Decode a basic block, keeping only real instructions."""
    decoded: List[Instruction] = []
    for line in lines:
        instruction = parse_instruction(line, index=len(decoded))
        if instruction is not None:
            decoded.append(instruction)
    return decoded


@dataclass
class DataFlowGraph:
    """Instructions of one block, with edges where a value is passed."""

    instructions: List[Instruction]
    edges: Set[Tuple[int, int]] = field(default_factory=set)
    producers: Dict[int, Dict[str, int]] = field(default_factory=dict)

    def successors(self, index: int) -> Set[int]:
        return {b for a, b in self.edges if a == index}

    def predecessors(self, index: int) -> Set[int]:
        return {a for a, b in self.edges if b == index}


def build_dfg(instructions: Sequence[Instruction]) -> DataFlowGraph:
    """Link each use to the instruction that last wrote that register.

    Only within the block: a value defined elsewhere is an external input,
    which is what makes it count against the encoding budget later.
    """
    graph = DataFlowGraph(instructions=list(instructions))
    last_write: Dict[str, int] = {}
    for instruction in instructions:
        sources: Dict[str, int] = {}
        for register in instruction.uses:
            producer = last_write.get(register)
            if producer is not None:
                graph.edges.add((producer, instruction.index))
                sources[register] = producer
        graph.producers[instruction.index] = sources
        for register in instruction.defs:
            last_write[register] = instruction.index
    return graph


def _reachability(graph: DataFlowGraph) -> Dict[int, Set[int]]:
    """Everything each instruction can reach, following data flow forward."""
    descendants: Dict[int, Set[int]] = {}
    for instruction in reversed(graph.instructions):
        node = instruction.index
        reached: Set[int] = set()
        for successor in graph.successors(node):
            reached.add(successor)
            reached |= descendants.get(successor, set())
        descendants[node] = reached
    return descendants


def is_convex(nodes: FrozenSet[int], descendants: Dict[int, Set[int]]) -> bool:
    """True when no value leaves the group and comes back.

    A group that is not convex cannot execute as a single instruction: some
    intermediate result would have to be visible to an instruction outside it
    and then re-enter, which one opcode cannot express. This is the constraint
    people forget, and forgetting it produces candidates that look profitable
    and cannot be built.
    """
    for outside, reached in descendants.items():
        if outside in nodes:
            continue
        # `outside` sits after some member of the group if any member reaches
        # it; if it also reaches a member, the path leaves and returns.
        if reached & nodes and any(outside in descendants[n] for n in nodes):
            return False
    return True


def external_operands(
    nodes: FrozenSet[int], graph: DataFlowGraph
) -> Tuple[Set[str], Set[str]]:
    """Registers the group must read from outside, and results it must expose.

    An output counts when a value written inside the group is read by an
    instruction outside it. A value that nothing in the block ever reads is
    also counted, because it is presumably read by a later block and this has
    no cross-block liveness to check against.

    What is *not* counted is a value produced and consumed entirely within the
    group, even when it is the last write to that register in the block. The
    stricter reading -- any final write is live-out -- rejects exactly the
    candidates worth having: in a normalize loop it counts the intermediate
    between `fsqrt` and `fdiv` as a second result and throws away reciprocal
    square root, the best known fusion in that kernel.
    """
    inputs: Set[str] = set()
    outputs: Set[str] = set()

    for index in sorted(nodes):
        instruction = graph.instructions[index]
        producers = graph.producers.get(index, {})
        for register in instruction.uses:
            producer = producers.get(register)
            if producer is None or producer not in nodes:
                inputs.add(register)

    for index in sorted(nodes):
        instruction = graph.instructions[index]
        for register in instruction.defs:
            consumers = [
                other
                for other in range(index + 1, len(graph.instructions))
                if graph.producers.get(other, {}).get(register) == index
            ]
            if not consumers or any(other not in nodes for other in consumers):
                outputs.add(register)
    return inputs, outputs


def pattern_key(nodes: FrozenSet[int], graph: DataFlowGraph) -> str:
    """A name for the *shape*, so the same shape counts wherever it appears.

    Two occurrences differing only in which registers they happen to use are
    the same candidate. The key is the mnemonics in order plus the internal
    edges as operand positions, so `fmul`->`fadd` on v1 and the same on v7
    collapse together while a different wiring of the same mnemonics does not.
    """
    ordered = sorted(nodes)
    position = {node: i for i, node in enumerate(ordered)}
    mnemonics = [graph.instructions[node].mnemonic for node in ordered]
    edges = sorted(
        f"{position[a]}>{position[b]}"
        for a, b in graph.edges
        if a in nodes and b in nodes
    )
    return " ".join(mnemonics) + (" | " + ",".join(edges) if edges else "")


@dataclass
class Candidate:
    """A recurring fusable shape, with the evidence for how often it runs."""

    key: str
    mnemonics: Tuple[str, ...]
    size: int
    inputs: int
    outputs: int
    static_occurrences: int = 0
    dynamic_occurrences: int = 0
    examples: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        return {
            "pattern": self.key,
            "mnemonics": list(self.mnemonics),
            "size": self.size,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "static_occurrences": self.static_occurrences,
            "dynamic_occurrences": self.dynamic_occurrences,
            "example": self.examples[0] if self.examples else "",
        }


def enumerate_candidates(
    graph: DataFlowGraph,
    *,
    max_nodes: int = 3,
    max_inputs: int = 2,
    max_outputs: int = 1,
) -> List[Tuple[FrozenSet[int], Set[str], Set[str]]]:
    """Every connected, convex group a single instruction could encode.

    Grown one instruction at a time from each seed along data-flow edges, so
    only connected groups are ever considered: a "pattern" of instructions
    that pass no values to each other is two patterns.
    """
    if max_nodes < 2:
        return []
    descendants = _reachability(graph)
    total = len(graph.instructions)
    skip = {
        i.index
        for i in graph.instructions
        if i.mnemonic in CONTROL_FLOW or CONDITIONAL_BRANCH.match(i.mnemonic)
    }

    seen: Set[FrozenSet[int]] = set()
    found: List[Tuple[FrozenSet[int], Set[str], Set[str]]] = []
    frontier = [frozenset({i}) for i in range(total) if i not in skip]
    while frontier:
        group = frontier.pop()
        neighbours: Set[int] = set()
        for node in group:
            neighbours |= graph.successors(node) | graph.predecessors(node)
        for neighbour in neighbours - group - skip:
            grown = group | {neighbour}
            if grown in seen or len(grown) > max_nodes:
                continue
            seen.add(grown)
            if not is_convex(grown, descendants):
                continue
            inputs, outputs = external_operands(grown, graph)
            if len(inputs) <= max_inputs and len(outputs) <= max_outputs:
                found.append((grown, inputs, outputs))
            # Grown further even when it does not fit: adding an instruction
            # can *reduce* the operand count by consuming a value that was an
            # output, which is exactly how a longer fusion becomes encodable.
            if len(grown) < max_nodes:
                frontier.append(grown)
    return found


def block_lines(block: Dict[str, object]) -> List[str]:
    """Pull assembly text out of a block however the producer spelled it.

    `hot_blocks` carries its disassembly in `listing`, as records with a
    `text` field, and uses `instructions` for the *number* of instructions --
    so reading `instructions` as lines gets an integer and silently mines
    nothing.
    """
    listing = block.get("listing")
    if isinstance(listing, list):
        rows = [
            str(row.get("text") or "") if isinstance(row, dict) else str(row)
            for row in listing
        ]
        if any(row.strip() for row in rows):
            return rows

    for field_name in ("instructions", "disassembly", "lines", "asm"):
        value = block.get(field_name)
        if isinstance(value, str):
            return value.splitlines()
        if isinstance(value, list) and all(isinstance(v, str) for v in value):
            return value
    return []


def mine_blocks(
    blocks: Sequence[Dict[str, object]],
    *,
    max_nodes: int = 3,
    max_inputs: int = 2,
    max_outputs: int = 1,
    min_dynamic: int = 0,
) -> List[Dict[str, object]]:
    """Rank fusable shapes across hot blocks by how often they really ran.

    Each block supplies its disassembly and its measured execution count. A
    shape occurring once in a block that ran ten million times outranks one
    occurring ten times in a block that ran twice, and only the second kind is
    visible to a static scan of the source.
    """
    candidates: Dict[str, Candidate] = {}
    for block in blocks:
        lines = block_lines(block)
        executions = int(block.get("executions") or block.get("count") or 0)
        decoded = parse_block(lines)
        if len(decoded) < 2:
            continue
        graph = build_dfg(decoded)
        for nodes, inputs, outputs in enumerate_candidates(
            graph,
            max_nodes=max_nodes,
            max_inputs=max_inputs,
            max_outputs=max_outputs,
        ):
            key = pattern_key(nodes, graph)
            entry = candidates.get(key)
            if entry is None:
                entry = Candidate(
                    key=key,
                    mnemonics=tuple(
                        graph.instructions[n].mnemonic for n in sorted(nodes)
                    ),
                    size=len(nodes),
                    inputs=len(inputs),
                    outputs=len(outputs),
                )
                candidates[key] = entry
            entry.static_occurrences += 1
            entry.dynamic_occurrences += executions
            if len(entry.examples) < 3:
                entry.examples.append(
                    " ; ".join(graph.instructions[n].text for n in sorted(nodes))
                )

    ranked = [c for c in candidates.values() if c.dynamic_occurrences >= min_dynamic]
    ranked.sort(key=lambda c: (-c.dynamic_occurrences, -c.size, c.key))
    return [c.as_dict() for c in ranked]
