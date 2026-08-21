"""Read a callgrind profile into per-instruction execution counts.

This is the dynamic half of instruction-set extension work, and the half that
matters: how often a sequence *executes*, not how often it appears in source.
Callgrind counts every instruction exactly and needs no performance counters,
so it works in a sandbox that drops the privileges a PMU would require.

The format is compressed in two ways that a naive reader gets wrong, and did:
position fields may be relative (``+4``), repeated (``*``) or absolute
(``0x1234``), and a cost line following ``calls=`` is the *inclusive* cost of a
call made at that position, not the cost of the instruction there. Reading only
the absolute lines and treating call costs as self cost produced numbers that
were quietly off by orders of magnitude while still looking like a profile.

Format reference: valgrind/callgrind/docs/cl-format.txt
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

# A metadata record: a bare keyword followed by a colon. Cost lines never look
# like this, and header lines such as "pid: 703" otherwise parse as a position
# and a cost -- which added 705 phantom instructions to a 51M profile, small
# enough to pass for rounding and wrong enough to matter.
_METADATA = re.compile(r"^[A-Za-z_][A-Za-z_0-9]*:")
# Name compression: "fn=(3) name" defines an id, "fn=(3)" refers to it later.
_COMPRESSED_NAME = re.compile(r"^\((\d+)\)\s*(.*)$")


@dataclass
class Profile:
    """Self cost per instruction address, and per function."""

    event: str = "Ir"
    total: int = 0
    by_address: Dict[int, int] = field(default_factory=dict)
    # Costs kept per binary object. Addresses are only meaningful relative to
    # the object they came from: a position-independent executable and every
    # shared library it loads all start near zero, so merging them into one map
    # produces addresses that belong to nothing. A run whose hot code was in
    # libc built "hot blocks" that could not be disassembled against the
    # program, and reported them as instructions the program had run.
    by_object: Dict[str, Dict[int, int]] = field(default_factory=dict)
    by_function: Dict[str, int] = field(default_factory=dict)
    # Recorded rather than inferred: a profile whose addresses are file-relative
    # cannot be matched against a disassembly without knowing that.
    positions: Tuple[str, ...] = ("line",)

    def hottest_addresses(self, limit: int = 20) -> List[Tuple[int, int]]:
        return sorted(self.by_address.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]

    def hottest_functions(self, limit: int = 20) -> List[Tuple[str, int]]:
        return sorted(self.by_function.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]


def _name(raw: str, names: Dict[str, str]) -> str:
    """Resolve callgrind's name compression to a readable name.

    Ids are defined once with their name and referenced bare afterwards, so a
    reader that keeps the raw text reports the hottest function in a profile as
    "(443)".
    """
    match = _COMPRESSED_NAME.match(raw)
    if not match:
        return raw
    key, text = match.group(1), match.group(2).strip()
    if text:
        names[key] = text
        return text
    return names.get(key, raw)


def _resolve(field_text: str, previous: Optional[int]) -> Optional[int]:
    """Resolve one position field against the previous value on that axis."""
    text = field_text.strip()
    if not text:
        return previous
    if text == "*":
        return previous
    if text.startswith(("+", "-")):
        if previous is None:
            return None
        try:
            return previous + int(text, 10)
        except ValueError:
            return None
    try:
        return int(text, 16) if text.startswith("0x") else int(text, 10)
    except ValueError:
        return None


def parse(lines: Iterable[str]) -> Profile:
    """Parse a callgrind output file into self costs.

    Only the first event column is read, which is ``Ir`` in a default run --
    the number of instructions executed.
    """
    profile = Profile()
    positions: Tuple[str, ...] = ("line",)
    event_names: List[str] = ["Ir"]
    current: Dict[str, Optional[int]] = {}
    names: Dict[str, str] = {}
    function = "???"
    pending_call = False
    current_object = "?"

    for raw in lines:
        line = raw.rstrip("\n")
        if not line or line.startswith("#"):
            continue

        if line.startswith("positions:"):
            positions = tuple(line.split(":", 1)[1].split())
            profile.positions = positions
            current = {}
            continue
        if line.startswith("events:"):
            event_names = line.split(":", 1)[1].split()
            profile.event = event_names[0] if event_names else "Ir"
            continue
        if line.startswith("fn="):
            function = _name(line.split("=", 1)[1].strip(), names)
            current = {}
            pending_call = False
            continue
        if line.startswith("ob="):
            current_object = _name(line.split("=", 1)[1].strip(), names)
            current = {}
            continue
        if line.startswith(("fl=", "fi=", "fe=")):
            _name(line.split("=", 1)[1].strip(), names)
            current = {}
            continue
        if line.startswith(("cfn=", "cfl=", "cob=", "cfi=")):
            _name(line.split("=", 1)[1].strip(), names)
            continue
        if line.startswith("calls="):
            # The next cost line belongs to the callee, not to this address.
            pending_call = True
            continue
        if _METADATA.match(line):
            continue  # summary:, totals:, pid:, version:, cmd:, desc: ...
        if "=" in line.split(" ", 1)[0]:
            continue  # any other metadata record

        parts = line.split()
        if len(parts) < len(positions):
            continue
        resolved: List[Optional[int]] = []
        for axis, text in zip(positions, parts[: len(positions)]):
            value = _resolve(text, current.get(axis))
            current[axis] = value
            resolved.append(value)

        if pending_call:
            # Cost of the call itself; the callee's own lines carry its self cost.
            pending_call = False
            continue

        counts = parts[len(positions) :]
        if not counts:
            continue
        try:
            cost = int(counts[0])
        except ValueError:
            continue

        profile.total += cost
        profile.by_function[function] = profile.by_function.get(function, 0) + cost
        if "instr" in positions:
            address = resolved[positions.index("instr")]
            if address is not None:
                profile.by_address[address] = profile.by_address.get(address, 0) + cost
                per_object = profile.by_object.setdefault(current_object, {})
                per_object[address] = per_object.get(address, 0) + cost

    return profile


def parse_disassembly(text: str) -> Dict[int, str]:
    """Map address to instruction text from ``objdump -d`` output."""
    import re

    listing: Dict[int, str] = {}
    pattern = re.compile(r"^\s*([0-9a-f]+):\s+(?:[0-9a-f]{2,8} )+\s*\t(.+?)\s*$")
    for line in text.splitlines():
        match = pattern.match(line)
        if match:
            listing[int(match.group(1), 16)] = match.group(2).strip()
    return listing


def program_addresses(profile: Profile, binary: str = "workload") -> Dict[int, int]:
    """Costs for the program's own object, not everything it linked against.

    Blocks are runs of contiguous addresses, and addresses only mean anything
    within one object. Mixing a PIE executable with the shared libraries it
    loads -- all of which start near zero -- yields runs that span objects and
    disassemble to nothing.
    """
    if not profile.by_object:
        return profile.by_address
    named = {
        obj: costs
        for obj, costs in profile.by_object.items()
        if binary in obj and not obj.endswith(".so")
    }
    if named:
        return max(named.values(), key=lambda costs: sum(costs.values()))
    # Nothing matched by name: fall back to the object carrying the most cost
    # that is not obviously a library, and say nothing more confident than that.
    ranked = sorted(
        profile.by_object.items(), key=lambda kv: -sum(kv[1].values())
    )
    for obj, costs in ranked:
        if ".so" not in obj:
            return costs
    return profile.by_address


def hot_blocks(
    profile: Profile,
    listing: Dict[int, str],
    *,
    limit: int = 10,
    max_gap: int = 8,
) -> List[Dict[str, object]]:
    """Group hot addresses into straight-line runs, hottest first.

    Every instruction in a basic block executes the same number of times, so a
    change in execution count is a control-flow boundary. Splitting on
    contiguity alone welded a function's prologue onto its inner loop and
    called the result one 91-instruction block that "ran 2,048,000 times" --
    true of the loop, false of the prologue beside it.

    This is a run of equally-executed contiguous instructions, which is the
    unit a dataflow graph is built from, so it carries its instructions rather
    than just its cost.
    """
    # The program's own object only: a run of contiguous addresses spanning two
    # objects is not a basic block, and cannot be disassembled against either.
    costs = program_addresses(profile)
    if not costs:
        return []
    addresses = sorted(costs)
    runs: List[List[int]] = [[addresses[0]]]
    for address in addresses[1:]:
        previous = runs[-1][-1]
        same_block = (
            address - previous <= max_gap
            and costs[address] == costs[previous]
        )
        if same_block:
            runs[-1].append(address)
        else:
            runs.append([address])

    blocks: List[Dict[str, object]] = []
    for run in runs:
        counts = [costs[a] for a in run]
        blocks.append(
            {
                "start": run[0],
                "end": run[-1],
                "instructions": len(run),
                # Execution count of the block, taken as its most-executed
                # instruction: a straight-line block runs as often as its
                # hottest instruction, and summing would count the block once
                # per instruction it contains.
                "executions": max(counts),
                "instruction_cost": sum(counts),
                "listing": [
                    {
                        "address": a,
                        "count": costs[a],
                        "text": listing.get(a, "?"),
                    }
                    for a in run
                ],
            }
        )
    blocks.sort(key=lambda b: (-int(b["instruction_cost"]), int(b["start"])))
    return blocks[:limit]
