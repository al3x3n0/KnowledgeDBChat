"""Read a gem5 stats.txt into the numbers a measurement needs.

gem5 emits several hundred statistics per run. A referee that hands all of them
back buries the two or three that settle a prediction, and one that hands back
a hand-picked few silently decides what the question was. This parses the file
whole and names the handful that answer "how long did this take", leaving the
rest addressable by name.

Cycles are the measurement to prefer over seconds: gem5's simSeconds depends on
the clock the config assigned, so comparing two runs by seconds compares their
configurations as much as their code.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, Optional

# "name    value    # description", with values that may be ints, floats, nan
# or inf. Distribution rows carry extra columns and are skipped by taking only
# the first value.
_STAT_LINE = re.compile(
    r"^(?P<name>[A-Za-z_][\w.:]*)\s+(?P<value>-?[\d.]+(?:[eE][-+]?\d+)?|nan|inf)\b"
)

# Names differ across gem5 versions, so each metric lists the spellings seen
# rather than assuming one. A referee that reports None because a key moved is
# worse than no referee.
CYCLE_KEYS = (
    "system.cpu.numCycles",
    "system.cpu0.numCycles",
    "system.switch_cpus.numCycles",
)
INSTRUCTION_KEYS = (
    "simInsts",
    "system.cpu.commitStats0.numInsts",
    "system.cpu.committedInsts",
    "system.cpu.commit.committedInsts",
)
SECONDS_KEYS = ("simSeconds",)
TICK_KEYS = ("simTicks",)


def parse(lines: Iterable[str]) -> Dict[str, float]:
    """Parse every scalar statistic in a stats.txt."""
    stats: Dict[str, float] = {}
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("-"):
            continue
        match = _STAT_LINE.match(line)
        if not match:
            continue
        text = match.group("value")
        try:
            value = float(text)
        except ValueError:
            continue
        stats.setdefault(match.group("name"), value)
    return stats


def first_present(stats: Dict[str, float], keys: Iterable[str]) -> Optional[float]:
    for key in keys:
        if key in stats:
            return stats[key]
    return None


def summarize(stats: Dict[str, float]) -> Dict[str, Any]:
    """Pull out the few numbers that settle a performance claim."""
    cycles = first_present(stats, CYCLE_KEYS)
    instructions = first_present(stats, INSTRUCTION_KEYS)
    ipc: Optional[float] = None
    if cycles and instructions is not None and cycles != 0:
        ipc = round(instructions / cycles, 4)

    return {
        "cycles": cycles,
        "instructions": instructions,
        "ipc": ipc,
        "sim_seconds": first_present(stats, SECONDS_KEYS),
        "sim_ticks": first_present(stats, TICK_KEYS),
        "stats_parsed": len(stats),
        # Say which spelling was found: comparing a cycle count from one gem5
        # version against another's is only safe if they came from the same
        # statistic.
        "cycles_stat": next((k for k in CYCLE_KEYS if k in stats), None),
        "instructions_stat": next((k for k in INSTRUCTION_KEYS if k in stats), None),
    }


def speedup(
    baseline_cycles: Optional[float], variant_cycles: Optional[float]
) -> Optional[float]:
    """Baseline over variant, or None when either side is missing or zero.

    Reporting a speedup against a missing measurement would invent a result;
    reporting one against zero cycles would invent an infinite one.
    """
    if not baseline_cycles or not variant_cycles:
        return None
    return round(baseline_cycles / variant_cycles, 4)
