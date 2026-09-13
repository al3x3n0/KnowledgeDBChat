"""What is limiting a simulated run, read off the statistics it already wrote.

A cycle count says how slow a kernel is and nothing about why, so choosing what
to do next is guesswork. In a live run the agent had a working way to measure a
mechanism and no way to pick one; it spent two of its nine iterations on
document search before trying a prefetcher, because nothing pointed anywhere.

This points. It does not conclude. Every signal here is a correlate -- stages
block for reasons that overlap, and the cycles cannot be partitioned between
them without double counting. So the output is a ranked set of suspects, each
naming the limit study that would settle it. Attribution suggests; headroom
proves. The same division as llvm-mca screening and gem5 refereeing.

Worth knowing how strong the signal can be: on a strided-scan kernel,
rename.IQFullEvents was 118,200 against ROBFullEvents 65 -- four orders of
magnitude, and widening the issue queue recovered 11.8% while widening the
reorder buffer recovered nothing at all.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

#: Which structure filling up stopped rename, and what to widen to test it.
#: gem5 counts these separately, which is the whole reason the question is
#: answerable rather than a matter of opinion.
BACKPRESSURE_SOURCES = {
    "IQ": ("issue_queue", "system.cpu.rename.IQFullEvents"),
    "ROB": ("reorder_buffer", "system.cpu.rename.ROBFullEvents"),
    "LQ": ("load_queue", "system.cpu.rename.LQFullEvents"),
    "SQ": ("store_queue", "system.cpu.rename.SQFullEvents"),
}

#: Cycle-accounted stage states, by the prefix they live under.
STAGE_STATES = {
    "fetch": "system.cpu.fetch.status::",
    "decode": "system.cpu.decode.status::",
    "rename": "system.cpu.rename.status::",
    "dispatch": "system.cpu.iew.dispatchStatus::",
}


def ticks_per_cycle(stats: Dict[str, float]) -> Optional[float]:
    """The tick/cycle ratio, derived rather than assumed.

    Miss latencies are reported in ticks and mean nothing to a reader thinking
    in cycles. The clock is not in stats.txt, but simTicks/numCycles is exactly
    it, and derived it cannot disagree with the run it came from -- a constant
    for 2 GHz would be silently wrong on any configuration that set a different
    clock.
    """
    cycles = stats.get("system.cpu.numCycles") or 0.0
    ticks = stats.get("simTicks") or 0.0
    if cycles <= 0 or ticks <= 0:
        return None
    return ticks / cycles


def _share(value: float, total: float) -> float:
    return (value / total) if total else 0.0


def stage_occupancy(stats: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    """Where each pipeline stage spent its cycles."""
    cycles = stats.get("system.cpu.numCycles") or 0.0
    occupancy: Dict[str, Dict[str, float]] = {}
    for stage, prefix in STAGE_STATES.items():
        states = {
            key[len(prefix) :]: value
            for key, value in stats.items()
            if key.startswith(prefix)
        }
        if not states:
            continue
        blocked = sum(
            value
            for name, value in states.items()
            if name.lower().startswith(("blocked", "block"))
        )
        occupancy[stage] = {
            "states": states,
            "blocked_cycles": blocked,
            "blocked_share": _share(blocked, cycles),
        }
    return occupancy


def backpressure(stats: Dict[str, float]) -> Dict[str, Any]:
    """Which structure filling up is what stopped rename.

    Reported as a share of the counted events rather than of cycles: these are
    occurrences, not durations, and presenting them as a fraction of runtime
    would invent a precision gem5 never measured.
    """
    events = {
        key: stats.get(stat, 0.0) for key, (_, stat) in BACKPRESSURE_SOURCES.items()
    }
    total = sum(events.values())
    ranked = sorted(events.items(), key=lambda kv: kv[1], reverse=True)
    dominant, top = ranked[0] if ranked else ("", 0.0)
    return {
        "events": events,
        "total_events": total,
        "dominant": dominant if top > 0 else None,
        "dominant_share": _share(top, total),
        "headroom_target": (
            BACKPRESSURE_SOURCES[dominant][0] if dominant and top > 0 else None
        ),
    }


def memory_pressure(stats: Dict[str, float]) -> Dict[str, Any]:
    """Miss counts and, in cycles rather than ticks, what a miss costs."""
    per_cycle = ticks_per_cycle(stats)
    out: Dict[str, Any] = {
        "lsq_blocked_by_cache": stats.get("system.cpu.lsq0.blockedByCache", 0.0),
    }
    for level, prefix in (
        ("l1d", "system.cpu.dcache."),
        ("l1i", "system.cpu.icache."),
        ("l2", "system.l2cache."),
    ):
        misses = stats.get(f"{prefix}demandMisses::total")
        if misses is None:
            continue
        latency_ticks = stats.get(f"{prefix}demandAvgMissLatency::total")
        out[level] = {
            "demand_misses": misses,
            "avg_miss_latency_cycles": (
                latency_ticks / per_cycle if latency_ticks and per_cycle else None
            ),
            "miss_rate": stats.get(f"{prefix}demandMissRate::total"),
        }
    return out


def memory_cost_ratio(stats: Dict[str, float]) -> Optional[float]:
    """Miss latency the run paid for, as a multiple of the cycles it took.

    MSHR misses rather than demand misses: a demand miss that merges into an
    existing outstanding request costs nothing extra, and counting it inflates
    the number by the degree of clustering. Even so this OVERSTATES, because
    misses overlap -- values above 1.0 are normal and mean the core had many in
    flight, not that it stalled longer than it ran.

    It is here because the alternative was worse. Ranking memory by miss COUNT
    against cycles put it fourth on a kernel where idealising the L1 recovered
    84% of the runtime, behind a full issue queue worth 3.4%. The queue fills
    because loads are missing; it is the symptom being counted.
    """
    cycles = stats.get("system.cpu.numCycles") or 0.0
    per_cycle = ticks_per_cycle(stats)
    if not cycles or not per_cycle:
        return None
    worst = 0.0
    for prefix in ("system.cpu.dcache.", "system.l2cache."):
        misses = stats.get(f"{prefix}demandMshrMisses::total") or stats.get(
            f"{prefix}demandMisses::total"
        )
        latency = stats.get(f"{prefix}demandAvgMissLatency::total")
        if misses and latency:
            worst = max(worst, (misses * (latency / per_cycle)) / cycles)
    return worst or None


def branch_behaviour(stats: Dict[str, float]) -> Dict[str, Any]:
    predicted = stats.get("system.cpu.branchPred.condPredicted", 0.0)
    incorrect = stats.get("system.cpu.branchPred.condIncorrect", 0.0)
    return {
        "conditional_predicted": predicted,
        "conditional_incorrect": incorrect,
        "mispredict_rate": _share(incorrect, predicted),
        "squash_cycles": stats.get("system.cpu.fetch.status::squashing", 0.0),
        "indirect_mispredicted": stats.get(
            "system.cpu.branchPred.indirectMispredicted", 0.0
        ),
    }


def front_end(stats: Dict[str, float]) -> Dict[str, Any]:
    cycles = stats.get("system.cpu.numCycles") or 0.0
    stall = stats.get("system.cpu.fetchStats0.icacheStallCycles", 0.0)
    return {
        "icache_stall_cycles": stall,
        "icache_stall_share": _share(stall, cycles),
        "fetch_rate": stats.get("system.cpu.fetchStats0.fetchRate"),
    }


def _signal(
    name: str, strength: float, evidence: str, target: Optional[str]
) -> Dict[str, Any]:
    return {
        "signal": name,
        "strength": round(strength, 4),
        "evidence": evidence,
        "headroom_target": target,
    }


def rank_signals(stats: Dict[str, float]) -> List[Dict[str, Any]]:
    """Suspects, strongest first, each naming the study that would settle it.

    `strength` is deliberately not a share of runtime. Each signal is scaled
    against the quantity it is actually a fraction of -- events for
    backpressure, cycles for a stall, predictions for a mispredict -- and they
    are comparable only as ranks. A number that looked like "37% of cycles"
    would be read as a budget and summed, and the sum would exceed the run.
    """
    cycles = stats.get("system.cpu.numCycles") or 0.0
    signals: List[Dict[str, Any]] = []

    # Memory first, because it is usually the cause and the queues are usually
    # the symptom. Capped at 1.0: overlap makes the raw ratio exceed the run,
    # and the uncapped figure is reported separately rather than ranked.
    ratio = memory_cost_ratio(stats)
    mem = memory_pressure(stats)
    if ratio:
        l1d = mem.get("l1d") if isinstance(mem.get("l1d"), dict) else {}
        signals.append(
            _signal(
                "waiting on memory",
                min(1.0, ratio),
                f"outstanding miss latency totals {ratio:.2f}x the run's "
                f"cycles (overlapping upper bound); L1D "
                f"{(l1d or {}).get('demand_misses', 0):.0f} demand misses"
                + (
                    f" at {l1d['avg_miss_latency_cycles']:.0f} cycles each"
                    if (l1d or {}).get("avg_miss_latency_cycles")
                    else ""
                ),
                "l1d_capacity",
            )
        )

    press = backpressure(stats)
    occupancy = stage_occupancy(stats)
    if press["dominant"]:
        events = press["events"][press["dominant"]]
        # The WORST-blocked stage, not rename's own. Backpressure propagates
        # backwards: a full structure blocks rename, which blocks decode,
        # which blocks fetch, and which of them records the stall depends on
        # the model. Scaling by rename ranked a load queue LAST on the PCE
        # INT8 attention kernel -- 9,342,457 LQ-full events against the issue
        # queue's 351,136 -- because rename there blocked 0.58% of cycles
        # while decode blocked 75.1%. Idealising that load queue recovered
        # 4.68%, more than every other structure combined, and the study that
        # followed the ranking measured three targets worth 0.00%.
        blocked = max(
            (
                float(stage.get("blocked_share", 0.0) or 0.0)
                for stage in occupancy.values()
                if isinstance(stage, dict)
            ),
            default=0.0,
        )
        signals.append(
            _signal(
                f"rename blocked by {press['dominant']} full",
                blocked,
                f"{events:.0f} of {press['total_events']:.0f} full events, "
                f"rename blocked {blocked * 100:.1f}% of cycles"
                + (
                    " -- but a full queue is usually downstream of memory "
                    "latency, not a limit of its own"
                    if ratio and ratio > 1.0
                    else ""
                ),
                press["headroom_target"],
            )
        )

    fe = front_end(stats)
    if fe["icache_stall_cycles"]:
        signals.append(
            _signal(
                "front end waiting on the instruction cache",
                fe["icache_stall_share"],
                f"{fe['icache_stall_cycles']:.0f} of {cycles:.0f} cycles",
                "l1i_capacity",
            )
        )

    br = branch_behaviour(stats)
    if br["conditional_predicted"]:
        signals.append(
            _signal(
                "branch mispredictions",
                br["mispredict_rate"],
                f"{br['conditional_incorrect']:.0f} of "
                f"{br['conditional_predicted']:.0f} conditional branches",
                "branch_prediction",
            )
        )

    signals.sort(key=lambda s: s["strength"], reverse=True)
    return signals


def _next_study(signals: List[Dict[str, Any]]) -> str:
    """What to measure next -- naming more than one target, deliberately.

    Attribution is heuristic and has already been wrong here: it ranked a full
    issue queue first on a kernel where idealising the L1 recovered 84% and the
    queue recovered 3.4%. `measure_headroom` takes a list, so recommending the
    top few costs one extra simulation each and cannot dead-end the study on a
    single bad rank. One suspect would be a stronger claim than this evidence
    supports.
    """
    targets: List[str] = []
    for signal in signals:
        target = signal.get("headroom_target")
        if target and target not in targets:
            targets.append(target)
    if not targets:
        return (
            "No signal stood out; the run may be limited by true dependencies "
            "rather than by any structure."
        )
    shortlist = targets[:3]
    return (
        "measure_headroom with targets "
        + ", ".join(repr(t) for t in shortlist)
        + " turns these suspects into bounds. Take more than the first: "
        "backpressure and memory latency are not independent, and the "
        "strongest-ranked signal is not reliably the one worth the most."
    )


def attribute(stats: Dict[str, float]) -> Dict[str, Any]:
    """The whole picture: where cycles went, what was binding, what to try."""
    cycles = stats.get("system.cpu.numCycles") or 0.0
    signals = rank_signals(stats)
    return {
        "cycles": cycles,
        "instructions": stats.get("simInsts"),
        "ipc": stats.get("system.cpu.ipc"),
        "idle_cycles": stats.get("system.cpu.idleCycles", 0.0),
        "stage_occupancy": stage_occupancy(stats),
        "backpressure": backpressure(stats),
        "front_end": front_end(stats),
        "branch": branch_behaviour(stats),
        "memory": memory_pressure(stats),
        "signals": signals,
        "next_study": _next_study(signals),
        "memory_cost_ratio": memory_cost_ratio(stats),
        "caveat": (
            "These signals overlap and are not a partition of the run's "
            "cycles: a stage blocks because the next one is full, and the "
            "same cycle is counted by both. Ranks are meaningful WITHIN one "
            "run; sums are not, and neither are comparisons between runs. "
            "Measured: two kernels whose memory signal both capped at 1.0 had "
            "L1 headroom of 84% and 29%, and their second-ranked queue signal "
            "ordered the opposite way from the queue headroom it predicted. "
            "The ordering here held in both, the magnitudes did not. Nothing "
            "here measures what a change would be worth -- that needs the "
            "change, or an idealised limit study."
        ),
    }
