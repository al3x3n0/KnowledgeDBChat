"""Comparing one microarchitectural mechanism against a baseline, in gem5.

`simulate_c_workload` tunes a core model: op latencies, widths, queue depths,
all of them values on objects that already exist. A *mechanism* is a different
thing -- a prefetcher, a replacement policy, a branch predictor -- and it is a
SimObject, not a value. gem5 will not let a value-setter install one:

    SimObjectCliWrapperException: tried to set unsettable object parameter:
    replacement_policy

so `-P` is out, and se.py's own flags reach almost nothing (it ships
`--list-rp-types` with no `--rp-type`, and on this build `--bp-type` offers
exactly GshareBP, because gem5 25.1 moved the direction predictors under
`ConditionalPredictor` while its ObjectList still filters on `BranchPredictor`
-- TAGE, LTAGE, TournamentBP and the perceptrons are all built and all
unreachable). Hence the config script below, which names the class in Python
where naming it works.

WHY THIS TOOL COMPARES INSTEAD OF RUNNING. The first prefetcher study attempted
here produced, in five minutes, the exact wrong answer this tool exists to
prevent. StridePrefetcher on the L1D with gem5's default `mshrs=4` issued
**35 of 503,959** identified candidates and came back bit-identical to no
prefetcher at all. Raising mshrs to 16 alongside it gave **2.59x** -- and
mshrs=32 with *no prefetcher* gave the same 2.59x. The whole effect was the
MSHRs. Reported as "StridePrefetcher is worth 2.59x" it would have been
fabrication, and nothing about the run looked wrong. So the unit here is a
pair: a baseline and a variant that differ in the mechanism and in nothing
else, refused outright when they differ in anything more.

The same study's honest result, for scale: the same prefetcher on the **L2**
issues 401,061 of 401,066 and is worth 1.78x. The mechanism was never the
problem; the missing control was.

gem5 is deterministic, and that is what makes this cheap. Two configurations
that should differ and do not are caught by comparing one run against one run
-- no trials, no spread, no quiet machine required. The M3 host's ~13%
measurement floor does not reach in here. What does not come free is absolute
fidelity: these models are 40% (O3CPU) to 77% (NeoverseV2) off the silicon this
project measures, so a number from here is a ratio against its own baseline and
never a claim about hardware.
"""

from __future__ import annotations

import json
import logging
import math
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from app.services import agent_sandbox_runtime

logger = logging.getLogger(__name__)

DEFAULT_IMAGE = "ghcr.io/al3x3n0/kdbc-gem5-research:latest"
GEM5_BINARY = "/opt/gem5/build/ARM/gem5.opt"
DEFAULT_TIMEOUT_SECONDS = 900
#: Budget per configuration when a caller does not set one. A study runs its
#: configurations one after another, so a flat budget means an eight-point
#: sweep gets what a single run gets -- and the timeout it reports reads as
#: "the workload is too big", sending the caller to shrink a kernel that was
#: never the problem. A pair still comes to 900s, unchanged.
PER_CONFIG_TIMEOUT_SECONDS = 450
MAX_TOTAL_TIMEOUT_SECONDS = 3600
DEFAULT_FLAGS = "-O2 -static"
MAX_OUTPUT_CHARS = 8000

SAFE_FLAGS = re.compile(r"^[A-Za-z0-9_\-=+./ ]*$")
#: Configuration names become shell words and directory names in the run.
SAFE_RUN_NAME = re.compile(r"^[A-Za-z0-9_]{1,40}$")

# Where a mechanism may be attached, and the spec key that attaches it. Anything
# outside this set is geometry, and geometry differing between the two arms is
# the confound that this module refuses.
MECHANISM_KEYS = ("prefetcher", "replacement_policy")


# ---------------------------------------------------------------------------
# The config script, staged into the run directory.
#
# It is a string constant rather than a file in the image because the image is
# not rebuilt for this: a 668 MB image carrying a stripped binary and gem5's
# configs is what exists, and staging costs nothing.
# ---------------------------------------------------------------------------
MECH_CONFIG_SCRIPT = '''
"""A syscall-emulation system whose mechanisms come from a spec, not from flags."""

import json
import sys

sys.path.append("/opt/gem5/configs")

import m5
from common import ObjectList
from m5.objects import (
    AddrRange, Cache, DDR3_1600_8x8, L2XBar, MemCtrl, Process, Root,
    SEWorkload, SrcClockDomain, System, SystemXBar, VoltageDomain,
)
from m5.objects import __dict__ as OBJECTS


class SpecError(Exception):
    """The spec asks for something this build cannot do."""


# What was asked for against what was built. Every value is read back off the
# instantiated object rather than echoed from the spec, so the caller can tell
# a configured experiment from a request the simulator quietly dropped.
MANIFEST = {"caches": {}, "cpu": {}, "applied": []}


def _resolve(kind, name, registry):
    if name not in registry:
        raise SpecError(
            "unknown %s %r. This gem5 build offers: %s"
            % (kind, name, ", ".join(sorted(registry)))
        )
    return registry[name]


def _build(kind, spec, registry):
    if spec is None:
        return None
    if isinstance(spec, str):
        spec = {"class": spec}
    name = spec.get("class")
    if not name:
        raise SpecError("%s spec has no 'class'" % kind)
    stray = sorted(k for k in spec if k not in ("class", "params"))
    if stray:
        raise SpecError(
            "%s spec has unrecognised key(s) %s. A mechanism takes only "
            "'class' and 'params' -- a parameter written at the top level is "
            "not applied, and the run would come back identical to no "
            "mechanism with nothing to say why. Write "
            "{'class': %r, 'params': {...}}." % (kind, ", ".join(stray), name)
        )
    cls = _resolve(kind, name, registry)
    obj = cls()
    for key, value in (spec.get("params") or {}).items():
        if key not in cls._params:
            raise SpecError(
                "%s has no parameter %r. It declares: %s"
                % (name, key, ", ".join(sorted(cls._params)))
            )
        setattr(obj, key, value)
        MANIFEST["applied"].append("%s.%s.%s=%s" % (kind, name, key, value))
    return obj


def _conditional_predictors():
    base = OBJECTS["ConditionalPredictor"]
    return {
        name: obj
        for name, obj in OBJECTS.items()
        if isinstance(obj, type) and issubclass(obj, base) and obj is not base
    }


L1 = {"size": "64KiB", "assoc": 2, "tag_latency": 2, "data_latency": 2,
      "response_latency": 2, "mshrs": 4, "tgts_per_mshr": 20}
L2 = dict(L1, size="2MiB", assoc=8, tag_latency=20, data_latency=20,
          response_latency=20, mshrs=20, tgts_per_mshr=12)


def _cache(level, spec, defaults):
    spec = dict(defaults, **(spec or {}))
    cache = Cache(
        size=str(spec["size"]), assoc=int(spec["assoc"]),
        tag_latency=int(spec["tag_latency"]),
        data_latency=int(spec["data_latency"]),
        response_latency=int(spec["response_latency"]),
        mshrs=int(spec["mshrs"]), tgts_per_mshr=int(spec["tgts_per_mshr"]),
    )
    repl = _build("replacement_policy", spec.get("replacement_policy"),
                  ObjectList.rp_list._sub_classes)
    if repl is not None:
        cache.replacement_policy = repl
    pref = _build("prefetcher", spec.get("prefetcher"),
                  ObjectList.hwp_list._sub_classes)
    if pref is not None:
        cache.prefetcher = pref
    MANIFEST["caches"][level] = {
        "size": str(cache.size), "assoc": int(cache.assoc),
        "mshrs": int(cache.mshrs),
        "replacement_policy": type(cache.replacement_policy).__name__,
        "prefetcher": type(cache.prefetcher).__name__ if pref is not None else "none",
    }
    return cache


def _apply_cpu_params(cpu, params):
    """Set core parameters, including the ones that live on a vector member.

    `numIQEntries` does not exist on O3CPU in gem5 25.1: the issue queue moved
    to `instQueues[N]` of class IQUnit, param `numEntries`, and instQueues is a
    SimObjectVector. A study that sets `numIQEntries` runs clean, changes
    nothing, and concludes the issue queue is not the bottleneck -- on kernels
    where rename.IQFullEvents outnumbers ROBFullEvents by four orders of
    magnitude. So a path may name a vector member explicitly or with [*], and
    an unknown parameter is refused rather than absorbed.
    """
    for path, value in params.items():
        name, _, member = path.partition("[")
        if member:
            index = member.rstrip("]")
            vector = getattr(cpu, name, None)
            if vector is None:
                raise SpecError("%s has no %r to index" % (type(cpu).__name__, name))
            index, _, field = index.partition("].")
            if not field:
                raise SpecError(
                    "%s names a vector member but no parameter on it; write "
                    "%s[*].numEntries" % (path, name)
                )
            members = list(vector) if index == "*" else [vector[int(index)]]
            if not members:
                raise SpecError("%s has no members to set" % name)
            for target in members:
                if field not in type(target)._params:
                    raise SpecError(
                        "%s has no parameter %r. It declares: %s"
                        % (type(target).__name__, field,
                           ", ".join(sorted(type(target)._params)))
                    )
                setattr(target, field, value)
            MANIFEST["applied"].append("cpu.%s=%s (%d members)"
                                       % (path, value, len(members)))
            continue

        if name not in type(cpu)._params:
            raise SpecError(
                "%s has no parameter %r. Structure sizes that moved in gem5 "
                "25.1 live on sub-objects: the issue queue is "
                "instQueues[*].numEntries, not numIQEntries." % (
                    type(cpu).__name__, name)
            )
        setattr(cpu, name, value)
        MANIFEST["applied"].append("cpu.%s=%s" % (name, value))
    MANIFEST["cpu"]["params"] = dict(params)


def main():
    with open("spec.json") as handle:
        spec = json.load(handle)

    cpu_name = spec.get("cpu_type", "O3CPU")
    cpu_cls = _resolve("cpu_type", cpu_name, ObjectList.cpu_list._sub_classes)

    system = System()
    system.clk_domain = SrcClockDomain(clock=spec.get("clock", "2GHz"),
                                       voltage_domain=VoltageDomain())
    system.mem_mode = "timing"
    system.mem_ranges = [AddrRange(spec.get("mem_size", "512MiB"))]
    system.cache_line_size = int(spec.get("cache_line_size", 64))
    system.cpu = cpu_cls()
    _apply_cpu_params(system.cpu, spec.get("cpu_params") or {})

    # In gem5 25.1 a BranchPredictor is a composition -- a conditional
    # predictor, a BTB, a RAS, an indirect predictor. Naming "LTAGE" as though
    # it were the predictor sets nothing; it is the conditionalBranchPred
    # member, which is why --bp-type cannot reach it.
    bp = spec.get("branch_pred") or {}
    if bp:
        pred = getattr(system.cpu, "branchPred", None)
        if pred is None:
            raise SpecError(
                "%s has no branch predictor to configure; it is not a "
                "speculative model." % cpu_name
            )
        cond = _build("conditional_predictor", bp.get("conditional"),
                      _conditional_predictors())
        if cond is not None:
            pred.conditionalBranchPred = cond
        MANIFEST["cpu"]["conditional_predictor"] = type(
            pred.conditionalBranchPred).__name__

    caches = spec.get("caches") or {}
    system.cpu.icache = _cache("l1i", caches.get("l1i"), dict(L1, size="32KiB"))
    system.cpu.dcache = _cache("l1d", caches.get("l1d"), L1)
    system.cpu.icache.cpu_side = system.cpu.icache_port
    system.cpu.dcache.cpu_side = system.cpu.dcache_port

    system.l2bus = L2XBar()
    system.cpu.icache.mem_side = system.l2bus.cpu_side_ports
    system.cpu.dcache.mem_side = system.l2bus.cpu_side_ports
    system.l2cache = _cache("l2", caches.get("l2"), L2)
    system.l2cache.cpu_side = system.l2bus.mem_side_ports

    system.membus = SystemXBar()
    system.l2cache.mem_side = system.membus.cpu_side_ports
    system.cpu.createInterruptController()
    system.mem_ctrl = MemCtrl()
    system.mem_ctrl.dram = DDR3_1600_8x8(range=system.mem_ranges[0])
    system.mem_ctrl.port = system.membus.mem_side_ports
    system.system_port = system.membus.cpu_side_ports

    binary = spec.get("binary", "./workload")
    system.workload = SEWorkload.init_compatible(binary)
    process = Process()
    process.cmd = [binary] + list(spec.get("args") or [])
    system.cpu.workload = process
    system.cpu.createThreads()

    MANIFEST["cpu"]["type"] = cpu_name
    MANIFEST["cpu"]["class"] = type(system.cpu).__name__

    Root(full_system=False, system=system)
    m5.instantiate()
    event = m5.simulate()
    MANIFEST["exit_cause"] = event.getCause()
    with open("manifest.json", "w") as handle:
        json.dump(MANIFEST, handle, indent=2)


try:
    main()
except SpecError as exc:
    sys.stderr.write("SPEC_ERROR %s\\n" % exc)
    sys.exit(93)
'''


# ---------------------------------------------------------------------------
# The three gates.
# ---------------------------------------------------------------------------
def find_confounds(baseline: Dict[str, Any], variant: Dict[str, Any]) -> List[str]:
    """Everything the two arms differ in that is not the mechanism itself.

    This is the whole point of the tool. A prefetcher added alongside a wider
    MSHR file measured 2.59x here, all of it the MSHRs; the two changes were in
    one spec and nothing distinguished them afterwards. A difference outside
    MECHANISM_KEYS means the comparison cannot attribute its result, so the run
    is refused rather than annotated -- a warning on a plausible number gets
    read past.
    """
    confounds: List[str] = []

    for key in sorted(set(baseline) | set(variant)):
        if key == "caches":
            continue
        if baseline.get(key) != variant.get(key):
            if key == "branch_pred":
                continue
            confounds.append(
                f"{key}: baseline {baseline.get(key)!r} vs variant "
                f"{variant.get(key)!r}"
            )

    base_caches = baseline.get("caches") or {}
    var_caches = variant.get("caches") or {}
    for level in sorted(set(base_caches) | set(var_caches)):
        base_level = base_caches.get(level) or {}
        var_level = var_caches.get(level) or {}
        if isinstance(base_level, dict) and isinstance(var_level, dict):
            for field in sorted(set(base_level) | set(var_level)):
                if field in MECHANISM_KEYS:
                    continue
                if base_level.get(field) != var_level.get(field):
                    confounds.append(
                        f"caches.{level}.{field}: baseline "
                        f"{base_level.get(field)!r} vs variant "
                        f"{var_level.get(field)!r}"
                    )
    return confounds


def parse_stats(text: str) -> Dict[str, float]:
    """gem5's stats.txt as a mapping, with the non-deterministic rows dropped.

    The parsing is `gem5_stats.parse`; what is added is the dropping. Host
    timings are the one part of a deterministic simulator's output that does
    not reproduce, so an equality test that included them would fail on every
    pair and prove nothing -- exactly backwards from the null control this
    comparison depends on.
    """
    from app.services import gem5_stats

    return {
        name: value
        for name, value in gem5_stats.parse(text.splitlines()).items()
        if not name.startswith("host")
    }


def mechanism_activity(
    stats: Dict[str, float], manifest: Dict[str, Any]
) -> Dict[str, Dict[str, float]]:
    """Activation counters for each mechanism the manifest says was attached.

    The stat path follows the attachment point -- `system.cpu.dcache.prefetcher`
    for an L1D prefetcher, `system.l2cache.prefetcher` for an L2 one -- so a
    hardcoded prefix reports a mechanism as inactive precisely when it is
    working. That happened here: the L2 prefetcher issuing 401,061 of 401,066
    candidates came back with an empty activation column because the extractor
    only looked under `dcache`.
    """
    level_paths = {
        "l1i": "system.cpu.icache",
        "l1d": "system.cpu.dcache",
        "l2": "system.l2cache",
    }
    activity: Dict[str, Dict[str, float]] = {}
    for level, described in (manifest.get("caches") or {}).items():
        if described.get("prefetcher", "none") == "none":
            continue
        prefix = f"{level_paths.get(level, '')}.prefetcher."
        counters = {
            key[len(prefix) :]: value
            for key, value in stats.items()
            if key.startswith(prefix)
            and key[len(prefix) :]
            in ("pfIdentified", "pfIssued", "pfUseful", "pfUnused")
        }
        if counters:
            activity[f"{level}.prefetcher"] = counters
    return activity


def judge_activation(activity: Dict[str, Dict[str, float]]) -> Optional[str]:
    """Why a zero result is not a result. None when the mechanism really ran.

    A mechanism that never fires produces a clean, reproducible, entirely
    wrong "no effect". The counter that matters is what the mechanism *did*
    (pfIssued), not what it noticed (pfIdentified): the starved L1D prefetcher
    identified 503,959 candidates and issued 35.
    """
    if not activity:
        return None
    for name, counters in sorted(activity.items()):
        issued = counters.get("pfIssued", 0.0)
        identified = counters.get("pfIdentified", 0.0)
        if issued <= 0:
            return (
                f"{name} never fired: pfIssued=0 against pfIdentified="
                f"{identified:.0f}. The mechanism is attached but did no work, "
                "so this run measures the baseline twice. Nothing about the "
                "cycle counts will say so."
            )
        if identified > 0 and issued / identified < 0.01:
            return (
                f"{name} issued {issued:.0f} of {identified:.0f} identified "
                f"candidates ({100 * issued / identified:.2f}%). At gem5's "
                "default L1 mshrs=4 a prefetcher has no spare miss-handling "
                "capacity to issue into, and the run comes back bit-identical "
                "to no prefetcher. Raise the cache's mshrs IN BOTH ARMS, or "
                "attach the mechanism to L2, where mshrs defaults to 20."
            )
    return None


# ---------------------------------------------------------------------------
# Running the pair.
# ---------------------------------------------------------------------------
def _spec_for(base: Dict[str, Any], binary: str, args: Sequence[str]) -> Dict[str, Any]:
    spec = json.loads(json.dumps(base or {}))
    spec["binary"] = f"./{binary}"
    spec["args"] = list(args)
    return spec


def _read(workdir: str, name: str) -> str:
    path = Path(workdir, name)
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def stats_identical(left: Dict[str, float], right: Dict[str, float]) -> bool:
    """Whether two runs produced the same statistics, NaN included.

    gem5 prints `nan` for every averaged statistic whose denominator was zero
    -- `avgBlocked::no_mshrs`, `ftq.occupancy::mean` and a dozen more appear in
    essentially every run. NaN is not equal to itself, so comparing the two
    dictionaries with `==` reports two byte-identical runs as different, and
    the check that exists to say "the variant changed nothing" instead said
    nothing at all. It never once fired on real output; the tests that passed
    used hand-written statistics with no NaN in them.

    Two NaNs in the same statistic are the same result. That is the whole fix.
    """
    if set(left) != set(right):
        return False
    for name, value in left.items():
        other = right[name]
        if value == other:
            continue
        if math.isnan(value) and math.isnan(other):
            continue
        return False
    return True


def compare_arms(
    baseline_stats: Dict[str, float],
    variant_stats: Dict[str, float],
) -> Dict[str, Any]:
    """Cycles, the ratio, and whether the two runs are the same run.

    Bit-identical stats is reported as its own outcome rather than as a 0.0%
    improvement, because the two mean different things and only one of them is
    a measurement. Three prefetcher parameters on this build -- on_miss,
    prefetch_on_access, use_virtual_addresses -- are accepted, recorded as
    changed in config.ini, and leave the simulation untouched. A run that says
    "0.0% faster" hides that; one that says "identical" points straight at it.
    """
    base_cycles = baseline_stats.get("system.cpu.numCycles", 0.0)
    var_cycles = variant_stats.get("system.cpu.numCycles", 0.0)
    identical = stats_identical(baseline_stats, variant_stats)
    result: Dict[str, Any] = {
        "baseline_cycles": base_cycles,
        "variant_cycles": var_cycles,
        "identical_stats": identical,
    }
    if base_cycles > 0 and var_cycles > 0:
        result["speedup"] = base_cycles / var_cycles
        result["cycle_change_percent"] = (
            100.0 * (var_cycles - base_cycles) / base_cycles
        )
    if identical:
        result["note"] = (
            "The two runs produced byte-identical statistics. gem5 is "
            "deterministic, so this is not noise averaging out: the variant "
            "changed nothing the simulator acted on. Check the manifest to see "
            "whether the mechanism was instantiated at all, and note that some "
            "parameters on this build are accepted and then ignored."
        )
    return result


def _differing_stats(
    baseline_stats: Dict[str, float], variant_stats: Dict[str, float], limit: int = 12
) -> List[Dict[str, Any]]:
    """The stats that moved most, as a fraction of the baseline."""
    rows = []
    for name, base in baseline_stats.items():
        var = variant_stats.get(name)
        if var is None or var == base:
            continue
        scale = abs(base) if base else abs(var)
        rows.append(
            {
                "stat": name,
                "baseline": base,
                "variant": var,
                "relative_change": (var - base) / scale if scale else 0.0,
            }
        )
    rows.sort(key=lambda row: abs(row["relative_change"]), reverse=True)
    return rows[:limit]


class SandboxRunFailed(Exception):
    """A run that produced no usable statistics, with why in `detail`."""

    def __init__(self, detail: Dict[str, Any]):
        super().__init__(detail.get("error", "run failed"))
        self.detail = detail


#: What a plugin of each kind must include and export, and a skeleton of one.
#:
#: A live run wrote gem5-style C++ three times running -- `#include
#: "mem/cache/prefetch/queued.hh"`, a class deriving from Queued, no extern
#: "C" -- because that is what a gem5 prefetcher looks like everywhere except
#: here. The tool description said the right thing and lost to a stronger
#: prior. So the correction carries a whole working skeleton rather than a
#: rule, which is the same fix the AXIS tools needed: describing a grammar
#: nobody has seen does not teach it.
PLUGIN_KINDS = {
    "replacement policy": {
        "header": "gem5_rp_plugin_abi.h",
        "entry": "gem5_rp_api_v1",
        "api": "Gem5RpApiV1",
        "select": '{"class": "PluginRP", "params": {"library": '
        '"/work/plugin.so", "config": "..."}}',
        "skeleton": """#include <gem5_rp_plugin_abi.h>
#include <new>
namespace {
struct P { uint64_t threshold; };
Gem5RpPolicy *create(const char *config) {
    P *p = new (std::nothrow) P(); if (!p) return nullptr;
    p->threshold = 2;              /* parse `config` yourself if you need it */
    return reinterpret_cast<Gem5RpPolicy *>(p);
}
void destroy(Gem5RpPolicy *s) { delete reinterpret_cast<P *>(s); }
void invalidate(Gem5RpPolicy *, Gem5RpEntry *e) { e->last_touch_tick = 0; e->touches = 0; }
void touch(Gem5RpPolicy *, Gem5RpEntry *e, uint64_t t) { e->last_touch_tick = t; e->touches++; }
void reset(Gem5RpPolicy *, Gem5RpEntry *e, uint64_t t) { e->last_touch_tick = t; e->touches = 1; }
size_t get_victim(Gem5RpPolicy *, Gem5RpEntry *const *e, size_t n) {
    size_t oldest = 0;                       /* return an INDEX, not a pointer */
    for (size_t i = 0; i < n; ++i)
        if (e[i]->last_touch_tick < e[oldest]->last_touch_tick) oldest = i;
    return oldest;
}
const Gem5RpApiV1 API = {GEM5_RP_ABI_VERSION, create, destroy, invalidate, touch, reset, get_victim};
}
extern "C" const Gem5RpApiV1 *gem5_rp_api_v1(void) { return &API; }""",
    },
    "prefetcher": {
        "header": "gem5_pf_plugin_abi.h",
        "entry": "gem5_pf_api_v1",
        "api": "Gem5PfApiV1",
        "select": '{"class": "PluginPrefetcher", "params": {"library": '
        '"/work/plugin.so", "config": "..."}}',
        "skeleton": """#include <gem5_pf_plugin_abi.h>
#include <new>
namespace {
struct P { uint32_t block_size; uint32_t degree; };
Gem5PfPrefetcher *create(const char *config, uint32_t block_size) {
    P *p = new (std::nothrow) P(); if (!p) return nullptr;
    p->block_size = block_size ? block_size : 64;
    p->degree = 2;                 /* parse `config` yourself if you need it */
    return reinterpret_cast<Gem5PfPrefetcher *>(p);
}
void destroy(Gem5PfPrefetcher *s) { delete reinterpret_cast<P *>(s); }
size_t calculate(Gem5PfPrefetcher *s, const Gem5PfAccess *a,
                 Gem5PfRequest *out, size_t max_out) {
    const P *p = reinterpret_cast<const P *>(s);
    uint64_t block = a->address & ~(uint64_t)(p->block_size - 1);
    size_t n = 0;                     /* next-N-lines; put your idea here */
    for (uint32_t i = 1; i <= p->degree && n < max_out; ++i, ++n) {
        out[n].address = block + (uint64_t)i * p->block_size;
        out[n].priority = 0;
    }
    return n;                                  /* how many you wrote to `out` */
}
const Gem5PfApiV1 API = {GEM5_PF_ABI_VERSION, create, destroy, calculate};
}
extern "C" const Gem5PfApiV1 *gem5_pf_api_v1(void) { return &API; }""",
    },
}


def check_plugin_source(source: str) -> Optional[str]:
    """Why this will not build as a plugin, before a compiler says it in C++.

    The compiler's answer to gem5-style source is a hundred lines about a
    missing header, and the correction it implies -- find the gem5 headers --
    is the opposite of the truth. There are none, and none are needed.
    """
    text = source or ""
    kind = "prefetcher" if "gem5_pf" in text or "Gem5Pf" in text else None
    if kind is None and ("gem5_rp" in text or "Gem5Rp" in text):
        kind = "replacement policy"

    gem5_includes = [
        line.strip()
        for line in text.splitlines()
        if line.strip().startswith("#include")
        and (
            '"mem/' in line
            or '"params/' in line
            or '"base/' in line
            or '"sim/' in line
            or '"cpu/' in line
        )
    ]
    if gem5_includes:
        guess = kind or ("prefetcher" if "refetch" in text else "replacement policy")
        spec = PLUGIN_KINDS[guess]
        return (
            "This is gem5's own source layout, not a plugin: "
            + ", ".join(gem5_includes[:3])
            + ". No gem5 headers exist in this sandbox and none are needed -- "
            "a plugin talks to gem5 across a C ABI and derives from nothing. "
            f"For a {guess}, include <{spec['header']}> alone and export "
            f"`extern \"C\" const {spec['api']} *{spec['entry']}(void)`. "
            f"Select it with {spec['select']}. Here is a complete one that "
            f"compiles:\n\n{spec['skeleton']}"
        )

    if kind is None:
        return (
            "This names neither plugin ABI. Include <gem5_rp_plugin_abi.h> for "
            "a cache replacement policy or <gem5_pf_plugin_abi.h> for a "
            "prefetcher -- one of them, and nothing else."
        )

    spec = PLUGIN_KINDS[kind]
    if spec["entry"] not in text:
        return (
            f"A {kind} plugin must export `{spec['entry']}`; this source never "
            f"defines it, so dlsym would find nothing. Here is a complete one "
            f"that compiles:\n\n{spec['skeleton']}"
        )
    if 'extern "C"' not in text:
        return (
            f"`{spec['entry']}` is defined but not `extern \"C\"`, so C++ "
            "will mangle its name and dlsym will not find it. The declaration "
            f"must read: extern \"C\" const {spec['api']} *{spec['entry']}(void)"
        )
    return None


#: What a configuration may contain. Anything else is a guess, and a guess
#: must be refused as one.
CONFIG_KEYS = (
    "cpu_type",
    "clock",
    "mem_size",
    "cache_line_size",
    "cpu_params",
    "caches",
    "branch_pred",
)
CACHE_KEYS = (
    "size",
    "assoc",
    "tag_latency",
    "data_latency",
    "response_latency",
    "mshrs",
    "tgts_per_mshr",
    "replacement_policy",
    "prefetcher",
)
CACHE_LEVELS = ("l1i", "l1d", "l2")

EXAMPLE_CONFIG = '{"caches": {"l2": {"prefetcher": "StridePrefetcher"}}}'


def check_config_shape(config: Dict[str, Any], which: str) -> Optional[str]:
    """Why this is not a configuration, said before anything else reads it.

    An unrecognised key used to fall through to the confound rule, which then
    reported an invented `name` field as a difference between the arms and
    lectured the caller about MSHRs. A live run spent two attempts on that:
    the mistake was the shape, and the diagnosis was about attribution.
    """
    if not isinstance(config, dict):
        return (
            f"{which} must be a configuration object, not "
            f"{type(config).__name__}. The mechanism goes inside it: "
            f"{EXAMPLE_CONFIG}"
        )
    stray = [k for k in config if k not in CONFIG_KEYS]
    if stray:
        return (
            f"{which} has key(s) {', '.join(sorted(stray))}, which are not "
            "part of a configuration. It takes: "
            + ", ".join(CONFIG_KEYS)
            + f". A mechanism is named inside `caches`, like {EXAMPLE_CONFIG}"
            + ". Use `label` on the call itself to name the comparison."
        )
    caches = config.get("caches")
    if caches is not None:
        if not isinstance(caches, dict):
            return f"{which}.caches must be an object keyed by cache level."
        bad_levels = [k for k in caches if k not in CACHE_LEVELS]
        if bad_levels:
            return (
                f"{which}.caches has level(s) {', '.join(sorted(bad_levels))}; "
                "this model has " + ", ".join(CACHE_LEVELS) + "."
            )
        for level, spec in caches.items():
            if not isinstance(spec, dict):
                return f"{which}.caches.{level} must be an object."
            stray = [k for k in spec if k not in CACHE_KEYS]
            if stray:
                return (
                    f"{which}.caches.{level} has key(s) "
                    f"{', '.join(sorted(stray))}. A cache level takes: "
                    + ", ".join(CACHE_KEYS)
                    + "."
                )
    return None


async def run_configs(
    *,
    code: str,
    configs: Dict[str, Dict[str, Any]],
    flags: str = DEFAULT_FLAGS,
    run_args: str = "",
    plugin_source: str = "",
    image: str = DEFAULT_IMAGE,
    timeout_seconds: Optional[int] = None,
) -> Dict[str, Dict[str, Any]]:
    """Compile once, then run every named configuration in one container.

    One compile matters more than it sounds: this machine has already produced
    a 20x disagreement between clang and gcc on the same kernel, so two arms
    built by two toolchain invocations are two experiments. One container also
    means one image, one binary and one set of paths behind every number a
    study compares.

    Raises SandboxRunFailed with a caller-ready result on any failure, so each
    study reports the same way without repeating the diagnosis.
    """
    safe_flags = (flags or DEFAULT_FLAGS).strip()
    if not SAFE_FLAGS.match(safe_flags):
        raise SandboxRunFailed(
            {
                "success": False,
                "error": f"flags contain unsupported characters: {flags!r}",
            }
        )
    if "-static" not in safe_flags.split():
        safe_flags = f"{safe_flags} -static"

    if not agent_sandbox_runtime.execution_enabled():
        raise SandboxRunFailed(
            {
                "success": False,
                "error": "Sandboxed execution is disabled on this server.",
            }
        )
    if image not in agent_sandbox_runtime.allowed_images():
        raise SandboxRunFailed(
            {
                "success": False,
                "error": agent_sandbox_runtime.image_not_allowlisted(image),
            }
        )

    cpu_types = {str(c.get("cpu_type") or "O3CPU") for c in configs.values()}
    from app.services.agent_gem5_sandbox import model_support

    for cpu_type in sorted(cpu_types):
        support = await model_support(image, cpu_type)
        if not support.get("usable", True):
            raise SandboxRunFailed(
                {"success": False, "error": support.get("reason", "")}
            )

    for name in configs:
        if not SAFE_RUN_NAME.match(name):
            raise SandboxRunFailed(
                {
                    "success": False,
                    "error": f"configuration name {name!r} is not a plain identifier",
                }
            )

    for name, config in configs.items():
        wrong = check_config_shape(config, name)
        if wrong:
            raise SandboxRunFailed({"success": False, "error": wrong})

    if plugin_source.strip():
        wrong = check_plugin_source(plugin_source)
        if wrong:
            raise SandboxRunFailed({"success": False, "error": wrong})

    args = [a for a in (run_args or "").split() if a]
    with tempfile.TemporaryDirectory(prefix="gem5_study_") as workdir:
        Path(workdir, "workload.c").write_text(code, encoding="utf-8")
        Path(workdir, "mech_se.py").write_text(MECH_CONFIG_SCRIPT, encoding="utf-8")
        if plugin_source.strip():
            Path(workdir, "plugin.cc").write_text(plugin_source, encoding="utf-8")
        for name, spec in configs.items():
            Path(workdir, f"{name}.json").write_text(
                json.dumps(_spec_for(spec, "workload", args)), encoding="utf-8"
            )

        # The plugin is built first and separately, so a mistake in it is
        # reported as a mistake in it. Compiled in the same container as the
        # simulation that dlopens it, which is the only way the two agree
        # about libstdc++.
        build_plugin = (
            "g++ -shared -fPIC -O2 -std=c++17 -o /work/plugin.so plugin.cc "
            "2>plugin_err.txt || { cat plugin_err.txt >&2; exit 89; }; "
            if plugin_source.strip()
            else ""
        )
        script = (
            build_plugin
            + f"gcc {safe_flags} -o workload workload.c -lm 2>compile_err.txt || "
            "{ cat compile_err.txt >&2; exit 90; }; "
            f"for arm in {' '.join(configs)}; do "
            "  cp $arm.json spec.json; "
            f"  {GEM5_BINARY} --outdir=$arm mech_se.py >$arm.log 2>&1 || "
            '   { echo "ARM_FAILED $arm" >&2; tail -25 $arm.log >&2; exit 91; }; '
            "  cp manifest.json $arm.manifest.json; "
            "done; echo OK"
        )
        try:
            returncode, _stdout, stderr = await agent_sandbox_runtime.run_in_sandbox(
                script,
                workdir,
                image=image,
                timeout_seconds=timeout_seconds,
                memory="4096m",
                cpus="2",
            )
        except TimeoutError:
            raise SandboxRunFailed(
                {
                    "success": False,
                    "error": (
                        f"{len(configs)} simulations timed out after "
                        f"{timeout_seconds}s, which is "
                        f"{timeout_seconds // max(1, len(configs))}s each. "
                        "They run one after another, so the cost is the "
                        "kernel times the number of configurations -- an "
                        "out-of-order model manages on the order of 100k "
                        "instructions a second. Shrink the kernel, or study "
                        "fewer points; the kernel alone may be fine."
                    ),
                }
            )
        except FileNotFoundError:
            raise SandboxRunFailed(
                {"success": False, "error": "Docker is not available to this process"}
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"run_configs failed: {exc}")
            raise SandboxRunFailed(
                {"success": False, "error": f"Simulation failed: {exc}"}
            )

        if returncode == 89:
            raise SandboxRunFailed(
                {
                    "success": False,
                    "error": (
                        "The plugin did not compile. It is built against one "
                        "ABI header alone -- <gem5_rp_plugin_abi.h> for a "
                        "replacement policy, <gem5_pf_plugin_abi.h> for a "
                        "prefetcher -- and no gem5 headers are available or "
                        "needed. The compiler output below is the whole story; "
                        "a missing gem5 header in it means the source is "
                        "written against gem5's internals rather than the ABI."
                    ),
                    "plugin_stderr": stderr[:MAX_OUTPUT_CHARS],
                }
            )
        if returncode == 90:
            from app.services.agent_compiler_sandbox import explain_compiler_failure

            raise SandboxRunFailed(
                {
                    "success": False,
                    "error": explain_compiler_failure(stderr),
                    "compiler_stderr": stderr[:MAX_OUTPUT_CHARS],
                }
            )
        if "SPEC_ERROR" in stderr:
            line = next((ln for ln in stderr.splitlines() if "SPEC_ERROR" in ln), "")
            raise SandboxRunFailed(
                {"success": False, "error": line.replace("SPEC_ERROR ", "", 1)}
            )
        if returncode != 0:
            raise SandboxRunFailed(
                {
                    "success": False,
                    "error": "A simulation failed.",
                    "stderr": stderr[-MAX_OUTPUT_CHARS:],
                }
            )

        runs: Dict[str, Dict[str, Any]] = {}
        for name in configs:
            stats = parse_stats(_read(workdir, f"{name}/stats.txt"))
            if not stats:
                raise SandboxRunFailed(
                    {
                        "success": False,
                        "error": f"Configuration {name!r} produced no statistics.",
                        "stderr": stderr[-MAX_OUTPUT_CHARS:],
                    }
                )
            try:
                manifest = json.loads(_read(workdir, f"{name}.manifest.json") or "{}")
            except json.JSONDecodeError:
                manifest = {}
            runs[name] = {"stats": stats, "manifest": manifest}
        return runs


async def simulate_mechanism(
    *,
    code: str,
    variant: Dict[str, Any],
    baseline: Optional[Dict[str, Any]] = None,
    flags: str = DEFAULT_FLAGS,
    run_args: str = "",
    label: str = "",
    plugin_source: str = "",
    image: str = DEFAULT_IMAGE,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    """Run one workload twice -- with the mechanism and without -- and compare."""
    if not (code or "").strip():
        return {"success": False, "error": "code is required"}
    if not isinstance(variant, dict) or not variant:
        return {
            "success": False,
            "error": (
                "variant must be a configuration object, e.g. "
                '{"caches": {"l2": {"prefetcher": "StridePrefetcher"}}}. '
                "Call describe_gem5_mechanisms for the classes this build has."
            ),
        }

    if baseline is None:
        # The natural baseline is the variant with its mechanisms removed, and
        # deriving it is safer than asking for it: a hand-written baseline is
        # where the geometry silently drifts apart.
        baseline = json.loads(json.dumps(variant))
        for level in list((baseline.get("caches") or {}).keys()):
            level_spec = baseline["caches"][level]
            if isinstance(level_spec, dict):
                for key in MECHANISM_KEYS:
                    level_spec.pop(key, None)
        baseline.pop("branch_pred", None)

    confounds = find_confounds(baseline, variant)
    if confounds:
        return {
            "success": False,
            "error": (
                "The two arms differ in more than the mechanism, so whatever "
                "this measured could not be attributed to it: "
                + "; ".join(confounds)
                + ". Measured here: adding a prefetcher and widening the MSHR "
                "file together read as a 2.59x prefetcher win, and the MSHRs "
                "alone gave the same 2.59x. Put every non-mechanism change in "
                "BOTH arms, or make it a separate comparison."
            ),
            "confounds": confounds,
        }

    try:
        runs = await run_configs(
            code=code,
            configs={"baseline": baseline, "variant": variant},
            flags=flags,
            run_args=run_args,
            plugin_source=plugin_source,
            image=image,
            timeout_seconds=timeout_seconds,
        )
    except SandboxRunFailed as failure:
        return failure.detail

    baseline_stats = runs["baseline"]["stats"]
    variant_stats = runs["variant"]["stats"]
    baseline_manifest = runs["baseline"]["manifest"]
    variant_manifest = runs["variant"]["manifest"]

    activity = mechanism_activity(variant_stats, variant_manifest)
    comparison = compare_arms(baseline_stats, variant_stats)
    inactive = judge_activation(activity)

    result: Dict[str, Any] = {
        "success": inactive is None,
        "label": label,
        "comparison": comparison,
        "mechanism_activity": activity,
        "variant_configuration": variant_manifest,
        "baseline_configuration": baseline_manifest,
        "moved_stats": _differing_stats(baseline_stats, variant_stats),
        "attribution": (
            "Both arms ran the same binary in the same model and differ only "
            "in the mechanism, so the cycle ratio is attributable to it. It is "
            "a ratio within this model, not a claim about silicon: these "
            "models measure 40-77% off the M3 this project calibrates against."
        ),
    }
    if inactive is not None:
        result["error"] = inactive
        return result

    # A finding, so a goal contract can count this the way it counts every
    # other measurement. Only a run that passed the gates writes one: a
    # comparison whose mechanism never fired is the very thing a contract must
    # not be able to satisfy itself with, and returning it as evidence would
    # let a run reach its required count without measuring anything.
    result["findings"] = [
        _finding(label, comparison, activity, variant_manifest, baseline_manifest)
    ]
    return result


def mechanisms_under_test(
    manifest: Dict[str, Any], baseline_manifest: Dict[str, Any]
) -> List[str]:
    """What the variant has that the baseline does not.

    Read off the two manifests rather than off the request, so this reports
    what was built. Diffing matters more than it looks: every cache carries a
    replacement policy whether or not one is being studied, so listing what the
    variant *has* would name LRURP as the mechanism under test in every
    prefetcher run.
    """
    mechanisms: List[str] = []
    baseline_caches = baseline_manifest.get("caches") or {}
    for level, described in sorted((manifest.get("caches") or {}).items()):
        against = baseline_caches.get(level) or {}
        for key in MECHANISM_KEYS:
            value = described.get(key)
            if value in (None, "none") or value == against.get(key):
                continue
            mechanisms.append(f"{level}.{key}={value}")

    predictor = (manifest.get("cpu") or {}).get("conditional_predictor")
    if predictor and predictor != (baseline_manifest.get("cpu") or {}).get(
        "conditional_predictor"
    ):
        mechanisms.append(f"cpu.conditional_predictor={predictor}")
    return mechanisms


def _finding(
    label: str,
    comparison: Dict[str, Any],
    activity: Dict[str, Dict[str, float]],
    manifest: Dict[str, Any],
    baseline_manifest: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """One mechanism comparison, in the shape the rest of the run reads."""
    mechanisms = mechanisms_under_test(manifest, baseline_manifest or {})
    subject = (label or "").strip() or (", ".join(mechanisms) or "mechanism")
    speedup = comparison.get("speedup")
    return {
        "type": "mechanism_comparison",
        "subject": subject,
        "title": (
            f"{subject}: {speedup:.4f}x "
            f"({comparison['baseline_cycles']:.0f} -> "
            f"{comparison['variant_cycles']:.0f} cycles)"
            if speedup
            else f"{subject}: no cycle counts"
        ),
        "mechanisms": mechanisms,
        "speedup": speedup,
        "baseline_cycles": comparison.get("baseline_cycles"),
        "variant_cycles": comparison.get("variant_cycles"),
        "identical_stats": comparison.get("identical_stats"),
        "mechanism_activity": activity,
        "cpu_type": (manifest.get("cpu") or {}).get("type"),
        "measurement_source": "gem5 mechanism pair",
    }


# ---------------------------------------------------------------------------
# What this build actually offers.
# ---------------------------------------------------------------------------
_MECHANISM_CATALOG: Dict[str, Dict[str, Any]] = {}

CATALOG_SCRIPT = """
import json, sys
sys.path.append("/opt/gem5/configs")
from common import ObjectList
from m5.objects import __dict__ as OBJECTS

base = OBJECTS["ConditionalPredictor"]
print("CATALOG " + json.dumps({
    "prefetcher": sorted(ObjectList.hwp_list._sub_classes),
    "replacement_policy": sorted(ObjectList.rp_list._sub_classes),
    "conditional_predictor": sorted(
        n for n, o in OBJECTS.items()
        if isinstance(o, type) and issubclass(o, base) and o is not base
    ),
    "cpu_type": sorted(ObjectList.cpu_list._sub_classes),
}))
"""


def forget_catalog(image: str = "") -> None:
    """Drop the cached catalog, for tests and after an image is rebuilt."""
    if image:
        _MECHANISM_CATALOG.pop(image, None)
    else:
        _MECHANISM_CATALOG.clear()


def _singular(word: str) -> str:
    """The singular of an English plural, to the depth this needs.

    "policies" -> "policy" is the case that matters: stripping a trailing `s`
    alone leaves "policie", which matches nothing, so the tolerant path would
    have been tolerant of every plural except the one in the category names.
    """
    if word.endswith("ies") and len(word) > 3:
        return word[:-3] + "y"
    return word[:-1] if word.endswith("s") else word


def resolve_kind(kind: str, catalog: Dict[str, Any]) -> Optional[str]:
    """Which category the caller meant, tolerating how it was spelled.

    A live run asked for "prefetchers" and was refused, then asked again for
    "prefetcher" and got the same list -- one action spent on the difference
    between a plural and a singular. The filter is a convenience; refusing it
    over an `s` costs more than the filter is worth. Genuinely unknown values
    still fail, because silently listing everything would hide a caller asking
    for a mechanism class this build does not have.
    """
    wanted = (kind or "").strip().lower().replace("-", "_")
    # "all" is what a caller writes when the schema says "omit for all", and a
    # live run wrote exactly that and was refused. Asking for everything and
    # saying nothing are the same request.
    if not wanted or wanted in ("all", "any", "everything", "*"):
        return ""
    wanted = _singular(wanted)
    for name in catalog:
        if wanted == _singular(name.lower()):
            return name
    return None


def explain_probe_failure(stderr: str) -> str:
    """Name the cause when it is in the output, rather than blaming gem5.

    "gem5 did not report its mechanism classes" is true and useless. The one
    time it fired in a live run the cause was a full disk, which is nobody's
    idea of a simulator problem and which no amount of retrying fixes.
    """
    text = stderr or ""
    if "No space left on device" in text:
        return (
            "The host has no disk space left, so gem5 could not create its "
            "output directory. This is not a problem with the simulator or "
            "with the request, and retrying will fail the same way until "
            "space is freed on the machine running the sandbox."
        )
    if "cannot execute" in text or "Exec format error" in text:
        return (
            "The gem5 binary in this image will not execute here, which "
            "usually means the image was built for another architecture."
        )
    if not text.strip():
        return (
            "gem5 produced no output at all when asked for its mechanism "
            "classes, and no error either."
        )
    return (
        "gem5 did not report its mechanism classes. Its own output is in "
        "`stderr` and is the place to look."
    )


async def describe_gem5_mechanisms(
    *,
    kind: str = "",
    image: str = DEFAULT_IMAGE,
    timeout_seconds: int = 120,
) -> Dict[str, Any]:
    """The mechanism classes this gem5 build carries, asked of the build itself.

    Not a hardcoded list. gem5 renames and moves these between releases -- 25.1
    moved every direction predictor from `BranchPredictor` to
    `ConditionalPredictor`, which is why the shipped `--bp-type` flag offers one
    predictor out of a dozen that are present. A list written down here would
    be wrong the same way, and silently.
    """
    if not agent_sandbox_runtime.execution_enabled():
        return {
            "success": False,
            "error": "Sandboxed execution is disabled on this server.",
        }
    if image not in agent_sandbox_runtime.allowed_images():
        return {
            "success": False,
            "error": agent_sandbox_runtime.image_not_allowlisted(image),
        }

    catalog = _MECHANISM_CATALOG.get(image)
    if catalog is None:
        with tempfile.TemporaryDirectory(prefix="gem5_catalog_") as workdir:
            Path(workdir, "catalog.py").write_text(CATALOG_SCRIPT, encoding="utf-8")
            try:
                # Not 2>/dev/null. gem5's banner is noise, but discarding
                # the whole stream discards the diagnosis with it: this probe
                # failed a live run with "gem5 did not report its mechanism
                # classes" and nothing else, while the real cause -- printed
                # on stderr and thrown away -- was `OSError: [Errno 28] No
                # space left on device`. The tool blamed the simulator for a
                # full disk.
                _, stdout, stderr = await agent_sandbox_runtime.run_in_sandbox(
                    f"{GEM5_BINARY} --outdir=/tmp/out catalog.py",
                    workdir,
                    image=image,
                    timeout_seconds=timeout_seconds,
                )
            except Exception as exc:  # pragma: no cover - defensive
                return {
                    "success": False,
                    "error": f"Could not inspect the build: {exc}",
                }
        line = next((ln for ln in stdout.splitlines() if ln.startswith("CATALOG ")), "")
        if not line:
            return {
                "success": False,
                "error": explain_probe_failure(stderr),
                "stderr": stderr[-MAX_OUTPUT_CHARS:],
            }
        catalog = json.loads(line[len("CATALOG ") :])
        _MECHANISM_CATALOG[image] = catalog

    # An unrecognised `kind` returns everything rather than failing. Three
    # separate live runs lost an action here -- to "prefetchers", to "all",
    # and to "cache" -- and in every case the whole catalogue was what the
    # caller wanted and what a second call then fetched. The filter is a
    # convenience over roughly sixty names; refusing one costs more than
    # sending them all, and the unmatched request is reported rather than
    # swallowed so a caller asking for something absent still learns that.
    wanted = resolve_kind(kind, catalog)
    unmatched = kind.strip() if (kind.strip() and wanted is None) else ""
    if unmatched:
        wanted = ""
    return {
        "success": True,
        "mechanisms": {wanted: catalog[wanted]} if wanted else catalog,
        **(
            {
                "note": (
                    f"{unmatched!r} is not one of the categories this build "
                    "describes ("
                    + ", ".join(sorted(catalog))
                    + "), so everything is listed instead."
                )
            }
            if unmatched
            else {}
        ),
        "how_to_use": (
            'Pass a class as {"caches": {"l2": {"prefetcher": '
            '"StridePrefetcher"}}}, or with parameters as {"class": '
            '"StridePrefetcher", "params": {"degree": 8}}. Attaching to L2 is '
            "the reliable place to start: gem5's default L1 mshrs=4 leaves a "
            "prefetcher no capacity to issue into, and it measures as no "
            "mechanism at all."
        ),
    }
