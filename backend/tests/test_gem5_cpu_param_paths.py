"""Walking to a parameter inside the staged gem5 config script.

`_apply_cpu_params` lives inside `MECH_CONFIG_SCRIPT`, a string staged into the
sandbox and run by gem5's own Python, so it is exercised here by pulling the
function out of the string and running it against stand-in objects. Testing it
through a simulation would cost minutes per case and would not say which path
shape failed.

The paths matter because gem5 25.1 buried the things a study wants to vary:
the issue queue moved to `instQueues[N]`, and functional unit timing is three
levels down at `instQueues[0].fuPool.FUList[3].opList[3].opLat`. A walker that
handled one level could name the issue queue and nothing about the units that
do the work.
"""

from __future__ import annotations

import re

import pytest

from app.services.agent_gem5_mechanism import MECH_CONFIG_SCRIPT


class Params(dict):
    """Stands in for a SimObject's `_params` mapping."""


def _load():
    """The script's own `_apply_cpu_params`, with its SpecError."""
    ns: dict = {}
    exec("class SpecError(Exception):\n    pass\n", ns)
    fn = re.search(
        r"\ndef _apply_cpu_params\(.*?(?=\ndef |\Z)", MECH_CONFIG_SCRIPT, re.S
    ).group(0)
    ns["MANIFEST"] = {"applied": [], "cpu": {}}
    exec(fn, ns)
    return ns["_apply_cpu_params"], ns["SpecError"], ns["MANIFEST"]


class Node:
    """A SimObject-ish node: named params, and attributes that may be vectors."""

    def __init__(self, params=None, **children):
        type(self)._params = Params(params or {})
        for k, v in children.items():
            setattr(self, k, v)

    def __init_subclass__(cls, **kw):  # pragma: no cover - not subclassed
        super().__init_subclass__(**kw)


def _cpu():
    """A cpu with a plain param, a vector of queues, and units three deep."""

    class Op:
        _params = Params({"opLat": 1, "pipelined": True})

    class FU:
        _params = Params({"count": 1})

    class IQ:
        _params = Params({"numEntries": 64})

    op_a, op_b = Op(), Op()
    fu = FU()
    fu.opList = [op_a, op_b]
    iq0, iq1 = IQ(), IQ()
    iq0.fuPool = type("Pool", (), {"_params": Params()})()
    iq0.fuPool.FUList = [fu]
    iq1.fuPool = type("Pool", (), {"_params": Params()})()
    iq1.fuPool.FUList = [fu]

    class CPU:
        _params = Params({"numROBEntries": 192})

    cpu = CPU()
    cpu.instQueues = [iq0, iq1]
    return cpu, op_a, op_b, iq0, iq1


class TestPathsAStudyNeeds:
    def test_a_plain_parameter_on_the_cpu(self):
        apply, _, _ = _load()
        cpu, *_ = _cpu()
        apply(cpu, {"numROBEntries": 512})
        assert cpu.numROBEntries == 512

    def test_one_level_of_vector_indexing_still_works(self):
        """The shape that existed before; breaking it would strand issue-queue
        studies."""
        apply, _, _ = _load()
        cpu, _, _, iq0, iq1 = _cpu()
        apply(cpu, {"instQueues[*].numEntries": 512})
        assert iq0.numEntries == 512 and iq1.numEntries == 512

    def test_a_functional_unit_three_levels_down(self):
        apply, _, _ = _load()
        cpu, op_a, op_b, *_ = _cpu()
        apply(cpu, {"instQueues[0].fuPool.FUList[0].opList[1].opLat": 24})
        assert op_b.opLat == 24
        assert not hasattr(op_a, "opLat") or op_a.opLat != 24

    def test_a_star_partway_down_fans_out(self):
        apply, _, _ = _load()
        cpu, op_a, op_b, *_ = _cpu()
        apply(cpu, {"instQueues[*].fuPool.FUList[0].opList[0].opLat": 7})
        assert op_a.opLat == 7


class TestWhatItRefuses:
    def test_an_unknown_parameter_on_the_cpu_names_where_things_moved(self):
        apply, SpecError, _ = _load()
        cpu, *_ = _cpu()
        with pytest.raises(SpecError) as exc:
            apply(cpu, {"numIQEntries": 64})
        assert "instQueues[*].numEntries" in str(exc.value)
        assert "fuPool" in str(exc.value)

    def test_a_missing_step_says_which_step(self):
        apply, SpecError, _ = _load()
        cpu, *_ = _cpu()
        with pytest.raises(SpecError) as exc:
            apply(cpu, {"nosuch.fuPool.FUList[0].opList[0].opLat": 1})
        assert "nosuch" in str(exc.value)

    def test_an_index_past_the_end_says_how_many_there_are(self):
        apply, SpecError, _ = _load()
        cpu, *_ = _cpu()
        with pytest.raises(SpecError) as exc:
            apply(cpu, {"instQueues[99].numEntries": 1})
        assert "out of range" in str(exc.value)

    def test_a_path_ending_at_a_vector_names_no_parameter(self):
        apply, SpecError, _ = _load()
        cpu, *_ = _cpu()
        with pytest.raises(SpecError) as exc:
            apply(cpu, {"instQueues[*]": 1})
        assert "names no parameter" in str(exc.value)

    def test_an_unknown_parameter_on_a_nested_object_lists_what_it_has(self):
        apply, SpecError, _ = _load()
        cpu, *_ = _cpu()
        with pytest.raises(SpecError) as exc:
            apply(cpu, {"instQueues[0].fuPool.FUList[0].opList[0].nosuch": 1})
        assert "opLat" in str(exc.value)
