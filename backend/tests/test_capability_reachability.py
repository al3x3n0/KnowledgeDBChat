"""Guards against capabilities that exist but cannot be reached.

One defect shape has now been found by hand four times in this project, and
each instance looked like finished work from the outside:

  - ``traces_one_regime`` was implemented, tested and documented, and the
    job-config normaliser dropped it, because its allowlist was a second
    registration point that nobody updated.
  - Two ``describe()`` functions carried methodology text nothing called.
  - Four counter tools existed and the evidence map did not list them, so a
    contract asking for their findings was told no tool produced them.
  - The counter sampler's schema offered five of the nine parameters its
    service accepts, so SMT experiments and C++ corpora -- the two things the
    tool had just been extended for -- were unreachable from the loop.

Nothing fails when this happens. A capability that no surface offers is not a
broken capability; it is an absent one, and absence produces no error, no log
line and no failing test. That is why every instance was found by reading.

What the four have in common is mechanical: a capability is defined in one
place and *registered* in another, and the two lists are maintained by hand.
The tests here assert the registrations agree. They deliberately check only
exact set equalities, not heuristics -- an earlier audit of the ``describe()``
class produced three false alarms out of four candidates, and a guard with that
signal-to-noise is worse than no guard, because it trains its readers to
override it.

The universes below are the real ones. Deriving a guard's universe from a
single registry is how this class evades its own guards: the metadata guard in
test_tool_effects_classification read AGENT_TOOLS, which is precisely the
registry the 21 data-analysis tools are not in, so they escaped the check
written to catch tools with no metadata.
"""

from typing import Set

import pytest

from app.agent_core.tool_catalog import iter_builtin_tools
from app.services import agent_evidence_map, agent_job_tool_policy
from app.services import agent_tool_dispatch as dispatch
from app.services.agent_tools import AGENT_TOOLS
from app.services.data_analysis_tools import exposed_data_analysis_tools

# Job types with their own tool list in agent_job_tool_policy.
JOB_TYPES = ("research", "coding", "data_analysis", "analysis", "custom")


class _Stub:
    """Stands in for the executor or service a provider is built around.

    The builders only close over it; nothing is called at build time.
    """

    def __getattr__(self, _name: str) -> "_Stub":
        return self

    def __call__(self, *_args, **_kwargs) -> "_Stub":
        return self


def advertised_tools() -> Set[str]:
    """Every tool name a run can be shown, across all registries that hold one."""
    names = {
        str(tool.get("name") or "").strip()
        for tool in AGENT_TOOLS
        if str(tool.get("name") or "").strip()
    }
    names |= {str(name).strip() for name in exposed_data_analysis_tools()}
    return names


def dispatchable_tools() -> Set[str]:
    """Every tool name a provider will actually answer to."""
    names: Set[str] = set()
    builders = [name for name in dir(dispatch) if name.startswith("build_")]
    assert builders, "no provider builders found; this guard would pass vacuously"
    for builder in builders:
        provider = getattr(dispatch, builder)(_Stub())
        names |= set(provider.supported_tools)
    return names


def test_every_advertised_tool_dispatches():
    """A tool a run is invited to call must have something that answers it."""
    orphaned = sorted(advertised_tools() - dispatchable_tools())
    assert not orphaned, (
        "These tools are advertised to a run but no provider handles them, so "
        "calling one fails at dispatch:\n" + "\n".join(f"  - {n}" for n in orphaned)
    )


def test_every_dispatchable_tool_is_advertised():
    """A capability no surface offers does not exist as far as a run is concerned.

    This is the direction that fails silently. ``create_chart_from_dataset``
    was dispatchable and advertised nowhere: the dataset-backed charting tool
    was renamed to end a name collision, and the prompt builder and the policy
    lists went on using the old name, so the capability the rename was meant to
    rescue became unreachable instead.
    """
    unreachable = sorted(dispatchable_tools() - advertised_tools())
    assert not unreachable, (
        "These tools can be dispatched but no registry offers them to a run, "
        "so nothing can ever call them:\n" + "\n".join(f"  - {n}" for n in unreachable)
    )


def test_no_tool_name_is_offered_by_two_registries():
    """One name, one contract.

    ``create_chart`` was defined in both registries with different parameters.
    Provider resolution is first-wins, so every call conforming to the
    advertised contract was answered by the other tool and failed on a field
    the run had not been asked for.
    """
    builtin = {
        str(tool.get("name") or "").strip()
        for tool in AGENT_TOOLS
        if str(tool.get("name") or "").strip()
    }
    collisions = sorted(builtin & set(exposed_data_analysis_tools()))
    assert not collisions, (
        "These names are defined in AGENT_TOOLS and in the data-analysis "
        "registry with different parameters; a run is shown one and gets the "
        "other:\n" + "\n".join(f"  - {n}" for n in collisions)
    )


def test_the_governed_universe_is_the_executable_universe():
    """Whatever a run can execute, tool governance must be able to see.

    The catalog is what the tool-policy UI lists, what effects classification
    reads, and what ``_constraints_ok`` fails closed on. A tool missing from it
    is ungovernable and denied as unknown wherever constraints are configured
    -- and, because the other guards in this suite derive their universe from
    it, invisible to them too.
    """
    ungoverned = sorted(dispatchable_tools() - {m.name for m in iter_builtin_tools()})
    assert (
        not ungoverned
    ), "These executable tools have no catalog metadata:\n" + "\n".join(
        f"  - {n}" for n in ungoverned
    )


@pytest.mark.parametrize("job_type", JOB_TYPES)
def test_job_type_tool_lists_name_real_tools(job_type):
    """A policy list is filtered against the supported set, so a name that is
    stale or misspelled is not an error -- it is a capability the job type
    quietly does without."""
    proposed = agent_job_tool_policy.get_tools_for_job_type(job_type, {})
    assert proposed, f"{job_type} proposes no tools at all"
    unknown = sorted(set(proposed) - dispatchable_tools())
    assert not unknown, f"{job_type} proposes tools nothing dispatches:\n" + "\n".join(
        f"  - {n}" for n in unknown
    )


def test_data_analysis_can_chart_a_dataset():
    """The specific reachability this suite was written after, pinned.

    A data_analysis job is the only job type that can build a dataset, so it is
    the only one for which dataset-backed charting means anything.
    """
    proposed = agent_job_tool_policy.get_tools_for_job_type("data_analysis", {})
    assert "create_chart_from_dataset" in proposed


def test_evidence_map_names_real_tools():
    """A chain derived from the evidence map has to be executable.

    The map is consulted to plan backwards from what a contract demands; an
    entry naming a tool that does not exist yields a plan that cannot run, and
    a tool missing from the map is a tool the planner believes produces
    nothing.
    """
    mapped = {entry.tool for entry in agent_evidence_map.EVIDENCE_TOOLS}
    assert mapped, "the evidence map is empty"
    phantom = sorted(mapped - dispatchable_tools())
    assert not phantom, (
        "The evidence map promises evidence from tools that do not exist:\n"
        + "\n".join(f"  - {n}" for n in phantom)
    )
