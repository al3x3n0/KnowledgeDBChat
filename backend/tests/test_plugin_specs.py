"""Promoting a contributed tool to one the model can see.

The property under test throughout is that a contributor's *claims* never
decide anything that matters. A manifest says what a tool is called and what it
does; the executor type decides what it is allowed to do, and the name is
checked against what a provider will actually accept.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.agent_core import plugin_specs as ps
from app.agent_core import tool_specs
from app.agent_core.tool_specs.spec import ToolSpec


def _tool(**overrides):
    base = dict(
        name="note",
        description="Record a note",
        tool_type="transform",
        parameters_schema={"type": "object", "properties": {}},
    )
    base.update(overrides)
    return SimpleNamespace(**base)


# --------------------------------------------------------------------------
# Governance is derived, not declared
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tool_type,effects,network",
    [
        ("transform", "read", "none"),
        ("webhook", "write", "external"),
        ("python", "write", "none"),
        ("docker_container", "write", "external"),
        ("llm_prompt", "read", "external"),
    ],
)
def test_the_executor_decides_what_a_tool_may_do(tool_type, effects, network):
    spec = ps.spec_from_user_tool(_tool(tool_type=tool_type), slug="acme")

    assert spec.effects == effects
    assert spec.network == network


def test_an_unknown_executor_is_treated_as_the_most_dangerous():
    """The case we know least about is not the case to be lenient with.

    ``ToolSpec``'s own defaults are lenient, which is right for a built-in --
    one that declares nothing was still written in this repository. A
    contributed tool of an executor type we do not recognise has had no such
    review.
    """
    spec = ps.spec_from_user_tool(_tool(tool_type="wasm_module"), slug="acme")

    assert spec.effects == "write"
    assert spec.network == "external"
    assert spec.cost_tier == "high"


def test_a_contributed_tool_is_offered_to_no_job_type_by_default():
    """Omitting job_types must not mean "every job type".

    ``ToolSpec.job_types is None`` means every job type, which is the right
    default for a measurement tool nobody owns and the wrong one for something
    a contributor can claim by leaving a field out.
    """
    spec = ps.spec_from_user_tool(_tool(), slug="acme")

    assert spec.job_types == ()
    assert "p_acme_note" not in tool_specs.STATIC_CATALOG.extended_with(
        [spec]
    ).tools_for_job_type("research")


# --------------------------------------------------------------------------
# Names
# --------------------------------------------------------------------------


def test_a_name_is_namespaced_exactly_once():
    """Regression: two call sites each namespaced, producing p_bench_p_bench_note.

    The doubled name is still a *valid* function name, for a tool that does not
    exist -- so nothing downstream rejects it, and the failure surfaces as a
    model calling a tool the dispatcher has never heard of.
    """
    already_namespaced = _tool(name="p_bench_note")

    spec = ps.spec_from_user_tool(
        already_namespaced, slug="bench", declared_name="note"
    )

    assert spec.name == "p_bench_note"


@pytest.mark.parametrize(
    "bad,fragment",
    [
        ("p_acme_my tool", "letters, digits, underscore"),
        ("p_acme_wat!", "letters, digits, underscore"),
        ("p_" + "x" * 80, "at most 64"),
        ("", "empty"),
    ],
)
def test_a_name_a_provider_would_reject_is_refused_here(bad, fragment):
    problem = ps.reject_contributed_name(bad)

    assert problem and fragment in problem


def test_a_contributed_tool_may_not_shadow_a_builtin():
    problem = ps.reject_contributed_name(
        "search_documents", reserved=tool_specs.spec_names()
    )

    assert problem and "already a built-in" in problem


# --------------------------------------------------------------------------
# Parameter schemas, which are user-supplied and therefore anything
# --------------------------------------------------------------------------


def test_a_missing_schema_becomes_an_empty_object_not_nothing():
    """A model handed ``{}`` calls the tool with no arguments and no warning."""
    spec = ps.spec_from_user_tool(_tool(parameters_schema=None), slug="acme")

    assert spec.parameters == {"type": "object", "properties": {}}


def test_a_bare_property_map_is_wrapped_rather_than_refused():
    """This is what the existing tool editor produces for a simple tool."""
    spec = ps.spec_from_user_tool(
        _tool(parameters_schema={"text": {"type": "string"}}), slug="acme"
    )

    assert spec.parameters == {
        "type": "object",
        "properties": {"text": {"type": "string"}},
    }


# --------------------------------------------------------------------------
# Promoting a batch never raises
# --------------------------------------------------------------------------


def test_one_unusable_tool_does_not_cost_a_job_the_others():
    tools = [_tool(name="good"), _tool(name="bad name!"), _tool(name="also_good")]

    specs, skipped = ps.specs_from_user_tools(tools, slug="acme")

    assert [s.name for s in specs] == ["p_acme_good", "p_acme_also_good"]
    assert [name for name, _why in skipped] == ["bad name!"]


# --------------------------------------------------------------------------
# The catalog the specs go into
# --------------------------------------------------------------------------


def test_extending_the_catalog_leaves_the_builtins_alone():
    """One user's plugin must never become another user's tool."""
    before = len(tool_specs.STATIC_CATALOG.all_specs())
    extra = ToolSpec(name="p_acme_x", description="d", parameters={})

    extended = tool_specs.STATIC_CATALOG.extended_with([extra])

    assert len(extended.all_specs()) == before + 1
    assert len(tool_specs.STATIC_CATALOG.all_specs()) == before
    assert tool_specs.spec_for("p_acme_x") is None


def test_the_catalog_refuses_a_spec_that_would_shadow_a_builtin():
    shadow = ToolSpec(name="search_documents", description="evil", parameters={})

    with pytest.raises(ValueError, match="may not be overridden"):
        tool_specs.STATIC_CATALOG.extended_with([shadow])


# --------------------------------------------------------------------------
# The namespace is reserved in both directions
# --------------------------------------------------------------------------


def test_no_builtin_occupies_the_contributed_namespace():
    """The provider claims every ``p_`` name on sight, because resolution needs
    the database and ``can_handle`` is synchronous.

    That is only sound while no built-in lives in the namespace. A built-in
    named ``p_something`` would be routed to the plugin provider, which would
    answer "no enabled plugin contributes this" for a tool that does exist --
    a failure that reads as a missing plugin rather than a naming collision,
    and would be found by a user rather than by CI.
    """
    clashing = sorted(
        name for name in tool_specs.spec_names() if name.startswith(ps.NAME_PREFIX)
    )

    assert clashing == [], (
        f"built-in tool(s) {clashing} occupy the {ps.NAME_PREFIX!r} namespace "
        "reserved for contributed tools; rename them"
    )
