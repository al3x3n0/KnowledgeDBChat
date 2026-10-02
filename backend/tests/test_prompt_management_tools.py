"""`switch_strategy`, `set_focus_directive` and `get_available_strategies`.

Each test calls the handler `build_autonomous_output_state_provider` registers,
with the real `AutonomousAgentExecutor` behind it, and judges the tool on what
the next thinking step is then given: the role profile and the focus directive
are only worth setting if the prompt a run is shown changes.
"""

from datetime import datetime
from uuid import uuid4

import pytest

from app.agent_core.tool_specs import agent_ops
from app.models.agent_job import AgentJob
from app.services.agent_tool_dispatch import (
    AgentToolExecutionContext,
    build_autonomous_output_state_provider,
)
from app.services.autonomous_agent_executor import AutonomousAgentExecutor

pytestmark = pytest.mark.unit

ROLES = ["researcher", "critic", "synthesizer", "verifier", "coder", "author"]
TOOLS = ["switch_strategy", "set_focus_directive", "get_available_strategies"]


@pytest.fixture(scope="module")
def executor():
    return AutonomousAgentExecutor()


def _job(config=None, iteration=3):
    return AgentJob(
        id=uuid4(),
        user_id=uuid4(),
        name="Prefetcher study",
        goal="Find out whether stride prefetching helps pointer chasing",
        job_type="research",
        status="running",
        config=config or {},
        iteration=iteration,
        max_iterations=20,
    )


async def _call(executor, tool, params, state=None, job=None):
    """Run one tool; returns (result, state)."""
    state = {} if state is None else state
    job = job or _job()
    provider = build_autonomous_output_state_provider(executor)
    ctx = AgentToolExecutionContext(
        mode="autonomous",
        db=None,
        service=None,
        user_id=str(job.user_id),
        job=job,
        state=state,
    )
    return await provider._handlers[tool](params, ctx), state


def _started_state(executor, job):
    """State as a run has it: the profile is resolved before the first step."""
    state = {}
    state["skill_profile"] = executor._resolve_agent_skill_profile(job, state=state)
    return state


def _system_prompt(executor, job, state):
    """The system prompt as the thinking service builds it for the next step."""
    return executor._build_thinking_prompt_stable(
        job, None, state, profile=state.get("skill_profile")
    )


def _spec(name):
    return next(spec for spec in agent_ops.SPECS if spec.name == name)


class TestSwitchStrategy:
    async def test_the_next_step_is_prompted_as_the_new_role(self, executor):
        job = _job(iteration=5)
        state = _started_state(executor, job)
        assert "ROLE PROFILE: Researcher" in _system_prompt(executor, job, state)

        result, _ = await _call(
            executor,
            "switch_strategy",
            {"role": "critic", "reason": "the claims need challenging"},
            state,
            job,
        )

        critic = executor._resolve_agent_skill_profile(job, override_role="critic")
        assert result["success"] is True
        assert result["data"] == {
            "previous_role": "researcher",
            "new_role": "critic",
            "display_name": "Critic",
            "preferred_tools": critic["preferred_tools"][:5],
        }
        assert state["skill_profile"]["role"] == "critic"
        assert (
            state["skill_profile"]["prompt_directives"] == critic["prompt_directives"]
        )
        prompt = _system_prompt(executor, job, state)
        assert "ROLE PROFILE: Critic" in prompt
        assert critic["prompt_directives"][0] in prompt
        assert "ROLE PROFILE: Researcher" not in prompt

    @pytest.mark.parametrize("role", ROLES)
    async def test_every_declared_role_can_be_switched_to(self, executor, role):
        result, state = await _call(executor, "switch_strategy", {"role": role})
        assert result["success"] is True
        assert result["data"]["new_role"] == role
        assert state["skill_profile"]["role"] == role
        assert len(result["data"]["preferred_tools"]) <= 5

    def test_the_schema_offers_exactly_the_roles_the_tool_accepts(self):
        enum = _spec("switch_strategy").parameters["properties"]["role"]["enum"]
        assert enum == ROLES

    async def test_the_role_is_normalised(self, executor):
        result, state = await _call(executor, "switch_strategy", {"role": "  Coder "})
        assert result["data"]["new_role"] == "coder"
        assert state["skill_profile"]["display_name"] == "Autonomous Coder"

    @pytest.mark.parametrize("params", [{}, {"role": ""}, {"role": "wizard"}])
    async def test_a_missing_or_unknown_role_is_refused(self, executor, params):
        job = _job()
        state = _started_state(executor, job)
        before = dict(state["skill_profile"])

        result, _ = await _call(executor, "switch_strategy", params, state, job)

        assert "Invalid role" in result["error"] and not result.get("success")
        for role in ROLES:
            assert role in result["error"]
        assert state["skill_profile"] == before
        assert "strategy_switches" not in state

    async def test_each_switch_is_logged_with_its_reason(self, executor):
        job = _job(iteration=7)
        state = _started_state(executor, job)
        await _call(
            executor,
            "switch_strategy",
            {"role": "critic", "reason": "r" * 900},
            state,
            job,
        )
        result, _ = await _call(
            executor, "switch_strategy", {"role": "synthesizer"}, state, job
        )

        assert result["data"]["previous_role"] == "critic"
        first, second = state["strategy_switches"]
        assert (first["from"], first["to"]) == ("researcher", "critic")
        assert (second["from"], second["to"]) == ("critic", "synthesizer")
        assert first["reason"] == "r" * 500
        assert second["reason"] == ""
        assert first["iteration"] == 7
        datetime.fromisoformat(first["timestamp"])

    @pytest.mark.xfail(
        strict=True,
        reason="run start re-resolves the profile with the job's configured "
        "role ranked above the switched one, so a resumed run with "
        "agent_role/swarm_role set silently reverts switch_strategy",
    )
    async def test_a_switch_survives_the_run_being_resumed(self, executor):
        job = _job(config={"swarm_role": "researcher"})
        state = _started_state(executor, job)
        await _call(executor, "switch_strategy", {"role": "critic"}, state, job)

        # What _run_autonomous_loop does when a run starts or resumes.
        resumed = executor._resolve_agent_skill_profile(job, state=state)

        assert resumed["role"] == "critic"


class TestSetFocusDirective:
    async def test_the_directive_reaches_the_next_prompt(self, executor):
        job = _job()
        assert "FOCUS DIRECTIVE" not in executor._build_thinking_prompt_volatile(
            job, {}
        )

        result, state = await _call(
            executor,
            "set_focus_directive",
            {"directive": "  Prioritize contradictions between sources  "},
            job=job,
        )

        assert result["success"] is True
        assert result["data"] == {
            "directive": "Prioritize contradictions between sources",
            "mode": "replaced",
        }
        assert state["focus_directive"] == "Prioritize contradictions between sources"
        prompt = executor._build_thinking_prompt_volatile(job, state)
        assert (
            "FOCUS DIRECTIVE (set by agent):\nPrioritize contradictions between sources"
            in prompt
        )

    async def test_it_replaces_by_default(self, executor):
        state = {"focus_directive": "Old focus"}
        result, _ = await _call(
            executor, "set_focus_directive", {"directive": "New focus"}, state
        )
        assert state["focus_directive"] == "New focus"
        assert result["data"]["mode"] == "replaced"

    async def test_append_keeps_the_existing_directive(self, executor):
        job = _job()
        state = {"focus_directive": "Focus on transformers"}
        result, _ = await _call(
            executor,
            "set_focus_directive",
            {"directive": "Also consider RNNs", "append": True},
            state,
            job,
        )

        assert result["data"]["mode"] == "appended"
        assert state["focus_directive"] == "Focus on transformers\nAlso consider RNNs"
        assert result["data"]["directive"] == state["focus_directive"]
        prompt = executor._build_thinking_prompt_volatile(job, state)
        assert "Focus on transformers\nAlso consider RNNs" in prompt

    async def test_append_with_nothing_set_just_sets_it(self, executor):
        _, state = await _call(
            executor, "set_focus_directive", {"directive": "First", "append": True}
        )
        assert state["focus_directive"] == "First"

    @pytest.mark.parametrize("params", [{}, {"directive": ""}, {"directive": "   "}])
    async def test_an_empty_directive_is_refused(self, executor, params):
        state = {"focus_directive": "Keep me"}
        result, _ = await _call(executor, "set_focus_directive", params, state)
        assert "directive" in result["error"] and not result.get("success")
        assert state["focus_directive"] == "Keep me"

    async def test_a_directive_is_capped_at_1000_characters(self, executor):
        result, state = await _call(
            executor, "set_focus_directive", {"directive": "D" * 1500}
        )
        assert state["focus_directive"] == "D" * 1000
        assert result["data"]["directive"] == "D" * 1000

    async def test_appended_directives_are_capped_at_2000_characters(self, executor):
        state = {}
        for letter in "ABC":
            await _call(
                executor,
                "set_focus_directive",
                {"directive": letter * 900, "append": True},
                state,
            )
        assert len(state["focus_directive"]) == 2000
        assert state["focus_directive"].startswith("A" * 900 + "\n" + "B" * 900)

    @pytest.mark.xfail(
        strict=True,
        reason="once 2000 characters are stored, an appended directive is "
        "truncated away entirely and the call still answers success/'appended'",
    )
    async def test_an_append_that_does_not_fit_is_not_reported_as_done(self, executor):
        state = {"focus_directive": "A" * 2000}
        result, _ = await _call(
            executor,
            "set_focus_directive",
            {"directive": "Stop reading surveys", "append": True},
            state,
        )
        stored = "Stop reading surveys" in state["focus_directive"]
        assert stored or (result.get("error") and not result.get("success"))

    @pytest.mark.xfail(
        strict=True,
        reason="a null directive is stringified: the focus directive becomes "
        "the literal text 'None' instead of the call being refused",
    )
    async def test_a_null_directive_is_refused(self, executor):
        state = {"focus_directive": "Keep me"}
        result, _ = await _call(
            executor, "set_focus_directive", {"directive": None}, state
        )
        assert result.get("error")
        assert state["focus_directive"] == "Keep me"


class TestGetAvailableStrategies:
    async def test_it_lists_every_role_switch_strategy_accepts(self, executor):
        job = _job()
        result, state = await _call(executor, "get_available_strategies", {}, job=job)

        assert result["success"] is True
        strategies = result["data"]["strategies"]
        assert [s["role"] for s in strategies] == ROLES
        for entry in strategies:
            profile = executor._resolve_agent_skill_profile(
                job, override_role=entry["role"]
            )
            assert entry["display_name"] == profile["display_name"]
            assert entry["preferred_tools"] == profile["preferred_tools"][:5]
            assert entry["discouraged_tools"] == profile["discouraged_tools"]
            assert entry["guidance"]
            assert len(entry["guidance"]) <= 300
            assert profile["prompt_directives"][0] in entry["guidance"]
        assert state == {}

    async def test_the_roles_are_told_apart(self, executor):
        result, _ = await _call(executor, "get_available_strategies", {})
        strategies = result["data"]["strategies"]
        assert len({s["display_name"] for s in strategies}) == len(ROLES)
        assert len({s["guidance"] for s in strategies}) == len(ROLES)

    async def test_the_current_role_follows_a_switch(self, executor):
        job = _job()
        state = _started_state(executor, job)
        result, _ = await _call(executor, "get_available_strategies", {}, state, job)
        assert result["data"]["current_role"] == "researcher"

        await _call(executor, "switch_strategy", {"role": "verifier"}, state, job)
        result, _ = await _call(executor, "get_available_strategies", {}, state, job)

        assert result["data"]["current_role"] == "verifier"
        assert state["skill_profile"]["role"] == "verifier"

    async def test_guidance_carries_every_directive_of_the_role(self, executor):
        job = _job()
        result, _ = await _call(executor, "get_available_strategies", {}, job=job)
        for entry in result["data"]["strategies"]:
            profile = executor._resolve_agent_skill_profile(
                job, override_role=entry["role"]
            )
            for directive in profile["prompt_directives"]:
                assert directive in entry["guidance"], entry["role"]


class TestHandoffContractPrompt:
    """`create_handoff` writes the contract; this is the prompt that reads it."""

    def test_a_handoff_contract_is_in_the_system_prompt(self, executor):
        job = _job(
            config={
                "handoff_contract": {
                    "from_job_id": "parent-123",
                    "context": "We found 5 papers on transformers. " + "C" * 2000,
                    "expected_outputs": [f"output_{i}" for i in range(15)],
                }
            }
        )
        prompt = executor._build_thinking_prompt_stable(job, None, {})

        assert "HANDOFF CONTRACT (from parent agent):" in prompt
        assert "We found 5 papers on transformers." in prompt
        assert "C" * 965 in prompt and "C" * 966 not in prompt  # 1000 in all
        assert "output_0, output_1" in prompt
        assert "output_9" in prompt and "output_10" not in prompt
        assert "You MUST produce results that satisfy the expected outputs." in prompt

    def test_no_contract_means_no_section(self, executor):
        prompt = executor._build_thinking_prompt_stable(_job(), None, {})
        assert "HANDOFF CONTRACT" not in prompt


class TestPromptManagementSchemas:
    """Tests for prompt management tool schema definitions."""

    def test_schemas_exist(self):
        from app.services.agent_tools import AGENT_TOOLS

        names = {t["name"] for t in AGENT_TOOLS}
        assert set(TOOLS) <= names

    def test_switch_strategy_requires_role(self):
        from app.services.agent_tools import get_tool_by_name

        tool = get_tool_by_name("switch_strategy")
        assert tool is not None
        assert "role" in tool["parameters"].get("required", [])

    def test_switch_strategy_has_role_enum(self):
        from app.services.agent_tools import get_tool_by_name

        tool = get_tool_by_name("switch_strategy")
        assert tool["parameters"]["properties"]["role"]["enum"] == ROLES

    def test_set_focus_directive_requires_directive(self):
        from app.services.agent_tools import get_tool_by_name

        tool = get_tool_by_name("set_focus_directive")
        assert tool is not None
        assert "directive" in tool["parameters"].get("required", [])

    def test_set_focus_directive_has_append(self):
        from app.services.agent_tools import get_tool_by_name

        tool = get_tool_by_name("set_focus_directive")
        assert tool["parameters"]["properties"]["append"]["type"] == "boolean"

    def test_get_available_strategies_no_required(self):
        from app.services.agent_tools import get_tool_by_name

        tool = get_tool_by_name("get_available_strategies")
        assert tool is not None
        assert tool["parameters"].get("required", []) == []


class TestPromptManagementRegistry:
    """Tests for prompt management tool registry classification."""

    @pytest.mark.parametrize("tool_name", TOOLS)
    def test_classification(self, tool_name):
        from app.services.tool_registry import get_tool_metadata

        meta = get_tool_metadata(tool_name)
        assert meta is not None
        assert meta.effects == "read"
        assert meta.cost_tier == "low"
        assert meta.network == "none"
