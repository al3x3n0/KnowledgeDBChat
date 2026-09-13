"""Direct tests for the extracted job-type tool policy.

Pure policy tables: they read only the job type and its config, so they are
exercised here without constructing an executor.
"""

import pytest

from app.models.agent_job import AgentJob
from app.services.agent_job_tool_policy import (
    get_tool_selection_config,
    get_tools_for_job_type,
)

SPECIALIZED_JOB_TYPES = [
    "research",
    "coding",
    "analysis",
    "monitor",
    "synthesis",
    "knowledge_expansion",
]


def _job(**config) -> AgentJob:
    return AgentJob(
        name="Policy Test",
        goal="Exercise the tool policy",
        job_type="research",
        config=config,
    )


# An unrecognised job type contributes no extras, so it yields the base set.
# "custom" is a configured type with extras of its own, not the base.
BASE_TOOLS = set(get_tools_for_job_type("no_such_job_type", {}))


def test_every_job_type_gets_the_shared_base_tools():
    assert {"search_documents", "save_research_finding"} <= BASE_TOOLS
    for job_type in SPECIALIZED_JOB_TYPES + ["custom"]:
        assert BASE_TOOLS <= set(get_tools_for_job_type(job_type, {}))


@pytest.mark.parametrize("job_type", SPECIALIZED_JOB_TYPES + ["custom"])
def test_specialized_job_types_add_tools_beyond_the_base(job_type):
    tools = set(get_tools_for_job_type(job_type, {}))

    assert tools - BASE_TOOLS, f"{job_type} should grant tools beyond the base"


def test_research_grants_the_paper_tools_and_coding_does_not():
    research = set(get_tools_for_job_type("research", {}))
    coding = set(get_tools_for_job_type("coding", {}))

    assert {"compare_methodologies", "batch_ingest_papers"} <= research
    assert "compare_methodologies" not in coding


def test_unknown_job_type_falls_back_to_the_base_set():
    assert set(get_tools_for_job_type("no_such_type", {})) == BASE_TOOLS


def test_allowlist_and_denylist_narrow_the_tool_set():
    allowed = get_tools_for_job_type(
        "research", {"allowed_tools": ["search_documents", "search_arxiv"]}
    )
    assert set(allowed) <= {"search_documents", "search_arxiv"}

    denied = get_tools_for_job_type("research", {"blocked_tools": ["search_documents"]})
    assert "search_documents" not in denied


def test_role_profile_puts_preferred_tools_first_and_drops_blocked_ones():
    ordered = get_tools_for_job_type(
        "research",
        {},
        profile={
            "preferred_tools": ["search_arxiv"],
            "blocked_tools": ["search_documents"],
        },
    )

    assert ordered[0] == "search_arxiv"
    assert "search_documents" not in ordered


def test_tool_list_has_no_duplicates():
    for job_type in SPECIALIZED_JOB_TYPES + ["custom"]:
        tools = get_tools_for_job_type(job_type, {})
        assert len(tools) == len(set(tools)), f"{job_type} repeats a tool"


def test_missing_config_is_tolerated():
    assert get_tools_for_job_type("research", None)


def test_selection_config_defaults_are_adaptive():
    config = get_tool_selection_config(_job())

    assert config["policy_mode"] == "adaptive"
    assert config["exploration_enabled"] is True
    assert 0.0 <= config["exploration_bonus"] <= 1.0


def test_selection_config_is_overridable_per_job():
    config = get_tool_selection_config(
        _job(
            tool_selection_policy_mode="thompson",
            tool_selection_exploration_bonus=0.4,
        )
    )

    assert config["policy_mode"] == "thompson"
    assert config["exploration_bonus"] == 0.4


def test_selection_config_clamps_out_of_range_overrides():
    config = get_tool_selection_config(_job(tool_selection_exploration_bonus=99))

    assert config["exploration_bonus"] == 2.0


def test_selection_config_ignores_unparseable_overrides():
    config = get_tool_selection_config(
        _job(
            tool_selection_policy_mode="not-a-mode",
            tool_selection_exploration_bonus="abc",
        )
    )

    assert config["policy_mode"] == "adaptive"
    assert config["exploration_bonus"] == 0.15
