"""Where a stalled run is pushed to look next.

Forced exploration defaults to six tools, five of which search documents. That
is a reasonable default for a research job and a detour for a simulation one:
three live runs of a microarchitecture study spent iterations on
`search_documents`, `search_arxiv` and `get_research_findings` in a job with no
document scope at all. Those runs ran out of iterations, not wall clock, and
never reached the second half of the study they were created for.
"""

from app.agent_core.tool_specs import TOOL_DOMAINS, tool_domain
from app.services.autonomous_agent_executor import AutonomousAgentExecutor

MEASUREMENT_LED = {
    "explain_bottleneck": {"success": 2, "failure": 0},
    "measure_headroom": {"success": 2, "failure": 0},
    "search_documents": {"success": 1, "failure": 0},
}

AVAILABLE = [
    "sweep_mechanism",
    "evaluate_across_kernels",
    "simulate_mechanism",
    "search_documents",
    "search_arxiv",
]


class TestTheDomainATooIBelongsTo:
    def test_a_tool_reports_the_module_that_declared_it(self):
        assert tool_domain("simulate_mechanism") == "measurement"
        assert tool_domain("search_arxiv") == "research"

    def test_a_tool_with_no_spec_has_no_domain(self):
        assert tool_domain("not_a_tool") == ""
        assert tool_domain("") == ""

    def test_every_declared_tool_has_one(self):
        """The map is a view of the specs, so a tool missing from it would
        mean a spec that no module declared."""
        assert all(domain for domain in TOOL_DOMAINS.values())


class TestAimingExploration:
    def setup_method(self):
        self.executor = AutonomousAgentExecutor()

    def test_a_measurement_run_explores_measurement_tools(self):
        aligned = self.executor._domain_aligned_exploration_tools(
            MEASUREMENT_LED, AVAILABLE
        )

        assert aligned == [
            "evaluate_across_kernels",
            "simulate_mechanism",
            "sweep_mechanism",
        ]

    def test_a_document_run_still_explores_documents(self):
        """Re-aiming must not break the job type the default was written for."""
        aligned = self.executor._domain_aligned_exploration_tools(
            {"search_documents": {"success": 5}}, AVAILABLE
        )

        assert aligned == ["search_documents"]

    def test_no_history_leaves_the_existing_default_alone(self):
        """In the first iterations there is nothing to aim at, and guessing
        would be worse than the default that at least covers a lot of ground."""
        assert self.executor._domain_aligned_exploration_tools({}, AVAILABLE) == []

    def test_a_run_that_has_only_failed_is_not_aimed_at_its_failures(self):
        """Successes say what is working; failures say the opposite."""
        aligned = self.executor._domain_aligned_exploration_tools(
            {"simulate_mechanism": {"success": 0, "failure": 4}}, AVAILABLE
        )

        assert aligned == []

    def test_malformed_stats_do_not_raise(self):
        assert self.executor._domain_aligned_exploration_tools(None, AVAILABLE) == []
        assert (
            self.executor._domain_aligned_exploration_tools(
                {"x": "not a dict"}, AVAILABLE
            )
            == []
        )


class TestWhoseListItIs:
    def test_a_job_that_named_its_tools_keeps_them(self):
        """Re-aiming applies to the default only. A job that chose its
        exploration tools meant them."""

        class _Job:
            config = {
                "tool_selection_forced_exploration_tools": ["search_arxiv"],
            }

        cfg = AutonomousAgentExecutor()._get_forced_exploration_config(_Job())

        assert cfg["tools"] == ["search_arxiv"]
        assert cfg["tools_are_default"] is False

    def test_a_job_that_said_nothing_gets_a_list_that_may_be_re_aimed(self):
        class _Job:
            config = {}

        cfg = AutonomousAgentExecutor()._get_forced_exploration_config(_Job())

        assert cfg["tools_are_default"] is True
        assert "search_documents" in cfg["tools"]
