"""Tests for extracted goal-contract service."""

from uuid import uuid4

from app.models.agent_job import AgentJob, AgentJobStatus
from app.services.agent_goal_contract_service import AgentGoalContractService
from app.services.autonomous_agent_executor import AutonomousAgentExecutor


def _make_job(config=None) -> AgentJob:
    return AgentJob(
        name="Goal Contract Test",
        goal="Summarize results",
        job_type="research",
        user_id=uuid4(),
        status=AgentJobStatus.RUNNING.value,
        config=config or {},
        max_iterations=20,
        max_tool_calls=20,
        max_llm_calls=20,
        max_runtime_minutes=30,
    )


def test_goal_contract_service_evaluates_required_counts_and_types():
    executor = AutonomousAgentExecutor()
    service = AgentGoalContractService()
    job = _make_job(
        {
            "goal_contract_enabled": True,
            "goal_contract_min_progress": 70,
            "goal_contract_min_findings": 2,
            "goal_contract_required_finding_types": ["paper"],
        }
    )
    state = {
        "goal_progress": 65,
        "findings": [{"type": "document", "id": "d1"}],
        "artifacts": [],
    }

    result = service.evaluate_goal_contract(executor, job, state)

    assert result["enabled"] is True
    assert result["satisfied"] is False
    assert "progress>=70" in result["missing"]
    assert "findings>=2" in result["missing"]
    assert "finding_type:paper" in result["missing"]


def test_goal_contract_service_builds_executive_digest():
    executor = AutonomousAgentExecutor()
    service = AgentGoalContractService()
    job = _make_job({"goal_contract_enabled": True, "goal_contract_min_findings": 2})
    job.results = {
        "summary": "Partial outcome",
        "research_bundle": {"next_steps": ["Validate metrics"]},
    }
    state = {
        "goal_progress": 60,
        "findings": [{"type": "document", "title": "Internal bottleneck"}],
        "artifacts": [{"type": "note", "id": "a1"}],
        "actions_taken": [
            {"action": {"tool": "search_documents"}, "result": {"success": True}},
            {
                "action": {"tool": "search_arxiv"},
                "result": {"success": False, "error": "timeout"},
            },
        ],
        "critic_notes": [{"severity": "high", "pivot": "Need stronger baselines"}],
    }

    digest = service.build_executive_digest(executor, job, state)

    assert digest["outcome"] == "Partial outcome"
    assert digest["metrics"]["failed_actions"] == 1
    assert digest["key_findings"]
    assert digest["risks"]
    assert digest["goal_contract"]["enabled"] is True
    assert digest["goal_contract"]["satisfied"] is False
    assert digest["next_actions"][0] == "Validate metrics"


def test_an_explicit_zero_progress_minimum_is_honoured():
    """A contract can require the deliverable instead of a percentage.

    `min_progress` defaults to 100 and was read as `value or 100`, so setting
    it to 0 -- because the real requirement is the measurement finding types --
    silently demanded 100 instead, and the job could never satisfy its contract
    however much it measured.
    """
    executor = AutonomousAgentExecutor()
    service = AgentGoalContractService()
    job = _make_job(
        {
            "goal_contract": {
                "enabled": True,
                "min_progress": 0,
                "min_findings": 2,
                "required_finding_types": [
                    "codegen_measurement",
                    "benchmark_measurement",
                ],
            }
        }
    )
    state = {
        "goal_progress": 44,
        "findings": [
            {"type": "codegen_measurement"},
            {"type": "benchmark_measurement"},
        ],
        "artifacts": [],
    }

    result = service.evaluate_goal_contract(executor, job, state)

    assert result["missing"] == []
    assert result["satisfied"] is True


def test_the_progress_minimum_still_defaults_to_a_complete_run():
    executor = AutonomousAgentExecutor()
    service = AgentGoalContractService()
    job = _make_job({"goal_contract": {"enabled": True, "min_findings": 1}})
    state = {"goal_progress": 44, "findings": [{"type": "paper"}], "artifacts": []}

    result = service.evaluate_goal_contract(executor, job, state)

    assert result["missing"] == ["progress>=100"]


def test_a_contract_can_say_how_many_of_each_finding_type_it_needs():
    """The contract is also a stopping rule, so it must match the deliverable.

    Asked for four measurements, a job stopped after one: "at least one of
    each type" was satisfied and auto-complete fired at three of four steps.
    """
    executor = AutonomousAgentExecutor()
    service = AgentGoalContractService()
    job = _make_job(
        {
            "goal_contract": {
                "enabled": True,
                "min_progress": 0,
                "required_finding_types": {
                    "codegen_measurement": 2,
                    "benchmark_measurement": 2,
                },
            }
        }
    )
    state = {
        "goal_progress": 44,
        "findings": [
            {"type": "codegen_measurement"},
            {"type": "codegen_measurement"},
            {"type": "benchmark_measurement"},
        ],
        "artifacts": [],
    }

    result = service.evaluate_goal_contract(executor, job, state)

    assert result["missing"] == ["finding_type:benchmark_measurement>=2"]

    state["findings"].append({"type": "benchmark_measurement"})
    assert service.evaluate_goal_contract(executor, job, state)["satisfied"] is True


def test_a_list_of_required_types_still_means_one_of_each():
    executor = AutonomousAgentExecutor()
    service = AgentGoalContractService()
    job = _make_job(
        {
            "goal_contract": {
                "enabled": True,
                "min_progress": 0,
                "required_finding_types": ["codegen_measurement"],
            }
        }
    )
    state = {"goal_progress": 1, "findings": [], "artifacts": []}

    assert service.evaluate_goal_contract(executor, job, state)["missing"] == [
        "finding_type:codegen_measurement"
    ]

    state["findings"].append({"type": "codegen_measurement"})
    assert service.evaluate_goal_contract(executor, job, state)["satisfied"] is True


class TestEvidenceIsTheCompletionCriterion:
    """A contract that names its evidence should not also need a percentage.

    `min_progress` defaulted to 100, so a run could produce every finding its
    contract asked for and still be recorded as falling short on the progress
    number alone. Measured across this deployment's 70 completed contracted
    runs: 17 of them -- a quarter -- were unsatisfied on `progress>=100` and
    nothing else, and only 23 ever reached 100 at all.

    It costs iterations too. A coding run declared success at iteration 7 with
    the suite green and every required finding in hand, was blocked for the
    progress number, and spent three more iterations arriving back where it
    already was.

    So when a contract names evidence, the evidence is the criterion. An
    author who wants a progress bar can still ask for one, and a contract that
    names no evidence keeps the old default -- there, the percentage is the
    only thing it has to go on.
    """

    def _evaluate(self, config, state):
        executor = AutonomousAgentExecutor()
        return AgentGoalContractService().evaluate_goal_contract(
            executor, _make_job(config), state
        )

    def test_required_findings_alone_can_satisfy_a_contract(self):
        result = self._evaluate(
            {
                "goal_contract_enabled": True,
                "goal_contract_required_finding_types": ["test_result"],
            },
            {
                "goal_progress": 40,
                "findings": [{"type": "test_result"}],
                "artifacts": [],
            },
        )
        assert result["satisfied"] is True, result["missing"]

    def test_the_evidence_is_still_required(self):
        # Dropping the progress floor must not drop the point of the contract.
        result = self._evaluate(
            {
                "goal_contract_enabled": True,
                "goal_contract_required_finding_types": ["test_result"],
            },
            {"goal_progress": 100, "findings": [{"type": "document"}], "artifacts": []},
        )
        assert result["satisfied"] is False
        assert any("test_result" in m for m in result["missing"])

    def test_an_explicit_progress_floor_is_still_honoured(self):
        result = self._evaluate(
            {
                "goal_contract_enabled": True,
                "goal_contract_min_progress": 90,
                "goal_contract_required_finding_types": ["test_result"],
            },
            {
                "goal_progress": 40,
                "findings": [{"type": "test_result"}],
                "artifacts": [],
            },
        )
        assert result["satisfied"] is False
        assert any("progress" in m for m in result["missing"])

    def test_a_contract_naming_no_evidence_keeps_the_old_default(self):
        # With nothing else to judge, the percentage is the whole contract.
        result = self._evaluate(
            {"goal_contract_enabled": True, "goal_contract_min_findings": 1},
            {"goal_progress": 50, "findings": [{"type": "document"}], "artifacts": []},
        )
        assert result["satisfied"] is False
        assert any("progress" in m for m in result["missing"])
