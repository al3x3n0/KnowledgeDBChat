"""The conditional tools: evaluate_condition, count_findings, check_goal_status.

These call the real handlers. The file used to restate each handler's logic
inline and assert on the restatement -- `state = {"findings": [...]}; count =
len(state.get("findings", [])); assert count >= threshold` -- so thirty tests
passed whatever the tools did, including while `documents_count` could never
report more than two documents and `plan_steps_total` was 0 for every plan a
real run has ever had.

The executor is the real `AutonomousAgentExecutor`. Its search service is the
real `SearchService`; only the vector store underneath it -- the external
edge -- is replaced, by a store that honours `limit` the way a real one does.
State is `initialize_runtime_state()`, which is what a run hands a tool.
"""

import copy
from types import SimpleNamespace

import pytest

from app.models.agent_job import AgentJob
from app.services.agent_runtime_state_service import initialize_runtime_state
from app.services.agent_tool_dispatch import (
    AgentToolExecutionContext,
    build_autonomous_observability_provider,
)
from app.services.agent_tools import AGENT_TOOLS, get_tool_by_name
from app.services.autonomous_agent_executor import AutonomousAgentExecutor
from app.services.tool_registry import get_tool_metadata

pytestmark = pytest.mark.unit

TOOLS = ("evaluate_condition", "count_findings", "check_goal_status")

CONDITIONS = tuple(
    get_tool_by_name("evaluate_condition")["parameters"]["properties"]["condition"][
        "enum"
    ]
)


class FakeVectorStore:
    """The index behind SearchService: one chunk per document."""

    def __init__(self, documents=(), fail=False):
        self.documents = list(documents)
        self.fail = fail
        self.calls = []

    async def initialize(self, *args, **kwargs):
        return None

    async def search(
        self, query, limit=10, filter_metadata=None, apply_postprocessing=True
    ):
        self.calls.append(
            {"query": query, "limit": limit, "filter_metadata": filter_metadata}
        )
        if self.fail:
            raise RuntimeError("vector store unreachable")
        wanted = filter_metadata or {}
        hits = []
        for index, doc in enumerate(self.documents):
            if any(doc.get(key) != value for key, value in wanted.items()):
                continue
            if query != "*" and query.lower() not in doc["content"].lower():
                continue
            hits.append(
                {
                    "id": f"chunk-{index}",
                    "content": doc["content"],
                    "score": 1.0 - index * 0.01,
                    "metadata": {
                        "document_id": f"doc-{index}",
                        "title": f"Doc {index}",
                        "source_id": doc.get("source_id"),
                    },
                }
            )
        return hits[:limit]


def _documents(count, source_id="src-a", content="sparse attention"):
    return [{"content": content, "source_id": source_id} for _ in range(count)]


def _executor(documents=(), fail=False):
    executor = AutonomousAgentExecutor()
    executor.search_service.vector_store = FakeVectorStore(documents, fail=fail)
    return executor


def _job(**fields):
    fields.setdefault("id", "job-1")
    fields.setdefault("user_id", "u")
    fields.setdefault("config", {})
    fields.setdefault("iteration", 0)
    fields.setdefault("max_iterations", 100)
    fields.setdefault("tool_calls_used", 0)
    fields.setdefault("max_tool_calls", 500)
    return SimpleNamespace(**fields)


async def _run(tool, params, state, executor=None, job=None):
    provider = build_autonomous_observability_provider(executor or _executor())
    ctx = AgentToolExecutionContext(
        mode="autonomous",
        db=None,
        service=None,
        user_id="u",
        job=job or _job(),
        state=state,
    )
    return await provider._handlers[tool](params, ctx)


def _state(**overrides):
    """The state a real run starts with."""
    state = initialize_runtime_state()
    state.update(overrides)
    return state


def _findings(*categories, confidence=0.8):
    return [
        {
            "id": f"f{i}",
            "title": f"finding {i}",
            "category": category,
            "confidence": confidence,
        }
        for i, category in enumerate(categories)
    ]


def _actions(count):
    return [
        {
            "action": {"tool": "search_documents"},
            "result": {"success": True, "data": {}},
            "iteration": i,
            "node": "act",
        }
        for i in range(count)
    ]


async def _evaluate(params, state=None, executor=None):
    result = await _run(
        "evaluate_condition", params, _state() if state is None else state, executor
    )
    assert result.get("success") is True, result
    return result["data"]


class TestEvaluateConditionRefusals:
    @pytest.mark.parametrize("params", [{}, {"condition": ""}, {"condition": "  "}])
    async def test_a_condition_is_required(self, params):
        result = await _run("evaluate_condition", params, _state())
        assert "success" not in result
        assert "condition" in result["error"].lower()

    async def test_an_unknown_condition_is_refused_and_names_every_valid_one(self):
        result = await _run(
            "evaluate_condition", {"condition": "nonexistent"}, _state()
        )
        assert "success" not in result
        assert "nonexistent" in result["error"]
        for condition in CONDITIONS:
            assert condition in result["error"]

    @pytest.mark.parametrize("condition", CONDITIONS)
    async def test_every_advertised_condition_is_implemented(self, condition):
        data = await _evaluate(
            {"condition": condition, "category": "result", "query": "attention"},
            executor=_executor(_documents(1)),
        )
        assert data["condition"] == condition
        assert isinstance(data["met"], bool)

    async def test_a_threshold_that_is_not_a_number_is_refused(self):
        result = await _run(
            "evaluate_condition",
            {"condition": "findings_count", "threshold": "many"},
            _state(findings=_findings("a", "b")),
        )
        assert "success" not in result
        assert "error" in result

    async def test_search_has_results_requires_a_query(self):
        executor = _executor(_documents(3))
        result = await _run(
            "evaluate_condition",
            {"condition": "search_has_results"},
            _state(),
            executor,
        )
        assert "query" in result["error"]
        assert executor.search_service.vector_store.calls == []

    async def test_findings_has_category_requires_a_category(self):
        result = await _run(
            "evaluate_condition",
            {"condition": "findings_has_category"},
            _state(findings=_findings("result")),
        )
        assert "success" not in result
        assert "category" in result["error"]

    async def test_a_failing_search_is_reported_not_raised(self):
        result = await _run(
            "evaluate_condition",
            {"condition": "documents_count"},
            _state(),
            _executor(fail=True),
        )
        assert "success" not in result
        assert "vector store unreachable" in result["error"]

    async def test_evaluating_does_not_change_the_state(self):
        state = _state(
            findings=_findings("a", "b"), actions_taken=_actions(3), goal_progress=40
        )
        before = copy.deepcopy(state)
        for condition in ("findings_count", "actions_count", "progress_above"):
            await _evaluate({"condition": condition}, state)
        await _evaluate({"condition": "findings_has_category", "category": "a"}, state)
        assert state == before


class TestEvaluateConditionOnState:
    @pytest.mark.parametrize(
        "threshold, met", [(2, True), (3, True), (4, False), (100, False)]
    )
    async def test_findings_count_is_met_at_or_above_the_threshold(
        self, threshold, met
    ):
        data = await _evaluate(
            {"condition": "findings_count", "threshold": threshold},
            _state(findings=_findings("a", "b", "c")),
        )
        assert data == {
            "met": met,
            "actual": 3,
            "threshold": threshold,
            "condition": "findings_count",
        }

    async def test_the_threshold_defaults_to_one(self):
        data = await _evaluate(
            {"condition": "findings_count"}, _state(findings=_findings("a"))
        )
        assert data["threshold"] == 1
        assert data["met"] is True

    async def test_a_fresh_run_has_no_findings(self):
        data = await _evaluate({"condition": "findings_count"})
        assert data["actual"] == 0
        assert data["met"] is False

    async def test_a_numeric_string_threshold_compares_as_a_number(self):
        # "10" must not compare as text, where it would sort below "3".
        data = await _evaluate(
            {"condition": "findings_count", "threshold": "10"},
            _state(findings=_findings("a", "b", "c")),
        )
        assert data["threshold"] == 10
        assert data["met"] is False

    async def test_a_threshold_of_zero_is_honoured(self):
        data = await _evaluate({"condition": "findings_count", "threshold": 0})
        assert data["threshold"] == 0
        assert data["met"] is True

    async def test_findings_has_category_counts_only_that_category(self):
        state = _state(findings=_findings("key_insight", "methodology", "key_insight"))
        data = await _evaluate(
            {"condition": "findings_has_category", "category": "key_insight"}, state
        )
        assert data["actual"] == 2
        assert data["category"] == "key_insight"
        assert data["met"] is True

    async def test_findings_has_category_applies_the_threshold_to_the_matches(self):
        state = _state(findings=_findings("result", "result", "other", "other"))
        data = await _evaluate(
            {
                "condition": "findings_has_category",
                "category": "result",
                "threshold": 3,
            },
            state,
        )
        assert data["actual"] == 2
        assert data["met"] is False

    async def test_findings_has_category_absent_category_is_not_met(self):
        data = await _evaluate(
            {"condition": "findings_has_category", "category": "key_insight"},
            _state(findings=_findings("methodology")),
        )
        assert data["actual"] == 0
        assert data["met"] is False

    @pytest.mark.parametrize("threshold, met", [(3, True), (5, True), (6, False)])
    async def test_actions_count(self, threshold, met):
        data = await _evaluate(
            {"condition": "actions_count", "threshold": threshold},
            _state(actions_taken=_actions(5)),
        )
        assert data["actual"] == 5
        assert data["met"] is met

    @pytest.mark.parametrize(
        "progress, threshold, met",
        [(75, 50, True), (50, 50, True), (20, 50, False), (0, 1, False)],
    )
    async def test_progress_above(self, progress, threshold, met):
        data = await _evaluate(
            {"condition": "progress_above", "threshold": threshold},
            _state(goal_progress=progress),
        )
        assert data["actual"] == progress
        assert data["met"] is met

    async def test_a_fresh_run_has_made_no_progress(self):
        data = await _evaluate({"condition": "progress_above"})
        assert data["actual"] == 0
        assert data["met"] is False


class TestEvaluateConditionOnTheKnowledgeBase:
    async def test_documents_count_on_an_empty_knowledge_base(self):
        data = await _evaluate({"condition": "documents_count"})
        assert data["actual"] == 0
        assert data["met"] is False

    async def test_documents_count_is_met_when_a_document_exists(self):
        data = await _evaluate(
            {"condition": "documents_count"}, executor=_executor(_documents(1))
        )
        assert data["actual"] == 1
        assert data["met"] is True

    async def test_documents_count_counts_every_document(self):
        data = await _evaluate(
            {"condition": "documents_count", "threshold": 5},
            executor=_executor(_documents(10)),
        )
        assert data["actual"] == 10
        assert data["met"] is True

    async def test_documents_count_is_filtered_by_source(self):
        executor = _executor(_documents(1, source_id="src-a"))
        data = await _evaluate(
            {"condition": "documents_count", "source_id": "src-b"}, executor=executor
        )
        assert data["actual"] == 0
        assert data["met"] is False
        assert executor.search_service.vector_store.calls[-1]["filter_metadata"] == {
            "source_id": "src-b"
        }

    async def test_search_has_results_runs_the_query_given(self):
        executor = _executor(
            _documents(1, content="sparse attention")
            + _documents(1, content="loop unrolling")
        )
        data = await _evaluate(
            {"condition": "search_has_results", "query": "unrolling"}, executor=executor
        )
        assert data["query"] == "unrolling"
        assert data["actual"] == 1
        assert data["met"] is True
        assert executor.search_service.vector_store.calls[-1]["query"] == "unrolling"

    async def test_search_has_results_with_no_match_is_not_met(self):
        data = await _evaluate(
            {"condition": "search_has_results", "query": "quantum"},
            executor=_executor(_documents(4)),
        )
        assert data["actual"] == 0
        assert data["met"] is False

    async def test_search_has_results_applies_the_threshold_to_every_match(self):
        """The search reports only what it fetched, so `actual` is a floor:
        enough to judge the threshold, not a count of every match."""
        data = await _evaluate(
            {"condition": "search_has_results", "query": "attention", "threshold": 3},
            executor=_executor(_documents(10)),
        )
        assert data["actual"] >= 3
        assert data["met"] is True


async def _count(params, findings):
    result = await _run("count_findings", params, _state(findings=findings))
    assert result.get("success") is True, result
    return result["data"]


class TestCountFindings:
    async def test_a_fresh_run_counts_nothing(self):
        result = await _run("count_findings", {}, _state())
        assert result == {
            "success": True,
            "data": {"total": 0, "by_category": {}, "categories": []},
        }

    async def test_counts_everything_grouped_by_category(self):
        data = await _count(
            {}, _findings("key_insight", "methodology", "key_insight", "result")
        )
        assert data["total"] == 4
        assert data["by_category"] == {"key_insight": 2, "methodology": 1, "result": 1}
        assert sorted(data["categories"]) == ["key_insight", "methodology", "result"]

    async def test_category_filter(self):
        data = await _count(
            {"category": "key_insight"},
            _findings("key_insight", "methodology", "key_insight"),
        )
        assert data["total"] == 2
        assert data["by_category"] == {"key_insight": 2}

    async def test_a_category_nothing_has(self):
        data = await _count({"category": "absent"}, _findings("key_insight"))
        assert data == {"total": 0, "by_category": {}, "categories": []}

    async def test_min_confidence_filter(self):
        findings = [
            {"category": "a", "confidence": 0.9},
            {"category": "a", "confidence": 0.3},
            {"category": "b", "confidence": 0.5},
        ]
        data = await _count({"min_confidence": 0.5}, findings)
        assert data["total"] == 2
        assert data["by_category"] == {"a": 1, "b": 1}

    async def test_category_and_confidence_combine(self):
        findings = [
            {"category": "a", "confidence": 0.9},
            {"category": "a", "confidence": 0.3},
            {"category": "b", "confidence": 0.9},
        ]
        data = await _count({"category": "a", "min_confidence": 0.5}, findings)
        assert data["total"] == 1
        assert data["by_category"] == {"a": 1}

    async def test_min_confidence_given_as_text_compares_as_a_number(self):
        findings = [{"category": "a", "confidence": 0.9}] + [
            {"category": "a", "confidence": 0.2}
        ]
        data = await _count({"min_confidence": "0.5"}, findings)
        assert data["total"] == 1

    async def test_a_min_confidence_that_is_not_a_number_is_refused(self):
        result = await _run(
            "count_findings",
            {"min_confidence": "high"},
            _state(findings=_findings("a")),
        )
        assert "success" not in result
        assert "error" in result

    async def test_a_finding_without_a_confidence_counts_at_the_save_default(self):
        # save_finding stores 0.8 when the caller gives none.
        findings = [{"category": "a"}]
        assert (await _count({"min_confidence": 0.8}, findings))["total"] == 1
        assert (await _count({"min_confidence": 0.81}, findings))["total"] == 0

    async def test_a_zero_confidence_finding_is_below_any_positive_floor(self):
        findings = [{"category": "a", "confidence": 0.0}]
        data = await _count({"min_confidence": 0.5}, findings)
        assert data["total"] == 0

    async def test_a_finding_with_no_category_key_is_uncategorized(self):
        data = await _count({}, [{"title": "t", "confidence": 0.9}])
        assert data["by_category"] == {"uncategorized": 1}

    async def test_a_finding_saved_without_a_category_is_uncategorized(self):
        data = await _count({}, [{"title": "t", "category": None, "confidence": 0.9}])
        assert data["by_category"] == {"uncategorized": 1}

    async def test_counting_does_not_change_the_state(self):
        state = _state(findings=_findings("a", "b"))
        before = copy.deepcopy(state)
        await _run("count_findings", {"category": "a", "min_confidence": 0.5}, state)
        assert state == before


async def _status(state, job=None):
    result = await _run("check_goal_status", {}, state, job=job)
    assert result.get("success") is True, result
    return result["data"]


class TestCheckGoalStatus:
    async def test_a_fresh_run(self):
        data = await _status(_state())
        assert data == {
            "iteration": 0,
            "max_iterations": 100,
            "iterations_remaining": 100,
            "tool_calls_used": 0,
            "max_tool_calls": 500,
            "tool_calls_remaining": 500,
            "goal_progress": 0,
            "findings_count": 0,
            "actions_count": 0,
            "has_execution_plan": False,
            "plan_steps_completed": 0,
            "plan_steps_total": 0,
            "goal_contract_enabled": False,
        }

    async def test_the_remaining_budget_is_what_is_left(self):
        job = _job(
            iteration=7, max_iterations=20, tool_calls_used=15, max_tool_calls=50
        )
        data = await _status(_state(), job)
        assert data["iterations_remaining"] == 13
        assert data["tool_calls_remaining"] == 35

    async def test_a_stored_job_reports_its_own_budget(self, db_session, test_user):
        job = AgentJob(
            user_id=test_user.id,
            name="job",
            goal="a goal",
            job_type="research",
            max_iterations=12,
            max_tool_calls=40,
        )
        db_session.add(job)
        await db_session.commit()
        await db_session.refresh(job)
        job.iteration = 3
        job.tool_calls_used = 9

        data = await _status(_state(), job)
        assert data["iterations_remaining"] == 9
        assert data["tool_calls_remaining"] == 31

    async def test_it_reports_what_the_run_has_gathered(self):
        state = _state(
            findings=_findings("a", "b", "c"),
            actions_taken=_actions(8),
            goal_progress=35,
        )
        data = await _status(state)
        assert data["findings_count"] == 3
        assert data["actions_count"] == 8
        assert data["goal_progress"] == 35

    def _planned_state(self, executor, completed=1):
        """A plan stored the way the executor stores one: a list of steps."""
        plan = executor._annotate_execution_plan_graph(
            [
                {"description": "search", "tool": "search_documents"},
                {"description": "read", "tool": "get_document_details"},
                {"description": "write up", "tool": "save_finding"},
            ]
        )
        assert len(plan) == 3
        return _state(execution_plan=plan, plan_step_index=completed)

    async def test_a_run_with_a_plan_says_so_and_how_far_it_is(self):
        executor = _executor()
        result = await _run(
            "check_goal_status",
            {},
            self._planned_state(executor, completed=1),
            executor,
        )
        assert result["data"]["has_execution_plan"] is True
        assert result["data"]["plan_steps_completed"] == 1

    async def test_the_plan_length_is_the_number_of_steps(self):
        executor = _executor()
        result = await _run(
            "check_goal_status", {}, self._planned_state(executor), executor
        )
        assert result["data"]["plan_steps_total"] == 3

    async def test_an_unmet_contract_names_what_is_missing(self):
        state = _state(
            goal_contract_last={
                "enabled": True,
                "satisfied": False,
                "missing": ["min_findings: 1/3", "finding_type: benchmark_result"],
            }
        )
        data = await _status(state)
        assert data["goal_contract_enabled"] is True
        assert data["goal_contract_satisfied"] is False
        assert data["goal_contract_missing"] == [
            "min_findings: 1/3",
            "finding_type: benchmark_result",
        ]

    async def test_a_met_contract_is_reported_as_satisfied(self):
        state = _state(
            goal_contract_last={"enabled": True, "satisfied": True, "missing": []}
        )
        data = await _status(state)
        assert data["goal_contract_satisfied"] is True
        assert data["goal_contract_missing"] == []

    async def test_the_missing_list_is_capped_at_ten(self):
        state = _state(
            goal_contract_last={
                "enabled": True,
                "satisfied": False,
                "missing": [f"requirement {i}" for i in range(25)],
            }
        )
        data = await _status(state)
        assert data["goal_contract_missing"] == [f"requirement {i}" for i in range(10)]

    async def test_a_disabled_contract_reports_only_that(self):
        state = _state(goal_contract_last={"enabled": False, "missing": ["x"]})
        data = await _status(state)
        assert data["goal_contract_enabled"] is False
        assert "goal_contract_missing" not in data

    async def test_checking_does_not_change_the_state(self):
        state = _state(findings=_findings("a"), actions_taken=_actions(2))
        before = copy.deepcopy(state)
        await _status(state)
        assert state == before


class TestConditionalToolSchemas:
    def test_schemas_exist(self):
        names = {t["name"] for t in AGENT_TOOLS}
        assert set(TOOLS) <= names

    def test_evaluate_condition_requires_condition(self):
        tool = get_tool_by_name("evaluate_condition")
        assert tool["parameters"].get("required", []) == ["condition"]

    def test_evaluate_condition_advertises_six_conditions(self):
        assert set(CONDITIONS) == {
            "findings_count",
            "findings_has_category",
            "documents_count",
            "search_has_results",
            "actions_count",
            "progress_above",
        }

    @pytest.mark.parametrize("name", ["count_findings", "check_goal_status"])
    def test_the_others_require_nothing(self, name):
        tool = get_tool_by_name(name)
        assert tool is not None
        assert tool["parameters"].get("required", []) == []

    def test_one_provider_answers_all_three(self):
        provider = build_autonomous_observability_provider(SimpleNamespace())
        assert set(TOOLS) <= set(provider._handlers)


class TestConditionalToolRegistry:
    @pytest.mark.parametrize("name", TOOLS)
    def test_classification(self, name):
        meta = get_tool_metadata(name)
        assert meta is not None
        assert meta.effects == "read"
        assert meta.cost_tier == "low"

    @pytest.mark.parametrize("name", ["count_findings", "check_goal_status"])
    def test_the_state_only_tools_touch_no_network(self, name):
        assert get_tool_metadata(name).network == "none"
