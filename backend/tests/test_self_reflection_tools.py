"""The self-reflection tools: get_job_history, get_job_metrics,
get_tool_usage_stats, get_tool_failure_analysis.

These call the real handlers against real `AgentJob` rows. The file used to
restate each handler inline and assert on the restatement -- `limit =
min(int(params.get("limit", 10) or 10), 50); assert limit == 50` -- so
thirty-four tests passed whatever the tools did.

Most tests here seed `execution_log` with the entry shape the handlers read
(`{"action": <tool>, "error": <text>}`), which pins their arithmetic. The
tests under `TestAgainstTheLogTheExecutorWrites` seed it with what the
executor actually appends, which is a different shape.
"""

from datetime import datetime, timedelta
from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.models.agent_job import AgentJob
from app.services.agent_tool_dispatch import (
    AgentToolExecutionContext,
    build_autonomous_observability_provider,
)

pytestmark = pytest.mark.unit

TOOLS = [
    "get_job_history",
    "get_job_metrics",
    "get_tool_usage_stats",
    "get_tool_failure_analysis",
]


async def _job(db, user, **fields):
    fields.setdefault("name", "job")
    fields.setdefault("goal", "a goal")
    fields.setdefault("job_type", "research")
    job = AgentJob(user_id=user.id, **fields)
    db.add(job)
    await db.commit()
    await db.refresh(job)
    return job


async def _run(tool, params, db, job):
    provider = build_autonomous_observability_provider(SimpleNamespace())
    ctx = AgentToolExecutionContext(
        mode="autonomous",
        db=db,
        service=None,
        user_id=str(job.user_id),
        job=job,
        state={},
    )
    return await provider._handlers[tool](params, ctx)


def _ok(tool, n=1):
    return [{"action": tool} for _ in range(n)]


def _failed(tool, error, n=1):
    return [{"action": tool, "error": error} for _ in range(n)]


class TestGetJobHistory:
    async def test_no_earlier_jobs_is_an_empty_history(self, db_session, test_user):
        current = await _job(db_session, test_user)

        result = await _run("get_job_history", {}, db_session, current)

        assert result == {"success": True, "data": {"jobs": [], "count": 0}}

    async def test_it_lists_the_users_other_jobs_newest_first(
        self, db_session, test_user, admin_user
    ):
        now = datetime.utcnow()
        old = await _job(
            db_session, test_user, goal="old", created_at=now - timedelta(days=3)
        )
        new = await _job(
            db_session, test_user, goal="new", created_at=now - timedelta(days=1)
        )
        await _job(db_session, admin_user, goal="someone else's")
        current = await _job(db_session, test_user, goal="current")

        result = await _run("get_job_history", {}, db_session, current)

        assert result["success"] is True
        assert [j["id"] for j in result["data"]["jobs"]] == [str(new.id), str(old.id)]
        assert result["data"]["count"] == 2

    async def test_a_job_reports_its_usage_and_duration(self, db_session, test_user):
        started = datetime(2026, 3, 20, 10, 0, 0)
        past = await _job(
            db_session,
            test_user,
            goal="G" * 500,
            job_type="analysis",
            status="failed",
            iteration=15,
            tool_calls_used=42,
            llm_calls_used=30,
            tokens_used=50000,
            error="E" * 500,
            started_at=started,
            completed_at=started + timedelta(minutes=45, seconds=30),
        )
        current = await _job(db_session, test_user)

        result = await _run("get_job_history", {}, db_session, current)

        (row,) = result["data"]["jobs"]
        assert row["id"] == str(past.id)
        assert row["job_type"] == "analysis" and row["status"] == "failed"
        assert row["iteration"] == 15
        assert row["tool_calls_used"] == 42
        assert row["llm_calls_used"] == 30
        assert row["tokens_used"] == 50000
        assert row["duration_minutes"] == 45.5
        assert row["goal"] == "G" * 200
        assert row["error"] == "E" * 200
        assert row["created_at"] and row["completed_at"]

    async def test_an_unfinished_job_has_no_duration_and_no_error(
        self, db_session, test_user
    ):
        await _job(db_session, test_user, started_at=datetime(2026, 3, 20, 10, 0, 0))
        current = await _job(db_session, test_user)

        result = await _run("get_job_history", {}, db_session, current)

        (row,) = result["data"]["jobs"]
        assert row["duration_minutes"] is None
        assert row["completed_at"] is None
        assert row["error"] is None

    async def test_job_type_and_status_filter(self, db_session, test_user):
        wanted = await _job(
            db_session, test_user, job_type="research", status="completed"
        )
        await _job(db_session, test_user, job_type="research", status="failed")
        await _job(db_session, test_user, job_type="coding", status="completed")
        current = await _job(db_session, test_user)

        by_type = await _run(
            "get_job_history", {"job_type": "research"}, db_session, current
        )
        by_status = await _run(
            "get_job_history", {"status": "completed"}, db_session, current
        )
        both = await _run(
            "get_job_history",
            {"job_type": " research ", "status": "completed"},
            db_session,
            current,
        )

        assert by_type["data"]["count"] == 2
        assert by_status["data"]["count"] == 2
        assert [j["id"] for j in both["data"]["jobs"]] == [str(wanted.id)]

    async def test_limit_defaults_to_10_and_is_capped_at_50(
        self, db_session, test_user
    ):
        db_session.add_all(
            [AgentJob(user_id=test_user.id, name=f"j{i}", goal="g") for i in range(55)]
        )
        await db_session.commit()
        current = await _job(db_session, test_user)

        default = await _run("get_job_history", {}, db_session, current)
        three = await _run("get_job_history", {"limit": 3}, db_session, current)
        huge = await _run("get_job_history", {"limit": 200}, db_session, current)

        assert default["data"]["count"] == 10
        assert three["data"]["count"] == 3 and len(three["data"]["jobs"]) == 3
        assert huge["data"]["count"] == 50

    async def test_a_negative_limit_does_not_lift_the_cap(self, db_session, test_user):
        db_session.add_all(
            [AgentJob(user_id=test_user.id, name=f"j{i}", goal="g") for i in range(55)]
        )
        await db_session.commit()
        current = await _job(db_session, test_user)

        result = await _run("get_job_history", {"limit": -1}, db_session, current)

        assert "error" in result or result["data"]["count"] <= 50

    async def test_a_limit_that_is_not_a_number_is_refused(self, db_session, test_user):
        current = await _job(db_session, test_user)

        result = await _run("get_job_history", {"limit": "many"}, db_session, current)

        assert "success" not in result
        assert result["error"].startswith("Failed to get job history")


class TestGetJobMetrics:
    async def test_it_defaults_to_the_current_job(self, db_session, test_user):
        started = datetime(2026, 3, 20, 10, 0, 0)
        current = await _job(
            db_session,
            test_user,
            goal="Analyze papers",
            job_type="analysis",
            status="completed",
            iteration=20,
            tool_calls_used=45,
            llm_calls_used=40,
            tokens_used=80000,
            error_count=2,
            started_at=started,
            completed_at=started + timedelta(minutes=15, seconds=30),
            execution_log=_ok("search_documents", 3)
            + _ok("summarize_document")
            + [{"phase": "replan"}],
        )

        result = await _run("get_job_metrics", {}, db_session, current)

        assert result["success"] is True
        data = result["data"]
        assert data["id"] == str(current.id)
        assert data["goal"] == "Analyze papers"
        assert data["status"] == "completed" and data["job_type"] == "analysis"
        assert data["iterations"] == 20
        assert data["tool_calls_used"] == 45
        assert data["llm_calls_used"] == 40
        assert data["tokens_used"] == 80000
        assert data["max_tool_calls"] == 500
        assert data["max_llm_calls"] == 200
        assert data["max_runtime_minutes"] == 60
        assert data["error_count"] == 2
        assert data["duration_minutes"] == 15.5
        assert data["tool_usage_breakdown"] == {
            "search_documents": 3,
            "summarize_document": 1,
        }
        assert data["started_at"] and data["completed_at"] and data["created_at"]

    async def test_a_job_with_no_log_and_no_end_has_empty_metrics(
        self, db_session, test_user
    ):
        current = await _job(db_session, test_user)

        result = await _run("get_job_metrics", {}, db_session, current)

        assert result["success"] is True
        assert result["data"]["tool_usage_breakdown"] == {}
        assert result["data"]["duration_minutes"] is None
        assert result["data"]["started_at"] is None
        assert result["data"]["completed_at"] is None

    async def test_it_reads_another_of_the_users_jobs_by_id(
        self, db_session, test_user
    ):
        other = await _job(
            db_session, test_user, goal="earlier", execution_log=_ok("search_web", 2)
        )
        current = await _job(db_session, test_user, goal="now")

        result = await _run(
            "get_job_metrics", {"job_id": f" {other.id} "}, db_session, current
        )

        assert result["data"]["id"] == str(other.id)
        assert result["data"]["goal"] == "earlier"
        assert result["data"]["tool_usage_breakdown"] == {"search_web": 2}

    async def test_another_users_job_is_refused(
        self, db_session, test_user, admin_user
    ):
        theirs = await _job(db_session, admin_user, goal="secret goal")
        current = await _job(db_session, test_user)

        result = await _run(
            "get_job_metrics", {"job_id": str(theirs.id)}, db_session, current
        )

        assert "success" not in result and "data" not in result
        assert "Not authorized" in result["error"]
        assert "secret goal" not in str(result)

    async def test_a_malformed_job_id_is_refused(self, db_session, test_user):
        current = await _job(db_session, test_user)

        result = await _run(
            "get_job_metrics", {"job_id": "not-valid"}, db_session, current
        )

        assert result == {"error": "Invalid job_id format"}

    async def test_an_unknown_job_id_is_not_found(self, db_session, test_user):
        current = await _job(db_session, test_user)
        missing = str(uuid4())

        result = await _run("get_job_metrics", {"job_id": missing}, db_session, current)

        assert result == {"error": f"Job not found: {missing}"}


class TestGetToolUsageStats:
    async def test_no_history_is_no_tools(self, db_session, test_user):
        current = await _job(db_session, test_user)

        result = await _run("get_tool_usage_stats", {}, db_session, current)

        assert result == {
            "success": True,
            "data": {"period_days": 7, "total_jobs_analyzed": 0, "tools": []},
        }

    async def test_it_aggregates_calls_and_success_rates_across_jobs(
        self, db_session, test_user
    ):
        await _job(
            db_session,
            test_user,
            execution_log=_ok("search_documents", 2)
            + _failed("search_documents", "timeout")
            + _failed("search_web", "503 Service Unavailable"),
        )
        current = await _job(
            db_session,
            test_user,
            execution_log=_ok("get_document_details")
            + _failed("search_web", "refused")
            + [{"phase": "replan"}, {"action": None}],
        )

        result = await _run("get_tool_usage_stats", {}, db_session, current)

        assert result["data"]["total_jobs_analyzed"] == 2
        tools = result["data"]["tools"]
        assert [t["name"] for t in tools][0] == "search_documents"
        assert {t["name"]: t for t in tools} == {
            "search_documents": {
                "name": "search_documents",
                "calls": 3,
                "successes": 2,
                "failures": 1,
                "success_rate": 0.667,
            },
            "search_web": {
                "name": "search_web",
                "calls": 2,
                "successes": 0,
                "failures": 2,
                "success_rate": 0.0,
            },
            "get_document_details": {
                "name": "get_document_details",
                "calls": 1,
                "successes": 1,
                "failures": 0,
                "success_rate": 1.0,
            },
        }

    async def test_tools_are_sorted_by_calls(self, db_session, test_user):
        current = await _job(
            db_session,
            test_user,
            execution_log=_ok("a", 5) + _ok("b", 20) + _ok("c", 10),
        )

        result = await _run("get_tool_usage_stats", {}, db_session, current)

        assert [(t["name"], t["calls"]) for t in result["data"]["tools"]] == [
            ("b", 20),
            ("c", 10),
            ("a", 5),
        ]

    async def test_tool_name_narrows_to_one_tool(self, db_session, test_user):
        current = await _job(
            db_session,
            test_user,
            execution_log=_ok("search_documents", 2) + _ok("search_web", 4),
        )

        result = await _run(
            "get_tool_usage_stats",
            {"tool_name": " search_documents "},
            db_session,
            current,
        )

        assert [(t["name"], t["calls"]) for t in result["data"]["tools"]] == [
            ("search_documents", 2)
        ]

    async def test_other_users_and_older_jobs_are_left_out(
        self, db_session, test_user, admin_user
    ):
        now = datetime.utcnow()
        await _job(db_session, admin_user, execution_log=_ok("theirs", 9))
        await _job(
            db_session,
            test_user,
            created_at=now - timedelta(days=10),
            execution_log=_ok("ten_days_ago", 4),
        )
        current = await _job(db_session, test_user, execution_log=_ok("today"))

        week = await _run("get_tool_usage_stats", {}, db_session, current)
        fortnight = await _run(
            "get_tool_usage_stats", {"days": 14}, db_session, current
        )

        assert [t["name"] for t in week["data"]["tools"]] == ["today"]
        assert week["data"]["total_jobs_analyzed"] == 1
        assert fortnight["data"]["period_days"] == 14
        assert {t["name"] for t in fortnight["data"]["tools"]} == {
            "today",
            "ten_days_ago",
        }

    async def test_days_is_capped_at_30(self, db_session, test_user):
        now = datetime.utcnow()
        await _job(
            db_session,
            test_user,
            created_at=now - timedelta(days=45),
            execution_log=_ok("long_ago"),
        )
        current = await _job(
            db_session,
            test_user,
            created_at=now - timedelta(days=20),
            execution_log=_ok("recent"),
        )

        result = await _run("get_tool_usage_stats", {"days": 90}, db_session, current)

        assert result["data"]["period_days"] == 30
        assert [t["name"] for t in result["data"]["tools"]] == ["recent"]

    async def test_at_most_50_tools_are_returned(self, db_session, test_user):
        log = []
        for i in range(60):
            log += _ok(f"tool_{i:02d}", 60 - i)
        current = await _job(db_session, test_user, execution_log=log)

        result = await _run("get_tool_usage_stats", {}, db_session, current)

        tools = result["data"]["tools"]
        assert len(tools) == 50
        assert tools[0] == {
            "name": "tool_00",
            "calls": 60,
            "successes": 60,
            "failures": 0,
            "success_rate": 1.0,
        }
        assert tools[-1]["name"] == "tool_49"


class TestGetToolFailureAnalysis:
    @pytest.mark.parametrize("params", [{}, {"tool_name": ""}, {"tool_name": "   "}])
    async def test_tool_name_is_required(self, db_session, test_user, params):
        current = await _job(db_session, test_user)

        result = await _run("get_tool_failure_analysis", params, db_session, current)

        assert result == {"error": "tool_name is required"}

    async def test_a_tool_never_called_has_a_zero_rate(self, db_session, test_user):
        current = await _job(db_session, test_user, execution_log=_ok("other", 3))

        result = await _run(
            "get_tool_failure_analysis",
            {"tool_name": "search_web"},
            db_session,
            current,
        )

        assert result == {
            "success": True,
            "data": {
                "tool_name": "search_web",
                "period_days": 7,
                "total_calls": 0,
                "total_failures": 0,
                "failure_rate": 0.0,
                "error_patterns": [],
                "recent_failures": [],
            },
        }

    async def test_it_counts_failures_and_groups_them_by_pattern(
        self, db_session, test_user
    ):
        current = await _job(
            db_session,
            test_user,
            job_type="research",
            iteration=4,
            execution_log=_ok("search_web", 4)
            + _failed("search_web", "Rate limit exceeded")
            + _failed("search_web", "Connection timeout after 30s", 4)
            + _failed("other_tool", "unrelated"),
        )

        result = await _run(
            "get_tool_failure_analysis",
            {"tool_name": "search_web"},
            db_session,
            current,
        )

        data = result["data"]
        assert data["total_calls"] == 9
        assert data["total_failures"] == 5
        assert data["failure_rate"] == 0.556
        assert [(p["pattern"], p["count"]) for p in data["error_patterns"]] == [
            ("Connection timeout after 30s", 4),
            ("Rate limit exceeded", 1),
        ]
        examples = data["error_patterns"][0]["examples"]
        assert len(examples) == 3
        assert examples[0]["job_id"] == str(current.id)
        assert examples[0]["job_type"] == "research"
        assert examples[0]["error"] == "Connection timeout after 30s"
        assert len(data["recent_failures"]) == 5

    async def test_long_errors_are_truncated_and_grouped_on_their_prefix(
        self, db_session, test_user
    ):
        current = await _job(
            db_session,
            test_user,
            execution_log=_failed("t", "E" * 500) + _failed("t", "E" * 90 + "tail"),
        )

        result = await _run(
            "get_tool_failure_analysis", {"tool_name": "t"}, db_session, current
        )

        data = result["data"]
        assert data["failure_rate"] == 1.0
        assert [(p["pattern"], p["count"]) for p in data["error_patterns"]] == [
            ("E" * 80, 2)
        ]
        assert data["recent_failures"][0]["error"] == "E" * 200

    async def test_patterns_and_recent_failures_are_capped_at_10(
        self, db_session, test_user
    ):
        log = []
        for i in range(25):
            log += _failed("t", f"err-{i:02d}", 2 if i == 24 else 1)
        current = await _job(db_session, test_user, execution_log=log)

        result = await _run(
            "get_tool_failure_analysis", {"tool_name": "t"}, db_session, current
        )

        data = result["data"]
        assert data["total_failures"] == 26
        assert len(data["error_patterns"]) == 10
        assert data["error_patterns"][0]["pattern"] == "err-24"
        assert [f["error"] for f in data["recent_failures"]] == [
            f"err-{i:02d}" for i in range(16, 25)
        ] + ["err-24"]

    async def test_recent_failures_are_the_most_recent_across_jobs(
        self, db_session, test_user
    ):
        now = datetime.utcnow()
        # The newer job is stored first, so storage order is not time order.
        current = await _job(
            db_session,
            test_user,
            created_at=now - timedelta(hours=1),
            execution_log=_failed("t", "new failure", 10),
        )
        await _job(
            db_session,
            test_user,
            created_at=now - timedelta(days=5),
            execution_log=_failed("t", "old failure", 10),
        )

        result = await _run(
            "get_tool_failure_analysis", {"tool_name": "t"}, db_session, current
        )

        assert {f["error"] for f in result["data"]["recent_failures"]} == {
            "new failure"
        }

    async def test_other_users_and_older_jobs_are_left_out(
        self, db_session, test_user, admin_user
    ):
        now = datetime.utcnow()
        await _job(db_session, admin_user, execution_log=_failed("t", "theirs", 5))
        await _job(
            db_session,
            test_user,
            created_at=now - timedelta(days=45),
            execution_log=_failed("t", "long ago", 5),
        )
        current = await _job(
            db_session,
            test_user,
            created_at=now - timedelta(days=20),
            execution_log=_ok("t") + _failed("t", "mine"),
        )

        week = await _run(
            "get_tool_failure_analysis", {"tool_name": "t"}, db_session, current
        )
        capped = await _run(
            "get_tool_failure_analysis",
            {"tool_name": "t", "days": 90},
            db_session,
            current,
        )

        assert week["data"]["period_days"] == 7 and week["data"]["total_calls"] == 0
        assert capped["data"]["period_days"] == 30
        assert capped["data"]["total_calls"] == 2
        assert capped["data"]["failure_rate"] == 0.5
        assert [f["error"] for f in capped["data"]["recent_failures"]] == ["mine"]


class TestAgainstTheLogTheExecutorWrites:
    """The same tools, fed the entries `autonomous_agent_executor` appends.

    Per iteration it writes `{"phase": "iteration_complete", "action": <tool>,
    "success": ..., "error": ...}`; a failing call that repeats adds
    `{"phase": "repeated_tool_failure", "tool": ..., "error_class": ...}`.
    Operator decisions write `{"phase": "swarm_review_decision", "action":
    "approve"}`. Entries go through the real `AgentJob.add_log_entry`.
    """

    @staticmethod
    async def _job_with_two_failed_calls(db, user):
        job = AgentJob(user_id=user.id, name="j", goal="g", job_type="research")
        for attempt in (1, 2):
            if attempt == 2:
                job.add_log_entry(
                    {
                        "phase": "repeated_tool_failure",
                        "tool": "search_arxiv",
                        "attempt": attempt,
                        "error_class": "http_406",
                    }
                )
            job.add_log_entry(
                {
                    "phase": "iteration_complete",
                    "action": "search_arxiv",
                    "success": False,
                    "error": "HTTP 406 from export.arxiv.org",
                    "progress": 0,
                    "findings_count": 0,
                }
            )
        db.add(job)
        await db.commit()
        await db.refresh(job)
        return job

    async def test_a_failed_call_lowers_the_success_rate(self, db_session, test_user):
        job = await self._job_with_two_failed_calls(db_session, test_user)

        result = await _run("get_tool_usage_stats", {}, db_session, job)

        (tool,) = result["data"]["tools"]
        assert tool["name"] == "search_arxiv" and tool["calls"] == 2
        assert tool["failures"] == 2
        assert tool["success_rate"] == 0.0

    async def test_a_failed_call_appears_in_the_failure_analysis(
        self, db_session, test_user
    ):
        job = await self._job_with_two_failed_calls(db_session, test_user)

        result = await _run(
            "get_tool_failure_analysis", {"tool_name": "search_arxiv"}, db_session, job
        )

        assert result["data"]["total_calls"] == 2
        assert result["data"]["total_failures"] == 2
        assert result["data"]["error_patterns"]

    async def test_an_operator_decision_is_not_a_tool_call(self, db_session, test_user):
        job = AgentJob(user_id=test_user.id, name="j", goal="g")
        job.add_log_entry({"phase": "iteration_complete", "action": "search_web"})
        job.add_log_entry(
            {"phase": "swarm_review_decision", "action": "approve", "note": "ok"}
        )
        db_session.add(job)
        await db_session.commit()
        await db_session.refresh(job)

        metrics = await _run("get_job_metrics", {}, db_session, job)
        stats = await _run("get_tool_usage_stats", {}, db_session, job)

        assert metrics["data"]["tool_usage_breakdown"] == {"search_web": 1}
        assert [t["name"] for t in stats["data"]["tools"]] == ["search_web"]


class TestSelfReflectionToolSchemas:
    """Tests for self-reflection tool schema definitions."""

    def test_schemas_exist(self):
        from app.services.agent_tools import AGENT_TOOLS

        names = {t["name"] for t in AGENT_TOOLS}
        assert set(TOOLS) <= names

    @pytest.mark.parametrize(
        "tool_name", ["get_job_history", "get_job_metrics", "get_tool_usage_stats"]
    )
    def test_nothing_is_required(self, tool_name):
        from app.services.agent_tools import get_tool_by_name

        tool = get_tool_by_name(tool_name)
        assert tool is not None
        assert tool["parameters"].get("required", []) == []

    def test_get_tool_failure_analysis_requires_tool_name(self):
        from app.services.agent_tools import get_tool_by_name

        tool = get_tool_by_name("get_tool_failure_analysis")
        assert tool is not None
        assert "tool_name" in tool["parameters"].get("required", [])

    def test_every_tool_has_a_handler(self):
        provider = build_autonomous_observability_provider(SimpleNamespace())
        assert set(TOOLS) <= set(provider._handlers)


class TestSelfReflectionToolRegistry:
    """Tests for self-reflection tool registry classification."""

    @pytest.mark.parametrize("tool_name", TOOLS)
    def test_read_only_low_cost_and_offline(self, tool_name):
        from app.services.tool_registry import get_tool_metadata

        meta = get_tool_metadata(tool_name)
        assert meta is not None
        assert meta.effects == "read", f"{tool_name} should be read-only"
        assert meta.cost_tier == "low", f"{tool_name} should be low cost"
        assert meta.network == "none", f"{tool_name} should not require network"
