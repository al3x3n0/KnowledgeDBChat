from fastapi.testclient import TestClient

from app.schemas.langgraph_issue_pr import (
    ExecutorOutput,
    LangGraphIssuePrResponse,
    PlannerOutput,
    ResearcherOutput,
    ReviewerOutput,
)
from app.services.langgraph_issue_pr_service import LangGraphIssuePrService


def _payload() -> dict:
    return {
        "issue": {
            "id": "431",
            "title": "Fix API timeout regression",
            "body": "Requests fail when retry counter resets incorrectly.",
            "labels": ["bug", "backend"],
        },
        "constraints": ["must add tests"],
        "repo_context": {"default_branch": "main"},
        "policy_profile": {},
    }


def test_langgraph_issue_pr_draft_success(
    client: TestClient,
    auth_headers: dict,
    monkeypatch,
):
    async def _run(self, request, **kwargs):
        return LangGraphIssuePrResponse(
            status="pr_ready",
            reason="ok",
            status_reason_code="passed",
            planner=PlannerOutput(),
            researcher=ResearcherOutput(),
            executor=ExecutorOutput(),
            reviewer=ReviewerOutput(decision="pass"),
            pr_draft={
                "title": "fix(scope): draft",
                "body_sections": {"Summary": "x"},
                "checklist": {"acceptance_criteria": [], "policy": []},
                "artifacts": [],
            },
            repo_context_meta={"cache_status": "hit", "scan_ms": 0},
            repo_context_summary={
                "repo_root": "/repo",
                "scanned_files": 10,
                "top_files": ["backend/app/services/example.py"],
                "top_tests": ["backend/tests/test_example.py"],
                "suggested_test_commands": ["pytest -q backend/tests/test_example.py"],
                "keywords": ["timeout", "retry"],
            },
            confidence_breakdown={
                "planner": 0.9,
                "researcher": 0.8,
                "executor": 0.85,
                "reviewer": 0.88,
                "overall": 0.859,
            },
            decision_trace=[
                "reviewer_decision:pass",
                "reviewer_confidence:0.88",
                "reason:ok",
            ],
            event_log=[],
        )

    monkeypatch.setattr(LangGraphIssuePrService, "run", _run)
    response = client.post(
        "/api/v1/langgraph/issue-pr/draft",
        json=_payload(),
        headers=auth_headers,
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "pr_ready"
    assert body["status_reason_code"] == "passed"
    assert body["pr_draft"]["title"] == "fix(scope): draft"
    assert body["repo_context_meta"]["cache_status"] == "hit"
    assert body["repo_context_summary"]["top_tests"][0] == "backend/tests/test_example.py"
    assert body["confidence_breakdown"]["overall"] > 0.8
    assert body["decision_trace"][0] == "reviewer_decision:pass"


def test_langgraph_issue_pr_draft_unavailable(
    client: TestClient,
    auth_headers: dict,
    monkeypatch,
):
    async def _run(self, request, **kwargs):
        raise RuntimeError("LangGraph is not installed")

    monkeypatch.setattr(LangGraphIssuePrService, "run", _run)
    response = client.post(
        "/api/v1/langgraph/issue-pr/draft",
        json=_payload(),
        headers=auth_headers,
    )
    assert response.status_code == 503
    assert "LangGraph is not installed" in response.json()["detail"]
