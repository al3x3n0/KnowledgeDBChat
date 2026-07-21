from app.schemas.langgraph_issue_pr import ReviewerOutput
from app.services.langgraph_issue_pr_service import LangGraphIssuePrService


def test_status_reason_code_passed():
    service = LangGraphIssuePrService()
    code = service._derive_status_reason_code(
        status="pr_ready",
        reviewer=ReviewerOutput(decision="pass", confidence=0.9),
        reason="PR draft package is ready.",
        needs_human=False,
    )
    assert code == "passed"


def test_status_reason_code_policy_escalation():
    service = LangGraphIssuePrService()
    code = service._derive_status_reason_code(
        status="needs_human_review",
        reviewer=ReviewerOutput(decision="escalate", confidence=0.8),
        reason="High-impact action detected; human approval required.",
        needs_human=True,
    )
    assert code == "policy_escalation"


def test_status_reason_code_revision_exhausted():
    service = LangGraphIssuePrService()
    code = service._derive_status_reason_code(
        status="blocked",
        reviewer=ReviewerOutput(decision="revise", confidence=0.6),
        reason="Revision loop exhausted before passing review.",
        needs_human=False,
    )
    assert code == "revision_exhausted"


def test_status_reason_code_coerce_invalid():
    service = LangGraphIssuePrService()
    assert service._coerce_status_reason_code("non_standard_code") == "unknown"
