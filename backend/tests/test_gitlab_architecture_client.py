"""The GitLab architecture service authenticates each call as its caller.

The service is a singleton and used to keep one HTTP client, built with the
token of whoever called first; every later request, for any user's source,
went out with that first token.
"""

import pytest

from app.services.gitlab_architecture_service import GitLabArchitectureService

pytestmark = pytest.mark.unit


async def test_each_token_gets_its_own_client():
    service = GitLabArchitectureService()
    try:
        first = await service._get_client("https://gitlab.example", "token-a")
        second = await service._get_client("https://gitlab.example", "token-b")

        assert first.headers["Authorization"] == "Bearer token-a"
        assert second.headers["Authorization"] == "Bearer token-b"
        assert await service._get_client("https://gitlab.example", "token-a") is first
    finally:
        await service.close()

    assert first.is_closed and second.is_closed


class _RecordingService:
    def __init__(self):
        self.tokens = []

    async def generate_architecture_diagram(self, *, token, **_kwargs):
        self.tokens.append(token)
        return {
            "project": "p",
            "mermaid_code": "graph TD",
            "diagram_type": "auto",
            "focus": None,
            "analysis_summary": "",
        }


async def test_the_agent_tool_uses_only_a_source_the_caller_may_use(
    db_session, test_user, admin_user, monkeypatch
):
    """A source carries a token. The tool took the first active GitLab source
    whoever asked; the /git endpoints allow an admin or whoever requested it."""
    from app.models.document import DocumentSource
    from app.services import gitlab_architecture_service as module
    from app.services.agent_service import AgentService

    db_session.add(
        DocumentSource(
            name="someone-elses",
            source_type="gitlab",
            is_active=True,
            config={
                "gitlab_url": "https://gitlab.example",
                "token": "not-yours",
                "requested_by": "somebody-else",
            },
        )
    )
    await db_session.commit()

    recorder = _RecordingService()
    monkeypatch.setattr(module, "get_gitlab_architecture_service", lambda: recorder)
    service = AgentService.__new__(AgentService)
    params = {"project_id": "group/project"}

    refused = await service._tool_generate_gitlab_architecture(
        params, test_user.id, db_session
    )
    assert "No active GitLab data source" in refused["error"]
    assert recorder.tokens == []

    allowed = await service._tool_generate_gitlab_architecture(
        params, admin_user.id, db_session
    )
    assert allowed["success"] is True
    assert recorder.tokens == ["not-yours"]

    db_session.add(
        DocumentSource(
            name="mine",
            source_type="gitlab",
            is_active=True,
            config={
                "gitlab_url": "https://gitlab.example",
                "token": "mine",
                "requested_by": test_user.username,
            },
        )
    )
    await db_session.commit()
    await service._tool_generate_gitlab_architecture(params, test_user.id, db_session)
    assert recorder.tokens[-1] == "mine"
