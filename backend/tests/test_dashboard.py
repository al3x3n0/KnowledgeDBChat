"""The dashboard service and its live socket, with rows in the database.

Neither had a test, and the route walk meets them with an empty database,
where they answer. With rows:

* the activity feed read `doc.source`, a lazy relationship (MissingGreenlet on
  an async session), and `execution.updated_at`, a column workflow executions
  do not have: one document or one workflow run and the feed -- and
  `/dashboard/all` with it -- was a 500;
* the socket took `user_id` from the client's first message and checked no
  token, so anyone who could reach the server was sent the overview, document
  titles and source health; it also kept one request database session for as
  long as it stayed open;
* the agent summary's "top agents" counted every user's conversations;
* trending tags counted `Paper` and `paper` apart, and one document whose
  tags were not a list of strings failed the whole request.
"""

import hashlib
from datetime import datetime, timedelta
from uuid import uuid4

import pytest

from app.api.endpoints import dashboard as dashboard_endpoints
from app.core.tokens import create_access_token
from app.models.agent_definition import AgentConversationContext, AgentDefinition
from app.models.document import Document, DocumentSource
from app.models.memory import AgentConversation
from app.models.workflow import Workflow, WorkflowExecution
from app.services.dashboard_service import dashboard_service

pytestmark = pytest.mark.unit


async def _document(db, title="Report", tags=None, source=None, **fields):
    if source is None:
        source = DocumentSource(
            name=f"src-{uuid4().hex[:6]}", source_type="file", config={}
        )
        db.add(source)
        await db.commit()
        await db.refresh(source)
    doc = Document(
        title=title,
        content="body",
        content_hash=hashlib.sha256(uuid4().bytes).hexdigest(),
        source_id=source.id,
        source_identifier=f"test:{uuid4().hex}",
        tags=tags,
        **fields,
    )
    db.add(doc)
    await db.commit()
    await db.refresh(doc)
    return doc, source


async def _execution(db, user, status="completed", **fields):
    workflow = Workflow(user_id=user.id, name=f"wf-{uuid4().hex[:6]}")
    db.add(workflow)
    await db.commit()
    await db.refresh(workflow)
    execution = WorkflowExecution(
        workflow_id=workflow.id,
        user_id=user.id,
        trigger_type="manual",
        status=status,
        **fields,
    )
    db.add(execution)
    await db.commit()
    await db.refresh(execution)
    return execution


# --------------------------------------------------------------------------
# activity feed
# --------------------------------------------------------------------------


async def test_the_feed_names_a_documents_source(db_session, test_user):
    doc, source = await _document(db_session, title="Handbook")
    # As a request sees it: nothing about the source is already loaded.
    db_session.expunge_all()

    feed = await dashboard_service.get_activity_feed(db_session, user_id=test_user.id)

    assert feed[0]["title"] == "Document added: Handbook"
    assert source.name in feed[0]["description"]


async def test_the_feed_shows_a_workflow_run_at_the_time_it_last_moved(
    db_session, test_user
):
    finished = datetime(2026, 10, 2, 12, 0, 0)
    await _execution(
        db_session,
        test_user,
        started_at=finished - timedelta(minutes=5),
        completed_at=finished,
    )

    feed = await dashboard_service.get_activity_feed(db_session, user_id=test_user.id)

    assert feed[0]["type"] == "workflow_completed"
    assert feed[0]["timestamp"].startswith("2026-10-02T12:00:00")


async def test_the_feed_shows_only_the_callers_workflow_runs(
    db_session, test_user, admin_user
):
    await _execution(db_session, admin_user)

    feed = await dashboard_service.get_activity_feed(db_session, user_id=test_user.id)

    assert feed == []


# --------------------------------------------------------------------------
# summaries
# --------------------------------------------------------------------------


async def test_every_widget_answers_with_rows_present(db_session, test_user):
    await _document(db_session, tags=["Paper"], file_type="application/pdf")
    await _execution(db_session, test_user, status="failed")

    overview = await dashboard_service.get_overview_stats(db_session, test_user.id)
    assert overview["documents"]["total"] == 1
    assert overview["workflows"]["total"] == 1
    by_type = await dashboard_service.get_documents_by_type_chart(db_session)
    assert by_type["datasets"][0]["data"] in ([], [1])
    by_source = await dashboard_service.get_documents_by_source_chart(db_session)
    assert by_source["datasets"][0]["data"] == [1]
    timeline = await dashboard_service.get_documents_timeline_chart(db_session, 7)
    assert sum(timeline["datasets"][0]["data"]) == 1
    health = await dashboard_service.get_source_health(db_session)
    assert health[0]["document_count"] == 1
    actions = await dashboard_service.get_quick_actions(db_session, test_user.id)
    assert actions[-1]["id"] == "upload_docs"
    summary = await dashboard_service.get_workflow_summary(db_session, test_user.id)
    assert summary["status_counts"] == {"failed": 1}
    assert summary["total_executions"] == 1


async def test_trending_tags_ignore_case_and_survive_a_malformed_document(
    db_session, test_user
):
    await _document(db_session, tags=["Paper", "cache"])
    await _document(db_session, tags=["paper", " "])
    await _document(db_session, tags=[["nested"], 7, None])
    await _document(db_session, tags="paper")

    tags = await dashboard_service.get_trending_tags(db_session)

    assert tags[0] == {"tag": "paper", "count": 2, "trend": "up"}
    assert {"tag": "cache", "count": 1, "trend": "up"} in tags
    assert all(isinstance(t["tag"], str) and t["tag"].strip() for t in tags)


async def test_top_agents_count_the_callers_conversations_only(
    db_session, test_user, admin_user
):
    agent = AgentDefinition(
        name=f"a-{uuid4().hex[:6]}",
        display_name="Analyst",
        system_prompt="x",
        capabilities=[],
    )
    db_session.add(agent)
    await db_session.commit()
    for owner, turns in ((test_user, 1), (admin_user, 3)):
        conversation = AgentConversation(user_id=owner.id, messages=[])
        db_session.add(conversation)
        await db_session.commit()
        for turn in range(turns):
            db_session.add(
                AgentConversationContext(
                    conversation_id=conversation.id,
                    agent_definition_id=agent.id,
                    turn_number=turn,
                )
            )
        await db_session.commit()

    summary = await dashboard_service.get_agent_usage_summary(
        db_session, user_id=test_user.id
    )

    assert summary["total_conversations"] == 1
    assert summary["top_agents"] == [{"name": "Analyst", "turns": 1}]


# --------------------------------------------------------------------------
# the live socket
# --------------------------------------------------------------------------


class _Socket:
    """A WebSocket as the handler uses one."""

    def __init__(self, messages, query=None):
        self._messages = list(messages)
        self.query_params = query or {}
        self.headers = {}
        self.sent = []
        self.closed = None
        self.accepted = False

    async def accept(self):
        self.accepted = True

    async def receive_json(self):
        from fastapi import WebSocketDisconnect

        if not self._messages:
            raise WebSocketDisconnect()
        return self._messages.pop(0)

    async def send_json(self, payload):
        self.sent.append(payload)

    async def close(self, code=1000, reason=""):
        if self.closed is None:
            self.closed = (code, reason)


@pytest.fixture
def sessions(db_session, monkeypatch):
    """The handler opens its own sessions; give it the test's database."""
    from contextlib import asynccontextmanager

    from app.utils import websocket_auth

    @asynccontextmanager
    async def session():
        yield db_session

    monkeypatch.setattr(dashboard_endpoints, "AsyncSessionLocal", session)
    monkeypatch.setattr(websocket_auth, "AsyncSessionLocal", session)


@pytest.mark.parametrize(
    "first_message",
    [
        {"type": "auth", "user_id": "00000000-0000-0000-0000-000000000001"},
        {"type": "auth", "token": "not-a-token"},
        {"type": "refresh"},
    ],
)
async def test_the_socket_sends_nothing_without_a_valid_token(
    db_session, test_user, sessions, first_message
):
    await _document(db_session, title="Confidential plan")
    socket = _Socket([first_message])

    await dashboard_endpoints.dashboard_websocket(socket)

    assert socket.sent == []
    assert socket.closed[0] == 4001


async def test_the_socket_answers_a_signed_in_user(db_session, test_user, sessions):
    await _document(db_session, title="Handbook")
    token = create_access_token(test_user.id)
    socket = _Socket(
        [{"type": "auth", "token": token}, {"type": "refresh", "widget": "activity"}]
    )

    await dashboard_endpoints.dashboard_websocket(socket)

    assert [m["type"] for m in socket.sent] == ["overview", "activity"]
    assert socket.sent[0]["data"]["documents"]["total"] == 1
    assert socket.sent[1]["data"][0]["title"] == "Document added: Handbook"


async def test_the_socket_accepts_a_token_in_the_query(db_session, test_user, sessions):
    token = create_access_token(test_user.id)
    socket = _Socket([], query={"token": token})

    await dashboard_endpoints.dashboard_websocket(socket)

    assert [m["type"] for m in socket.sent] == ["overview"]


def test_the_socket_does_not_hold_a_request_session():
    """`Depends(get_db)` on a socket keeps one of the limited sessions open
    for as long as the client stays connected."""
    import inspect

    parameters = inspect.signature(dashboard_endpoints.dashboard_websocket).parameters
    assert list(parameters) == ["websocket"]
