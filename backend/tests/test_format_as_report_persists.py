"""`format_as_report` with `persist` stores a document.

It called a document-service method that does not exist and built the
`Document` with a keyword the model does not have and without two columns it
requires. All three sat inside `except Exception`, so the tool reported
success with `document_id: None`.
"""

from types import SimpleNamespace
from uuid import uuid4

import pytest
from sqlalchemy import select

from app.models.document import Document
from app.services.agent_tool_dispatch import (
    AgentToolExecutionContext,
    build_autonomous_output_state_provider,
)
from app.services.document_service import DocumentService

pytestmark = pytest.mark.unit


async def test_a_persisted_report_is_a_document(db_session, test_user):
    executor = SimpleNamespace(document_service=DocumentService())
    provider = build_autonomous_output_state_provider(executor)
    job = SimpleNamespace(
        id=uuid4(),
        user_id=test_user.id,
        goal="Study the L2 prefetcher",
        iteration=3,
        progress=40,
        max_iterations=10,
        config={},
    )
    state = {}
    handler = provider._handlers["format_as_report"]

    result = await handler(
        {"title": "Cache study", "executive_summary": "It helps.", "persist": True},
        AgentToolExecutionContext(
            mode="autonomous",
            db=db_session,
            service=None,
            user_id=str(test_user.id),
            job=job,
            state=state,
        ),
    )

    assert result.get("success") is True, result
    document_id = result["data"]["document_id"]
    assert document_id, "the report was formatted but not stored"

    stored = (await db_session.execute(select(Document))).scalars().all()
    assert [str(d.id) for d in stored] == [document_id]
    assert stored[0].content.startswith("# Cache study")
    assert stored[0].extra_metadata["job_id"] == str(job.id)
    assert state["artifacts"][0]["id"] == document_id
