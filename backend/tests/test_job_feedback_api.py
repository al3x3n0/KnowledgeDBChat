from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import HTTPException

from app.modules.autonomy.api import job_feedback
from app.modules.autonomy.application import feedback_presenters
from app.schemas.agent_job import AgentJobFeedbackCreate


class _Result:
    def __init__(self, value):
        self.value = value

    def scalar_one_or_none(self):
        return self.value


class _Db:
    def __init__(self, value=None):
        self.value = value

    async def execute(self, _query):
        return _Result(self.value)


def test_sanitize_tool_names_deduplicates_and_rejects_unsafe_values():
    assert feedback_presenters.sanitize_tool_names(
        [" read_file ", "read_file", "../shell", "x", "mcp:search"]
    ) == ["read_file", "mcp:search"]


@pytest.mark.asyncio
async def test_create_feedback_hides_missing_or_foreign_job():
    with pytest.raises(HTTPException) as exc_info:
        await job_feedback.create_agent_job_feedback(
            uuid4(),
            AgentJobFeedbackCreate(rating=4, feedback="Useful"),
            db=_Db(),
            current_user=SimpleNamespace(id=uuid4()),
        )

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Agent job not found"


@pytest.mark.asyncio
async def test_list_learning_feedback_rejects_invalid_scope_before_query():
    with pytest.raises(HTTPException) as exc_info:
        await job_feedback.list_learning_feedback(
            scope="organization",
            limit=100,
            db=_Db(),
            current_user=SimpleNamespace(id=uuid4()),
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "scope must be user, customer, or team"
