"""Regression coverage for modular decision-trace reporting routes."""

from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import HTTPException

from app.api.endpoints import agent_jobs


@pytest.mark.asyncio
async def test_decision_trace_export_rejects_an_unknown_format():
    with pytest.raises(HTTPException) as exc_info:
        await agent_jobs.export_decision_trace(
            format="xml",
            source_kind=None,
            decision_type=None,
            customer=None,
            status=None,
            severity=None,
            actor_mode=None,
            triage_status=None,
            assigned_to_user_id=None,
            unassigned_only=False,
            escalation_state=None,
            pinned=None,
            actionable_only=False,
            start_at=None,
            end_at=None,
            db=object(),
            current_user=SimpleNamespace(id=uuid4()),
        )

    assert exc_info.value.status_code == 422
    assert exc_info.value.detail == "Unsupported export format"
