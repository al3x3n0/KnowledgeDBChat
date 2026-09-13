from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import APIRouter, HTTPException

from app.api.endpoints import agent_jobs
from app.modules.autonomy.api.swarm_outcomes import build_swarm_outcomes_api


class _Scalars:
    def all(self):
        return []


class _Result:
    def scalars(self):
        return _Scalars()


class _EmptyDb:
    async def execute(self, _query):
        return _Result()


@pytest.mark.asyncio
async def test_swarm_outcomes_rejects_unknown_preset_before_query():
    with pytest.raises(HTTPException) as exc_info:
        await agent_jobs.get_swarm_outcomes(
            source_id=None,
            preset_key="unknown-swarm",
            terminal_outcome=None,
            promotion_mode=None,
            visibility_scope="mine",
            date_from=None,
            date_to=None,
            db=SimpleNamespace(),
            current_user=SimpleNamespace(id=uuid4()),
        )

    assert exc_info.value.status_code == 422
    assert exc_info.value.detail == "Unknown coding swarm preset"


@pytest.mark.asyncio
async def test_swarm_outcomes_returns_zeroed_rows_for_empty_dataset():
    async def load_users(_db, *, current_user):
        return {str(current_user.id): current_user}

    api = build_swarm_outcomes_api(
        router=APIRouter(),
        presets={
            "bug_triage_swarm": {
                "launch_mode": "quick_start_bug_triage_swarm",
                "label": "Bug triage",
            }
        },
        is_job_visible=lambda _job, _user: True,
        is_backlog_visible=lambda _item, _user: True,
        extract_launch_mode=lambda config: str(config.get("launch_mode") or ""),
        infer_preset_key=lambda _job: "bug_triage_swarm",
        derive_outcome_case=lambda *_args, **_kwargs: None,
        extract_swarm_summary=lambda _job: {},
        datetime_sort_key=lambda _value: 0.0,
        iso_or_none=lambda value: value.isoformat() if value else None,
        load_collaboration_user_lookup=load_users,
    )
    response = await api.get_swarm_outcomes(
        source_id=None,
        preset_key=None,
        terminal_outcome=None,
        promotion_mode=None,
        visibility_scope="mine",
        date_from=None,
        date_to=None,
        db=_EmptyDb(),
        current_user=SimpleNamespace(id=uuid4()),
    )

    assert len(response.preset_rows) == 1
    assert response.totals["total_swarm_roots"] == 0
    assert response.totals["avg_confidence"] is None
    assert response.cases == []
