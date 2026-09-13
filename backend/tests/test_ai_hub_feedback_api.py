from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import HTTPException

from app.modules.autonomy.api import ai_hub_feedback


class _Result:
    def scalar_one_or_none(self):
        return None


class _Db:
    async def execute(self, _query):
        return _Result()


@pytest.mark.asyncio
async def test_list_ai_hub_feedback_hides_missing_or_foreign_job():
    with pytest.raises(HTTPException) as exc_info:
        await ai_hub_feedback.list_ai_hub_recommendation_feedback(
            uuid4(),
            db=_Db(),
            current_user=SimpleNamespace(id=uuid4()),
        )

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Agent job not found"
