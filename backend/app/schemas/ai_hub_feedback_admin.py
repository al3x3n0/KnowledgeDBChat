"""
Admin schemas for AI Hub recommendation feedback analytics.
"""

from __future__ import annotations

from typing import List, Literal
from uuid import UUID

from pydantic import BaseModel, Field


ItemType = Literal["dataset_preset", "eval_template"]


class AIHubFeedbackStatsRow(BaseModel):
    item_type: ItemType
    item_id: str
    accepts: int = 0
    rejects: int = 0
    net: int = 0


class AIHubFeedbackStatsResponse(BaseModel):
    profile_id: UUID
    rows: List[AIHubFeedbackStatsRow]


class AIHubFeedbackBackfillResponse(BaseModel):
    ok: bool
    profile_id: UUID
    updated: int = Field(0, description="Number of rows updated (customer_profile_id backfilled)")

