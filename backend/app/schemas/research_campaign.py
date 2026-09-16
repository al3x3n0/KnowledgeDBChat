"""Schemas for research campaigns.

The campaign machinery existed with no HTTP surface at all: a model, a service
that could create and advance one, and a beat task driving them -- reachable
only from inside the process. These are what let a person start one.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ResearchCampaignItemCreate(BaseModel):
    """One seed question the campaign starts from."""

    title: str = Field(..., min_length=1, max_length=300)
    detail: Optional[str] = None


class ResearchCampaignCreate(BaseModel):
    """A campaign is a goal plus the seeds to pursue it with.

    `goal` is required by the service and deliberately so: it is what
    completion is judged against, and a campaign without one runs until it
    exhausts its budget.
    """

    name: str = Field(..., min_length=1, max_length=300)
    goal: str = Field(..., min_length=1)
    items: List[ResearchCampaignItemCreate] = Field(default_factory=list)
    max_jobs: int = Field(10, ge=1, le=500)
    job_template: Optional[Dict[str, Any]] = None


class ResearchCampaignItemResponse(BaseModel):
    id: UUID
    title: str
    detail: Optional[str] = None
    status: str
    origin: str
    generation: int = 0
    job_id: Optional[UUID] = None

    model_config = ConfigDict(from_attributes=True)


class ResearchCampaignResponse(BaseModel):
    id: UUID
    name: str
    goal: str
    status: str
    max_jobs: int
    jobs_launched: int
    conclusion: Optional[str] = None
    #: The same conclusion with its working -- evidence, gaps, confidence.
    #: An answer with no way to see what it rested on is one nobody should
    #: act on.
    conclusion_detail: Optional[Dict[str, Any]] = None

    #: The questions this campaign is working through. Present on the detail
    #: response, empty on the list.
    items: List[ResearchCampaignItemResponse] = Field(default_factory=list)
    created_at: datetime
    updated_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    #: The service's own summary -- item counts by status, what is next up,
    #: how much budget is left. Carried through rather than recomputed so the
    #: API and the scheduler cannot disagree about a campaign's state.
    summary: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(from_attributes=True)


class ResearchCampaignListResponse(BaseModel):
    items: List[ResearchCampaignResponse]
    total: int
