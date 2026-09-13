"""A programme of research that outlives any one job.

Everything the agent runtime does is scoped to a job: a goal, an iteration
budget, a contract, and an ending. That is the right unit for one experiment
and the wrong one for a line of enquiry. A question worth a week is a sequence
of experiments where each one's result decides the next, and nothing here held
that sequence -- job chaining fires children at completion and then no one is
watching, and a machine restart ends the whole thing silently.

A campaign is the missing holder: a standing goal, a backlog of work to do
under it, and a record of what each piece of work found. It is advanced a step
at a time by a caller that can be a scheduler, so progress survives restarts --
all the state that matters is here rather than in a running process.

Two decisions worth naming. A campaign has a **job budget**, because an agent
that can create work from its own findings can create it without end, and
"until it is done" is not a limit. And an item that produced a job keeps the
job's id whatever became of it, so a campaign that went wrong can be read
afterwards rather than only counted.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import UUID

from app.core.database import Base


class CampaignStatus:
    ACTIVE = "active"
    COMPLETED = "completed"
    EXHAUSTED = "exhausted"
    CANCELLED = "cancelled"


class CampaignItemStatus:
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    DROPPED = "dropped"


class ResearchCampaign(Base):
    __tablename__ = "research_campaigns"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    name = Column(String(300), nullable=False)
    # The standing question. Each item's job gets its own goal; this is what
    # they are all in service of, and what completion is judged against.
    goal = Column(Text, nullable=False)

    status = Column(
        String(40), default=CampaignStatus.ACTIVE, nullable=False, index=True
    )

    # A campaign whose jobs can create further work needs a ceiling that is not
    # "until it is done": findings beget items beget jobs beget findings.
    max_jobs = Column(Integer, default=10, nullable=False)
    jobs_launched = Column(Integer, default=0, nullable=False)

    # How to build each job: contract, job type, iteration budget, and which
    # finding types spawn further items.
    job_template = Column(JSON, nullable=True)

    conclusion = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    __table_args__ = (Index("ix_research_campaigns_status_user", "status", "user_id"),)


class ResearchCampaignItem(Base):
    __tablename__ = "research_campaign_items"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    campaign_id = Column(
        UUID(as_uuid=True),
        ForeignKey("research_campaigns.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    title = Column(String(300), nullable=False)
    detail = Column(Text, nullable=True)
    status = Column(
        String(40), default=CampaignItemStatus.PENDING, nullable=False, index=True
    )

    # "seed" for what the campaign was started with, "discovered" for work a
    # job's own findings created. Kept apart so a campaign that only ever
    # chased its own tail is visible as one.
    origin = Column(String(40), default="seed", nullable=False)

    # Which item's job revealed this one, and how far from the seed list it
    # is. Without these a cold *line* cannot be told from a single cold item,
    # and the tenth speculative offshoot looks like the first.
    parent_item_id = Column(UUID(as_uuid=True), nullable=True, index=True)
    generation = Column(Integer, default=0, nullable=False)

    # Not a foreign key: the job may be deleted, and losing the record that
    # this item was worked on would be the wrong repair.
    job_id = Column(UUID(as_uuid=True), nullable=True, index=True)
    launched_at = Column(DateTime, nullable=True)
    outcome = Column(JSON, nullable=True)

    # What the campaign thought of this item when it last looked, and why. Kept
    # so that a choice can be read afterwards and argued with, rather than only
    # observed.
    priority = Column(Float, nullable=True)
    priority_reason = Column(String(400), nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index("ix_research_campaign_items_campaign_status", "campaign_id", "status"),
    )
