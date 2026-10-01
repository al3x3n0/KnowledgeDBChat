"""Schemas for sandbox skills and the images authored for them.

A skill's response leads with the three things a person needs before
activating one: who wrote it (`origin`), whether its control has passed against
the content it has *now* (`verified`), and what the last control run said.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class SandboxSkillWrite(BaseModel):
    """Create or replace a skill from its manifest.

    Validated before anything is stored, and refused with the reason -- see
    `services/sandbox_skill_manifest.py`.
    """

    manifest: Dict[str, Any] = Field(
        ...,
        description=(
            "The skill: id, name, description, image, procedure, files, "
            "result, judge_command, control, timeout_seconds"
        ),
    )


class SandboxSkillResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    slug: str
    name: str
    description: str
    manifest: Dict[str, Any]
    status: str
    origin: str
    origin_job_id: Optional[UUID] = None
    #: The finding type a contract writes to require this skill's result.
    produces: str
    #: True only when the control passed against the current content.
    verified: bool
    last_dry_run: Optional[Dict[str, Any]] = None
    notes: List[str] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime


class SandboxSkillListResponse(BaseModel):
    items: List[SandboxSkillResponse]
    #: Every image a skill may name right now, so an editor offers a list
    #: instead of a text box that accepts a plausible string.
    images: List[str]
    #: Whether a skill can run at all on this server. When false, nothing can
    #: be dry-run and therefore nothing can be activated -- worth saying on the
    #: page rather than on each failed button.
    execution_enabled: bool
    authoring_enabled: bool
    image_build_enabled: bool


class SandboxSkillDryRunResponse(BaseModel):
    skill: SandboxSkillResponse
    ok: bool
    #: False when nothing could be tested, which is not the skill's fault.
    ran: bool
    detail: str
    returncode: Optional[int] = None
    stdout: str = ""
    stderr: str = ""
    result: Optional[Dict[str, Any]] = None


class SandboxSkillDraftRequest(BaseModel):
    """Ask for a skill in words; with `current`, ask for a revision."""

    description: str = Field(
        ..., min_length=3, max_length=4000, description="What the skill should do"
    )
    current: Optional[Dict[str, Any]] = Field(
        default=None,
        description="The skill in the editor now. Present means: revise this.",
    )


class SandboxSkillDraftQueued(BaseModel):
    task_id: str
    poll_url: str


class SandboxSkillDraftStatus(BaseModel):
    state: str
    #: 'drafting' | 'checking' | 'done', or None before the task starts.
    stage: Optional[str] = None
    attempt: int = 0
    notes: List[str] = Field(default_factory=list)
    manifest: Optional[Dict[str, Any]] = None
    #: The last control outcome, when one was attempted.
    dry_run: Optional[Dict[str, Any]] = None
    attempts: int = 0
    pending: bool = True


class SandboxSkillImagePropose(BaseModel):
    slug: str = Field(..., min_length=2, max_length=48)
    dockerfile: str = Field(..., min_length=5, max_length=20000)
    description: Optional[str] = Field(default=None, max_length=2000)


class SandboxSkillImageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    slug: str
    description: Optional[str] = None
    dockerfile: str
    base_image: str
    tag: str
    status: str
    proposed_by: Optional[UUID] = None
    approved_by: Optional[UUID] = None
    build_log: Optional[str] = None
    built_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime


class SandboxSkillImageListResponse(BaseModel):
    items: List[SandboxSkillImageResponse]
    build_enabled: bool
