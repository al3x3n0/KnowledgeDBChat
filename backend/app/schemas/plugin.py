"""Schemas for plugins.

A plugin is a manifest plus a per-user decision to run it, and the API keeps
those separate: `PluginResponse` describes the thing, `installed`/`enabled`
describe this caller's relationship to it. Merging them would make "a plugin
someone else wrote that I have not installed" unrepresentable, which is the
normal state of every builtin bundle.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class PluginCreate(BaseModel):
    """Author a plugin from a manifest.

    The manifest is validated before anything is stored, and refused with the
    reason -- see `services/plugin_manifest.py`.
    """

    manifest: Dict[str, Any] = Field(
        ..., description="The plugin manifest: id, name, version, contributes"
    )


class PluginUpdate(BaseModel):
    manifest: Dict[str, Any]


class ContributedToolView(BaseModel):
    """One tool a plugin offers, and how it is actually classified.

    `effects` and `network` are derived from the executor type, not read from
    the manifest, so this is what the tool will really be governed as rather
    than what its author claimed.
    """

    name: str
    declared_name: str
    description: str
    tool_type: str
    effects: str
    network: str
    cost_tier: str
    job_types: List[str]


class PluginResponse(BaseModel):
    id: UUID
    slug: str
    name: str
    description: Optional[str] = None
    version: str
    source: str
    owner_id: Optional[UUID] = None
    created_at: datetime
    updated_at: Optional[datetime] = None

    #: This caller's relationship to the plugin.
    installed: bool = False
    enabled: bool = False

    #: What it contributes, with governance resolved.
    tools: List[ContributedToolView] = Field(default_factory=list)
    #: Anything declared that cannot currently be offered, and why.
    unavailable: List[Dict[str, str]] = Field(default_factory=list)

    model_config = ConfigDict(from_attributes=True)


class PluginListResponse(BaseModel):
    items: List[PluginResponse]
    total: int


class PluginInstallRequest(BaseModel):
    enabled: bool = True
    settings: Dict[str, Any] = Field(default_factory=dict)


class PluginInstallationUpdate(BaseModel):
    enabled: Optional[bool] = None
    settings: Optional[Dict[str, Any]] = None


class PluginDraftRequest(BaseModel):
    """Ask for a manifest in words."""

    description: str = Field(
        ..., min_length=3, max_length=4000, description="What the plugin should do"
    )


class PluginDraftResponse(BaseModel):
    """A drafted manifest, and an account of how it got there.

    `notes` is not decoration. A draft that took three attempts, or that
    validates but reads a path its own tool does not produce, is one a person
    should look harder at -- and saying so is more useful than presenting every
    draft as equally trustworthy.
    """

    manifest: Optional[Dict[str, Any]] = None
    notes: List[str] = Field(default_factory=list)
    attempts: int = 0
