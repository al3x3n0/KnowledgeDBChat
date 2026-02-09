"""
Schemas for creating AI Hub plugins (dataset presets + eval templates).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Literal

from pydantic import BaseModel, Field


class CreateAIHubPluginRequest(BaseModel):
    plugin_type: Literal["dataset_preset", "eval_template"]
    plugin: Dict[str, Any] = Field(..., description="The plugin JSON content to persist")
    overwrite: bool = Field(False, description="Allow overwriting an existing plugin file")


class CreateAIHubPluginResponse(BaseModel):
    ok: bool
    plugin_type: str
    plugin_id: str
    path: str
    overwritten: bool = False
    warnings: Optional[list[str]] = None

