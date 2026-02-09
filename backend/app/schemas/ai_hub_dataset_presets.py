"""
Schemas for AI Hub dataset generation presets.
"""

from typing import List

from pydantic import BaseModel


class DatasetPresetInfo(BaseModel):
    id: str
    name: str
    description: str
    dataset_type: str


class DatasetPresetsResponse(BaseModel):
    presets: List[DatasetPresetInfo]

