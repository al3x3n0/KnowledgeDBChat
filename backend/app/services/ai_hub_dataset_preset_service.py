"""
AI Hub dataset generation presets (pluggable).

Presets are JSON files on disk. This allows per-customer configuration without
hard-coding domain specifics into the UI.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from app.core.config import settings
from app.core.feature_flags import get_str as get_feature_str


@dataclass(frozen=True)
class DatasetPreset:
    id: str
    name: str
    description: str
    dataset_type: str
    generation_prompt: str


class AIHubDatasetPresetService:
    def __init__(self) -> None:
        self._dir = (
            Path(settings.AI_HUB_DATASET_PRESETS_DIR)
            if getattr(settings, "AI_HUB_DATASET_PRESETS_DIR", None)
            else Path(__file__).resolve().parents[1] / "plugins" / "ai_hub" / "dataset_presets"
        )

    def list_presets(self) -> List[DatasetPreset]:
        if not self._dir.exists():
            return []

        presets: List[DatasetPreset] = []
        for path in sorted(self._dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                presets.append(
                    DatasetPreset(
                        id=data["id"],
                        name=data.get("name", data["id"]),
                        description=data.get("description", ""),
                        dataset_type=data.get("dataset_type", "instruction"),
                        generation_prompt=data.get("generation_prompt", ""),
                    )
                )
            except Exception as exc:
                logger.warning(f"Failed to load dataset preset {path}: {exc}")
        return presets

    def get_preset(self, preset_id: str) -> Optional[DatasetPreset]:
        for p in self.list_presets():
            if p.id == preset_id:
                return p
        return None

    async def list_enabled_presets(self) -> List[DatasetPreset]:
        """
        Enabled preset allowlist.

        Priority:
        1) Redis feature flag `ai_hub_enabled_dataset_presets` (CSV)
        2) Env/config `AI_HUB_DATASET_ENABLED_PRESET_IDS` (CSV)
        3) Default: all presets
        """
        raw = await get_feature_str("ai_hub_enabled_dataset_presets")
        # If an admin has explicitly set the override (even to empty), do not fall back to env.
        if raw is None:
            raw = getattr(settings, "AI_HUB_DATASET_ENABLED_PRESET_IDS", None)

        enabled_ids = [x.strip() for x in (raw or "").split(",") if x and x.strip()]
        presets = self.list_presets()
        if enabled_ids:
            allow = set(enabled_ids)
            presets = [p for p in presets if p.id in allow]
        return presets


ai_hub_dataset_preset_service = AIHubDatasetPresetService()
