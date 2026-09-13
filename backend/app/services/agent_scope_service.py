"""Canonical source-scope normalization for agent jobs and chains."""

from copy import deepcopy
from typing import Any, Optional


def normalize_scope_config(config: Any) -> Any:
    """Promote the legacy ``target_source_id`` key to ``source_id``."""
    if not isinstance(config, dict):
        return config

    normalized = dict(config)
    source_id = str(normalized.get("source_id") or "").strip()
    legacy_source_id = str(normalized.get("target_source_id") or "").strip()
    if not source_id and legacy_source_id:
        normalized["source_id"] = legacy_source_id
    normalized.pop("target_source_id", None)
    return normalized


def normalize_scope_keys_deep(value: Any) -> Any:
    """Recursively canonicalize source-scope keys in dictionaries and lists."""
    if isinstance(value, dict):
        normalized = normalize_scope_config(value) or {}
        return {
            key: normalize_scope_keys_deep(item) for key, item in normalized.items()
        }
    if isinstance(value, list):
        return [normalize_scope_keys_deep(item) for item in value]
    return value


def merge_chain_step_config(
    default_settings: Optional[dict],
    step_config: Optional[dict],
) -> dict:
    """
    Merge chain defaults with a step while preserving the inherited root scope.

    Nested step dictionaries retain normal override semantics. Only the
    canonical top-level ``source_id`` from chain defaults is protected.
    """

    def merge(base: Any, override: Any, *, preserve_source_id: bool) -> Any:
        if isinstance(base, dict) and isinstance(override, dict):
            merged = deepcopy(base)
            for key, value in override.items():
                if (
                    key == "source_id"
                    and preserve_source_id
                    and str(merged.get("source_id") or "").strip()
                ):
                    continue
                merged[key] = merge(
                    merged.get(key),
                    value,
                    preserve_source_id=False,
                )
            return merged
        return deepcopy(override)

    defaults = normalize_scope_keys_deep(default_settings) or {}
    overrides = normalize_scope_keys_deep(step_config) or {}
    return merge(defaults, overrides, preserve_source_id=True)
