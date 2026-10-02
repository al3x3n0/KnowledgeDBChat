"""Reading values out of a job's ``config``, which is JSON a person wrote.

Nine config normalisers each defined the same closures over ``cfg`` -- six
copies of the clamped integer, four of the clamped float, three of the string
list -- so "what does a non-numeric ``critic_max_retries`` mean" had six
answers that happened to agree. A value that cannot be read falls back to the
default, and every number is clamped: a config is a request, and the range is
what the runtime is prepared to do.
"""

from __future__ import annotations

from typing import Any, List, Mapping, Optional
from uuid import UUID


def clamped_int(
    cfg: Mapping[str, Any], key: str, default: int, lo: int, hi: int
) -> int:
    try:
        val = int(cfg.get(key, default))
    except Exception:
        val = default
    return max(lo, min(val, hi))


def clamped_float(
    cfg: Mapping[str, Any], key: str, default: float, lo: float, hi: float
) -> float:
    try:
        val = float(cfg.get(key, default))
    except Exception:
        val = default
    return max(lo, min(val, hi))


def string_list(value: Any) -> List[str]:
    """A list of non-empty strings, from a list or a comma-separated string."""
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str):
        return [str(x).strip() for x in value.split(",") if str(x).strip()]
    return []


def coerce_bool(value: Any, default: bool = False) -> bool:
    """A flag as a person might have written it; anything unrecognised is the default."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off"}:
            return False
    return default


def safe_float(value: Any, default: float = 0.0) -> float:
    """A float, or the default -- never an exception."""
    try:
        return float(value)
    except Exception:
        return default


def safe_int(value: Any, default: int = 0) -> int:
    """An int, or the default -- never an exception."""
    try:
        return int(value)
    except Exception:
        return default


def positive_int(value: Any, default: int) -> int:
    """An int greater than zero, or the default."""
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def as_number(value: Any) -> Optional[float]:
    """A measurement as a float, or None. A bool is not a number here:
    ``float(True)`` is 1.0, and a flag is not a reading of anything."""
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def uuid_list(values: Any, limit: int) -> List[str]:
    """Canonical UUID strings from a list, in order, without repeats, capped.
    Anything that is not a UUID is dropped rather than refused."""
    if not isinstance(values, list):
        return []
    out: List[str] = []
    for raw in values:
        try:
            value = str(UUID(str(raw)))
        except Exception:
            continue
        if value in out:
            continue
        out.append(value)
        if len(out) >= limit:
            break
    return out


def unique_strings(value: Any, limit: int) -> List[str]:
    """Non-empty stripped strings from a list, first occurrence kept, capped.
    Anything that is not a list is no strings at all."""
    if not isinstance(value, list):
        return []
    out: List[str] = []
    for item in value:
        text = str(item or "").strip()
        if not text or text in out:
            continue
        out.append(text)
        if len(out) >= limit:
            break
    return out
