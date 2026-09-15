"""What a person may change about their own interface, and its bounds.

The stored value is a JSON column the client writes, which makes it the one
place in this application where a user's own text becomes configuration the
application then acts on. That earns a normalizer rather than a pass-through:
unknown keys are dropped, lists are capped, and strings are trimmed to a
length, so a malformed or hostile preferences document cannot become an
unbounded row or a navigation entry with a megabyte of label.

Nothing here refuses a *route*. A preference naming a destination that no
longer exists is inert -- it matches nothing when applied -- and that is the
right behaviour: routes come and go, and a person who hid a page that was
later renamed should not have their whole navigation rejected because of it.
The catalog decides what exists; these preferences only ever reorder, hide,
rename and pin what the catalog already offers.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

#: Bounds. Generous enough that nobody legitimately customizing their nav will
#: meet them, small enough that the column cannot be used as storage.
MAX_ENTRIES = 200
MAX_KEY_LENGTH = 200
MAX_LABEL_LENGTH = 60


def _clean_key(value: Any) -> Optional[str]:
    # A key must already be a string. Coercing would turn `7` into "7" and
    # `None` into "None" -- keys that match no destination but still occupy a
    # row, which is storing junk rather than rejecting it.
    if not isinstance(value, str):
        return None
    key = value.strip()
    if not key or len(key) > MAX_KEY_LENGTH:
        return None
    return key


def _clean_keys(value: Any) -> List[str]:
    """A de-duplicated, order-preserving list of destination keys."""
    if not isinstance(value, list):
        return []
    out: List[str] = []
    for item in value:
        key = _clean_key(item)
        if key and key not in out:
            out.append(key)
        if len(out) >= MAX_ENTRIES:
            break
    return out


def _clean_labels(value: Any) -> Dict[str, str]:
    """Renames, as ``destination key -> label``."""
    if not isinstance(value, dict):
        return {}
    out: Dict[str, str] = {}
    for raw_key, raw_label in value.items():
        key = _clean_key(raw_key)
        label = str(raw_label or "").strip()[:MAX_LABEL_LENGTH]
        # An empty rename is a removal, not a blank label: a nav entry with no
        # name is unreachable by anything except position.
        if key and label:
            out[key] = label
        if len(out) >= MAX_ENTRIES:
            break
    return out


def normalize(raw: Any) -> Optional[Dict[str, Any]]:
    """The stored shape, or ``None`` when nothing was actually customized.

    Returning ``None`` for an empty document matters: null means "has never
    customized anything", which is what lets the interface evolve for people
    who never opened these settings. A row holding an empty object would
    instead mean "customized it to exactly the defaults", and would pin them
    to whatever the defaults were on the day they saved.
    """
    if not isinstance(raw, dict):
        return None

    nav_raw = raw.get("nav")
    nav_raw = nav_raw if isinstance(nav_raw, dict) else {}

    nav: Dict[str, Any] = {}
    door_order = _clean_keys(nav_raw.get("doorOrder"))
    hidden = _clean_keys(nav_raw.get("hidden"))
    pinned = _clean_keys(nav_raw.get("pinned"))
    renamed = _clean_labels(nav_raw.get("renamed"))
    landing = _clean_key(nav_raw.get("landing"))

    if door_order:
        nav["doorOrder"] = door_order
    if hidden:
        nav["hidden"] = hidden
    if pinned:
        nav["pinned"] = pinned
    if renamed:
        nav["renamed"] = renamed
    if landing:
        nav["landing"] = landing

    return {"nav": nav} if nav else None
