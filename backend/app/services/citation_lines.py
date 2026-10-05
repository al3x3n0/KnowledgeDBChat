"""Which lines of generated markdown are expected to carry a citation.

Four copies of this predicate existed: three inside one endpoint module and
one in a monitoring task. They decide what counts toward "citation coverage",
so a copy that drifted would report a different coverage for the same note
depending on which path produced it.
"""

from __future__ import annotations

import re


def is_line_citable(line: str) -> bool:
    s = (line or "").strip()
    if not s:
        return False
    if s.startswith("#"):
        return False
    if s.startswith("```") or s.startswith(">"):
        return False
    return bool(re.search(r"[A-Za-z0-9]", s))
