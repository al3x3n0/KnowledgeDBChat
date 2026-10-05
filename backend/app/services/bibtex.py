"""BibTeX helpers shared by the LaTeX endpoint and the LaTeX agent runner.

Both built citations from the knowledge base and each carried its own copy of
these four functions, identical to the character. A fix to how a title is
escaped, or which month macro a date maps to, would have reached the bibliography
a person exports and not the one an agent writes, or the other way round.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Optional


def _sanitize_bib_filename(name: str) -> str:
    s = (name or "").strip()
    if not s:
        return "refs.bib"
    if "/" in s or "\\" in s or s.startswith("."):
        return "refs.bib"
    if not s.lower().endswith(".bib"):
        s = s + ".bib"
    if len(s) > 100:
        s = s[:100]
    return s


def _escape_bibtex(s: str) -> str:
    """
    Escape user/content strings for safe inclusion inside BibTeX fields / LaTeX text.

    Note: We intentionally do not try to preserve existing LaTeX macros. This endpoint
    is meant for plain-text metadata pulled from the Knowledge DB.
    """
    t = (s or "").strip()
    if not t:
        return ""
    # Collapse whitespace/newlines to keep entries tidy.
    t = re.sub(r"\s+", " ", t).strip()
    # LaTeX special chars commonly appearing in titles/authors.
    t = t.replace("\\", r"\textbackslash{}")
    t = t.replace("{", r"\{").replace("}", r"\}")
    t = t.replace("&", r"\&")
    t = t.replace("%", r"\%")
    t = t.replace("$", r"\$")
    t = t.replace("#", r"\#")
    t = t.replace("_", r"\_")
    t = t.replace("~", r"\textasciitilde{}")
    t = t.replace("^", r"\textasciicircum{}")
    return t


def _extract_arxiv_id(url: str) -> Optional[str]:
    """
    Extract an arXiv identifier from a URL if present.

    Supports:
    - https://arxiv.org/abs/1234.56789
    - https://arxiv.org/abs/1234.56789v2
    - https://arxiv.org/pdf/1234.56789.pdf
    - https://arxiv.org/pdf/1234.56789v2.pdf
    """
    u = (url or "").strip()
    if not u:
        return None
    m = re.search(
        r"arxiv\.org/(abs|pdf)/(?P<id>\d{4}\.\d{4,5}(v\d+)?)(?:\.pdf)?", u, flags=re.I
    )
    if not m:
        return None
    return (m.group("id") or "").strip() or None


def _bibtex_month_macro(dt: Optional[datetime]) -> Optional[str]:
    if not dt:
        return None
    try:
        month = int(dt.month)
    except Exception:
        return None
    months = [
        "jan",
        "feb",
        "mar",
        "apr",
        "may",
        "jun",
        "jul",
        "aug",
        "sep",
        "oct",
        "nov",
        "dec",
    ]
    if 1 <= month <= 12:
        return months[month - 1]
    return None


def _bib_key_from_uuid(doc_id) -> str:
    """Durable, reversible cite key (no guessing or prefix matching needed):
    ``\\cite{KDB:<uuid>}``."""
    return f"KDB:{str(doc_id)}"
