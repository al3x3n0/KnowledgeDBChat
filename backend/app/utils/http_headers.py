"""Header values built from data.

A download's filename comes from a document title or an upload's own name,
and thirteen routes wrote it into ``Content-Disposition`` with an f-string.
Three things are wrong with that. A header is latin-1, so a filename in any
other script -- ``отчёт.pdf`` -- raises ``UnicodeEncodeError`` when the
response is built, and the download is a 500. A double quote in the name ends
the quoted value early. And a line break in it is a second header.

``content_disposition`` sends an ASCII fallback every client understands and,
when the name needs it, the real name in the RFC 5987 form (``filename*``),
which current browsers prefer.
"""

from __future__ import annotations

import re
import unicodedata
from urllib.parse import quote

_UNSAFE = re.compile(r'[\x00-\x1f\x7f"\\/]')


def _clean(filename: object) -> str:
    """One line, no quotes or separators; never empty."""
    name = _UNSAFE.sub("_", str(filename or "")).strip().strip(".")
    return name or "download"


def content_disposition(filename: object, *, inline: bool = False) -> str:
    """The ``Content-Disposition`` value for a file of this name."""
    name = _clean(filename)
    kind = "inline" if inline else "attachment"
    # Closest ASCII: accents dropped, anything still outside ASCII replaced.
    folded = unicodedata.normalize("NFKD", name)
    ascii_name = "".join(
        ch if ord(ch) < 128 else "" if unicodedata.combining(ch) else "_"
        for ch in folded
    )
    ascii_name = ascii_name.strip() or "download"
    value = f'{kind}; filename="{ascii_name}"'
    if ascii_name != name:
        value += f"; filename*=UTF-8''{quote(name, safe='')}"
    return value


__all__ = ["content_disposition"]
