"""Path and LaTeX-document helpers shared by the deterministic agent runners.

Both functions here were defined inline, byte-identically, in several runners:
``safe_relpath`` in two and ``insert_before_end_document`` in four. That is worth
fixing for its own sake, but ``safe_relpath`` especially — it is a path-traversal
guard, and a copy that gets fixed while its twins do not is the shape of a real
vulnerability rather than untidiness.
"""

from __future__ import annotations

END_DOCUMENT = "\\end{document}"

# Long enough for any real repository path, short enough to bound what a model
# can talk the runner into writing.
MAX_RELPATH_LENGTH = 240


def safe_relpath(path: str) -> str:
    """Reduce a model-supplied path to a bounded, relative, traversal-free form.

    Backslashes are normalized, leading slashes and ``./`` prefixes stripped,
    and ``..`` segments dropped entirely rather than resolved — a path that
    tried to escape simply loses the attempt instead of landing somewhere else.
    Returns "" when nothing usable remains, which callers treat as "skip".
    """
    normalized = (path or "").replace("\\", "/").strip()
    normalized = normalized.lstrip("/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    parts = [part for part in normalized.split("/") if part not in {"", ".", ".."}]
    return "/".join(parts)[:MAX_RELPATH_LENGTH]


def insert_before_end_document(source: str, addition: str) -> str:
    """Insert ``addition`` just before ``\\end{document}``.

    Appends to the end when the document has no such marker, so a partial or
    fragment file still receives the content rather than silently dropping it.
    """
    text = source or ""
    index = text.rfind(END_DOCUMENT)
    if index == -1:
        return (text.rstrip() + "\n\n" + addition.strip() + "\n").lstrip("\n")
    before = text[:index].rstrip()
    after = text[index:]
    return f"{before}\n\n{addition.strip()}\n\n{after}"
