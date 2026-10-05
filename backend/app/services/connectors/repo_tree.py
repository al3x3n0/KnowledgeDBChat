"""What the GitHub and GitLab connectors do identically with a repository.

Both filter paths against ``ignore_globs`` and both render a file tree as
text; the two classes carried a copy of each.
"""

from fnmatch import fnmatch
from typing import Dict, List


class RepoTreeMixin:
    """Expects ``self.ignore_globs`` to be a list of glob patterns."""

    ignore_globs: List[str]

    def _should_ignore(self, path: str) -> bool:
        try:
            for pat in self.ignore_globs:
                if fnmatch(path, pat):
                    return True
        except Exception:
            return False
        return False

    def _tree_to_text(self, node: Dict, prefix: str, lines: List[str]) -> None:
        """Convert tree node to text lines recursively."""
        children = node.get("children", [])
        # Sort: directories first, then files, alphabetically
        children = sorted(
            children,
            key=lambda x: (x.get("type") != "directory", x.get("name", "").lower()),
        )

        for i, child in enumerate(children):
            is_last = i == len(children) - 1
            connector = "└── " if is_last else "├── "
            name = child.get("name", "")
            if child.get("type") == "directory":
                name += "/"
            lines.append(f"{prefix}{connector}{name}")

            if child.get("type") == "directory" and child.get("children"):
                extension = "    " if is_last else "│   "
                self._tree_to_text(child, prefix + extension, lines)
