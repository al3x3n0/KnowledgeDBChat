"""Copying a repository tree into a sandbox directory.

A repository is somebody else's content, and it can contain symbolic links
that point anywhere on this machine. `shutil.copytree` follows them by
default: a link named `notes.c` pointing at the server's `.env` arrives in
the sandbox directory as a copy of that file, where the tool about to run
will read it and quote it back. Keeping them as links is no better -- a tool
that then writes a file of that name writes through the link.

So links are left out, along with `.git`.
"""

from __future__ import annotations

import os
from typing import List


def without_git_or_links(directory: str, names: List[str]) -> List[str]:
    """An `ignore` for `shutil.copytree`: drop `.git` and every symlink."""
    return [
        name
        for name in names
        if name == ".git" or os.path.islink(os.path.join(directory, name))
    ]


__all__ = ["without_git_or_links"]
