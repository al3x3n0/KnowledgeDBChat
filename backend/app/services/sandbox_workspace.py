"""Filesystem permissions shared by Docker and Kubernetes sandbox backends."""

from __future__ import annotations

import os

#: The uid and gid a sandboxed program runs as.
SANDBOX_UID = SANDBOX_GID = 65534


def open_to_sandbox_user(workdir: str, *, read_only: bool = False) -> None:
    """Let the sandbox's uid use a work directory this process made.

    The container runs as 65534 and this process usually as root, and
    ``tempfile`` makes a directory only its owner may enter. On a Linux
    daemon -- the stack's own, or any production one -- the sandbox then
    cannot write its first file, and every tool said so in its own words:
    "cannot create compile_err.txt: Permission denied" read as a kernel that
    did not compile. It went unseen because Docker Desktop's file sharing,
    which the first live runs went through, does not carry ownership.

    Permission bits are not enough: a script that runs ``chmod +x prog`` on a
    file this process wrote is refused unless it *owns* the file. So where
    this process is root the tree is handed to the sandbox's uid, which root
    can still read, write and remove. Where it is not (a developer's machine)
    the bits are opened instead.

    Only skills opened their directory; this is called where the directory is
    mounted, so no caller has to remember. Best effort: a path that is not
    there, or not ours to change, is left for the run to report.
    """
    add_dir, add_file = (0o055, 0o044) if read_only else (0o077, 0o066)
    give = not read_only and hasattr(os, "geteuid") and os.geteuid() == 0
    try:
        for root, _dirs, files in os.walk(workdir, followlinks=False):
            for path, add in [(root, add_dir)] + [
                (os.path.join(root, name), add_file) for name in files
            ]:
                try:
                    if os.path.islink(path):
                        continue
                    if give:
                        os.chown(path, SANDBOX_UID, SANDBOX_GID)
                    os.chmod(path, os.stat(path).st_mode | add)
                except OSError:
                    continue
    except OSError:
        return
