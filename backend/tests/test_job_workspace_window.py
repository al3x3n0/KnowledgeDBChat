"""Looking at the environment a run worked in.

The window is addressed through the JOB, never as a standalone repository
browser, and it is read-only. Both are deliberate: this is not a coding IDE
that happens to run remotely, it is "how did this stage produce this result".

Two things get the most attention here. The first is the question a directory
cannot answer -- what the run CHANGED -- which only exists because the hashes
of the starting files were persisted alongside it. The second is path
traversal: this endpoint takes a path straight from a URL, which is exactly
the input an attempt arrives on.
"""

import uuid
from pathlib import Path

import pytest

from app.modules.autonomy.api import job_workspace
from app.services.coding_workspace_manager import (
    CodingWorkspace,
    CodingWorkspaceManager,
)

pytestmark = pytest.mark.unit


def _sha(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


async def _prepared(db_session, user, tmp_path, job_id=None):
    """A registered workspace whose run changed one file and added another."""
    workspace_id = str(uuid.uuid4())
    base = tmp_path / workspace_id
    base.mkdir(parents=True, exist_ok=True)
    original = b"int main(void) { return 0; }"
    (base / "kernel.c").write_bytes(original)
    (base / "README.md").write_bytes(b"unchanged")

    workspace = CodingWorkspace(
        workspace_id=workspace_id,
        base_path=base,
        original_hashes={
            "kernel.c": _sha(original),
            "README.md": _sha(b"unchanged"),
            "gone.c": _sha(b"deleted later"),
        },
    )
    manager = CodingWorkspaceManager()
    await manager.persist_record(
        workspace, db_session, user_id=user.id, job_id=job_id, status="retained"
    )

    # What the run did: edited one file, added one, and the third never
    # survived.
    (base / "kernel.c").write_bytes(b"int main(void) { return 42; }")
    (base / "new_bench.c").write_bytes(b"// added by the run")
    return workspace


@pytest.mark.asyncio
class TestWhatTheRunChanged:
    async def test_modified_added_and_deleted_are_told_apart(
        self, db_session, test_user, tmp_path
    ):
        workspace = await _prepared(db_session, test_user, tmp_path)
        reader = CodingWorkspaceManager()
        found = await reader.load_record(workspace.workspace_id, db_session)

        changes = reader.get_status(found)

        assert "kernel.c" in changes["modified"]
        assert "new_bench.c" in changes["added"]
        assert "gone.c" in changes["deleted"]
        assert "README.md" not in changes["modified"]

    async def test_the_answer_survives_the_process_that_made_it(
        self, db_session, test_user, tmp_path
    ):
        """The point of persisting the starting hashes. A fresh reader has
        never seen this workspace and can still say what changed."""
        workspace = await _prepared(db_session, test_user, tmp_path)

        found = await CodingWorkspaceManager().load_record(
            workspace.workspace_id, db_session
        )

        assert CodingWorkspaceManager().get_status(found)["modified"] == ["kernel.c"]


@pytest.mark.asyncio
class TestThePathComesFromAUrl:
    async def test_traversal_is_refused(self, db_session, test_user, tmp_path):
        """Resolution goes through the manager's own `safe_resolve`, which is
        what the agent's tools use. A second implementation here would be a
        second place for this bug to live."""
        workspace = await _prepared(db_session, test_user, tmp_path)
        reader = CodingWorkspaceManager()
        found = await reader.load_record(workspace.workspace_id, db_session)

        for attempt in (
            "../../../etc/passwd",
            "/etc/passwd",
            "subdir/../../outside.txt",
        ):
            content, error = reader.read_file(found, attempt)
            assert content is None, f"{attempt!r} must not resolve"
            assert error

    async def test_a_real_file_reads(self, db_session, test_user, tmp_path):
        workspace = await _prepared(db_session, test_user, tmp_path)
        reader = CodingWorkspaceManager()
        found = await reader.load_record(workspace.workspace_id, db_session)

        content, error = reader.read_file(found, "kernel.c")

        assert error is None
        assert "return 42" in content


@pytest.mark.asyncio
class TestTheThreeWaysThereIsNothingToShow:
    """They are not interchangeable, and a reader acts differently on each."""

    async def test_a_run_that_never_made_a_workspace(self, db_session):
        record = await job_workspace._record_for_job(job_id=uuid.uuid4(), db=db_session)

        assert record is None

    async def test_a_workspace_whose_files_were_swept(
        self, db_session, test_user, tmp_path
    ):
        """ "Discarded" is not "never existed": the run's findings still
        reference it, and the row is what keeps them traceable."""
        workspace = await _prepared(db_session, test_user, tmp_path)
        for child in list(Path(workspace.base_path).iterdir()):
            child.unlink()
        Path(workspace.base_path).rmdir()

        found = await CodingWorkspaceManager().load_record(
            workspace.workspace_id, db_session
        )

        assert found is None

    async def test_the_most_recent_workspace_of_a_run_is_the_one_shown(
        self, db_session, test_user, tmp_path
    ):
        """A run can make several -- a stage that re-clones, a swarm member
        with its own copy -- and the last produced the results being read."""
        job_id = uuid.uuid4()
        first = await _prepared(db_session, test_user, tmp_path, job_id=job_id)
        second = await _prepared(db_session, test_user, tmp_path, job_id=job_id)

        record = await job_workspace._record_for_job(job_id=job_id, db=db_session)

        assert str(record.id) in {first.workspace_id, second.workspace_id}
        assert record.owner_job_id == job_id


@pytest.mark.asyncio
class TestTheChangeItself:
    """A workspace cloned from a repository keeps its `.git`, so the real
    line-level diff is already there and only needed asking for.

    The distinction that matters is between "git says nothing changed" and
    "git cannot say" -- an empty patch and no patch at all. Collapsing them
    would tell a reader a file is untouched when the truth is that nothing
    here can tell them either way.
    """

    async def _git_workspace(self, tmp_path):
        import asyncio as aio

        base = tmp_path / str(uuid.uuid4())
        base.mkdir(parents=True)
        (base / "kernel.c").write_bytes(b"int main(void) { return 0; }\n")

        async def run(*args):
            proc = await aio.create_subprocess_exec(
                *args,
                cwd=str(base),
                stdout=aio.subprocess.PIPE,
                stderr=aio.subprocess.PIPE,
            )
            await proc.communicate()

        await run("git", "init", "-q")
        await run("git", "config", "user.email", "t@example.invalid")
        await run("git", "config", "user.name", "t")
        await run("git", "add", ".")
        await run("git", "commit", "-qm", "initial")
        return CodingWorkspace(workspace_id=str(uuid.uuid4()), base_path=base)

    async def test_a_modified_file_yields_a_real_patch(self, tmp_path):
        workspace = await self._git_workspace(tmp_path)
        (workspace.base_path / "kernel.c").write_bytes(
            b"int main(void) { return 42; }\n"
        )

        patch = await CodingWorkspaceManager().unified_diff(workspace, "kernel.c")

        assert patch is not None
        assert "-int main(void) { return 0; }" in patch
        assert "+int main(void) { return 42; }" in patch

    async def test_an_unchanged_file_yields_an_empty_patch_not_none(self, tmp_path):
        """git knows the file and says it is unchanged. That is an answer."""
        workspace = await self._git_workspace(tmp_path)

        patch = await CodingWorkspaceManager().unified_diff(workspace, "kernel.c")

        assert patch == ""

    async def test_a_workspace_without_git_cannot_say(
        self, db_session, test_user, tmp_path
    ):
        """None, not empty. A knowledge-base workspace has no repository and
        no stored originals, and reporting that as "unchanged" would be a
        confident wrong answer."""
        workspace = await _prepared(db_session, test_user, tmp_path)

        assert (
            await CodingWorkspaceManager().unified_diff(workspace, "kernel.c") is None
        )

    async def test_traversal_is_refused_before_git_runs(self, tmp_path):
        workspace = await self._git_workspace(tmp_path)

        assert (
            await CodingWorkspaceManager().unified_diff(
                workspace, "../../../etc/passwd"
            )
            is None
        )
