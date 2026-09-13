"""A workspace one process made, found by another.

This is the whole point of the registry, and it is worth stating what it
replaces: a workspace was a `tempfile.mkdtemp()` directory in a process-local
dict inside the celery worker that ran the job, so `manager.get(workspace_id)`
from an API request returned None -- always, not intermittently.

For a research pipeline that matters beyond convenience. The environment a
number was measured in is part of the evidence, and a benchmark whose
workspace has been deleted is a number nobody can re-derive.

The tests do not simulate two containers. They do the thing that actually
breaks: build a handle, register it, throw the in-memory manager away, and ask
a fresh one to find it.
"""

import uuid
from pathlib import Path

import pytest

from app.services.coding_workspace_manager import (
    CodingWorkspace,
    CodingWorkspaceManager,
)

pytestmark = pytest.mark.unit


def _workspace(tmp_path: Path, **over) -> CodingWorkspace:
    workspace_id = over.pop("workspace_id", str(uuid.uuid4()))
    base = tmp_path / workspace_id
    base.mkdir(parents=True, exist_ok=True)
    (base / "kernel.c").write_text("int main(void) { return 0; }")
    fields = {
        "workspace_id": workspace_id,
        "base_path": base,
        "source_id": "src-1",
        "repo_url": "https://example.invalid/repo.git",
        "branch": "main",
        "original_hashes": {"kernel.c": "abc123"},
    }
    fields.update(over)
    return CodingWorkspace(**fields)


@pytest.mark.asyncio
class TestAcrossProcesses:
    async def test_a_fresh_manager_finds_a_registered_workspace(
        self, db_session, test_user, tmp_path
    ):
        """The failure this exists to fix: the API is not the process that
        made the workspace, so its own dict is empty."""
        manager = CodingWorkspaceManager()
        workspace = _workspace(tmp_path)
        await manager.persist_record(workspace, db_session, user_id=test_user.id)

        # A different manager, as a different process would have.
        reader = CodingWorkspaceManager()
        assert (
            reader.get(workspace.workspace_id) is None
        ), "the in-memory path must stay process-local; that is not the bug"

        found = await reader.load_record(workspace.workspace_id, db_session)

        assert found is not None
        assert found.base_path == workspace.base_path
        assert (found.base_path / "kernel.c").is_file()

    async def test_what_the_run_started_with_survives(
        self, db_session, test_user, tmp_path
    ):
        """Reading the directory shows what it ENDS with. Without the original
        hashes there is no way to say what the run changed, which is the
        question a reviewer actually asks."""
        manager = CodingWorkspaceManager()
        workspace = _workspace(tmp_path, original_hashes={"kernel.c": "deadbeef"})
        await manager.persist_record(workspace, db_session, user_id=test_user.id)

        found = await CodingWorkspaceManager().load_record(
            workspace.workspace_id, db_session
        )

        assert found.original_hashes == {"kernel.c": "deadbeef"}

    async def test_provenance_travels_with_it(self, db_session, test_user, tmp_path):
        """A reader who did not launch the run needs to know where the code
        came from."""
        manager = CodingWorkspaceManager()
        workspace = _workspace(tmp_path)
        job_id = uuid.uuid4()
        await manager.persist_record(
            workspace, db_session, user_id=test_user.id, job_id=job_id
        )

        found = await CodingWorkspaceManager().load_record(
            workspace.workspace_id, db_session
        )

        assert found.repo_url == "https://example.invalid/repo.git"
        assert found.branch == "main"
        assert found.owner_job_id == str(job_id)

    async def test_registering_twice_updates_rather_than_duplicates(
        self, db_session, test_user, tmp_path
    ):
        """A workspace is registered when made and again as it changes; a
        second row would make the id ambiguous."""
        manager = CodingWorkspaceManager()
        workspace = _workspace(tmp_path)
        await manager.persist_record(workspace, db_session, user_id=test_user.id)
        workspace.original_hashes["extra.c"] = "ffff"
        await manager.persist_record(
            workspace, db_session, user_id=test_user.id, status="retained"
        )

        found = await CodingWorkspaceManager().load_record(
            workspace.workspace_id, db_session
        )

        assert "extra.c" in found.original_hashes


@pytest.mark.asyncio
class TestWhatItRefusesToReturn:
    async def test_a_workspace_whose_files_are_gone(
        self, db_session, test_user, tmp_path
    ):
        """ "The workspace was discarded" is a different answer from "there is
        no such workspace", and a handle onto files that no longer exist would
        fail on the first read with a confusing error instead."""
        manager = CodingWorkspaceManager()
        workspace = _workspace(tmp_path)
        await manager.persist_record(workspace, db_session, user_id=test_user.id)

        (workspace.base_path / "kernel.c").unlink()
        workspace.base_path.rmdir()

        assert await manager.load_record(workspace.workspace_id, db_session) is None

    async def test_an_unknown_id(self, db_session):
        manager = CodingWorkspaceManager()

        assert await manager.load_record(str(uuid.uuid4()), db_session) is None

    async def test_an_id_that_is_not_a_uuid_does_not_raise(self, db_session):
        """The id reaches this from a URL. A malformed one is a 404, not a
        500."""
        manager = CodingWorkspaceManager()

        assert await manager.load_record("not-a-uuid", db_session) is None


@pytest.mark.asyncio
class TestRegistrationNeverStopsTheAgent:
    async def test_a_failed_write_is_survivable(self, tmp_path, test_user):
        """A workspace that cannot be registered is still a workspace the
        agent can work in. Refusing to run because the registry write failed
        would trade a missing view for a missing capability."""

        class _BrokenSession:
            async def execute(self, *a, **kw):
                raise RuntimeError("no database today")

            def add(self, *a, **kw):
                raise RuntimeError("no database today")

            async def commit(self):
                raise RuntimeError("no database today")

        manager = CodingWorkspaceManager()
        workspace = _workspace(tmp_path)

        # Must not raise.
        await manager.persist_record(workspace, _BrokenSession(), user_id=test_user.id)


@pytest.mark.asyncio
class TestFinishingWithAWorkspaceWithoutDestroyingIt:
    """The old finaliser deleted every directory the moment the run ended.

    That single line is why the environment a number was measured in could
    never be inspected afterwards -- the registry would have pointed at a
    directory that no longer existed. Retention is what makes the rest of this
    worth having, and a sweep is what stops retention being a slower leak.
    """

    async def test_files_survive_the_job_that_made_them(
        self, db_session, test_user, tmp_path, monkeypatch
    ):
        from app.core.config import settings

        monkeypatch.setattr(settings, "CODING_WORKSPACE_RETENTION_HOURS", 72)
        manager = CodingWorkspaceManager()
        workspace = _workspace(tmp_path)
        manager._workspaces[workspace.workspace_id] = workspace

        await manager.release_all(db_session, user_id=test_user.id)

        assert (workspace.base_path / "kernel.c").is_file(), "the evidence stays"
        found = await CodingWorkspaceManager().load_record(
            workspace.workspace_id, db_session
        )
        assert found is not None, "and it is still findable"

    async def test_the_agent_lets_go_even_though_the_files_remain(
        self, db_session, test_user, tmp_path, monkeypatch
    ):
        """Held in the process dict for ever, a long-lived worker would
        accumulate every workspace it ever made."""
        from app.core.config import settings

        monkeypatch.setattr(settings, "CODING_WORKSPACE_RETENTION_HOURS", 72)
        manager = CodingWorkspaceManager()
        workspace = _workspace(tmp_path)
        manager._workspaces[workspace.workspace_id] = workspace

        await manager.release_all(db_session, user_id=test_user.id)

        assert manager.get(workspace.workspace_id) is None

    async def test_retention_of_zero_still_deletes(
        self, db_session, test_user, tmp_path, monkeypatch
    ):
        """The previous behaviour, still available to anyone who wants it."""
        from app.core.config import settings

        monkeypatch.setattr(settings, "CODING_WORKSPACE_RETENTION_HOURS", 0)
        manager = CodingWorkspaceManager()
        workspace = _workspace(tmp_path)
        manager._workspaces[workspace.workspace_id] = workspace

        await manager.release_all(db_session, user_id=test_user.id)

        assert not workspace.base_path.exists()

    async def test_a_deleted_workspace_is_recorded_as_discarded_not_forgotten(
        self, db_session, test_user, tmp_path, monkeypatch
    ):
        """A finding that cites it should still be traceable to something."""
        from sqlalchemy import select

        from app.core.config import settings
        from app.models.coding_workspace import CodingWorkspaceRecord

        monkeypatch.setattr(settings, "CODING_WORKSPACE_RETENTION_HOURS", 0)
        manager = CodingWorkspaceManager()
        workspace = _workspace(tmp_path)
        manager._workspaces[workspace.workspace_id] = workspace

        await manager.release_all(db_session, user_id=test_user.id)

        row = (
            await db_session.execute(
                select(CodingWorkspaceRecord).where(
                    CodingWorkspaceRecord.id == uuid.UUID(workspace.workspace_id)
                )
            )
        ).scalar_one_or_none()
        assert row is not None, "the row outlives the directory"
        assert row.status == "discarded"


@pytest.mark.asyncio
class TestTheSweepKeepsRetentionFromBecomingALeak:
    async def test_an_expired_workspace_is_removed(
        self, db_session, test_user, tmp_path
    ):
        from datetime import datetime, timedelta, timezone

        from sqlalchemy import select

        from app.models.coding_workspace import CodingWorkspaceRecord

        manager = CodingWorkspaceManager()
        workspace = _workspace(tmp_path)
        await manager.persist_record(
            workspace, db_session, user_id=test_user.id, status="retained"
        )
        row = (
            await db_session.execute(
                select(CodingWorkspaceRecord).where(
                    CodingWorkspaceRecord.id == uuid.UUID(workspace.workspace_id)
                )
            )
        ).scalar_one()
        row.last_used_at = datetime.now(timezone.utc) - timedelta(hours=200)
        await db_session.commit()

        removed = await manager.sweep_expired(db_session, hours=72)

        assert removed == 1
        assert not workspace.base_path.exists()

    async def test_a_workspace_inside_the_window_is_left_alone(
        self, db_session, test_user, tmp_path
    ):
        manager = CodingWorkspaceManager()
        workspace = _workspace(tmp_path)
        await manager.persist_record(
            workspace, db_session, user_id=test_user.id, status="retained"
        )

        removed = await manager.sweep_expired(db_session, hours=72)

        assert removed == 0
        assert (workspace.base_path / "kernel.c").is_file()

    async def test_an_active_workspace_is_never_swept(
        self, db_session, test_user, tmp_path
    ):
        """Only workspaces a job has finished with are candidates. Sweeping one
        still in use would delete the ground under a running stage."""
        from datetime import datetime, timedelta, timezone

        from sqlalchemy import select

        from app.models.coding_workspace import CodingWorkspaceRecord

        manager = CodingWorkspaceManager()
        workspace = _workspace(tmp_path)
        await manager.persist_record(
            workspace, db_session, user_id=test_user.id, status="active"
        )
        row = (
            await db_session.execute(
                select(CodingWorkspaceRecord).where(
                    CodingWorkspaceRecord.id == uuid.UUID(workspace.workspace_id)
                )
            )
        ).scalar_one()
        row.last_used_at = datetime.now(timezone.utc) - timedelta(hours=500)
        await db_session.commit()

        assert await manager.sweep_expired(db_session, hours=72) == 0
        assert (workspace.base_path / "kernel.c").is_file()


@pytest.mark.asyncio
class TestTheDiskBudget:
    """Age alone does not bound disk.

    "Keep everything for three days" costs however many jobs run in three days
    times up to 100 MB each, and that number is not knowable in advance. A byte
    ceiling is the guarantee an operator actually wants.
    """

    async def _sized(self, db_session, user, tmp_path, kb: int, hours_old: float):
        from datetime import datetime, timedelta, timezone

        from sqlalchemy import select

        from app.models.coding_workspace import CodingWorkspaceRecord

        workspace = _workspace(tmp_path)
        (workspace.base_path / "blob.bin").write_bytes(b"x" * (kb * 1024))
        manager = CodingWorkspaceManager()
        await manager.persist_record(
            workspace, db_session, user_id=user.id, status="retained"
        )
        row = (
            await db_session.execute(
                select(CodingWorkspaceRecord).where(
                    CodingWorkspaceRecord.id == uuid.UUID(workspace.workspace_id)
                )
            )
        ).scalar_one()
        row.last_used_at = datetime.now(timezone.utc) - timedelta(hours=hours_old)
        await db_session.commit()
        return workspace

    async def test_the_oldest_go_first_when_over_budget(
        self, db_session, test_user, tmp_path
    ):
        """The least recently used workspace is the one least likely to be the
        subject of a question someone is asking now."""
        newest = await self._sized(db_session, test_user, tmp_path, kb=600, hours_old=1)
        oldest = await self._sized(db_session, test_user, tmp_path, kb=600, hours_old=9)

        removed = await CodingWorkspaceManager().sweep_over_budget(
            db_session, max_total_mb=1
        )

        assert removed == 1
        assert newest.base_path.exists(), "the one in use is kept"
        assert not oldest.base_path.exists()

    async def test_nothing_is_released_when_it_all_fits(
        self, db_session, test_user, tmp_path
    ):
        kept = await self._sized(db_session, test_user, tmp_path, kb=100, hours_old=1)

        removed = await CodingWorkspaceManager().sweep_over_budget(
            db_session, max_total_mb=64
        )

        assert removed == 0
        assert kept.base_path.exists()

    async def test_a_ceiling_of_zero_disables_the_rule(
        self, db_session, test_user, tmp_path
    ):
        """Leaving only the age rule, for anyone who wants time and not bytes
        to be the policy."""
        kept = await self._sized(db_session, test_user, tmp_path, kb=600, hours_old=9)

        assert (
            await CodingWorkspaceManager().sweep_over_budget(db_session, max_total_mb=0)
            == 0
        )
        assert kept.base_path.exists()

    async def test_an_active_workspace_is_never_released_for_space(
        self, db_session, test_user, tmp_path
    ):
        """Same rule as the age sweep: releasing one still in use would delete
        the ground under a running stage, and being short of disk is not a
        reason to do that silently."""
        workspace = _workspace(tmp_path)
        (workspace.base_path / "blob.bin").write_bytes(b"x" * (2 * 1024 * 1024))
        await CodingWorkspaceManager().persist_record(
            workspace, db_session, user_id=test_user.id, status="active"
        )

        removed = await CodingWorkspaceManager().sweep_over_budget(
            db_session, max_total_mb=1
        )

        assert removed == 0
        assert workspace.base_path.exists()
