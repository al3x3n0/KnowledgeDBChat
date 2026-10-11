"""A workspace written out from knowledge-base documents can say what changed.

Such a workspace kept hashes of its original files and not the files, and had
no git, so `propose_code_patch` answered "Not a git repository" on every one:
a coding job working from an ingested source could edit and never propose.
`create_from_source` now commits the starting files, so the diff a cloned
workspace gives is available here too.

Real rows, real git, the real handler. Nothing is replaced except the
directory workspaces are made in.
"""

import hashlib
import shutil
import subprocess
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest

from app.core.config import settings
from app.models.agent_job import AgentJob
from app.models.code_patch_proposal import CodePatchProposal
from app.models.document import Document, DocumentSource
from app.services import coding_workspace_manager as manager_module
from app.services.agent_tool_providers.base import AgentToolExecutionContext
from app.services.agent_tool_providers.workspace_mutation import (
    build_autonomous_workspace_mutation_provider,
)
from app.services.coding_workspace_manager import CodingWorkspaceManager

pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(shutil.which("git") is None, reason="needs git"),
]

FILES = {
    "src/parser.py": "end = len(items) - 1\nreturn items[:end]\n",
    "README.md": "# Parser\n",
}


@pytest.fixture
def workspace_root(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "CODING_WORKSPACE_ROOT", str(tmp_path / "ws"))
    return tmp_path / "ws"


async def _source(db, files):
    source = DocumentSource(
        name=f"repo-{uuid4().hex[:8]}", source_type="file", config={}
    )
    db.add(source)
    await db.commit()
    await db.refresh(source)
    for rel, text in files.items():
        db.add(
            Document(
                title=rel,
                content=text,
                content_hash=hashlib.sha256(text.encode()).hexdigest(),
                source_id=source.id,
                source_identifier=rel,
            )
        )
    await db.commit()
    return source


def _git(ws, *args):
    return subprocess.run(
        ["git", *args],
        cwd=str(ws.base_path),
        capture_output=True,
        text=True,
        timeout=30,
    )


async def test_the_starting_files_are_committed(db_session, workspace_root):
    source = await _source(db_session, FILES)

    ws = await CodingWorkspaceManager().create_from_source(source.id, db_session)

    assert (ws.base_path / ".git").is_dir()
    tracked = _git(ws, "ls-files").stdout.split()
    assert sorted(tracked) == sorted(FILES)
    # Nothing has been edited yet, so there is nothing to show.
    assert _git(ws, "status", "--porcelain").stdout == ""
    # The baseline is not counted among the workspace's own files.
    assert sorted(ws.original_hashes) == sorted(FILES)


async def test_an_edit_shows_as_a_diff(db_session, workspace_root):
    source = await _source(db_session, FILES)
    manager = CodingWorkspaceManager()
    ws = await manager.create_from_source(source.id, db_session)

    (ws.base_path / "src/parser.py").write_text(
        "end = len(items)\nreturn items[:end]\n"
    )

    patch = await manager.unified_diff(ws, "src/parser.py")
    assert "-end = len(items) - 1" in patch
    assert "+end = len(items)" in patch
    assert await manager.unified_diff(ws, "README.md") == ""


async def test_a_gitignore_among_the_documents_hides_nothing(
    db_session, workspace_root
):
    """An ignored file left out of the baseline would show an edit as no change."""
    files = {**FILES, ".gitignore": "*.log\n", "build.log": "ok\n"}
    source = await _source(db_session, files)
    manager = CodingWorkspaceManager()
    ws = await manager.create_from_source(source.id, db_session)

    (ws.base_path / "build.log").write_text("failed\n")

    assert "+failed" in await manager.unified_diff(ws, "build.log")


async def test_a_workspace_is_still_made_when_git_cannot_run(
    db_session, workspace_root, monkeypatch
):
    """The baseline is a convenience; the files are the workspace."""
    source = await _source(db_session, FILES)

    async def no_git(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(
        manager_module.asyncio, "create_subprocess_exec", no_git, raising=True
    )
    manager = CodingWorkspaceManager()
    ws = await manager.create_from_source(source.id, db_session)

    assert (ws.base_path / "src/parser.py").is_file()
    assert not (ws.base_path / ".git").exists()
    assert await manager.unified_diff(ws, "src/parser.py") is None


async def test_a_run_can_propose_what_it_changed(db_session, test_user, workspace_root):
    """The whole path: ingested source, an edit, a new file, a proposal row."""
    source = await _source(db_session, FILES)
    job = AgentJob(
        name="Fix the parser",
        goal="Fix the off-by-one in the parser",
        job_type="coding",
        user_id=test_user.id,
        status="running",
        config={},
    )
    db_session.add(job)
    await db_session.commit()
    await db_session.refresh(job)

    manager = CodingWorkspaceManager()
    ws = await manager.create_from_source(source.id, db_session)
    ws.owner_job_id = str(job.id)
    (ws.base_path / "src/parser.py").write_text(
        "end = len(items)\nreturn items[:end]\n"
    )
    (ws.base_path / "tests").mkdir()
    (ws.base_path / "tests/test_parser.py").write_text("def test_end():\n    pass\n")

    handler = build_autonomous_workspace_mutation_provider(
        SimpleNamespace(workspace_manager=manager)
    )._handlers["propose_code_patch"]
    result = await handler(
        {"title": "Fix the off-by-one", "workspace_id": ws.workspace_id},
        AgentToolExecutionContext(
            mode="autonomous",
            db=db_session,
            service=None,
            user_id=None,
            job=job,
            state={},
        ),
    )

    assert result.get("success") is True, result
    assert sorted(result["data"]["files"]) == [
        "src/parser.py",
        "tests/test_parser.py",
    ]
    row = await db_session.get(CodePatchProposal, UUID(result["data"]["proposal_id"]))
    assert "+end = len(items)" in row.diff_unified
    assert "tests/test_parser.py" in row.diff_unified
