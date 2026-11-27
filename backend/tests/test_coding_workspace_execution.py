from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from app.services import coding_workspace_execution as execution


@pytest.mark.asyncio
async def test_run_uses_a_disposable_repository_copy(monkeypatch, tmp_path):
    workspace = tmp_path / "repo"
    workspace.mkdir()
    (workspace / ".git").mkdir()
    (workspace / "app.py").write_text("before")
    (workspace / "external").symlink_to(tmp_path)
    sandbox_root = tmp_path / "sandboxes"
    monkeypatch.setenv("WORKER_SECRET", "worker-only-value")
    monkeypatch.setattr(execution, "get_flag", AsyncMock(return_value=True))
    monkeypatch.setattr(
        execution.sandbox_skill_runtime, "root_dir", lambda: sandbox_root
    )
    monkeypatch.setattr(
        execution.agent_sandbox_runtime,
        "allowed_images",
        lambda: [execution.settings.CODING_WORKSPACE_SANDBOX_IMAGE],
    )

    async def sandbox(script, workdir, *, image, timeout_seconds):
        copy = Path(workdir)
        assert copy.parent == sandbox_root and copy != workspace
        assert not (copy / ".git").exists() and not (copy / "external").exists()
        assert "worker-only-value" not in script and "WORKER_SECRET" not in script
        assert (
            image == execution.settings.CODING_WORKSPACE_SANDBOX_IMAGE
            and timeout_seconds == 15
        )
        (copy / "app.py").write_text("changed in the sandbox")
        return 0, "done", ""

    monkeypatch.setattr(execution.agent_sandbox_runtime, "run_in_sandbox", sandbox)
    result = await execution.run_workspace_command(workspace, "pytest", timeout=15)
    assert result == (0, "done", "")
    assert (workspace / "app.py").read_text() == "before"
    assert list(sandbox_root.iterdir()) == []
