from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.models.agent_job import AgentJob, AgentJobStatus
from app.services.agent_coding_harness_service import (
    MUTATING_TOOL_NAMES,
    agent_coding_harness_service,
)
from app.services.autonomous_agent_executor import AutonomousAgentExecutor
from app.services.agent_tool_dispatch import (
    AgentToolExecutionContext,
    build_autonomous_workspace_mutation_provider,
)
from app.services.coding_workspace_manager import (
    CodingWorkspace,
    CodingWorkspaceManager,
)


def test_launch_defaults_add_modern_loop_and_preserve_overrides():
    config = agent_coding_harness_service.apply_launch_defaults(
        {
            "source_id": str(uuid4()),
            "native_tool_loop": {"max_tool_calls": 2},
            "coding_harness": {
                "verification": {"max_repair_attempts": 5},
            },
        },
        preset_key="bug_triage_swarm",
    )

    harness = config["coding_harness"]
    assert config["coding_harness_enabled"] is True
    assert harness["version"] == "v2"
    assert harness["loop"]["phases"] == [
        "gather_context",
        "act",
        "verify",
    ]
    assert harness["delegation"]["single_mutation_owner"] is True
    assert harness["verification"]["max_repair_attempts"] == 5
    assert harness["verification"]["allow_unverified_completion"] is False
    assert config["native_tool_loop"]["enabled"] is True
    assert config["native_tool_loop"]["max_tool_calls"] == 2
    assert config["auto_compaction"]["enabled"] is True
    assert config["verification_required_for_completion"] is True


def test_role_catalog_enforces_one_mutation_owner():
    roles = agent_coding_harness_service.get_role_catalog()

    for role_name in ("reproducer", "root_cause", "verifier"):
        config = roles[role_name]["config"]
        assert config["coding_harness_may_mutate"] is False
        assert set(MUTATING_TOOL_NAMES).issubset(config["blocked_tools"])
        assert not set(MUTATING_TOOL_NAMES).intersection(config["allowed_tools"])

    patcher_config = roles["patcher"]["config"]
    assert patcher_config["coding_harness_may_mutate"] is True
    assert set(MUTATING_TOOL_NAMES).issubset(patcher_config["allowed_tools"])
    assert patcher_config["blocked_tools"] == []
    assert "create_workspace_checkpoint" in patcher_config["allowed_tools"]
    assert "restore_workspace_checkpoint" in patcher_config["allowed_tools"]
    assert "hydrate_candidate_snapshot" in patcher_config["allowed_tools"]
    assert "persist_durable_workspace_checkpoint" in patcher_config["allowed_tools"]
    assert "restore_durable_workspace_checkpoint" in patcher_config["allowed_tools"]
    assert "hydrate_candidate_snapshot" in roles["verifier"]["config"]["allowed_tools"]
    assert (
        "restore_workspace_checkpoint" in roles["verifier"]["config"]["blocked_tools"]
    )
    assert (
        "restore_durable_workspace_checkpoint"
        in roles["verifier"]["config"]["blocked_tools"]
    )


def test_execution_evidence_requires_patch_and_successful_verification():
    evidence = agent_coding_harness_service.build_execution_evidence(
        {
            "coding_harness_role": "patcher",
            "coding_harness_may_mutate": True,
        },
        {
            "coding_modified_files": ["backend/app/parser.py"],
            "coding_pre_mutation_checkpoint_id": "checkpoint-baseline",
            "coding_hydrated_candidate_snapshot_id": "candidate-parser",
            "coding_command_history": [
                {
                    "command": "pytest -q backend/tests/test_parser.py",
                    "exit_code": 0,
                    "stdout_preview": "4 passed",
                }
            ],
        },
    )

    assert evidence["completion_eligible"] is True
    assert evidence["verification_state"] == "verified"
    assert evidence["modified_files"] == ["backend/app/parser.py"]
    assert evidence["successful_commands"] == 1
    assert evidence["workspace_recovery"]["baseline_checkpoint_id"] == (
        "checkpoint-baseline"
    )
    assert evidence["workspace_recovery"]["hydrated_candidate_snapshot_id"] == (
        "candidate-parser"
    )

    unverified = agent_coding_harness_service.build_execution_evidence(
        {
            "coding_harness_role": "patcher",
            "coding_harness_may_mutate": True,
        },
        {
            "coding_modified_files": ["backend/app/parser.py"],
            "coding_command_history": [
                {
                    "command": "pytest -q backend/tests/test_parser.py",
                    "exit_code": 1,
                }
            ],
        },
    )
    assert unverified["completion_eligible"] is False
    assert unverified["verification_state"] == "failed"


def test_discovers_bounded_repository_instruction_context(tmp_path):
    (tmp_path / ".kimi-code").mkdir()
    (tmp_path / "AGENTS.md").write_text(
        "Run focused tests before the full suite.",
        encoding="utf-8",
    )
    (tmp_path / "CLAUDE.md").write_text(
        "Read AGENTS.md and keep changes modular.",
        encoding="utf-8",
    )
    (tmp_path / ".kimi-code" / "AGENTS.md").write_text(
        "Use the repository sandbox.",
        encoding="utf-8",
    )
    workspace = CodingWorkspace(
        workspace_id="workspace-1",
        base_path=tmp_path,
    )

    context = agent_coding_harness_service.discover_project_instructions(workspace)

    assert context["trust"] == "untrusted_repository_context"
    assert context["permission_effect"] == "none"
    assert [item["path"] for item in context["files"]] == [
        "AGENTS.md",
        "CLAUDE.md",
        ".kimi-code/AGENTS.md",
    ]
    assert context["total_chars"] > 0


def test_prompt_and_swarm_children_receive_harness_contract():
    executor = AutonomousAgentExecutor()
    job = AgentJob(
        name="Harness test",
        goal="Repair the parser regression",
        job_type="analysis",
        user_id=uuid4(),
        status=AgentJobStatus.RUNNING.value,
        config=agent_coding_harness_service.apply_launch_defaults(
            {
                "launch_mode": "quick_start_bug_triage_swarm",
                "coding_swarm_enabled": True,
                "swarm_child_jobs_enabled": True,
                "swarm_max_agents": 4,
                "swarm_roles": [
                    "reproducer",
                    "root_cause",
                    "patcher",
                    "verifier",
                ],
            },
            preset_key="bug_triage_swarm",
        ),
        max_iterations=20,
        max_tool_calls=20,
        max_llm_calls=20,
        max_runtime_minutes=30,
    )
    state = {
        "swarm_chain_configured": False,
        "step_events": [],
        "coding_harness_context": {
            "files": [
                {
                    "path": "AGENTS.md",
                    "content": "Keep changes modular.",
                }
            ]
        },
    }

    prompt = agent_coding_harness_service.format_prompt_context(job, state)
    executor._ensure_swarm_chain_config(job, state)

    assert "gather context → act → verify" in prompt
    assert "AGENTS.md" in prompt
    children = job.chain_config["child_jobs"]
    assert children[0]["config"]["coding_harness_role"] == "reproducer"
    assert children[0]["config"]["coding_harness_may_mutate"] is False
    assert "write_file" not in children[0]["config"]["allowed_tools"]
    assert children[2]["config"]["coding_harness_role"] == "patcher"
    assert children[2]["config"]["coding_harness_may_mutate"] is True
    assert "apply_patch" in children[2]["config"]["allowed_tools"]
    assert children[2]["config"]["coding_workspace_session_id"] == (
        f"coding-session-{job.id}"
    )
    assert children[3]["config"]["handoff_contract"]["expected_outputs"] == [
        "verification_commands",
        "verification_evidence",
        "accept_or_reject",
    ]


def test_harness_fan_in_only_promotes_verified_mutation_owner():
    executor = AutonomousAgentExecutor()

    def _result(*, eligible: bool = False):
        return {
            "findings": [{"title": "Parser state is reset too early"}],
            "modified_files": ["backend/app/parser.py"],
            "verification_commands": ["pytest -q backend/tests/test_parser.py"],
            "coding_harness": {
                "completion_eligible": eligible,
                "verification_state": "verified" if eligible else "not_run",
                "modified_files": ["backend/app/parser.py"] if eligible else [],
                "candidate_snapshot": (
                    {
                        "type": "workspace_delta_snapshot",
                        "snapshot_id": "candidate-parser",
                        "session_id": "coding-session-root",
                        "files": [
                            {
                                "path": "backend/app/parser.py",
                                "sha256": "abc123",
                            }
                        ],
                        "immutable": True,
                    }
                    if eligible
                    else None
                ),
            },
        }

    payload = {
        "coding_swarm_enabled": True,
        "coding_harness_enabled": True,
        "expected_siblings": 4,
        "terminal_siblings": 4,
        "sibling_jobs": [
            {
                "job_id": "reproducer",
                "role": "Reproducer",
                "status": "completed",
                "results": _result(),
            },
            {
                "job_id": "root-cause",
                "role": "Root Cause Analyst",
                "status": "completed",
                "results": _result(),
            },
            {
                "job_id": "patcher",
                "role": "Primary Implementer",
                "status": "completed",
                "results": _result(eligible=True),
            },
            {
                "job_id": "verifier",
                "role": "Independent Verifier",
                "status": "completed",
                "results": _result(),
            },
        ],
    }

    promoted = executor._build_swarm_fan_in_result(payload)

    assert promoted["review_state"] == "auto_promoted"
    assert promoted["winning_slice_id"] == "patcher"
    assert promoted["verification_guardrail_met"] is True
    assert promoted["winning_candidate_snapshot"]["snapshot_id"] == ("candidate-parser")

    payload["sibling_jobs"][2]["results"] = _result(eligible=False)
    rejected = executor._build_swarm_fan_in_result(payload)

    assert rejected["review_state"] == "needs_review"
    assert rejected["winning_slice_id"] == ""
    assert rejected["verification_guardrail_met"] is False


@pytest.mark.asyncio
async def test_command_tool_rejects_destructive_commands_before_execution(tmp_path):
    manager = CodingWorkspaceManager()
    workspace = CodingWorkspace(
        workspace_id="workspace-unsafe",
        base_path=tmp_path,
    )
    manager._workspaces[workspace.workspace_id] = workspace
    provider = build_autonomous_workspace_mutation_provider(
        SimpleNamespace(workspace_manager=manager)
    )
    context = AgentToolExecutionContext(
        mode="autonomous",
        db=None,
        service=None,
        job=SimpleNamespace(config={}),
        state={"coding_workspace_id": workspace.workspace_id},
    )

    result = await provider.execute(
        "run_command",
        {"command": "sudo rm -rf /tmp/project"},
        context,
    )

    assert result["success"] is False
    assert result["data"]["blocked_commands"] == ["sudo rm -rf /tmp/project"]


@pytest.mark.asyncio
async def test_provider_rechecks_role_tool_policy_before_mutation(tmp_path):
    manager = CodingWorkspaceManager()
    workspace = CodingWorkspace(
        workspace_id="workspace-read-only",
        base_path=tmp_path,
    )
    manager._workspaces[workspace.workspace_id] = workspace
    provider = build_autonomous_workspace_mutation_provider(
        SimpleNamespace(workspace_manager=manager)
    )
    read_only_config = agent_coding_harness_service.get_role_catalog()["root_cause"][
        "config"
    ]
    context = AgentToolExecutionContext(
        mode="autonomous",
        db=None,
        service=None,
        job=SimpleNamespace(config=read_only_config),
        state={"coding_workspace_id": workspace.workspace_id},
    )

    result = await provider.execute(
        "write_file",
        {"path": "should-not-exist.py", "content": "unsafe = True\n"},
        context,
    )

    assert result["success"] is False
    assert "not permitted" in result["error"]
    assert not (tmp_path / "should-not-exist.py").exists()

    restore_result = await provider.execute(
        "restore_workspace_checkpoint",
        {"checkpoint_id": "checkpoint-forbidden"},
        context,
    )
    assert restore_result["success"] is False
    assert "not permitted" in restore_result["error"]


@pytest.mark.asyncio
async def test_command_tool_marks_nonzero_exit_as_failed_observation(
    tmp_path,
    monkeypatch,
):
    import asyncio

    from app.core import feature_flags

    manager = CodingWorkspaceManager()
    workspace = CodingWorkspace(
        workspace_id="workspace-command",
        base_path=tmp_path,
    )
    manager._workspaces[workspace.workspace_id] = workspace
    provider = build_autonomous_workspace_mutation_provider(
        SimpleNamespace(workspace_manager=manager)
    )
    context = AgentToolExecutionContext(
        mode="autonomous",
        db=None,
        service=None,
        job=SimpleNamespace(config={}),
        state={"coding_workspace_id": workspace.workspace_id},
    )

    async def _enabled(_name):
        return True

    class _Process:
        returncode = 1

        async def communicate(self):
            return b"", b"1 failed"

    async def _create_process(*_args, **_kwargs):
        return _Process()

    monkeypatch.setattr(feature_flags, "get_flag", _enabled)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", _create_process)

    result = await provider.execute(
        "run_command",
        {"command": "pytest -q", "timeout_seconds": 5},
        context,
    )

    assert result["success"] is False
    assert result["error"] == "Command exited with status 1"
    assert result["data"]["exit_code"] == 1
    assert result["data"]["stderr"] == "1 failed"
