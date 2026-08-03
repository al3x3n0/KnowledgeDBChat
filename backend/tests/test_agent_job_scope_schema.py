from app.schemas.agent_job import (
    AgentJobChainDefinitionCreate,
    AgentJobCreate,
    AgentJobFromChainCreate,
    AgentJobFromTemplate,
    AgentJobQuickStartClaudeBackendRequest,
    AgentJobQuickStartRepoBugTriageRequest,
    AgentJobUpdate,
    ChainStepConfig,
)


def test_agent_job_create_normalizes_target_source_id_to_source_id():
    payload = AgentJobCreate(
        name="Scope Test",
        goal="Test normalization",
        config={"target_source_id": "abc-123"},
    )

    assert payload.config is not None
    assert payload.config.get("source_id") == "abc-123"
    assert "target_source_id" not in payload.config


def test_agent_job_update_normalizes_scope_config():
    payload = AgentJobUpdate(config={"target_source_id": "xyz-789"})

    assert payload.config is not None
    assert payload.config.get("source_id") == "xyz-789"
    assert "target_source_id" not in payload.config


def test_agent_job_template_override_normalizes_scope_config():
    payload = AgentJobFromTemplate(
        template_id="00000000-0000-0000-0000-000000000001",
        name="Template Scope Test",
        config={"target_source_id": "tpl-1"},
    )

    assert payload.config is not None
    assert payload.config.get("source_id") == "tpl-1"
    assert "target_source_id" not in payload.config


def test_chain_schema_normalizes_step_and_default_scope_keys():
    chain = AgentJobChainDefinitionCreate(
        name="chain_scope",
        display_name="Chain Scope",
        chain_steps=[
            ChainStepConfig(
                step_name="Step 1",
                goal_template="Do thing",
                config={"target_source_id": "step-42"},
            )
        ],
        default_settings={"target_source_id": "default-42"},
    )

    step_cfg = chain.chain_steps[0].config or {}
    assert step_cfg.get("source_id") == "step-42"
    assert "target_source_id" not in step_cfg
    assert chain.default_settings is not None
    assert chain.default_settings.get("source_id") == "default-42"
    assert "target_source_id" not in chain.default_settings


def test_from_chain_overrides_normalize_scope_keys():
    payload = AgentJobFromChainCreate(
        chain_definition_id="00000000-0000-0000-0000-000000000002",
        name_prefix="Chain Run",
        variables={},
        config_overrides={"target_source_id": "ovr-1"},
    )

    assert payload.config_overrides is not None
    assert payload.config_overrides.get("source_id") == "ovr-1"
    assert "target_source_id" not in payload.config_overrides


def test_quick_start_request_normalizes_config_override_scope_keys():
    payload = AgentJobQuickStartClaudeBackendRequest(
        goal="Fix backend API tests",
        source_id="00000000-0000-0000-0000-000000000123",
        config_overrides={"target_source_id": "ovr-quick"},
    )

    assert payload.config_overrides is not None
    assert payload.config_overrides.get("source_id") == "ovr-quick"
    assert "target_source_id" not in payload.config_overrides


def test_quick_start_request_normalizes_commands_and_file_paths():
    payload = AgentJobQuickStartClaudeBackendRequest(
        goal="Fix backend API tests",
        source_id="00000000-0000-0000-0000-000000000123",
        commands=[
            "  python -m pytest -q  ",
            "python -m pytest -q",
            "npm test",
            "",
            "   ",
        ],
        file_paths=[
            " backend/app/main.py ",
            "backend/app/main.py",
            "",
            "backend/tests/test_api.py",
        ],
        search_query="   backend regression tests   ",
    )

    assert payload.commands == ["python -m pytest -q", "npm test"]
    assert payload.file_paths == ["backend/app/main.py", "backend/tests/test_api.py"]
    assert payload.search_query == "backend regression tests"


def test_quick_start_request_drops_unsafe_file_paths():
    payload = AgentJobQuickStartClaudeBackendRequest(
        goal="Fix backend API tests",
        source_id="00000000-0000-0000-0000-000000000123",
        file_paths=[
            "/etc/passwd",
            "../backend/app/main.py",
            "C:\\Windows\\system32\\cmd.exe",
            "./backend/app/api/endpoints/agent_jobs.py",
            "backend//tests/./test_agent_job_scope_schema.py",
        ],
    )

    assert payload.file_paths == [
        "backend/app/api/endpoints/agent_jobs.py",
        "backend/tests/test_agent_job_scope_schema.py",
    ]


def test_repo_bug_triage_request_normalizes_scope_config_and_lists():
    payload = AgentJobQuickStartRepoBugTriageRequest(
        failure_symptom="Frontend login spinner never stops",
        source_id="00000000-0000-0000-0000-000000000123",
        scope=" Front-End ",
        commands=["  npm test  ", "npm test", "CI=true npm test -- --watchAll=false"],
        file_paths=[
            " frontend/src/App.tsx ",
            "./frontend/src/pages/LoginPage.tsx",
            "../oops",
        ],
        config_overrides={"target_source_id": "ovr-repo"},
    )

    assert payload.scope == "frontend"
    assert payload.commands == ["npm test", "CI=true npm test -- --watchAll=false"]
    assert payload.file_paths == [
        "frontend/src/App.tsx",
        "frontend/src/pages/LoginPage.tsx",
    ]
    assert payload.config_overrides is not None
    assert payload.config_overrides.get("source_id") == "ovr-repo"
    assert "target_source_id" not in payload.config_overrides


def test_repo_bug_triage_request_requires_goal_or_symptom():
    try:
        AgentJobQuickStartRepoBugTriageRequest(
            source_id="00000000-0000-0000-0000-000000000123",
        )
    except Exception as exc:
        assert "goal or failure_symptom" in str(exc).lower()
    else:
        raise AssertionError(
            "Expected validation failure when goal and failure_symptom are both missing"
        )
