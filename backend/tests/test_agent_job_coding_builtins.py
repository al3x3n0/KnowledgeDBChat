from app.services.agent_job_chain_templates import (
    CLAUDE_CODE_BACKEND_CHAIN_ID,
    REPO_BUG_TRIAGE_REPAIR_CHAIN_ID,
    get_builtin_agent_job_chain_definition,
)
from app.services.agent_job_templates import (
    CLAUDE_CODE_BACKEND_TEMPLATE_ID,
    REPO_BUG_TRIAGE_REPAIR_TEMPLATE_ID,
    get_builtin_agent_job_template,
)


def _assert_no_target_source_id(value):
    if isinstance(value, dict):
        assert "target_source_id" not in value
        for v in value.values():
            _assert_no_target_source_id(v)
    elif isinstance(value, list):
        for item in value:
            _assert_no_target_source_id(item)


def test_claude_code_backend_chain_is_registered_with_expected_steps():
    chain = get_builtin_agent_job_chain_definition(CLAUDE_CODE_BACKEND_CHAIN_ID)

    assert chain is not None
    assert chain.name == "claude_code_backend_chain"
    assert chain.get_step_count() == 6

    step0 = chain.get_step(0) or {}
    step1 = chain.get_step(1) or {}
    step2 = chain.get_step(2) or {}
    step3 = chain.get_step(3) or {}
    step4 = chain.get_step(4) or {}
    step5 = chain.get_step(5) or {}

    assert (step0.get("config") or {}).get(
        "deterministic_runner"
    ) == "code_patch_proposer"
    assert (step1.get("config") or {}).get(
        "deterministic_runner"
    ) == "experiment_runner"
    assert (step2.get("config") or {}).get(
        "deterministic_runner"
    ) == "code_patch_proposer"
    assert (step3.get("config") or {}).get(
        "deterministic_runner"
    ) == "experiment_runner"
    assert (step4.get("config") or {}).get(
        "deterministic_runner"
    ) == "code_patch_apply_to_kb"
    assert (step5.get("config") or {}).get(
        "deterministic_runner"
    ) == "code_patch_apply_to_kb"

    defaults = chain.default_settings or {}
    assert defaults.get("source_id") == ""
    assert defaults.get("search_query") == "backend"
    assert defaults.get("auto_commands_from_project_profile") is True
    assert defaults.get("apply_patch_to_kb") is False
    assert defaults.get("apply_patch_to_kb_confirm") is False


def test_claude_code_backend_template_is_registered_with_chain():
    template = get_builtin_agent_job_template(CLAUDE_CODE_BACKEND_TEMPLATE_ID)

    assert template is not None
    assert template.name == "claude_code_backend"
    assert template.category == "code"
    assert template.default_config.get("deterministic_runner") == "code_patch_proposer"
    assert template.default_config.get("search_query") == "backend"
    assert template.default_config.get("auto_commands_from_project_profile") is True

    chain_cfg = template.default_chain_config or {}
    child_jobs = chain_cfg.get("child_jobs") or []
    assert child_jobs
    verify_round_1 = child_jobs[0]
    assert (verify_round_1.get("config") or {}).get(
        "deterministic_runner"
    ) == "experiment_runner"

    nested = (verify_round_1.get("chain_config") or {}).get("child_jobs") or []
    assert nested
    refine_round_2 = nested[0]
    assert (refine_round_2.get("config") or {}).get(
        "deterministic_runner"
    ) == "code_patch_proposer"


def test_claude_code_backend_builtins_do_not_use_legacy_target_source_id_keys():
    chain = get_builtin_agent_job_chain_definition(CLAUDE_CODE_BACKEND_CHAIN_ID)
    template = get_builtin_agent_job_template(CLAUDE_CODE_BACKEND_TEMPLATE_ID)

    assert chain is not None
    assert template is not None
    _assert_no_target_source_id(chain.default_settings)
    _assert_no_target_source_id(chain.chain_steps)
    _assert_no_target_source_id(template.default_config)
    _assert_no_target_source_id(template.default_chain_config)


def test_repo_bug_triage_chain_is_registered_with_expected_steps():
    chain = get_builtin_agent_job_chain_definition(REPO_BUG_TRIAGE_REPAIR_CHAIN_ID)

    assert chain is not None
    assert chain.name == "repo_bug_triage_repair_chain"
    assert chain.get_step_count() == 4

    step0 = chain.get_step(0) or {}
    step1 = chain.get_step(1) or {}
    step2 = chain.get_step(2) or {}
    step3 = chain.get_step(3) or {}

    assert (step0.get("config") or {}).get(
        "deterministic_runner"
    ) == "code_patch_proposer"
    assert (step1.get("config") or {}).get(
        "deterministic_runner"
    ) == "experiment_runner"
    assert (step2.get("config") or {}).get(
        "deterministic_runner"
    ) == "code_patch_proposer"
    assert (step3.get("config") or {}).get(
        "deterministic_runner"
    ) == "experiment_runner"

    defaults = chain.default_settings or {}
    assert defaults.get("source_id") == ""
    assert defaults.get("failure_symptom") == ""
    assert defaults.get("scope") == "auto"
    assert defaults.get("auto_commands_from_project_profile") is True
    assert defaults.get("create_workspace_from_source") is True
    assert defaults.get("emit_execution_plan") is True
    assert defaults.get("max_verification_commands") == 3
    assert defaults.get("apply_patch_to_kb") is False


def test_repo_bug_triage_template_is_registered_with_chain():
    template = get_builtin_agent_job_template(REPO_BUG_TRIAGE_REPAIR_TEMPLATE_ID)

    assert template is not None
    assert template.name == "repo_bug_triage_repair"
    assert template.category == "code"
    assert template.default_config.get("deterministic_runner") == "code_patch_proposer"
    assert template.default_config.get("scope") == "auto"
    assert template.default_config.get("auto_commands_from_project_profile") is True
    assert template.default_config.get("create_workspace_from_source") is True
    assert template.default_config.get("emit_execution_plan") is True
    assert template.default_config.get("max_verification_commands") == 3
    assert template.default_config.get("apply_patch_to_kb") is False

    chain_cfg = template.default_chain_config or {}
    child_jobs = chain_cfg.get("child_jobs") or []
    assert child_jobs
    verify_round_1 = child_jobs[0]
    assert (verify_round_1.get("config") or {}).get(
        "deterministic_runner"
    ) == "experiment_runner"


def test_repo_bug_triage_builtins_do_not_use_legacy_target_source_id_keys():
    chain = get_builtin_agent_job_chain_definition(REPO_BUG_TRIAGE_REPAIR_CHAIN_ID)
    template = get_builtin_agent_job_template(REPO_BUG_TRIAGE_REPAIR_TEMPLATE_ID)

    assert chain is not None
    assert template is not None
    _assert_no_target_source_id(chain.default_settings)
    _assert_no_target_source_id(chain.chain_steps)
    _assert_no_target_source_id(template.default_config)
    _assert_no_target_source_id(template.default_chain_config)
