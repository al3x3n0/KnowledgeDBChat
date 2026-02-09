"""
Built-in Agent Job Chain Definitions.

We keep some chain definitions in code (instead of DB) so they can ship as product
features without requiring migrations or manual seeding.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID


@dataclass(frozen=True)
class BuiltinAgentJobChainDefinition:
    id: UUID
    name: str
    display_name: str
    description: Optional[str]
    chain_steps: List[Dict[str, Any]]
    default_settings: Optional[Dict[str, Any]] = None
    owner_user_id: Optional[UUID] = None
    is_system: bool = True
    is_active: bool = True
    created_at: datetime = datetime(2026, 1, 1)
    updated_at: datetime = datetime(2026, 1, 1)

    def get_step_count(self) -> int:
        return len(self.chain_steps) if self.chain_steps else 0

    def get_step(self, index: int) -> Optional[Dict[str, Any]]:
        if not self.chain_steps or index >= len(self.chain_steps):
            return None
        return self.chain_steps[index]

    def create_job_config_for_step(
        self,
        step_index: int,
        variables: Dict[str, str],
        parent_results: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Create job configuration for a specific step.

        Mirrors AgentJobChainDefinition.create_job_config_for_step, but for built-in chains.
        """
        step = self.get_step(step_index)
        if not step:
            return None

        # Start with default settings
        config = dict(self.default_settings) if self.default_settings else {}

        # Apply step-specific config
        if step.get("config"):
            config.update(step["config"])

        # Build goal from template
        goal_template = step.get("goal_template", "")
        goal = goal_template
        for key, value in (variables or {}).items():
            goal = goal.replace(f"{{{key}}}", value)

        job_config: Dict[str, Any] = {
            "name": step.get("step_name", f"Chain Step {step_index + 1}"),
            "job_type": step.get("job_type", "custom"),
            "goal": goal,
            "config": config,
        }

        # Add template reference if specified
        if step.get("template_id"):
            job_config["template_id"] = step["template_id"]

        # Add chain configuration for next step trigger
        if step_index < len(self.chain_steps) - 1:
            job_config["chain_config"] = {
                "trigger_condition": step.get("trigger_condition", "on_complete"),
                "inherit_results": config.get("inherit_results", True),
            }
            if step.get("trigger_thresholds"):
                job_config["chain_config"].update(step["trigger_thresholds"])

        # Pass parent results if configured
        if parent_results and config.get("inherit_results", True):
            job_config["inherited_data"] = {"parent_results": parent_results}

        return job_config


CUSTOMER_RESEARCH_SCOUT_DEEP_DIVE_CHAIN_ID = UUID("9a5e6c41-6c04-4f5b-9adf-2d7fb3cbb5b8")
ARXIV_REPO_CODE_PATCH_CHAIN_ID = UUID("0b5a8bb2-7a2c-4a2b-8d51-02f9a4f20a77")
ARXIV_ALGORITHM_PROJECT_CHAIN_ID = UUID("44c0efc8-1f6b-4b67-8e0a-2bbf7fbb0d1d")
ARXIV_REPO_ALGORITHM_PROJECT_CHAIN_ID = UUID("7d8c4a5a-9d5b-4a55-9b41-6f39d12fb4b8")
RESEARCH_ENGINEER_CHAIN_ID = UUID("d2d9d7a5-2b0c-4c0c-bc2c-2e8d1b0f1d4c")
RESEARCH_ENGINEER_LOOP_CHAIN_ID = UUID("e2a1c11e-4b6d-4104-a3ba-fa66d8321a6b")
PAPER_PIPELINE_CHAIN_ID = UUID("9d62b9e2-1ed8-4e90-9a6d-2f0a6c0c6db5")
EXPERIMENT_LOOP_CHAIN_ID = UUID("8c38aa0e-92f6-4e58-9e57-6aaac7cfe0b2")
EXPERIMENT_LOOP_SEEDED_CHAIN_ID = UUID("9e267663-48d6-4a69-9679-984d1cdf6205")


BUILTIN_AGENT_JOB_CHAIN_DEFINITIONS: List[BuiltinAgentJobChainDefinition] = [
    BuiltinAgentJobChainDefinition(
        id=CUSTOMER_RESEARCH_SCOUT_DEEP_DIVE_CHAIN_ID,
        name="customer_research_scout_deep_dive_chain",
        display_name="Customer Research: Scout → Deep Dive",
        description=(
            "Two-step customer-specific research: a scout job to collect signals, then a deep-dive job "
            "that produces hypotheses + an experiment plan and persists a brief (optional)."
        ),
        chain_steps=[
            {
                "step_name": "Scout",
                "job_type": "research",
                "goal_template": "Customer-specific research scout: {goal}",
                "config": {
                    "prefer_sources": ["documents", "arxiv"],
                    "max_documents": 12,
                    "max_papers": 8,
                    "persist_artifacts": False,
                    "reading_list_name": "Customer Research",
                },
                "trigger_condition": "on_complete",
            },
            {
                "step_name": "Deep Dive",
                "job_type": "research",
                "goal_template": (
                    "Deep dive on the best sources from the scout. Goal: {goal}\n\n"
                    "Output: (1) 3-5 hypotheses, (2) risks/unknowns, (3) minimal experiment plan (metrics + timeline), "
                    "and (4) a short brief document if allowed."
                ),
                "config": {
                    "prefer_sources": ["documents"],
                    "max_documents": 6,
                    "max_papers": 0,
                },
            },
        ],
        default_settings={
            "inherit_results": True,
            "inherit_config": True,
            "max_iterations": 8,
            "max_tool_calls": 60,
            "max_llm_calls": 20,
            "max_runtime_minutes": 12,
        },
    ),
    BuiltinAgentJobChainDefinition(
        id=ARXIV_REPO_CODE_PATCH_CHAIN_ID,
        name="arxiv_repo_code_patch_chain",
        display_name="Paper → Repo → Code Patch",
        description=(
            "Extract repository links from an arXiv item, ingest the repository into the knowledge base, "
            "then generate a reviewable code patch proposal."
        ),
        chain_steps=[
            {
                "step_name": "Extract Repos",
                "job_type": "monitor",
                "goal_template": "Extract code repository links for arXiv inbox item {inbox_item_id}",
                "config": {
                    "deterministic_runner": "arxiv_inbox_extract_repos",
                },
                "trigger_condition": "on_complete",
            },
            {
                "step_name": "Ingest Repo",
                "job_type": "monitor",
                "goal_template": "Ingest the paper's code repository (auto-select if needed) and wait until files are available",
                "config": {
                    "deterministic_runner": "git_repo_ingest_wait",
                },
                "trigger_condition": "on_complete",
            },
            {
                "step_name": "Propose Patch",
                "job_type": "analysis",
                "goal_template": "Produce a minimal patch for: {goal}",
                "config": {
                    "deterministic_runner": "code_patch_proposer",
                },
            },
        ],
        default_settings={
            # Step parameters passed via config_overrides:
            # - inbox_item_id, optional provider/repo override, and optional token/gitlab_url.
            "inherit_results": True,
            "inherit_config": True,
            "max_iterations": 1,
            "max_tool_calls": 0,
            "max_llm_calls": 2,
            "max_runtime_minutes": 15,
            # Repo ingest tuning
            "git_ingest_max_pages": 5,
            "wait_seconds": 120,
            # Code patch proposer defaults
            "max_files": 8,
            "max_chars_per_file": 8000,
        },
    ),
    BuiltinAgentJobChainDefinition(
        id=ARXIV_ALGORITHM_PROJECT_CHAIN_ID,
        name="arxiv_algorithm_project_chain",
        display_name="Paper → Algorithm Implementation",
        description="Generate a small runnable reference implementation (and tests) of the paper's core algorithm.",
        chain_steps=[
            {
                "step_name": "Implement Algorithm",
                "job_type": "analysis",
                "goal_template": "Implement the core algorithm from arXiv inbox item {inbox_item_id} as a runnable reference project",
                "config": {
                    "deterministic_runner": "paper_algorithm_project",
                },
            }
        ],
        default_settings={
            "inherit_results": True,
            "inherit_config": True,
            "max_iterations": 1,
            "max_tool_calls": 0,
            "max_llm_calls": 2,
            "max_runtime_minutes": 15,
            # Parameters passed via config_overrides:
            # - inbox_item_id, language, include_tests
        },
    ),
    BuiltinAgentJobChainDefinition(
        id=ARXIV_REPO_ALGORITHM_PROJECT_CHAIN_ID,
        name="arxiv_repo_algorithm_project_chain",
        display_name="Paper → Repo → Algorithm Implementation",
        description="Extract the paper's repo, ingest it, then generate a runnable reference implementation using the repo as guidance.",
        chain_steps=[
            {
                "step_name": "Extract Repos",
                "job_type": "monitor",
                "goal_template": "Extract code repository links for arXiv inbox item {inbox_item_id}",
                "config": {
                    "deterministic_runner": "arxiv_inbox_extract_repos",
                },
                "trigger_condition": "on_complete",
            },
            {
                "step_name": "Ingest Repo",
                "job_type": "monitor",
                "goal_template": "Ingest the paper's code repository (auto-select if needed) and wait until files are available",
                "config": {
                    "deterministic_runner": "git_repo_ingest_wait",
                },
                "trigger_condition": "on_complete",
            },
            {
                "step_name": "Implement Algorithm",
                "job_type": "analysis",
                "goal_template": "Implement the core algorithm from arXiv inbox item {inbox_item_id} as a runnable reference project (use ingested repo as guidance)",
                "config": {
                    "deterministic_runner": "paper_algorithm_project",
                },
            },
        ],
        default_settings={
            "inherit_results": True,
            "inherit_config": True,
            "max_iterations": 1,
            "max_tool_calls": 0,
            "max_llm_calls": 2,
            "max_runtime_minutes": 20,
            # Repo ingest tuning
            "git_ingest_max_pages": 5,
            "wait_seconds": 180,
            # Paper algorithm project tuning
            "language": "python",
            "include_tests": True,
            "use_repo_context": True,
            "max_repo_files": 8,
            "max_chars_per_repo_file": 8000,
        },
    ),
    BuiltinAgentJobChainDefinition(
        id=RESEARCH_ENGINEER_CHAIN_ID,
        name="research_engineer_chain",
        display_name="ResearchEngineer: Scientist → Code Patch → Paper Update",
        description=(
            "A combined AI Scientist + Code Agent workflow. Step 1 drafts a minimal experiment plan and optionally appends "
            "a LaTeX section into a LaTeX Studio project. Step 2 generates a reviewable code patch proposal against a git "
            "DocumentSource. Step 3 appends implementation notes into the same LaTeX project.\n\n"
            "Required config_overrides to start:\n"
            "- latex_project_id: UUID (LaTeX Studio project)\n"
            "- target_source_id: UUID (git DocumentSource for code patch proposer)\n"
            "Optional:\n"
            "- search_query: string (KB query)\n"
            "- file_paths: [string]\n"
        ),
        chain_steps=[
            {
                "step_name": "Scientist Plan",
                "job_type": "analysis",
                "goal_template": "Draft a minimal hypothesis + experiment plan for: {goal}",
                "config": {
                    "deterministic_runner": "research_engineer_scientist",
                },
                "trigger_condition": "on_complete",
            },
            {
                "step_name": "Propose Patch",
                "job_type": "analysis",
                "goal_template": "Implement the requested change as a minimal patch: {goal}",
                "config": {
                    "deterministic_runner": "code_patch_proposer",
                },
                "trigger_condition": "on_complete",
            },
            {
                "step_name": "Update Paper",
                "job_type": "analysis",
                "goal_template": "Update the LaTeX project with implementation notes from the patch proposal",
                "config": {
                    "deterministic_runner": "research_engineer_paper_update",
                },
            },
        ],
        default_settings={
            "inherit_results": True,
            "max_iterations": 1,
            "max_tool_calls": 0,
            "max_llm_calls": 2,
            "max_runtime_minutes": 15,
            "max_documents": 8,
            "max_files": 8,
            "max_chars_per_file": 8000,
            # Provided via config_overrides at launch:
            "latex_project_id": "",
            "target_source_id": "",
            "search_query": "",
            "file_paths": [],
        },
    ),
    BuiltinAgentJobChainDefinition(
        id=RESEARCH_ENGINEER_LOOP_CHAIN_ID,
        name="research_engineer_loop_chain",
        display_name="ResearchEngineer Loop: Plan → Patch → Run → Patch → Run → Paper",
        description=(
            "A combined AI Scientist + Code Agent loop grounded in a git DocumentSource and a LaTeX Studio project.\n\n"
            "Flow:\n"
            "1) Scientist drafts a minimal hypothesis + experiment plan.\n"
            "2) Code Agent proposes a unified diff patch.\n"
            "3) ExperimentRunner applies the patch (in-memory) and runs commands/tests.\n"
            "4) Code Agent refines the patch using experiment output.\n"
            "5) ExperimentRunner re-runs the commands/tests against the refined patch.\n"
            "6) Paper update appends implementation notes (and experiment summary).\n\n"
            "Required config_overrides to start:\n"
            "- latex_project_id: UUID (LaTeX Studio project)\n"
            "- target_source_id: UUID (git DocumentSource for code patch proposer)\n"
            "Optional:\n"
            "- search_query: string (KB query)\n"
            "- file_paths: [string]\n"
            "- commands: [string] (experiment commands; defaults to tests_to_run from patch)\n"
            "- enable_experiments: bool (default true)\n"
        ),
        chain_steps=[
            {
                "step_name": "Scientist Plan",
                "job_type": "analysis",
                "goal_template": "Draft a minimal hypothesis + experiment plan for: {goal}",
                "config": {"deterministic_runner": "research_engineer_scientist"},
                "trigger_condition": "on_complete",
            },
            {
                "step_name": "Propose Patch (Round 1)",
                "job_type": "analysis",
                "goal_template": "Implement the requested change as a minimal patch: {goal}",
                "config": {"deterministic_runner": "code_patch_proposer"},
                "trigger_condition": "on_complete",
            },
            {
                "step_name": "Run Experiments (Round 1)",
                "job_type": "analysis",
                "goal_template": "Run the proposed tests/experiments and record results for: {goal}",
                "config": {"deterministic_runner": "experiment_runner"},
                "trigger_condition": "on_any_end",
            },
            {
                "step_name": "Refine Patch (Round 2)",
                "job_type": "analysis",
                "goal_template": "Refine the patch based on experiment output for: {goal}",
                "config": {"deterministic_runner": "code_patch_proposer"},
                "trigger_condition": "on_complete",
            },
            {
                "step_name": "Run Experiments (Round 2)",
                "job_type": "analysis",
                "goal_template": "Re-run the tests/experiments against the refined patch and record results for: {goal}",
                "config": {"deterministic_runner": "experiment_runner"},
                "trigger_condition": "on_any_end",
            },
            {
                "step_name": "KB Apply Dry-Run (Optional)",
                "job_type": "analysis",
                "goal_template": "Dry-run apply the final patch proposal to KnowledgeDB code documents",
                "config": {
                    "deterministic_runner": "code_patch_apply_to_kb",
                    "enabled_key": "apply_patch_to_kb",
                    "dry_run": True,
                    "require_experiments_ok": False,
                },
                "trigger_condition": "on_any_end",
            },
            {
                "step_name": "KB Apply Write (Optional)",
                "job_type": "analysis",
                "goal_template": "Apply the final patch proposal to KnowledgeDB code documents (write)",
                "config": {
                    "deterministic_runner": "code_patch_apply_to_kb",
                    "enabled_key": "apply_patch_to_kb_confirm",
                    "dry_run": False,
                },
                "trigger_condition": "on_any_end",
            },
            {
                "step_name": "Update Paper",
                "job_type": "analysis",
                "goal_template": "Update the LaTeX project with implementation notes and experiment summary",
                "config": {"deterministic_runner": "research_engineer_paper_update"},
            },
        ],
        default_settings={
            "inherit_results": True,
            "max_iterations": 1,
            "max_tool_calls": 0,
            "max_llm_calls": 2,
            "max_runtime_minutes": 25,
            # KB grounding
            "max_documents": 8,
            # Code patch proposer
            "max_files": 8,
            "max_chars_per_file": 8000,
            # Experiments defaults
            "enable_experiments": True,
            "commands": [],
            # Optional: apply patch proposal to KB
            "apply_patch_to_kb": False,
            "apply_patch_to_kb_confirm": False,
            "proposal_strategy": "best_passing",
            "require_experiments_ok": True,
            "require_dry_run_first": True,
            "fail_on_block": False,
            # Provided via config_overrides at launch:
            "latex_project_id": "",
            "target_source_id": "",
            "search_query": "",
            "file_paths": [],
        },
    ),
    BuiltinAgentJobChainDefinition(
        id=EXPERIMENT_LOOP_SEEDED_CHAIN_ID,
        name="experiment_loop_seeded_chain",
        display_name="Experiment Loop: Seeded (configurable)",
        description=(
            "Seeds a configurable experiment loop as a chained job sequence.\n\n"
            "This version supports a configurable number of runs via config_overrides.max_runs.\n\n"
            "Required config_overrides to start:\n"
            "- research_note_id: UUID\n"
            "- source_id: UUID (git DocumentSource)\n"
            "- commands: [string] baseline commands (one per command)\n"
            "Optional:\n"
            "- max_runs: int (default 3)\n"
            "- command_variants: [[string]] or [{name, commands}] to drive ablations deterministically\n"
            "- use_llm_decider: bool (let LLM propose next commands)\n"
        ),
        chain_steps=[
            {
                "step_name": "Seed Loop",
                "job_type": "analysis",
                "goal_template": "Seed experiment loop for research note {research_note_id}",
                "config": {"deterministic_runner": "experiment_loop_seed"},
            }
        ],
        default_settings={
            "inherit_results": True,
            "inherit_config": True,
            "max_iterations": 1,
            "max_tool_calls": 0,
            "max_llm_calls": 2,
            "max_runtime_minutes": 12,
            # Required via config_overrides:
            "research_note_id": "",
            "source_id": "",
            "commands": [],
            # Optional:
            "max_runs": 3,
            "command_variants": [],
            "use_llm_decider": False,
            "append_to_note": True,
            "enable_experiments": True,
            "timeout_seconds": 60,
        },
    ),
    BuiltinAgentJobChainDefinition(
        id=EXPERIMENT_LOOP_CHAIN_ID,
        name="experiment_loop_chain",
        display_name="Experiment Loop: Plan → Decide → Run → Persist (x3)",
        description=(
            "Autonomous experiment loop grounded in a Research Note and a git DocumentSource.\n\n"
            "Flow:\n"
            "1) Generate an ExperimentPlan from the note (Hypothesis).\n"
            "2) Decide next command variant + run name.\n"
            "3) Run the commands via ExperimentRunner.\n"
            "4) Persist results into ExperimentRun and append to the note.\n"
            "Repeats steps 2-4 three times (baseline + 2 ablations).\n\n"
            "Required config_overrides to start:\n"
            "- research_note_id: UUID\n"
            "- source_id: UUID (git DocumentSource)\n"
            "- commands: [string] baseline commands (one per command)\n"
            "Optional:\n"
            "- command_variants: [[string]] or [{name, commands}] to drive ablations deterministically\n"
            "- use_llm_decider: bool (let LLM propose next commands)\n"
        ),
        chain_steps=[
            {
                "step_name": "Generate Plan",
                "job_type": "analysis",
                "goal_template": "Generate an experiment plan from research note {research_note_id}",
                "config": {"deterministic_runner": "experiment_plan_generate"},
                "trigger_condition": "on_complete",
            },
            {
                "step_name": "Decide Next (1)",
                "job_type": "analysis",
                "goal_template": "Decide baseline commands for research note {research_note_id}",
                "config": {"deterministic_runner": "experiment_decide_next"},
                "trigger_condition": "on_complete",
            },
            {
                "step_name": "Run (1)",
                "job_type": "analysis",
                "goal_template": "Run baseline commands for research note {research_note_id}",
                "config": {"deterministic_runner": "experiment_runner"},
                "trigger_condition": "on_any_end",
            },
            {
                "step_name": "Persist (1)",
                "job_type": "analysis",
                "goal_template": "Persist baseline results for research note {research_note_id}",
                "config": {"deterministic_runner": "experiment_persist_results", "append_to_note": True},
                "trigger_condition": "on_any_end",
            },
            {
                "step_name": "Decide Next (2)",
                "job_type": "analysis",
                "goal_template": "Decide next ablation commands for research note {research_note_id}",
                "config": {"deterministic_runner": "experiment_decide_next"},
                "trigger_condition": "on_complete",
            },
            {
                "step_name": "Run (2)",
                "job_type": "analysis",
                "goal_template": "Run ablation commands for research note {research_note_id}",
                "config": {"deterministic_runner": "experiment_runner"},
                "trigger_condition": "on_any_end",
            },
            {
                "step_name": "Persist (2)",
                "job_type": "analysis",
                "goal_template": "Persist ablation results for research note {research_note_id}",
                "config": {"deterministic_runner": "experiment_persist_results", "append_to_note": True},
                "trigger_condition": "on_any_end",
            },
            {
                "step_name": "Decide Next (3)",
                "job_type": "analysis",
                "goal_template": "Decide final ablation commands for research note {research_note_id}",
                "config": {"deterministic_runner": "experiment_decide_next"},
                "trigger_condition": "on_complete",
            },
            {
                "step_name": "Run (3)",
                "job_type": "analysis",
                "goal_template": "Run final ablation commands for research note {research_note_id}",
                "config": {"deterministic_runner": "experiment_runner"},
                "trigger_condition": "on_any_end",
            },
            {
                "step_name": "Persist (3)",
                "job_type": "analysis",
                "goal_template": "Persist final ablation results for research note {research_note_id}",
                "config": {"deterministic_runner": "experiment_persist_results", "append_to_note": True},
            },
        ],
        default_settings={
            "inherit_results": True,
            "inherit_config": True,
            "max_iterations": 1,
            "max_tool_calls": 0,
            "max_llm_calls": 2,
            "max_runtime_minutes": 12,
            # Required via config_overrides:
            "research_note_id": "",
            "source_id": "",
            "commands": [],
            # Optional:
            "command_variants": [],
            "use_llm_decider": False,
            "append_to_note": True,
            "enable_experiments": True,
            "timeout_seconds": 60,
        },
    ),
    BuiltinAgentJobChainDefinition(
        id=PAPER_PIPELINE_CHAIN_ID,
        name="paper_pipeline_chain",
        display_name="PaperPipeline: Plan → Patch → Run → Cite → Review → Compile → Publish",
        description=(
            "End-to-end paper workflow grounded in the Knowledge DB and a LaTeX Studio project.\n\n"
            "Required config_overrides to start:\n"
            "- latex_project_id: UUID (LaTeX Studio project)\n"
            "- target_source_id: UUID (git DocumentSource)\n"
            "Optional:\n"
            "- search_query: string (KB query)\n"
            "- file_paths: [string] (limit patch context)\n"
            "- commands: [string] (experiment commands; defaults to inherited tests_to_run)\n"
        ),
        chain_steps=[
            {
                "step_name": "Scientist Plan",
                "job_type": "analysis",
                "goal_template": "Draft a minimal hypothesis + experiment plan for: {goal}",
                "config": {"deterministic_runner": "research_engineer_scientist"},
                "trigger_condition": "on_complete",
            },
            {
                "step_name": "Propose Patch",
                "job_type": "analysis",
                "goal_template": "Implement the requested change as a minimal patch: {goal}",
                "config": {"deterministic_runner": "code_patch_proposer"},
                "trigger_condition": "on_complete",
            },
            {
                "step_name": "Run Experiments",
                "job_type": "analysis",
                "goal_template": "Run the proposed tests/experiments and record results for: {goal}",
                "config": {"deterministic_runner": "experiment_runner"},
                "trigger_condition": "on_complete",
            },
            {
                "step_name": "Sync Citations",
                "job_type": "analysis",
                "goal_template": "Synchronize KDB citations into refs.bib / thebibliography",
                "config": {"deterministic_runner": "latex_citation_sync"},
                "trigger_condition": "on_complete",
            },
            {
                "step_name": "Reviewer/Critic",
                "job_type": "analysis",
                "goal_template": "Review paper.tex for missing citations and clarity; propose minimal diff fixes",
                "config": {"deterministic_runner": "latex_reviewer_critic"},
                "trigger_condition": "on_complete",
            },
            {
                "step_name": "Apply Review Diff (Optional)",
                "job_type": "analysis",
                "goal_template": "Apply the reviewer’s suggested diff to paper.tex (optional)",
                "config": {"deterministic_runner": "latex_apply_unified_diff"},
                "trigger_condition": "on_complete",
            },
            {
                "step_name": "Compile PDF",
                "job_type": "analysis",
                "goal_template": "Compile the LaTeX project to PDF (queue compile job if needed)",
                "config": {"deterministic_runner": "latex_compile_project"},
                "trigger_condition": "on_complete",
            },
            {
                "step_name": "Publish",
                "job_type": "analysis",
                "goal_template": "Publish the LaTeX and PDF into the Knowledge DB",
                "config": {"deterministic_runner": "latex_publish_project"},
            },
        ],
        default_settings={
            "inherit_results": True,
            "max_iterations": 1,
            "max_tool_calls": 0,
            "max_llm_calls": 2,
            "max_runtime_minutes": 25,
            # KB grounding
            "max_documents": 8,
            # Code patch proposer
            "max_files": 8,
            "max_chars_per_file": 8000,
            # Citation sync defaults
            "mode": "bibtex",
            "bib_filename": "refs.bib",
            "enable_citation_sync": True,
            # Reviewer defaults
            "focus": "",
            "enable_reviewer": True,
            # Optional: apply reviewer diff
            "apply_review_diff": False,
            # Compile defaults
            "safe_mode": True,
            "preferred_engine": None,
            "use_worker": True,
            "wait_seconds": 120,
            "skip_if_unavailable": True,
            "enable_compile": True,
            # Publish defaults
            "include_tex": True,
            "include_pdf": True,
            "publish_tags": "latex,paper",
            "enable_publish": True,
            # Experiments defaults
            "enable_experiments": True,
            # Provided via config_overrides at launch:
            "latex_project_id": "",
            "target_source_id": "",
            "search_query": "",
            "file_paths": [],
            "commands": [],
        },
    ),
]


def list_builtin_agent_job_chain_definitions() -> List[BuiltinAgentJobChainDefinition]:
    return [c for c in BUILTIN_AGENT_JOB_CHAIN_DEFINITIONS if c.is_active]


def get_builtin_agent_job_chain_definition(chain_id: UUID) -> Optional[BuiltinAgentJobChainDefinition]:
    for c in BUILTIN_AGENT_JOB_CHAIN_DEFINITIONS:
        if c.id == chain_id and c.is_active:
            return c
    return None
