"""
Built-in Agent Job Templates.

We keep some templates in code (instead of DB) so they can ship as product
features without requiring migrations or manual seeding.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID


@dataclass(frozen=True)
class BuiltinAgentJobTemplate:
    id: UUID
    name: str
    display_name: str
    description: str
    category: str
    job_type: str
    default_goal: str
    default_config: Dict[str, Any]
    agent_definition_id: Optional[UUID] = None
    default_chain_config: Optional[Dict[str, Any]] = None
    default_max_iterations: int = 10
    default_max_tool_calls: int = 50
    default_max_llm_calls: int = 0
    default_max_runtime_minutes: int = 5
    is_system: bool = True
    is_active: bool = True
    owner_user_id: Optional[UUID] = None
    created_at: datetime = datetime(2026, 1, 1)
    updated_at: datetime = datetime(2026, 1, 1)


AI_HUB_SCIENTIST_TEMPLATE_ID = UUID("b5967d2e-2f8c-4d65-9c6b-1a4f8c0a5d2d")
CUSTOMER_RESEARCH_SCOUT_TEMPLATE_ID = UUID("4b2b7a57-4c2f-4e2b-8b10-6d041a1a41df")
CUSTOMER_RESEARCH_SCOUT_DEEP_DIVE_TEMPLATE_ID = UUID("9c4a92a2-36a6-4f0a-8d19-7f9dcaf26434")
RESEARCH_INBOX_MONITOR_TEMPLATE_ID = UUID("2a4f2b12-3e7a-4a64-9f1a-6b4b0f0b3b52")
CODE_PATCH_PROPOSER_TEMPLATE_ID = UUID("6b5ddc3b-2d6c-4a48-9a70-8b6a4c1f55a9")
LATEX_CITATION_SYNC_TEMPLATE_ID = UUID("c6c6e9c2-0f6d-4e64-9e70-9d0d5a5c1a11")
LATEX_REVIEWER_CRITIC_TEMPLATE_ID = UUID("8f0d4d6f-0a93-4a4b-9b18-7f2f3b5b7c8d")
EXPERIMENT_RUNNER_TEMPLATE_ID = UUID("4e3c2b1a-9f8e-4d3c-b2a1-1c2d3e4f5a6b")
LATEX_COMPILE_PROJECT_TEMPLATE_ID = UUID("f0b7c0e7-3a39-4a9b-9b8e-3dd89a2dcbf4")
LATEX_PUBLISH_PROJECT_TEMPLATE_ID = UUID("3e8b4f4a-9d2f-4c73-9f0e-7d5ac97acdb3")
CLAUDE_CODE_BACKEND_TEMPLATE_ID = UUID("5b24fc8a-1e6f-4a49-9c90-f53b6fdac2a1")
REPO_BUG_TRIAGE_REPAIR_TEMPLATE_ID = UUID("e9fb641d-8ca2-4f17-aedd-df5e76d6a38e")
DOMAIN_RESEARCH_TEMPLATE_ID = UUID("51f2bb76-33d2-4df7-b58f-7f2fb68b3f03")
AUTONOMOUS_CODER_TEMPLATE_ID = UUID("a1b2c3d4-5e6f-7a8b-9c0d-e1f2a3b4c5d6")
DOCUMENT_AUTHOR_TEMPLATE_ID = UUID("d6c5b4a3-f2e1-0d9c-8b7a-6f5e4d3c2b1a")


BUILTIN_AGENT_JOB_TEMPLATES: List[BuiltinAgentJobTemplate] = [
    BuiltinAgentJobTemplate(
        id=AI_HUB_SCIENTIST_TEMPLATE_ID,
        name="ai_hub_scientist_propose_bundle",
        display_name="AI Scientist: Propose AI Hub Bundle",
        description=(
            "Proposes a customer-specific AI Hub configuration (enabled dataset presets + eval templates) "
            "and a 3-workflow happy-path demo plan."
        ),
        category="ai_hub",
        job_type="analysis",
        default_goal=(
            "Propose a minimal, measurable AI Hub setup for the current customer. "
            "Output an 'ai_hub_bundle' with enabled preset IDs, enabled eval template IDs, "
            "and recommended happy-path demos."
        ),
        default_config={
            "deterministic_runner": "ai_hub_scientist",
            "workflows": ["triage", "extraction", "literature"],
            "apply": False,
            "llm_tier": "balanced",
            "llm_fallback_tiers": ["fast"],
            "llm_timeout_seconds": 120,
        },
        agent_definition_id=None,
        default_max_iterations=1,
        default_max_tool_calls=0,
        default_max_llm_calls=0,
        default_max_runtime_minutes=2,
    )
    ,
    BuiltinAgentJobTemplate(
        id=CUSTOMER_RESEARCH_SCOUT_TEMPLATE_ID,
        name="customer_research_scout",
        display_name="Customer Research: Scout + Brief (Optional)",
        description=(
            "Runs an autonomous customer-specific research loop using the deployment customer profile "
            "(Admin → AI Hub) and optional per-job customer context. Produces a structured plan + key findings, "
            "and can optionally persist a short brief document and/or update a reading list."
        ),
        category="research",
        job_type="research",
        default_goal=(
            "Using the customer profile + the user's knowledge base, perform customer-specific research on the goal. "
            "Prioritize internal documents first; use external sources only if needed. "
            "Return: (1) recommended queries, (2) shortlist of relevant internal documents, "
            "(3) key findings + open questions, (4) next-step experiments, and "
            "(5) optionally create a brief document and/or reading list updates."
        ),
        default_config={
            # Optional customer-specific hint in addition to deployment profile.
            "customer_context": "",
            # If true, allow the agent to create a persisted document (Agent Notes source).
            "persist_artifacts": False,
            # If set, the agent can add documents (and ingested papers) into this reading list.
            "reading_list_name": "Customer Research",
            # Guardrails for research behavior
            "prefer_sources": ["documents", "arxiv"],
            "max_documents": 12,
            "max_papers": 8,
            "llm_tier": "balanced",
            "llm_fallback_tiers": ["fast"],
            "llm_timeout_seconds": 120,
        },
        agent_definition_id=None,
        default_max_iterations=8,
        default_max_tool_calls=60,
        default_max_llm_calls=20,
        default_max_runtime_minutes=10,
    ),
    BuiltinAgentJobTemplate(
        id=CUSTOMER_RESEARCH_SCOUT_DEEP_DIVE_TEMPLATE_ID,
        name="customer_research_scout_deep_dive",
        display_name="Customer Research: Scout → Deep Dive",
        description=(
            "Runs a customer-specific research scout, then automatically triggers a deep-dive follow-up job "
            "that synthesizes an experiment plan and persists a brief document (optional)."
        ),
        category="research",
        job_type="research",
        default_goal=(
            "Run a customer-specific research scout for the goal. Prefer internal documents; use external sources "
            "only if needed. Then deep-dive on the most relevant sources and produce a concrete experiment plan."
        ),
        default_config={
            "customer_context": "",
            "persist_artifacts": False,
            "reading_list_name": "Customer Research",
            "prefer_sources": ["documents", "arxiv"],
            "max_documents": 12,
            "max_papers": 8,
            "llm_tier": "balanced",
            "llm_fallback_tiers": ["fast"],
            "llm_timeout_seconds": 120,
        },
        default_chain_config={
            "trigger_condition": "on_complete",
            "inherit_results": True,
            "inherit_config": True,
            "child_jobs": [
                {
                    "name": "Customer Research — Deep Dive",
                    "job_type": "research",
                    "goal": (
                        "Deep-dive using inherited results from the scout job. Focus on the top internal documents "
                        "and any high-signal papers. Output: (1) 3-5 hypotheses, (2) risks/unknowns, "
                        "(3) a minimal experiment plan (metrics + timeline), and (4) a short brief document if allowed."
                    ),
                    "config": {
                        "prefer_sources": ["documents"],
                        "max_documents": 6,
                        "max_papers": 0,
                        "llm_tier": "deep",
                        "llm_fallback_tiers": ["balanced"],
                        "llm_timeout_seconds": 180,
                    },
                }
            ],
        },
        agent_definition_id=None,
        default_max_iterations=8,
        default_max_tool_calls=60,
        default_max_llm_calls=20,
        default_max_runtime_minutes=12,
    ),
    BuiltinAgentJobTemplate(
        id=RESEARCH_INBOX_MONITOR_TEMPLATE_ID,
        name="research_inbox_monitor",
        display_name="Research Inbox: Continuous Monitor",
        description=(
            "Continuously monitors your internal knowledge base (and optionally arXiv) for customer-relevant updates, "
            "and files new items into a Research Inbox for triage (accept/reject)."
        ),
        category="research",
        job_type="monitor",
        default_goal=(
            "Monitor for new, customer-relevant internal documents and papers, and file discoveries into the Research Inbox "
            "so the user can triage and convert into follow-up research."
        ),
        default_config={
            "deterministic_runner": "research_inbox_monitor",
            "customer_context": "",
            "customer": "",
            "prefer_sources": ["documents", "arxiv"],
            "monitor_queries": [],
            "max_documents": 8,
            "max_papers": 8,
            "use_feedback_bias": True,
            # If true, add newly discovered internal docs to a reading list.
            "auto_add_to_reading_list": False,
            "reading_list_name": "Research Inbox",
            # If true, keep a rolling weekly brief document in Agent Notes.
            "persist_artifacts": False,
            # For schedule_type == "continuous": interval in minutes.
            "interval_minutes": 60,
            "automation_profile": "balanced",
            "automation_policy": {
                "follow_up_review_mode": "manual_only",
                "allowed_recommendations": [
                    "deep_dive_chain",
                    "single_research_job",
                ],
            },
            "autonomy_budget": {
                "auto_launch_limit_24h": 3,
                "approval_queue_limit_24h": 6,
                "alert_limit_24h": 4,
                "queue_backlog_cap": 8,
            },
        },
        agent_definition_id=None,
        default_max_iterations=1,
        default_max_tool_calls=0,
        default_max_llm_calls=0,
        default_max_runtime_minutes=5,
    ),
    BuiltinAgentJobTemplate(
        id=DOMAIN_RESEARCH_TEMPLATE_ID,
        name="domain_research_orchestrator",
        display_name="Domain Research: Ideas + Notes",
        description=(
            "Research a scientific or technical domain, rank evidence-backed ideas, generate a brief and a report, "
            "persist outputs as Research Notes, and optionally auto-launch a deeper follow-up."
        ),
        category="research",
        job_type="research",
        default_goal=(
            "Research the target domain using Knowledge DB sources first, then arXiv as needed. "
            "Produce ranked ideas with supporting evidence, a short brief, a longer report, and recommended next experiments."
        ),
        default_config={
            "deterministic_runner": "domain_research_orchestrator",
            "domain_research_mode": True,
            "domain": "",
            "objective": "",
            "customer_context": "",
            "source_scope": "kb_plus_arxiv",
            "track_type": "generic",
            "monitor_queries": [],
            "repo_source_ids": [],
            "benchmark_queries": [],
            "report_format": "brief_and_report",
            "persist_artifacts": True,
            "persist_target": "research_notes",
            "automation_profile": "balanced",
            "automation_policy": {
                "confidence_threshold": 0.7,
                "experiment_readiness_threshold": 0.8,
                "max_auto_follow_up_launches": 2,
                "auto_create_experiment_plans": True,
                "auto_launch_follow_up": True,
                "auto_execute_validation_runs": False,
                "max_concurrent_validation_runs": 1,
                "max_validation_runtime_minutes": 20,
                "max_validation_budget_per_run": 25.0,
                "follow_up_review_mode": "auto_launch_safe",
                "validation_backoff_policy": {
                    "max_consecutive_failures": 2,
                    "cooldown_minutes": 180,
                },
                "auto_launch_experiment_runs": False,
            },
            "auto_launch_follow_up": True,
            "follow_up_type": "deep_dive_chain",
            "prefer_sources": ["documents", "arxiv"],
            "max_documents": 10,
            "max_papers": 8,
            "confidence_threshold": 0.7,
            "llm_tier": "deep",
            "llm_fallback_tiers": ["balanced"],
            "llm_timeout_seconds": 180,
        },
        agent_definition_id=None,
        default_max_iterations=1,
        default_max_tool_calls=0,
        default_max_llm_calls=2,
        default_max_runtime_minutes=15,
    ),
    BuiltinAgentJobTemplate(
        id=CODE_PATCH_PROPOSER_TEMPLATE_ID,
        name="code_patch_proposer",
        display_name="Code Agent: Patch Proposal (MVP)",
        description=(
            "Generates a reviewable unified diff (patch) against a target git code source in your Knowledge Base. "
            "Use this after accepting Research Inbox items to turn findings into concrete code changes."
        ),
        category="code",
        job_type="analysis",
        default_goal=(
            "Implement the requested change as a minimal patch. Output a unified diff, risks, and tests to run."
        ),
        default_config={
            "deterministic_runner": "code_patch_proposer",
            "source_id": "",
            "search_query": "",
            "file_paths": [],
            "max_files": 6,
            "max_chars_per_file": 8000,
            "llm_tier": "deep",
            "llm_fallback_tiers": ["balanced"],
            "llm_timeout_seconds": 180,
        },
        agent_definition_id=None,
        default_max_iterations=1,
        default_max_tool_calls=0,
        default_max_llm_calls=1,
        default_max_runtime_minutes=10,
    ),
    BuiltinAgentJobTemplate(
        id=LATEX_CITATION_SYNC_TEMPLATE_ID,
        name="latex_citation_sync",
        display_name="LaTeX Agent: Citation Sync (KDB → Bib)",
        description=(
            "Scans a LaTeX Studio project for \\cite{KDB:<uuid>} keys (or legacy \\cite{KDB........}) and synchronizes the bibliography "
            "by updating refs.bib (BibTeX mode) or inserting/replacing a thebibliography block."
        ),
        category="latex",
        job_type="analysis",
        default_goal="Synchronize citations for a LaTeX Studio project (update refs.bib or thebibliography).",
        default_config={
            "deterministic_runner": "latex_citation_sync",
            # Required:
            "latex_project_id": "",
            # Optional:
            "mode": "bibtex",  # bibtex|thebibliography
            "bib_filename": "refs.bib",
        },
        agent_definition_id=None,
        default_max_iterations=1,
        default_max_tool_calls=0,
        default_max_llm_calls=0,
        default_max_runtime_minutes=3,
    ),
    BuiltinAgentJobTemplate(
        id=LATEX_REVIEWER_CRITIC_TEMPLATE_ID,
        name="latex_reviewer_critic",
        display_name="LaTeX Agent: Reviewer/Critic (Diff Suggestions)",
        description=(
            "Reviews a LaTeX Studio project for missing citations, unclear claims, and inconsistent notation, "
            "and suggests improvements as a unified diff against paper.tex."
        ),
        category="latex",
        job_type="analysis",
        default_goal="Review this LaTeX paper for citation gaps, clarity, and notation consistency; propose minimal fixes as a unified diff.",
        default_config={
            "deterministic_runner": "latex_reviewer_critic",
            # Required:
            "latex_project_id": "",
            # Optional:
            "focus": "",
            "llm_tier": "deep",
            "llm_fallback_tiers": ["balanced"],
            "llm_timeout_seconds": 180,
        },
        agent_definition_id=None,
        default_max_iterations=1,
        default_max_tool_calls=0,
        default_max_llm_calls=2,
        default_max_runtime_minutes=6,
    ),
    BuiltinAgentJobTemplate(
        id=EXPERIMENT_RUNNER_TEMPLATE_ID,
        name="experiment_runner",
        display_name="Code Agent: Experiment Runner (Unsafe)",
        description=(
            "Runs a small list of commands (e.g. tests) against a git DocumentSource in a sandboxed execution backend "
            "(explicitly gated by server unsafe-code-execution settings). Optionally appends a Results section to a LaTeX project."
        ),
        category="code",
        job_type="analysis",
        default_goal="Run the experiment commands/tests and record results.",
        default_config={
            "deterministic_runner": "experiment_runner",
            # Required:
            "source_id": "",
            # Optional:
            "commands": [],
            "latex_project_id": "",
            "timeout_seconds": 30,
        },
        agent_definition_id=None,
        default_max_iterations=1,
        default_max_tool_calls=0,
        default_max_llm_calls=0,
        default_max_runtime_minutes=10,
    ),
    BuiltinAgentJobTemplate(
        id=CLAUDE_CODE_BACKEND_TEMPLATE_ID,
        name="claude_code_backend",
        display_name="Code Agent: Claude-Code Backend Loop",
        description=(
            "Runs a Claude-code-style backend loop: propose patch, run verification commands, "
            "refine patch, and optionally apply to KB code documents."
        ),
        category="code",
        job_type="analysis",
        default_goal=(
            "Implement the backend change as a minimal patch, verify with focused checks/tests, "
            "and provide a safer final patch candidate."
        ),
        default_config={
            "deterministic_runner": "code_patch_proposer",
            "source_id": "",
            "search_query": "backend",
            "file_paths": [],
            "commands": [],
            "auto_commands_from_project_profile": True,
            "auto_commands_profile_max_files": 300,
            "max_files": 10,
            "max_chars_per_file": 10000,
            "apply_patch_to_kb": False,
            "apply_patch_to_kb_confirm": False,
            "proposal_strategy": "best_passing",
            "require_experiments_ok": True,
            "require_dry_run_first": True,
            "fail_on_block": False,
            "llm_tier": "deep",
            "llm_fallback_tiers": ["balanced"],
            "llm_timeout_seconds": 180,
        },
        default_chain_config={
            "trigger_condition": "on_complete",
            "inherit_results": True,
            "inherit_config": True,
            "child_jobs": [
                {
                    "name": "Claude-Code Backend — Verify (Round 1)",
                    "job_type": "analysis",
                    "goal": "Run backend verification commands/tests for the proposed patch.",
                    "config": {"deterministic_runner": "experiment_runner"},
                    "chain_config": {
                        "trigger_condition": "on_any_end",
                        "inherit_results": True,
                        "inherit_config": True,
                        "child_jobs": [
                            {
                                "name": "Claude-Code Backend — Refine (Round 2)",
                                "job_type": "analysis",
                                "goal": "Refine the backend patch using experiment output.",
                                "config": {"deterministic_runner": "code_patch_proposer"},
                                "chain_config": {
                                    "trigger_condition": "on_complete",
                                    "inherit_results": True,
                                    "inherit_config": True,
                                    "child_jobs": [
                                        {
                                            "name": "Claude-Code Backend — Verify (Round 2)",
                                            "job_type": "analysis",
                                            "goal": "Re-run backend verification commands/tests for the refined patch.",
                                            "config": {"deterministic_runner": "experiment_runner"},
                                        }
                                    ],
                                },
                            }
                        ],
                    },
                }
            ],
        },
        agent_definition_id=None,
        default_max_iterations=1,
        default_max_tool_calls=0,
        default_max_llm_calls=2,
        default_max_runtime_minutes=20,
    ),
    BuiltinAgentJobTemplate(
        id=REPO_BUG_TRIAGE_REPAIR_TEMPLATE_ID,
        name="repo_bug_triage_repair",
        display_name="Code Agent: Repo Bug Triage + Repair",
        description=(
            "Runs a repo-wide bug triage loop from a git-backed KB source: inspect symptom context, "
            "propose a minimal patch, run bounded verification commands, refine from failures, "
            "and stop at a reviewable patch proposal by default."
        ),
        category="code",
        job_type="analysis",
        default_goal=(
            "Triage the reported repo bug, identify the minimal fix, verify with focused checks/tests, "
            "and provide a safer final patch proposal with verification notes."
        ),
        default_config={
            "deterministic_runner": "code_patch_proposer",
            "source_id": "",
            "failure_symptom": "",
            "error_output": "",
            "scope": "auto",
            "search_query": "",
            "file_paths": [],
            "commands": [],
            "auto_commands_from_project_profile": True,
            "auto_commands_profile_max_files": 400,
            "create_workspace_from_source": True,
            "emit_execution_plan": True,
            "max_verification_commands": 3,
            "max_files": 12,
            "max_chars_per_file": 10000,
            "apply_patch_to_kb": False,
            "apply_patch_to_kb_confirm": False,
            "proposal_strategy": "best_passing",
            "require_experiments_ok": True,
            "require_dry_run_first": False,
            "fail_on_block": False,
            "llm_tier": "deep",
            "llm_fallback_tiers": ["balanced"],
            "llm_timeout_seconds": 180,
        },
        default_chain_config={
            "trigger_condition": "on_complete",
            "inherit_results": True,
            "inherit_config": True,
            "child_jobs": [
                {
                    "name": "Repo Bug Triage — Verify (Round 1)",
                    "job_type": "analysis",
                    "goal": "Run focused verification commands/tests for the proposed bug fix.",
                    "config": {"deterministic_runner": "experiment_runner"},
                    "chain_config": {
                        "trigger_condition": "on_any_end",
                        "inherit_results": True,
                        "inherit_config": True,
                        "child_jobs": [
                            {
                                "name": "Repo Bug Triage — Refine (Round 2)",
                                "job_type": "analysis",
                                "goal": "Refine the bug-fix patch using verification output.",
                                "config": {"deterministic_runner": "code_patch_proposer"},
                                "chain_config": {
                                    "trigger_condition": "on_complete",
                                    "inherit_results": True,
                                    "inherit_config": True,
                                    "child_jobs": [
                                        {
                                            "name": "Repo Bug Triage — Verify (Round 2)",
                                            "job_type": "analysis",
                                            "goal": "Re-run verification commands/tests for the refined patch.",
                                            "config": {"deterministic_runner": "experiment_runner"},
                                        }
                                    ],
                                },
                            }
                        ],
                    },
                }
            ],
        },
        agent_definition_id=None,
        default_max_iterations=1,
        default_max_tool_calls=0,
        default_max_llm_calls=2,
        default_max_runtime_minutes=25,
    ),
    BuiltinAgentJobTemplate(
        id=LATEX_COMPILE_PROJECT_TEMPLATE_ID,
        name="latex_compile_project",
        display_name="LaTeX Agent: Compile Project (PDF)",
        description=(
            "Compiles a LaTeX Studio project to PDF. Uses the dedicated LaTeX Celery worker when enabled; "
            "otherwise attempts an in-process compile (requires TeX tools in the API container)."
        ),
        category="latex",
        job_type="analysis",
        default_goal="Compile this LaTeX Studio project to PDF.",
        default_config={
            "deterministic_runner": "latex_compile_project",
            # Required:
            "latex_project_id": "",
            # Optional:
            "safe_mode": True,
            "preferred_engine": None,
            "use_worker": True,
            "wait_seconds": 120,
            "skip_if_unavailable": True,
        },
        agent_definition_id=None,
        default_max_iterations=1,
        default_max_tool_calls=0,
        default_max_llm_calls=0,
        default_max_runtime_minutes=5,
    ),
    BuiltinAgentJobTemplate(
        id=LATEX_PUBLISH_PROJECT_TEMPLATE_ID,
        name="latex_publish_project",
        display_name="LaTeX Agent: Publish Project (KB)",
        description="Publishes paper.tex and/or paper.pdf from a LaTeX Studio project into the Knowledge DB.",
        category="latex",
        job_type="analysis",
        default_goal="Publish this LaTeX Studio project into the Knowledge DB.",
        default_config={
            "deterministic_runner": "latex_publish_project",
            # Required:
            "latex_project_id": "",
            # Optional:
            "include_tex": True,
            "include_pdf": True,
            "publish_tags": "latex,paper",
        },
        agent_definition_id=None,
        default_max_iterations=1,
        default_max_tool_calls=0,
        default_max_llm_calls=0,
        default_max_runtime_minutes=6,
    ),
    BuiltinAgentJobTemplate(
        id=AUTONOMOUS_CODER_TEMPLATE_ID,
        name="autonomous_coder",
        display_name="Autonomous Coder",
        description=(
            "Clones a repository or loads code from a KB source, then iteratively "
            "browses, edits, tests, and patches code to achieve a coding goal. "
            "Supports running commands, applying patches, and tracking file changes."
        ),
        category="coding",
        job_type="analysis",
        default_goal="Clone the repository, understand the codebase, implement the requested changes, and verify with tests.",
        default_config={
            "agent_role": "coder",
            "auto_clone": True,
            "auto_run_tests": True,
            "max_test_retries": 3,
            "llm_tier": "deep",
            "llm_fallback_tiers": ["balanced"],
            "llm_timeout_seconds": 180,
        },
        agent_definition_id=None,
        default_max_iterations=20,
        default_max_tool_calls=100,
        default_max_llm_calls=40,
        default_max_runtime_minutes=30,
    ),
    BuiltinAgentJobTemplate(
        id=DOCUMENT_AUTHOR_TEMPLATE_ID,
        name="document_author",
        display_name="Document Author",
        description=(
            "Plans, writes, revises, and exports structured documents with citations. "
            "Uses RAG search to find relevant sources, supports multi-section documents "
            "with figures, and exports to DOCX, PDF, PPTX, or LaTeX formats."
        ),
        category="authoring",
        job_type="synthesis",
        default_goal="Plan, write, and export a well-structured document based on the given topic and requirements.",
        default_config={
            "agent_role": "author",
            "output_format": "markdown",
            "export_formats": ["docx"],
            "persist_to_kb": True,
            "llm_tier": "deep",
            "llm_fallback_tiers": ["balanced"],
            "llm_timeout_seconds": 180,
        },
        agent_definition_id=None,
        default_max_iterations=15,
        default_max_tool_calls=80,
        default_max_llm_calls=30,
        default_max_runtime_minutes=20,
    ),
]


def list_builtin_agent_job_templates(category: Optional[str] = None) -> List[BuiltinAgentJobTemplate]:
    templates = [t for t in BUILTIN_AGENT_JOB_TEMPLATES if t.is_active]
    if category:
        templates = [t for t in templates if t.category == category]
    return templates


def get_builtin_agent_job_template(template_id: UUID) -> Optional[BuiltinAgentJobTemplate]:
    for t in BUILTIN_AGENT_JOB_TEMPLATES:
        if t.id == template_id and t.is_active:
            return t
    return None
