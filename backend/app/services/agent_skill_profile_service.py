"""Skill-profile policy helpers extracted from AutonomousAgentExecutor."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional


class AgentSkillProfileService:
    """Resolve active role profiles for autonomous jobs."""

    def resolve_agent_skill_profile(
        self,
        executor: Any,
        job: Any,
        *,
        state: Optional[Dict[str, Any]] = None,
        override_role: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Resolve the active role profile controlling prompt/tool behavior."""
        cfg = job.config if isinstance(job.config, dict) else {}

        if override_role and override_role in {"researcher", "critic", "synthesizer", "verifier", "coder", "author"}:
            role = override_role
        else:
            prior_role = executor._normalize_role_token(
                (state or {}).get("skill_profile", {}).get("role") if isinstance(state, dict) else ""
            )
            role_candidates = [
                executor._normalize_role_token(cfg.get("agent_role")),
                executor._normalize_role_token(cfg.get("swarm_role")),
                executor._normalize_role_token(cfg.get("role")),
                prior_role,
                executor._normalize_role_token(job.name),
                executor._normalize_role_token(job.goal),
            ]

            role = "researcher"
            for candidate in role_candidates:
                if candidate in {"researcher", "critic", "synthesizer", "verifier", "coder", "author"}:
                    role = candidate
                    break
                if "critic" in candidate:
                    role = "critic"
                    break
                if "synth" in candidate:
                    role = "synthesizer"
                    break
                if "verif" in candidate or "validat" in candidate:
                    role = "verifier"
                    break
                if "cod" in candidate or "develop" in candidate or "engineer" in candidate:
                    role = "coder"
                    break
                if "author" in candidate or "writ" in candidate or "document" in candidate:
                    role = "author"
                    break

        profiles: Dict[str, Dict[str, Any]] = {
            "researcher": {
                "role": "researcher",
                "display_name": "Researcher",
                "prompt_directives": [
                    "Prioritize discovery and evidence coverage before conclusions.",
                    "Favor retrieval and analysis tools to expand breadth.",
                    "Capture uncertainties explicitly for downstream roles.",
                ],
                "preferred_tools": [
                    "search_documents", "search_with_filters", "search_arxiv", "find_related_papers",
                    "get_document_details", "read_document_content", "summarize_document", "extract_paper_insights",
                    "build_research_graph", "identify_research_gaps",
                ],
                "discouraged_tools": [
                    "create_synthesis_document", "create_document_from_text", "generate_research_presentation",
                ],
                "blocked_tools": [],
                "metric_focus": ["evidence_actions", "evidence_findings"],
            },
            "critic": {
                "role": "critic",
                "display_name": "Critic",
                "prompt_directives": [
                    "Challenge assumptions and seek disconfirming evidence.",
                    "Validate claims with direct source checks before acceptance.",
                    "Call out risks, contradictions, and missing controls.",
                ],
                "preferred_tools": [
                    "compare_documents", "compare_methodologies", "identify_research_gaps", "build_research_graph",
                    "read_document_content", "get_document_details", "get_research_findings", "search_with_filters",
                ],
                "discouraged_tools": [
                    "create_synthesis_document", "create_document_from_text", "generate_research_presentation",
                ],
                "blocked_tools": [],
                "metric_focus": ["challenge_actions", "risk_findings"],
            },
            "synthesizer": {
                "role": "synthesizer",
                "display_name": "Synthesizer",
                "prompt_directives": [
                    "Convert evidence into concise, actionable outputs.",
                    "Merge overlapping findings and reduce redundancy.",
                    "Always provide clear next-step recommendations.",
                ],
                "preferred_tools": [
                    "create_synthesis_document", "create_document_from_text", "generate_research_presentation",
                    "write_progress_report", "link_entities", "save_research_finding",
                ],
                "discouraged_tools": [
                    "batch_ingest_papers", "monitor_arxiv_topic",
                ],
                "blocked_tools": [],
                "metric_focus": ["synthesis_actions", "artifacts_created"],
            },
            "verifier": {
                "role": "verifier",
                "display_name": "Verifier",
                "prompt_directives": [
                    "Verify outputs against goal criteria and source evidence.",
                    "Prefer reproducible checks over broad exploration.",
                    "Surface confidence level and unresolved validation gaps.",
                ],
                "preferred_tools": [
                    "read_document_content", "get_document_details", "compare_documents", "search_with_filters",
                    "get_research_findings", "compare_methodologies", "build_research_graph",
                ],
                "discouraged_tools": [
                    "batch_ingest_papers", "monitor_arxiv_topic",
                ],
                "blocked_tools": [],
                "metric_focus": ["verification_actions", "failed_checks"],
            },
            "coder": {
                "role": "coder",
                "display_name": "Autonomous Coder",
                "prompt_directives": [
                    "Clone the repository first, then explore before editing.",
                    "Run tests after each significant code change.",
                    "Keep changes minimal and focused on the goal.",
                    "Use search_code to understand existing patterns before writing new code.",
                ],
                "preferred_tools": [
                    "clone_and_index_repo", "browse_repo_files", "read_file", "write_file",
                    "apply_patch", "run_command", "search_code", "get_workspace_status",
                    "retrieve_repo_symbols", "get_symbol_context", "find_tests_for_symbol",
                    "project_bootstrap",
                ],
                "discouraged_tools": [
                    "search_arxiv", "create_synthesis_document", "generate_research_presentation",
                    "plan_document", "write_section", "export_document",
                ],
                "blocked_tools": [],
                "metric_focus": ["files_modified", "tests_passed", "commands_run"],
            },
            "author": {
                "role": "author",
                "display_name": "Document Author",
                "prompt_directives": [
                    "Plan the document structure before writing any content.",
                    "Search for and cite relevant sources when writing sections.",
                    "Revise sections based on quality criteria before assembling.",
                    "Always assemble the full document before exporting.",
                ],
                "preferred_tools": [
                    "plan_document", "write_section", "revise_section", "assemble_document",
                    "export_document", "insert_figure",
                    "search_documents", "read_document_content",
                ],
                "discouraged_tools": [
                    "run_command", "clone_and_index_repo", "execute_python",
                    "batch_ingest_papers", "monitor_arxiv_topic",
                ],
                "blocked_tools": [],
                "metric_focus": ["sections_written", "citations_added", "revisions_made"],
            },
        }
        profile = dict(profiles.get(role, profiles["researcher"]))
        profile["role"] = role
        profile["resolved_at"] = datetime.utcnow().isoformat()
        return profile
