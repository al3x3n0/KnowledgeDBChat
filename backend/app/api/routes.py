"""
Main API router configuration.
"""

from fastapi import APIRouter

from app.api.endpoints import (
    admin,
    admin_tool_policies,
    agent,
    agent_control_plane,
    agent_jobs,
    ai_hub_eval,
    analytics,
    api_keys,
    artifact_drafts,
    auth,
    autonomous_rnd_evals,
    chat,
    code_patches,
    coding_backlog,
    coding_swarm_profiles,
    content,
    dashboard,
    document_folders,
    documents,
    docx_editor,
    domain_research_profiles,
    experiments,
    export,
    external_agents,
    git,
    knowledge_graph,
    langgraph,
    latex,
    llm_snapshots,
    mcp_config,
    memory,
    model_registry,
    notifications,
    patch_prs,
    personas,
    presentations,
    reading_lists,
    repo_reports,
    research,
    research_inbox,
    research_monitor_profiles,
    research_notes,
    research_papers,
    research_portfolios,
    retrieval_traces,
    scientific_sandbox_profiles,
    searches,
    secrets,
    synthesis,
    system,
    templates,
    tool_audit,
    tool_policies,
    training_datasets,
    training_jobs,
    upload,
    usage,
    user_tools,
    users,
    workflows,
)

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(auth.router, prefix="/auth", tags=["authentication"])
api_router.include_router(users.router, prefix="/users", tags=["users"])
api_router.include_router(chat.router, prefix="/chat", tags=["chat"])
api_router.include_router(documents.router, prefix="/documents", tags=["documents"])
api_router.include_router(
    document_folders.router, prefix="/document-folders", tags=["document-folders"]
)
api_router.include_router(upload.router, prefix="/upload", tags=["upload"])
api_router.include_router(memory.router, prefix="/memory", tags=["memory"])
api_router.include_router(admin.router, prefix="/admin", tags=["administration"])
api_router.include_router(system.router, prefix="/system", tags=["system"])
api_router.include_router(
    knowledge_graph.router, prefix="/kg", tags=["knowledge-graph"]
)
api_router.include_router(git.router, prefix="/git", tags=["git"])
api_router.include_router(personas.router, prefix="/personas", tags=["personas"])
api_router.include_router(templates.router, prefix="/templates", tags=["templates"])
api_router.include_router(docx_editor.router, prefix="/documents", tags=["docx-editor"])
api_router.include_router(agent.router, prefix="/agent", tags=["agent"])
api_router.include_router(user_tools.router, prefix="/user-tools", tags=["user-tools"])
api_router.include_router(workflows.router, prefix="/workflows", tags=["workflows"])
api_router.include_router(
    presentations.router, prefix="/presentations", tags=["presentations"]
)
api_router.include_router(
    notifications.router, prefix="/notifications", tags=["notifications"]
)
api_router.include_router(research.router, prefix="/research", tags=["research"])
api_router.include_router(
    research_papers.router, prefix="/research/papers", tags=["research-papers"]
)
api_router.include_router(
    research_inbox.router, prefix="/research/inbox", tags=["research-inbox"]
)
api_router.include_router(
    research_monitor_profiles.router,
    prefix="/research/monitor-profiles",
    tags=["research-monitor-profiles"],
)
api_router.include_router(
    domain_research_profiles.router,
    prefix="/domain-research-profiles",
    tags=["domain-research-profiles"],
)
api_router.include_router(
    research_portfolios.router,
    prefix="/research-portfolios",
    tags=["research-portfolios"],
)
api_router.include_router(
    scientific_sandbox_profiles.router,
    prefix="/scientific-sandbox-profiles",
    tags=["scientific-sandbox-profiles"],
)
api_router.include_router(
    code_patches.router, prefix="/code-patches", tags=["code-patches"]
)
api_router.include_router(patch_prs.router, prefix="/patch-prs", tags=["patch-prs"])
api_router.include_router(
    reading_lists.router, prefix="/reading-lists", tags=["reading-lists"]
)
api_router.include_router(
    research_notes.router, prefix="/research-notes", tags=["research-notes"]
)
api_router.include_router(
    experiments.router, prefix="/experiments", tags=["experiments"]
)
api_router.include_router(
    external_agents.router, prefix="/external-agents", tags=["external-agents"]
)
api_router.include_router(secrets.router, prefix="/secrets", tags=["secrets"])
api_router.include_router(tool_audit.router, prefix="/audit", tags=["audit"])
api_router.include_router(tool_policies.router, prefix="/tools", tags=["tools"])
api_router.include_router(
    admin_tool_policies.router, prefix="/admin", tags=["admin-tools"]
)
api_router.include_router(searches.router, prefix="/searches", tags=["searches"])
api_router.include_router(usage.router, prefix="/usage", tags=["usage"])
api_router.include_router(analytics.router, prefix="/analytics", tags=["analytics"])
api_router.include_router(
    content.router, prefix="/content", tags=["content-generation"]
)
api_router.include_router(dashboard.router, prefix="/dashboard", tags=["dashboard"])
api_router.include_router(api_keys.router, prefix="/api-keys", tags=["api-keys"])
api_router.include_router(export.router, prefix="/export", tags=["export"])
api_router.include_router(
    repo_reports.router, prefix="/repo-reports", tags=["repo-reports"]
)
api_router.include_router(
    artifact_drafts.router, prefix="/artifact-drafts", tags=["artifact-drafts"]
)
api_router.include_router(
    retrieval_traces.router, prefix="/retrieval-traces", tags=["retrieval-traces"]
)
api_router.include_router(
    llm_snapshots.router, prefix="/llm-snapshots", tags=["llm-snapshots"]
)
api_router.include_router(mcp_config.router, prefix="/mcp-config", tags=["mcp-config"])
api_router.include_router(agent_jobs.router, prefix="/agent-jobs", tags=["agent-jobs"])
api_router.include_router(
    autonomous_rnd_evals.router,
    prefix="/autonomous-rnd-evals",
    tags=["autonomous-rnd-evals"],
)
api_router.include_router(
    agent_control_plane.router,
    prefix="/agent-control-plane",
    tags=["agent-control-plane"],
)
api_router.include_router(
    coding_backlog.router, prefix="/coding-backlog", tags=["coding-backlog"]
)
api_router.include_router(
    coding_swarm_profiles.router,
    prefix="/coding-swarm-profiles",
    tags=["coding-swarm-profiles"],
)
api_router.include_router(langgraph.router, prefix="/langgraph", tags=["langgraph"])
api_router.include_router(synthesis.router, prefix="/synthesis", tags=["synthesis"])
api_router.include_router(latex.router, prefix="/latex", tags=["latex"])

# AI Hub / Training endpoints
api_router.include_router(
    training_datasets.router, prefix="/training/datasets", tags=["training-datasets"]
)
api_router.include_router(
    training_jobs.router, prefix="/training/jobs", tags=["training-jobs"]
)
api_router.include_router(
    model_registry.router, prefix="/training/models", tags=["model-registry"]
)
api_router.include_router(
    ai_hub_eval.router, prefix="/training/evals", tags=["training-evals"]
)
