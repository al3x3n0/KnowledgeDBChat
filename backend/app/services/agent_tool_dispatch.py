"""App-layer tool provider registry for agent services.

This was one 12,000-line module holding every provider. Each provider now
lives in its own module under ``app/services/agent_tool_providers/`` (the registry and
provider types in ``base``, helpers shared by several providers in
``common``), and this module re-exports every name, so existing imports keep
working. New code should import from the module that owns the name; a test
patching a module global must patch it where the provider looks it up.
"""

from __future__ import annotations

from app.services.agent_tool_providers.base import (  # noqa: F401
    AgentToolExecutionContext,
    AgentToolProvider,
    AgentToolRegistry,
    FunctionToolProvider,
    _unimplemented_tool,
)
from app.services.agent_tool_providers.chat import (  # noqa: F401
    build_agent_service_analytics_content_provider,
    build_agent_service_chat_core_provider,
    build_agent_service_document_provider,
    build_agent_service_knowledge_graph_provider,
    build_agent_service_research_provider,
    build_agent_service_workflow_provider,
)
from app.services.agent_tool_providers.collaboration import (  # noqa: F401
    build_autonomous_collaboration_provider,
)
from app.services.agent_tool_providers.common import (  # noqa: F401
    _load_documents_for_analysis,
    _tool_snapshot_context,
)
from app.services.agent_tool_providers.data_analysis import (  # noqa: F401
    build_autonomous_data_analysis_provider,
)
from app.services.agent_tool_providers.document import (  # noqa: F401
    MERGE_MAX_DOCUMENTS,
    build_autonomous_document_provider,
)
from app.services.agent_tool_providers.document_authoring import (  # noqa: F401
    build_autonomous_document_authoring_provider,
)
from app.services.agent_tool_providers.kg import (  # noqa: F401
    build_autonomous_kg_provider,
)
from app.services.agent_tool_providers.media import (  # noqa: F401
    build_autonomous_media_provider,
)
from app.services.agent_tool_providers.memory import (  # noqa: F401
    build_autonomous_memory_provider,
)
from app.services.agent_tool_providers.notification_visualization import (  # noqa: F401
    MAX_NOTIFICATIONS_PER_RUN,
    build_autonomous_notification_visualization_provider,
)
from app.services.agent_tool_providers.observability import (  # noqa: F401
    build_autonomous_observability_provider,
)
from app.services.agent_tool_providers.output_state import (  # noqa: F401
    build_autonomous_output_state_provider,
)
from app.services.agent_tool_providers.project_bootstrap import (  # noqa: F401
    build_autonomous_project_bootstrap_provider,
)
from app.services.agent_tool_providers.reasoning import (  # noqa: F401
    build_autonomous_reasoning_provider,
)
from app.services.agent_tool_providers.research import (  # noqa: F401
    _CLUSTER_ANALYSIS_SCHEMA,
    _INGEST_POLL_SECONDS,
    _METHODOLOGY_COMPARISON_SCHEMA,
    _RESEARCH_GAPS_SCHEMA,
    INGEST_WAIT_SECONDS,
    _document_excerpts,
    _wait_for_ingested_documents,
    build_autonomous_research_provider,
)
from app.services.agent_tool_providers.sandbox_skill import (  # noqa: F401
    build_autonomous_sandbox_skill_provider,
)
from app.services.agent_tool_providers.scheduling import (  # noqa: F401
    MAX_SCHEDULED_JOBS_PER_RUN,
    build_autonomous_scheduling_provider,
)
from app.services.agent_tool_providers.snapshot import (  # noqa: F401
    build_autonomous_snapshot_provider,
)
from app.services.agent_tool_providers.symbol_retrieval import (  # noqa: F401
    build_autonomous_symbol_retrieval_provider,
)
from app.services.agent_tool_providers.web_research import (  # noqa: F401
    build_autonomous_web_research_provider,
)
from app.services.agent_tool_providers.workflow import (  # noqa: F401
    build_autonomous_workflow_provider,
)
from app.services.agent_tool_providers.workspace_mutation import (  # noqa: F401
    _as_float,
    _diff_files,
    _new_file_diffs,
    _store_patch_proposal,
    build_autonomous_workspace_mutation_provider,
    hot_blocks_from_findings,
)
from app.services.agent_tool_providers.workspace_read import (  # noqa: F401
    build_autonomous_workspace_read_provider,
)

__all__ = [
    "AgentToolExecutionContext",
    "AgentToolProvider",
    "AgentToolRegistry",
    "FunctionToolProvider",
    "INGEST_WAIT_SECONDS",
    "MAX_NOTIFICATIONS_PER_RUN",
    "MAX_SCHEDULED_JOBS_PER_RUN",
    "MERGE_MAX_DOCUMENTS",
    "_CLUSTER_ANALYSIS_SCHEMA",
    "_INGEST_POLL_SECONDS",
    "_METHODOLOGY_COMPARISON_SCHEMA",
    "_RESEARCH_GAPS_SCHEMA",
    "_as_float",
    "_diff_files",
    "_document_excerpts",
    "_load_documents_for_analysis",
    "_new_file_diffs",
    "_store_patch_proposal",
    "_tool_snapshot_context",
    "_unimplemented_tool",
    "_wait_for_ingested_documents",
    "build_agent_service_analytics_content_provider",
    "build_agent_service_chat_core_provider",
    "build_agent_service_document_provider",
    "build_agent_service_knowledge_graph_provider",
    "build_agent_service_research_provider",
    "build_agent_service_workflow_provider",
    "build_autonomous_collaboration_provider",
    "build_autonomous_data_analysis_provider",
    "build_autonomous_document_authoring_provider",
    "build_autonomous_document_provider",
    "build_autonomous_kg_provider",
    "build_autonomous_media_provider",
    "build_autonomous_memory_provider",
    "build_autonomous_notification_visualization_provider",
    "build_autonomous_observability_provider",
    "build_autonomous_output_state_provider",
    "build_autonomous_project_bootstrap_provider",
    "build_autonomous_reasoning_provider",
    "build_autonomous_research_provider",
    "build_autonomous_sandbox_skill_provider",
    "build_autonomous_scheduling_provider",
    "build_autonomous_snapshot_provider",
    "build_autonomous_symbol_retrieval_provider",
    "build_autonomous_web_research_provider",
    "build_autonomous_workflow_provider",
    "build_autonomous_workspace_mutation_provider",
    "build_autonomous_workspace_read_provider",
    "hot_blocks_from_findings",
]
