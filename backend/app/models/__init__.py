"""
Database models for the Knowledge Database application.
"""

from .document import Document, DocumentChunk, DocumentSource
from .chat import ChatSession, ChatMessage
from .user import User
from .knowledge_graph import Entity, EntityMention, Relationship
from .upload_session import UploadSession
from .persona import Persona, DocumentPersonaDetection, PersonaEditRequest
from .template import TemplateJob
from .memory import ConversationMemory, MemoryInteraction, UserPreferences, AgentConversation, AgentToolExecution
from .agent_definition import AgentDefinition, AgentConversationContext, AgentMemoryInjection
from .workflow import UserTool, Workflow, WorkflowNode, WorkflowEdge, WorkflowExecution, WorkflowNodeExecution
from .presentation import PresentationJob, PresentationTemplate
from .notification import Notification, NotificationPreferences, NotificationType
from .reading_list import ReadingList, ReadingListItem
from .secret import UserSecret
from .tool_audit import ToolExecutionAudit
from .tool_policy import ToolPolicy
from .agent_tool_prior import AgentToolPrior
from .saved_search import SavedSearch, SearchShare
from .llm_usage import LLMUsageEvent
from .export_job import ExportJob
from .repo_report import RepoReportJob
from .artifact_draft import ArtifactDraft
from .retrieval_trace import RetrievalTrace
from .agent_job import (
    AgentJob,
    AgentJobCheckpoint,
    AgentJobTemplate,
    AgentJobChainDefinition,
    AgentJobStatus,
    AgentJobType,
    ChainTriggerCondition,
)
from .synthesis_job import SynthesisJob, SynthesisJobType, SynthesisJobStatus
from .research_note import ResearchNote
from .experiment import ExperimentPlan, ExperimentRun
from .research_inbox import ResearchInboxItem
from .research_monitor_profile import ResearchMonitorProfile
from .code_patch_proposal import CodePatchProposal
from .patch_pr import PatchPR
from .training_dataset import (
    TrainingDataset,
    DatasetSample,
    DatasetType,
    DatasetFormat,
    DatasetStatus,
)
from .training_job import (
    TrainingJob,
    TrainingCheckpoint,
    TrainingMethod,
    TrainingBackend,
    TrainingJobStatus,
)
from .model_registry import (
    ModelAdapter,
    AdapterType,
    AdapterStatus,
)
from .ai_hub_recommendation_feedback import AIHubRecommendationFeedback
from .latex_project import LatexProject
from .latex_project_file import LatexProjectFile

__all__ = [
    "Document",
    "DocumentChunk",
    "DocumentSource",
    "ChatSession",
    "ChatMessage",
    "User",
    "UploadSession",
    "Entity",
    "EntityMention",
    "Relationship",
    "Persona",
    "DocumentPersonaDetection",
    "PersonaEditRequest",
    "TemplateJob",
    "ConversationMemory",
    "MemoryInteraction",
    "UserPreferences",
    "AgentConversation",
    "AgentToolExecution",
    # Agent definition models
    "AgentDefinition",
    "AgentConversationContext",
    "AgentMemoryInjection",
    # Workflow models
    "UserTool",
    "Workflow",
    "WorkflowNode",
    "WorkflowEdge",
    "WorkflowExecution",
    "WorkflowNodeExecution",
    # Presentation models
    "PresentationJob",
    "PresentationTemplate",
    # Notification models
    "Notification",
    "NotificationPreferences",
    "NotificationType",
    "ReadingList",
    "ReadingListItem",
    "UserSecret",
    "ToolExecutionAudit",
    "AgentToolPrior",
    "SavedSearch",
    "SearchShare",
    "LLMUsageEvent",
    # Export models
    "ExportJob",
    # Repository report models
    "RepoReportJob",
    "ArtifactDraft",
    "RetrievalTrace",
    # Autonomous agent job models
    "AgentJob",
    "AgentJobCheckpoint",
    "AgentJobTemplate",
    "AgentJobChainDefinition",
    "AgentJobStatus",
    "AgentJobType",
    "ChainTriggerCondition",
    # Synthesis job models
    "SynthesisJob",
    "SynthesisJobType",
    "SynthesisJobStatus",
    # Research notes
    "ResearchNote",
    "ExperimentPlan",
    "ExperimentRun",
    "ResearchInboxItem",
    "ResearchMonitorProfile",
    "CodePatchProposal",
    "PatchPR",
    # Training dataset models
    "TrainingDataset",
    "DatasetSample",
    "DatasetType",
    "DatasetFormat",
    "DatasetStatus",
    # Training job models
    "TrainingJob",
    "TrainingCheckpoint",
    "TrainingMethod",
    "TrainingBackend",
    "TrainingJobStatus",
    # Model registry models
    "ModelAdapter",
    "AdapterType",
    "AdapterStatus",
    "AIHubRecommendationFeedback",
    "LatexProject",
    "LatexProjectFile",
]
