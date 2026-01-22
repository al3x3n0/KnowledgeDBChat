"""
Tool definitions for the agentic chat system.

Defines available tools that the agent can use to perform document operations.
"""

from typing import List, Dict, Any


# Tool definitions following function-calling conventions
AGENT_TOOLS: List[Dict[str, Any]] = [
    {
        "name": "search_documents",
        "description": "Search for documents in the knowledge base using semantic search. Use this to find relevant documents based on a query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to find relevant documents"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 5, max: 20)",
                    "default": 5
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "get_document_details",
        "description": "Get detailed information about a specific document including title, content preview, metadata, and processing status.",
        "parameters": {
            "type": "object",
            "properties": {
                "document_id": {
                    "type": "string",
                    "description": "The UUID of the document to retrieve"
                }
            },
            "required": ["document_id"]
        }
    },
    {
        "name": "summarize_document",
        "description": "Generate or retrieve a summary for a specific document. If a summary already exists, returns it unless force_regenerate is true.",
        "parameters": {
            "type": "object",
            "properties": {
                "document_id": {
                    "type": "string",
                    "description": "The UUID of the document to summarize"
                },
                "force_regenerate": {
                    "type": "boolean",
                    "description": "Force regeneration of summary even if one already exists",
                    "default": False
                }
            },
            "required": ["document_id"]
        }
    },
    {
        "name": "delete_document",
        "description": "Delete a document from the knowledge base. This action is irreversible and requires explicit confirmation.",
        "parameters": {
            "type": "object",
            "properties": {
                "document_id": {
                    "type": "string",
                    "description": "The UUID of the document to delete"
                },
                "confirm": {
                    "type": "boolean",
                    "description": "Must be set to true to confirm deletion. If false or missing, will only return document info for confirmation.",
                    "default": False
                }
            },
            "required": ["document_id"]
        }
    },
    {
        "name": "list_recent_documents",
        "description": "List the most recently added or updated documents in the knowledge base.",
        "parameters": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of documents to return (default: 10, max: 50)",
                    "default": 10
                }
            },
            "required": []
        }
    },
    {
        "name": "list_document_sources",
        "description": "List available document sources with type and status.",
        "parameters": {
            "type": "object",
            "properties": {
                "active_only": {
                    "type": "boolean",
                    "description": "Only include active sources (default: false)",
                    "default": False
                }
            },
            "required": []
        }
    },
    {
        "name": "list_documents_by_source",
        "description": "List documents from a specific source (by source ID, name, or type).",
        "parameters": {
            "type": "object",
            "properties": {
                "source_id": {
                    "type": "string",
                    "description": "UUID of the document source"
                },
                "source_name": {
                    "type": "string",
                    "description": "Name of the document source (case-insensitive, partial match)"
                },
                "source_type": {
                    "type": "string",
                    "description": "Source type (e.g., gitlab, confluence, web, file)"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of documents to return (default: 20, max: 50)",
                    "default": 20
                },
                "offset": {
                    "type": "integer",
                    "description": "Pagination offset (default: 0)",
                    "default": 0
                }
            },
            "required": []
        }
    },
    {
        "name": "request_file_upload",
        "description": "Request the user to upload a file. Use this when the user wants to add a new document to the knowledge base.",
        "parameters": {
            "type": "object",
            "properties": {
                "suggested_title": {
                    "type": "string",
                    "description": "Suggested title for the document (optional)"
                },
                "suggested_tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Suggested tags for categorization (optional)"
                }
            },
            "required": []
        }
    },
    {
        "name": "create_document_from_text",
        "description": "Create a new document directly from text content. Useful for saving notes, code snippets, or any text the user wants to store in the knowledge base.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Title for the new document"
                },
                "content": {
                    "type": "string",
                    "description": "The text content to save as a document"
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tags for categorization (optional)"
                }
            },
            "required": ["title", "content"]
        }
    },
    {
        "name": "find_similar_documents",
        "description": "Find documents that are semantically similar to a given document. Useful for discovering related content.",
        "parameters": {
            "type": "object",
            "properties": {
                "document_id": {
                    "type": "string",
                    "description": "The UUID of the reference document"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of similar documents to return (default: 5)",
                    "default": 5
                }
            },
            "required": ["document_id"]
        }
    },
    {
        "name": "search_documents_by_author",
        "description": "Find documents authored by a person. Uses case-insensitive matching.",
        "parameters": {
            "type": "object",
            "properties": {
                "author": {
                    "type": "string",
                    "description": "Author name or substring to search for"
                },
                "match_type": {
                    "type": "string",
                    "enum": ["contains", "exact", "starts_with"],
                    "description": "Match strategy for author names",
                    "default": "contains"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of documents to return (default: 20, max: 50)",
                    "default": 20
                }
            },
            "required": ["author"]
        }
    },
    {
        "name": "update_document_tags",
        "description": "Add, remove, or replace tags on a document. Use action 'add' to add tags, 'remove' to remove tags, or 'replace' to replace all tags.",
        "parameters": {
            "type": "object",
            "properties": {
                "document_id": {
                    "type": "string",
                    "description": "The UUID of the document to update"
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tags to add, remove, or set"
                },
                "action": {
                    "type": "string",
                    "enum": ["add", "remove", "replace"],
                    "description": "Action to perform: 'add' (default), 'remove', or 'replace'"
                }
            },
            "required": ["document_id", "tags"]
        }
    },
    {
        "name": "get_knowledge_base_stats",
        "description": "Get statistics about the knowledge base including document counts, storage usage, and processing status.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "batch_delete_documents",
        "description": "Delete multiple documents at once. Requires explicit confirmation. Use with caution as this action is irreversible.",
        "parameters": {
            "type": "object",
            "properties": {
                "document_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of document UUIDs to delete"
                },
                "confirm": {
                    "type": "boolean",
                    "description": "Must be set to true to confirm batch deletion",
                    "default": False
                }
            },
            "required": ["document_ids"]
        }
    },
    {
        "name": "batch_summarize_documents",
        "description": "Queue summarization for multiple documents at once. Useful for processing several documents that lack summaries.",
        "parameters": {
            "type": "object",
            "properties": {
                "document_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of document UUIDs to summarize"
                },
                "force_regenerate": {
                    "type": "boolean",
                    "description": "Force regeneration even if summaries exist",
                    "default": False
                }
            },
            "required": ["document_ids"]
        }
    },
    {
        "name": "search_by_tags",
        "description": "Find documents that have specific tags. Useful for filtering documents by category or topic.",
        "parameters": {
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tags to search for"
                },
                "match_all": {
                    "type": "boolean",
                    "description": "If true, documents must have ALL specified tags. If false (default), documents with ANY of the tags are returned.",
                    "default": False
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of documents to return (default: 20)",
                    "default": 20
                }
            },
            "required": ["tags"]
        }
    },
    {
        "name": "search_documents_by_tag",
        "description": "Find documents that have specific tags. Useful for filtering documents by category or topic.",
        "parameters": {
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tags to search for"
                },
                "match_all": {
                    "type": "boolean",
                    "description": "If true, documents must have ALL specified tags. If false (default), documents with ANY of the tags are returned.",
                    "default": False
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of documents to return (default: 20)",
                    "default": 20
                }
            },
            "required": ["tags"]
        }
    },
    {
        "name": "list_all_tags",
        "description": "Get a list of all unique tags used across documents in the knowledge base.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "compare_documents",
        "description": "Compare two documents to find similarities and differences. Analyzes content overlap, unique sections, and provides a similarity score.",
        "parameters": {
            "type": "object",
            "properties": {
                "document_id_1": {
                    "type": "string",
                    "description": "The UUID of the first document to compare"
                },
                "document_id_2": {
                    "type": "string",
                    "description": "The UUID of the second document to compare"
                },
                "comparison_type": {
                    "type": "string",
                    "enum": ["semantic", "keyword", "full"],
                    "description": "Type of comparison: 'semantic' (meaning-based), 'keyword' (word overlap), or 'full' (both). Default: 'full'"
                }
            },
            "required": ["document_id_1", "document_id_2"]
        }
    },
    {
        "name": "start_template_fill",
        "description": "Start a template fill job. Analyzes a template document and fills it with content extracted from source documents using AI. Use this when the user wants to fill a template with information from their documents.",
        "parameters": {
            "type": "object",
            "properties": {
                "source_document_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of document UUIDs to use as source content for filling the template"
                }
            },
            "required": ["source_document_ids"]
        }
    },
    {
        "name": "list_template_jobs",
        "description": "List the user's template fill jobs with their status and progress.",
        "parameters": {
            "type": "object",
            "properties": {
                "status_filter": {
                    "type": "string",
                    "enum": ["all", "pending", "processing", "completed", "failed"],
                    "description": "Filter jobs by status. Default: 'all'"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of jobs to return (default: 10)",
                    "default": 10
                }
            },
            "required": []
        }
    },
    {
        "name": "get_template_job_status",
        "description": "Get detailed status of a specific template fill job including progress, current section being processed, and download link if completed.",
        "parameters": {
            "type": "object",
            "properties": {
                "job_id": {
                    "type": "string",
                    "description": "The UUID of the template job to check"
                }
            },
            "required": ["job_id"]
        }
    },
    # =========================================================================
    # RAG / Q&A Tools
    # =========================================================================
    {
        "name": "answer_question",
        "description": "Answer a question using RAG (Retrieval-Augmented Generation) by searching the knowledge base and generating a response based on relevant document content. Use this when the user asks a factual question that should be answered using information from their documents.",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question to answer using document context"
                },
                "max_sources": {
                    "type": "integer",
                    "description": "Maximum number of source documents to use for context (default: 5, max: 10)",
                    "default": 5
                }
            },
            "required": ["question"]
        }
    },
    # =========================================================================
    # Document Content Tools
    # =========================================================================
    {
        "name": "read_document_content",
        "description": "Read the full text content of a document. Use this when you need to see the actual content of a document, not just metadata or preview.",
        "parameters": {
            "type": "object",
            "properties": {
                "document_id": {
                    "type": "string",
                    "description": "The UUID of the document to read"
                },
                "max_length": {
                    "type": "integer",
                    "description": "Maximum number of characters to return (default: 10000, max: 50000)",
                    "default": 10000
                },
                "include_chunks": {
                    "type": "boolean",
                    "description": "If true, return content split by chunks with metadata",
                    "default": False
                }
            },
            "required": ["document_id"]
        }
    },
    # =========================================================================
    # Knowledge Graph Tools
    # =========================================================================
    {
        "name": "search_entities",
        "description": "Search for entities (people, organizations, locations, technologies, etc.) mentioned in the knowledge base documents.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query for entity names"
                },
                "entity_type": {
                    "type": "string",
                    "enum": ["person", "organization", "location", "product", "technology", "concept", "other"],
                    "description": "Filter by entity type (optional)"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of entities to return (default: 10)",
                    "default": 10
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "get_entity_relationships",
        "description": "Get relationships for a specific entity, showing how it connects to other entities in the knowledge graph.",
        "parameters": {
            "type": "object",
            "properties": {
                "entity_id": {
                    "type": "string",
                    "description": "The UUID of the entity"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of relationships to return (default: 20)",
                    "default": 20
                }
            },
            "required": ["entity_id"]
        }
    },
    {
        "name": "find_documents_by_entity",
        "description": "Find all documents that mention a specific entity. Useful for exploring all content related to a person, organization, or concept.",
        "parameters": {
            "type": "object",
            "properties": {
                "entity_id": {
                    "type": "string",
                    "description": "The UUID of the entity"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of documents to return (default: 10)",
                    "default": 10
                }
            },
            "required": ["entity_id"]
        }
    },
    {
        "name": "get_document_knowledge_graph",
        "description": "Get the knowledge graph (entities and relationships) extracted from a specific document.",
        "parameters": {
            "type": "object",
            "properties": {
                "document_id": {
                    "type": "string",
                    "description": "The UUID of the document"
                }
            },
            "required": ["document_id"]
        }
    },
    {
        "name": "get_global_knowledge_graph",
        "description": "Get the global knowledge graph across all documents (entities and relationships), with optional filters and limits. Useful for building an overview graph or answering questions like 'what are the key entities and how are they connected?'.",
        "parameters": {
            "type": "object",
            "properties": {
                "entity_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Entity types to include (optional)"
                },
                "relation_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Relationship types to include (optional)"
                },
                "min_confidence": {
                    "type": "number",
                    "description": "Minimum relationship confidence (0.0-1.0)",
                    "default": 0.0
                },
                "min_mentions": {
                    "type": "integer",
                    "description": "Minimum mention count for an entity to be included",
                    "default": 1
                },
                "limit_nodes": {
                    "type": "integer",
                    "description": "Maximum number of nodes to return (default: 300, max: 1000)",
                    "default": 300
                },
                "limit_edges": {
                    "type": "integer",
                    "description": "Maximum number of edges to return (default: 1000, max: 5000)",
                    "default": 1000
                },
                "search": {
                    "type": "string",
                    "description": "Search entity names (optional)"
                }
            },
            "required": []
        }
    },
    {
        "name": "get_entity_mentions",
        "description": "Get the document mentions for a specific entity (snippets and metadata). Useful to ground an entity in the underlying sources.",
        "parameters": {
            "type": "object",
            "properties": {
                "entity_id": {
                    "type": "string",
                    "description": "The UUID of the entity"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of mentions to return (default: 25, max: 200)",
                    "default": 25
                },
                "offset": {
                    "type": "integer",
                    "description": "Pagination offset (default: 0)",
                    "default": 0
                }
            },
            "required": ["entity_id"]
        }
    },
    {
        "name": "get_kg_stats",
        "description": "Get knowledge graph statistics: counts of entities, relationships, and mentions.",
        "parameters": {"type": "object", "properties": {}, "required": []}
    },
    {
        "name": "rebuild_document_knowledge_graph",
        "description": "Admin-only: delete and rebuild the knowledge graph extracted from a document (re-extract entities/relationships from its chunks). Use when extraction rules/models changed or graph looks wrong.",
        "parameters": {
            "type": "object",
            "properties": {
                "document_id": {
                    "type": "string",
                    "description": "The UUID of the document"
                }
            },
            "required": ["document_id"]
        }
    },
    {
        "name": "merge_entities",
        "description": "Admin-only: merge a source entity into a target entity (repairs duplicates). Mentions and relationships are repointed; duplicates are deduplicated.",
        "parameters": {
            "type": "object",
            "properties": {
                "source_id": {"type": "string", "description": "Source entity UUID to merge from"},
                "target_id": {"type": "string", "description": "Target entity UUID to merge into"}
            },
            "required": ["source_id", "target_id"]
        }
    },
    {
        "name": "delete_entity",
        "description": "Admin-only: delete an entity from the knowledge graph (cascades mentions/relationships). Requires confirm_name to prevent accidental deletion.",
        "parameters": {
            "type": "object",
            "properties": {
                "entity_id": {"type": "string", "description": "Entity UUID to delete"},
                "confirm_name": {"type": "string", "description": "Must exactly match the entity's canonical name"}
            },
            "required": ["entity_id", "confirm_name"]
        }
    },
    {
        "name": "generate_diagram",
        "description": "Generate a visual diagram (architecture, flowchart, sequence, ER diagram, mind map, etc.) from documents or a description. Returns Mermaid diagram code that can be rendered visually. Use this when the user asks for architecture diagrams, system diagrams, flowcharts, or any visual representation of information from documents.",
        "parameters": {
            "type": "object",
            "properties": {
                "diagram_type": {
                    "type": "string",
                    "enum": ["flowchart", "sequence", "class", "state", "er", "gantt", "pie", "mindmap", "architecture", "auto"],
                    "description": "Type of diagram to generate. Use 'auto' to let AI choose the best type based on content.",
                    "default": "auto"
                },
                "source": {
                    "type": "string",
                    "enum": ["documents", "description", "search"],
                    "description": "Source for diagram generation: 'documents' (use specific doc IDs), 'description' (use provided text), 'search' (search and use results)",
                    "default": "description"
                },
                "document_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of document UUIDs to analyze (required if source is 'documents')"
                },
                "search_query": {
                    "type": "string",
                    "description": "Search query to find relevant documents (required if source is 'search')"
                },
                "description": {
                    "type": "string",
                    "description": "Text description of what to diagram (required if source is 'description')"
                },
                "focus": {
                    "type": "string",
                    "description": "Specific aspect to focus on (e.g., 'data flow', 'components', 'user interactions', 'dependencies')"
                },
                "detail_level": {
                    "type": "string",
                    "enum": ["high", "medium", "low"],
                    "description": "Level of detail in the diagram",
                    "default": "medium"
                }
            },
            "required": ["source"]
        }
    },
    # Workflow and Custom Tool Integration
    {
        "name": "run_workflow",
        "description": "Execute a saved workflow by name or ID. Workflows are user-defined automation sequences that can perform multiple operations.",
        "parameters": {
            "type": "object",
            "properties": {
                "workflow_name": {
                    "type": "string",
                    "description": "Name of the workflow to execute (case-insensitive search)"
                },
                "workflow_id": {
                    "type": "string",
                    "description": "UUID of the workflow to execute (alternative to name)"
                },
                "inputs": {
                    "type": "object",
                    "description": "Input parameters to pass to the workflow"
                }
            },
            "required": []
        }
    },
    {
        "name": "propose_workflow_from_description",
        "description": "Generate a workflow draft from a natural language description WITHOUT saving it. Use this to propose a workflow for the user to review/approve before saving.",
        "parameters": {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "Natural language description of the workflow to generate"
                },
                "name": {
                    "type": "string",
                    "description": "Optional name for the workflow"
                },
                "is_active": {
                    "type": "boolean",
                    "description": "Whether the workflow should be active (default: true)",
                    "default": True
                },
                "trigger_config": {
                    "type": "object",
                    "description": "Optional trigger configuration (manual, schedule, event, webhook)"
                }
            },
            "required": ["description"]
        }
    },
    {
        "name": "create_workflow_from_description",
        "description": "Generate and save a workflow from a natural language description. Returns the new workflow ID and summary.",
        "parameters": {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "Natural language description of the workflow to generate"
                },
                "name": {
                    "type": "string",
                    "description": "Optional name for the workflow"
                },
                "is_active": {
                    "type": "boolean",
                    "description": "Whether the workflow should be active (default: true)",
                    "default": True
                },
                "trigger_config": {
                    "type": "object",
                    "description": "Optional trigger configuration (manual, schedule, event, webhook)"
                }
            },
            "required": ["description"]
        }
    },
    {
        "name": "list_workflows",
        "description": "List available workflows that can be executed.",
        "parameters": {
            "type": "object",
            "properties": {
                "active_only": {
                    "type": "boolean",
                    "description": "Only list active workflows (default: true)",
                    "default": True
                }
            },
            "required": []
        }
    },
    {
        "name": "run_custom_tool",
        "description": "Execute a user-defined custom tool by name. Custom tools include webhooks, data transformers, Python scripts, and LLM prompts.",
        "parameters": {
            "type": "object",
            "properties": {
                "tool_name": {
                    "type": "string",
                    "description": "Name of the custom tool to execute"
                },
                "inputs": {
                    "type": "object",
                    "description": "Input parameters for the tool"
                }
            },
            "required": ["tool_name"]
        }
    },
    {
        "name": "list_custom_tools",
        "description": "List available custom tools that can be executed.",
        "parameters": {
            "type": "object",
            "properties": {
                "tool_type": {
                    "type": "string",
                    "enum": ["webhook", "transform", "python", "llm_prompt"],
                    "description": "Filter by tool type (optional)"
                }
            },
            "required": []
        }
    },
    # Agent Collaboration Tool
    {
        "name": "delegate_to_agent",
        "description": "Delegate a specific subtask to another specialized agent. Use when the task requires expertise outside your specialty. The other agent will process the request and return results. Available agents: qa_specialist (answering questions), document_expert (document operations), code_expert (code analysis), research_assistant (deep research), data_analyst (insights and visualizations), report_generator (creating reports), workflow_assistant (automation).",
        "parameters": {
            "type": "object",
            "properties": {
                "target_agent": {
                    "type": "string",
                    "description": "Name of the agent to delegate to (e.g., 'qa_specialist', 'code_expert', 'research_assistant')"
                },
                "task_description": {
                    "type": "string",
                    "description": "Clear description of what you need the other agent to do"
                },
                "context": {
                    "type": "string",
                    "description": "Relevant context from your current analysis to pass to the other agent (optional)"
                }
            },
            "required": ["target_agent", "task_description"]
        }
    },
    {
        "name": "list_available_agents",
        "description": "List all available specialized agents that can be delegated to, including their capabilities and descriptions.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
]


def get_tools_description() -> str:
    """Generate a text description of available tools for the LLM prompt."""
    descriptions = []
    for tool in AGENT_TOOLS:
        params = tool["parameters"]["properties"]
        param_list = []
        for name, info in params.items():
            required = name in tool["parameters"].get("required", [])
            param_str = f"  - {name} ({info['type']}{'*' if required else ''}): {info['description']}"
            param_list.append(param_str)

        tool_desc = f"""Tool: {tool['name']}
Description: {tool['description']}
Parameters:
{chr(10).join(param_list) if param_list else '  (no parameters)'}"""
        descriptions.append(tool_desc)

    return "\n\n".join(descriptions)


def get_tool_by_name(name: str) -> Dict[str, Any] | None:
    """Get a tool definition by name."""
    for tool in AGENT_TOOLS:
        if tool["name"] == name:
            return tool
    return None


def validate_tool_params(tool_name: str, params: Dict[str, Any]) -> tuple[bool, str]:
    """
    Validate parameters for a tool call.

    Returns:
        Tuple of (is_valid, error_message)
    """
    tool = get_tool_by_name(tool_name)
    if not tool:
        return False, f"Unknown tool: {tool_name}"

    required_params = tool["parameters"].get("required", [])
    for param in required_params:
        if param not in params:
            return False, f"Missing required parameter: {param}"

    return True, ""
