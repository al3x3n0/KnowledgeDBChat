"""
Tool definitions for the agentic chat system.

Defines available tools that the agent can use to perform document operations.
"""

from typing import Any, Dict, List

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
                    "description": "The search query to find relevant documents",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 5, max: 20)",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "web_scrape",
        "description": "Fetch a web page (or a small set of pages) and extract readable text and links. Useful for wikis/portals.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch (http/https)",
                },
                "follow_links": {
                    "type": "boolean",
                    "description": "Whether to crawl links from the page (bounded by max_pages/max_depth)",
                    "default": False,
                },
                "max_pages": {
                    "type": "integer",
                    "description": "Maximum pages to fetch when crawling (default: 1, max: 25)",
                    "default": 1,
                },
                "max_depth": {
                    "type": "integer",
                    "description": "Maximum crawl depth when follow_links is true (default: 0, max: 5)",
                    "default": 0,
                },
                "same_domain_only": {
                    "type": "boolean",
                    "description": "Only follow links on the same domain as the start URL",
                    "default": True,
                },
                "include_links": {
                    "type": "boolean",
                    "description": "Include extracted links in the response",
                    "default": True,
                },
                "allow_private_networks": {
                    "type": "boolean",
                    "description": "Allow private-network hosts (admin only)",
                    "default": False,
                },
                "max_content_chars": {
                    "type": "integer",
                    "description": "Maximum characters to return per page (default: 50000, max: 500000)",
                    "default": 50000,
                },
            },
            "required": ["url"],
        },
    },
    {
        "name": "ingest_url",
        "description": "Scrape a URL and ingest the extracted text into the KnowledgeDB as document(s) (optionally crawling a few linked pages).",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to ingest (http/https)",
                },
                "title": {
                    "type": "string",
                    "description": "Optional title override for the created document (single-document mode only)",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tags to attach to created/updated documents (optional)",
                },
                "ingest_mode": {
                    "type": "string",
                    "enum": ["auto", "web", "youtube"],
                    "description": "Ingestion mode; auto routes YouTube URLs to media download + transcription",
                    "default": "auto",
                },
                "youtube_audio_only": {
                    "type": "boolean",
                    "description": "When ingesting YouTube, prefer audio-only stream for faster transcription",
                    "default": True,
                },
                "follow_links": {
                    "type": "boolean",
                    "description": "Whether to crawl links from the page (bounded by max_pages/max_depth)",
                    "default": False,
                },
                "max_pages": {
                    "type": "integer",
                    "description": "Maximum pages to fetch when crawling (default: 1, max: 25)",
                    "default": 1,
                },
                "max_depth": {
                    "type": "integer",
                    "description": "Maximum crawl depth when follow_links is true (default: 0, max: 5)",
                    "default": 0,
                },
                "same_domain_only": {
                    "type": "boolean",
                    "description": "Only follow links on the same domain as the start URL",
                    "default": True,
                },
                "one_document_per_page": {
                    "type": "boolean",
                    "description": "If crawling, create/update one document per page URL instead of combining into one",
                    "default": False,
                },
                "allow_private_networks": {
                    "type": "boolean",
                    "description": "Allow private-network hosts (admin only, or allowlisted web sources)",
                    "default": False,
                },
                "max_content_chars": {
                    "type": "integer",
                    "description": "Maximum characters to return per page before ingesting (default: 50000, max: 500000)",
                    "default": 50000,
                },
            },
            "required": ["url"],
        },
    },
    {
        "name": "get_document_details",
        "description": "Get detailed information about a specific document including title, content preview, metadata, and processing status.",
        "parameters": {
            "type": "object",
            "properties": {
                "document_id": {
                    "type": "string",
                    "description": "The UUID of the document to retrieve",
                }
            },
            "required": ["document_id"],
        },
    },
    {
        "name": "summarize_document",
        "description": "Generate or retrieve a summary for a specific document. If a summary already exists, returns it unless force_regenerate is true.",
        "parameters": {
            "type": "object",
            "properties": {
                "document_id": {
                    "type": "string",
                    "description": "The UUID of the document to summarize",
                },
                "force_regenerate": {
                    "type": "boolean",
                    "description": "Force regeneration of summary even if one already exists",
                    "default": False,
                },
            },
            "required": ["document_id"],
        },
    },
    {
        "name": "delete_document",
        "description": "Delete a document from the knowledge base. This action is irreversible and requires explicit confirmation.",
        "parameters": {
            "type": "object",
            "properties": {
                "document_id": {
                    "type": "string",
                    "description": "The UUID of the document to delete",
                },
                "confirm": {
                    "type": "boolean",
                    "description": "Must be set to true to confirm deletion. If false or missing, will only return document info for confirmation.",
                    "default": False,
                },
            },
            "required": ["document_id"],
        },
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
                    "default": 10,
                }
            },
            "required": [],
        },
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
                    "default": False,
                }
            },
            "required": [],
        },
    },
    {
        "name": "list_documents_by_source",
        "description": "List documents from a specific source (by source ID, name, or type).",
        "parameters": {
            "type": "object",
            "properties": {
                "source_id": {
                    "type": "string",
                    "description": "UUID of the document source",
                },
                "source_name": {
                    "type": "string",
                    "description": "Name of the document source (case-insensitive, partial match)",
                },
                "source_type": {
                    "type": "string",
                    "description": "Source type (e.g., gitlab, confluence, web, file)",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of documents to return (default: 20, max: 50)",
                    "default": 20,
                },
                "offset": {
                    "type": "integer",
                    "description": "Pagination offset (default: 0)",
                    "default": 0,
                },
            },
            "required": [],
        },
    },
    {
        "name": "request_file_upload",
        "description": "Request the user to upload a file. Use this when the user wants to add a new document to the knowledge base.",
        "parameters": {
            "type": "object",
            "properties": {
                "suggested_title": {
                    "type": "string",
                    "description": "Suggested title for the document (optional)",
                },
                "suggested_tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Suggested tags for categorization (optional)",
                },
            },
            "required": [],
        },
    },
    {
        "name": "create_document_from_text",
        "description": "Create a new document directly from text content. Useful for saving notes, code snippets, or any text the user wants to store in the knowledge base.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Title for the new document",
                },
                "content": {
                    "type": "string",
                    "description": "The text content to save as a document",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tags for categorization (optional)",
                },
            },
            "required": ["title", "content"],
        },
    },
    {
        "name": "find_similar_documents",
        "description": "Find documents that are semantically similar to a given document. Useful for discovering related content.",
        "parameters": {
            "type": "object",
            "properties": {
                "document_id": {
                    "type": "string",
                    "description": "The UUID of the reference document",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of similar documents to return (default: 5)",
                    "default": 5,
                },
            },
            "required": ["document_id"],
        },
    },
    {
        "name": "search_documents_by_author",
        "description": "Find documents authored by a person. Uses case-insensitive matching.",
        "parameters": {
            "type": "object",
            "properties": {
                "author": {
                    "type": "string",
                    "description": "Author name or substring to search for",
                },
                "match_type": {
                    "type": "string",
                    "enum": ["contains", "exact", "starts_with"],
                    "description": "Match strategy for author names",
                    "default": "contains",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of documents to return (default: 20, max: 50)",
                    "default": 20,
                },
            },
            "required": ["author"],
        },
    },
    {
        "name": "update_document_tags",
        "description": "Add, remove, or replace tags on a document. Use action 'add' to add tags, 'remove' to remove tags, or 'replace' to replace all tags.",
        "parameters": {
            "type": "object",
            "properties": {
                "document_id": {
                    "type": "string",
                    "description": "The UUID of the document to update",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tags to add, remove, or set",
                },
                "action": {
                    "type": "string",
                    "enum": ["add", "remove", "replace"],
                    "description": "Action to perform: 'add' (default), 'remove', or 'replace'",
                },
            },
            "required": ["document_id", "tags"],
        },
    },
    {
        "name": "get_knowledge_base_stats",
        "description": "Get statistics about the knowledge base including document counts, storage usage, and processing status.",
        "parameters": {"type": "object", "properties": {}, "required": []},
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
                    "description": "List of document UUIDs to delete",
                },
                "confirm": {
                    "type": "boolean",
                    "description": "Must be set to true to confirm batch deletion",
                    "default": False,
                },
            },
            "required": ["document_ids"],
        },
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
                    "description": "List of document UUIDs to summarize",
                },
                "force_regenerate": {
                    "type": "boolean",
                    "description": "Force regeneration even if summaries exist",
                    "default": False,
                },
            },
            "required": ["document_ids"],
        },
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
                    "description": "Tags to search for",
                },
                "match_all": {
                    "type": "boolean",
                    "description": "If true, documents must have ALL specified tags. If false (default), documents with ANY of the tags are returned.",
                    "default": False,
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of documents to return (default: 20)",
                    "default": 20,
                },
            },
            "required": ["tags"],
        },
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
                    "description": "Tags to search for",
                },
                "match_all": {
                    "type": "boolean",
                    "description": "If true, documents must have ALL specified tags. If false (default), documents with ANY of the tags are returned.",
                    "default": False,
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of documents to return (default: 20)",
                    "default": 20,
                },
            },
            "required": ["tags"],
        },
    },
    {
        "name": "list_all_tags",
        "description": "Get a list of all unique tags used across documents in the knowledge base.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "compare_documents",
        "description": "Compare two documents to find similarities and differences. Analyzes content overlap, unique sections, and provides a similarity score.",
        "parameters": {
            "type": "object",
            "properties": {
                "document_id_1": {
                    "type": "string",
                    "description": "The UUID of the first document to compare",
                },
                "document_id_2": {
                    "type": "string",
                    "description": "The UUID of the second document to compare",
                },
                "comparison_type": {
                    "type": "string",
                    "enum": ["semantic", "keyword", "full"],
                    "description": "Type of comparison: 'semantic' (meaning-based), 'keyword' (word overlap), or 'full' (both). Default: 'full'",
                },
            },
            "required": ["document_id_1", "document_id_2"],
        },
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
                    "description": "List of document UUIDs to use as source content for filling the template",
                }
            },
            "required": ["source_document_ids"],
        },
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
                    "description": "Filter jobs by status. Default: 'all'",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of jobs to return (default: 10)",
                    "default": 10,
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_template_job_status",
        "description": "Get detailed status of a specific template fill job including progress, current section being processed, and download link if completed.",
        "parameters": {
            "type": "object",
            "properties": {
                "job_id": {
                    "type": "string",
                    "description": "The UUID of the template job to check",
                }
            },
            "required": ["job_id"],
        },
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
                    "description": "The question to answer using document context",
                },
                "max_sources": {
                    "type": "integer",
                    "description": "Maximum number of source documents to use for context (default: 5, max: 10)",
                    "default": 5,
                },
            },
            "required": ["question"],
        },
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
                    "description": "The UUID of the document to read",
                },
                "max_length": {
                    "type": "integer",
                    "description": "Maximum number of characters to return (default: 10000, max: 50000)",
                    "default": 10000,
                },
                "include_chunks": {
                    "type": "boolean",
                    "description": "If true, return content split by chunks with metadata",
                    "default": False,
                },
            },
            "required": ["document_id"],
        },
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
                    "description": "Search query for entity names",
                },
                "entity_type": {
                    "type": "string",
                    "enum": [
                        "person",
                        "organization",
                        "location",
                        "product",
                        "technology",
                        "concept",
                        "other",
                    ],
                    "description": "Filter by entity type (optional)",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of entities to return (default: 10)",
                    "default": 10,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_entity_relationships",
        "description": "Get relationships for a specific entity, showing how it connects to other entities in the knowledge graph.",
        "parameters": {
            "type": "object",
            "properties": {
                "entity_id": {
                    "type": "string",
                    "description": "The UUID of the entity",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of relationships to return (default: 20)",
                    "default": 20,
                },
            },
            "required": ["entity_id"],
        },
    },
    {
        "name": "find_documents_by_entity",
        "description": "Find all documents that mention a specific entity. Useful for exploring all content related to a person, organization, or concept.",
        "parameters": {
            "type": "object",
            "properties": {
                "entity_id": {
                    "type": "string",
                    "description": "The UUID of the entity",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of documents to return (default: 10)",
                    "default": 10,
                },
            },
            "required": ["entity_id"],
        },
    },
    {
        "name": "get_document_knowledge_graph",
        "description": "Get the knowledge graph (entities and relationships) extracted from a specific document.",
        "parameters": {
            "type": "object",
            "properties": {
                "document_id": {
                    "type": "string",
                    "description": "The UUID of the document",
                }
            },
            "required": ["document_id"],
        },
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
                    "description": "Entity types to include (optional)",
                },
                "relation_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Relationship types to include (optional)",
                },
                "min_confidence": {
                    "type": "number",
                    "description": "Minimum relationship confidence (0.0-1.0)",
                    "default": 0.0,
                },
                "min_mentions": {
                    "type": "integer",
                    "description": "Minimum mention count for an entity to be included",
                    "default": 1,
                },
                "limit_nodes": {
                    "type": "integer",
                    "description": "Maximum number of nodes to return (default: 300, max: 1000)",
                    "default": 300,
                },
                "limit_edges": {
                    "type": "integer",
                    "description": "Maximum number of edges to return (default: 1000, max: 5000)",
                    "default": 1000,
                },
                "search": {
                    "type": "string",
                    "description": "Search entity names (optional)",
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_entity_mentions",
        "description": "Get the document mentions for a specific entity (snippets and metadata). Useful to ground an entity in the underlying sources.",
        "parameters": {
            "type": "object",
            "properties": {
                "entity_id": {
                    "type": "string",
                    "description": "The UUID of the entity",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of mentions to return (default: 25, max: 200)",
                    "default": 25,
                },
                "offset": {
                    "type": "integer",
                    "description": "Pagination offset (default: 0)",
                    "default": 0,
                },
            },
            "required": ["entity_id"],
        },
    },
    {
        "name": "get_kg_stats",
        "description": "Get knowledge graph statistics: counts of entities, relationships, and mentions.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "rebuild_document_knowledge_graph",
        "description": "Admin-only: delete and rebuild the knowledge graph extracted from a document (re-extract entities/relationships from its chunks). Use when extraction rules/models changed or graph looks wrong.",
        "parameters": {
            "type": "object",
            "properties": {
                "document_id": {
                    "type": "string",
                    "description": "The UUID of the document",
                }
            },
            "required": ["document_id"],
        },
    },
    {
        "name": "merge_entities",
        "description": "Admin-only: merge a source entity into a target entity (repairs duplicates). Mentions and relationships are repointed; duplicates are deduplicated.",
        "parameters": {
            "type": "object",
            "properties": {
                "source_id": {
                    "type": "string",
                    "description": "Source entity UUID to merge from",
                },
                "target_id": {
                    "type": "string",
                    "description": "Target entity UUID to merge into",
                },
            },
            "required": ["source_id", "target_id"],
        },
    },
    {
        "name": "delete_entity",
        "description": "Admin-only: delete an entity from the knowledge graph (cascades mentions/relationships). Requires confirm_name to prevent accidental deletion.",
        "parameters": {
            "type": "object",
            "properties": {
                "entity_id": {"type": "string", "description": "Entity UUID to delete"},
                "confirm_name": {
                    "type": "string",
                    "description": "Must exactly match the entity's canonical name",
                },
            },
            "required": ["entity_id", "confirm_name"],
        },
    },
    {
        "name": "generate_diagram",
        "description": "Generate a visual diagram (architecture, flowchart, sequence, ER diagram, mind map, etc.) from documents or a description. Returns Mermaid diagram code that can be rendered visually. Use this when the user asks for architecture diagrams, system diagrams, flowcharts, or any visual representation of information from documents.",
        "parameters": {
            "type": "object",
            "properties": {
                "diagram_type": {
                    "type": "string",
                    "enum": [
                        "flowchart",
                        "sequence",
                        "class",
                        "state",
                        "er",
                        "gantt",
                        "pie",
                        "mindmap",
                        "architecture",
                        "auto",
                    ],
                    "description": "Type of diagram to generate. Use 'auto' to let AI choose the best type based on content.",
                    "default": "auto",
                },
                "source": {
                    "type": "string",
                    "enum": ["documents", "description", "search", "gitlab_repo"],
                    "description": "Source for diagram generation: 'documents' (use specific doc IDs), 'description' (use provided text), 'search' (search and use results), 'gitlab_repo' (analyze GitLab repository)",
                    "default": "description",
                },
                "document_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of document UUIDs to analyze (required if source is 'documents')",
                },
                "search_query": {
                    "type": "string",
                    "description": "Search query to find relevant documents (required if source is 'search')",
                },
                "description": {
                    "type": "string",
                    "description": "Text description of what to diagram (required if source is 'description')",
                },
                "gitlab_project": {
                    "type": "string",
                    "description": "GitLab project ID or path (required if source is 'gitlab_repo')",
                },
                "gitlab_branch": {
                    "type": "string",
                    "description": "Branch to analyze (optional, defaults to default branch)",
                },
                "focus": {
                    "type": "string",
                    "description": "Specific aspect to focus on (e.g., 'data flow', 'components', 'user interactions', 'dependencies')",
                },
                "detail_level": {
                    "type": "string",
                    "enum": ["high", "medium", "low"],
                    "description": "Level of detail in the diagram",
                    "default": "medium",
                },
            },
            "required": ["source"],
        },
    },
    {
        "name": "generate_gitlab_architecture",
        "description": "Generate an architecture diagram from a GitLab repository. Analyzes the repository structure, README, config files (docker-compose, package.json, requirements.txt, etc.) and code to understand the system architecture and generate a visual diagram. Use this when the user asks to create an architecture diagram from a GitLab repo.",
        "parameters": {
            "type": "object",
            "properties": {
                "project_id": {
                    "type": "string",
                    "description": "GitLab project ID or path (e.g., 'group/project' or numeric ID)",
                },
                "branch": {
                    "type": "string",
                    "description": "Branch to analyze (optional, defaults to default branch)",
                },
                "diagram_type": {
                    "type": "string",
                    "enum": ["flowchart", "architecture", "c4", "auto"],
                    "description": "Type of architecture diagram to generate",
                    "default": "auto",
                },
                "focus": {
                    "type": "string",
                    "description": "Specific aspect to focus on: 'services', 'data_flow', 'dependencies', 'deployment', 'components'",
                },
                "detail_level": {
                    "type": "string",
                    "enum": ["high", "medium", "low"],
                    "description": "Level of detail: 'high' (all components), 'medium' (main components), 'low' (overview only)",
                    "default": "medium",
                },
            },
            "required": ["project_id"],
        },
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
                    "description": "Name of the workflow to execute (case-insensitive search)",
                },
                "workflow_id": {
                    "type": "string",
                    "description": "UUID of the workflow to execute (alternative to name)",
                },
                "inputs": {
                    "type": "object",
                    "description": "Input parameters to pass to the workflow",
                },
            },
            "required": [],
        },
    },
    {
        "name": "propose_workflow_from_description",
        "description": "Generate a workflow draft from a natural language description WITHOUT saving it. Use this to propose a workflow for the user to review/approve before saving.",
        "parameters": {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "Natural language description of the workflow to generate",
                },
                "name": {
                    "type": "string",
                    "description": "Optional name for the workflow",
                },
                "is_active": {
                    "type": "boolean",
                    "description": "Whether the workflow should be active (default: true)",
                    "default": True,
                },
                "trigger_config": {
                    "type": "object",
                    "description": "Optional trigger configuration (manual, schedule, event, webhook)",
                },
                "synthesize_custom_tools": {
                    "type": "boolean",
                    "description": "Allow generating custom tool drafts alongside the workflow (including docker_container tools)",
                    "default": False,
                },
                "preferred_tool_type": {
                    "type": "string",
                    "enum": [
                        "webhook",
                        "transform",
                        "python",
                        "llm_prompt",
                        "docker_container",
                    ],
                    "description": "Bias synthesized custom tools toward this type",
                },
                "expose_workflow_as_tool": {
                    "type": "boolean",
                    "description": "Also generate a workflow_runner tool draft wrapping this workflow",
                    "default": False,
                },
                "workflow_tool_name": {
                    "type": "string",
                    "description": "Optional custom name for the synthesized workflow_runner tool",
                },
            },
            "required": ["description"],
        },
    },
    {
        "name": "create_workflow_from_description",
        "description": "Generate and save a workflow from a natural language description. Returns the new workflow ID and summary.",
        "parameters": {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "Natural language description of the workflow to generate",
                },
                "name": {
                    "type": "string",
                    "description": "Optional name for the workflow",
                },
                "is_active": {
                    "type": "boolean",
                    "description": "Whether the workflow should be active (default: true)",
                    "default": True,
                },
                "trigger_config": {
                    "type": "object",
                    "description": "Optional trigger configuration (manual, schedule, event, webhook)",
                },
                "synthesize_custom_tools": {
                    "type": "boolean",
                    "description": "Generate and persist custom tools from the description (supports docker_container)",
                    "default": False,
                },
                "preferred_tool_type": {
                    "type": "string",
                    "enum": [
                        "webhook",
                        "transform",
                        "python",
                        "llm_prompt",
                        "docker_container",
                    ],
                    "description": "Bias synthesized custom tools toward this type",
                },
                "expose_workflow_as_tool": {
                    "type": "boolean",
                    "description": "Create a workflow_runner custom tool for the saved workflow",
                    "default": False,
                },
                "workflow_tool_name": {
                    "type": "string",
                    "description": "Optional name for the created workflow_runner tool",
                },
            },
            "required": ["description"],
        },
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
                    "default": True,
                }
            },
            "required": [],
        },
    },
    {
        "name": "create_custom_tool",
        "description": (
            "Create a reusable custom tool for later use by you, by workflows, "
            "or by future jobs. Use it when you find yourself repeating the "
            "same shaped work. The tool is owned by this user and persists "
            "after the job ends. Types: transform (Jinja2/JSONPath over "
            "inputs), llm_prompt (templated model call), webhook (HTTP call to "
            "an external API), python (sandboxed, no subprocess/filesystem/"
            "network)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Tool name, unique for this user",
                },
                "description": {
                    "type": "string",
                    "description": "What the tool does and when to use it",
                },
                "tool_type": {
                    "type": "string",
                    "description": ("One of: transform, llm_prompt, webhook, python"),
                },
                "parameters_schema": {
                    "type": "object",
                    "description": "JSON Schema for the tool's inputs",
                },
                "config": {
                    "type": "object",
                    "description": (
                        "Type-specific configuration. llm_prompt: "
                        "{'user_prompt': '...'} and optional 'system_prompt'. "
                        "python: {'code': '...'}. webhook: {'url': ..., "
                        "'method': ...}. transform: {'expression': ...}. "
                        "Templates use Jinja2, so reference inputs as "
                        "{{ input_name }} with double braces; single braces are "
                        "left as literal text."
                    ),
                },
            },
            "required": ["name", "tool_type", "config"],
        },
    },
    {
        "name": "run_custom_tool",
        "description": "Execute a user-defined custom tool by name. Custom tools include webhooks, data transformers, Python scripts, and LLM prompts.",
        "parameters": {
            "type": "object",
            "properties": {
                "tool_name": {
                    "type": "string",
                    "description": "Name of the custom tool to execute",
                },
                "inputs": {
                    "type": "object",
                    "description": "Input parameters for the tool",
                },
            },
            "required": ["tool_name"],
        },
    },
    {
        "name": "list_custom_tools",
        "description": "List available custom tools that can be executed.",
        "parameters": {
            "type": "object",
            "properties": {
                "tool_type": {
                    "type": "string",
                    "enum": [
                        "webhook",
                        "external_agent",
                        "transform",
                        "python",
                        "llm_prompt",
                        "docker_container",
                        "workflow_runner",
                    ],
                    "description": "Filter by tool type (optional)",
                }
            },
            "required": [],
        },
    },
    {
        "name": "search_arxiv",
        "description": "Search scientific papers on arXiv (metadata + abstracts). Use arXiv query syntax such as 'all:transformers AND cat:cs.CL'.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "arXiv API search query (e.g. 'all:diffusion AND cat:cs.CV')",
                },
                "start": {
                    "type": "integer",
                    "description": "Pagination start offset (default: 0)",
                    "default": 0,
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum results to return (default: 10, max: 25)",
                    "default": 10,
                },
                "sort_by": {
                    "type": "string",
                    "enum": ["relevance", "lastUpdatedDate", "submittedDate"],
                    "description": "Sort by field",
                    "default": "relevance",
                },
                "sort_order": {
                    "type": "string",
                    "enum": ["ascending", "descending"],
                    "description": "Sort order",
                    "default": "descending",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "ingest_arxiv_papers",
        "description": "Ingest arXiv papers into the Knowledge DB by creating an arXiv document source and running ingestion (async). Provide either paper_ids, search_queries, or categories.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Optional display name for the ingestion source",
                },
                "search_queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "arXiv API search_query expressions",
                },
                "paper_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Explicit arXiv identifiers (e.g. 2401.12345)",
                },
                "categories": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "arXiv categories (e.g. cs.CL, cs.CV)",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum results per query (default: 25, max: 200)",
                    "default": 25,
                },
                "start": {
                    "type": "integer",
                    "description": "Pagination start offset for queries (default: 0)",
                    "default": 0,
                },
                "sort_by": {
                    "type": "string",
                    "enum": ["relevance", "lastUpdatedDate", "submittedDate"],
                    "description": "Sort by field",
                    "default": "submittedDate",
                },
                "sort_order": {
                    "type": "string",
                    "enum": ["ascending", "descending"],
                    "description": "Sort order",
                    "default": "descending",
                },
                "auto_sync": {
                    "type": "boolean",
                    "description": "Trigger ingestion immediately (default: true)",
                    "default": True,
                },
            },
            "required": [],
        },
    },
    {
        "name": "literature_review_arxiv",
        "description": "Search arXiv for a topic, optionally ingest top papers into the Knowledge DB, and return a compact literature review starter set (papers + links).",
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "Research topic (free text)",
                },
                "query": {
                    "type": "string",
                    "description": "Optional explicit arXiv query; if omitted, derived from topic",
                },
                "categories": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional arXiv categories to constrain the search",
                },
                "max_papers": {
                    "type": "integer",
                    "description": "How many papers to return (default: 5, max: 25)",
                    "default": 5,
                },
                "ingest": {
                    "type": "boolean",
                    "description": "Whether to ingest the returned papers into the Knowledge DB (default: true)",
                    "default": True,
                },
                "sort_by": {
                    "type": "string",
                    "enum": ["relevance", "lastUpdatedDate", "submittedDate"],
                    "description": "Sort by field",
                    "default": "relevance",
                },
                "sort_order": {
                    "type": "string",
                    "enum": ["ascending", "descending"],
                    "description": "Sort order",
                    "default": "descending",
                },
            },
            "required": ["topic"],
        },
    },
    {
        "name": "summarize_documents_in_source",
        "description": "Queue summarization for documents in a source (e.g., an arXiv import). Use this after ingestion to generate summaries and paper insights.",
        "parameters": {
            "type": "object",
            "properties": {
                "source_id": {"type": "string", "description": "Document source UUID"},
                "force": {
                    "type": "boolean",
                    "description": "Force re-summarize even if summary exists",
                    "default": False,
                },
                "only_missing": {
                    "type": "boolean",
                    "description": "Only summarize documents missing a summary (ignored if force=true)",
                    "default": True,
                },
                "limit": {
                    "type": "integer",
                    "description": "Max documents to queue",
                    "default": 500,
                },
            },
            "required": ["source_id"],
        },
    },
    {
        "name": "enrich_arxiv_metadata_for_source",
        "description": "Enrich arXiv papers in a source with BibTeX and DOI metadata (venue, keywords, affiliations) when available.",
        "parameters": {
            "type": "object",
            "properties": {
                "source_id": {"type": "string", "description": "arXiv source UUID"},
                "force": {
                    "type": "boolean",
                    "description": "Force refresh even if already enriched",
                    "default": False,
                },
                "limit": {
                    "type": "integer",
                    "description": "Max documents to queue",
                    "default": 500,
                },
            },
            "required": ["source_id"],
        },
    },
    {
        "name": "generate_literature_review_for_source",
        "description": "Generate a literature review document for an arXiv import source (uses available summaries and extracted paper insights).",
        "parameters": {
            "type": "object",
            "properties": {
                "source_id": {"type": "string", "description": "arXiv source UUID"},
                "topic": {
                    "type": "string",
                    "description": "Optional topic label for the report",
                },
            },
            "required": ["source_id"],
        },
    },
    {
        "name": "generate_slides_for_source",
        "description": "Generate slides (presentation job) for an arXiv import source. Prefers the literature review document if available.",
        "parameters": {
            "type": "object",
            "properties": {
                "source_id": {"type": "string", "description": "arXiv source UUID"},
                "title": {"type": "string", "description": "Presentation title"},
                "topic": {"type": "string", "description": "Presentation topic"},
                "slide_count": {
                    "type": "integer",
                    "description": "Slide count (3-40)",
                    "default": 10,
                },
                "style": {
                    "type": "string",
                    "description": "Presentation style",
                    "default": "professional",
                },
                "include_diagrams": {
                    "type": "boolean",
                    "description": "Include diagrams",
                    "default": True,
                },
                "prefer_review_document": {
                    "type": "boolean",
                    "description": "Use the literature review as the only source doc when available",
                    "default": True,
                },
            },
            "required": ["source_id"],
        },
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
                    "description": "Name of the agent to delegate to (e.g., 'qa_specialist', 'code_expert', 'research_assistant')",
                },
                "task_description": {
                    "type": "string",
                    "description": "Clear description of what you need the other agent to do",
                },
                "context": {
                    "type": "string",
                    "description": "Relevant context from your current analysis to pass to the other agent (optional)",
                },
            },
            "required": ["target_agent", "task_description"],
        },
    },
    {
        "name": "list_available_agents",
        "description": "List all available specialized agents that can be delegated to, including their capabilities and descriptions.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    # =========================================================================
    # Data Analysis & Visualization Tools
    # =========================================================================
    {
        "name": "get_collection_statistics",
        "description": "Get comprehensive statistics for a document collection including document counts, file sizes, word counts, processing status, top tags, top authors, and timeline data. Useful for understanding the knowledge base composition.",
        "parameters": {
            "type": "object",
            "properties": {
                "source_id": {
                    "type": "string",
                    "description": "Filter statistics to a specific document source UUID",
                },
                "tag": {
                    "type": "string",
                    "description": "Filter statistics to documents with a specific tag",
                },
                "date_from": {
                    "type": "string",
                    "description": "Start date filter (ISO format: YYYY-MM-DD)",
                },
                "date_to": {
                    "type": "string",
                    "description": "End date filter (ISO format: YYYY-MM-DD)",
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_source_analytics",
        "description": "Get detailed analytics for document sources including document counts, sizes, processing rates, and health status. Useful for monitoring data source performance.",
        "parameters": {
            "type": "object",
            "properties": {
                "source_id": {
                    "type": "string",
                    "description": "Specific source UUID to analyze (optional, returns all sources if not specified)",
                }
            },
            "required": [],
        },
    },
    {
        "name": "get_trending_topics",
        "description": "Find trending topics based on recent document tags and content. Shows which topics are rising, stable, or declining in frequency.",
        "parameters": {
            "type": "object",
            "properties": {
                "days": {
                    "type": "integer",
                    "description": "Number of days to look back for trends (default: 7)",
                    "default": 7,
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of trending topics to return (default: 10)",
                    "default": 10,
                },
            },
            "required": [],
        },
    },
    {
        "name": "generate_chart_data",
        "description": "Generate data for charts and visualizations. Returns structured data that can be used to create bar, line, pie, or area charts.",
        "parameters": {
            "type": "object",
            "properties": {
                "chart_type": {
                    "type": "string",
                    "enum": ["bar", "line", "pie", "area"],
                    "description": "Type of chart to generate data for",
                    "default": "bar",
                },
                "metric": {
                    "type": "string",
                    "enum": ["document_count", "file_size", "content_size"],
                    "description": "Metric to visualize",
                    "default": "document_count",
                },
                "group_by": {
                    "type": "string",
                    "enum": ["source_type", "file_type", "author", "date"],
                    "description": "Field to group the data by",
                    "default": "source_type",
                },
                "date_from": {
                    "type": "string",
                    "description": "Start date filter (ISO format)",
                },
                "date_to": {
                    "type": "string",
                    "description": "End date filter (ISO format)",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum data points (default: 10)",
                    "default": 10,
                },
            },
            "required": ["metric", "group_by"],
        },
    },
    {
        "name": "export_data",
        "description": "Export document data to various formats (JSON, CSV, JSONL). Useful for data analysis, backup, or integration with external tools.",
        "parameters": {
            "type": "object",
            "properties": {
                "format": {
                    "type": "string",
                    "enum": ["json", "csv", "jsonl"],
                    "description": "Export format",
                    "default": "json",
                },
                "source_id": {
                    "type": "string",
                    "description": "Filter to specific document source UUID",
                },
                "tag": {
                    "type": "string",
                    "description": "Filter to documents with specific tag",
                },
                "include_content": {
                    "type": "boolean",
                    "description": "Include full document content (default: false)",
                    "default": False,
                },
                "include_chunks": {
                    "type": "boolean",
                    "description": "Include document chunks (default: false)",
                    "default": False,
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum documents to export (default: 1000)",
                    "default": 1000,
                },
            },
            "required": [],
        },
    },
    # =========================================================================
    # Advanced Search Tools
    # =========================================================================
    {
        "name": "faceted_search",
        "description": "Execute a search with faceted results showing aggregations by source type, file type, author, tags, and date. Useful for exploring and filtering large result sets.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "page": {
                    "type": "integer",
                    "description": "Page number (default: 1)",
                    "default": 1,
                },
                "page_size": {
                    "type": "integer",
                    "description": "Results per page (default: 10)",
                    "default": 10,
                },
                "filters": {
                    "type": "object",
                    "description": "Filter criteria: {source_id, file_type, author, tags, date_range}",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_search_suggestions",
        "description": "Get search suggestions and autocomplete for a partial query. Returns suggestions from document titles, tags, and authors.",
        "parameters": {
            "type": "object",
            "properties": {
                "partial_query": {
                    "type": "string",
                    "description": "Partial search query to get suggestions for",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum suggestions (default: 5)",
                    "default": 5,
                },
            },
            "required": ["partial_query"],
        },
    },
    {
        "name": "get_related_searches",
        "description": "Get related search queries based on the current search. Useful for discovering related topics and expanding research.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Current search query"},
                "limit": {
                    "type": "integer",
                    "description": "Maximum related searches (default: 5)",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
    # =========================================================================
    # Content Generation Tools
    # =========================================================================
    {
        "name": "draft_email",
        "description": "Generate a professional email draft based on context and documents. Can reference knowledge base content for accurate information.",
        "parameters": {
            "type": "object",
            "properties": {
                "subject": {"type": "string", "description": "Email subject or topic"},
                "recipient": {
                    "type": "string",
                    "description": "Intended recipient (for context)",
                },
                "context": {
                    "type": "string",
                    "description": "Additional context or instructions for the email",
                },
                "document_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Document UUIDs to reference in the email",
                },
                "search_query": {
                    "type": "string",
                    "description": "Search query to find relevant documents to reference",
                },
                "tone": {
                    "type": "string",
                    "enum": ["professional", "casual", "formal", "friendly"],
                    "description": "Email tone (default: professional)",
                    "default": "professional",
                },
                "length": {
                    "type": "string",
                    "enum": ["short", "medium", "long"],
                    "description": "Email length (default: medium)",
                    "default": "medium",
                },
            },
            "required": ["subject"],
        },
    },
    {
        "name": "generate_meeting_notes",
        "description": "Generate structured meeting notes from a transcript or documents. Includes summary, key points, action items, and decisions.",
        "parameters": {
            "type": "object",
            "properties": {
                "transcript": {
                    "type": "string",
                    "description": "Meeting transcript text",
                },
                "document_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Document UUIDs containing meeting content",
                },
                "meeting_title": {
                    "type": "string",
                    "description": "Title of the meeting",
                },
                "participants": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of meeting participants",
                },
                "include_action_items": {
                    "type": "boolean",
                    "description": "Include action items section (default: true)",
                    "default": True,
                },
                "include_decisions": {
                    "type": "boolean",
                    "description": "Include decisions section (default: true)",
                    "default": True,
                },
            },
            "required": [],
        },
    },
    {
        "name": "generate_documentation",
        "description": "Generate technical or user documentation from source documents. Supports various documentation types and target audiences.",
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "Documentation topic"},
                "doc_type": {
                    "type": "string",
                    "enum": ["technical", "user_guide", "api", "how_to"],
                    "description": "Type of documentation (default: technical)",
                    "default": "technical",
                },
                "document_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Source document UUIDs",
                },
                "search_query": {
                    "type": "string",
                    "description": "Search query to find relevant source content",
                },
                "target_audience": {
                    "type": "string",
                    "enum": ["developers", "end_users", "admins"],
                    "description": "Target reader (default: developers)",
                    "default": "developers",
                },
                "include_examples": {
                    "type": "boolean",
                    "description": "Include code/usage examples (default: true)",
                    "default": True,
                },
            },
            "required": ["topic"],
        },
    },
    {
        "name": "generate_executive_summary",
        "description": "Generate a concise executive summary for leadership. Includes key findings, metrics, recommendations, and next steps.",
        "parameters": {
            "type": "object",
            "properties": {
                "document_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Source document UUIDs to summarize",
                },
                "search_query": {
                    "type": "string",
                    "description": "Search query to find relevant content",
                },
                "topic": {
                    "type": "string",
                    "description": "Focus topic for the summary",
                },
                "max_length": {
                    "type": "integer",
                    "description": "Maximum word count (default: 500)",
                    "default": 500,
                },
                "include_recommendations": {
                    "type": "boolean",
                    "description": "Include recommendations section (default: true)",
                    "default": True,
                },
                "include_metrics": {
                    "type": "boolean",
                    "description": "Include key metrics (default: true)",
                    "default": True,
                },
            },
            "required": [],
        },
    },
    {
        "name": "generate_report",
        "description": "Generate a structured report (status, analysis, research, or summary) from documents. Includes proper sections and formatting.",
        "parameters": {
            "type": "object",
            "properties": {
                "report_type": {
                    "type": "string",
                    "enum": ["status", "analysis", "research", "summary"],
                    "description": "Type of report to generate",
                },
                "document_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Source document UUIDs",
                },
                "search_query": {
                    "type": "string",
                    "description": "Search query to find relevant content",
                },
                "title": {"type": "string", "description": "Report title"},
                "sections": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Custom sections to include (optional)",
                },
            },
            "required": ["report_type"],
        },
    },
]


# =========================================================================
# Autonomous Agent Tools
# =========================================================================
AUTONOMOUS_AGENT_TOOLS: List[Dict[str, Any]] = [
    {
        "name": "project_bootstrap",
        "description": "Build a lightweight project profile from ingested repository files (stack, key files, test paths, and suggested commands).",
        "parameters": {
            "type": "object",
            "properties": {
                "source_id": {
                    "type": "string",
                    "description": "Optional document source UUID to scope profiling",
                },
                "max_files": {
                    "type": "integer",
                    "description": "Maximum source files to sample (default: 400, max: 2000)",
                    "default": 400,
                },
            },
            "required": [],
        },
    },
    {
        "name": "add_to_reading_list",
        "description": "Add papers or documents to a reading list for later review. Creates a new reading list if it doesn't exist.",
        "parameters": {
            "type": "object",
            "properties": {
                "list_name": {
                    "type": "string",
                    "description": "Name of the reading list to add to (will be created if it doesn't exist)",
                },
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "document_id": {
                                "type": "string",
                                "description": "Document UUID (if already in KB)",
                            },
                            "arxiv_id": {
                                "type": "string",
                                "description": "arXiv ID (e.g., '2301.12345')",
                            },
                            "title": {
                                "type": "string",
                                "description": "Title of the paper/document",
                            },
                            "notes": {
                                "type": "string",
                                "description": "Notes about why this was added",
                            },
                            "priority": {
                                "type": "integer",
                                "description": "Priority 1-5 (1=highest)",
                                "default": 3,
                            },
                        },
                    },
                    "description": "Items to add to the reading list",
                },
            },
            "required": ["list_name", "items"],
        },
    },
    {
        "name": "get_reading_lists",
        "description": "Get all reading lists and their items. Useful for checking existing research collections.",
        "parameters": {
            "type": "object",
            "properties": {
                "list_name": {
                    "type": "string",
                    "description": "Filter to a specific list name (optional)",
                },
                "include_items": {
                    "type": "boolean",
                    "description": "Include list items (default: true)",
                    "default": True,
                },
            },
            "required": [],
        },
    },
    {
        "name": "save_research_finding",
        "description": "Save a research finding or insight discovered during analysis. Findings are stored for later synthesis and reporting.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Brief title for the finding",
                },
                "content": {
                    "type": "string",
                    "description": "Detailed description of the finding",
                },
                "category": {
                    "type": "string",
                    "enum": [
                        "key_insight",
                        "methodology",
                        "result",
                        "gap",
                        "connection",
                        "contradiction",
                        "trend",
                    ],
                    "description": "Category of finding",
                },
                "source_document_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Document UUIDs that support this finding",
                },
                "confidence": {
                    "type": "number",
                    "description": "Confidence score 0.0-1.0",
                    "default": 0.8,
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tags for categorization",
                },
            },
            "required": ["title", "content", "category"],
        },
    },
    {
        "name": "get_research_findings",
        "description": "Retrieve saved research findings. Useful for reviewing what has been discovered so far.",
        "parameters": {
            "type": "object",
            "properties": {
                "category": {"type": "string", "description": "Filter by category"},
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Filter by tags",
                },
                "min_confidence": {
                    "type": "number",
                    "description": "Minimum confidence threshold",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum findings to return",
                    "default": 50,
                },
            },
            "required": [],
        },
    },
    {
        "name": "create_synthesis_document",
        "description": "Create a synthesis document from collected findings and sources. Generates a structured research report.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Title for the synthesis document",
                },
                "topic": {
                    "type": "string",
                    "description": "Research topic being synthesized",
                },
                "findings_to_include": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "IDs of specific findings to include (optional, includes all if empty)",
                },
                "document_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Source document UUIDs to reference",
                },
                "sections": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Custom sections to include (default: introduction, key findings, methodology, gaps, conclusion)",
                },
                "format": {
                    "type": "string",
                    "enum": ["markdown", "structured", "academic"],
                    "description": "Output format",
                    "default": "structured",
                },
            },
            "required": ["title", "topic"],
        },
    },
    {
        "name": "extract_paper_insights",
        "description": "Extract structured insights from a research paper including methodology, key findings, limitations, and future work.",
        "parameters": {
            "type": "object",
            "properties": {
                "document_id": {
                    "type": "string",
                    "description": "Document UUID of the paper",
                },
                "focus_areas": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": [
                            "methodology",
                            "results",
                            "limitations",
                            "future_work",
                            "contributions",
                            "related_work",
                            "datasets",
                            "metrics",
                        ],
                    },
                    "description": "Specific areas to focus extraction on",
                },
                "extract_entities": {
                    "type": "boolean",
                    "description": "Extract named entities (authors, institutions, methods, datasets)",
                    "default": True,
                },
            },
            "required": ["document_id"],
        },
    },
    {
        "name": "find_related_papers",
        "description": "Find papers related to a given paper through citations, shared authors, or semantic similarity.",
        "parameters": {
            "type": "object",
            "properties": {
                "document_id": {
                    "type": "string",
                    "description": "Document UUID of the reference paper",
                },
                "arxiv_id": {
                    "type": "string",
                    "description": "arXiv ID (alternative to document_id)",
                },
                "relation_type": {
                    "type": "string",
                    "enum": [
                        "semantic",
                        "citations",
                        "shared_authors",
                        "shared_topics",
                        "all",
                    ],
                    "description": "Type of relationship to find",
                    "default": "semantic",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum papers to return",
                    "default": 10,
                },
                "search_external": {
                    "type": "boolean",
                    "description": "Search external sources (arXiv) in addition to knowledge base",
                    "default": True,
                },
            },
            "required": [],
        },
    },
    {
        "name": "build_research_graph",
        "description": "Build a knowledge graph of concepts, methods, and relationships from a set of papers. Useful for understanding the research landscape.",
        "parameters": {
            "type": "object",
            "properties": {
                "document_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Document UUIDs to analyze",
                },
                "source_id": {
                    "type": "string",
                    "description": "Document source UUID (analyze all papers in source)",
                },
                "focus_on": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": [
                            "methods",
                            "datasets",
                            "metrics",
                            "authors",
                            "concepts",
                            "tools",
                        ],
                    },
                    "description": "Entity types to focus on",
                },
                "include_relationships": {
                    "type": "boolean",
                    "description": "Include relationships between entities",
                    "default": True,
                },
            },
            "required": [],
        },
    },
    {
        "name": "compare_methodologies",
        "description": "Compare methodologies across multiple papers. Useful for understanding different approaches to a problem.",
        "parameters": {
            "type": "object",
            "properties": {
                "document_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Document UUIDs of papers to compare",
                },
                "comparison_aspects": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": [
                            "approach",
                            "datasets",
                            "metrics",
                            "results",
                            "limitations",
                            "computational_cost",
                        ],
                    },
                    "description": "Aspects to compare",
                },
                "output_format": {
                    "type": "string",
                    "enum": ["table", "narrative", "structured"],
                    "description": "Output format for comparison",
                    "default": "structured",
                },
            },
            "required": ["document_ids"],
        },
    },
    {
        "name": "identify_research_gaps",
        "description": "Analyze papers to identify potential research gaps and opportunities.",
        "parameters": {
            "type": "object",
            "properties": {
                "document_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Document UUIDs to analyze",
                },
                "source_id": {
                    "type": "string",
                    "description": "Document source UUID (analyze all papers in source)",
                },
                "topic": {
                    "type": "string",
                    "description": "Research topic to focus gap analysis on",
                },
                "gap_types": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": [
                            "methodological",
                            "empirical",
                            "theoretical",
                            "application",
                            "dataset",
                            "evaluation",
                        ],
                    },
                    "description": "Types of gaps to look for",
                },
            },
            "required": [],
        },
    },
    {
        "name": "generate_research_presentation",
        "description": "Generate a presentation from research findings. Creates a presentation job that can be downloaded.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Presentation title"},
                "topic": {"type": "string", "description": "Research topic"},
                "document_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Source document UUIDs",
                },
                "slide_count": {
                    "type": "integer",
                    "description": "Number of slides (5-30)",
                    "default": 12,
                },
                "style": {
                    "type": "string",
                    "enum": ["academic", "professional", "technical"],
                    "description": "Presentation style",
                    "default": "academic",
                },
                "include_diagrams": {
                    "type": "boolean",
                    "description": "Include auto-generated diagrams",
                    "default": True,
                },
            },
            "required": ["title", "topic"],
        },
    },
    {
        "name": "monitor_arxiv_topic",
        "description": "Set up or check monitoring for new papers on a topic. Returns recent papers matching the criteria.",
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "Topic to monitor"},
                "query": {"type": "string", "description": "arXiv query expression"},
                "categories": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "arXiv categories to monitor",
                },
                "since_days": {
                    "type": "integer",
                    "description": "Look back this many days for new papers",
                    "default": 7,
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum papers to return",
                    "default": 20,
                },
            },
            "required": ["topic"],
        },
    },
    {
        "name": "ingest_paper_by_id",
        "description": "Ingest a specific paper into the knowledge base by its arXiv ID.",
        "parameters": {
            "type": "object",
            "properties": {
                "arxiv_id": {
                    "type": "string",
                    "description": "arXiv paper ID (e.g., '2301.12345')",
                },
                "add_to_reading_list": {
                    "type": "string",
                    "description": "Name of reading list to add paper to (optional)",
                },
                "extract_insights": {
                    "type": "boolean",
                    "description": "Extract and save insights after ingestion",
                    "default": True,
                },
            },
            "required": ["arxiv_id"],
        },
    },
    {
        "name": "write_progress_report",
        "description": "Write a progress report for the current job. Useful for documenting what has been accomplished so far.",
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Brief summary of progress",
                },
                "completed_tasks": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of completed tasks",
                },
                "pending_tasks": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of pending tasks",
                },
                "key_findings": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Key findings so far",
                },
                "blockers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Any blockers or issues",
                },
                "next_steps": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Planned next steps",
                },
            },
            "required": ["summary"],
        },
    },
    {
        "name": "analyze_document_cluster",
        "description": "Analyze a cluster of related documents to find common themes, differences, and patterns.",
        "parameters": {
            "type": "object",
            "properties": {
                "document_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Document UUIDs to analyze as a cluster",
                },
                "analysis_type": {
                    "type": "string",
                    "enum": ["themes", "evolution", "comparison", "comprehensive"],
                    "description": "Type of cluster analysis",
                    "default": "comprehensive",
                },
                "extract_timeline": {
                    "type": "boolean",
                    "description": "Extract temporal evolution of topics",
                    "default": False,
                },
            },
            "required": ["document_ids"],
        },
    },
    {
        "name": "suggest_next_action",
        "description": "Get AI suggestions for the next action based on current job state and findings. Useful when uncertain about how to proceed.",
        "parameters": {
            "type": "object",
            "properties": {
                "current_goal": {
                    "type": "string",
                    "description": "Current goal being worked on",
                },
                "progress_so_far": {
                    "type": "string",
                    "description": "Description of progress made",
                },
                "available_resources": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Available resources (documents, sources, etc.)",
                },
                "constraints": {
                    "type": "string",
                    "description": "Any constraints to consider",
                },
            },
            "required": ["current_goal"],
        },
    },
    {
        "name": "create_knowledge_base_entry",
        "description": "Create a new structured entry in the knowledge base (not a raw document). Good for storing curated knowledge from research.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Entry title"},
                "content": {
                    "type": "string",
                    "description": "Main content (markdown supported)",
                },
                "entry_type": {
                    "type": "string",
                    "enum": [
                        "concept",
                        "method",
                        "dataset",
                        "tool",
                        "finding",
                        "synthesis",
                        "comparison",
                    ],
                    "description": "Type of knowledge entry",
                },
                "related_documents": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Related document UUIDs",
                },
                "related_entities": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Related entity UUIDs from knowledge graph",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tags for categorization",
                },
                "metadata": {
                    "type": "object",
                    "description": "Additional structured metadata",
                },
            },
            "required": ["title", "content", "entry_type"],
        },
    },
    {
        "name": "link_entities",
        "description": "Create or strengthen a relationship between two entities in the knowledge graph.",
        "parameters": {
            "type": "object",
            "properties": {
                "source_entity_id": {
                    "type": "string",
                    "description": "Source entity UUID",
                },
                "target_entity_id": {
                    "type": "string",
                    "description": "Target entity UUID",
                },
                "source_name": {
                    "type": "string",
                    "description": "Source entity name (alternative to ID)",
                },
                "target_name": {
                    "type": "string",
                    "description": "Target entity name (alternative to ID)",
                },
                "relationship_type": {
                    "type": "string",
                    "description": "Type of relationship (e.g., 'uses', 'extends', 'compares_to', 'improves')",
                },
                "evidence": {
                    "type": "string",
                    "description": "Evidence or explanation for this relationship",
                },
                "confidence": {
                    "type": "number",
                    "description": "Confidence score 0.0-1.0",
                    "default": 0.8,
                },
            },
            "required": ["relationship_type"],
        },
    },
    {
        "name": "search_with_filters",
        "description": "Advanced search with multiple filters. More flexible than basic search.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "source_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Limit to specific sources",
                },
                "file_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Filter by file types",
                },
                "date_from": {
                    "type": "string",
                    "description": "Start date (ISO format)",
                },
                "date_to": {"type": "string", "description": "End date (ISO format)"},
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Filter by tags",
                },
                "min_relevance": {
                    "type": "number",
                    "description": "Minimum relevance score 0.0-1.0",
                    "default": 0.5,
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum results",
                    "default": 20,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "batch_ingest_papers",
        "description": "Ingest multiple papers at once into the knowledge base.",
        "parameters": {
            "type": "object",
            "properties": {
                "arxiv_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of arXiv IDs to ingest",
                },
                "source_name": {
                    "type": "string",
                    "description": "Name for the document source",
                },
                "add_to_reading_list": {
                    "type": "string",
                    "description": "Reading list to add papers to",
                },
            },
            "required": ["arxiv_ids"],
        },
    },
    # ── Structured Reasoning Tools ──────────────────────────────────
    {
        "name": "reflect",
        "description": "Self-reflect on current progress, approach quality, and potential blind spots. Stores reflection in state for future reference.",
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "What to reflect on (e.g., 'search strategy', 'evidence quality', 'goal alignment')",
                },
                "assessment": {
                    "type": "string",
                    "description": "Your self-assessment of the topic",
                },
                "blind_spots": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Identified blind spots or assumptions",
                },
                "suggested_corrections": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Corrective actions to consider",
                },
            },
            "required": ["topic", "assessment"],
        },
    },
    {
        "name": "hypothesize",
        "description": "Formulate and track a hypothesis. Hypotheses can later be confirmed, refuted, or updated with evidence.",
        "parameters": {
            "type": "object",
            "properties": {
                "hypothesis": {
                    "type": "string",
                    "description": "The hypothesis statement",
                },
                "rationale": {
                    "type": "string",
                    "description": "Why this hypothesis is plausible",
                },
                "testable_predictions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Observable predictions if the hypothesis is true",
                },
                "status": {
                    "type": "string",
                    "enum": [
                        "proposed",
                        "testing",
                        "supported",
                        "refuted",
                        "inconclusive",
                    ],
                    "description": "Current status of the hypothesis",
                },
                "hypothesis_id": {
                    "type": "string",
                    "description": "ID of existing hypothesis to update (leave empty for new)",
                },
            },
            "required": ["hypothesis"],
        },
    },
    {
        "name": "weigh_evidence",
        "description": "Score and record evidence for or against a claim or hypothesis. Maintains a running evidence ledger.",
        "parameters": {
            "type": "object",
            "properties": {
                "claim": {"type": "string", "description": "The claim being evaluated"},
                "hypothesis_id": {
                    "type": "string",
                    "description": "Link to a tracked hypothesis (optional)",
                },
                "evidence_for": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "statement": {"type": "string"},
                            "source_document_id": {"type": "string"},
                            "strength": {"type": "number", "description": "0.0-1.0"},
                        },
                    },
                    "description": "Evidence supporting the claim",
                },
                "evidence_against": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "statement": {"type": "string"},
                            "source_document_id": {"type": "string"},
                            "strength": {"type": "number"},
                        },
                    },
                    "description": "Evidence against the claim",
                },
                "verdict": {
                    "type": "string",
                    "enum": [
                        "strongly_supported",
                        "weakly_supported",
                        "neutral",
                        "weakly_refuted",
                        "strongly_refuted",
                    ],
                    "description": "Overall assessment",
                },
            },
            "required": ["claim", "verdict"],
        },
    },
    {
        "name": "critique_plan",
        "description": "Challenge the current execution plan. Identify weaknesses, missing steps, or questionable assumptions.",
        "parameters": {
            "type": "object",
            "properties": {
                "plan_summary": {
                    "type": "string",
                    "description": "Summary of the plan being critiqued",
                },
                "weaknesses": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Identified weaknesses in the plan",
                },
                "missing_steps": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Steps the plan is missing",
                },
                "assumptions_challenged": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Assumptions that should be questioned",
                },
                "severity": {
                    "type": "string",
                    "enum": ["minor", "moderate", "major"],
                    "description": "How severe the critique is",
                },
            },
            "required": ["plan_summary", "weaknesses"],
        },
    },
    # ── Multi-Agent Coordination Tools ──────────────────────────────
    {
        "name": "delegate_subtask",
        "description": "Spawn a child agent job to work on a specific subtask. The child runs asynchronously as a background task.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name for the subtask job"},
                "goal": {
                    "type": "string",
                    "description": "Goal for the child agent job",
                },
                "job_type": {
                    "type": "string",
                    "enum": ["research", "analysis", "synthesis", "custom"],
                    "description": "Type of job for the child agent",
                },
                "config": {
                    "type": "object",
                    "description": "Job-specific config to pass to child",
                },
                "max_iterations": {
                    "type": "integer",
                    "description": "Max iterations for child (capped at parent's remaining)",
                },
                "share_findings": {
                    "type": "boolean",
                    "description": "Share parent findings with child (default: true)",
                },
                "wait": {
                    "type": "boolean",
                    "description": "If true, poll for completion (blocks up to 60s)",
                },
            },
            "required": ["name", "goal"],
        },
    },
    {
        "name": "wait_for_subtask",
        "description": "Check status or wait for a delegated subtask to complete. Returns current status and results if available.",
        "parameters": {
            "type": "object",
            "properties": {
                "subtask_job_id": {
                    "type": "string",
                    "description": "Job ID of the delegated subtask",
                },
                "timeout_seconds": {
                    "type": "integer",
                    "description": "How long to poll (max 120, default 30)",
                },
            },
            "required": ["subtask_job_id"],
        },
    },
    {
        "name": "share_findings",
        "description": "Push findings to sibling agent jobs (those sharing the same parent). Used for coordination in multi-agent workflows.",
        "parameters": {
            "type": "object",
            "properties": {
                "findings": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "content": {"type": "string"},
                            "category": {"type": "string"},
                        },
                    },
                    "description": "Findings to share with siblings",
                },
                "target_job_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific sibling job IDs (optional, shares with all siblings if empty)",
                },
            },
            "required": ["findings"],
        },
    },
    {
        "name": "request_review",
        "description": "Ask another agent job or a human operator to review the current work. Creates a review checkpoint.",
        "parameters": {
            "type": "object",
            "properties": {
                "review_type": {
                    "type": "string",
                    "enum": ["peer_agent", "human"],
                    "description": "Whether to request review from a peer agent or human",
                },
                "content_to_review": {
                    "type": "string",
                    "description": "The content or summary to be reviewed",
                },
                "review_criteria": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Criteria for the reviewer to evaluate",
                },
                "reviewer_job_id": {
                    "type": "string",
                    "description": "Specific sibling job to request review from (for peer_agent type)",
                },
            },
            "required": ["content_to_review"],
        },
    },
    # ── Code & Execution Tools ──────────────────────────────────────
    {
        "name": "compile_c_snippet",
        "description": (
            "Compile a C snippet in the compiler research sandbox and return "
            "the generated assembly plus codegen counts (vector instructions, "
            "conditional branches, calls). Use this to check what the compiler "
            "actually emitted before drawing conclusions from any timing: a "
            "loop may be vectorized or if-converted, leaving no branch to "
            "measure. Prefer this over execute_python for anything involving a "
            "compiler."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "C source to compile"},
                "flags": {
                    "type": "string",
                    "description": (
                        "Compiler flags, e.g. '-O2' or '-O3 -ffast-math'. The "
                        "sandbox targets aarch64: use '-mcpu=native' to tune "
                        "for the host, as clang there rejects '-march=native'."
                    ),
                },
                "emit": {
                    "type": "string",
                    "description": "'asm' (default) returns assembly, 'ir' returns LLVM IR",
                },
                "label": {
                    "type": "string",
                    "description": (
                        "Short name for what this snippet is, e.g. 'float sum "
                        "reduction'. Recorded with the measurement; without it "
                        "several measurements cannot be told apart afterwards."
                    ),
                },
            },
            "required": ["code"],
        },
    },
    {
        "name": "profile_c_workload",
        "description": (
            "Compile a self-contained C program, run it under callgrind, and "
            "report what actually executed: exact dynamic instruction counts "
            "per function, and the hottest straight-line blocks with their "
            "disassembly. Use this to find where the time really goes before "
            "proposing anything -- source occurrence is not execution "
            "frequency, and a sequence appearing often in cold code is worth "
            "less than one appearing twice in an inner loop. Instrumented "
            "execution is ~50x slower than native, so give the program a "
            "bounded input. It counts instructions; it does not time them."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Self-contained C program including main()",
                },
                "flags": {
                    "type": "string",
                    "description": (
                        "Compiler flags (default '-O3 -g'). Debug info is added "
                        "if absent, since without it nothing can be attributed "
                        "to a function. This sandbox targets aarch64: use "
                        "'-mcpu=native', not '-march=native'."
                    ),
                },
                "run_args": {
                    "type": "string",
                    "description": "Arguments passed to the program (optional)",
                },
                "label": {
                    "type": "string",
                    "description": (
                        "Short name for this workload, recorded with the "
                        "profile so several can be told apart afterwards."
                    ),
                },
                "top_functions": {
                    "type": "integer",
                    "description": "How many functions to rank (default 8)",
                },
                "top_blocks": {
                    "type": "integer",
                    "description": "How many hot blocks to return (default 5)",
                },
            },
            "required": ["code"],
        },
    },
    {
        "name": "cost_fusion_candidate",
        "description": (
            "Cost a mined fusion candidate and bound what fusing it could "
            "save, per occurrence, on a named core. Costs the sequence as it "
            "stands and each operation in it alone, then reports the saving as "
            "a range: at best the sequence's cost minus the slowest operation "
            "the fused form still has to perform, at worst nothing. It does "
            "not ask you to name an instruction to stand for the fused one, "
            "because the answer would then depend on that choice -- picking a "
            "slow stand-in manufactures a regression. Multiply the range by "
            "the candidate's dynamic_occurrences for the benefit."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": (
                        "The candidate, as find_fusion_candidates spells it, "
                        "e.g. 'fsqrt fdiv | 0>1'."
                    ),
                },
                "cpu": {
                    "type": "string",
                    "description": (
                        "The core model to cost against, e.g. neoverse-n1. "
                        "Required: a cycle count is a property of a core."
                    ),
                },
                "mode": {
                    "type": "string",
                    "description": (
                        "'dependent' (default) measures the chain's latency, "
                        "what a loop-carried computation meets; 'independent' "
                        "measures throughput, what an unrolled loop meets. "
                        "They disagree by a lot."
                    ),
                },
                "copies": {
                    "type": "integer",
                    "description": "Repetitions inside the region (default 20).",
                },
                "label": {"type": "string", "description": "A name for the run."},
            },
            "required": ["pattern", "cpu"],
        },
    },
    {
        "name": "find_fusion_candidates",
        "description": (
            "Mine hot blocks for instruction sequences that could become one "
            "instruction. Builds the data-flow graph of each block, finds the "
            "connected groups a single opcode could encode -- convex, and "
            "within the operand budget -- and ranks them by how often the "
            "containing block actually executed. This is the step between "
            "profiling and proposing: it answers 'which sequences recur on "
            "the hot path', which reading disassembly by hand does not scale "
            "to. Feed it the blocks from profile_c_workload. A result is a "
            "claim that a shape is frequent, not that fusing it pays; cost it "
            "with analyze_snippet_cycles before proposing anything."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "blocks": {
                    # Either shape: the schema check is what refused a model
                    # that described its blocks in prose instead of omitting
                    # the field, before the handler could fall back to the
                    # profile this run had already taken.
                    "type": ["array", "string"],
                    "items": {"type": "object"},
                    "description": (
                        "Optional. If you have already run profile_c_workload "
                        "in this job, leave this out and its hot blocks are "
                        "used automatically -- do not copy the disassembly "
                        "across, since a truncated copy mines a different "
                        "program than the one profiled. Otherwise pass objects "
                        "with an `instructions` list of assembly lines and an "
                        "`executions` count."
                    ),
                },
                "max_instructions": {
                    "type": "integer",
                    "description": (
                        "Largest group to consider (default 3). Bigger groups "
                        "are harder to encode and rarer."
                    ),
                },
                "max_inputs": {
                    "type": "integer",
                    "description": (
                        "External registers the fused instruction may read "
                        "(default 2, the usual budget for a 32-bit encoding)."
                    ),
                },
                "max_outputs": {
                    "type": "integer",
                    "description": "Results it may write (default 1).",
                },
                "min_executions": {
                    "type": "integer",
                    "description": (
                        "Drop candidates whose blocks ran fewer times than " "this."
                    ),
                },
            },
            "required": [],
        },
    },
    {
        "name": "describe_model_parameters",
        "description": (
            "List the tunable parameters of a simulated core model and the "
            "exact paths that set them. Tuning a model is impossible without "
            "this: the per-op-class latencies live in a functional-unit pool "
            "whose layout differs per model, and the paths gem5 prints in its "
            "own config.ini are not the paths that can be assigned to. Returns "
            "op-class latencies (what a single-instruction benchmark "
            "constrains) separately from widths and queue depths (which only "
            "whole-kernel behaviour can pin down). Feed the `parameter` "
            "strings to simulate_c_workload's param_overrides with =<value> "
            "appended. Costs one short simulation, so call it once per model "
            "rather than per candidate."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "cpu_type": {
                    "type": "string",
                    "description": (
                        "Which model to inspect: O3CPU, MinorCPU, "
                        "TimingSimpleCPU, AtomicSimpleCPU, NeoverseV2, "
                        "O3_ARM_v7a_3, HPI, ex5_big or ex5_LITTLE."
                    ),
                },
                "op_classes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional filter, e.g. ['FloatSqrt','FloatDiv']. "
                        "Omit to see every op class the model defines."
                    ),
                },
            },
            "required": [],
        },
    },
    {
        "name": "simulate_c_workload",
        "description": (
            "Run a self-contained C program in a simulated out-of-order core "
            "with caches and a branch predictor, and report the cycles it "
            "took. This is the referee for a performance claim: "
            "analyze_snippet_cycles estimates how a sequence issues assuming a "
            "warm front end and no cache misses, while this executes it and "
            "measures. Simulation runs on the order of 100k instructions a "
            "second, so bring a kernel with a bounded input, and screen "
            "candidates with analyze_snippet_cycles first. Compare runs by "
            "cycles rather than sim_seconds."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Self-contained C program including main()",
                },
                "flags": {
                    "type": "string",
                    "description": (
                        "Compiler flags (default '-O3 -static'). Static linking "
                        "is added if absent: syscall-emulation mode has no "
                        "dynamic loader."
                    ),
                },
                "cpu_type": {
                    "type": "string",
                    "description": (
                        "Which core to model. Generic: O3CPU (out-of-order, the "
                        "one a timing claim needs), MinorCPU (in-order), "
                        "TimingSimpleCPU, AtomicSimpleCPU (no timing model at "
                        "all). Named ARM cores: NeoverseV2, O3_ARM_v7a_3, HPI, "
                        "ex5_big, ex5_LITTLE. The generic models carry gem5's "
                        "default latencies, which match no shipped silicon -- "
                        "measured against an Apple M3 host, O3CPU is 40% off "
                        "per instruction and NeoverseV2 77%, so name the core a "
                        "claim is about and calibrate it. NeoverseV2, ex5_big "
                        "and ex5_LITTLE have no functional unit for scalar "
                        "fused multiply-add: this tool refuses such workloads "
                        "rather than hanging, and -ffp-contract=off avoids it."
                    ),
                },
                "param_overrides": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Tune the core model for this run: full parameter "
                        "assignments such as "
                        "'system.cpu[0].instQueues[0].fuPool.FUList[3]"
                        ".opList[4].opLat=10' or 'system.cpu[0].issueWidth=6'. "
                        "Call describe_model_parameters to get the exact paths "
                        "a model exposes -- they are not guessable, and the "
                        "flattened names in gem5's config.ini cannot be "
                        "assigned to. This is how a model is calibrated "
                        "against measured silicon without forking a config."
                    ),
                },
                "run_args": {
                    "type": "string",
                    "description": "Arguments passed to the program (optional)",
                },
                "label": {
                    "type": "string",
                    "description": (
                        "Short name for this run, recorded with the "
                        "measurement so variants can be told apart."
                    ),
                },
            },
            "required": ["code"],
        },
    },
    {
        "name": "verify_run_bundle",
        "description": (
            "Check this run's evidence bundle, which is written as the run "
            "goes: every measurement call, its parameters, its result and the "
            "image it ran in. Without replay it confirms the artifacts are the "
            "ones this run produced. With replay=true it re-runs each recorded "
            "call and reports whether the same results come back, judging "
            "nothing that reports wall clock, since two honest runs of a "
            "benchmark disagree. Use it before claiming a result is "
            "reproducible."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "replay": {
                    "type": "boolean",
                    "description": (
                        "Re-run the recorded calls and compare (slow: it "
                        "repeats every measurement). Default false, which "
                        "checks integrity only."
                    ),
                },
            },
            "required": [],
        },
    },
    {
        "name": "record_prediction",
        "description": (
            "State what you expect a measurement to show, and how you reached "
            "that, BEFORE running the thing that measures it. This is what "
            "makes a methodology scoreable: the error between this number and "
            "what is measured is the score, and a prediction written after the "
            "outcome is known scores perfectly while teaching nothing. Returns "
            "a prediction_id to settle later with record_measurement."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "subject": {
                    "type": "string",
                    "description": "What this is about, e.g. 'fused ldp+fmla in saxpy loop'",
                },
                "metric": {
                    "type": "string",
                    "description": (
                        "The quantity, e.g. 'speedup' or 'cycles_per_iteration'. "
                        "Errors on different quantities cannot be compared."
                    ),
                },
                "predicted_value": {
                    "type": "number",
                    "description": "The number you expect the measurement to produce",
                },
                "methodology": {
                    "type": "string",
                    "description": (
                        "How you arrived at it. This is what is being scored, "
                        "so describe the approach, not just the answer."
                    ),
                },
                "methodology_tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Short tags for the approach, e.g. ['mca', 'sampled']. "
                        "Later runs group errors by these to see which "
                        "approach predicts well."
                    ),
                },
                "prediction_basis": {
                    "type": "string",
                    "description": "The evidence behind the number (optional)",
                },
                "derived_from": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Required. The finding types this prediction is "
                        "computed from, e.g. ['cycle_model_measurement']. Each "
                        "must already exist in this run or the call is "
                        "refused: a prediction citing a measurement it never "
                        "obtained is worse than no prediction. If this is a "
                        "judgement with no measurement behind it, pass "
                        "['none'] and explain in the methodology -- that is "
                        "recorded as such, and is honest where a silent guess "
                        "is not."
                    ),
                },
            },
            "required": [
                "subject",
                "metric",
                "predicted_value",
                "methodology",
                "derived_from",
            ],
        },
    },
    {
        "name": "record_measurement",
        "description": (
            "Settle a prediction with what was actually measured, naming the "
            "referee that produced it. A prediction can only be settled once, "
            "so the flattering measurement cannot be the one kept."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "prediction_id": {
                    "type": "string",
                    "description": "The UUID returned by record_prediction",
                },
                "measured_value": {
                    "type": "number",
                    "description": "What the measurement produced",
                },
                "measurement_source": {
                    "type": "string",
                    "description": (
                        "What produced it, e.g. 'gem5 O3 neoverse-n1' or "
                        "'wall clock, 5 trials'. A number without its source "
                        "cannot be compared with another."
                    ),
                },
                "notes": {"type": "string", "description": "Optional context"},
            },
            "required": ["prediction_id", "measured_value", "measurement_source"],
        },
    },
    {
        "name": "calibration_report",
        "description": (
            "Read how past predictions held up, overall and grouped by "
            "methodology tag. Use this before choosing an approach: it says "
            "which methods have been predicting well and which have not, "
            "including how many predictions were never checked."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "metric": {"type": "string", "description": "Filter by metric"},
                "subject": {"type": "string", "description": "Filter by subject"},
                "limit": {"type": "integer", "description": "Max rows (default 50)"},
            },
            "required": [],
        },
    },
    {
        "name": "axis_check",
        "description": (
            "Validate an AXIS architecture description (.axisl) of an "
            "instruction-set extension. AXIS is the source of truth for a "
            "proposal: one description elaborates into encoder, decoder, "
            "semantics, compiler patterns and SMT-LIB semantics, so a proposal "
            "is checkable and its artifacts are regenerable rather than "
            "hand-written per candidate. Run this before emitting anything."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "The AXIS description source (.axisl text)",
                },
            },
            "required": ["source"],
        },
    },
    {
        "name": "axis_emit",
        "description": (
            "Generate one artifact from an AXIS description: 'smt2' for formal "
            "semantics, 'decode-c'/'encode-c' to get the instruction into and "
            "out of a binary, 'semantics-c'/'sim-c' for a simulator, "
            "'golden-python' for a reference model, 'llvm-patterns'/'tablegen' "
            "for compiler support, 'pyrtl' for hardware. Note that the TableGen "
            "backend emits RISC-V instruction formats; check it before relying "
            "on it for another target."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "The AXIS description source (.axisl text)",
                },
                "target": {
                    "type": "string",
                    "description": (
                        "What to emit: json, bundle-manifest, legality-json, "
                        "encode-c, encode-json, decode-c, decode-json, "
                        "roundtrip-json, asm-disasm-json, semantics-c, "
                        "semantics-rust, semantics-json, sim-c, exec-c, "
                        "exec-python, golden-python, smt2, tablegen, llvm-ir, "
                        "llvm-patterns, llvm-intrinsics, intrinsics, pyrtl"
                    ),
                },
            },
            "required": ["source", "target"],
        },
    },
    {
        "name": "axis_prove",
        "description": (
            "Prove a claim about a proposed instruction against the formal "
            "semantics AXIS emits for it, using an SMT solver. This is the "
            "strongest gate available: a cycle count says a sequence is "
            "faster, a proof says the replacement computes the same thing for "
            "every input. Use it to show a fused instruction is equivalent to "
            "the sequence it replaces before spending simulation on it."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "The AXIS description source (.axisl text)",
                },
                "obligation": {
                    "type": "string",
                    "description": (
                        "SMT-LIB appended to the emitted semantics, so it can "
                        "call the generated functions by name. Assert the "
                        "NEGATION of your claim and end with (check-sat): "
                        "'unsat' then means no counterexample exists and the "
                        "claim holds for all inputs, while 'sat' returns a "
                        "counterexample showing the candidate is wrong."
                    ),
                },
            },
            "required": ["source", "obligation"],
        },
    },
    {
        "name": "analyze_snippet_cycles",
        "description": (
            "Cost a code sequence against a named core's scheduling model with "
            "llvm-mca, without running it: cycles per iteration, IPC, uops and "
            "block reciprocal throughput. Use this to compare two sequences "
            "that do the same work, and to cost a sequence no hardware here "
            "can run -- a proposed instruction, or a target that is not this "
            "host. These are modelled estimates, not measurements: mca assumes "
            "a warm front end and no cache misses."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": (
                        "C source to compile and analyse. A function is enough; "
                        "no main() is needed."
                    ),
                },
                "asm": {
                    "type": "string",
                    "description": (
                        "Assembly to analyse directly, instead of code. Use this "
                        "to cost a hypothetical sequence, such as one where an "
                        "idiom is replaced by the instruction being proposed. "
                        "If you intend to compare the estimate against a "
                        "measurement of a compiled program, this must be the "
                        "compiler's own output with your edit applied, never "
                        "assembly you wrote by hand: a run that hand-wrote a "
                        "plausible-looking loop estimated 36.26 cycles per "
                        "iteration where the code that actually ran cost 59.05, "
                        "and blamed the estimate rather than the substitution. "
                        "To cost a loop rather than a whole function, fence it "
                        "with '# LLVM-MCA-BEGIN name' and a bare '# LLVM-MCA-END' "
                        "(no name after END, or llvm-mca rejects it): the same "
                        "kernel measures 24.14 cycles as a function and 7.18 as "
                        "its inner loop. These markers are assembly comments and "
                        "must go in 'asm', never in 'code'."
                    ),
                },
                "cpu": {
                    "type": "string",
                    "description": (
                        "Required. The core model to cost against, e.g. "
                        "'neoverse-n1', 'cortex-a78', 'cortex-x2'. A cycle "
                        "count without a named model cannot be compared."
                    ),
                },
                "flags": {
                    "type": "string",
                    "description": "Compiler flags used when code is given (default -O3)",
                },
                "target": {
                    "type": "string",
                    "description": (
                        "Target TRIPLE, e.g. aarch64-linux-gnu (the default). "
                        "Not a core model and not a name for the run: the core "
                        "goes in 'cpu' and a name goes in 'label'. Cross-target "
                        "analysis works, since the code is never executed."
                    ),
                },
                "iterations": {
                    "type": "integer",
                    "description": "Iterations to simulate (default 100)",
                },
                "label": {
                    "type": "string",
                    "description": (
                        "Short name for what this sequence is. Recorded with the "
                        "estimate; without it several estimates cannot be told "
                        "apart afterwards."
                    ),
                },
            },
            "required": ["cpu"],
        },
    },
    {
        "name": "measure_predictability",
        "description": (
            "Measure how much signal the sampled hardware counters carry about "
            "a target counter's NEXT interval -- the ceiling on any predictor "
            "tapping them, established before designing one. Reads the trace "
            "from the sample_hardware_counters call this run already made; do "
            "not paste it back. The number that decides anything is "
            "'information beyond persistence': what a counter adds over simply "
            "predicting the same value as last interval. Programs run in "
            "phases, so almost every counter looks predictive until you ask "
            "what it contributes, and a predictor that cannot beat last-value "
            "is not worth building in hardware. A trace too short to estimate "
            "on is refused rather than answered."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "description": (
                        "Counter to predict, e.g. system.cpu.numCycles. Must be "
                        "present in the trace."
                    ),
                },
                "bins": {
                    "type": "integer",
                    "description": "Discretisation levels, default 3. More bins need a longer trace",
                },
                "from_interval": {
                    "type": "integer",
                    "description": (
                        "Ignore intervals before this one. Use it when "
                        "sample_hardware_counters warned that the trace "
                        "changes regime -- the intervals before the break "
                        "describe a machine that does not recur (a co-runner "
                        "still initialising, a cache still cold), and a number "
                        "taken across it largely measures the break rather "
                        "than the workload. Pass the interval the warning "
                        "names to study the steady side."
                    ),
                },
            },
            "required": ["target"],
        },
    },
    {
        "name": "select_counter_taps",
        "description": (
            "Which counters, TOGETHER, a predictor should tap -- and how many "
            "of them are worth the wires. Reads the trace this run already "
            "sampled. measure_predictability scores counters one at a time; "
            "this does greedy forward selection from persistence, stopping at "
            "the depth the trace can support, because conditioning on one more "
            "counter multiplies the cells the estimate needs by the bin count "
            "and a trace is hundreds of intervals, not millions. Every tap is "
            "placed against a null that runs the SAME selection on permuted "
            "counters, so the threshold contains the advantage that picking "
            "the best of fifty confers -- and each tap is judged on what IT "
            "added, not on the running total, because its own increment is "
            "what its area is being bought with. Use this once a single "
            "counter has shown signal, to decide whether a second tap is a "
            "design or a coincidence."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "description": (
                        "Counter to predict, e.g. derived.thread0_ipc. Must be "
                        "present in the trace."
                    ),
                },
                "bins": {
                    "type": "integer",
                    "description": (
                        "Discretisation levels, default 3. More bins cost "
                        "supported taps: the depth this trace allows falls as "
                        "the bin count rises."
                    ),
                },
                "from_interval": {
                    "type": "integer",
                    "description": (
                        "Ignore intervals before this one. Use it when "
                        "sample_hardware_counters warned that the trace "
                        "changes regime -- the intervals before the break "
                        "describe a machine that does not recur (a co-runner "
                        "still initialising, a cache still cold), and a number "
                        "taken across it largely measures the break rather "
                        "than the workload. Pass the interval the warning "
                        "names to study the steady side."
                    ),
                },
            },
            "required": ["target"],
        },
    },
    {
        "name": "evaluate_predictor_design",
        "description": (
            "Run the predictor a measurement indicated and score it against "
            "its own ceiling, on intervals it was not trained on. Use after "
            "select_counter_taps has named a tap. An information ceiling says "
            "what is available; this says what a table of saturating counters "
            "actually reaches, which is the number that decides whether to "
            "stop or to spend more. Reports a design's cost in bits of state "
            "next to what it gains OVER PREDICTING THE SAME AS LAST INTERVAL "
            "-- never over chance, which flatters every predictor on an "
            "autocorrelated target. The split is contiguous, never random, "
            "because adjacent intervals are near-identical and a random split "
            "puts each scored row's twin in the warm-up. If the cheap design "
            "captures most of its ceiling, a learned model is competing for "
            "the remainder and no training corpus needs generating to find "
            "that out."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "description": "Counter to predict, e.g. derived.thread0_ipc.",
                },
                "tap": {
                    "type": "string",
                    "description": (
                        "The counter the predictor reads alongside the target's "
                        "own last value. Normally the tap select_counter_taps "
                        "recommended."
                    ),
                },
                "bins": {
                    "type": "integer",
                    "description": "Discretisation levels, default 3.",
                },
                "split": {
                    "type": "number",
                    "description": (
                        "Fraction of the trace that warms the tables, default "
                        "0.5. The rest is scored and never trained on."
                    ),
                },
                "from_interval": {
                    "type": "integer",
                    "description": (
                        "Ignore intervals before this one. Use it when "
                        "sample_hardware_counters warned that the trace "
                        "changes regime -- the intervals before the break "
                        "describe a machine that does not recur (a co-runner "
                        "still initialising, a cache still cold), and a number "
                        "taken across it largely measures the break rather "
                        "than the workload. Pass the interval the warning "
                        "names to study the steady side."
                    ),
                },
            },
            "required": ["target", "tap"],
        },
    },
    {
        "name": "sample_hardware_counters",
        "description": (
            "Run a C workload in a simulated core and return its hardware "
            "counters SAMPLED OVER TIME, rather than as run totals. Call "
            "M5_SAMPLE() in the program wherever a sample should be taken -- "
            "typically once per outer-loop iteration or per phase -- and each "
            "interval reports what happened since the previous call. This is "
            "the shape a hardware predictor reads: run totals cannot train or "
            "evaluate one. The macro is injected; do not define it. Only "
            "counters that vary across the trace are returned, because a "
            "counter that never changes cannot predict anything that does."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": (
                        "Self-contained C program calling M5_SAMPLE() at each "
                        "sampling point. Fewer than 4 samples is a total, not "
                        "a trace, and the result says so."
                    ),
                },
                "cpu_type": {
                    "type": "string",
                    "description": "Core model to simulate; see describe_model_parameters",
                },
                "flags": {"type": "string", "description": "Compiler flags"},
                "label": {
                    "type": "string",
                    "description": "Short name for what this workload is",
                },
                "max_counters": {
                    "type": "integer",
                    "description": "How many varying counters to return (default 60)",
                },
            },
            "required": ["code"],
        },
    },
    {
        "name": "benchmark_c_snippet",
        "description": (
            "Compile and run a self-contained C program in the compiler "
            "research sandbox, returning its stdout and wall-clock time over "
            "repeated trials (minimum reported). The program must print its "
            "own measurements; there are no performance counters in the "
            "sandbox, and there is no network."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Self-contained C program including main()",
                },
                "flags": {
                    "type": "string",
                    "description": (
                        "Compiler flags. The sandbox targets aarch64: use "
                        "'-mcpu=native', not '-march=native'."
                    ),
                },
                "repeat": {
                    "type": "integer",
                    "description": "Trials to run, 1-10 (default 3); the fastest is reported",
                },
                "label": {
                    "type": "string",
                    "description": (
                        "Short name for what this snippet is, e.g. 'float sum "
                        "reduction'. Recorded with the measurement; without it "
                        "several measurements cannot be told apart afterwards."
                    ),
                },
            },
            "required": ["code"],
        },
    },
    {
        "name": "execute_python",
        "description": (
            "Run Python code in a RestrictedPython sandbox. Output via the "
            "'result' variable. Only whitelisted pure-Python modules are "
            "importable: there is no subprocess, no filesystem and no network, "
            "so this CANNOT compile code or invoke a toolchain. Use "
            "compile_c_snippet or benchmark_c_snippet for compiler work."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute"},
                "timeout_seconds": {
                    "type": "integer",
                    "description": "Execution timeout in seconds (max 30, default 10)",
                },
            },
            "required": ["code"],
        },
    },
    {
        "name": "execute_data_pipeline",
        "description": "Process data with pandas/numpy operations in a Docker sandbox. Pass data via 'input_data' variable, output via 'result' variable.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code using pandas/numpy for data processing",
                },
                "input_data": {
                    "type": "object",
                    "description": "Data to process (passed as dict to the code)",
                },
                "input_document_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Document IDs whose content should be loaded as input",
                },
                "timeout_seconds": {
                    "type": "integer",
                    "description": "Execution timeout (max 300, default 60)",
                },
            },
            "required": ["code"],
        },
    },
    {
        "name": "write_and_run_script",
        "description": "Write a Python script, execute it in a Docker sandbox, and return the results. For multi-file or complex logic.",
        "parameters": {
            "type": "object",
            "properties": {
                "script_name": {
                    "type": "string",
                    "description": "Filename for the script (e.g., 'analysis.py')",
                },
                "script_content": {
                    "type": "string",
                    "description": "Full script content",
                },
                "requirements": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "pip packages to install from whitelist (pandas, numpy, scipy, etc.)",
                },
                "arguments": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Command-line arguments to pass to the script",
                },
                "timeout_seconds": {
                    "type": "integer",
                    "description": "Execution timeout (max 300, default 120)",
                },
                "input_data": {
                    "type": "object",
                    "description": "Data passed as /workspace/input.json",
                },
            },
            "required": ["script_name", "script_content"],
        },
    },
    # ==================== Autonomous Coding Tools ====================
    {
        "name": "clone_and_index_repo",
        "description": "Clone a git repository into a temporary coding workspace and index its file tree. Returns a workspace_id for subsequent file operations.",
        "parameters": {
            "type": "object",
            "properties": {
                "source_id": {
                    "type": "string",
                    "description": "UUID of a git DocumentSource in KB (preferred)",
                },
                "repo_url": {
                    "type": "string",
                    "description": "Git clone URL (alternative to source_id, requires code execution enabled)",
                },
                "branch": {
                    "type": "string",
                    "description": "Branch to check out (default: main/master)",
                },
            },
            "required": [],
        },
    },
    {
        "name": "browse_repo_files",
        "description": "List files and directories in the coding workspace with optional glob filtering.",
        "parameters": {
            "type": "object",
            "properties": {
                "workspace_id": {
                    "type": "string",
                    "description": "Workspace ID (optional if only one workspace)",
                },
                "path": {
                    "type": "string",
                    "description": "Directory path to list (default: root)",
                    "default": ".",
                },
                "glob_pattern": {
                    "type": "string",
                    "description": "Glob pattern to filter (e.g., '**/*.py')",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max files to return (default 200)",
                    "default": 200,
                },
            },
            "required": [],
        },
    },
    {
        "name": "read_file",
        "description": "Read file contents from the coding workspace, optionally limited to a line range.",
        "parameters": {
            "type": "object",
            "properties": {
                "workspace_id": {
                    "type": "string",
                    "description": "Workspace ID (optional if only one workspace)",
                },
                "path": {
                    "type": "string",
                    "description": "Relative file path in workspace",
                },
                "start_line": {
                    "type": "integer",
                    "description": "Start line (1-based, optional)",
                },
                "end_line": {
                    "type": "integer",
                    "description": "End line (1-based, optional)",
                },
                "max_chars": {
                    "type": "integer",
                    "description": "Max chars to return (default 20000)",
                    "default": 20000,
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write or overwrite a file in the coding workspace.",
        "parameters": {
            "type": "object",
            "properties": {
                "workspace_id": {
                    "type": "string",
                    "description": "Workspace ID (optional if only one workspace)",
                },
                "path": {"type": "string", "description": "Relative file path"},
                "content": {
                    "type": "string",
                    "description": "Full file content to write",
                },
                "create_dirs": {
                    "type": "boolean",
                    "description": "Create parent directories if missing (default true)",
                    "default": True,
                },
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "apply_patch",
        "description": "Apply a unified diff to files in the coding workspace. Supports fuzzy hunk matching.",
        "parameters": {
            "type": "object",
            "properties": {
                "workspace_id": {
                    "type": "string",
                    "description": "Workspace ID (optional if only one workspace)",
                },
                "diff": {"type": "string", "description": "Unified diff text"},
                "dry_run": {
                    "type": "boolean",
                    "description": "Validate patch without applying (default false)",
                    "default": False,
                },
            },
            "required": ["diff"],
        },
    },
    {
        "name": "run_command",
        "description": "Run a shell command in the coding workspace. Gated by unsafe_code_execution_enabled feature flag.",
        "parameters": {
            "type": "object",
            "properties": {
                "workspace_id": {
                    "type": "string",
                    "description": "Workspace ID (optional if only one workspace)",
                },
                "command": {
                    "type": "string",
                    "description": "Shell command to execute",
                },
                "timeout_seconds": {
                    "type": "integer",
                    "description": "Execution timeout in seconds (max 120, default 30)",
                    "default": 30,
                },
                "env": {
                    "type": "object",
                    "description": "Additional environment variables",
                },
            },
            "required": ["command"],
        },
    },
    {
        "name": "search_code",
        "description": "Search for text patterns in workspace files using regex (grep-like).",
        "parameters": {
            "type": "object",
            "properties": {
                "workspace_id": {
                    "type": "string",
                    "description": "Workspace ID (optional if only one workspace)",
                },
                "pattern": {
                    "type": "string",
                    "description": "Regex pattern to search for",
                },
                "path": {
                    "type": "string",
                    "description": "Subdirectory to search in (default: root)",
                    "default": ".",
                },
                "file_glob": {
                    "type": "string",
                    "description": "Glob to limit files (e.g., '*.py')",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max matches to return (default 50)",
                    "default": 50,
                },
                "context_lines": {
                    "type": "integer",
                    "description": "Lines of context around each match (default 2)",
                    "default": 2,
                },
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "get_workspace_status",
        "description": "Show modified, added, and deleted files in the coding workspace compared to original state.",
        "parameters": {
            "type": "object",
            "properties": {
                "workspace_id": {
                    "type": "string",
                    "description": "Workspace ID (optional if only one workspace)",
                },
                "show_diff_summary": {
                    "type": "boolean",
                    "description": "Include change statistics (default true)",
                    "default": True,
                },
            },
            "required": [],
        },
    },
    {
        "name": "create_workspace_checkpoint",
        "description": "Create a bounded recovery checkpoint of the active coding workspace before risky edits.",
        "parameters": {
            "type": "object",
            "properties": {
                "workspace_id": {
                    "type": "string",
                    "description": "Workspace ID (optional if only one workspace)",
                },
                "label": {
                    "type": "string",
                    "description": "Short reason for the checkpoint",
                },
            },
            "required": [],
        },
    },
    {
        "name": "list_workspace_checkpoints",
        "description": "List available recovery checkpoints for the active coding workspace.",
        "parameters": {
            "type": "object",
            "properties": {
                "workspace_id": {
                    "type": "string",
                    "description": "Workspace ID (optional if only one workspace)",
                },
            },
            "required": [],
        },
    },
    {
        "name": "restore_workspace_checkpoint",
        "description": "Restore a recovery checkpoint. By default, first preserves the current workspace as another checkpoint.",
        "parameters": {
            "type": "object",
            "properties": {
                "workspace_id": {
                    "type": "string",
                    "description": "Workspace ID (optional if only one workspace)",
                },
                "checkpoint_id": {
                    "type": "string",
                    "description": "Checkpoint ID returned by create/list checkpoint",
                },
                "preserve_current": {
                    "type": "boolean",
                    "description": "Checkpoint the current state before restore (default true)",
                    "default": True,
                },
            },
            "required": ["checkpoint_id"],
        },
    },
    {
        "name": "hydrate_candidate_snapshot",
        "description": "Load the system-provided immutable candidate snapshot into the active workspace after verifying its baseline and file hashes.",
        "parameters": {
            "type": "object",
            "properties": {
                "workspace_id": {
                    "type": "string",
                    "description": "Workspace ID (optional if only one workspace)",
                },
                "snapshot_id": {
                    "type": "string",
                    "description": "Candidate snapshot ID when multiple system-provided candidates are available",
                },
            },
            "required": [],
        },
    },
    {
        "name": "persist_durable_workspace_checkpoint",
        "description": "Persist the active mutation-owner workspace as an immutable restart-safe session checkpoint.",
        "parameters": {
            "type": "object",
            "properties": {
                "workspace_id": {
                    "type": "string",
                    "description": "Workspace ID (optional if only one workspace)",
                },
                "label": {
                    "type": "string",
                    "description": "Short reason for preserving this state",
                },
            },
            "required": [],
        },
    },
    {
        "name": "list_durable_workspace_checkpoints",
        "description": "List restart-safe checkpoints bound to the current coding session.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "restore_durable_workspace_checkpoint",
        "description": "Restore a restart-safe checkpoint belonging to the current coding session into a clean reconstructed workspace.",
        "parameters": {
            "type": "object",
            "properties": {
                "checkpoint_id": {
                    "type": "string",
                    "description": "Durable checkpoint ID from the current job session",
                },
            },
            "required": ["checkpoint_id"],
        },
    },
    # ==================== Document Authoring Tools ====================
    {
        "name": "plan_document",
        "description": "Create a structured document outline with sections and subsections. Stores the plan in document workspace state for subsequent write_section calls.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Document title"},
                "abstract": {
                    "type": "string",
                    "description": "Document abstract or summary",
                },
                "doc_type": {
                    "type": "string",
                    "enum": [
                        "research_report",
                        "technical_doc",
                        "executive_brief",
                        "design_doc",
                        "tutorial",
                        "decision_memo",
                    ],
                    "description": "Document type (default: research_report)",
                    "default": "research_report",
                },
                "sections": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {
                                "type": "string",
                                "description": "Section identifier",
                            },
                            "title": {"type": "string", "description": "Section title"},
                            "description": {
                                "type": "string",
                                "description": "What this section should cover",
                            },
                        },
                        "required": ["id", "title"],
                    },
                    "description": "List of document sections",
                },
                "style": {
                    "type": "string",
                    "enum": ["academic", "professional", "technical", "informal"],
                    "description": "Writing style (default: professional)",
                    "default": "professional",
                },
            },
            "required": ["title", "sections"],
        },
    },
    {
        "name": "write_section",
        "description": "Write content for a specific document section, optionally using RAG search for KB context and citations.",
        "parameters": {
            "type": "object",
            "properties": {
                "section_id": {
                    "type": "string",
                    "description": "Section ID from the document plan",
                },
                "content": {
                    "type": "string",
                    "description": "Section content in markdown",
                },
                "search_query": {
                    "type": "string",
                    "description": "Optional query to search KB for relevant context before writing",
                },
                "citations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "ref_id": {
                                "type": "string",
                                "description": "Citation reference ID (e.g., [1])",
                            },
                            "document_id": {
                                "type": "string",
                                "description": "KB document UUID",
                            },
                            "title": {"type": "string", "description": "Source title"},
                            "excerpt": {
                                "type": "string",
                                "description": "Relevant excerpt from source",
                            },
                        },
                        "required": ["ref_id", "document_id", "title"],
                    },
                    "description": "Citations for this section",
                },
            },
            "required": ["section_id", "content"],
        },
    },
    {
        "name": "revise_section",
        "description": "Rewrite a previously written document section with specific feedback or corrections.",
        "parameters": {
            "type": "object",
            "properties": {
                "section_id": {"type": "string", "description": "Section ID to revise"},
                "feedback": {
                    "type": "string",
                    "description": "Specific feedback or revision instructions",
                },
                "new_content": {
                    "type": "string",
                    "description": "Revised section content in markdown",
                },
                "additional_citations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "ref_id": {"type": "string"},
                            "document_id": {"type": "string"},
                            "title": {"type": "string"},
                            "excerpt": {"type": "string"},
                        },
                    },
                    "description": "Additional citations to add",
                },
            },
            "required": ["section_id", "new_content"],
        },
    },
    {
        "name": "assemble_document",
        "description": "Combine all written sections into a final document with table of contents and references.",
        "parameters": {
            "type": "object",
            "properties": {
                "include_toc": {
                    "type": "boolean",
                    "description": "Include table of contents (default true)",
                    "default": True,
                },
                "include_references": {
                    "type": "boolean",
                    "description": "Include references section (default true)",
                    "default": True,
                },
                "include_abstract": {
                    "type": "boolean",
                    "description": "Include abstract (default true)",
                    "default": True,
                },
                "section_order": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Custom section ordering by ID (optional, uses plan order by default)",
                },
            },
            "required": [],
        },
    },
    {
        "name": "export_document",
        "description": "Export the assembled document to DOCX, PDF, PPTX, or LaTeX format. Optionally persist to the knowledge base.",
        "parameters": {
            "type": "object",
            "properties": {
                "format": {
                    "type": "string",
                    "enum": ["docx", "pdf", "pptx", "latex"],
                    "description": "Export format",
                },
                "persist_to_kb": {
                    "type": "boolean",
                    "description": "Also save the document to the knowledge base (default false)",
                    "default": False,
                },
                "latex_project_id": {
                    "type": "string",
                    "description": "Existing LaTeX project to export into (latex format only)",
                },
            },
            "required": ["format"],
        },
    },
    {
        "name": "insert_figure",
        "description": "Insert a chart, table, or diagram into a document section.",
        "parameters": {
            "type": "object",
            "properties": {
                "section_id": {
                    "type": "string",
                    "description": "Section to insert figure into",
                },
                "figure_type": {
                    "type": "string",
                    "enum": ["chart", "table", "diagram", "flowchart"],
                    "description": "Type of figure",
                },
                "caption": {"type": "string", "description": "Figure caption"},
                "data": {
                    "type": "object",
                    "description": "Data for chart or table generation",
                },
                "diagram_spec": {
                    "type": "string",
                    "description": "Mermaid or PlantUML spec for diagram generation",
                },
            },
            "required": ["section_id", "figure_type", "caption"],
        },
    },
    # ==================== Workspace Artifact Retrieval ====================
    {
        "name": "get_workspace_artifact_url",
        "description": "Get a download URL for a file persisted from a previous job's coding workspace. Use this to access code or documents saved by earlier jobs.",
        "parameters": {
            "type": "object",
            "properties": {
                "job_id": {
                    "type": "string",
                    "description": "The job ID whose workspace was persisted",
                },
                "file_path": {
                    "type": "string",
                    "description": "Relative file path within the persisted workspace",
                },
            },
            "required": ["job_id", "file_path"],
        },
    },
    # ==================== Memory Tools ====================
    {
        "name": "record_method",
        "description": (
            "Record how to investigate something, with the evidence that it "
            "works, so later jobs inherit it. Findings say what you learned "
            "about the subject; this is for what you learned about method -- "
            "the part that transfers to a different subject. Record one when "
            "an approach turned out to be necessary, when an obvious approach "
            "produced a wrong answer, or when you found a check that catches a "
            "class of mistake. You must name the finding types in this run "
            "that establish it: a method claimed without evidence can only be "
            "stored by passing derived_from=['none'], and is marked unvalidated."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": (
                        "Short name, e.g. 'measure instruction latency with "
                        "inline-asm dependent chains'."
                    ),
                },
                "procedure": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "The steps in order, concrete enough that another run "
                        "can follow them without rediscovering the method."
                    ),
                },
                "prevents": {
                    "type": "string",
                    "description": (
                        "The wrong answer this exists to stop, stated "
                        "specifically. This is what tells a future reader "
                        "whether their situation is the same one, so 'improves "
                        "accuracy' is not an answer -- 'the compiler vectorises "
                        "a C loop so the measurement is of different code' is."
                    ),
                },
                "derived_from": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Finding types produced in THIS run that establish the "
                        "method. Refused if no such finding exists. Pass "
                        "['none'] to record an untested method, which is stored "
                        "as unvalidated."
                    ),
                },
                "applies_to": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "What it applies to -- tools, domains or workload kinds "
                        "-- so it is recalled by jobs that need it."
                    ),
                },
                "limits": {
                    "type": "string",
                    "description": (
                        "Where it stops working, and what would falsify it."
                    ),
                },
                "builds_on": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Names of methods recalled into this run that you "
                        "actually followed. Saying so is what lets a method "
                        "earn a track record: a method merely present in your "
                        "context is weak evidence about this run, and one you "
                        "name is strong."
                    ),
                },
            },
            "required": ["name", "procedure", "prevents", "derived_from"],
        },
    },
    {
        "name": "create_memory",
        "description": "Store a persistent memory for the current user. Use this to save important facts, insights, decisions, or context that should be available to future jobs.",
        "parameters": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "Memory content (concise, factual, and actionable)",
                },
                "importance": {
                    "type": "number",
                    "description": "Importance score 0.0-1.0 (default 0.5)",
                },
                "category": {
                    "type": "string",
                    "enum": [
                        "fact",
                        "preference",
                        "context",
                        "summary",
                        "goal",
                        "constraint",
                    ],
                    "description": "Memory category (default: fact)",
                },
                "metadata": {
                    "type": "object",
                    "description": "Optional metadata key-value pairs",
                },
            },
            "required": ["content"],
        },
    },
    {
        "name": "search_memories",
        "description": "Search the user's stored memories using semantic similarity. Returns ranked results matching the query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query for finding relevant memories",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results (default 10, max 50)",
                },
                "category_filter": {
                    "type": "string",
                    "enum": [
                        "fact",
                        "preference",
                        "context",
                        "summary",
                        "goal",
                        "constraint",
                    ],
                    "description": "Filter by memory category (optional)",
                },
                "min_importance": {
                    "type": "number",
                    "description": "Minimum importance score filter (0.0-1.0)",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "recall_memories",
        "description": "Recall memories related to a topic using broad semantic matching. Similar to search_memories but without filters, useful for open-ended context gathering.",
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "Topic to recall memories about",
                },
                "limit": {"type": "integer", "description": "Max results (default 10)"},
            },
            "required": ["topic"],
        },
    },
    {
        "name": "get_memory_stats",
        "description": "Get statistics about the user's memory store including counts by type, recent activity, and most accessed memories.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    # ==================== Symbol-aware Code Retrieval ====================
    {
        "name": "retrieve_repo_symbols",
        "description": "Search for code symbols (functions, classes, methods) in the coding workspace. Returns ranked matches with file locations and line numbers.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query (function names, class names, keywords)",
                },
                "workspace_id": {
                    "type": "string",
                    "description": "Workspace ID (optional, uses active workspace)",
                },
                "language_filter": {
                    "type": "string",
                    "enum": ["python", "typescript", "javascript"],
                    "description": "Filter by programming language (optional)",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max symbols to return (default 20)",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_symbol_context",
        "description": "Get a symbol's full definition, surrounding code context, and related symbols in the same file.",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol_name": {
                    "type": "string",
                    "description": "Name of the symbol (function, class, method)",
                },
                "file_path": {
                    "type": "string",
                    "description": "File path containing the symbol",
                },
                "workspace_id": {
                    "type": "string",
                    "description": "Workspace ID (optional, uses active workspace)",
                },
            },
            "required": ["symbol_name", "file_path"],
        },
    },
    {
        "name": "find_tests_for_symbol",
        "description": "Find test files and test functions that reference or cover a given code symbol.",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol_name": {
                    "type": "string",
                    "description": "Name of the symbol to find tests for",
                },
                "workspace_id": {
                    "type": "string",
                    "description": "Workspace ID (optional, uses active workspace)",
                },
            },
            "required": ["symbol_name"],
        },
    },
    # ── Workflow orchestration tools ──────────────────────────────
    {
        "name": "list_available_workflows",
        "description": "List DAG workflows available to the current user. Returns workflow IDs, names, and descriptions.",
        "parameters": {
            "type": "object",
            "properties": {
                "is_active": {
                    "type": "boolean",
                    "description": "Filter by active status (default true)",
                },
            },
            "required": [],
        },
    },
    {
        "name": "execute_workflow",
        "description": "Launch a DAG workflow by ID. The workflow executes its node graph and returns an execution ID for status tracking.",
        "parameters": {
            "type": "object",
            "properties": {
                "workflow_id": {
                    "type": "string",
                    "description": "UUID of the workflow to execute",
                },
                "trigger_data": {
                    "type": "object",
                    "description": "Data to pass as trigger context",
                },
                "inputs": {
                    "type": "object",
                    "description": "Initial context variables for the workflow",
                },
            },
            "required": ["workflow_id"],
        },
    },
    {
        "name": "get_workflow_status",
        "description": "Check the status of a workflow execution by its execution ID.",
        "parameters": {
            "type": "object",
            "properties": {
                "execution_id": {
                    "type": "string",
                    "description": "UUID of the workflow execution to check",
                },
            },
            "required": ["execution_id"],
        },
    },
    {
        "name": "enqueue_external_agent_call",
        "description": (
            "Durably enqueue a capability-scoped call to a configured external "
            "agent through the transactional outbox. Delivery occurs only after "
            "the current agent checkpoint commits."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "tool_id": {
                    "type": "string",
                    "description": "UUID of an enabled external-agent connection",
                },
                "capability": {
                    "type": "string",
                    "description": "Capability declared by the connection manifest",
                },
                "payload": {
                    "type": "object",
                    "description": "Bounded JSON request payload",
                },
                "idempotency_key": {
                    "type": "string",
                    "description": (
                        "Optional stable key; the current journal key is used by "
                        "default"
                    ),
                },
                "max_attempts": {
                    "type": "integer",
                    "description": "Delivery attempts before dead-lettering (1-8)",
                },
            },
            "required": ["tool_id", "capability", "payload"],
        },
    },
    {
        "name": "get_external_call_status",
        "description": (
            "Read delivery, retry, dead-letter, or response state for an "
            "external-agent outbox request created by this job."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "outbox_id": {
                    "type": "string",
                    "description": "UUID returned by enqueue_external_agent_call",
                },
            },
            "required": ["outbox_id"],
        },
    },
    # ── Agent-to-agent communication tools ────────────────────────
    {
        "name": "send_message_to_agent",
        "description": "Send a message to another agent job. The target agent can read it via read_agent_messages. Works across any jobs owned by the same user.",
        "parameters": {
            "type": "object",
            "properties": {
                "target_job_id": {
                    "type": "string",
                    "description": "UUID of the target agent job",
                },
                "message": {"type": "string", "description": "Message content to send"},
                "category": {
                    "type": "string",
                    "description": "Optional category tag (e.g. 'question', 'finding', 'request')",
                },
            },
            "required": ["target_job_id", "message"],
        },
    },
    {
        "name": "read_agent_messages",
        "description": "Read messages sent to this agent by other agent jobs. Returns messages from the specified index onward.",
        "parameters": {
            "type": "object",
            "properties": {
                "since_index": {
                    "type": "integer",
                    "description": "Start reading from this message index (default 0)",
                },
            },
            "required": [],
        },
    },
    # ── Research tools ────────────────────────────────────────────
    {
        "name": "search_web",
        "description": "Search the web using DuckDuckGo. Returns titles, URLs, and snippets for the top results.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "max_results": {
                    "type": "integer",
                    "description": "Maximum results to return (default 5, max 10)",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "summarize_url",
        "description": "Fetch content from a URL and produce an LLM-generated summary. Optionally focus the summary on a specific topic.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "URL to fetch and summarize"},
                "focus": {
                    "type": "string",
                    "description": "Optional focus topic for the summary",
                },
            },
            "required": ["url"],
        },
    },
    {
        "name": "fetch_url_content",
        "description": "Fetch and extract text content from a URL. Returns raw text without summarization.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "URL to fetch"},
                "max_chars": {
                    "type": "integer",
                    "description": "Maximum characters to return (default 50000)",
                },
            },
            "required": ["url"],
        },
    },
    # ── Notification/alerting tools ───────────────────────────────
    {
        "name": "send_notification",
        "description": "Send an in-app notification to the job owner. Delivered via WebSocket push and visible in the notification bell.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Notification title"},
                "message": {
                    "type": "string",
                    "description": "Notification message body",
                },
                "priority": {
                    "type": "string",
                    "description": "Priority level: low, normal, high, urgent (default normal)",
                },
                "action_url": {
                    "type": "string",
                    "description": "Optional URL for click-through action",
                },
            },
            "required": ["title", "message"],
        },
    },
    {
        "name": "send_email_alert",
        "description": "Send an email alert to the job owner. Falls back to in-app notification if SMTP is not configured.",
        "parameters": {
            "type": "object",
            "properties": {
                "subject": {"type": "string", "description": "Email subject line"},
                "body": {"type": "string", "description": "Email body text"},
                "priority": {
                    "type": "string",
                    "description": "Priority level: low, normal, high, urgent (default normal)",
                },
            },
            "required": ["subject", "body"],
        },
    },
    # ── Data Visualization Tools ──────────────────────────────────────
    {
        "name": "create_chart",
        "description": "Generate a data chart (bar, line, pie, scatter, histogram, heatmap, box, area) from structured data. The chart is rendered as an image and persisted to storage. Returns a download URL.",
        "parameters": {
            "type": "object",
            "properties": {
                "chart_type": {
                    "type": "string",
                    "description": "Chart type: bar, line, pie, scatter, histogram, heatmap, box, or area",
                },
                "data": {
                    "type": "object",
                    "description": "Chart data object. For most charts: {labels: [...], values: [...]} or {labels: [...], datasets: [{label, values}, ...]}. For heatmap: {labels: [...], matrix: [[...]]}. For scatter: {points: [{x, y}, ...]}",
                },
                "title": {"type": "string", "description": "Chart title (optional)"},
                "x_label": {"type": "string", "description": "X-axis label (optional)"},
                "y_label": {"type": "string", "description": "Y-axis label (optional)"},
                "format": {
                    "type": "string",
                    "description": "Output format: png or svg (default png)",
                },
            },
            "required": ["chart_type", "data"],
        },
    },
    {
        "name": "render_diagram",
        "description": "Render diagram source code (Mermaid or Graphviz) to an image and persist to storage. Returns a download URL. Use this after generate_diagram to produce a viewable image.",
        "parameters": {
            "type": "object",
            "properties": {
                "diagram_code": {
                    "type": "string",
                    "description": "The diagram source code (Mermaid or Graphviz DOT syntax)",
                },
                "diagram_type": {
                    "type": "string",
                    "description": "Diagram language: mermaid or graphviz (default mermaid)",
                },
                "format": {
                    "type": "string",
                    "description": "Output format: png or svg (default png)",
                },
            },
            "required": ["diagram_code"],
        },
    },
    # ── Knowledge Graph Tools ─────────────────────────────────────────
    {
        "name": "query_kg_entities",
        "description": "Search for entities in the knowledge graph by name or keyword. Returns matching entities with their types and descriptions.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query to match against entity names and descriptions",
                },
                "entity_type": {
                    "type": "string",
                    "description": "Filter by entity type (e.g. person, org, location, product)",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results (default 20, max 100)",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_entity_context",
        "description": "Get a knowledge graph entity with all its relationships and connected entities. Useful for understanding how an entity relates to others.",
        "parameters": {
            "type": "object",
            "properties": {
                "entity_id": {
                    "type": "string",
                    "description": "UUID of the entity to get context for",
                },
            },
            "required": ["entity_id"],
        },
    },
    {
        "name": "create_kg_entity",
        "description": "Create a new entity in the knowledge graph. Use this to add discovered concepts, people, organizations, or other entities during research.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Canonical name for the entity",
                },
                "entity_type": {
                    "type": "string",
                    "description": "Entity type: person, org, location, product, concept, technology, event, or other",
                },
                "description": {
                    "type": "string",
                    "description": "Brief description of the entity",
                },
            },
            "required": ["name", "entity_type"],
        },
    },
    {
        "name": "create_kg_relationship",
        "description": "Create a relationship between two entities in the knowledge graph. Both entities must already exist.",
        "parameters": {
            "type": "object",
            "properties": {
                "source_entity_id": {
                    "type": "string",
                    "description": "UUID of the source entity",
                },
                "target_entity_id": {
                    "type": "string",
                    "description": "UUID of the target entity",
                },
                "relation_type": {
                    "type": "string",
                    "description": "Type of relationship (e.g. works_at, authored, related_to, part_of)",
                },
                "confidence": {
                    "type": "number",
                    "description": "Confidence score 0.0-1.0 (default 0.8)",
                },
                "evidence": {
                    "type": "string",
                    "description": "Evidence or reason for this relationship",
                },
            },
            "required": ["source_entity_id", "target_entity_id", "relation_type"],
        },
    },
    {
        "name": "query_kg_graph",
        "description": "Query the global knowledge graph with filters. Returns nodes and edges for visualization or analysis. Useful for exploring the broader knowledge structure.",
        "parameters": {
            "type": "object",
            "properties": {
                "entity_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": 'Filter by entity types (e.g. ["person", "org"])',
                },
                "relation_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Filter by relationship types",
                },
                "min_confidence": {
                    "type": "number",
                    "description": "Minimum confidence threshold (0.0-1.0)",
                },
                "search": {
                    "type": "string",
                    "description": "Text search filter on entity names",
                },
                "limit_nodes": {
                    "type": "integer",
                    "description": "Maximum number of nodes to return (default 50, max 200)",
                },
            },
            "required": [],
        },
    },
    # ── Scheduling Tools ──────────────────────────────────────────────
    {
        "name": "schedule_job",
        "description": "Schedule a new agent job for future execution. Supports one-time runs at a specific datetime or recurring runs with a cron expression. The scheduled job will be picked up automatically by the scheduler.",
        "parameters": {
            "type": "object",
            "properties": {
                "goal": {
                    "type": "string",
                    "description": "The goal/task description for the scheduled job",
                },
                "job_type": {
                    "type": "string",
                    "description": "Job type: research, monitor, analysis, synthesis, coding (default research)",
                },
                "schedule_type": {
                    "type": "string",
                    "description": "Schedule type: once (run at specific time) or recurring (cron-based)",
                },
                "run_at": {
                    "type": "string",
                    "description": "ISO datetime for one-time execution (required if schedule_type=once)",
                },
                "cron": {
                    "type": "string",
                    "description": "Cron expression for recurring execution (required if schedule_type=recurring), e.g. '0 9 * * 1' for every Monday at 9am",
                },
                "config": {
                    "type": "object",
                    "description": "Optional job configuration (max_iterations, tool overrides, etc.)",
                },
            },
            "required": ["goal", "schedule_type"],
        },
    },
    {
        "name": "cancel_scheduled_job",
        "description": "Cancel a scheduled or recurring agent job. Prevents future executions and marks the job as cancelled.",
        "parameters": {
            "type": "object",
            "properties": {
                "job_id": {
                    "type": "string",
                    "description": "UUID of the scheduled job to cancel",
                },
            },
            "required": ["job_id"],
        },
    },
    # ── Document Authoring Enhancements ───────────────────────────────
    {
        "name": "list_documents_by_tag",
        "description": "List documents matching specified tags. Supports matching any tag (OR) or all tags (AND). Useful for finding related documents for synthesis or analysis.",
        "parameters": {
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tags to search for",
                },
                "match_all": {
                    "type": "boolean",
                    "description": "If true, documents must have ALL specified tags (AND). If false (default), documents matching ANY tag are returned (OR)",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results (default 20, max 100)",
                },
            },
            "required": ["tags"],
        },
    },
    {
        "name": "merge_documents",
        "description": "Merge content from multiple documents into a single new document. Each source document becomes a section with its title as heading. Useful for creating comprehensive reports from multiple sources.",
        "parameters": {
            "type": "object",
            "properties": {
                "document_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "UUIDs of documents to merge (max 20)",
                },
                "title": {
                    "type": "string",
                    "description": "Title for the merged document",
                },
                "separator": {
                    "type": "string",
                    "description": "Separator between document sections (default '\\n\\n---\\n\\n')",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tags for the merged document",
                },
            },
            "required": ["document_ids", "title"],
        },
    },
    # ── Agent Self-Reflection Tools ───────────────────────────────────
    {
        "name": "get_job_history",
        "description": "Query past agent job runs for the same user. Useful for learning from previous attempts, avoiding repeated mistakes, and understanding what has been done before.",
        "parameters": {
            "type": "object",
            "properties": {
                "job_type": {
                    "type": "string",
                    "description": "Filter by job type (research, monitor, analysis, synthesis, coding)",
                },
                "status": {
                    "type": "string",
                    "description": "Filter by status (completed, failed, cancelled)",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results (default 10, max 50)",
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_job_metrics",
        "description": "Get detailed metrics for a specific job including resource usage, timing, error rates, and per-tool usage breakdown. Defaults to the current job if no job_id is provided.",
        "parameters": {
            "type": "object",
            "properties": {
                "job_id": {
                    "type": "string",
                    "description": "UUID of the job to get metrics for (defaults to current job)",
                },
            },
            "required": [],
        },
    },
    # ── Tool Usage Analytics ──────────────────────────────────────────
    {
        "name": "get_tool_usage_stats",
        "description": "Get aggregated tool usage statistics across recent jobs. Shows which tools are used most, success/failure rates, and trends. Useful for optimizing tool selection strategies.",
        "parameters": {
            "type": "object",
            "properties": {
                "days": {
                    "type": "integer",
                    "description": "Number of days to analyze (default 7, max 30)",
                },
                "tool_name": {
                    "type": "string",
                    "description": "Filter to a specific tool (optional)",
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_tool_failure_analysis",
        "description": "Analyze failure patterns for a specific tool. Groups errors by pattern, shows frequency and examples. Useful for understanding why a tool is failing and how to work around issues.",
        "parameters": {
            "type": "object",
            "properties": {
                "tool_name": {
                    "type": "string",
                    "description": "Name of the tool to analyze failures for",
                },
                "days": {
                    "type": "integer",
                    "description": "Number of days to analyze (default 7, max 30)",
                },
            },
            "required": ["tool_name"],
        },
    },
    # ── Batch Processing Tools ────────────────────────────────────────
    {
        "name": "batch_search",
        "description": "Run multiple search queries against the knowledge base in a single call. Returns results grouped by query with optional deduplication. Much more efficient than calling search_documents multiple times.",
        "parameters": {
            "type": "object",
            "properties": {
                "queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of search queries to execute (max 10)",
                },
                "limit_per_query": {
                    "type": "integer",
                    "description": "Maximum results per query (default 5, max 20)",
                },
                "source_id": {
                    "type": "string",
                    "description": "Optional source ID to filter results",
                },
                "deduplicate": {
                    "type": "boolean",
                    "description": "Remove duplicate documents across queries (default true)",
                },
            },
            "required": ["queries"],
        },
    },
    {
        "name": "batch_summarize",
        "description": "Get summaries for multiple documents in a single call. Returns existing pre-generated summaries immediately. Use generate_missing=true to generate summaries for documents that don't have one (slower, uses LLM).",
        "parameters": {
            "type": "object",
            "properties": {
                "document_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "UUIDs of documents to summarize (max 20)",
                },
                "generate_missing": {
                    "type": "boolean",
                    "description": "If true, generate summaries for documents that lack one (default false)",
                },
            },
            "required": ["document_ids"],
        },
    },
    # ── Conditional Execution Tools ──
    {
        "name": "evaluate_condition",
        "description": "Evaluate a structured condition against current job state. Returns a boolean result with context data. Use to check findings count, category presence, document count, search coverage, action count, or progress level before deciding next steps.",
        "parameters": {
            "type": "object",
            "properties": {
                "condition": {
                    "type": "string",
                    "enum": [
                        "findings_count",
                        "findings_has_category",
                        "documents_count",
                        "search_has_results",
                        "actions_count",
                        "progress_above",
                    ],
                    "description": "The condition type to evaluate",
                },
                "threshold": {
                    "type": "integer",
                    "description": "Minimum value for the condition to be met (default 1)",
                },
                "category": {
                    "type": "string",
                    "description": "Finding category to check (for findings_has_category)",
                },
                "query": {
                    "type": "string",
                    "description": "Search query to test (for search_has_results)",
                },
                "source_id": {
                    "type": "string",
                    "description": "Optional source ID filter (for documents_count, search_has_results)",
                },
            },
            "required": ["condition"],
        },
    },
    {
        "name": "count_findings",
        "description": "Count accumulated research findings with optional filtering by category and confidence threshold. Returns totals grouped by category.",
        "parameters": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Only count findings of this category",
                },
                "min_confidence": {
                    "type": "number",
                    "description": "Minimum confidence score to include (default 0.0)",
                },
            },
            "required": [],
        },
    },
    {
        "name": "check_goal_status",
        "description": "Get current job progress, iteration budget remaining, resource usage, and plan status. Use to decide whether to continue, wrap up, or change strategy.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    # ── Context Window Management Tools ──
    {
        "name": "compress_history",
        "description": "Summarize past action history into a condensed narrative using LLM. The compressed summary persists across iterations so the agent retains awareness of earlier work. Use when action history is getting long and you want to preserve context without losing track of what was done.",
        "parameters": {
            "type": "object",
            "properties": {
                "keep_last": {
                    "type": "integer",
                    "description": "Number of recent actions to keep verbatim (default 5, max 20)",
                },
            },
            "required": [],
        },
    },
    {
        "name": "summarize_findings",
        "description": "Synthesize accumulated research findings into a coherent summary using LLM. Optionally consolidate findings into a single synthesized finding to reduce clutter. Can filter by category to synthesize specific finding types.",
        "parameters": {
            "type": "object",
            "properties": {
                "consolidate": {
                    "type": "boolean",
                    "description": "If true, replace target findings with one synthesized finding (default false)",
                },
                "category": {
                    "type": "string",
                    "description": "Only summarize findings of this category",
                },
            },
            "required": [],
        },
    },
    # ── Agent Collaboration Protocol Tools ──
    {
        "name": "create_handoff",
        "description": "Create a structured handoff to spawn a child agent job with a typed contract specifying what the child should produce. The child will see the contract in its system prompt. Use instead of delegate_subtask when you need structured output expectations.",
        "parameters": {
            "type": "object",
            "properties": {
                "goal": {"type": "string", "description": "The child agent's goal"},
                "job_type": {
                    "type": "string",
                    "enum": ["research", "analysis", "synthesis", "custom"],
                    "description": "Job type for the child (default research)",
                },
                "context": {
                    "type": "string",
                    "description": "Situation briefing — what the child needs to know about the current state",
                },
                "expected_outputs": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "What the child must produce (e.g., summary, key_findings, recommendations)",
                },
                "share_findings": {
                    "type": "boolean",
                    "description": "Share current findings with the child (default true)",
                },
                "max_iterations": {
                    "type": "integer",
                    "description": "Maximum iterations for the child job (default 10, max 20)",
                },
            },
            "required": ["goal", "expected_outputs"],
        },
    },
    {
        "name": "get_sibling_status",
        "description": "Check status, progress, and optionally findings of sibling agent jobs (jobs with the same parent). Use to coordinate with peer agents running in parallel.",
        "parameters": {
            "type": "object",
            "properties": {
                "include_findings": {
                    "type": "boolean",
                    "description": "Also return finding titles from siblings (default false)",
                },
            },
            "required": [],
        },
    },
    {
        "name": "broadcast_to_siblings",
        "description": "Send a message to all sibling agent jobs at once. Use for coordination announcements, status updates, or sharing discoveries with all peer agents.",
        "parameters": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "The message to broadcast",
                },
                "category": {
                    "type": "string",
                    "description": "Message category (default broadcast)",
                },
            },
            "required": ["message"],
        },
    },
    # ── Prompt Template Management Tools ──
    {
        "name": "switch_strategy",
        "description": "Change the agent's role/skill profile mid-run. Different roles prioritize different tools and approaches: researcher (discovery), critic (challenge/validate), synthesizer (combine/summarize), verifier (check/test), coder (code changes), author (document writing). The new profile takes effect on the next thinking step.",
        "parameters": {
            "type": "object",
            "properties": {
                "role": {
                    "type": "string",
                    "enum": [
                        "researcher",
                        "critic",
                        "synthesizer",
                        "verifier",
                        "coder",
                        "author",
                    ],
                    "description": "The role to switch to",
                },
                "reason": {
                    "type": "string",
                    "description": "Why switching strategy (logged for transparency)",
                },
            },
            "required": ["role"],
        },
    },
    {
        "name": "set_focus_directive",
        "description": "Set a custom focus directive that gets injected into the system prompt on every subsequent iteration. Use to steer attention toward specific aspects (e.g., contradictions, recent papers, practical applications).",
        "parameters": {
            "type": "object",
            "properties": {
                "directive": {
                    "type": "string",
                    "description": "The focus instruction (e.g., 'Prioritize finding contradictions between sources')",
                },
                "append": {
                    "type": "boolean",
                    "description": "If true, append to existing directive; if false, replace (default false)",
                },
            },
            "required": ["directive"],
        },
    },
    {
        "name": "get_available_strategies",
        "description": "List all available role profiles with their descriptions, tool preferences, and guidance. Use before switch_strategy to understand available options.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    # ── Agent Output Formatting Tools ──
    {
        "name": "format_as_table",
        "description": "Convert data into a formatted markdown table, stored as an artifact. Use source='findings' to auto-extract from accumulated findings, or provide custom columns and rows.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Table title"},
                "columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Column headers (required for custom source)",
                },
                "rows": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "string"}},
                    "description": "Row data — each inner array matches columns (required for custom source)",
                },
                "source": {
                    "type": "string",
                    "enum": ["findings", "custom"],
                    "description": "Data source: 'findings' auto-extracts from state, 'custom' uses columns/rows (default custom)",
                },
                "finding_fields": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Which finding fields to use as columns when source=findings (default: title, category, confidence)",
                },
            },
            "required": ["title"],
        },
    },
    {
        "name": "format_as_report",
        "description": "Compile findings, progress reports, and custom sections into a structured markdown report. Optionally persist to the knowledge base as a document.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Report title"},
                "sections": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "heading": {"type": "string"},
                            "content": {"type": "string"},
                        },
                    },
                    "description": "Custom report sections",
                },
                "include_findings": {
                    "type": "boolean",
                    "description": "Include accumulated findings (default true)",
                },
                "include_progress": {
                    "type": "boolean",
                    "description": "Include progress report history (default true)",
                },
                "executive_summary": {
                    "type": "string",
                    "description": "Executive summary text",
                },
                "persist": {
                    "type": "boolean",
                    "description": "Save as a document in the knowledge base (default false)",
                },
            },
            "required": ["title"],
        },
    },
    {
        "name": "set_output_schema",
        "description": "Define or update a structured JSON output for the final job results. Set key-value pairs that will be included in job.results['structured_output'] at completion. Use to build structured output progressively throughout execution.",
        "parameters": {
            "type": "object",
            "properties": {
                "schema": {
                    "type": "object",
                    "description": "Key-value pairs to set in the structured output",
                },
                "merge": {
                    "type": "boolean",
                    "description": "If true, merge with existing schema; if false, replace (default true)",
                },
            },
            "required": ["schema"],
        },
    },
    # ── Multi-Modal Ingestion ──────────────────────────────────────────
    {
        "name": "transcribe_document",
        "description": "Trigger Whisper transcription on an existing audio or video document in the knowledge base. Creates a linked transcript document asynchronously. Check the document's extra_metadata for transcription status afterwards.",
        "parameters": {
            "type": "object",
            "properties": {
                "document_id": {
                    "type": "string",
                    "description": "UUID of the audio/video document to transcribe",
                },
                "language": {
                    "type": "string",
                    "description": "Language code (e.g. 'en', 'ru') or 'auto' for detection (default 'auto')",
                },
            },
            "required": ["document_id"],
        },
    },
    {
        "name": "analyze_image",
        "description": "Analyze an image document using a vision-capable LLM. Downloads the image and sends it to the model with a custom prompt. Useful for describing images, extracting text (OCR), identifying diagrams, or visual analysis.",
        "parameters": {
            "type": "object",
            "properties": {
                "document_id": {
                    "type": "string",
                    "description": "UUID of the image document to analyze",
                },
                "prompt": {
                    "type": "string",
                    "description": "Analysis prompt (default: describe the image in detail)",
                },
                "model": {
                    "type": "string",
                    "description": "Vision model override (e.g. 'llava', 'llava:13b'). Defaults to configured VISION_MODEL.",
                },
            },
            "required": ["document_id"],
        },
    },
    {
        "name": "get_media_info",
        "description": "Get media-specific metadata for a document. For audio/video: duration, codec, bitrate, dimensions. For images: width, height, format, color mode. Also includes transcription status.",
        "parameters": {
            "type": "object",
            "properties": {
                "document_id": {
                    "type": "string",
                    "description": "UUID of the media document to inspect",
                },
            },
            "required": ["document_id"],
        },
    },
    # ── Workspace Snapshot ─────────────────────────────────────────────
    {
        "name": "capture_snapshot",
        "description": "Capture a named snapshot of current workspace state metrics (findings count, progress, tool stats, etc.) for later comparison or drift detection.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Snapshot label (e.g. 'after_search', 'before_synthesis')",
                },
                "keys": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Additional state keys to capture beyond default metrics",
                },
            },
            "required": ["name"],
        },
    },
    {
        "name": "compare_snapshots",
        "description": "Compare two named snapshots and return a structured diff showing what changed between them (findings delta, progress change, new tools used, etc.).",
        "parameters": {
            "type": "object",
            "properties": {
                "snapshot_a": {
                    "type": "string",
                    "description": "Name of the earlier snapshot",
                },
                "snapshot_b": {
                    "type": "string",
                    "description": "Name of the later snapshot",
                },
            },
            "required": ["snapshot_a", "snapshot_b"],
        },
    },
    {
        "name": "detect_drift",
        "description": "Compare current state against a named baseline snapshot and flag significant changes or problems (stalling, progress regression, high tool failure rates).",
        "parameters": {
            "type": "object",
            "properties": {
                "baseline": {
                    "type": "string",
                    "description": "Name of the baseline snapshot to compare against",
                },
                "thresholds": {
                    "type": "object",
                    "description": 'Custom thresholds for drift alerts (e.g. {"stalled_iterations": 3, "goal_progress_drop": 10})',
                },
            },
            "required": ["baseline"],
        },
    },
]

# Combine all tools
AGENT_TOOLS = AGENT_TOOLS + AUTONOMOUS_AGENT_TOOLS


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
