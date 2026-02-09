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
        "name": "web_scrape",
        "description": "Fetch a web page (or a small set of pages) and extract readable text and links. Useful for wikis/portals.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch (http/https)"
                },
                "follow_links": {
                    "type": "boolean",
                    "description": "Whether to crawl links from the page (bounded by max_pages/max_depth)",
                    "default": False
                },
                "max_pages": {
                    "type": "integer",
                    "description": "Maximum pages to fetch when crawling (default: 1, max: 25)",
                    "default": 1
                },
                "max_depth": {
                    "type": "integer",
                    "description": "Maximum crawl depth when follow_links is true (default: 0, max: 5)",
                    "default": 0
                },
                "same_domain_only": {
                    "type": "boolean",
                    "description": "Only follow links on the same domain as the start URL",
                    "default": True
                },
                "include_links": {
                    "type": "boolean",
                    "description": "Include extracted links in the response",
                    "default": True
                },
                "allow_private_networks": {
                    "type": "boolean",
                    "description": "Allow private-network hosts (admin only)",
                    "default": False
                },
                "max_content_chars": {
                    "type": "integer",
                    "description": "Maximum characters to return per page (default: 50000, max: 500000)",
                    "default": 50000
                },
            },
            "required": ["url"]
        }
    },
    {
        "name": "ingest_url",
        "description": "Scrape a URL and ingest the extracted text into the KnowledgeDB as document(s) (optionally crawling a few linked pages).",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to ingest (http/https)"
                },
                "title": {
                    "type": "string",
                    "description": "Optional title override for the created document (single-document mode only)"
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tags to attach to created/updated documents (optional)"
                },
                "follow_links": {
                    "type": "boolean",
                    "description": "Whether to crawl links from the page (bounded by max_pages/max_depth)",
                    "default": False
                },
                "max_pages": {
                    "type": "integer",
                    "description": "Maximum pages to fetch when crawling (default: 1, max: 25)",
                    "default": 1
                },
                "max_depth": {
                    "type": "integer",
                    "description": "Maximum crawl depth when follow_links is true (default: 0, max: 5)",
                    "default": 0
                },
                "same_domain_only": {
                    "type": "boolean",
                    "description": "Only follow links on the same domain as the start URL",
                    "default": True
                },
                "one_document_per_page": {
                    "type": "boolean",
                    "description": "If crawling, create/update one document per page URL instead of combining into one",
                    "default": False
                },
                "allow_private_networks": {
                    "type": "boolean",
                    "description": "Allow private-network hosts (admin only, or allowlisted web sources)",
                    "default": False
                },
                "max_content_chars": {
                    "type": "integer",
                    "description": "Maximum characters to return per page before ingesting (default: 50000, max: 500000)",
                    "default": 50000
                },
            },
            "required": ["url"]
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
                    "enum": ["documents", "description", "search", "gitlab_repo"],
                    "description": "Source for diagram generation: 'documents' (use specific doc IDs), 'description' (use provided text), 'search' (search and use results), 'gitlab_repo' (analyze GitLab repository)",
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
                "gitlab_project": {
                    "type": "string",
                    "description": "GitLab project ID or path (required if source is 'gitlab_repo')"
                },
                "gitlab_branch": {
                    "type": "string",
                    "description": "Branch to analyze (optional, defaults to default branch)"
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
    {
        "name": "generate_gitlab_architecture",
        "description": "Generate an architecture diagram from a GitLab repository. Analyzes the repository structure, README, config files (docker-compose, package.json, requirements.txt, etc.) and code to understand the system architecture and generate a visual diagram. Use this when the user asks to create an architecture diagram from a GitLab repo.",
        "parameters": {
            "type": "object",
            "properties": {
                "project_id": {
                    "type": "string",
                    "description": "GitLab project ID or path (e.g., 'group/project' or numeric ID)"
                },
                "branch": {
                    "type": "string",
                    "description": "Branch to analyze (optional, defaults to default branch)"
                },
                "diagram_type": {
                    "type": "string",
                    "enum": ["flowchart", "architecture", "c4", "auto"],
                    "description": "Type of architecture diagram to generate",
                    "default": "auto"
                },
                "focus": {
                    "type": "string",
                    "description": "Specific aspect to focus on: 'services', 'data_flow', 'dependencies', 'deployment', 'components'"
                },
                "detail_level": {
                    "type": "string",
                    "enum": ["high", "medium", "low"],
                    "description": "Level of detail: 'high' (all components), 'medium' (main components), 'low' (overview only)",
                    "default": "medium"
                }
            },
            "required": ["project_id"]
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
    {
        "name": "search_arxiv",
        "description": "Search scientific papers on arXiv (metadata + abstracts). Use arXiv query syntax such as 'all:transformers AND cat:cs.CL'.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "arXiv API search query (e.g. 'all:diffusion AND cat:cs.CV')"
                },
                "start": {
                    "type": "integer",
                    "description": "Pagination start offset (default: 0)",
                    "default": 0
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum results to return (default: 10, max: 25)",
                    "default": 10
                },
                "sort_by": {
                    "type": "string",
                    "enum": ["relevance", "lastUpdatedDate", "submittedDate"],
                    "description": "Sort by field",
                    "default": "relevance"
                },
                "sort_order": {
                    "type": "string",
                    "enum": ["ascending", "descending"],
                    "description": "Sort order",
                    "default": "descending"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "ingest_arxiv_papers",
        "description": "Ingest arXiv papers into the Knowledge DB by creating an arXiv document source and running ingestion (async). Provide either paper_ids, search_queries, or categories.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Optional display name for the ingestion source"
                },
                "search_queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "arXiv API search_query expressions"
                },
                "paper_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Explicit arXiv identifiers (e.g. 2401.12345)"
                },
                "categories": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "arXiv categories (e.g. cs.CL, cs.CV)"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum results per query (default: 25, max: 200)",
                    "default": 25
                },
                "start": {
                    "type": "integer",
                    "description": "Pagination start offset for queries (default: 0)",
                    "default": 0
                },
                "sort_by": {
                    "type": "string",
                    "enum": ["relevance", "lastUpdatedDate", "submittedDate"],
                    "description": "Sort by field",
                    "default": "submittedDate"
                },
                "sort_order": {
                    "type": "string",
                    "enum": ["ascending", "descending"],
                    "description": "Sort order",
                    "default": "descending"
                },
                "auto_sync": {
                    "type": "boolean",
                    "description": "Trigger ingestion immediately (default: true)",
                    "default": True
                }
            },
            "required": []
        }
    },
    {
        "name": "literature_review_arxiv",
        "description": "Search arXiv for a topic, optionally ingest top papers into the Knowledge DB, and return a compact literature review starter set (papers + links).",
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "Research topic (free text)"
                },
                "query": {
                    "type": "string",
                    "description": "Optional explicit arXiv query; if omitted, derived from topic"
                },
                "categories": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional arXiv categories to constrain the search"
                },
                "max_papers": {
                    "type": "integer",
                    "description": "How many papers to return (default: 5, max: 25)",
                    "default": 5
                },
                "ingest": {
                    "type": "boolean",
                    "description": "Whether to ingest the returned papers into the Knowledge DB (default: true)",
                    "default": True
                },
                "sort_by": {
                    "type": "string",
                    "enum": ["relevance", "lastUpdatedDate", "submittedDate"],
                    "description": "Sort by field",
                    "default": "relevance"
                },
                "sort_order": {
                    "type": "string",
                    "enum": ["ascending", "descending"],
                    "description": "Sort order",
                    "default": "descending"
                }
            },
            "required": ["topic"]
        }
    },
    {
        "name": "summarize_documents_in_source",
        "description": "Queue summarization for documents in a source (e.g., an arXiv import). Use this after ingestion to generate summaries and paper insights.",
        "parameters": {
            "type": "object",
            "properties": {
                "source_id": {"type": "string", "description": "Document source UUID"},
                "force": {"type": "boolean", "description": "Force re-summarize even if summary exists", "default": False},
                "only_missing": {"type": "boolean", "description": "Only summarize documents missing a summary (ignored if force=true)", "default": True},
                "limit": {"type": "integer", "description": "Max documents to queue", "default": 500}
            },
            "required": ["source_id"]
        }
    },
    {
        "name": "enrich_arxiv_metadata_for_source",
        "description": "Enrich arXiv papers in a source with BibTeX and DOI metadata (venue, keywords, affiliations) when available.",
        "parameters": {
            "type": "object",
            "properties": {
                "source_id": {"type": "string", "description": "arXiv source UUID"},
                "force": {"type": "boolean", "description": "Force refresh even if already enriched", "default": False},
                "limit": {"type": "integer", "description": "Max documents to queue", "default": 500}
            },
            "required": ["source_id"]
        }
    },
    {
        "name": "generate_literature_review_for_source",
        "description": "Generate a literature review document for an arXiv import source (uses available summaries and extracted paper insights).",
        "parameters": {
            "type": "object",
            "properties": {
                "source_id": {"type": "string", "description": "arXiv source UUID"},
                "topic": {"type": "string", "description": "Optional topic label for the report"}
            },
            "required": ["source_id"]
        }
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
                "slide_count": {"type": "integer", "description": "Slide count (3-40)", "default": 10},
                "style": {"type": "string", "description": "Presentation style", "default": "professional"},
                "include_diagrams": {"type": "boolean", "description": "Include diagrams", "default": True},
                "prefer_review_document": {"type": "boolean", "description": "Use the literature review as the only source doc when available", "default": True}
            },
            "required": ["source_id"]
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
                    "description": "Filter statistics to a specific document source UUID"
                },
                "tag": {
                    "type": "string",
                    "description": "Filter statistics to documents with a specific tag"
                },
                "date_from": {
                    "type": "string",
                    "description": "Start date filter (ISO format: YYYY-MM-DD)"
                },
                "date_to": {
                    "type": "string",
                    "description": "End date filter (ISO format: YYYY-MM-DD)"
                }
            },
            "required": []
        }
    },
    {
        "name": "get_source_analytics",
        "description": "Get detailed analytics for document sources including document counts, sizes, processing rates, and health status. Useful for monitoring data source performance.",
        "parameters": {
            "type": "object",
            "properties": {
                "source_id": {
                    "type": "string",
                    "description": "Specific source UUID to analyze (optional, returns all sources if not specified)"
                }
            },
            "required": []
        }
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
                    "default": 7
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of trending topics to return (default: 10)",
                    "default": 10
                }
            },
            "required": []
        }
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
                    "default": "bar"
                },
                "metric": {
                    "type": "string",
                    "enum": ["document_count", "file_size", "content_size"],
                    "description": "Metric to visualize",
                    "default": "document_count"
                },
                "group_by": {
                    "type": "string",
                    "enum": ["source_type", "file_type", "author", "date"],
                    "description": "Field to group the data by",
                    "default": "source_type"
                },
                "date_from": {
                    "type": "string",
                    "description": "Start date filter (ISO format)"
                },
                "date_to": {
                    "type": "string",
                    "description": "End date filter (ISO format)"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum data points (default: 10)",
                    "default": 10
                }
            },
            "required": ["metric", "group_by"]
        }
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
                    "default": "json"
                },
                "source_id": {
                    "type": "string",
                    "description": "Filter to specific document source UUID"
                },
                "tag": {
                    "type": "string",
                    "description": "Filter to documents with specific tag"
                },
                "include_content": {
                    "type": "boolean",
                    "description": "Include full document content (default: false)",
                    "default": False
                },
                "include_chunks": {
                    "type": "boolean",
                    "description": "Include document chunks (default: false)",
                    "default": False
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum documents to export (default: 1000)",
                    "default": 1000
                }
            },
            "required": []
        }
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
                "query": {
                    "type": "string",
                    "description": "Search query"
                },
                "page": {
                    "type": "integer",
                    "description": "Page number (default: 1)",
                    "default": 1
                },
                "page_size": {
                    "type": "integer",
                    "description": "Results per page (default: 10)",
                    "default": 10
                },
                "filters": {
                    "type": "object",
                    "description": "Filter criteria: {source_id, file_type, author, tags, date_range}"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "get_search_suggestions",
        "description": "Get search suggestions and autocomplete for a partial query. Returns suggestions from document titles, tags, and authors.",
        "parameters": {
            "type": "object",
            "properties": {
                "partial_query": {
                    "type": "string",
                    "description": "Partial search query to get suggestions for"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum suggestions (default: 5)",
                    "default": 5
                }
            },
            "required": ["partial_query"]
        }
    },
    {
        "name": "get_related_searches",
        "description": "Get related search queries based on the current search. Useful for discovering related topics and expanding research.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Current search query"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum related searches (default: 5)",
                    "default": 5
                }
            },
            "required": ["query"]
        }
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
                "subject": {
                    "type": "string",
                    "description": "Email subject or topic"
                },
                "recipient": {
                    "type": "string",
                    "description": "Intended recipient (for context)"
                },
                "context": {
                    "type": "string",
                    "description": "Additional context or instructions for the email"
                },
                "document_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Document UUIDs to reference in the email"
                },
                "search_query": {
                    "type": "string",
                    "description": "Search query to find relevant documents to reference"
                },
                "tone": {
                    "type": "string",
                    "enum": ["professional", "casual", "formal", "friendly"],
                    "description": "Email tone (default: professional)",
                    "default": "professional"
                },
                "length": {
                    "type": "string",
                    "enum": ["short", "medium", "long"],
                    "description": "Email length (default: medium)",
                    "default": "medium"
                }
            },
            "required": ["subject"]
        }
    },
    {
        "name": "generate_meeting_notes",
        "description": "Generate structured meeting notes from a transcript or documents. Includes summary, key points, action items, and decisions.",
        "parameters": {
            "type": "object",
            "properties": {
                "transcript": {
                    "type": "string",
                    "description": "Meeting transcript text"
                },
                "document_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Document UUIDs containing meeting content"
                },
                "meeting_title": {
                    "type": "string",
                    "description": "Title of the meeting"
                },
                "participants": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of meeting participants"
                },
                "include_action_items": {
                    "type": "boolean",
                    "description": "Include action items section (default: true)",
                    "default": True
                },
                "include_decisions": {
                    "type": "boolean",
                    "description": "Include decisions section (default: true)",
                    "default": True
                }
            },
            "required": []
        }
    },
    {
        "name": "generate_documentation",
        "description": "Generate technical or user documentation from source documents. Supports various documentation types and target audiences.",
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "Documentation topic"
                },
                "doc_type": {
                    "type": "string",
                    "enum": ["technical", "user_guide", "api", "how_to"],
                    "description": "Type of documentation (default: technical)",
                    "default": "technical"
                },
                "document_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Source document UUIDs"
                },
                "search_query": {
                    "type": "string",
                    "description": "Search query to find relevant source content"
                },
                "target_audience": {
                    "type": "string",
                    "enum": ["developers", "end_users", "admins"],
                    "description": "Target reader (default: developers)",
                    "default": "developers"
                },
                "include_examples": {
                    "type": "boolean",
                    "description": "Include code/usage examples (default: true)",
                    "default": True
                }
            },
            "required": ["topic"]
        }
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
                    "description": "Source document UUIDs to summarize"
                },
                "search_query": {
                    "type": "string",
                    "description": "Search query to find relevant content"
                },
                "topic": {
                    "type": "string",
                    "description": "Focus topic for the summary"
                },
                "max_length": {
                    "type": "integer",
                    "description": "Maximum word count (default: 500)",
                    "default": 500
                },
                "include_recommendations": {
                    "type": "boolean",
                    "description": "Include recommendations section (default: true)",
                    "default": True
                },
                "include_metrics": {
                    "type": "boolean",
                    "description": "Include key metrics (default: true)",
                    "default": True
                }
            },
            "required": []
        }
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
                    "description": "Type of report to generate"
                },
                "document_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Source document UUIDs"
                },
                "search_query": {
                    "type": "string",
                    "description": "Search query to find relevant content"
                },
                "title": {
                    "type": "string",
                    "description": "Report title"
                },
                "sections": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Custom sections to include (optional)"
                }
            },
            "required": ["report_type"]
        }
    }
]


# =========================================================================
# Autonomous Agent Tools
# =========================================================================
AUTONOMOUS_AGENT_TOOLS: List[Dict[str, Any]] = [
    {
        "name": "add_to_reading_list",
        "description": "Add papers or documents to a reading list for later review. Creates a new reading list if it doesn't exist.",
        "parameters": {
            "type": "object",
            "properties": {
                "list_name": {
                    "type": "string",
                    "description": "Name of the reading list to add to (will be created if it doesn't exist)"
                },
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "document_id": {"type": "string", "description": "Document UUID (if already in KB)"},
                            "arxiv_id": {"type": "string", "description": "arXiv ID (e.g., '2301.12345')"},
                            "title": {"type": "string", "description": "Title of the paper/document"},
                            "notes": {"type": "string", "description": "Notes about why this was added"},
                            "priority": {"type": "integer", "description": "Priority 1-5 (1=highest)", "default": 3}
                        }
                    },
                    "description": "Items to add to the reading list"
                }
            },
            "required": ["list_name", "items"]
        }
    },
    {
        "name": "get_reading_lists",
        "description": "Get all reading lists and their items. Useful for checking existing research collections.",
        "parameters": {
            "type": "object",
            "properties": {
                "list_name": {
                    "type": "string",
                    "description": "Filter to a specific list name (optional)"
                },
                "include_items": {
                    "type": "boolean",
                    "description": "Include list items (default: true)",
                    "default": True
                }
            },
            "required": []
        }
    },
    {
        "name": "save_research_finding",
        "description": "Save a research finding or insight discovered during analysis. Findings are stored for later synthesis and reporting.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Brief title for the finding"
                },
                "content": {
                    "type": "string",
                    "description": "Detailed description of the finding"
                },
                "category": {
                    "type": "string",
                    "enum": ["key_insight", "methodology", "result", "gap", "connection", "contradiction", "trend"],
                    "description": "Category of finding"
                },
                "source_document_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Document UUIDs that support this finding"
                },
                "confidence": {
                    "type": "number",
                    "description": "Confidence score 0.0-1.0",
                    "default": 0.8
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tags for categorization"
                }
            },
            "required": ["title", "content", "category"]
        }
    },
    {
        "name": "get_research_findings",
        "description": "Retrieve saved research findings. Useful for reviewing what has been discovered so far.",
        "parameters": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Filter by category"
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Filter by tags"
                },
                "min_confidence": {
                    "type": "number",
                    "description": "Minimum confidence threshold"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum findings to return",
                    "default": 50
                }
            },
            "required": []
        }
    },
    {
        "name": "create_synthesis_document",
        "description": "Create a synthesis document from collected findings and sources. Generates a structured research report.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Title for the synthesis document"
                },
                "topic": {
                    "type": "string",
                    "description": "Research topic being synthesized"
                },
                "findings_to_include": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "IDs of specific findings to include (optional, includes all if empty)"
                },
                "document_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Source document UUIDs to reference"
                },
                "sections": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Custom sections to include (default: introduction, key findings, methodology, gaps, conclusion)"
                },
                "format": {
                    "type": "string",
                    "enum": ["markdown", "structured", "academic"],
                    "description": "Output format",
                    "default": "structured"
                }
            },
            "required": ["title", "topic"]
        }
    },
    {
        "name": "extract_paper_insights",
        "description": "Extract structured insights from a research paper including methodology, key findings, limitations, and future work.",
        "parameters": {
            "type": "object",
            "properties": {
                "document_id": {
                    "type": "string",
                    "description": "Document UUID of the paper"
                },
                "focus_areas": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["methodology", "results", "limitations", "future_work", "contributions", "related_work", "datasets", "metrics"]
                    },
                    "description": "Specific areas to focus extraction on"
                },
                "extract_entities": {
                    "type": "boolean",
                    "description": "Extract named entities (authors, institutions, methods, datasets)",
                    "default": True
                }
            },
            "required": ["document_id"]
        }
    },
    {
        "name": "find_related_papers",
        "description": "Find papers related to a given paper through citations, shared authors, or semantic similarity.",
        "parameters": {
            "type": "object",
            "properties": {
                "document_id": {
                    "type": "string",
                    "description": "Document UUID of the reference paper"
                },
                "arxiv_id": {
                    "type": "string",
                    "description": "arXiv ID (alternative to document_id)"
                },
                "relation_type": {
                    "type": "string",
                    "enum": ["semantic", "citations", "shared_authors", "shared_topics", "all"],
                    "description": "Type of relationship to find",
                    "default": "semantic"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum papers to return",
                    "default": 10
                },
                "search_external": {
                    "type": "boolean",
                    "description": "Search external sources (arXiv) in addition to knowledge base",
                    "default": True
                }
            },
            "required": []
        }
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
                    "description": "Document UUIDs to analyze"
                },
                "source_id": {
                    "type": "string",
                    "description": "Document source UUID (analyze all papers in source)"
                },
                "focus_on": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["methods", "datasets", "metrics", "authors", "concepts", "tools"]
                    },
                    "description": "Entity types to focus on"
                },
                "include_relationships": {
                    "type": "boolean",
                    "description": "Include relationships between entities",
                    "default": True
                }
            },
            "required": []
        }
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
                    "description": "Document UUIDs of papers to compare"
                },
                "comparison_aspects": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["approach", "datasets", "metrics", "results", "limitations", "computational_cost"]
                    },
                    "description": "Aspects to compare"
                },
                "output_format": {
                    "type": "string",
                    "enum": ["table", "narrative", "structured"],
                    "description": "Output format for comparison",
                    "default": "structured"
                }
            },
            "required": ["document_ids"]
        }
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
                    "description": "Document UUIDs to analyze"
                },
                "source_id": {
                    "type": "string",
                    "description": "Document source UUID (analyze all papers in source)"
                },
                "topic": {
                    "type": "string",
                    "description": "Research topic to focus gap analysis on"
                },
                "gap_types": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["methodological", "empirical", "theoretical", "application", "dataset", "evaluation"]
                    },
                    "description": "Types of gaps to look for"
                }
            },
            "required": []
        }
    },
    {
        "name": "generate_research_presentation",
        "description": "Generate a presentation from research findings. Creates a presentation job that can be downloaded.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Presentation title"
                },
                "topic": {
                    "type": "string",
                    "description": "Research topic"
                },
                "document_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Source document UUIDs"
                },
                "slide_count": {
                    "type": "integer",
                    "description": "Number of slides (5-30)",
                    "default": 12
                },
                "style": {
                    "type": "string",
                    "enum": ["academic", "professional", "technical"],
                    "description": "Presentation style",
                    "default": "academic"
                },
                "include_diagrams": {
                    "type": "boolean",
                    "description": "Include auto-generated diagrams",
                    "default": True
                }
            },
            "required": ["title", "topic"]
        }
    },
    {
        "name": "monitor_arxiv_topic",
        "description": "Set up or check monitoring for new papers on a topic. Returns recent papers matching the criteria.",
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "Topic to monitor"
                },
                "query": {
                    "type": "string",
                    "description": "arXiv query expression"
                },
                "categories": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "arXiv categories to monitor"
                },
                "since_days": {
                    "type": "integer",
                    "description": "Look back this many days for new papers",
                    "default": 7
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum papers to return",
                    "default": 20
                }
            },
            "required": ["topic"]
        }
    },
    {
        "name": "ingest_paper_by_id",
        "description": "Ingest a specific paper into the knowledge base by its arXiv ID.",
        "parameters": {
            "type": "object",
            "properties": {
                "arxiv_id": {
                    "type": "string",
                    "description": "arXiv paper ID (e.g., '2301.12345')"
                },
                "add_to_reading_list": {
                    "type": "string",
                    "description": "Name of reading list to add paper to (optional)"
                },
                "extract_insights": {
                    "type": "boolean",
                    "description": "Extract and save insights after ingestion",
                    "default": True
                }
            },
            "required": ["arxiv_id"]
        }
    },
    {
        "name": "write_progress_report",
        "description": "Write a progress report for the current job. Useful for documenting what has been accomplished so far.",
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Brief summary of progress"
                },
                "completed_tasks": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of completed tasks"
                },
                "pending_tasks": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of pending tasks"
                },
                "key_findings": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Key findings so far"
                },
                "blockers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Any blockers or issues"
                },
                "next_steps": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Planned next steps"
                }
            },
            "required": ["summary"]
        }
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
                    "description": "Document UUIDs to analyze as a cluster"
                },
                "analysis_type": {
                    "type": "string",
                    "enum": ["themes", "evolution", "comparison", "comprehensive"],
                    "description": "Type of cluster analysis",
                    "default": "comprehensive"
                },
                "extract_timeline": {
                    "type": "boolean",
                    "description": "Extract temporal evolution of topics",
                    "default": False
                }
            },
            "required": ["document_ids"]
        }
    },
    {
        "name": "suggest_next_action",
        "description": "Get AI suggestions for the next action based on current job state and findings. Useful when uncertain about how to proceed.",
        "parameters": {
            "type": "object",
            "properties": {
                "current_goal": {
                    "type": "string",
                    "description": "Current goal being worked on"
                },
                "progress_so_far": {
                    "type": "string",
                    "description": "Description of progress made"
                },
                "available_resources": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Available resources (documents, sources, etc.)"
                },
                "constraints": {
                    "type": "string",
                    "description": "Any constraints to consider"
                }
            },
            "required": ["current_goal"]
        }
    },
    {
        "name": "create_knowledge_base_entry",
        "description": "Create a new structured entry in the knowledge base (not a raw document). Good for storing curated knowledge from research.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Entry title"
                },
                "content": {
                    "type": "string",
                    "description": "Main content (markdown supported)"
                },
                "entry_type": {
                    "type": "string",
                    "enum": ["concept", "method", "dataset", "tool", "finding", "synthesis", "comparison"],
                    "description": "Type of knowledge entry"
                },
                "related_documents": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Related document UUIDs"
                },
                "related_entities": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Related entity UUIDs from knowledge graph"
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tags for categorization"
                },
                "metadata": {
                    "type": "object",
                    "description": "Additional structured metadata"
                }
            },
            "required": ["title", "content", "entry_type"]
        }
    },
    {
        "name": "link_entities",
        "description": "Create or strengthen a relationship between two entities in the knowledge graph.",
        "parameters": {
            "type": "object",
            "properties": {
                "source_entity_id": {
                    "type": "string",
                    "description": "Source entity UUID"
                },
                "target_entity_id": {
                    "type": "string",
                    "description": "Target entity UUID"
                },
                "source_name": {
                    "type": "string",
                    "description": "Source entity name (alternative to ID)"
                },
                "target_name": {
                    "type": "string",
                    "description": "Target entity name (alternative to ID)"
                },
                "relationship_type": {
                    "type": "string",
                    "description": "Type of relationship (e.g., 'uses', 'extends', 'compares_to', 'improves')"
                },
                "evidence": {
                    "type": "string",
                    "description": "Evidence or explanation for this relationship"
                },
                "confidence": {
                    "type": "number",
                    "description": "Confidence score 0.0-1.0",
                    "default": 0.8
                }
            },
            "required": ["relationship_type"]
        }
    },
    {
        "name": "search_with_filters",
        "description": "Advanced search with multiple filters. More flexible than basic search.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                },
                "source_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Limit to specific sources"
                },
                "file_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Filter by file types"
                },
                "date_from": {
                    "type": "string",
                    "description": "Start date (ISO format)"
                },
                "date_to": {
                    "type": "string",
                    "description": "End date (ISO format)"
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Filter by tags"
                },
                "min_relevance": {
                    "type": "number",
                    "description": "Minimum relevance score 0.0-1.0",
                    "default": 0.5
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum results",
                    "default": 20
                }
            },
            "required": ["query"]
        }
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
                    "description": "List of arXiv IDs to ingest"
                },
                "source_name": {
                    "type": "string",
                    "description": "Name for the document source"
                },
                "add_to_reading_list": {
                    "type": "string",
                    "description": "Reading list to add papers to"
                }
            },
            "required": ["arxiv_ids"]
        }
    }
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
