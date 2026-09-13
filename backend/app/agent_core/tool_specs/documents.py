"""Document, search and ingestion tools.

Generated from the literals these tools were declared as; the descriptions
and parameter schemas are the original text.
"""

from __future__ import annotations

from app.agent_core.tool_specs.spec import ToolSpec

SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="search_documents",
        description="Search for documents in the knowledge base using semantic search. Use this to find relevant documents based on a query.",
        parameters={
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
    ),
    ToolSpec(
        name="web_scrape",
        description="Fetch a web page (or a small set of pages) and extract readable text and links. Useful for wikis/portals.",
        parameters={
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
        network="egress",
        pii_risk="medium",
        job_types=(),
    ),
    ToolSpec(
        name="ingest_url",
        produces=("documents_ingested",),
        typical_seconds=30,
        consumes="A URL; leaves the fetched page in the corpus.",
        description="Scrape a URL and ingest the extracted text into the KnowledgeDB as document(s) (optionally crawling a few linked pages).",
        parameters={
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
        effects="write",
        network="egress",
        pii_risk="medium",
        job_types=(),
    ),
    ToolSpec(
        name="get_document_details",
        description="Get detailed information about a specific document including title, content preview, metadata, and processing status.",
        parameters={
            "type": "object",
            "properties": {
                "document_id": {
                    "type": "string",
                    "description": "The UUID of the document to retrieve",
                }
            },
            "required": ["document_id"],
        },
    ),
    ToolSpec(
        name="summarize_document",
        produces=("document_summary",),
        typical_seconds=20,
        consumes="One document id already in the corpus.",
        description="Generate or retrieve a summary for a specific document. If a summary already exists, returns it unless force_regenerate is true.",
        parameters={
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
        job_types=("research", "analysis", "synthesis", "document_authoring", "custom"),
    ),
    ToolSpec(
        name="delete_document",
        description="Delete a document from the knowledge base. This action is irreversible and requires explicit confirmation.",
        parameters={
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
        effects="write",
        job_types=(),
    ),
    ToolSpec(
        name="list_recent_documents",
        description="List the most recently added or updated documents in the knowledge base.",
        parameters={
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
        job_types=(),
    ),
    ToolSpec(
        name="list_document_sources",
        description="List available document sources with type and status.",
        parameters={
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
        job_types=(),
    ),
    ToolSpec(
        name="list_documents_by_source",
        description="List documents from a specific source (by source ID, name, or type).",
        parameters={
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
        job_types=(),
    ),
    ToolSpec(
        name="request_file_upload",
        description="Request the user to upload a file. Use this when the user wants to add a new document to the knowledge base.",
        parameters={
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
        job_types=(),
    ),
    ToolSpec(
        name="create_document_from_text",
        produces=("documents_ingested",),
        typical_seconds=5,
        consumes="Title and body; leaves a document in the corpus.",
        description="Create a new document directly from text content. Useful for saving notes, code snippets, or any text the user wants to store in the knowledge base.",
        parameters={
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
        effects="write",
        job_types=("research", "analysis", "synthesis", "document_authoring", "custom"),
    ),
    ToolSpec(
        name="find_similar_documents",
        description="Find documents that are semantically similar to a given document. Useful for discovering related content.",
        parameters={
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
        job_types=("research", "analysis", "knowledge_expansion", "custom"),
    ),
    ToolSpec(
        name="search_documents_by_author",
        description="Find documents authored by a person. Uses case-insensitive matching.",
        parameters={
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
        job_types=(),
    ),
    ToolSpec(
        name="update_document_tags",
        description="Add, remove, or replace tags on a document. Use action 'add' to add tags, 'remove' to remove tags, or 'replace' to replace all tags.",
        parameters={
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
        effects="write",
        job_types=(),
    ),
    ToolSpec(
        name="get_knowledge_base_stats",
        description="Get statistics about the knowledge base including document counts, storage usage, and processing status.",
        parameters={"type": "object", "properties": {}, "required": []},
        job_types=("research", "monitor", "knowledge_expansion"),
    ),
    ToolSpec(
        name="batch_delete_documents",
        description="Delete multiple documents at once. Requires explicit confirmation. Use with caution as this action is irreversible.",
        parameters={
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
        effects="write",
        job_types=(),
    ),
    ToolSpec(
        name="batch_summarize_documents",
        description="Queue summarization for multiple documents at once. Useful for processing several documents that lack summaries.",
        parameters={
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
        job_types=(),
    ),
    ToolSpec(
        name="search_by_tags",
        description="Find documents that have specific tags. Useful for filtering documents by category or topic.",
        parameters={
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
        job_types=(),
    ),
    ToolSpec(
        name="search_documents_by_tag",
        description="Find documents that have specific tags. Useful for filtering documents by category or topic.",
        parameters={
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
        job_types=(),
    ),
    ToolSpec(
        name="list_all_tags",
        description="Get a list of all unique tags used across documents in the knowledge base.",
        parameters={"type": "object", "properties": {}, "required": []},
        job_types=(),
    ),
    ToolSpec(
        name="compare_documents",
        description="Compare two documents to find similarities and differences. Analyzes content overlap, unique sections, and provides a similarity score.",
        parameters={
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
        job_types=("analysis",),
    ),
    ToolSpec(
        name="answer_question",
        description="Answer a question using RAG (Retrieval-Augmented Generation) by searching the knowledge base and generating a response based on relevant document content. Use this when the user asks a factual question that should be answered using information from their documents.",
        parameters={
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
        job_types=(),
    ),
    ToolSpec(
        name="read_document_content",
        description="Read the full text content of a document. Use this when you need to see the actual content of a document, not just metadata or preview.",
        parameters={
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
    ),
    ToolSpec(
        name="summarize_documents_in_source",
        description="Queue summarization for documents in a source (e.g., an arXiv import). Use this after ingestion to generate summaries and paper insights.",
        parameters={
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
        job_types=(),
    ),
    ToolSpec(
        name="faceted_search",
        description="Execute a search with faceted results showing aggregations by source type, file type, author, tags, and date. Useful for exploring and filtering large result sets.",
        parameters={
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
        job_types=(),
    ),
    ToolSpec(
        name="get_search_suggestions",
        description="Get search suggestions and autocomplete for a partial query. Returns suggestions from document titles, tags, and authors.",
        parameters={
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
        job_types=(),
    ),
    ToolSpec(
        name="get_related_searches",
        description="Get related search queries based on the current search. Useful for discovering related topics and expanding research.",
        parameters={
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
        job_types=(),
    ),
    ToolSpec(
        name="search_with_filters",
        description="Advanced search with multiple filters. More flexible than basic search.",
        parameters={
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
    ),
    ToolSpec(
        name="transcribe_document",
        description="Trigger Whisper transcription on an existing audio or video document in the knowledge base. Creates a linked transcript document asynchronously. Check the document's extra_metadata for transcription status afterwards.",
        parameters={
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
        effects="write",
        cost_tier="medium",
    ),
    ToolSpec(
        name="analyze_image",
        description="Analyze an image document using a vision-capable LLM. Downloads the image and sends it to the model with a custom prompt. Useful for describing images, extracting text (OCR), identifying diagrams, or visual analysis.",
        parameters={
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
        cost_tier="medium",
    ),
    ToolSpec(
        name="get_media_info",
        description="Get media-specific metadata for a document. For audio/video: duration, codec, bitrate, dimensions. For images: width, height, format, color mode. Also includes transcription status.",
        parameters={
            "type": "object",
            "properties": {
                "document_id": {
                    "type": "string",
                    "description": "UUID of the media document to inspect",
                },
            },
            "required": ["document_id"],
        },
    ),
)
