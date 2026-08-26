"""Research tools: papers, reading lists, synthesis and literature review.

Generated from the literals these tools were declared as; the descriptions
and parameter schemas are the original text.
"""

from __future__ import annotations

from app.agent_core.tool_specs.spec import ToolSpec

SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="search_arxiv",
        description="Search scientific papers on arXiv (metadata + abstracts). Use arXiv query syntax such as 'all:transformers AND cat:cs.CL'.",
        parameters={
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
        network="egress",
        job_types=("research", "monitor", "knowledge_expansion", "custom"),
    ),
    ToolSpec(
        name="ingest_arxiv_papers",
        description="Ingest arXiv papers into the Knowledge DB by creating an arXiv document source and running ingestion (async). Provide either paper_ids, search_queries, or categories.",
        parameters={
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
        effects="write",
        network="egress",
        job_types=(),
    ),
    ToolSpec(
        name="literature_review_arxiv",
        description="Search arXiv for a topic, optionally ingest top papers into the Knowledge DB, and return a compact literature review starter set (papers + links).",
        parameters={
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
        network="egress",
        job_types=(),
    ),
    ToolSpec(
        name="enrich_arxiv_metadata_for_source",
        description="Enrich arXiv papers in a source with BibTeX and DOI metadata (venue, keywords, affiliations) when available.",
        parameters={
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
        job_types=(),
    ),
    ToolSpec(
        name="generate_literature_review_for_source",
        description="Generate a literature review document for an arXiv import source (uses available summaries and extracted paper insights).",
        parameters={
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
        effects="write",
        job_types=(),
    ),
    ToolSpec(
        name="add_to_reading_list",
        description="Add papers or documents to a reading list for later review. Creates a new reading list if it doesn't exist.",
        parameters={
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
        effects="write",
        job_types=("research", "monitor", "custom"),
    ),
    ToolSpec(
        name="get_reading_lists",
        description="Get all reading lists and their items. Useful for checking existing research collections.",
        parameters={
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
        job_types=("research", "monitor"),
    ),
    ToolSpec(
        name="save_research_finding",
        description="Save a research finding or insight discovered during analysis. Findings are stored for later synthesis and reporting.",
        parameters={
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
                "finding_type": {
                    "type": "string",
                    "description": (
                        "What kind of finding this is, e.g. "
                        "'latency_measurement'. A goal contract can require "
                        "findings of a named type and can bound the numbers "
                        "they carry, so a conclusion recorded without a type "
                        "cannot be checked against either."
                    ),
                },
                "metrics": {
                    "type": "object",
                    "description": (
                        "The numbers this finding asserts, as name to value, "
                        'e.g. {"cycles_per_multiply": 6.0}. State them here '
                        "as well as in the text: a contract can only check a "
                        "number it can read, and prose is not readable."
                    ),
                },
            },
            "required": ["title", "content", "category"],
        },
    ),
    ToolSpec(
        name="get_research_findings",
        description="Retrieve saved research findings. Useful for reviewing what has been discovered so far.",
        parameters={
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
    ),
    ToolSpec(
        name="create_synthesis_document",
        description="Create a synthesis document from collected findings and sources. Generates a structured research report.",
        parameters={
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
        effects="write",
        job_types=("research", "analysis", "synthesis", "custom"),
    ),
    ToolSpec(
        name="extract_paper_insights",
        description="Extract structured insights from a research paper including methodology, key findings, limitations, and future work.",
        parameters={
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
        job_types=("research", "analysis", "custom"),
    ),
    ToolSpec(
        name="find_related_papers",
        description="Find papers related to a given paper through citations, shared authors, or semantic similarity.",
        parameters={
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
        job_types=("research", "knowledge_expansion"),
    ),
    ToolSpec(
        name="build_research_graph",
        description="Build a knowledge graph of concepts, methods, and relationships from a set of papers. Useful for understanding the research landscape.",
        parameters={
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
        effects="write",
        job_types=("research", "analysis", "knowledge_expansion"),
    ),
    ToolSpec(
        name="compare_methodologies",
        description="Compare methodologies across multiple papers. Useful for understanding different approaches to a problem.",
        parameters={
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
        job_types=("research", "analysis"),
    ),
    ToolSpec(
        name="identify_research_gaps",
        description="Analyze papers to identify potential research gaps and opportunities.",
        parameters={
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
        job_types=("research", "analysis"),
    ),
    ToolSpec(
        name="generate_research_presentation",
        description="Generate a presentation from research findings. Creates a presentation job that can be downloaded.",
        parameters={
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
        effects="write",
        job_types=("research", "synthesis"),
    ),
    ToolSpec(
        name="monitor_arxiv_topic",
        description="Set up or check monitoring for new papers on a topic. Returns recent papers matching the criteria.",
        parameters={
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
        job_types=("monitor",),
    ),
    ToolSpec(
        name="ingest_paper_by_id",
        description="Ingest a specific paper into the knowledge base by its arXiv ID.",
        parameters={
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
        effects="write",
        job_types=("research", "monitor", "knowledge_expansion"),
    ),
    ToolSpec(
        name="analyze_document_cluster",
        description="Analyze a cluster of related documents to find common themes, differences, and patterns.",
        parameters={
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
        job_types=("research", "analysis"),
    ),
    ToolSpec(
        name="create_knowledge_base_entry",
        description="Create a new structured entry in the knowledge base (not a raw document). Good for storing curated knowledge from research.",
        parameters={
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
        effects="write",
        job_types=("research", "synthesis", "knowledge_expansion"),
    ),
    ToolSpec(
        name="batch_ingest_papers",
        description="Ingest multiple papers at once into the knowledge base.",
        parameters={
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
        job_types=("research", "knowledge_expansion"),
    ),
    ToolSpec(
        name="search_web",
        description="Search the web using DuckDuckGo. Returns titles, URLs, and snippets for the top results.",
        parameters={
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
        network="egress",
        cost_tier="medium",
    ),
    ToolSpec(
        name="summarize_url",
        description="Fetch content from a URL and produce an LLM-generated summary. Optionally focus the summary on a specific topic.",
        parameters={
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
        network="egress",
        cost_tier="medium",
        job_types=("research", "analysis"),
    ),
    ToolSpec(
        name="fetch_url_content",
        description="Fetch and extract text content from a URL. Returns raw text without summarization.",
        parameters={
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
        network="egress",
    ),
)
