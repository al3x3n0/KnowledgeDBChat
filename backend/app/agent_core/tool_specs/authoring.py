"""Generation and authoring tools: documents, diagrams, charts, exports.

Generated from the literals these tools were declared as; the descriptions
and parameter schemas are the original text.
"""

from __future__ import annotations

from app.agent_core.tool_specs.spec import ToolSpec

SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="start_template_fill",
        description="Start a template fill job. Analyzes a template document and fills it with content extracted from source documents using AI. Use this when the user wants to fill a template with information from their documents.",
        parameters={
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
        job_types=(),
    ),
    ToolSpec(
        name="list_template_jobs",
        description="List the user's template fill jobs with their status and progress.",
        parameters={
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
        job_types=(),
    ),
    ToolSpec(
        name="get_template_job_status",
        description="Get detailed status of a specific template fill job including progress, current section being processed, and download link if completed.",
        parameters={
            "type": "object",
            "properties": {
                "job_id": {
                    "type": "string",
                    "description": "The UUID of the template job to check",
                }
            },
            "required": ["job_id"],
        },
        job_types=(),
    ),
    ToolSpec(
        name="generate_diagram",
        description="Generate a visual diagram (architecture, flowchart, sequence, ER diagram, mind map, etc.) from documents or a description. Returns Mermaid diagram code that can be rendered visually. Use this when the user asks for architecture diagrams, system diagrams, flowcharts, or any visual representation of information from documents.",
        parameters={
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
        effects="write",
        job_types=(),
    ),
    ToolSpec(
        name="generate_gitlab_architecture",
        description="Generate an architecture diagram from a GitLab repository. Analyzes the repository structure, README, config files (docker-compose, package.json, requirements.txt, etc.) and code to understand the system architecture and generate a visual diagram. Use this when the user asks to create an architecture diagram from a GitLab repo.",
        parameters={
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
        effects="write",
        job_types=(),
    ),
    ToolSpec(
        name="generate_slides_for_source",
        description="Generate slides (presentation job) for an arXiv import source. Prefers the literature review document if available.",
        parameters={
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
        effects="write",
        job_types=(),
    ),
    ToolSpec(
        name="generate_chart_data",
        description="Generate data for charts and visualizations. Returns structured data that can be used to create bar, line, pie, or area charts.",
        parameters={
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
        effects="write",
        job_types=(),
    ),
    ToolSpec(
        name="export_data",
        description="Export document data to various formats (JSON, CSV, JSONL). Useful for data analysis, backup, or integration with external tools.",
        parameters={
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
        effects="write",
        job_types=(),
    ),
    ToolSpec(
        name="draft_email",
        description="Generate a professional email draft based on context and documents. Can reference knowledge base content for accurate information.",
        parameters={
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
        job_types=(),
    ),
    ToolSpec(
        name="generate_meeting_notes",
        description="Generate structured meeting notes from a transcript or documents. Includes summary, key points, action items, and decisions.",
        parameters={
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
        effects="write",
        job_types=(),
    ),
    ToolSpec(
        name="generate_documentation",
        description="Generate technical or user documentation from source documents. Supports various documentation types and target audiences.",
        parameters={
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
        effects="write",
        job_types=(),
    ),
    ToolSpec(
        name="generate_executive_summary",
        description="Generate a concise executive summary for leadership. Includes key findings, metrics, recommendations, and next steps.",
        parameters={
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
        effects="write",
        job_types=(),
    ),
    ToolSpec(
        name="generate_report",
        description="Generate a structured report (status, analysis, research, or summary) from documents. Includes proper sections and formatting.",
        parameters={
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
        effects="write",
        cost_tier="medium",
        job_types=(),
    ),
    ToolSpec(
        name="plan_document",
        description="Create a structured document outline with sections and subsections. Stores the plan in document workspace state for subsequent write_section calls.",
        parameters={
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
        job_types=("synthesis", "document_authoring"),
    ),
    ToolSpec(
        name="write_section",
        description="Write content for a specific document section, optionally using RAG search for KB context and citations.",
        parameters={
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
        job_types=("synthesis", "document_authoring"),
    ),
    ToolSpec(
        name="revise_section",
        description="Rewrite a previously written document section with specific feedback or corrections.",
        parameters={
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
        job_types=("synthesis", "document_authoring"),
    ),
    ToolSpec(
        name="assemble_document",
        description="Combine all written sections into a final document with table of contents and references.",
        parameters={
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
        job_types=("synthesis", "document_authoring"),
    ),
    ToolSpec(
        name="export_document",
        description="Export the assembled document to DOCX, PDF, PPTX, or LaTeX format. Optionally persist to the knowledge base.",
        parameters={
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
        effects="write",
        cost_tier="medium",
        job_types=("synthesis", "document_authoring"),
    ),
    ToolSpec(
        name="insert_figure",
        description="Insert a chart, table, or diagram into a document section.",
        parameters={
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
        job_types=("synthesis", "document_authoring"),
    ),
    ToolSpec(
        name="get_workspace_artifact_url",
        description="Get a download URL for a file persisted from a previous job's coding workspace. Use this to access code or documents saved by earlier jobs.",
        parameters={
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
        job_types=("analysis", "coding"),
    ),
    ToolSpec(
        name="create_chart",
        description="Generate a data chart (bar, line, pie, scatter, histogram, heatmap, box, area) from structured data. The chart is rendered as an image and persisted to storage. Returns a download URL.",
        parameters={
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
        effects="write",
        cost_tier="medium",
    ),
    ToolSpec(
        name="render_diagram",
        description="Render diagram source code (Mermaid or Graphviz) to an image and persist to storage. Returns a download URL. Use this after generate_diagram to produce a viewable image.",
        parameters={
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
        effects="write",
        network="egress",
    ),
    ToolSpec(
        name="list_documents_by_tag",
        description="List documents matching specified tags. Supports matching any tag (OR) or all tags (AND). Useful for finding related documents for synthesis or analysis.",
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
                    "description": "If true, documents must have ALL specified tags (AND). If false (default), documents matching ANY tag are returned (OR)",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results (default 20, max 100)",
                },
            },
            "required": ["tags"],
        },
    ),
    ToolSpec(
        name="merge_documents",
        description="Merge content from multiple documents into a single new document. Each source document becomes a section with its title as heading. Useful for creating comprehensive reports from multiple sources.",
        parameters={
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
        effects="write",
        job_types=("research", "analysis", "synthesis"),
    ),
)
