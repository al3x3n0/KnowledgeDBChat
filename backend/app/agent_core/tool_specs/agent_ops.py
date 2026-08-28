"""The agent operating on itself: reasoning, batching, context, formatting.

Generated from the literals these tools were declared as; the descriptions
and parameter schemas are the original text.
"""

from __future__ import annotations

from app.agent_core.tool_specs.spec import ToolSpec

SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="get_collection_statistics",
        description="Get comprehensive statistics for a document collection including document counts, file sizes, word counts, processing status, top tags, top authors, and timeline data. Useful for understanding the knowledge base composition.",
        parameters={
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
        job_types=(),
    ),
    ToolSpec(
        name="get_source_analytics",
        description="Get detailed analytics for document sources including document counts, sizes, processing rates, and health status. Useful for monitoring data source performance.",
        parameters={
            "type": "object",
            "properties": {
                "source_id": {
                    "type": "string",
                    "description": "Specific source UUID to analyze (optional, returns all sources if not specified)",
                }
            },
            "required": [],
        },
        job_types=(),
    ),
    ToolSpec(
        name="get_trending_topics",
        description="Find trending topics based on recent document tags and content. Shows which topics are rising, stable, or declining in frequency.",
        parameters={
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
        job_types=(),
    ),
    ToolSpec(
        name="write_progress_report",
        description="Write a progress report for the current job. Useful for documenting what has been accomplished so far.",
        parameters={
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
    ),
    ToolSpec(
        name="suggest_next_action",
        description="Get AI suggestions for the next action based on current job state and findings. Useful when uncertain about how to proceed.",
        parameters={
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
    ),
    ToolSpec(
        name="reflect",
        description="Self-reflect on current progress, approach quality, and potential blind spots. Stores reflection in state for future reference.",
        parameters={
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
    ),
    ToolSpec(
        name="hypothesize",
        description="Formulate and track a hypothesis. Hypotheses can later be confirmed, refuted, or updated with evidence.",
        parameters={
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
    ),
    ToolSpec(
        name="weigh_evidence",
        description="Score and record evidence for or against a claim or hypothesis. Maintains a running evidence ledger.",
        parameters={
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
    ),
    ToolSpec(
        name="critique_plan",
        description="Challenge the current execution plan. Identify weaknesses, missing steps, or questionable assumptions.",
        parameters={
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
    ),
    ToolSpec(
        name="calibration_report",
        description="Read how past predictions held up, overall and grouped by "
        "methodology tag. Use this before choosing an approach: it says "
        "which methods have been predicting well and which have not, "
        "including how many predictions were never checked.",
        parameters={
            "type": "object",
            "properties": {
                "metric": {"type": "string", "description": "Filter by metric"},
                "subject": {"type": "string", "description": "Filter by subject"},
                "limit": {"type": "integer", "description": "Max rows (default 50)"},
            },
            "required": [],
        },
        job_types=(),
    ),
    ToolSpec(
        name="recall_prior_findings",
        description="Read what EARLIER runs measured, with the numbers intact. Every "
        "run's findings are stored -- the cycles, the speedup, the subject, "
        "the measurement source -- and this is the only way to reach them; "
        "get_research_findings returns this run's own findings and nothing "
        "else. Use it before measuring something that may already have been "
        "measured, and to get a baseline you would otherwise re-derive. What "
        "comes back is citable in derived_from and each finding says which "
        "job produced it. It does NOT count toward this run's goal contract: "
        "recalling a number is not establishing one. Call with no filters to "
        "see what kinds of evidence exist before asking for a kind.",
        parameters={
            "type": "object",
            "properties": {
                "finding_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Which evidence types to return, e.g. "
                        "['mechanism_comparison', 'headroom_bound']. Omit to "
                        "see every type earlier runs produced, which is the "
                        "way to learn the vocabulary -- the types are whatever "
                        "the tools that ran chose to emit."
                    ),
                },
                "subject": {
                    "type": "string",
                    "description": (
                        "Substring matched against a finding's subject, title "
                        "and measurement source, e.g. 'attention' or 'O3CPU'."
                    ),
                },
                "job_type": {
                    "type": "string",
                    "description": "Only look at jobs of this type (optional)",
                },
                "limit": {
                    "type": "integer",
                    "description": "How many findings to return, 1-25 (default 10)",
                },
            },
            "required": [],
        },
        effects="read",
        cost_tier="low",
        pii_risk="low",
        produces=(),
        typical_seconds=2,
        consumes="nothing; returns evidence earlier runs established.",
    ),
    ToolSpec(
        name="get_job_history",
        description="Query past agent job runs for the same user. Useful for learning from previous attempts, avoiding repeated mistakes, and understanding what has been done before.",
        parameters={
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
    ),
    ToolSpec(
        name="get_job_metrics",
        description="Get detailed metrics for a specific job including resource usage, timing, error rates, and per-tool usage breakdown. Defaults to the current job if no job_id is provided.",
        parameters={
            "type": "object",
            "properties": {
                "job_id": {
                    "type": "string",
                    "description": "UUID of the job to get metrics for (defaults to current job)",
                },
            },
            "required": [],
        },
    ),
    ToolSpec(
        name="get_tool_usage_stats",
        description="Get aggregated tool usage statistics across recent jobs. Shows which tools are used most, success/failure rates, and trends. Useful for optimizing tool selection strategies.",
        parameters={
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
    ),
    ToolSpec(
        name="get_tool_failure_analysis",
        description="Analyze failure patterns for a specific tool. Groups errors by pattern, shows frequency and examples. Useful for understanding why a tool is failing and how to work around issues.",
        parameters={
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
    ),
    ToolSpec(
        name="batch_search",
        description="Run multiple search queries against the knowledge base in a single call. Returns results grouped by query with optional deduplication. Much more efficient than calling search_documents multiple times.",
        parameters={
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
        cost_tier="medium",
    ),
    ToolSpec(
        name="batch_summarize",
        description="Get summaries for multiple documents in a single call. Returns existing pre-generated summaries immediately. Use generate_missing=true to generate summaries for documents that don't have one (slower, uses LLM).",
        parameters={
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
        cost_tier="medium",
    ),
    ToolSpec(
        name="evaluate_condition",
        description="Evaluate a structured condition against current job state. Returns a boolean result with context data. Use to check findings count, category presence, document count, search coverage, action count, or progress level before deciding next steps.",
        parameters={
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
    ),
    ToolSpec(
        name="count_findings",
        description="Count accumulated research findings with optional filtering by category and confidence threshold. Returns totals grouped by category.",
        parameters={
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
    ),
    ToolSpec(
        name="check_goal_status",
        description="Get current job progress, iteration budget remaining, resource usage, and plan status. Use to decide whether to continue, wrap up, or change strategy.",
        parameters={"type": "object", "properties": {}, "required": []},
    ),
    ToolSpec(
        name="compress_history",
        description="Summarize past action history into a condensed narrative using LLM. The compressed summary persists across iterations so the agent retains awareness of earlier work. Use when action history is getting long and you want to preserve context without losing track of what was done.",
        parameters={
            "type": "object",
            "properties": {
                "keep_last": {
                    "type": "integer",
                    "description": "Number of recent actions to keep verbatim (default 5, max 20)",
                },
            },
            "required": [],
        },
        cost_tier="medium",
    ),
    ToolSpec(
        name="summarize_findings",
        description="Synthesize accumulated research findings into a coherent summary using LLM. Optionally consolidate findings into a single synthesized finding to reduce clutter. Can filter by category to synthesize specific finding types.",
        parameters={
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
        cost_tier="medium",
    ),
    ToolSpec(
        name="switch_strategy",
        description="Change the agent's role/skill profile mid-run. Different roles prioritize different tools and approaches: researcher (discovery), critic (challenge/validate), synthesizer (combine/summarize), verifier (check/test), coder (code changes), author (document writing). The new profile takes effect on the next thinking step.",
        parameters={
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
    ),
    ToolSpec(
        name="set_focus_directive",
        description="Set a custom focus directive that gets injected into the system prompt on every subsequent iteration. Use to steer attention toward specific aspects (e.g., contradictions, recent papers, practical applications).",
        parameters={
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
    ),
    ToolSpec(
        name="get_available_strategies",
        description="List all available role profiles with their descriptions, tool preferences, and guidance. Use before switch_strategy to understand available options.",
        parameters={"type": "object", "properties": {}, "required": []},
    ),
    ToolSpec(
        name="format_as_table",
        description="Convert data into a formatted markdown table, stored as an artifact. Use source='findings' to auto-extract from accumulated findings, or provide custom columns and rows.",
        parameters={
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
    ),
    ToolSpec(
        name="format_as_report",
        description="Compile findings, progress reports, and custom sections into a structured markdown report. Optionally persist to the knowledge base as a document.",
        parameters={
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
    ),
    ToolSpec(
        name="set_output_schema",
        description="Define or update a structured JSON output for the final job results. Set key-value pairs that will be included in job.results['structured_output'] at completion. Use to build structured output progressively throughout execution.",
        parameters={
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
    ),
)
