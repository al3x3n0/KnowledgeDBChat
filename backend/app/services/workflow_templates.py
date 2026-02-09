"""
Pre-built workflow templates for common automation tasks.

These templates leverage the new tools (analytics, search, content generation)
to provide ready-to-use automation workflows.
"""

from typing import List, Dict, Any


# Template definitions
WORKFLOW_TEMPLATES: List[Dict[str, Any]] = [
    # =========================================================================
    # Weekly Knowledge Digest
    # =========================================================================
    {
        "template_id": "weekly_digest",
        "name": "Weekly Knowledge Digest",
        "description": "Generates a weekly summary of trending topics, recent documents, and key insights from the knowledge base. Perfect for team updates.",
        "category": "reporting",
        "trigger_config": {
            "type": "schedule",
            "schedule": "0 9 * * 1"  # Every Monday at 9 AM
        },
        "nodes": [
            {
                "node_id": "start",
                "node_type": "start",
                "config": {},
                "position_x": 250,
                "position_y": 50
            },
            {
                "node_id": "get_trending",
                "node_type": "tool",
                "builtin_tool": "get_trending_topics",
                "config": {
                    "input_mapping": {
                        "days": 7,
                        "limit": 10
                    },
                    "output_key": "trending_topics"
                },
                "position_x": 100,
                "position_y": 150
            },
            {
                "node_id": "get_stats",
                "node_type": "tool",
                "builtin_tool": "get_collection_statistics",
                "config": {
                    "input_mapping": {},
                    "output_key": "kb_stats"
                },
                "position_x": 400,
                "position_y": 150
            },
            {
                "node_id": "merge_parallel",
                "node_type": "parallel",
                "config": {
                    "output_key": "parallel_results"
                },
                "position_x": 250,
                "position_y": 150
            },
            {
                "node_id": "list_recent",
                "node_type": "tool",
                "builtin_tool": "list_recent_documents",
                "config": {
                    "input_mapping": {
                        "limit": 20
                    },
                    "output_key": "recent_docs"
                },
                "position_x": 250,
                "position_y": 280
            },
            {
                "node_id": "generate_summary",
                "node_type": "tool",
                "builtin_tool": "generate_executive_summary",
                "config": {
                    "input_mapping": {
                        "topic": "Weekly Knowledge Base Digest",
                        "max_length": 600,
                        "include_recommendations": True,
                        "include_metrics": True
                    },
                    "output_key": "digest_summary"
                },
                "position_x": 250,
                "position_y": 400
            },
            {
                "node_id": "end",
                "node_type": "end",
                "config": {},
                "position_x": 250,
                "position_y": 520
            }
        ],
        "edges": [
            {"source_node_id": "start", "target_node_id": "merge_parallel"},
            {"source_node_id": "merge_parallel", "target_node_id": "get_trending"},
            {"source_node_id": "merge_parallel", "target_node_id": "get_stats"},
            {"source_node_id": "get_trending", "target_node_id": "list_recent"},
            {"source_node_id": "get_stats", "target_node_id": "list_recent"},
            {"source_node_id": "list_recent", "target_node_id": "generate_summary"},
            {"source_node_id": "generate_summary", "target_node_id": "end"}
        ]
    },

    # =========================================================================
    # Research Paper Pipeline
    # =========================================================================
    {
        "template_id": "research_pipeline",
        "name": "Research Paper Pipeline",
        "description": "Searches arXiv for papers on a topic, ingests them into the knowledge base, generates summaries, and creates a literature review. Input: topic (research topic).",
        "category": "research",
        "trigger_config": {"type": "manual"},
        "nodes": [
            {
                "node_id": "start",
                "node_type": "start",
                "config": {},
                "position_x": 250,
                "position_y": 50
            },
            {
                "node_id": "search_arxiv",
                "node_type": "tool",
                "builtin_tool": "literature_review_arxiv",
                "config": {
                    "input_mapping": {
                        "topic": "{{context.trigger_data.topic}}",
                        "max_papers": 10,
                        "ingest": True
                    },
                    "output_key": "arxiv_results"
                },
                "position_x": 250,
                "position_y": 170
            },
            {
                "node_id": "check_results",
                "node_type": "condition",
                "config": {
                    "condition": {
                        "type": "comparison",
                        "left": "{{context.arxiv_results.papers}}",
                        "operator": "exists",
                        "right": True
                    },
                    "output_key": "has_papers"
                },
                "position_x": 250,
                "position_y": 290
            },
            {
                "node_id": "wait_for_ingestion",
                "node_type": "wait",
                "config": {
                    "wait_seconds": 30
                },
                "position_x": 250,
                "position_y": 410
            },
            {
                "node_id": "summarize_source",
                "node_type": "tool",
                "builtin_tool": "summarize_documents_in_source",
                "config": {
                    "input_mapping": {
                        "source_id": "{{context.arxiv_results.ingest.source_id}}",
                        "force": False,
                        "only_missing": True
                    },
                    "output_key": "summarization_result"
                },
                "position_x": 250,
                "position_y": 530
            },
            {
                "node_id": "generate_review",
                "node_type": "tool",
                "builtin_tool": "generate_literature_review_for_source",
                "config": {
                    "input_mapping": {
                        "source_id": "{{context.arxiv_results.ingest.source_id}}",
                        "topic": "{{context.trigger_data.topic}}"
                    },
                    "output_key": "literature_review"
                },
                "position_x": 250,
                "position_y": 650
            },
            {
                "node_id": "no_papers_end",
                "node_type": "end",
                "config": {
                    "output_key": "no_papers_found"
                },
                "position_x": 450,
                "position_y": 410
            },
            {
                "node_id": "end",
                "node_type": "end",
                "config": {},
                "position_x": 250,
                "position_y": 770
            }
        ],
        "edges": [
            {"source_node_id": "start", "target_node_id": "search_arxiv"},
            {"source_node_id": "search_arxiv", "target_node_id": "check_results"},
            {"source_node_id": "check_results", "target_node_id": "wait_for_ingestion", "source_handle": "true"},
            {"source_node_id": "check_results", "target_node_id": "no_papers_end", "source_handle": "false"},
            {"source_node_id": "wait_for_ingestion", "target_node_id": "summarize_source"},
            {"source_node_id": "summarize_source", "target_node_id": "generate_review"},
            {"source_node_id": "generate_review", "target_node_id": "end"}
        ]
    },

    # =========================================================================
    # Automated Report Generation
    # =========================================================================
    {
        "template_id": "auto_report",
        "name": "Automated Report Generator",
        "description": "Generates a comprehensive report from a document source or search query. Input: source_id OR search_query, report_type (status/analysis/research/summary), title.",
        "category": "reporting",
        "trigger_config": {"type": "manual"},
        "nodes": [
            {
                "node_id": "start",
                "node_type": "start",
                "config": {},
                "position_x": 250,
                "position_y": 50
            },
            {
                "node_id": "search_documents",
                "node_type": "tool",
                "builtin_tool": "faceted_search",
                "config": {
                    "input_mapping": {
                        "query": "{{context.trigger_data.search_query}}",
                        "page_size": 20
                    },
                    "output_key": "search_results"
                },
                "position_x": 250,
                "position_y": 170
            },
            {
                "node_id": "get_stats",
                "node_type": "tool",
                "builtin_tool": "get_collection_statistics",
                "config": {
                    "input_mapping": {
                        "source_id": "{{context.trigger_data.source_id}}"
                    },
                    "output_key": "source_stats"
                },
                "position_x": 250,
                "position_y": 290
            },
            {
                "node_id": "generate_report",
                "node_type": "tool",
                "builtin_tool": "generate_report",
                "config": {
                    "input_mapping": {
                        "report_type": "{{context.trigger_data.report_type}}",
                        "search_query": "{{context.trigger_data.search_query}}",
                        "title": "{{context.trigger_data.title}}"
                    },
                    "output_key": "generated_report"
                },
                "position_x": 250,
                "position_y": 410
            },
            {
                "node_id": "end",
                "node_type": "end",
                "config": {},
                "position_x": 250,
                "position_y": 530
            }
        ],
        "edges": [
            {"source_node_id": "start", "target_node_id": "search_documents"},
            {"source_node_id": "search_documents", "target_node_id": "get_stats"},
            {"source_node_id": "get_stats", "target_node_id": "generate_report"},
            {"source_node_id": "generate_report", "target_node_id": "end"}
        ]
    },

    # =========================================================================
    # Document Analysis Pipeline
    # =========================================================================
    {
        "template_id": "document_analysis",
        "name": "Document Analysis Pipeline",
        "description": "Analyzes a document: generates summary, extracts key entities, finds similar documents, and creates an executive brief. Input: document_id.",
        "category": "analysis",
        "trigger_config": {"type": "manual"},
        "nodes": [
            {
                "node_id": "start",
                "node_type": "start",
                "config": {},
                "position_x": 250,
                "position_y": 50
            },
            {
                "node_id": "get_details",
                "node_type": "tool",
                "builtin_tool": "get_document_details",
                "config": {
                    "input_mapping": {
                        "document_id": "{{context.trigger_data.document_id}}"
                    },
                    "output_key": "doc_details"
                },
                "position_x": 250,
                "position_y": 150
            },
            {
                "node_id": "parallel_analysis",
                "node_type": "parallel",
                "config": {
                    "output_key": "analysis_results"
                },
                "position_x": 250,
                "position_y": 250
            },
            {
                "node_id": "summarize",
                "node_type": "tool",
                "builtin_tool": "summarize_document",
                "config": {
                    "input_mapping": {
                        "document_id": "{{context.trigger_data.document_id}}",
                        "force_regenerate": False
                    },
                    "output_key": "summary"
                },
                "position_x": 100,
                "position_y": 350
            },
            {
                "node_id": "find_similar",
                "node_type": "tool",
                "builtin_tool": "find_similar_documents",
                "config": {
                    "input_mapping": {
                        "document_id": "{{context.trigger_data.document_id}}",
                        "limit": 5
                    },
                    "output_key": "similar_docs"
                },
                "position_x": 250,
                "position_y": 350
            },
            {
                "node_id": "get_kg",
                "node_type": "tool",
                "builtin_tool": "get_document_knowledge_graph",
                "config": {
                    "input_mapping": {
                        "document_id": "{{context.trigger_data.document_id}}"
                    },
                    "output_key": "knowledge_graph"
                },
                "position_x": 400,
                "position_y": 350
            },
            {
                "node_id": "generate_brief",
                "node_type": "tool",
                "builtin_tool": "generate_executive_summary",
                "config": {
                    "input_mapping": {
                        "document_ids": ["{{context.trigger_data.document_id}}"],
                        "max_length": 400,
                        "include_recommendations": True
                    },
                    "output_key": "executive_brief"
                },
                "position_x": 250,
                "position_y": 470
            },
            {
                "node_id": "end",
                "node_type": "end",
                "config": {},
                "position_x": 250,
                "position_y": 590
            }
        ],
        "edges": [
            {"source_node_id": "start", "target_node_id": "get_details"},
            {"source_node_id": "get_details", "target_node_id": "parallel_analysis"},
            {"source_node_id": "parallel_analysis", "target_node_id": "summarize"},
            {"source_node_id": "parallel_analysis", "target_node_id": "find_similar"},
            {"source_node_id": "parallel_analysis", "target_node_id": "get_kg"},
            {"source_node_id": "summarize", "target_node_id": "generate_brief"},
            {"source_node_id": "find_similar", "target_node_id": "generate_brief"},
            {"source_node_id": "get_kg", "target_node_id": "generate_brief"},
            {"source_node_id": "generate_brief", "target_node_id": "end"}
        ]
    },

    # =========================================================================
    # Meeting Follow-up Workflow
    # =========================================================================
    {
        "template_id": "meeting_followup",
        "name": "Meeting Follow-up Generator",
        "description": "Processes meeting transcript or recording to generate meeting notes with action items, then drafts follow-up email. Input: transcript OR document_id, meeting_title, participants (list).",
        "category": "productivity",
        "trigger_config": {"type": "manual"},
        "nodes": [
            {
                "node_id": "start",
                "node_type": "start",
                "config": {},
                "position_x": 250,
                "position_y": 50
            },
            {
                "node_id": "generate_notes",
                "node_type": "tool",
                "builtin_tool": "generate_meeting_notes",
                "config": {
                    "input_mapping": {
                        "transcript": "{{context.trigger_data.transcript}}",
                        "document_ids": "{{context.trigger_data.document_ids}}",
                        "meeting_title": "{{context.trigger_data.meeting_title}}",
                        "participants": "{{context.trigger_data.participants}}",
                        "include_action_items": True,
                        "include_decisions": True
                    },
                    "output_key": "meeting_notes"
                },
                "position_x": 250,
                "position_y": 170
            },
            {
                "node_id": "draft_email",
                "node_type": "tool",
                "builtin_tool": "draft_email",
                "config": {
                    "input_mapping": {
                        "subject": "Meeting Follow-up: {{context.trigger_data.meeting_title}}",
                        "context": "Meeting notes:\n{{context.meeting_notes.notes}}\n\nAction items:\n{{context.meeting_notes.action_items}}",
                        "tone": "professional",
                        "length": "medium"
                    },
                    "output_key": "followup_email"
                },
                "position_x": 250,
                "position_y": 310
            },
            {
                "node_id": "end",
                "node_type": "end",
                "config": {},
                "position_x": 250,
                "position_y": 430
            }
        ],
        "edges": [
            {"source_node_id": "start", "target_node_id": "generate_notes"},
            {"source_node_id": "generate_notes", "target_node_id": "draft_email"},
            {"source_node_id": "draft_email", "target_node_id": "end"}
        ]
    },

    # =========================================================================
    # Batch Document Summarization
    # =========================================================================
    {
        "template_id": "batch_summarize",
        "name": "Batch Document Summarization",
        "description": "Summarizes all documents from a source that are missing summaries. Input: source_id.",
        "category": "processing",
        "trigger_config": {"type": "manual"},
        "nodes": [
            {
                "node_id": "start",
                "node_type": "start",
                "config": {},
                "position_x": 250,
                "position_y": 50
            },
            {
                "node_id": "get_source_stats",
                "node_type": "tool",
                "builtin_tool": "get_source_analytics",
                "config": {
                    "input_mapping": {
                        "source_id": "{{context.trigger_data.source_id}}"
                    },
                    "output_key": "source_info"
                },
                "position_x": 250,
                "position_y": 150
            },
            {
                "node_id": "list_docs",
                "node_type": "tool",
                "builtin_tool": "list_documents_by_source",
                "config": {
                    "input_mapping": {
                        "source_id": "{{context.trigger_data.source_id}}",
                        "limit": 50
                    },
                    "output_key": "source_documents"
                },
                "position_x": 250,
                "position_y": 270
            },
            {
                "node_id": "queue_summaries",
                "node_type": "tool",
                "builtin_tool": "summarize_documents_in_source",
                "config": {
                    "input_mapping": {
                        "source_id": "{{context.trigger_data.source_id}}",
                        "force": False,
                        "only_missing": True,
                        "limit": 100
                    },
                    "output_key": "summarization_queued"
                },
                "position_x": 250,
                "position_y": 390
            },
            {
                "node_id": "end",
                "node_type": "end",
                "config": {},
                "position_x": 250,
                "position_y": 510
            }
        ],
        "edges": [
            {"source_node_id": "start", "target_node_id": "get_source_stats"},
            {"source_node_id": "get_source_stats", "target_node_id": "list_docs"},
            {"source_node_id": "list_docs", "target_node_id": "queue_summaries"},
            {"source_node_id": "queue_summaries", "target_node_id": "end"}
        ]
    },

    # =========================================================================
    # Knowledge Base Health Check
    # =========================================================================
    {
        "template_id": "kb_health_check",
        "name": "Knowledge Base Health Check",
        "description": "Runs a comprehensive health check on the knowledge base: statistics, processing status, and generates a health report.",
        "category": "maintenance",
        "trigger_config": {
            "type": "schedule",
            "schedule": "0 6 * * *"  # Daily at 6 AM
        },
        "nodes": [
            {
                "node_id": "start",
                "node_type": "start",
                "config": {},
                "position_x": 250,
                "position_y": 50
            },
            {
                "node_id": "get_stats",
                "node_type": "tool",
                "builtin_tool": "get_knowledge_base_stats",
                "config": {
                    "input_mapping": {},
                    "output_key": "kb_stats"
                },
                "position_x": 250,
                "position_y": 150
            },
            {
                "node_id": "get_source_analytics",
                "node_type": "tool",
                "builtin_tool": "get_source_analytics",
                "config": {
                    "input_mapping": {},
                    "output_key": "source_health"
                },
                "position_x": 250,
                "position_y": 270
            },
            {
                "node_id": "get_kg_stats",
                "node_type": "tool",
                "builtin_tool": "get_kg_stats",
                "config": {
                    "input_mapping": {},
                    "output_key": "kg_stats"
                },
                "position_x": 250,
                "position_y": 390
            },
            {
                "node_id": "generate_health_report",
                "node_type": "tool",
                "builtin_tool": "generate_report",
                "config": {
                    "input_mapping": {
                        "report_type": "status",
                        "title": "Knowledge Base Health Report",
                        "sections": [
                            "Overview",
                            "Document Statistics",
                            "Source Health",
                            "Knowledge Graph Status",
                            "Recommendations"
                        ]
                    },
                    "output_key": "health_report"
                },
                "position_x": 250,
                "position_y": 510
            },
            {
                "node_id": "end",
                "node_type": "end",
                "config": {},
                "position_x": 250,
                "position_y": 630
            }
        ],
        "edges": [
            {"source_node_id": "start", "target_node_id": "get_stats"},
            {"source_node_id": "get_stats", "target_node_id": "get_source_analytics"},
            {"source_node_id": "get_source_analytics", "target_node_id": "get_kg_stats"},
            {"source_node_id": "get_kg_stats", "target_node_id": "generate_health_report"},
            {"source_node_id": "generate_health_report", "target_node_id": "end"}
        ]
    },

    # =========================================================================
    # Email Draft from Search
    # =========================================================================
    {
        "template_id": "email_from_search",
        "name": "Email Draft from Search",
        "description": "Searches the knowledge base and drafts an email based on the results. Input: search_query, email_subject, recipient, tone.",
        "category": "productivity",
        "trigger_config": {"type": "manual"},
        "nodes": [
            {
                "node_id": "start",
                "node_type": "start",
                "config": {},
                "position_x": 250,
                "position_y": 50
            },
            {
                "node_id": "search",
                "node_type": "tool",
                "builtin_tool": "search_documents",
                "config": {
                    "input_mapping": {
                        "query": "{{context.trigger_data.search_query}}",
                        "limit": 10
                    },
                    "output_key": "search_results"
                },
                "position_x": 250,
                "position_y": 150
            },
            {
                "node_id": "draft_email",
                "node_type": "tool",
                "builtin_tool": "draft_email",
                "config": {
                    "input_mapping": {
                        "subject": "{{context.trigger_data.email_subject}}",
                        "recipient": "{{context.trigger_data.recipient}}",
                        "search_query": "{{context.trigger_data.search_query}}",
                        "tone": "{{context.trigger_data.tone}}",
                        "length": "medium"
                    },
                    "output_key": "email_draft"
                },
                "position_x": 250,
                "position_y": 270
            },
            {
                "node_id": "end",
                "node_type": "end",
                "config": {},
                "position_x": 250,
                "position_y": 390
            }
        ],
        "edges": [
            {"source_node_id": "start", "target_node_id": "search"},
            {"source_node_id": "search", "target_node_id": "draft_email"},
            {"source_node_id": "draft_email", "target_node_id": "end"}
        ]
    },

    # =========================================================================
    # CPU OPTIMIZATION R&D TEMPLATES
    # =========================================================================

    # =========================================================================
    # CPU Optimization Research Monitor
    # =========================================================================
    {
        "template_id": "cpu_research_monitor",
        "name": "CPU Optimization Research Monitor",
        "description": "Weekly monitoring of arXiv for new research on CPU optimization topics (SIMD, vectorization, cache optimization, branch prediction, compiler optimizations). Ingests papers and generates a digest.",
        "category": "research",
        "trigger_config": {
            "type": "schedule",
            "schedule": "0 8 * * 1"  # Every Monday at 8 AM
        },
        "nodes": [
            {
                "node_id": "start",
                "node_type": "start",
                "config": {},
                "position_x": 250,
                "position_y": 50
            },
            {
                "node_id": "parallel_searches",
                "node_type": "parallel",
                "config": {
                    "output_key": "all_searches"
                },
                "position_x": 250,
                "position_y": 150
            },
            {
                "node_id": "search_simd",
                "node_type": "tool",
                "builtin_tool": "search_arxiv",
                "config": {
                    "input_mapping": {
                        "query": "SIMD vectorization CPU optimization",
                        "max_results": 5,
                        "sort_by": "submittedDate",
                        "sort_order": "descending"
                    },
                    "output_key": "simd_papers"
                },
                "position_x": 50,
                "position_y": 250
            },
            {
                "node_id": "search_cache",
                "node_type": "tool",
                "builtin_tool": "search_arxiv",
                "config": {
                    "input_mapping": {
                        "query": "cache optimization memory hierarchy performance",
                        "max_results": 5,
                        "sort_by": "submittedDate",
                        "sort_order": "descending"
                    },
                    "output_key": "cache_papers"
                },
                "position_x": 200,
                "position_y": 250
            },
            {
                "node_id": "search_compiler",
                "node_type": "tool",
                "builtin_tool": "search_arxiv",
                "config": {
                    "input_mapping": {
                        "query": "compiler optimization code generation performance",
                        "max_results": 5,
                        "sort_by": "submittedDate",
                        "sort_order": "descending"
                    },
                    "output_key": "compiler_papers"
                },
                "position_x": 350,
                "position_y": 250
            },
            {
                "node_id": "search_microarch",
                "node_type": "tool",
                "builtin_tool": "search_arxiv",
                "config": {
                    "input_mapping": {
                        "query": "microarchitecture branch prediction instruction level parallelism",
                        "max_results": 5,
                        "sort_by": "submittedDate",
                        "sort_order": "descending"
                    },
                    "output_key": "microarch_papers"
                },
                "position_x": 500,
                "position_y": 250
            },
            {
                "node_id": "generate_digest",
                "node_type": "tool",
                "builtin_tool": "generate_executive_summary",
                "config": {
                    "input_mapping": {
                        "topic": "Weekly CPU Optimization Research Digest",
                        "search_query": "CPU optimization SIMD vectorization cache compiler microarchitecture",
                        "max_length": 800,
                        "include_recommendations": True,
                        "include_metrics": False
                    },
                    "output_key": "research_digest"
                },
                "position_x": 250,
                "position_y": 400
            },
            {
                "node_id": "end",
                "node_type": "end",
                "config": {},
                "position_x": 250,
                "position_y": 520
            }
        ],
        "edges": [
            {"source_node_id": "start", "target_node_id": "parallel_searches"},
            {"source_node_id": "parallel_searches", "target_node_id": "search_simd"},
            {"source_node_id": "parallel_searches", "target_node_id": "search_cache"},
            {"source_node_id": "parallel_searches", "target_node_id": "search_compiler"},
            {"source_node_id": "parallel_searches", "target_node_id": "search_microarch"},
            {"source_node_id": "search_simd", "target_node_id": "generate_digest"},
            {"source_node_id": "search_cache", "target_node_id": "generate_digest"},
            {"source_node_id": "search_compiler", "target_node_id": "generate_digest"},
            {"source_node_id": "search_microarch", "target_node_id": "generate_digest"},
            {"source_node_id": "generate_digest", "target_node_id": "end"}
        ]
    },

    # =========================================================================
    # Performance Benchmark Analysis
    # =========================================================================
    {
        "template_id": "benchmark_analysis",
        "name": "Performance Benchmark Analysis",
        "description": "Analyzes benchmark results: searches for related documents, compares with historical data, identifies regressions/improvements, and generates a detailed performance report. Input: benchmark_name, search_query (benchmark context).",
        "category": "analysis",
        "trigger_config": {"type": "manual"},
        "nodes": [
            {
                "node_id": "start",
                "node_type": "start",
                "config": {},
                "position_x": 250,
                "position_y": 50
            },
            {
                "node_id": "search_benchmarks",
                "node_type": "tool",
                "builtin_tool": "faceted_search",
                "config": {
                    "input_mapping": {
                        "query": "{{context.trigger_data.search_query}} benchmark performance results",
                        "file_types": ["md", "txt", "json", "csv"],
                        "page_size": 20
                    },
                    "output_key": "benchmark_docs"
                },
                "position_x": 250,
                "position_y": 150
            },
            {
                "node_id": "search_baseline",
                "node_type": "tool",
                "builtin_tool": "search_documents",
                "config": {
                    "input_mapping": {
                        "query": "{{context.trigger_data.benchmark_name}} baseline reference performance",
                        "limit": 10
                    },
                    "output_key": "baseline_docs"
                },
                "position_x": 250,
                "position_y": 270
            },
            {
                "node_id": "get_trending",
                "node_type": "tool",
                "builtin_tool": "get_trending_topics",
                "config": {
                    "input_mapping": {
                        "days": 30,
                        "limit": 10
                    },
                    "output_key": "recent_trends"
                },
                "position_x": 250,
                "position_y": 390
            },
            {
                "node_id": "generate_report",
                "node_type": "tool",
                "builtin_tool": "generate_report",
                "config": {
                    "input_mapping": {
                        "report_type": "analysis",
                        "search_query": "{{context.trigger_data.search_query}} performance benchmark",
                        "title": "Performance Benchmark Analysis: {{context.trigger_data.benchmark_name}}",
                        "sections": [
                            "Executive Summary",
                            "Benchmark Configuration",
                            "Performance Results",
                            "Comparison with Baseline",
                            "Regression Analysis",
                            "Optimization Opportunities",
                            "Recommendations"
                        ]
                    },
                    "output_key": "benchmark_report"
                },
                "position_x": 250,
                "position_y": 510
            },
            {
                "node_id": "end",
                "node_type": "end",
                "config": {},
                "position_x": 250,
                "position_y": 630
            }
        ],
        "edges": [
            {"source_node_id": "start", "target_node_id": "search_benchmarks"},
            {"source_node_id": "search_benchmarks", "target_node_id": "search_baseline"},
            {"source_node_id": "search_baseline", "target_node_id": "get_trending"},
            {"source_node_id": "get_trending", "target_node_id": "generate_report"},
            {"source_node_id": "generate_report", "target_node_id": "end"}
        ]
    },

    # =========================================================================
    # Optimization Technique Deep Dive
    # =========================================================================
    {
        "template_id": "optimization_deep_dive",
        "name": "Optimization Technique Deep Dive",
        "description": "Comprehensive analysis of a specific optimization technique: searches internal docs, finds arXiv papers, analyzes similar techniques, and generates a technical brief. Input: technique_name (e.g., 'loop unrolling', 'cache blocking', 'SIMD vectorization').",
        "category": "research",
        "trigger_config": {"type": "manual"},
        "nodes": [
            {
                "node_id": "start",
                "node_type": "start",
                "config": {},
                "position_x": 250,
                "position_y": 50
            },
            {
                "node_id": "parallel_research",
                "node_type": "parallel",
                "config": {
                    "output_key": "research_results"
                },
                "position_x": 250,
                "position_y": 150
            },
            {
                "node_id": "search_internal",
                "node_type": "tool",
                "builtin_tool": "faceted_search",
                "config": {
                    "input_mapping": {
                        "query": "{{context.trigger_data.technique_name}} optimization implementation",
                        "page_size": 15
                    },
                    "output_key": "internal_docs"
                },
                "position_x": 100,
                "position_y": 250
            },
            {
                "node_id": "search_arxiv",
                "node_type": "tool",
                "builtin_tool": "literature_review_arxiv",
                "config": {
                    "input_mapping": {
                        "topic": "{{context.trigger_data.technique_name}} CPU performance optimization",
                        "max_papers": 8,
                        "ingest": True
                    },
                    "output_key": "arxiv_papers"
                },
                "position_x": 400,
                "position_y": 250
            },
            {
                "node_id": "find_related",
                "node_type": "tool",
                "builtin_tool": "search_documents",
                "config": {
                    "input_mapping": {
                        "query": "{{context.trigger_data.technique_name}} related techniques alternatives comparison",
                        "limit": 10
                    },
                    "output_key": "related_techniques"
                },
                "position_x": 250,
                "position_y": 370
            },
            {
                "node_id": "generate_documentation",
                "node_type": "tool",
                "builtin_tool": "generate_documentation",
                "config": {
                    "input_mapping": {
                        "topic": "{{context.trigger_data.technique_name}}",
                        "doc_type": "technical",
                        "search_query": "{{context.trigger_data.technique_name}} optimization",
                        "target_audience": "developers",
                        "include_examples": True
                    },
                    "output_key": "technical_doc"
                },
                "position_x": 250,
                "position_y": 490
            },
            {
                "node_id": "generate_brief",
                "node_type": "tool",
                "builtin_tool": "generate_executive_summary",
                "config": {
                    "input_mapping": {
                        "topic": "{{context.trigger_data.technique_name}} - Technical Deep Dive",
                        "search_query": "{{context.trigger_data.technique_name}}",
                        "max_length": 600,
                        "include_recommendations": True,
                        "include_metrics": True
                    },
                    "output_key": "executive_brief"
                },
                "position_x": 250,
                "position_y": 610
            },
            {
                "node_id": "end",
                "node_type": "end",
                "config": {},
                "position_x": 250,
                "position_y": 730
            }
        ],
        "edges": [
            {"source_node_id": "start", "target_node_id": "parallel_research"},
            {"source_node_id": "parallel_research", "target_node_id": "search_internal"},
            {"source_node_id": "parallel_research", "target_node_id": "search_arxiv"},
            {"source_node_id": "search_internal", "target_node_id": "find_related"},
            {"source_node_id": "search_arxiv", "target_node_id": "find_related"},
            {"source_node_id": "find_related", "target_node_id": "generate_documentation"},
            {"source_node_id": "generate_documentation", "target_node_id": "generate_brief"},
            {"source_node_id": "generate_brief", "target_node_id": "end"}
        ]
    },

    # =========================================================================
    # Architecture Decision Record (ADR) Generator
    # =========================================================================
    {
        "template_id": "adr_generator",
        "name": "Architecture Decision Record Generator",
        "description": "Generates an Architecture Decision Record (ADR) for optimization decisions. Searches for context, analyzes alternatives, and creates a structured ADR. Input: decision_title, context_query (what problem we're solving), alternatives (comma-separated list).",
        "category": "documentation",
        "trigger_config": {"type": "manual"},
        "nodes": [
            {
                "node_id": "start",
                "node_type": "start",
                "config": {},
                "position_x": 250,
                "position_y": 50
            },
            {
                "node_id": "search_context",
                "node_type": "tool",
                "builtin_tool": "faceted_search",
                "config": {
                    "input_mapping": {
                        "query": "{{context.trigger_data.context_query}}",
                        "page_size": 15
                    },
                    "output_key": "context_docs"
                },
                "position_x": 250,
                "position_y": 150
            },
            {
                "node_id": "search_alternatives",
                "node_type": "tool",
                "builtin_tool": "search_documents",
                "config": {
                    "input_mapping": {
                        "query": "{{context.trigger_data.alternatives}} comparison tradeoffs",
                        "limit": 15
                    },
                    "output_key": "alternatives_docs"
                },
                "position_x": 250,
                "position_y": 270
            },
            {
                "node_id": "get_similar",
                "node_type": "tool",
                "builtin_tool": "search_documents",
                "config": {
                    "input_mapping": {
                        "query": "architecture decision record ADR optimization",
                        "limit": 5
                    },
                    "output_key": "similar_adrs"
                },
                "position_x": 250,
                "position_y": 390
            },
            {
                "node_id": "generate_adr",
                "node_type": "tool",
                "builtin_tool": "generate_report",
                "config": {
                    "input_mapping": {
                        "report_type": "analysis",
                        "search_query": "{{context.trigger_data.context_query}} {{context.trigger_data.alternatives}}",
                        "title": "ADR: {{context.trigger_data.decision_title}}",
                        "sections": [
                            "Status",
                            "Context",
                            "Problem Statement",
                            "Decision Drivers",
                            "Considered Options",
                            "Decision Outcome",
                            "Pros and Cons",
                            "Technical Details",
                            "Consequences",
                            "References"
                        ]
                    },
                    "output_key": "adr_document"
                },
                "position_x": 250,
                "position_y": 510
            },
            {
                "node_id": "end",
                "node_type": "end",
                "config": {},
                "position_x": 250,
                "position_y": 630
            }
        ],
        "edges": [
            {"source_node_id": "start", "target_node_id": "search_context"},
            {"source_node_id": "search_context", "target_node_id": "search_alternatives"},
            {"source_node_id": "search_alternatives", "target_node_id": "get_similar"},
            {"source_node_id": "get_similar", "target_node_id": "generate_adr"},
            {"source_node_id": "generate_adr", "target_node_id": "end"}
        ]
    },

    # =========================================================================
    # Technical Specification Generator
    # =========================================================================
    {
        "template_id": "tech_spec_generator",
        "name": "Technical Specification Generator",
        "description": "Generates a technical specification document for an optimization feature. Input: feature_name, requirements_query (search for requirements docs), scope (e.g., 'kernel optimization', 'library API').",
        "category": "documentation",
        "trigger_config": {"type": "manual"},
        "nodes": [
            {
                "node_id": "start",
                "node_type": "start",
                "config": {},
                "position_x": 250,
                "position_y": 50
            },
            {
                "node_id": "search_requirements",
                "node_type": "tool",
                "builtin_tool": "faceted_search",
                "config": {
                    "input_mapping": {
                        "query": "{{context.trigger_data.requirements_query}} requirements specification",
                        "page_size": 20
                    },
                    "output_key": "requirements_docs"
                },
                "position_x": 250,
                "position_y": 150
            },
            {
                "node_id": "search_existing",
                "node_type": "tool",
                "builtin_tool": "search_documents",
                "config": {
                    "input_mapping": {
                        "query": "{{context.trigger_data.feature_name}} implementation design",
                        "limit": 10
                    },
                    "output_key": "existing_docs"
                },
                "position_x": 250,
                "position_y": 270
            },
            {
                "node_id": "search_patterns",
                "node_type": "tool",
                "builtin_tool": "search_documents",
                "config": {
                    "input_mapping": {
                        "query": "{{context.trigger_data.scope}} design pattern best practices",
                        "limit": 10
                    },
                    "output_key": "pattern_docs"
                },
                "position_x": 250,
                "position_y": 390
            },
            {
                "node_id": "generate_spec",
                "node_type": "tool",
                "builtin_tool": "generate_documentation",
                "config": {
                    "input_mapping": {
                        "topic": "{{context.trigger_data.feature_name}} Technical Specification",
                        "doc_type": "technical",
                        "search_query": "{{context.trigger_data.feature_name}} {{context.trigger_data.scope}}",
                        "target_audience": "developers",
                        "include_examples": True
                    },
                    "output_key": "tech_spec"
                },
                "position_x": 250,
                "position_y": 510
            },
            {
                "node_id": "end",
                "node_type": "end",
                "config": {},
                "position_x": 250,
                "position_y": 630
            }
        ],
        "edges": [
            {"source_node_id": "start", "target_node_id": "search_requirements"},
            {"source_node_id": "search_requirements", "target_node_id": "search_existing"},
            {"source_node_id": "search_existing", "target_node_id": "search_patterns"},
            {"source_node_id": "search_patterns", "target_node_id": "generate_spec"},
            {"source_node_id": "generate_spec", "target_node_id": "end"}
        ]
    },

    # =========================================================================
    # Competitive Analysis Pipeline
    # =========================================================================
    {
        "template_id": "competitive_analysis",
        "name": "Competitive Analysis Pipeline",
        "description": "Analyzes competitor solutions or alternative approaches. Searches for documentation, benchmarks, and generates a comparative analysis. Input: competitor_name, our_solution (name of our approach), focus_area (e.g., 'performance', 'features', 'architecture').",
        "category": "analysis",
        "trigger_config": {"type": "manual"},
        "nodes": [
            {
                "node_id": "start",
                "node_type": "start",
                "config": {},
                "position_x": 250,
                "position_y": 50
            },
            {
                "node_id": "parallel_search",
                "node_type": "parallel",
                "config": {
                    "output_key": "search_results"
                },
                "position_x": 250,
                "position_y": 150
            },
            {
                "node_id": "search_competitor",
                "node_type": "tool",
                "builtin_tool": "faceted_search",
                "config": {
                    "input_mapping": {
                        "query": "{{context.trigger_data.competitor_name}} {{context.trigger_data.focus_area}}",
                        "page_size": 15
                    },
                    "output_key": "competitor_docs"
                },
                "position_x": 100,
                "position_y": 250
            },
            {
                "node_id": "search_our_solution",
                "node_type": "tool",
                "builtin_tool": "faceted_search",
                "config": {
                    "input_mapping": {
                        "query": "{{context.trigger_data.our_solution}} {{context.trigger_data.focus_area}}",
                        "page_size": 15
                    },
                    "output_key": "our_docs"
                },
                "position_x": 400,
                "position_y": 250
            },
            {
                "node_id": "search_benchmarks",
                "node_type": "tool",
                "builtin_tool": "search_documents",
                "config": {
                    "input_mapping": {
                        "query": "{{context.trigger_data.competitor_name}} {{context.trigger_data.our_solution}} benchmark comparison",
                        "limit": 10
                    },
                    "output_key": "benchmark_docs"
                },
                "position_x": 250,
                "position_y": 370
            },
            {
                "node_id": "generate_analysis",
                "node_type": "tool",
                "builtin_tool": "generate_report",
                "config": {
                    "input_mapping": {
                        "report_type": "analysis",
                        "search_query": "{{context.trigger_data.competitor_name}} vs {{context.trigger_data.our_solution}} {{context.trigger_data.focus_area}}",
                        "title": "Competitive Analysis: {{context.trigger_data.our_solution}} vs {{context.trigger_data.competitor_name}}",
                        "sections": [
                            "Executive Summary",
                            "Overview of Solutions",
                            "Feature Comparison",
                            "Performance Analysis",
                            "Architecture Differences",
                            "Strengths and Weaknesses",
                            "Market Positioning",
                            "Recommendations"
                        ]
                    },
                    "output_key": "competitive_report"
                },
                "position_x": 250,
                "position_y": 490
            },
            {
                "node_id": "end",
                "node_type": "end",
                "config": {},
                "position_x": 250,
                "position_y": 610
            }
        ],
        "edges": [
            {"source_node_id": "start", "target_node_id": "parallel_search"},
            {"source_node_id": "parallel_search", "target_node_id": "search_competitor"},
            {"source_node_id": "parallel_search", "target_node_id": "search_our_solution"},
            {"source_node_id": "search_competitor", "target_node_id": "search_benchmarks"},
            {"source_node_id": "search_our_solution", "target_node_id": "search_benchmarks"},
            {"source_node_id": "search_benchmarks", "target_node_id": "generate_analysis"},
            {"source_node_id": "generate_analysis", "target_node_id": "end"}
        ]
    },

    # =========================================================================
    # Code Optimization Review
    # =========================================================================
    {
        "template_id": "code_optimization_review",
        "name": "Code Optimization Review",
        "description": "Reviews code or patches for optimization opportunities. Searches for best practices, analyzes patterns, and generates recommendations. Input: code_description (what the code does), optimization_goals (e.g., 'latency', 'throughput', 'memory').",
        "category": "analysis",
        "trigger_config": {"type": "manual"},
        "nodes": [
            {
                "node_id": "start",
                "node_type": "start",
                "config": {},
                "position_x": 250,
                "position_y": 50
            },
            {
                "node_id": "search_best_practices",
                "node_type": "tool",
                "builtin_tool": "faceted_search",
                "config": {
                    "input_mapping": {
                        "query": "{{context.trigger_data.code_description}} best practices optimization",
                        "page_size": 15
                    },
                    "output_key": "best_practices"
                },
                "position_x": 250,
                "position_y": 150
            },
            {
                "node_id": "search_patterns",
                "node_type": "tool",
                "builtin_tool": "search_documents",
                "config": {
                    "input_mapping": {
                        "query": "{{context.trigger_data.optimization_goals}} optimization patterns techniques",
                        "limit": 15
                    },
                    "output_key": "optimization_patterns"
                },
                "position_x": 250,
                "position_y": 270
            },
            {
                "node_id": "search_antipatterns",
                "node_type": "tool",
                "builtin_tool": "search_documents",
                "config": {
                    "input_mapping": {
                        "query": "{{context.trigger_data.code_description}} antipatterns pitfalls performance bugs",
                        "limit": 10
                    },
                    "output_key": "antipatterns"
                },
                "position_x": 250,
                "position_y": 390
            },
            {
                "node_id": "generate_review",
                "node_type": "tool",
                "builtin_tool": "generate_report",
                "config": {
                    "input_mapping": {
                        "report_type": "analysis",
                        "search_query": "{{context.trigger_data.code_description}} {{context.trigger_data.optimization_goals}} optimization",
                        "title": "Code Optimization Review: {{context.trigger_data.code_description}}",
                        "sections": [
                            "Summary",
                            "Current Implementation Analysis",
                            "Optimization Opportunities",
                            "Recommended Patterns",
                            "Anti-patterns to Avoid",
                            "Performance Considerations",
                            "Implementation Roadmap",
                            "Testing Recommendations"
                        ]
                    },
                    "output_key": "review_report"
                },
                "position_x": 250,
                "position_y": 510
            },
            {
                "node_id": "end",
                "node_type": "end",
                "config": {},
                "position_x": 250,
                "position_y": 630
            }
        ],
        "edges": [
            {"source_node_id": "start", "target_node_id": "search_best_practices"},
            {"source_node_id": "search_best_practices", "target_node_id": "search_patterns"},
            {"source_node_id": "search_patterns", "target_node_id": "search_antipatterns"},
            {"source_node_id": "search_antipatterns", "target_node_id": "generate_review"},
            {"source_node_id": "generate_review", "target_node_id": "end"}
        ]
    },

    # =========================================================================
    # Patent/Prior Art Research
    # =========================================================================
    {
        "template_id": "prior_art_research",
        "name": "Prior Art Research",
        "description": "Searches for prior art and existing research on an optimization technique for patent or novelty analysis. Input: technique_description, keywords (comma-separated key terms).",
        "category": "research",
        "trigger_config": {"type": "manual"},
        "nodes": [
            {
                "node_id": "start",
                "node_type": "start",
                "config": {},
                "position_x": 250,
                "position_y": 50
            },
            {
                "node_id": "search_internal",
                "node_type": "tool",
                "builtin_tool": "faceted_search",
                "config": {
                    "input_mapping": {
                        "query": "{{context.trigger_data.technique_description}} {{context.trigger_data.keywords}}",
                        "page_size": 20
                    },
                    "output_key": "internal_results"
                },
                "position_x": 250,
                "position_y": 150
            },
            {
                "node_id": "search_arxiv",
                "node_type": "tool",
                "builtin_tool": "literature_review_arxiv",
                "config": {
                    "input_mapping": {
                        "topic": "{{context.trigger_data.technique_description}}",
                        "max_papers": 15,
                        "ingest": True
                    },
                    "output_key": "arxiv_results"
                },
                "position_x": 250,
                "position_y": 270
            },
            {
                "node_id": "check_papers",
                "node_type": "condition",
                "config": {
                    "condition": {
                        "type": "comparison",
                        "left": "{{context.arxiv_results.papers}}",
                        "operator": "exists",
                        "right": True
                    },
                    "output_key": "has_papers"
                },
                "position_x": 250,
                "position_y": 390
            },
            {
                "node_id": "wait_ingestion",
                "node_type": "wait",
                "config": {
                    "wait_seconds": 20
                },
                "position_x": 250,
                "position_y": 490
            },
            {
                "node_id": "generate_review",
                "node_type": "tool",
                "builtin_tool": "generate_report",
                "config": {
                    "input_mapping": {
                        "report_type": "research",
                        "search_query": "{{context.trigger_data.technique_description}} {{context.trigger_data.keywords}}",
                        "title": "Prior Art Research: {{context.trigger_data.technique_description}}",
                        "sections": [
                            "Executive Summary",
                            "Technique Overview",
                            "Related Academic Research",
                            "Internal Documentation",
                            "Key Publications",
                            "Timeline of Developments",
                            "Novelty Assessment",
                            "Recommendations"
                        ]
                    },
                    "output_key": "prior_art_report"
                },
                "position_x": 250,
                "position_y": 610
            },
            {
                "node_id": "no_papers_report",
                "node_type": "tool",
                "builtin_tool": "generate_report",
                "config": {
                    "input_mapping": {
                        "report_type": "research",
                        "search_query": "{{context.trigger_data.technique_description}}",
                        "title": "Prior Art Research: {{context.trigger_data.technique_description}} (Limited External Sources)",
                        "sections": [
                            "Executive Summary",
                            "Technique Overview",
                            "Internal Documentation",
                            "Novelty Assessment",
                            "Recommendations"
                        ]
                    },
                    "output_key": "prior_art_report"
                },
                "position_x": 450,
                "position_y": 490
            },
            {
                "node_id": "end",
                "node_type": "end",
                "config": {},
                "position_x": 250,
                "position_y": 730
            },
            {
                "node_id": "end_no_papers",
                "node_type": "end",
                "config": {},
                "position_x": 450,
                "position_y": 610
            }
        ],
        "edges": [
            {"source_node_id": "start", "target_node_id": "search_internal"},
            {"source_node_id": "search_internal", "target_node_id": "search_arxiv"},
            {"source_node_id": "search_arxiv", "target_node_id": "check_papers"},
            {"source_node_id": "check_papers", "target_node_id": "wait_ingestion", "source_handle": "true"},
            {"source_node_id": "check_papers", "target_node_id": "no_papers_report", "source_handle": "false"},
            {"source_node_id": "wait_ingestion", "target_node_id": "generate_review"},
            {"source_node_id": "generate_review", "target_node_id": "end"},
            {"source_node_id": "no_papers_report", "target_node_id": "end_no_papers"}
        ]
    },

    # =========================================================================
    # Research Presentation Generator
    # =========================================================================
    {
        "template_id": "research_presentation",
        "name": "Research Presentation Generator",
        "description": "Generates a professional presentation from a research topic. Searches the knowledge base and optionally arXiv for papers, then creates slides with diagrams. Input: topic, slide_count (5-20), include_arxiv (true/false).",
        "category": "productivity",
        "trigger_config": {"type": "manual"},
        "nodes": [
            {
                "node_id": "start",
                "node_type": "start",
                "config": {},
                "position_x": 250,
                "position_y": 50
            },
            {
                "node_id": "search_kb",
                "node_type": "tool",
                "builtin_tool": "faceted_search",
                "config": {
                    "input_mapping": {
                        "query": "{{context.trigger_data.topic}}",
                        "page_size": 20
                    },
                    "output_key": "kb_results"
                },
                "position_x": 250,
                "position_y": 150
            },
            {
                "node_id": "check_arxiv",
                "node_type": "condition",
                "config": {
                    "condition": {
                        "type": "comparison",
                        "left": "{{context.trigger_data.include_arxiv}}",
                        "operator": "eq",
                        "right": True
                    },
                    "output_key": "should_search_arxiv"
                },
                "position_x": 250,
                "position_y": 270
            },
            {
                "node_id": "search_arxiv",
                "node_type": "tool",
                "builtin_tool": "search_arxiv",
                "config": {
                    "input_mapping": {
                        "query": "{{context.trigger_data.topic}}",
                        "max_results": 5,
                        "sort_by": "relevance"
                    },
                    "output_key": "arxiv_results"
                },
                "position_x": 100,
                "position_y": 370
            },
            {
                "node_id": "skip_arxiv",
                "node_type": "tool",
                "builtin_tool": "get_knowledge_base_stats",
                "config": {
                    "input_mapping": {},
                    "output_key": "kb_stats"
                },
                "position_x": 400,
                "position_y": 370
            },
            {
                "node_id": "generate_summary",
                "node_type": "tool",
                "builtin_tool": "generate_executive_summary",
                "config": {
                    "input_mapping": {
                        "topic": "{{context.trigger_data.topic}}",
                        "search_query": "{{context.trigger_data.topic}}",
                        "max_length": 500,
                        "include_recommendations": True,
                        "include_metrics": False
                    },
                    "output_key": "executive_summary"
                },
                "position_x": 250,
                "position_y": 490
            },
            {
                "node_id": "generate_presentation",
                "node_type": "tool",
                "builtin_tool": "generate_report",
                "config": {
                    "input_mapping": {
                        "report_type": "presentation",
                        "search_query": "{{context.trigger_data.topic}}",
                        "title": "{{context.trigger_data.topic}}",
                        "sections": [
                            "Title & Overview",
                            "Background & Motivation",
                            "Key Concepts",
                            "Current State of the Art",
                            "Technical Deep Dive",
                            "Applications & Use Cases",
                            "Challenges & Limitations",
                            "Future Directions",
                            "Conclusions",
                            "References"
                        ]
                    },
                    "output_key": "presentation_content"
                },
                "position_x": 250,
                "position_y": 610
            },
            {
                "node_id": "end",
                "node_type": "end",
                "config": {},
                "position_x": 250,
                "position_y": 730
            }
        ],
        "edges": [
            {"source_node_id": "start", "target_node_id": "search_kb"},
            {"source_node_id": "search_kb", "target_node_id": "check_arxiv"},
            {"source_node_id": "check_arxiv", "target_node_id": "search_arxiv", "source_handle": "true"},
            {"source_node_id": "check_arxiv", "target_node_id": "skip_arxiv", "source_handle": "false"},
            {"source_node_id": "search_arxiv", "target_node_id": "generate_summary"},
            {"source_node_id": "skip_arxiv", "target_node_id": "generate_summary"},
            {"source_node_id": "generate_summary", "target_node_id": "generate_presentation"},
            {"source_node_id": "generate_presentation", "target_node_id": "end"}
        ]
    },

    # =========================================================================
    # Quick Paper Brief
    # =========================================================================
    {
        "template_id": "quick_paper_brief",
        "name": "Quick Paper Brief",
        "description": "Generates a quick technical brief from a specific arXiv paper. Fetches the paper, summarizes it, extracts key points, and creates a one-page brief. Input: arxiv_id (e.g., '2401.12345').",
        "category": "research",
        "trigger_config": {"type": "manual"},
        "nodes": [
            {
                "node_id": "start",
                "node_type": "start",
                "config": {},
                "position_x": 250,
                "position_y": 50
            },
            {
                "node_id": "fetch_paper",
                "node_type": "tool",
                "builtin_tool": "search_arxiv",
                "config": {
                    "input_mapping": {
                        "query": "id:{{context.trigger_data.arxiv_id}}",
                        "max_results": 1
                    },
                    "output_key": "paper_info"
                },
                "position_x": 250,
                "position_y": 150
            },
            {
                "node_id": "search_related",
                "node_type": "tool",
                "builtin_tool": "search_documents",
                "config": {
                    "input_mapping": {
                        "query": "{{context.paper_info.title}}",
                        "limit": 5
                    },
                    "output_key": "related_docs"
                },
                "position_x": 250,
                "position_y": 270
            },
            {
                "node_id": "generate_brief",
                "node_type": "tool",
                "builtin_tool": "generate_executive_summary",
                "config": {
                    "input_mapping": {
                        "topic": "{{context.paper_info.title}}",
                        "search_query": "{{context.paper_info.title}}",
                        "max_length": 400,
                        "include_recommendations": True,
                        "include_metrics": False
                    },
                    "output_key": "paper_brief"
                },
                "position_x": 250,
                "position_y": 390
            },
            {
                "node_id": "end",
                "node_type": "end",
                "config": {},
                "position_x": 250,
                "position_y": 510
            }
        ],
        "edges": [
            {"source_node_id": "start", "target_node_id": "fetch_paper"},
            {"source_node_id": "fetch_paper", "target_node_id": "search_related"},
            {"source_node_id": "search_related", "target_node_id": "generate_brief"},
            {"source_node_id": "generate_brief", "target_node_id": "end"}
        ]
    }
]


def get_template_by_id(template_id: str) -> Dict[str, Any] | None:
    """Get a workflow template by its ID."""
    for template in WORKFLOW_TEMPLATES:
        if template["template_id"] == template_id:
            return template
    return None


def get_templates_by_category(category: str) -> List[Dict[str, Any]]:
    """Get all templates in a category."""
    return [t for t in WORKFLOW_TEMPLATES if t.get("category") == category]


def list_template_categories() -> List[str]:
    """List all available template categories."""
    categories = set(t.get("category", "other") for t in WORKFLOW_TEMPLATES)
    return sorted(categories)


def get_template_summary() -> List[Dict[str, Any]]:
    """Get a summary of all available templates."""
    return [
        {
            "template_id": t["template_id"],
            "name": t["name"],
            "description": t["description"],
            "category": t.get("category", "other"),
            "trigger_type": t.get("trigger_config", {}).get("type", "manual"),
            "node_count": len(t.get("nodes", [])),
        }
        for t in WORKFLOW_TEMPLATES
    ]
