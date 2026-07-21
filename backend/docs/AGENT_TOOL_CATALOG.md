# Agent Tool Catalog

Complete reference for all 167 agent tools available in KnowledgeDBChat. Tools are grouped by functional category with parameters, classifications, and usage examples.

**Legend**: `*` = required parameter | Effects: `read`/`write` | Network: `none`/`egress` | Cost: `low`/`medium`/`high`

---

## 1. Document & Search (26 tools)

Core knowledge base operations — search, read, create, delete, and tag documents.

### search_documents
Search for documents using semantic vector search.
```
Params: query* (string), limit (integer, default 5)
Effects: read | Network: none | Cost: low
```
**Example:** `{"query": "transformer attention mechanisms", "limit": 10}`

### get_document_details
Retrieve full metadata for a document by ID.
```
Params: document_id* (string)
Effects: read | Network: none | Cost: low
```

### read_document_content
Read raw content of a document, optionally with chunk boundaries.
```
Params: document_id* (string), max_length (integer), include_chunks (boolean)
Effects: read | Network: none | Cost: low
```

### summarize_document
Generate or retrieve an LLM summary of a document.
```
Params: document_id* (string), force_regenerate (boolean)
Effects: read | Network: none | Cost: low
```

### find_similar_documents
Find documents similar to a given document using vector similarity.
```
Params: document_id* (string), limit (integer)
Effects: read | Network: none | Cost: low
```

### compare_documents
Compare two documents by content, structure, or methodology.
```
Params: document_id_1* (string), document_id_2* (string), comparison_type (string)
Effects: read | Network: none | Cost: low
```

### answer_question
Answer a question using RAG (retrieval-augmented generation).
```
Params: question* (string), max_sources (integer)
Effects: read | Network: none | Cost: low
```
**Example:** `{"question": "What are the key differences between GPT-4 and Claude?", "max_sources": 5}`

### create_document_from_text
Create a new document from raw text content.
```
Params: title* (string), content* (string), tags (array)
Effects: write | Network: none | Cost: low
```

### delete_document / batch_delete_documents
Delete one or multiple documents by ID.
```
delete_document — Params: document_id* (string), confirm (boolean)
batch_delete_documents — Params: document_ids* (array), confirm (boolean)
Effects: write | Network: none | Cost: low
```

### update_document_tags
Add or replace tags on a document.
```
Params: document_id* (string), tags* (array), action (string: "add"|"replace")
Effects: write | Network: none | Cost: low
```

### list_recent_documents
List recently added/modified documents.
```
Params: limit (integer)
Effects: read | Network: none | Cost: low
```

### list_document_sources / list_documents_by_source
Browse data sources and their documents.
```
list_document_sources — Params: active_only (boolean)
list_documents_by_source — Params: source_id (string), source_name (string), source_type (string), limit (integer), offset (integer)
Effects: read | Network: none | Cost: low
```

### search_documents_by_author
Find documents by author name.
```
Params: author* (string), match_type (string), limit (integer)
Effects: read | Network: none | Cost: low
```

### search_by_tags / search_documents_by_tag
Find documents matching specific tags.
```
Params: tags* (array), match_all (boolean), limit (integer)
Effects: read | Network: none | Cost: low
```

### list_all_tags
List all unique tags in the knowledge base.
```
Params: (none)
Effects: read | Network: none | Cost: low
```

### get_knowledge_base_stats
Get summary statistics about the knowledge base.
```
Params: (none)
Effects: read | Network: none | Cost: low
```

### batch_summarize_documents
Generate summaries for multiple documents at once.
```
Params: document_ids* (array), force_regenerate (boolean)
Effects: read | Network: none | Cost: low
```

### request_file_upload
Request a file upload from the user (interactive chat only).
```
Params: suggested_title (string), suggested_tags (array)
Effects: read | Network: none | Cost: low
```

---

## 2. Web & URL Ingestion (5 tools)

Fetch, scrape, and ingest web content.

### web_scrape
Fetch a web page and extract readable text. Supports crawling.
```
Params: url* (string), follow_links (boolean), max_pages (integer, max 25), max_depth (integer, max 5), same_domain_only (boolean), include_links (boolean), allow_private_networks (boolean), max_content_chars (integer)
Effects: read | Network: egress | Cost: low | PII: medium
```
**Example:** `{"url": "https://arxiv.org/abs/2301.00001", "max_content_chars": 50000}`

### ingest_url
Scrape a URL and save it as a document in the knowledge base. Supports YouTube transcription.
```
Params: url* (string), title (string), tags (array), ingest_mode (string), youtube_audio_only (boolean), follow_links (boolean), max_pages (integer), max_depth (integer), same_domain_only (boolean), one_document_per_page (boolean), allow_private_networks (boolean), max_content_chars (integer)
Effects: write | Network: egress | Cost: low | PII: medium
```
**Example:** `{"url": "https://example.com/research-paper", "title": "Research Paper 2024", "tags": ["ml", "research"]}`

### fetch_url_content
Fetch and extract text from a URL without ingesting.
```
Params: url* (string), max_chars (integer, default 50000, max 100000)
Effects: read | Network: egress | Cost: low
```

### summarize_url
Fetch URL content and produce an LLM-generated summary with optional focus.
```
Params: url* (string), focus (string)
Effects: read | Network: egress | Cost: medium
```
**Example:** `{"url": "https://blog.example.com/post", "focus": "security implications"}`

### search_web
Search the web using DuckDuckGo (no API key needed).
```
Params: query* (string), max_results (integer, default 5, max 10)
Effects: read | Network: egress | Cost: medium
```
**Example:** `{"query": "latest advances in protein folding 2026", "max_results": 5}`

---

## 3. Knowledge Graph (16 tools)

Query and manipulate the entity-relationship knowledge graph.

### search_entities
Search for entities by name or type.
```
Params: query* (string), entity_type (string), limit (integer)
Effects: read | Network: none | Cost: low
```

### get_entity_relationships
Get relationships for a specific entity.
```
Params: entity_id* (string), limit (integer)
Effects: read | Network: none | Cost: low
```

### get_entity_mentions
Get source mentions (provenance) for an entity.
```
Params: entity_id* (string), limit (integer), offset (integer)
Effects: read | Network: none | Cost: low
```

### find_documents_by_entity
Find documents associated with a specific entity.
```
Params: entity_id* (string), limit (integer)
Effects: read | Network: none | Cost: low
```

### get_document_knowledge_graph
Get the KG subgraph for a specific document.
```
Params: document_id* (string)
Effects: read | Network: none | Cost: low
```

### get_global_knowledge_graph
Query the full knowledge graph with filters.
```
Params: entity_types (array), relation_types (array), min_confidence (number), min_mentions (integer), limit_nodes (integer), limit_edges (integer), search (string)
Effects: read | Network: none | Cost: low
```

### get_kg_stats
Get knowledge graph statistics (entity/relationship counts).
```
Params: (none)
Effects: read | Network: none | Cost: low
```

### rebuild_document_knowledge_graph
Re-extract entities and relationships from a document.
```
Params: document_id* (string)
Effects: write | Network: none | Cost: low
```

### merge_entities
Merge two entities into one (deduplication).
```
Params: source_id* (string), target_id* (string)
Effects: write | Network: none | Cost: low
```

### delete_entity
Delete an entity with confirmation.
```
Params: entity_id* (string), confirm_name* (string)
Effects: write | Network: none | Cost: low
```

### link_entities
Create a relationship between two entities.
```
Params: relationship_type* (string), source_entity_id (string), target_entity_id (string), source_name (string), target_name (string), evidence (string), confidence (number)
Effects: read | Network: none | Cost: low
```

### query_kg_entities *(autonomous)*
Search entities in the knowledge graph by name or keyword.
```
Params: query* (string), entity_type (string), limit (integer, default 20, max 100)
Effects: read | Network: none | Cost: low
```
**Example:** `{"query": "transformer", "entity_type": "concept", "limit": 10}`

### get_entity_context *(autonomous)*
Get an entity with all its relationships and connected entities.
```
Params: entity_id* (string)
Effects: read | Network: none | Cost: low
```

### create_kg_entity *(autonomous)*
Create a new entity in the knowledge graph.
```
Params: name* (string), entity_type* (string: person|org|location|product|concept|technology|event|other), description (string)
Effects: write | Network: none | Cost: low
```
**Example:** `{"name": "GPT-4", "entity_type": "technology", "description": "Large language model by OpenAI"}`

### create_kg_relationship *(autonomous)*
Create a relationship between two existing entities.
```
Params: source_entity_id* (string), target_entity_id* (string), relation_type* (string), confidence (number, default 0.8), evidence (string)
Effects: write | Network: none | Cost: low
```

### query_kg_graph *(autonomous)*
Query the global knowledge graph with filters for visualization or analysis.
```
Params: entity_types (array), relation_types (array), min_confidence (number), search (string), limit_nodes (integer, default 50, max 200)
Effects: read | Network: none | Cost: low
```

---

## 4. ArXiv & Research (16 tools)

Scientific paper discovery, ingestion, and analysis.

### search_arxiv
Search ArXiv with full query syntax.
```
Params: query* (string), start (integer), max_results (integer), sort_by (string), sort_order (string)
Effects: read | Network: egress | Cost: low
```
**Example:** `{"query": "all:transformers AND cat:cs.CL", "max_results": 10, "sort_by": "submittedDate"}`

### ingest_arxiv_papers
Create an ArXiv source and bulk-ingest papers.
```
Params: name (string), search_queries (array), paper_ids (array), categories (array), max_results (integer), sort_by (string), auto_sync (boolean)
Effects: read | Network: egress | Cost: low
```

### literature_review_arxiv
Combined search + optional ingest + review generation.
```
Params: topic* (string), query (string), categories (array), max_papers (integer), ingest (boolean), sort_by (string)
Effects: read | Network: egress | Cost: low
```

### ingest_paper_by_id / batch_ingest_papers
Ingest specific papers by ArXiv ID.
```
ingest_paper_by_id — Params: arxiv_id* (string), add_to_reading_list (string), extract_insights (boolean)
batch_ingest_papers — Params: arxiv_ids* (array), source_name (string), add_to_reading_list (string)
Effects: read | Network: none | Cost: low
```

### monitor_arxiv_topic
Monitor ArXiv for new papers on a topic.
```
Params: topic* (string), query (string), categories (array), since_days (integer), max_results (integer)
Effects: read | Network: none | Cost: low
```

### extract_paper_insights
Extract key insights, methods, and entities from a paper.
```
Params: document_id* (string), focus_areas (array), extract_entities (boolean)
Effects: read | Network: none | Cost: low
```

### find_related_papers
Find papers related to a given paper.
```
Params: document_id (string), arxiv_id (string), relation_type (string), limit (integer), search_external (boolean)
Effects: read | Network: none | Cost: low
```

### compare_methodologies
Compare research methodologies across papers.
```
Params: document_ids* (array), comparison_aspects (array), output_format (string)
Effects: read | Network: none | Cost: low
```

### identify_research_gaps
Identify gaps in a research corpus.
```
Params: document_ids (array), source_id (string), topic (string), gap_types (array)
Effects: read | Network: none | Cost: low
```

### build_research_graph
Build a research graph from documents.
```
Params: document_ids (array), source_id (string), focus_on (array), include_relationships (boolean)
Effects: read | Network: none | Cost: low
```

### analyze_document_cluster
Analyze a group of documents for themes and timelines.
```
Params: document_ids* (array), analysis_type (string), extract_timeline (boolean)
Effects: read | Network: none | Cost: low
```

### create_synthesis_document
Generate a synthesis document from findings and sources.
```
Params: title* (string), topic* (string), findings_to_include (array), document_ids (array), sections (array), format (string)
Effects: read | Network: none | Cost: low
```

### add_to_reading_list / get_reading_lists
Manage reading lists.
```
add_to_reading_list — Params: list_name* (string), items* (array)
get_reading_lists — Params: list_name (string), include_items (boolean)
Effects: read | Network: none | Cost: low
```

### save_research_finding / get_research_findings
Save and retrieve research findings.
```
save_research_finding — Params: title* (string), content* (string), category* (string), source_document_ids (array), confidence (number), tags (array)
get_research_findings — Params: category (string), tags (array), min_confidence (number), limit (integer)
Effects: read | Network: none | Cost: low
```

---

## 5. Content Generation (8 tools)

Generate reports, summaries, presentations, and other documents.

### generate_report
Generate a structured report.
```
Params: report_type* (string), document_ids (array), search_query (string), title (string), sections (array)
Effects: read | Network: none | Cost: medium
```

### generate_executive_summary
Generate an executive summary from documents.
```
Params: document_ids (array), search_query (string), topic (string), max_length (integer), include_recommendations (boolean), include_metrics (boolean)
Effects: read | Network: none | Cost: low
```

### generate_documentation
Generate technical documentation on a topic.
```
Params: topic* (string), doc_type (string), document_ids (array), search_query (string), target_audience (string), include_examples (boolean)
Effects: read | Network: none | Cost: low
```

### generate_meeting_notes
Generate meeting notes from a transcript.
```
Params: transcript (string), document_ids (array), meeting_title (string), participants (array), include_action_items (boolean), include_decisions (boolean)
Effects: read | Network: none | Cost: low
```

### draft_email
Draft an email using knowledge base context.
```
Params: subject* (string), recipient (string), context (string), document_ids (array), search_query (string), tone (string), length (string)
Effects: read | Network: none | Cost: low
```

### generate_diagram
Generate a Mermaid diagram from documents or descriptions.
```
Params: diagram_type (string), source* (string), document_ids (array), search_query (string), description (string), focus (string), detail_level (string)
Effects: read | Network: none | Cost: low
```

### generate_research_presentation
Generate a slide presentation from research.
```
Params: title* (string), topic* (string), document_ids (array), slide_count (integer), style (string), include_diagrams (boolean)
Effects: read | Network: none | Cost: low
```

### generate_slides_for_source / generate_literature_review_for_source
Generate slides or a literature review from a data source.
```
generate_slides_for_source — Params: source_id* (string), title (string), topic (string), slide_count (integer), style (string), include_diagrams (boolean), prefer_review_document (boolean)
generate_literature_review_for_source — Params: source_id* (string), topic (string)
Effects: read | Network: none | Cost: low
```

---

## 6. Analytics & Search Enhancement (8 tools)

Advanced search, statistics, and data export.

### faceted_search
Search with structured filters (source, type, date, tags).
```
Params: query* (string), page (integer), page_size (integer), filters (object)
Effects: read | Network: none | Cost: low
```

### search_with_filters
Search with source, date, tag, and relevance filters.
```
Params: query* (string), source_ids (array), file_types (array), date_from (string), date_to (string), tags (array), min_relevance (number), limit (integer)
Effects: read | Network: none | Cost: low
```

### get_search_suggestions / get_related_searches
Search autocomplete and related queries.
```
get_search_suggestions — Params: partial_query* (string), limit (integer)
get_related_searches — Params: query* (string), limit (integer)
Effects: read | Network: none | Cost: low
```

### get_collection_statistics / get_source_analytics / get_trending_topics
Analytics on document collections.
```
get_collection_statistics — Params: source_id (string), tag (string), date_from (string), date_to (string)
get_source_analytics — Params: source_id (string)
get_trending_topics — Params: days (integer), limit (integer)
Effects: read | Network: none | Cost: low
```

### generate_chart_data
Generate data for charts and visualizations.
```
Params: chart_type (string), metric* (string), group_by* (string), date_from (string), date_to (string), limit (integer)
Effects: read | Network: none | Cost: low
```

### export_data
Export documents and metadata in bulk.
```
Params: format (string), source_id (string), tag (string), include_content (boolean), include_chunks (boolean), limit (integer)
Effects: read | Network: none | Cost: low
```

---

## 7. Workflows & Custom Tools (8 tools)

Workflow management and custom tool execution.

### run_workflow
Run a named workflow (chat-level).
```
Params: workflow_name (string), workflow_id (string), inputs (object)
Effects: write | Network: none | Cost: low
```

### list_available_workflows
List DAG workflows available to the current user (agent-level).
```
Params: is_active (boolean, default true)
Effects: read | Network: none | Cost: low
```
**Example:** `{"is_active": true}`

### execute_workflow
Launch a DAG workflow by ID (agent-level).
```
Params: workflow_id* (string), trigger_data (object), inputs (object)
Effects: write | Network: none | Cost: medium
```
**Example:** `{"workflow_id": "550e8400-e29b-41d4-a716-446655440000", "inputs": {"topic": "quantum computing"}}`

### get_workflow_status
Check the status of a workflow execution.
```
Params: execution_id* (string)
Effects: read | Network: none | Cost: low
```

### propose_workflow_from_description / create_workflow_from_description
Generate a workflow from a natural language description.
```
Params: description* (string), name (string), is_active (boolean), trigger_config (object), synthesize_custom_tools (boolean), preferred_tool_type (string), expose_workflow_as_tool (boolean), workflow_tool_name (string)
Effects: read | Network: none | Cost: low
```

### list_workflows
List workflows (chat-level).
```
Params: active_only (boolean)
Effects: read | Network: none | Cost: low
```

### run_custom_tool / list_custom_tools
Execute or list user-defined custom tools.
```
run_custom_tool — Params: tool_name* (string), inputs (object)
list_custom_tools — Params: tool_type (string)
Effects: write/read | Network: none | Cost: low | PII: medium
```

---

## 8. Reasoning (4 tools)

Structured thinking tools for agent self-reflection and hypothesis testing.

### reflect
Record a reflection on a topic with self-assessment.
```
Params: topic* (string), assessment* (string), blind_spots (array), suggested_corrections (array)
Effects: read | Network: none | Cost: low
```
**Example:** `{"topic": "initial research approach", "assessment": "Too focused on recent papers, missing foundational work", "blind_spots": ["Pre-2020 literature"], "suggested_corrections": ["Broaden date range"]}`

### hypothesize
Create or update a hypothesis with testable predictions.
```
Params: hypothesis* (string), rationale (string), testable_predictions (array), status (string), hypothesis_id (string)
Effects: read | Network: none | Cost: low
```
**Example:** `{"hypothesis": "Transformer models perform better on long sequences due to attention", "rationale": "Attention captures long-range dependencies", "testable_predictions": ["Performance gap increases with sequence length"]}`

### weigh_evidence
Evaluate evidence for/against a claim or hypothesis.
```
Params: claim* (string), hypothesis_id (string), evidence_for (array), evidence_against (array), verdict* (string)
Effects: read | Network: none | Cost: low
```

### critique_plan
Self-critique a plan by identifying weaknesses and missing steps.
```
Params: plan_summary* (string), weaknesses* (array), missing_steps (array), assumptions_challenged (array), severity (string)
Effects: read | Network: none | Cost: low
```

---

## 9. Multi-Agent Coordination (6 tools)

Delegate work, share findings, and communicate between agents.

### delegate_subtask
Spawn a child agent job for a specific subtask. Max depth 3, max 5 children per parent.
```
Params: name* (string), goal* (string), job_type (string: research|analysis|synthesis|custom), config (object), max_iterations (integer), share_findings (boolean), wait (boolean)
Effects: write | Network: none | Cost: medium
```
**Example:** `{"name": "Literature Survey", "goal": "Find all papers on RLHF published in 2025", "job_type": "research", "share_findings": true}`

### wait_for_subtask
Poll for a delegated subtask's completion.
```
Params: subtask_job_id* (string), timeout_seconds (integer, max 120)
Effects: read | Network: none | Cost: low
```

### share_findings
Push findings to sibling agent jobs (same parent).
```
Params: findings* (array of {title, content, category}), target_job_ids (array)
Effects: write | Network: none | Cost: low
```

### request_review
Request review from a peer agent or human operator.
```
Params: content_to_review* (string), review_type (string: peer_agent|human), review_criteria (array), reviewer_job_id (string)
Effects: write | Network: none | Cost: medium
```

### send_message_to_agent
Send a direct message to any agent job owned by the same user.
```
Params: target_job_id* (string), message* (string), category (string)
Effects: write | Network: none | Cost: low
```
**Example:** `{"target_job_id": "550e8400-...", "message": "Found 3 relevant papers, check shared findings", "category": "update"}`

### read_agent_messages
Read messages sent to this agent by other agents.
```
Params: since_index (integer, default 0)
Effects: read | Network: none | Cost: low
```

---

## 10. Code Execution (3 tools)

Execute Python code and data pipelines.

### execute_python
Run Python code in a sandboxed environment.
```
Params: code* (string), timeout_seconds (integer)
Effects: read | Network: none | Cost: medium | PII: medium
```
**Example:** `{"code": "import pandas as pd\ndf = pd.DataFrame({'a': [1,2,3]})\nprint(df.describe())", "timeout_seconds": 30}`

### execute_data_pipeline
Run a data processing pipeline with input documents.
```
Params: code* (string), input_data (object), input_document_ids (array), timeout_seconds (integer)
Effects: write | Network: none | Cost: high | PII: medium
```

### write_and_run_script
Write a script file with dependencies and execute it.
```
Params: script_name* (string), script_content* (string), requirements (array), arguments (array), timeout_seconds (integer), input_data (object)
Effects: write | Network: none | Cost: high | PII: high
```

---

## 11. Coding Workspace (8 tools)

Clone repos, read/write files, run commands in isolated workspaces.

### clone_and_index_repo
Clone a git repository and index it for code search.
```
Params: source_id (string), repo_url (string), branch (string)
Effects: read | Network: egress | Cost: medium
```
**Example:** `{"repo_url": "https://github.com/org/repo.git", "branch": "main"}`

### browse_repo_files
List files in a workspace with optional glob filtering.
```
Params: workspace_id (string), path (string), glob_pattern (string), max_results (integer)
Effects: read | Network: none | Cost: low
```

### read_file
Read file content with optional line range.
```
Params: path* (string), workspace_id (string), start_line (integer), end_line (integer), max_chars (integer)
Effects: read | Network: none | Cost: low
```

### write_file
Write or create a file in the workspace.
```
Params: path* (string), content* (string), workspace_id (string), create_dirs (boolean)
Effects: write | Network: none | Cost: low
```

### apply_patch
Apply a unified diff patch to workspace files.
```
Params: diff* (string), workspace_id (string), dry_run (boolean)
Effects: write | Network: none | Cost: low
```
**Example:** `{"diff": "--- a/main.py\n+++ b/main.py\n@@ -1 +1 @@\n-print('old')\n+print('new')", "dry_run": false}`

### run_command
Execute a shell command in the workspace.
```
Params: command* (string), workspace_id (string), timeout_seconds (integer), env (object)
Effects: write | Network: none | Cost: high | PII: medium
```

### search_code
Search workspace files by regex pattern.
```
Params: pattern* (string), workspace_id (string), path (string), file_glob (string), max_results (integer), context_lines (integer)
Effects: read | Network: none | Cost: low | PII: medium
```

### get_workspace_status
Show workspace change status (modified/added/deleted files).
```
Params: workspace_id (string), show_diff_summary (boolean)
Effects: read | Network: none | Cost: low
```

---

## 12. Symbol-Aware Code Retrieval (3 tools)

Navigate code by function/class symbols.

### retrieve_repo_symbols
Search for code symbols (functions, classes) by keyword.
```
Params: query* (string), workspace_id (string), language_filter (string), max_results (integer)
Effects: read | Network: none | Cost: low
```
**Example:** `{"query": "UserService authenticate", "language_filter": "python", "max_results": 10}`

### get_symbol_context
Get the source code surrounding a specific symbol in a file.
```
Params: symbol_name* (string), file_path* (string), workspace_id (string)
Effects: read | Network: none | Cost: low
```

### find_tests_for_symbol
Find test files and test functions that cover a given symbol.
```
Params: symbol_name* (string), workspace_id (string)
Effects: read | Network: none | Cost: low
```

---

## 13. Document Authoring (8 tools)

Plan, write, revise, and export structured documents.

### plan_document
Create a document plan with title, sections, and style.
```
Params: title* (string), abstract (string), doc_type (string), sections* (array), style (string)
Effects: read | Network: none | Cost: low
```
**Example:** `{"title": "Survey of LLM Agents", "doc_type": "survey", "sections": [{"id": "intro", "title": "Introduction"}, {"id": "methods", "title": "Methods"}], "style": "academic"}`

### write_section
Write content for a planned section.
```
Params: section_id* (string), content* (string), search_query (string), citations (array)
Effects: read | Network: none | Cost: low
```

### revise_section
Revise a section with feedback and new content.
```
Params: section_id* (string), feedback (string), new_content* (string), additional_citations (array)
Effects: read | Network: none | Cost: low
```

### assemble_document
Assemble all sections into a complete markdown document.
```
Params: include_toc (boolean), include_references (boolean), include_abstract (boolean), section_order (array)
Effects: read | Network: none | Cost: low
```

### export_document
Export assembled document to DOCX, PDF, PPTX, or LaTeX.
```
Params: format* (string: docx|pdf|pptx|latex), persist_to_kb (boolean), latex_project_id (string)
Effects: write | Network: none | Cost: medium
```
**Example:** `{"format": "docx", "persist_to_kb": true}`

### insert_figure
Insert a figure into a document section.
```
Params: section_id* (string), figure_type* (string), caption* (string), data (object), diagram_spec (string)
Effects: read | Network: none | Cost: low
```

### list_documents_by_tag
List documents matching specified tags with AND/OR matching.
```
Params: tags* (array), match_all (boolean, default false), limit (integer, default 20, max 100)
Effects: read | Network: none | Cost: low
```
**Example:** `{"tags": ["machine-learning", "survey"], "match_all": true}`

### merge_documents
Merge content from multiple documents into a single new document.
```
Params: document_ids* (array, max 20), title* (string), separator (string, default "\n\n---\n\n"), tags (array)
Effects: write | Network: none | Cost: low
```
**Example:** `{"document_ids": ["uuid1", "uuid2"], "title": "Combined Research Report", "tags": ["synthesis"]}`

---

## 14. Workspace Persistence (1 tool)

Access artifacts from persisted workspaces.

### get_workspace_artifact_url
Get a presigned download URL for a file from a persisted workspace.
```
Params: job_id* (string), file_path* (string)
Effects: read | Network: none | Cost: low
```
**Example:** `{"job_id": "550e8400-...", "file_path": "src/main.py"}`

---

## 15. Memory (4 tools)

Create and search persistent memories across agent job sessions.

### create_memory
Store a memory for future retrieval.
```
Params: content* (string), importance (number, 0-1), category (string: fact|preference|context|summary|goal|constraint), metadata (object)
Effects: write | Network: none | Cost: low
```
**Example:** `{"content": "The project uses OAuth2 for authentication", "importance": 0.9, "category": "fact", "metadata": {"source": "codebase analysis"}}`

### search_memories
Search memories with optional category and importance filters.
```
Params: query* (string), limit (integer, max 50), category_filter (string), min_importance (number)
Effects: read | Network: none | Cost: low
```

### recall_memories
Broadly recall memories on a topic (no filters).
```
Params: topic* (string), limit (integer, max 50)
Effects: read | Network: none | Cost: low
```

### get_memory_stats
Get memory statistics (counts by type, recent activity).
```
Params: (none)
Effects: read | Network: none | Cost: low
```

---

## 16. Notification & Alerting (2 tools)

Proactively notify users from agent jobs.

### send_notification
Send an in-app notification with WebSocket push.
```
Params: title* (string), message* (string), priority (string: low|normal|high|urgent), action_url (string)
Effects: write | Network: none | Cost: low
```
**Example:** `{"title": "Research Complete", "message": "Found 15 relevant papers on quantum ML", "priority": "high", "action_url": "/agents/job-123"}`

### send_email_alert
Send an email alert (falls back to in-app notification if SMTP not configured).
```
Params: subject* (string), body* (string), priority (string: low|normal|high|urgent)
Effects: write | Network: none | Cost: low
```

---

## 17. Miscellaneous (6 tools)

Template filling, progress reporting, agent delegation, and project bootstrap.

### start_template_fill / list_template_jobs / get_template_job_status
Manage template-based document generation jobs.
```
start_template_fill — Params: source_document_ids* (array)
list_template_jobs — Params: status_filter (string), limit (integer)
get_template_job_status — Params: job_id* (string)
Effects: read | Network: none | Cost: low
```

### write_progress_report
Write a structured progress report.
```
Params: summary* (string), completed_tasks (array), pending_tasks (array), key_findings (array), blockers (array), next_steps (array)
Effects: read | Network: none | Cost: low
```

### suggest_next_action
Get an LLM suggestion for the next action given current progress.
```
Params: current_goal* (string), progress_so_far (string), available_resources (array), constraints (string)
Effects: read | Network: none | Cost: low
```

### project_bootstrap
Bootstrap a project workspace from a data source.
```
Params: source_id (string), max_files (integer)
Effects: read | Network: none | Cost: low
```

### delegate_to_agent / list_available_agents
Delegate to a named agent (chat-level).
```
delegate_to_agent — Params: target_agent* (string), task_description* (string), context (string)
list_available_agents — Params: (none)
Effects: read | Network: none | Cost: low
```

### create_knowledge_base_entry
Create a structured knowledge base entry.
```
Params: title* (string), content* (string), entry_type* (string), related_documents (array), related_entities (array), tags (array), metadata (object)
Effects: read | Network: none | Cost: low
```

### summarize_documents_in_source / enrich_arxiv_metadata_for_source
Batch operations on data sources.
```
summarize_documents_in_source — Params: source_id* (string), force (boolean), only_missing (boolean), limit (integer)
enrich_arxiv_metadata_for_source — Params: source_id* (string), force (boolean), limit (integer)
Effects: read | Network: none | Cost: low
```

### generate_gitlab_architecture
Generate architecture diagrams from a GitLab project.
```
Params: project_id* (string), branch (string), diagram_type (string), focus (string), detail_level (string)
Effects: read | Network: none | Cost: low
```

---

## 18. Data Visualization (2 tools)

Generate charts and render diagrams to images, persisted to MinIO.

### create_chart
Generate a data chart and persist to storage.
```
Params: chart_type* (string: bar|line|pie|scatter|histogram|heatmap|box|area), data* (object), title (string), x_label (string), y_label (string), format (string: png|svg, default png)
Effects: write | Network: none | Cost: medium
```
**Example:** `{"chart_type": "bar", "data": {"labels": ["Q1","Q2","Q3"], "values": [10,25,15]}, "title": "Revenue"}`

### render_diagram
Render Mermaid or Graphviz diagram source code to an image.
```
Params: diagram_code* (string), diagram_type (string: mermaid|graphviz, default mermaid), format (string: png|svg, default png)
Effects: write | Network: egress | Cost: low
```
**Example:** `{"diagram_code": "graph TD\n  A-->B\n  B-->C", "format": "svg"}`

---

## 19. Scheduling (2 tools)

Schedule future agent job executions with one-time or cron-based recurrence.

### schedule_job
Schedule a new agent job for future execution.
```
Params: goal* (string), job_type (string, default research), schedule_type* (string: once|recurring), run_at (string, ISO datetime), cron (string, cron expression), config (object)
Effects: write | Network: none | Cost: low
```
**Example (one-time):** `{"goal": "Search for new papers on RAG", "schedule_type": "once", "run_at": "2026-04-01T09:00:00Z"}`
**Example (recurring):** `{"goal": "Monitor arxiv for LLM papers", "schedule_type": "recurring", "cron": "0 9 * * 1", "job_type": "monitor"}`

### cancel_scheduled_job
Cancel a scheduled or recurring agent job.
```
Params: job_id* (string)
Effects: write | Network: none | Cost: low
```

---

## 20. Agent Self-Reflection (2 tools)

Let agents introspect past job history and per-job metrics to learn from experience.

### get_job_history
Query past agent job runs for the same user.
```
Params: job_type (string), status (string: completed|failed|cancelled), limit (integer, default 10, max 50)
Effects: read | Network: none | Cost: low
```
**Example:** `{"job_type": "research", "status": "completed", "limit": 5}`

### get_job_metrics
Get detailed metrics for a specific job including per-tool usage breakdown.
```
Params: job_id (string, defaults to current job)
Effects: read | Network: none | Cost: low
```

---

## 21. Tool Usage Analytics (2 tools)

Analyze tool usage patterns, success rates, and failure modes across jobs.

### get_tool_usage_stats
Get aggregated tool usage statistics across recent jobs.
```
Params: days (integer, default 7, max 30), tool_name (string, optional filter)
Effects: read | Network: none | Cost: low
```
**Example:** `{"days": 14}` → returns top tools by call count with success rates

### get_tool_failure_analysis
Analyze failure patterns for a specific tool with error grouping.
```
Params: tool_name* (string), days (integer, default 7, max 30)
Effects: read | Network: none | Cost: low
```
**Example:** `{"tool_name": "search_web", "days": 7}` → groups errors by pattern, shows frequency

---

## 22. Batch Processing (2 tools)

Process multiple items efficiently in a single tool call.

### batch_search
Run multiple search queries against the knowledge base with cross-query deduplication.
```
Params: queries* (array, max 10), limit_per_query (integer, default 5, max 20), source_id (string), deduplicate (boolean, default true)
Effects: read | Network: none | Cost: medium
```
**Example:** `{"queries": ["transformer architecture", "attention mechanism", "BERT fine-tuning"], "limit_per_query": 5}`

### batch_summarize
Get summaries for multiple documents in a single call.
```
Params: document_ids* (array, max 20), generate_missing (boolean, default false)
Effects: read | Network: none | Cost: medium
```
**Example:** `{"document_ids": ["uuid1", "uuid2", "uuid3"], "generate_missing": true}`

---

## 23. Conditional Execution (3 tools)

Structured state introspection for branching decisions — check findings, progress, document counts, and search coverage.

### evaluate_condition
Evaluate a structured condition against current job state.
```
Params: condition* (string: findings_count|findings_has_category|documents_count|search_has_results|actions_count|progress_above), threshold (integer, default 1), category (string), query (string), source_id (string)
Effects: read | Network: none | Cost: low
```
**Example:** `{"condition": "findings_has_category", "category": "key_insight", "threshold": 3}`

### count_findings
Count accumulated findings with grouping by category.
```
Params: category (string), min_confidence (number, default 0.0)
Effects: read | Network: none | Cost: low
```
**Example:** `{"min_confidence": 0.7}` → returns `{total: 5, by_category: {key_insight: 3, methodology: 2}}`

### check_goal_status
Get current job progress, iteration budget, resource usage, and plan status.
```
Params: (none)
Effects: read | Network: none | Cost: low
```

---

## 24. Context Window Management (2 tools)

Compress action history and synthesize findings to maintain context awareness across long-running jobs.

### compress_history
LLM-summarize past actions into a condensed narrative that persists across iterations.
```
Params: keep_last (integer, default 5, max 20)
Effects: read | Network: none | Cost: medium
```
**Example:** `{"keep_last": 3}` → compresses all but last 3 actions into a narrative summary

### summarize_findings
Synthesize accumulated findings into a coherent summary, optionally consolidating into a single finding.
```
Params: consolidate (boolean, default false), category (string)
Effects: read | Network: none | Cost: medium
```
**Example:** `{"category": "key_insight", "consolidate": true}` → replaces key_insight findings with one synthesized finding

---

## 25. Agent Collaboration Protocols (3 tools)

Structured handoff patterns, sibling awareness, and broadcast messaging for multi-agent coordination.

### create_handoff
Spawn a child agent job with a typed contract specifying expected outputs.
```
Params: goal* (string), job_type (string: research|analysis|synthesis|custom, default research), context (string), expected_outputs* (array, max 10), share_findings (boolean, default true), max_iterations (integer, default 10, max 20)
Effects: write | Network: none | Cost: medium
```
**Example:** `{"goal": "Summarize transformer papers", "expected_outputs": ["summary", "key_findings", "open_questions"], "context": "Found 5 relevant papers in previous search"}`

### get_sibling_status
Check status, progress, and findings of sibling agent jobs (same parent).
```
Params: include_findings (boolean, default false)
Effects: read | Network: none | Cost: low
```
**Example:** `{"include_findings": true}` → returns sibling names, statuses, iteration progress, and finding titles

### broadcast_to_siblings
Send a message to all sibling agent jobs at once.
```
Params: message* (string), category (string, default "broadcast")
Effects: write | Network: none | Cost: low
```
**Example:** `{"message": "Found critical paper on attention mechanisms — ID doc-123", "category": "discovery"}`

---

## 26. Prompt Template Management (3 tools)

Switch agent strategies mid-run and inject custom focus directives into the system prompt.

### switch_strategy
Change the agent's role/skill profile mid-run. Affects tool prioritization and prompt guidance.
```
Params: role* (string: researcher|critic|synthesizer|verifier|coder|author), reason (string)
Effects: read | Network: none | Cost: low
```
**Example:** `{"role": "synthesizer", "reason": "Enough findings gathered, switching to synthesis phase"}`

### set_focus_directive
Set a custom focus directive injected into the system prompt on every subsequent iteration.
```
Params: directive* (string), append (boolean, default false)
Effects: read | Network: none | Cost: low
```
**Example:** `{"directive": "Prioritize finding contradictions between sources", "append": true}`

### get_available_strategies
List all available role profiles with guidance and tool preferences.
```
Params: (none)
Effects: read | Network: none | Cost: low
```

---

## 27. Output Formatting (3 tools)

Format agent results as structured markdown tables, reports, or custom JSON schemas. Artifacts are stored in state and merged into `job.results` at completion.

### format_as_table
Convert data into a formatted markdown table. Can auto-extract from accumulated findings.
```
Params: title* (string), columns (array of strings), rows (array of arrays), source (string: "custom"|"findings", default "custom"), finding_fields (array of strings, default ["title","category","confidence"])
Effects: read | Network: none | Cost: low
```
**Example:** `{"title": "Top Papers", "columns": ["Title", "Score"], "rows": [["Attention Is All You Need", "0.95"], ["BERT", "0.91"]]}`

### format_as_report
Compile findings, progress reports, and custom sections into a structured markdown report.
```
Params: title* (string), sections (array of {heading, content}), include_findings (boolean, default true), include_progress (boolean, default true), executive_summary (string), persist (boolean, default false)
Effects: read | Network: none | Cost: low
```
**Example:** `{"title": "Transformer Survey", "executive_summary": "This report covers...", "sections": [{"heading": "Background", "content": "..."}]}`

### set_output_schema
Define a structured JSON schema for the final job results. Fields are populated progressively and merged into `job.results["structured_output"]` at completion.
```
Params: schema* (object), merge (boolean, default true)
Effects: read | Network: none | Cost: low
```
**Example:** `{"schema": {"title": "Analysis Report", "key_findings": [], "recommendations": []}}`

---

## 28. Multi-Modal Ingestion (3 tools)

Process media files (images, audio, video) already in the knowledge base — trigger transcription, analyze images via vision LLM, or inspect media metadata.

### transcribe_document
Trigger Whisper transcription on an existing audio or video document. Creates a linked transcript document asynchronously via Celery.
```
Params: document_id* (string), language (string, default "auto")
Effects: write | Network: none | Cost: medium
```
**Example:** `{"document_id": "abc-123", "language": "en"}`

### analyze_image
Analyze an image document using a vision-capable LLM (e.g. llava). Downloads the image and sends it with a custom prompt.
```
Params: document_id* (string), prompt (string), model (string)
Effects: read | Network: none | Cost: medium
```
**Example:** `{"document_id": "img-456", "prompt": "Extract all text from this image"}`

### get_media_info
Get media-specific metadata: duration/codec/bitrate for audio/video (via ffprobe), width/height/format for images (via Pillow). Includes transcription status.
```
Params: document_id* (string)
Effects: read | Network: none | Cost: low
```

---

## 29. Workspace Snapshots (3 tools)

Capture, compare, and detect drift in agent workspace state across iterations for self-monitoring.

### capture_snapshot
Capture a named snapshot of current workspace metrics (findings count, progress, tool stats, etc.) for later comparison.
```
Params: name* (string), keys (array of strings)
Effects: read | Network: none | Cost: low
```
**Example:** `{"name": "after_search", "keys": ["hypotheses"]}`

### compare_snapshots
Compare two named snapshots and return a structured diff showing deltas for all metrics.
```
Params: snapshot_a* (string), snapshot_b* (string)
Effects: read | Network: none | Cost: low
```
**Example:** `{"snapshot_a": "after_search", "snapshot_b": "after_synthesis"}`

### detect_drift
Compare current state against a baseline snapshot and flag problems (stalling, progress regression, high failure rates).
```
Params: baseline* (string), thresholds (object)
Effects: read | Network: none | Cost: low
```
**Example:** `{"baseline": "initial", "thresholds": {"stalled_iterations": 5}}`

---

## Job Type → Tool Routing

Tools are routed to agents based on job type. `base_tools` are available to all job types.

| Job Type | Additional Tools |
|----------|-----------------|
| **base** (all) | search_documents, get_document_details, list_recent_documents, answer_question, read_document_content, reflect, hypothesize, weigh_evidence, critique_plan, suggest_next_action, write_progress_report, save_research_finding, get_research_findings, execute_python, create_memory, search_memories, recall_memories, get_memory_stats, list_available_workflows, execute_workflow, get_workflow_status, send_message_to_agent, read_agent_messages, search_web, fetch_url_content, send_notification, send_email_alert, create_chart, render_diagram, query_kg_entities, get_entity_context, query_kg_graph, schedule_job, cancel_scheduled_job, list_documents_by_tag, get_job_history, get_job_metrics, get_tool_usage_stats, get_tool_failure_analysis, batch_search, batch_summarize, evaluate_condition, count_findings, check_goal_status, compress_history, summarize_findings, create_handoff, get_sibling_status, broadcast_to_siblings, switch_strategy, set_focus_directive, get_available_strategies, format_as_table, format_as_report, set_output_schema, transcribe_document, analyze_image, get_media_info, capture_snapshot, compare_snapshots, detect_drift |
| **research** | search_arxiv, summarize_document, find_similar_documents, extract_paper_insights, find_related_papers, build_research_graph, compare_methodologies, identify_research_gaps, create_synthesis_document, ingest_paper_by_id, batch_ingest_papers, summarize_url, create_kg_entity, create_kg_relationship, merge_documents, ... |
| **analysis** | search_documents, summarize_document, compare_documents, clone_and_index_repo, browse_repo_files, read_file, write_file, apply_patch, run_command, search_code, retrieve_repo_symbols, get_symbol_context, find_tests_for_symbol, summarize_url, create_kg_entity, create_kg_relationship, merge_documents, ... |
| **coding** | clone_and_index_repo, browse_repo_files, read_file, write_file, apply_patch, run_command, search_code, get_workspace_status, retrieve_repo_symbols, get_symbol_context, find_tests_for_symbol, get_workspace_artifact_url |
| **synthesis** | search_documents, summarize_document, generate_diagram, create_synthesis_document, plan_document, write_section, revise_section, assemble_document, export_document, insert_figure, create_kg_entity, create_kg_relationship, merge_documents, ... |
| **document_authoring** | plan_document, write_section, revise_section, assemble_document, export_document, insert_figure |
| **monitor** | search_arxiv, search_documents, monitor_arxiv_topic, ingest_paper_by_id, add_to_reading_list, get_reading_lists |

---

## Registry Classifications Summary

| Classification | Tools |
|---------------|-------|
| **write** (side effects) | delete_document, batch_delete_documents, update_document_tags, create_document_from_text, ingest_url, rebuild_document_knowledge_graph, merge_entities, delete_entity, run_workflow, run_custom_tool, delegate_subtask, share_findings, request_review, execute_data_pipeline, write_and_run_script, write_file, apply_patch, run_command, export_document, create_memory, execute_workflow, send_message_to_agent, send_notification, send_email_alert, create_chart, render_diagram, create_kg_entity, create_kg_relationship, schedule_job, cancel_scheduled_job, merge_documents, create_handoff, broadcast_to_siblings, transcribe_document |
| **network** (egress) | web_scrape, ingest_url, search_arxiv, ingest_arxiv_papers, literature_review_arxiv, create_repo_report, docker_execute, clone_and_index_repo, search_web, fetch_url_content, summarize_url, render_diagram |
| **high cost** | docker_execute, execute_data_pipeline, write_and_run_script, run_command |
| **medium cost** | generate_report, create_repo_report, create_presentation, delegate_subtask, request_review, execute_python, clone_and_index_repo, export_document, execute_workflow, search_web, summarize_url, create_chart, batch_search, batch_summarize, compress_history, summarize_findings, create_handoff, transcribe_document, analyze_image |
| **high PII** | docker_execute, write_and_run_script |
| **medium PII** | web_scrape, ingest_url, run_custom_tool, execute_python, execute_data_pipeline, run_command, search_code |
