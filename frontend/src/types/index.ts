/**
 * TypeScript type definitions
 */

export interface User {
  id: string;
  username: string;
  email: string;
  full_name?: string;
  role: string;
  is_active: boolean;
  is_verified: boolean;
  avatar_url?: string;
  last_login?: string;
  login_count: number;
  created_at: string;
}

export interface ChatSession {
  id: string;
  title?: string;
  is_active: boolean;
  extra_metadata?: Record<string, any> | null;
  created_at: string;
  updated_at: string;
  last_message_at: string;
  messages?: ChatMessage[];
}

export interface ChatMessage {
  id: string;
  content: string;
  role: 'user' | 'assistant' | 'system';
  message_type?: string;
  model_used?: string;
  response_time?: number;
  token_count?: number;
  source_documents?: SourceDocument[];
  context_used?: string;
  search_query?: string;
  retrieval_trace_id?: string;
  user_rating?: number;
  user_feedback?: string;
  created_at: string;
}

export interface SourceDocument {
  id: string;
  title: string;
  score: number;
  source: string;
  url?: string;
  download_url?: string;
  chunk_id?: string;
  chunk_index?: number;
  snippet?: string;
}

export interface InstantArxivIngestResponse {
  document_id: string;
  arxiv_id: string;
  title: string;
  authors: string[];
  abstract: string;
  categories: string[];
  url: string;
  pdf_url: string;
  chunks_created: number;
  ready_for_chat: boolean;
  background_tasks?: string[];
}

export interface Persona {
  id: string;
  name: string;
  platform_id?: string | null;
  user_id?: string | null;
  avatar_url?: string | null;
  description?: string | null;
  extra_metadata?: Record<string, any> | null;
  is_active?: boolean;
  is_system?: boolean;
  created_at?: string;
  updated_at?: string;
}

export interface DocumentPersonaDetection {
  id: string;
  persona_id: string;
  role: string;
  detection_type?: string | null;
  confidence?: number | null;
  start_time?: number | null;
  end_time?: number | null;
  details?: Record<string, any> | null;
  created_at: string;
  persona: Persona;
}

// Agent definitions for multi-agent system
export interface AgentDefinition {
  id: string;
  name: string;
  display_name: string;
  description?: string | null;
  system_prompt: string | null;
  capabilities: string[];
  tool_whitelist?: string[] | null;
  routing_defaults?: Record<string, any> | null;
  priority: number;
  is_active: boolean;
  is_system: boolean;
  owner_user_id?: string | null;
  version?: number | null;
  lifecycle_status?: 'draft' | 'published' | 'archived' | string;
  created_at: string;
  updated_at: string;
}

export interface AgentDefinitionSummary {
  id: string;
  name: string;
  display_name: string;
  description?: string | null;
  capabilities: string[];
  priority: number;
  is_active: boolean;
  is_system: boolean;
  created_at?: string | null;
  updated_at?: string | null;
}

export interface AgentDefinitionCreate {
  name: string;
  display_name: string;
  description?: string | null;
  system_prompt: string;
  capabilities: string[];
  tool_whitelist?: string[] | null;
  routing_defaults?: Record<string, any> | null;
  priority?: number;
  is_active?: boolean;
}

export interface AgentDefinitionUpdate {
  display_name?: string;
  description?: string | null;
  system_prompt?: string;
  capabilities?: string[];
  tool_whitelist?: string[] | null;
  routing_defaults?: Record<string, any> | null;
  priority?: number;
  is_active?: boolean;
}

export interface CapabilityInfo {
  name: string;
  description: string;
  keywords: string[];
}

export interface ArxivPaper {
  id: string;
  entry_url: string;
  pdf_url?: string | null;
  title: string;
  summary?: string;
  authors?: string[];
  published?: string;
  updated?: string;
  categories?: string[];
  primary_category?: string | null;
  doi?: string | null;
  comments?: string | null;
}

export interface ArxivSearchResponse {
  total_results: number;
  start: number;
  max_results: number;
  items: ArxivPaper[];
}

export interface PaperClaim {
  id: string;
  kind: 'performance' | 'compile_time' | 'code_size' | 'energy' | 'correctness' | 'robustness' | 'other' | string;
  statement: string;
  mechanism?: string | null;
  target_layer: 'source' | 'ir' | 'midend' | 'backend' | 'runtime' | 'hardware' | 'unknown' | string;
  conditions?: string[] | null;
  assumptions?: string[] | null;
  expected_effect?: string | null;
  evidence_summary?: string | null;
  confidence?: number | null;
  tags?: string[] | null;
  rank?: number | null;
  created_at?: string | null;
  updated_at?: string | null;
}

export interface PaperExtractionJob {
  id: string;
  user_id: string;
  document_id: string;
  source_id?: string | null;
  paper_id?: string | null;
  status: 'pending' | 'running' | 'completed' | 'failed' | string;
  extractor_version?: string | null;
  error?: string | null;
  request_payload?: Record<string, any> | null;
  result_summary?: Record<string, any> | null;
  created_at?: string | null;
  started_at?: string | null;
  completed_at?: string | null;
  updated_at?: string | null;
}

export interface ResearchPaper {
  id: string;
  user_id: string;
  document_id: string;
  source_id?: string | null;
  arxiv_id: string;
  title: string;
  authors?: string[] | null;
  abstract?: string | null;
  published_at?: string | null;
  categories?: string[] | null;
  paper_url?: string | null;
  pdf_url?: string | null;
  extraction_status: 'pending' | 'running' | 'completed' | 'failed' | string;
  extracted_at?: string | null;
  extractor_version?: string | null;
  summary?: string | null;
  mechanisms?: string[] | null;
  assumptions?: string[] | null;
  benchmarks?: string[] | null;
  metrics?: string[] | null;
  limitations?: string[] | null;
  raw_extraction_payload?: Record<string, any> | null;
  claims: PaperClaim[];
  latest_job?: PaperExtractionJob | null;
  created_at?: string | null;
  updated_at?: string | null;
}

export interface ResearchPaperListResponse {
  items: ResearchPaper[];
  total: number;
  limit: number;
  offset: number;
}

export interface Document {
  id: string;
  title: string;
  content?: string;
  content_hash: string;
  url?: string;
  file_path?: string;
  file_type?: string;
  file_size?: number;
  source_identifier: string;
  author?: string;
  tags?: string[];
  metadata?: any;
  extra_metadata?: {
    is_transcribing?: boolean;
    is_transcribed?: boolean;
    transcription_metadata?: any;
    [key: string]: any;
  };
  is_processed: boolean;
  processing_error?: string;
  summary?: string;
  summary_model?: string;
  summary_generated_at?: string;
  created_at: string;
  updated_at: string;
  last_modified?: string;
  source: DocumentSource;
  chunks?: DocumentChunk[];
  download_url?: string;
  owner_persona?: Persona | null;
  persona_detections?: DocumentPersonaDetection[];
}

export interface DocumentSource {
  id: string;
  name: string;
  source_type: 'gitlab' | 'github' | 'confluence' | 'web' | 'file' | 'arxiv';
  config: any;
  is_active: boolean;
  is_syncing?: boolean;
  last_sync?: string;
  last_error?: string;
  created_at: string;
  updated_at: string;
}

// LaTeX Studio
export interface LatexStatusResponse {
  enabled: boolean;
  admin_only: boolean;
  use_celery_worker?: boolean;
  celery_queue?: string | null;
  timeout_seconds: number;
  max_source_chars: number;
  available_engines: Record<string, boolean>;
  available_tools?: Record<string, boolean>;
}

export interface LatexCompileJobCreateRequest {
  safe_mode?: boolean;
  preferred_engine?: string | null;
}

export interface LatexCompileJobResponse {
  id: string;
  project_id?: string | null;
  status: string;
  safe_mode?: boolean;
  preferred_engine?: string | null;
  engine?: string | null;
  log?: string | null;
  violations?: string[];
  pdf_file_path?: string | null;
  pdf_download_url?: string | null;
  created_at?: string | null;
  updated_at?: string | null;
  started_at?: string | null;
  finished_at?: string | null;
}

export interface LatexCompileRequest {
  tex_source: string;
  safe_mode?: boolean;
  preferred_engine?: string | null;
}

export interface LatexCompileResponse {
  success: boolean;
  engine?: string | null;
  pdf_base64?: string | null;
  log: string;
  violations: string[];
}

export interface LatexCopilotRequest {
  prompt: string;
  search_query?: string;
  citation_mode?: 'thebibliography' | 'bibtex';
}

export interface LatexCopilotResponse {
  tex_snippet: string;
  bibtex: string;
  references_tex?: string;
  bibtex_entries?: string;
}

export interface LatexCopilotFixRequest {
  tex_source: string;
  compile_log: string;
  safe_mode?: boolean;
}

export interface LatexCopilotFixResponse {
  tex_source_fixed: string;
  notes: string;
  unsafe_warnings: string[];
}

export interface LatexMathCopilotRequest {
  tex_source: string;
  mode?: 'analyze' | 'autocomplete';
  goal?: string;
  selection?: string | null;
  cursor_context?: string | null;
  enforce_siunitx?: boolean;
  enforce_shapes?: boolean;
  enforce_bold_italic_conventions?: boolean;
  enforce_equation_labels?: boolean;
  max_source_chars?: number;
  return_patched_source?: boolean;
}

export interface LatexMathCopilotResponse {
  conventions: Record<string, string>;
  suggestions: Array<Record<string, string>>;
  diff_unified: string;
  notes: string;
  base_sha256: string;
  diff_applies: boolean;
  patched_sha256?: string | null;
  tex_source_patched?: string | null;
  diff_warnings?: string[];
}

export interface LatexCitationsRequest {
  document_ids: string[];
  mode?: 'bibtex' | 'thebibliography';
  bib_filename?: string;
}

export interface LatexCitationsResponse {
  mode: 'bibtex' | 'thebibliography';
  cite_keys_by_doc_id: Record<string, string>;
  cite_command: string;
  bibliography_scaffold?: string;
  bibtex_entries?: string;
  references_tex?: string;
}

export interface LatexApplyUnifiedDiffRequest {
  diff_unified: string;
  expected_base_sha256?: string | null;
}

export interface LatexApplyUnifiedDiffResponse {
  applied: boolean;
  tex_source: string;
  base_sha256: string;
  new_sha256: string;
  warnings: string[];
}

export interface LatexProjectListItem {
  id: string;
  title: string;
  updated_at?: string | null;
  last_compiled_at?: string | null;
}

export interface LatexProjectListResponse {
  items: LatexProjectListItem[];
  total: number;
  limit: number;
  offset: number;
}

export interface LatexProjectCreateRequest {
  title: string;
  tex_source: string;
}

export interface LatexProjectUpdateRequest {
  title?: string;
  tex_source?: string;
}

export interface LatexProjectResponse {
  id: string;
  user_id: string;
  title: string;
  tex_source: string;
  tex_file_path?: string | null;
  pdf_file_path?: string | null;
  pdf_download_url?: string | null;
  last_compile_engine?: string | null;
  last_compile_log?: string | null;
  last_compiled_at?: string | null;
  created_at?: string | null;
  updated_at?: string | null;
}

export interface LatexProjectCompileRequest {
  safe_mode?: boolean;
  preferred_engine?: string | null;
}

export interface LatexProjectCompileResponse {
  success: boolean;
  engine?: string | null;
  pdf_file_path?: string | null;
  pdf_download_url?: string | null;
  log: string;
  violations: string[];
}

export interface LatexProjectPublishRequest {
  include_tex?: boolean;
  include_pdf?: boolean;
  safe_mode?: boolean;
  tags?: string[];
}

export interface LatexProjectPublishItem {
  kind: 'tex' | 'pdf';
  document_id: string;
  title: string;
  file_type?: string | null;
  file_path?: string | null;
}

export interface LatexProjectPublishSkipped {
  kind: 'tex' | 'pdf';
  reason: string;
}

export interface LatexProjectPublishResponse {
  project_id: string;
  published: LatexProjectPublishItem[];
  skipped: LatexProjectPublishSkipped[];
}

export interface LatexProjectFileResponse {
  id: string;
  project_id: string;
  filename: string;
  content_type?: string | null;
  file_size: number;
  sha256?: string | null;
  file_path: string;
  download_url?: string | null;
  created_at?: string | null;
}

export interface LatexProjectFileListResponse {
  items: LatexProjectFileResponse[];
  total: number;
}

export interface LatexProjectFileUploadResponse {
  file: LatexProjectFileResponse;
  replaced: boolean;
}

export interface ActiveGitSource {
  source: DocumentSource;
  pending: boolean;
  task_id?: string;
}

export interface GitBranch {
  repository: string;
  name: string;
  commit_sha?: string;
  commit_message?: string;
  commit_author?: string;
  commit_date?: string;
  protected?: boolean;
}

export type GitCompareStatus =
  | 'queued'
  | 'running'
  | 'completed'
  | 'failed'
  | 'cancel_requested'
  | 'canceled';

export interface GitCompareJob {
  id: string;
  source_id: string;
  repository: string;
  base_branch: string;
  compare_branch: string;
  status: GitCompareStatus;
  diff_summary?: Record<string, any>;
  llm_summary?: string;
  error?: string;
  created_at: string;
  updated_at: string;
  completed_at?: string;
}

export interface DocumentChunk {
  id: string;
  content: string;
  chunk_index: number;
  start_pos?: number;
  end_pos?: number;
  embedding_id?: string;
  metadata?: any;
  created_at: string;
}

export interface SystemHealth {
  timestamp: string;
  overall_status: 'healthy' | 'degraded' | 'unhealthy';
  services: Record<string, ServiceHealth>;
}

export interface ServiceHealth {
  status: 'healthy' | 'degraded' | 'unhealthy' | 'unknown';
  message?: string;
  error?: string;
}

export interface SystemStats {
  timestamp: string;
  documents?: DocumentStats;
  chat?: ChatStats;
  sources?: SourceStats;
  vector_store?: VectorStoreStats;
  processing?: ProcessingStats;
  error?: string;
}

export interface AdminIngestionDBStatus {
  documents_total: number;
  documents_processed: number;
  documents_pending: number;
  documents_failed: number;
  documents_without_chunks: number;
  chunks_total: number;
  chunks_embedded: number;
  chunks_missing_embedding: number;
}

export interface AdminIngestionVectorStoreStatus {
  provider: string;
  collection_name?: string | null;
  collection_exists?: boolean | null;
  points_total?: number | null;
  error?: string | null;
}

export interface AdminIngestionSourceStatus {
  source_id: string;
  name: string;
  source_type: string;
  is_active: boolean;
  is_syncing: boolean;
  last_sync?: string | null;
  last_error?: string | null;
  docs_total: number;
  docs_processed: number;
  docs_pending: number;
  docs_failed: number;
  chunks_total: number;
  chunks_embedded: number;
  chunks_missing_embedding: number;
  last_sync_log?: Record<string, any> | null;
}

export interface AdminIngestionStatus {
  timestamp: string;
  db: AdminIngestionDBStatus;
  vector_store: AdminIngestionVectorStoreStatus;
  sources: AdminIngestionSourceStatus[];
  recent_document_errors: Array<{
    document_id: string;
    title?: string;
    source_id?: string;
    updated_at?: string | null;
    error?: string | null;
  }>;
}

export interface UnsafeExecStatusResponse {
  enabled: boolean;
  backend: 'subprocess' | 'docker' | string;
  docker?: {
    available: boolean;
    image?: string;
    image_present?: boolean | null;
  };
  limits?: {
    timeout_seconds?: number;
    max_memory_mb?: number;
  };
}

export interface ToolAudit {
  id: string;
  user_id: string;
  agent_definition_id?: string | null;
  conversation_id?: string | null;
  tool_name: string;
  tool_input?: Record<string, any> | null;
  tool_output?: any;
  status: string;
  error?: string | null;
  execution_time_ms?: number | null;
  approval_required: boolean;
  approval_status?: string | null;
  approved_by?: string | null;
  approved_at?: string | null;
  approval_note?: string | null;
  created_at: string;
  updated_at: string;
}

export interface LLMUsageEvent {
  id: string;
  user_id?: string | null;
  provider: string;
  model?: string | null;
  task_type?: string | null;
  prompt_tokens?: number | null;
  completion_tokens?: number | null;
  total_tokens?: number | null;
  input_chars?: number | null;
  output_chars?: number | null;
  latency_ms?: number | null;
  error?: string | null;
  extra?: Record<string, any> | null;
  created_at: string;
}

export interface LLMUsageSummaryItem {
  provider: string;
  model?: string | null;
  task_type?: string | null;
  request_count: number;
  total_prompt_tokens: number;
  total_completion_tokens: number;
  total_tokens: number;
  avg_latency_ms?: number | null;
}

export interface LLMUsageSummaryResponse {
  items: LLMUsageSummaryItem[];
  date_from?: string | null;
  date_to?: string | null;
}


export interface LLMRoutingSummaryItem {
  provider: string;
  model?: string | null;
  task_type?: string | null;

  routing_tier?: string | null;
  routing_requested_tier?: string | null;
  routing_attempt?: number | null;
  routing_attempts?: number | null;
  routing_tier_provider?: string | null;
  routing_tier_model?: string | null;

  routing_experiment_id?: string | null;
  routing_experiment_variant_id?: string | null;

  request_count: number;
  success_count: number;
  error_count: number;
  success_rate: number;

  total_tokens: number;
  avg_latency_ms?: number | null;
  p50_latency_ms?: number | null;
  p95_latency_ms?: number | null;
}

export interface LLMRoutingSummaryResponse {
  items: LLMRoutingSummaryItem[];
  date_from?: string | null;
  date_to?: string | null;
  scanned_events: number;
  truncated: boolean;
}


export interface LLMRoutingExperimentVariantStat {
  experiment_id: string;
  variant_id: string;
  request_count: number;
  success_count: number;
  error_count: number;
  success_rate: number;
  avg_latency_ms?: number | null;
  p95_latency_ms?: number | null;
}

export interface LLMRoutingExperimentRecommendationResponse {
  experiment_id: string;
  agent_id?: string | null;
  recommended_variant_id?: string | null;
  rationale: string;
  variants: LLMRoutingExperimentVariantStat[];
  date_from?: string | null;
  date_to?: string | null;
  scanned_events: number;
  truncated: boolean;
}

export interface DocumentStats {
  total: number;
  processed: number;
  failed: number;
  pending: number;
  success_rate: number;
  without_summary?: number;
}
export interface LLMRoutingExperimentListItem {
  agent_id: string;
  agent_name: string;
  agent_display_name: string;
  agent_is_system: boolean;
  agent_owner_user_id?: string | null;
  agent_lifecycle_status?: string | null;
  routing_defaults?: Record<string, any> | null;
  experiment: Record<string, any>;
}

export interface LLMRoutingExperimentListResponse {
  items: LLMRoutingExperimentListItem[];
  total: number;
}



export interface ChatStats {
  total_sessions: number;
  active_sessions_24h: number;
  total_messages: number;
  avg_messages_per_session: number;
}

export interface SourceStats {
  total: number;
  active: number;
  by_type: Record<string, number>;
}

export interface VectorStoreStats {
  total_chunks?: number;
  collection_name?: string;
  embedding_model?: string;
  error?: string;
}

export interface ProcessingStats {
  documents_last_7_days: Array<{ date: string; count: number }>;
  total_documents_last_7_days: number;
}

export interface WebSocketMessage {
  type: 'message' | 'typing' | 'error';
  data?: any;
  message?: string;
}

export interface TaskStatus {
  active_tasks?: Record<string, any>;
  scheduled_tasks?: Record<string, any>;
  reserved_tasks?: Record<string, any>;
}

export interface Memory {
  id: string;
  memory_type: 'fact' | 'preference' | 'context' | 'summary' | 'goal' | 'constraint';
  content: string;
  importance_score: number;
  context?: any;
  tags?: string[];
  session_id?: string;
  source_message_id?: string;
  created_at: string;
  last_accessed_at: string;
  access_count: number;
  is_active: boolean;
}

export interface MemoryStats {
  total_memories: number;
  memories_by_type: Record<string, number>;
  recent_memories: number;
  most_accessed_memories: Memory[];
  memory_usage_trend: Array<{ date: string; count: number }>;
}

export interface MemorySummary {
  summary: string;
  key_facts: string[];
  preferences: string[];
  context_items: string[];
  memory_count: number;
  time_range: string;
}

// Template Types
export interface TemplateSection {
  title: string;
  level: number;
  placeholder_text?: string;
}

export type TemplateJobStatus = 'pending' | 'analyzing' | 'extracting' | 'filling' | 'completed' | 'failed';

export interface TemplateJob {
  id: string;
  template_filename: string;
  sections?: TemplateSection[];
  source_document_ids: string[];
  status: TemplateJobStatus;
  progress: number;
  current_section?: string;
  filled_filename?: string;
  error_message?: string;
  created_at: string;
  updated_at: string;
  completed_at?: string;
  download_url?: string;
}

export interface TemplateJobListResponse {
  jobs: TemplateJob[];
  total: number;
}

export interface TemplateProgressUpdate {
  type: 'progress' | 'complete' | 'error';
  job_id: string;
  data?: {
    stage?: string;
    progress?: number;
    current_section?: string;
    section_index?: number;
    total_sections?: number;
    filled_filename?: string;
  };
  error?: string;
  result?: {
    filled_filename?: string;
    filled_file_path?: string;
  };
}

// DOCX Editor types
export interface DocxEditResponse {
  html_content: string;
  document_title: string;
  document_id: string;
  version: string;
  editable: boolean;
  warnings?: string[];
}

export interface DocxEditRequest {
  html_content: string;
  version: string;
  create_backup?: boolean;
}

export interface DocxSaveResponse {
  success: boolean;
  document_id: string;
  new_version: string;
  message: string;
  backup_path?: string;
}

// Presentation Types
export type PresentationStatus = 'pending' | 'generating' | 'completed' | 'failed' | 'cancelled';
export type PresentationStyle = 'professional' | 'casual' | 'technical' | 'modern' | 'minimal' | 'corporate' | 'creative' | 'dark';

export interface ThemeColors {
  title_color: string;
  accent_color: string;
  text_color: string;
  bg_color: string;
}

export interface ThemeFonts {
  title_font: string;
  body_font: string;
}

export interface ThemeSizes {
  title_size: number;
  subtitle_size: number;
  heading_size: number;
  body_size: number;
  bullet_size: number;
}

export interface ThemeConfig {
  colors: ThemeColors;
  fonts: ThemeFonts;
  sizes: ThemeSizes;
}

export interface PresentationTemplate {
  id: string;
  user_id?: string;
  name: string;
  description?: string;
  template_type: 'theme' | 'pptx';
  theme_config?: ThemeConfig;
  preview_url?: string;
  is_system: boolean;
  is_public: boolean;
  is_active: boolean;
  created_at: string;
  updated_at: string;
}

export interface PresentationSlideContent {
  slide_number: number;
  slide_type: 'title' | 'content' | 'diagram' | 'summary' | 'two_column';
  title: string;
  content: string[];
  subtitle?: string;
  diagram_code?: string;
  diagram_description?: string;
  notes?: string;
}

export interface PresentationOutline {
  title: string;
  subtitle?: string;
  slides: PresentationSlideContent[];
}

export interface PresentationJob {
  id: string;
  user_id: string;
  title: string;
  topic: string;
  source_document_ids: string[];
  slide_count: number;
  style: PresentationStyle;
  include_diagrams: boolean;
  status: PresentationStatus;
  progress: number;
  current_stage?: string;
  generated_outline?: PresentationOutline;
  file_path?: string;
  file_size?: number;
  error?: string;
  created_at: string;
  started_at?: string;
  completed_at?: string;
  download_url?: string;
}

export interface PresentationProgressUpdate {
  type: 'progress';
  progress: number;
  stage: string;
  status: PresentationStatus;
  error?: string;
}

// Knowledge Graph Relationship Types
export interface KGRelationshipDetail {
  id: string;
  relation_type: string;
  source_entity_id: string;
  target_entity_id: string;
  source_entity_name: string;
  target_entity_name: string;
  confidence: number;
  evidence?: string | null;
  document_id?: string | null;
  chunk_id?: string | null;
  is_manual: boolean;
  created_at: string;
}

export interface KGRelationshipCreate {
  source_entity_id: string;
  target_entity_id: string;
  relation_type: string;
  confidence?: number;
  evidence?: string;
}

export interface KGRelationshipUpdate {
  relation_type?: string;
  confidence?: number;
  evidence?: string;
}

// Search Types
export type SearchMode = 'smart' | 'keyword' | 'exact';
export type SearchSortBy = 'relevance' | 'date' | 'title';
export type SearchSortOrder = 'asc' | 'desc';

export interface SearchResult {
  id: string;
  title: string;
  source: string;
  source_type: string;
  file_type?: string;
  author?: string;
  snippet: string;
  relevance_score: number;
  created_at: string;
  updated_at: string;
  url?: string;
  download_url?: string;
  chunk_id?: string;
}

export interface SearchResponse {
  results: SearchResult[];
  total: number;
  page: number;
  page_size: number;
  query: string;
  mode: string;
  took_ms: number;
}

export interface SearchParams {
  q: string;
  mode?: SearchMode;
  sort_by?: SearchSortBy;
  sort_order?: SearchSortOrder;
  page?: number;
  page_size?: number;
  source_id?: string;
  file_type?: string;
}

// Notification Types
export type NotificationType =
  | 'document_processing_complete'
  | 'document_processing_error'
  | 'source_sync_complete'
  | 'source_sync_error'
  | 'ingestion_complete'
  | 'ingestion_error'
  | 'transcription_complete'
  | 'transcription_error'
  | 'summarization_complete'
  | 'research_note_citation_issue'
  | 'experiment_run_update'
  | 'autonomous_rnd_verification_update'
  | 'hypothesis_reevaluation_update'
  | 'queue_urgency_alert'
  | 'follow_up_outcome_alert'
  | 'policy_guardrail_alert'
  | 'autonomy_budget_alert'
  | 'customer_autonomy_budget_alert'
  | 'system_maintenance'
  | 'quota_warning'
  | 'admin_broadcast'
  | 'mention'
  | 'share'
  | 'comment';

export type NotificationPriority = 'low' | 'normal' | 'high' | 'urgent';

export interface ExperimentRunNotificationData {
  launch_mode?: string | null;
  final_phase?: string | null;
  source_name?: string | null;
  source_id?: string | null;
  bootstrap_attempted?: boolean;
  bootstrap_ok?: boolean | null;
  fallback_attempted?: boolean;
  fallback_ok?: boolean | null;
  failed_command_count?: number;
  first_failed_command?: string | null;
  recovery_open?: boolean;
  recovery_reason?: string | null;
  recommended_action?: string | null;
  latest_operator_action?: string | null;
  latest_operator_note?: string | null;
  latest_operator_status_before?: string | null;
  latest_operator_status_after?: string | null;
  latest_operator_at?: string | null;
  latest_operator_outcome?: string | null;
  latest_operator_outcome_reason?: string | null;
  experiment_run_id?: string;
  experiment_plan_id?: string;
  agent_job_id?: string | null;
  status?: string;
  note_id?: string | null;
}

export interface QueueUrgencyNotificationData {
  queue_key?: string | null;
  queue_item_type?: string | null;
  job_id?: string | null;
  sla_bucket?: string | null;
  escalation_level?: string | null;
  priority_score?: number | null;
  recommended_action?: string | null;
  reason_label?: string | null;
  customer?: string | null;
  age_minutes?: number | null;
  is_overdue?: boolean;
  is_stale?: boolean;
  evidence_summary?: string | null;
  scheduler_state?: Record<string, any> | null;
}

export interface FollowUpOutcomeNotificationData {
  inbox_item_id?: string | null;
  follow_up_job_id?: string | null;
  follow_up_last_job_id?: string | null;
  follow_up_recommendation_key?: string | null;
  follow_up_outcome_status?: string | null;
  follow_up_outcome_summary?: string | null;
  customer?: string | null;
  follow_up_policy_mode?: string | null;
  origin_source_kind?: 'profile' | 'portfolio' | null;
  origin_source_id?: string | null;
  origin_opportunity_id?: string | null;
}

export interface HypothesisReevaluationNotificationData {
  note_id?: string | null;
  note_title?: string | null;
  reevaluation_job_id?: string | null;
  reevaluation_status?: string | null;
  source_run_ids?: string[] | null;
  reprioritization_summary?: string | null;
  pending_reevaluation_created_at?: string | null;
  pending_reevaluation_completed_at?: string | null;
  reevaluation_error?: string | null;
  origin_source_kind?: 'profile' | 'portfolio' | null;
  origin_source_id?: string | null;
  origin_opportunity_id?: string | null;
  origin_action_url?: string | null;
}

export interface PolicyGuardrailNotificationData {
  queue_key?: string | null;
  monitor_job_id?: string | null;
  history_entry_id?: string | null;
  policy_guardrail_action?: string | null;
  policy_guardrail_reasons?: string[] | null;
  customer?: string | null;
}

export interface AutonomyBudgetNotificationData {
  queue_key?: string | null;
  job_id?: string | null;
  monitor_job_id?: string | null;
  budget_throttle_state?: string | null;
  budget_throttle_reasons?: string[] | null;
  customer?: string | null;
}

export interface CustomerAutonomyBudgetNotificationData {
  customer?: string | null;
  customer_budget_throttle_state?: string | null;
  customer_budget_throttle_reasons?: string[] | null;
}

export interface Notification {
  id: string;
  notification_type: NotificationType;
  title: string;
  message: string;
  priority: NotificationPriority;
  related_entity_type?: string;
  related_entity_id?: string;
  data?: Record<string, any> | ExperimentRunNotificationData | QueueUrgencyNotificationData | FollowUpOutcomeNotificationData | HypothesisReevaluationNotificationData | PolicyGuardrailNotificationData | AutonomyBudgetNotificationData | CustomerAutonomyBudgetNotificationData;
  action_url?: string;
  is_read: boolean;
  read_at?: string;
  created_at: string;
}

export interface NotificationListResponse {
  items: Notification[];
  total: number;
  page: number;
  page_size: number;
  unread_count: number;
}

export interface NotificationPreferences {
  id: string;
  user_id: string;
  notify_document_processing: boolean;
  notify_document_errors: boolean;
  notify_sync_complete: boolean;
  notify_ingestion_complete: boolean;
  notify_transcription_complete: boolean;
  notify_summarization_complete: boolean;
  notify_research_note_citation_issues: boolean;
  notify_experiment_run_updates: boolean;
  notify_hypothesis_reevaluation_updates: boolean;
  notify_queue_urgency_alerts: boolean;
  notify_follow_up_outcome_alerts: boolean;
  notify_policy_guardrail_alerts: boolean;
  notify_autonomy_budget_alerts: boolean;
  notify_customer_autonomy_budget_alerts: boolean;
  research_note_citation_coverage_threshold: number;
  research_note_citation_notify_cooldown_hours: number;
  queue_urgency_alert_reminder_cooldown_hours: number;
  research_note_citation_notify_on_unknown_keys: boolean;
  research_note_citation_notify_on_low_coverage: boolean;
  research_note_citation_notify_on_missing_bibliography: boolean;
  notify_maintenance: boolean;
  notify_quota_warnings: boolean;
  notify_admin_broadcasts: boolean;
  notify_mentions: boolean;
  notify_shares: boolean;
  notify_comments: boolean;
  play_sound: boolean;
  show_desktop_notification: boolean;
  created_at: string;
  updated_at: string;
}

export interface NotificationPreferencesUpdate {
  notify_document_processing?: boolean;
  notify_document_errors?: boolean;
  notify_sync_complete?: boolean;
  notify_ingestion_complete?: boolean;
  notify_transcription_complete?: boolean;
  notify_summarization_complete?: boolean;
  notify_research_note_citation_issues?: boolean;
  notify_experiment_run_updates?: boolean;
  notify_hypothesis_reevaluation_updates?: boolean;
  notify_queue_urgency_alerts?: boolean;
  notify_follow_up_outcome_alerts?: boolean;
  notify_policy_guardrail_alerts?: boolean;
  notify_autonomy_budget_alerts?: boolean;
  notify_customer_autonomy_budget_alerts?: boolean;
  research_note_citation_coverage_threshold?: number;
  research_note_citation_notify_cooldown_hours?: number;
  queue_urgency_alert_reminder_cooldown_hours?: number;
  research_note_citation_notify_on_unknown_keys?: boolean;
  research_note_citation_notify_on_low_coverage?: boolean;
  research_note_citation_notify_on_missing_bibliography?: boolean;
  notify_maintenance?: boolean;
  notify_quota_warnings?: boolean;
  notify_admin_broadcasts?: boolean;
  notify_mentions?: boolean;
  notify_shares?: boolean;
  notify_comments?: boolean;
  play_sound?: boolean;
  show_desktop_notification?: boolean;
}

// API Key Types
export interface APIKey {
  id: string;
  name: string;
  description?: string;
  key_prefix: string;
  scopes?: string[];
  rate_limit_per_minute: number;
  rate_limit_per_day: number;
  is_active: boolean;
  expires_at?: string;
  last_used_at?: string;
  last_used_ip?: string;
  usage_count: number;
  created_at: string;
  revoked_at?: string;
}

export interface APIKeyCreate {
  name: string;
  description?: string;
  scopes?: string[];
  expires_in_days?: number;
  rate_limit_per_minute?: number;
  rate_limit_per_day?: number;
}

export interface APIKeyCreateResponse extends APIKey {
  api_key: string; // The actual key - only shown once!
  message: string;
}

export interface APIKeyUpdate {
  name?: string;
  description?: string;
  scopes?: string[];
  rate_limit_per_minute?: number;
  rate_limit_per_day?: number;
  is_active?: boolean;
}

export interface APIKeyListResponse {
  api_keys: APIKey[];
  total: number;
}

export interface APIKeyUsageStats {
  key_id: string;
  key_name: string;
  period_days: number;
  total_requests: number;
  lifetime_requests: number;
  last_used_at?: string;
  top_endpoints: Array<{ endpoint: string; count: number }>;
}

// ==================== Repository Report Types ====================

export type RepoReportStatus = 'pending' | 'analyzing' | 'generating' | 'uploading' | 'completed' | 'failed' | 'cancelled';
export type RepoReportOutputFormat = 'docx' | 'pdf' | 'pptx';
export type RepoReportStyle = 'professional' | 'casual' | 'technical' | 'modern' | 'minimal' | 'corporate' | 'creative' | 'dark';

export interface RepoReportSection {
  id: string;
  name: string;
  description: string;
  default: boolean;
}

export interface RepoReportJobCreate {
  source_id?: string;
  repo_url?: string;
  repo_token?: string;
  output_format: RepoReportOutputFormat;
  title?: string;
  sections?: string[];
  slide_count?: number;
  include_diagrams?: boolean;
  style?: RepoReportStyle;
  custom_theme?: ThemeConfig;
}

export interface RepoReportJob {
  id: string;
  user_id: string;
  source_id?: string;
  adhoc_url?: string;
  repo_name: string;
  repo_url: string;
  repo_type: 'github' | 'gitlab';
  output_format: RepoReportOutputFormat;
  title: string;
  sections: string[];
  slide_count?: number;
  include_diagrams: boolean;
  style: RepoReportStyle;
  custom_theme?: ThemeConfig;
  status: RepoReportStatus;
  progress: number;
  current_stage?: string;
  file_path?: string;
  file_size?: number;
  error?: string;
  created_at: string;
  started_at?: string;
  completed_at?: string;
}

export interface RepoReportJobListItem {
  id: string;
  user_id: string;
  repo_name: string;
  repo_url: string;
  repo_type: 'github' | 'gitlab';
  output_format: RepoReportOutputFormat;
  title: string;
  status: RepoReportStatus;
  progress: number;
  file_size?: number;
  error?: string;
  created_at: string;
  started_at?: string;
  completed_at?: string;
}

export interface RepoReportJobListResponse {
  jobs: RepoReportJobListItem[];
  total: number;
}

export interface AvailableSectionsResponse {
  sections: RepoReportSection[];
}

export interface RepoReportProgressUpdate {
  type: 'progress';
  progress: number;
  stage: string;
  status: RepoReportStatus;
  error?: string;
}

// ==================== MCP Configuration Types ====================

export interface MCPToolInfo {
  name: string;
  display_name: string;
  description: string;
  category: string;
  required_scope: string;
  config_schema: Record<string, any>;
}

export interface MCPToolConfigResponse {
  tool_name: string;
  display_name: string;
  description: string;
  category: string;
  is_enabled: boolean;
  config?: Record<string, any>;
}

export interface MCPSourceAccessResponse {
  source_id: string;
  source_name: string;
  source_type: string;
  can_read: boolean;
  can_search: boolean;
  can_chat: boolean;
}

export interface MCPKeyConfigResponse {
  api_key_id: string;
  api_key_name: string;
  mcp_enabled: boolean;
  allowed_tools?: string[];
  source_access_mode: string;
  tool_configs: MCPToolConfigResponse[];
  source_access: MCPSourceAccessResponse[];
}

export interface MCPKeyConfigUpdate {
  mcp_enabled?: boolean;
  allowed_tools?: string[];
  source_access_mode?: string;
}

export interface MCPToolConfigUpdate {
  tool_name: string;
  is_enabled: boolean;
  config?: Record<string, any>;
}

export interface MCPSourceAccessUpdate {
  source_id: string;
  can_read: boolean;
  can_search: boolean;
  can_chat: boolean;
}

// ==================== Autonomous Agent Job Types ====================

export type AgentJobStatus = 'pending' | 'running' | 'paused' | 'completed' | 'failed' | 'cancelled';
export type AgentJobType = 'research' | 'monitor' | 'analysis' | 'synthesis' | 'knowledge_expansion' | 'data_analysis' | 'custom';

export type ChainTriggerCondition = 'on_complete' | 'on_fail' | 'on_any_end' | 'on_progress' | 'on_findings';

export interface ChainConfig {
  trigger_condition: ChainTriggerCondition;
  progress_threshold?: number;
  findings_threshold?: number;
  inherit_results?: boolean;
  inherit_config?: boolean;
  child_jobs?: Array<{
    name: string;
    job_type: AgentJobType;
    goal: string;
    config?: Record<string, any>;
    chain_config?: ChainConfig;
    max_iterations?: number;
    max_tool_calls?: number;
    max_llm_calls?: number;
    max_runtime_minutes?: number;
  }>;
}

export interface AgentJobCreate {
  name: string;
  description?: string;
  job_type: AgentJobType;
  goal: string;
  goal_criteria?: Record<string, any>;
  config?: Record<string, any>;
  agent_definition_id?: string;
  max_iterations?: number;
  max_tool_calls?: number;
  max_llm_calls?: number;
  max_runtime_minutes?: number;
  schedule_type?: string;
  schedule_cron?: string;
  start_immediately?: boolean;
  // Chain configuration
  chain_config?: ChainConfig;
  parent_job_id?: string;
}

export interface AgentJobFromTemplate {
  template_id: string;
  name: string;
  goal?: string;
  config?: Record<string, any>;
  start_immediately?: boolean;
  chain_config?: ChainConfig;
}

export interface AgentJobUpdate {
  name?: string;
  description?: string;
  goal?: string;
  goal_criteria?: Record<string, any>;
  config?: Record<string, any>;
  max_iterations?: number;
  max_tool_calls?: number;
  max_llm_calls?: number;
  max_runtime_minutes?: number;
  schedule_type?: string;
  schedule_cron?: string;
}

export interface AgentJobSwarmSummary {
  enabled: boolean;
  configured: boolean;
  fan_in_enabled: boolean;
  fan_in_group_id?: string;
  roles?: string[];
  role_count?: number;
  expected_siblings?: number;
  received_siblings?: number;
  terminal_siblings?: number;
  consensus_count?: number;
  consensus_findings?: string[];
  conflict_count?: number;
  conflicts?: Array<Record<string, any>>;
  action_plan?: Array<Record<string, any>>;
  confidence?: Record<string, any>;
  winning_slice_id?: string;
  winning_role?: string;
  promotion_reason?: string;
  review_state?: string;
  review_reason?: string;
  review_required?: boolean;
  tie_breaker_attempted?: boolean;
  tie_breaker_job_id?: string;
  tie_breaker_source_job_id?: string;
  file_converged?: boolean;
  file_convergence_support?: number;
  top_file_cluster?: Record<string, any> | null;
  command_converged?: boolean;
  command_convergence_support?: number;
  top_command_cluster?: Record<string, any> | null;
  repair_chain_job_id?: string;
  candidate_paths?: Array<Record<string, any>>;
  recommended_commands?: string[];
  owner_user_id?: string;
  shared_review?: boolean;
  shared_with_user_ids?: string[];
  assigned_user_id?: string;
  assigned_at?: string;
  assigned_by_user_id?: string;
  review_note?: string;
}

export interface AgentJobGoalContractSummary {
  enabled: boolean;
  satisfied: boolean;
  missing_count?: number;
  missing?: string[];
  strict_completion?: boolean;
  satisfied_iteration?: number;
  metrics?: Record<string, any>;
}

export interface AgentJobApprovalCheckpoint {
  required: boolean;
  status?: string;
  current_phase?: string;
  message?: string;
  iteration?: number;
  reasons?: string[];
  action?: Record<string, any>;
  created_at?: string;
}

export interface AgentJobExecutiveDigest {
  goal?: string;
  status?: string;
  outcome?: string;
  metrics?: Record<string, any>;
  key_findings?: string[];
  risks?: string[];
  next_actions?: string[];
  goal_contract?: Record<string, any>;
}

export interface AgentJobExecutionGraphDagStats {
  total_nodes?: number;
  total_edges?: number;
  node_type_counts?: Record<string, number>;
  edge_type_counts?: Record<string, number>;
  root_nodes?: number;
  leaf_nodes?: number;
  orphan_nodes?: number;
  blocked_nodes?: number;
  successful_nodes?: number;
  has_cycle?: boolean;
  critical_path_length?: number;
}

export interface AgentJobExecutionGraphHealth {
  status?: 'ok' | 'warning' | 'critical' | 'unknown' | string;
  reasons?: string[];
  severity_score?: number;
  blocked_ratio?: number;
}

export interface AgentJobExecutionGraph {
  verification_attempts?: number;
  verification_successes?: number;
  summarization_attempts?: number;
  summarization_successes?: number;
  nodes?: Array<Record<string, any>>;
  edges?: Array<Record<string, any>>;
  dag_stats?: AgentJobExecutionGraphDagStats;
  graph_health?: AgentJobExecutionGraphHealth;
  recommended_actions?: string[];
}

export interface AgentJobFeedback {
  id: string;
  job_id?: string;
  rating: number;
  feedback?: string;
  target_type: 'job' | 'checkpoint' | 'finding' | 'action' | 'tool' | string;
  target_id?: string;
  scope: 'user' | 'customer' | 'team' | string;
  preferred_tools: string[];
  discouraged_tools: string[];
  checkpoint?: string;
  created_at?: string;
}

export interface AgentJobFeedbackCreate {
  rating: number;
  feedback?: string;
  target_type?: 'job' | 'checkpoint' | 'finding' | 'action' | 'tool' | string;
  target_id?: string;
  scope?: 'user' | 'customer' | 'team' | string;
  team_key?: string;
  preferred_tools?: string[];
  discouraged_tools?: string[];
  checkpoint?: string;
}

export interface AgentJobFeedbackListResponse {
  items: AgentJobFeedback[];
  total: number;
}

export interface AgentJobExtractedMemory {
  id: string;
  type: string;
  content: string;
  importance_score: number;
  tags: string[];
}

export interface AgentJobMemoryExtractResponse {
  job_id: string;
  memories_created: number;
  parsed_count: number;
  candidate_count: number;
  skipped_duplicates: number;
  is_relaunch_chain: boolean;
  relaunch_root_job_id?: string | null;
  memories: AgentJobExtractedMemory[];
}

export interface AgentJobMemory {
  id: string;
  job_id: string;
  type: string;
  content: string;
  importance_score: number;
  tags: string[];
  context?: Record<string, any>;
  access_count: number;
  created_at?: string | null;
}

export interface AgentJobMemoryListResponse {
  job_id: string;
  memories: AgentJobMemory[];
  total: number;
}

export interface AgentJobMemoryDeleteResponse {
  job_id: string;
  deleted_count: number;
}

export interface AgentJobMemoryStatsMostAccessedItem {
  id: string;
  type: string;
  content: string;
  access_count: number;
}

export interface AgentJobMemoryStatsMostImportantItem {
  id: string;
  type: string;
  content: string;
  importance: number;
}

export interface AgentJobMemoryStatsResponse {
  total_memories: number;
  by_type: Record<string, number>;
  job_sourced: number;
  chat_sourced: number;
  manual: number;
  most_accessed: AgentJobMemoryStatsMostAccessedItem[];
  most_important: AgentJobMemoryStatsMostImportantItem[];
}

export interface AgentJobMemorySearchItem {
  id: string;
  type: string;
  content: string;
  importance_score: number;
  tags: string[];
  job_id?: string | null;
  access_count: number;
  created_at?: string | null;
}

export interface AgentJobMemorySearchResponse {
  query: string;
  memories: AgentJobMemorySearchItem[];
  total: number;
}

export interface AgentTaskMemoryGraphNode {
  id: string;
  type: string;
  content: string;
  importance_score: number;
  tags: string[];
  job_id?: string | null;
  created_at?: string | null;
  project_scope?: string | null;
  execution_outcome?: string | null;
  strategy_signal?: string | null;
  access_count?: number;
}

export interface AgentTaskMemoryGraphEdge {
  source: string;
  target: string;
  weight: number;
  reasons?: string[];
}

export interface AgentTaskMemoryGraph {
  nodes: AgentTaskMemoryGraphNode[];
  edges: AgentTaskMemoryGraphEdge[];
  stats: Record<string, any>;
  job_id?: string;
}

export interface AgentControlRunRoutingSummary {
  provider?: string | null;
  model?: string | null;
  routing_tier?: string | null;
  requested_tier?: string | null;
  request_count: number;
  summary?: string | null;
}

export interface AgentControlRunNode {
  id: string;
  kind: string;
  label: string;
  status?: string | null;
  stage?: string | null;
  timestamp?: string | null;
  metadata: Record<string, any>;
}

export interface AgentControlRunEdge {
  source: string;
  target: string;
  relation: string;
  metadata: Record<string, any>;
}

export interface AgentControlRunReplaySummary {
  replayability_status: string;
  planner_summary?: string | null;
  router_summary?: string | null;
  executor_summary?: string | null;
  ended_at?: string | null;
}

export interface AgentControlRunLink {
  label: string;
  path: string;
}

export interface AgentControlRunReviewItem {
  run_id?: string | null;
  run_title?: string | null;
  run_source_type?: string | null;
  run_status?: string | null;
  review_type?: string | null;
  review_status?: string | null;
  reason_code?: string | null;
  reason_label?: string | null;
  source_kind?: string | null;
  source_id?: string | null;
  opportunity_id?: string | null;
  canonical_key?: string | null;
  title?: string | null;
  evidence_revision?: string | null;
  autonomy_state?: string | null;
  operator_note?: string | null;
  created_at?: string | null;
  action_path?: string | null;
  queue_path?: string | null;
  note_path?: string | null;
  synthesis_path?: string | null;
  item_type?: string | null;
  queue_item_key?: string | null;
  status?: string | null;
  summary?: string | null;
  evidence_summary?: string | null;
  customer?: string | null;
  job_id?: string | null;
  job_name?: string | null;
  job_type?: string | null;
  age_minutes?: number | null;
  priority_score?: number | null;
  sla_bucket?: string | null;
  escalation_level?: string | null;
  next_run_at?: string | null;
  backoff_until?: string | null;
  checkpoint?: Record<string, any> | null;
  checkpoint_action_draft?: Record<string, any> | null;
  scheduler_state?: Record<string, any> | null;
  follow_up_launch_status?: string | null;
  follow_up_review_status?: string | null;
  follow_up_recommendation_key?: string | null;
  recommendation_score?: number | null;
  follow_up_block_reason?: string | null;
  follow_up_budget_decision?: string | null;
  follow_up_budget_reason?: string | null;
  follow_up_customer_budget_decision?: string | null;
  follow_up_customer_budget_reason?: string | null;
  recommended_action?: string | null;
  policy_update_payload?: Record<string, any> | null;
  policy_rollback_payload?: Record<string, any> | null;
  policy_guardrail_action?: string | null;
  policy_guardrail_target_history_entry_id?: string | null;
  policy_guardrail_reasons?: string[] | null;
  budget_throttle_state?: string | null;
  budget_reason?: string | null;
  customer_budget_throttle_state?: string | null;
  customer_budget_reason?: string | null;
  available_actions?: string[] | null;
  can_acknowledge?: boolean;
  can_approve?: boolean;
  can_reject?: boolean;
  can_defer?: boolean;
  can_launch_follow_up?: boolean;
  can_relaunch_follow_up?: boolean;
  can_skip?: boolean;
  can_restart?: boolean;
  can_resume?: boolean;
  can_cancel?: boolean;
  metadata: Record<string, any>;
}

export interface AgentControlRunReviewActionRequest {
  review_type: string;
  source_kind: string;
  source_id: string;
  opportunity_id: string;
  action: string;
  operator_note?: string | null;
  reason_code?: string | null;
  checkpoint_action_patch?: Record<string, any> | null;
}

export interface AgentControlRunReviewActionResponse {
  ok: boolean;
  action: string;
  review_type?: string | null;
  source_kind?: string | null;
  source_id?: string | null;
  opportunity_id?: string | null;
  detail?: string | null;
  monitor_job_id?: string | null;
  follow_up_launch_status?: string | null;
  follow_up_operator_decision?: string | null;
  follow_up_job_id?: string | null;
}

export interface AgentControlRunBulkReviewActionRequest {
  item_type: string;
  action: string;
  job_ids?: string[];
  domain_research_profile_id?: string | null;
  profile_opportunity_ids?: string[];
  portfolio_id?: string | null;
  portfolio_opportunity_ids?: string[];
  operator_note?: string | null;
}

export interface AgentControlRunBulkReviewActionResult {
  item_key?: string | null;
  job_id?: string | null;
  opportunity_id?: string | null;
  ok: boolean;
  detail?: string | null;
  error?: string | null;
  status?: string | null;
  follow_up_launch_status?: string | null;
  follow_up_operator_decision?: string | null;
  follow_up_job_id?: string | null;
}

export interface AgentControlRunBulkReviewActionResponse {
  ok: boolean;
  item_type: string;
  action: string;
  requested_count: number;
  applied: number;
  failed: number;
  results: AgentControlRunBulkReviewActionResult[];
}

export interface AgentControlRunSummary {
  id: string;
  source_type: string;
  title: string;
  subtitle?: string | null;
  status: string;
  outcome?: string | null;
  created_at: string;
  started_at?: string | null;
  completed_at?: string | null;
  root_job_id?: string | null;
  workflow_execution_id?: string | null;
  child_job_count: number;
  child_execution_count: number;
  linked_note_count: number;
  linked_experiment_count: number;
  decision_count: number;
  replayability_status: string;
  automation_profile?: string | null;
  routing?: AgentControlRunRoutingSummary | null;
  queued_operator_review_count: number;
  queued_operator_reviews_by_type?: Record<string, number> | null;
}

export interface AgentControlRunListResponse {
  items: AgentControlRunSummary[];
  total: number;
}

export interface AgentControlRunView {
  id: string;
  user_id: string;
  name: string;
  filters: Record<string, any>;
  is_default: boolean;
  created_at: string;
  updated_at: string;
}

export interface AgentControlRunReviewListResponse {
  items: AgentControlRunReviewItem[];
  total: number;
  summary?: {
    total: number;
    by_type?: Record<string, number> | null;
    by_sla_bucket?: Record<string, number> | null;
    by_status?: Record<string, number> | null;
    by_customer?: Record<string, number> | null;
    by_escalation?: Record<string, number> | null;
  } | null;
  offset?: number;
  limit?: number;
  has_more?: boolean;
}

export interface AgentControlRunViewListResponse {
  items: AgentControlRunView[];
  total: number;
}

export interface AgentControlRunViewCreateRequest {
  name: string;
  filters: Record<string, any>;
  is_default?: boolean;
}

export interface AgentControlRunViewUpdateRequest {
  name?: string;
  filters?: Record<string, any>;
  is_default?: boolean;
}

export interface AgentControlRunDetail {
  run: AgentControlRunSummary;
  nodes: AgentControlRunNode[];
  edges: AgentControlRunEdge[];
  decision_trace: AgentDecisionTraceEvent[];
  memory_graph?: AgentTaskMemoryGraph | null;
  routing?: AgentControlRunRoutingSummary | null;
  replay: AgentControlRunReplaySummary;
  related_links: AgentControlRunLink[];
  queued_operator_review_count: number;
  queued_operator_reviews: AgentControlRunReviewItem[];
  policy_summary: Record<string, any>;
  metadata: Record<string, any>;
}

export interface AgentJobExperimentRun {
  source_id?: string;
  source_name?: string;
  enabled?: boolean;
  backend?: string;
  commands?: string[];
  verification_commands?: string[];
  bootstrap_commands?: string[];
  fallback_commands?: string[];
  runs?: Array<Record<string, any>>;
  ok?: boolean | null;
  final_phase?: string;
  phases?: string[];
  verification_phases?: string[];
  failed_commands?: string[];
  proposal_id?: string | null;
  latex_project_id?: string | null;
  latex_updated?: boolean;
  inferred_project_profile?: Record<string, any> | null;
  bootstrap_attempted?: boolean;
  bootstrap_ok?: boolean | null;
  bootstrap_used?: boolean;
  fallback_attempted?: boolean;
  fallback_ok?: boolean | null;
  fallback_used?: boolean;
  note?: string;
  summary?: string;
}

export interface AgentJobOperatorIntervention {
  action: string;
  actor_user_id?: string | null;
  at?: string;
  note?: string | null;
  job_status_before?: string | null;
  job_status_after?: string | null;
  outcome_status?: string | null;
  outcome_reason?: string | null;
  resolved_at?: string | null;
  metadata?: Record<string, any> | null;
}

export interface AgentJobCodingWorkspaceSummary {
  created?: boolean;
  workspace_id?: string | null;
  source_type?: string | null;
  file_count?: number;
  base_path?: string | null;
  error?: string | null;
}

export interface AgentJobVerificationPlan {
  commands?: string[];
  bootstrap_commands?: string[];
  fallback_commands?: string[];
  auto_inferred?: boolean;
}

export interface AgentJobExecutionPlanStep {
  step_id?: string;
  title?: string;
  status?: string;
  objective?: string;
  commands?: string[];
}

export interface AgentJobCodePatchRecovery {
  recovery_state?: string;
  last_failed_commands?: string[];
  retry_reason?: string | null;
  resume_hint?: string | null;
  suggested_operator_actions?: string[];
  can_retry_with_refined_plan?: boolean;
  can_resume_verification?: boolean;
  latest_failed_output?: string | null;
}

export interface AgentJobCodePatchExecution {
  mode?: string;
  source_id?: string;
  source_name?: string;
  source_type?: string;
  scope?: string;
  failure_symptom?: string;
  error_output?: string;
  workspace?: AgentJobCodingWorkspaceSummary | null;
  inferred_project_profile?: Record<string, any> | null;
  verification_plan?: AgentJobVerificationPlan | null;
  execution_plan?: AgentJobExecutionPlanStep[];
  proposal_strategy?: string;
  recovery?: AgentJobCodePatchRecovery | null;
}

export interface CollaborationSummary {
  owner_user_id?: string | null;
  owner_label?: string | null;
  assigned_user_id?: string | null;
  assignee_label?: string | null;
  assigned_by_user_id?: string | null;
  assigned_at?: string | null;
  shared_with_user_ids?: string[] | null;
  visibility_scope?: string | null;
  is_owned_by_current_user?: boolean;
  is_assigned_to_current_user?: boolean;
  is_shared_with_current_user?: boolean;
  note?: string | null;
}

export interface CodingBacklogItem {
  id: string;
  user_id: string;
  source_id?: string | null;
  title: string;
  portfolio_goal: string;
  status: string;
  priority: number;
  scope?: string | null;
  failure_symptom?: string | null;
  error_output?: string | null;
  file_paths?: string[] | null;
  commands?: string[] | null;
  auto_apply_enabled: boolean;
  require_patch_pr: boolean;
  visibility?: string;
  shared_with_user_ids?: string[] | null;
  assigned_user_id?: string | null;
  assigned_by_user_id?: string | null;
  assigned_at?: string | null;
  collaboration?: Record<string, any> | null;
  collaboration_summary?: CollaborationSummary | null;
  operator_queue_state?: string | null;
  closure_reason?: string | null;
  why_not_repair?: Record<string, any> | null;
  policy?: CodingBacklogPolicy | null;
  lineage?: Record<string, any> | null;
  decomposition?: CodingBacklogDecomposition | null;
  child_job_ids?: string[] | null;
  latest_summary?: CodingBacklogLatestSummary | null;
  orchestrator_job_id?: string | null;
  current_job_id?: string | null;
  latest_apply_job_id?: string | null;
  latest_proposal_id?: string | null;
  created_at: string;
  updated_at: string;
  started_at?: string | null;
  completed_at?: string | null;
}

export interface CodingBacklogItemCreate {
  title: string;
  portfolio_goal: string;
  source_id: string;
  scope?: string;
  priority?: number;
  failure_symptom?: string;
  error_output?: string;
  file_paths?: string[];
  commands?: string[];
  auto_apply_enabled?: boolean;
  require_patch_pr?: boolean;
  visibility?: string;
  shared_with_user_ids?: string[];
  assigned_user_id?: string;
  assigned_by_user_id?: string;
  assigned_at?: string;
  collaboration?: Record<string, any>;
  policy?: CodingBacklogPolicy;
  lineage?: Record<string, any>;
  start_immediately?: boolean;
}

export interface CodingBacklogItemUpdate {
  title?: string;
  portfolio_goal?: string;
  scope?: string;
  priority?: number;
  failure_symptom?: string;
  error_output?: string;
  file_paths?: string[];
  commands?: string[];
  auto_apply_enabled?: boolean;
  require_patch_pr?: boolean;
  visibility?: string;
  shared_with_user_ids?: string[];
  assigned_user_id?: string;
  assigned_by_user_id?: string;
  assigned_at?: string;
  collaboration?: Record<string, any>;
  policy?: CodingBacklogPolicy;
  lineage?: Record<string, any>;
  decomposition?: CodingBacklogDecomposition;
}

export interface CodingBacklogItemListResponse {
  items: CodingBacklogItem[];
  total: number;
  limit: number;
  offset: number;
}

export interface CodingSwarmProfile {
  id: string;
  user_id: string;
  source_id: string;
  title: string;
  description?: string | null;
  status: string;
  preset_key: string;
  scope_default: string;
  default_commands?: string[] | null;
  default_file_paths?: string[] | null;
  max_agents: number;
  safe_command_policy: string;
  saved_search_query?: string | null;
  is_default: boolean;
  visibility?: 'private' | 'shared' | string;
  shared_with_user_ids?: string[];
  collaboration_summary?: CollaborationSummary | null;
  latest_job_id?: string | null;
  profile_metadata?: Record<string, any> | null;
  created_at: string;
  updated_at: string;
}

export interface CodingSwarmProfileCreate {
  title: string;
  source_id: string;
  preset_key: string;
  description?: string;
  scope_default?: string;
  default_commands?: string[];
  default_file_paths?: string[];
  max_agents?: number;
  safe_command_policy?: string;
  saved_search_query?: string;
  is_default?: boolean;
  visibility?: 'private' | 'shared' | string;
  shared_with_user_ids?: string[];
  profile_metadata?: Record<string, any>;
}

export interface CodingSwarmProfileUpdate {
  title?: string;
  description?: string;
  preset_key?: string;
  scope_default?: string;
  default_commands?: string[];
  default_file_paths?: string[];
  max_agents?: number;
  safe_command_policy?: string;
  saved_search_query?: string;
  is_default?: boolean;
  status?: string;
  visibility?: 'private' | 'shared' | string;
  shared_with_user_ids?: string[];
  profile_metadata?: Record<string, any>;
}

export interface CodingSwarmProfileListResponse {
  items: CodingSwarmProfile[];
  total: number;
  limit: number;
  offset: number;
}

export interface DomainResearchProfile {
  id: string;
  user_id: string;
  title: string;
  domain: string;
  objective: string;
  customer_context?: string | null;
  status: string;
  source_scope: string;
  track_type: 'compiler' | 'microarchitecture' | 'generic' | string;
  research_mode: string;
  monitor_queries?: string[] | null;
  repo_source_ids?: string[] | null;
  benchmark_queries?: string[] | null;
  report_format: string;
  scoring_policy?: Record<string, any> | null;
  selection_policy?: Record<string, any> | null;
  validation_policy?: Record<string, any> | null;
  automation_profile?: 'balanced' | 'max_autonomy' | string | null;
  automation_policy?: Record<string, any> | null;
  effective_policy?: Record<string, any> | null;
  sandbox_profile_id?: string | null;
  interval_minutes: number;
  persist_artifacts: boolean;
  auto_launch_follow_up: boolean;
  auto_create_experiment_plans: boolean;
  confidence_threshold: number;
  max_documents: number;
  max_papers: number;
  opportunities?: ResearchOpportunity[] | null;
  latest_summary?: SharedAutonomySummary | null;
  latest_note_ids?: string[] | null;
  latest_experiment_plan_ids?: string[] | null;
  latest_validation_run_ids?: string[] | null;
  latest_validation_runs?: ScientificValidationRunSummary[] | null;
  latest_run_job_id?: string | null;
  active_job_id?: string | null;
  created_at: string;
  updated_at: string;
  started_at?: string | null;
  paused_at?: string | null;
  last_run_at?: string | null;
}

export interface DomainResearchProfileCreate {
  title: string;
  domain: string;
  objective: string;
  customer_context?: string;
  source_scope?: 'kb_only' | 'arxiv_only' | 'kb_plus_arxiv' | 'kb_plus_arxiv_plus_repo' | string;
  track_type?: 'compiler' | 'microarchitecture' | 'generic' | string;
  research_mode?: 'literature_to_hypothesis' | string;
  monitor_queries?: string[];
  repo_source_ids?: string[];
  benchmark_queries?: string[];
  report_format?: 'brief_only' | 'report_only' | 'brief_and_report' | string;
  scoring_policy?: Record<string, any>;
  selection_policy?: Record<string, any>;
  validation_policy?: Record<string, any>;
  automation_profile?: 'balanced' | 'max_autonomy' | string;
  automation_policy?: Record<string, any>;
  sandbox_profile_id?: string;
  interval_minutes?: number;
  persist_artifacts?: boolean;
  auto_launch_follow_up?: boolean;
  auto_create_experiment_plans?: boolean;
  confidence_threshold?: number;
  max_documents?: number;
  max_papers?: number;
  start_immediately?: boolean;
}

export interface DomainResearchProfileUpdate {
  title?: string;
  objective?: string;
  customer_context?: string;
  source_scope?: 'kb_only' | 'arxiv_only' | 'kb_plus_arxiv' | 'kb_plus_arxiv_plus_repo' | string;
  track_type?: 'compiler' | 'microarchitecture' | 'generic' | string;
  research_mode?: 'literature_to_hypothesis' | string;
  monitor_queries?: string[];
  repo_source_ids?: string[];
  benchmark_queries?: string[];
  report_format?: 'brief_only' | 'report_only' | 'brief_and_report' | string;
  scoring_policy?: Record<string, any>;
  selection_policy?: Record<string, any>;
  validation_policy?: Record<string, any>;
  automation_profile?: 'balanced' | 'max_autonomy' | string;
  automation_policy?: Record<string, any>;
  sandbox_profile_id?: string;
  interval_minutes?: number;
  persist_artifacts?: boolean;
  auto_launch_follow_up?: boolean;
  auto_create_experiment_plans?: boolean;
  confidence_threshold?: number;
  max_documents?: number;
  max_papers?: number;
}

export interface DomainResearchProfileActionRequest {
  action: 'start' | 'pause' | 'resume' | 'cancel' | 'run_now' | string;
}

export type ResearchOpportunityStage =
  | 'discovered'
  | 'accepted'
  | 'suppressed'
  | 'planned'
  | 'validating'
  | 'completed'
  | 'blocked'
  | string;

export type ResearchOpportunityDecisionState =
  | 'pending_review'
  | 'accepted'
  | 'suppressed'
  | 'auto_accepted'
  | string;

export interface ResearchOpportunity {
  opportunity_id: string;
  canonical_key: string;
  title: string;
  hypothesis: string;
  stage: ResearchOpportunityStage;
  decision_state: ResearchOpportunityDecisionState;
  decision_source?: 'system' | 'operator' | string;
  operator_note?: string | null;
  supporting_evidence?: string[];
  supporting_sources?: Array<Record<string, any>>;
  next_steps?: string[];
  source_profile_ids?: string[];
  source_job_ids?: string[];
  source_note_ids?: string[];
  linked_experiment_plan_ids?: string[];
  linked_validation_run_ids?: string[];
  latest_experiment_plan_id?: string | null;
  latest_validation_run_id?: string | null;
  latest_validation_job_id?: string | null;
  latest_validation_status?: string | null;
  latest_validation_blocked_reason_code?: string | null;
  child_job_ids?: string[];
  source_repo_ids?: string[];
  confidence?: number;
  novelty?: number;
  readiness?: number;
  track_type?: string | null;
  autonomy_state?: 'eligible' | 'cooldown' | 'blocked_structural' | 'completed_waiting_change' | 'active' | string | null;
  last_evaluated_at?: string | null;
  next_eligible_at?: string | null;
  evidence_revision?: string | null;
  last_material_change_at?: string | null;
  last_decision_type?: string | null;
  last_decision_reason_code?: string | null;
  portfolio_config_revision?: string | null;
  last_skip_reason_code?: string | null;
  last_blocked_reason_code?: string | null;
  follow_up_review_status?: string | null;
  follow_up_reviewed_at?: string | null;
  follow_up_reviewed_by_user_id?: string | null;
  follow_up_review_note?: string | null;
  follow_up_review_evidence_revision?: string | null;
  last_reevaluation_review_outcome?: string | null;
  last_reevaluation_reviewed_at?: string | null;
  last_reevaluation_review_job_id?: string | null;
  last_reevaluation_review_note?: string | null;
  last_reevaluation_review_source_note_id?: string | null;
  last_reevaluation_review_target_note_id?: string | null;
  updated_at?: string | null;
}

export interface AutonomySchedulerSummary {
  scheduling_mode?: string | null;
  next_run_at?: string | null;
  last_evaluated_at?: string | null;
  last_dispatched_at?: string | null;
  launches_count?: number;
  queued_approvals_count?: number;
  pending_follow_up_approvals_count?: number;
  manual_recommendations_count?: number;
  manual_follow_up_recommendations_count?: number;
  blocked_by_policy_count?: number;
  blocked_by_budget_count?: number;
  suppressed_relaunches_count?: number;
  [key: string]: any;
}

export interface AutonomyReviewSummaryItem {
  opportunity_id?: string | null;
  canonical_key?: string | null;
  title?: string | null;
  review_type?: string | null;
  reason_code?: string | null;
  operator_note?: string | null;
  [key: string]: any;
}

export interface SharedAutonomySummary {
  autonomy_mode?: string | null;
  effective_policy?: Record<string, any> | null;
  stage_counts?: Record<string, number> | null;
  autonomy_state_counts?: Record<string, number> | null;
  scheduler_summary?: AutonomySchedulerSummary | null;
  follow_up_review_counts?: Record<string, number> | null;
  queued_operator_reviews_count?: number;
  queued_operator_reviews_by_type?: Record<string, number> | null;
  queued_operator_reviews?: AutonomyReviewSummaryItem[] | null;
  pending_follow_up_approvals?: AutonomyReviewSummaryItem[] | null;
  manual_follow_up_recommendations?: AutonomyReviewSummaryItem[] | null;
  suppressed_relaunches?: AutonomyReviewSummaryItem[] | null;
  [key: string]: any;
}

export interface ResearchOpportunityActionRequest {
  action:
    | 'accept'
    | 'suppress'
    | 'reopen'
    | 'create_plan'
    | 'launch_validation'
    | 'materialize_experiment'
    | 'launch_follow_up'
    | 'relaunch_follow_up'
    | string;
  operator_note?: string;
  start_immediately?: boolean;
}

export interface DomainResearchProfileListResponse {
  items: DomainResearchProfile[];
  total: number;
}

export interface ResearchPortfolio {
  id: string;
  user_id: string;
  title: string;
  objective: string;
  status: string;
  linked_profile_ids?: string[] | null;
  automation_profile?: 'balanced' | 'max_autonomy' | string | null;
  automation_policy?: Record<string, any> | null;
  effective_policy?: Record<string, any> | null;
  sandbox_profile_id?: string | null;
  opportunities?: ResearchOpportunity[] | null;
  latest_summary?: SharedAutonomySummary | null;
  latest_note_ids?: string[] | null;
  latest_experiment_plan_ids?: string[] | null;
  latest_validation_run_ids?: string[] | null;
  latest_validation_runs?: ScientificValidationRunSummary[] | null;
  child_job_ids?: string[] | null;
  active_job_id?: string | null;
  latest_run_job_id?: string | null;
  created_at: string;
  updated_at: string;
  started_at?: string | null;
  paused_at?: string | null;
  last_run_at?: string | null;
}

export interface ResearchPortfolioCreate {
  title: string;
  objective: string;
  linked_profile_ids: string[];
  automation_profile?: 'balanced' | 'max_autonomy' | string;
  automation_policy?: Record<string, any>;
  sandbox_profile_id?: string;
  start_immediately?: boolean;
}

export interface AgentJobPromoteDomainResearchProfileRequest {
  title?: string;
  domain?: string;
  objective?: string;
  customer_context?: string;
  source_scope?: 'kb_only' | 'arxiv_only' | 'kb_plus_arxiv' | 'kb_plus_arxiv_plus_repo' | string;
  track_type?: 'compiler' | 'microarchitecture' | 'generic' | string;
  research_mode?: 'literature_to_hypothesis' | string;
  monitor_queries?: string[];
  repo_source_ids?: string[];
  benchmark_queries?: string[];
  report_format?: 'brief_only' | 'report_only' | 'brief_and_report' | string;
  scoring_policy?: Record<string, any>;
  selection_policy?: Record<string, any>;
  automation_profile?: 'balanced' | 'max_autonomy' | string;
  automation_policy?: Record<string, any>;
  sandbox_profile_id?: string;
  interval_minutes?: number;
  persist_artifacts?: boolean;
  auto_launch_follow_up?: boolean;
  auto_create_experiment_plans?: boolean;
  confidence_threshold?: number;
  max_documents?: number;
  max_papers?: number;
}

export interface AgentJobPromoteDomainResearchPortfolioRequest {
  title?: string;
  objective?: string;
  sandbox_profile_id?: string;
  automation_profile?: 'balanced' | 'max_autonomy' | string;
  automation_policy?: Record<string, any>;
}

export interface AgentJobPromoteDomainResearchRequest {
  target_mode?: 'profile_only' | 'profile_with_portfolio' | string;
  profile?: AgentJobPromoteDomainResearchProfileRequest;
  portfolio_id?: string;
  portfolio?: AgentJobPromoteDomainResearchPortfolioRequest;
  start_profile_now?: boolean;
  run_portfolio_now?: boolean;
}

export interface AgentJobPromoteDomainResearchResponse {
  source_job_id: string;
  promotion_status: string;
  domain_research_profile_id: string;
  research_portfolio_id?: string | null;
  profile: DomainResearchProfile;
  portfolio?: ResearchPortfolio | null;
  source_job?: AgentJob | null;
}

export interface ResearchPortfolioUpdate {
  title?: string;
  objective?: string;
  linked_profile_ids?: string[];
  automation_profile?: 'balanced' | 'max_autonomy' | string;
  automation_policy?: Record<string, any>;
  sandbox_profile_id?: string;
}

export interface ResearchPortfolioActionRequest {
  action: 'start' | 'pause' | 'resume' | 'cancel' | 'run_now' | string;
}

export interface ResearchPortfolioListResponse {
  items: ResearchPortfolio[];
  total: number;
}

export interface CodingBacklogPolicy {
  max_auto_retries?: number;
  max_files_touched?: number;
  blocked_path_prefixes?: string[];
  require_experiments_ok?: boolean;
  confidence_threshold?: number;
  [key: string]: any;
}

export interface CodingBacklogPortfolioProgress {
  total_slices: number;
  pending_slices: number;
  completed_slices: number;
  failed_slices: number;
  auto_applied_slices: number;
  proposal_only_slices: number;
}

export interface CodingBacklogSlice {
  slice_id: string;
  title: string;
  status: string;
  scope?: string | null;
  file_paths?: string[] | null;
  commands?: string[] | null;
  search_query?: string | null;
  goal?: string | null;
  retry_count?: number;
  selected_proposal_id?: string | null;
  promotion_decision?: string | null;
  blocked_reason?: string | null;
  child_job_id?: string | null;
  apply_job_id?: string | null;
  proposal_confidence?: number | null;
  files_touched?: string[] | null;
  started_at?: string | null;
  completed_at?: string | null;
  status_reason?: string | null;
  awaiting_operator_action?: boolean;
  allowed_slice_actions?: string[] | null;
  recommended_next_action?: string | null;
  operator_decision?: string | null;
  operator_note?: string | null;
  operator_acted_at?: string | null;
  patch_pr_id?: string | null;
  timeline?: CodingBacklogTimelineEntry[] | null;
  job_lineage?: CodingBacklogJobLineage | null;
  artifact_history?: CodingBacklogArtifactHistoryEntry[] | null;
  manual_promotion_history?: CodingBacklogManualPromotionEntry[] | null;
}

export interface CodingBacklogPromotionDecision {
  slice_id: string;
  title?: string | null;
  decision: string;
  proposal_id?: string | null;
  job_id?: string | null;
  apply_job_id?: string | null;
  blocked_reason?: string | null;
  proposal_confidence?: number | null;
  files_touched_count?: number | null;
}

export interface CodingBacklogDecomposition {
  strategy?: string | null;
  planned_slices: CodingBacklogSlice[];
  active_slice_id?: string | null;
  completed_slices?: string[] | null;
  failed_slices?: string[] | null;
  promotion_decisions?: CodingBacklogPromotionDecision[] | null;
  backlog_timeline?: CodingBacklogTimelineEntry[] | null;
  lineage_summary?: CodingBacklogLineageSummary | null;
  portfolio_progress?: CodingBacklogPortfolioProgress | null;
  [key: string]: any;
}

export interface CodingBacklogLatestSummary {
  status?: string | null;
  current_child_job_id?: string | null;
  retry_from_job_id?: string | null;
  selected_proposal_id?: string | null;
  promotion_decision?: string | null;
  blocked_reason?: string | null;
  active_slice_id?: string | null;
  active_slice_title?: string | null;
  promotion_evaluation?: Record<string, any> | null;
  portfolio_progress?: CodingBacklogPortfolioProgress | null;
  waiting_on_operator_action?: boolean;
  allowed_slice_actions?: string[] | null;
  recommended_next_action?: string | null;
  note?: string | null;
  [key: string]: any;
}

export interface CodingBacklogTimelineEntry {
  at: string;
  actor: string;
  action: string;
  slice_id?: string | null;
  previous_status?: string | null;
  new_status?: string | null;
  note?: string | null;
  job_id?: string | null;
  proposal_id?: string | null;
  patch_pr_id?: string | null;
  metadata?: Record<string, any> | null;
}

export interface CodingBacklogJobLineage {
  repair_job_ids?: string[] | null;
  apply_job_ids?: string[] | null;
  patch_pr_ids?: string[] | null;
  proposal_ids?: string[] | null;
  retry_from_job_ids?: string[] | null;
}

export interface CodingBacklogArtifactHistoryEntry {
  at: string;
  artifact_type: string;
  artifact_id: string;
  label?: string | null;
}

export interface CodingBacklogManualPromotionEntry {
  at: string;
  action: string;
  operator_note?: string | null;
  proposal_id?: string | null;
  patch_pr_id?: string | null;
  apply_job_id?: string | null;
}

export interface CodingBacklogLineageSummary {
  repair_job_count: number;
  apply_job_count: number;
  patch_pr_count: number;
  proposal_count: number;
  operator_action_count: number;
}

export interface CodingBacklogActionRequest {
  action:
    | 'start'
    | 'pause'
    | 'resume'
    | 'cancel'
    | 'close'
    | 'assign_backlog'
    | 'clear_backlog_assignment'
    | 'update_backlog_note'
    | 'apply_override'
    | 'create_patch_pr'
    | 'keep_proposal_only'
    | 'relaunch_slice'
    | 'skip_slice';
  slice_id?: string;
  assigned_user_id?: string;
  closure_reason?: string;
  operator_note?: string;
}

export interface AgentJob {
  id: string;
  name: string;
  description?: string;
  job_type: AgentJobType;
  goal: string;
  goal_criteria?: Record<string, any>;
  config?: Record<string, any>;
  launch_mode?: string;
  relaunch_from_job_id?: string;
  relaunch_children_count?: number;
  promotion_status?: string | null;
  promoted_domain_research_profile_id?: string | null;
  promoted_research_portfolio_id?: string | null;
  agent_definition_id?: string;
  agent_definition_name?: string;
  user_id: string;
  status: AgentJobStatus;
  progress: number;
  current_phase?: string;
  phase_details?: string;
  iteration: number;
  max_iterations: number;
  max_tool_calls: number;
  max_llm_calls: number;
  max_runtime_minutes: number;
  tool_calls_used: number;
  llm_calls_used: number;
  tokens_used: number;
  error?: string;
  error_count: number;
  schedule_type?: string;
  schedule_cron?: string;
  next_run_at?: string;
  scheduler_state?: Record<string, any> | null;
  results?: Record<string, any>;
  experiment_run?: AgentJobExperimentRun | null;
  experiment_runs?: AgentJobExperimentRun[] | null;
  operator_interventions?: AgentJobOperatorIntervention[] | null;
  output_artifacts?: Array<{type: string; id: string; title: string}>;
  created_at: string;
  started_at?: string;
  completed_at?: string;
  last_activity_at?: string;
  celery_task_id?: string;
  execution_log?: Array<Record<string, any>>;
  // Chain fields
  parent_job_id?: string;
  root_job_id?: string;
  chain_depth: number;
  chain_triggered: boolean;
  chain_config?: ChainConfig;
  swarm_summary?: AgentJobSwarmSummary;
  goal_contract_summary?: AgentJobGoalContractSummary;
  approval_checkpoint?: AgentJobApprovalCheckpoint;
  executive_digest?: AgentJobExecutiveDigest;
}

export interface AgentJobListResponse {
  jobs: AgentJob[];
  total: number;
  page: number;
  page_size: number;
  has_more: boolean;
}

export interface AutonomousRndVerificationBudget {
  repeat_count?: number;
  timeout_seconds?: number;
  max_runtime_minutes?: number;
  budget_limit?: number;
}

export interface AutonomousRndVerificationTask {
  task_id: string;
  evidence_id?: string | null;
  evidence_status?: string | null;
  priority?: string | null;
  priority_score?: number | null;
  required_checks: string[];
  launch_status: string;
  job_status?: string | null;
  approval_status?: string | null;
  reconciliation_status?: string | null;
  reconciliation_recorded_at?: string | null;
  experiment_plan_id?: string | null;
  experiment_run_id?: string | null;
  agent_job_id?: string | null;
  audit_id?: string | null;
  budget: AutonomousRndVerificationBudget;
}

export interface AutonomousRndVerificationTimelineEvent {
  event_id: string;
  task_id: string;
  event_type: string;
  at: string;
  actor: string;
  label: string;
  status?: string | null;
  entity_type?: string | null;
  entity_id?: string | null;
}

export interface AutonomousRndVerificationLifecycle {
  task_count: number;
  launch_status_counts: Record<string, number>;
  evidence_status_counts: Record<string, number>;
  tasks: AutonomousRndVerificationTask[];
  timeline: AutonomousRndVerificationTimelineEvent[];
}

export interface AutonomousRndJobOutcomeResponse {
  job_id: string;
  job_status: string;
  outcome: Record<string, any>;
  verification_lifecycle: AutonomousRndVerificationLifecycle;
}

export interface AutonomousRndVerificationLaunchRequest {
  approval_confirmed: true;
  approval_note: string;
  research_note_id: string;
  source_id: string;
  sandbox_profile_id: string;
  commands: string[];
  repeat_count: number;
  timeout_seconds: number;
  max_runtime_minutes: number;
  budget_limit: number;
  start_immediately: boolean;
}

export interface AutonomousRndVerificationLaunchResponse {
  created: boolean;
  queued: boolean;
  experiment_plan_id: string;
  experiment_run_id: string;
  agent_job_id: string;
  audit_id: string;
  status: string;
  budget: AutonomousRndVerificationBudget;
}

export interface AutonomousRndVerificationAuditEnvelope {
  snapshot: Record<string, any>;
  integrity: {
    canonicalization: string;
    sha256: string;
    signature_algorithm?: string;
    signature_encoding?: string;
    signature?: string;
    key_id?: string;
    public_key?: string;
  };
}

export interface ExternalAgentConnection {
  id: string;
  name: string;
  description?: string | null;
  provider_type: 'generic_agent' | 'compops' | 'mlflow';
  endpoint_url: string;
  capabilities: string[];
  auth_type: 'none' | 'bearer' | 'api_key' | 'basic' | string;
  secret_id?: string | null;
  auth_header_name?: string | null;
  timeout_seconds: number;
  is_enabled: boolean;
  version: number;
  created_at: string;
  updated_at: string;
}

export interface ExternalAgentConnectionList {
  agents: ExternalAgentConnection[];
  total: number;
}

export interface ExternalAgentInvokeResult {
  status: 'completed' | 'requires_approval' | 'failed';
  audit_id: string;
  output?: Record<string, any> | null;
  error?: string | null;
  evidence_linked?: boolean;
}

export interface CompOpsEvidenceSubscription {
  id: string;
  user_id: string;
  job_id: string;
  tool_id: string;
  capability: string;
  remote_id: string;
  payload: Record<string, any>;
  interval_minutes: number;
  is_enabled: boolean;
  status: string;
  last_response_sha256?: string | null;
  last_audit_id?: string | null;
  last_attempt_at?: string | null;
  last_success_at?: string | null;
  next_sync_at?: string | null;
  last_error?: string | null;
  webhook_enabled: boolean;
  last_webhook_at?: string | null;
  last_webhook_event_id?: string | null;
  created_at: string;
  updated_at: string;
}

export interface CompOpsEvidenceSubscriptionList {
  subscriptions: CompOpsEvidenceSubscription[];
  total: number;
}

export interface CompOpsEvidenceSyncResult {
  subscription: CompOpsEvidenceSubscription;
  evidence_changed: boolean;
}

export interface CompOpsWebhookSetup {
  subscription: CompOpsEvidenceSubscription;
  callback_path: string;
  signing_secret: string;
  signature_header: string;
  timestamp_header: string;
  event_id_header: string;
  signing_format: string;
}

export interface SecretSummary {
  id: string;
  name: string;
  created_at: string;
  updated_at: string;
}

export interface AgentCheckpointQueueAction {
  kind: string;
  label: string;
  description?: string | null;
  action?: string | null;
  recommended?: boolean;
  launch_label?: string | null;
  recommendation_key?: string | null;
  autonomy_eligibility?: string | null;
  recommendation_score?: number | null;
  recommendation_reasons?: string[] | null;
  job_create_payload?: Record<string, any> | null;
  chain_create_payload?: Record<string, any> | null;
  follow_up_action_payload?: Record<string, any> | null;
  policy_update_payload?: Record<string, any> | null;
  policy_rollback_payload?: Record<string, any> | null;
}

export interface AgentCheckpointQueueItem {
  queue_key: string;
  item_type: 'approval_checkpoint' | 'job_recovery' | 'follow_up_recommendation' | 'policy_review' | 'budget_review' | string;
  priority: number;
  priority_score?: number;
  title: string;
  summary?: string | null;
  evidence_summary?: string | null;
  status?: string | null;
  customer?: string | null;
  job_name?: string | null;
  job_type?: string | null;
  reason_code?: string | null;
  reason_label?: string | null;
  recommended_action?: string | null;
  age_minutes?: number;
  sla_bucket?: 'normal' | 'at_risk' | 'overdue' | string | null;
  escalation_level?: 'normal' | 'medium' | 'high' | string | null;
  is_overdue?: boolean;
  is_stale?: boolean;
  next_run_at?: string | null;
  backoff_until?: string | null;
  action_count?: number;
  created_at?: string | null;
  job_id?: string | null;
  inbox_item_id?: string | null;
  portfolio_id?: string | null;
  portfolio_title?: string | null;
  portfolio_opportunity_id?: string | null;
  portfolio_opportunity_key?: string | null;
  domain_research_profile_id?: string | null;
  domain_research_profile_title?: string | null;
  profile_opportunity_id?: string | null;
  profile_opportunity_key?: string | null;
  domain?: string | null;
  objective?: string | null;
  track_type?: string | null;
  source_scope?: string | null;
  repo_source_ids?: string[] | null;
  benchmark_queries?: string[] | null;
  sandbox_profile_id?: string | null;
  automation_profile?: string | null;
  effective_policy?: Record<string, any> | null;
  confidence?: number | null;
  readiness?: number | null;
  linked_note_ids?: string[] | null;
  linked_experiment_plan_ids?: string[] | null;
  linked_validation_run_ids?: string[] | null;
  child_job_ids?: string[] | null;
  job?: AgentJob | null;
  checkpoint?: Record<string, any> | null;
  scheduler_state?: Record<string, any> | null;
  inbox_item?: ResearchInboxItem | Record<string, any> | null;
  follow_up_decision?: string | null;
  follow_up_policy_mode?: string | null;
  follow_up_launch_status?: string | null;
  follow_up_block_reason?: string | null;
  follow_up_budget_decision?: string | null;
  follow_up_budget_reason?: string | null;
  follow_up_budget_throttle_state?: string | null;
  follow_up_customer_budget_decision?: string | null;
  follow_up_customer_budget_reason?: string | null;
  follow_up_customer_budget_throttle_state?: string | null;
  follow_up_recommendation_key?: string | null;
  follow_up_job_id?: string | null;
  follow_up_chain_definition_id?: string | null;
  follow_up_operator_decision?: string | null;
  follow_up_operator_note?: string | null;
  follow_up_operator_acted_at?: string | null;
  follow_up_operator_user_id?: string | null;
  policy_guardrail_status?: string | null;
  policy_guardrail_action?: string | null;
  policy_guardrail_target_history_entry_id?: string | null;
  policy_guardrail_reasons?: string[] | null;
  policy_guardrail_follow_up_autonomy?: Record<string, any> | null;
  policy_guardrail_target_policy?: Record<string, any> | null;
  budget_throttle_state?: string | null;
  budget_reason?: string | null;
  customer_budget_throttle_state?: string | null;
  customer_budget_reason?: string | null;
  actions: AgentCheckpointQueueAction[];
}

export interface AgentCheckpointQueueFollowUpActionRequest {
  inbox_item_id?: string;
  domain_research_profile_id?: string;
  profile_opportunity_id?: string;
  portfolio_id?: string;
  portfolio_opportunity_id?: string;
  action: 'approve_launch' | 'reject_launch' | string;
  operator_note?: string;
}

export interface AgentCheckpointQueueFollowUpActionResponse {
  inbox_item_id?: string | null;
  domain_research_profile_id?: string | null;
  profile_opportunity_id?: string | null;
  portfolio_id?: string | null;
  portfolio_opportunity_id?: string | null;
  ok: boolean;
  follow_up_launch_status?: string | null;
  follow_up_operator_decision?: string | null;
  follow_up_job_id?: string | null;
  follow_up_chain_definition_id?: string | null;
  detail?: string | null;
}

export interface AgentCheckpointQueueBulkFollowUpActionRequest {
  domain_research_profile_id?: string;
  profile_opportunity_ids?: string[];
  portfolio_id?: string;
  portfolio_opportunity_ids?: string[];
  action: 'approve_launch' | 'reject_launch' | string;
  operator_note?: string;
}

export interface AgentCheckpointQueueBulkFollowUpActionResult {
  domain_research_profile_id?: string | null;
  profile_opportunity_id?: string | null;
  portfolio_id?: string | null;
  portfolio_opportunity_id?: string | null;
  ok: boolean;
  follow_up_launch_status?: string | null;
  follow_up_operator_decision?: string | null;
  follow_up_job_id?: string | null;
  detail?: string | null;
  error?: string | null;
}

export interface AgentCheckpointQueueBulkFollowUpActionResponse {
  requested_count: number;
  applied: number;
  failed: number;
  results: AgentCheckpointQueueBulkFollowUpActionResult[];
}

export interface AgentCheckpointQueueResponse {
  items: AgentCheckpointQueueItem[];
  total: number;
  limit: number;
  offset: number;
  approvals: number;
  recoveries: number;
  follow_ups: number;
  policy_reviews: number;
  budget_reviews: number;
  by_type: Record<string, number>;
  by_status: Record<string, number>;
  by_customer: Record<string, number>;
  by_sla_bucket: Record<string, number>;
  by_escalation_level: Record<string, number>;
}

export interface AgentDecisionTraceDeepLink {
  target_tab: string;
  job_id?: string | null;
  params?: Record<string, string> | null;
  label?: string | null;
}

export interface AgentDecisionTraceEvent {
  event_id: string;
  event_type: string;
  event_time: string;
  source_kind: string;
  source_id?: string | null;
  source_label?: string | null;
  customer?: string | null;
  decision_type: string;
  reason_code?: string | null;
  reason_label?: string | null;
  scheduler_state?: Record<string, any> | null;
  status?: string | null;
  severity?: string | null;
  actor_mode?: string | null;
  summary: string;
  operator_note?: string | null;
  before_state?: Record<string, any> | null;
  after_state?: Record<string, any> | null;
  deep_link?: AgentDecisionTraceDeepLink | null;
  metadata?: Record<string, any> | null;
  is_derived?: boolean;
  record_origin?: string | null;
  triage_status?: string | null;
  acknowledged_at?: string | null;
  acknowledged_by_user_id?: string | null;
  resolved_at?: string | null;
  resolved_by_user_id?: string | null;
  resolution_note?: string | null;
  pinned?: boolean;
  last_viewed_at?: string | null;
  owner_user_id?: string | null;
  owner_label?: string | null;
  assigned_to_user_id?: string | null;
  assigned_at?: string | null;
  assigned_by_user_id?: string | null;
  assignee_label?: string | null;
  is_owned_by_current_user?: boolean;
  is_assigned_to_current_user?: boolean;
  team_bucket?: string | null;
  due_at?: string | null;
  escalation_state?: string | null;
  escalation_reason?: string | null;
  escalated_at?: string | null;
  domain?: string | null;
  objective?: string | null;
  track_type?: string | null;
  source_scope?: string | null;
  repo_source_ids?: string[] | null;
  benchmark_queries?: string[] | null;
  sandbox_profile_id?: string | null;
  automation_profile?: string | null;
  effective_policy?: Record<string, any> | null;
  confidence?: number | null;
  readiness?: number | null;
  linked_note_ids?: string[] | null;
  linked_experiment_plan_ids?: string[] | null;
  linked_validation_run_ids?: string[] | null;
  child_job_ids?: string[] | null;
}

export interface AgentDecisionTraceResponse {
  items: AgentDecisionTraceEvent[];
  total: number;
  limit: number;
  offset: number;
  by_source_kind: Record<string, number>;
  by_decision_type: Record<string, number>;
  by_status: Record<string, number>;
  by_customer: Record<string, number>;
  by_severity: Record<string, number>;
  by_actor_mode: Record<string, number>;
  by_triage_status: Record<string, number>;
  by_assignee: Record<string, number>;
  by_escalation_state: Record<string, number>;
  overdue_count: number;
  has_more: boolean;
}

export interface AgentDecisionTraceAnalyticsBucket {
  value: string;
  count: number;
}

export interface AgentDecisionTraceAnalyticsTrendPoint {
  day: string;
  count: number;
}

export interface AgentDecisionTraceAnalyticsResponse {
  window_days: number;
  total: number;
  by_source_kind: Record<string, number>;
  by_triage_status: Record<string, number>;
  top_decision_types: AgentDecisionTraceAnalyticsBucket[];
  top_reason_labels: AgentDecisionTraceAnalyticsBucket[];
  top_queue_reasons: AgentDecisionTraceAnalyticsBucket[];
  daily_trend: AgentDecisionTraceAnalyticsTrendPoint[];
}

export interface AgentDecisionTraceActionRequest {
  action: 'acknowledge' | 'start_investigation' | 'resolve' | 'reopen' | 'toggle_pin' | 'assign' | 'unassign' | 'set_due_at' | 'clear_due_at' | 'approve_launch' | 'reject_launch' | 'relaunch_follow_up';
  note?: string;
  assigned_to_user_id?: string;
  due_at?: string;
}

export interface AgentDecisionTraceActionResponse {
  event: AgentDecisionTraceEvent;
}

export interface AgentDecisionTraceView {
  id: string;
  user_id: string;
  name: string;
  filters: Record<string, any>;
  is_default: boolean;
  created_at: string;
  updated_at: string;
}

export interface AgentDecisionTraceViewListResponse {
  items: AgentDecisionTraceView[];
  total: number;
}

export interface AgentDecisionTraceViewCreateRequest {
  name: string;
  filters: Record<string, any>;
  is_default?: boolean;
}

export interface AgentDecisionTraceViewUpdateRequest {
  name?: string;
  filters?: Record<string, any>;
  is_default?: boolean;
}

export interface AgentCheckpointQueueBulkActionRequest {
  item_type: 'approval_checkpoint' | 'job_recovery' | string;
  action: 'approve' | 'reject' | 'skip' | 'restart' | 'resume' | 'cancel' | string;
  job_ids: string[];
  checkpoint_note?: string;
}

export interface AgentCheckpointQueueBulkActionResult {
  job_id: string;
  ok: boolean;
  status?: string | null;
  error?: string | null;
  queue_key?: string | null;
}

export interface AgentCheckpointQueueBulkActionResponse {
  requested_count: number;
  applied: number;
  failed: number;
  results: AgentCheckpointQueueBulkActionResult[];
}

export interface AgentJobTemplate {
  id: string;
  name: string;
  display_name: string;
  description?: string;
  category?: string;
  job_type: AgentJobType;
  default_goal?: string;
  default_config?: Record<string, any>;
  default_chain_config?: ChainConfig;
  agent_definition_id?: string;
  default_max_iterations: number;
  default_max_tool_calls: number;
  default_max_llm_calls: number;
  default_max_runtime_minutes: number;
  is_system: boolean;
  is_active: boolean;
  owner_user_id?: string;
  recommended?: boolean;
  recommendation_score?: number;
  recommendation_reasons?: string[];
  created_at: string;
  updated_at: string;
}

export interface AgentJobTemplateListResponse {
  templates: AgentJobTemplate[];
  total: number;
}

export interface AgentJobQuickStartClaudeBackendRequest {
  name?: string;
  goal: string;
  source_id: string;
  search_query?: string;
  file_paths?: string[];
  commands?: string[];
  start_immediately?: boolean;
  config_overrides?: Record<string, any>;
}

export interface AgentJobQuickStartDomainResearchRequest {
  name?: string;
  domain: string;
  objective: string;
  customer_context?: string;
  source_scope?: 'kb_only' | 'arxiv_only' | 'kb_plus_arxiv' | 'kb_plus_arxiv_plus_repo' | string;
  track_type?: 'compiler' | 'microarchitecture' | 'generic' | string;
  monitor_queries?: string[];
  repo_source_ids?: string[];
  benchmark_queries?: string[];
  sandbox_profile_id?: string;
  report_format?: 'brief_only' | 'report_only' | 'brief_and_report' | string;
  scoring_policy?: Record<string, any>;
  selection_policy?: Record<string, any>;
  automation_profile?: 'balanced' | 'max_autonomy' | string;
  automation_policy?: Record<string, any>;
  validation_policy?: Record<string, any>;
  persist_artifacts?: boolean;
  auto_launch_follow_up?: boolean;
  auto_create_experiment_plans?: boolean;
  max_documents?: number;
  max_papers?: number;
  profile_id?: string;
  confidence_threshold?: number;
  start_immediately?: boolean;
  config_overrides?: Record<string, any>;
}

export interface ScientificSandboxProfile {
  id: string;
  name: string;
  description?: string | null;
  track_type: string;
  backend: string;
  docker_image?: string | null;
  timeout_seconds: number;
  resource_caps: Record<string, any>;
  allowed_benchmark_families: string[];
  allowed_perf_collectors: string[];
  required_capabilities: string[];
  toolchains: string[];
  budget_limit_default: number;
  enabled: boolean;
  system_managed: boolean;
  is_default: boolean;
  created_by_user_id?: string | null;
  created_at?: string | null;
  updated_at?: string | null;
}

export interface ScientificValidationRunSummary {
  id: string;
  agent_job_id?: string | null;
  name: string;
  status: string;
  progress: number;
  validation_kind?: string | null;
  sandbox_profile_id?: string | null;
  sandbox_profile_name?: string | null;
  recipe_family?: string | null;
  recipe_id?: string | null;
  benchmark_family?: string | null;
  benchmark_suite_id?: string | null;
  benchmark_case_ids?: string[];
  blocked_reason_code?: string | null;
  hypothesis_id?: string | null;
  track_type?: string | null;
  domain_research_profile_id?: string | null;
  research_portfolio_id?: string | null;
  parent_run_id?: string | null;
  latest_child_run_id?: string | null;
  retry_count?: number;
  latest_operator_action?: string | null;
  latest_operator_outcome_status?: string | null;
  compiler_artifact_summary?: {
    source_run_ids?: string[];
    primary_run_id?: string | null;
    comparison_run_id?: string | null;
    explanation_note_id?: string | null;
    explanation_synthesis_job_id?: string | null;
    explanation_synthesis_status?: string | null;
    proposal_note_id?: string | null;
    proposal_synthesis_job_id?: string | null;
    proposal_synthesis_status?: string | null;
    patch_draft_note_id?: string | null;
    patch_draft_synthesis_job_id?: string | null;
    patch_draft_synthesis_status?: string | null;
    source_explanation_note_id?: string | null;
    source_proposal_note_id?: string | null;
    source_id?: string | null;
    source_name?: string | null;
    available_actions?: string[];
  } | null;
  created_at: string;
  started_at?: string | null;
  completed_at?: string | null;
}

export interface ScientificSandboxProfileListResponse {
  items: ScientificSandboxProfile[];
  total: number;
}

export interface ScientificSandboxProfileCreate {
  id: string;
  name: string;
  description?: string;
  track_type?: string;
  backend?: string;
  docker_image?: string;
  timeout_seconds?: number;
  resource_caps?: Record<string, any>;
  allowed_benchmark_families?: string[];
  allowed_perf_collectors?: string[];
  required_capabilities?: string[];
  toolchains?: string[];
  budget_limit_default?: number;
  enabled?: boolean;
  is_default?: boolean;
}

export interface ScientificSandboxProfileUpdate {
  name?: string;
  description?: string;
  track_type?: string;
  backend?: string;
  docker_image?: string;
  timeout_seconds?: number;
  resource_caps?: Record<string, any>;
  allowed_benchmark_families?: string[];
  allowed_perf_collectors?: string[];
  required_capabilities?: string[];
  toolchains?: string[];
  budget_limit_default?: number;
  enabled?: boolean;
  is_default?: boolean;
}

export interface AgentJobQuickStartRepoBugTriageRequest {
  name?: string;
  goal?: string;
  failure_symptom?: string;
  source_id: string;
  scope?: 'auto' | 'backend' | 'frontend' | 'worker' | string;
  search_query?: string;
  file_paths?: string[];
  commands?: string[];
  error_output?: string;
  start_immediately?: boolean;
  config_overrides?: Record<string, any>;
}

export interface AgentJobQuickStartBugTriageSwarmRequest {
  name?: string;
  goal?: string;
  failure_symptom?: string;
  source_id: string;
  scope?: 'auto' | 'backend' | 'frontend' | 'worker' | string;
  search_query?: string;
  file_paths?: string[];
  commands?: string[];
  error_output?: string;
  max_agents?: number;
  profile_id?: string;
  start_immediately?: boolean;
  config_overrides?: Record<string, any>;
}

export interface AgentJobQuickStartBuildBreakSwarmRequest {
  name?: string;
  goal?: string;
  failure_symptom?: string;
  source_id: string;
  scope?: 'auto' | 'backend' | 'frontend' | 'worker' | string;
  search_query?: string;
  file_paths?: string[];
  commands?: string[];
  error_output?: string;
  max_agents?: number;
  profile_id?: string;
  start_immediately?: boolean;
  config_overrides?: Record<string, any>;
}

export interface AgentJobQuickStartFrontendRegressionSwarmRequest {
  name?: string;
  goal?: string;
  failure_symptom?: string;
  source_id: string;
  scope?: 'auto' | 'backend' | 'frontend' | 'worker' | string;
  search_query?: string;
  file_paths?: string[];
  commands?: string[];
  error_output?: string;
  max_agents?: number;
  profile_id?: string;
  start_immediately?: boolean;
  config_overrides?: Record<string, any>;
}

export interface AgentJobQuickStartRoleWorkflowRequest {
  name?: string;
  goal: string;
  roles?: string[];
  max_agents?: number;
  memory_profile?: 'off' | 'minimal' | 'balanced' | 'evidence' | 'synthesis' | string;
  approval_mode?: 'high_impact' | 'none' | string;
  execution_mode?: 'plan_and_execute' | 'adaptive' | string;
  extract_memory_on_failure?: boolean;
  memory_failed_types?: string[];
  memory_completed_types?: string[];
  start_immediately?: boolean;
  config_overrides?: Record<string, any>;
}

export interface AgentJobRelaunchLineageNode {
  id: string;
  name: string;
  status: AgentJobStatus;
  created_at: string;
  launch_mode?: string;
}

export interface AgentJobRelaunchLineage {
  job_id: string;
  root_job_id: string;
  parent_job_id?: string;
  latest_child_job_id?: string;
  ancestors_truncated?: boolean;
  descendants_truncated?: boolean;
  ancestors: AgentJobRelaunchLineageNode[];
  descendants: AgentJobRelaunchLineageNode[];
}

export interface AgentJobStats {
  total_jobs: number;
  running_jobs: number;
  pending_jobs: number;
  completed_jobs: number;
  failed_jobs: number;
  total_iterations: number;
  total_tool_calls: number;
  total_llm_calls: number;
  avg_completion_time_minutes?: number;
  success_rate?: number;
  launch_mode_counts?: Record<string, number>;
  launch_mode_none_count?: number;
}

export interface AgentJobSwarmAnalyticsPresetRow {
  preset_key: string;
  launch_mode: string;
  label: string;
  total_runs: number;
  avg_confidence?: number | null;
  high_confidence_runs: number;
  medium_confidence_runs: number;
  low_confidence_runs: number;
  auto_promoted_runs: number;
  review_needed_runs: number;
  tie_breaker_runs: number;
  manual_promotion_runs: number;
  repair_handoff_runs: number;
  backlog_handoff_runs: number;
  auto_backlog_handoff_runs: number;
  manual_backlog_handoff_runs: number;
  backlog_auto_suppressed_runs: number;
  promotion_rate?: number | null;
  review_rate?: number | null;
  tie_breaker_rate?: number | null;
}

export interface AgentJobSwarmAnalytics {
  preset_rows: AgentJobSwarmAnalyticsPresetRow[];
  totals: Record<string, any>;
  filters: Record<string, any>;
}

export interface AgentJobSwarmOutcomeCase {
  swarm_job_id: string;
  swarm_job_name?: string | null;
  preset_key: string;
  launch_mode: string;
  source_id?: string | null;
  source_label?: string | null;
  swarm_status?: string | null;
  swarm_completed_at?: string | null;
  review_state?: string | null;
  review_reason?: string | null;
  owner_user_id?: string | null;
  assigned_user_id?: string | null;
  assigned_at?: string | null;
  assigned_by_user_id?: string | null;
  review_note?: string | null;
  collaboration_summary?: CollaborationSummary | null;
  promotion_mode: 'auto' | 'manual' | 'none' | string;
  confidence_overall?: number | null;
  tie_breaker_attempted?: boolean;
  repair_job_id?: string | null;
  repair_job_name?: string | null;
  repair_status?: string | null;
  repair_handoff_at?: string | null;
  verification_status?: string | null;
  verification_reason?: string | null;
  backlog_item_id?: string | null;
  backlog_title?: string | null;
  backlog_status?: string | null;
  backlog_route_mode?: string | null;
  backlog_routed_at?: string | null;
  latest_downstream_at?: string | null;
  handoff_latency_minutes?: number | null;
  terminal_outcome: 'verified_fix' | 'repair_failed' | 'backlog_routed' | 'needs_review' | 'stalled_after_handoff' | string;
  terminal_reason?: string | null;
}

export interface AgentJobSwarmOutcomePresetRow {
  preset_key: string;
  launch_mode: string;
  label: string;
  total_swarm_roots: number;
  auto_promoted_runs: number;
  manual_promoted_runs: number;
  tie_breaker_runs: number;
  repair_handoff_runs: number;
  verified_fix_runs: number;
  repair_failed_runs: number;
  backlog_routed_runs: number;
  auto_backlog_routed_runs: number;
  manual_backlog_routed_runs: number;
  backlog_auto_suppressed_runs: number;
  needs_review_runs: number;
  stalled_after_handoff_runs: number;
  avg_confidence?: number | null;
  avg_handoff_minutes?: number | null;
}

export interface AgentJobSwarmOutcomeAnalytics {
  preset_rows: AgentJobSwarmOutcomePresetRow[];
  cases: AgentJobSwarmOutcomeCase[];
  totals: Record<string, any>;
  filters: Record<string, any>;
}

export interface AgentJobProgressUpdate {
  type: 'progress';
  job_id: string;
  progress: number;
  phase: string;
  status: AgentJobStatus;
  iteration: number;
  phase_details?: string;
  execution_graph_runtime?: Record<string, any>;
  scope_observability_runtime?: Record<string, any>;
  error?: string;
  timestamp: string;
}

export interface AgentJobCheckpoint {
  id: string;
  job_id: string;
  iteration: number;
  phase?: string;
  created_at: string;
}

// Research Inbox Types
export type ResearchInboxItemStatus = 'new' | 'accepted' | 'rejected';

export interface ResearchInboxItem {
  id: string;
  user_id: string;
  job_id?: string;
  customer?: string;
  item_type: string;
  item_key: string;
  title: string;
  summary?: string;
  url?: string;
  published_at?: string;
  discovered_at: string;
  status: ResearchInboxItemStatus;
  feedback?: string;
  metadata?: Record<string, any>;
  follow_up_decision?: string;
  follow_up_policy_mode?: string;
  follow_up_launch_status?: string;
  follow_up_block_reason?: string;
  follow_up_budget_decision?: string;
  follow_up_budget_reason?: string;
  follow_up_budget_throttle_state?: string;
  follow_up_customer_budget_decision?: string;
  follow_up_customer_budget_reason?: string;
  follow_up_customer_budget_throttle_state?: string;
  follow_up_recommendation_key?: string;
  follow_up_operator_decision?: string;
  follow_up_operator_note?: string;
  follow_up_operator_acted_at?: string;
  follow_up_operator_user_id?: string;
  follow_up_job_id?: string;
  follow_up_last_job_id?: string;
  follow_up_chain_definition_id?: string;
  follow_up_launched_at?: string;
  follow_up_outcome_status?: string;
  follow_up_outcome_recorded_at?: string;
  follow_up_outcome_summary?: string;
  origin_source_kind?: string;
  origin_source_id?: string;
  origin_opportunity_id?: string;
  created_at: string;
  updated_at: string;
}

export interface ResearchInboxListResponse {
  items: ResearchInboxItem[];
  total: number;
  limit: number;
  offset: number;
}

export interface ResearchInboxItemUpdateRequest {
  status?: ResearchInboxItemStatus;
  feedback?: string;
  metadata_patch?: Record<string, any>;
}

export interface ResearchInboxBulkFollowUpRelaunchRequest {
  item_ids: string[];
  operator_note?: string;
}

export interface ResearchInboxBulkFollowUpRelaunchResult {
  item_id: string;
  ok: boolean;
  follow_up_job_id?: string | null;
  error?: string | null;
}

export interface ResearchInboxBulkFollowUpRelaunchResponse {
  requested_count: number;
  applied: number;
  failed: number;
  results: ResearchInboxBulkFollowUpRelaunchResult[];
}

export interface ResearchInboxStats {
  total: number;
  new: number;
  accepted: number;
  rejected: number;
}

export interface ResearchMonitorProfile {
  id: string;
  user_id: string;
  customer?: string;
  token_scores?: Record<string, number>;
  phrase_scores?: Record<string, number>;
  recommendation_scores?: Record<string, number>;
  source_type_scores?: Record<string, number>;
  outcome_counters?: Record<string, number>;
  customer_budget_config?: ResearchMonitorBudgetConfig;
  muted_tokens?: string[];
  muted_patterns?: string[];
  notes?: string;
  created_at: string;
  updated_at: string;
}

export interface ResearchMonitorRecommendationAnalytics {
  recommendation_key: string;
  launch_count: number;
  auto_launch_count: number;
  approval_launch_count: number;
  blocked_count: number;
  completed_count: number;
  failed_count: number;
  cancelled_count: number;
  success_rate: number;
  score_trend: string;
  monitor_count: number;
}

export interface ResearchMonitorBudgetConfig {
  auto_launch_limit_24h: number;
  approval_queue_limit_24h: number;
  alert_limit_24h: number;
  queue_backlog_cap: number;
}

export interface ResearchMonitorBudgetUsage {
  auto_launch_count_24h: number;
  approval_queue_count_24h: number;
  alert_count_24h: number;
  queue_backlog_count: number;
}

export interface ResearchMonitorCustomerTopContributor {
  monitor_job_id?: string;
  monitor_name: string;
  customer?: string;
  value: number;
  throttle_state?: string | null;
}

export interface ResearchMonitorBudgetHistoryEntry {
  id: string;
  at: string;
  actor_user_id?: string;
  change_source: string;
  change_reason?: string;
  previous_autonomy_budget: ResearchMonitorBudgetConfig;
  next_autonomy_budget: ResearchMonitorBudgetConfig;
  guidance_context: Record<string, any>;
}

export interface ResearchMonitorCustomerRebalanceChange {
  monitor_job_id: string;
  monitor_name: string;
  customer?: string;
  current_budget: ResearchMonitorBudgetConfig;
  proposed_budget: ResearchMonitorBudgetConfig;
  delta_budget: ResearchMonitorBudgetConfig;
  reasons: string[];
}

export interface ResearchMonitorCustomerRebalancePreview {
  customer: string;
  guidance_status: string;
  guidance_summary?: string;
  guidance_reasons: string[];
  before_capacity: ResearchMonitorBudgetConfig;
  after_capacity: ResearchMonitorBudgetConfig;
  changes: ResearchMonitorCustomerRebalanceChange[];
}

export interface ResearchMonitorCustomerRebalanceEvaluationCounts {
  accepted_count: number;
  blocked_count: number;
  follow_up_completed_count: number;
  follow_up_failed_count: number;
  follow_up_cancelled_count: number;
  auto_launch_used_24h: number;
  approval_queue_used_24h: number;
  alert_used_24h: number;
  backlog_used: number;
  throttled_monitor_count: number;
}

export interface ResearchMonitorCustomerRebalanceHistoryEntry {
  id: string;
  at: string;
  actor_user_id?: string;
  change_source: string;
  change_reason?: string;
  changes: ResearchMonitorCustomerRebalanceChange[];
  before_capacity: ResearchMonitorBudgetConfig;
  after_capacity: ResearchMonitorBudgetConfig;
  evaluation_target_count: number;
  evaluation_state: string;
  evaluation_status?: string;
  evaluation_sample_count: number;
  evaluation_reasons: string[];
  before_counts: ResearchMonitorCustomerRebalanceEvaluationCounts;
  after_counts: ResearchMonitorCustomerRebalanceEvaluationCounts;
  delta_counts: ResearchMonitorCustomerRebalanceEvaluationCounts;
}

export interface ResearchMonitorCustomerRebalanceEvaluationSample {
  item_id: string;
  title: string;
  period: string;
  launch_status?: string;
  outcome_status?: string;
  recommendation_key?: string;
  summary?: string;
  monitor_job_id?: string;
  monitor_name?: string;
}

export interface ResearchMonitorCustomerRebalanceEvaluationDetail {
  customer: string;
  history_entry_id: string;
  evaluation_status: string;
  evaluation_sample_count: number;
  evaluation_target_count: number;
  evaluation_reasons: string[];
  before_counts: ResearchMonitorCustomerRebalanceEvaluationCounts;
  after_counts: ResearchMonitorCustomerRebalanceEvaluationCounts;
  delta_counts: ResearchMonitorCustomerRebalanceEvaluationCounts;
  sample_items: ResearchMonitorCustomerRebalanceEvaluationSample[];
}

export interface ResearchMonitorCustomerPortfolio {
  customer: string;
  monitor_count: number;
  strong_monitor_count: number;
  mixed_monitor_count: number;
  weak_monitor_count: number;
  auto_launch_used_24h: number;
  auto_launch_capacity_24h: number;
  approval_queue_used_24h: number;
  approval_queue_capacity_24h: number;
  alert_used_24h: number;
  alert_capacity_24h: number;
  backlog_used: number;
  backlog_capacity: number;
  throttled_monitor_count: number;
  customer_budget: ResearchMonitorBudgetConfig;
  customer_budget_usage: ResearchMonitorBudgetUsage;
  customer_budget_remaining: ResearchMonitorBudgetUsage;
  customer_budget_throttle_state: string;
  customer_budget_throttle_reasons: string[];
  accepted_count: number;
  blocked_count: number;
  follow_up_completed_count: number;
  follow_up_failed_count: number;
  follow_up_cancelled_count: number;
  portfolio_status: string;
  portfolio_reasons: string[];
  top_launch_monitors: ResearchMonitorCustomerTopContributor[];
  top_backlog_monitors: ResearchMonitorCustomerTopContributor[];
  top_alert_monitors: ResearchMonitorCustomerTopContributor[];
  throttled_monitors: ResearchMonitorCustomerTopContributor[];
  rebalance_guidance_status: string;
  rebalance_guidance_reasons: string[];
  rebalance_guidance_summary?: string;
  rebalance_guidance_changes: ResearchMonitorCustomerRebalanceChange[];
  latest_rebalance_evaluation_status?: string;
  latest_rebalance_evaluation_sample_count: number;
  latest_rebalance_evaluation_target_count: number;
  latest_rebalance_evaluation_reasons: string[];
  recent_rebalance_history: ResearchMonitorCustomerRebalanceHistoryEntry[];
}

export interface ResearchMonitorHealthSummary {
  monitor_job_id?: string;
  monitor_name: string;
  monitor_job_type?: string;
  customer?: string;
  discovered_count: number;
  accepted_count: number;
  rejected_count: number;
  acceptance_rate: number;
  auto_launched_count: number;
  approval_launched_count: number;
  queued_for_approval_count: number;
  manual_only_count: number;
  blocked_count: number;
  follow_up_completed_count: number;
  follow_up_failed_count: number;
  follow_up_cancelled_count: number;
  relaunch_count: number;
  health_score: number;
  health_bucket: string;
  health_reasons: string[];
  automation_profile?: string;
  automation_policy?: Record<string, any>;
  effective_policy?: Record<string, any>;
  autonomy_mode?: string;
  current_policy_mode?: string;
  current_allowed_recommendations?: string[];
  autonomy_budget: ResearchMonitorBudgetConfig;
  budget_usage: ResearchMonitorBudgetUsage;
  budget_remaining: ResearchMonitorBudgetUsage;
  budget_throttle_state: string;
  budget_throttle_reasons: string[];
  budget_clamp_state?: string | null;
  budget_clamp_reasons?: string[];
  budget_history_count: number;
  latest_budget_changed_at?: string;
  latest_budget_change_source?: string;
  latest_budget_actor_user_id?: string;
  latest_budget_change_reason?: string;
  recommended_policy_mode: string;
  recommended_allowed_recommendations: string[];
  policy_reasons: string[];
  policy_confidence: string;
  policy_history_count: number;
  latest_policy_changed_at?: string;
  latest_policy_change_source?: string;
  latest_policy_actor_user_id?: string;
  latest_policy_evaluation_status?: string;
  latest_policy_evaluation_sample_count: number;
  latest_policy_evaluation_target_count: number;
  latest_policy_evaluation_reasons: string[];
  policy_guardrail_status?: string | null;
  policy_guardrail_action?: string | null;
  policy_guardrail_reasons: string[];
  policy_guardrail_target_history_entry_id?: string | null;
  policy_guardrail_follow_up_autonomy?: ResearchMonitorPolicyConfig | null;
  policy_guardrail_state?: string | null;
  policy_guardrail_target_policy?: Record<string, any> | null;
  policy_mode_counts: Record<string, number>;
  follow_up_review_counts?: Record<string, number>;
  scheduler_summary?: Record<string, any>;
  suppressed_relaunches_count?: number;
  recent_policy_history: ResearchMonitorPolicyHistoryEntry[];
  top_recommendations: ResearchMonitorRecommendationAnalytics[];
}

export interface ResearchMonitorCustomerBudgetUpdateRequest {
  customer: string;
  auto_launch_limit_24h?: number;
  approval_queue_limit_24h?: number;
  alert_limit_24h?: number;
  queue_backlog_cap?: number;
  reset_to_default?: boolean;
}

export interface ResearchMonitorCustomerBudgetUpdateResponse {
  customer: string;
  customer_budget: ResearchMonitorBudgetConfig;
}

export interface ResearchMonitorAnalyticsTotals {
  total_monitors: number;
  discovered_count: number;
  accepted_count: number;
  rejected_count: number;
  auto_launched_count: number;
  approval_launched_count: number;
  blocked_count: number;
  follow_up_completed_count: number;
  follow_up_failed_count: number;
  follow_up_cancelled_count: number;
  strong_monitors: number;
  mixed_monitors: number;
  weak_monitors: number;
}

export interface ResearchMonitorAnalyticsResponse {
  generated_at: string;
  totals: ResearchMonitorAnalyticsTotals;
  customers: ResearchMonitorCustomerPortfolio[];
  monitors: ResearchMonitorHealthSummary[];
  recommendations: ResearchMonitorRecommendationAnalytics[];
}

export interface ResearchMonitorPolicyConfig {
  mode: string;
  allowed_recommendations: string[];
  automation_profile?: string;
  automation_policy?: Record<string, any>;
  effective_policy?: Record<string, any>;
}

export interface ResearchMonitorPolicyEvaluationCounts {
  accepted_count: number;
  auto_launched_count: number;
  approval_launched_count: number;
  queued_for_approval_count: number;
  manual_only_count: number;
  blocked_count: number;
  follow_up_completed_count: number;
  follow_up_failed_count: number;
  follow_up_cancelled_count: number;
}

export interface ResearchMonitorPolicyEvaluationSample {
  item_id: string;
  title: string;
  period: string;
  launch_status?: string;
  outcome_status?: string;
  recommendation_key?: string;
  summary?: string;
}

export interface ResearchMonitorPolicyHistoryEntry {
  id: string;
  at: string;
  actor_user_id?: string;
  change_source: string;
  change_reason?: string;
  previous_follow_up_autonomy?: ResearchMonitorPolicyConfig;
  next_follow_up_autonomy?: ResearchMonitorPolicyConfig;
  previous_automation_profile?: string;
  next_automation_profile?: string;
  previous_automation_policy?: Record<string, any>;
  next_automation_policy?: Record<string, any>;
  previous_effective_policy?: Record<string, any>;
  next_effective_policy?: Record<string, any>;
  effective_clamp_state?: string | null;
  effective_clamp_reasons?: string[];
  analytics_context: Record<string, any>;
  evaluation_target_count: number;
  evaluation_state: string;
  evaluation_status?: string;
  evaluation_sample_count: number;
  evaluation_reasons: string[];
  before_counts: ResearchMonitorPolicyEvaluationCounts;
  after_counts: ResearchMonitorPolicyEvaluationCounts;
  delta_counts: ResearchMonitorPolicyEvaluationCounts;
}

export interface ResearchMonitorPolicyUpdateRequest {
  automation_profile?: string;
  automation_policy?: Record<string, any>;
  mode?: string;
  allowed_recommendations?: string[];
  reset_to_default?: boolean;
  change_source?: string;
  change_reason?: string;
  analytics_context?: Record<string, any>;
}

export interface ResearchMonitorPolicyRollbackRequest {
  history_entry_id: string;
  change_reason?: string;
}

export interface ResearchMonitorPolicySimulationRequest {
  automation_profile?: string;
  automation_policy?: Record<string, any>;
  mode?: string;
  allowed_recommendations?: string[];
  history_limit?: number;
}

export interface ResearchMonitorPolicyUpdateResponse {
  monitor_job_id: string;
  follow_up_autonomy?: ResearchMonitorPolicyConfig;
  automation_profile?: string;
  automation_policy?: Record<string, any>;
  effective_policy?: Record<string, any>;
  latest_history_entry?: ResearchMonitorPolicyHistoryEntry;
  policy_history_count: number;
}

export interface ResearchMonitorBudgetUpdateRequest {
  auto_launch_limit_24h?: number;
  approval_queue_limit_24h?: number;
  alert_limit_24h?: number;
  queue_backlog_cap?: number;
  reset_to_default?: boolean;
}

export interface ResearchMonitorBudgetUpdateResponse {
  monitor_job_id: string;
  autonomy_budget: ResearchMonitorBudgetConfig;
  latest_history_entry?: ResearchMonitorBudgetHistoryEntry;
}

export interface ResearchMonitorCustomerRebalanceApplyMonitorRequest {
  monitor_job_id: string;
  auto_launch_limit_24h: number;
  approval_queue_limit_24h: number;
  alert_limit_24h: number;
  queue_backlog_cap: number;
}

export interface ResearchMonitorCustomerRebalancePreviewRequest {
  customer: string;
  monitor_budget_updates?: ResearchMonitorCustomerRebalanceApplyMonitorRequest[];
}

export interface ResearchMonitorCustomerRebalanceApplyRequest {
  customer: string;
  monitor_budget_updates: ResearchMonitorCustomerRebalanceApplyMonitorRequest[];
  change_reason?: string;
}

export interface ResearchMonitorCustomerRebalanceApplyResponse {
  customer: string;
  updated_monitor_ids: string[];
  guidance_status: string;
  guidance_summary?: string;
  latest_history_entries: ResearchMonitorBudgetHistoryEntry[];
}

export interface ResearchMonitorPolicySimulationCounts {
  auto_launch_safe_count: number;
  queue_for_approval_count: number;
  manual_only_count: number;
  blocked_count: number;
  insufficient_context_count: number;
}

export interface ResearchMonitorPolicySimulationRecommendationDelta {
  recommendation_key: string;
  baseline_count: number;
  simulated_count: number;
  delta_count: number;
}

export interface ResearchMonitorPolicySimulationSample {
  item_id: string;
  title: string;
  recommendation_key?: string;
  current_outcome: string;
  simulated_outcome: string;
  reason: string;
}

export interface ResearchMonitorPolicySimulationResponse {
  monitor_job_id: string;
  current_policy: ResearchMonitorPolicyConfig;
  proposed_policy: ResearchMonitorPolicyConfig;
  current_automation_profile?: string;
  current_automation_policy?: Record<string, any>;
  current_effective_policy?: Record<string, any>;
  proposed_automation_profile?: string;
  proposed_automation_policy?: Record<string, any>;
  proposed_effective_policy?: Record<string, any>;
  history_limit: number;
  baseline_counts: ResearchMonitorPolicySimulationCounts;
  simulated_counts: ResearchMonitorPolicySimulationCounts;
  delta_counts: ResearchMonitorPolicySimulationCounts;
  top_recommendation_deltas: ResearchMonitorPolicySimulationRecommendationDelta[];
  sample_items: ResearchMonitorPolicySimulationSample[];
  insufficient_context_count: number;
}

export interface ResearchMonitorPolicyEvaluationDetail {
  monitor_job_id: string;
  history_entry_id: string;
  evaluation_status: string;
  evaluation_sample_count: number;
  evaluation_target_count: number;
  evaluation_reasons: string[];
  before_counts: ResearchMonitorPolicyEvaluationCounts;
  after_counts: ResearchMonitorPolicyEvaluationCounts;
  delta_counts: ResearchMonitorPolicyEvaluationCounts;
  sample_items: ResearchMonitorPolicyEvaluationSample[];
}

export type CodePatchProposalStatus = 'proposed' | 'applied' | 'rejected';

export interface CodePatchProposal {
  id: string;
  user_id: string;
  job_id?: string;
  source_id?: string;
  title: string;
  summary?: string;
  diff_unified: string;
  metadata?: Record<string, any>;
  status: CodePatchProposalStatus;
  created_at: string;
  updated_at: string;
}


export type PatchPRStatus = 'draft' | 'open' | 'approved' | 'merged' | 'rejected';

export interface PatchPRListItem {
  id: string;
  source_id?: string;
  title: string;
  status: PatchPRStatus | string;
  selected_proposal_id?: string;
  created_at: string;
  updated_at: string;
}

export interface PatchPRListResponse {
  items: PatchPRListItem[];
  total: number;
  limit: number;
  offset: number;
}

export interface PatchPR {
  id: string;
  user_id: string;
  source_id?: string;
  title: string;
  description?: string;
  status: PatchPRStatus | string;
  selected_proposal_id?: string;
  proposal_ids: string[];
  checks?: Record<string, any>;
  approvals?: Array<Record<string, any>>;
  merged_at?: string;
  created_at: string;
  updated_at: string;
}

export interface PatchPRCreateRequest {
  title: string;
  description?: string;
  source_id?: string;
  initial_proposal_id?: string;
}

export interface PatchPRFromChainRequest {
  root_job_id: string;
  title?: string;
  description?: string;
  proposal_strategy?: 'best_passing' | 'latest';
  open_after_create?: boolean;
}

export interface PatchPRUpdateRequest {
  title?: string;
  description?: string;
  status?: PatchPRStatus | string;
  selected_proposal_id?: string;
}

export interface PatchPRApproveRequest {
  note?: string;
}

export interface PatchPRMergeRequest {
  dry_run?: boolean;
  require_approved?: boolean;
}

export interface PatchPRMergeResponse {
  pr_id: string;
  dry_run: boolean;
  ok: boolean;
  selected_proposal_id?: string;
  applied_files: any[];
  errors: any[];
}

// Artifact Drafts (generic review flow)
export type ArtifactDraftStatus = 'draft' | 'in_review' | 'approved' | 'published' | 'rejected';
export type ArtifactDraftType = 'presentation' | 'repo_report' | string;

export interface ArtifactDraftListItem {
  id: string;
  artifact_type: ArtifactDraftType;
  source_id?: string;
  title: string;
  status: ArtifactDraftStatus | string;
  created_at: string;
  updated_at: string;
  published_at?: string;
}

export interface ArtifactDraftListResponse {
  items: ArtifactDraftListItem[];
  total: number;
  limit: number;
  offset: number;
}

export interface ArtifactDraft {
  id: string;
  user_id: string;
  artifact_type: ArtifactDraftType;
  source_id?: string;
  title: string;
  description?: string;
  status: ArtifactDraftStatus | string;
  draft_payload: Record<string, any>;
  published_payload?: Record<string, any> | null;
  approvals?: Array<Record<string, any>>;
  created_at: string;
  updated_at: string;
  published_at?: string;
}

export interface RetrievalTrace {
  id: string;
  user_id?: string | null;
  session_id?: string | null;
  chat_message_id?: string | null;
  trace_type: string;
  query: string;
  processed_query?: string | null;
  provider?: string | null;
  settings_snapshot?: Record<string, any> | null;
  trace: Record<string, any>;
  created_at: string;
}

// Chain Definition Types
export interface ChainStepConfig {
  step_name: string;
  template_id?: string;
  job_type: AgentJobType;
  goal_template: string;
  config?: Record<string, any>;
  trigger_condition: ChainTriggerCondition;
  trigger_thresholds?: {
    progress_threshold?: number;
    findings_threshold?: number;
  };
}

export interface AgentJobChainDefinition {
  id: string;
  name: string;
  display_name: string;
  description?: string;
  chain_steps: ChainStepConfig[];
  default_settings?: Record<string, any>;
  owner_user_id?: string;
  is_system: boolean;
  is_active: boolean;
  created_at: string;
  updated_at: string;
}

export interface AgentJobChainDefinitionCreate {
  name: string;
  display_name: string;
  description?: string;
  chain_steps: ChainStepConfig[];
  default_settings?: Record<string, any>;
}

export interface AgentJobChainDefinitionUpdate {
  display_name?: string;
  description?: string;
  chain_steps?: ChainStepConfig[];
  default_settings?: Record<string, any>;
  is_active?: boolean;
}

export interface AgentJobChainDefinitionListResponse {
  chains: AgentJobChainDefinition[];
  total: number;
}

export interface AgentJobFromChainCreate {
  chain_definition_id: string;
  name_prefix: string;
  variables: Record<string, string>;
  config_overrides?: Record<string, any>;
  start_immediately?: boolean;
}

export interface AgentJobChainStatus {
  root_job_id: string;
  chain_definition_id?: string;
  total_steps: number;
  completed_steps: number;
  current_step: number;
  overall_progress: number;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'partially_completed';
  jobs: AgentJob[];
}

// ==================== Synthesis Types ====================

export type SynthesisJobType =
  | 'multi_doc_summary'
  | 'comparative_analysis'
  | 'theme_extraction'
  | 'knowledge_synthesis'
  | 'research_report'
  | 'executive_brief'
  | 'decision_memo'
  | 'gap_analysis_hypotheses'
  | 'hypothesis_reevaluation'
  | 'compiler_regression_explanation'
  | 'compiler_patch_proposal'
  | 'compiler_patch_draft';

export type SynthesisJobStatus =
  | 'pending'
  | 'analyzing'
  | 'synthesizing'
  | 'generating'
  | 'completed'
  | 'failed'
  | 'cancelled';

export interface SynthesisJob {
  id: string;
  user_id: string;
  job_type: SynthesisJobType;
  title: string;
  description?: string;
  document_ids: string[];
  paper_ids?: string[];
  /** Autonomous runs whose findings were used as source material. */
  agent_job_ids?: string[];
  research_note_id?: string;
  source_id?: string;
  search_query?: string;
  topic?: string;
  options?: Record<string, any>;
  output_format: 'markdown' | 'docx' | 'pdf' | 'pptx';
  output_style: string;
  status: SynthesisJobStatus;
  progress: number;
  current_stage?: string;
  result_content?: string;
  result_metadata?: {
    documents_analyzed?: number;
    word_count?: number;
    themes_found?: string[];
    key_findings?: string[];
    audience?: string;
    summary?: string;
    reprioritization_summary?: string;
    priority_deltas?: Array<Record<string, any>>;
    archived_hypothesis_ids?: string[];
    structured_hypotheses?: Array<Record<string, any>>;
    source_note_id?: string;
    regression_type?: string;
    metric_deltas?: Array<Record<string, any>>;
    artifact_deltas?: Array<Record<string, any>>;
    likely_causes?: Array<Record<string, any>>;
    supporting_signals?: string[];
    confounders?: string[];
    recommended_next_steps?: string[];
    source_run_ids?: string[];
    primary_run_id?: string;
    comparison_run_id?: string;
    primary_run_summary?: Record<string, any>;
    comparison_run_summary?: Record<string, any>;
    proposal_summary?: string;
    target_area?: string;
    candidate_change?: string;
    expected_effect?: string;
    mechanism?: string;
    supporting_evidence?: string[];
    validation_plan?: string[];
    risk_assessment?: string[];
    rollback_or_guardrail?: string;
    source_explanation_note_id?: string;
    draft_summary?: string;
    source_proposal_note_id?: string;
    source_name?: string;
    target_files?: string[];
    target_symbols?: string[];
    change_plan?: string[];
    proposed_code_regions?: Array<Record<string, any>>;
    validation_commands?: string[];
    benchmark_validation_scope?: string[];
    risk_checks?: string[];
    rollback_steps?: string[];
  };
  artifacts?: Array<{
    type: string;
    format?: string;
    code?: string;
    title?: string;
  }>;
  file_path?: string;
  file_size?: number;
  error?: string;
  review_outcome_status?: 'applied_to_source_note' | 'saved_as_new_note' | 'dismissed' | string | null;
  review_recorded_at?: string | null;
  review_note?: string | null;
  review_target_note_id?: string | null;
  can_apply?: boolean;
  can_dismiss?: boolean;
  created_at?: string;
  started_at?: string;
  completed_at?: string;
}

// ==================== Research Notes ====================

export interface ResearchHypothesis {
  id: string;
  rank: number;
  title: string;
  claim: string;
  rationale: string;
  supporting_evidence?: string[];
  supporting_sources?: Array<Record<string, any>>;
  counterarguments?: string[];
  novelty_score: number;
  evidence_score: number;
  testability_score: number;
  overall_score: number;
  recommended_next_step: string;
  autonomous_origin?: {
    source_kind?: 'profile' | 'portfolio' | string;
    source_id?: string | null;
    opportunity_id?: string | null;
    evidence_revision_at_launch?: string | null;
  } | null;
  experiment_evidence?: ResearchHypothesisExperimentEvidence[];
}

export interface ResearchHypothesisExperimentEvidence {
  run_id: string;
  experiment_plan_id?: string;
  plan_scope?: string | null;
  status?: string | null;
  summary?: string | null;
  appended_at?: string | null;
  selected_hypothesis_ids?: string[];
  supporting_sources?: Array<Record<string, any>>;
  source_paper_ids?: string[];
  source_document_ids?: string[];
  verification_commands?: string[];
  failed_commands?: string[];
  result_highlights?: string[];
  measurement_summary?: Record<string, any> | null;
  compiler_artifacts?: Record<string, any> | null;
  perf_counters?: Record<string, any> | null;
  artifact_diff_summary?: string | null;
  artifact_inventory?: string[];
  repeat_count?: number | null;
  autonomous_origin?: {
    source_kind?: 'profile' | 'portfolio' | string;
    source_id?: string | null;
    opportunity_id?: string | null;
    evidence_revision_at_launch?: string | null;
  } | null;
}

export interface ResearchNoteReevaluationHistoryEntry {
  job_id: string;
  saved_at?: string | null;
  note_title?: string | null;
  source_note_id?: string | null;
  target_note_id?: string | null;
  origin_source_kind?: 'profile' | 'portfolio' | string | null;
  origin_source_id?: string | null;
  origin_opportunity_id?: string | null;
  source_run_ids?: string[];
  reprioritization_summary?: string | null;
  priority_deltas?: Array<Record<string, any>>;
  archived_hypothesis_ids?: string[];
  outcome_status?: 'applied_to_source_note' | 'saved_as_new_note' | 'dismissed' | string | null;
  outcome_recorded_at?: string | null;
  outcome_note?: string | null;
}

export interface ResearchNote {
  id: string;
  user_id: string;
  title: string;
  content_markdown: string;
  tags?: string[];
  attribution?: Record<string, any> | null;
  structured_payload?: {
    research_mode?: string;
    summary?: string;
    hypotheses?: ResearchHypothesis[];
    source_paper_ids?: string[];
    source_document_ids?: string[];
    gaps?: string[];
    solution_sketches?: string[];
    scoring_policy?: Record<string, any>;
    selection_policy?: Record<string, any>;
    open_questions?: string[];
    ranked_opportunities?: string[];
    artifact_type?: string;
    reprioritization_summary?: string;
    priority_deltas?: Array<Record<string, any>>;
    archived_hypothesis_ids?: string[];
    previous_hypotheses?: ResearchHypothesis[];
    previous_summary?: string;
    previous_artifact_type?: string;
    reevaluation_history?: ResearchNoteReevaluationHistoryEntry[];
    last_appended_run_id?: string;
    last_appended_at?: string;
    pending_reevaluation_job_id?: string;
    pending_reevaluation_created_at?: string;
    pending_reevaluation_reason?: string;
    pending_reevaluation_source_run_ids?: string[];
    pending_reevaluation_status?: string;
    pending_reevaluation_completed_at?: string;
    pending_reevaluation_error?: string;
    regression_type?: string;
    source_run_ids?: string[];
    primary_run_id?: string;
    comparison_run_id?: string;
    metric_deltas?: Array<Record<string, any>>;
    artifact_deltas?: Array<Record<string, any>>;
    likely_causes?: Array<Record<string, any>>;
    supporting_signals?: string[];
    confounders?: string[];
    recommended_next_steps?: string[];
    benchmark_family?: string;
    benchmark_suite_id?: string;
    benchmark_case_ids?: string[];
    benchmark_baseline_id?: string;
    primary_run_summary?: Record<string, any> | null;
    comparison_run_summary?: Record<string, any> | null;
    proposal_summary?: string;
    target_area?: string;
    candidate_change?: string;
    expected_effect?: string;
    mechanism?: string;
    supporting_evidence?: string[];
    validation_plan?: string[];
    risk_assessment?: string[];
    rollback_or_guardrail?: string;
    source_explanation_note_id?: string;
    draft_summary?: string;
    source_proposal_note_id?: string;
    source_id?: string;
    source_name?: string;
    target_files?: string[];
    target_symbols?: string[];
    change_plan?: string[];
    proposed_code_regions?: Array<Record<string, any>>;
    validation_commands?: string[];
    benchmark_validation_scope?: string[];
    risk_checks?: string[];
    rollback_steps?: string[];
  } | null;
  source_synthesis_job_id?: string | null;
  source_document_ids?: string[] | null;
  created_at?: string | null;
  updated_at?: string | null;
}

export interface ResearchNoteListResponse {
  items: ResearchNote[];
  total: number;
  limit: number;
  offset: number;
}

// ==================== Experiments ====================

export interface ExperimentPlan {
  id: string;
  user_id: string;
  research_note_id: string;
  title: string;
  hypothesis_text?: string | null;
  plan: Record<string, any>;
  generator?: string | null;
  benchmark_family?: string | null;
  benchmark_suite_id?: string | null;
  benchmark_case_ids?: string[];
  benchmark_baseline_id?: string | null;
  generator_details?: {
    generated_at?: string;
    plan_mode?: 'aggregate_note' | 'single_hypothesis' | 'compiler_regression_followup';
    hypothesis_id?: string | null;
    selected_hypothesis_ids?: string[];
    source_paper_ids?: string[];
    source_document_ids?: string[];
    source_run_ids?: string[];
    primary_run_id?: string | null;
    comparison_run_id?: string | null;
    regression_type?: string | null;
    supporting_sources?: Array<Record<string, any>>;
    benchmark_family?: string | null;
    benchmark_suite_id?: string | null;
    benchmark_case_ids?: string[];
    benchmark_baseline_id?: string | null;
    benchmark_suite_name?: string | null;
    benchmark_case_names?: string[];
    benchmark_default_commands?: string[];
    reevaluation_mode?: boolean;
    reevaluation_source_job_id?: string | null;
    explanation_mode?: boolean;
    likely_causes?: Array<Record<string, any>>;
    recommended_next_steps?: string[];
    [key: string]: any;
  } | null;
  created_at: string;
  updated_at: string;
}

export interface ExperimentPlanListResponse {
  plans: ExperimentPlan[];
}

export interface ExperimentPlanGenerateRequest {
  note_id: string;
  max_note_chars?: number;
  prefer_section?: 'hypothesis' | 'full_note';
  plan_mode?: 'aggregate_note' | 'single_hypothesis' | 'compiler_regression_followup';
  hypothesis_id?: string;
  benchmark_suite_id?: string;
  benchmark_case_ids?: string[];
  include_ablations?: boolean;
  include_timeline?: boolean;
  include_risks?: boolean;
  include_repro_checklist?: boolean;
}

export interface ExperimentPlanUpdateRequest {
  title?: string;
  hypothesis_text?: string | null;
  plan?: Record<string, any>;
}

export interface ExperimentRun {
  id: string;
  user_id: string;
  experiment_plan_id: string;
  agent_job_id?: string | null;
  parent_run_id?: string | null;
  latest_child_run_id?: string | null;
  name: string;
  status:
    | 'pending'
    | 'planned'
    | 'queued'
    | 'provisioning'
    | 'running'
    | 'paused'
    | 'succeeded'
    | 'completed'
    | 'failed'
    | 'blocked'
    | 'cancelled';
  config?: Record<string, any> | null;
  results?: Record<string, any> | null;
  validation_kind?: string | null;
  sandbox_profile_id?: string | null;
  recipe_family?: string | null;
  recipe_id?: string | null;
  recipe_version?: number | null;
  domain_research_profile_id?: string | null;
  research_portfolio_id?: string | null;
  hypothesis_id?: string | null;
  originating_job_id?: string | null;
  blocked_reason_code?: string | null;
  capability_check?: Record<string, any> | null;
  profile_snapshot?: Record<string, any> | null;
  recipe_snapshot?: Record<string, any> | null;
  benchmark_family?: string | null;
  benchmark_suite_id?: string | null;
  benchmark_case_ids?: string[];
  benchmark_baseline_id?: string | null;
  measurement_summary?: Record<string, any> | null;
  compiler_artifacts?: Record<string, any> | null;
  perf_counters?: Record<string, any> | null;
  artifact_inventory?: string[];
  repeat_count?: number | null;
  experiment_run?: AgentJobExperimentRun | null;
  operator_interventions?: AgentJobOperatorIntervention[] | null;
  operator_actions?: ExperimentRunOperatorAction[] | null;
  summary?: string | null;
  progress: number;
  retry_count?: number;
  started_at?: string | null;
  completed_at?: string | null;
  created_at: string;
  updated_at: string;
}

export interface ExperimentRunListResponse {
  runs: ExperimentRun[];
}

export interface ExperimentRunPostRunActions {
  auto_append_to_note?: boolean;
  target_note_id?: string;
  append_status?: 'pending' | 'completed' | 'failed' | string;
  appended_at?: string | null;
  append_error?: string | null;
}

export interface ExperimentRunCreateRequest {
  name?: string;
  config?: Record<string, any> | null;
  summary?: string | null;
}

export interface BenchmarkCase {
  id: string;
  suite_id: string;
  name: string;
  description?: string | null;
  rank: number;
  source_ref?: string | null;
  benchmark_query?: string | null;
  compile_command_template?: string | null;
  run_command_template?: string | null;
  expected_artifacts: string[];
  metrics: Array<Record<string, any>>;
  observability: Record<string, any>;
  metadata: Record<string, any>;
}

export interface BenchmarkBaseline {
  id: string;
  suite_id: string;
  case_id?: string | null;
  name: string;
  description?: string | null;
  compiler_revision?: string | null;
  toolchain_id?: string | null;
  sandbox_profile_id?: string | null;
  measurements: Record<string, any>;
  environment_snapshot: Record<string, any>;
  enabled: boolean;
  system_managed: boolean;
}

export interface BenchmarkSuite {
  id: string;
  user_id?: string | null;
  name: string;
  description?: string | null;
  track_type: string;
  benchmark_family: string;
  suite_version: number;
  tags: string[];
  metadata: Record<string, any>;
  enabled: boolean;
  system_managed: boolean;
  cases: BenchmarkCase[];
  baselines: BenchmarkBaseline[];
}

export interface BenchmarkSuiteListResponse {
  items: BenchmarkSuite[];
  total: number;
}

export interface ExperimentRunUpdateRequest {
  name?: string;
  status?:
    | 'pending'
    | 'planned'
    | 'queued'
    | 'provisioning'
    | 'running'
    | 'paused'
    | 'succeeded'
    | 'completed'
    | 'failed'
    | 'blocked'
    | 'cancelled';
  progress?: number;
  config?: Record<string, any> | null;
  results?: Record<string, any> | null;
  summary?: string | null;
  started_at?: string | null;
  completed_at?: string | null;
}

export interface ExperimentRunStartRequest {
  source_id: string;
  commands: string[];
  latex_project_id?: string | null;
  timeout_seconds?: number;
  start_immediately?: boolean;
}

export interface ExperimentRunStartResponse {
  run: ExperimentRun;
  agent_job_id: string;
}

export interface ExperimentRunOperatorAction {
  action: string;
  actor_user_id?: string | null;
  at?: string | null;
  note?: string | null;
  previous_status?: string | null;
  new_status?: string | null;
  linked_job_id?: string | null;
  linked_job_action?: string | null;
  outcome_status?: string | null;
  outcome_reason?: string | null;
  parent_run_id?: string | null;
  child_run_id?: string | null;
}

export interface ExperimentRunActionRequest {
  action: 'start' | 'sync' | 'pause' | 'resume' | 'cancel' | 'retry' | 'requeue';
  note?: string | null;
  start_immediately?: boolean;
}

export interface ExperimentRunActionResponse {
  run: ExperimentRun;
  agent_job_id?: string | null;
}

// ============================================================================
// AI Hub / Training Types
// ============================================================================

// Dataset types
export type DatasetType = 'instruction' | 'chat' | 'completion' | 'preference';
export type DatasetFormat = 'alpaca' | 'sharegpt' | 'custom';
export type DatasetStatus = 'draft' | 'validating' | 'ready' | 'error' | 'archived';

export interface DatasetSample {
  id: string;
  dataset_id: string;
  sample_index: number;
  content: {
    instruction: string;
    input?: string;
    output: string;
  };
  source_document_id?: string;
  input_tokens: number;
  output_tokens: number;
  is_flagged: boolean;
  flag_reason?: string;
  created_at: string;
}

export interface DatasetSampleCreate {
  instruction: string;
  input?: string;
  output: string;
  source_document_id?: string;
}

export interface AddSamplesResponse {
  added_count: number;
  total_count: number;
  token_count: number;
}

export interface DatasetSamplesResponse {
  samples: DatasetSample[];
  total: number;
  page: number;
  page_size: number;
  has_more: boolean;
}

export interface DatasetStats {
  id: string;
  name: string;
  status: DatasetStatus;
  sample_count: number;
  token_count: number;
  input_tokens: number;
  output_tokens: number;
  avg_input_tokens: number;
  avg_output_tokens: number;
  flagged_count: number;
  is_validated: boolean;
  file_size?: number | null;
}

export interface TrainingDataset {
  id: string;
  name: string;
  description?: string;
  dataset_type: DatasetType;
  format: DatasetFormat;
  source_document_ids?: string[];
  file_path?: string;
  file_size?: number;
  sample_count: number;
  token_count: number;
  is_validated: boolean;
  validation_errors?: Array<{ code: string; message: string }>;
  version: number;
  parent_dataset_id?: string;
  user_id: string;
  is_public: boolean;
  status: DatasetStatus;
  created_at: string;
  updated_at?: string;
}

export interface TrainingDatasetCreate {
  name: string;
  description?: string;
  dataset_type?: DatasetType;
  format?: DatasetFormat;
  samples?: DatasetSampleCreate[];
  is_public?: boolean;
}

export interface TrainingDatasetListResponse {
  datasets: TrainingDataset[];
  total: number;
  page: number;
  page_size: number;
  has_more: boolean;
}

export interface DatasetValidationResult {
  is_valid: boolean;
  sample_count: number;
  token_count: number;
  errors: Array<{ code: string; message: string }>;
  warnings: Array<{ code: string; message: string }>;
}

export interface GenerateDatasetRequest {
  name: string;
  description?: string;
  document_ids: string[];
  dataset_type?: DatasetType;
  samples_per_document?: number;
  generation_prompt?: string;
  preset_id?: string;
  extra_instructions?: string;
}

// Training job types
export type TrainingMethod = 'lora' | 'qlora' | 'full_finetune';
export type TrainingBackend = 'local' | 'simulated' | 'modal' | 'runpod';
export type TrainingJobStatus =
  | 'pending'
  | 'queued'
  | 'preparing'
  | 'training'
  | 'saving'
  | 'completed'
  | 'failed'
  | 'cancelled';

export interface HyperparametersConfig {
  lora_r?: number;
  lora_alpha?: number;
  lora_dropout?: number;
  target_modules?: string[];
  learning_rate?: number;
  num_epochs?: number;
  batch_size?: number;
  gradient_accumulation_steps?: number;
  warmup_steps?: number;
  max_seq_length?: number;
  weight_decay?: number;
  max_grad_norm?: number;
}

export interface ResourceConfig {
  device?: string;
  max_memory_gb?: number;
  mixed_precision?: string;
  gradient_checkpointing?: boolean;
}

export interface TrainingJob {
  id: string;
  name: string;
  description?: string;
  training_method: TrainingMethod;
  training_backend: TrainingBackend;
  base_model: string;
  base_model_provider: string;
  dataset_id: string;
  hyperparameters?: HyperparametersConfig;
  resource_config?: ResourceConfig;
  user_id: string;
  status: TrainingJobStatus;
  progress: number;
  current_step?: number;
  total_steps?: number;
  current_epoch?: number;
  total_epochs?: number;
  training_metrics?: {
    current_loss?: number;
    best_loss?: number;
    loss_history?: number[];
    learning_rate?: number;
  };
  final_metrics?: Record<string, any>;
  output_adapter_id?: string;
  error?: string;
  celery_task_id?: string;
  created_at: string;
  started_at?: string;
  completed_at?: string;
}

export interface TrainingJobCreate {
  name: string;
  description?: string;
  training_method?: TrainingMethod;
  training_backend?: TrainingBackend;
  base_model: string;
  base_model_provider?: string;
  dataset_id: string;
  hyperparameters?: HyperparametersConfig;
  resource_config?: ResourceConfig;
  start_immediately?: boolean;
}

export interface TrainingJobListResponse {
  jobs: TrainingJob[];
  total: number;
  page: number;
  page_size: number;
  has_more: boolean;
}

export interface TrainingJobDetail extends TrainingJob {
  dataset_name?: string | null;
  dataset_sample_count?: number | null;
  adapter_name?: string | null;
}

export interface TrainingCheckpoint {
  id: string;
  job_id: string;
  step: number;
  epoch?: number;
  checkpoint_path?: string;
  loss?: number;
  metrics?: Record<string, any>;
  created_at: string;
}

export interface TrainingStats {
  total_jobs: number;
  running_jobs: number;
  completed_jobs: number;
  failed_jobs: number;
  total_training_hours: number;
  total_samples_trained: number;
  avg_final_loss?: number;
}

export type TrainingStatsResponse = TrainingStats;

export interface BaseModelInfo {
  name: string;
  display_name: string;
  provider: string;
  size_gb?: number;
  parameters?: string;
  context_length?: number;
  is_available: boolean;
}

// Model adapter types
export type AdapterType = 'lora' | 'qlora';
export type AdapterStatus = 'training' | 'ready' | 'deploying' | 'deployed' | 'failed' | 'archived';

export interface ModelAdapter {
  id: string;
  name: string;
  display_name: string;
  description?: string;
  base_model: string;
  adapter_type: AdapterType;
  adapter_config?: HyperparametersConfig;
  adapter_path?: string;
  adapter_size?: number;
  training_job_id?: string;
  training_metrics?: Record<string, any>;
  user_id: string;
  is_public: boolean;
  status: AdapterStatus;
  is_deployed: boolean;
  deployment_config?: {
    ollama_model_name?: string;
    deployed_at?: string;
  };
  version: number;
  tags?: string[];
  usage_count: number;
  created_at: string;
  updated_at?: string;
}

export interface ModelAdapterUpdate {
  display_name?: string;
  description?: string;
  is_public?: boolean;
  tags?: string[];
}

export interface ModelAdapterListResponse {
  adapters: ModelAdapter[];
  total: number;
  page: number;
  page_size: number;
  has_more: boolean;
}

export interface ModelAdapterStats {
  total_adapters: number;
  deployed_adapters: number;
  total_usage: number;
}

export interface DeployAdapterRequest {
  ollama_model_name?: string;
}

export interface DeploymentStatusResponse {
  adapter_id: string;
  is_deployed: boolean;
  ollama_model_name?: string | null;
  deployed_at?: string | null;
  status: AdapterStatus | string;
}

export interface TestAdapterRequest {
  prompt: string;
  max_tokens?: number;
  temperature?: number;
}

export interface TestAdapterResponse {
  prompt: string;
  response: string;
  tokens_generated: number;
  generation_time_ms: number;
}

export interface TrainingProgressUpdate {
  type: 'progress';
  job_id: string;
  progress: number;
  status: string;
  current_step?: number;
  total_steps?: number;
  current_epoch?: number;
  total_epochs?: number;
  current_loss?: number;
  learning_rate?: number;
  eta_seconds?: number;
  timestamp: string;
}

// AI Hub eval templates (pluggable)
export interface TrainingEvalTemplateInfo {
  id: string;
  name: string;
  description: string;
  version: number;
  rubric?: Record<string, any>;
  case_count: number;
}

export interface TrainingEvalTemplatesResponse {
  templates: TrainingEvalTemplateInfo[];
}

export interface TrainingEvalRunResponse {
  template_id: string;
  template_version: number;
  base_model: string;
  candidate_model: string;
  judge_model: string;
  avg_score: number;
  num_cases: number;
  results: Array<Record<string, any>>;
}

// ---------------------------------------------------------------- Document folders

/** A node in the document folder tree.
 *
 *  `key` is the whole addressing story and the only thing the documents list
 *  needs: 'all' | 'unfiled' | 'user:<id>' | 'source:<id>' | 'type:<ext>' |
 *  'recent:today|week|month' | 'tag:<t>'. System nodes have no `id`, which is
 *  why the key rather than the id is what gets passed around.
 */
export interface DocumentFolderNode {
  key: string;
  name: string;
  /** 'user' can be edited and filled; 'system' is computed and read-only;
   *  'group' is a heading that holds system nodes and selects nothing. */
  kind: 'user' | 'system' | 'group';
  document_count: number;
  subtree_count: number;
  children: DocumentFolderNode[];
  id?: string | null;
  description?: string | null;
  color?: string | null;
  icon?: string | null;
  position?: number | null;
}

export interface DocumentFolderTree {
  system: DocumentFolderNode[];
  folders: DocumentFolderNode[];
}

export interface DocumentFolderRef {
  key: string;
  id: string;
  name: string;
  color?: string | null;
}

export interface DocumentFolder {
  id: string;
  name: string;
  key: string;
  parent_id?: string | null;
  description?: string | null;
  color?: string | null;
  position: number;
}

export interface DocumentFolderItemsResult {
  added: number;
  already_present: number;
  not_found: number;
  removed: number;
}

// ------------------------------------------------------------ Agent pipelines

/** One stage's compiled plan: the tools derived from its contract, and cost. */
export interface PipelineStagePlan {
  stage_id: string;
  tools: string[];
  iterations: number;
  seconds: number;
  checkpoint: boolean;
  /** Tools with no recorded cost. Counted as zero because nothing knows
   *  better, which is not the same as being free. */
  unpriced: string[];
}

export interface PipelinePlan {
  order: string[];
  stages: PipelineStagePlan[];
  total_seconds: number;
  critical_path_seconds: number;
  checkpoints: string[];
}

/** What is wrong with a pipeline, decided before anything expensive runs.
 *
 *  Three separate answers on purpose: a pipeline can be valid, compile to a
 *  chain, and still be unaffordable. */
export interface PipelineCheck {
  valid: boolean;
  problems: string[];
  expressible: boolean;
  binding_problems: string[];
  description: string[];
  plan?: PipelinePlan | null;
  budget?: {
    affordable: boolean;
    budget_seconds: number;
    estimated_seconds: number;
    critical_path_seconds: number;
    unpriced_tools: string[];
    caveat?: string;
  } | null;
}

export interface PipelineBinding {
  name: string;
  chain_config: Record<string, any>;
  deferred_edges: Array<{ after: string; launch: string; reason: string }>;
  checkpoints: string[];
  description: string[];
}
