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
  | 'system_maintenance'
  | 'quota_warning'
  | 'admin_broadcast'
  | 'mention'
  | 'share'
  | 'comment';

export type NotificationPriority = 'low' | 'normal' | 'high' | 'urgent';

export interface Notification {
  id: string;
  notification_type: NotificationType;
  title: string;
  message: string;
  priority: NotificationPriority;
  related_entity_type?: string;
  related_entity_id?: string;
  data?: Record<string, any>;
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
  research_note_citation_coverage_threshold: number;
  research_note_citation_notify_cooldown_hours: number;
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
  research_note_citation_coverage_threshold?: number;
  research_note_citation_notify_cooldown_hours?: number;
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

export interface AgentJob {
  id: string;
  name: string;
  description?: string;
  job_type: AgentJobType;
  goal: string;
  goal_criteria?: Record<string, any>;
  config?: Record<string, any>;
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
  results?: Record<string, any>;
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
  created_at: string;
  updated_at: string;
}

export interface AgentJobTemplateListResponse {
  templates: AgentJobTemplate[];
  total: number;
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
}

export interface AgentJobProgressUpdate {
  type: 'progress';
  job_id: string;
  progress: number;
  phase: string;
  status: AgentJobStatus;
  iteration: number;
  phase_details?: string;
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
  muted_tokens?: string[];
  muted_patterns?: string[];
  notes?: string;
  created_at: string;
  updated_at: string;
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
  | 'gap_analysis_hypotheses';

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
  created_at?: string;
  started_at?: string;
  completed_at?: string;
}

// ==================== Research Notes ====================

export interface ResearchNote {
  id: string;
  user_id: string;
  title: string;
  content_markdown: string;
  tags?: string[];
  attribution?: Record<string, any> | null;
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
  generator_details?: Record<string, any> | null;
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
  name: string;
  status: 'planned' | 'running' | 'completed' | 'failed' | 'cancelled';
  config?: Record<string, any> | null;
  results?: Record<string, any> | null;
  summary?: string | null;
  progress: number;
  started_at?: string | null;
  completed_at?: string | null;
  created_at: string;
  updated_at: string;
}

export interface ExperimentRunListResponse {
  runs: ExperimentRun[];
}

export interface ExperimentRunCreateRequest {
  name: string;
  config?: Record<string, any> | null;
  summary?: string | null;
}

export interface ExperimentRunUpdateRequest {
  name?: string;
  status?: 'planned' | 'running' | 'completed' | 'failed' | 'cancelled';
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
