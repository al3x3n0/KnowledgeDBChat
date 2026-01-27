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
  priority?: number;
  is_active?: boolean;
}

export interface AgentDefinitionUpdate {
  display_name?: string;
  description?: string | null;
  system_prompt?: string;
  capabilities?: string[];
  tool_whitelist?: string[] | null;
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

export interface DocumentStats {
  total: number;
  processed: number;
  failed: number;
  pending: number;
  success_rate: number;
  without_summary?: number;
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
