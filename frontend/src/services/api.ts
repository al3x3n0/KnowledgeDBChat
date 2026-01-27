/**
 * API client for Knowledge Database backend
 */

import axios, { AxiosInstance, AxiosError, AxiosProgressEvent } from 'axios';
import toast from 'react-hot-toast';
import {
  User,
  ChatSession,
  ChatMessage,
  Document,
  DocumentSource,
  ActiveGitSource,
  GitBranch,
  GitCompareJob,
  DocumentChunk,
  SystemHealth,
  SystemStats,
  Memory,
  MemoryStats,
  MemorySummary,
  Persona,
  TemplateJob,
  TemplateJobListResponse,
  PresentationJob,
  PresentationTemplate,
  ThemeConfig,
  PresentationStyle,
  AgentDefinition,
  AgentDefinitionCreate,
  AgentDefinitionUpdate,
  AgentDefinitionSummary,
  CapabilityInfo,
  KGRelationshipDetail,
  KGRelationshipCreate,
  KGRelationshipUpdate,
  SearchParams,
  SearchResponse,
  Notification,
  NotificationListResponse,
  NotificationPreferences,
  NotificationPreferencesUpdate,
  ArxivSearchResponse,
  ToolAudit,
  LLMUsageSummaryResponse,
  LLMUsageEvent,
  APIKey,
  APIKeyCreate,
  APIKeyCreateResponse,
  APIKeyUpdate,
  APIKeyListResponse,
  APIKeyUsageStats,
  RepoReportJob,
  RepoReportJobCreate,
  RepoReportJobListResponse,
  AvailableSectionsResponse,
} from '../types';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// API error response structure from backend
interface ApiErrorResponse {
  detail?: string;
  message?: string;
}

// Helper to extract error message from axios error
function getErrorMessage(error: AxiosError<ApiErrorResponse>): string {
  return error.response?.data?.detail || error.response?.data?.message || error.message || 'An error occurred';
}

export interface GitRepoRequestPayload {
  provider: 'github' | 'gitlab';
  name?: string;
  token?: string;
  repositories: string[];
  include_files?: boolean;
  include_issues?: boolean;
  include_pull_requests?: boolean;
  include_wiki?: boolean;
  incremental_files?: boolean;
  use_gitignore?: boolean;
  max_pages?: number;
  gitlab_url?: string;
  auto_sync?: boolean;
}

export interface ArxivRequestPayload {
  name?: string;
  search_queries?: string[];
  paper_ids?: string[];
  categories?: string[];
  max_results?: number;
  start?: number;
  sort_by?: 'relevance' | 'lastUpdatedDate' | 'submittedDate';
  sort_order?: 'ascending' | 'descending';
  auto_sync?: boolean;
}

// API-specific types
export interface LoginResponse {
  access_token: string;
  token_type: string;
  user: User;
}

class ApiClient {
  private client: AxiosInstance;
  private token: string | null = null;

  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Request interceptor to add auth token
    this.client.interceptors.request.use(
      (config) => {
        if (this.token) {
          config.headers.Authorization = `Bearer ${this.token}`;
        }
        return config;
      },
      (error) => Promise.reject(error)
    );

    // Response interceptor for error handling
    this.client.interceptors.response.use(
      (response) => response,
      (error: AxiosError<ApiErrorResponse>) => {
        const status = error.response?.status;
        if (status === 401) {
          this.clearToken();
          toast.error('Session expired. Please login again.');
          window.location.href = '/login';
        } else if (status && status >= 500) {
          toast.error('Server error. Please try again later.');
        } else if (error.response?.data) {
          toast.error(getErrorMessage(error));
        }
        return Promise.reject(error);
      }
    );

    // Load token from localStorage
    this.loadToken();
  }

  private loadToken() {
    const savedToken = localStorage.getItem('access_token');
    if (savedToken) {
      this.token = savedToken;
    }
  }

  setToken(token: string) {
    this.token = token;
    localStorage.setItem('access_token', token);
  }

  clearToken() {
    this.token = null;
    localStorage.removeItem('access_token');
  }

  // Generic HTTP methods for workflows and other dynamic endpoints
  async get<T = any>(url: string, config?: any): Promise<{ data: T }> {
    return this.client.get(`/api/v1${url}`, config);
  }

  async post<T = any>(url: string, data?: any, config?: any): Promise<{ data: T }> {
    return this.client.post(`/api/v1${url}`, data, config);
  }

  async put<T = any>(url: string, data?: any, config?: any): Promise<{ data: T }> {
    return this.client.put(`/api/v1${url}`, data, config);
  }

  async delete<T = any>(url: string, config?: any): Promise<{ data: T }> {
    return this.client.delete(`/api/v1${url}`, config);
  }

  // Authentication endpoints
  async login(username: string, password: string): Promise<LoginResponse> {
    try {
      const response = await this.client.post('/api/v1/auth/login', {
        username,
        password,
      });
      return response.data;
    } catch (error) {
      const axiosError = error as AxiosError<ApiErrorResponse>;
      console.error('API: Login error:', getErrorMessage(axiosError));
      throw error;
    }
  }

  async register(userData: {
    username: string;
    email: string;
    password: string;
    full_name?: string;
  }): Promise<User> {
    const response = await this.client.post('/api/v1/auth/register', userData);
    return response.data;
  }

  async getCurrentUser(): Promise<User> {
    const response = await this.client.get('/api/v1/auth/me');
    return response.data;
  }

  async logout(): Promise<void> {
    await this.client.post('/api/v1/auth/logout');
  }

  // Chat endpoints
  async getChatSessions(): Promise<ChatSession[]> {
    try {
      console.log('API: Fetching chat sessions from /api/v1/chat/sessions');
      const response = await this.client.get('/api/v1/chat/sessions');
      console.log('API: Chat sessions response:', response.data);
      // Backend returns PaginatedResponse with items array
      const sessions = response.data?.items || response.data || [];
      console.log('API: Returning sessions:', sessions.length);
      return sessions;
    } catch (error: any) {
      console.error('API: Error fetching chat sessions:', error);
      console.error('API: Error response:', error.response?.data);
      throw error;
    }
  }

  async createChatSession(title?: string): Promise<ChatSession> {
    const response = await this.client.post('/api/v1/chat/sessions', {
      title,
    });
    return response.data;
  }

  async getChatSession(sessionId: string): Promise<ChatSession> {
    const response = await this.client.get(`/api/v1/chat/sessions/${sessionId}`);
    return response.data;
  }

  async sendMessage(sessionId: string, content: string): Promise<ChatMessage> {
    const response = await this.client.post(
      `/api/v1/chat/sessions/${sessionId}/messages`,
      { content }
    );
    return response.data;
  }

  async deleteChatSession(sessionId: string): Promise<void> {
    try {
      console.log('API: Deleting chat session:', sessionId);
      const response = await this.client.delete(`/api/v1/chat/sessions/${sessionId}`);
      console.log('API: Chat session deleted successfully:', response.data);
      return response.data;
    } catch (error: any) {
      console.error('API: Error deleting chat session:', error);
      console.error('API: Error status:', error.response?.status);
      console.error('API: Error response:', error.response?.data);
      console.error('API: Error message:', error.message);
      throw error;
    }
  }

  async submitMessageFeedback(
    messageId: string,
    rating: number,
    feedback?: string
  ): Promise<void> {
    await this.client.put(`/api/v1/chat/messages/${messageId}/feedback`, {
      rating,
      feedback,
    });
  }

  // Document endpoints
  async getDocuments(params?: {
    skip?: number;
    limit?: number;
    source_id?: string;
    search?: string;
    owner_persona_id?: string;
    persona_id?: string;
    persona_role?: string;
  }): Promise<Document[]> {
    const response = await this.client.get('/api/v1/documents/', { params });
    // Backend returns PaginatedResponse with items array
    return response.data?.items || response.data || [];
  }

  async searchDocuments(params: SearchParams): Promise<SearchResponse> {
    const response = await this.client.get('/api/v1/documents/search', { params });
    return response.data;
  }

  async getDocument(documentId: string): Promise<Document> {
    const response = await this.client.get(`/api/v1/documents/${documentId}`);
    return response.data;
  }

  // Knowledge Graph endpoints
  async getKGStats(): Promise<{ entities: number; relationships: number; mentions: number }> {
    const response = await this.client.get('/api/v1/kg/stats');
    return response.data;
  }

  // Vector store / embedding model
  async getVectorStoreStats(): Promise<{ total_chunks?: number; collection_name?: string; embedding_model?: string; available_models?: string[]; error?: string }>{
    const response = await this.client.get('/api/v1/admin/vector-store/stats');
    return response.data;
  }

  async switchEmbeddingModel(modelName: string): Promise<{ message: string }>{
    const response = await this.client.post('/api/v1/admin/vector-store/switch-model', null, { params: { model_name: modelName } });
    return response.data;
  }

  // LLM models
  async listLLMModels(): Promise<{ models: string[]; default_model?: string }>{
    const response = await this.client.get('/api/v1/admin/llm/models');
    return response.data;
  }

  async switchLLMModel(modelName: string): Promise<{ message: string; model: string }>{
    const response = await this.client.post('/api/v1/admin/llm/switch-model', null, { params: { model_name: modelName } });
    return response.data;
  }

  // User LLM Settings
  async getUserLLMSettings(): Promise<{
    llm_provider: string | null;
    llm_model: string | null;
    llm_api_url: string | null;
    llm_api_key_set: boolean;
    llm_temperature: number | null;
    llm_max_tokens: number | null;
    llm_task_models: Record<string, string> | null;
    llm_task_providers: Record<string, string> | null;
  }> {
    const response = await this.client.get('/api/v1/users/me/llm-settings');
    return response.data;
  }

  async updateUserLLMSettings(settings: {
    llm_provider?: string | null;
    llm_model?: string | null;
    llm_api_url?: string | null;
    llm_api_key?: string | null;
    llm_temperature?: number | null;
    llm_max_tokens?: number | null;
    llm_task_models?: Record<string, string> | null;
    llm_task_providers?: Record<string, string> | null;
  }): Promise<{
    llm_provider: string | null;
    llm_model: string | null;
    llm_api_url: string | null;
    llm_api_key_set: boolean;
    llm_temperature: number | null;
    llm_max_tokens: number | null;
    llm_task_models: Record<string, string> | null;
    llm_task_providers: Record<string, string> | null;
  }> {
    const response = await this.client.put('/api/v1/users/me/llm-settings', settings);
    return response.data;
  }

  async clearUserLLMSettings(): Promise<{ message: string }> {
    const response = await this.client.delete('/api/v1/users/me/llm-settings');
    return response.data;
  }

  async listMyLLMModels(params?: { provider?: string }): Promise<{ provider: string; models: string[]; default_model?: string | null; error?: string | null }> {
    const response = await this.client.get('/api/v1/users/me/llm-models', { params });
    return response.data;
  }

  // Agent Chat API
  async agentChat(request: {
    message: string;
    agent_id?: string;
    conversation_history?: Array<{
      id: string;
      role: string;
      content: string;
      tool_calls?: any[];
      created_at: string;
    }>;
  }): Promise<{
    message: {
      id: string;
      role: string;
      content: string;
      tool_calls?: any[];
      created_at: string;
    };
    tool_results?: Array<{
      id: string;
      tool_name: string;
      tool_input: Record<string, any>;
      tool_output?: any;
      status: string;
      error?: string;
      execution_time_ms?: number;
    }>;
    requires_user_action: boolean;
    action_type?: string;
    routing_info?: {
      agent_id: string;
      agent_name: string;
      agent_display_name: string;
      routing_reason: string;
      handoff_from?: string | null;
    };
  }> {
    const response = await this.client.post('/api/v1/agent/chat', request);
    return response.data;
  }

  async agentGetTools(): Promise<{
    tools: Array<{
      name: string;
      description: string;
      parameters: any;
    }>;
  }> {
    const response = await this.client.get('/api/v1/agent/tools');
    return response.data;
  }

  async agentConfirmDelete(documentId: string): Promise<{
    action: string;
    document_id: string;
    title?: string;
    message?: string;
  }> {
    const response = await this.client.post(`/api/v1/agent/confirm-delete/${documentId}`);
    return response.data;
  }

  async searchArxiv(params: {
    q: string;
    start?: number;
    max_results?: number;
    sort_by?: 'relevance' | 'lastUpdatedDate' | 'submittedDate';
    sort_order?: 'ascending' | 'descending';
  }): Promise<ArxivSearchResponse> {
    const response = await this.client.get('/api/v1/research/arxiv/search', { params });
    return response.data;
  }

  async translateArxivQuery(payload: {
    text: string;
    categories?: string[];
  }): Promise<{ query: string }>{
    const response = await this.client.post('/api/v1/research/arxiv/translate-query', payload);
    return response.data;
  }

  async listArxivImports(params?: {
    limit?: number;
    offset?: number;
  }): Promise<{ items: Array<any>; total: number; limit: number; offset: number }>{
    const response = await this.client.get('/api/v1/research/arxiv/imports', { params });
    return response.data;
  }

  async ingestArxivPapers(payload: {
    name?: string;
    search_queries?: string[];
    paper_ids?: string[];
    categories?: string[];
    max_results?: number;
    start?: number;
    sort_by?: 'relevance' | 'lastUpdatedDate' | 'submittedDate';
    sort_order?: 'ascending' | 'descending';
    auto_sync?: boolean;
    auto_summarize?: boolean;
    auto_literature_review?: boolean;
    topic?: string;
  }): Promise<DocumentSource> {
    const response = await this.client.post('/api/v1/documents/sources/arxiv', payload);
    return response.data;
  }

  async summarizeArxivImport(
    sourceId: string,
    payload?: { force?: boolean; limit?: number; only_missing?: boolean }
  ): Promise<{ message: string; source_id: string; queued: number; considered: number }>{
    const response = await this.client.post(`/api/v1/research/arxiv/imports/${sourceId}/summarize`, {
      force: payload?.force ?? false,
      limit: payload?.limit ?? 200,
      only_missing: payload?.only_missing ?? true,
    });
    return response.data;
  }

  async generateReviewForArxivImport(
    sourceId: string,
    payload?: { topic?: string | null }
  ): Promise<{ message: string; source_id: string; task_id: string }>{
    const response = await this.client.post(`/api/v1/research/arxiv/imports/${sourceId}/generate-review`, {
      topic: payload?.topic ?? null,
    });
    return response.data;
  }

  async generateSlidesForArxivImport(
    sourceId: string,
    payload?: {
      title?: string;
      topic?: string | null;
      slide_count?: number;
      style?: string;
      include_diagrams?: boolean;
      prefer_review_document?: boolean;
    }
  ): Promise<{ message: string; source_id: string; presentation_job_id: string }>{
    const response = await this.client.post(`/api/v1/research/arxiv/imports/${sourceId}/generate-slides`, payload || {});
    return response.data;
  }

  async enrichMetadataForArxivImport(
    sourceId: string,
    payload?: { force?: boolean; limit?: number }
  ): Promise<{ message: string; source_id: string; task_id: string }>{
    const response = await this.client.post(`/api/v1/research/arxiv/imports/${sourceId}/enrich-metadata`, {
      force: payload?.force ?? false,
      limit: payload?.limit ?? 500,
    });
    return response.data;
  }

  async listReadingLists(params?: { limit?: number; offset?: number }): Promise<{ items: any[]; total: number; limit: number; offset: number }>{
    const response = await this.client.get('/api/v1/reading-lists', { params });
    return response.data;
  }

  async getReadingList(readingListId: string): Promise<any> {
    const response = await this.client.get(`/api/v1/reading-lists/${readingListId}`);
    return response.data;
  }

  async createReadingList(payload: {
    name: string;
    description?: string | null;
    source_id?: string | null;
    auto_populate_from_source?: boolean;
  }): Promise<any> {
    const response = await this.client.post('/api/v1/reading-lists', payload);
    return response.data;
  }

  async updateReadingListItem(
    readingListId: string,
    itemId: string,
    payload: { status?: 'to-read' | 'reading' | 'done'; priority?: number; position?: number; notes?: string | null }
  ): Promise<any> {
    const response = await this.client.put(`/api/v1/reading-lists/${readingListId}/items/${itemId}`, payload);
    return response.data;
  }

  async createWorkflow(payload: any): Promise<any> {
    const response = await this.client.post('/api/v1/workflows', payload);
    return response.data;
  }

  async searchKGEntities(
    q?: string,
    limit: number = 50,
    offset: number = 0
  ): Promise<{ items: Array<{ id: string; canonical_name: string; entity_type: string }>; total: number; limit: number; offset: number }> {
    const response = await this.client.get('/api/v1/kg/entities', { params: { q, limit, offset } });
    return response.data;
  }

  async getKGDocumentGraph(documentId: string): Promise<{ nodes: Array<{ id: string; name: string; type: string }>; edges: Array<{ id: string; type: string; source: string; target: string; confidence?: number; evidence?: string | null; chunk_id?: string | null }> }> {
    const response = await this.client.get(`/api/v1/kg/document/${documentId}/graph`);
    return response.data;
  }

  async rebuildKGForDocument(documentId: string): Promise<{ message: string; mentions: number; relationships: number }> {
    const response = await this.client.post(`/api/v1/kg/document/${documentId}/rebuild`);
    return response.data;
  }

  async summarizeDocument(documentId: string, force: boolean = false): Promise<{ message: string; task_id: string }>{
    const response = await this.client.post(`/api/v1/documents/${documentId}/summarize`, null, { params: { force } });
    return response.data;
  }

  async getRelatedDocuments(documentId: string, limit: number = 8): Promise<{ items: any[]; limit: number }>{
    const response = await this.client.get(`/api/v1/documents/${documentId}/related`, { params: { limit } });
    return response.data;
  }

  async summarizeMissingDocuments(limit: number = 500): Promise<{ queued: number }>{
    const response = await this.client.post(`/api/v1/documents/summarize-missing`, null, { params: { limit } });
    return response.data;
  }

  async listPersonas(params?: {
    search?: string;
    page?: number;
    page_size?: number;
    include_inactive?: boolean;
  }): Promise<{ items: Persona[]; total: number; page: number; page_size: number }> {
    const response = await this.client.get('/api/v1/personas/', { params });
    return response.data;
  }

  async getPersona(personaId: string): Promise<Persona> {
    const response = await this.client.get(`/api/v1/personas/${personaId}`);
    return response.data;
  }

  async createPersona(data: {
    name: string;
    platform_id?: string | null;
    user_id?: string | null;
    description?: string | null;
    avatar_url?: string | null;
    is_active?: boolean;
    is_system?: boolean;
  }): Promise<Persona> {
    const response = await this.client.post('/api/v1/personas/', data);
    return response.data;
  }

  async updatePersona(
    personaId: string,
    data: {
      name?: string;
      platform_id?: string | null;
      user_id?: string | null;
      description?: string | null;
      avatar_url?: string | null;
      is_active?: boolean;
      is_system?: boolean;
    }
  ): Promise<Persona> {
    const response = await this.client.put(`/api/v1/personas/${personaId}`, data);
    return response.data;
  }

  async deletePersona(personaId: string): Promise<void> {
    await this.client.delete(`/api/v1/personas/${personaId}`);
  }

  async requestPersonaEdit(
    personaId: string,
    data: { message: string; document_id?: string | null }
  ): Promise<{ id: string }> {
    const response = await this.client.post(`/api/v1/personas/${personaId}/edit-request`, data);
    return response.data;
  }

  // Agent Definition Admin API
  async listAgentDefinitions(params?: {
    search?: string;
    active_only?: boolean;
  }): Promise<{ agents: AgentDefinitionSummary[]; total: number }> {
    const response = await this.client.get('/api/v1/agent/agents', { params });
    return response.data;
  }

  async getAgentDefinition(agentId: string): Promise<AgentDefinition> {
    const response = await this.client.get(`/api/v1/agent/agents/${agentId}`);
    return {
      ...response.data,
      system_prompt: response.data?.system_prompt ?? null,
    };
  }

  async createAgentDefinition(data: AgentDefinitionCreate): Promise<AgentDefinition> {
    const response = await this.client.post('/api/v1/agent/agents', data);
    return response.data;
  }

  async updateAgentDefinition(
    agentId: string,
    data: AgentDefinitionUpdate
  ): Promise<AgentDefinition> {
    const response = await this.client.put(`/api/v1/agent/agents/${agentId}`, data);
    return response.data;
  }

  async deleteAgentDefinition(agentId: string): Promise<{ status: string; deleted: string; name: string }> {
    const response = await this.client.delete(`/api/v1/agent/agents/${agentId}`);
    return response.data;
  }

  async duplicateAgentDefinition(agentId: string): Promise<AgentDefinition> {
    const response = await this.client.post(`/api/v1/agent/agents/${agentId}/duplicate`);
    return response.data;
  }

  async listAgentCapabilities(): Promise<{ capabilities: CapabilityInfo[] }> {
    const response = await this.client.get('/api/v1/agent/capabilities');
    return response.data;
  }

  async listAgentTools(): Promise<{ tools: Array<{ name: string; description: string; parameters: any }> }> {
    const response = await this.client.get('/api/v1/agent/tools');
    return response.data;
  }

  async mergeKGEntities(sourceId: string, targetId: string): Promise<{ message: string }> {
    const response = await this.client.post('/api/v1/kg/entities/merge', { source_id: sourceId, target_id: targetId });
    return response.data;
  }

  async getKGEntity(entityId: string): Promise<{ id: string; canonical_name: string; entity_type: string; description?: string | null; properties?: string | null }> {
    const response = await this.client.get(`/api/v1/kg/entity/${entityId}`);
    return response.data;
  }

  async updateKGEntity(
    entityId: string,
    data: Partial<{ canonical_name: string; entity_type: string; description: string | null; properties: string | null }>
  ): Promise<{ id: string; canonical_name: string; entity_type: string; description?: string | null; properties?: string | null }> {
    const response = await this.client.patch(`/api/v1/kg/entity/${entityId}`, data);
    return response.data;
  }

  async deleteKGEntity(entityId: string, confirmName: string): Promise<{ message: string }> {
    const response = await this.client.delete(`/api/v1/kg/entity/${entityId}`, { params: { confirm_name: confirmName } });
    return response.data;
  }

  async getKGAuditLogs(params: {
    action?: string;
    user_id?: string;
    date_from?: string;
    date_to?: string;
    limit?: number;
    offset?: number;
  }): Promise<{ items: Array<{ id: string; user_id: string; user_name?: string; action: string; details?: string; created_at: string }>; total: number; limit: number; offset: number }>{
    const response = await this.client.get('/api/v1/kg/audit', { params });
    return response.data;
  }

  async getGlobalKGGraph(params?: {
    entity_types?: string;
    relation_types?: string;
    min_confidence?: number;
    min_mentions?: number;
    limit_nodes?: number;
    limit_edges?: number;
    search?: string;
  }): Promise<{
    nodes: Array<{
      id: string;
      name: string;
      type: string;
      mention_count?: number;
      description?: string;
    }>;
    edges: Array<{
      id: string;
      type: string;
      source: string;
      target: string;
      confidence?: number;
      evidence?: string | null;
      chunk_id?: string | null;
    }>;
    metadata: {
      total_entities: number;
      total_relationships: number;
      filtered_nodes: number;
      filtered_edges: number;
      entity_types: string[];
      relation_types: string[];
    };
  }> {
    const response = await this.client.get('/api/v1/kg/global/graph', { params });
    return response.data;
  }

  // Users (admin)
  async searchUsers(search: string, page: number = 1, page_size: number = 10): Promise<{ items: Array<User>; total: number; page: number; page_size: number }>{
    const response = await this.client.get('/api/v1/users/', { params: { search, page, page_size } });
    return response.data;
  }

  // Tool audit & approvals (admin)
  async listToolAudits(params?: { status?: string; limit?: number }): Promise<ToolAudit[]> {
    const response = await this.client.get('/api/v1/audit/tools', { params });
    return response.data;
  }

  async approveToolAudit(auditId: string, note?: string): Promise<ToolAudit> {
    const response = await this.client.post(`/api/v1/audit/tools/${auditId}/approve`, { note });
    return response.data;
  }

  async rejectToolAudit(auditId: string, note?: string): Promise<ToolAudit> {
    const response = await this.client.post(`/api/v1/audit/tools/${auditId}/reject`, { note });
    return response.data;
  }

  async runToolAudit(auditId: string): Promise<ToolAudit> {
    const response = await this.client.post(`/api/v1/audit/tools/${auditId}/run`);
    return response.data;
  }

  // LLM usage (tokens/latency)
  async getLLMUsageSummary(params?: {
    provider?: string;
    model?: string;
    task_type?: string;
    user_id?: string;
    date_from?: string;
    date_to?: string;
  }): Promise<LLMUsageSummaryResponse> {
    const response = await this.client.get('/api/v1/usage/llm/summary', { params });
    return response.data;
  }

  async listLLMUsageEvents(params?: {
    provider?: string;
    model?: string;
    task_type?: string;
    user_id?: string;
    date_from?: string;
    date_to?: string;
    page?: number;
    page_size?: number;
  }): Promise<{ items: LLMUsageEvent[]; total: number; page: number; page_size: number; total_pages: number }>{
    const response = await this.client.get('/api/v1/usage/llm/events', { params });
    return response.data;
  }

  async getKGChunk(
    chunkId: string,
    evidence?: string
  ): Promise<{ id: string; document_id: string; document_title?: string; chunk_index: number; content: string; match_start?: number | null; match_end?: number | null }> {
    const response = await this.client.get(`/api/v1/kg/chunk/${chunkId}`, { params: evidence ? { evidence } : undefined });
    return response.data;
  }

  async getKGMentions(
    entityId: string,
    limit: number = 25,
    offset: number = 0
  ): Promise<{ items: Array<{ id: string; entity_id: string; document_id: string; document_title?: string; chunk_id?: string | null; text: string; sentence?: string | null; start_pos?: number | null; end_pos?: number | null; created_at?: string | null }>; total: number; limit: number; offset: number }> {
    const response = await this.client.get(`/api/v1/kg/entity/${entityId}/mentions`, { params: { limit, offset } });
    return response.data;
  }

  // Knowledge Graph Relationship CRUD
  async listKGRelationTypes(): Promise<{ types: string[] }> {
    const response = await this.client.get('/api/v1/kg/relation-types');
    return response.data;
  }

  async getKGRelationship(relId: string): Promise<KGRelationshipDetail> {
    const response = await this.client.get(`/api/v1/kg/relationship/${relId}`);
    return response.data;
  }

  async createKGRelationship(data: KGRelationshipCreate): Promise<KGRelationshipDetail> {
    const response = await this.client.post('/api/v1/kg/relationship', data);
    return response.data;
  }

  async updateKGRelationship(relId: string, data: KGRelationshipUpdate): Promise<KGRelationshipDetail> {
    const response = await this.client.patch(`/api/v1/kg/relationship/${relId}`, data);
    return response.data;
  }

  async deleteKGRelationship(relId: string, confirm: boolean = false): Promise<{ message: string; deleted?: any; relationship?: any }> {
    const response = await this.client.delete(`/api/v1/kg/relationship/${relId}`, { params: { confirm } });
    return response.data;
  }

  async getDocumentDownloadUrl(documentId: string, useProxy: boolean = true, useVideoStreamer: boolean = false): Promise<string> {
    // Use video streamer for video/audio files
    if (useVideoStreamer) {
      // 1) Prefer explicit env override
      const envVideoBase = (process.env.REACT_APP_VIDEO_STREAM_URL || '').replace(/\/$/, '');
      if (envVideoBase) {
        const url = envVideoBase.match(/\/stream$/)
          ? `${envVideoBase}/${documentId}`
          : envVideoBase.match(/\/video$/)
            ? `${envVideoBase}/${documentId}`
            : `${envVideoBase}/video/${documentId}`;
        console.log('Video streamer URL (env):', url, 'for document:', documentId);
        return url;
      }

      // 2) Derive from known base URLs
      let baseUrl = this.client.defaults.baseURL || API_BASE_URL;
      baseUrl = baseUrl.replace(/\/$/, '');
      if (baseUrl.endsWith('/api')) baseUrl = baseUrl.slice(0, -4);

      // If pointing to nginx (port 3000), use /video
      if (/(:|\/)3000(\/|$)/.test(baseUrl)) {
        const url = `${baseUrl}/video/${documentId}`;
        console.log('Video streamer URL (nginx base):', url, 'for document:', documentId);
        return url;
      }

      // 3) Try current page origin (useful when running behind nginx)
      if (typeof window !== 'undefined' && window.location?.origin) {
        const origin = window.location.origin.replace(/\/$/, '');
        if (/(:|\/)3000(\/|$)/.test(origin)) {
          const url = `${origin}/video/${documentId}`;
          console.log('Video streamer URL (window origin):', url, 'for document:', documentId);
          return url;
        }
      }

      // 4) Fallback to direct video-streamer (dev without nginx)
      const fallback = `http://localhost:8080/stream/${documentId}`;
      console.log('Video streamer URL (fallback direct):', fallback, 'for document:', documentId);
      return fallback;
    }
    
    if (useProxy) {
      // Return the direct download URL (backend will stream the file)
      // The URL will be used with an anchor tag, and auth token will be added via Authorization header
      // or query parameter if needed
      const baseUrl = this.client.defaults.baseURL || API_BASE_URL;
      return `${baseUrl}/api/v1/documents/${documentId}/download?use_proxy=true`;
    } else {
      // Legacy mode: get presigned URL
      const response = await this.client.get(`/api/v1/documents/${documentId}/download?use_proxy=false`);
      return response.data.download_url;
    }
  }

  async downloadDocument(documentId: string, useProxy: boolean = true): Promise<{ blob: Blob; filename: string }> {
    // Download document as blob with authentication
    const response = await this.client.get(
      `/api/v1/documents/${documentId}/download?use_proxy=${useProxy}`,
      {
        responseType: 'blob', // Important: tell axios to handle as blob
      }
    );

    // Get filename from Content-Disposition header
    const contentDisposition = response.headers['content-disposition'] || response.headers['Content-Disposition'];
    let filename = `document_${documentId}`;
    if (contentDisposition) {
      const filenameMatch = contentDisposition.match(/filename[^;=\n]*=((['"]).*?\2|[^;\n]*)/);
      if (filenameMatch && filenameMatch[1]) {
        filename = filenameMatch[1].replace(/['"]/g, '');
      }
    }

    return {
      blob: response.data,
      filename,
    };
  }

  async uploadDocument(
    file: File,
    title?: string,
    tags?: string[]
  ): Promise<{ message: string; document_id: string }> {
    const formData = new FormData();
    formData.append('file', file);
    if (title) formData.append('title', title);
    if (tags) formData.append('tags', tags.join(','));

    const response = await this.client.post('/api/v1/documents/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  }

  async uploadDocumentWithProgress(
    file: File,
    title?: string,
    tags?: string[],
    onProgress?: (progress: number) => void,
    onStatusChange?: (status: string) => void,
    onBytesProgress?: (uploaded: number, total: number) => void
  ): Promise<{ message: string; document_id: string }> {
    // For large files (especially videos), use chunked upload
    const LARGE_FILE_THRESHOLD = 50 * 1024 * 1024; // 50MB
    if (file.size > LARGE_FILE_THRESHOLD) {
      return this.uploadDocumentChunked(file, title, tags, onProgress, onStatusChange);
    }
    
    // For smaller files, use regular upload with progress
    const formData = new FormData();
    formData.append('file', file);
    if (title) formData.append('title', title);
    if (tags) formData.append('tags', tags.join(','));

    const response = await this.client.post('/api/v1/documents/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress: (progressEvent) => {
        if (progressEvent.total && onProgress) {
          const percentCompleted = Math.round(
            (progressEvent.loaded * 100) / progressEvent.total
          );
          onProgress(percentCompleted);
        }
        if (progressEvent.total && onBytesProgress) {
          onBytesProgress(progressEvent.loaded, progressEvent.total);
        }
      },
    });
    return response.data;
  }

  /**
   * Chunked upload for large files with resume capability
   */
  async uploadDocumentChunked(
    file: File,
    title?: string,
    tags?: string[],
    onProgress?: (progress: number) => void,
    onStatusChange?: (status: string) => void,
    onBytesProgress?: (uploaded: number, total: number) => void
  ): Promise<{ message: string; document_id: string }> {
    const CHUNK_SIZE = 5 * 1024 * 1024; // 5MB chunks
    const totalChunks = Math.ceil(file.size / CHUNK_SIZE);
    
    // Check for existing upload session
    let sessionId: string | null = null;
    let uploadedChunks: number[] = [];
    
    try {
      // Initialize upload session
      onStatusChange?.('Initializing upload...');
      const initFormData = new FormData();
      
      // Ensure filename is not empty
      const filename = file.name || `upload_${Date.now()}`;
      initFormData.append('filename', filename);
      initFormData.append('file_size', file.size.toString());
      initFormData.append('file_type', file.type || '');
      initFormData.append('content_type', file.type || '');
      if (title) initFormData.append('title', title);
      if (tags) initFormData.append('tags', tags.join(','));
      
      const initResponse = await this.client.post('/api/v1/upload/init', initFormData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      sessionId = initResponse.data.session_id;
      uploadedChunks = initResponse.data.uploaded_chunks || [];

      // Bytes already uploaded (resume)
      const sizeForIndex = (index: number) => index < totalChunks - 1
        ? CHUNK_SIZE
        : file.size - (totalChunks - 1) * CHUNK_SIZE;
      let uploadedSoFar = uploadedChunks.reduce((sum: number, idx: number) => sum + sizeForIndex(idx), 0);
      onBytesProgress?.(uploadedSoFar, file.size);
      
      onStatusChange?.('Uploading chunks...');
      
      // Upload each chunk
      for (let chunkIndex = 0; chunkIndex < totalChunks; chunkIndex++) {
        // Skip if already uploaded
        if (uploadedChunks.includes(chunkIndex)) {
          const progress = ((chunkIndex + 1) / totalChunks) * 100;
          onProgress?.(Math.round(progress));
          uploadedSoFar += sizeForIndex(chunkIndex);
          onBytesProgress?.(uploadedSoFar, file.size);
          continue;
        }
        
        const start = chunkIndex * CHUNK_SIZE;
        const end = Math.min(start + CHUNK_SIZE, file.size);
        const chunk = file.slice(start, end);
        
        const chunkFormData = new FormData();
        chunkFormData.append('chunk_number', chunkIndex.toString());
        chunkFormData.append('chunk', chunk, file.name);
        
        try {
          await this.client.post(`/api/v1/upload/${sessionId}/chunk`, chunkFormData, {
            headers: {
              'Content-Type': 'multipart/form-data',
            },
            onUploadProgress: (progressEvent) => {
              if (progressEvent.total) {
                // Progress for this chunk
                const chunkProgress = (progressEvent.loaded / progressEvent.total) * 100;
                // Overall progress
                const overallProgress = ((chunkIndex + chunkProgress / 100) / totalChunks) * 100;
                onProgress?.(Math.round(overallProgress));
                onBytesProgress?.(uploadedSoFar + progressEvent.loaded, file.size);
              }
            },
          });
          
          uploadedChunks.push(chunkIndex);
          uploadedSoFar += (end - start);
          onBytesProgress?.(uploadedSoFar, file.size);
        } catch (error: any) {
          // If chunk upload fails, we can resume later
          console.error(`Failed to upload chunk ${chunkIndex}:`, error);
          throw new Error(`Failed to upload chunk ${chunkIndex}: ${error.message}`);
        }
      }
      
      // Complete upload
      onStatusChange?.('Finalizing upload...');
      const completeResponse = await this.client.post(`/api/v1/upload/${sessionId}/complete`);
      
      onStatusChange?.('Upload complete!');
      onProgress?.(100);
      onBytesProgress?.(file.size, file.size);
      
      return {
        message: completeResponse.data.message,
        document_id: completeResponse.data.document_id,
      };
    } catch (error: any) {
      console.error('Chunked upload error:', error);
      throw error;
    }
  }

  /**
   * Get upload session status (for resume)
   */
  async getUploadStatus(sessionId: string): Promise<{
    session_id: string;
    filename: string;
    file_size: number;
    total_chunks: number;
    uploaded_chunks: number[];
    uploaded_bytes: number;
    progress: number;
    status: string;
    can_resume: boolean;
  }> {
    const response = await this.client.get(`/api/v1/upload/${sessionId}/status`);
    return response.data;
  }

  async deleteDocument(documentId: string): Promise<void> {
    await this.client.delete(`/api/v1/documents/${documentId}`);
  }

  async reprocessDocument(documentId: string): Promise<void> {
    await this.client.post(`/api/v1/documents/reprocess/${documentId}`);
  }

  async transcribeDocument(documentId: string): Promise<{ message: string }> {
    const response = await this.client.post(`/api/v1/documents/${documentId}/transcribe`);
    return response.data;
  }

  // Document sources endpoints
  async getDocumentSources(): Promise<DocumentSource[]> {
    const response = await this.client.get('/api/v1/documents/sources/');
    return response.data;
  }

  async getActiveGitSources(): Promise<ActiveGitSource[]> {
    const response = await this.client.get('/api/v1/documents/sources/git-active');
    return response.data;
  }

  async attachPresentationAudio(
    documentId: string,
    file: File,
    language?: string
  ): Promise<{
    audio_url: string;
    alignment: any[];
    duration?: number;
    audio_track: any;
    transcript_document_id?: string;
  }> {
    const formData = new FormData();
    formData.append('audio', file);
    if (language) {
      formData.append('language', language);
    }
    const response = await this.client.post(
      `/api/v1/documents/${documentId}/presentation/audio`,
      formData,
      {
        headers: { 'Content-Type': 'multipart/form-data' },
      }
    );
    return response.data;
  }

  async getPresentationAudio(documentId: string): Promise<{
    audio_url: string;
    alignment: any[];
    duration?: number;
    transcript_document_id?: string;
    file_name?: string;
    content_type?: string;
  }> {
    const response = await this.client.get(`/api/v1/documents/${documentId}/presentation/audio`);
    return response.data;
  }

  async getGitBranches(sourceId: string, repository: string): Promise<GitBranch[]> {
    const response = await this.client.get(`/api/v1/git/sources/${sourceId}/branches`, {
      params: { repository },
    });
    return response.data;
  }

  async startGitComparison(
    sourceId: string,
    payload: { repository: string; base_branch: string; compare_branch: string; include_files?: boolean; explain?: boolean }
  ): Promise<GitCompareJob> {
    const response = await this.client.post(`/api/v1/git/sources/${sourceId}/compare`, payload);
    return response.data;
  }

  async getGitComparisonJobs(sourceId?: string): Promise<GitCompareJob[]> {
    const response = await this.client.get('/api/v1/git/compare/', {
      params: sourceId ? { source_id: sourceId } : undefined,
    });
    return response.data;
  }

  async getGitComparisonJob(jobId: string): Promise<GitCompareJob> {
    const response = await this.client.get(`/api/v1/git/compare/${jobId}`);
    return response.data;
  }

  async cancelGitComparisonJob(jobId: string): Promise<{ message: string; task_id?: string }> {
    const response = await this.client.post(`/api/v1/git/compare/${jobId}/cancel`);
    return response.data;
  }

  async createDocumentSource(sourceData: {
    name: string;
    source_type: string;
    config: any;
  }): Promise<DocumentSource> {
    const response = await this.client.post('/api/v1/documents/sources/', sourceData);
    return response.data;
  }

  async updateDocumentSource(
    sourceId: string,
    sourceData: {
      name: string;
      source_type: string;
      config: any;
    }
  ): Promise<DocumentSource> {
    const response = await this.client.put(
      `/api/v1/documents/sources/${sourceId}`,
      sourceData
    );
    return response.data;
  }

  async deleteDocumentSource(sourceId: string): Promise<void> {
    await this.client.delete(`/api/v1/documents/sources/${sourceId}`);
  }

  async syncDocumentSource(sourceId: string): Promise<void> {
    await this.client.post(`/api/v1/documents/sources/${sourceId}/sync`);
  }

  async requestGitRepository(payload: GitRepoRequestPayload): Promise<DocumentSource> {
    const response = await this.client.post('/api/v1/documents/sources/git-repo', payload);
    return response.data;
  }

  async requestArxivSource(payload: ArxivRequestPayload): Promise<DocumentSource> {
    const response = await this.client.post('/api/v1/documents/sources/arxiv', payload);
    return response.data;
  }

  async deleteRepoDocuments(sourceId: string): Promise<{ deleted: number; message: string }> {
    const response = await this.client.delete(`/api/v1/documents/sources/${sourceId}/documents`);
    return response.data;
  }

  async cancelUserSource(sourceId: string): Promise<{ message: string; task_id?: string }> {
    const response = await this.client.post(`/api/v1/documents/sources/${sourceId}/cancel`);
    return response.data;
  }

  // Admin endpoints
  async getSystemHealth(): Promise<SystemHealth> {
    const response = await this.client.get('/api/v1/admin/health');
    return response.data;
  }

  // Non-admin system health (for degraded-mode banner)
  async getSystemHealthStatus(): Promise<SystemHealth> {
    const response = await this.client.get('/api/v1/system/health');
    return response.data;
  }

  async getSystemStats(): Promise<SystemStats> {
    const response = await this.client.get('/api/v1/admin/stats');
    return response.data;
  }

  async getSourcesNextRun(): Promise<{ items: Array<{ source_id: string; next_run: string | null }>; count: number }>{
    const response = await this.client.get('/api/v1/admin/sources/next-run');
    return response.data;
  }

  async validateCron(cron: string): Promise<{ valid: boolean; next_run?: string; error?: string }>{
    const response = await this.client.post('/api/v1/admin/validate-cron', cron, {
      headers: { 'Content-Type': 'text/plain' },
    });
    return response.data;
  }

  async getSourceSyncLogs(sourceId: string, limit: number = 20, offset: number = 0): Promise<{ items: any[] }>{
    const response = await this.client.get(`/api/v1/admin/sources/${sourceId}/sync-logs`, { params: { limit, offset } });
    return response.data;
  }

  async exportSourceSyncLogsCSV(sourceId: string, limit: number = 1000, offset: number = 0): Promise<Blob> {
    const response = await this.client.get(`/api/v1/admin/sources/${sourceId}/sync-logs.csv`, {
      params: { limit, offset },
      responseType: 'blob',
    });
    return response.data as Blob;
  }

  async getFeatureFlags(): Promise<{ knowledge_graph_enabled: boolean; summarization_enabled: boolean; auto_summarize_on_process: boolean }>{
    const response = await this.client.get('/api/v1/admin/flags');
    return response.data;
  }

  async updateFeatureFlags(flags: Partial<{ knowledge_graph_enabled: boolean; summarization_enabled: boolean; auto_summarize_on_process: boolean }>): Promise<{ updated: Record<string, boolean> }>{
    const response = await this.client.post('/api/v1/admin/flags', flags);
    return response.data;
  }

  async triggerFullSync(): Promise<{ task_id: string; message: string; status: string }> {
    const response = await this.client.post('/api/v1/admin/sync/all');
    return response.data;
  }

  async triggerSourceSync(sourceId: string, options?: { forceFull?: boolean }): Promise<{ task_id: string; message: string; status: string }> {
    const response = await this.client.post(`/api/v1/admin/sync/source/${sourceId}`, null, {
      params: options?.forceFull ? { force_full: true } : undefined,
    });
    return response.data;
  }

  async getTaskStatus(): Promise<any> {
    const response = await this.client.get('/api/v1/admin/tasks/status');
    return response.data;
  }

  async getSystemLogs(lines: number = 100): Promise<{ logs: string[]; total_lines: number; returned_lines: number }> {
    const response = await this.client.get('/api/v1/admin/logs', {
      params: { lines },
    });
    return response.data;
  }

  async resetVectorStore(): Promise<void> {
    await this.client.post('/api/v1/admin/vector-store/reset');
  }

  

  // Memory endpoints
  async searchMemories(params: {
    query: string;
    memory_types?: string[];
    tags?: string[];
    min_importance?: number;
    limit?: number;
  }): Promise<Memory[]> {
    const response = await this.client.post('/api/v1/memory/search', params);
    return response.data;
  }

  async getMemories(params?: {
    session_id?: string;
    memory_types?: string;
    limit?: number;
  }): Promise<Memory[]> {
    const response = await this.client.get('/api/v1/memory/', { params });
    return response.data;
  }

  async getMemory(memoryId: string): Promise<Memory> {
    const response = await this.client.get(`/api/v1/memory/${memoryId}`);
    return response.data;
  }

  async createMemory(memoryData: {
    memory_type: string;
    content: string;
    importance_score?: number;
    context?: any;
    tags?: string[];
    session_id?: string;
  }): Promise<Memory> {
    const response = await this.client.post('/api/v1/memory/', memoryData);
    return response.data;
  }

  async updateMemory(memoryId: string, memoryData: {
    content?: string;
    importance_score?: number;
    context?: any;
    tags?: string[];
    is_active?: boolean;
  }): Promise<Memory> {
    const response = await this.client.put(`/api/v1/memory/${memoryId}`, memoryData);
    return response.data;
  }

  async deleteMemory(memoryId: string): Promise<void> {
    await this.client.delete(`/api/v1/memory/${memoryId}`);
  }

  async extractMemoriesFromSession(sessionId: string): Promise<{ message: string; memories: Memory[] }> {
    const response = await this.client.post(`/api/v1/memory/extract/${sessionId}`);
    return response.data;
  }

  async getMemoryStats(): Promise<MemoryStats> {
    const response = await this.client.get('/api/v1/memory/stats/overview');
    return response.data;
  }

  async getMemorySummary(params: {
    session_id?: string;
    time_range_days?: number;
    include_types?: string[];
  }): Promise<MemorySummary> {
    const response = await this.client.post('/api/v1/memory/summary', params);
    return response.data;
  }

  // Template filling endpoints
  async createTemplateFillJob(
    templateFile: File,
    sourceDocumentIds: string[]
  ): Promise<TemplateJob> {
    const formData = new FormData();
    formData.append('template', templateFile);
    formData.append('source_document_ids', sourceDocumentIds.join(','));

    const response = await this.client.post('/api/v1/templates/fill', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  }

  async getTemplateJob(jobId: string): Promise<TemplateJob> {
    const response = await this.client.get(`/api/v1/templates/${jobId}`);
    return response.data;
  }

  async listTemplateJobs(params?: {
    skip?: number;
    limit?: number;
    status?: string;
  }): Promise<TemplateJobListResponse> {
    const response = await this.client.get('/api/v1/templates/', { params });
    return response.data;
  }

  async deleteTemplateJob(jobId: string): Promise<void> {
    await this.client.delete(`/api/v1/templates/${jobId}`);
  }

  async downloadFilledTemplate(jobId: string): Promise<{ blob: Blob; filename: string }> {
    const response = await this.client.get(
      `/api/v1/templates/${jobId}/download`,
      {
        responseType: 'blob',
        maxRedirects: 0,
        validateStatus: (status) => status >= 200 && status < 400,
      }
    );

    // Handle redirect - get the URL and fetch from there
    if (response.status === 307 || response.status === 302) {
      const redirectUrl = response.headers['location'];
      if (redirectUrl) {
        const redirectResponse = await fetch(redirectUrl);
        const blob = await redirectResponse.blob();
        return { blob, filename: `filled_template.docx` };
      }
    }

    // Get filename from Content-Disposition header
    const contentDisposition = response.headers['content-disposition'];
    let filename = `filled_template_${jobId}.docx`;
    if (contentDisposition) {
      const filenameMatch = contentDisposition.match(/filename[^;=\n]*=((['"]).*?\2|[^;\n]*)/);
      if (filenameMatch && filenameMatch[1]) {
        filename = filenameMatch[1].replace(/['"]/g, '');
      }
    }

    return {
      blob: response.data,
      filename,
    };
  }

  // WebSocket connection for template progress
  createTemplateProgressWebSocket(jobId: string): WebSocket {
    const token = localStorage.getItem('access_token');
    if (!token) {
      throw new Error('No authentication token found. Please log in.');
    }
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    const wsUrl = `${protocol}//${host}/api/v1/templates/${jobId}/progress?token=${encodeURIComponent(token)}`;
    console.log('Creating template progress WebSocket connection to:', wsUrl);
    return new WebSocket(wsUrl);
  }

  // WebSocket connection for chat
  createWebSocket(sessionId: string): WebSocket {
    // Get token from localStorage
    const token = localStorage.getItem('access_token');
    if (!token) {
      throw new Error('No authentication token found. Please log in.');
    }
    
    // Use current window origin to avoid mixed content issues
    // This ensures WebSocket URL matches the page origin (http://127.0.0.1:3000 or http://localhost:3000)
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host; // Includes port (e.g., "127.0.0.1:3000" or "localhost:3000")
    const wsUrl = `${protocol}//${host}/api/v1/chat/sessions/${sessionId}/ws?token=${encodeURIComponent(token)}`;
    
    console.log('Creating WebSocket connection to:', wsUrl);
    return new WebSocket(wsUrl);
  }

  // WebSocket connection for transcription progress
  createTranscriptionProgressWebSocket(documentId: string): WebSocket {
    // Get token from localStorage
    const token = localStorage.getItem('access_token');
    if (!token) {
      throw new Error('No authentication token found. Please log in.');
    }
    
    // Use current window origin to avoid mixed content issues
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    const wsUrl = `${protocol}//${host}/api/v1/documents/${documentId}/transcription-progress?token=${encodeURIComponent(token)}`;
    
    console.log('Creating transcription progress WebSocket connection to:', wsUrl);
    return new WebSocket(wsUrl);
  }

  // WebSocket connection for summarization progress
  createSummarizationProgressWebSocket(documentId: string): WebSocket {
    const token = localStorage.getItem('access_token');
    if (!token) {
      throw new Error('No authentication token found. Please log in.');
    }
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    const wsUrl = `${protocol}//${host}/api/v1/documents/${documentId}/summarization-progress?token=${encodeURIComponent(token)}`;
    console.log('Creating summarization progress WebSocket connection to:', wsUrl);
    return new WebSocket(wsUrl);
  }

  // WebSocket connection for ingestion (source) progress
  createIngestionProgressWebSocket(sourceId: string, options?: { admin?: boolean }): WebSocket {
    const useAdminEndpoint = options?.admin ?? true;
    const token = localStorage.getItem('access_token');
    if (!token) {
      throw new Error('No authentication token found. Please log in.');
    }
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    const basePath = useAdminEndpoint
      ? `/api/v1/admin/sources/${sourceId}/ingestion-progress`
      : `/api/v1/documents/sources/${sourceId}/ingestion-progress`;
    const wsUrl = `${protocol}//${host}${basePath}?token=${encodeURIComponent(token)}`;
    console.log('Creating ingestion progress WebSocket connection to:', wsUrl);
    return new WebSocket(wsUrl);
  }

  async clearSourceError(sourceId: string): Promise<{ message: string }> {
    const response = await this.client.post(`/api/v1/admin/sources/${sourceId}/clear-error`);
    return response.data;
  }

  async dryRunSource(sourceId: string, overrides?: any): Promise<{ success: boolean; total?: number; estimated_existing?: number; estimated_new?: number; sample?: any[]; error?: string; mode?: string; by_type?: Record<string, number> }>{
    const response = await this.client.post(`/api/v1/admin/sources/${sourceId}/dry-run`, overrides || {});
    return response.data;
  }

  async cancelSourceSync(sourceId: string): Promise<{ message: string; task_id?: string }>{
    const response = await this.client.post(`/api/v1/admin/sources/${sourceId}/cancel`);
    return response.data;
  }

  // DOCX Editor endpoints
  async getDocumentForEditing(documentId: string): Promise<{
    html_content: string;
    document_title: string;
    document_id: string;
    version: string;
    editable: boolean;
    warnings?: string[];
  }> {
    const response = await this.client.get(`/api/v1/documents/${documentId}/edit`);
    return response.data;
  }

  async saveDocumentEdits(
    documentId: string,
    content: {
      html_content: string;
      version: string;
      create_backup?: boolean;
    }
  ): Promise<{
    success: boolean;
    document_id: string;
    new_version: string;
    message: string;
    backup_path?: string;
  }> {
    const response = await this.client.put(`/api/v1/documents/${documentId}/edit`, content);
    return response.data;
  }

  // Workflow schema endpoints
  async getBuiltinToolSchemas(): Promise<{
    tools: Array<{
      name: string;
      description: string;
      parameters: Record<string, any>;
      parameter_list: Array<{
        name: string;
        type: string;
        description?: string;
        required: boolean;
        default?: any;
        enum?: string[];
      }>;
      tool_type: 'builtin' | 'custom';
    }>;
  }> {
    const response = await this.client.get('/api/v1/workflows/tools/builtin');
    return response.data;
  }

  async getBuiltinToolSchema(toolName: string): Promise<{
    name: string;
    description: string;
    parameters: Record<string, any>;
    parameter_list: Array<{
      name: string;
      type: string;
      description?: string;
      required: boolean;
      default?: any;
      enum?: string[];
    }>;
    tool_type: 'builtin' | 'custom';
  }> {
    const response = await this.client.get(`/api/v1/workflows/tools/builtin/${toolName}`);
    return response.data;
  }

  async getWorkflowContextSchema(workflowId: string): Promise<{
    nodes: Record<string, Array<{
      path: string;
      type: string;
      from_node: string;
      description?: string;
    }>>;
  }> {
    const response = await this.client.get(`/api/v1/workflows/${workflowId}/context-schema`);
    return response.data;
  }

  async validateWorkflow(workflowId: string): Promise<{
    valid: boolean;
    issues: Array<{
      severity: 'error' | 'warning' | 'info';
      node_id?: string;
      field?: string;
      message: string;
    }>;
  }> {
    const response = await this.client.post(`/api/v1/workflows/${workflowId}/validate`);
    return response.data;
  }

  async getWorkflow(workflowId: string): Promise<{
    id: string;
    name: string;
    description?: string;
    is_active: boolean;
    trigger_config?: Record<string, any>;
    nodes: Array<any>;
    edges: Array<any>;
    created_at: string;
    updated_at: string;
  }> {
    const response = await this.client.get(`/api/v1/workflows/${workflowId}`);
    return response.data;
  }

  async getWorkflowsForSelection(excludeId?: string): Promise<Array<{
    id: string;
    name: string;
    description?: string;
    is_active: boolean;
  }>> {
    const params = excludeId ? { exclude_id: excludeId } : {};
    const response = await this.client.get('/api/v1/workflows/list-for-selection', { params });
    return response.data;
  }

  async synthesizeWorkflow(payload: {
    description: string;
    name?: string;
    is_active?: boolean;
    trigger_config?: Record<string, any>;
  }): Promise<{
    workflow: {
      name: string;
      description?: string;
      is_active: boolean;
      trigger_config?: Record<string, any>;
      nodes: Array<{
        node_id: string;
        node_type: 'start' | 'end' | 'tool' | 'condition' | 'parallel' | 'loop' | 'wait';
        tool_id?: string | null;
        builtin_tool?: string | null;
        config: Record<string, any>;
        position_x: number;
        position_y: number;
      }>;
      edges: Array<{
        source_node_id: string;
        target_node_id: string;
        source_handle?: string | null;
        condition?: Record<string, any> | null;
      }>;
    };
    warnings: string[];
  }> {
    const response = await this.client.post('/api/v1/workflows/synthesize', payload);
    return response.data;
  }

  // Presentation generation endpoints
  async createPresentation(data: {
    title: string;
    topic: string;
    source_document_ids?: string[];
    slide_count?: number;
    style?: PresentationStyle;
    include_diagrams?: boolean;
    template_id?: string;
    custom_theme?: ThemeConfig;
  }): Promise<PresentationJob> {
    const response = await this.client.post('/api/v1/presentations', data);
    return response.data;
  }

  async listPresentations(params?: {
    limit?: number;
    offset?: number;
    status_filter?: string;
  }): Promise<PresentationJob[]> {
    const response = await this.client.get('/api/v1/presentations', { params });
    return response.data;
  }

  async getPresentationJob(jobId: string): Promise<PresentationJob> {
    const response = await this.client.get(`/api/v1/presentations/${jobId}`);
    return response.data;
  }

  async downloadPresentation(jobId: string): Promise<void> {
    // Open the download URL directly in the browser
    // The backend will redirect to a presigned MinIO URL
    const token = localStorage.getItem('access_token');
    const baseUrl = this.client.defaults.baseURL || API_BASE_URL;
    const url = `${baseUrl}/api/v1/presentations/${jobId}/download`;

    // Use fetch with auth header and handle redirect
    const response = await fetch(url, {
      headers: {
        'Authorization': `Bearer ${token}`,
      },
      redirect: 'follow',
    });

    if (response.ok) {
      const blob = await response.blob();
      const contentDisposition = response.headers.get('content-disposition');
      let filename = `presentation_${jobId}.pptx`;
      if (contentDisposition) {
        const filenameMatch = contentDisposition.match(/filename[^;=\n]*=((['"]).*?\2|[^;\n]*)/);
        if (filenameMatch && filenameMatch[1]) {
          filename = filenameMatch[1].replace(/['"]/g, '');
        }
      }

      // Trigger download
      const downloadUrl = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = downloadUrl;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(downloadUrl);
    } else {
      throw new Error(`Download failed: ${response.statusText}`);
    }
  }

  async deletePresentation(jobId: string): Promise<void> {
    await this.client.delete(`/api/v1/presentations/${jobId}`);
  }

  async cancelPresentation(jobId: string): Promise<PresentationJob> {
    const response = await this.client.post(`/api/v1/presentations/${jobId}/cancel`);
    return response.data;
  }

  // WebSocket connection for presentation progress
  createPresentationProgressWebSocket(jobId: string): WebSocket {
    const token = localStorage.getItem('access_token');
    if (!token) {
      throw new Error('No authentication token found. Please log in.');
    }
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    const wsUrl = `${protocol}//${host}/api/v1/presentations/${jobId}/progress?token=${encodeURIComponent(token)}`;
    console.log('Creating presentation progress WebSocket connection to:', wsUrl);
    return new WebSocket(wsUrl);
  }

  // Presentation Template endpoints
  async listPresentationTemplates(): Promise<PresentationTemplate[]> {
    const response = await this.client.get('/api/v1/presentations/templates');
    return response.data;
  }

  async getPresentationTemplate(templateId: string): Promise<PresentationTemplate> {
    const response = await this.client.get(`/api/v1/presentations/templates/${templateId}`);
    return response.data;
  }

  async createPresentationTemplate(data: {
    name: string;
    description?: string;
    template_type?: 'theme' | 'pptx';
    theme_config?: ThemeConfig;
    is_public?: boolean;
  }): Promise<PresentationTemplate> {
    const response = await this.client.post('/api/v1/presentations/templates', data);
    return response.data;
  }

  async updatePresentationTemplate(
    templateId: string,
    data: {
      name?: string;
      description?: string;
      theme_config?: ThemeConfig;
      is_public?: boolean;
      is_active?: boolean;
    }
  ): Promise<PresentationTemplate> {
    const response = await this.client.put(`/api/v1/presentations/templates/${templateId}`, data);
    return response.data;
  }

  async deletePresentationTemplate(templateId: string): Promise<void> {
    await this.client.delete(`/api/v1/presentations/templates/${templateId}`);
  }

  async uploadPresentationTemplate(
    file: File,
    name: string,
    description?: string,
    isPublic?: boolean,
    onProgress?: (progress: number) => void
  ): Promise<PresentationTemplate> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('name', name);
    if (description) formData.append('description', description);
    formData.append('is_public', String(isPublic ?? false));

    const response = await this.client.post(
      '/api/v1/presentations/templates/upload',
      formData,
      {
        headers: { 'Content-Type': 'multipart/form-data' },
        onUploadProgress: (progressEvent) => {
          if (progressEvent.total && onProgress) {
            const percentCompleted = Math.round(
              (progressEvent.loaded * 100) / progressEvent.total
            );
            onProgress(percentCompleted);
          }
        },
      }
    );
    return response.data;
  }

  // ==================== Notification endpoints ====================

  async getNotifications(params?: {
    page?: number;
    page_size?: number;
    unread_only?: boolean;
    notification_types?: string;
  }): Promise<NotificationListResponse> {
    const response = await this.client.get('/api/v1/notifications/', { params });
    return response.data;
  }

  async getUnreadCount(): Promise<{ unread_count: number }> {
    const response = await this.client.get('/api/v1/notifications/unread-count');
    return response.data;
  }

  async markNotificationRead(notificationId: string): Promise<void> {
    await this.client.put(`/api/v1/notifications/${notificationId}/read`);
  }

  async markAllNotificationsRead(): Promise<{ count: number }> {
    const response = await this.client.put('/api/v1/notifications/read-all');
    return response.data;
  }

  async dismissNotification(notificationId: string): Promise<void> {
    await this.client.put(`/api/v1/notifications/${notificationId}/dismiss`);
  }

  async deleteNotification(notificationId: string): Promise<void> {
    await this.client.delete(`/api/v1/notifications/${notificationId}`);
  }

  async getNotificationPreferences(): Promise<NotificationPreferences> {
    const response = await this.client.get('/api/v1/notifications/preferences');
    return response.data;
  }

  async updateNotificationPreferences(
    preferences: NotificationPreferencesUpdate
  ): Promise<NotificationPreferences> {
    const response = await this.client.put('/api/v1/notifications/preferences', preferences);
    return response.data;
  }

  // WebSocket for real-time notifications
  createNotificationsWebSocket(): WebSocket {
    const token = localStorage.getItem('access_token');
    if (!token) {
      throw new Error('No authentication token found. Please log in.');
    }
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    const wsUrl = `${protocol}//${host}/api/v1/notifications/ws?token=${encodeURIComponent(token)}`;
    console.log('Creating notifications WebSocket connection to:', wsUrl);
    return new WebSocket(wsUrl);
  }

  // ==================== API Key endpoints ====================

  async createAPIKey(data: APIKeyCreate): Promise<APIKeyCreateResponse> {
    const response = await this.client.post('/api/v1/api-keys/', data);
    return response.data;
  }

  async listAPIKeys(includeRevoked: boolean = false): Promise<APIKeyListResponse> {
    const response = await this.client.get('/api/v1/api-keys/', {
      params: { include_revoked: includeRevoked },
    });
    return response.data;
  }

  async getAPIKey(keyId: string): Promise<APIKey> {
    const response = await this.client.get(`/api/v1/api-keys/${keyId}`);
    return response.data;
  }

  async updateAPIKey(keyId: string, data: APIKeyUpdate): Promise<APIKey> {
    const response = await this.client.patch(`/api/v1/api-keys/${keyId}`, data);
    return response.data;
  }

  async revokeAPIKey(keyId: string): Promise<void> {
    await this.client.delete(`/api/v1/api-keys/${keyId}`);
  }

  async getAPIKeyUsage(keyId: string, days: number = 30): Promise<APIKeyUsageStats> {
    const response = await this.client.get(`/api/v1/api-keys/${keyId}/usage`, {
      params: { days },
    });
    return response.data;
  }

  // ==================== Repository Report endpoints ====================

  async getRepoReportSections(): Promise<AvailableSectionsResponse> {
    const response = await this.client.get('/api/v1/repo-reports/sections');
    return response.data;
  }

  async createRepoReport(data: RepoReportJobCreate): Promise<RepoReportJob> {
    const response = await this.client.post('/api/v1/repo-reports', data);
    return response.data;
  }

  async listRepoReports(params?: {
    limit?: number;
    offset?: number;
    status_filter?: string;
    output_format?: string;
  }): Promise<RepoReportJobListResponse> {
    const response = await this.client.get('/api/v1/repo-reports', { params });
    return response.data;
  }

  async getRepoReport(jobId: string, includeAnalysis: boolean = false): Promise<RepoReportJob> {
    const response = await this.client.get(`/api/v1/repo-reports/${jobId}`, {
      params: { include_analysis: includeAnalysis },
    });
    return response.data;
  }

  async downloadRepoReport(jobId: string): Promise<void> {
    const token = localStorage.getItem('access_token');
    const baseUrl = this.client.defaults.baseURL || API_BASE_URL;
    const url = `${baseUrl}/api/v1/repo-reports/${jobId}/download`;

    const response = await fetch(url, {
      headers: {
        'Authorization': `Bearer ${token}`,
      },
      redirect: 'follow',
    });

    if (response.ok) {
      const blob = await response.blob();
      const contentDisposition = response.headers.get('content-disposition');
      let filename = `repo_report_${jobId}`;
      if (contentDisposition) {
        const filenameMatch = contentDisposition.match(/filename[^;=\n]*=((['"]).*?\2|[^;\n]*)/);
        if (filenameMatch && filenameMatch[1]) {
          filename = filenameMatch[1].replace(/['"]/g, '');
        }
      }

      // Trigger download
      const downloadUrl = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = downloadUrl;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(downloadUrl);
    } else {
      throw new Error(`Download failed: ${response.statusText}`);
    }
  }

  async deleteRepoReport(jobId: string): Promise<void> {
    await this.client.delete(`/api/v1/repo-reports/${jobId}`);
  }

  async cancelRepoReport(jobId: string): Promise<RepoReportJob> {
    const response = await this.client.post(`/api/v1/repo-reports/${jobId}/cancel`);
    return response.data;
  }

  // WebSocket for repo report progress
  createRepoReportProgressWebSocket(jobId: string): WebSocket {
    const token = localStorage.getItem('access_token');
    if (!token) {
      throw new Error('No authentication token found. Please log in.');
    }
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    const wsUrl = `${protocol}//${host}/api/v1/repo-reports/${jobId}/progress?token=${encodeURIComponent(token)}`;
    console.log('Creating repo report progress WebSocket connection to:', wsUrl);
    return new WebSocket(wsUrl);
  }
}

// Create singleton instance
export const apiClient = new ApiClient();
export default apiClient;
