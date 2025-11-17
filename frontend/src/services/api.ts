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
  DocumentChunk,
  SystemHealth,
  SystemStats,
  Memory,
  MemoryStats,
  MemorySummary,
} from '../types';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

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
      (error: AxiosError) => {
        const status = error.response?.status;
        if (status === 401) {
          this.clearToken();
          toast.error('Session expired. Please login again.');
          window.location.href = '/login';
        } else if (status && status >= 500) {
          toast.error('Server error. Please try again later.');
        } else if (error.response?.data) {
          const errorMessage = (error.response.data as any)?.detail || 'An error occurred';
          toast.error(errorMessage);
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

  // Authentication endpoints
  async login(username: string, password: string): Promise<LoginResponse> {
    try {
      console.log('API: Attempting login for user:', username);
      const response = await this.client.post('/api/v1/auth/login', {
        username,
        password,
      });
      console.log('API: Login response received:', {
        status: response.status,
        hasAccessToken: !!response.data?.access_token,
        hasUser: !!response.data?.user,
        data: response.data
      });
      return response.data;
    } catch (error: any) {
      console.error('API: Login error:', error);
      console.error('API: Login error response:', error.response?.data);
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
  }): Promise<Document[]> {
    const response = await this.client.get('/api/v1/documents/', { params });
    // Backend returns PaginatedResponse with items array
    return response.data?.items || response.data || [];
  }

  async getDocument(documentId: string): Promise<Document> {
    const response = await this.client.get(`/api/v1/documents/${documentId}`);
    return response.data;
  }

  async getDocumentDownloadUrl(documentId: string, useProxy: boolean = true): Promise<string> {
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
    onProgress?: (progress: number) => void
  ): Promise<{ message: string; document_id: string }> {
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
      },
    });
    return response.data;
  }

  async deleteDocument(documentId: string): Promise<void> {
    await this.client.delete(`/api/v1/documents/${documentId}`);
  }

  async reprocessDocument(documentId: string): Promise<void> {
    await this.client.post(`/api/v1/documents/reprocess/${documentId}`);
  }

  // Document sources endpoints
  async getDocumentSources(): Promise<DocumentSource[]> {
    const response = await this.client.get('/api/v1/documents/sources/');
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

  // Admin endpoints
  async getSystemHealth(): Promise<SystemHealth> {
    const response = await this.client.get('/api/v1/admin/health');
    return response.data;
  }

  async getSystemStats(): Promise<SystemStats> {
    const response = await this.client.get('/api/v1/admin/stats');
    return response.data;
  }

  async triggerFullSync(): Promise<{ task_id: string; message: string; status: string }> {
    const response = await this.client.post('/api/v1/admin/sync/all');
    return response.data;
  }

  async triggerSourceSync(sourceId: string): Promise<{ task_id: string; message: string; status: string }> {
    const response = await this.client.post(`/api/v1/admin/sync/source/${sourceId}`);
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

  async getVectorStoreStats(): Promise<any> {
    const response = await this.client.get('/api/v1/admin/vector-store/stats');
    return response.data;
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
}

// Create singleton instance
export const apiClient = new ApiClient();
export default apiClient;
