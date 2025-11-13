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
  is_processed: boolean;
  processing_error?: string;
  created_at: string;
  updated_at: string;
  last_modified?: string;
  source: DocumentSource;
  chunks?: DocumentChunk[];
}

export interface DocumentSource {
  id: string;
  name: string;
  source_type: 'gitlab' | 'confluence' | 'web' | 'file';
  config: any;
  is_active: boolean;
  last_sync?: string;
  created_at: string;
  updated_at: string;
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

export interface DocumentStats {
  total: number;
  processed: number;
  failed: number;
  pending: number;
  success_rate: number;
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


