/**
 * WebSocket-based hook for agent chat with real-time streaming.
 *
 * Provides streaming tool execution feedback for better UX.
 * Includes conversation memory persistence.
 */

import { useState, useCallback, useRef, useEffect } from 'react';
import toast from 'react-hot-toast';

// ============================================================================
// Conversation Memory API
// ============================================================================

interface ConversationData {
  id: string;
  title: string | null;
  status: string;
  messages: any[];
  summary: string | null;
  message_count: number;
  tool_calls_count: number;
  created_at: string;
  updated_at: string;
  last_message_at: string;
}

interface ConversationListItem {
  id: string;
  title: string | null;
  status: string;
  message_count: number;
  tool_calls_count: number;
  summary: string | null;
  last_message_at: string;
  created_at: string;
}

const getAuthHeaders = () => ({
  'Authorization': `Bearer ${localStorage.getItem('access_token')}`,
  'Content-Type': 'application/json',
});

const conversationApi = {
  async getActive(): Promise<ConversationData | null> {
    try {
      const response = await fetch('/api/v1/agent/conversations/active', {
        headers: getAuthHeaders(),
      });
      if (!response.ok) return null;
      return response.json();
    } catch (error) {
      console.error('Failed to get active conversation:', error);
      return null;
    }
  },

  async appendMessage(conversationId: string, message: any, toolCalls?: any[]): Promise<void> {
    try {
      await fetch(`/api/v1/agent/conversations/${conversationId}/messages`, {
        method: 'POST',
        headers: getAuthHeaders(),
        body: JSON.stringify({
          message: message,
          tool_calls: toolCalls || null,
        }),
      });
    } catch (error) {
      console.error('Failed to append message:', error);
    }
  },

  async startNew(): Promise<ConversationData | null> {
    try {
      const response = await fetch('/api/v1/agent/conversations/new', {
        method: 'POST',
        headers: getAuthHeaders(),
      });
      if (!response.ok) return null;
      return response.json();
    } catch (error) {
      console.error('Failed to start new conversation:', error);
      return null;
    }
  },

  async listConversations(limit = 20): Promise<ConversationListItem[]> {
    try {
      const response = await fetch(`/api/v1/agent/conversations?limit=${limit}`, {
        headers: getAuthHeaders(),
      });
      if (!response.ok) return [];
      const data = await response.json();
      return data.conversations || [];
    } catch (error) {
      console.error('Failed to list conversations:', error);
      return [];
    }
  },

  async getConversation(id: string): Promise<ConversationData | null> {
    try {
      const response = await fetch(`/api/v1/agent/conversations/${id}`, {
        headers: getAuthHeaders(),
      });
      if (!response.ok) return null;
      return response.json();
    } catch (error) {
      console.error('Failed to get conversation:', error);
      return null;
    }
  },

  async deleteConversation(id: string): Promise<boolean> {
    try {
      const response = await fetch(`/api/v1/agent/conversations/${id}`, {
        method: 'DELETE',
        headers: getAuthHeaders(),
      });
      return response.ok;
    } catch (error) {
      console.error('Failed to delete conversation:', error);
      return false;
    }
  },
};

export interface AgentToolCall {
  id: string;
  tool_name: string;
  tool_input: Record<string, any>;
  tool_output?: any;
  status: 'pending' | 'running' | 'completed' | 'failed';
  error?: string;
  execution_time_ms?: number;
}

export interface AgentMessage {
  id: string;
  role: 'user' | 'assistant' | 'system' | 'tool';
  content: string;
  tool_calls?: AgentToolCall[];
  created_at: string;
}

export type ConnectionStatus = 'disconnected' | 'connecting' | 'connected' | 'error';
export type ProcessingPhase = 'idle' | 'thinking' | 'planning' | 'executing' | 'generating';

export interface StreamingState {
  phase: ProcessingPhase;
  message: string;
  toolCount?: number;
  currentTool?: AgentToolCall;
  completedTools: AgentToolCall[];
}

// Multi-agent routing info
export interface AgentRoutingInfo {
  agent_id: string;
  agent_name: string;
  agent_display_name: string;
  routing_reason: string;
  handoff_from?: string | null;
}

// Agent info for display
export interface AgentInfo {
  id: string;
  name: string;
  displayName: string;
  description?: string;
  capabilities?: string[];
}

export interface TemplateUploadContext {
  sourceDocumentIds: string[];
  sourceDocuments: { id: string; title: string }[];
}

export interface TemplateJobProgress {
  jobId: string;
  templateFilename: string;
  status: string;
  progress: number;
  currentSection?: string;
  downloadUrl?: string;
  error?: string;
}

export interface UseAgentWebSocketReturn {
  messages: AgentMessage[];
  isLoading: boolean;
  connectionStatus: ConnectionStatus;
  streamingState: StreamingState;
  sendMessage: (content: string) => void;
  pendingUpload: boolean;
  uploadContext: { suggestedTitle?: string; suggestedTags?: string[] } | null;
  handleFileUpload: (file: File) => Promise<void>;
  clearPendingUpload: () => void;
  clearMessages: () => void;
  confirmDelete: (documentId: string) => Promise<void>;
  pendingDelete: { documentId: string; title: string } | null;
  clearPendingDelete: () => void;
  pendingTemplateUpload: boolean;
  templateUploadContext: TemplateUploadContext | null;
  handleTemplateUpload: (file: File) => Promise<void>;
  clearPendingTemplateUpload: () => void;
  activeTemplateJob: TemplateJobProgress | null;
  connect: () => void;
  disconnect: () => void;
  // Conversation memory
  conversationId: string | null;
  conversationHistory: ConversationListItem[];
  isLoadingConversation: boolean;
  startNewConversation: () => Promise<void>;
  loadConversation: (id: string) => Promise<void>;
  refreshConversationHistory: () => Promise<void>;
  // Multi-agent support
  currentAgent: AgentInfo | null;
  injectedMemoriesCount: number;
  lastRoutingReason: string | null;
  availableAgents: AgentInfo[];
  loadAvailableAgents: () => Promise<void>;
}

const getWsUrl = (): string => {
  const token = localStorage.getItem('access_token');
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const host = window.location.host;
  return `${protocol}//${host}/api/v1/agent/ws?token=${token}`;
};

const WELCOME_MESSAGE: AgentMessage = {
  id: 'welcome',
  role: 'assistant',
  content: 'Hello! I can help you manage documents in the knowledge base. You can ask me to:\n\n- **Search** for documents by content or tags\n- **Summarize** documents (single or batch)\n- **Upload** new documents or create from text\n- **Delete** documents (single or batch)\n- **Find similar** documents\n- **Compare** two documents\n- **Manage tags** (add, remove, list)\n- **Fill templates** with document content\n- **View stats** about the knowledge base\n\nHow can I help you today?',
  created_at: new Date().toISOString(),
};

export function useAgentWebSocket(): UseAgentWebSocketReturn {
  const [messages, setMessages] = useState<AgentMessage[]>([WELCOME_MESSAGE]);

  const [isLoading, setIsLoading] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>('disconnected');
  const [streamingState, setStreamingState] = useState<StreamingState>({
    phase: 'idle',
    message: '',
    completedTools: []
  });

  const [pendingUpload, setPendingUpload] = useState(false);
  const [uploadContext, setUploadContext] = useState<{ suggestedTitle?: string; suggestedTags?: string[] } | null>(null);
  const [pendingDelete, setPendingDelete] = useState<{ documentId: string; title: string } | null>(null);
  const [pendingTemplateUpload, setPendingTemplateUpload] = useState(false);
  const [templateUploadContext, setTemplateUploadContext] = useState<TemplateUploadContext | null>(null);
  const [activeTemplateJob, setActiveTemplateJob] = useState<TemplateJobProgress | null>(null);

  // Conversation memory state
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [conversationHistory, setConversationHistory] = useState<ConversationListItem[]>([]);
  const [isLoadingConversation, setIsLoadingConversation] = useState(false);
  const conversationIdRef = useRef<string | null>(null);

  // Multi-agent state
  const [currentAgent, setCurrentAgent] = useState<AgentInfo | null>(null);
  const [injectedMemoriesCount, setInjectedMemoriesCount] = useState(0);
  const [lastRoutingReason, setLastRoutingReason] = useState<string | null>(null);
  const [availableAgents, setAvailableAgents] = useState<AgentInfo[]>([]);
  const turnNumberRef = useRef(0);

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const pingIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const reconnectAttemptsRef = useRef(0);
  const maxReconnectAttempts = 3;

  // Load active conversation on mount
  useEffect(() => {
    const loadActiveConversation = async () => {
      setIsLoadingConversation(true);
      try {
        const conversation = await conversationApi.getActive();
        if (conversation) {
          setConversationId(conversation.id);
          conversationIdRef.current = conversation.id;

          // Load messages from conversation
          if (conversation.messages && conversation.messages.length > 0) {
            const loadedMessages: AgentMessage[] = conversation.messages.map((msg: any) => ({
              id: msg.id,
              role: msg.role,
              content: msg.content,
              tool_calls: msg.tool_calls as AgentToolCall[] | undefined,
              created_at: msg.created_at,
            }));
            setMessages(loadedMessages);
          }
        }

        // Load conversation history
        const history = await conversationApi.listConversations(20);
        setConversationHistory(history);
      } catch (error) {
        console.error('Failed to load conversation:', error);
      } finally {
        setIsLoadingConversation(false);
      }
    };

    loadActiveConversation();
  }, []);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    const token = localStorage.getItem('access_token');
    if (!token) {
      setConnectionStatus('error');
      return;
    }

    try {
      setConnectionStatus('connecting');
      const ws = new WebSocket(getWsUrl());
      wsRef.current = ws;

      ws.onopen = () => {
        setConnectionStatus('connected');
        reconnectAttemptsRef.current = 0;

        // Start ping interval to keep connection alive
        pingIntervalRef.current = setInterval(() => {
          if (ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'ping' }));
          }
        }, 30000);
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          handleWebSocketMessage(data);
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error);
        }
      };

      ws.onclose = () => {
        setConnectionStatus('disconnected');
        setIsLoading(false);

        if (pingIntervalRef.current) {
          clearInterval(pingIntervalRef.current);
        }

        // Attempt reconnect
        if (reconnectAttemptsRef.current < maxReconnectAttempts) {
          reconnectAttemptsRef.current += 1;
          reconnectTimeoutRef.current = setTimeout(() => {
            connect();
          }, 3000);
        }
      };

      ws.onerror = () => {
        setConnectionStatus('error');
      };

    } catch (error) {
      console.error('WebSocket connection error:', error);
      setConnectionStatus('error');
    }
  }, []);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }
    if (pingIntervalRef.current) {
      clearInterval(pingIntervalRef.current);
    }
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setConnectionStatus('disconnected');
  }, []);

  const handleWebSocketMessage = useCallback((data: any) => {
    switch (data.type) {
      case 'connected':
        // Connection established
        break;

      case 'pong':
        // Ping response, connection is alive
        break;

      case 'thinking':
        setStreamingState({
          phase: 'thinking',
          message: data.message,
          completedTools: []
        });
        break;

      case 'planning':
        setStreamingState(prev => ({
          ...prev,
          phase: 'planning',
          message: data.message,
          toolCount: data.tool_count
        }));
        break;

      case 'tool_start':
        setStreamingState(prev => ({
          ...prev,
          phase: 'executing',
          message: `Executing: ${formatToolName(data.tool.tool_name)}`,
          currentTool: data.tool
        }));
        break;

      case 'tool_progress':
        setStreamingState(prev => ({
          ...prev,
          currentTool: prev.currentTool ? {
            ...prev.currentTool,
            status: data.status
          } : undefined
        }));
        break;

      case 'tool_complete':
        setStreamingState(prev => ({
          ...prev,
          completedTools: [...prev.completedTools, data.tool],
          currentTool: undefined
        }));
        break;

      case 'tool_error':
        setStreamingState(prev => ({
          ...prev,
          currentTool: prev.currentTool ? {
            ...prev.currentTool,
            status: 'failed',
            error: data.error
          } : undefined
        }));
        break;

      case 'generating':
        setStreamingState(prev => ({
          ...prev,
          phase: 'generating',
          message: data.message,
          currentTool: undefined
        }));
        break;

      case 'response':
        setIsLoading(false);
        setStreamingState({
          phase: 'idle',
          message: '',
          completedTools: []
        });

        // Capture routing info from multi-agent system
        if (data.routing_info) {
          const routingInfo = data.routing_info as AgentRoutingInfo;
          setCurrentAgent({
            id: routingInfo.agent_id,
            name: routingInfo.agent_name,
            displayName: routingInfo.agent_display_name,
          });
          setLastRoutingReason(routingInfo.routing_reason);
        }

        // Capture injected memories count
        if (data.injected_memories) {
          setInjectedMemoriesCount(data.injected_memories.length);
        } else {
          setInjectedMemoriesCount(0);
        }

        // Increment turn number for tracking
        turnNumberRef.current += 1;

        // Add assistant message
        const assistantMessage: AgentMessage = {
          id: data.message.id,
          role: 'assistant',
          content: data.message.content,
          tool_calls: data.message.tool_calls as AgentToolCall[] | undefined,
          created_at: data.message.created_at,
        };
        setMessages(prev => [...prev, assistantMessage]);

        // Persist assistant message to conversation
        if (conversationIdRef.current) {
          const toolCalls = data.message.tool_calls?.map((tc: any) => ({
            id: tc.id,
            tool_name: tc.tool_name,
            tool_input: tc.tool_input,
            tool_output: tc.tool_output,
            status: tc.status,
            error: tc.error,
            execution_time_ms: tc.execution_time_ms,
          }));
          conversationApi.appendMessage(conversationIdRef.current, assistantMessage, toolCalls);
        }

        // Check for special actions
        if (data.requires_user_action) {
          if (data.action_type === 'upload_file') {
            setPendingUpload(true);
            const uploadTool = data.tool_results?.find(
              (t: any) => t.tool_name === 'request_file_upload'
            );
            if (uploadTool?.tool_output) {
              setUploadContext({
                suggestedTitle: uploadTool.tool_output.suggested_title,
                suggestedTags: uploadTool.tool_output.suggested_tags,
              });
            }
          }
        }

        // Check for template upload requirement
        const templateTool = data.tool_results?.find(
          (t: any) => t.tool_name === 'start_template_fill' && t.tool_output?.action === 'template_upload_required'
        );
        if (templateTool?.tool_output) {
          setPendingTemplateUpload(true);
          setTemplateUploadContext({
            sourceDocumentIds: templateTool.tool_output.source_document_ids || [],
            sourceDocuments: templateTool.tool_output.source_documents || [],
          });
        }

        // Check for pending delete confirmation
        const deleteTool = data.tool_results?.find(
          (t: any) => t.tool_name === 'delete_document' && t.tool_output?.action === 'confirmation_required'
        );
        if (deleteTool?.tool_output) {
          setPendingDelete({
            documentId: deleteTool.tool_output.document_id,
            title: deleteTool.tool_output.title,
          });
        }
        break;

      case 'error':
        setIsLoading(false);
        setStreamingState({
          phase: 'idle',
          message: '',
          completedTools: []
        });
        toast.error(data.message || 'An error occurred');

        const errorMessage: AgentMessage = {
          id: `error-${Date.now()}`,
          role: 'assistant',
          content: 'Sorry, I encountered an error processing your request. Please try again.',
          created_at: new Date().toISOString(),
        };
        setMessages(prev => [...prev, errorMessage]);
        break;
    }
  }, []);

  const sendMessage = useCallback((content: string) => {
    if (!content.trim()) return;

    // Add user message immediately
    const userMessage: AgentMessage = {
      id: `user-${Date.now()}`,
      role: 'user',
      content: content.trim(),
      created_at: new Date().toISOString(),
    };
    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);

    // Persist user message to conversation
    if (conversationIdRef.current) {
      conversationApi.appendMessage(conversationIdRef.current, userMessage);
    }

    // Ensure connection
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      connect();
      // Wait a bit for connection
      setTimeout(() => {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
          const history = messages.slice(-10).map(m => ({
            role: m.role,
            content: m.content,
            created_at: m.created_at
          }));
          wsRef.current.send(JSON.stringify({
            type: 'message',
            content: content,
            conversation_history: history
          }));
        } else {
          setIsLoading(false);
          toast.error('Failed to connect to agent');
        }
      }, 1000);
      return;
    }

    // Send message via WebSocket
    const history = messages.slice(-10).map(m => ({
      role: m.role,
      content: m.content,
      created_at: m.created_at
    }));

    wsRef.current.send(JSON.stringify({
      type: 'message',
      content: content,
      conversation_history: history
    }));
  }, [messages, connect]);

  const handleFileUpload = useCallback(async (file: File) => {
    setIsLoading(true);

    try {
      const formData = new FormData();
      formData.append('file', file);
      if (uploadContext?.suggestedTitle) {
        formData.append('title', uploadContext.suggestedTitle);
      }
      if (uploadContext?.suggestedTags?.length) {
        formData.append('tags', JSON.stringify(uploadContext.suggestedTags));
      }

      const token = localStorage.getItem('access_token');
      const response = await fetch('/api/v1/upload/document', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`
        },
        body: formData
      });

      if (!response.ok) throw new Error('Upload failed');
      const result = await response.json();

      const successMessage: AgentMessage = {
        id: `upload-${Date.now()}`,
        role: 'assistant',
        content: `Successfully uploaded **${file.name}**! The document is now being processed and will be searchable shortly.\n\nDocument ID: \`${result.document_id}\``,
        created_at: new Date().toISOString(),
      };
      setMessages(prev => [...prev, successMessage]);
      setPendingUpload(false);
      setUploadContext(null);
      toast.success('Document uploaded successfully');

    } catch (error: any) {
      console.error('Upload error:', error);
      toast.error('Failed to upload document');

      const errorMessage: AgentMessage = {
        id: `upload-error-${Date.now()}`,
        role: 'assistant',
        content: `Failed to upload **${file.name}**. Please try again or contact support if the issue persists.`,
        created_at: new Date().toISOString(),
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  }, [uploadContext]);

  const confirmDelete = useCallback(async (documentId: string) => {
    if (!pendingDelete) return;

    setIsLoading(true);
    try {
      const token = localStorage.getItem('access_token');
      const response = await fetch(`/api/v1/agent/confirm-delete/${documentId}`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        }
      });

      if (!response.ok) throw new Error('Delete failed');

      const successMessage: AgentMessage = {
        id: `delete-${Date.now()}`,
        role: 'assistant',
        content: `Successfully deleted **${pendingDelete.title}**.`,
        created_at: new Date().toISOString(),
      };
      setMessages(prev => [...prev, successMessage]);
      setPendingDelete(null);
      toast.success('Document deleted');

    } catch (error: any) {
      console.error('Delete error:', error);
      toast.error('Failed to delete document');

      const errorMessage: AgentMessage = {
        id: `delete-error-${Date.now()}`,
        role: 'assistant',
        content: `Failed to delete the document. Please try again.`,
        created_at: new Date().toISOString(),
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  }, [pendingDelete]);

  const clearPendingUpload = useCallback(() => {
    setPendingUpload(false);
    setUploadContext(null);
  }, []);

  const clearPendingDelete = useCallback(() => {
    setPendingDelete(null);
  }, []);

  const handleTemplateUpload = useCallback(async (file: File) => {
    if (!templateUploadContext) return;

    setIsLoading(true);

    try {
      const formData = new FormData();
      formData.append('template', file);
      formData.append('source_document_ids', templateUploadContext.sourceDocumentIds.join(','));

      const token = localStorage.getItem('access_token');
      const response = await fetch('/api/v1/templates/fill', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`
        },
        body: formData
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || 'Template upload failed');
      }

      const result = await response.json();

      // Set active job for progress tracking
      setActiveTemplateJob({
        jobId: result.id,
        templateFilename: result.template_filename,
        status: result.status,
        progress: result.progress || 0,
      });

      const successMessage: AgentMessage = {
        id: `template-${Date.now()}`,
        role: 'assistant',
        content: `Template **${file.name}** uploaded successfully! The template is now being filled with content from your source documents.\n\nJob ID: \`${result.id}\`\n\nI'll notify you when it's ready for download. You can also ask "What's the status of my template job?" to check progress.`,
        created_at: new Date().toISOString(),
      };
      setMessages(prev => [...prev, successMessage]);
      setPendingTemplateUpload(false);
      setTemplateUploadContext(null);
      toast.success('Template job started');

      // Start polling for job status
      pollTemplateJobStatus(result.id);

    } catch (error: any) {
      console.error('Template upload error:', error);
      toast.error(error.message || 'Failed to upload template');

      const errorMessage: AgentMessage = {
        id: `template-error-${Date.now()}`,
        role: 'assistant',
        content: `Failed to upload template **${file.name}**. ${error.message || 'Please try again.'}`,
        created_at: new Date().toISOString(),
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  }, [templateUploadContext]);

  const pollTemplateJobStatus = useCallback(async (jobId: string) => {
    const token = localStorage.getItem('access_token');
    let attempts = 0;
    const maxAttempts = 120; // 10 minutes max (5s intervals)

    const checkStatus = async () => {
      try {
        const response = await fetch(`/api/v1/templates/${jobId}`, {
          headers: {
            'Authorization': `Bearer ${token}`
          }
        });

        if (!response.ok) return;

        const job = await response.json();

        setActiveTemplateJob({
          jobId: job.id,
          templateFilename: job.template_filename,
          status: job.status,
          progress: job.progress || 0,
          currentSection: job.current_section,
          downloadUrl: job.download_url,
          error: job.error_message,
        });

        if (job.status === 'completed') {
          const completedMessage: AgentMessage = {
            id: `template-complete-${Date.now()}`,
            role: 'assistant',
            content: `Your template **${job.template_filename}** has been filled successfully!\n\n[Download filled document](/api/v1/templates/${jobId}/download)`,
            created_at: new Date().toISOString(),
          };
          setMessages(prev => [...prev, completedMessage]);
          toast.success('Template filled successfully!');
          setActiveTemplateJob(null);
          return;
        }

        if (job.status === 'failed') {
          const failedMessage: AgentMessage = {
            id: `template-failed-${Date.now()}`,
            role: 'assistant',
            content: `Template filling failed for **${job.template_filename}**.\n\nError: ${job.error_message || 'Unknown error'}`,
            created_at: new Date().toISOString(),
          };
          setMessages(prev => [...prev, failedMessage]);
          toast.error('Template filling failed');
          setActiveTemplateJob(null);
          return;
        }

        // Continue polling if still processing
        attempts++;
        if (attempts < maxAttempts) {
          setTimeout(checkStatus, 5000);
        }

      } catch (error) {
        console.error('Error polling template status:', error);
      }
    };

    // Start polling after a short delay
    setTimeout(checkStatus, 2000);
  }, []);

  const clearPendingTemplateUpload = useCallback(() => {
    setPendingTemplateUpload(false);
    setTemplateUploadContext(null);
  }, []);

  const clearMessages = useCallback(() => {
    setMessages([{
      id: 'welcome',
      role: 'assistant',
      content: 'Chat cleared! I can search, summarize, upload, delete, compare documents, fill templates, manage tags, and more. What would you like to do?',
      created_at: new Date().toISOString(),
    }]);
    setPendingUpload(false);
    setUploadContext(null);
    setPendingDelete(null);
  }, []);

  // Conversation management methods
  const startNewConversation = useCallback(async () => {
    setIsLoadingConversation(true);
    try {
      const newConversation = await conversationApi.startNew();
      if (newConversation) {
        setConversationId(newConversation.id);
        conversationIdRef.current = newConversation.id;
        setMessages([WELCOME_MESSAGE]);
        setPendingUpload(false);
        setUploadContext(null);
        setPendingDelete(null);
        setPendingTemplateUpload(false);
        setTemplateUploadContext(null);

        // Refresh conversation history
        const history = await conversationApi.listConversations(20);
        setConversationHistory(history);

        toast.success('Started new conversation');
      }
    } catch (error) {
      console.error('Failed to start new conversation:', error);
      toast.error('Failed to start new conversation');
    } finally {
      setIsLoadingConversation(false);
    }
  }, []);

  const loadConversation = useCallback(async (id: string) => {
    setIsLoadingConversation(true);
    try {
      const conversation = await conversationApi.getConversation(id);
      if (conversation) {
        setConversationId(conversation.id);
        conversationIdRef.current = conversation.id;

        if (conversation.messages && conversation.messages.length > 0) {
          const loadedMessages: AgentMessage[] = conversation.messages.map((msg: any) => ({
            id: msg.id,
            role: msg.role,
            content: msg.content,
            tool_calls: msg.tool_calls as AgentToolCall[] | undefined,
            created_at: msg.created_at,
          }));
          setMessages(loadedMessages);
        } else {
          setMessages([WELCOME_MESSAGE]);
        }

        setPendingUpload(false);
        setUploadContext(null);
        setPendingDelete(null);
        setPendingTemplateUpload(false);
        setTemplateUploadContext(null);
      }
    } catch (error) {
      console.error('Failed to load conversation:', error);
      toast.error('Failed to load conversation');
    } finally {
      setIsLoadingConversation(false);
    }
  }, []);

  const refreshConversationHistory = useCallback(async () => {
    try {
      const history = await conversationApi.listConversations(20);
      setConversationHistory(history);
    } catch (error) {
      console.error('Failed to refresh conversation history:', error);
    }
  }, []);

  // Load available agents from the server
  const loadAvailableAgents = useCallback(async () => {
    try {
      const response = await fetch('/api/v1/agent/agents?active_only=true', {
        headers: getAuthHeaders(),
      });
      if (!response.ok) return;
      const data = await response.json();
      const agents: AgentInfo[] = (data.agents || []).map((a: any) => ({
        id: a.id,
        name: a.name,
        displayName: a.display_name,
        description: a.description,
        capabilities: a.capabilities,
      }));
      setAvailableAgents(agents);
    } catch (error) {
      console.error('Failed to load available agents:', error);
    }
  }, []);

  // Auto-connect on mount and load agents
  useEffect(() => {
    connect();
    loadAvailableAgents();
    return () => {
      disconnect();
    };
  }, []);

  return {
    messages,
    isLoading,
    connectionStatus,
    streamingState,
    sendMessage,
    pendingUpload,
    uploadContext,
    handleFileUpload,
    clearPendingUpload,
    clearMessages,
    confirmDelete,
    pendingDelete,
    clearPendingDelete,
    pendingTemplateUpload,
    templateUploadContext,
    handleTemplateUpload,
    clearPendingTemplateUpload,
    activeTemplateJob,
    connect,
    disconnect,
    // Conversation memory
    conversationId,
    conversationHistory,
    isLoadingConversation,
    startNewConversation,
    loadConversation,
    refreshConversationHistory,
    // Multi-agent support
    currentAgent,
    injectedMemoriesCount,
    lastRoutingReason,
    availableAgents,
    loadAvailableAgents,
  };
}

function formatToolName(name: string): string {
  return name.split('_').map(word =>
    word.charAt(0).toUpperCase() + word.slice(1)
  ).join(' ');
}
