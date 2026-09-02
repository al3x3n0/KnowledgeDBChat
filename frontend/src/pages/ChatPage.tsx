/**
 * Chat page with session management and WebSocket communication
 */

import React, { useState, useEffect, useRef } from 'react';
import { useParams, useNavigate, useLocation } from 'react-router-dom';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import {
  Plus,
  Send,
  Trash2,
  MessageCircle,
  Bot,
  User,
  ExternalLink,
  ThumbsUp,
  ThumbsDown,
  Clock,
  Download,
  Loader2,
  Eye,
  Sparkles,
  Network,
  ChevronDown,
  ChevronUp,
  X
} from 'lucide-react';
import { formatDistanceToNow } from 'date-fns';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

import { apiClient } from '../services/api';
import { ChatMessage, RetrievalTrace, WebSocketMessage } from '../types';
import { useAuth } from '../contexts/AuthContext';
import Button from '../components/common/Button';
import Input from '../components/common/Input';
import LoadingSpinner from '../components/common/LoadingSpinner';
import ConfirmationModal from '../components/common/ConfirmationModal';
import { useKeyboardShortcuts } from '../hooks/useKeyboardShortcuts';
import toast from 'react-hot-toast';

const ChatPage: React.FC = () => {
  const { sessionId } = useParams<{ sessionId: string }>();
  const navigate = useNavigate();
  const location = useLocation();
  const { user } = useAuth();
  const queryClient = useQueryClient();
  
  const [message, setMessage] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [wsConnection, setWsConnection] = useState<WebSocket | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const [deleteConfirmOpen, setDeleteConfirmOpen] = useState(false);
  const [sessionToDelete, setSessionToDelete] = useState<string | null>(null);

  // Allow other pages (e.g., Context Pack) to prefill the input via navigation state.
  useEffect(() => {
    const st = (location as any)?.state as { prefillMessage?: string } | undefined;
    const prefill = (st?.prefillMessage || '').toString();
    if (!prefill) return;
    setMessage((prev) => (prev && prev.trim().length ? prev : prefill));
    // Clear state to avoid re-applying on re-render/back-forward.
    // Note: no reliable way to clear location.state without a navigation. We just
    // apply once on mount.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Fetch chat sessions
  const { data: sessions, isLoading: sessionsLoading, error: sessionsError } = useQuery(
    'chatSessions',
    () => apiClient.getChatSessions(),
    {
      enabled: !!user, // Only fetch if user is authenticated
      refetchOnWindowFocus: false,
      refetchInterval: 10000, // Refetch every 10 seconds to catch title updates
      retry: 2,
      onError: () => {
        toast.error('Failed to load chat sessions');
      },
    }
  );

  // Fetch current session messages
  const { data: currentSession, isLoading: sessionLoading } = useQuery(
    ['chatSession', sessionId],
    () => sessionId ? apiClient.getChatSession(sessionId) : null,
    {
      enabled: !!sessionId,
      refetchOnWindowFocus: false,
    }
  );

  const updateSessionMutation = useMutation(
    ({
      sessionId,
      payload,
    }: {
      sessionId: string;
      payload: { title?: string | null; extra_metadata?: Record<string, any> | null };
    }) => apiClient.updateChatSession(sessionId, payload),
    {
      onSuccess: () => {
        queryClient.invalidateQueries(['chatSession', sessionId]);
        queryClient.invalidateQueries('chatSessions');
      },
      onError: (error: any) => {
        const errorMessage =
          error?.response?.data?.detail || error?.message || 'Failed to update chat session';
        toast.error(errorMessage);
      },
    }
  );

  // Create new session mutation
  const createSessionMutation = useMutation(
    () => apiClient.createChatSession(),
    {
      onSuccess: (newSession) => {
        queryClient.invalidateQueries('chatSessions');
        navigate(`/chat/${newSession.id}`);
      },
    }
  );

  // Delete session mutation
  const deleteSessionMutation = useMutation(
    (sessionId: string) => apiClient.deleteChatSession(sessionId),
    {
      onSuccess: () => {
        queryClient.invalidateQueries('chatSessions');
        // If we deleted the current session, navigate away
        if (sessionToDelete === sessionId) {
          navigate('/chat');
        }
        setDeleteConfirmOpen(false);
        setSessionToDelete(null);
        toast.success('Chat session deleted');
      },
      onError: (error: any) => {
        const errorMessage = error?.response?.data?.detail || error?.message || 'Failed to delete chat session';
        toast.error(errorMessage);
        setDeleteConfirmOpen(false);
        setSessionToDelete(null);
      },
    }
  );

  // Send message mutation
  const sendMessageMutation = useMutation(
    ({ sessionId, content }: { sessionId: string; content: string }) =>
      apiClient.sendMessage(sessionId, content),
    {
      onSuccess: () => {
        queryClient.invalidateQueries(['chatSession', sessionId]);
      },
    }
  );

  // Keyboard shortcuts
  useKeyboardShortcuts([
    {
      key: 'k',
      ctrlKey: true,
      handler: () => {
        inputRef.current?.focus();
      },
      description: 'Focus message input',
    },
    {
      key: 'n',
      ctrlKey: true,
      handler: () => {
        if (!createSessionMutation.isLoading) {
          handleCreateSession();
        }
      },
      description: 'Create new chat session',
    },
        {
          key: '/',
          ctrlKey: true,
          handler: () => {
            // Show shortcuts help (can be implemented later)
            toast('Keyboard shortcuts: Ctrl+K (focus input), Ctrl+N (new chat), Ctrl+/ (help)', {
              icon: 'ℹ️',
              duration: 4000,
            });
          },
          description: 'Show keyboard shortcuts',
        },
  ]);

  // WebSocket connection
  useEffect(() => {
    if (!sessionId) return;

    const ws = apiClient.createWebSocket(sessionId);
    
    ws.onopen = () => {
      setWsConnection(ws);
    };

    ws.onmessage = (event) => {
      try {
        const data: WebSocketMessage = JSON.parse(event.data);

        if (data.type === 'typing') {
          setIsTyping(true);
        } else if (data.type === 'message') {
          setIsTyping(false);
          queryClient.invalidateQueries(['chatSession', sessionId]);
        } else if (data.type === 'error') {
          setIsTyping(false);
          toast.error(data.message || 'An error occurred');
        }
      } catch {
        // Ignore malformed messages
      }
    };

    ws.onclose = () => {
      setWsConnection(null);
      setIsTyping(false);
    };

    ws.onerror = () => {
      setIsTyping(false);
    };

    return () => {
      ws.close();
    };
  }, [sessionId, queryClient]);

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [currentSession?.messages, isTyping]);

  // Focus input on session change
  useEffect(() => {
    inputRef.current?.focus();
  }, [sessionId]);

  const handleSendMessage = async (e?: React.FormEvent) => {
    e?.preventDefault();
    
    if (!message.trim() || !sessionId) return;

    const messageContent = message.trim();
    setMessage('');

    try {
      if (wsConnection && wsConnection.readyState === WebSocket.OPEN) {
        // Send via WebSocket for real-time response
        wsConnection.send(JSON.stringify({ message: messageContent }));
      } else {
        // Fallback to HTTP API
        await sendMessageMutation.mutateAsync({ sessionId, content: messageContent });
      }
    } catch {
      toast.error('Failed to send message');
    }
  };

  // Handle Enter key in message input (Shift+Enter for new line)
  const handleInputKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const handleCreateSession = () => {
    createSessionMutation.mutate();
  };

  const handleDeleteSession = (sessionToDeleteId: string) => {
    setSessionToDelete(sessionToDeleteId);
    setDeleteConfirmOpen(true);
  };

  const confirmDeleteSession = () => {
    if (sessionToDelete) {
      deleteSessionMutation.mutate(sessionToDelete);
    } else {
      toast.error('No session selected for deletion');
    }
  };

  const handleFeedback = async (messageId: string, rating: number) => {
    try {
      await apiClient.submitMessageFeedback(messageId, rating);
      queryClient.invalidateQueries(['chatSession', sessionId]);
      toast.success('Feedback submitted');
    } catch (error) {
      toast.error('Failed to submit feedback');
    }
  };

  const pinned = (() => {
    const meta = (currentSession as any)?.extra_metadata || {};
    const chatModel = meta?.llm_task_models?.chat;
    const chatProvider = meta?.llm_task_providers?.chat;
    const adapterId = meta?.ai_hub?.adapter_id;
    const parts: string[] = [];
    if (chatProvider) parts.push(String(chatProvider));
    if (chatModel) parts.push(String(chatModel));
    if (!chatModel && adapterId) parts.push(`adapter:${String(adapterId).slice(0, 8)}`);
    return { isPinned: Boolean(chatModel || adapterId), label: parts.join(' • '), meta };
  })();

  const clearPinned = () => {
    if (!sessionId || !pinned.isPinned) return;
    const next = { ...(pinned.meta || {}) };
    delete (next as any).ai_hub;
    delete (next as any).llm_task_models;
    delete (next as any).llm_task_providers;
    delete (next as any).llm_provider;
    delete (next as any).llm_model;
    updateSessionMutation.mutate({ sessionId, payload: { extra_metadata: next } });
  };

  if (sessionsLoading) {
    return <LoadingSpinner className="h-full" text="Loading chat sessions..." />;
  }

  // Error already handled by onError callback

  return (
    <div className="flex h-full">
      {/* Sidebar - Chat Sessions */}
      <div className="w-80 bg-white border-r border-gray-200 flex flex-col h-full">
        {/* Header */}
        <div className="p-4 border-b border-gray-200 flex-shrink-0">
          <Button
            onClick={handleCreateSession}
            fullWidth
            icon={<Plus className="w-4 h-4" />}
            loading={createSessionMutation.isLoading}
          >
            New Chat
          </Button>
        </div>

        {/* Sessions List */}
        <div className="flex-1 overflow-y-auto min-h-0 scroll-smooth scrollbar-thin">
          {sessionsError ? (
            <div className="p-4 text-center text-red-500">
              <p>Error loading sessions</p>
              <p className="text-xs mt-1">Please refresh the page</p>
            </div>
          ) : sessions?.length === 0 ? (
            <div className="p-4 text-center text-gray-500">
              <MessageCircle className="w-12 h-12 mx-auto mb-2 text-gray-300" />
              <p>No chat sessions yet</p>
              <p className="text-sm">Start a new conversation!</p>
            </div>
          ) : (
            <div className="space-y-2 p-2">
              {sessions?.map((session) => (
                <div
                  key={session.id}
                  className={`group p-3 rounded-lg cursor-pointer transition-colors duration-200 ${
                    session.id === sessionId
                      ? 'bg-primary-50 border border-primary-200'
                      : 'hover:bg-gray-50'
                  }`}
                  onClick={() => navigate(`/chat/${session.id}`)}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1 min-w-0">
                      <h3 className="text-sm font-medium text-gray-900 truncate">
                        {session.title || 'Untitled Chat'}
                      </h3>
                      <p className="text-xs text-gray-500 mt-1">
                        {session.last_message_at 
                          ? `${formatDistanceToNow(new Date(session.last_message_at))} ago`
                          : 'Just now'}
                      </p>
                    </div>
                    <button
                      type="button"
                      className="opacity-0 group-hover:opacity-100 p-1 hover:bg-gray-200 rounded transition-opacity flex-shrink-0"
                      onClick={(e) => {
                        e.stopPropagation();
                        e.preventDefault();
                        handleDeleteSession(session.id);
                      }}
                      title="Delete session"
                      aria-label="Delete session"
                    >
                      <Trash2 className="w-4 h-4 text-gray-500" />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col">
        {sessionId ? (
          <>
            {/* Chat Header */}
            <div className="bg-white border-b border-gray-200 p-4">
              <div className="flex items-center justify-between">
                <div>
                  <h1 className="text-lg font-semibold text-gray-900">
                    {currentSession?.title || 'Chat Session'}
                  </h1>
                  {pinned.isPinned && (
                    <div className="mt-2 inline-flex items-center gap-2 text-xs bg-indigo-50 text-indigo-700 border border-indigo-100 rounded-full px-3 py-1">
                      <Sparkles className="w-3 h-3" />
                      <span className="font-medium">AI Hub pinned</span>
                      <span className="text-indigo-600">{pinned.label}</span>
                      <button
                        className="ml-1 text-indigo-600 hover:text-indigo-800"
                        onClick={clearPinned}
                        disabled={updateSessionMutation.isLoading}
                        title="Clear pinned AI Hub model for this session"
                        aria-label="Clear pinned model"
                      >
                        <X className="w-3 h-3" />
                      </button>
                    </div>
                  )}
                </div>
                <div className="flex items-center space-x-2">
                  {wsConnection && (
                    <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800">
                      <div className="w-2 h-2 bg-green-400 rounded-full mr-1"></div>
                      Connected
                    </span>
                  )}
                </div>
              </div>
            </div>

            {/* Messages */}
            <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-gray-50">
              {sessionLoading ? (
                <LoadingSpinner className="h-32" text="Loading messages..." />
              ) : (
                <>
                  {currentSession?.messages?.map((msg, index) => (
                    <ChatMessageComponent
                      key={msg.id}
                      message={msg}
                      question={
                        msg.role === 'assistant'
                          ? currentSession?.messages
                              ?.slice(0, index)
                              .reverse()
                              .find((m) => m.role === 'user')?.content
                          : undefined
                      }
                      onFeedback={(rating) => handleFeedback(msg.id, rating)}
                    />
                  ))}
                  
                  {isTyping && (
                    <div className="flex items-start space-x-3">
                      <div className="flex-shrink-0">
                        <div className="w-8 h-8 bg-primary-600 rounded-full flex items-center justify-center">
                          <Bot className="w-4 h-4 text-white" />
                        </div>
                      </div>
                      <div className="bg-white rounded-lg p-3 shadow-sm max-w-xs">
                        <div className="loading-dots">
                          <div></div>
                          <div></div>
                          <div></div>
                        </div>
                      </div>
                    </div>
                  )}
                  
                  <div ref={messagesEndRef} />
                </>
              )}
            </div>

            {/* Message Input */}
            <div className="bg-white border-t border-gray-200 p-4">
              <form onSubmit={handleSendMessage} className="flex space-x-2">
                <Input
                  ref={inputRef}
                  value={message}
                  onChange={(e) => setMessage(e.target.value)}
                  onKeyDown={handleInputKeyDown}
                  placeholder="Ask me anything about your documents... (Ctrl+K to focus, Enter to send)"
                  className="flex-1"
                  disabled={sendMessageMutation.isLoading}
                />
                <Button
                  type="submit"
                  disabled={!message.trim() || sendMessageMutation.isLoading}
                  icon={<Send className="w-4 h-4" />}
                >
                  Send
                </Button>
              </form>
            </div>
          </>
        ) : (
          // No session selected
          <div className="flex-1 flex items-center justify-center bg-gray-50">
            <div className="text-center">
              <MessageCircle className="w-24 h-24 mx-auto mb-4 text-gray-300" />
              <h2 className="text-xl font-semibold text-gray-900 mb-2">
                Welcome to Knowledge Database Chat
              </h2>
              <p className="text-gray-600 mb-6 max-w-md">
                Start a new conversation to search and get answers from your organizational knowledge base.
              </p>
              <Button
                onClick={handleCreateSession}
                icon={<Plus className="w-4 h-4" />}
                loading={createSessionMutation.isLoading}
              >
                Start New Chat
              </Button>
            </div>
          </div>
        )}
      </div>

      {/* Delete Confirmation Modal */}
      <ConfirmationModal
        isOpen={deleteConfirmOpen}
        onClose={() => {
          setDeleteConfirmOpen(false);
          setSessionToDelete(null);
        }}
        onConfirm={confirmDeleteSession}
        title="Delete Chat Session"
        message="Are you sure you want to delete this chat session? This action cannot be undone."
        confirmText="Delete"
        cancelText="Cancel"
        variant="danger"
        isLoading={deleteSessionMutation.isLoading}
      />
    </div>
  );
};

// Chat Message Component
interface ChatMessageProps {
  message: ChatMessage;
  /** The question this answer replied to, so an answer can become a run's goal. */
  question?: string;
  onFeedback: (rating: number) => void;
}

const ChatMessageComponent: React.FC<ChatMessageProps> = ({ message, question, onFeedback }) => {
  const navigate = useNavigate();
  const isUser = message.role === 'user';
  const isAssistant = message.role === 'assistant';
  const [downloadingDocs, setDownloadingDocs] = React.useState<Set<string>>(new Set());
  const [kgOpen, setKgOpen] = React.useState(false);
  const [startingRun, setStartingRun] = React.useState(false);
  const [kgLoading, setKgLoading] = React.useState(false);
  const [kgTrace, setKgTrace] = React.useState<RetrievalTrace | null>(null);
  const [kgError, setKgError] = React.useState<string | null>(null);
  const sourceDocs = (message.source_documents || []) as Array<{
    id: string;
    title?: string;
    score?: number;
    source?: string;
    chunk_id?: string;
    snippet?: string;
    download_url?: string;
    url?: string;
  }>;

  const citationNumbers = React.useMemo(() => {
    if (!isAssistant || !message.content) return [];
    const matches = message.content.match(/\[(\d+)\]/g) || [];
    const nums = matches
      .map((m) => parseInt(m.replace(/\[|\]/g, ''), 10))
      .filter((n) => Number.isFinite(n) && n >= 1 && n <= sourceDocs.length);
    return Array.from(new Set(nums)).sort((a, b) => a - b);
  }, [isAssistant, message.content, sourceDocs.length]);

  const handleViewSource = (docId: string, chunkId?: string) => {
    navigate('/documents', {
      state: {
        openDocId: docId,
        highlightChunkId: chunkId
      }
    });
  };

  const handleOpenGlobalKG = (name: string, id?: string) => {
    const params = new URLSearchParams();
    if (name) params.set('search', name);
    if (id) params.set('sel', id);
    navigate(`/kg/global?${params.toString()}`);
  };

  const handleOpenDocumentGraph = (docId: string) => {
    navigate(`/documents/${encodeURIComponent(docId)}/graph`);
  };

  const handleDownload = async (docId: string, downloadUrl?: string) => {
    try {
      setDownloadingDocs(prev => new Set(prev).add(docId));

      // Use the API client method to download as blob
      const { blob, filename } = await apiClient.downloadDocument(docId, true);

      // Create download link and trigger download
      const url = window.URL.createObjectURL(blob);
      const link = window.document.createElement('a');
      link.href = url;
      link.download = filename;
      window.document.body.appendChild(link);
      link.click();
      window.document.body.removeChild(link);
      window.URL.revokeObjectURL(url);

      toast.success('Download started');
    } catch (error: any) {
      const errorMessage = error.response?.data
        ? (error.response.data instanceof Blob
            ? 'Download failed: Server error'
            : error.response.data.detail || error.response.data.message || 'Download failed')
        : error.message || 'Failed to download document. Please try again.';
      toast.error(errorMessage);
    } finally {
      setDownloadingDocs(prev => {
        const next = new Set(prev);
        next.delete(docId);
        return next;
      });
    }
  };

  const toggleKG = async () => {
    const next = !kgOpen;
    setKgOpen(next);
    if (!next) return;
    if (!isAssistant || !message.retrieval_trace_id) return;
    if (kgTrace || kgLoading) return;

    try {
      setKgError(null);
      setKgLoading(true);
      const trace = await apiClient.getRetrievalTrace(message.retrieval_trace_id);
      setKgTrace(trace);
    } catch (e: any) {
      setKgError(e?.message || 'Failed to load retrieval trace');
    } finally {
      setKgLoading(false);
    }
  };

  const kgPack = React.useMemo(() => {
    const t: any = kgTrace?.trace;
    return t?.kg_context_pack || null;
  }, [kgTrace]);

  return (
    <div className={`flex items-start space-x-3 ${isUser ? 'flex-row-reverse space-x-reverse' : ''}`}>
      {/* Avatar */}
      <div className="flex-shrink-0">
        <div className={`w-8 h-8 rounded-full flex items-center justify-center ${
          isUser ? 'bg-gray-600' : 'bg-primary-600'
        }`}>
          {isUser ? (
            <User className="w-4 h-4 text-white" />
          ) : (
            <Bot className="w-4 h-4 text-white" />
          )}
        </div>
      </div>

      {/* Message Content */}
      <div className={`flex-1 max-w-3xl ${isUser ? 'text-right' : ''}`}>
        <div className={`inline-block rounded-lg p-3 shadow-sm ${
          isUser 
            ? 'bg-primary-600 text-white' 
            : 'bg-gray-100 text-gray-900 border border-gray-200 shadow-none'
        }`}>
          {isAssistant ? (
            <ReactMarkdown 
              remarkPlugins={[remarkGfm]}
              className="prose prose-sm max-w-none"
            >
              {message.content}
            </ReactMarkdown>
          ) : (
            <p className="text-sm">{message.content}</p>
          )}
        </div>

        {/* Message metadata */}
        <div className={`mt-1 text-xs text-gray-500 ${isUser ? 'text-right' : ''}`}>
          <span>{formatDistanceToNow(new Date(message.created_at))} ago</span>
          {message.response_time && (
            <span className="ml-2">
              <Clock className="w-3 h-3 inline mr-1" />
              {message.response_time.toFixed(1)}s
            </span>
          )}
        </div>

        {/* Inline citations */}
        {isAssistant && citationNumbers.length > 0 && (
          <div className="mt-2 flex items-center gap-2 flex-wrap">
            <span className="text-xs text-gray-600">Citations:</span>
            {citationNumbers.map((n) => {
              const doc = sourceDocs[n - 1];
              const canOpen = Boolean(doc?.id);
              return (
                <button
                  key={n}
                  type="button"
                  disabled={!canOpen}
                  className="text-xs px-2 py-1 rounded border border-gray-300 hover:bg-gray-50 disabled:opacity-50"
                  title={doc?.title || `Source ${n}`}
                  onClick={() => {
                    if (doc?.id) handleViewSource(doc.id, doc.chunk_id);
                  }}
                >
                  [{n}]
                </button>
              );
            })}
          </div>
        )}

        {/* Measure it. The corpus answered as far as it can; the next move is
            a run, and it should start from what was just established rather
            than rediscovering it. */}
        {isAssistant && question && sourceDocs.length > 0 && (
          <div className="mt-3 pt-3 border-t border-gray-200">
            <button
              type="button"
              disabled={startingRun}
              className="text-xs px-3 py-1.5 rounded-md border border-primary-500 text-primary-700 hover:bg-gray-100 disabled:opacity-50"
              onClick={async () => {
                setStartingRun(true);
                try {
                  const sources = sourceDocs
                    .map((doc, i) => `[${i + 1}] ${doc.title || 'Untitled'}`)
                    .join('\n');
                  const job = await apiClient.createAgentJob({
                    name: question.slice(0, 120),
                    goal: question,
                    job_type: 'research',
                    config: {
                      starting_context: `A search of the knowledge base answered:\n\n${message.content}\n\nIt read:\n${sources}`,
                      // Kept so a run can be traced back to the question that
                      // started it.
                      origin: { kind: 'chat', message_id: message.id },
                    },
                  } as any);
                  toast.success('Run started from this answer');
                  navigate(`/autonomous-agents?job=${job.id}`);
                } catch {
                  toast.error('Could not start a run');
                } finally {
                  setStartingRun(false);
                }
              }}
            >
              {startingRun ? 'Starting…' : 'Measure this'}
            </button>
            <p className="text-xs text-gray-500 mt-1">
              Starts a run with this question as its goal, carrying this answer and its
              sources as context.
            </p>
          </div>
        )}

        {/* Source documents */}
        {sourceDocs.length > 0 && (
          <div className="mt-2 space-y-1">
            <p className="text-xs font-medium text-gray-700">Sources:</p>
            {sourceDocs.map((doc, index) => (
              <div key={index} className="text-xs bg-gray-100 rounded p-2 hover:bg-gray-200 transition-colors">
                <div className="flex items-center justify-between">
                  <span className="font-medium truncate">Source {index + 1}: {doc.title}</span>
                  <span className="text-gray-500 ml-2">
                    {(((doc.score ?? 0) as number) * 100).toFixed(0)}% match
                  </span>
                </div>
                {doc.snippet && (
                  <div className="text-gray-600 mt-1 text-xs italic line-clamp-2">
                    "{doc.snippet}..."
                  </div>
                )}
                <div className="text-gray-600 mt-1 flex items-center gap-2 flex-wrap">
                  <span className="truncate">{doc.source}</span>
                  <div className="flex items-center gap-1 ml-auto">
                    {doc.id && doc.chunk_id && (
                      <button
                        onClick={() => handleViewSource(doc.id, doc.chunk_id)}
                        className="text-primary-600 hover:text-primary-800 p-1 rounded hover:bg-primary-50 transition-colors"
                        title="View in document"
                      >
                        <Eye className="w-3 h-3" />
                      </button>
                    )}
                    {doc.url && (
                      <button
                        onClick={() => window.open(doc.url, '_blank', 'noopener,noreferrer')}
                        className="text-primary-600 hover:text-primary-800 p-1 rounded hover:bg-primary-50 transition-colors"
                        title="Open source"
                      >
                        <ExternalLink className="w-3 h-3" />
                      </button>
                    )}
                    {doc.id && (
                      <button
                        onClick={() => handleDownload(doc.id)}
                        disabled={downloadingDocs.has(doc.id)}
                        className="text-primary-600 hover:text-primary-800 p-1 rounded hover:bg-primary-50 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                        title="Download document"
                      >
                        {downloadingDocs.has(doc.id) ? (
                          <Loader2 className="w-3 h-3 animate-spin" />
                        ) : (
                          <Download className="w-3 h-3" />
                        )}
                      </button>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* KG context pack (debug/inspection) */}
        {isAssistant && message.retrieval_trace_id && (
          <div className="mt-2">
            <button
              type="button"
              onClick={toggleKG}
              className="text-xs px-2 py-1 rounded border border-gray-300 hover:bg-gray-50 inline-flex items-center gap-1"
              title="View KG context pack grounded in retrieved sources"
            >
              <span>KG context pack</span>
              {kgOpen ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
            </button>

            {kgOpen && (
              <div className="mt-2 text-xs bg-gray-50 border border-gray-200 rounded p-2">
                {kgLoading && <div className="text-gray-600">Loading…</div>}
                {!kgLoading && kgError && <div className="text-red-600">{kgError}</div>}
                {!kgLoading && !kgError && !kgPack && (
                  <div className="text-gray-600">No KG context pack in trace.</div>
                )}
                {!kgLoading && !kgError && kgPack && (
                  <div className="space-y-2">
                    <div className="flex items-center justify-between gap-2">
                      <div className="text-gray-700">
                        <span className="font-medium">Stats:</span>{' '}
                        {JSON.stringify(kgPack.stats || {})}
                      </div>
                      <div className="flex items-center gap-2">
                        <button
                          type="button"
                          className="text-xs px-2 py-1 rounded border border-gray-300 hover:bg-gray-50"
                          onClick={() => {
                            if (message.retrieval_trace_id) navigate(`/context-packs/${encodeURIComponent(String(message.retrieval_trace_id))}`);
                          }}
                          title="Open full context pack page"
                        >
                          Open
                        </button>
                        <button
                          type="button"
                          className="text-xs px-2 py-1 rounded border border-gray-300 hover:bg-gray-50"
                          onClick={async () => {
                            try {
                              const ctx = (kgTrace?.trace as any)?.kg_context_pack?.kg_context;
                              const text = typeof ctx === 'string' ? ctx : JSON.stringify(kgPack, null, 2);
                              await navigator.clipboard.writeText(text);
                              toast.success('Copied KG context');
                            } catch {
                              toast.error('Copy failed');
                            }
                          }}
                          title="Copy KG context to clipboard"
                        >
                          Copy
                        </button>
                      </div>
                    </div>

                    <div>
                      <div className="font-medium text-gray-700">Entities</div>
                      <div className="mt-1 space-y-1">
                        {(kgPack.entities || []).slice(0, 20).map((e: any) => (
                          <div key={String(e.id)} className="bg-white border border-gray-200 rounded px-2 py-1">
                            <div className="flex items-center justify-between gap-2">
                              <div className="truncate">
                                <button
                                  type="button"
                                  className="font-medium text-primary-700 hover:text-primary-900 hover:underline"
                                  onClick={() => handleOpenGlobalKG(String(e.name || ''), String(e.id || ''))}
                                  title="Open in Global KG"
                                >
                                  {e.name}
                                </button>{' '}
                                <span className="text-gray-500">({e.type})</span>
                              </div>
                              {(typeof e.mention_count === 'number' || typeof e.document_count === 'number') && (
                                <div className="text-gray-500 shrink-0">
                                  {typeof e.mention_count === 'number' ? `${e.mention_count} mentions` : ''}
                                  {typeof e.document_count === 'number' ? ` · ${e.document_count} docs` : ''}
                                </div>
                              )}
                            </div>
                            {Array.isArray(e.evidence) && e.evidence.length > 0 && (
                              <div className="mt-1 space-y-1">
                                {e.evidence.slice(0, 2).map((ev: any, idx: number) => (
                                  <div key={idx} className="flex items-start gap-2">
                                    <div className="text-gray-600 line-clamp-2 flex-1">
                                      evidence: {String(ev?.text || '')}
                                    </div>
                                    {ev?.document_id && (
                                      <div className="flex items-center gap-1 shrink-0">
                                        <button
                                          type="button"
                                          className="text-primary-700 hover:text-primary-900 p-1 rounded hover:bg-primary-50"
                                          title="Open document at evidence"
                                          onClick={() => handleViewSource(String(ev.document_id), ev?.chunk_id ? String(ev.chunk_id) : undefined)}
                                        >
                                          <Eye className="w-3 h-3" />
                                        </button>
                                        <button
                                          type="button"
                                          className="text-primary-700 hover:text-primary-900 p-1 rounded hover:bg-primary-50"
                                          title="Open document knowledge graph"
                                          onClick={() => handleOpenDocumentGraph(String(ev.document_id))}
                                        >
                                          <Network className="w-3 h-3" />
                                        </button>
                                      </div>
                                    )}
                                  </div>
                                ))}
                              </div>
                            )}
                          </div>
                        ))}
                      </div>
                    </div>

                    <div>
                      <div className="font-medium text-gray-700">Relationships</div>
                      <div className="mt-1 space-y-1">
                        {(kgPack.relationships || []).slice(0, 40).map((r: any) => (
                          <div key={String(r.id)} className="bg-white border border-gray-200 rounded px-2 py-1">
                            <div className="text-gray-800">
                              <button
                                type="button"
                                className="font-mono text-primary-700 hover:text-primary-900 hover:underline"
                                title="Open source in Global KG"
                                onClick={() => {
                                  const src = (kgPack.entities || []).find((e: any) => String(e.id) === String(r.source));
                                  handleOpenGlobalKG(String(src?.name || ''), String(r.source || ''));
                                }}
                              >
                                {String(
                                  (kgPack.entities || []).find((e: any) => String(e.id) === String(r.source))?.name ||
                                    String(r.source).slice(0, 8)
                                )}
                              </button>{' '}
                              <span className="text-gray-500">--[{r.type}]--&gt;</span>{' '}
                              <button
                                type="button"
                                className="font-mono text-primary-700 hover:text-primary-900 hover:underline"
                                title="Open target in Global KG"
                                onClick={() => {
                                  const tgt = (kgPack.entities || []).find((e: any) => String(e.id) === String(r.target));
                                  handleOpenGlobalKG(String(tgt?.name || ''), String(r.target || ''));
                                }}
                              >
                                {String(
                                  (kgPack.entities || []).find((e: any) => String(e.id) === String(r.target))?.name ||
                                    String(r.target).slice(0, 8)
                                )}
                              </button>{' '}
                              {typeof r.confidence === 'number' && (
                                <span className="text-gray-500">({(r.confidence * 100).toFixed(0)}%)</span>
                              )}
                            </div>
                            {r.evidence && (
                              <div className="mt-1 text-gray-600 line-clamp-2">
                                evidence: {String(r.evidence)}
                              </div>
                            )}
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* Feedback buttons for assistant messages */}
        {isAssistant && (
          <div className="mt-2 flex items-center space-x-2">
            <span className="text-xs text-gray-500">Was this helpful?</span>
            <button
              onClick={() => onFeedback(5)}
              className="p-1 hover:bg-gray-100 rounded"
              title="Thumbs up"
            >
              <ThumbsUp className="w-3 h-3 text-gray-400 hover:text-green-500" />
            </button>
            <button
              onClick={() => onFeedback(1)}
              className="p-1 hover:bg-gray-100 rounded"
              title="Thumbs down"
            >
              <ThumbsDown className="w-3 h-3 text-gray-400 hover:text-red-500" />
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default ChatPage;
