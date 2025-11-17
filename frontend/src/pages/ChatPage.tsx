/**
 * Chat page with session management and WebSocket communication
 */

import React, { useState, useEffect, useRef } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import { 
  Plus, 
  Send, 
  MoreVertical, 
  Trash2, 
  MessageCircle,
  Bot,
  User,
  ExternalLink,
  ThumbsUp,
  ThumbsDown,
  Clock,
  Download,
  Loader2
} from 'lucide-react';
import { formatDistanceToNow } from 'date-fns';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

import { apiClient } from '../services/api';
import { ChatSession, ChatMessage, WebSocketMessage } from '../types';
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
  const { user } = useAuth();
  const queryClient = useQueryClient();
  
  const [message, setMessage] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [wsConnection, setWsConnection] = useState<WebSocket | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const [deleteConfirmOpen, setDeleteConfirmOpen] = useState(false);
  const [sessionToDelete, setSessionToDelete] = useState<string | null>(null);

  // Fetch chat sessions
  const { data: sessions, isLoading: sessionsLoading, error: sessionsError } = useQuery(
    'chatSessions',
    async () => {
      console.log('Fetching chat sessions...');
      try {
        const result = await apiClient.getChatSessions();
        console.log('Chat sessions fetched:', result);
        return result;
      } catch (error) {
        console.error('Error in getChatSessions:', error);
        throw error;
      }
    },
    {
      enabled: !!user, // Only fetch if user is authenticated
      refetchOnWindowFocus: false,
      retry: 2,
      onError: (error: any) => {
        console.error('Error fetching chat sessions:', error);
        console.error('Error details:', {
          message: error?.message,
          response: error?.response?.data,
          status: error?.response?.status,
        });
        toast.error('Failed to load chat sessions');
      },
      onSuccess: (data) => {
        console.log('Chat sessions loaded successfully:', data?.length || 0, 'sessions');
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
    apiClient.deleteChatSession,
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
        console.error('Error deleting session:', error);
        const errorMessage = error?.response?.data?.detail || error?.message || 'Failed to delete chat session';
        console.error('Delete error details:', {
          status: error?.response?.status,
          data: error?.response?.data,
          message: errorMessage,
        });
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
      console.log('WebSocket connected');
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
      } catch (error) {
        console.error('WebSocket message parsing error:', error);
      }
    };

    ws.onclose = () => {
      console.log('WebSocket disconnected');
      setWsConnection(null);
      setIsTyping(false);
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
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
    } catch (error) {
      console.error('Error sending message:', error);
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
    console.log('Delete session clicked:', sessionToDeleteId);
    setSessionToDelete(sessionToDeleteId);
    setDeleteConfirmOpen(true);
  };

  const confirmDeleteSession = () => {
    console.log('Confirm delete called, sessionToDelete:', sessionToDelete);
    if (sessionToDelete) {
      console.log('Calling delete mutation for:', sessionToDelete);
      deleteSessionMutation.mutate(sessionToDelete);
    } else {
      console.error('No session to delete!');
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

  if (sessionsLoading) {
    return <LoadingSpinner className="h-full" text="Loading chat sessions..." />;
  }

  if (sessionsError) {
    console.error('Sessions error:', sessionsError);
  }

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
        <div className="flex-1 overflow-y-auto min-h-0">
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
                <h1 className="text-lg font-semibold text-gray-900">
                  {currentSession?.title || 'Chat Session'}
                </h1>
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
                  {currentSession?.messages?.map((msg) => (
                    <ChatMessageComponent
                      key={msg.id}
                      message={msg}
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
  onFeedback: (rating: number) => void;
}

const ChatMessageComponent: React.FC<ChatMessageProps> = ({ message, onFeedback }) => {
  const isUser = message.role === 'user';
  const isAssistant = message.role === 'assistant';
  const [downloadingDocs, setDownloadingDocs] = React.useState<Set<string>>(new Set());

  const handleDownload = async (docId: string, downloadUrl?: string) => {
    try {
      setDownloadingDocs(prev => new Set(prev).add(docId));
      
      // Use backend proxy endpoint - streams file through backend
      // This avoids presigned URL signature issues
      console.log('Starting download for document:', docId);
      
      // Use the API client method to download as blob
      const { blob, filename } = await apiClient.downloadDocument(docId, true);
      
      console.log('Download successful, filename:', filename);
      
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
      console.error('Error downloading document:', error);
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
            : 'bg-white text-gray-900'
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

        {/* Source documents */}
        {message.source_documents && message.source_documents.length > 0 && (
          <div className="mt-2 space-y-1">
            <p className="text-xs font-medium text-gray-700">Sources:</p>
            {message.source_documents.map((doc, index) => (
              <div key={index} className="text-xs bg-gray-100 rounded p-2 hover:bg-gray-200 transition-colors">
                <div className="flex items-center justify-between">
                  <span className="font-medium truncate">{doc.title}</span>
                  <span className="text-gray-500 ml-2">
                    {(doc.score * 100).toFixed(0)}% match
                  </span>
                </div>
                <div className="text-gray-600 mt-1 flex items-center gap-2 flex-wrap">
                  <span className="truncate">{doc.source}</span>
                  <div className="flex items-center gap-1 ml-auto">
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


