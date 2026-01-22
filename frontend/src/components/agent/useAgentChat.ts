/**
 * Custom hook for managing agent chat state and interactions.
 */

import { useState, useCallback } from 'react';
import { apiClient } from '../../services/api';
import toast from 'react-hot-toast';

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

export interface UseAgentChatReturn {
  messages: AgentMessage[];
  isLoading: boolean;
  sendMessage: (content: string) => Promise<void>;
  pendingUpload: boolean;
  uploadContext: { suggestedTitle?: string; suggestedTags?: string[] } | null;
  handleFileUpload: (file: File) => Promise<void>;
  clearPendingUpload: () => void;
  clearMessages: () => void;
  confirmDelete: (documentId: string) => Promise<void>;
  pendingDelete: { documentId: string; title: string } | null;
  clearPendingDelete: () => void;
}

export function useAgentChat(): UseAgentChatReturn {
  const [messages, setMessages] = useState<AgentMessage[]>([
    {
      id: 'welcome',
      role: 'assistant',
      content: 'Hello! I can help you manage documents in the knowledge base. You can ask me to:\n\n- **Search** for documents by content or tags\n- **Summarize** documents (single or batch)\n- **Upload** new documents or create from text\n- **Delete** documents (single or batch)\n- **Find similar** documents\n- **Manage tags** (add, remove, list)\n- **View stats** about the knowledge base\n\nHow can I help you today?',
      created_at: new Date().toISOString(),
    }
  ]);
  const [isLoading, setIsLoading] = useState(false);
  const [pendingUpload, setPendingUpload] = useState(false);
  const [uploadContext, setUploadContext] = useState<{ suggestedTitle?: string; suggestedTags?: string[] } | null>(null);
  const [pendingDelete, setPendingDelete] = useState<{ documentId: string; title: string } | null>(null);

  const sendMessage = useCallback(async (content: string) => {
    if (!content.trim()) return;

    // Add user message
    const userMessage: AgentMessage = {
      id: `user-${Date.now()}`,
      role: 'user',
      content: content.trim(),
      created_at: new Date().toISOString(),
    };
    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);

    try {
      // Call agent API
      const response = await apiClient.agentChat({
        message: content,
        conversation_history: messages.slice(-10), // Last 10 messages for context
      });

      // Check for special actions
      if (response.requires_user_action) {
        if (response.action_type === 'upload_file') {
          setPendingUpload(true);
          // Extract upload context from tool results
          const uploadTool = response.tool_results?.find(t => t.tool_name === 'request_file_upload');
          if (uploadTool?.tool_output) {
            setUploadContext({
              suggestedTitle: uploadTool.tool_output.suggested_title,
              suggestedTags: uploadTool.tool_output.suggested_tags,
            });
          }
        }
      }

      // Check for pending delete confirmation
      const deleteTool = response.tool_results?.find(
        t => t.tool_name === 'delete_document' && t.tool_output?.action === 'confirmation_required'
      );
      if (deleteTool?.tool_output) {
        setPendingDelete({
          documentId: deleteTool.tool_output.document_id,
          title: deleteTool.tool_output.title,
        });
      }

      // Add assistant message
      const assistantMessage: AgentMessage = {
        id: response.message.id,
        role: 'assistant',
        content: response.message.content,
        tool_calls: response.tool_results as AgentToolCall[] | undefined,
        created_at: response.message.created_at,
      };
      setMessages(prev => [...prev, assistantMessage]);

    } catch (error: any) {
      console.error('Agent chat error:', error);
      toast.error('Failed to process your request');

      // Add error message
      const errorMessage: AgentMessage = {
        id: `error-${Date.now()}`,
        role: 'assistant',
        content: 'Sorry, I encountered an error processing your request. Please try again.',
        created_at: new Date().toISOString(),
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  }, [messages]);

  const handleFileUpload = useCallback(async (file: File) => {
    setIsLoading(true);

    try {
      const result = await apiClient.uploadDocument(
        file,
        uploadContext?.suggestedTitle,
        uploadContext?.suggestedTags
      );

      // Add success message
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
      await apiClient.agentConfirmDelete(documentId);

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

  const clearMessages = useCallback(() => {
    setMessages([{
      id: 'welcome',
      role: 'assistant',
      content: 'Chat cleared! I can search, summarize, upload, delete, find similar documents, manage tags, and show stats. What would you like to do?',
      created_at: new Date().toISOString(),
    }]);
    setPendingUpload(false);
    setUploadContext(null);
    setPendingDelete(null);
  }, []);

  return {
    messages,
    isLoading,
    sendMessage,
    pendingUpload,
    uploadContext,
    handleFileUpload,
    clearPendingUpload,
    clearMessages,
    confirmDelete,
    pendingDelete,
    clearPendingDelete,
  };
}
