/**
 * Documents management page
 */

import React, { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import { 
  Upload, 
  Search, 
  Filter, 
  Download,
  Trash2,
  RefreshCw,
  FileText,
  CheckCircle,
  XCircle,
  Clock,
  Eye,
  MoreVertical,
  Plus,
  X
} from 'lucide-react';
import { formatDistanceToNow } from 'date-fns';

import { apiClient } from '../services/api';
import { Document, DocumentSource } from '../types';
import { useAuth } from '../contexts/AuthContext';
import Button from '../components/common/Button';
import Input from '../components/common/Input';
import LoadingSpinner from '../components/common/LoadingSpinner';
import ConfirmationModal from '../components/common/ConfirmationModal';
import ProgressBar from '../components/common/ProgressBar';
import { formatFileSize } from '../utils/formatting';
import toast from 'react-hot-toast';

const DocumentsPage: React.FC = () => {
  const { user } = useAuth();
  const queryClient = useQueryClient();
  
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedSource, setSelectedSource] = useState<string>('');
  const [showUploadModal, setShowUploadModal] = useState(false);
  const [selectedDocument, setSelectedDocument] = useState<Document | null>(null);
  const [deleteConfirmOpen, setDeleteConfirmOpen] = useState(false);
  const [documentToDelete, setDocumentToDelete] = useState<string | null>(null);

  // Fetch documents
  const { data: documents, isLoading: documentsLoading, refetch: refetchDocuments } = useQuery(
    ['documents', searchQuery, selectedSource],
    () => apiClient.getDocuments({
      search: searchQuery || undefined,
      source_id: selectedSource || undefined,
      limit: 100,
    }),
    {
      refetchOnWindowFocus: false,
    }
  );

  // Fetch document sources
  const { data: sources } = useQuery(
    'documentSources',
    apiClient.getDocumentSources,
    {
      refetchOnWindowFocus: false,
    }
  );

  // Delete document mutation
  const deleteDocumentMutation = useMutation(
    (documentId: string) => apiClient.deleteDocument(documentId),
    {
      onSuccess: () => {
        queryClient.invalidateQueries('documents');
        toast.success('Document deleted successfully');
      },
      onError: (error: any) => {
        console.error('Delete document error:', error);
        const errorMessage = error?.response?.data?.detail || error?.message || 'Failed to delete document';
        toast.error(errorMessage);
      },
    }
  );

  // Reprocess document mutation
  const reprocessDocumentMutation = useMutation(
    apiClient.reprocessDocument,
    {
      onSuccess: () => {
        queryClient.invalidateQueries('documents');
        toast.success('Document reprocessing started');
      },
      onError: () => {
        toast.error('Failed to reprocess document');
      },
    }
  );

  const handleDeleteDocument = (documentId: string) => {
    setDocumentToDelete(documentId);
    setDeleteConfirmOpen(true);
  };

  const confirmDeleteDocument = () => {
    if (documentToDelete) {
      deleteDocumentMutation.mutate(documentToDelete);
      setDeleteConfirmOpen(false);
      setDocumentToDelete(null);
    }
  };

  const handleReprocessDocument = (documentId: string) => {
    reprocessDocumentMutation.mutate(documentId);
  };

  const getStatusIcon = (document: Document) => {
    if (document.is_processed) {
      return <CheckCircle className="w-4 h-4 text-green-500" />;
    } else if (document.processing_error) {
      return <XCircle className="w-4 h-4 text-red-500" />;
    } else {
      return <Clock className="w-4 h-4 text-yellow-500" />;
    }
  };

  const getStatusText = (document: Document) => {
    if (document.is_processed) {
      return 'Processed';
    } else if (document.processing_error) {
      return 'Failed';
    } else {
      return 'Processing';
    }
  };

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="bg-white border-b border-gray-200 p-6">
        <div className="flex items-center justify-between mb-4">
          <h1 className="text-2xl font-bold text-gray-900">Documents</h1>
          <div className="flex items-center space-x-3">
            <Button
              onClick={() => refetchDocuments()}
              variant="ghost"
              icon={<RefreshCw className="w-4 h-4" />}
            >
              Refresh
            </Button>
            <Button
              onClick={() => setShowUploadModal(true)}
              icon={<Upload className="w-4 h-4" />}
            >
              Upload Document
            </Button>
          </div>
        </div>

        {/* Search and filters */}
        <div className="flex items-center space-x-4">
          <div className="flex-1">
            <Input
              placeholder="Search documents..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              leftIcon={<Search className="w-4 h-4" />}
            />
          </div>
          <div className="w-64">
            <select
              value={selectedSource}
              onChange={(e) => setSelectedSource(e.target.value)}
              className="block w-full rounded-lg border-gray-300 shadow-sm focus:border-primary-500 focus:ring-primary-500"
            >
              <option value="">All Sources</option>
              {sources?.map((source) => (
                <option key={source.id} value={source.id}>
                  {source.name}
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-auto">
        {documentsLoading ? (
          <LoadingSpinner className="h-64" text="Loading documents..." />
        ) : documents?.length === 0 ? (
          <div className="flex items-center justify-center h-64">
            <div className="text-center">
              <FileText className="w-16 h-16 mx-auto mb-4 text-gray-300" />
              <h3 className="text-lg font-medium text-gray-900 mb-2">No documents found</h3>
              <p className="text-gray-500 mb-6">
                {searchQuery || selectedSource 
                  ? 'Try adjusting your search or filter criteria'
                  : 'Upload documents or configure data sources to get started'
                }
              </p>
              <Button
                onClick={() => setShowUploadModal(true)}
                icon={<Upload className="w-4 h-4" />}
              >
                Upload Document
              </Button>
            </div>
          </div>
        ) : (
          <div className="p-6">
            <div className="grid gap-4">
              {(documents || []).map((document) => (
                <div
                  key={document.id}
                  className="bg-white border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow duration-200"
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1 min-w-0">
                      {/* Title and status */}
                      <div className="flex items-center space-x-2 mb-2">
                        <h3 className="text-lg font-medium text-gray-900 truncate">
                          {document.title}
                        </h3>
                        <div className="flex items-center space-x-1">
                          {getStatusIcon(document)}
                          <span className="text-sm text-gray-500">
                            {getStatusText(document)}
                          </span>
                        </div>
                      </div>

                      {/* Metadata */}
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm text-gray-600 mb-3">
                        <div>
                          <span className="font-medium">Source:</span>
                          <span className="ml-1">{document.source.name}</span>
                        </div>
                        <div>
                          <span className="font-medium">Type:</span>
                          <span className="ml-1">{document.file_type || 'Unknown'}</span>
                        </div>
                        <div>
                          <span className="font-medium">Size:</span>
                          <span className="ml-1">{formatFileSize(document.file_size)}</span>
                        </div>
                        <div>
                          <span className="font-medium">Updated:</span>
                          <span className="ml-1">
                            {formatDistanceToNow(new Date(document.updated_at))} ago
                          </span>
                        </div>
                      </div>

                      {/* Tags */}
                      {document.tags && document.tags.length > 0 && (
                        <div className="flex flex-wrap gap-1 mb-3">
                          {document.tags.map((tag, index) => (
                            <span
                              key={index}
                              className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-primary-100 text-primary-800"
                            >
                              {tag}
                            </span>
                          ))}
                        </div>
                      )}

                      {/* Author */}
                      {document.author && (
                        <div className="text-sm text-gray-600">
                          <span className="font-medium">Author:</span>
                          <span className="ml-1">{document.author}</span>
                        </div>
                      )}

                      {/* Error message */}
                      {document.processing_error && (
                        <div className="mt-2 p-2 bg-red-50 border border-red-200 rounded text-sm text-red-700">
                          <strong>Processing Error:</strong> {document.processing_error}
                        </div>
                      )}
                    </div>

                    {/* Actions */}
                    <div className="flex items-center space-x-2 ml-4">
                      <Button
                        variant="ghost"
                        size="sm"
                        icon={<Eye className="w-4 h-4" />}
                        onClick={() => setSelectedDocument(document)}
                      >
                        View
                      </Button>
                      
                      {document.url && (
                        <Button
                          variant="ghost"
                          size="sm"
                          icon={<Eye className="w-4 h-4" />}
                          onClick={() => window.open(document.url, '_blank')}
                        >
                          Open
                        </Button>
                      )}
                      
                      {document.file_path && (
                        <Button
                          variant="ghost"
                          size="sm"
                          icon={<Download className="w-4 h-4" />}
                          onClick={async () => {
                            try {
                              const downloadUrl = await apiClient.getDocumentDownloadUrl(document.id, true);
                              // Use fetch to download with authentication
                              const response = await fetch(downloadUrl, {
                                method: 'GET',
                                headers: {
                                  'Authorization': `Bearer ${localStorage.getItem('access_token') || ''}`,
                                },
                                credentials: 'include',
                              });
                              
                              if (!response.ok) {
                                throw new Error(`Download failed: ${response.statusText}`);
                              }
                              
                              const blob = await response.blob();
                              const url = window.URL.createObjectURL(blob);
                              const link = window.document.createElement('a');
                              link.href = url;
                              link.download = document.title || `document_${document.id}`;
                              window.document.body.appendChild(link);
                              link.click();
                              window.document.body.removeChild(link);
                              window.URL.revokeObjectURL(url);
                            } catch (error) {
                              toast.error('Failed to generate download URL');
                            }
                          }}
                        >
                          Download
                        </Button>
                      )}
                      
                      <Button
                        variant="ghost"
                        size="sm"
                        icon={<Trash2 className="w-4 h-4" />}
                        onClick={() => handleDeleteDocument(document.id)}
                        loading={deleteDocumentMutation.isLoading}
                      >
                        Delete
                      </Button>
                      
                      {user?.role === 'admin' && (
                        <Button
                          variant="ghost"
                          size="sm"
                          icon={<RefreshCw className="w-4 h-4" />}
                          onClick={() => handleReprocessDocument(document.id)}
                          loading={reprocessDocumentMutation.isLoading}
                        >
                          Reprocess
                        </Button>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Upload Modal */}
      {showUploadModal && (
        <UploadModal
          onClose={() => setShowUploadModal(false)}
          onSuccess={() => {
            setShowUploadModal(false);
            queryClient.invalidateQueries('documents');
          }}
        />
      )}

      {/* Document Details Modal */}
      {selectedDocument && (
        <DocumentDetailsModal
          document={selectedDocument}
          onClose={() => setSelectedDocument(null)}
        />
      )}

      {/* Delete Confirmation Modal */}
      <ConfirmationModal
        isOpen={deleteConfirmOpen}
        onClose={() => {
          setDeleteConfirmOpen(false);
          setDocumentToDelete(null);
        }}
        onConfirm={confirmDeleteDocument}
        title="Delete Document"
        message="Are you sure you want to delete this document? This action cannot be undone."
        confirmText="Delete"
        cancelText="Cancel"
        variant="danger"
        isLoading={deleteDocumentMutation.isLoading}
      />
    </div>
  );
};

// Upload Modal Component
interface UploadModalProps {
  onClose: () => void;
  onSuccess: () => void;
}

const UploadModal: React.FC<UploadModalProps> = ({ onClose, onSuccess }) => {
  const [file, setFile] = useState<File | null>(null);
  const [title, setTitle] = useState('');
  const [tags, setTags] = useState('');
  const [uploadProgress, setUploadProgress] = useState(0);
  const [isUploading, setIsUploading] = useState(false);

  const uploadMutation = useMutation(
    async ({ file, title, tags }: { file: File; title?: string; tags?: string[] }) => {
      setIsUploading(true);
      setUploadProgress(0);
      
      try {
        const result = await apiClient.uploadDocumentWithProgress(
          file,
          title,
          tags,
          (progress) => {
            setUploadProgress(progress);
          }
        );
        return result;
      } finally {
        setIsUploading(false);
      }
    },
    {
      onSuccess: () => {
        toast.success('Document uploaded successfully');
        setUploadProgress(0);
        onSuccess();
      },
      onError: () => {
        toast.error('Failed to upload document');
        setUploadProgress(0);
        setIsUploading(false);
      },
    }
  );

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) return;

    const tagArray = tags
      .split(',')
      .map(tag => tag.trim())
      .filter(tag => tag.length > 0);

    uploadMutation.mutate({
      file,
      title: title || undefined,
      tags: tagArray.length > 0 ? tagArray : undefined,
    });
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg p-6 w-full max-w-md">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">Upload Document</h2>
        
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              File
            </label>
            <input
              type="file"
              accept=".pdf,.doc,.docx,.ppt,.pptx,.txt,.html,.htm,.md,.markdown,application/pdf,application/msword,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/vnd.openxmlformats-officedocument.presentationml.presentation,application/vnd.ms-powerpoint,text/plain,text/html,text/markdown"
              onChange={(e) => setFile(e.target.files?.[0] || null)}
              className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-medium file:bg-primary-50 file:text-primary-700 hover:file:bg-primary-100"
              required
            />
          </div>

          <Input
            label="Title (Optional)"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            placeholder="Enter document title"
          />

          <Input
            label="Tags (Optional)"
            value={tags}
            onChange={(e) => setTags(e.target.value)}
            placeholder="Enter tags separated by commas"
            helpText="e.g. documentation, guide, reference"
          />

          {/* Upload Progress */}
          {isUploading && (
            <div className="space-y-2">
              <div className="flex items-center justify-between text-sm">
                <span className="text-gray-600">Uploading...</span>
                <span className="text-gray-600 font-medium">{uploadProgress}%</span>
              </div>
              <ProgressBar
                value={uploadProgress}
                showLabel={false}
                size="md"
                variant="primary"
              />
              {file && (
                <p className="text-xs text-gray-500">
                  {file.name} ({(file.size / 1024 / 1024).toFixed(2)} MB)
                </p>
              )}
            </div>
          )}

          <div className="flex justify-end space-x-3 pt-4">
            <Button
              type="button"
              variant="ghost"
              onClick={onClose}
              disabled={isUploading}
            >
              Cancel
            </Button>
            <Button
              type="submit"
              loading={uploadMutation.isLoading}
              disabled={!file || isUploading}
            >
              {isUploading ? 'Uploading...' : 'Upload'}
            </Button>
          </div>
        </form>
      </div>
    </div>
  );
};

// Document Details Modal Component
interface DocumentDetailsModalProps {
  document: Document;
  onClose: () => void;
}

const DocumentDetailsModal: React.FC<DocumentDetailsModalProps> = ({ document, onClose }) => {
  const handleDownload = async () => {
    try {
      const downloadUrl = await apiClient.getDocumentDownloadUrl(document.id, true);
      // Use fetch to download with authentication
      const response = await fetch(downloadUrl, {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('access_token') || ''}`,
        },
        credentials: 'include',
      });
      
      if (!response.ok) {
        throw new Error(`Download failed: ${response.statusText}`);
      }
      
      // Get filename from Content-Disposition header
      const contentDisposition = response.headers.get('Content-Disposition');
      let filename = document.title || `document_${document.id}`;
      if (contentDisposition) {
        const filenameMatch = contentDisposition.match(/filename[^;=\n]*=((['"]).*?\2|[^;\n]*)/);
        if (filenameMatch && filenameMatch[1]) {
          filename = filenameMatch[1].replace(/['"]/g, '');
        }
      }
      
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const link = window.document.createElement('a');
      link.href = url;
      link.download = filename;
      window.document.body.appendChild(link);
      link.click();
      window.document.body.removeChild(link);
      window.URL.revokeObjectURL(url);
    } catch (error) {
      toast.error('Failed to generate download URL');
    }
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg p-6 w-full max-w-4xl max-h-[90vh] overflow-auto">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-gray-900">{document.title}</h2>
          <div className="flex items-center gap-2">
            {document.file_path && (
              <Button
                variant="ghost"
                size="sm"
                icon={<Download className="w-4 h-4" />}
                onClick={handleDownload}
              >
                Download
              </Button>
            )}
            <button
              onClick={onClose}
              className="text-gray-400 hover:text-gray-600"
            >
              <X className="w-6 h-6" />
            </button>
          </div>
        </div>

        <div className="space-y-4">
          {/* Metadata */}
          <div className="grid grid-cols-2 gap-4 p-4 bg-gray-50 rounded-lg">
            <div>
              <span className="font-medium text-gray-700">Source:</span>
              <span className="ml-2">{document.source.name}</span>
            </div>
            <div>
              <span className="font-medium text-gray-700">Type:</span>
              <span className="ml-2">{document.file_type || 'Unknown'}</span>
            </div>
            <div>
              <span className="font-medium text-gray-700">Size:</span>
              <span className="ml-2">{formatFileSize(document.file_size)}</span>
            </div>
            <div>
              <span className="font-medium text-gray-700">Status:</span>
              <span className="ml-2">{document.is_processed ? 'Processed' : 'Processing'}</span>
            </div>
          </div>

          {/* Content */}
          {document.content && (
            <div>
              <h3 className="font-medium text-gray-900 mb-2">Content Preview</h3>
              <div className="p-4 bg-gray-50 rounded-lg max-h-96 overflow-auto">
                <pre className="whitespace-pre-wrap text-sm text-gray-700">
                  {document.content.substring(0, 2000)}
                  {document.content.length > 2000 && '...'}
                </pre>
              </div>
            </div>
          )}

          {/* Chunks */}
          {document.chunks && document.chunks.length > 0 && (
            <div>
              <h3 className="font-medium text-gray-900 mb-2">
                Processed Chunks ({document.chunks.length})
              </h3>
              <div className="space-y-2 max-h-64 overflow-auto">
                {document.chunks.map((chunk) => (
                  <div key={chunk.id} className="p-3 bg-gray-50 rounded border">
                    <div className="text-xs text-gray-500 mb-1">
                      Chunk {chunk.chunk_index + 1}
                    </div>
                    <p className="text-sm text-gray-700">
                      {chunk.content.substring(0, 200)}
                      {chunk.content.length > 200 && '...'}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default DocumentsPage;
