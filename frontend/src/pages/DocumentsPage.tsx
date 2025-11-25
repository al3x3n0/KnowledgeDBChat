/**
 * Documents management page
 */

import React, { useState, useEffect, useRef } from 'react';
import { useQuery, useMutation, useQueryClient } from 'react-query';
// Use built distribution to avoid ESM fully specified resolution issues
import videojs from 'video.js';
import 'video.js/dist/video-js.css';
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
  X,
  Video,
  FileVideo
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

// Helpers for naming
const getBaseName = (document: Document): string => {
  const original = document.extra_metadata?.original_filename || document.title || '';
  return original.replace(/\.[^.]+$/, '');
};

const getDisplayTitle = (document: Document): string => {
  // Display without extension while converting/after upload
  return getBaseName(document) || document.title || '';
};

const getDownloadFilename = (document: Document): string => {
  const base = getBaseName(document) || `document_${document.id}`;
  // Choose extension based on current file_type or original
  if (document.file_type === 'video/mp4') return `${base}.mp4`;
  const original = document.extra_metadata?.original_filename || document.title || '';
  const m = original.match(/\.([^.]+)$/);
  return m ? `${base}.${m[1]}` : base;
};

const DocumentsPage: React.FC = () => {
  const { user } = useAuth();
  const queryClient = useQueryClient();
  
  const [activeTab, setActiveTab] = useState<'documents' | 'videos'>('documents');
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedSource, setSelectedSource] = useState<string>('');
  const [showUploadModal, setShowUploadModal] = useState(false);
  const [selectedDocument, setSelectedDocument] = useState<Document | null>(null);
  const [deleteConfirmOpen, setDeleteConfirmOpen] = useState(false);
  const [documentToDelete, setDocumentToDelete] = useState<string | null>(null);
  const [transcriptionProgress, setTranscriptionProgress] = useState<Record<string, { progress: number; message: string; stage: string }>>({});
  const transcriptionWebSockets = React.useRef<Record<string, WebSocket>>({});
  const [uploadProgress, setUploadProgress] = useState<Record<string, { progress: number; status: string }>>({});
  const [uploadStatus, setUploadStatus] = useState<Record<string, string>>({});
  const [streamingSegments, setStreamingSegments] = useState<Record<string, Array<{ start: number; text: string }>>>({});
  // Live document status overrides from WebSocket to avoid refetch loops
  const [docStatus, setDocStatus] = useState<Record<string, { is_transcoding?: boolean; is_transcribing?: boolean; is_transcribed?: boolean; failed?: boolean; error?: string }>>({});

  const getDocFlags = (doc: Document) => {
    const override = docStatus[doc.id] || {};
    const isTranscoding = override.is_transcoding ?? (doc.extra_metadata?.is_transcoding === true);
    const isTranscribing = override.is_transcribing ?? (doc.extra_metadata?.is_transcribing === true);
    const isTranscribed = override.is_transcribed ?? (doc.extra_metadata?.is_transcribed === true);
    return { isTranscoding, isTranscribing, isTranscribed };
  };

  // Helper function to check if document is video/audio
  const isVideoAudio = (doc: Document): boolean => {
    if (doc.file_type) {
      return doc.file_type.startsWith('video/') || doc.file_type.startsWith('audio/');
    }
    const ext = doc.title?.toLowerCase().split('.').pop() || '';
    return ['.mp4', '.avi', '.mkv', '.mov', '.webm', '.flv', '.wmv', '.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac']
      .some(e => ext === e.substring(1));
  };

  // Fetch documents
  const { data: allDocuments, isLoading: documentsLoading, refetch: refetchDocuments } = useQuery(
    ['documents', searchQuery, selectedSource],
    () => apiClient.getDocuments({
      search: searchQuery || undefined,
      source_id: selectedSource || undefined,
      limit: 100,
    }),
    {
      refetchOnWindowFocus: false,
      refetchInterval: 10000, // Refetch every 10 seconds to catch transcription updates
    }
  );

  // Filter documents based on active tab
  const documents = allDocuments?.filter(doc => {
    if (activeTab === 'videos') {
      return isVideoAudio(doc);
    } else {
      return !isVideoAudio(doc);
    }
  }) || [];

  // Connect WebSocket for transcription progress on video/audio documents
  useEffect(() => {
    // Connect to WebSocket for each video/audio document that is being transcribed
    documents.forEach((doc) => {
      const flags = getDocFlags(doc);
      const wantsProgress = flags.isTranscribing || flags.isTranscoding;
      if (isVideoAudio(doc) && wantsProgress && !transcriptionWebSockets.current[doc.id]) {
        try {
          const ws = apiClient.createTranscriptionProgressWebSocket(doc.id);
          
          ws.onopen = () => {
            console.log(`Transcription progress WebSocket connected for document ${doc.id}`);
          };
          
          ws.onmessage = (event) => {
            try {
              const data = JSON.parse(event.data);
              
              if (data.type === 'transcription_progress') {
                const progress = data.progress || {};
                setTranscriptionProgress(prev => ({
                  ...prev,
                  [doc.id]: {
                    progress: progress.progress || 0,
                    message: progress.message || 'Processing...',
                    stage: progress.stage || 'unknown'
                  }
                }));
              } else if (data.type === 'document_status') {
                // Update local override for flags; avoid refetching to prevent WS churn
                const status = data.status || {};
                setDocStatus(prev => ({
                  ...prev,
                  [doc.id]: { ...(prev[doc.id] || {}), ...status }
                }));
              } else if (data.type === 'transcription_segment') {
                const seg = data.segment || {};
                if (typeof seg.start === 'number' && typeof seg.text === 'string') {
                  setStreamingSegments(prev => {
                    const cur = prev[doc.id] || [];
                    // Avoid duplicate adjacent same text
                    if (cur.length > 0 && cur[cur.length - 1].text === seg.text) {
                      return prev;
                    }
                    return { ...prev, [doc.id]: [...cur, { start: seg.start, text: seg.text }] };
                  });
                }
              } else if (data.type === 'transcription_complete') {
                // Transcription complete, refetch documents
                queryClient.invalidateQueries('documents');
                // Close WebSocket
                if (transcriptionWebSockets.current[doc.id]) {
                  transcriptionWebSockets.current[doc.id].close();
                  delete transcriptionWebSockets.current[doc.id];
                }
                // Clear progress
                setTranscriptionProgress(prev => {
                  const newProgress = { ...prev };
                  delete newProgress[doc.id];
                  return newProgress;
                });
                // Clear status override
                setDocStatus(prev => {
                  const next = { ...prev } as any;
                  delete next[doc.id];
                  return next;
                });
                setStreamingSegments(prev => {
                  const next = { ...prev } as any;
                  delete next[doc.id];
                  return next;
                });
              } else if (data.type === 'transcription_error') {
                toast.error(`Transcription error: ${data.error || 'Unknown error'}`);
                // Mark failed locally so status shows immediately
                setDocStatus(prev => ({
                  ...prev,
                  [doc.id]: { ...(prev[doc.id] || {}), failed: true, error: data.error || 'Unknown error' }
                }));
                // Close WebSocket
                if (transcriptionWebSockets.current[doc.id]) {
                  transcriptionWebSockets.current[doc.id].close();
                  delete transcriptionWebSockets.current[doc.id];
                }
                // Clear progress
                setTranscriptionProgress(prev => {
                  const newProgress = { ...prev };
                  delete newProgress[doc.id];
                  return newProgress;
                });
                // Clear status override
                setDocStatus(prev => {
                  const next = { ...prev } as any;
                  // keep failed flag until next fetch updates it
                  return next;
                });
                setStreamingSegments(prev => {
                  const next = { ...prev } as any;
                  delete next[doc.id];
                  return next;
                });
              }
            } catch (error) {
              console.error('Error parsing transcription progress message:', error);
            }
          };
          
          ws.onclose = () => {
            console.log(`Transcription progress WebSocket closed for document ${doc.id}`);
            delete transcriptionWebSockets.current[doc.id];
          };
          
          ws.onerror = (error) => {
            console.error(`Transcription progress WebSocket error for document ${doc.id}:`, error);
          };
          
          transcriptionWebSockets.current[doc.id] = ws;
        } catch (error) {
          console.error(`Failed to create transcription progress WebSocket for document ${doc.id}:`, error);
        }
      }
    });
    
    // Cleanup: close WebSockets for documents that are no longer being transcribed
    Object.keys(transcriptionWebSockets.current).forEach((docId) => {
      const doc = documents.find(d => d.id === docId);
      const flags = doc ? getDocFlags(doc) : { isTranscoding: false, isTranscribing: false };
      const wantsProgress = flags.isTranscribing || flags.isTranscoding;
      if (!doc || !isVideoAudio(doc) || !wantsProgress) {
        transcriptionWebSockets.current[docId].close();
        delete transcriptionWebSockets.current[docId];
        setTranscriptionProgress(prev => {
          const newProgress = { ...prev };
          delete newProgress[docId];
          return newProgress;
        });
        setDocStatus(prev => {
          const next = { ...prev } as any;
          delete next[docId];
          return next;
        });
        setStreamingSegments(prev => {
          const next = { ...prev } as any;
          delete next[docId];
          return next;
        });
      }
    });
    
    // Cleanup on unmount
    return () => {
      Object.values(transcriptionWebSockets.current).forEach(ws => ws.close());
      transcriptionWebSockets.current = {};
    };
  }, [documents, queryClient]);

  // Fetch document sources
  const { data: sources } = useQuery(
    'documentSources',
    () => apiClient.getDocumentSources(),
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
    (documentId: string) => apiClient.reprocessDocument(documentId),
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
    const isVideoAudioDoc = isVideoAudio(document);
    const { isTranscoding, isTranscribing, isTranscribed } = getDocFlags(document);
    const failed = docStatus[document.id]?.failed || !!document.processing_error || !!document.extra_metadata?.transcription_error;
    
    // Error has priority
    if (failed) {
      return 'Failed';
    }

    // Check if we have real-time progress
    const progress = transcriptionProgress[document.id];
    if (progress && (isTranscribing || isTranscoding)) {
      return progress.message || (isTranscribing ? 'Transcribing...' : 'Transcoding to MP4...');
    }
    
    if (isTranscoding && isVideoAudioDoc) {
      return 'Transcoding to MP4...';
    }

    if (isTranscribing) {
      return 'Transcribing...';
    } else if (isTranscribed && isVideoAudioDoc) {
      return 'Transcribed';
    } else if (document.is_processed) {
      return 'Processed';
    } else if (document.processing_error) {
      return 'Failed';
    } else {
      return 'Processing';
    }
  };

  const getProgressPercentage = (document: Document): number | null => {
    const progress = transcriptionProgress[document.id];
    const { isTranscribing, isTranscoding } = getDocFlags(document);
    if (progress && (isTranscribing || isTranscoding)) {
      return progress.progress;
    }
    return null;
  };

  const tabs = [
    { id: 'documents' as const, name: 'Documents', icon: FileText },
    { id: 'videos' as const, name: 'Videos & Audio', icon: Video },
  ];

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="bg-white border-b border-gray-200 p-6">
        <div className="flex items-center justify-between mb-4">
          <h1 className="text-2xl font-bold text-gray-900">
            {activeTab === 'videos' ? 'Videos & Audio' : 'Documents'}
          </h1>
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
              Upload {activeTab === 'videos' ? 'Video/Audio' : 'Document'}
            </Button>
          </div>
        </div>

        {/* Tabs */}
        <div className="flex space-x-1 border-b border-gray-200 mb-4">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
                  activeTab === tab.id
                    ? 'border-primary-500 text-primary-700'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                <Icon className="w-4 h-4 mr-2" />
                {tab.name}
                {tab.id === 'videos' && (
                  <span className="ml-2 px-2 py-0.5 text-xs font-medium bg-gray-100 text-gray-700 rounded-full">
                    {allDocuments?.filter(isVideoAudio).length || 0}
                  </span>
                )}
                {tab.id === 'documents' && (
                  <span className="ml-2 px-2 py-0.5 text-xs font-medium bg-gray-100 text-gray-700 rounded-full">
                    {allDocuments?.filter(d => !isVideoAudio(d)).length || 0}
                  </span>
                )}
              </button>
            );
          })}
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
          <LoadingSpinner className="h-64" text={`Loading ${activeTab === 'videos' ? 'videos' : 'documents'}...`} />
        ) : documents?.length === 0 ? (
          <div className="flex items-center justify-center h-64">
            <div className="text-center">
              {activeTab === 'videos' ? (
                <Video className="w-16 h-16 mx-auto mb-4 text-gray-300" />
              ) : (
                <FileText className="w-16 h-16 mx-auto mb-4 text-gray-300" />
              )}
              <h3 className="text-lg font-medium text-gray-900 mb-2">
                No {activeTab === 'videos' ? 'videos or audio files' : 'documents'} found
              </h3>
              <p className="text-gray-500 mb-6">
                {searchQuery || selectedSource 
                  ? 'Try adjusting your search or filter criteria'
                  : activeTab === 'videos'
                    ? 'Upload video or audio files to get started'
                    : 'Upload documents or configure data sources to get started'
                }
              </p>
              <Button
                onClick={() => setShowUploadModal(true)}
                icon={<Upload className="w-4 h-4" />}
              >
                Upload {activeTab === 'videos' ? 'Video/Audio' : 'Document'}
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
                        {isVideoAudio(document) && (
                          <Video className="w-5 h-5 text-primary-600 flex-shrink-0" />
                        )}
                        <h3 className="text-lg font-medium text-gray-900 truncate">
                          {getDisplayTitle(document)}
                        </h3>
                        <div className="flex items-center space-x-1 ml-auto">
                          {getStatusIcon(document)}
                          <span className="text-sm text-gray-500">
                            {getStatusText(document)}
                          </span>
                        </div>
                      </div>

                      {/* Progress bar for transcription/transcoding */}
                      {(document.extra_metadata?.is_transcribing || document.extra_metadata?.is_transcoding) && getProgressPercentage(document) !== null && (
                        <div className="mt-2">
                          <div className="flex items-center justify-between mb-1">
                            <span className="text-sm font-medium text-gray-700">
                              {transcriptionProgress[document.id]?.stage 
                                ? transcriptionProgress[document.id].stage.charAt(0).toUpperCase() + 
                                  transcriptionProgress[document.id].stage.slice(1).replace(/_/g, ' ')
                                : 'Processing'}
                            </span>
                            <span className="text-sm text-gray-600">
                              {getProgressPercentage(document)?.toFixed(1)}%
                            </span>
                          </div>
                          <div className="w-full bg-gray-200 rounded-full h-2">
                            <div
                              className="bg-primary-600 h-2 rounded-full transition-all duration-300"
                              style={{ width: `${getProgressPercentage(document)}%` }}
                            />
                          </div>
                          <div className="text-xs text-gray-500 mt-1">
                            {transcriptionProgress[document.id]?.message || 'Processing...'}
                          </div>
                        </div>
                      )}

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
                        onClick={() => {
                          if (document.extra_metadata?.is_transcoding) {
                            toast.error('Video is converting to MP4. Please wait.');
                            return;
                          }
                          setSelectedDocument(document);
                        }}
                        disabled={document.extra_metadata?.is_transcoding === true}
                      >
                        View
                      </Button>
                      {/* Retry transcription if failed */}
                      {isVideoAudio(document) && (docStatus[document.id]?.failed || !!document.extra_metadata?.transcription_error) && !document.extra_metadata?.is_transcribing && !document.extra_metadata?.is_transcoding && (
                        <Button
                          variant="ghost"
                          size="sm"
                          icon={<RefreshCw className="w-4 h-4" />}
                          onClick={async () => {
                            try {
                              const res = await apiClient.transcribeDocument(document.id);
                              toast.success(res.message || 'Transcription scheduled');
                              queryClient.invalidateQueries('documents');
                            } catch (e: any) {
                              toast.error(e?.response?.data?.detail || e?.message || 'Failed to schedule transcription');
                            }
                          }}
                        >
                          Retry Transcription
                        </Button>
                      )}
                      
                      {document.url && (
                        <Button
                          variant="ghost"
                          size="sm"
                          icon={<Eye className="w-4 h-4" />}
                          onClick={() => window.open(document.url, '_blank')}
                          disabled={document.extra_metadata?.is_transcoding === true}
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
                            if (document.extra_metadata?.is_transcoding) {
                              toast.error('Video is converting to MP4. Please wait.');
                              return;
                            }
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
                              link.download = getDownloadFilename(document) || `document_${document.id}`;
                              window.document.body.appendChild(link);
                              link.click();
                              window.document.body.removeChild(link);
                              window.URL.revokeObjectURL(url);
                            } catch (error) {
                              toast.error('Failed to generate download URL');
                            }
                          }}
                          disabled={document.extra_metadata?.is_transcoding === true}
                        >
                          Download
                        </Button>
                      )}
                      
                      <Button
                        variant="ghost"
                        size="sm"
                        icon={<Trash2 className="w-4 h-4" />}
                        onClick={() => handleDeleteDocument(document.id)}
                        disabled={document.extra_metadata?.is_transcoding === true}
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
          liveSegments={streamingSegments[selectedDocument.id]}
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
  const [uploadStatus, setUploadStatus] = useState<string>('');
  const [uploadedBytes, setUploadedBytes] = useState(0);
  const [isUploading, setIsUploading] = useState(false);

  const uploadMutation = useMutation(
    async ({ file, title, tags }: { file: File; title?: string; tags?: string[] }) => {
      setIsUploading(true);
      setUploadProgress(0);
      setUploadStatus('Preparing upload...');
      
      try {
        const result = await apiClient.uploadDocumentWithProgress(
          file,
          title,
          tags,
          (progress) => {
            setUploadProgress(progress);
          },
          (status) => {
            setUploadStatus(status);
          },
          (uploaded, total) => {
            setUploadedBytes(uploaded);
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
        setUploadStatus('');
        onSuccess();
      },
      onError: (error: any) => {
        const errorMessage = error?.response?.data?.detail || error?.message || 'Failed to upload document';
        toast.error(errorMessage);
        setUploadProgress(0);
        setUploadStatus('');
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
              accept=".pdf,.doc,.docx,.ppt,.pptx,.txt,.html,.htm,.md,.markdown,.mp4,.avi,.mkv,.mov,.webm,.flv,.wmv,.mp3,.wav,.m4a,.flac,.ogg,.aac,application/pdf,application/msword,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/vnd.openxmlformats-officedocument.presentationml.presentation,application/vnd.ms-powerpoint,text/plain,text/html,text/markdown,video/mp4,video/x-msvideo,video/x-matroska,video/quicktime,video/webm,video/x-flv,video/x-ms-wmv,audio/mpeg,audio/wav,audio/x-m4a,audio/flac,audio/ogg,audio/aac"
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
            <span className="text-gray-600">{uploadStatus || 'Uploading...'}</span>
            <span className="text-gray-600 font-medium">{uploadProgress}%</span>
          </div>
          <ProgressBar
            value={uploadProgress}
            showLabel={false}
            size="md"
            variant="primary"
          />
          {file && (
            <div className="flex items-center justify-between text-xs text-gray-500">
              <span>{formatFileSize(uploadedBytes)} / {formatFileSize(file.size)}</span>
            </div>
          )}
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
  liveSegments?: Array<{ start: number; text: string }>;
}

const DocumentDetailsModal: React.FC<DocumentDetailsModalProps> = ({ document, onClose, liveSegments }) => {
  const mainPlayerRef = useRef<any>(null);
  const secondaryPlayerRef = useRef<any>(null);
  const transcriptRef = useRef<HTMLDivElement>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [isVideoLoading, setIsVideoLoading] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [activeSegmentIndex, setActiveSegmentIndex] = useState<number | null>(null);
  // Track only basic state; player handles loading internally
  
  // Check if document is video/audio
  const isVideoAudio = (doc: Document): boolean => {
    if (doc.file_type) {
      return doc.file_type.startsWith('video/') || doc.file_type.startsWith('audio/');
    }
    const ext = doc.title?.toLowerCase().split('.').pop() || '';
    return ['.mp4', '.avi', '.mkv', '.mov', '.webm', '.flv', '.wmv', '.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac']
      .some(e => ext === e.substring(1));
  };
  
  const isMediaFile = isVideoAudio(document);
  
  // Load video URL when modal opens - use streaming URL for better performance
  useEffect(() => {
    if (isMediaFile && !videoUrl && !document.extra_metadata?.is_transcoding) {
      setIsVideoLoading(true);
      // Use streaming URL - native player supports HTTP range requests for streaming
      // This allows the video to start playing while still downloading
      const token = localStorage.getItem('access_token');
      
      if (!token) {
        console.error('No authentication token found');
        toast.error('Please log in to view videos');
        setIsVideoLoading(false);
        return;
      }
      
      // Use video streamer for video/audio files
      apiClient.getDocumentDownloadUrl(document.id, true, true)
        .then(url => {
          // Add token as query parameter for authentication
          const separator = url.includes('?') ? '&' : '?';
          const streamingUrl = `${url}${separator}token=${encodeURIComponent(token)}`;
          console.log('Video URL with token:', streamingUrl);
          console.log('Token length:', token.length);
          setVideoUrl(streamingUrl);
          setIsVideoLoading(false);
        })
        .catch(error => {
          console.error('Failed to get video URL:', error);
          toast.error(`Failed to load video: ${error.message || 'Unknown error'}`);
          setIsVideoLoading(false);
        });
    }
  }, [isMediaFile, document.id, videoUrl, document.extra_metadata?.is_transcoding]);

  // Initialize Video.js players when URL is ready
  useEffect(() => {
    if (!videoUrl || document.extra_metadata?.is_transcoding) return;
    
    const players: any[] = [];
    
    // Wait for DOM elements to be available
    const timer = setTimeout(() => {
      const mainEl = window.document.getElementById('vjs-player-main') as HTMLVideoElement | null;
      const secEl = window.document.getElementById('vjs-player-secondary') as HTMLVideoElement | null;
      const type = (document.file_type?.startsWith('audio/') ? document.file_type : (document.file_type || 'video/mp4')) as string;
      
      const setup = (el: HTMLVideoElement | null) => {
        if (!el) return;
        try {
          // Check if player already exists
          const existingPlayer = (videojs as any).getPlayer(el);
          if (existingPlayer) {
            existingPlayer.dispose();
          }
          
          // Ensure the URL includes the token query parameter
          const urlWithToken = videoUrl; // videoUrl already has token from useEffect
          console.log('Initializing Video.js with URL:', urlWithToken);
          
          const player = videojs(el, {
            controls: true,
            preload: 'metadata',
            autoplay: false,
            fluid: true,
            responsive: true,
            sources: [{ src: urlWithToken, type }],
          });
          
          // Intercept video element's src changes to ensure token is always present
          player.ready(() => {
            const videoEl = player.el().querySelector('video') as HTMLVideoElement;
            if (videoEl) {
              // Ensure src has token
              const currentSrc = videoEl.src || videoEl.currentSrc;
              if (currentSrc && !currentSrc.includes('token=')) {
                const token = localStorage.getItem('access_token');
                if (token) {
                  const separator = currentSrc.includes('?') ? '&' : '?';
                  const newSrc = `${currentSrc}${separator}token=${encodeURIComponent(token)}`;
                  videoEl.src = newSrc;
                  console.log('Added token to video element src:', newSrc);
                }
              }
              
              // Monitor for src changes (e.g., during range requests)
              const observer = new MutationObserver(() => {
                const src = videoEl.src || videoEl.currentSrc;
                if (src && !src.includes('token=')) {
                  const token = localStorage.getItem('access_token');
                  if (token) {
                    const separator = src.includes('?') ? '&' : '?';
                    const newSrc = `${src}${separator}token=${encodeURIComponent(token)}`;
                    if (videoEl.src !== newSrc) {
                      videoEl.src = newSrc;
                      console.log('Added token to video element src (after change):', newSrc);
                    }
                  }
                }
              });
              
              observer.observe(videoEl, {
                attributes: true,
                attributeFilter: ['src'],
              });
              
              // Also listen to loadstart to catch range requests
              videoEl.addEventListener('loadstart', () => {
                const src = videoEl.src || videoEl.currentSrc;
                if (src && !src.includes('token=')) {
                  const token = localStorage.getItem('access_token');
                  if (token) {
                    const separator = src.includes('?') ? '&' : '?';
                    const newSrc = `${src}${separator}token=${encodeURIComponent(token)}`;
                    if (videoEl.src !== newSrc) {
                      videoEl.src = newSrc;
                      console.log('Added token to video element src (loadstart):', newSrc);
                    }
                  }
                }
              });
            }
          });
          
          player.on('timeupdate', () => {
            try { 
              const time = player.currentTime() || 0;
              setCurrentTime(time);
              handleProgress({ playedSeconds: time });
            } catch {}
          });
          
          player.on('loadedmetadata', () => {
            try {
              // Seek to 0.1 seconds to show preview frame
              player.currentTime(0.1);
              player.pause();
            } catch {}
          });
          
          player.on('error', (error: any) => {
            console.error('Video.js player error:', error);
            const playerError = player.error();
            if (playerError) {
              console.error('Player error code:', playerError.code);
              console.error('Player error message:', playerError.message);
            }
          });
          
          players.push(player);
          if (el.id === 'vjs-player-main') mainPlayerRef.current = player;
          if (el.id === 'vjs-player-secondary') secondaryPlayerRef.current = player;
        } catch (e) {
          console.warn('Video.js init error', e);
        }
      };
      
      setup(mainEl);
      setup(secEl);
    }, 100);
    
    return () => {
      clearTimeout(timer);
      players.forEach(p => { 
        try { 
          if (p && typeof p.dispose === 'function') {
            p.dispose(); 
          }
        } catch {} 
      });
    };
  }, [videoUrl, document.extra_metadata?.is_transcoding]);
  
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
      let filename = getDownloadFilename(document) || `document_${document.id}`;
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

  const jumpToTimestamp = (seconds: number) => {
    try {
      const player = (mainPlayerRef.current || secondaryPlayerRef.current) as any;
      if (player && typeof player.currentTime === 'function') {
        player.currentTime(seconds);
      }
    } catch {}
  };
  
  // Get segments from metadata
  const segments = document.extra_metadata?.transcription_metadata?.segments || [];
  const liveSegs = liveSegments || [];
  // Pull streaming segments from parent state via window var (quick bridge) if available
  
  // Update active segment based on current playback time
  useEffect(() => {
    if (segments.length === 0) return;
    
    const activeIndex = segments.findIndex((segment: any) => {
      const start = segment.start || 0;
      const end = segment.end || 0;
      return currentTime >= start && currentTime < end;
    });
    
    if (activeIndex !== -1 && activeIndex !== activeSegmentIndex) {
      setActiveSegmentIndex(activeIndex);
      
      // Auto-scroll to active segment
      if (transcriptRef.current) {
        const segmentElement = transcriptRef.current.children[activeIndex] as HTMLElement;
        if (segmentElement) {
          segmentElement.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
      }
    } else if (activeIndex === -1) {
      setActiveSegmentIndex(null);
    }
  }, [currentTime, segments, activeSegmentIndex]);
  
  const handleProgress = (state: any) => {
    if (state && typeof state === 'object' && 'playedSeconds' in state) {
      setCurrentTime(state.playedSeconds);
    }
  };
  
  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg p-6 w-full max-w-6xl max-h-[90vh] overflow-auto">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-gray-900">{getDisplayTitle(document)}</h2>
          <div className="flex items-center gap-2">
            {document.file_path && (
              <Button
                variant="ghost"
                size="sm"
                icon={<Download className="w-4 h-4" />}
                onClick={handleDownload}
                disabled={document.extra_metadata?.is_transcoding === true}
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
          {/* Video/Audio Player with YouTube-style Layout */}
          {document.extra_metadata?.is_transcoding ? (
            <div className="p-6 bg-yellow-50 border border-yellow-200 rounded">
              <h3 className="font-medium text-yellow-900 mb-2">Converting to MP4</h3>
              <p className="text-yellow-800">This video is being converted to MP4 for playback. Viewing and downloading are temporarily disabled. Please close this dialog and check back shortly.</p>
            </div>
          ) : isMediaFile && segments.length > 0 ? (
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
              {/* Video Player - Takes 2/3 width on large screens */}
              <div className="lg:col-span-2">
                <h3 className="font-medium text-gray-900 mb-2">
                  {document.file_type?.startsWith('video/') ? 'Video Player' : 'Audio Player'}
                </h3>
                <div className="bg-black rounded-lg overflow-hidden">
                  {isVideoLoading ? (
                    <div className="aspect-video flex items-center justify-center bg-gray-900">
                      <div className="text-white">Loading player...</div>
                    </div>
                  ) : videoUrl ? (
                    <div style={{ position: 'relative', width: '100%', aspectRatio: '16/9' }}>
                      <video id="vjs-player-main" className="video-js vjs-default-skin w-full h-full" playsInline preload="metadata" />
                    </div>
                  ) : (
                    <div className="aspect-video flex items-center justify-center bg-gray-900">
                      <div className="text-white">Failed to load media</div>
                    </div>
                  )}
                </div>
              </div>
              
          {/* Transcript Sidebar - Takes 1/3 width on large screens */}
          <div className="lg:col-span-1">
            <h3 className="font-medium text-gray-900 mb-2">
              Transcript ({segments.length || liveSegs.length} segments)
            </h3>
            <div 
              ref={transcriptRef}
              className="bg-gray-50 rounded-lg p-4 max-h-[600px] overflow-y-auto"
              style={{ scrollBehavior: 'smooth' }}
            >
              <div className="space-y-2">
                {(segments.length > 0 ? segments : liveSegs).map((segment: any, index: number) => {
                  const formatTime = (seconds: number) => {
                    const hours = Math.floor(seconds / 3600);
                    const minutes = Math.floor((seconds % 3600) / 60);
                    const secs = Math.floor(seconds % 60);
                    if (hours > 0) {
                      return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
                    }
                    return `${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
                  };
                  
                  const startTime = segment.start || 0;
                  const endTime = segment.end || startTime + 1;
                  const text = segment.text || '';
                  const speaker = segment.speaker;
                  const isActive = activeSegmentIndex === index;
                  
                  return (
                    <div 
                      key={segment.id || index} 
                          className={`p-3 rounded-lg border transition-all cursor-pointer ${
                            isActive 
                              ? 'bg-primary-50 border-primary-300 shadow-md' 
                              : 'bg-white border-gray-200 hover:border-primary-300 hover:shadow-sm'
                          }`}
                          onClick={() => {
                            jumpToTimestamp(startTime);
                          }}
                        >
                          <div className="flex items-center space-x-2 mb-1">
                            <span className={`text-xs font-mono font-medium px-2 py-1 rounded ${
                              isActive 
                                ? 'text-primary-700 bg-primary-100' 
                                : 'text-primary-600 bg-primary-50'
                            }`}>
                              {formatTime(startTime)}
                            </span>
                            {speaker && (
                              <span className="text-xs font-medium text-gray-600 bg-gray-100 px-2 py-1 rounded">
                                {speaker}
                              </span>
                            )}
                          </div>
                          <p className={`text-sm mt-1 ${
                            isActive ? 'text-gray-900 font-medium' : 'text-gray-700'
                          }`}>
                            {text}
                          </p>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>
            </div>
          ) : isMediaFile ? (
            <div>
              <h3 className="font-medium text-gray-900 mb-2">
                {document.file_type?.startsWith('video/') ? 'Video Player' : 'Audio Player'}
              </h3>
              <div className="bg-black rounded-lg overflow-hidden">
                {isVideoLoading ? (
                  <div className="aspect-video flex items-center justify-center bg-gray-900">
                    <div className="text-white">Loading player...</div>
                  </div>
                ) : videoUrl ? (
                  <video id="vjs-player-secondary" className="video-js vjs-default-skin w-full" playsInline preload="metadata" />
                ) : (
                  <div className="aspect-video flex items-center justify-center bg-gray-900">
                    <div className="text-white">Failed to load media</div>
                  </div>
                )}
              </div>
            </div>
          ) : null}

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

          {/* Transcription Segments (only show if not in YouTube-style layout) */}
          {!isMediaFile && document.extra_metadata?.transcription_metadata?.segments && 
           document.extra_metadata.transcription_metadata.segments.length > 0 && (
            <div>
              <h3 className="font-medium text-gray-900 mb-2">
                Transcript with Time Codes ({document.extra_metadata.transcription_metadata.segments.length} segments)
              </h3>
              <div className="p-4 bg-gray-50 rounded-lg max-h-96 overflow-auto">
                <div className="space-y-3">
                  {document.extra_metadata.transcription_metadata.segments.map((segment: any, index: number) => {
                    const formatTime = (seconds: number) => {
                      const hours = Math.floor(seconds / 3600);
                      const minutes = Math.floor((seconds % 3600) / 60);
                      const secs = Math.floor(seconds % 60);
                      if (hours > 0) {
                        return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
                      }
                      return `${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
                    };
                    
                    const startTime = segment.start || 0;
                    const endTime = segment.end || 0;
                    const text = segment.text || '';
                    const speaker = segment.speaker;
                    
                    return (
                      <div 
                        key={segment.id || index} 
                        className="p-3 bg-white rounded border border-gray-200 hover:border-primary-300 hover:shadow-sm transition-all cursor-pointer"
                      >
                        <div className="flex items-start justify-between mb-1">
                          <div className="flex items-center space-x-2">
                            <span className="text-xs font-mono font-medium text-primary-600 bg-primary-50 px-2 py-1 rounded">
                              {formatTime(startTime)} - {formatTime(endTime)}
                            </span>
                            {speaker && (
                              <span className="text-xs font-medium text-gray-600 bg-gray-100 px-2 py-1 rounded">
                                {speaker}
                              </span>
                            )}
                          </div>
                        </div>
                        <p className="text-sm text-gray-700 mt-1">{text}</p>
                      </div>
                    );
                  })}
                </div>
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
