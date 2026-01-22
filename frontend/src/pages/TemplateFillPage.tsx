/**
 * Template Fill page for AI-powered document generation
 */

import React, { useState, useEffect, useCallback, useRef } from 'react';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import {
  Upload,
  FileText,
  CheckCircle,
  XCircle,
  Clock,
  Download,
  Trash2,
  RefreshCw,
  Loader2,
  Search,
  Plus,
  X,
  FileCheck,
} from 'lucide-react';
import { formatDistanceToNow } from 'date-fns';
import toast from 'react-hot-toast';

import { apiClient } from '../services/api';
import type {
  Document as KnowledgeDocument,
  TemplateJob,
  TemplateProgressUpdate,
} from '../types';
import { useAuth } from '../contexts/AuthContext';
import Button from '../components/common/Button';
import Input from '../components/common/Input';
import LoadingSpinner from '../components/common/LoadingSpinner';
import { formatFileSize } from '../utils/formatting';

const TemplateFillPage: React.FC = () => {
  const { user } = useAuth();
  const queryClient = useQueryClient();

  // State
  const [templateFile, setTemplateFile] = useState<File | null>(null);
  const [selectedDocuments, setSelectedDocuments] = useState<string[]>([]);
  const [documentSearch, setDocumentSearch] = useState('');
  const [showDocumentSelector, setShowDocumentSelector] = useState(false);
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [jobProgress, setJobProgress] = useState<TemplateProgressUpdate | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const wsRef = useRef<WebSocket | null>(null);

  // Fetch template jobs
  const { data: jobsData, isLoading: jobsLoading, refetch: refetchJobs } = useQuery(
    'templateJobs',
    () => apiClient.listTemplateJobs({ limit: 50 }),
    {
      refetchInterval: activeJobId ? 5000 : false,
    }
  );

  // Fetch documents for selection
  const { data: documents, isLoading: documentsLoading } = useQuery(
    ['documents', documentSearch],
    () => apiClient.getDocuments({ search: documentSearch, limit: 100 }),
    {
      enabled: showDocumentSelector,
    }
  );

  // Create job mutation
  const createJobMutation = useMutation(
    async () => {
      if (!templateFile || selectedDocuments.length === 0) {
        throw new Error('Please select a template and source documents');
      }
      return apiClient.createTemplateFillJob(templateFile, selectedDocuments);
    },
    {
      onSuccess: (job) => {
        toast.success('Template fill job started');
        setActiveJobId(job.id);
        setTemplateFile(null);
        setSelectedDocuments([]);
        refetchJobs();
      },
      onError: (error: any) => {
        toast.error(error.message || 'Failed to create job');
      },
    }
  );

  // Delete job mutation
  const deleteJobMutation = useMutation(
    (jobId: string) => apiClient.deleteTemplateJob(jobId),
    {
      onSuccess: () => {
        toast.success('Job deleted');
        refetchJobs();
      },
      onError: () => {
        toast.error('Failed to delete job');
      },
    }
  );

  // Download filled template
  const handleDownload = async (job: TemplateJob) => {
    try {
      const { blob, filename } = await apiClient.downloadFilledTemplate(job.id);
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
      toast.success('Download started');
    } catch (error) {
      toast.error('Failed to download file');
    }
  };

  // WebSocket for progress updates
  useEffect(() => {
    if (!activeJobId) {
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
      return;
    }

    try {
      const ws = apiClient.createTemplateProgressWebSocket(activeJobId);

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data) as TemplateProgressUpdate;
          setJobProgress(data);

          if (data.type === 'complete') {
            toast.success('Template filled successfully!');
            setActiveJobId(null);
            refetchJobs();
          } else if (data.type === 'error') {
            toast.error(data.error || 'Job failed');
            setActiveJobId(null);
            refetchJobs();
          }
        } catch (e) {
          console.error('Failed to parse WebSocket message:', e);
        }
      };

      ws.onerror = () => {
        // Reset progress state on WebSocket error
        setJobProgress(null);
      };

      ws.onclose = () => {
        wsRef.current = null;
      };

      wsRef.current = ws;
    } catch {
      // Reset state if WebSocket creation fails
      setJobProgress(null);
    }

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
      // Cleanup progress state
      setJobProgress(null);
    };
  }, [activeJobId, refetchJobs]);

  // Handle file selection
  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      if (!file.name.endsWith('.docx') && !file.name.endsWith('.doc')) {
        toast.error('Please select a Word document (.docx or .doc)');
        return;
      }
      setTemplateFile(file);
    }
  };

  // Toggle document selection
  const toggleDocument = (docId: string) => {
    setSelectedDocuments((prev) =>
      prev.includes(docId) ? prev.filter((id) => id !== docId) : [...prev, docId]
    );
  };

  // Get status icon
  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="w-5 h-5 text-green-500" />;
      case 'failed':
        return <XCircle className="w-5 h-5 text-red-500" />;
      case 'pending':
        return <Clock className="w-5 h-5 text-yellow-500" />;
      default:
        return <Loader2 className="w-5 h-5 text-blue-500 animate-spin" />;
    }
  };

  // Get status label
  const getStatusLabel = (job: TemplateJob) => {
    if (job.status === 'completed') return 'Completed';
    if (job.status === 'failed') return 'Failed';
    if (job.status === 'pending') return 'Pending';
    if (job.current_section) {
      return `${job.status}: ${job.current_section}`;
    }
    return job.status.charAt(0).toUpperCase() + job.status.slice(1);
  };

  return (
    <div className="p-6 max-w-6xl mx-auto">
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
          Template Fill
        </h1>
        <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
          Upload a template document and select source materials to automatically fill it with AI-generated content.
        </p>
      </div>

      {/* Create New Job Section */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6 mb-6">
        <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
          Create New Job
        </h2>

        {/* Template Upload */}
        <div className="mb-4">
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Template Document (DOCX)
          </label>
          <div className="flex items-center gap-4">
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileSelect}
              accept=".docx,.doc"
              className="hidden"
            />
            <Button
              variant="secondary"
              onClick={() => fileInputRef.current?.click()}
              className="flex items-center gap-2"
            >
              <Upload className="w-4 h-4" />
              Select Template
            </Button>
            {templateFile && (
              <div className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
                <FileText className="w-4 h-4" />
                {templateFile.name}
                <button
                  onClick={() => setTemplateFile(null)}
                  className="text-red-500 hover:text-red-700"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
            )}
          </div>
        </div>

        {/* Source Documents */}
        <div className="mb-4">
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Source Documents ({selectedDocuments.length} selected)
          </label>
          <Button
            variant="secondary"
            onClick={() => setShowDocumentSelector(!showDocumentSelector)}
            className="flex items-center gap-2"
          >
            <Plus className="w-4 h-4" />
            Select Source Documents
          </Button>

          {/* Selected documents chips */}
          {selectedDocuments.length > 0 && (
            <div className="mt-2 flex flex-wrap gap-2">
              {selectedDocuments.map((docId) => {
                const doc = documents?.find((d) => d.id === docId);
                return (
                  <span
                    key={docId}
                    className="inline-flex items-center gap-1 px-2 py-1 bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-200 rounded text-sm"
                  >
                    {doc?.title || docId.slice(0, 8)}
                    <button
                      onClick={() => toggleDocument(docId)}
                      className="hover:text-blue-600"
                    >
                      <X className="w-3 h-3" />
                    </button>
                  </span>
                );
              })}
            </div>
          )}

          {/* Document selector modal */}
          {showDocumentSelector && (
            <div className="mt-4 border border-gray-200 dark:border-gray-700 rounded-lg p-4 max-h-64 overflow-y-auto">
              <div className="mb-3">
                <Input
                  placeholder="Search documents..."
                  value={documentSearch}
                  onChange={(e) => setDocumentSearch(e.target.value)}
                  className="w-full"
                />
              </div>
              {documentsLoading ? (
                <LoadingSpinner />
              ) : documents && documents.length > 0 ? (
                <div className="space-y-2">
                  {documents.map((doc) => (
                    <label
                      key={doc.id}
                      className="flex items-center gap-2 p-2 hover:bg-gray-50 dark:hover:bg-gray-700 rounded cursor-pointer"
                    >
                      <input
                        type="checkbox"
                        checked={selectedDocuments.includes(doc.id)}
                        onChange={() => toggleDocument(doc.id)}
                        className="rounded text-blue-500"
                      />
                      <span className="text-sm text-gray-700 dark:text-gray-300 truncate">
                        {doc.title}
                      </span>
                      <span className="text-xs text-gray-400 ml-auto">
                        {doc.file_type}
                      </span>
                    </label>
                  ))}
                </div>
              ) : (
                <p className="text-sm text-gray-500 text-center py-4">
                  No documents found
                </p>
              )}
              <div className="mt-3 flex justify-end">
                <Button
                  variant="secondary"
                  size="sm"
                  onClick={() => setShowDocumentSelector(false)}
                >
                  Done
                </Button>
              </div>
            </div>
          )}
        </div>

        {/* Submit Button */}
        <Button
          onClick={() => createJobMutation.mutate()}
          disabled={!templateFile || selectedDocuments.length === 0 || createJobMutation.isLoading}
          className="flex items-center gap-2"
        >
          {createJobMutation.isLoading ? (
            <Loader2 className="w-4 h-4 animate-spin" />
          ) : (
            <FileCheck className="w-4 h-4" />
          )}
          Start Filling
        </Button>
      </div>

      {/* Active Job Progress */}
      {activeJobId && jobProgress && (
        <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4 mb-6">
          <div className="flex items-center gap-2 mb-2">
            <Loader2 className="w-5 h-5 text-blue-500 animate-spin" />
            <span className="font-medium text-blue-700 dark:text-blue-300">
              Processing: {jobProgress.data?.stage || 'Starting...'}
            </span>
          </div>
          {jobProgress.data?.current_section && (
            <p className="text-sm text-blue-600 dark:text-blue-400 mb-2">
              Current section: {jobProgress.data.current_section}
              {jobProgress.data.section_index && jobProgress.data.total_sections && (
                <span className="ml-2">
                  ({jobProgress.data.section_index}/{jobProgress.data.total_sections})
                </span>
              )}
            </p>
          )}
          <div className="w-full bg-blue-200 dark:bg-blue-800 rounded-full h-2">
            <div
              className="bg-blue-500 h-2 rounded-full transition-all duration-300"
              style={{ width: `${jobProgress.data?.progress || 0}%` }}
            />
          </div>
        </div>
      )}

      {/* Jobs List */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow">
        <div className="px-6 py-4 border-b border-gray-200 dark:border-gray-700 flex items-center justify-between">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
            Recent Jobs
          </h2>
          <Button variant="secondary" size="sm" onClick={() => refetchJobs()}>
            <RefreshCw className="w-4 h-4" />
          </Button>
        </div>

        {jobsLoading ? (
          <div className="p-8">
            <LoadingSpinner />
          </div>
        ) : jobsData?.jobs && jobsData.jobs.length > 0 ? (
          <div className="divide-y divide-gray-200 dark:divide-gray-700">
            {jobsData.jobs.map((job) => (
              <div
                key={job.id}
                className="p-4 hover:bg-gray-50 dark:hover:bg-gray-700/50"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    {getStatusIcon(job.status)}
                    <div>
                      <p className="font-medium text-gray-900 dark:text-white">
                        {job.template_filename}
                      </p>
                      <p className="text-sm text-gray-500 dark:text-gray-400">
                        {getStatusLabel(job)}
                        {job.progress > 0 && job.status !== 'completed' && (
                          <span className="ml-2">({job.progress}%)</span>
                        )}
                      </p>
                      <p className="text-xs text-gray-400">
                        Created {formatDistanceToNow(new Date(job.created_at))} ago
                        {job.completed_at && (
                          <span className="ml-2">
                            | Completed {formatDistanceToNow(new Date(job.completed_at))} ago
                          </span>
                        )}
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    {job.status === 'completed' && (
                      <Button
                        variant="secondary"
                        size="sm"
                        onClick={() => handleDownload(job)}
                        className="flex items-center gap-1"
                      >
                        <Download className="w-4 h-4" />
                        Download
                      </Button>
                    )}
                    <Button
                      variant="secondary"
                      size="sm"
                      onClick={() => deleteJobMutation.mutate(job.id)}
                      className="text-red-500 hover:text-red-700"
                    >
                      <Trash2 className="w-4 h-4" />
                    </Button>
                  </div>
                </div>
                {job.error_message && (
                  <p className="mt-2 text-sm text-red-500 dark:text-red-400">
                    Error: {job.error_message}
                  </p>
                )}
              </div>
            ))}
          </div>
        ) : (
          <div className="p-8 text-center text-gray-500 dark:text-gray-400">
            No template jobs yet. Create one above!
          </div>
        )}
      </div>
    </div>
  );
};

export default TemplateFillPage;
