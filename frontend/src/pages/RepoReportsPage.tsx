/**
 * Repository Reports & Presentations Generator Page
 */

import React, { useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import toast from 'react-hot-toast';
import apiClient from '../services/api';
import {
  RepoReportJob,
  RepoReportJobListItem,
  RepoReportStatus,
  RepoReportOutputFormat,
  RepoReportStyle,
  RepoReportSection,
  DocumentSource,
  ThemeConfig,
} from '../types';

// Status badge colors
const STATUS_COLORS: Record<RepoReportStatus, string> = {
  pending: 'bg-yellow-100 text-yellow-800',
  analyzing: 'bg-blue-100 text-blue-800',
  generating: 'bg-indigo-100 text-indigo-800',
  uploading: 'bg-purple-100 text-purple-800',
  completed: 'bg-green-100 text-green-800',
  failed: 'bg-red-100 text-red-800',
  cancelled: 'bg-gray-100 text-gray-800',
};

// Stage labels for display
const STAGE_LABELS: Record<string, string> = {
  'Starting analysis': 'Starting...',
  'Fetching repository info': 'Fetching repo info...',
  'Fetching README': 'Loading README...',
  'Fetching file tree': 'Building file tree...',
  'Fetching commits': 'Loading commits...',
  'Fetching issues': 'Loading issues...',
  'Fetching pull requests': 'Loading PRs...',
  'Fetching contributors': 'Loading contributors...',
  'Fetching languages': 'Analyzing languages...',
  'Generating insights': 'Generating insights...',
  'Building document content': 'Building content...',
  'Rendering DOCX document': 'Rendering DOCX...',
  'Rendering PDF document': 'Rendering PDF...',
  'Generating presentation outline': 'Creating outline...',
  'Generating diagrams': 'Generating diagrams...',
  'Building PPTX presentation': 'Building slides...',
  'Uploading to storage': 'Uploading...',
  'Completed': 'Completed',
};

// Output format options
const OUTPUT_FORMATS: { value: RepoReportOutputFormat; label: string; icon: string }[] = [
  { value: 'docx', label: 'Word Document (.docx)', icon: 'W' },
  { value: 'pdf', label: 'PDF Document (.pdf)', icon: 'P' },
  { value: 'pptx', label: 'PowerPoint (.pptx)', icon: 'S' },
];

// Style options
const AVAILABLE_STYLES: { value: RepoReportStyle; label: string; description: string }[] = [
  { value: 'professional', label: 'Professional', description: 'Clean corporate look' },
  { value: 'technical', label: 'Technical', description: 'Developer-focused style' },
  { value: 'modern', label: 'Modern', description: 'Contemporary design' },
  { value: 'minimal', label: 'Minimal', description: 'Simple and clean' },
  { value: 'corporate', label: 'Corporate', description: 'Traditional business' },
  { value: 'creative', label: 'Creative', description: 'Artistic approach' },
  { value: 'dark', label: 'Dark', description: 'Dark theme' },
];

interface CreateModalProps {
  isOpen: boolean;
  onClose: () => void;
  onCreate: (data: CreateReportData) => void;
  sources: DocumentSource[];
  sections: RepoReportSection[];
  isLoading: boolean;
}

interface CreateReportData {
  source_id?: string;
  repo_url?: string;
  repo_token?: string;
  output_format: RepoReportOutputFormat;
  title?: string;
  sections: string[];
  slide_count?: number;
  include_diagrams: boolean;
  style: RepoReportStyle;
  custom_theme?: ThemeConfig;
}

const CreateReportModal: React.FC<CreateModalProps> = ({
  isOpen,
  onClose,
  onCreate,
  sources,
  sections,
  isLoading,
}) => {
  const [sourceType, setSourceType] = useState<'existing' | 'adhoc'>('adhoc');
  const [selectedSourceId, setSelectedSourceId] = useState<string>('');
  const [repoUrl, setRepoUrl] = useState('');
  const [repoToken, setRepoToken] = useState('');
  const [title, setTitle] = useState('');
  const [outputFormat, setOutputFormat] = useState<RepoReportOutputFormat>('docx');
  const [selectedSections, setSelectedSections] = useState<string[]>([]);
  const [slideCount, setSlideCount] = useState(10);
  const [includeDiagrams, setIncludeDiagrams] = useState(true);
  const [style, setStyle] = useState<RepoReportStyle>('professional');

  // Filter to only git sources
  const gitSources = sources.filter(
    (s) => s.source_type === 'github' || s.source_type === 'gitlab'
  );

  // Initialize selected sections with defaults
  useEffect(() => {
    if (sections.length > 0 && selectedSections.length === 0) {
      setSelectedSections(
        sections.filter((s) => s.default).map((s) => s.id)
      );
    }
  }, [sections, selectedSections.length]);

  const toggleSection = (sectionId: string) => {
    setSelectedSections((prev) =>
      prev.includes(sectionId)
        ? prev.filter((id) => id !== sectionId)
        : [...prev, sectionId]
    );
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    if (sourceType === 'adhoc' && !repoUrl.trim()) {
      toast.error('Please enter a repository URL');
      return;
    }

    if (sourceType === 'existing' && !selectedSourceId) {
      toast.error('Please select a repository source');
      return;
    }

    if (selectedSections.length === 0) {
      toast.error('Please select at least one section');
      return;
    }

    const data: CreateReportData = {
      output_format: outputFormat,
      sections: selectedSections,
      include_diagrams: includeDiagrams,
      style,
      title: title.trim() || undefined,
    };

    if (sourceType === 'existing') {
      data.source_id = selectedSourceId;
    } else {
      data.repo_url = repoUrl.trim();
      if (repoToken.trim()) {
        data.repo_token = repoToken.trim();
      }
    }

    if (outputFormat === 'pptx') {
      data.slide_count = slideCount;
    }

    onCreate(data);
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl w-full max-w-2xl max-h-[90vh] overflow-hidden">
        <div className="px-6 py-4 border-b border-gray-200 flex justify-between items-center">
          <h2 className="text-xl font-semibold text-gray-900">Generate Repository Report</h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600"
            disabled={isLoading}
          >
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        <form onSubmit={handleSubmit} className="p-6 overflow-y-auto max-h-[calc(90vh-8rem)]">
          {/* Source Selection */}
          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Repository Source
            </label>
            <div className="flex space-x-4">
              <label className="flex items-center">
                <input
                  type="radio"
                  value="adhoc"
                  checked={sourceType === 'adhoc'}
                  onChange={() => setSourceType('adhoc')}
                  className="text-blue-600 focus:ring-blue-500"
                  disabled={isLoading}
                />
                <span className="ml-2 text-sm text-gray-700">Enter URL</span>
              </label>
              {gitSources.length > 0 && (
                <label className="flex items-center">
                  <input
                    type="radio"
                    value="existing"
                    checked={sourceType === 'existing'}
                    onChange={() => setSourceType('existing')}
                    className="text-blue-600 focus:ring-blue-500"
                    disabled={isLoading}
                  />
                  <span className="ml-2 text-sm text-gray-700">Use Existing Source</span>
                </label>
              )}
            </div>
          </div>

          {/* Ad-hoc URL input */}
          {sourceType === 'adhoc' && (
            <>
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Repository URL *
                </label>
                <input
                  type="text"
                  value={repoUrl}
                  onChange={(e) => setRepoUrl(e.target.value)}
                  placeholder="https://github.com/owner/repo"
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
                  disabled={isLoading}
                />
                <p className="mt-1 text-xs text-gray-500">
                  Supports GitHub and GitLab repository URLs
                </p>
              </div>

              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Access Token (optional)
                </label>
                <input
                  type="password"
                  value={repoToken}
                  onChange={(e) => setRepoToken(e.target.value)}
                  placeholder="ghp_xxxx or glpat-xxxx"
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
                  disabled={isLoading}
                />
                <p className="mt-1 text-xs text-gray-500">
                  Required for private repositories
                </p>
              </div>
            </>
          )}

          {/* Existing source selection */}
          {sourceType === 'existing' && (
            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Select Repository Source *
              </label>
              <select
                value={selectedSourceId}
                onChange={(e) => setSelectedSourceId(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
                disabled={isLoading}
              >
                <option value="">Choose a source...</option>
                {gitSources.map((source) => (
                  <option key={source.id} value={source.id}>
                    {source.name} ({source.source_type})
                  </option>
                ))}
              </select>
            </div>
          )}

          {/* Title */}
          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Report Title (optional)
            </label>
            <input
              type="text"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              placeholder="Auto-generated from repo name"
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
              disabled={isLoading}
            />
          </div>

          {/* Output Format */}
          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Output Format
            </label>
            <div className="grid grid-cols-3 gap-3">
              {OUTPUT_FORMATS.map((format) => (
                <label
                  key={format.value}
                  className={`flex items-center p-3 border rounded-lg cursor-pointer transition-colors ${
                    outputFormat === format.value
                      ? 'border-blue-500 bg-blue-50'
                      : 'border-gray-200 hover:bg-gray-50'
                  }`}
                >
                  <input
                    type="radio"
                    value={format.value}
                    checked={outputFormat === format.value}
                    onChange={() => setOutputFormat(format.value)}
                    className="sr-only"
                    disabled={isLoading}
                  />
                  <div className={`w-8 h-8 flex items-center justify-center rounded text-sm font-bold ${
                    outputFormat === format.value
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-200 text-gray-600'
                  }`}>
                    {format.icon}
                  </div>
                  <span className="ml-3 text-sm font-medium text-gray-700">
                    {format.label}
                  </span>
                </label>
              ))}
            </div>
          </div>

          {/* Sections */}
          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Include Sections
            </label>
            <div className="grid grid-cols-2 gap-2 max-h-48 overflow-y-auto border border-gray-200 rounded-md p-3">
              {sections.map((section) => (
                <label
                  key={section.id}
                  className="flex items-center p-2 rounded hover:bg-gray-50 cursor-pointer"
                >
                  <input
                    type="checkbox"
                    checked={selectedSections.includes(section.id)}
                    onChange={() => toggleSection(section.id)}
                    className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                    disabled={isLoading}
                  />
                  <div className="ml-2">
                    <span className="text-sm font-medium text-gray-700">{section.name}</span>
                    <p className="text-xs text-gray-500">{section.description}</p>
                  </div>
                </label>
              ))}
            </div>
            <p className="mt-1 text-xs text-gray-500">
              {selectedSections.length} section(s) selected
            </p>
          </div>

          {/* Style */}
          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Visual Style
            </label>
            <select
              value={style}
              onChange={(e) => setStyle(e.target.value as RepoReportStyle)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
              disabled={isLoading}
            >
              {AVAILABLE_STYLES.map((s) => (
                <option key={s.value} value={s.value}>
                  {s.label} - {s.description}
                </option>
              ))}
            </select>
          </div>

          {/* PPTX-specific options */}
          {outputFormat === 'pptx' && (
            <>
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Number of Slides: {slideCount}
                </label>
                <input
                  type="range"
                  min="5"
                  max="20"
                  value={slideCount}
                  onChange={(e) => setSlideCount(parseInt(e.target.value))}
                  className="w-full"
                  disabled={isLoading}
                />
                <div className="flex justify-between text-xs text-gray-500">
                  <span>5</span>
                  <span>20</span>
                </div>
              </div>

              <div className="mb-4">
                <label className="flex items-center">
                  <input
                    type="checkbox"
                    checked={includeDiagrams}
                    onChange={(e) => setIncludeDiagrams(e.target.checked)}
                    className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                    disabled={isLoading}
                  />
                  <span className="ml-2 text-sm text-gray-700">Include Architecture Diagrams</span>
                </label>
              </div>
            </>
          )}
        </form>

        <div className="px-6 py-4 bg-gray-50 border-t border-gray-200 flex justify-end space-x-3">
          <button
            type="button"
            onClick={onClose}
            className="px-4 py-2 text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50"
            disabled={isLoading}
          >
            Cancel
          </button>
          <button
            onClick={handleSubmit}
            disabled={isLoading}
            className="px-4 py-2 text-white bg-blue-600 rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center"
          >
            {isLoading && (
              <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
              </svg>
            )}
            Generate Report
          </button>
        </div>
      </div>
    </div>
  );
};

const RepoReportsPage: React.FC = () => {
  const navigate = useNavigate();
  const [jobs, setJobs] = useState<RepoReportJobListItem[]>([]);
  const [sources, setSources] = useState<DocumentSource[]>([]);
  const [sections, setSections] = useState<RepoReportSection[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isCreating, setIsCreating] = useState(false);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [activeWebSockets, setActiveWebSockets] = useState<Map<string, WebSocket>>(new Map());

  // Fetch jobs
  const fetchJobs = useCallback(async () => {
    try {
      const data = await apiClient.listRepoReports({ limit: 50 });
      setJobs(data.jobs);
    } catch (error) {
      console.error('Failed to fetch repo report jobs:', error);
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Fetch sources (for existing source selection)
  const fetchSources = useCallback(async () => {
    try {
      const data = await apiClient.getDocumentSources();
      setSources(data);
    } catch (error) {
      console.error('Failed to fetch document sources:', error);
    }
  }, []);

  // Fetch available sections
  const fetchSections = useCallback(async () => {
    try {
      const data = await apiClient.getRepoReportSections();
      setSections(data.sections);
    } catch (error) {
      console.error('Failed to fetch sections:', error);
    }
  }, []);

  useEffect(() => {
    fetchJobs();
    fetchSources();
    fetchSections();
  }, [fetchJobs, fetchSources, fetchSections]);

  // Setup WebSocket for in-progress jobs
  useEffect(() => {
    const inProgressJobs = jobs.filter(
      (j) => j.status === 'pending' || j.status === 'analyzing' || j.status === 'generating' || j.status === 'uploading'
    );

    inProgressJobs.forEach((job) => {
      if (!activeWebSockets.has(job.id)) {
        try {
          const ws = apiClient.createRepoReportProgressWebSocket(job.id);

          ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            setJobs((prev) =>
              prev.map((j) =>
                j.id === job.id
                  ? {
                      ...j,
                      progress: data.progress,
                      status: data.status,
                      error: data.error,
                    }
                  : j
              )
            );

            if (data.status === 'completed' || data.status === 'failed' || data.status === 'cancelled') {
              ws.close();
              setActiveWebSockets((prev) => {
                const newMap = new Map(prev);
                newMap.delete(job.id);
                return newMap;
              });
              fetchJobs();
            }
          };

          ws.onerror = (error) => {
            console.error('WebSocket error:', error);
          };

          ws.onclose = () => {
            setActiveWebSockets((prev) => {
              const newMap = new Map(prev);
              newMap.delete(job.id);
              return newMap;
            });
          };

          setActiveWebSockets((prev) => new Map(prev).set(job.id, ws));
        } catch (error) {
          console.error('Failed to create WebSocket:', error);
        }
      }
    });

    return () => {
      activeWebSockets.forEach((ws) => ws.close());
    };
  }, [jobs, activeWebSockets, fetchJobs]);

  const handleCreate = async (data: CreateReportData) => {
    setIsCreating(true);
    try {
      const job = await apiClient.createRepoReport({
        source_id: data.source_id,
        repo_url: data.repo_url,
        repo_token: data.repo_token,
        output_format: data.output_format,
        title: data.title,
        sections: data.sections,
        slide_count: data.slide_count,
        include_diagrams: data.include_diagrams,
        style: data.style,
        custom_theme: data.custom_theme,
      });
      setJobs((prev) => [{
        id: job.id,
        user_id: job.user_id,
        repo_name: job.repo_name,
        repo_url: job.repo_url,
        repo_type: job.repo_type,
        output_format: job.output_format,
        title: job.title,
        status: job.status,
        progress: job.progress,
        file_size: job.file_size,
        error: job.error,
        created_at: job.created_at,
        started_at: job.started_at,
        completed_at: job.completed_at,
      }, ...prev]);
      setShowCreateModal(false);
      toast.success('Repository report generation started!');
    } catch (error) {
      console.error('Failed to create repo report:', error);
      toast.error('Failed to start report generation');
    } finally {
      setIsCreating(false);
    }
  };

  const handleDownload = async (job: RepoReportJobListItem) => {
    try {
      await apiClient.downloadRepoReport(job.id);
      toast.success('Download started');
    } catch (error) {
      console.error('Download failed:', error);
      toast.error('Download failed');
    }
  };

  const handleCreateDraft = async (job: RepoReportJobListItem) => {
    try {
      const draft = await apiClient.createArtifactDraftFromRepoReport(job.id);
      toast.success('Draft created');
      navigate('/artifact-drafts', { state: { selectedDraftId: draft.id } as any });
    } catch (error: any) {
      console.error('Failed to create draft:', error);
      toast.error(error?.response?.data?.detail || 'Failed to create draft');
    }
  };

  const handleDelete = async (jobId: string) => {
    if (!window.confirm('Are you sure you want to delete this report?')) return;
    try {
      await apiClient.deleteRepoReport(jobId);
      setJobs((prev) => prev.filter((j) => j.id !== jobId));
      toast.success('Report deleted');
    } catch (error) {
      console.error('Delete failed:', error);
      toast.error('Delete failed');
    }
  };

  const handleCancel = async (jobId: string) => {
    try {
      await apiClient.cancelRepoReport(jobId);
      toast.success('Report generation cancelled');
      fetchJobs();
    } catch (error) {
      console.error('Cancel failed:', error);
      toast.error('Cancel failed');
    }
  };

  const formatDate = (dateStr: string) => {
    return new Date(dateStr).toLocaleString();
  };

  const formatFileSize = (bytes?: number) => {
    if (!bytes) return '-';
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const getFormatIcon = (format: RepoReportOutputFormat) => {
    switch (format) {
      case 'docx':
        return (
          <span className="inline-flex items-center justify-center w-6 h-6 bg-blue-100 text-blue-700 rounded text-xs font-bold">
            W
          </span>
        );
      case 'pdf':
        return (
          <span className="inline-flex items-center justify-center w-6 h-6 bg-red-100 text-red-700 rounded text-xs font-bold">
            P
          </span>
        );
      case 'pptx':
        return (
          <span className="inline-flex items-center justify-center w-6 h-6 bg-orange-100 text-orange-700 rounded text-xs font-bold">
            S
          </span>
        );
    }
  };

  const getRepoIcon = (repoType: 'github' | 'gitlab') => {
    if (repoType === 'github') {
      return (
        <svg className="w-5 h-5 text-gray-700" fill="currentColor" viewBox="0 0 24 24">
          <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z" />
        </svg>
      );
    }
    return (
      <svg className="w-5 h-5 text-orange-600" fill="currentColor" viewBox="0 0 24 24">
        <path d="M22.65 14.39L12 22.13 1.35 14.39a.84.84 0 0 1-.3-.94l1.22-3.78 2.44-7.51A.42.42 0 0 1 4.82 2a.43.43 0 0 1 .58 0 .42.42 0 0 1 .11.18l2.44 7.49h8.1l2.44-7.51A.42.42 0 0 1 18.6 2a.43.43 0 0 1 .58 0 .42.42 0 0 1 .11.18l2.44 7.51L23 13.45a.84.84 0 0 1-.35.94z" />
      </svg>
    );
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Header */}
      <div className="flex justify-between items-center mb-8">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Repository Reports</h1>
          <p className="mt-1 text-sm text-gray-500">
            Generate comprehensive reports and presentations from GitHub/GitLab repositories
          </p>
        </div>
        <button
          onClick={() => setShowCreateModal(true)}
          className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 flex items-center"
        >
          <svg className="w-5 h-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
          </svg>
          Generate Report
        </button>
      </div>

      {/* Jobs List */}
      {isLoading ? (
        <div className="flex justify-center py-12">
          <svg className="animate-spin h-8 w-8 text-blue-600" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
          </svg>
        </div>
      ) : jobs.length === 0 ? (
        <div className="text-center py-12 bg-gray-50 rounded-lg">
          <svg className="mx-auto h-12 w-12 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
          </svg>
          <h3 className="mt-2 text-sm font-medium text-gray-900">No reports yet</h3>
          <p className="mt-1 text-sm text-gray-500">
            Generate your first repository report or presentation.
          </p>
          <div className="mt-6">
            <button
              onClick={() => setShowCreateModal(true)}
              className="inline-flex items-center px-4 py-2 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700"
            >
              Generate your first report
            </button>
          </div>
        </div>
      ) : (
        <div className="bg-white shadow sm:rounded-lg overflow-x-auto">
          <table className="w-full divide-y divide-gray-200 table-fixed">
            <thead className="bg-gray-50">
              <tr>
                <th className="w-1/4 px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Repository
                </th>
                <th className="w-16 px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Format
                </th>
                <th className="w-24 px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Status
                </th>
                <th className="w-36 px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Progress
                </th>
                <th className="w-20 px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Size
                </th>
                <th className="w-40 px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Created
                </th>
                <th className="w-24 px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Actions
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {jobs.map((job) => (
                <tr key={job.id} className="hover:bg-gray-50">
                  <td className="px-4 py-4">
                    <div className="flex items-center min-w-0">
                      <div className="flex-shrink-0">
                        {getRepoIcon(job.repo_type)}
                      </div>
                      <div className="ml-3 min-w-0 flex-1">
                        <div className="text-sm font-medium text-gray-900 truncate">
                          {job.title}
                        </div>
                        <div className="text-xs text-gray-500 truncate">
                          {job.repo_name}
                        </div>
                      </div>
                    </div>
                  </td>
                  <td className="px-4 py-4">
                    {getFormatIcon(job.output_format)}
                  </td>
                  <td className="px-4 py-4">
                    <span
                      className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium ${
                        STATUS_COLORS[job.status]
                      }`}
                    >
                      {job.status}
                    </span>
                  </td>
                  <td className="px-4 py-4">
                    {['pending', 'analyzing', 'generating', 'uploading'].includes(job.status) ? (
                      <div className="flex items-center">
                        <div className="flex-1 bg-gray-200 rounded-full h-2 mr-2">
                          <div
                            className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                            style={{ width: `${job.progress}%` }}
                          />
                        </div>
                        <span className="text-xs text-gray-600 w-8">{job.progress}%</span>
                      </div>
                    ) : job.status === 'failed' ? (
                      <span className="text-xs text-red-600 truncate block" title={job.error}>
                        {job.error?.substring(0, 30)}...
                      </span>
                    ) : (
                      <span className="text-xs text-gray-500">-</span>
                    )}
                  </td>
                  <td className="px-4 py-4 text-sm text-gray-500">
                    {formatFileSize(job.file_size)}
                  </td>
                  <td className="px-4 py-4 text-sm text-gray-500">
                    <span className="truncate block">{formatDate(job.created_at)}</span>
                  </td>
                  <td className="px-4 py-4 text-right text-sm font-medium">
                    <div className="flex justify-end space-x-1">
                      {job.status === 'completed' && (
                        <button
                          onClick={() => handleDownload(job)}
                          className="text-blue-600 hover:text-blue-900 p-1"
                          title="Download"
                        >
                          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                          </svg>
                        </button>
                      )}
                      {job.status === 'completed' && (
                        <button
                          onClick={() => handleCreateDraft(job)}
                          className="text-primary-600 hover:text-primary-800 p-1"
                          title="Create draft review"
                        >
                          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                          </svg>
                        </button>
                      )}
                      {['pending', 'analyzing', 'generating', 'uploading'].includes(job.status) && (
                        <button
                          onClick={() => handleCancel(job.id)}
                          className="text-yellow-600 hover:text-yellow-900 p-1"
                          title="Cancel"
                        >
                          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 9v6m4-6v6m7-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                          </svg>
                        </button>
                      )}
                      <button
                        onClick={() => handleDelete(job.id)}
                        className="text-red-600 hover:text-red-900 p-1"
                        title="Delete"
                      >
                        <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                        </svg>
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Create Modal */}
      <CreateReportModal
        isOpen={showCreateModal}
        onClose={() => setShowCreateModal(false)}
        onCreate={handleCreate}
        sources={sources}
        sections={sections}
        isLoading={isCreating}
      />
    </div>
  );
};

export default RepoReportsPage;
