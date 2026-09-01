/**
 * Document Synthesis Page
 *
 * Create multi-document summaries, comparative analyses, theme extractions,
 * and research reports from selected documents.
 */

import React, { useMemo, useRef, useState, useEffect } from 'react';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import { useLocation, useNavigate } from 'react-router-dom';
import {
  FileText,
  Plus,
  Trash2,
  Download,
  Copy,
  Eye,
  Clock,
  CheckCircle2,
  AlertCircle,
  Loader2,
  RefreshCw,
  Layers,
  BarChart3,
  BookOpen,
  Lightbulb,
  FileSearch,
  Briefcase,
  Target,
  X,
  Search,
  ChevronDown,
  ChevronUp,
  Save,
} from 'lucide-react';
import toast from 'react-hot-toast';
import { apiClient } from '../services/api';
import type { SynthesisJob, SynthesisJobType, SynthesisJobStatus, Document, AgentJob } from '../types';
import Button from '../components/common/Button';
import LoadingSpinner from '../components/common/LoadingSpinner';
import { useNotifications } from '../contexts/NotificationContext';

// Job type configuration
const JOB_TYPE_CONFIG: Record<SynthesisJobType, { icon: React.ComponentType<any>; label: string; color: string; description: string }> = {
  multi_doc_summary: {
    icon: Layers,
    label: 'Multi-Document Summary',
    color: 'text-blue-600 bg-blue-100',
    description: 'Synthesize multiple documents into one cohesive summary',
  },
  comparative_analysis: {
    icon: BarChart3,
    label: 'Comparative Analysis',
    color: 'text-purple-600 bg-purple-100',
    description: 'Compare and contrast documents to identify similarities and differences',
  },
  theme_extraction: {
    icon: Lightbulb,
    label: 'Theme Extraction',
    color: 'text-yellow-600 bg-yellow-100',
    description: 'Extract and analyze common themes across documents',
  },
  knowledge_synthesis: {
    icon: BookOpen,
    label: 'Knowledge Synthesis',
    color: 'text-green-600 bg-green-100',
    description: 'Synthesize knowledge from sources into new insights',
  },
  research_report: {
    icon: FileSearch,
    label: 'Research Report',
    color: 'text-indigo-600 bg-indigo-100',
    description: 'Generate formal research report from documents',
  },
  executive_brief: {
    icon: Briefcase,
    label: 'Executive Brief',
    color: 'text-orange-600 bg-orange-100',
    description: 'Create concise executive briefing for leadership',
  },
  decision_memo: {
    icon: FileText,
    label: 'Decision Memo',
    color: 'text-cyan-700 bg-cyan-100',
    description: 'Compare claims across sources, surface conflicts, and produce a short memo with citations',
  },
  gap_analysis_hypotheses: {
    icon: Target,
    label: 'Gap Analysis & Hypotheses',
    color: 'text-rose-600 bg-rose-100',
    description: 'Identify research gaps, propose testable hypotheses, and outline experiment plans',
  },
  hypothesis_reevaluation: {
    icon: RefreshCw,
    label: 'Hypothesis Re-evaluation',
    color: 'text-emerald-700 bg-emerald-100',
    description: 'Re-score and re-rank structured hypotheses using appended experiment evidence',
  },
  compiler_regression_explanation: {
    icon: FileSearch,
    label: 'Compiler Regression Explanation',
    color: 'text-sky-700 bg-sky-100',
    description: 'Compare two benchmark-backed compiler runs and explain the likely causes and next steps',
  },
  compiler_patch_proposal: {
    icon: Target,
    label: 'Compiler Patch Proposal',
    color: 'text-teal-700 bg-teal-100',
    description: 'Turn a compiler regression explanation into a bounded compiler-change proposal with validation and rollback guidance',
  },
  compiler_patch_draft: {
    icon: FileText,
    label: 'Compiler Patch Draft',
    color: 'text-violet-700 bg-violet-100',
    description: 'Turn a compiler patch proposal into a repo-aware draft with target files, symbols, validation commands, and rollback steps',
  },
};

// Status configuration
const STATUS_CONFIG: Record<SynthesisJobStatus, { color: string; bgColor: string; icon: React.ComponentType<any> }> = {
  pending: { color: 'text-yellow-700', bgColor: 'bg-yellow-100', icon: Clock },
  analyzing: { color: 'text-blue-700', bgColor: 'bg-blue-100', icon: Loader2 },
  synthesizing: { color: 'text-purple-700', bgColor: 'bg-purple-100', icon: Loader2 },
  generating: { color: 'text-indigo-700', bgColor: 'bg-indigo-100', icon: Loader2 },
  completed: { color: 'text-green-700', bgColor: 'bg-green-100', icon: CheckCircle2 },
  failed: { color: 'text-red-700', bgColor: 'bg-red-100', icon: AlertCircle },
  cancelled: { color: 'text-gray-700', bgColor: 'bg-gray-100', icon: X },
};

const DEFAULT_NOTE_TAGS: Record<SynthesisJobType, string[]> = {
  multi_doc_summary: ['summary', 'synthesis'],
  comparative_analysis: ['comparison', 'analysis'],
  theme_extraction: ['themes', 'analysis'],
  knowledge_synthesis: ['knowledge-synthesis', 'insights'],
  research_report: ['research-report', 'analysis'],
  executive_brief: ['executive-brief', 'briefing'],
  decision_memo: ['decision-memo', 'research-synthesis', 'citations'],
  gap_analysis_hypotheses: ['gap-analysis', 'hypotheses'],
  hypothesis_reevaluation: ['hypothesis-reevaluation', 'hypotheses'],
  compiler_regression_explanation: ['compiler-regression-explanation', 'performance-analysis'],
  compiler_patch_proposal: ['compiler-patch-proposal', 'compiler-proposal'],
  compiler_patch_draft: ['compiler-patch-draft', 'compiler-change-plan'],
};

const SynthesisPage: React.FC = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [selectedJob, setSelectedJob] = useState<SynthesisJob | null>(null);
  const [statusFilter, setStatusFilter] = useState<string>('');
  const [typeFilter, setTypeFilter] = useState<string>('');
  const queryClient = useQueryClient();
  const { fetchNotifications, refreshUnreadCount } = useNotifications();
  const focusedJobId = useMemo(() => new URLSearchParams(location.search).get('job'), [location.search]);

  // Fetch jobs
  const { data: jobsData, isLoading: jobsLoading, refetch: refetchJobs } = useQuery(
    ['synthesis-jobs', statusFilter, typeFilter],
    () => apiClient.listSynthesisJobs({
      status: statusFilter || undefined,
      job_type: typeFilter || undefined,
      page_size: 50,
    }),
    {
      refetchInterval: 5000, // Auto-refresh every 5 seconds
    }
  );

  // Mutations
  const deleteMutation = useMutation(
    (jobId: string) => apiClient.deleteSynthesisJob(jobId),
    {
      onSuccess: () => {
        queryClient.invalidateQueries(['synthesis-jobs']);
        toast.success('Job deleted');
        setSelectedJob(null);
      },
      onError: (error: any) => {
        toast.error(error.message || 'Delete failed');
      },
    }
  );

  const cancelMutation = useMutation(
    (jobId: string) => apiClient.cancelSynthesisJob(jobId),
    {
      onSuccess: () => {
        queryClient.invalidateQueries(['synthesis-jobs']);
        toast.success('Job cancelled');
      },
      onError: (error: any) => {
        toast.error(error.message || 'Cancel failed');
      },
    }
  );

  const dismissReviewMutation = useMutation(
    ({ jobId, outcomeNote }: { jobId: string; outcomeNote?: string }) =>
      apiClient.reviewSynthesisJob(jobId, { outcome_status: 'dismissed', outcome_note: outcomeNote }),
    {
      onSuccess: (job) => {
        queryClient.invalidateQueries(['synthesis-jobs']);
        queryClient.invalidateQueries(['research-notes']);
        void fetchNotifications();
        void refreshUnreadCount();
        if (job?.research_note_id) {
          queryClient.invalidateQueries(['synthesis-source-note', job.id, job.research_note_id]);
        }
        if (selectedJob?.research_note_id) {
          queryClient.invalidateQueries(['domain-research-profiles']);
          queryClient.invalidateQueries(['research-portfolios']);
        }
        toast.success('Reevaluation draft dismissed');
      },
      onError: (error: any) => {
        toast.error(error.message || 'Dismiss failed');
      },
    }
  );

  // Job card component
  const JobCard: React.FC<{ job: SynthesisJob }> = ({ job }) => {
    const typeConfig = JOB_TYPE_CONFIG[job.job_type] || JOB_TYPE_CONFIG.multi_doc_summary;
    const statusConfig = STATUS_CONFIG[job.status] || STATUS_CONFIG.pending;
    const StatusIcon = statusConfig.icon;
    const TypeIcon = typeConfig.icon;
    const isRunning = ['analyzing', 'synthesizing', 'generating'].includes(job.status);

    const sourceCount = job.research_note_id ? 1 : (job.paper_ids?.length ? job.paper_ids.length : job.document_ids.length);
    const sourceLabel = job.research_note_id ? 'note' : (job.paper_ids?.length ? 'papers' : 'docs');

    return (
      <div
        className={`bg-white border rounded-lg p-4 hover:shadow-md transition-shadow cursor-pointer ${
          selectedJob?.id === job.id ? 'border-primary-500 ring-2 ring-primary-200' : 'border-gray-200'
        }`}
        onClick={() => setSelectedJob(job)}
      >
        <div className="flex items-start justify-between mb-3">
          <div className="flex items-center gap-2">
            <div className={`p-2 rounded-lg ${typeConfig.color}`}>
              <TypeIcon className="w-4 h-4" />
            </div>
            <div>
              <h3 className="font-medium text-gray-900 truncate max-w-[200px]">{job.title}</h3>
              <p className="text-xs text-gray-500">{typeConfig.label}</p>
            </div>
          </div>
          <div className={`flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium ${statusConfig.bgColor} ${statusConfig.color}`}>
            <StatusIcon className={`w-3 h-3 ${isRunning ? 'animate-spin' : ''}`} />
            <span className="capitalize">{job.status}</span>
          </div>
        </div>

        {/* Progress bar */}
        <div className="mb-3">
          <div className="flex items-center justify-between text-xs text-gray-500 mb-1">
            <span>Progress</span>
            <span>{job.progress}%</span>
          </div>
          <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
            <div
              className={`h-full rounded-full transition-all ${
                job.status === 'completed' ? 'bg-green-500' :
                job.status === 'failed' ? 'bg-red-500' :
                'bg-primary-500'
              }`}
              style={{ width: `${job.progress}%` }}
            />
          </div>
        </div>

        {/* Current stage */}
        {job.current_stage && isRunning && (
          <p className="text-xs text-gray-600 mb-2 truncate">
            <span className="font-medium">Stage:</span> {job.current_stage}
          </p>
        )}

        {/* Stats row */}
        <div className="flex items-center gap-4 text-xs text-gray-500 mt-2">
          <span className="flex items-center gap-1">
            <FileText className="w-3 h-3" />
            {sourceCount} {sourceLabel}
          </span>
          {job.result_metadata?.word_count && (
            <span>{job.result_metadata.word_count} words</span>
          )}
          <span className="capitalize">{job.output_format}</span>
        </div>
      </div>
    );
  };

  useEffect(() => {
    if (!focusedJobId || !jobsData?.jobs?.length) return;
    const match = jobsData.jobs.find((job) => job.id === focusedJobId);
    if (match) setSelectedJob(match);
  }, [focusedJobId, jobsData?.jobs]);

  // Job detail panel
  const JobDetailPanel: React.FC<{ job: SynthesisJob }> = ({ job }) => {
    const [showFullContent, setShowFullContent] = useState(false);
    const typeConfig = JOB_TYPE_CONFIG[job.job_type] || JOB_TYPE_CONFIG.multi_doc_summary;
    const statusConfig = STATUS_CONFIG[job.status] || STATUS_CONFIG.pending;
    const StatusIcon = statusConfig.icon;
    const TypeIcon = typeConfig.icon;
    const isRunning = ['analyzing', 'synthesizing', 'generating'].includes(job.status);
    const contentContainerRef = useRef<HTMLDivElement | null>(null);
    const isReevaluationJob = job.job_type === 'hypothesis_reevaluation';
    const isCompilerExplanationJob = job.job_type === 'compiler_regression_explanation';
    const isCompilerPatchProposalJob = job.job_type === 'compiler_patch_proposal';
    const isCompilerPatchDraftJob = job.job_type === 'compiler_patch_draft';
    const isCompletedReevaluationDraft = isReevaluationJob && job.status === 'completed' && !!job.research_note_id && Boolean(job.can_apply);
    const reviewOutcomeStatus = String(job.review_outcome_status || '').trim();
    const isDismissedDraft = reviewOutcomeStatus === 'dismissed';
    const isAppliedDraft = reviewOutcomeStatus === 'applied_to_source_note';
    const isSavedAsNewNoteDraft = reviewOutcomeStatus === 'saved_as_new_note';
    const reviewTargetNoteId = String(job.review_target_note_id || '').trim();
    const { data: sourceNote } = useQuery(
      ['synthesis-source-note', job.id, job.research_note_id],
      () => apiClient.getResearchNote(job.research_note_id!),
      {
        enabled: Boolean(isReevaluationJob && job.research_note_id),
        staleTime: 5000,
      }
    );
    const primaryAutonomousOpportunityTarget = useMemo(() => {
      const hypotheses = Array.isArray(sourceNote?.structured_payload?.hypotheses) ? sourceNote.structured_payload.hypotheses : [];
      for (const hypothesis of hypotheses) {
        const origins = [
          hypothesis?.autonomous_origin,
          ...(Array.isArray(hypothesis?.experiment_evidence)
            ? hypothesis.experiment_evidence.map((row: any) => row?.autonomous_origin)
            : []),
        ];
        for (const origin of origins) {
          const sourceKind = String(origin?.source_kind || '').trim().toLowerCase();
          const sourceId = String(origin?.source_id || '').trim();
          const opportunityId = String(origin?.opportunity_id || '').trim();
          if ((sourceKind === 'profile' || sourceKind === 'portfolio') && sourceId && opportunityId) {
            return {
              sourceKind: sourceKind as 'profile' | 'portfolio',
              sourceId,
              opportunityId,
            };
          }
        }
      }
      return null;
    }, [sourceNote]);
    const reviewOutcomeLabel = useMemo(() => {
      if (isAppliedDraft) return 'Applied to source note';
      if (isSavedAsNewNoteDraft) return 'Saved as new note';
      if (isDismissedDraft) return 'Dismissed';
      return '';
    }, [isAppliedDraft, isDismissedDraft, isSavedAsNewNoteDraft]);
    const shouldShowSavedNoteLink = Boolean(
      reviewTargetNoteId
      && reviewTargetNoteId !== String(job.research_note_id || '').trim()
    );

    const reevaluationDiffRows = useMemo(() => {
      if (!isReevaluationJob) return [];
      const previousRows = Array.isArray(sourceNote?.structured_payload?.hypotheses) ? sourceNote!.structured_payload!.hypotheses! : [];
      const previousById = previousRows.reduce<Record<string, any>>((acc, hypothesis) => {
        const id = String(hypothesis?.id || '').trim();
        if (id) acc[id] = hypothesis;
        return acc;
      }, {});
      const updatedRows = Array.isArray(job.result_metadata?.structured_hypotheses) ? job.result_metadata!.structured_hypotheses! : [];
      const archivedIds = Array.isArray(job.result_metadata?.archived_hypothesis_ids) ? job.result_metadata!.archived_hypothesis_ids! : [];
      const priorityDeltas = Array.isArray(job.result_metadata?.priority_deltas) ? job.result_metadata!.priority_deltas! : [];

      const rows = updatedRows.map((hypothesis: any, index: number) => {
        const id = String(hypothesis?.id || '').trim() || `hypothesis-${index + 1}`;
        const previous = previousById[id];
        const delta = priorityDeltas.find((item: any) => String(item?.hypothesis_id || '').trim() === id) || null;
        return {
          id,
          title: String(hypothesis?.title || previous?.title || id),
          previous,
          next: hypothesis,
          delta,
          archived: archivedIds.includes(id),
        };
      });

      for (const previous of previousRows) {
        const id = String(previous?.id || '').trim();
        if (!id || rows.some((row) => row.id === id)) continue;
        rows.push({
          id,
          title: String(previous?.title || id),
          previous,
          next: null,
          delta: priorityDeltas.find((item: any) => String(item?.hypothesis_id || '').trim() === id) || null,
          archived: archivedIds.includes(id),
        });
      }

      return rows;
    }, [isReevaluationJob, sourceNote, job.result_metadata]);

    const sections = useMemo(() => {
      const content = job.result_content || '';
      const lines = content.split('\n');
      const parsed: Array<{ id: string; title: string; level: number; lineIndex: number }> = [];
      const seen = new Map<string, number>();

      const slugify = (s: string) =>
        s
          .toLowerCase()
          .trim()
          .replace(/[^\w\s-]/g, '')
          .replace(/\s+/g, '-')
          .replace(/-+/g, '-')
          .slice(0, 60);

      for (let i = 0; i < lines.length; i += 1) {
        const match = lines[i].match(/^(#{1,3})\s+(.+)\s*$/);
        if (!match) continue;
        const level = match[1].length;
        const title = match[2].trim();
        const base = slugify(title) || `section-${i}`;
        const n = (seen.get(base) || 0) + 1;
        seen.set(base, n);
        const id = n === 1 ? base : `${base}-${n}`;
        parsed.push({ id, title, level, lineIndex: i });
      }

      return parsed;
    }, [job.result_content]);

    const contentBlocks = useMemo(() => {
      const content = job.result_content || '';
      if (!content) return [];

      const lines = content.split('\n');
      const blocks: Array<{ heading?: { id: string; title: string; level: number }; body: string }> = [];

      const headingByLine = new Map<number, { id: string; title: string; level: number }>();
      for (const s of sections) {
        headingByLine.set(s.lineIndex, { id: s.id, title: s.title, level: s.level });
      }

      let currentHeading: { id: string; title: string; level: number } | undefined;
      let currentBody: string[] = [];

      const pushBlock = () => {
        if (currentHeading || currentBody.join('\n').trim()) {
          blocks.push({
            heading: currentHeading,
            body: currentBody.join('\n').trimEnd(),
          });
        }
      };

      for (let i = 0; i < lines.length; i += 1) {
        const heading = headingByLine.get(i);
        if (heading) {
          pushBlock();
          currentHeading = heading;
          currentBody = [];
          continue;
        }
        currentBody.push(lines[i]);
      }
      pushBlock();

      return blocks;
    }, [job.result_content, sections]);

    const handleDownload = async () => {
      try {
        await apiClient.downloadSynthesisResult(job.id, job.title);
        toast.success('Downloaded successfully');
      } catch (error: any) {
        toast.error(error.message || 'Download failed');
      }
    };

    const handleDownloadMarkdown = () => {
      if (!job.result_content) return;
      const blob = new Blob([job.result_content], { type: 'text/markdown;charset=utf-8' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      const safeTitle = (job.title || 'synthesis').replace(/[^\w\s-]/g, '').trim().replace(/\s+/g, '_');
      a.download = `${safeTitle || 'synthesis'}.md`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      toast.success('Markdown downloaded');
    };

    const handleCopy = async () => {
      if (!job.result_content) return;
      try {
        await navigator.clipboard.writeText(job.result_content);
        toast.success('Copied to clipboard');
      } catch (e: any) {
        toast.error(e?.message || 'Copy failed');
      }
    };

    const handleSaveAsResearchNote = async (targetNoteId?: string) => {
      if (!job.result_content) return;
      try {
        const suggestedTags = DEFAULT_NOTE_TAGS[job.job_type].join(', ');
        const tags = suggestedTags
          .split(',')
          .map((t) => t.trim())
          .filter(Boolean);
        await apiClient.saveSynthesisJobAsResearchNote(job.id, {
          title: job.title || JOB_TYPE_CONFIG[job.job_type].label,
          tags: tags.length > 0 ? tags : undefined,
          target_note_id: targetNoteId,
        });
        toast.success(targetNoteId ? 'Applied reevaluation to source note' : 'Saved as new note');
        queryClient.invalidateQueries(['synthesis-jobs']);
        queryClient.invalidateQueries(['research-notes']);
        void fetchNotifications();
        void refreshUnreadCount();
        if (job.research_note_id) {
          queryClient.invalidateQueries(['synthesis-source-note', job.id, job.research_note_id]);
        }
        if (primaryAutonomousOpportunityTarget?.sourceKind === 'profile') {
          queryClient.invalidateQueries(['domain-research-profiles']);
        }
        if (primaryAutonomousOpportunityTarget?.sourceKind === 'portfolio') {
          queryClient.invalidateQueries(['research-portfolios']);
        }
      } catch (e: any) {
        toast.error(e?.message || 'Failed to save note');
      }
    };

    const renderReevaluationHandoffActions = (includeSavedTarget: boolean) => {
      if (!sourceNote && !primaryAutonomousOpportunityTarget && !(includeSavedTarget && shouldShowSavedNoteLink)) {
        return null;
      }
      return (
        <div className="mb-2 flex flex-wrap gap-2">
          {sourceNote ? (
            <Button
              size="sm"
              variant="secondary"
              onClick={() => navigate(`/research-notes?note=${encodeURIComponent(sourceNote.id)}`)}
            >
              Open source note
            </Button>
          ) : null}
          {includeSavedTarget && shouldShowSavedNoteLink ? (
            <Button
              size="sm"
              variant="secondary"
              onClick={() => navigate(`/research-notes?note=${encodeURIComponent(reviewTargetNoteId)}`)}
            >
              Open saved note
            </Button>
          ) : null}
          {primaryAutonomousOpportunityTarget ? (
            <Button
              size="sm"
              variant="ghost"
              onClick={() => navigate(
                primaryAutonomousOpportunityTarget.sourceKind === 'profile'
                  ? `/autonomous-agents?tab=domain&profileId=${encodeURIComponent(primaryAutonomousOpportunityTarget.sourceId)}&opportunityId=${encodeURIComponent(primaryAutonomousOpportunityTarget.opportunityId)}`
                  : `/autonomous-agents?tab=fleet&fleetId=${encodeURIComponent(primaryAutonomousOpportunityTarget.sourceId)}&opportunityId=${encodeURIComponent(primaryAutonomousOpportunityTarget.opportunityId)}`
              )}
            >
              Open originating opportunity
            </Button>
          ) : null}
        </div>
      );
    };

    const jumpToSection = (id: string) => {
      const container = contentContainerRef.current;
      if (!container) return;
      const el = container.querySelector(`[data-section-id="${id}"]`) as HTMLElement | null;
      if (el) {
        el.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    };

    return (
      <div className="bg-white border border-gray-200 rounded-lg h-full overflow-hidden flex flex-col">
        {/* Header */}
        <div className="p-4 border-b border-gray-200">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-3">
              <div className={`p-2 rounded-lg ${typeConfig.color}`}>
                <TypeIcon className="w-5 h-5" />
              </div>
              <div>
                <h2 className="text-lg font-semibold">{job.title}</h2>
                <p className="text-sm text-gray-500">{typeConfig.label}</p>
              </div>
            </div>
            <div className={`flex items-center gap-1 px-3 py-1.5 rounded-full ${statusConfig.bgColor} ${statusConfig.color}`}>
              <StatusIcon className={`w-4 h-4 ${isRunning ? 'animate-spin' : ''}`} />
              <span className="font-medium capitalize">{job.status}</span>
            </div>
          </div>

          {/* Actions */}
          <div className="flex items-center gap-2 mt-3">
            {isRunning && (
              <Button
                size="sm"
                variant="secondary"
                onClick={() => cancelMutation.mutate(job.id)}
                disabled={cancelMutation.isLoading}
              >
                <X className="w-4 h-4 mr-1" />
                Cancel
              </Button>
            )}
            {isReevaluationJob && reviewOutcomeLabel ? (
              <span className="inline-flex items-center rounded-full bg-emerald-100 px-3 py-1 text-xs font-medium text-emerald-800">
                Outcome: {reviewOutcomeLabel}
              </span>
            ) : null}
            {isCompletedReevaluationDraft ? (
              <>
                <Button size="sm" variant="primary" onClick={() => handleSaveAsResearchNote(job.research_note_id || undefined)}>
                  <Save className="w-4 h-4 mr-1" />
                  Apply to source note
                </Button>
                <Button size="sm" variant="secondary" onClick={() => handleSaveAsResearchNote(undefined)}>
                  <Save className="w-4 h-4 mr-1" />
                  Save as new note
                </Button>
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={() => dismissReviewMutation.mutate({ jobId: job.id })}
                  disabled={dismissReviewMutation.isLoading}
                >
                  <X className="w-4 h-4 mr-1" />
                  Dismiss draft
                </Button>
              </>
            ) : job.status === 'completed' && job.result_content ? (
              <Button size="sm" variant="secondary" onClick={() => handleSaveAsResearchNote()}>
                <Save className="w-4 h-4 mr-1" />
                {job.job_type === 'decision_memo'
                  ? 'Save Memo Note'
                  : job.job_type === 'compiler_regression_explanation'
                    ? 'Save Explanation Note'
                    : job.job_type === 'compiler_patch_proposal'
                      ? 'Save Proposal Note'
                      : job.job_type === 'compiler_patch_draft'
                        ? 'Save Patch Draft Note'
                      : 'Save Note'}
              </Button>
            ) : null}
            {job.status === 'completed' && job.result_content && (
              <Button size="sm" variant="secondary" onClick={handleDownloadMarkdown}>
                <Download className="w-4 h-4 mr-1" />
                Download MD
              </Button>
            )}
            {job.status === 'completed' && job.result_content && (
              <Button size="sm" variant="secondary" onClick={handleCopy}>
                <Copy className="w-4 h-4 mr-1" />
                Copy
              </Button>
            )}
            {job.status === 'completed' && job.file_path && (
              <Button
                size="sm"
                variant="primary"
                onClick={handleDownload}
              >
                <Download className="w-4 h-4 mr-1" />
                Download {job.output_format.toUpperCase()}
              </Button>
            )}
            <Button
              size="sm"
              variant="ghost"
              onClick={() => {
                if (window.confirm('Are you sure you want to delete this job?')) {
                  deleteMutation.mutate(job.id);
                }
              }}
              disabled={isRunning || deleteMutation.isLoading}
            >
              <Trash2 className="w-4 h-4 mr-1" />
              Delete
            </Button>
          </div>
          {isReevaluationJob && reviewOutcomeLabel ? (
            <div className="mt-3 rounded border border-gray-200 bg-gray-50 px-3 py-2 text-xs text-gray-700">
              <div className="font-medium text-gray-900">Review outcome</div>
              <div className="mt-1">
                {reviewOutcomeLabel}
                {job.review_recorded_at ? ` · ${new Date(job.review_recorded_at).toLocaleString()}` : ''}
              </div>
              {job.review_note ? <div className="mt-1 text-gray-600">{job.review_note}</div> : null}
              <div className="mt-2">
                {renderReevaluationHandoffActions(true)}
              </div>
            </div>
          ) : null}
        </div>

        {/* Content */}
        <div ref={contentContainerRef} className="flex-1 overflow-y-auto p-4">
          {/* Progress */}
          <div className="mb-4">
            <h3 className="text-sm font-medium text-gray-700 mb-2">Progress</h3>
            <div className="h-3 bg-gray-200 rounded-full overflow-hidden">
              <div
                className={`h-full rounded-full transition-all ${
                  job.status === 'completed' ? 'bg-green-500' :
                  job.status === 'failed' ? 'bg-red-500' :
                  'bg-primary-500'
                }`}
                style={{ width: `${job.progress}%` }}
              />
            </div>
            <div className="flex justify-between text-xs text-gray-500 mt-1">
              <span>{job.progress}% complete</span>
              {job.current_stage && <span>{job.current_stage}</span>}
            </div>
          </div>

          {/* Topic */}
          {job.topic && (
            <div className="mb-4">
              <h3 className="text-sm font-medium text-gray-700 mb-1">Topic</h3>
              <p className="text-sm text-gray-600 bg-gray-50 rounded-lg p-3">{job.topic}</p>
            </div>
          )}

          {/* Research Hub options */}
          {job.job_type === 'gap_analysis_hypotheses' && job.options && (
            <div className="mb-4">
              <h3 className="text-sm font-medium text-gray-700 mb-2">Research Hub Options</h3>
              <div className="text-sm text-gray-700 bg-gray-50 rounded-lg p-3 space-y-1">
                {job.options.domain && (
                  <p>
                    <span className="text-gray-500">Domain:</span> {String(job.options.domain)}
                  </p>
                )}
                {job.options.constraints && (
                  <p className="whitespace-pre-wrap">
                    <span className="text-gray-500">Constraints:</span> {String(job.options.constraints)}
                  </p>
                )}
                {job.options.desired_outcomes && (
                  <p className="whitespace-pre-wrap">
                    <span className="text-gray-500">Desired Outcomes:</span> {String(job.options.desired_outcomes)}
                  </p>
                )}
                {job.options.include_bibliography !== undefined && (
                  <p>
                    <span className="text-gray-500">Include bibliography:</span>{' '}
                    {job.options.include_bibliography ? 'Yes' : 'No'}
                  </p>
                )}
              </div>
            </div>
          )}

          {/* Decision memo options */}
          {job.job_type === 'decision_memo' && job.options && (
            <div className="mb-4">
              <h3 className="text-sm font-medium text-gray-700 mb-2">Decision Memo Options</h3>
              <div className="text-sm text-gray-700 bg-gray-50 rounded-lg p-3 space-y-1">
                {job.options.audience && (
                  <p>
                    <span className="text-gray-500">Audience:</span>{' '}
                    {String(job.options.audience).replace(/_/g, ' ')}
                  </p>
                )}
                {job.options.include_recommendations !== undefined && (
                  <p>
                    <span className="text-gray-500">Recommended actions:</span>{' '}
                    {job.options.include_recommendations ? 'Included' : 'Omitted'}
                  </p>
                )}
                {job.options.include_watchlist !== undefined && (
                  <p>
                    <span className="text-gray-500">Watchlist:</span>{' '}
                    {job.options.include_watchlist ? 'Included' : 'Omitted'}
                  </p>
                )}
              </div>
            </div>
          )}

          {/* Documents */}
          <div className="mb-4">
            <h3 className="text-sm font-medium text-gray-700 mb-2">
              {job.paper_ids?.length ? `Papers (${job.paper_ids.length})` : `Documents (${job.document_ids.length})`}
            </h3>
            <div className="text-xs text-gray-500 bg-gray-50 rounded-lg p-3 max-h-24 overflow-y-auto">
              {(job.paper_ids?.length ? job.paper_ids : job.document_ids).map((id, idx) => (
                <div key={id} className="truncate">{id}</div>
              ))}
            </div>
          </div>

          {/* Metadata */}
          {job.result_metadata && (
            <div className="mb-4">
              <h3 className="text-sm font-medium text-gray-700 mb-2">Results</h3>
              <div className="grid grid-cols-2 gap-3">
                {job.result_metadata.word_count !== undefined && (
                  <div className="bg-gray-50 rounded-lg p-3 text-center">
                    <p className="text-xs text-gray-500">Word Count</p>
                    <p className="text-lg font-semibold">{job.result_metadata.word_count}</p>
                  </div>
                )}
                {job.result_metadata.documents_analyzed !== undefined && (
                  <div className="bg-gray-50 rounded-lg p-3 text-center">
                    <p className="text-xs text-gray-500">Docs Analyzed</p>
                    <p className="text-lg font-semibold">{job.result_metadata.documents_analyzed}</p>
                  </div>
                )}
                {job.paper_ids?.length ? (
                  <div className="bg-gray-50 rounded-lg p-3 text-center">
                    <p className="text-xs text-gray-500">Selected Papers</p>
                    <p className="text-lg font-semibold">{job.paper_ids.length}</p>
                  </div>
                ) : null}
                {job.job_type === 'decision_memo' && job.result_metadata.audience && (
                  <div className="bg-gray-50 rounded-lg p-3 text-center">
                    <p className="text-xs text-gray-500">Audience</p>
                    <p className="text-sm font-semibold capitalize">
                      {String(job.result_metadata.audience).replace(/_/g, ' ')}
                    </p>
                  </div>
                )}
              </div>

              {/* Themes found */}
              {job.result_metadata.themes_found && job.result_metadata.themes_found.length > 0 && (
                <div className="mt-3">
                  <p className="text-xs text-gray-500 mb-1">Themes Found</p>
                  <div className="flex flex-wrap gap-1">
                    {job.result_metadata.themes_found.map((theme, idx) => (
                      <span key={idx} className="text-xs bg-primary-100 text-primary-700 px-2 py-1 rounded">
                        {theme}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {isCompletedReevaluationDraft && sourceNote && (
            <div className="mb-4">
              {renderReevaluationHandoffActions(false)}
              <h3 className="text-sm font-medium text-gray-700 mb-2">Changes vs source note</h3>
              <div className="space-y-2">
                {reevaluationDiffRows.map((row) => {
                  const previousRank = row.previous?.rank;
                  const nextRank = row.next?.rank;
                  const previousOverall = row.previous?.overall_score;
                  const nextOverall = row.next?.overall_score;
                  const previousEvidence = row.previous?.evidence_score;
                  const nextEvidence = row.next?.evidence_score;
                  const previousTestability = row.previous?.testability_score;
                  const nextTestability = row.next?.testability_score;
                  return (
                    <div key={row.id} className="rounded border border-gray-200 bg-gray-50 p-3">
                      <div className="flex items-center justify-between gap-2">
                        <div className="font-medium text-gray-900">{row.title}</div>
                        <div className="text-[11px] text-gray-600">
                          {row.archived
                            ? 'Archived'
                            : row.previous && row.next
                              ? 'Updated'
                              : row.next
                                ? 'New'
                                : 'Removed'}
                        </div>
                      </div>
                      <div className="mt-1 text-xs text-gray-700">
                        Rank {previousRank ?? '—'} → {nextRank ?? '—'}
                        {' · '}
                        Overall {previousOverall ?? '—'} → {nextOverall ?? '—'}
                        {' · '}
                        Evidence {previousEvidence ?? '—'} → {nextEvidence ?? '—'}
                        {' · '}
                        Testability {previousTestability ?? '—'} → {nextTestability ?? '—'}
                      </div>
                      {row.delta?.reason ? (
                        <div className="mt-1 text-xs text-gray-600">{String(row.delta.reason)}</div>
                      ) : null}
                      {row.next?.recommended_next_step && row.next?.recommended_next_step !== row.previous?.recommended_next_step ? (
                        <div className="mt-1 text-xs text-gray-700">
                          Next step: {String(row.next.recommended_next_step)}
                        </div>
                      ) : null}
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {isCompilerExplanationJob && job.result_metadata && (
            <div className="mb-4">
              <h3 className="text-sm font-medium text-gray-700 mb-2">Compiler Regression Explanation</h3>
              <div className="rounded border border-sky-200 bg-sky-50 p-4 space-y-3">
                {job.result_metadata.summary ? (
                  <div className="text-sm text-sky-900">{String(job.result_metadata.summary)}</div>
                ) : null}
                <div className="text-xs text-sky-800">
                  {job.result_metadata.regression_type ? `Type: ${String(job.result_metadata.regression_type)}` : 'Type: mixed'}
                  {job.result_metadata.primary_run_id ? ` · Primary ${String(job.result_metadata.primary_run_id)}` : ''}
                  {job.result_metadata.comparison_run_id ? ` · Comparison ${String(job.result_metadata.comparison_run_id)}` : ''}
                </div>
                {Array.isArray(job.result_metadata.metric_deltas) && job.result_metadata.metric_deltas.length > 0 ? (
                  <div>
                    <div className="text-xs font-medium uppercase tracking-wide text-sky-700">Metric Deltas</div>
                    <ul className="mt-1 space-y-1 text-sm text-sky-900">
                      {job.result_metadata.metric_deltas.slice(0, 6).map((item: any, idx: number) => (
                        <li key={`metric-delta-${idx}`}>
                          {String(item?.metric || 'metric')}: {String(item?.comparison ?? '?')} → {String(item?.primary ?? '?')}
                          {item?.interpretation ? ` · ${String(item.interpretation)}` : ''}
                        </li>
                      ))}
                    </ul>
                  </div>
                ) : null}
                {Array.isArray(job.result_metadata.artifact_deltas) && job.result_metadata.artifact_deltas.length > 0 ? (
                  <div>
                    <div className="text-xs font-medium uppercase tracking-wide text-sky-700">Artifact Deltas</div>
                    <ul className="mt-1 space-y-1 text-sm text-sky-900">
                      {job.result_metadata.artifact_deltas.slice(0, 5).map((item: any, idx: number) => (
                        <li key={`artifact-delta-${idx}`}>
                          {String(item?.kind || 'artifact')}: {String(item?.summary || '')}
                        </li>
                      ))}
                    </ul>
                  </div>
                ) : null}
                {Array.isArray(job.result_metadata.likely_causes) && job.result_metadata.likely_causes.length > 0 ? (
                  <div>
                    <div className="text-xs font-medium uppercase tracking-wide text-sky-700">Likely Causes</div>
                    <ul className="mt-1 space-y-1 text-sm text-sky-900">
                      {job.result_metadata.likely_causes.slice(0, 5).map((item: any, idx: number) => (
                        <li key={`likely-cause-${idx}`}>
                          {String(item?.title || 'Cause')}
                          {item?.confidence ? ` [${String(item.confidence)}]` : ''}
                          {item?.reason ? ` · ${String(item.reason)}` : ''}
                        </li>
                      ))}
                    </ul>
                  </div>
                ) : null}
                {Array.isArray(job.result_metadata.confounders) && job.result_metadata.confounders.length > 0 ? (
                  <div>
                    <div className="text-xs font-medium uppercase tracking-wide text-sky-700">Confounders</div>
                    <ul className="mt-1 space-y-1 text-sm text-sky-900">
                      {job.result_metadata.confounders.slice(0, 5).map((item: any, idx: number) => (
                        <li key={`confounder-${idx}`}>{String(item)}</li>
                      ))}
                    </ul>
                  </div>
                ) : null}
                {Array.isArray(job.result_metadata.recommended_next_steps) && job.result_metadata.recommended_next_steps.length > 0 ? (
                  <div>
                    <div className="text-xs font-medium uppercase tracking-wide text-sky-700">Recommended Next Steps</div>
                    <ul className="mt-1 space-y-1 text-sm text-sky-900">
                      {job.result_metadata.recommended_next_steps.slice(0, 5).map((item: any, idx: number) => (
                        <li key={`next-step-${idx}`}>{String(item)}</li>
                      ))}
                    </ul>
                  </div>
                ) : null}
              </div>
            </div>
          )}

          {isCompilerPatchProposalJob && job.result_metadata && (
            <div className="mb-4">
              <h3 className="text-sm font-medium text-gray-700 mb-2">Compiler Patch Proposal</h3>
              <div className="rounded border border-teal-200 bg-teal-50 p-4 space-y-3">
                {job.result_metadata.proposal_summary ? (
                  <div className="text-sm text-teal-900">{String(job.result_metadata.proposal_summary)}</div>
                ) : null}
                <div className="text-xs text-teal-800">
                  {job.result_metadata.target_area ? `Target area: ${String(job.result_metadata.target_area)}` : 'Target area: compiler'}
                  {job.result_metadata.source_explanation_note_id ? ` · Source note ${String(job.result_metadata.source_explanation_note_id)}` : ''}
                </div>
                {job.result_metadata.candidate_change ? (
                  <div>
                    <div className="text-xs font-medium uppercase tracking-wide text-teal-700">Candidate Change</div>
                    <div className="mt-1 text-sm text-teal-900 whitespace-pre-wrap">{String(job.result_metadata.candidate_change)}</div>
                  </div>
                ) : null}
                {job.result_metadata.expected_effect ? (
                  <div>
                    <div className="text-xs font-medium uppercase tracking-wide text-teal-700">Expected Effect</div>
                    <div className="mt-1 text-sm text-teal-900 whitespace-pre-wrap">{String(job.result_metadata.expected_effect)}</div>
                  </div>
                ) : null}
                {job.result_metadata.mechanism ? (
                  <div>
                    <div className="text-xs font-medium uppercase tracking-wide text-teal-700">Mechanism</div>
                    <div className="mt-1 text-sm text-teal-900 whitespace-pre-wrap">{String(job.result_metadata.mechanism)}</div>
                  </div>
                ) : null}
                {job.result_metadata.validation_plan ? (
                  <div>
                    <div className="text-xs font-medium uppercase tracking-wide text-teal-700">Validation Plan</div>
                    <div className="mt-1 text-sm text-teal-900 whitespace-pre-wrap">{String(job.result_metadata.validation_plan)}</div>
                  </div>
                ) : null}
                {job.result_metadata.risk_assessment ? (
                  <div>
                    <div className="text-xs font-medium uppercase tracking-wide text-teal-700">Risk Assessment</div>
                    <div className="mt-1 text-sm text-teal-900 whitespace-pre-wrap">{String(job.result_metadata.risk_assessment)}</div>
                  </div>
                ) : null}
                {job.result_metadata.rollback_or_guardrail ? (
                  <div>
                    <div className="text-xs font-medium uppercase tracking-wide text-teal-700">Rollback Or Guardrail</div>
                    <div className="mt-1 text-sm text-teal-900 whitespace-pre-wrap">{String(job.result_metadata.rollback_or_guardrail)}</div>
                  </div>
                ) : null}
              </div>
            </div>
          )}

          {isCompilerPatchDraftJob && job.result_metadata && (
            <div className="mb-4">
              <h3 className="text-sm font-medium text-gray-700 mb-2">Compiler Patch Draft</h3>
              <div className="rounded border border-violet-200 bg-violet-50 p-4 space-y-3">
                {job.result_metadata.draft_summary ? (
                  <div className="text-sm text-violet-900">{String(job.result_metadata.draft_summary)}</div>
                ) : null}
                <div className="text-xs text-violet-800">
                  {job.result_metadata.source_name ? `Repo source: ${String(job.result_metadata.source_name)}` : 'Repo source selected'}
                  {(job.result_metadata as { source_id?: string | null }).source_id
                    ? ` · ${String((job.result_metadata as { source_id?: string | null }).source_id)}`
                    : ''}
                </div>
                {Array.isArray(job.result_metadata.target_files) && job.result_metadata.target_files.length > 0 ? (
                  <div>
                    <div className="text-xs font-medium uppercase tracking-wide text-violet-700">Target Files</div>
                    <ul className="mt-1 space-y-1 text-sm text-violet-900">
                      {job.result_metadata.target_files.slice(0, 8).map((item: any, idx: number) => (
                        <li key={`draft-file-${idx}`}>{String(item)}</li>
                      ))}
                    </ul>
                  </div>
                ) : null}
                {Array.isArray(job.result_metadata.target_symbols) && job.result_metadata.target_symbols.length > 0 ? (
                  <div>
                    <div className="text-xs font-medium uppercase tracking-wide text-violet-700">Target Symbols</div>
                    <div className="mt-1 text-sm text-violet-900">
                      {job.result_metadata.target_symbols.slice(0, 10).map((item: any) => String(item)).join(', ')}
                    </div>
                  </div>
                ) : null}
                {Array.isArray(job.result_metadata.change_plan) && job.result_metadata.change_plan.length > 0 ? (
                  <div>
                    <div className="text-xs font-medium uppercase tracking-wide text-violet-700">Change Plan</div>
                    <ul className="mt-1 space-y-1 text-sm text-violet-900">
                      {job.result_metadata.change_plan.slice(0, 10).map((item: any, idx: number) => (
                        <li key={`draft-step-${idx}`}>{String(item)}</li>
                      ))}
                    </ul>
                  </div>
                ) : null}
                {Array.isArray(job.result_metadata.validation_commands) && job.result_metadata.validation_commands.length > 0 ? (
                  <div>
                    <div className="text-xs font-medium uppercase tracking-wide text-violet-700">Validation Commands</div>
                    <ul className="mt-1 space-y-1 text-sm text-violet-900">
                      {job.result_metadata.validation_commands.slice(0, 8).map((item: any, idx: number) => (
                        <li key={`draft-validation-${idx}`}>{String(item)}</li>
                      ))}
                    </ul>
                  </div>
                ) : null}
                {Array.isArray(job.result_metadata.rollback_steps) && job.result_metadata.rollback_steps.length > 0 ? (
                  <div>
                    <div className="text-xs font-medium uppercase tracking-wide text-violet-700">Rollback Steps</div>
                    <ul className="mt-1 space-y-1 text-sm text-violet-900">
                      {job.result_metadata.rollback_steps.slice(0, 8).map((item: any, idx: number) => (
                        <li key={`draft-rollback-${idx}`}>{String(item)}</li>
                      ))}
                    </ul>
                  </div>
                ) : null}
              </div>
            </div>
          )}

          {/* Error */}
          {job.error && (
            <div className="mb-4">
              <h3 className="text-sm font-medium text-red-700 mb-1">Error</h3>
              <p className="text-sm text-red-600 bg-red-50 rounded-lg p-3">{job.error}</p>
            </div>
          )}

          {/* Content preview */}
          {job.result_content && (
            <div className="mb-4">
              <div className="flex items-center justify-between mb-2">
                <h3 className="text-sm font-medium text-gray-700">Content Preview</h3>
                <div className="flex items-center gap-2">
                  {job.job_type === 'gap_analysis_hypotheses' && sections.length > 0 && (
                    <select
                      className="border border-gray-300 rounded-lg px-2 py-1 text-xs"
                      defaultValue=""
                      onChange={(e) => {
                        const v = e.target.value;
                        if (v) jumpToSection(v);
                        e.target.value = '';
                      }}
                    >
                      <option value="">Jump to section…</option>
                      {sections.map((s) => (
                        <option key={s.id} value={s.id}>
                          {`${'  '.repeat(Math.max(0, s.level - 1))}${s.title}`}
                        </option>
                      ))}
                    </select>
                  )}
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={() => setShowFullContent(!showFullContent)}
                  >
                    {showFullContent ? (
                      <>
                        <ChevronUp className="w-3 h-3 mr-1" />
                        Collapse
                      </>
                    ) : (
                      <>
                        <ChevronDown className="w-3 h-3 mr-1" />
                        Expand
                      </>
                    )}
                  </Button>
                </div>
              </div>
              <div
                className={`bg-gray-50 rounded-lg p-3 text-sm overflow-hidden ${
                  showFullContent ? 'max-h-96 overflow-y-auto' : 'max-h-48'
                }`}
              >
                {job.job_type === 'gap_analysis_hypotheses' ? (
                  <div className="space-y-3">
                    {contentBlocks.map((b, idx) => (
                      <div
                        key={`${b.heading?.id || 'no-heading'}-${idx}`}
                        data-section-id={b.heading?.id || undefined}
                      >
                        {b.heading && (
                          <div
                            className={`font-semibold text-gray-900 ${
                              b.heading.level === 1
                                ? 'text-base'
                                : b.heading.level === 2
                                ? 'text-sm'
                                : 'text-sm'
                            }`}
                          >
                            {b.heading.title}
                          </div>
                        )}
                        {b.body && (
                          <pre className="whitespace-pre-wrap font-sans text-gray-700">
                            {b.body}
                          </pre>
                        )}
                      </div>
                    ))}
                  </div>
                ) : (
                  <pre className="whitespace-pre-wrap font-sans text-gray-700">
                    {job.result_content}
                  </pre>
                )}
              </div>
            </div>
          )}

          {/* Timestamps */}
          <div className="text-xs text-gray-500 space-y-1">
            <p>Created: {job.created_at ? new Date(job.created_at).toLocaleString() : '-'}</p>
            {job.started_at && <p>Started: {new Date(job.started_at).toLocaleString()}</p>}
            {job.completed_at && <p>Completed: {new Date(job.completed_at).toLocaleString()}</p>}
          </div>
        </div>
      </div>
    );
  };

  // Create modal
  const CreateModal: React.FC = () => {
    const [step, setStep] = useState<'type' | 'documents' | 'config'>('type');
    const [selectedType, setSelectedType] = useState<SynthesisJobType>('multi_doc_summary');
    const [selectedDocs, setSelectedDocs] = useState<string[]>([]);
    const [selectedRuns, setSelectedRuns] = useState<string[]>([]);
    const [searchQuery, setSearchQuery] = useState('');
    const [sourceSearchQuery, setSourceSearchQuery] = useState('');
    const [title, setTitle] = useState('');
    const [autoTitle, setAutoTitle] = useState(true);
    const [topic, setTopic] = useState('');
    const [outputFormat, setOutputFormat] = useState<'markdown' | 'docx' | 'pdf' | 'pptx'>('markdown');
    const [outputStyle, setOutputStyle] = useState('professional');
    const [isSubmitting, setIsSubmitting] = useState(false);
    // Research Hub options (Gap Analysis & Hypotheses)
    const [domain, setDomain] = useState('compilers');
    const [constraints, setConstraints] = useState('');
    const [desiredOutcomes, setDesiredOutcomes] = useState('');
    const [includeBibliography, setIncludeBibliography] = useState(true);
    const [decisionAudience, setDecisionAudience] = useState('product_and_strategy');
    const [includeDecisionRecommendations, setIncludeDecisionRecommendations] = useState(true);
    const [includeDecisionWatchlist, setIncludeDecisionWatchlist] = useState(true);

    useEffect(() => {
      if (autoTitle) {
        setTitle(JOB_TYPE_CONFIG[selectedType].label);
      }

      if (selectedType === 'gap_analysis_hypotheses') {
        // Default to a more technical tone for research
        if (outputStyle === 'professional') {
          setOutputStyle('technical');
        }
      } else if (selectedType === 'decision_memo') {
        if (outputStyle === 'casual') {
          setOutputStyle('professional');
        }
      }
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [selectedType]);

    // Fetch documents for selection
    const { data: documentsData, isLoading: docsLoading } = useQuery(
      ['documents-for-synthesis', searchQuery],
      () => apiClient.getDocuments({ limit: 50, search: searchQuery || undefined }),
      { enabled: step === 'documents' }
    );

    // Completed runs only: a run still going has findings that will change,
    // and a document citing a moving number is worse than one citing none.
    const { data: runsData, isLoading: runsLoading } = useQuery(
      ['agent-runs-for-synthesis'],
      () => apiClient.listAgentJobs({ status: 'completed', page_size: 25, sort_by: 'recent' }),
      { enabled: step === 'documents' }
    );

    const handleSubmit = async () => {
      if (selectedDocs.length === 0 && selectedRuns.length === 0 && !sourceSearchQuery.trim()) {
        toast.error('Select a document or a run, or provide a search query');
        return;
      }
      if (!title.trim()) {
        toast.error('Enter a title');
        return;
      }

      setIsSubmitting(true);
      try {
        await apiClient.createSynthesisJob({
          job_type: selectedType,
          title,
          document_ids: selectedDocs,
          agent_job_ids: selectedRuns.length ? selectedRuns : undefined,
          search_query: sourceSearchQuery.trim() || undefined,
          topic: topic || undefined,
          output_format: outputFormat,
          output_style: outputStyle,
          options:
            selectedType === 'gap_analysis_hypotheses'
              ? {
                  domain: domain.trim() || undefined,
                  constraints: constraints.trim() || undefined,
                  desired_outcomes: desiredOutcomes.trim() || undefined,
                  include_bibliography: includeBibliography,
                }
              : selectedType === 'decision_memo'
                ? {
                    audience: decisionAudience,
                    include_recommendations: includeDecisionRecommendations,
                    include_watchlist: includeDecisionWatchlist,
                  }
              : undefined,
        });
        toast.success('Synthesis job created');
        queryClient.invalidateQueries(['synthesis-jobs']);
        setShowCreateModal(false);
      } catch (error: any) {
        toast.error(error.message || 'Failed to create job');
      } finally {
        setIsSubmitting(false);
      }
    };

    return (
      <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
        <div className="bg-white rounded-lg shadow-xl w-full max-w-2xl max-h-[90vh] overflow-hidden flex flex-col">
          <div className="p-6 border-b border-gray-200">
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-semibold">Create Synthesis</h2>
              <Button variant="ghost" size="sm" onClick={() => setShowCreateModal(false)}>
                <X className="w-5 h-5" />
              </Button>
            </div>

            {/* Step indicators */}
            <div className="flex items-center gap-4 mt-4">
              {['type', 'documents', 'config'].map((s, idx) => (
                <div
                  key={s}
                  className={`flex items-center gap-2 ${step === s ? 'text-primary-600' : 'text-gray-400'}`}
                >
                  <div
                    className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-medium ${
                      step === s ? 'bg-primary-600 text-white' : 'bg-gray-200'
                    }`}
                  >
                    {idx + 1}
                  </div>
                  <span className="text-sm capitalize">{s}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="flex-1 overflow-y-auto p-6">
            {/* Step 1: Select type */}
            {step === 'type' && (
              <div className="grid grid-cols-2 gap-3">
                {Object.entries(JOB_TYPE_CONFIG)
                  .filter(([type]) => type !== 'hypothesis_reevaluation' && type !== 'compiler_patch_proposal' && type !== 'compiler_patch_draft')
                  .map(([type, config]) => {
                  const Icon = config.icon;
                  return (
                    <div
                      key={type}
                      className={`border rounded-lg p-4 cursor-pointer transition-colors ${
                        selectedType === type
                          ? 'border-primary-500 bg-primary-50'
                          : 'border-gray-200 hover:border-gray-300'
                      }`}
                      onClick={() => setSelectedType(type as SynthesisJobType)}
                    >
                      <div className="flex items-start gap-3">
                        <div className={`p-2 rounded-lg ${config.color}`}>
                          <Icon className="w-5 h-5" />
                        </div>
                        <div>
                          <h3 className="font-medium text-gray-900">{config.label}</h3>
                          <p className="text-xs text-gray-500 mt-1">{config.description}</p>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}

            {/* Step 2: Select documents */}
            {step === 'documents' && (
              <div>
                <div className="mb-4">
                  <div className="relative">
                    <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" />
                    <input
                      type="text"
                      placeholder="Search documents..."
                      className="w-full border border-gray-300 rounded-lg pl-10 pr-4 py-2 text-sm bg-white text-gray-900 placeholder-gray-400"
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                    />
                  </div>
                </div>

                <p className="text-sm text-gray-500 mb-3">
                  Selected: {selectedDocs.length} documents
                </p>

                {/* Runs as sources. A run's findings reach the synthesis with
                    their measurements attached, so a number in the document is
                    the one the run recorded rather than one typed in again. */}
                <div className="mb-4">
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Or build on what a run measured
                  </label>
                  {runsLoading ? (
                    <div className="flex justify-center py-4">
                      <LoadingSpinner />
                    </div>
                  ) : (runsData?.jobs || []).length === 0 ? (
                    <p className="text-xs text-gray-500">
                      No completed runs yet. A run's findings become citable here once it finishes.
                    </p>
                  ) : (
                    <div className="space-y-2 max-h-48 overflow-y-auto">
                      {(runsData?.jobs || []).map((run: AgentJob) => (
                        <label
                          key={run.id}
                          className={`flex items-center gap-3 p-3 border rounded-lg cursor-pointer ${
                            selectedRuns.includes(run.id)
                              ? 'border-primary-500 bg-primary-50'
                              : 'border-gray-200 hover:bg-gray-50'
                          }`}
                        >
                          <input
                            type="checkbox"
                            checked={selectedRuns.includes(run.id)}
                            onChange={(e) => {
                              if (e.target.checked) {
                                setSelectedRuns([...selectedRuns, run.id]);
                              } else {
                                setSelectedRuns(selectedRuns.filter((id) => id !== run.id));
                              }
                            }}
                            className="rounded"
                          />
                          <div className="flex-1 min-w-0">
                            <p className="font-medium text-gray-900 truncate">{run.name}</p>
                            <p className="text-xs text-gray-500 truncate">
                              {run.job_type}
                              {typeof run.iteration === 'number' ? ` | ${run.iteration} iterations` : ''}
                            </p>
                          </div>
                        </label>
                      ))}
                    </div>
                  )}
                </div>

                <div className="mb-4">
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Or search existing knowledge base
                  </label>
                  <input
                    type="text"
                    className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm bg-white text-gray-900 placeholder-gray-400"
                    value={sourceSearchQuery}
                    onChange={(e) => setSourceSearchQuery(e.target.value)}
                    placeholder="Optional query to pull matching documents automatically"
                  />
                  <p className="text-xs text-gray-500 mt-1">
                    Useful for decision memos when you want to synthesize by topic instead of manually selecting every document.
                  </p>
                </div>

                {docsLoading ? (
                  <div className="flex justify-center py-8">
                    <LoadingSpinner />
                  </div>
                ) : (
                  <div className="space-y-2 max-h-64 overflow-y-auto">
                    {(documentsData || []).map((doc: Document) => (
                      <label
                        key={doc.id}
                        className={`flex items-center gap-3 p-3 border rounded-lg cursor-pointer ${
                          selectedDocs.includes(doc.id)
                            ? 'border-primary-500 bg-primary-50'
                            : 'border-gray-200 hover:bg-gray-50'
                        }`}
                      >
                        <input
                          type="checkbox"
                          checked={selectedDocs.includes(doc.id)}
                          onChange={(e) => {
                            if (e.target.checked) {
                              setSelectedDocs([...selectedDocs, doc.id]);
                            } else {
                              setSelectedDocs(selectedDocs.filter((id) => id !== doc.id));
                            }
                          }}
                          className="rounded"
                        />
	                        <div className="flex-1 min-w-0">
	                          <p className="font-medium text-gray-900 truncate">{doc.title}</p>
	                          <p className="text-xs text-gray-500">
	                            {doc.file_type || doc.source?.source_type || 'document'}
	                            {doc.file_size ? ` | ${(doc.file_size / 1024).toFixed(0)} KB` : ''}
	                          </p>
	                        </div>
	                      </label>
                    ))}
                  </div>
                )}
              </div>
            )}

            {/* Step 3: Configuration */}
            {step === 'config' && (
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Title *</label>
                  <input
                    type="text"
                    className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                    value={title}
                    onChange={(e) => {
                      setTitle(e.target.value);
                      setAutoTitle(false);
                    }}
                    placeholder="My Document Synthesis"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Topic/Focus</label>
                  <input
                    type="text"
                    className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                    value={topic}
                    onChange={(e) => setTopic(e.target.value)}
                    placeholder="Optional focus area for the synthesis"
                  />
                </div>

                {selectedType === 'gap_analysis_hypotheses' && (
                  <div className="border border-gray-200 rounded-lg p-4 bg-rose-50/30">
                    <h4 className="text-sm font-medium text-gray-900 mb-3">Research Hub Options</h4>
                    <div className="space-y-3">
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">Domain</label>
                        <input
                          type="text"
                          className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                          value={domain}
                          onChange={(e) => setDomain(e.target.value)}
                          placeholder="e.g., compilers, CPU microarchitecture, program analysis"
                        />
                      </div>

                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">Constraints</label>
                        <textarea
                          className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                          rows={3}
                          value={constraints}
                          onChange={(e) => setConstraints(e.target.value)}
                          placeholder="e.g., must integrate with LLVM; must run on SPEC CPU; limited to compile-time overhead < 5%"
                        />
                      </div>

                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">Desired Outcomes</label>
                        <textarea
                          className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                          rows={3}
                          value={desiredOutcomes}
                          onChange={(e) => setDesiredOutcomes(e.target.value)}
                          placeholder="e.g., novel pass ideas; benchmark plan; falsifiable hypotheses; threat-to-validity checklist"
                        />
                      </div>

                      <label className="flex items-center gap-2 text-sm text-gray-700">
                        <input
                          type="checkbox"
                          className="rounded"
                          checked={includeBibliography}
                          onChange={(e) => setIncludeBibliography(e.target.checked)}
                        />
                        Include bibliography / source list
                      </label>
                    </div>
                  </div>
                )}

                {selectedType === 'decision_memo' && (
                  <div className="border border-gray-200 rounded-lg p-4 bg-cyan-50/40">
                    <h4 className="text-sm font-medium text-gray-900 mb-3">Decision Memo Options</h4>
                    <div className="space-y-3">
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">Audience</label>
                        <select
                          className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                          value={decisionAudience}
                          onChange={(e) => setDecisionAudience(e.target.value)}
                        >
                          <option value="product_and_strategy">Product and Strategy</option>
                          <option value="executive_leadership">Executive Leadership</option>
                          <option value="research_ops">Research Ops</option>
                          <option value="procurement">Procurement</option>
                        </select>
                      </div>

                      <label className="flex items-center gap-2 text-sm text-gray-700">
                        <input
                          type="checkbox"
                          className="rounded"
                          checked={includeDecisionRecommendations}
                          onChange={(e) => setIncludeDecisionRecommendations(e.target.checked)}
                        />
                        Include recommended actions
                      </label>

                      <label className="flex items-center gap-2 text-sm text-gray-700">
                        <input
                          type="checkbox"
                          className="rounded"
                          checked={includeDecisionWatchlist}
                          onChange={(e) => setIncludeDecisionWatchlist(e.target.checked)}
                        />
                        Include watchlist / monitoring triggers
                      </label>
                    </div>
                  </div>
                )}

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Output Format</label>
                    <select
                      className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                      value={outputFormat}
                      onChange={(e) => setOutputFormat(e.target.value as any)}
                    >
                      <option value="markdown">Markdown</option>
                      <option value="docx">Word (DOCX)</option>
                      <option value="pdf">PDF</option>
                      <option value="pptx">PowerPoint</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Style</label>
                    <select
                      className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                      value={outputStyle}
                      onChange={(e) => setOutputStyle(e.target.value)}
                    >
                      <option value="professional">Professional</option>
                      <option value="technical">Technical</option>
                      <option value="casual">Casual</option>
                    </select>
                  </div>
                </div>

                {/* Summary */}
                <div className="bg-gray-50 rounded-lg p-4">
                  <h4 className="text-sm font-medium text-gray-700 mb-2">Summary</h4>
                  <div className="text-sm text-gray-600 space-y-1">
                    <p><span className="text-gray-500">Type:</span> {JOB_TYPE_CONFIG[selectedType].label}</p>
                    <p><span className="text-gray-500">Documents:</span> {selectedDocs.length}</p>
                    {selectedRuns.length > 0 && (
                      <p><span className="text-gray-500">Runs:</span> {selectedRuns.length}</p>
                    )}
                    {sourceSearchQuery.trim() && (
                      <p><span className="text-gray-500">Search query:</span> {sourceSearchQuery.trim()}</p>
                    )}
                    <p><span className="text-gray-500">Output:</span> {outputFormat.toUpperCase()}</p>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Footer */}
          <div className="p-6 border-t border-gray-200 flex justify-between">
            <Button
              variant="secondary"
              onClick={() => {
                if (step === 'documents') setStep('type');
                else if (step === 'config') setStep('documents');
                else setShowCreateModal(false);
              }}
            >
              {step === 'type' ? 'Cancel' : 'Back'}
            </Button>

            <Button
              onClick={() => {
                if (step === 'type') setStep('documents');
                else if (step === 'documents') {
                  if (selectedDocs.length === 0 && selectedRuns.length === 0 && !sourceSearchQuery.trim()) {
                    toast.error('Select at least one document or provide a search query');
                    return;
                  }
                  setStep('config');
                }
                else handleSubmit();
              }}
              disabled={isSubmitting}
            >
              {step === 'config' ? (
                isSubmitting ? 'Creating...' : 'Create Synthesis'
              ) : 'Next'}
            </Button>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="p-6 h-full flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Document Synthesis</h1>
          <p className="text-gray-500">Generate multi-document summaries, comparisons, and reports</p>
        </div>
        <Button onClick={() => setShowCreateModal(true)}>
          <Plus className="w-4 h-4 mr-2" />
          New Synthesis
        </Button>
      </div>

      {/* Filters */}
      <div className="flex gap-3 mb-4">
        <select
          className="border border-gray-300 rounded-lg px-3 py-2 text-sm"
          value={statusFilter}
          onChange={(e) => setStatusFilter(e.target.value)}
        >
          <option value="">All Status</option>
          <option value="pending">Pending</option>
          <option value="analyzing">Analyzing</option>
          <option value="synthesizing">Synthesizing</option>
          <option value="generating">Generating</option>
          <option value="completed">Completed</option>
          <option value="failed">Failed</option>
          <option value="cancelled">Cancelled</option>
        </select>
        <select
          className="border border-gray-300 rounded-lg px-3 py-2 text-sm"
          value={typeFilter}
          onChange={(e) => setTypeFilter(e.target.value)}
        >
          <option value="">All Types</option>
          {Object.entries(JOB_TYPE_CONFIG).map(([type, config]) => (
            <option key={type} value={type}>{config.label}</option>
          ))}
        </select>
        <Button variant="ghost" size="sm" onClick={() => refetchJobs()}>
          <RefreshCw className="w-4 h-4" />
        </Button>
      </div>

      {/* Content */}
      <div className="flex-1 flex gap-6 min-h-0">
        {/* Jobs list */}
        <div className="w-2/3 overflow-y-auto">
          {jobsLoading ? (
            <div className="flex justify-center items-center h-full">
              <LoadingSpinner />
            </div>
          ) : jobsData?.jobs.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-gray-500">
              <Layers className="w-12 h-12 mb-3 text-gray-400" />
              <p className="text-lg font-medium">No synthesis jobs yet</p>
              <p className="text-sm">Create a new synthesis to get started</p>
            </div>
          ) : (
            <div className="grid grid-cols-2 gap-4">
              {jobsData?.jobs.map((job) => (
                <JobCard key={job.id} job={job} />
              ))}
            </div>
          )}
        </div>

        {/* Detail panel */}
        <div className="w-1/3">
          {selectedJob ? (
            <JobDetailPanel job={selectedJob} />
          ) : (
            <div className="bg-gray-50 border border-gray-200 rounded-lg h-full flex flex-col items-center justify-center text-gray-500">
              <Eye className="w-10 h-10 mb-3 text-gray-400" />
              <p className="font-medium">Select a job</p>
              <p className="text-sm">Click on a job to view details</p>
            </div>
          )}
        </div>
      </div>

      {/* Create modal */}
      {showCreateModal && <CreateModal />}
    </div>
  );
};

export default SynthesisPage;
