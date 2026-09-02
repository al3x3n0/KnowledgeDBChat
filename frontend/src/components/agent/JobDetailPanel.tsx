/**
 * The detail panel for one agent run: its phases, findings, evidence,
 * experiment runs, memories, patches and the actions available on it.
 *
 * The largest of the eighteen components AutonomousAgentsPage declared inside
 * its own render body — 4,224 lines of it. A component declared in a render
 * body gets a new function identity every render, which React treats as a new
 * type and remounts rather than reconciles; this one sits beside the job list
 * on the tab most used, so it was being torn down and rebuilt on every
 * keystroke anywhere on the page.
 *
 * For its size it reached for surprisingly little: about nineteen values from
 * the page, plus navigation and the query client, which it now takes from
 * their hooks directly like any other component. What is left is declared
 * below.
 *
 * It is still 4,000 lines and wants splitting again — findings, evidence,
 * experiment runs and patches are separate concerns sharing only a job. This
 * move is the precondition for that, not a substitute for it.
 */

import {
  AlertCircle,
  BarChart3,
  Brain,
  Download,
  GitBranch,
  Layers,
  Link2,
  Pause,
  Play,
  RefreshCw,
  RotateCcw,
  Sparkles,
  Target,
  Trash2,
  XCircle,
} from 'lucide-react';
import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import toast from 'react-hot-toast';
import { useMutation, useQuery, useQueryClient } from 'react-query';
import { useLocation, useNavigate } from 'react-router-dom';

import { apiClient } from '../../services/api';
import type {
  AgentJob,
  AgentJobOperatorIntervention,
  AgentJobPromoteDomainResearchRequest,
  AgentJobStatus,
  AgentJobType,
  ResearchPortfolio,
} from '../../types';
import {
  normalizeJobMemoryPersistenceSummary,
  type JobMemoryExtractionSummary,
} from '../../utils/agentMemoryExtraction';
import { copyText } from '../../utils/clipboard';
import {
  buildDomainResearchPromotionDraft,
  codePatchView,
  executionGraphView,
  experimentRunsView,
  swarmSummaryOf,
  humanizeSwarmOutcome,
  slugifyText,
  summarizeSchedulerState,
  swarmOutcomeBadgeClass,
  type DomainResearchPromotionDraft,
} from '../../utils/agentJobDetail';
import {
  isExperimentRecoveryOpen,
  summarizeExperimentRun,
  summarizeOperatorInterventions,
} from '../../utils/experimentRunSummary';
import AIHubBundleSection from './AIHubBundleSection';
import JobMemoriesSection from './JobMemoriesSection';
import Button from '../common/Button';
import LoadingSpinner from '../common/LoadingSpinner';
import AutonomousRndVerificationPanel from './AutonomousRndVerificationPanel';
import JobExecutionGraphSection from './JobExecutionGraphSection';
import RecoveryAuditPanel from './RecoveryAuditPanel';
import {
  JOB_TYPE_CONFIG,
  STATUS_CONFIG,
  type AgentJobsTab,
  type UnsafeExecBadge,
} from './jobConfig';

export interface JobDetailPanelProps {
  job: AgentJob;

  /** Builds a link to this page, carrying the current filters. Owned by the
   *  page because the page owns what the URL means. */
  buildAutonomousAgentsUrl: (
    jobId?: string,
    extras?: Record<string, string | null | undefined>
  ) => string;
  formatDuration: (startedAt?: string, completedAt?: string) => string;

  /** Run actions: pause, resume, cancel and the rest. */
  actionMutation: any;
  createMutation: any;
  deleteMutation: any;
  createCodingBacklogMutation: any;
  promoteDomainResearchMutation: any;

  setSelectedJob: (job: AgentJob | null) => void;
  setActiveTab: (tab: AgentJobsTab) => void;
  setExportingJob: (job: AgentJob | null) => void;
  setShowExportModal: (show: boolean) => void;
  setHasRelaunchChildrenFilter: (value: string) => void;
  setRelaunchFromJobIdFilter: (value: string) => void;

  /** Swarm outcomes indexed both ways, so a run can find its counterpart. */
  swarmOutcomeByRepairJobId: Record<string, any>;
  swarmOutcomeBySwarmJobId: Record<string, any>;
  /** The demo-check availability badge: its status drives what the panel
   *  is allowed to offer, so it is data here rather than rendered markup. */
  unsafeExecBadge: UnsafeExecBadge;
}

export const JobDetailPanel: React.FC<JobDetailPanelProps> = ({
  job,
  buildAutonomousAgentsUrl,
  formatDuration,
  actionMutation,
  createMutation,
  deleteMutation,
  createCodingBacklogMutation,
  promoteDomainResearchMutation,
  setSelectedJob,
  setActiveTab,
  setExportingJob,
  setShowExportModal,
  setHasRelaunchChildrenFilter,
  setRelaunchFromJobIdFilter,
  swarmOutcomeByRepairJobId,
  swarmOutcomeBySwarmJobId,
  unsafeExecBadge,
}) => {
  const navigate = useNavigate();
  const location = useLocation();
  const queryClient = useQueryClient();

  // Fourteen values the graph and scope views read off the job, derived once
  // in a pure function so they can be tested without rendering, and shared
  // rather than recomputed by each section that reads them.
  const {
    graphHealth,
    graphHealthStatus,
    graphRecommendedActions,
  } = useMemo(() => executionGraphView(job), [job]);

  // Sixteen code-patch values, previously four clusters and six separate
  // useMemos over the same job.results.
  const {
    codePatchProposal,
    codePatchExecution,
    codePatchWorkspace,
    codePatchVerificationPlan,
    codePatchExecutionPlan,
    codePatchRecovery,
    codePatchDetectedStack,
    codePatchVerificationCommands,
    codePatchBootstrapCommands,
    codePatchFallbackCommands,
    codePatchFailedCommands,
    codePatchSuggestedActions,
    codingRecoveryState,
    codePatchProposals,
    codePatchApply,
    codePatchKbApply,
  } = useMemo(() => codePatchView(job), [job]);

  const swarmSummary = useMemo(() => swarmSummaryOf(job), [job]);
  // latestExperimentRun and latestExperimentSummary are not destructured: the
  // panel never read them directly, only the recovery flag they feed. They
  // looked used while they were locals because another local consumed them.
  const {
    experimentRuns,
    latestExperimentRunIndex,
    latestExperimentRecoveryOpen,
  } = useMemo(() => experimentRunsView(job), [job]);
    // location.search belongs in the deps: while this panel was declared
    // inside the page's render body it remounted constantly, so reading the
    // URL once per mount happened to look live. As a stable component it would
    // have kept the value from the first render for ever.
    const lineageModeFromUrl = useMemo(() => {
      const raw = String(new URLSearchParams(location.search).get('lx') || '').trim().toLowerCase();
      return raw === 'full' ? 'full' : 'compact';
    }, [location.search]);
    const [logData, setLogData] = useState<{ entries: Array<Record<string, any>>; total: number } | null>(null);
    const [loadingLog, setLoadingLog] = useState(false);
    const [stepEventsData, setStepEventsData] = useState<{
      items: Array<Record<string, any>>;
      total: number;
      source?: string;
    } | null>(null);
    const [loadingStepEvents, setLoadingStepEvents] = useState(false);
    const [showStepEvents, setShowStepEvents] = useState(false);
    const [showExecutionLog, setShowExecutionLog] = useState(false);
    const [manualExtractionSummary, setManualExtractionSummary] = useState<JobMemoryExtractionSummary | null>(null);
    const [lineageExpanded, setLineageExpanded] = useState<boolean>(lineageModeFromUrl === 'full');
    const [showShortcutHelp, setShowShortcutHelp] = useState(false);
    const [approvalNote, setApprovalNote] = useState<string>('');
    const [showApprovalEdit, setShowApprovalEdit] = useState<boolean>(false);
    const [approvalEditTool, setApprovalEditTool] = useState<string>('');
    const [approvalEditPurpose, setApprovalEditPurpose] = useState<string>('');
    const [approvalEditParams, setApprovalEditParams] = useState<string>('{}');
    const [showPromotionPanel, setShowPromotionPanel] = useState(false);
    const [promotionDraft, setPromotionDraft] = useState<DomainResearchPromotionDraft>(() => buildDomainResearchPromotionDraft(job));
    const detailPanelMountedRef = useRef(true);
    const logRequestIdRef = useRef(0);
    const stepEventsRequestIdRef = useRef(0);
    useEffect(() => {
      const shouldExpand = lineageModeFromUrl === 'full';
      setLineageExpanded((prev) => (prev === shouldExpand ? prev : shouldExpand));
    }, [lineageModeFromUrl]);
    useEffect(() => {
      detailPanelMountedRef.current = true;
      return () => {
        detailPanelMountedRef.current = false;
      };
    }, []);

    const typeConfig = JOB_TYPE_CONFIG[job.job_type as AgentJobType] || JOB_TYPE_CONFIG.custom;
    const statusConfig = STATUS_CONFIG[job.status as AgentJobStatus] || STATUS_CONFIG.pending;
    const StatusIcon = statusConfig.icon;
    const TypeIcon = typeConfig.icon;
    const domainResearch = (job.results as any)?.domain_research;
    const customerProfile = (job.results as any)?.customer_profile;
    const customerContext = (job.results as any)?.customer_context;
    const researchBundle = (job.results as any)?.research_bundle;
    const goalContractSummary = (((job as any)?.goal_contract_summary && typeof (job as any)?.goal_contract_summary === 'object')
      ? (job as any).goal_contract_summary
      : (((job.results as any)?.goal_contract && typeof (job.results as any)?.goal_contract === 'object')
          ? (job.results as any).goal_contract
          : null));
    const approvalCheckpoint = (((job as any)?.approval_checkpoint && typeof (job as any)?.approval_checkpoint === 'object')
      ? (job as any).approval_checkpoint
      : (((job.results as any)?.approval_checkpoint && typeof (job.results as any)?.approval_checkpoint === 'object')
          ? (job.results as any).approval_checkpoint
          : (((job.results as any)?.execution_strategy?.approval_checkpoints?.pending && typeof (job.results as any)?.execution_strategy?.approval_checkpoints?.pending === 'object')
              ? (job.results as any).execution_strategy.approval_checkpoints.pending
              : null)));
    const executionPlan = (
      Array.isArray((job.results as any)?.execution_strategy?.execution_plan)
        ? ((job.results as any).execution_strategy.execution_plan as Array<Record<string, any>>)
        : []
    );
    const planStepIndex = Number((job.results as any)?.execution_strategy?.plan_step_index || 0);
    const activePlanIndex = Math.max(0, Math.min(planStepIndex, Math.max(0, executionPlan.length - 1)));
    const operatorInterventions = (
      Array.isArray(job.operator_interventions)
        ? (job.operator_interventions as AgentJobOperatorIntervention[])
        : Array.isArray((job.results as any)?.execution_strategy?.operator_interventions)
          ? ((job.results as any).execution_strategy.operator_interventions as AgentJobOperatorIntervention[])
        : []
    );
    const operatorInterventionSummary = summarizeOperatorInterventions(operatorInterventions);
    const launchMode = String((job as any)?.launch_mode || ((job.config as any)?.launch_mode || '')).trim().toLowerCase();
    const verificationOrigin = (
      (job.config as any)?.verification_origin
      && typeof (job.config as any).verification_origin === 'object'
    )
      ? ((job.config as any).verification_origin as Record<string, any>)
      : null;
    const verificationParentJobId = String(verificationOrigin?.parent_job_id || '').trim();
    const verificationTaskId = String(verificationOrigin?.verification_task_id || '').trim();
    const focusedVerificationTaskId = String(
      new URLSearchParams(location.search).get('verification_task') || ''
    ).trim();
    const promotionStatus = String((job as any)?.promotion_status || '').trim().toLowerCase();
    const promotedProfileId = String((job as any)?.promoted_domain_research_profile_id || '').trim();
    const promotedPortfolioId = String((job as any)?.promoted_research_portfolio_id || '').trim();
    const launchQuickStart = (((job.config as any)?.quick_start && typeof ((job.config as any)?.quick_start) === 'object')
      ? ((job.config as any).quick_start as Record<string, any>)
      : null);
    const executionMode = String(
      (job.results as any)?.execution_strategy?.execution_mode
      || (job.config as any)?.execution_mode
      || launchQuickStart?.execution_mode
      || ''
    ).trim().toLowerCase();
    const planCompleted = Boolean((job.results as any)?.execution_strategy?.plan_completed);
    const memoryPersistence = normalizeJobMemoryPersistenceSummary(
      (job.results as any)?.execution_strategy?.memory_persistence
    );
    const memoryRuntime = memoryPersistence?.runtime || null;
    const memoryExtractionSummary = memoryPersistence?.extraction || null;
    const memoryExtractionView = manualExtractionSummary || memoryExtractionSummary;
    const promotionDraftJobName = job.name;
    const promotionDraftDomain = String((job.config as any)?.domain || '');
    const promotionDraftInterval = (job.config as any)?.interval_minutes;
    useEffect(() => {
      setManualExtractionSummary(null);
    }, [job.id]);
    useEffect(() => {
      setShowPromotionPanel(false);
      setPromotionDraft(buildDomainResearchPromotionDraft({
        name: promotionDraftJobName,
        config: {
          domain: promotionDraftDomain,
          interval_minutes: promotionDraftInterval,
        },
      }));
    }, [job.id, promotionDraftDomain, promotionDraftInterval, promotionDraftJobName]);
    useEffect(() => {
      const action = (approvalCheckpoint?.action && typeof approvalCheckpoint.action === 'object')
        ? (approvalCheckpoint.action as Record<string, any>)
        : {};
      setApprovalEditTool(String(action.tool || '').trim());
      setApprovalEditPurpose(String(action.purpose || '').trim());
      setApprovalEditParams(JSON.stringify((action.params && typeof action.params === 'object') ? action.params : {}, null, 2));
    }, [approvalCheckpoint]);
    const launchLog = useMemo(() => {
      const rows = Array.isArray(job.execution_log) ? job.execution_log : [];
      for (let i = rows.length - 1; i >= 0; i--) {
        const row = rows[i] as any;
        const phase = String(row?.phase || '').toLowerCase();
        const action = String(row?.action || '').toLowerCase();
        if (phase === 'launch' || action === 'job_launch') return row;
      }
      return null;
    }, [job.execution_log]);
    const launchResult = (launchLog?.result && typeof launchLog.result === 'object')
      ? (launchLog.result as Record<string, any>)
      : null;
    const launchSourceId = String(
      ((job.config as any)?.source_id || launchResult?.source_id || '')
    ).trim();
    const launchDomain = String(((job.config as any)?.domain || '')).trim();
    const launchObjective = String(((job.config as any)?.objective || '')).trim();
    const canPromoteDomainResearch = (
      launchMode === 'quick_start_domain_research'
      && job.status === 'completed'
      && !promotedProfileId
      && !String(((job.config as any)?.profile_id || '')).trim()
    );
    const shouldLoadPromotionPortfolios = (
      showPromotionPanel
      && canPromoteDomainResearch
      && promotionDraft.target_mode === 'profile_with_portfolio'
      && promotionDraft.portfolio_mode === 'existing'
    );
    const {
      data: promotionResearchPortfoliosData,
      isLoading: promotionResearchPortfoliosLoading,
    } = useQuery(
      ['research-portfolios', 'promotion-picker'],
      () => apiClient.listResearchPortfolios({ limit: 100, offset: 0 }),
      {
        enabled: shouldLoadPromotionPortfolios,
        refetchOnWindowFocus: false,
      }
    );
    const promotionResearchPortfolios = (
      (((promotionResearchPortfoliosData as any)?.items || []) as ResearchPortfolio[])
    );
    const launchRelaunchFromJobId = String(
      ((job.config as any)?.relaunch_from_job_id || launchResult?.relaunch_from_job_id || '')
    ).trim();
    const {
      data: relaunchLineage,
    } = useQuery(
      ['agent-job-relaunch-lineage', job.id, lineageExpanded ? 'full' : 'compact'],
      () =>
        apiClient.getAgentJobRelaunchLineage(
          String(job.id),
          lineageExpanded
            ? { ancestor_limit: 300, descendant_limit: 2000 }
            : { ancestor_limit: 60, descendant_limit: 180 }
        ),
      {
        enabled: ['quick_start_claude_backend', 'quick_start_domain_research', 'quick_start_bug_triage_swarm', 'quick_start_build_break_swarm', 'quick_start_frontend_regression_swarm', 'quick_start_repo_bug_triage'].includes(launchMode) || !!launchRelaunchFromJobId,
        staleTime: 15000,
      }
    );
    const lineageParentJobId = String((relaunchLineage as any)?.parent_job_id || launchRelaunchFromJobId || '').trim();
    const lineageLatestChildJobId = String((relaunchLineage as any)?.latest_child_job_id || '').trim();
    const lineageAllAncestors = useMemo(
      () => Array.isArray((relaunchLineage as any)?.ancestors)
        ? ((relaunchLineage as any).ancestors as any[])
        : [],
      [relaunchLineage]
    );
    const lineageAllDescendants = useMemo(
      () => Array.isArray((relaunchLineage as any)?.descendants)
        ? ((relaunchLineage as any).descendants as any[])
        : [],
      [relaunchLineage]
    );
    const lineageAncestors = lineageAllAncestors.slice(0, 4);
    const lineageDescendants = lineageAllDescendants.slice(0, 4);
    const lineageAncestorsTruncated = Boolean((relaunchLineage as any)?.ancestors_truncated);
    const lineageDescendantsTruncated = Boolean((relaunchLineage as any)?.descendants_truncated);
    const lineageAnyTruncated = lineageAncestorsTruncated || lineageDescendantsTruncated;
    const lineageRootNode = lineageAllAncestors.length > 0 ? lineageAllAncestors[lineageAllAncestors.length - 1] : null;
    const lineageParentNode = lineageAllAncestors.length > 0 ? lineageAllAncestors[0] : null;
    const copyLineageLink = useCallback(async () => {
      const params = new URLSearchParams(location.search);
      params.set('job', String(job.id));
      if (lineageExpanded) params.set('lx', 'full');
      else params.delete('lx');
      const qs = params.toString();
      const link = `${window.location.origin}${location.pathname}${qs ? `?${qs}` : ''}`;
      try {
        if (navigator?.clipboard?.writeText) {
          await navigator.clipboard.writeText(link);
          toast.success('Lineage link copied');
        } else {
          toast.error('Clipboard not supported');
        }
      } catch {
        toast.error('Failed to copy lineage link');
      }
    }, [job.id, lineageExpanded, location.pathname, location.search]);
    const lineagePathNodes = useMemo(() => {
      const path = [...lineageAllAncestors].reverse();
      path.push({
        id: job.id,
        name: job.name,
        status: job.status,
      });
      return path;
    }, [lineageAllAncestors, job.id, job.name, job.status]);
    const lineageLatestChildNode = useMemo(() => {
      if (!lineageLatestChildJobId) return null;
      const pool = [...lineageAllDescendants, ...lineageAllAncestors];
      return pool.find((n: any) => String(n?.id || '') === lineageLatestChildJobId) || null;
    }, [lineageLatestChildJobId, lineageAllDescendants, lineageAllAncestors]);
    useEffect(() => {
      const onKeyDown = (ev: KeyboardEvent) => {
        if (ev.metaKey || ev.ctrlKey || ev.altKey) return;
        const target = ev.target as HTMLElement | null;
        const tag = String(target?.tagName || '').toLowerCase();
        const isEditable = Boolean(
          target?.isContentEditable ||
          tag === 'input' ||
          tag === 'textarea' ||
          tag === 'select'
        );
        if (isEditable) return;

        if (ev.key === '[' && lineageParentJobId) {
          ev.preventDefault();
          navigate(buildAutonomousAgentsUrl(lineageParentJobId));
          return;
        }
        if (ev.key === ']' && lineageLatestChildJobId && lineageLatestChildJobId !== String(job.id)) {
          ev.preventDefault();
          navigate(buildAutonomousAgentsUrl(lineageLatestChildJobId));
          return;
        }
        const isQuestion = ev.key === '?' || (ev.key === '/' && ev.shiftKey);
        if (isQuestion) {
          ev.preventDefault();
          setShowShortcutHelp((v) => !v);
          return;
        }
        if (ev.key.toLowerCase() === 'c') {
          ev.preventDefault();
          void copyLineageLink();
          return;
        }
        if (ev.key.toLowerCase() === 'x') {
          ev.preventDefault();
          setSelectedJob(null);
          navigate(buildAutonomousAgentsUrl(), { replace: true });
          return;
        }
        if (ev.key === 'Escape') {
          setShowShortcutHelp(false);
        }
      };
      window.addEventListener('keydown', onKeyDown);
      return () => window.removeEventListener('keydown', onKeyDown);
    }, [
      lineageParentJobId,
      lineageLatestChildJobId,
      job.id,
      copyLineageLink,
      // Without these the 'x' shortcut navigates with whatever URL the panel
      // first rendered against.
      buildAutonomousAgentsUrl,
      navigate,
      setSelectedJob,
    ]);
    const launchSearchQuery = String(((job.config as any)?.search_query || launchResult?.search_query || '')).trim();
    const launchCommands = Array.isArray((job.config as any)?.commands)
      ? ((job.config as any).commands as any[]).map((x) => String(x || '').trim()).filter(Boolean)
      : [];
    const launchFilePaths = Array.isArray((job.config as any)?.file_paths)
      ? ((job.config as any).file_paths as any[]).map((x) => String(x || '').trim()).filter(Boolean)
      : [];
    const launchCommandsCount = launchCommands.length > 0
      ? launchCommands.length
      : Math.max(0, Number(launchResult?.commands_count || 0));
    const launchFilePathsCount = launchFilePaths.length > 0
      ? launchFilePaths.length
      : Math.max(0, Number(launchResult?.file_paths_count || 0));
    const graphHealthReasons = useMemo(
      () => Array.isArray(graphHealth?.reasons)
        ? (graphHealth?.reasons || []).filter((x: any) => String(x || '').trim()).slice(0, 6)
        : [],
      [graphHealth?.reasons]
    );
    const schedulerState = (job as any)?.scheduler_state && typeof (job as any).scheduler_state === 'object'
      ? ((job as any).scheduler_state as Record<string, any>)
      : null;
    const isRecoveryPlaybookCandidate = useMemo(() => {
      const normalizedStatus = String(job.status || '').trim().toLowerCase();
      const queueReason = String(schedulerState?.queue_reason || '').trim().toLowerCase();
      const hasFailedOrRecoverySignal = (
        normalizedStatus === 'failed'
        || normalizedStatus === 'cancelled'
        || ['execution_failure', 'stalled_run', 'scheduled_recovery', 'scheduler_backoff'].includes(queueReason)
        || Boolean(job.error && String(job.error).trim())
        || Boolean(job.phase_details && String(job.phase_details).trim() && normalizedStatus !== 'paused')
      );
      if (hasFailedOrRecoverySignal) return true;
      return Boolean(graphHealthStatus && ['warning', 'critical'].includes(graphHealthStatus) && graphHealthReasons.length > 0);
    }, [graphHealthReasons.length, graphHealthStatus, job.error, job.phase_details, job.status, schedulerState?.queue_reason]);
    const recoveryPlaybookDefaults = useMemo(() => {
      if (!isRecoveryPlaybookCandidate) return null;
      const jobName = String(job.name || 'Recovery Job').trim() || 'Recovery Job';
      const queueReason = String(schedulerState?.queue_reason || '').trim().replace(/_/g, ' ');
      const reasonText = queueReason || String(graphHealthReasons[0] || '').trim();
      const summaryParts = [
        reasonText ? `Recovery context: ${reasonText}.` : '',
        String(job.error || '').trim() ? `Error: ${String(job.error).trim().slice(0, 240)}.` : '',
        String(job.phase_details || '').trim() ? `Details: ${String(job.phase_details).trim().slice(0, 240)}.` : '',
      ].filter(Boolean);
      return {
        name: `playbook_recovery_${slugifyText(jobName) || 'job'}`,
        display_name: `${jobName} (Recovery Playbook)`,
        description: summaryParts.length > 0
          ? `Saved from recovery job ${String(job.id || '')}. ${summaryParts.join(' ')}`
          : `Saved from recovery job ${String(job.id || '')}.`,
      };
    }, [graphHealthReasons, isRecoveryPlaybookCandidate, job.error, job.id, job.name, job.phase_details, schedulerState?.queue_reason]);
    const schedulerSummaryLines = summarizeSchedulerState(schedulerState);
    // The maps are rebuilt by the page when the outcomes query refreshes;
    // keyed only on job.id this kept the outcome it saw first.
    const swarmOutcomeCase = useMemo(
      () => swarmOutcomeBySwarmJobId[String(job.id)] || swarmOutcomeByRepairJobId[String(job.id)] || null,
      [job.id, swarmOutcomeBySwarmJobId, swarmOutcomeByRepairJobId]
    );
    const canSaveAsPlaybook = Boolean((job as any)?.chain_config?.child_jobs?.length)
      || Boolean((job as any)?.root_job_id)
      || Boolean((job as any)?.parent_job_id)
      || Boolean(recoveryPlaybookDefaults);

    const saveAsPlaybookMutation = useMutation(
      () => apiClient.saveAgentJobAsChain(String(job.id), recoveryPlaybookDefaults || {}),
      {
        onSuccess: () => {
          toast.success('Saved as playbook');
          queryClient.invalidateQueries(['agent-job-chains']);
          setActiveTab('chains');
        },
        onError: (e: any) => {
          toast.error(e?.message || 'Failed to save playbook');
        },
      }
    );

    const documentArtifact = useMemo(() => {
      const arts = (job.output_artifacts as any[]) || [];
      return arts.find((a) => a?.type === 'document' && (a?.id || a?.document_id));
    }, [job.output_artifacts]);

    const readingListArtifact = useMemo(() => {
      const arts = (job.output_artifacts as any[]) || [];
      return arts.find((a) => a?.type === 'reading_list' && a?.id);
    }, [job.output_artifacts]);

    const arxivSourceArtifacts = useMemo(() => {
      const arts = (job.output_artifacts as any[]) || [];
      return arts.filter((a) => a?.type === 'document_source' && (a?.source_type === 'arxiv' || a?.sourceType === 'arxiv'));
    }, [job.output_artifacts]);
    const hasArxivImportRequestArtifact = useMemo(() => {
      const arts = (job.output_artifacts as any[]) || [];
      return arts.some((a) => a?.type === 'arxiv_ingest_requested' && a?.source_id);
    }, [job.output_artifacts]);

    const isRepoBugTriageJob = launchMode === 'quick_start_repo_bug_triage';
    const codingRecoveryDirectControls = Boolean(
      isRepoBugTriageJob && (
        codePatchRecovery?.can_retry_with_refined_plan
        || codePatchRecovery?.can_resume_verification
        || ['completed', 'failed', 'cancelled'].includes(job.status)
      )
    );


    const queueManagedApproval = Boolean(approvalCheckpoint && job.status === 'paused');
    const queueManagedRecovery = Boolean(latestExperimentRecoveryOpen && !codingRecoveryDirectControls);
    const supportsQuickStartRelaunch = ['quick_start_claude_backend', 'quick_start_domain_research', 'quick_start_bug_triage_swarm', 'quick_start_build_break_swarm', 'quick_start_frontend_regression_swarm', 'quick_start_repo_bug_triage', 'quick_start_role_workflow'].includes(launchMode);
    const terminalJob = ['completed', 'failed', 'cancelled'].includes(job.status);
    const isCodingSwarmJob = ['quick_start_bug_triage_swarm', 'quick_start_build_break_swarm', 'quick_start_frontend_regression_swarm'].includes(launchMode);
    const swarmReviewState = String(swarmSummary?.review_state || '').trim().toLowerCase();
    const swarmReviewReason = String(swarmSummary?.review_reason || swarmSummary?.promotion_reason || '').trim();
    const swarmNeedsReview = Boolean(
      swarmSummary?.review_required
      || ['needs_review', 'insufficient_swarm_consensus', 'consensus_failed'].includes(swarmReviewState)
      || (job.status === 'paused' && isCodingSwarmJob)
    );
    const canLaunchSwarmTieBreaker = Boolean(
      isCodingSwarmJob
      && (terminalJob || job.status === 'paused')
      && ['tie_break_needed', 'needs_review', 'insufficient_swarm_consensus', 'consensus_failed'].includes(swarmReviewState || 'consensus_failed')
      && !swarmSummary?.repair_chain_job_id
    );
    const canManualPromoteSwarmCandidate = Boolean(
      isCodingSwarmJob
      && (terminalJob || job.status === 'paused')
      && Array.isArray(swarmSummary?.candidate_paths)
      && swarmSummary.candidate_paths.length > 0
      && !swarmSummary?.repair_chain_job_id
    );
    const openCheckpointQueue = () => {
      setActiveTab('queue');
      navigate(buildAutonomousAgentsUrl(String(job.id)), { replace: true });
    };


    const generatedProject = useMemo(() => {
      const fromResults = (job.results as any)?.generated_project;
      if (fromResults?.source_id) {
        const behavioral = fromResults?.sanity_check?.behavioral;
        return {
          source_id: String(fromResults.source_id),
          source_name: String(fromResults.source_name || 'Generated project'),
          project_name: String(fromResults.project_name || fromResults.source_name || 'Generated project'),
          entrypoint: String(fromResults.entrypoint || 'demo.py'),
          file_count: Number(fromResults.file_count || 0),
          sanity_ok: fromResults?.sanity_check?.ok === true,
          sanity_errors_count: Array.isArray(fromResults?.sanity_check?.syntax_errors) ? fromResults.sanity_check.syntax_errors.length : 0,
          behavioral,
        };
      }
      const arts = (job.output_artifacts as any[]) || [];
      const art = arts.find((a) => a?.type === 'generated_project' && a?.source_id);
      if (art?.source_id) {
        return {
          source_id: String(art.source_id),
          source_name: String(art.title || 'Generated project'),
          project_name: String(art.title || 'Generated project'),
          entrypoint: 'demo.py',
          file_count: 0,
          sanity_ok: false,
          sanity_errors_count: 0,
          behavioral: null,
        };
      }
      return null;
    }, [job.output_artifacts, job.results]);

    const demoCheck = useMemo(() => {
      const fromResults = (job.results as any)?.demo_check;
      if (fromResults?.source_id) {
        return {
          source_id: String(fromResults.source_id),
          source_name: String(fromResults.source_name || ''),
          entrypoint: String(fromResults.entrypoint || 'demo.py'),
          ok: fromResults.ok === true,
          behavioral: fromResults.behavioral,
        };
      }
      return null;
    }, [job.results]);

    const { data: recentImports } = useQuery(
      ['arxiv-imports', 'recent'],
      () => apiClient.listArxivImports({ limit: 50, offset: 0 }),
      { enabled: hasArxivImportRequestArtifact && arxivSourceArtifacts.length === 0, staleTime: 30000 }
    );

    const arxivImportsFallback = useMemo(() => {
      if (arxivSourceArtifacts.length > 0) return [];
      const ids = new Set<string>();
      const arts = (job.output_artifacts as any[]) || [];
      for (const a of arts) {
        if (a?.type === 'arxiv_ingest_requested' && a?.source_id) ids.add(String(a.source_id));
      }
      const items = (recentImports as any)?.items || [];
      const found: any[] = [];
      for (const it of items) {
        if (ids.has(String(it?.id))) found.push(it);
      }
      return found;
    }, [arxivSourceArtifacts.length, job.output_artifacts, recentImports]);

    const desiredReadingListName = useMemo(() => {
      const cfgName = (job.config as any)?.reading_list_name;
      if (typeof cfgName === 'string' && cfgName.trim()) return cfgName.trim();
      return '';
    }, [job.config]);

    const { data: readingListsLookup } = useQuery(
      ['reading-lists', 'lookup', desiredReadingListName],
      () => apiClient.listReadingLists({ limit: 200, offset: 0 }),
      { enabled: !!desiredReadingListName && !readingListArtifact?.id, staleTime: 30000 }
    );

    const readingListByName = useMemo(() => {
      if (!desiredReadingListName) return null;
      const items = (readingListsLookup as any)?.items || [];
      const match = items.find((x: any) => String(x?.name || '').trim() === desiredReadingListName);
      return match || null;
    }, [readingListsLookup, desiredReadingListName]);

    const openDocument = (docId: string) => {
      if (!docId) return;
      navigate('/documents', { state: { openDocId: String(docId) } });
    };

    const openReadingList = (rlId: string) => {
      if (!rlId) return;
      navigate(`/reading-lists/${encodeURIComponent(String(rlId))}`);
    };

    const loadLog = useCallback(async () => {
      const requestId = ++logRequestIdRef.current;
      if (detailPanelMountedRef.current) {
        setLoadingLog(true);
      }
      try {
        const data = await apiClient.getAgentJobLog(job.id, 20);
        if (!detailPanelMountedRef.current || logRequestIdRef.current !== requestId) return;
        setLogData(data);
      } catch (error) {
        if (!detailPanelMountedRef.current || logRequestIdRef.current !== requestId) return;
        console.error('Failed to load log:', error);
      } finally {
        if (detailPanelMountedRef.current && logRequestIdRef.current === requestId) {
          setLoadingLog(false);
        }
      }
    }, [job.id]);

    const loadStepEvents = useCallback(async () => {
      const requestId = ++stepEventsRequestIdRef.current;
      if (detailPanelMountedRef.current) {
        setLoadingStepEvents(true);
      }
      try {
        const data = await apiClient.getAgentJobStepEvents(job.id, 120, 0);
        if (!detailPanelMountedRef.current || stepEventsRequestIdRef.current !== requestId) return;
        setStepEventsData(data);
      } catch (error) {
        if (!detailPanelMountedRef.current || stepEventsRequestIdRef.current !== requestId) return;
        console.error('Failed to load step events:', error);
      } finally {
        if (detailPanelMountedRef.current && stepEventsRequestIdRef.current === requestId) {
          setLoadingStepEvents(false);
        }
      }
    }, [job.id]);

    useEffect(() => {
      if (!showExecutionLog || loadingLog || logData) return;
      loadLog();
    }, [showExecutionLog, loadingLog, logData, loadLog]);

    useEffect(() => {
      if (!showStepEvents || loadingStepEvents || stepEventsData) return;
      loadStepEvents();
    }, [showStepEvents, loadingStepEvents, stepEventsData, loadStepEvents]);

    const submitCheckpointAction = (
      nextAction: 'approve' | 'reject' | 'edit' | 'skip' | 'resume'
    ) => {
      actionMutation.mutate({
        jobId: job.id,
        action: nextAction,
        checkpointNote: approvalNote.trim() || undefined,
      });
    };

    const submitEditedCheckpointAction = () => {
      let parsedParams: Record<string, any> = {};
      try {
        parsedParams = approvalEditParams.trim() ? JSON.parse(approvalEditParams) : {};
      } catch (e: any) {
        toast.error(`Invalid JSON params: ${e?.message || 'parse error'}`);
        return;
      }

      const patch: Record<string, any> = {
        params: parsedParams,
      };
      if (approvalEditTool.trim()) patch.tool = approvalEditTool.trim();
      if (approvalEditPurpose.trim()) patch.purpose = approvalEditPurpose.trim();

      actionMutation.mutate({
        jobId: job.id,
        action: 'edit',
        checkpointNote: approvalNote.trim() || undefined,
        checkpointActionPatch: patch,
      });
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
                <h2 className="text-lg font-semibold">{job.name}</h2>
                <p className="text-sm text-gray-500">{typeConfig.label}</p>
              </div>
            </div>
            <div className={`flex items-center gap-1 px-3 py-1.5 rounded-full ${statusConfig.bgColor} ${statusConfig.color}`}>
              <StatusIcon className={`w-4 h-4 ${job.status === 'running' ? 'animate-spin' : ''}`} />
              <span className="font-medium capitalize">{job.status}</span>
            </div>
          </div>
          {(lineageRootNode || lineageParentNode) && (
            <div className="mt-1 mb-2 flex flex-wrap items-center gap-2 text-xs text-gray-600">
              <span className="text-gray-500">Relaunch lineage:</span>
              {lineageRootNode && (
                <button
                  type="button"
                  className="underline decoration-dotted hover:opacity-80"
                  onClick={() => navigate(buildAutonomousAgentsUrl(String(lineageRootNode?.id || '')))}
                >
                  Root {String(lineageRootNode?.name || 'job')}
                </button>
              )}
              {lineageParentNode && String(lineageParentNode?.id || '') !== String(lineageRootNode?.id || '') && (
                <>
                  <span>{'>'}</span>
                  <button
                    type="button"
                    className="underline decoration-dotted hover:opacity-80"
                    onClick={() => navigate(buildAutonomousAgentsUrl(String(lineageParentNode?.id || '')))}
                  >
                    Parent {String(lineageParentNode?.name || 'job')}
                  </button>
                </>
              )}
              <span>{'>'}</span>
              <span className="font-medium text-gray-800">Current {job.name}</span>
              <button
                type="button"
                className="ml-1 underline decoration-dotted hover:opacity-80"
                onClick={() => { void copyLineageLink(); }}
                title="Copy deep link with current lineage mode"
              >
                Copy lineage link
              </button>
              <button
                type="button"
                className="underline decoration-dotted hover:opacity-80"
                onClick={() => setShowShortcutHelp((v) => !v)}
                title="Show keyboard shortcuts"
              >
                Shortcuts (?)
              </button>
            </div>
          )}
          {showShortcutHelp && (
            <div className="mt-2 mb-2 rounded border border-gray-200 bg-gray-50 p-2 text-xs text-gray-700">
              <div><span className="font-mono">[</span> open parent job</div>
              <div><span className="font-mono">]</span> open latest child job</div>
              <div><span className="font-mono">c</span> copy lineage link</div>
              <div><span className="font-mono">x</span> close job detail</div>
              <div><span className="font-mono">?</span> toggle this help</div>
              <div><span className="font-mono">Esc</span> close this help</div>
            </div>
          )}

          {(queueManagedApproval || queueManagedRecovery) && (
            <div className="mt-3 rounded-lg border border-amber-200 bg-amber-50 p-3">
              <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
                <div className="min-w-0">
                  <div className="text-sm font-medium text-amber-900">
                    {queueManagedApproval ? 'Approval action is managed in Checkpoint Queue' : 'Recovery action is managed in Checkpoint Queue'}
                  </div>
                  <div className="mt-1 text-xs text-amber-800">
                    {queueManagedApproval
                      ? String(approvalCheckpoint?.message || 'This job is paused on a pending approval checkpoint.')
                      : String(graphHealthReasons[0] || 'This job has an open recovery path that should be triaged from the queue.')}
                  </div>
                  {queueManagedRecovery && graphRecommendedActions.length > 0 ? (
                    <div className="mt-1 text-xs text-amber-800">
                      Recommended next step: {graphRecommendedActions[0]}
                    </div>
                  ) : null}
                </div>
                <div className="flex shrink-0 gap-2">
                  <Button size="sm" variant="primary" onClick={openCheckpointQueue}>
                    Open Queue
                  </Button>
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={() => {
                      setSelectedJob(null);
                      navigate(buildAutonomousAgentsUrl(), { replace: true });
                    }}
                  >
                    Queue Only
                  </Button>
                </div>
              </div>
            </div>
          )}

          {/* Actions */}
          <div className="flex items-center gap-2 mt-3">
            <Button
              size="sm"
              variant="ghost"
              onClick={() => {
                setSelectedJob(null);
                navigate(buildAutonomousAgentsUrl(), { replace: true });
              }}
              title="Close details"
            >
              <XCircle className="w-4 h-4 mr-1" />
              Close
            </Button>
            {job.status === 'running' && (
              <Button
                size="sm"
                variant="secondary"
                onClick={() => actionMutation.mutate({ jobId: job.id, action: 'pause' })}
                disabled={actionMutation.isLoading}
              >
                <Pause className="w-4 h-4 mr-1" />
                Pause
              </Button>
            )}
            {job.status === 'paused' && !approvalCheckpoint && !queueManagedRecovery && !swarmNeedsReview && (
              <Button
                size="sm"
                variant="primary"
                onClick={() => actionMutation.mutate({ jobId: job.id, action: 'resume' })}
                disabled={actionMutation.isLoading}
              >
                <Play className="w-4 h-4 mr-1" />
                {isRepoBugTriageJob && codePatchRecovery?.can_resume_verification ? 'Resume verification' : 'Resume'}
              </Button>
            )}
            {queueManagedApproval ? (
              <Button size="sm" variant="primary" onClick={openCheckpointQueue}>
                <AlertCircle className="w-4 h-4 mr-1" />
                Open Checkpoint Queue
              </Button>
            ) : null}
            {queueManagedRecovery && terminalJob ? (
              <Button
                size="sm"
                variant="secondary"
                onClick={() => actionMutation.mutate({ jobId: job.id, action: 'restart' })}
                disabled={actionMutation.isLoading}
              >
                <RotateCcw className="w-4 h-4 mr-1" />
                Restart recovery
              </Button>
            ) : null}
            {queueManagedRecovery && terminalJob && supportsQuickStartRelaunch ? (
              <Button
                size="sm"
                variant="primary"
                onClick={() => actionMutation.mutate({ jobId: job.id, action: 'relaunch' })}
                disabled={actionMutation.isLoading}
                title="Create a new quick-start run from the same launch context"
              >
                <Play className="w-4 h-4 mr-1" />
                Relaunch clean run
              </Button>
            ) : null}
            {queueManagedRecovery && (
              <Button size="sm" variant="ghost" onClick={openCheckpointQueue}>
                <AlertCircle className="w-4 h-4 mr-1" />
                Open Checkpoint Queue
              </Button>
            )}
            {canLaunchSwarmTieBreaker && (
              <Button
                size="sm"
                variant="secondary"
                onClick={() => actionMutation.mutate({ jobId: job.id, action: 'launch_tie_breaker' })}
                disabled={actionMutation.isLoading}
              >
                <RotateCcw className="w-4 h-4 mr-1" />
                Relaunch verifier
              </Button>
            )}
            {canManualPromoteSwarmCandidate && (
              <Button
                size="sm"
                variant="primary"
                onClick={() =>
                  actionMutation.mutate({
                    jobId: job.id,
                    action: 'promote_swarm_candidate',
                    actionPayload: {
                      candidate_job_id: String((swarmSummary?.candidate_paths || [])[0]?.job_id || ''),
                    },
                  })
                }
                disabled={actionMutation.isLoading}
              >
                <Play className="w-4 h-4 mr-1" />
                Promote top path
              </Button>
            )}
            {swarmNeedsReview && isCodingSwarmJob && !createCodingBacklogMutation.isLoading && (
              <Button
                size="sm"
                variant="ghost"
                onClick={() => {
                  const cfg = (job.config || {}) as Record<string, any>;
                  const topCandidate = ((swarmSummary?.candidate_paths || [])[0] || {}) as Record<string, any>;
                  const presetLabel =
                    launchMode === 'quick_start_build_break_swarm'
                      ? 'Build swarm'
                      : launchMode === 'quick_start_frontend_regression_swarm'
                        ? 'Frontend swarm'
                        : 'Bug swarm';
                  createCodingBacklogMutation.mutate({
                    title: `${presetLabel} review - ${String(job.name || 'autonomous job').slice(0, 72)}`,
                    portfolio_goal: String(job.goal || 'Review bug triage swarm findings and implement the best repair path').slice(0, 2000),
                    source_id: String(cfg.source_id || ''),
                    scope: String(cfg.scope || 'auto') || 'auto',
                    failure_symptom: String(cfg.failure_symptom || '').trim() || undefined,
                    error_output: String(cfg.error_output || '').trim() || undefined,
                    file_paths: Array.from(
                      new Set(
                        ((swarmSummary?.candidate_paths || []) as Array<Record<string, any>>)
                          .flatMap((row) => (Array.isArray(row?.suspect_files) ? row.suspect_files : []))
                          .map((value) => String(value || '').trim())
                          .filter(Boolean)
                      )
                    ).slice(0, 12),
                    commands: Array.isArray(swarmSummary?.recommended_commands)
                      ? swarmSummary.recommended_commands.slice(0, 6).map((value: unknown) => String(value || '').trim()).filter(Boolean)
                      : [],
                    visibility: Array.isArray(swarmSummary?.shared_with_user_ids) && swarmSummary.shared_with_user_ids.length > 0 ? 'shared' : 'private',
                    shared_with_user_ids: Array.isArray(swarmSummary?.shared_with_user_ids) ? swarmSummary.shared_with_user_ids.slice(0, 200).map((value: unknown) => String(value || '').trim()).filter(Boolean) : [],
                    assigned_user_id: String(swarmSummary?.assigned_user_id || '').trim() || undefined,
                    assigned_by_user_id: String(swarmSummary?.assigned_by_user_id || '').trim() || undefined,
                    assigned_at: String(swarmSummary?.assigned_at || '').trim() || undefined,
                    collaboration: {
                      owner_user_id: String(swarmSummary?.owner_user_id || job.user_id || '').trim() || undefined,
                      visibility: Array.isArray(swarmSummary?.shared_with_user_ids) && swarmSummary.shared_with_user_ids.length > 0 ? 'shared' : 'private',
                      shared_with_user_ids: Array.isArray(swarmSummary?.shared_with_user_ids) ? swarmSummary.shared_with_user_ids.slice(0, 200).map((value: unknown) => String(value || '').trim()).filter(Boolean) : [],
                      assigned_user_id: String(swarmSummary?.assigned_user_id || '').trim() || undefined,
                      assigned_by_user_id: String(swarmSummary?.assigned_by_user_id || '').trim() || undefined,
                      assigned_at: String(swarmSummary?.assigned_at || '').trim() || undefined,
                      note: swarmReviewReason || undefined,
                    },
                    lineage: {
                      originating_swarm_job_id: String(job.id || ''),
                      originating_swarm_preset: String(((cfg.quick_start as any)?.preset_key || cfg.coding_swarm_preset_key || launchMode || '')).trim(),
                      originating_swarm_review_reason: swarmReviewReason || undefined,
                      originating_swarm_candidate_job_id: String(topCandidate.job_id || '').trim() || undefined,
                      originating_swarm_candidate_role: String(topCandidate.role || '').trim() || undefined,
                      originating_swarm_candidate_index: 0,
                      originating_swarm_route_mode: 'manual',
                    },
                    start_immediately: false,
                  });
                }}
                disabled={createCodingBacklogMutation.isLoading || !String((job.config as any)?.source_id || '').trim()}
              >
                <Layers className="w-4 h-4 mr-1" />
                Create backlog item
              </Button>
            )}
            {['pending', 'running', 'paused'].includes(job.status) && (
              <Button
                size="sm"
                variant="ghost"
                onClick={() => actionMutation.mutate({ jobId: job.id, action: 'cancel' })}
                disabled={actionMutation.isLoading}
              >
                <XCircle className="w-4 h-4 mr-1" />
                Cancel
              </Button>
            )}
            {terminalJob && !queueManagedRecovery && (
              <Button
                size="sm"
                variant="secondary"
                onClick={() => actionMutation.mutate({ jobId: job.id, action: 'restart' })}
                disabled={actionMutation.isLoading}
              >
                <RotateCcw className="w-4 h-4 mr-1" />
                {isRepoBugTriageJob ? 'Retry with refined plan' : 'Restart'}
              </Button>
            )}
            {terminalJob &&
              supportsQuickStartRelaunch &&
              !queueManagedRecovery && (
              <Button
                size="sm"
                variant="primary"
                onClick={() => actionMutation.mutate({ jobId: job.id, action: 'relaunch' })}
                disabled={actionMutation.isLoading}
                title="Create a new quick-start run from the same launch context"
              >
                <Play className="w-4 h-4 mr-1" />
                {isRepoBugTriageJob ? 'Relaunch clean run' : 'Relaunch'}
              </Button>
            )}
            {canPromoteDomainResearch && (
              <Button
                size="sm"
                variant="primary"
                onClick={() => setShowPromotionPanel((prev) => !prev)}
                disabled={promoteDomainResearchMutation.isLoading}
              >
                <Sparkles className="w-4 h-4 mr-1" />
                {showPromotionPanel ? 'Hide Promotion' : 'Promote to Monitor'}
              </Button>
            )}
            {/* Export button - available for completed or failed jobs with results */}
            {['completed', 'failed'].includes(job.status) && (
              <Button
                size="sm"
                variant="secondary"
                onClick={() => {
                  setExportingJob(job);
                  setShowExportModal(true);
                }}
              >
                <Download className="w-4 h-4 mr-1" />
                Export
              </Button>
            )}
            <Button
              size="sm"
              variant="secondary"
              onClick={async () => {
                const link = `${window.location.origin}${buildAutonomousAgentsUrl(String(job.id))}`;
                try {
                  if (navigator?.clipboard?.writeText) {
                    await navigator.clipboard.writeText(link);
                    toast.success('Job link copied');
                  } else {
                    toast.error('Clipboard copy is not available in this browser');
                  }
                } catch {
                  toast.error('Failed to copy job link');
                }
              }}
              title="Copy deep link to this job (preserves current filters)"
            >
              <Link2 className="w-4 h-4 mr-1" />
              Copy Link
            </Button>
            {canSaveAsPlaybook && (
              <Button
                size="sm"
                variant="secondary"
                onClick={() => saveAsPlaybookMutation.mutate()}
                disabled={saveAsPlaybookMutation.isLoading}
                title={recoveryPlaybookDefaults
                  ? 'Save this recovery job as a reusable playbook (chain definition)'
                  : 'Save this job chain as a reusable playbook (chain definition)'}
              >
                <GitBranch className="w-4 h-4 mr-1" />
                {saveAsPlaybookMutation.isLoading
                  ? 'Saving…'
                  : recoveryPlaybookDefaults
                    ? 'Save recovery playbook'
                    : 'Save playbook'}
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
              disabled={job.status === 'running' || deleteMutation.isLoading}
            >
              <Trash2 className="w-4 h-4 mr-1" />
              Delete
            </Button>
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-4">
          {verificationParentJobId && verificationTaskId && (
            <div className="mb-4 rounded-lg border border-violet-200 bg-violet-50 p-3">
              <div className="text-sm font-medium text-violet-900">R&D evidence verifier</div>
              <div className="mt-1 font-mono text-xs text-violet-700">{verificationTaskId}</div>
              <Button
                size="sm"
                variant="secondary"
                className="mt-2"
                onClick={() => {
                  setSelectedJob(null);
                  navigate(
                    `/autonomous-agents?job=${encodeURIComponent(verificationParentJobId)}`
                      + `&verification_task=${encodeURIComponent(verificationTaskId)}`
                  );
                }}
              >
                <Target className="mr-1 h-3.5 w-3.5" />
                Open parent evidence
              </Button>
            </div>
          )}

          {Boolean((job.results as any)?.evaluation_outcome) && (
            <AutonomousRndVerificationPanel
              jobId={String(job.id)}
              defaultResearchNoteId={String((job.config as any)?.research_note_id || '')}
              defaultSourceId={String((job.config as any)?.source_id || '')}
              focusTaskId={focusedVerificationTaskId}
              onOpenAgentJob={(agentJobId) => {
                setSelectedJob(null);
                navigate(buildAutonomousAgentsUrl(agentJobId));
              }}
            />
          )}

          {/* Goal */}
          <div className="mb-4">
            <h3 className="text-sm font-medium text-gray-700 mb-1 flex items-center gap-1">
              <Target className="w-4 h-4" />
              Goal
            </h3>
            <p className="text-sm text-gray-600 bg-gray-50 rounded-lg p-3">{job.goal}</p>
          </div>

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
              {job.current_phase && <span>{job.current_phase}</span>}
            </div>
          </div>

          {/* Resource usage */}
          <div className="mb-4">
            <h3 className="text-sm font-medium text-gray-700 mb-2">Resource Usage</h3>
            <div className="grid grid-cols-3 gap-3">
              <div className="bg-gray-50 rounded-lg p-3 text-center">
                <p className="text-xs text-gray-500">Iterations</p>
                <p className="text-lg font-semibold">{job.iteration}/{job.max_iterations}</p>
              </div>
              <div className="bg-gray-50 rounded-lg p-3 text-center">
                <p className="text-xs text-gray-500">Tool Calls</p>
                <p className="text-lg font-semibold">{job.tool_calls_used}/{job.max_tool_calls}</p>
              </div>
              <div className="bg-gray-50 rounded-lg p-3 text-center">
                <p className="text-xs text-gray-500">LLM Calls</p>
                <p className="text-lg font-semibold">{job.llm_calls_used}/{job.max_llm_calls}</p>
              </div>
            </div>
          </div>

          {/* Timing */}
          <div className="mb-4">
            <h3 className="text-sm font-medium text-gray-700 mb-2">Timing</h3>
            <div className="grid grid-cols-2 gap-3 text-sm">
              <div>
                <span className="text-gray-500">Created:</span>
                <span className="ml-2">{new Date(job.created_at).toLocaleString()}</span>
              </div>
              {job.started_at && (
                <div>
                  <span className="text-gray-500">Started:</span>
                  <span className="ml-2">{new Date(job.started_at).toLocaleString()}</span>
                </div>
              )}
              {job.completed_at && (
                <div>
                  <span className="text-gray-500">Completed:</span>
                  <span className="ml-2">{new Date(job.completed_at).toLocaleString()}</span>
                </div>
              )}
              <div>
                <span className="text-gray-500">Duration:</span>
                <span className="ml-2">{formatDuration(job.started_at, job.completed_at)}</span>
              </div>
            </div>
          </div>

          {schedulerSummaryLines.length > 0 && (
            <div className="mb-4">
              <h3 className="text-sm font-medium text-gray-700 mb-2">Scheduler</h3>
              <div className="rounded-lg border border-gray-200 bg-gray-50 p-3 text-sm text-gray-700 space-y-1">
                {schedulerSummaryLines.slice(0, 10).map((line) => (
                  <div key={line}>{line}</div>
                ))}
              </div>
            </div>
          )}

          {/* Launch context */}
          {(launchMode || launchQuickStart || launchResult) && (
            <div className="mb-4">
              <h3 className="text-sm font-medium text-gray-700 mb-2">Launch Context</h3>
              <div className="bg-indigo-50 border border-indigo-100 rounded-lg p-3">
                <div className="flex flex-wrap gap-3 text-xs text-indigo-800">
                  <span>Mode: {launchMode || 'manual/legacy'}</span>
                  {(launchQuickStart?.profile || launchResult?.quick_start_profile) && (
                    <span>Profile: {String(launchQuickStart?.profile || launchResult?.quick_start_profile)}</span>
                  )}
                  {(launchQuickStart?.version || launchResult?.quick_start_version) && (
                    <span>Version: {String(launchQuickStart?.version || launchResult?.quick_start_version)}</span>
                  )}
                  {executionMode && (
                    <span>Execution mode: {executionMode.replace(/_/g, ' ')}</span>
                  )}
                  {(launchQuickStart?.source_type || launchResult?.source_type) && (
                    <span>Source type: {String(launchQuickStart?.source_type || launchResult?.source_type)}</span>
                  )}
                  {(launchQuickStart?.source_name || launchResult?.source_name) && (
                    <span>Source: {String(launchQuickStart?.source_name || launchResult?.source_name)}</span>
                  )}
                  {launchSourceId && (
                    <span title={launchSourceId}>Source ID: {launchSourceId.slice(0, 8)}</span>
                  )}
                  {launchRelaunchFromJobId && (
                    <button
                      type="button"
                      className="underline decoration-dotted hover:opacity-80"
                      onClick={() => navigate(buildAutonomousAgentsUrl(launchRelaunchFromJobId))}
                      title="Open original job"
                    >
                      Relaunched from: {launchRelaunchFromJobId.slice(0, 8)}
                    </button>
                  )}
                  {launchLog?.timestamp && (
                    <span>Logged: {new Date(String(launchLog.timestamp)).toLocaleString()}</span>
                  )}
                  {launchSearchQuery && (
                    <span className="truncate max-w-[420px]">Search: {launchSearchQuery}</span>
                  )}
                  {launchCommandsCount > 0 && (
                    <span>Commands: {launchCommandsCount}</span>
                  )}
                  {launchFilePathsCount > 0 && (
                    <span>Focused files: {launchFilePathsCount}</span>
                  )}
                </div>
                {launchFilePaths.length > 0 && (
                  <div className="mt-2 text-xs text-indigo-800">
                    <div className="font-medium mb-1">Focused file paths</div>
                    <ul className="space-y-1">
                      {launchFilePaths.slice(0, 6).map((p, idx) => (
                        <li key={`${idx}-${p.slice(0, 24)}`} className="font-mono truncate">- {p}</li>
                      ))}
                    </ul>
                  </div>
                )}
                {launchCommands.length > 0 && (
                  <div className="mt-2 text-xs text-indigo-800">
                    <div className="font-medium mb-1">Launch commands</div>
                    <ul className="space-y-1">
                      {launchCommands.slice(0, 4).map((cmd, idx) => (
                        <li key={`${idx}-${cmd.slice(0, 24)}`} className="font-mono truncate">- {cmd}</li>
                      ))}
                    </ul>
                  </div>
                )}
                {(canPromoteDomainResearch || promotedProfileId || promotedPortfolioId || promotionStatus) && (
                  <div className="mt-3 rounded-lg border border-cyan-200 bg-cyan-50 p-3">
                    <div className="flex flex-wrap items-center gap-2 text-xs text-cyan-900">
                      <span className="font-medium">Promotion</span>
                      {promotionStatus ? (
                        <span className="rounded-full border border-cyan-200 bg-white px-2 py-0.5">
                          {promotionStatus.replace(/_/g, ' ')}
                        </span>
                      ) : (
                        <span className="rounded-full border border-cyan-200 bg-white px-2 py-0.5">
                          Eligible
                        </span>
                      )}
                      {promotedProfileId ? (
                        <Button
                          size="sm"
                          variant="ghost"
                          onClick={() => navigate(buildAutonomousAgentsUrl(undefined, { tab: 'domain', profileId: promotedProfileId }), { replace: true })}
                          className="!px-2 !py-1 !h-auto text-xs"
                        >
                          Open monitor
                        </Button>
                      ) : null}
                      {promotedPortfolioId ? (
                        <Button
                          size="sm"
                          variant="ghost"
                          onClick={() => navigate(buildAutonomousAgentsUrl(undefined, { tab: 'fleet', fleetId: promotedPortfolioId }), { replace: true })}
                          className="!px-2 !py-1 !h-auto text-xs"
                        >
                          Open fleet
                        </Button>
                      ) : null}
                    </div>
                    {showPromotionPanel && canPromoteDomainResearch && (
                      <div className="mt-3 grid grid-cols-1 md:grid-cols-2 gap-3 text-sm text-cyan-950">
                        <div>
                          <label className="block text-xs font-medium mb-1">Monitor title</label>
                          <input
                            className="w-full rounded border border-cyan-200 bg-white px-3 py-2 text-sm"
                            value={promotionDraft.title}
                            onChange={(e) => setPromotionDraft((prev) => ({ ...prev, title: e.target.value }))}
                          />
                        </div>
                        <div>
                          <label className="block text-xs font-medium mb-1">Cadence in minutes</label>
                          <input
                            className="w-full rounded border border-cyan-200 bg-white px-3 py-2 text-sm"
                            value={promotionDraft.interval_minutes}
                            onChange={(e) => setPromotionDraft((prev) => ({ ...prev, interval_minutes: e.target.value }))}
                          />
                        </div>
                        <div>
                          <label className="block text-xs font-medium mb-1">Promotion scope</label>
                          <select
                            className="w-full rounded border border-cyan-200 bg-white px-3 py-2 text-sm"
                            value={promotionDraft.target_mode}
                            onChange={(e) => setPromotionDraft((prev) => ({ ...prev, target_mode: e.target.value as 'profile_only' | 'profile_with_portfolio' }))}
                          >
                            <option value="profile_only">Profile only</option>
                            <option value="profile_with_portfolio">Profile + fleet</option>
                          </select>
                        </div>
                        {promotionDraft.target_mode === 'profile_with_portfolio' ? (
                          <div>
                            <label className="block text-xs font-medium mb-1">Fleet target</label>
                            <select
                              className="w-full rounded border border-cyan-200 bg-white px-3 py-2 text-sm"
                              value={promotionDraft.portfolio_mode}
                              onChange={(e) => setPromotionDraft((prev) => ({ ...prev, portfolio_mode: e.target.value as 'existing' | 'new' }))}
                            >
                              <option value="new">Create new fleet</option>
                              <option value="existing">Attach to existing fleet</option>
                            </select>
                          </div>
                        ) : null}
                        {promotionDraft.target_mode === 'profile_with_portfolio' && promotionDraft.portfolio_mode === 'existing' ? (
                          <div className="md:col-span-2">
                            <label className="block text-xs font-medium mb-1">Existing fleet</label>
                            <select
                              className="w-full rounded border border-cyan-200 bg-white px-3 py-2 text-sm"
                              value={promotionDraft.portfolio_id}
                              disabled={promotionResearchPortfoliosLoading}
                              onChange={(e) => setPromotionDraft((prev) => ({ ...prev, portfolio_id: e.target.value }))}
                            >
                              <option value="">{promotionResearchPortfoliosLoading ? 'Loading fleets…' : 'Select a fleet'}</option>
                              {promotionResearchPortfolios.map((portfolio) => (
                                <option key={String(portfolio.id)} value={String(portfolio.id)}>
                                  {String(portfolio.title || portfolio.id)}
                                </option>
                              ))}
                            </select>
                          </div>
                        ) : null}
                        {promotionDraft.target_mode === 'profile_with_portfolio' && promotionDraft.portfolio_mode === 'new' ? (
                          <div className="md:col-span-2">
                            <label className="block text-xs font-medium mb-1">New fleet title</label>
                            <input
                              className="w-full rounded border border-cyan-200 bg-white px-3 py-2 text-sm"
                              value={promotionDraft.portfolio_title}
                              onChange={(e) => setPromotionDraft((prev) => ({ ...prev, portfolio_title: e.target.value }))}
                            />
                          </div>
                        ) : null}
                        <label className="flex items-center gap-2 text-xs">
                          <input
                            type="checkbox"
                            checked={promotionDraft.start_profile_now}
                            onChange={(e) => setPromotionDraft((prev) => ({ ...prev, start_profile_now: e.target.checked }))}
                          />
                          Start recurring monitor now
                        </label>
                        {promotionDraft.target_mode === 'profile_with_portfolio' ? (
                          <label className="flex items-center gap-2 text-xs">
                            <input
                              type="checkbox"
                              checked={promotionDraft.run_portfolio_now}
                              onChange={(e) => setPromotionDraft((prev) => ({ ...prev, run_portfolio_now: e.target.checked }))}
                            />
                            Run fleet once now
                          </label>
                        ) : null}
                        <div className="md:col-span-2 text-xs text-cyan-900">
                          <div>Domain: {launchDomain || 'Unknown'}</div>
                          <div>Objective: {launchObjective || 'Unknown'}</div>
                        </div>
                        <div className="md:col-span-2 flex gap-2">
                          <Button
                            size="sm"
                            variant="primary"
                            disabled={
                              promoteDomainResearchMutation.isLoading
                              || !promotionDraft.title.trim()
                              || (promotionDraft.target_mode === 'profile_with_portfolio' && promotionDraft.portfolio_mode === 'existing' && !promotionDraft.portfolio_id)
                              || (promotionDraft.target_mode === 'profile_with_portfolio' && promotionDraft.portfolio_mode === 'new' && !promotionDraft.portfolio_title.trim())
                            }
                            onClick={() => {
                              const cfg = ((job.config || {}) as Record<string, any>) || {};
                              const payload: AgentJobPromoteDomainResearchRequest = {
                                target_mode: promotionDraft.target_mode,
                                profile: {
                                  title: promotionDraft.title.trim(),
                                  interval_minutes: Math.max(15, Number(promotionDraft.interval_minutes || 1440) || 1440),
                                  domain: String(cfg.domain || '').trim() || undefined,
                                  objective: String(cfg.objective || '').trim() || undefined,
                                  customer_context: String(cfg.customer_context || '').trim() || undefined,
                                  source_scope: String(cfg.source_scope || '').trim() || undefined,
                                  track_type: String(cfg.track_type || '').trim() || undefined,
                                  research_mode: String(cfg.research_mode || '').trim() || undefined,
                                  monitor_queries: Array.isArray(cfg.monitor_queries) ? cfg.monitor_queries : undefined,
                                  repo_source_ids: Array.isArray(cfg.repo_source_ids) ? cfg.repo_source_ids : undefined,
                                  benchmark_queries: Array.isArray(cfg.benchmark_queries) ? cfg.benchmark_queries : undefined,
                                  report_format: String(cfg.report_format || '').trim() || undefined,
                                  scoring_policy: cfg.scoring_policy,
                                  selection_policy: cfg.selection_policy,
                                  sandbox_profile_id: String(cfg.sandbox_profile_id || '').trim() || undefined,
                                  automation_profile: String(cfg.automation_profile || '').trim() || undefined,
                                  automation_policy: (cfg.automation_policy && typeof cfg.automation_policy === 'object') ? cfg.automation_policy : undefined,
                                  persist_artifacts: Boolean(cfg.persist_artifacts ?? true),
                                  auto_launch_follow_up: Boolean(cfg.auto_launch_follow_up ?? true),
                                  auto_create_experiment_plans: Boolean(cfg.auto_create_experiment_plans ?? true),
                                  confidence_threshold: Number(cfg.confidence_threshold || 0) || undefined,
                                  max_documents: Number(cfg.max_documents || 0) || undefined,
                                  max_papers: Number(cfg.max_papers || 0) || undefined,
                                },
                                start_profile_now: promotionDraft.start_profile_now,
                                run_portfolio_now: promotionDraft.target_mode === 'profile_with_portfolio' ? promotionDraft.run_portfolio_now : false,
                              };
                              if (promotionDraft.target_mode === 'profile_with_portfolio') {
                                if (promotionDraft.portfolio_mode === 'existing') {
                                  payload.portfolio_id = promotionDraft.portfolio_id;
                                } else {
                                  payload.portfolio = {
                                    title: promotionDraft.portfolio_title.trim(),
                                    objective: String(cfg.objective || '').trim() || undefined,
                                    sandbox_profile_id: String(cfg.sandbox_profile_id || '').trim() || undefined,
                                    automation_profile: String(cfg.automation_profile || '').trim() || undefined,
                                    automation_policy: (cfg.automation_policy && typeof cfg.automation_policy === 'object') ? cfg.automation_policy : undefined,
                                  };
                                }
                              }
                              promoteDomainResearchMutation.mutate({ jobId: String(job.id), data: payload });
                            }}
                          >
                            {promoteDomainResearchMutation.isLoading ? 'Promoting…' : 'Create monitor'}
                          </Button>
                          <Button
                            size="sm"
                            variant="ghost"
                            onClick={() => {
                              setShowPromotionPanel(false);
                              setPromotionDraft(buildDomainResearchPromotionDraft(job));
                            }}
                          >
                            Cancel
                          </Button>
                        </div>
                      </div>
                    )}
                  </div>
                )}
                {(lineageParentJobId || lineageLatestChildJobId) && (
                  <div className="mt-2 flex flex-wrap gap-2">
                    {lineageParentJobId && (
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={() => navigate(buildAutonomousAgentsUrl(lineageParentJobId))}
                        className="!px-2 !py-1 !h-auto text-xs"
                      >
                        Open parent
                      </Button>
                    )}
                    {lineageLatestChildJobId && (
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={() => navigate(buildAutonomousAgentsUrl(lineageLatestChildJobId))}
                        className="!px-2 !py-1 !h-auto text-xs"
                      >
                        Open latest child
                      </Button>
                    )}
                    <Button
                      size="sm"
                      variant="ghost"
                      onClick={() => {
                        setActiveTab('jobs');
                        setHasRelaunchChildrenFilter('');
                        setRelaunchFromJobIdFilter(String(job.id));
                      }}
                      className="!px-2 !py-1 !h-auto text-xs"
                      title="Show jobs relaunched from this job in the list"
                    >
                      Filter children
                    </Button>
                    {launchRelaunchFromJobId && (
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={() => {
                          setActiveTab('jobs');
                          setHasRelaunchChildrenFilter('');
                          setRelaunchFromJobIdFilter(launchRelaunchFromJobId);
                        }}
                        className="!px-2 !py-1 !h-auto text-xs"
                        title="Show sibling jobs (same relaunch parent) in the list"
                      >
                        Filter siblings
                      </Button>
                    )}
                  </div>
                )}
                {(lineageParentJobId || lineageLatestChildJobId) && (
                  <div className="mt-1 text-[11px] text-indigo-700/80">
                    Keyboard: <span className="font-mono">[</span> parent, <span className="font-mono">]</span> latest child
                  </div>
                )}
                {(lineageAnyTruncated || lineageExpanded) && (
                  <div className="mt-2">
                    <Button
                      size="sm"
                      variant="ghost"
                      onClick={() => {
                        const nextExpanded = !lineageExpanded;
                        setLineageExpanded(nextExpanded);
                        const params = new URLSearchParams(location.search);
                        if (nextExpanded) params.set('lx', 'full');
                        else params.delete('lx');
                        const qs = params.toString();
                        navigate(`${location.pathname}${qs ? `?${qs}` : ''}`, { replace: true });
                      }}
                      className="!px-2 !py-1 !h-auto text-xs"
                    >
                      {lineageExpanded ? 'Show compact lineage' : 'Load full lineage'}
                    </Button>
                  </div>
                )}
                {(lineageAncestors.length > 0 || lineageDescendants.length > 0) && (
                  <div className="mt-2 grid grid-cols-1 md:grid-cols-2 gap-2 text-xs text-indigo-900">
                    {lineageAncestors.length > 0 && (
                      <div className="bg-white/60 border border-indigo-100 rounded p-2">
                        <div className="font-medium mb-1">
                          Ancestors ({lineageAncestors.length}/{lineageAllAncestors.length})
                          {lineageAncestorsTruncated ? ' (truncated)' : ''}
                        </div>
                        <ul className="space-y-1">
                          {lineageAncestors.map((n: any) => (
                            <li key={`anc-${String(n?.id || '')}`}>
                              <button
                                type="button"
                                className="font-mono underline decoration-dotted hover:opacity-80"
                                onClick={() => navigate(buildAutonomousAgentsUrl(String(n?.id || '')))}
                              >
                                {String(n?.name || 'job')} ({String(n?.status || '')})
                              </button>
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}
                    {lineageDescendants.length > 0 && (
                      <div className="bg-white/60 border border-indigo-100 rounded p-2">
                        <div className="font-medium mb-1">
                          Descendants ({lineageDescendants.length}/{lineageAllDescendants.length})
                          {lineageDescendantsTruncated ? ' (truncated)' : ''}
                        </div>
                        <ul className="space-y-1">
                          {lineageDescendants.map((n: any) => (
                            <li key={`des-${String(n?.id || '')}`}>
                              <button
                                type="button"
                                className="font-mono underline decoration-dotted hover:opacity-80"
                                onClick={() => navigate(buildAutonomousAgentsUrl(String(n?.id || '')))}
                              >
                                {String(n?.name || 'job')} ({String(n?.status || '')})
                              </button>
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                )}
                {lineagePathNodes.length > 1 && (
                  <div className="mt-2 text-xs text-indigo-900">
                    <div className="font-medium mb-1">Lineage path</div>
                    <div className="flex flex-wrap items-center gap-1.5">
                      {lineagePathNodes.map((n: any, idx: number) => (
                        <React.Fragment key={`path-${String(n?.id || idx)}`}>
                          {idx > 0 && <span className="text-indigo-500">{'>'}</span>}
                          <button
                            type="button"
                            className="px-2 py-0.5 rounded border border-indigo-200 bg-white/70 hover:bg-white"
                            onClick={() => navigate(buildAutonomousAgentsUrl(String(n?.id || '')))}
                            title={String(n?.id || '')}
                          >
                            {String(n?.name || 'job')}
                          </button>
                        </React.Fragment>
                      ))}
                      {lineageLatestChildJobId && String(lineageLatestChildJobId) !== String(job.id) && (
                        <>
                          <span className="text-indigo-500">{'>'}</span>
                          <button
                            type="button"
                            className="px-2 py-0.5 rounded border border-cyan-200 bg-cyan-50 hover:bg-cyan-100"
                            onClick={() => navigate(buildAutonomousAgentsUrl(lineageLatestChildJobId))}
                            title={lineageLatestChildJobId}
                          >
                            Latest {String(lineageLatestChildNode?.name || lineageLatestChildJobId).slice(0, 28)}
                          </button>
                        </>
                      )}
                    </div>
                  </div>
                )}
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

          {/* Results summary */}
          {job.results && (
            <div className="mb-4">
              <h3 className="text-sm font-medium text-gray-700 mb-2">Results Summary</h3>
              <div className="bg-gray-50 rounded-lg p-3">
                {job.results.summary && (
                  <p className="text-sm text-gray-600 mb-2">{job.results.summary}</p>
                )}
                <div className="flex gap-4 text-sm text-gray-500">
                  {job.results.findings_count !== undefined && (
                    <span>Findings: {job.results.findings_count}</span>
                  )}
                  {job.results.actions_count !== undefined && (
                    <span>Actions: {job.results.actions_count}</span>
                  )}
                </div>
              </div>
            </div>
          )}

          {executionPlan.length > 0 && (
            <div className="mb-4">
              <div className="mb-2 flex flex-wrap items-center gap-2">
                <h3 className="text-sm font-medium text-gray-700">Execution Plan</h3>
                {executionMode && (
                  <span className="px-2 py-0.5 rounded-full border border-indigo-100 bg-indigo-50 text-indigo-700 text-[11px]">
                    {executionMode.replace(/_/g, ' ')}
                  </span>
                )}
                <span className={`px-2 py-0.5 rounded-full border text-[11px] ${
                  planCompleted
                    ? 'border-emerald-100 bg-emerald-50 text-emerald-700'
                    : 'border-amber-100 bg-amber-50 text-amber-700'
                }`}>
                  {planCompleted ? 'Plan complete' : 'Plan in progress'}
                </span>
              </div>
              <div className="bg-indigo-50 border border-indigo-100 rounded-lg p-3 space-y-2">
                {executionPlan.slice(0, 12).map((step: Record<string, any>, idx: number) => {
                  const status = String(step?.status || (idx === activePlanIndex ? 'in_progress' : 'pending')).toLowerCase();
                  const statusClass =
                    status === 'done'
                      ? 'bg-emerald-50 text-emerald-700 border-emerald-100'
                      : status === 'failed'
                        ? 'bg-red-50 text-red-700 border-red-100'
                        : status === 'skipped'
                          ? 'bg-gray-50 text-gray-700 border-gray-100'
                          : status === 'waiting_approval'
                            ? 'bg-rose-50 text-rose-700 border-rose-100'
                            : 'bg-indigo-50 text-indigo-700 border-indigo-100';
                  const title = String(step?.title || step?.objective || step?.step_id || `Step ${idx + 1}`).trim();
                  const isActive = idx === activePlanIndex;
                  return (
                    <div key={String(step?.step_id || idx)} className="bg-white border border-indigo-100 rounded p-2">
                      <div className="flex items-center justify-between gap-2">
                        <div className="text-xs text-gray-800 truncate">
                          <span className="font-medium">Step {idx + 1}.</span> {title}
                        </div>
                        <span className={`shrink-0 px-2 py-0.5 rounded-full text-[11px] border ${statusClass}`}>
                          {status.replace('_', ' ')}
                        </span>
                      </div>
                      {step?.objective && (
                        <div className="text-xs text-gray-600 mt-1 line-clamp-2">{String(step.objective)}</div>
                      )}
                      {Array.isArray(step?.suggested_tools) && step.suggested_tools.length > 0 && (
                        <div className="text-[11px] text-indigo-700 mt-1">
                          Tools: {step.suggested_tools.slice(0, 4).map((x: any) => String(x)).join(', ')}
                        </div>
                      )}
                      {isActive && (
                        <div className="text-[11px] text-indigo-600 mt-1">Active step</div>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {memoryPersistence && (
            <div className="mb-4">
              <h3 className="text-sm font-medium text-gray-700 mb-2">Memory Persistence</h3>
              <div className="bg-violet-50 border border-violet-100 rounded-lg p-3">
                <div className="flex flex-wrap gap-3 text-xs text-violet-800">
                  <span>Enabled: {memoryPersistence?.enabled ? 'yes' : 'no'}</span>
                  {memoryRuntime?.profile && (
                    <span>Profile: {String(memoryRuntime.profile)}</span>
                  )}
                  {memoryRuntime?.role && (
                    <span>Role: {String(memoryRuntime.role)}</span>
                  )}
                  {memoryRuntime?.limit !== undefined && memoryRuntime?.limit !== null && (
                    <span>Runtime limit: {Number(memoryRuntime.limit || 0)}</span>
                  )}
                  {memoryPersistence?.injected_count !== undefined && (
                    <span>Injected: {Number(memoryPersistence.injected_count || 0)}</span>
                  )}
                  {memoryExtractionView?.status && (
                    <span>
                      Extraction: {String(memoryExtractionView.status)}
                      {memoryExtractionView?.created_count !== undefined
                        ? ` (${Number(memoryExtractionView.created_count || 0)})`
                        : ''}
                    </span>
                  )}
                  {memoryExtractionView?.parsed_count !== undefined && (
                    <span>Parsed: {Number(memoryExtractionView.parsed_count || 0)}</span>
                  )}
                  {memoryExtractionView?.candidate_count !== undefined && (
                    <span>Candidates: {Number(memoryExtractionView.candidate_count || 0)}</span>
                  )}
                  {memoryExtractionView?.skipped_duplicates !== undefined && (
                    <span>Skipped duplicates: {Number(memoryExtractionView.skipped_duplicates || 0)}</span>
                  )}
                  {memoryExtractionView?.dedup_existing_signature_count !== undefined && (
                    <span>Existing signatures: {Number(memoryExtractionView.dedup_existing_signature_count || 0)}</span>
                  )}
                  {memoryExtractionView?.is_relaunch_chain && (
                    <span>Relaunch lineage dedup: yes</span>
                  )}
                </div>
                {memoryExtractionView?.relaunch_root_job_id && (
                  <div className="mt-2 text-xs text-violet-700 break-all">
                    Relaunch root: {String(memoryExtractionView.relaunch_root_job_id)}
                  </div>
                )}
                {memoryExtractionView?.error && (
                  <div className="mt-2 text-xs text-rose-700">
                    Error: {String(memoryExtractionView.error)}
                  </div>
                )}
              </div>
            </div>
          )}

          <div className="mb-4">
            <h3 className="text-sm font-medium text-gray-700 mb-2">Operator Interventions</h3>
            <div className="bg-amber-50 border border-amber-200 rounded-lg p-3">
              <RecoveryAuditPanel
                className="mb-3"
                latestAction={operatorInterventionSummary.latestLabel}
                latestOutcome={operatorInterventionSummary.latestOutcome}
                latestOutcomeReason={operatorInterventionSummary.latestOutcomeReason}
                recoveryReason={graphHealthReasons[0]}
                nextStep={graphRecommendedActions[0]}
              />
              {operatorInterventions.length === 0 ? (
                <div className="text-xs text-gray-500">No operator interventions recorded yet.</div>
              ) : (
                <>
                  <div className="text-xs text-gray-500 mb-2">
                    {operatorInterventions.length} intervention(s)
                  </div>
                  <div className="space-y-1.5 max-h-48 overflow-y-auto">
                    {operatorInterventions
                      .slice(-10)
                      .reverse()
                      .map((entry: AgentJobOperatorIntervention, idx: number) => {
                        const action = String(entry?.action || 'intervention').replace(/_/g, ' ');
                        const eventAt = entry?.at ? new Date(String(entry.at)).toLocaleString() : '';
                        const statusBefore = String(entry?.job_status_before || '').trim();
                        const statusAfter = String(entry?.job_status_after || '').trim();
                        const note = String(entry?.note || '').trim();
                        const metadata = (entry?.metadata && typeof entry.metadata === 'object')
                          ? entry.metadata
                          : null;
                        const tool = metadata?.tool ? String(metadata.tool) : '';
                        const stepId = metadata?.plan_step_id ? String(metadata.plan_step_id) : '';
                        const newJobId = metadata?.new_job_id ? String(metadata.new_job_id) : '';
                        return (
                          <div key={`${idx}-${String(entry?.at || '')}-${String(entry?.action || '')}`} className="text-xs text-amber-900 bg-white border border-amber-200 rounded p-2">
                            <div className="font-medium text-amber-900">
                              {action}
                              {statusBefore || statusAfter ? ` • ${statusBefore || '?'} -> ${statusAfter || '?'}` : ''}
                            </div>
                            <div className="text-amber-800">
                              {eventAt}
                              {tool ? ` • tool ${tool}` : ''}
                              {stepId ? ` • step ${stepId}` : ''}
                              {newJobId ? ` • new job ${newJobId}` : ''}
                            </div>
                            {note && (
                              <div className="text-amber-700 mt-1">{note}</div>
                            )}
                          </div>
                        );
                      })}
                  </div>
                </>
              )}
            </div>
          </div>

          <div className="mb-4">
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-sm font-medium text-gray-700">Step Events</h3>
              <div className="flex items-center gap-2">
                {showStepEvents ? (
                  <Button size="sm" variant="ghost" onClick={loadStepEvents} disabled={loadingStepEvents}>
                    <RefreshCw className={`w-3 h-3 mr-1 ${loadingStepEvents ? 'animate-spin' : ''}`} />
                    Refresh
                  </Button>
                ) : null}
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={() => setShowStepEvents((current) => !current)}
                >
                  {showStepEvents ? 'Hide Step Events' : 'Show Step Events'}
                </Button>
              </div>
            </div>
            <div className="bg-slate-50 border border-slate-200 rounded-lg p-3">
              {!showStepEvents ? (
                <div className="text-xs text-gray-500">Step events load on demand.</div>
              ) : loadingStepEvents ? (
                <div className="text-xs text-gray-500">Loading step events…</div>
              ) : !stepEventsData || !Array.isArray(stepEventsData.items) || stepEventsData.items.length === 0 ? (
                <div className="text-xs text-gray-500">No step events yet.</div>
              ) : (
                <>
                  <div className="text-xs text-gray-500 mb-2">
                    {stepEventsData.total} event(s)
                    {stepEventsData.source ? ` • source: ${String(stepEventsData.source)}` : ''}
                  </div>
                  <div className="space-y-1.5 max-h-56 overflow-y-auto">
                    {stepEventsData.items
                      .slice(-30)
                      .reverse()
                      .map((ev: Record<string, any>, idx: number) => {
                        const evType = String(ev?.type || 'event').replace(/_/g, ' ');
                        const evStep = ev?.plan_step_id ? ` (${String(ev.plan_step_id)})` : '';
                        const evTool = ev?.tool ? ` • ${String(ev.tool)}` : '';
                        const evNote = ev?.note ? ` • ${String(ev.note).slice(0, 120)}` : '';
                        const evAt = ev?.at ? new Date(String(ev.at)).toLocaleString() : '';
                        return (
                          <div key={`${idx}-${String(ev?.at || '')}-${String(ev?.type || '')}`} className="text-xs text-slate-700 bg-white border border-slate-200 rounded p-2">
                            <div className="font-medium text-slate-800">
                              {evType}{evStep}{evTool}
                            </div>
                            <div className="text-slate-600">
                              {evAt}
                              {ev?.iteration !== undefined ? ` • iter ${Number(ev.iteration || 0)}` : ''}
                              {ev?.plan_step_index !== undefined && Number(ev.plan_step_index) >= 0 ? ` • plan idx ${Number(ev.plan_step_index)}` : ''}
                              {evNote}
                            </div>
                          </div>
                        );
                      })}
                  </div>
                </>
              )}
            </div>
          </div>

          {/* Approval checkpoint */}
          {approvalCheckpoint && (
            <div className="mb-4">
              <h3 className="text-sm font-medium text-gray-700 mb-2">Approval Checkpoint</h3>
              <div className="bg-rose-50 border border-rose-100 rounded-lg p-3">
                <p className="text-sm text-rose-800">
                  {String(approvalCheckpoint?.message || 'Human approval required before next action.')}
                </p>
                <div className="mt-2 flex flex-wrap gap-3 text-xs text-rose-700">
                  {approvalCheckpoint?.iteration !== undefined && (
                    <span>Iteration: {Number(approvalCheckpoint.iteration || 0)}</span>
                  )}
                  {approvalCheckpoint?.action?.tool && (
                    <span>Next tool: {String(approvalCheckpoint.action.tool)}</span>
                  )}
                  {approvalCheckpoint?.created_at && (
                    <span>Created: {new Date(String(approvalCheckpoint.created_at)).toLocaleString()}</span>
                  )}
                </div>
                {Array.isArray(approvalCheckpoint?.reasons) && approvalCheckpoint.reasons.length > 0 && (
                  <ul className="mt-2 text-xs text-rose-700 space-y-1">
                    {approvalCheckpoint.reasons.slice(0, 6).map((reason: string, idx: number) => (
                      <li key={`${idx}-${reason.slice(0, 24)}`}>- {reason}</li>
                    ))}
                  </ul>
                )}
                {job.status === 'paused' && (
                  <div className="mt-3 space-y-3">
                    <div className="flex flex-wrap gap-2">
                      <Button size="sm" variant="primary" onClick={openCheckpointQueue}>
                        Open in Checkpoint Queue
                      </Button>
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={() => setShowApprovalEdit((v) => !v)}
                        disabled={actionMutation.isLoading}
                      >
                        {showApprovalEdit ? 'Hide Advanced Edit' : 'Advanced Edit'}
                      </Button>
                    </div>
                    {showApprovalEdit && (
                      <div className="bg-white border border-rose-200 rounded p-2 space-y-2">
                        <div className="text-xs text-rose-700">
                          Use the queue for the normal approval path. This form is kept for inspection and exceptional edits.
                        </div>
                        <textarea
                          className="w-full rounded border border-rose-200 bg-white p-2 text-xs text-rose-900"
                          rows={2}
                          placeholder="Optional operator note (saved with approval event)"
                          value={approvalNote}
                          onChange={(e) => setApprovalNote(e.target.value)}
                        />
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                          <input
                            className="rounded border border-gray-300 px-2 py-1 text-xs"
                            placeholder="Tool name"
                            value={approvalEditTool}
                            onChange={(e) => setApprovalEditTool(e.target.value)}
                          />
                          <input
                            className="rounded border border-gray-300 px-2 py-1 text-xs"
                            placeholder="Purpose"
                            value={approvalEditPurpose}
                            onChange={(e) => setApprovalEditPurpose(e.target.value)}
                          />
                        </div>
                        <textarea
                          className="w-full rounded border border-gray-300 p-2 text-xs font-mono"
                          rows={6}
                          value={approvalEditParams}
                          onChange={(e) => setApprovalEditParams(e.target.value)}
                          placeholder='{"query":"...","limit":5}'
                        />
                        <div className="flex flex-wrap justify-end gap-2">
                          <Button
                            size="sm"
                            variant="ghost"
                            onClick={() => submitCheckpointAction('reject')}
                            disabled={actionMutation.isLoading}
                          >
                            Reject
                          </Button>
                          <Button
                            size="sm"
                            variant="secondary"
                            onClick={() => submitCheckpointAction('skip')}
                            disabled={actionMutation.isLoading}
                          >
                            Skip Step
                          </Button>
                          <Button
                            size="sm"
                            variant="primary"
                            onClick={submitEditedCheckpointAction}
                            disabled={actionMutation.isLoading}
                          >
                            Apply Edit + Resume
                          </Button>
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Goal contract */}
          {goalContractSummary && (
            <div className="mb-4">
              <h3 className="text-sm font-medium text-gray-700 mb-2">Goal Contract</h3>
              <div className="bg-amber-50 border border-amber-100 rounded-lg p-3">
                <div className="flex flex-wrap gap-3 text-xs text-amber-700 mb-2">
                  <span>Enabled: {goalContractSummary?.enabled ? 'yes' : 'no'}</span>
                  <span>Satisfied: {goalContractSummary?.satisfied ? 'yes' : 'no'}</span>
                  {goalContractSummary?.strict_completion !== undefined && (
                    <span>Strict: {goalContractSummary.strict_completion ? 'yes' : 'no'}</span>
                  )}
                  {goalContractSummary?.satisfied_iteration ? (
                    <span>Satisfied at iteration: {Number(goalContractSummary.satisfied_iteration || 0)}</span>
                  ) : null}
                </div>
                {Array.isArray(goalContractSummary?.missing) && goalContractSummary.missing.length > 0 && (
                  <div>
                    <div className="text-xs font-medium text-amber-800 mb-1">Missing requirements</div>
                    <ul className="text-xs text-amber-700 space-y-1">
                      {goalContractSummary.missing.slice(0, 8).map((m: string, idx: number) => (
                        <li key={`${idx}-${m.slice(0, 24)}`}>- {m}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            </div>
          )}

          <JobExecutionGraphSection job={job} />
          {/* Swarm summary */}
          {swarmSummary && (
            <div className="mb-4">
              <h3 className="text-sm font-medium text-gray-700 mb-2 flex items-center gap-1">
                <Layers className="w-4 h-4" />
                Swarm Summary
              </h3>
              <div className="bg-slate-50 rounded-lg p-3 space-y-2">
                <div className="flex flex-wrap gap-3 text-xs text-slate-600">
                  <span>Roles: {Number(swarmSummary?.role_count || 0)}</span>
                  <span>Siblings: {Number(swarmSummary?.terminal_siblings || 0)}/{Number(swarmSummary?.expected_siblings || 0)}</span>
                  <span>Consensus: {Number(swarmSummary?.consensus_count || 0)}</span>
                  <span>Conflicts: {Number(swarmSummary?.conflict_count || 0)}</span>
                  {swarmSummary?.confidence?.overall !== undefined && (
                    <span>Confidence: {(Number(swarmSummary.confidence.overall) * 100).toFixed(0)}%</span>
                  )}
                </div>
                {Array.isArray(swarmSummary?.roles) && swarmSummary.roles.length > 0 && (
                  <div className="text-xs text-slate-700">
                    Roles: {swarmSummary.roles.slice(0, 8).join(', ')}
                  </div>
                )}
                {(swarmSummary?.winning_role || swarmSummary?.repair_chain_job_id) && (
                  <div className="flex flex-wrap gap-2 text-xs text-slate-700">
                    {swarmSummary?.winning_role ? (
                      <span className="px-2 py-1 rounded-full bg-orange-100 text-orange-800">
                        Winning role: {String(swarmSummary.winning_role)}
                      </span>
                    ) : null}
                    {swarmSummary?.repair_chain_job_id ? (
                      <span className="px-2 py-1 rounded-full bg-emerald-100 text-emerald-800">
                        Repair handoff: {String(swarmSummary.repair_chain_job_id).slice(0, 8)}
                      </span>
                    ) : null}
                  </div>
                )}
                {swarmSummary?.review_state ? (
                  <div className="flex flex-wrap gap-2 text-xs text-slate-700">
                    <span className="px-2 py-1 rounded-full bg-slate-200 text-slate-800">
                      Review state: {String(swarmSummary.review_state).replace(/_/g, ' ')}
                    </span>
                    {swarmSummary?.tie_breaker_job_id ? (
                      <span className="px-2 py-1 rounded-full bg-amber-100 text-amber-800">
                        Tie-breaker: {String(swarmSummary.tie_breaker_job_id).slice(0, 8)}
                      </span>
                    ) : null}
                  </div>
                ) : null}
                {swarmSummary?.promotion_reason ? (
                  <div className="text-xs text-slate-700">
                    Promotion: {String(swarmSummary.promotion_reason)}
                  </div>
                ) : null}
                {swarmReviewReason ? (
                  <div className="text-xs text-slate-700">
                    Review: {swarmReviewReason}
                  </div>
                ) : null}
                {(swarmSummary?.top_file_cluster || swarmSummary?.top_command_cluster) ? (
                  <div className="flex flex-wrap gap-2 text-xs text-slate-700">
                    {swarmSummary?.top_file_cluster ? (
                      <span className="px-2 py-1 rounded-full bg-sky-100 text-sky-800">
                        File cluster: {String((swarmSummary.top_file_cluster as any)?.cluster || 'unknown')} · {Number((swarmSummary.top_file_cluster as any)?.support_count || 0)}
                      </span>
                    ) : null}
                    {swarmSummary?.top_command_cluster ? (
                      <span className="px-2 py-1 rounded-full bg-violet-100 text-violet-800">
                        Command support: {Number((swarmSummary.top_command_cluster as any)?.support_count || 0)}
                      </span>
                    ) : null}
                  </div>
                ) : null}
                {Array.isArray(swarmSummary?.candidate_paths) && swarmSummary.candidate_paths.length > 0 && (
                  <div>
                    <div className="text-xs font-medium text-slate-700 mb-1">Candidate paths</div>
                    <ul className="text-xs text-slate-700 space-y-2">
                      {swarmSummary.candidate_paths.slice(0, 3).map((row: any, idx: number) => {
                        const suspectFiles = Array.isArray(row?.suspect_files) ? row.suspect_files : [];
                        return (
                          <li key={`${idx}-${String(row?.job_id || row?.role || 'candidate')}`} className="flex items-start justify-between gap-3">
                            <span>
                              - {String(row?.role || 'Candidate')} · {suspectFiles.slice(0, 3).join(', ') || 'No file hints'}
                            </span>
                            {canManualPromoteSwarmCandidate ? (
                              <Button
                                size="sm"
                                variant="ghost"
                                onClick={() =>
                                  actionMutation.mutate({
                                    jobId: job.id,
                                    action: 'promote_swarm_candidate',
                                    actionPayload: {
                                      candidate_job_id: String(row?.job_id || ''),
                                      candidate_index: idx,
                                    },
                                  })
                                }
                                disabled={actionMutation.isLoading}
                              >
                                Promote
                              </Button>
                            ) : null}
                          </li>
                        );
                      })}
                    </ul>
                  </div>
                )}
                {Array.isArray(swarmSummary?.recommended_commands) && swarmSummary.recommended_commands.length > 0 && (
                  <div>
                    <div className="text-xs font-medium text-slate-700 mb-1">Recommended commands</div>
                    <ul className="text-xs text-slate-700 space-y-1">
                      {swarmSummary.recommended_commands.slice(0, 3).map((command: string, idx: number) => (
                        <li key={`${idx}-${command.slice(0, 24)}`}>- {command}</li>
                      ))}
                    </ul>
                  </div>
                )}
                {Array.isArray(swarmSummary?.consensus_findings) && swarmSummary.consensus_findings.length > 0 && (
                  <div>
                    <div className="text-xs font-medium text-slate-700 mb-1">Top consensus findings</div>
                    <ul className="text-xs text-slate-700 space-y-1">
                      {swarmSummary.consensus_findings.slice(0, 4).map((finding: string, idx: number) => (
                        <li key={`${idx}-${finding.slice(0, 24)}`}>- {finding}</li>
                      ))}
                    </ul>
                  </div>
                )}
                {Array.isArray(swarmSummary?.conflicts) && swarmSummary.conflicts.length > 0 && (
                  <div>
                    <div className="text-xs font-medium text-slate-700 mb-1">Conflicts</div>
                    <ul className="text-xs text-slate-700 space-y-1">
                      {swarmSummary.conflicts.slice(0, 3).map((c: any, idx: number) => (
                        <li key={`${idx}-${String(c?.type || 'conflict')}`}>
                          - {String(c?.description || c?.type || 'Conflict').slice(0, 220)}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
                {Array.isArray(swarmSummary?.action_plan) && swarmSummary.action_plan.length > 0 && (
                  <div>
                    <div className="text-xs font-medium text-slate-700 mb-1">Action plan</div>
                    <ul className="text-xs text-slate-700 space-y-1">
                      {swarmSummary.action_plan.slice(0, 4).map((step: any, idx: number) => (
                        <li key={`${idx}-${String(step?.action || 'step')}`}>
                          - {String(step?.action || '').slice(0, 220)}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            </div>
          )}

          {swarmOutcomeCase && (
            <div className="mb-4">
              <h3 className="text-sm font-medium text-gray-700 mb-2 flex items-center gap-1">
                <BarChart3 className="w-4 h-4" />
                Outcome Funnel
              </h3>
              <div className="bg-white border border-gray-200 rounded-lg p-3 space-y-2">
                <div className="flex flex-wrap gap-2 text-xs">
                  <span className={`px-2 py-1 rounded-full ${swarmOutcomeBadgeClass(swarmOutcomeCase.terminal_outcome)}`}>
                    {humanizeSwarmOutcome(swarmOutcomeCase.terminal_outcome)}
                  </span>
                  <span className="px-2 py-1 rounded-full bg-violet-100 text-violet-800">
                    Promotion: {humanizeSwarmOutcome(swarmOutcomeCase.promotion_mode)}
                  </span>
                  {swarmOutcomeCase.verification_status ? (
                    <span className="px-2 py-1 rounded-full bg-cyan-100 text-cyan-800">
                      Verification: {humanizeSwarmOutcome(swarmOutcomeCase.verification_status)}
                    </span>
                  ) : null}
                  {swarmOutcomeCase.repair_status ? (
                    <span className="px-2 py-1 rounded-full bg-slate-100 text-slate-700">
                      Repair: {humanizeSwarmOutcome(swarmOutcomeCase.repair_status)}
                    </span>
                  ) : null}
                </div>
                <div className="flex flex-wrap gap-3 text-xs text-gray-600">
                  <span>Preset: {humanizeSwarmOutcome(swarmOutcomeCase.preset_key)}</span>
                  {swarmOutcomeCase.repair_job_id ? <span>Repair job: {String(swarmOutcomeCase.repair_job_id).slice(0, 8)}</span> : null}
                  {swarmOutcomeCase.backlog_item_id ? <span>Backlog: {String(swarmOutcomeCase.backlog_title || swarmOutcomeCase.backlog_item_id)}</span> : null}
                  {typeof swarmOutcomeCase.handoff_latency_minutes === 'number' ? <span>Handoff: {Number(swarmOutcomeCase.handoff_latency_minutes).toFixed(0)}m</span> : null}
                </div>
                {swarmOutcomeCase.terminal_reason ? (
                  <div className="text-xs text-gray-700">{String(swarmOutcomeCase.terminal_reason)}</div>
                ) : null}
                {swarmOutcomeCase.verification_reason ? (
                  <div className="text-xs text-gray-500">{String(swarmOutcomeCase.verification_reason)}</div>
                ) : null}
              </div>
            </div>
          )}

          {/* Customer research context */}
          {(customerProfile || customerContext || documentArtifact || readingListArtifact || arxivSourceArtifacts.length > 0 || domainResearch) && (
            <div className="mb-4">
              <h3 className="text-sm font-medium text-gray-700 mb-2 flex items-center gap-1">
                <Brain className="w-4 h-4" />
                {domainResearch ? 'Domain Research' : 'Customer Research'}
              </h3>
              <div className="bg-white border border-gray-200 rounded-lg p-3 space-y-3">
                {customerProfile?.name && (
                  <div className="text-sm text-gray-800">
                    <span className="text-gray-500">Profile:</span> {customerProfile.name}
                  </div>
                )}
                {Array.isArray(customerProfile?.keywords) && customerProfile.keywords.length > 0 && (
                  <div className="text-xs text-gray-600">
                    <span className="text-gray-500">Keywords:</span> {customerProfile.keywords.slice(0, 20).join(', ')}
                  </div>
                )}
                {customerContext && (
                  <div className="text-xs text-gray-600 whitespace-pre-wrap">
                    <span className="text-gray-500">Context:</span> {String(customerContext).slice(0, 1200)}
                  </div>
                )}

                {researchBundle && (
                  <details className="bg-gray-50 border border-gray-200 rounded-lg p-2">
                    <summary className="cursor-pointer text-xs font-medium text-gray-800">
                      Research bundle ({(researchBundle?.top_documents || []).length} docs • {(researchBundle?.top_papers || []).length} papers •{' '}
                      {(researchBundle?.key_insights || []).length} insights)
                    </summary>
                    <div className="mt-2 grid grid-cols-1 md:grid-cols-2 gap-3">
                      <div className="bg-white border border-gray-200 rounded p-2">
                        <div className="text-xs font-medium text-gray-800 mb-1">Top documents</div>
                        <div className="space-y-1">
                          {(researchBundle?.top_documents || []).slice(0, 6).map((d: any) => (
                            <div key={String(d?.id)} className="flex items-start justify-between gap-2">
                              <div className="text-xs text-gray-700 min-w-0">
                                <div className="truncate">{d?.title || d?.id}</div>
                                <div className="text-gray-500 font-mono truncate">{String(d?.id || '')}</div>
                              </div>
                              {d?.id && (
                                <div className="shrink-0 flex gap-1">
                                  <Button size="sm" variant="secondary" onClick={() => openDocument(String(d.id))}>
                                    Open
                                  </Button>
                                </div>
                              )}
                            </div>
                          ))}
                        </div>
                      </div>
                      <div className="bg-white border border-gray-200 rounded p-2">
                        <div className="text-xs font-medium text-gray-800 mb-1">Key insights</div>
                        <div className="space-y-1">
                          {(researchBundle?.key_insights || []).slice(0, 8).map((it: any, idx: number) => (
                            <div key={String(it?.id || idx)} className="text-xs text-gray-700">
                              <div className="truncate">{it?.title || '(untitled insight)'}</div>
                              {(it?.category || it?.confidence !== undefined) && (
                                <div className="text-gray-500">
                                  {it?.category ? String(it.category) : ''}
                                  {it?.confidence !== undefined ? ` • ${(Number(it.confidence) * 100).toFixed(0)}%` : ''}
                                </div>
                              )}
                            </div>
                          ))}
                        </div>
                      </div>
                      <div className="bg-white border border-gray-200 rounded p-2 md:col-span-2">
                        <div className="text-xs font-medium text-gray-800 mb-1">Top papers</div>
                        <div className="space-y-1">
                          {(researchBundle?.top_papers || []).slice(0, 6).map((p: any, idx: number) => (
                            <div key={String(p?.arxiv_id || idx)} className="flex items-start justify-between gap-2">
                              <div className="text-xs text-gray-700 min-w-0">
                                <div className="truncate">{p?.title || p?.arxiv_id}</div>
                                <div className="text-gray-500 font-mono truncate">{String(p?.arxiv_id || '')}</div>
                              </div>
                              {p?.arxiv_id && (
                                <div className="shrink-0">
                                  <Button size="sm" variant="ghost" onClick={() => copyText(String(p.arxiv_id), 'arXiv ID')}>
                                    Copy ID
                                  </Button>
                                </div>
                              )}
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                    {Array.isArray(researchBundle?.suggested_queries) && researchBundle.suggested_queries.length > 0 && (
                      <div className="mt-3 bg-white border border-gray-200 rounded p-2">
                        <div className="text-xs font-medium text-gray-800 mb-1">Suggested queries</div>
                        <div className="text-xs text-gray-600 whitespace-pre-wrap">
                          {researchBundle.suggested_queries.slice(0, 8).map((q: string) => `- ${q}`).join('\n')}
                        </div>
                        <div className="pt-2">
                          <Button
                            size="sm"
                            variant="secondary"
                            onClick={() => copyText((researchBundle.suggested_queries || []).join('\n'), 'Suggested queries')}
                          >
                            Copy queries
                          </Button>
                        </div>
                      </div>
                    )}
                  </details>
                )}

                {domainResearch && (
                  <details className="bg-gray-50 border border-gray-200 rounded-lg p-2" open>
                    <summary className="cursor-pointer text-xs font-medium text-gray-800">
                      Domain ideas ({(domainResearch?.proposed_ideas || []).length} ideas)
                    </summary>
                    <div className="mt-2 space-y-3">
                      {domainResearch?.domain_summary ? (
                        <div className="bg-white border border-gray-200 rounded p-2 text-xs text-gray-700 whitespace-pre-wrap">
                          {String(domainResearch.domain_summary)}
                        </div>
                      ) : null}
                      {Array.isArray(domainResearch?.proposed_ideas) && domainResearch.proposed_ideas.length > 0 && (
                        <div className="space-y-2">
                          {domainResearch.proposed_ideas.slice(0, 5).map((idea: any, idx: number) => (
                            <div key={String(idea?.id || idx)} className="bg-white border border-gray-200 rounded p-2 text-xs text-gray-700 space-y-1">
                              <div className="font-medium text-gray-800">{idea?.title || `Idea ${idx + 1}`}</div>
                              {idea?.hypothesis ? <div>{String(idea.hypothesis)}</div> : null}
                              {idea?.opportunity ? <div className="text-gray-600">{String(idea.opportunity)}</div> : null}
                              {idea?.confidence !== undefined ? (
                                <div className="text-gray-500">Confidence {(Number(idea.confidence) * 100).toFixed(0)}%</div>
                              ) : null}
                            </div>
                          ))}
                        </div>
                      )}
                      {(domainResearch?.delta_since_last_run || domainResearch?.novelty_summary) && (
                        <div className="bg-white border border-gray-200 rounded p-2 text-xs text-gray-700">
                          <div className="font-medium text-gray-800 mb-1">Run-to-run delta</div>
                          <div>
                            New ideas {Number((domainResearch?.novelty_summary || {}).new_idea_count || 0)}
                            {' '}· Repeated {Number((domainResearch?.novelty_summary || {}).repeated_idea_count || 0)}
                            {' '}· New evidence {Number((domainResearch?.novelty_summary || {}).new_evidence_count || 0)}
                          </div>
                          {Array.isArray(domainResearch?.delta_since_last_run?.new_idea_titles) && domainResearch.delta_since_last_run.new_idea_titles.length > 0 ? (
                            <div className="mt-1 text-gray-600">
                              {domainResearch.delta_since_last_run.new_idea_titles.slice(0, 3).join(', ')}
                            </div>
                          ) : null}
                        </div>
                      )}
                      {Array.isArray(domainResearch?.research_note_ids) && domainResearch.research_note_ids.length > 0 && (
                        <div className="bg-white border border-gray-200 rounded p-2">
                          <div className="text-xs font-medium text-gray-800 mb-1">Research Notes</div>
                          <div className="space-y-1">
                            {domainResearch.research_note_ids.slice(0, 6).map((noteId: string) => (
                              <div key={noteId} className="flex items-center justify-between gap-2">
                                <div className="text-xs text-gray-600 font-mono truncate">{String(noteId)}</div>
                                <div className="flex gap-1 shrink-0">
                                  <Button size="sm" variant="secondary" onClick={() => navigate(`/research-notes?note=${encodeURIComponent(String(noteId))}`)}>
                                    Open
                                  </Button>
                                  <Button size="sm" variant="ghost" onClick={() => copyText(String(noteId), 'Research Note ID')}>
                                    Copy ID
                                  </Button>
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                      {Array.isArray(domainResearch?.experiment_plan_ids) && domainResearch.experiment_plan_ids.length > 0 && (
                        <div className="bg-white border border-gray-200 rounded p-2">
                          <div className="text-xs font-medium text-gray-800 mb-1">Experiment Plans</div>
                          <div className="space-y-1">
                            {domainResearch.experiment_plan_ids.slice(0, 6).map((planId: string) => (
                              <div key={planId} className="flex items-center justify-between gap-2">
                                <div className="text-xs text-gray-600 font-mono truncate">{String(planId)}</div>
                                <Button size="sm" variant="ghost" onClick={() => copyText(String(planId), 'Experiment Plan ID')}>
                                  Copy ID
                                </Button>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                      {Array.isArray(domainResearch?.follow_up_launches) && domainResearch.follow_up_launches.length > 0 && (
                        <div className="bg-white border border-gray-200 rounded p-2">
                          <div className="text-xs font-medium text-gray-800 mb-1">Follow-up launches</div>
                          <div className="space-y-1">
                            {domainResearch.follow_up_launches.slice(0, 4).map((item: any, idx: number) => (
                              <div key={String(item?.job_id || idx)} className="flex items-center justify-between gap-2">
                                <div className="text-xs text-gray-700">
                                  {String(item?.name || item?.job_id || 'Deep-dive follow-up')}
                                </div>
                                {item?.job_id ? (
                                  <Button size="sm" variant="secondary" onClick={() => navigate(buildAutonomousAgentsUrl(String(item.job_id)), { replace: true })}>
                                    Open Job
                                  </Button>
                                ) : null}
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  </details>
                )}

                {(documentArtifact?.id || documentArtifact?.document_id) && (
                  <div className="flex items-center justify-between gap-3 bg-gray-50 border border-gray-200 rounded-lg p-2">
                    <div className="text-xs text-gray-700">
                      <div className="font-medium text-gray-800">Brief document</div>
                      <div className="text-gray-600 font-mono">
                        {String(documentArtifact.id || documentArtifact.document_id)}
                      </div>
                    </div>
                    <div className="flex gap-2">
                      <Button
                        size="sm"
                        variant="secondary"
                        onClick={() => openDocument(String(documentArtifact.id || documentArtifact.document_id))}
                      >
                        Open
                      </Button>
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={() => copyText(String(documentArtifact.id || documentArtifact.document_id), 'Document ID')}
                      >
                        Copy ID
                      </Button>
                    </div>
                  </div>
                )}

                {codePatchProposal?.proposal_id && (
                  <div className="flex items-center justify-between gap-3 bg-gray-50 border border-gray-200 rounded-lg p-2">
                    <div className="text-xs text-gray-700 min-w-0">
                      <div className="font-medium text-gray-800">Code patch</div>
                      <div className="text-gray-600 truncate">{codePatchProposal.title}</div>
                      <div className="text-gray-600 font-mono truncate">{codePatchProposal.proposal_id}</div>
                    </div>
                    <div className="flex gap-2 shrink-0">
                      <Button
                        size="sm"
                        variant="secondary"
                        onClick={() =>
                          apiClient.downloadCodePatchProposal(codePatchProposal.proposal_id, codePatchProposal.title)
                        }
                      >
                        Download
                      </Button>
                      <Button
                        size="sm"
                        variant="secondary"
                        onClick={async () => {
                          const ok = window.confirm(
                            'Apply this patch to KnowledgeDB code documents now? This updates the stored file contents.'
                          );
                          if (!ok) return;
                          try {
                            const res = await apiClient.applyCodePatchProposal(codePatchProposal.proposal_id);
                            if ((res.errors || []).length > 0) {
                              toast.error(`Applied with errors: ${(res.errors || []).length}`);
                            } else {
                              toast.success('Patch applied to KB');
                            }
                            queryClient.invalidateQueries(['agent-jobs']);
                          } catch (e: any) {
                            toast.error(e?.response?.data?.detail || e?.message || 'Failed to apply patch');
                          }
                        }}
                      >
                        Apply to KB
                      </Button>
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={() => copyText(codePatchProposal.proposal_id, 'Proposal ID')}
                      >
                        Copy ID
                      </Button>
                    </div>
                  </div>
                )}

                {codePatchProposals.length > 1 ? (
                  <details className="bg-gray-50 border border-gray-200 rounded-lg p-2">
                    <summary className="cursor-pointer text-xs font-medium text-gray-800">Code patch history</summary>
                    <div className="mt-2 space-y-2">
                      {codePatchProposals.map((p) => (
                        <div key={p.proposal_id} className="flex items-center justify-between gap-3 bg-white border border-gray-200 rounded-lg p-2">
                          <div className="text-xs text-gray-700 min-w-0">
                            <div className="text-gray-600 truncate">{p.title}</div>
                            <div className="text-gray-600 font-mono truncate">{p.proposal_id}</div>
                          </div>
                          <div className="flex gap-2 shrink-0">
                            <Button size="sm" variant="secondary" onClick={() => apiClient.downloadCodePatchProposal(p.proposal_id, p.title)}>
                              Download
                            </Button>
                            <Button
                              size="sm"
                              variant="secondary"
                              onClick={async () => {
                                const ok = window.confirm(
                                  'Apply this patch to KnowledgeDB code documents now? This updates the stored file contents.'
                                );
                                if (!ok) return;
                                try {
                                  const res = await apiClient.applyCodePatchProposal(p.proposal_id);
                                  if ((res.errors || []).length > 0) toast.error(`Applied with errors: ${(res.errors || []).length}`);
                                  else toast.success('Patch applied to KB');
                                  queryClient.invalidateQueries(['agent-jobs']);
                                } catch (e: any) {
                                  toast.error(e?.response?.data?.detail || e?.message || 'Failed to apply patch');
                                }
                              }}
                            >
                              Apply to KB
                            </Button>
                            <Button size="sm" variant="ghost" onClick={() => copyText(p.proposal_id, 'Proposal ID')}>
                              Copy ID
                            </Button>
                          </div>
                        </div>
                      ))}
                    </div>
                  </details>
                ) : null}

                {codePatchExecution ? (
                  <details className="bg-gray-50 border border-gray-200 rounded-lg p-2" open={launchMode === 'quick_start_repo_bug_triage'}>
                    <summary className="cursor-pointer text-xs font-medium text-gray-800">Coding execution</summary>
                    <div className="mt-2 space-y-3 text-xs text-gray-700">
                      <div className="flex flex-wrap gap-1.5">
                        {codePatchExecution.mode ? (
                          <span className="px-2 py-0.5 rounded-full bg-slate-50 text-slate-700 border border-slate-100">
                            {String(codePatchExecution.mode)}
                          </span>
                        ) : null}
                        {codePatchExecution.scope ? (
                          <span className="px-2 py-0.5 rounded-full bg-amber-50 text-amber-700 border border-amber-100">
                            Scope {String(codePatchExecution.scope)}
                          </span>
                        ) : null}
                        {codePatchWorkspace?.created ? (
                          <span className="px-2 py-0.5 rounded-full bg-sky-50 text-sky-700 border border-sky-100">
                            Workspace {Number(codePatchWorkspace.file_count || 0)} files
                          </span>
                        ) : null}
                        {codePatchVerificationPlan?.auto_inferred ? (
                          <span className="px-2 py-0.5 rounded-full bg-violet-50 text-violet-700 border border-violet-100">
                            Verification auto-inferred
                          </span>
                        ) : null}
                        {codePatchExecution.proposal_strategy ? (
                          <span className="px-2 py-0.5 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-100">
                            Strategy {String(codePatchExecution.proposal_strategy)}
                          </span>
                        ) : null}
                        {codingRecoveryState ? (
                          <span className="px-2 py-0.5 rounded-full bg-rose-50 text-rose-700 border border-rose-100">
                            Recovery {codingRecoveryState.replace(/_/g, ' ')}
                          </span>
                        ) : null}
                      </div>

                      {codePatchRecovery ? (
                        <div className="bg-white border border-gray-200 rounded p-2 space-y-2">
                          <div className="font-medium text-gray-800">Recovery</div>
                          {codePatchRecovery.retry_reason ? (
                            <div className="text-gray-600">{String(codePatchRecovery.retry_reason)}</div>
                          ) : null}
                          {codePatchRecovery.resume_hint ? (
                            <div className="text-gray-600">Resume hint: {String(codePatchRecovery.resume_hint)}</div>
                          ) : null}
                          {codePatchFailedCommands.length > 0 ? (
                            <div>
                              <div className="text-gray-700 font-medium mb-1">Failed commands</div>
                              <div className="space-y-1">
                                {codePatchFailedCommands.map((cmd, idx) => (
                                  <div key={`recovery-failed-${idx}`} className="font-mono text-gray-600 break-all">{cmd}</div>
                                ))}
                              </div>
                            </div>
                          ) : null}
                          {codePatchRecovery.latest_failed_output ? (
                            <div>
                              <div className="text-gray-700 font-medium mb-1">Latest failed output</div>
                              <div className="text-gray-600 font-mono whitespace-pre-wrap break-words">
                                {String(codePatchRecovery.latest_failed_output)}
                              </div>
                            </div>
                          ) : null}
                          {codePatchSuggestedActions.length > 0 ? (
                            <div className="text-gray-500">
                              Suggested actions: {codePatchSuggestedActions.map((item) => String(item).replace(/_/g, ' ')).join(', ')}
                            </div>
                          ) : null}
                        </div>
                      ) : null}

                      {codePatchExecution.failure_symptom ? (
                        <div>
                          <div className="font-medium text-gray-800 mb-1">Failure symptom</div>
                          <div className="text-gray-600 whitespace-pre-wrap">{String(codePatchExecution.failure_symptom)}</div>
                        </div>
                      ) : null}

                      {codePatchExecution.error_output ? (
                        <div>
                          <div className="font-medium text-gray-800 mb-1">Error output</div>
                          <div className="text-gray-600 font-mono whitespace-pre-wrap break-words">
                            {String(codePatchExecution.error_output)}
                          </div>
                        </div>
                      ) : null}

                      {codePatchWorkspace ? (
                        <div className="bg-white border border-gray-200 rounded p-2 space-y-1">
                          <div className="font-medium text-gray-800">Workspace</div>
                          <div className="text-gray-600">
                            {codePatchWorkspace.created ? 'ready' : 'not created'}
                            {codePatchWorkspace.source_type ? ` • source ${String(codePatchWorkspace.source_type)}` : ''}
                            {codePatchWorkspace.file_count !== undefined ? ` • files ${Number(codePatchWorkspace.file_count || 0)}` : ''}
                          </div>
                          {codePatchWorkspace.workspace_id ? (
                            <div className="text-gray-500 font-mono break-all">{String(codePatchWorkspace.workspace_id)}</div>
                          ) : null}
                          {codePatchWorkspace.error ? (
                            <div className="text-rose-700">{String(codePatchWorkspace.error)}</div>
                          ) : null}
                        </div>
                      ) : null}

                      {codePatchDetectedStack.length > 0 ? (
                        <div className="bg-white border border-gray-200 rounded p-2">
                          <div className="font-medium text-gray-800 mb-1">Inferred project profile</div>
                          <div className="text-gray-600">Detected stack: {codePatchDetectedStack.join(', ')}</div>
                        </div>
                      ) : null}

                      {codePatchVerificationCommands.length > 0 ? (
                        <div className="bg-white border border-gray-200 rounded p-2 space-y-2">
                          <div className="font-medium text-gray-800">Verification plan</div>
                          <div>
                            <div className="text-gray-700 font-medium mb-1">Primary commands</div>
                            <div className="text-gray-600 font-mono whitespace-pre-wrap">{codePatchVerificationCommands.join('\n')}</div>
                          </div>
                          {codePatchBootstrapCommands.length > 0 ? (
                            <div>
                              <div className="text-gray-700 font-medium mb-1">Bootstrap</div>
                              <div className="text-gray-600 font-mono whitespace-pre-wrap">{codePatchBootstrapCommands.join('\n')}</div>
                            </div>
                          ) : null}
                          {codePatchFallbackCommands.length > 0 ? (
                            <div>
                              <div className="text-gray-700 font-medium mb-1">Fallback</div>
                              <div className="text-gray-600 font-mono whitespace-pre-wrap">{codePatchFallbackCommands.join('\n')}</div>
                            </div>
                          ) : null}
                        </div>
                      ) : null}

                      {codePatchExecutionPlan.length > 0 ? (
                        <div className="bg-white border border-gray-200 rounded p-2 space-y-2">
                          <div className="font-medium text-gray-800">Execution plan</div>
                          <div className="space-y-2">
                            {codePatchExecutionPlan.map((step, idx) => (
                              <div key={String(step?.step_id || idx)} className="border border-gray-100 rounded p-2">
                                <div className="flex items-center justify-between gap-2">
                                  <div className="font-medium text-gray-800">{String(step?.title || `Step ${idx + 1}`)}</div>
                                  {step?.status ? (
                                    <span className="px-2 py-0.5 rounded-full bg-slate-50 text-slate-700 border border-slate-100">
                                      {String(step.status)}
                                    </span>
                                  ) : null}
                                </div>
                                {step?.objective ? (
                                  <div className="mt-1 text-gray-600">{String(step.objective)}</div>
                                ) : null}
                                {Array.isArray(step?.commands) && step.commands.length > 0 ? (
                                  <div className="mt-1 text-gray-600 font-mono whitespace-pre-wrap">
                                    {step.commands.join('\n')}
                                  </div>
                                ) : null}
                              </div>
                            ))}
                          </div>
                        </div>
                      ) : null}
                    </div>
                  </details>
                ) : null}

                {experimentRuns.length > 0 ? (
                  <details className="bg-gray-50 border border-gray-200 rounded-lg p-2">
                    <summary className="cursor-pointer text-xs font-medium text-gray-800">Experiment runs</summary>
                    <div className="mt-2 space-y-2">
                      {experimentRuns.map((er, idx) => {
                        const okVal = er?.ok;
                        const label = okVal === true ? 'PASS' : okVal === false ? 'FAIL' : 'SKIP';
                        const labelClass = okVal === true ? 'text-green-700' : okVal === false ? 'text-red-700' : 'text-amber-700';
                        const cmds = Array.isArray(er?.commands) ? er.commands : [];
                        const isLatestExperimentRun = idx === latestExperimentRunIndex;
                        const {
                          verificationCommands: verificationCmds,
                          bootstrapCommands: bootstrapCmds,
                          fallbackCommands: fallbackCmds,
                          phases,
                          failedCommands,
                          finalPhase,
                          sourceId,
                          sourceName,
                          detectedStack,
                        } = summarizeExperimentRun(er);
                        const recoveryOpen = isExperimentRecoveryOpen(er, {
                          verificationCommands: verificationCmds,
                          bootstrapCommands: bootstrapCmds,
                          fallbackCommands: fallbackCmds,
                          phases,
                          failedCommands,
                          finalPhase,
                          sourceId,
                          sourceName,
                          detectedStack,
                        });
                        const pid = String(er?.proposal_id || '').trim();
                        return (
                          <div key={idx} className="bg-white border border-gray-200 rounded-lg p-2">
                            <div className="flex items-center justify-between gap-2 text-xs">
                              <div className="text-gray-700 min-w-0">
                                <span className={`font-medium ${labelClass}`}>{label}</span>
                                {er?.source_name ? <span className="text-gray-500"> — {String(er.source_name)}</span> : null}
                                {pid ? <span className="text-gray-500"> • </span> : null}
                                {pid ? <span className="text-gray-500 font-mono truncate">{pid}</span> : null}
                              </div>
                              {cmds.length > 0 ? <div className="text-gray-500">{cmds.length} cmd(s)</div> : null}
                            </div>
                            <div className="mt-2 flex flex-wrap gap-1 text-[11px]">
                              {finalPhase ? (
                                <span className="px-2 py-0.5 rounded-full bg-slate-100 text-slate-700 border border-slate-200">
                                  Final {finalPhase}
                                </span>
                              ) : null}
                              {Boolean(er?.bootstrap_attempted) ? (
                                <span className={`px-2 py-0.5 rounded-full border ${er?.bootstrap_ok ? 'bg-blue-50 text-blue-700 border-blue-200' : 'bg-amber-50 text-amber-700 border-amber-200'}`}>
                                  Bootstrap {er?.bootstrap_ok ? 'ok' : 'attempted'}
                                </span>
                              ) : null}
                              {Boolean(er?.fallback_attempted) ? (
                                <span className={`px-2 py-0.5 rounded-full border ${er?.fallback_ok ? 'bg-indigo-50 text-indigo-700 border-indigo-200' : 'bg-amber-50 text-amber-700 border-amber-200'}`}>
                                  Fallback {er?.fallback_ok ? 'ok' : 'attempted'}
                                </span>
                              ) : null}
                              {recoveryOpen ? (
                                <span className="px-2 py-0.5 rounded-full bg-rose-100 text-rose-800 border border-rose-200">
                                  Recovery open
                                </span>
                              ) : null}
                              {recoveryOpen && graphRecommendedActions.length > 0 ? (
                                <span className="px-2 py-0.5 rounded-full bg-amber-50 text-amber-700 border border-amber-200">
                                  Next {graphRecommendedActions[0]}
                                </span>
                              ) : null}
                              {phases.length > 0 ? (
                                <span className="px-2 py-0.5 rounded-full bg-gray-100 text-gray-700 border border-gray-200">
                                  Phases {phases.join(' -> ')}
                                </span>
                              ) : null}
                              {sourceName ? (
                                <span className="px-2 py-0.5 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200">
                                  Source {sourceName}
                                </span>
                              ) : null}
                              {sourceId ? (
                                <span className="px-2 py-0.5 rounded-full bg-white text-gray-700 border border-gray-200 font-mono">
                                  {sourceId}
                                </span>
                              ) : null}
                              {detectedStack.length > 0 ? (
                                <span className="px-2 py-0.5 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200">
                                  Stack {detectedStack.join(', ')}
                                </span>
                              ) : null}
                              {isLatestExperimentRun && operatorInterventionSummary.latestLabel ? (
                                <span className="px-2 py-0.5 rounded-full bg-amber-50 text-amber-800 border border-amber-200">
                                  Last {operatorInterventionSummary.latestLabel}
                                </span>
                              ) : null}
                              {isLatestExperimentRun && operatorInterventionSummary.latestOutcome ? (
                                <span className="px-2 py-0.5 rounded-full bg-orange-50 text-orange-700 border border-orange-100">
                                  Outcome {operatorInterventionSummary.latestOutcome}
                                </span>
                              ) : null}
                            </div>
                            {isLatestExperimentRun && operatorInterventionSummary.recentItems.length > 1 ? (
                              <div className="mt-2 text-[11px] text-amber-800">
                                <div className="font-medium mb-1">Recent intervention timeline</div>
                                <ul className="space-y-1">
                                  {operatorInterventionSummary.recentItems.map((item, itemIdx) => (
                                    <li key={`${idx}-timeline-${itemIdx}`}>- {item}</li>
                                  ))}
                                </ul>
                              </div>
                            ) : null}
                            {isLatestExperimentRun && operatorInterventionSummary.latestOutcomeReason ? (
                              <div className="mt-2 text-[11px] text-orange-700">
                                <span className="font-medium">Outcome reason:</span> {operatorInterventionSummary.latestOutcomeReason}
                              </div>
                            ) : null}
                            {recoveryOpen && graphHealthReasons.length > 0 ? (
                              <div className="mt-2 text-[11px] text-rose-700">
                                <span className="font-medium">Reason:</span> {graphHealthReasons[0]}
                              </div>
                            ) : null}
                            {recoveryOpen && isLatestExperimentRun ? (
                              <div className="mt-3 flex flex-wrap items-center gap-2">
                                <Button size="sm" variant="primary" onClick={openCheckpointQueue}>
                                  Open in Checkpoint Queue
                                </Button>
                                {failedCommands.length > 0 ? (
                                  <Button
                                    size="sm"
                                    variant="ghost"
                                    onClick={() => copyText(String(failedCommands[0]), 'Failed command')}
                                  >
                                    Copy failed command
                                  </Button>
                                ) : null}
                                {graphRecommendedActions.length > 0 ? (
                                  <Button
                                    size="sm"
                                    variant="ghost"
                                    onClick={() => copyText(String(graphRecommendedActions[0]), 'Recovery next step')}
                                  >
                                    Copy next step
                                  </Button>
                                ) : null}
                              </div>
                            ) : null}
                            {verificationCmds.length > 0 ? (
                              <div className="mt-2 space-y-1">
                                <div className="text-[11px] font-medium text-gray-700">Verification</div>
                                <div className="text-[11px] text-gray-600 font-mono whitespace-pre-wrap">
                                  {verificationCmds.slice(0, 6).join('\n')}
                                </div>
                              </div>
                            ) : null}
                            {bootstrapCmds.length > 0 ? (
                              <div className="mt-2 space-y-1">
                                <div className="text-[11px] font-medium text-gray-700">Bootstrap</div>
                                <div className="text-[11px] text-gray-600 font-mono whitespace-pre-wrap">
                                  {bootstrapCmds.slice(0, 4).join('\n')}
                                </div>
                              </div>
                            ) : null}
                            {fallbackCmds.length > 0 ? (
                              <div className="mt-2 space-y-1">
                                <div className="text-[11px] font-medium text-gray-700">Fallback verification</div>
                                <div className="text-[11px] text-gray-600 font-mono whitespace-pre-wrap">
                                  {fallbackCmds.slice(0, 4).join('\n')}
                                </div>
                              </div>
                            ) : null}
                            {failedCommands.length > 0 ? (
                              <div className="mt-2 space-y-1">
                                <div className="text-[11px] font-medium text-rose-700">Failed commands</div>
                                <div className="text-[11px] text-rose-700 font-mono whitespace-pre-wrap">
                                  {failedCommands.slice(0, 4).join('\n')}
                                </div>
                              </div>
                            ) : null}
                          </div>
                        );
                      })}
                    </div>
                  </details>
                ) : null}

                {codePatchApply ? (
                  <div className="flex items-center justify-between gap-3 bg-gray-50 border border-gray-200 rounded-lg p-2">
                    <div className="text-xs text-gray-700 min-w-0">
                      <div className="font-medium text-gray-800">Patch apply (sandbox)</div>
                      <div className="text-gray-600">
                        applied: {Array.isArray(codePatchApply.applied) ? codePatchApply.applied.length : 0} • errors:{' '}
                        {Array.isArray(codePatchApply.errors) ? codePatchApply.errors.length : 0}
                      </div>
                      {codePatchApply.proposal_id ? (
                        <div className="text-gray-600 font-mono truncate">{String(codePatchApply.proposal_id)}</div>
                      ) : null}
                    </div>
                  </div>
                ) : null}

                {codePatchKbApply ? (
                  <details className="bg-gray-50 border border-gray-200 rounded-lg p-2">
                    <summary className="cursor-pointer text-xs font-medium text-gray-800">Patch apply (Knowledge DB)</summary>
                    <div className="mt-2 space-y-2 text-xs text-gray-700">
                      <div className="text-gray-600">
                        {codePatchKbApply.enabled === false
                          ? 'skipped'
                          : codePatchKbApply.dry_run
                            ? `dry-run — ok: ${String(codePatchKbApply.ok)}`
                            : `applied: ${String(codePatchKbApply.did_apply)} — ok: ${String(codePatchKbApply.ok)}`}
                        {' • '}
                        errors: {Array.isArray(codePatchKbApply.errors) ? codePatchKbApply.errors.length : 0}
                        {' • '}
                        files: {Array.isArray(codePatchKbApply.applied_files) ? codePatchKbApply.applied_files.length : 0}
                      </div>
                      {codePatchKbApply.blocked_reason ? (
                        <div className="text-yellow-800 bg-yellow-50 border border-yellow-200 rounded px-2 py-1">
                          Blocked: {String(codePatchKbApply.blocked_reason)}
                        </div>
                      ) : null}
                      {codePatchKbApply.proposal_strategy ? (
                        <div className="text-gray-500">strategy: {String(codePatchKbApply.proposal_strategy)}</div>
                      ) : null}
                      {codePatchKbApply.proposal_id ? (
                        <div className="text-gray-600 font-mono truncate">{String(codePatchKbApply.proposal_id)}</div>
                      ) : null}

                      {Array.isArray(codePatchKbApply.applied_files) && codePatchKbApply.applied_files.length > 0 ? (
                        <div className="space-y-1">
                          <div className="font-medium text-gray-800">Applied files</div>
                          <div className="space-y-1">
                            {codePatchKbApply.applied_files.slice(0, 50).map((f: any, i: number) => (
                              <div key={String(f?.document_id || f?.path || i)} className="flex items-center justify-between gap-2 bg-white border border-gray-200 rounded px-2 py-1">
                                <div className="min-w-0">
                                  <div className="text-gray-600 font-mono truncate">{String(f?.path || '(unknown path)')}</div>
                                  {f?.document_id ? (
                                    <div className="text-gray-500 font-mono truncate">{String(f.document_id)}</div>
                                  ) : null}
                                </div>
                                <div className="flex gap-2 shrink-0">
                                  {f?.document_id ? (
                                    <Button size="sm" variant="secondary" onClick={() => openDocument(String(f.document_id))}>
                                      Open
                                    </Button>
                                  ) : null}
                                  {f?.document_id ? (
                                    <Button size="sm" variant="ghost" onClick={() => copyText(String(f.document_id), 'Document ID')}>
                                      Copy ID
                                    </Button>
                                  ) : null}
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      ) : null}

                      {Array.isArray(codePatchKbApply.errors) && codePatchKbApply.errors.length > 0 ? (
                        <div className="space-y-1">
                          <div className="font-medium text-gray-800">Errors</div>
                          <div className="space-y-1">
                            {codePatchKbApply.errors.slice(0, 50).map((e: any, i: number) => (
                              <div key={String(e?.path || i)} className="bg-white border border-red-200 rounded px-2 py-1">
                                <div className="text-red-800 font-mono">{String(e?.path || '(unknown file)')}</div>
                                <div className="text-red-700">{String(e?.error || e?.message || '')}</div>
                              </div>
                            ))}
                          </div>
                        </div>
                      ) : null}
                    </div>
                  </details>
                ) : null}

                {generatedProject?.source_id && (
                  <div className="flex items-center justify-between gap-3 bg-gray-50 border border-gray-200 rounded-lg p-2">
                    <div className="text-xs text-gray-700 min-w-0">
                      <div className="font-medium text-gray-800">Generated project</div>
                      <div className="text-gray-600 truncate">{generatedProject.project_name}</div>
                      <div className="text-gray-600 font-mono truncate">{generatedProject.source_id}</div>
                      {generatedProject.file_count ? (
                        <div className="text-gray-500">{generatedProject.file_count} files</div>
                      ) : null}
                      {generatedProject.sanity_errors_count ? (
                        <div className="text-red-600">Syntax errors: {generatedProject.sanity_errors_count}</div>
                      ) : generatedProject.sanity_ok ? (
                        <div className="text-green-700">Sanity check: OK</div>
                      ) : null}
                      {generatedProject.sanity_ok && generatedProject.behavioral?.enabled === false ? (
                        <div className="text-amber-700">Behavioral check: skipped (server disabled)</div>
                      ) : generatedProject.behavioral?.ran ? (
                        generatedProject.behavioral?.ok ? (
                          <div className="text-green-700">Behavioral check: OK</div>
                        ) : (
                          <div className="text-red-600">Behavioral check: failed</div>
                        )
                      ) : null}
                      {generatedProject.behavioral?.ran ? (
                        <details className="mt-2">
                          <summary className="cursor-pointer text-gray-700">Behavior details</summary>
                          <div className="mt-2 space-y-2">
                            <div className="text-gray-700">
                              Backend: <span className="font-mono">{String(generatedProject.behavioral.backend || '')}</span>
                              {typeof generatedProject.behavioral.duration_ms === 'number' ? (
                                <span className="ml-2">({generatedProject.behavioral.duration_ms}ms)</span>
                              ) : null}
                              {generatedProject.behavioral.timed_out ? <span className="ml-2 text-red-600">timeout</span> : null}
                            </div>
                            {generatedProject.behavioral.error ? (
                              <div className="text-red-700">Error: {String(generatedProject.behavioral.error)}</div>
                            ) : null}
                            {typeof generatedProject.behavioral.exit_code === 'number' ? (
                              <div className="text-gray-700">
                                Exit code: <span className="font-mono">{String(generatedProject.behavioral.exit_code)}</span>
                              </div>
                            ) : null}
                            {typeof generatedProject.behavioral.stdout === 'string' && generatedProject.behavioral.stdout.trim() ? (
                              <div>
                                <div className="flex items-center justify-between">
                                  <div className="text-gray-700">stdout</div>
                                  <Button
                                    size="sm"
                                    variant="ghost"
                                    onClick={() => copyText(String(generatedProject.behavioral.stdout || ''), 'stdout')}
                                  >
                                    Copy
                                  </Button>
                                </div>
                                <pre className="mt-1 p-2 bg-white border border-gray-200 rounded whitespace-pre-wrap max-h-48 overflow-auto">
                                  {String(generatedProject.behavioral.stdout)}
                                </pre>
                              </div>
                            ) : null}
                            {typeof generatedProject.behavioral.stderr === 'string' && generatedProject.behavioral.stderr.trim() ? (
                              <div>
                                <div className="flex items-center justify-between">
                                  <div className="text-gray-700">stderr</div>
                                  <Button
                                    size="sm"
                                    variant="ghost"
                                    onClick={() => copyText(String(generatedProject.behavioral.stderr || ''), 'stderr')}
                                  >
                                    Copy
                                  </Button>
                                </div>
                                <pre className="mt-1 p-2 bg-white border border-gray-200 rounded whitespace-pre-wrap max-h-48 overflow-auto">
                                  {String(generatedProject.behavioral.stderr)}
                                </pre>
                              </div>
                            ) : null}
                          </div>
                        </details>
                      ) : null}
                    </div>
                    <div className="flex gap-2 shrink-0">
                      <Button
                        size="sm"
                        variant="secondary"
                        onClick={() =>
                          apiClient.downloadDocumentSourceZip(generatedProject.source_id, generatedProject.project_name)
                        }
                      >
                        Download ZIP
                      </Button>
                      <Button
                        size="sm"
                        variant="secondary"
                        onClick={() => navigate('/documents', { state: { selectedSourceId: generatedProject.source_id } })}
                      >
                        Open
                      </Button>
                      <Button
                        size="sm"
                        variant="secondary"
                        disabled={unsafeExecBadge.status !== 'ready' || createMutation.isLoading}
                        title={
                          unsafeExecBadge.status === 'ready'
                            ? 'Run sandboxed demo check again'
                            : 'Demo check not available (see badge on Implement Algorithm)'
                        }
                        onClick={() =>
                          createMutation.mutate({
                            name: `Demo check — ${generatedProject.project_name}`.slice(0, 120),
                            job_type: 'monitor' as any,
                            goal: `Run demo check (${generatedProject.project_name})`,
                            config: {
                              deterministic_runner: 'generated_project_demo_check',
                              source_id: generatedProject.source_id,
                              entrypoint: generatedProject.entrypoint || 'demo.py',
                            },
                            max_iterations: 1,
                            max_tool_calls: 0,
                            max_llm_calls: 0,
                            max_runtime_minutes: 5,
                            start_immediately: true,
                          })
                        }
                      >
                        Re-run demo
                      </Button>
                      <Button size="sm" variant="ghost" onClick={() => copyText(generatedProject.source_id, 'Source ID')}>
                        Copy ID
                      </Button>
                    </div>
                  </div>
                )}

                {demoCheck?.source_id && (
                  <div className="flex items-start justify-between gap-3 bg-gray-50 border border-gray-200 rounded-lg p-2">
                    <div className="text-xs text-gray-700 min-w-0">
                      <div className="font-medium text-gray-800">Demo check</div>
                      <div className="text-gray-600 font-mono truncate">{demoCheck.source_id}</div>
                      <div className={demoCheck.ok ? 'text-green-700' : 'text-red-600'}>
                        {demoCheck.ok ? 'OK' : 'FAILED'} • {demoCheck.entrypoint}
                      </div>
                      {demoCheck.behavioral?.ran ? (
                        <details className="mt-2">
                          <summary className="cursor-pointer text-gray-700">Details</summary>
                          <div className="mt-2 space-y-2">
                            <div className="text-gray-700">
                              Backend: <span className="font-mono">{String(demoCheck.behavioral.backend || '')}</span>
                              {typeof demoCheck.behavioral.duration_ms === 'number' ? (
                                <span className="ml-2">({demoCheck.behavioral.duration_ms}ms)</span>
                              ) : null}
                              {demoCheck.behavioral.timed_out ? <span className="ml-2 text-red-600">timeout</span> : null}
                            </div>
                            {demoCheck.behavioral.error ? (
                              <div className="text-red-700">Error: {String(demoCheck.behavioral.error)}</div>
                            ) : null}
                            {typeof demoCheck.behavioral.exit_code === 'number' ? (
                              <div className="text-gray-700">
                                Exit code: <span className="font-mono">{String(demoCheck.behavioral.exit_code)}</span>
                              </div>
                            ) : null}
                            {typeof demoCheck.behavioral.stdout === 'string' && demoCheck.behavioral.stdout.trim() ? (
                              <div>
                                <div className="flex items-center justify-between">
                                  <div className="text-gray-700">stdout</div>
                                  <Button size="sm" variant="ghost" onClick={() => copyText(String(demoCheck.behavioral.stdout || ''), 'stdout')}>
                                    Copy
                                  </Button>
                                </div>
                                <pre className="mt-1 p-2 bg-white border border-gray-200 rounded whitespace-pre-wrap max-h-48 overflow-auto">
                                  {String(demoCheck.behavioral.stdout)}
                                </pre>
                              </div>
                            ) : null}
                            {typeof demoCheck.behavioral.stderr === 'string' && demoCheck.behavioral.stderr.trim() ? (
                              <div>
                                <div className="flex items-center justify-between">
                                  <div className="text-gray-700">stderr</div>
                                  <Button size="sm" variant="ghost" onClick={() => copyText(String(demoCheck.behavioral.stderr || ''), 'stderr')}>
                                    Copy
                                  </Button>
                                </div>
                                <pre className="mt-1 p-2 bg-white border border-gray-200 rounded whitespace-pre-wrap max-h-48 overflow-auto">
                                  {String(demoCheck.behavioral.stderr)}
                                </pre>
                              </div>
                            ) : null}
                          </div>
                        </details>
                      ) : null}
                    </div>
                    <div className="flex gap-2 shrink-0">
                      <Button size="sm" variant="ghost" onClick={() => copyText(demoCheck.source_id, 'Source ID')}>
                        Copy source
                      </Button>
                    </div>
                  </div>
                )}

                {readingListArtifact?.id && (
                  <div className="flex items-center justify-between gap-3 bg-gray-50 border border-gray-200 rounded-lg p-2">
                    <div className="text-xs text-gray-700">
                      <div className="font-medium text-gray-800">Reading list</div>
                      <div className="text-gray-600">
                        {readingListArtifact.name || 'Reading List'} •{' '}
                        <span className="font-mono">{String(readingListArtifact.id)}</span>
                      </div>
                    </div>
                    <div className="flex gap-2">
                      <Button size="sm" variant="secondary" onClick={() => openReadingList(String(readingListArtifact.id))}>
                        Open
                      </Button>
                      <Button size="sm" variant="ghost" onClick={() => copyText(String(readingListArtifact.id), 'Reading list ID')}>
                        Copy ID
                      </Button>
                    </div>
                  </div>
                )}

                {!readingListArtifact?.id && readingListByName?.id && (
                  <div className="flex items-center justify-between gap-3 bg-gray-50 border border-gray-200 rounded-lg p-2">
                    <div className="text-xs text-gray-700">
                      <div className="font-medium text-gray-800">Reading list</div>
                      <div className="text-gray-600">
                        {readingListByName.name || 'Reading List'} •{' '}
                        <span className="font-mono">{String(readingListByName.id)}</span>
                      </div>
                      <div className="text-gray-500">Resolved by name from job config</div>
                    </div>
                    <div className="flex gap-2">
                      <Button size="sm" variant="secondary" onClick={() => openReadingList(String(readingListByName.id))}>
                        Open
                      </Button>
                      <Button size="sm" variant="ghost" onClick={() => copyText(String(readingListByName.id), 'Reading list ID')}>
                        Copy ID
                      </Button>
                    </div>
                  </div>
                )}

                {(arxivSourceArtifacts.length > 0 || arxivImportsFallback.length > 0) && (
                  <div className="bg-gray-50 border border-gray-200 rounded-lg p-2">
                    <div className="text-xs font-medium text-gray-800 mb-2">arXiv imports</div>
                    <div className="space-y-2">
                      {(arxivSourceArtifacts.length > 0 ? arxivSourceArtifacts : arxivImportsFallback).slice(0, 5).map((s: any, idx: number) => (
                        <div key={`${s.id || idx}`} className="flex items-center justify-between gap-3">
                          <div className="text-xs text-gray-700">
                            <div className="text-gray-800">{s.name || 'ArXiv Import'}</div>
                            <div className="text-gray-600 font-mono">{String(s.id)}</div>
                          </div>
                          <div className="flex gap-2">
                            <Button size="sm" variant="secondary" onClick={() => navigate(`/papers?source_id=${encodeURIComponent(String(s.id))}`)}>
                              Open Papers
                            </Button>
                            <Button size="sm" variant="ghost" onClick={() => copyText(String(s.id), 'Source ID')}>
                              Copy ID
                            </Button>
                          </div>
                        </div>
                      ))}
                      {(arxivSourceArtifacts.length > 5 || arxivImportsFallback.length > 5) && (
                        <div className="text-xs text-gray-500">
                          +{(arxivSourceArtifacts.length > 0 ? arxivSourceArtifacts.length : arxivImportsFallback.length) - 5} more imports
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* The bundle a research run proposed, and its review. */}
          <AIHubBundleSection job={job} />

          {/* Output Artifacts - Charts and Diagrams */}
          {job.output_artifacts && job.output_artifacts.length > 0 && (
            <div className="mb-4">
              <h3 className="text-sm font-medium text-gray-700 mb-2 flex items-center gap-1">
                <BarChart3 className="w-4 h-4" />
                Generated Visualizations ({job.output_artifacts.filter(a => a.type === 'chart' || a.type === 'diagram').length})
              </h3>
              <div className="space-y-3">
                {job.output_artifacts
                  .filter((artifact: any) => artifact.type === 'chart' || artifact.type === 'diagram')
                  .map((artifact: any, idx: number) => (
                    <div key={idx} className="border border-gray-200 rounded-lg overflow-hidden">
                      <div className="px-3 py-2 bg-gray-50 border-b border-gray-200 flex items-center justify-between">
                        <span className="text-xs font-medium text-gray-600 flex items-center gap-1">
                          {artifact.type === 'chart' ? (
                            <BarChart3 className="w-3 h-3" />
                          ) : (
                            <Layers className="w-3 h-3" />
                          )}
                          {artifact.tool || artifact.type}
                          {artifact.format && ` (${artifact.format})`}
                        </span>
                        {artifact.edit_url && (
                          <a
                            href={artifact.edit_url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-xs text-primary-600 hover:text-primary-800"
                          >
                            Edit in Draw.io
                          </a>
                        )}
                      </div>
                      {artifact.image_base64 && (
                        <div className="p-2 bg-white">
                          <img
                            src={`data:${artifact.mime_type || 'image/png'};base64,${artifact.image_base64}`}
                            alt={artifact.tool || 'Visualization'}
                            className="max-w-full h-auto mx-auto"
                            style={{ maxHeight: '300px' }}
                          />
                        </div>
                      )}
                      {artifact.code && artifact.format === 'mermaid' && (
                        <div className="p-2 bg-gray-50 text-gray-900 overflow-x-auto border-t border-gray-200">
                          <pre className="text-xs font-mono whitespace-pre-wrap">{artifact.code}</pre>
                        </div>
                      )}
                      {artifact.code && artifact.format === 'graphviz' && (
                        <div className="p-2 bg-gray-50 text-gray-900 overflow-x-auto border-t border-gray-200">
                          <pre className="text-xs font-mono whitespace-pre-wrap">{artifact.code}</pre>
                        </div>
                      )}
                    </div>
                  ))}
              </div>
            </div>
          )}

          {/* What the run remembered, and the button that extracts it. */}
          <JobMemoriesSection job={job} onExtracted={setManualExtractionSummary} />

          {/* Execution log */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-sm font-medium text-gray-700">Execution Log</h3>
              <div className="flex items-center gap-2">
                {showExecutionLog ? (
                  <Button size="sm" variant="ghost" onClick={loadLog} disabled={loadingLog}>
                    <RefreshCw className={`w-3 h-3 mr-1 ${loadingLog ? 'animate-spin' : ''}`} />
                    Refresh
                  </Button>
                ) : null}
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={() => setShowExecutionLog((current) => !current)}
                >
                  {showExecutionLog ? 'Hide Execution Log' : 'Show Execution Log'}
                </Button>
              </div>
            </div>
            {!showExecutionLog ? (
              <p className="text-sm text-gray-500 text-center py-4">Execution log loads on demand</p>
            ) : loadingLog ? (
              <div className="flex justify-center py-4">
                <LoadingSpinner size="sm" />
              </div>
            ) : logData && logData.entries.length > 0 ? (
              <div className="space-y-2 max-h-60 overflow-y-auto">
                {logData.entries.map((entry, idx) => (
                  <div key={idx} className="text-xs bg-gray-50 rounded p-2">
                    <div className="flex items-center justify-between text-gray-500 mb-1">
                      <span className="font-medium">
                        Iteration {entry.iteration} - {entry.phase}
                      </span>
                      <span>{entry.timestamp}</span>
                    </div>
                    {entry.action && <p className="text-gray-600">Action: {entry.action}</p>}
                    {entry.thought && <p className="text-gray-600 truncate">Thought: {entry.thought}</p>}
                    {entry.error && <p className="text-red-600">Error: {entry.error}</p>}
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-sm text-gray-500 text-center py-4">No log entries yet</p>
            )}
          </div>
        </div>
      </div>
    );
};

export default JobDetailPanel;
